from __future__ import annotations

import copy
import json
import math
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours
from navirl.overseer import ProviderConfig, run_aegis_rerank
from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario
from navirl.verify.judge import run_visual_judge, write_judge_output
from navirl.verify.validators import (
    build_visual_summary,
    run_numeric_invariants,
    sample_key_frames,
)

DEFAULT_SCENARIOS = {
    "quick": [
        "hallway_pass.yaml",
        "doorway_token_yield.yaml",
        "kitchen_congestion.yaml",
        "group_cohesion.yaml",
    ],
    "full": [
        "hallway_pass.yaml",
        "doorway_token_yield.yaml",
        "kitchen_congestion.yaml",
        "group_cohesion.yaml",
        "robot_comfort_avoidance.yaml",
        "routine_cook_dinner_micro.yaml",
    ],
}

DEFAULT_TUNE_RETENTION_HOURS = 168.0


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning experiments.

    Groups all tuning parameters into logical categories for better
    maintainability and type safety.
    """

    # Output configuration
    out_root: str | Path
    retention_hours: float | None = None

    # Scenario configuration
    suite: str = "quick"
    scenarios: list[str] | None = None

    # Search configuration
    trials: int = 24
    seed: int = 17
    search_space_path: str | Path | None = None

    # Judge configuration
    judge_mode: str = "heuristic"
    judge_confidence_min: float = 0.7
    judge_provider: str = "codex"
    judge_model: str | None = None
    judge_endpoint: str | None = None
    judge_api_key_env: str = "NAVIRL_VLM_API_KEY"
    judge_native_cmd: str | None = None
    judge_allow_fallback: bool = True

    # Output format
    max_frames: int = 10
    video: bool = False

    # Aegis configuration
    aegis_rerank: bool = True
    aegis_top_k: int = 6


DEFAULT_SEARCH_SPACE = {
    "scene.orca.neighbor_dist": [3.5, 4.5, 5.5, 6.5],
    "scene.orca.max_neighbors": [10, 14, 18, 24],
    "scene.orca.time_horizon": [2.5, 3.5, 4.5],
    "scene.orca.time_horizon_obst": [2.5, 3.5, 4.5, 5.5],
    "scene.orca.pref_velocity_smoothing": [0.25, 0.35, 0.5, 0.65],
    "scene.orca.wall_clearance_buffer_m": [0.0, 0.005, 0.015, 0.03],
    "evaluation.wall_clearance_buffer": [0.0, 0.005, 0.015, 0.03],
    "evaluation.deadlock_resample_attempts": [4, 6, 8],
    "evaluation.traversability_offset_step": [0.003, 0.005, 0.008],
    "evaluation.traversability_offset_max": [0.02, 0.03, 0.05],
    "humans.controller.params.lookahead": [2, 3, 4],
    "humans.controller.params.waypoint_tolerance": [0.18, 0.22, 0.28],
    "humans.controller.params.slowdown_dist": [0.7, 0.9, 1.2],
    "humans.controller.params.velocity_smoothing": [0.25, 0.4, 0.55, 0.7],
    "humans.controller.params.min_speed": [0.04, 0.08, 0.12],
    "robot.controller.params.target_lookahead": [2, 3, 4],
    "robot.controller.params.slowdown_dist": [0.35, 0.5, 0.7],
    "robot.controller.params.velocity_smoothing": [0.4, 0.55, 0.7, 0.82],
}


@dataclass(slots=True)
class TrialScenarioResult:
    scenario_id: str
    bundle_dir: str
    invariants_pass: bool
    judge_pass: bool
    judge_status: str
    judge_confidence: float
    score: float
    metrics: dict


def _emit_progress(message: str) -> None:
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[navirl tune {ts}] {message}", file=sys.stderr, flush=True)


def _resolve_default_scenario_paths(suite: str) -> list[Path]:
    base = Path(__file__).resolve().parents[1] / "scenarios" / "library"
    return [base / name for name in DEFAULT_SCENARIOS[suite]]


def _resolve_scenarios(scenarios: list[str] | None, suite: str) -> list[Path]:
    if not scenarios:
        return _resolve_default_scenario_paths(suite)

    out: list[Path] = []
    for raw in scenarios:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.yaml")))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"Scenario path does not exist: {raw}")
    if not out:
        raise FileNotFoundError("No scenarios resolved for tuning.")
    return out


def _load_search_space(path: str | Path | None) -> dict[str, list]:
    if path is None:
        return copy.deepcopy(DEFAULT_SEARCH_SPACE)

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".json":
            raw = json.load(f)
        else:
            raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Search space must be an object mapping dotted paths to value lists.")

    out: dict[str, list] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError("Search-space keys must be strings.")
        if not isinstance(v, list) or not v:
            raise ValueError(f"Search-space entry '{k}' must be a non-empty list.")
        out[k] = v
    return out


def _set_dotted_path(obj: dict, path: str, value) -> None:
    keys = path.split(".")
    cur = obj
    for key in keys[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _sample_overrides(rng: random.Random, search_space: dict[str, list]) -> dict[str, object]:
    return {k: rng.choice(v) for k, v in search_space.items()}


def _apply_overrides(scenario: dict, overrides: dict[str, object]) -> dict:
    out = copy.deepcopy(scenario)
    human_type = str(out.get("humans", {}).get("controller", {}).get("type", ""))
    robot_type = str(out.get("robot", {}).get("controller", {}).get("type", ""))

    for path, value in overrides.items():
        if path.startswith("humans.controller.params.") and human_type not in {"orca", "orca_plus"}:
            continue
        if path.startswith("robot.controller.params.") and robot_type != "baseline_astar":
            continue
        _set_dotted_path(out, path, value)

    # Keep geometry offsets consistent between backend runtime and invariant thresholds.
    scene_wall = overrides.get("scene.orca.wall_clearance_buffer_m")
    eval_wall = overrides.get("evaluation.wall_clearance_buffer")
    if scene_wall is not None and eval_wall is None:
        _set_dotted_path(out, "evaluation.wall_clearance_buffer", scene_wall)
    elif eval_wall is not None and scene_wall is None:
        _set_dotted_path(out, "scene.orca.wall_clearance_buffer_m", eval_wall)
    return out


def _score_scenario(metrics: dict, invariants: dict, judge_payload: dict) -> float:
    checks = {
        str(c.get("name", "")): c for c in invariants.get("checks", []) if isinstance(c, dict)
    }
    horizon = max(1, int(metrics.get("horizon_steps", 1)))
    pair_rate = float(metrics.get("collisions_agent_agent", 0)) / float(horizon)
    intrusion = float(metrics.get("intrusion_rate", 0.0))
    obst = float(metrics.get("collisions_agent_obstacle", 0))
    deadlock_count = float(metrics.get("deadlock_count", 0))
    retry_count = float(metrics.get("_retry_count", 0))
    oscillation = float(metrics.get("oscillation_score", 0.0))
    jerk = float(metrics.get("jerk_proxy", 0.0))
    success = float(metrics.get("success_rate", 0.0))
    robot_progress = float(checks.get("robot_progress", {}).get("progress_fraction", success))
    wall_proximity = float(checks.get("wall_proximity_fraction", {}).get("near_wall_fraction", 0.0))
    jitter_worst = float(checks.get("motion_jitter", {}).get("worst_flip_rate", 0.0))
    stop_check = checks.get("agent_stop_duration", {})
    longest_stop_s = 0.0
    if isinstance(stop_check, dict):
        for item in stop_check.get("top_longest_stops", []):
            if isinstance(item, dict):
                longest_stop_s = max(longest_stop_s, float(item.get("max_stopped_seconds", 0.0)))

    judge_conf = float(judge_payload.get("confidence", 0.0))
    inv_ok = bool(invariants.get("overall_pass", False))
    judge_ok = bool(judge_payload.get("overall_pass", False))

    score = 0.0
    score += 2.8 * judge_conf
    score += 1.8 * success
    score += 1.4 * max(0.0, min(1.0, robot_progress))
    score -= 7.5 * deadlock_count
    score -= 3.8 * obst
    score -= 1.4 * intrusion
    score -= 1.1 * pair_rate
    score -= 0.5 * oscillation
    score -= 0.08 * jerk
    score -= 2.4 * wall_proximity
    score -= 1.3 * max(0.0, jitter_worst - 0.15)
    score -= 0.3 * max(0.0, longest_stop_s - 2.0)
    score -= 0.4 * retry_count

    if not inv_ok:
        score -= 8.0
    if not judge_ok:
        score -= 5.0
    if str(judge_payload.get("status")) == "needs_human_review":
        score -= 2.0

    return float(score)


def _sanitize_json_value(value):
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    return value


def _write_report(
    report_path: Path,
    *,
    suite: str,
    scenarios: list[Path],
    trials: int,
    seed: int,
    judge_mode: str,
    judge_confidence_min: float,
    search_space: dict[str, list],
    ranking: list[dict],
) -> None:
    lines = [
        f"# NavIRL Hyperparameter Tuning Report ({suite})",
        "",
        f"- trials: `{trials}`",
        f"- seed: `{seed}`",
        f"- judge_mode: `{judge_mode}`",
        f"- judge_confidence_min: `{judge_confidence_min}`",
        "- scenarios:",
    ]
    for sp in scenarios:
        lines.append(f"  - `{sp}`")

    lines.extend(
        [
            "",
            "## Search Space",
            "",
            "```yaml",
            yaml.safe_dump(search_space, sort_keys=True).rstrip(),
            "```",
            "",
            "## Top Trials",
            "",
            "| Rank | Trial | Score | Pass Rate | Mean Judge Conf | Aegis Realism | Overrides |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(ranking[:5], start=1):
        aegis_val = row.get("aegis_realism_score")
        aegis_txt = f"{float(aegis_val):.3f}" if aegis_val is not None else "-"
        lines.append(
            f"| {idx} | {row['trial_idx']} | {row['aggregate_score']:.3f} | "
            f"{row['pass_rate']:.2f} | {row['mean_judge_confidence']:.2f} | "
            f"{aegis_txt} | `{json.dumps(row['overrides'], sort_keys=True)}` |"
        )

    if ranking:
        best = ranking[0]
        lines.extend(
            [
                "",
                "## Best Overrides",
                "",
                "```yaml",
                yaml.safe_dump(best["overrides"], sort_keys=True).rstrip(),
                "```",
                "",
                "## Reproduction",
                "",
                "```bash",
                "python -m navirl tune --suite "
                f"{suite} --trials {trials} --seed {seed} --judge-mode {judge_mode} "
                f"--judge-confidence-min {judge_confidence_min}",
                "```",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tuning(config: TuningConfig) -> dict:
    if config.trials <= 0:
        raise ValueError("config.trials must be positive")
    if config.max_frames <= 0:
        raise ValueError("config.max_frames must be positive")

    out_root = Path(config.out_root)
    resolved_retention_hours = resolve_retention_hours(
        config.retention_hours,
        env_var="NAVIRL_TUNE_TTL_HOURS",
        default_hours=DEFAULT_TUNE_RETENTION_HOURS,
    )
    pruned_runs = prune_old_run_dirs(
        out_root,
        ttl_hours=resolved_retention_hours,
        prefixes=("tune_",),
        keep_latest=3,
    )

    run_tag = (
        f"tune_{config.suite}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    run_dir = out_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    scenario_paths = _resolve_scenarios(config.scenarios, suite=config.suite)
    scenario_templates = [load_scenario(p) for p in scenario_paths]
    search_space = _load_search_space(config.search_space_path)
    _emit_progress(f"run_dir={run_dir}")
    _emit_progress(
        "config: "
        f"suite={config.suite} config.trials={config.trials} seed={config.seed} scenarios={len(scenario_templates)} "
        f"config.judge_mode={config.judge_mode} config.judge_provider={config.judge_provider} "
        f"config.judge_allow_fallback={config.judge_allow_fallback} config.max_frames={config.max_frames} "
        f"config.video={config.video} retention_hours={resolved_retention_hours}"
    )
    if pruned_runs:
        _emit_progress(f"pruned_old_runs={len(pruned_runs)}")

    rng = random.Random(config.seed)
    ranking: list[dict] = []
    trials_path = run_dir / "trials.jsonl"

    with trials_path.open("w", encoding="utf-8") as trials_file:
        for trial_idx in range(config.trials):
            trial_t0 = perf_counter()
            _emit_progress(f"trial {trial_idx + 1}/{config.trials}: start")
            overrides = _sample_overrides(rng, search_space)
            trial_scenarios: list[TrialScenarioResult] = []

            for template in scenario_templates:
                scenario = _apply_overrides(template, overrides)
                scenario_id = str(scenario["id"])
                trial_out = run_dir / "trials" / f"trial_{trial_idx:03d}"
                run_id = f"trial_{trial_idx:03d}_{scenario_id}"
                _emit_progress(
                    f"trial {trial_idx + 1}/{config.trials} scenario {scenario_id}: run simulation"
                )

                try:
                    log = run_scenario_dict(
                        scenario=scenario,
                        out_root=trial_out,
                        run_id=run_id,
                        render_override=True,
                        video_override=config.video,
                    )
                    bundle_dir = Path(log.bundle_dir)
                except Exception as exc:
                    trial_scenarios.append(
                        TrialScenarioResult(
                            scenario_id=scenario_id,
                            bundle_dir=str(trial_out / run_id / "bundle"),
                            invariants_pass=False,
                            judge_pass=False,
                            judge_status="fail",
                            judge_confidence=0.0,
                            score=-120.0,
                            metrics={"run_error": str(exc)},
                        )
                    )
                    _emit_progress(
                        f"trial {trial_idx + 1}/{config.trials} scenario {scenario_id}: run_error={exc}"
                    )
                    continue

                invariants = run_numeric_invariants(bundle_dir)
                with (bundle_dir / "invariants.json").open("w", encoding="utf-8") as f:
                    json.dump(invariants, f, indent=2, sort_keys=True)

                summary = build_visual_summary(bundle_dir, invariants)
                frame_paths = sample_key_frames(bundle_dir, num_frames=config.max_frames)
                judge_payload = run_visual_judge(
                    bundle_dir=bundle_dir,
                    summary=summary,
                    frame_paths=frame_paths,
                    confidence_threshold=config.judge_confidence_min,
                    mode=config.judge_mode,
                    require_video=config.video,
                    provider=config.judge_provider,
                    model=config.judge_model,
                    endpoint=config.judge_endpoint,
                    api_key_env=config.judge_api_key_env,
                    native_cmd=config.judge_native_cmd,
                    allow_fallback=config.judge_allow_fallback,
                )
                write_judge_output(bundle_dir / "judge.json", judge_payload)

                metrics = dict(summary.get("metrics", {}))
                metrics["_retry_count"] = int(len(summary.get("retry_history", [])))
                scenario_score = _score_scenario(metrics, invariants, judge_payload)
                provider_trace = judge_payload.get("provider_trace", {})
                provider_used = bool(provider_trace.get("provider_used", False))
                fallback_used = bool(provider_trace.get("fallback_used", False))
                trial_scenarios.append(
                    TrialScenarioResult(
                        scenario_id=scenario_id,
                        bundle_dir=str(bundle_dir),
                        invariants_pass=bool(invariants.get("overall_pass", False)),
                        judge_pass=bool(judge_payload.get("overall_pass", False)),
                        judge_status=str(judge_payload.get("status", "fail")),
                        judge_confidence=float(judge_payload.get("confidence", 0.0)),
                        score=scenario_score,
                        metrics=metrics,
                    )
                )
                _emit_progress(
                    f"trial {trial_idx + 1}/{config.trials} scenario {scenario_id}: "
                    f"invariants={'pass' if invariants.get('overall_pass', False) else 'fail'} "
                    f"judge={judge_payload.get('status', 'fail')} "
                    f"conf={float(judge_payload.get('confidence', 0.0)):.2f} "
                    f"provider_used={provider_used} fallback_used={fallback_used}"
                )

            pass_count = sum(1 for r in trial_scenarios if r.invariants_pass and r.judge_pass)
            scenario_count = max(1, len(trial_scenarios))
            aggregate_score = float(sum(r.score for r in trial_scenarios))
            pass_rate = float(pass_count / scenario_count)
            mean_judge_conf = float(
                sum(r.judge_confidence for r in trial_scenarios) / scenario_count
            )

            trial_record = {
                "trial_idx": trial_idx,
                "overrides": overrides,
                "aggregate_score": aggregate_score,
                "pass_count": pass_count,
                "pass_rate": pass_rate,
                "mean_judge_confidence": mean_judge_conf,
                "scenarios": [
                    {
                        "scenario_id": r.scenario_id,
                        "bundle_dir": r.bundle_dir,
                        "invariants_pass": r.invariants_pass,
                        "judge_pass": r.judge_pass,
                        "judge_status": r.judge_status,
                        "judge_confidence": r.judge_confidence,
                        "score": r.score,
                        "metrics": r.metrics,
                    }
                    for r in trial_scenarios
                ],
            }
            trial_record = _sanitize_json_value(trial_record)
            ranking.append(trial_record)
            trials_file.write(json.dumps(trial_record, sort_keys=True) + "\n")
            _emit_progress(
                f"trial {trial_idx + 1}/{config.trials}: done "
                f"score={aggregate_score:.3f} pass_rate={pass_rate:.2f} "
                f"mean_judge_conf={mean_judge_conf:.2f} "
                f"elapsed_s={perf_counter() - trial_t0:.1f}"
            )

    ranking.sort(
        key=lambda r: (
            int(r["pass_count"]),
            float(r["aggregate_score"]),
            float(r["mean_judge_confidence"]),
        ),
        reverse=True,
    )

    rerank_info = {
        "applied": False,
        "provider_used": False,
        "provider_error": "",
        "provider_ranking": [],
        "heuristic_scores": {},
        "blended_scores": {},
        "status": "skipped",
    }
    if config.aegis_rerank and ranking:
        _emit_progress(f"config.aegis_rerank: start top_k={max(1, int(config.aegis_top_k))}")
        rerank_info = run_aegis_rerank(
            ranking,
            mode=config.judge_mode,
            provider_config=ProviderConfig(
                provider=config.judge_provider,
                model=config.judge_model,
                endpoint=config.judge_endpoint,
                api_key_env=config.judge_api_key_env,
                native_cmd=config.judge_native_cmd,
            ),
            top_k=max(1, int(config.aegis_top_k)),
            allow_fallback=config.judge_allow_fallback,
        )
        score_map = {int(k): float(v) for k, v in rerank_info.get("blended_scores", {}).items()}
        for row in ranking:
            trial_idx = int(row["trial_idx"])
            row["aegis_realism_score"] = float(score_map.get(trial_idx, 0.0))
        ranking.sort(
            key=lambda r: (
                int(r["pass_count"]),
                float(r["aggregate_score"]) + 0.35 * float(r.get("aegis_realism_score", 0.0)),
                float(r["mean_judge_confidence"]),
            ),
            reverse=True,
        )
        _emit_progress(
            "config.aegis_rerank: "
            f"status={rerank_info.get('status', 'unknown')} "
            f"provider_used={bool(rerank_info.get('provider_used', False))}"
        )

    best = _sanitize_json_value(ranking[0]) if ranking else {}

    best_path = run_dir / "best_params.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)

    report_path = run_dir / "REPORT.md"
    _write_report(
        report_path,
        suite=config.suite,
        scenarios=scenario_paths,
        trials=config.trials,
        seed=config.seed,
        judge_mode=config.judge_mode,
        judge_confidence_min=config.judge_confidence_min,
        search_space=search_space,
        ranking=ranking,
    )
    rerank_path = run_dir / "AEGIS_RERANK.json"
    with rerank_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(rerank_info), f, indent=2, sort_keys=True)
    _emit_progress(
        "completed: "
        f"report={report_path} best_params={best_path} "
        f"best_trial_idx={best.get('trial_idx', 'n/a')}"
    )

    return _sanitize_json_value(
        {
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "best_params_path": str(best_path),
            "trials_path": str(trials_path),
            "best_trial": best,
            "retention_hours": resolved_retention_hours,
            "pruned_runs": [str(p) for p in pruned_runs],
            "aegis_rerank_path": str(rerank_path),
        }
    )
