from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours
from navirl.pipeline import run_scenario_file
from navirl.verify.judge import run_visual_judge, write_judge_output
from navirl.verify.validators import (
    build_visual_summary,
    check_video_artifact,
    run_numeric_invariants,
    sample_key_frames,
)

PASS = 0
FAIL = 10
NEEDS_HUMAN_REVIEW = 20

DEFAULT_VERIFY_RETENTION_HOURS = 168.0

CANONICAL_SCENARIOS = [
    "hallway_pass.yaml",
    "doorway_token_yield.yaml",
    "kitchen_congestion.yaml",
    "group_cohesion.yaml",
    "robot_comfort_avoidance.yaml",
    "routine_cook_dinner_micro.yaml",
]


@dataclass(slots=True)
class VerifyResult:
    scenario_id: str
    bundle_dir: str
    test_pass: bool
    invariants_pass: bool
    judge_status: str
    judge_confidence: float
    overall_pass: bool
    video_check_pass: bool | None = None
    notes: str = ""


def _run_pytest() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    return (proc.returncode == 0), out


def _scenario_file_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "scenarios" / "library" / name


def _write_report(report_path: Path, suite: str, pytest_ok: bool, pytest_out: str, rows: list[VerifyResult], thresholds: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# NavIRL Verify Report ({suite})",
        "",
        f"- Pytest pass: `{pytest_ok}`",
        "- Thresholds:",
        f"  - judge_confidence_min: `{thresholds['judge_confidence_min']}`",
        "  - teleport_thresh: scenario-specific (`evaluation.teleport_thresh`)",
        "  - max_speed: scenario-specific (`evaluation.max_speed`)",
        "  - max_accel: scenario-specific (`evaluation.max_accel`)",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Invariants | Judge | Confidence | Video | Overall | Notes | Bundle |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row.scenario_id} | {row.invariants_pass} | {row.judge_status} | {row.judge_confidence:.2f} | {row.video_check_pass} | {row.overall_pass} | {row.notes or '-'} | `{row.bundle_dir}` |"
        )

    lines.extend(["", "## Failing Details", ""])
    failing = [r for r in rows if not r.overall_pass]
    if not failing:
        lines.append("- None")
    else:
        for row in failing:
            lines.append(f"### {row.scenario_id}")
            inv_path = Path(row.bundle_dir) / "invariants.json"
            if inv_path.exists():
                with inv_path.open("r", encoding="utf-8") as f:
                    inv = json.load(f)
                failed_checks = [c for c in inv.get("checks", []) if not c.get("pass", False)]
                if not failed_checks:
                    lines.append("- Invariants: none failed.")
                else:
                    for check in failed_checks:
                        lines.append(f"- Failed check: `{check.get('name', 'unknown')}`")
                        if check.get("name") == "scenario_feasibility":
                            for sug in check.get("suggestions", [])[:6]:
                                lines.append(f"- Suggested fix: {sug}")
                        elif "num_violations" in check:
                            lines.append(f"- Violations: `{check.get('num_violations')}`")
            judge_path = Path(row.bundle_dir) / "judge.json"
            if judge_path.exists():
                with judge_path.open("r", encoding="utf-8") as f:
                    judge = json.load(f)
                for viol in judge.get("violations", [])[:6]:
                    lines.append(
                        f"- Judge: `{viol.get('type', 'unknown')}` ({viol.get('severity', 'n/a')}) - {viol.get('evidence', '')}"
                    )
            lines.append("")

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "pytest -q",
            f"python -m navirl verify --suite {suite}",
            "```",
            "",
            "## Pytest Output",
            "",
            "```text",
            pytest_out.strip(),
            "```",
        ]
    )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_verify(
    suite: str,
    out_root: str | Path,
    *,
    judge_mode: str = "heuristic",
    judge_confidence_min: float = 0.6,
    judge_provider: str = "codex",
    judge_model: str | None = None,
    judge_endpoint: str | None = None,
    judge_api_key_env: str = "NAVIRL_VLM_API_KEY",
    judge_native_cmd: str | None = None,
    judge_allow_fallback: bool = True,
    retention_hours: float | None = None,
) -> int:
    out_root = Path(out_root)
    verify_root = out_root / suite
    verify_root.mkdir(parents=True, exist_ok=True)
    resolved_retention_hours = resolve_retention_hours(
        retention_hours,
        env_var="NAVIRL_VERIFY_TTL_HOURS",
        default_hours=DEFAULT_VERIFY_RETENTION_HOURS,
    )
    prune_old_run_dirs(
        verify_root,
        ttl_hours=resolved_retention_hours,
        prefixes=(f"verify_{suite}_",),
        keep_latest=2,
    )

    pytest_ok, pytest_out = _run_pytest()

    scenario_rows: list[VerifyResult] = []
    for name in CANONICAL_SCENARIOS:
        scenario_path = _scenario_file_path(name)
        scenario_id = scenario_path.stem
        run_id = f"verify_{suite}_{scenario_id}"

        # Full suite always requests video artifacts.
        video_override = True if suite == "full" else None
        render_override = True

        try:
            log = run_scenario_file(
                scenario_path=scenario_path,
                out_root=verify_root,
                run_id=run_id,
                render_override=render_override,
                video_override=video_override,
            )
        except Exception as exc:
            scenario_rows.append(
                VerifyResult(
                    scenario_id=scenario_id,
                    bundle_dir=str(verify_root / run_id / "bundle"),
                    test_pass=pytest_ok,
                    invariants_pass=False,
                    judge_status="fail",
                    judge_confidence=0.0,
                    overall_pass=False,
                    video_check_pass=False if suite == "full" else None,
                    notes=f"run_error={str(exc)[:180]}",
                )
            )
            continue

        bundle_dir = Path(log.bundle_dir)
        invariants = run_numeric_invariants(bundle_dir)
        with (bundle_dir / "invariants.json").open("w", encoding="utf-8") as f:
            json.dump(invariants, f, indent=2, sort_keys=True)

        visual_summary = build_visual_summary(bundle_dir, invariants)
        frame_paths = sample_key_frames(bundle_dir, num_frames=10 if suite == "full" else 8)
        judge_payload = run_visual_judge(
            bundle_dir,
            visual_summary,
            frame_paths,
            confidence_threshold=judge_confidence_min,
            mode=judge_mode,
            require_video=(suite == "full"),
            provider=judge_provider,
            model=judge_model,
            endpoint=judge_endpoint,
            api_key_env=judge_api_key_env,
            native_cmd=judge_native_cmd,
            allow_fallback=judge_allow_fallback,
        )
        write_judge_output(bundle_dir / "judge.json", judge_payload)

        video_check_pass = None
        if suite == "full":
            video_check = check_video_artifact(bundle_dir)
            video_check_pass = bool(video_check["pass"])
            with (bundle_dir / "video_check.json").open("w", encoding="utf-8") as f:
                json.dump(video_check, f, indent=2, sort_keys=True)

        overall = bool(invariants.get("overall_pass", False)) and bool(judge_payload.get("overall_pass", False))
        if suite == "full":
            overall = overall and bool(video_check_pass)

        failed_checks = [c for c in invariants.get("checks", []) if not c.get("pass", False)]
        feasibility = next((c for c in invariants.get("checks", []) if c.get("name") == "scenario_feasibility"), None)
        note_parts: list[str] = []
        if failed_checks:
            note_parts.append("failed=" + ",".join(c.get("name", "unknown") for c in failed_checks[:3]))
        if isinstance(feasibility, dict):
            sugs = feasibility.get("suggestions", [])
            if sugs:
                note_parts.append("fix=" + str(sugs[0]))
        notes = "; ".join(note_parts)[:220]

        scenario_rows.append(
            VerifyResult(
                scenario_id=scenario_id,
                bundle_dir=str(bundle_dir),
                test_pass=pytest_ok,
                invariants_pass=bool(invariants.get("overall_pass", False)),
                judge_status=str(judge_payload.get("status", "fail")),
                judge_confidence=float(judge_payload.get("confidence", 0.0)),
                overall_pass=overall,
                video_check_pass=video_check_pass,
                notes=notes,
            )
        )

    report_path = verify_root / "REPORT.md"
    _write_report(
        report_path,
        suite=suite,
        pytest_ok=pytest_ok,
        pytest_out=pytest_out,
        rows=scenario_rows,
        thresholds={"judge_confidence_min": judge_confidence_min},
    )

    if any(r.judge_status == "needs_human_review" for r in scenario_rows):
        return NEEDS_HUMAN_REVIEW

    if (not pytest_ok) or any(not r.overall_pass for r in scenario_rows):
        return FAIL

    return PASS
