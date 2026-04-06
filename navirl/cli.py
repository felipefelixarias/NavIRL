"""Command-line interface for NavIRL.

This module provides the main CLI entry point for NavIRL operations including
running scenarios, batch experiments, verification, tuning, visualization,
and artifact management.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours
from navirl.experiments import BatchTemplate, run_batch_template
from navirl.metrics import aggregate_reports, compute_metrics_from_bundle
from navirl.overseer import apply_layout_to_scenario, suggest_layout
from navirl.pipeline import expand_state_paths, run_batch, run_scenario_file
from navirl.scenarios import load_scenario
from navirl.scenarios.validate import validate_scenario_dict
from navirl.tune import run_tuning
from navirl.verify import run_verify
from navirl.viz.viewer import replay_log

DEFAULT_LOG_RETENTION_HOURS = 168.0


def _cmd_run(args: argparse.Namespace) -> int:
    retention_hours = resolve_retention_hours(
        args.retention_hours,
        env_var="NAVIRL_LOG_TTL_HOURS",
        default_hours=DEFAULT_LOG_RETENTION_HOURS,
    )
    prune_old_run_dirs(args.out, ttl_hours=retention_hours, keep_latest=20)

    log = run_scenario_file(
        scenario_path=args.scenario,
        out_root=args.out,
        run_id=args.run_id,
        render_override=args.render,
        video_override=args.video,
    )
    print(log.bundle_dir)
    return 0


def _cmd_run_batch(args: argparse.Namespace) -> int:
    retention_hours = resolve_retention_hours(
        args.retention_hours,
        env_var="NAVIRL_LOG_TTL_HOURS",
        default_hours=DEFAULT_LOG_RETENTION_HOURS,
    )
    prune_old_run_dirs(args.out, ttl_hours=retention_hours, keep_latest=20)

    logs = run_batch(args)
    for log in logs:
        print(log.bundle_dir)
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    template = BatchTemplate.from_yaml(args.template)
    summary = run_batch_template(
        template,
        out_root=args.out,
        render=args.render,
        video=args.video,
    )
    print(
        f"Completed {summary.completed_runs}/{summary.total_runs} runs "
        f"({summary.failed_runs} failed)"
    )
    print(f"Report: {args.out}/REPORT.md")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    paths = expand_state_paths(args.inputs)
    if not paths:
        raise FileNotFoundError("No state logs found from provided inputs")

    out_dir = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for sp in paths:
        metrics = compute_metrics_from_bundle(sp)
        rec = {
            "state_path": str(sp),
            "bundle_dir": str(sp.parent),
            "metrics": metrics,
        }
        per_run.append(rec)

    aggregate = aggregate_reports([r["metrics"] for r in per_run])

    with (out_dir / "per_run.json").open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2, sort_keys=True)
    with (out_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    report_md = out_dir / "REPORT.md"
    lines = [
        "# NavIRL Eval Report",
        "",
        f"- num_runs: `{len(per_run)}`",
        f"- avg_success_rate: `{aggregate.get('avg_success_rate', 0.0):.3f}`",
        f"- avg_intrusion_rate: `{aggregate.get('avg_intrusion_rate', 0.0):.3f}`",
        "",
        "## Artifacts",
        "",
        f"- `{out_dir / 'per_run.json'}`",
        f"- `{out_dir / 'aggregate.json'}`",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(report_md))
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    out = replay_log(
        state_path=args.state,
        out_dir=args.out,
        fps=args.fps,
        video=args.video,
        max_frames=args.max_frames,
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, validate=False)
    validate_scenario_dict(scenario)
    print("valid")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    judge_conf_min = args.judge_confidence_min
    if judge_conf_min is None:
        judge_conf_min = 0.7 if args.suite == "quick" else 0.78
    return run_verify(
        suite=args.suite,
        out_root=args.out,
        judge_mode=args.judge_mode,
        judge_confidence_min=judge_conf_min,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_endpoint=args.judge_endpoint,
        judge_api_key_env=args.judge_api_key_env,
        judge_native_cmd=args.judge_native_cmd,
        judge_allow_fallback=args.judge_allow_fallback,
        retention_hours=args.retention_hours,
    )


def _cmd_tune(args: argparse.Namespace) -> int:
    result = run_tuning(
        out_root=args.out,
        suite=args.suite,
        scenarios=args.scenarios,
        trials=args.trials,
        seed=args.seed,
        judge_mode=args.judge_mode,
        judge_confidence_min=args.judge_confidence_min,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_endpoint=args.judge_endpoint,
        judge_api_key_env=args.judge_api_key_env,
        judge_native_cmd=args.judge_native_cmd,
        judge_allow_fallback=args.judge_allow_fallback,
        max_frames=args.max_frames,
        video=args.video,
        search_space_path=args.search_space,
        retention_hours=args.retention_hours,
        aegis_rerank=args.aegis_rerank,
        aegis_top_k=args.aegis_top_k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_overseer_layout(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    with scenario_path.open("r", encoding="utf-8") as f:
        raw_scenario = yaml.safe_load(f)
    if not isinstance(raw_scenario, dict):
        raise ValueError("Scenario file must decode into an object.")
    validate_scenario_dict(raw_scenario)

    resolved = load_scenario(scenario_path, validate=False)
    suggestion = suggest_layout(
        resolved,
        objective=args.objective,
        humans_count=args.humans_count,
        seed=args.seed,
    )

    out = {"suggestion": suggestion}
    if args.write_scenario:
        patched = apply_layout_to_scenario(raw_scenario, suggestion)
        out_path = Path(args.write_scenario)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
        out["scenario_out"] = str(out_path)

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _add_judge_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_mode: str = "heuristic",
    default_confidence_min: float | None = None,
    default_provider: str = "codex",
) -> None:
    """Add common judge-related arguments to a parser."""
    parser.add_argument("--judge-mode", choices=["heuristic", "vlm"], default=default_mode)
    parser.add_argument("--judge-confidence-min", type=float, default=default_confidence_min)
    parser.add_argument(
        "--judge-provider",
        choices=["codex", "claude", "native", "openai_compatible", "kimi"],
        default=default_provider,
    )
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-endpoint", type=str, default=None)
    parser.add_argument("--judge-api-key-env", type=str, default="NAVIRL_VLM_API_KEY")
    parser.add_argument("--judge-native-cmd", type=str, default=None)
    parser.add_argument(
        "--judge-allow-fallback", action=argparse.BooleanOptionalAction, default=True
    )


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, out_default: str = "logs", retention_hours: bool = True
) -> None:
    """Add common arguments like --out and --retention-hours."""
    parser.add_argument("--out", type=str, default=out_default)
    if retention_hours:
        parser.add_argument("--retention-hours", type=float, default=None)


def _create_run_parser(subparsers) -> None:
    """Create the 'run' command parser."""
    p_run = subparsers.add_parser("run", help="Run a scenario and emit a trace bundle")
    p_run.add_argument("scenario", type=str)
    _add_common_arguments(p_run, out_default="logs")
    p_run.add_argument("--run-id", type=str, default=None)
    p_run.add_argument("--render", action=argparse.BooleanOptionalAction, default=None)
    p_run.add_argument("--video", action=argparse.BooleanOptionalAction, default=None)
    p_run.set_defaults(func=_cmd_run)


def _create_batch_parser(subparsers) -> None:
    """Create the 'run-batch' command parser."""
    p_batch = subparsers.add_parser("run-batch", help="Run a directory of scenarios")
    p_batch.add_argument("scenarios", type=str)
    _add_common_arguments(p_batch, out_default="logs")
    p_batch.add_argument("--seeds", type=str, default="7")
    p_batch.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes to use for batch execution (default: 1, sequential)",
    )
    p_batch.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p_batch.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_batch.set_defaults(func=_cmd_run_batch)


def _create_eval_parser(subparsers) -> None:
    """Create the 'eval' command parser."""
    p_eval = subparsers.add_parser("eval", help="Evaluate one or more state logs")
    p_eval.add_argument("inputs", nargs="+", help="state.jsonl files, run dirs, or globs")
    p_eval.add_argument("--report", type=str, default="out/eval")
    p_eval.set_defaults(func=_cmd_eval)


def _create_view_parser(subparsers) -> None:
    """Create the 'view' command parser."""
    p_view = subparsers.add_parser("view", help="Render replay overlays from a state log")
    p_view.add_argument("state", type=str)
    p_view.add_argument("--out", type=str, default="out/view")
    p_view.add_argument("--fps", type=int, default=12)
    p_view.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p_view.add_argument("--max-frames", type=int, default=None)
    p_view.set_defaults(func=_cmd_view)


def _create_validate_parser(subparsers) -> None:
    """Create the 'validate' command parser."""
    p_validate = subparsers.add_parser(
        "validate", help="Validate a scenario file against ScenarioSpec"
    )
    p_validate.add_argument("scenario", type=str)
    p_validate.set_defaults(func=_cmd_validate)


def _create_verify_parser(subparsers) -> None:
    """Create the 'verify' command parser."""
    p_verify = subparsers.add_parser("verify", help="Run NavIRL victory gate suites")
    p_verify.add_argument("--suite", choices=["quick", "full"], default="quick")
    _add_common_arguments(p_verify, out_default="out/verify")
    _add_judge_arguments(p_verify)
    p_verify.set_defaults(func=_cmd_verify)


def _create_tune_parser(subparsers) -> None:
    """Create the 'tune' command parser."""
    p_tune = subparsers.add_parser(
        "tune",
        help="Tune ORCA/social-nav hyperparameters with visual judge feedback",
    )
    p_tune.add_argument("--suite", choices=["quick", "full"], default="quick")
    p_tune.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional scenario files/directories",
    )
    p_tune.add_argument("--trials", type=int, default=20)
    p_tune.add_argument("--seed", type=int, default=17)
    _add_common_arguments(p_tune, out_default="out/tune")
    _add_judge_arguments(p_tune, default_confidence_min=0.7)
    p_tune.add_argument("--aegis-rerank", action=argparse.BooleanOptionalAction, default=True)
    p_tune.add_argument("--aegis-top-k", type=int, default=6)
    p_tune.add_argument("--max-frames", type=int, default=10)
    p_tune.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_tune.add_argument("--search-space", type=str, default=None)
    p_tune.set_defaults(func=_cmd_tune)


def _create_experiment_parser(subparsers) -> None:
    """Create the 'experiment' command parser."""
    p_exp = subparsers.add_parser(
        "experiment",
        help="Run a batch experiment from a YAML template",
    )
    p_exp.add_argument("template", type=str, help="Path to a batch template YAML file")
    _add_common_arguments(p_exp, out_default="out/experiment")
    p_exp.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p_exp.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_exp.set_defaults(func=_cmd_experiment)


def _create_layout_parser(subparsers) -> None:
    """Create the 'overseer-layout' command parser."""
    p_layout = subparsers.add_parser(
        "overseer-layout",
        help="Suggest map-aware starts/goals for realistic showcase scenarios",
    )
    p_layout.add_argument("scenario", type=str)
    p_layout.add_argument(
        "--objective",
        choices=["auto", "cross_flow", "bottleneck_showcase", "comfort_showcase", "comfort"],
        default="auto",
    )
    p_layout.add_argument("--humans-count", type=int, default=None)
    p_layout.add_argument("--seed", type=int, default=17)
    p_layout.add_argument("--write-scenario", type=str, default=None)
    p_layout.set_defaults(func=_cmd_overseer_layout)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete NavIRL command-line argument parser.

    Creates the main parser and all subcommands (run, eval, verify, etc.)
    using helper functions to reduce code duplication.

    Returns:
        Configured ArgumentParser ready for parsing command line arguments.
    """
    parser = argparse.ArgumentParser(prog="navirl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create all command parsers using dedicated helper functions
    _create_run_parser(subparsers)
    _create_batch_parser(subparsers)
    _create_eval_parser(subparsers)
    _create_view_parser(subparsers)
    _create_validate_parser(subparsers)
    _create_verify_parser(subparsers)
    _create_tune_parser(subparsers)
    _create_experiment_parser(subparsers)
    _create_layout_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
