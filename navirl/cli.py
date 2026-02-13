from __future__ import annotations

import argparse
import json
from pathlib import Path

from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours
from navirl.metrics import aggregate_reports, compute_metrics_from_bundle
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
    _ = args.parallel  # currently sequential; kept for CLI compatibility.
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
        max_frames=args.max_frames,
        video=args.video,
        search_space_path=args.search_space,
        retention_hours=args.retention_hours,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navirl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a scenario and emit a trace bundle")
    p_run.add_argument("scenario", type=str)
    p_run.add_argument("--out", type=str, default="logs")
    p_run.add_argument("--run-id", type=str, default=None)
    p_run.add_argument("--render", action=argparse.BooleanOptionalAction, default=None)
    p_run.add_argument("--video", action=argparse.BooleanOptionalAction, default=None)
    p_run.add_argument("--retention-hours", type=float, default=None)
    p_run.set_defaults(func=_cmd_run)

    p_batch = sub.add_parser("run-batch", help="Run a directory of scenarios")
    p_batch.add_argument("scenarios", type=str)
    p_batch.add_argument("--out", type=str, default="logs")
    p_batch.add_argument("--seeds", type=str, default="7")
    p_batch.add_argument("--parallel", type=int, default=1)
    p_batch.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p_batch.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_batch.add_argument("--retention-hours", type=float, default=None)
    p_batch.set_defaults(func=_cmd_run_batch)

    p_eval = sub.add_parser("eval", help="Evaluate one or more state logs")
    p_eval.add_argument("inputs", nargs="+", help="state.jsonl files, run dirs, or globs")
    p_eval.add_argument("--report", type=str, default="out/eval")
    p_eval.set_defaults(func=_cmd_eval)

    p_view = sub.add_parser("view", help="Render replay overlays from a state log")
    p_view.add_argument("state", type=str)
    p_view.add_argument("--out", type=str, default="out/view")
    p_view.add_argument("--fps", type=int, default=12)
    p_view.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p_view.add_argument("--max-frames", type=int, default=None)
    p_view.set_defaults(func=_cmd_view)

    p_validate = sub.add_parser("validate", help="Validate a scenario file against ScenarioSpec")
    p_validate.add_argument("scenario", type=str)
    p_validate.set_defaults(func=_cmd_validate)

    p_verify = sub.add_parser("verify", help="Run NavIRL victory gate suites")
    p_verify.add_argument("--suite", choices=["quick", "full"], default="quick")
    p_verify.add_argument("--out", type=str, default="out/verify")
    p_verify.add_argument("--judge-mode", choices=["heuristic", "vlm"], default="heuristic")
    p_verify.add_argument("--judge-confidence-min", type=float, default=None)
    p_verify.set_defaults(func=_cmd_verify)

    p_tune = sub.add_parser(
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
    p_tune.add_argument("--out", type=str, default="out/tune")
    p_tune.add_argument("--judge-mode", choices=["heuristic", "vlm"], default="heuristic")
    p_tune.add_argument("--judge-confidence-min", type=float, default=0.7)
    p_tune.add_argument("--max-frames", type=int, default=10)
    p_tune.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_tune.add_argument("--search-space", type=str, default=None)
    p_tune.add_argument("--retention-hours", type=float, default=None)
    p_tune.set_defaults(func=_cmd_tune)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
