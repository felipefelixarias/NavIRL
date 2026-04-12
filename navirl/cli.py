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
from navirl.orchestration import Orchestrator, OrchestratorConfig
from navirl.overseer import apply_layout_to_scenario, suggest_layout
from navirl.packs import load_pack, run_pack, write_pack_json, write_pack_markdown
from navirl.pipeline import expand_state_paths, run_batch, run_scenario_file
from navirl.repro import build_repro_package, run_checklist, verify_repro_package
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


def _cmd_pack_run(args: argparse.Namespace) -> int:
    manifest = load_pack(args.manifest)
    result = run_pack(
        manifest,
        out_root=args.out,
        render=args.render,
        video=args.video,
    )
    completed = sum(1 for r in result.runs if r.status == "completed")
    failed = sum(1 for r in result.runs if r.status == "failed")

    out_dir = Path(args.out)
    write_pack_json(result, out_dir / "pack_results.json")
    write_pack_markdown(result, out_dir / "PACK_REPORT.md", manifest.metrics)

    print(
        f"Pack '{manifest.name}' v{manifest.version}: "
        f"{completed}/{len(result.runs)} completed ({failed} failed)"
    )
    print(f"Checksum: {result.manifest_checksum[:16]}...")
    print(f"Report:   {out_dir / 'PACK_REPORT.md'}")
    return 0


def _cmd_pack_validate(args: argparse.Namespace) -> int:
    manifest = load_pack(args.manifest)
    print(f"Pack '{manifest.name}' v{manifest.version}")
    print(f"  Scenarios: {len(manifest.scenarios)}")
    print(f"  Total runs: {manifest.total_runs}")
    print(f"  Metrics: {len(manifest.metrics)}")
    print(f"  Checksum: {manifest.checksum()[:16]}...")
    for entry in manifest.scenarios:
        print(f"  - {entry.id}: {len(entry.seeds)} seeds, path={entry.path}")
    print("valid")
    return 0


def _create_pack_parser(subparsers) -> None:
    """Create the 'pack' command parser with run/validate subcommands."""
    p_pack = subparsers.add_parser(
        "pack",
        help="Work with standardized experiment packs",
    )
    pack_sub = p_pack.add_subparsers(dest="pack_command", required=True)

    # pack run
    p_run = pack_sub.add_parser("run", help="Execute an experiment pack")
    p_run.add_argument("manifest", type=str, help="Path to pack manifest YAML")
    _add_common_arguments(p_run, out_default="out/pack")
    p_run.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p_run.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_run.set_defaults(func=_cmd_pack_run)

    # pack validate
    p_val = pack_sub.add_parser("validate", help="Validate a pack manifest")
    p_val.add_argument("manifest", type=str, help="Path to pack manifest YAML")
    p_val.set_defaults(func=_cmd_pack_validate)


def _cmd_orchestrate(args: argparse.Namespace) -> int:
    template = BatchTemplate.from_yaml(args.template)
    config = OrchestratorConfig(
        num_shards=args.shards,
        max_retries=args.retries,
        max_workers=args.workers if args.workers > 0 else None,
        render=args.render,
        video=args.video,
    )
    orch = Orchestrator(template=template, out_root=args.out, config=config)
    if args.resume:
        summary = orch.resume()
    else:
        summary = orch.run()
    print(
        f"Completed {summary.completed_runs}/{summary.total_runs} runs "
        f"({summary.failed_runs} failed)"
    )
    print(f"Report: {args.out}/REPORT.md")
    return 0


def _create_orchestrate_parser(subparsers) -> None:
    """Create the 'orchestrate' command parser."""
    p_orch = subparsers.add_parser(
        "orchestrate",
        help="Run a batch experiment with distributed orchestration and resumability",
    )
    p_orch.add_argument("template", type=str, help="Path to a batch template YAML file")
    _add_common_arguments(p_orch, out_default="out/orchestrate")
    p_orch.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Number of shards to partition work into (default: 4)",
    )
    p_orch.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Max concurrent workers (default: 0 = one per shard)",
    )
    p_orch.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max retry attempts per shard (default: 2)",
    )
    p_orch.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p_orch.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    p_orch.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume a previously interrupted run",
    )
    p_orch.set_defaults(func=_cmd_orchestrate)


def _cmd_repro_build(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    scenario_paths = None
    if args.scenarios:
        scenario_paths = [Path(s) for s in args.scenarios]

    pack_result_path = Path(args.pack_results) if args.pack_results else None
    metadata = {}
    if args.author:
        metadata["author"] = args.author
    if args.study:
        metadata["study"] = args.study

    package = build_repro_package(
        name=args.name,
        version=args.version,
        description=args.description or "",
        run_dir=run_dir,
        scenario_paths=scenario_paths,
        pack_result_path=pack_result_path,
        out_dir=Path(args.out),
        metadata=metadata,
    )
    print(f"Reproducibility package '{package.name}' v{package.version}")
    print(f"  Artifacts: {len(package.artifacts)}")
    print(f"  Checksum: {package.checksum()[:16]}...")
    print(f"  Output: {args.out}/MANIFEST.json")
    return 0


def _cmd_repro_check(args: argparse.Namespace) -> int:
    package_dir = Path(args.package_dir)
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")

    report = run_checklist(package_dir)

    if args.format == "markdown":
        print(report.to_markdown())
    else:
        print(json.dumps(report.to_dict(), indent=2))

    return 0 if report.passed else 1


def _cmd_repro_verify(args: argparse.Namespace) -> int:
    package_dir = Path(args.package_dir)
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Package directory not found: {package_dir}")

    ok, issues = verify_repro_package(package_dir)

    if ok:
        print("All artifact checksums verified.")
        return 0

    print(f"Integrity check failed ({len(issues)} issue(s)):")
    for issue in issues:
        print(f"  - {issue}")
    return 1


def _create_repro_parser(subparsers) -> None:
    """Create the 'repro' command parser with build/check/verify subcommands."""
    p_repro = subparsers.add_parser(
        "repro",
        help="Build and verify reproducibility packages",
    )
    repro_sub = p_repro.add_subparsers(dest="repro_command", required=True)

    # repro build
    p_build = repro_sub.add_parser("build", help="Build a reproducibility package")
    p_build.add_argument("name", type=str, help="Package name")
    p_build.add_argument("run_dir", type=str, help="Directory with experiment run outputs")
    p_build.add_argument("--out", type=str, default="out/repro")
    p_build.add_argument("--version", type=str, default="1.0")
    p_build.add_argument("--description", type=str, default=None)
    p_build.add_argument(
        "--scenarios", nargs="*", default=None, help="Explicit scenario YAML paths"
    )
    p_build.add_argument("--pack-results", type=str, default=None, help="pack_results.json path")
    p_build.add_argument("--author", type=str, default=None)
    p_build.add_argument("--study", type=str, default=None)
    p_build.set_defaults(func=_cmd_repro_build)

    # repro check
    p_check = repro_sub.add_parser("check", help="Run publication readiness checklist")
    p_check.add_argument("package_dir", type=str, help="Path to reproducibility package")
    p_check.add_argument(
        "--format", choices=["json", "markdown"], default="markdown", help="Output format"
    )
    p_check.set_defaults(func=_cmd_repro_check)

    # repro verify
    p_verify = repro_sub.add_parser("verify", help="Verify artifact integrity")
    p_verify.add_argument("package_dir", type=str, help="Path to reproducibility package")
    p_verify.set_defaults(func=_cmd_repro_verify)


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
    _create_pack_parser(subparsers)
    _create_orchestrate_parser(subparsers)
    _create_repro_parser(subparsers)
    _create_layout_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
