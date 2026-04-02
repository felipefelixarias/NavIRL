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


def _write_report(
    report_path: Path,
    suite: str,
    pytest_ok: bool,
    pytest_out: str,
    rows: list[VerifyResult],
    thresholds: dict,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate summary statistics
    total_scenarios = len(rows)
    passed_scenarios = sum(1 for r in rows if r.overall_pass)
    failed_scenarios = total_scenarios - passed_scenarios
    success_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0

    lines = [
        f"# 🔍 NavIRL Verification Report: {suite.title()} Suite",
        "",
        "## 📊 Executive Summary",
        "",
        f"- **Overall Status:** {'✅ PASS' if failed_scenarios == 0 and pytest_ok else '❌ FAIL'}",
        f"- **Success Rate:** {success_rate:.1f}% ({passed_scenarios}/{total_scenarios} scenarios)",
        f"- **Failed Scenarios:** {failed_scenarios}",
        f"- **Test Suite:** {'✅ Pass' if pytest_ok else '❌ Fail'}",
        "",
        f"### 🎯 Quick Actions",
        "",
    ]

    if failed_scenarios == 0:
        lines.extend([
            "🎉 **All scenarios passed!** Your implementation meets verification standards.",
            "",
            "- Consider running the full suite if you ran quick verification",
            "- Review any warnings in the detailed results below",
            ""
        ])
    else:
        lines.extend([
            f"⚠️ **{failed_scenarios} scenario(s) need attention.** Common next steps:",
            "",
            "1. 📖 Review the **Failure Analysis** section below for specific issues",
            "2. 🔧 Apply the suggested fixes for each failed scenario",
            f"3. 🔄 Re-run verification: `python -m navirl verify --suite {suite}`",
            "4. 📝 Check the reproduction commands for individual scenarios",
            ""
        ])

    lines.extend([
        "",
        "## ⚙️ Configuration",
        "",
        f"- **Judge confidence threshold:** {thresholds['judge_confidence_min']}",
        "- **Teleport threshold:** scenario-specific (`evaluation.teleport_thresh`)",
        "- **Max speed:** scenario-specific (`evaluation.max_speed`)",
        "- **Max acceleration:** scenario-specific (`evaluation.max_accel`)",
        "",
        "## 📋 Scenario Results",
        "",
        "| Scenario | 📊 Invariants | 👁️ Judge | 🎯 Confidence | 🎥 Video | ✅ Overall | 📝 Notes | 📁 Bundle |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ])

    for row in rows:
        # Format status with visual indicators
        invariants_status = "✅ Pass" if row.invariants_pass else "❌ Fail"
        judge_status_icon = {"pass": "✅", "fail": "❌", "needs_human_review": "⚠️"}.get(
            row.judge_status, "❓"
        )
        judge_status_text = f"{judge_status_icon} {row.judge_status.replace('_', ' ').title()}"
        video_status = (
            ("✅ Pass" if row.video_check_pass else "❌ Fail")
            if row.video_check_pass is not None
            else "➖"
        )
        overall_status = "✅ Pass" if row.overall_pass else "❌ Fail"
        notes_display = (
            (row.notes[:60] + "..." if len(row.notes) > 60 else row.notes) if row.notes else "➖"
        )

        lines.append(
            f"| {row.scenario_id} | {invariants_status} | {judge_status_text} | {row.judge_confidence:.2f} | {video_status} | {overall_status} | {notes_display} | `{Path(row.bundle_dir).name}` |"
        )

    lines.extend(["", "## ❌ Failure Analysis", ""])
    failing = [r for r in rows if not r.overall_pass]
    if not failing:
        lines.append("🎉 **All scenarios passed!** No failures to report.")
    else:
        lines.extend([
            f"**{len(failing)} scenario(s) failed.** See detailed analysis below:",
            "",
            "*💡 Tip: Each failure includes suggested fixes and reproduction steps.*",
            ""
        ])

        for i, row in enumerate(failing, 1):
            lines.extend([
                f"### {i}. {row.scenario_id} 🔴",
                "",
                f"**Bundle:** `{Path(row.bundle_dir).name}`",
                f"**Overall Status:** {'❌ Failed' if not row.overall_pass else '✅ Passed'}",
                ""
            ])

            # Analyze failure types
            failure_types = []
            if not row.invariants_pass:
                failure_types.append("📊 Invariant Violations")
            if row.judge_status in ["fail", "needs_human_review"]:
                failure_types.append("👁️ Visual Judge Issues")
            if row.video_check_pass is False:
                failure_types.append("🎥 Video Generation")

            if failure_types:
                lines.extend([
                    "**Failure Categories:**",
                    ""
                ])
                for ft in failure_types:
                    lines.append(f"- {ft}")
                lines.append("")

            # Detailed invariant analysis
            inv_path = Path(row.bundle_dir) / "invariants.json"
            if inv_path.exists():
                with inv_path.open("r", encoding="utf-8") as f:
                    inv = json.load(f)
                failed_checks = [c for c in inv.get("checks", []) if not c.get("pass", False)]

                if failed_checks:
                    lines.extend([
                        "#### 📊 Invariant Failures",
                        ""
                    ])
                    for check in failed_checks:
                        check_name = check.get('name', 'unknown')
                        lines.append(f"**❌ {check_name.replace('_', ' ').title()}**")

                        # Add context for specific check types
                        if "num_violations" in check:
                            violations = check.get('num_violations', 0)
                            lines.append(f"- Violation count: **{violations}**")

                        if "message" in check:
                            lines.append(f"- Details: {check['message']}")

                        # Enhanced suggestions formatting
                        if check.get("name") == "scenario_feasibility":
                            suggestions = check.get("suggestions", [])[:6]
                            if suggestions:
                                lines.extend([
                                    "- **🔧 Suggested Fixes:**"
                                ])
                                for j, sug in enumerate(suggestions, 1):
                                    lines.append(f"  {j}. {sug}")

                        lines.append("")

            # Enhanced judge analysis
            judge_path = Path(row.bundle_dir) / "judge.json"
            if judge_path.exists():
                with judge_path.open("r", encoding="utf-8") as f:
                    judge = json.load(f)
                violations = judge.get("violations", [])

                if violations:
                    lines.extend([
                        "#### 👁️ Visual Judge Analysis",
                        "",
                        f"**Confidence:** {row.judge_confidence:.2f}",
                        f"**Status:** {row.judge_status.replace('_', ' ').title()}",
                        ""
                    ])

                    for viol in violations[:6]:
                        viol_type = viol.get('type', 'unknown').replace('_', ' ').title()
                        severity = viol.get('severity', 'n/a').upper()
                        evidence = viol.get('evidence', 'No details provided')

                        lines.extend([
                            f"**🚨 {viol_type}** (Severity: {severity})",
                            f"- Evidence: {evidence}",
                            ""
                        ])

            # Video check details
            if row.video_check_pass is False:
                lines.extend([
                    "#### 🎥 Video Generation Issues",
                    "",
                    "- Video artifact generation failed",
                    "- This may affect visual analysis capabilities",
                    ""
                ])

            # Quick reproduction steps
            lines.extend([
                "#### 🔄 Quick Reproduction",
                "",
                "```bash",
                f"# Re-run this specific scenario",
                f"cd {Path(row.bundle_dir).parent}",
                f"python -m navirl pipeline --scenario {row.scenario_id}",
                "```",
                "",
                "---",
                ""
            ])

    lines.extend(
        [
            "",
            "## 🔄 Reproduction Guide",
            "",
            "### Re-run Full Verification",
            "```bash",
            "# Run the complete verification suite",
            f"python -m navirl verify --suite {suite}",
            "",
            "# Run with different judge settings",
            f"python -m navirl verify --suite {suite} --judge-confidence-min 0.5",
            "",
            "# Run tests only",
            "pytest -q",
            "```",
            "",
            "### Debug Individual Scenarios",
            "```bash",
            "# Example: Debug a specific scenario",
            "python -m navirl pipeline --scenario hallway_pass --render",
            "",
            "# Run with detailed logging",
            f"NAVIRL_LOG_LEVEL=DEBUG python -m navirl verify --suite {suite}",
            "```",
            "",
            "### 📚 Additional Resources",
            "",
            "- **Troubleshooting Guide:** `docs/troubleshooting.md`",
            "- **Scenario Library:** `navirl/scenarios/library/`",
            "- **Verification Config:** `navirl/verify/`",
            "",
            "---",
            "",
            "## 🧪 Test Suite Output",
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

        overall = bool(invariants.get("overall_pass", False)) and bool(
            judge_payload.get("overall_pass", False)
        )
        if suite == "full":
            overall = overall and bool(video_check_pass)

        failed_checks = [c for c in invariants.get("checks", []) if not c.get("pass", False)]
        feasibility = next(
            (c for c in invariants.get("checks", []) if c.get("name") == "scenario_feasibility"),
            None,
        )
        note_parts: list[str] = []
        if failed_checks:
            note_parts.append(
                "failed=" + ",".join(c.get("name", "unknown") for c in failed_checks[:3])
            )
        if isinstance(feasibility, dict):
            sugs = feasibility.get("suggestions", [])
            if sugs:
                note_parts.append("fix=" + str(sugs[0]))
        notes = "; ".join(note_parts)[:220]

        scenario_rows.append(
            VerifyResult(
                scenario_id=scenario_id,
                bundle_dir=str(bundle_dir),
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
