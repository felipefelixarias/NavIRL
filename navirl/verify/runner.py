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
    verify_root: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate enhanced summary statistics
    total_scenarios = len(rows)
    passed_scenarios = sum(1 for r in rows if r.overall_pass)
    failed_scenarios = total_scenarios - passed_scenarios
    needs_review = sum(1 for r in rows if r.judge_status == "needs_human_review")
    invariant_failures = sum(1 for r in rows if not r.invariants_pass)
    judge_failures = sum(1 for r in rows if r.judge_status == "fail")
    success_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0

    lines = [
        f"# 🔍 NavIRL Verification Report: {suite.title()} Suite",
        "",
        "## 📊 Executive Summary",
        "",
        f"- **Overall Status:** {'✅ PASS' if failed_scenarios == 0 and pytest_ok else '❌ FAIL'}",
        f"- **Success Rate:** {success_rate:.1f}% ({passed_scenarios}/{total_scenarios} scenarios)",
        f"- **Failed Scenarios:** {failed_scenarios}",
        f"- **Needs Review:** {needs_review}",
        f"- **Test Suite:** {'✅ Pass' if pytest_ok else '❌ Fail'}",
        "",
        "### 🎯 Quick Actions",
        "",
    ]

    if failed_scenarios == 0 and pytest_ok:
        lines.extend(
            [
                "🎉 **All scenarios passed!** Your implementation meets verification standards.",
                "",
                "**✅ Next Steps:**",
                "- Consider running the full suite if you ran quick verification",
                "- Review any warnings in the detailed results below",
                "- Ready for production deployment consideration",
                "",
            ]
        )
    elif failed_scenarios == 0 and not pytest_ok:
        lines.extend(
            [
                "⚠️ **Scenarios passed but test suite failed.** Core functionality may be compromised.",
                "",
                "**🚨 Immediate Actions:**",
                "1. 🔍 Fix failing tests - see Test Suite Output section",
                "2. 🧪 Ensure all unit tests pass before scenario verification",
                f"3. 🔄 Re-run: `python -m navirl verify --suite {suite}`",
                "",
            ]
        )
    else:
        failure_types = []
        if invariant_failures:
            failure_types.append(f"📊 {invariant_failures} invariant violation(s)")
        if judge_failures:
            failure_types.append(f"👁️ {judge_failures} visual issue(s)")
        if needs_review:
            failure_types.append(f"⚠️ {needs_review} scenario(s) need human review")

        lines.extend(
            [
                f"❌ **{failed_scenarios} scenario(s) need attention.** Failure breakdown:",
                "",
                *[f"- {ft}" for ft in failure_types],
                "",
                "**📋 Recommended Workflow:**",
                "1. 📖 Review **Failure Analysis** section below for specific issues",
                "2. 🔧 Start with the first failed scenario - fixes often resolve multiple issues",
                "3. 🔄 Test individually: `python -m navirl pipeline --scenario <scenario_name>`",
                f"4. 🧪 Re-run full suite: `python -m navirl verify --suite {suite}`",
                "5. 🎯 Focus on highest-impact fixes first (invariant violations > visual issues)",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## ⚙️ Configuration",
            "",
            f"- **Judge confidence threshold:** {thresholds['judge_confidence_min']} (higher = stricter)",
            "- **Suite type:** {suite} ({'full artifacts + video' if suite == 'full' else 'essential checks only'})",
            "- **Teleport threshold:** scenario-specific (`evaluation.teleport_thresh`)",
            "- **Max speed:** scenario-specific (`evaluation.max_speed`)",
            "- **Max acceleration:** scenario-specific (`evaluation.max_accel`)",
            "",
            "## 📋 Scenario Results",
            "",
            "*💡 Tip: Click on bundle names to explore detailed artifacts and logs.*",
            "",
            "| Scenario | 📊 Invariants | 👁️ Judge | 🎯 Confidence | 🎥 Video | ✅ Overall | 📝 Key Issues | 📁 Bundle |",
            "|---|:---:|:---:|:---:|:---:|:---:|---|:---:|",
        ]
    )

    for row in rows:
        # Enhanced status formatting with better visual hierarchy
        invariants_status = "✅ Pass" if row.invariants_pass else "❌ **Fail**"

        judge_icons = {"pass": "✅", "fail": "❌", "needs_human_review": "⚠️"}
        judge_status_icon = judge_icons.get(row.judge_status, "❓")
        judge_text = row.judge_status.replace("_", " ").title()
        judge_status_text = f"{judge_status_icon} {judge_text}"

        # Confidence with visual indicators
        confidence_display = f"{row.judge_confidence:.2f}"
        if row.judge_confidence < 0.3:
            confidence_display = f"🔴 {confidence_display}"
        elif row.judge_confidence < 0.6:
            confidence_display = f"🟡 {confidence_display}"
        elif row.judge_confidence >= 0.8:
            confidence_display = f"🟢 {confidence_display}"

        video_status = (
            ("✅ Pass" if row.video_check_pass else "❌ **Fail**")
            if row.video_check_pass is not None
            else "➖"
        )
        overall_status = "✅ **Pass**" if row.overall_pass else "❌ **Fail**"

        # Enhanced notes with better truncation and key info
        if row.notes:
            # Extract the most important part of the notes
            if "failed=" in row.notes:
                failed_part = row.notes.split("failed=")[1].split(";")[0]
                notes_display = f"Failed: {failed_part}"
            elif "fix=" in row.notes:
                fix_part = row.notes.split("fix=")[1][:50]
                notes_display = f"Fix: {fix_part}"
            else:
                notes_display = row.notes[:50] + "..." if len(row.notes) > 50 else row.notes
        else:
            notes_display = "✅ No issues"

        lines.append(
            f"| **{row.scenario_id}** | {invariants_status} | {judge_status_text} | {confidence_display} | {video_status} | {overall_status} | {notes_display} | [`{Path(row.bundle_dir).name}`]({Path(row.bundle_dir).name}) |"
        )

    lines.extend(["", "## ❌ Failure Analysis", ""])
    failing = [r for r in rows if not r.overall_pass]
    if not failing:
        lines.append("🎉 **All scenarios passed!** No failures to report.")
    else:
        # Categorize failures for better organization
        critical_failures = [r for r in failing if not r.invariants_pass]
        visual_issues = [r for r in failing if r.judge_status == "fail"]
        needs_review_issues = [r for r in failing if r.judge_status == "needs_human_review"]

        lines.extend(
            [
                f"**{len(failing)} scenario(s) failed.** Organized by priority:",
                "",
                "| Priority | Category | Count | Description |",
                "|:---:|---|:---:|---|",
                f"| 🔴 | **Critical** | {len(critical_failures)} | Invariant violations (core functionality broken) |",
                f"| 🟡 | **Visual** | {len(visual_issues)} | Visual judge failures (behavior concerns) |",
                f"| 🟠 | **Review** | {len(needs_review_issues)} | Needs human evaluation (edge cases) |",
                "",
                "*💡 Strategy: Fix critical issues first - they often resolve visual problems too.*",
                "",
            ]
        )

        # Process failures in priority order
        prioritized_failures = []
        prioritized_failures.extend([(r, "🔴 Critical") for r in critical_failures])
        prioritized_failures.extend(
            [(r, "🟡 Visual") for r in visual_issues if r not in critical_failures]
        )
        prioritized_failures.extend(
            [
                (r, "🟠 Review")
                for r in needs_review_issues
                if r not in critical_failures and r not in visual_issues
            ]
        )

        for i, (row, priority_label) in enumerate(prioritized_failures, 1):
            lines.extend(
                [
                    f"### {i}. {row.scenario_id} {priority_label}",
                    "",
                    f"**Bundle:** `{Path(row.bundle_dir).name}`",
                    f"**Judge Confidence:** {row.judge_confidence:.2f}",
                    "",
                ]
            )

            # Enhanced failure type analysis
            failure_types = []
            severity_indicators = []

            if not row.invariants_pass:
                failure_types.append("📊 **Invariant Violations** (High Priority)")
                severity_indicators.append("Core simulation rules broken")
            if row.judge_status == "fail":
                failure_types.append("👁️ **Visual Judge Issues** (Medium Priority)")
                severity_indicators.append("Behavior appears problematic")
            elif row.judge_status == "needs_human_review":
                failure_types.append("⚠️ **Needs Human Review** (Low Priority)")
                severity_indicators.append("Ambiguous behavior detected")
            if row.video_check_pass is False:
                failure_types.append("🎥 **Video Generation** (Infrastructure)")
                severity_indicators.append("Artifact generation failed")

            if failure_types:
                lines.extend(["**Issue Summary:**", ""])
                for ft, sev in zip(failure_types, severity_indicators, strict=True):
                    lines.extend(
                        [
                            f"- {ft}",
                            f"  - *Impact:* {sev}",
                        ]
                    )
                lines.append("")

            # Enhanced detailed invariant analysis
            inv_path = Path(row.bundle_dir) / "invariants.json"
            if inv_path.exists():
                with inv_path.open("r", encoding="utf-8") as f:
                    inv = json.load(f)
                failed_checks = [c for c in inv.get("checks", []) if not c.get("pass", False)]

                if failed_checks:
                    lines.extend(
                        [
                            "#### 📊 Invariant Analysis",
                            "",
                            f"**{len(failed_checks)}** check(s) failed out of {len(inv.get('checks', []))} total.",
                            "",
                        ]
                    )

                    # Categorize checks by severity
                    critical_checks = []
                    warning_checks = []

                    for check in failed_checks:
                        check_name = check.get("name", "unknown")
                        severity = check.get("severity", "unknown")

                        if severity in ["critical", "high"] or check_name in [
                            "collision_free",
                            "safety_constraints",
                        ]:
                            critical_checks.append(check)
                        else:
                            warning_checks.append(check)

                    # Process critical checks first
                    if critical_checks:
                        lines.extend(["**🚨 Critical Issues (Fix Immediately):**", ""])
                        for check in critical_checks:
                            check_name = check.get("name", "unknown").replace("_", " ").title()
                            lines.append(f"**❌ {check_name}**")

                            if "num_violations" in check:
                                violations = check.get("num_violations", 0)
                                lines.append(f"- **Violations:** {violations}")

                            if "message" in check:
                                lines.append(f"- **Issue:** {check['message']}")

                            # Enhanced suggestions with actionability
                            if check.get("name") == "scenario_feasibility":
                                suggestions = check.get("suggestions", [])[:4]
                                if suggestions:
                                    lines.append("- **🛠️ Immediate Actions:**")
                                    for j, sug in enumerate(suggestions, 1):
                                        lines.append(f"  {j}. {sug}")

                            lines.append("")

                    # Process warning checks
                    if warning_checks:
                        lines.extend(["**⚠️ Warnings (Address After Critical Issues):**", ""])
                        for check in warning_checks:
                            check_name = check.get("name", "unknown").replace("_", " ").title()
                            lines.append(f"**🟡 {check_name}**")

                            if "num_violations" in check:
                                violations = check.get("num_violations", 0)
                                lines.append(f"- Count: {violations}")

                            if "message" in check:
                                lines.append(f"- Details: {check['message']}")

                            lines.append("")

            # Enhanced judge analysis with actionable insights
            judge_path = Path(row.bundle_dir) / "judge.json"
            if judge_path.exists():
                with judge_path.open("r", encoding="utf-8") as f:
                    judge = json.load(f)
                violations = judge.get("violations", [])

                if violations:
                    confidence_desc = (
                        "Very Low"
                        if row.judge_confidence < 0.3
                        else (
                            "Low"
                            if row.judge_confidence < 0.6
                            else "Moderate" if row.judge_confidence < 0.8 else "High"
                        )
                    )

                    lines.extend(
                        [
                            "#### 👁️ Visual Behavior Analysis",
                            "",
                            f"**Judge Confidence:** {row.judge_confidence:.2f} ({confidence_desc})",
                            f"**Decision:** {row.judge_status.replace('_', ' ').title()}",
                            f"**Violations Found:** {len(violations)}",
                            "",
                        ]
                    )

                    # Group violations by severity
                    high_sev = [
                        v
                        for v in violations
                        if v.get("severity", "").lower() in ["high", "critical"]
                    ]
                    med_sev = [v for v in violations if v.get("severity", "").lower() == "medium"]
                    low_sev = [
                        v for v in violations if v.get("severity", "").lower() in ["low", "warning"]
                    ]

                    for sev_group, sev_name, sev_icon in [
                        (high_sev, "High Priority", "🔴"),
                        (med_sev, "Medium Priority", "🟡"),
                        (low_sev, "Low Priority", "🟠"),
                    ]:
                        if sev_group:
                            lines.extend([f"**{sev_icon} {sev_name} Issues:**", ""])
                            for viol in sev_group[:3]:  # Limit to top 3 per category
                                viol_type = viol.get("type", "unknown").replace("_", " ").title()
                                evidence = viol.get("evidence", "No details provided")

                                lines.extend(
                                    [
                                        f"- **{viol_type}**",
                                        f"  - *Observation:* {evidence[:100]}{'...' if len(evidence) > 100 else ''}",
                                    ]
                                )
                            lines.append("")

            # Video check details with actionable guidance
            if row.video_check_pass is False:
                lines.extend(
                    [
                        "#### 🎥 Video Generation Issues",
                        "",
                        "**Problem:** Video artifact generation failed",
                        "**Impact:** Visual analysis capabilities reduced",
                        "**Next Steps:**",
                        "- Check rendering pipeline configuration",
                        "- Verify sufficient disk space and memory",
                        "- Review scenario complexity settings",
                        "",
                    ]
                )

            # Enhanced reproduction steps with debugging options
            lines.extend(
                [
                    "#### 🔄 Debugging This Scenario",
                    "",
                    "```bash",
                    "# Quick re-run (same configuration)",
                    f"python -m navirl pipeline --scenario {row.scenario_id}",
                    "",
                    "# Debug run with detailed logging",
                    f"NAVIRL_LOG_LEVEL=DEBUG python -m navirl pipeline --scenario {row.scenario_id} --render",
                    "",
                    "# Interactive exploration",
                    f"cd {Path(row.bundle_dir).parent}",
                    f"python -m navirl interactive --bundle {Path(row.bundle_dir).name}",
                    "```",
                    "",
                    "---",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## 🔄 Comprehensive Reproduction Guide",
            "",
            "### 🎯 Quick Fixes (Most Common Issues)",
            "```bash",
            "# 1. Regenerate scenarios with fresh configuration",
            f"python -m navirl verify --suite {suite} --clean",
            "",
            "# 2. Lower judge sensitivity (if too many false positives)",
            f"python -m navirl verify --suite {suite} --judge-confidence-min 0.4",
            "",
            "# 3. Re-run with full debugging enabled",
            f"NAVIRL_LOG_LEVEL=DEBUG python -m navirl verify --suite {suite} --verbose",
            "```",
            "",
            "### 🔍 Systematic Debugging Workflow",
            "",
            "**Step 1: Isolate the Issue**",
            "```bash",
            "# Test one scenario at a time",
            "python -m navirl pipeline --scenario <failed_scenario> --render --debug",
            "",
            "# Check if it's a test infrastructure issue",
            "pytest tests/ -v -k 'not slow'",
            "```",
            "",
            "**Step 2: Environment Validation**",
            "```bash",
            "# Verify NavIRL installation",
            "python -c \"import navirl; print('NavIRL version:', navirl.__version__)\"",
            "",
            "# Check core dependencies",
            "python -m navirl doctor  # If available",
            "```",
            "",
            "**Step 3: Configuration Tuning**",
            "```bash",
            "# Run with looser thresholds",
            f"python -m navirl verify --suite {suite} \\",
            "  --judge-confidence-min 0.3 \\",
            "  --teleport-thresh-multiplier 1.5",
            "",
            "# Skip video generation (faster iteration)",
            "python -m navirl verify --suite quick",  # Always available
            "```",
            "",
            "### 🛠️ Advanced Troubleshooting",
            "",
            "**Performance Issues:**",
            "```bash",
            "# Reduce simulation complexity",
            "export NAVIRL_MAX_AGENTS=10",
            "export NAVIRL_RENDER_FPS=30",
            f"python -m navirl verify --suite {suite}",
            "```",
            "",
            "**Judge/Visual Issues:**",
            "```bash",
            "# Switch to heuristic judge (no AI required)",
            f"python -m navirl verify --suite {suite} --judge-mode heuristic",
            "",
            "# Use different visual judge provider",
            f"python -m navirl verify --suite {suite} --judge-provider openai",
            "```",
            "",
            "### 📊 Understanding Results",
            "",
            "| Result Type | Next Action | Time Investment |",
            "|---|---|---|",
            "| ✅ All Pass | Ready for production | Continue development |",
            "| ❌ 1-2 Failures | Debug specific scenarios | 15-30 minutes |",
            "| ❌ 3+ Failures | Check configuration/environment | 30-60 minutes |",
            "| ⚠️ Needs Review | Human evaluation required | Variable |",
            "",
            "### 📚 Resources & Support",
            "",
            "- **📖 Documentation:** [NavIRL Verification Guide](docs/verification.md)",
            "- **🧰 Troubleshooting:** [Common Issues & Solutions](docs/troubleshooting.md)",
            "- **🏗️ Scenario Library:** `navirl/scenarios/library/` (reference implementations)",
            "- **⚙️ Config Templates:** `navirl/verify/configs/` (tuning examples)",
            "- **🐛 Issue Tracker:** [Report bugs/unexpected behavior](https://github.com/navirl/issues)",
            "",
            "### 🔧 Configuration Files",
            f"- **Verification config:** `{verify_root}/config.yaml`",
            f"- **Last run logs:** `{verify_root}/verify_{suite}_*/logs/`",
            f"- **Artifacts:** `{verify_root}/verify_{suite}_*/bundle/`",
            "",
            "---",
            "",
            "## 🧪 Test Suite Output",
            "",
            "*Note: Test suite results provide additional context for verification failures.*",
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
        verify_root=verify_root,
    )

    if any(r.judge_status == "needs_human_review" for r in scenario_rows):
        return NEEDS_HUMAN_REVIEW

    if (not pytest_ok) or any(not r.overall_pass for r in scenario_rows):
        return FAIL

    return PASS
