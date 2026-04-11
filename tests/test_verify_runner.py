"""Tests for navirl/verify/runner.py helper and formatting functions.

Covers VerifyResult, _calculate_verification_stats, _format_executive_summary,
_format_configuration_section, _format_scenario_row, _format_scenario_results_table,
_write_report, and _scenario_file_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navirl.verify.runner import (
    CANONICAL_SCENARIOS,
    FAIL,
    NEEDS_HUMAN_REVIEW,
    PASS,
    VerifyResult,
    _calculate_verification_stats,
    _format_configuration_section,
    _format_executive_summary,
    _format_scenario_results_table,
    _format_scenario_row,
    _scenario_file_path,
    _write_report,
)

# ===================================================================
# Fixtures
# ===================================================================


def _make_result(
    scenario_id: str = "test_scenario",
    overall_pass: bool = True,
    invariants_pass: bool = True,
    judge_status: str = "pass",
    judge_confidence: float = 0.85,
    video_check_pass: bool | None = None,
    notes: str = "",
    bundle_dir: str = "/tmp/fake_bundle",
) -> VerifyResult:
    return VerifyResult(
        scenario_id=scenario_id,
        bundle_dir=bundle_dir,
        invariants_pass=invariants_pass,
        judge_status=judge_status,
        judge_confidence=judge_confidence,
        overall_pass=overall_pass,
        video_check_pass=video_check_pass,
        notes=notes,
    )


@pytest.fixture
def all_pass_rows():
    return [
        _make_result("scenario_a", overall_pass=True),
        _make_result("scenario_b", overall_pass=True),
        _make_result("scenario_c", overall_pass=True),
    ]


@pytest.fixture
def mixed_rows():
    return [
        _make_result("pass_scenario", overall_pass=True),
        _make_result(
            "fail_invariant",
            overall_pass=False,
            invariants_pass=False,
            judge_status="fail",
            judge_confidence=0.2,
        ),
        _make_result(
            "needs_review",
            overall_pass=False,
            judge_status="needs_human_review",
            judge_confidence=0.45,
        ),
    ]


# ===================================================================
# Constants
# ===================================================================


class TestConstants:
    def test_exit_codes(self):
        assert PASS == 0
        assert FAIL == 10
        assert NEEDS_HUMAN_REVIEW == 20

    def test_canonical_scenarios_non_empty(self):
        assert len(CANONICAL_SCENARIOS) > 0
        for name in CANONICAL_SCENARIOS:
            assert name.endswith(".yaml")


# ===================================================================
# VerifyResult
# ===================================================================


class TestVerifyResult:
    def test_defaults(self):
        r = VerifyResult(
            scenario_id="s1",
            bundle_dir="/tmp/b",
            invariants_pass=True,
            judge_status="pass",
            judge_confidence=0.9,
            overall_pass=True,
        )
        assert r.video_check_pass is None
        assert r.notes == ""

    def test_all_fields(self):
        r = _make_result(
            video_check_pass=True,
            notes="some note",
        )
        assert r.scenario_id == "test_scenario"
        assert r.video_check_pass is True
        assert r.notes == "some note"


# ===================================================================
# _calculate_verification_stats
# ===================================================================


class TestCalculateVerificationStats:
    def test_all_passing(self, all_pass_rows):
        stats = _calculate_verification_stats(all_pass_rows)
        assert stats["total_scenarios"] == 3
        assert stats["passed_scenarios"] == 3
        assert stats["failed_scenarios"] == 0
        assert stats["success_rate"] == pytest.approx(100.0)
        assert stats["needs_review"] == 0
        assert stats["invariant_failures"] == 0
        assert stats["judge_failures"] == 0

    def test_mixed_results(self, mixed_rows):
        stats = _calculate_verification_stats(mixed_rows)
        assert stats["total_scenarios"] == 3
        assert stats["passed_scenarios"] == 1
        assert stats["failed_scenarios"] == 2
        assert stats["success_rate"] == pytest.approx(100 / 3)
        assert stats["needs_review"] == 1
        assert stats["invariant_failures"] == 1
        assert stats["judge_failures"] == 1

    def test_empty_rows(self):
        stats = _calculate_verification_stats([])
        assert stats["total_scenarios"] == 0
        assert stats["success_rate"] == 0


# ===================================================================
# _format_executive_summary
# ===================================================================


class TestFormatExecutiveSummary:
    def test_all_pass_with_pytest_ok(self, all_pass_rows):
        stats = _calculate_verification_stats(all_pass_rows)
        lines = _format_executive_summary("quick", True, stats)
        text = "\n".join(lines)
        assert "PASS" in text
        assert "All scenarios passed" in text

    def test_all_pass_pytest_fail(self, all_pass_rows):
        stats = _calculate_verification_stats(all_pass_rows)
        lines = _format_executive_summary("quick", False, stats)
        text = "\n".join(lines)
        assert "test suite failed" in text.lower() or "FAIL" in text

    def test_scenario_failures(self, mixed_rows):
        stats = _calculate_verification_stats(mixed_rows)
        lines = _format_executive_summary("full", True, stats)
        text = "\n".join(lines)
        assert "2 scenario(s) need attention" in text
        assert "invariant violation" in text.lower()

    def test_suite_name_in_title(self):
        stats = _calculate_verification_stats([_make_result()])
        lines = _format_executive_summary("smoke", True, stats)
        text = "\n".join(lines)
        assert "Smoke" in text


# ===================================================================
# _format_configuration_section
# ===================================================================


class TestFormatConfigurationSection:
    def test_contains_threshold(self):
        lines = _format_configuration_section("quick", {"judge_confidence_min": 0.6})
        text = "\n".join(lines)
        assert "0.6" in text
        assert "quick" in text

    def test_full_suite_mentions_artifacts(self):
        lines = _format_configuration_section("full", {"judge_confidence_min": 0.5})
        text = "\n".join(lines)
        assert "full artifacts" in text or "full" in text


# ===================================================================
# _format_scenario_row
# ===================================================================


class TestFormatScenarioRow:
    def test_passing_row(self):
        row = _make_result(judge_confidence=0.9)
        text = _format_scenario_row(row)
        assert "test_scenario" in text
        assert "Pass" in text
        assert "0.90" in text

    def test_failing_row_with_invariant_fail(self):
        row = _make_result(invariants_pass=False, judge_status="fail", judge_confidence=0.15)
        text = _format_scenario_row(row)
        assert "Fail" in text
        assert "0.15" in text

    def test_confidence_indicators(self):
        # Low confidence
        row_low = _make_result(judge_confidence=0.2)
        assert "🔴" in _format_scenario_row(row_low)

        # Medium confidence
        row_med = _make_result(judge_confidence=0.5)
        assert "🟡" in _format_scenario_row(row_med)

        # High confidence
        row_high = _make_result(judge_confidence=0.9)
        assert "🟢" in _format_scenario_row(row_high)

    def test_video_status_none(self):
        row = _make_result(video_check_pass=None)
        text = _format_scenario_row(row)
        assert "- |" in text or "-" in text

    def test_video_pass_and_fail(self):
        row_pass = _make_result(video_check_pass=True)
        assert "Pass" in _format_scenario_row(row_pass)

        row_fail = _make_result(video_check_pass=False)
        assert "Fail" in _format_scenario_row(row_fail)

    def test_notes_with_failed_prefix(self):
        row = _make_result(notes="failed=collision_free,safety_constraints; fix=adjust radius")
        text = _format_scenario_row(row)
        assert "Failed:" in text

    def test_notes_with_fix_prefix(self):
        row = _make_result(notes="fix=increase agent radius to avoid collisions")
        text = _format_scenario_row(row)
        assert "Fix:" in text

    def test_long_notes_truncated(self):
        row = _make_result(notes="x" * 100)
        text = _format_scenario_row(row)
        assert "..." in text

    def test_empty_notes(self):
        row = _make_result(notes="")
        text = _format_scenario_row(row)
        assert "No issues" in text

    def test_needs_human_review_status(self):
        row = _make_result(judge_status="needs_human_review")
        text = _format_scenario_row(row)
        assert "⚠️" in text
        assert "Needs Human Review" in text


# ===================================================================
# _format_scenario_results_table
# ===================================================================


class TestFormatScenarioResultsTable:
    def test_table_header(self, all_pass_rows):
        lines = _format_scenario_results_table(all_pass_rows)
        text = "\n".join(lines)
        assert "Scenario" in text
        assert "Invariants" in text
        assert "Judge" in text

    def test_table_has_rows(self, all_pass_rows):
        lines = _format_scenario_results_table(all_pass_rows)
        # Header lines + separator + one row per scenario
        row_lines = [line for line in lines if line.startswith("| **")]
        assert len(row_lines) == 3


# ===================================================================
# _write_report
# ===================================================================


class TestWriteReport:
    def test_writes_markdown_file(self, tmp_path, all_pass_rows):
        report_path = tmp_path / "reports" / "REPORT.md"
        _write_report(
            report_path=report_path,
            suite="quick",
            pytest_ok=True,
            pytest_out="3 passed",
            rows=all_pass_rows,
            thresholds={"judge_confidence_min": 0.6},
            verify_root=tmp_path,
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "Quick" in content
        assert "3 passed" in content

    def test_report_with_failures(self, tmp_path, mixed_rows):
        report_path = tmp_path / "REPORT.md"
        _write_report(
            report_path=report_path,
            suite="full",
            pytest_ok=False,
            pytest_out="1 failed",
            rows=mixed_rows,
            thresholds={"judge_confidence_min": 0.6},
            verify_root=tmp_path,
        )
        content = report_path.read_text()
        assert "Failure Analysis" in content
        assert "fail_invariant" in content
        assert "needs_review" in content

    def test_report_no_failures(self, tmp_path, all_pass_rows):
        report_path = tmp_path / "REPORT.md"
        _write_report(
            report_path=report_path,
            suite="smoke",
            pytest_ok=True,
            pytest_out="all passed",
            rows=all_pass_rows,
            thresholds={"judge_confidence_min": 0.5},
            verify_root=tmp_path,
        )
        content = report_path.read_text()
        assert "All scenarios passed" in content

    def test_report_with_invariants_json(self, tmp_path):
        """Test report generation when invariants.json exists for a failed scenario."""
        bundle_dir = tmp_path / "bundle_fail"
        bundle_dir.mkdir()
        invariants_data = {
            "overall_pass": False,
            "checks": [
                {
                    "name": "collision_free",
                    "pass": False,
                    "severity": "critical",
                    "num_violations": 5,
                },
                {
                    "name": "speed_limit",
                    "pass": False,
                    "severity": "low",
                    "message": "minor overshoot",
                },
                {"name": "goal_reached", "pass": True},
            ],
        }
        (bundle_dir / "invariants.json").write_text(json.dumps(invariants_data))

        rows = [
            _make_result(
                scenario_id="inv_fail",
                overall_pass=False,
                invariants_pass=False,
                bundle_dir=str(bundle_dir),
            )
        ]
        report_path = tmp_path / "REPORT.md"
        _write_report(
            report_path=report_path,
            suite="quick",
            pytest_ok=True,
            pytest_out="ok",
            rows=rows,
            thresholds={"judge_confidence_min": 0.6},
            verify_root=tmp_path,
        )
        content = report_path.read_text()
        assert "Invariant Analysis" in content
        assert "Collision Free" in content
        assert "Speed Limit" in content

    def test_report_with_judge_json(self, tmp_path):
        """Test report generation when judge.json exists with violations."""
        bundle_dir = tmp_path / "bundle_judge"
        bundle_dir.mkdir()
        judge_data = {
            "status": "fail",
            "confidence": 0.4,
            "violations": [
                {
                    "type": "unsafe_proximity",
                    "severity": "high",
                    "evidence": "Robot too close to human at step 42",
                },
                {
                    "type": "odd_trajectory",
                    "severity": "medium",
                    "evidence": "Zigzag pattern detected",
                },
                {
                    "type": "minor_wobble",
                    "severity": "low",
                    "evidence": "Slight oscillation observed",
                },
            ],
        }
        (bundle_dir / "judge.json").write_text(json.dumps(judge_data))

        rows = [
            _make_result(
                scenario_id="judge_fail",
                overall_pass=False,
                judge_status="fail",
                judge_confidence=0.4,
                bundle_dir=str(bundle_dir),
            )
        ]
        report_path = tmp_path / "REPORT.md"
        _write_report(
            report_path=report_path,
            suite="quick",
            pytest_ok=True,
            pytest_out="ok",
            rows=rows,
            thresholds={"judge_confidence_min": 0.6},
            verify_root=tmp_path,
        )
        content = report_path.read_text()
        assert "Visual Behavior Analysis" in content
        assert "Unsafe Proximity" in content
        assert "High Priority" in content

    def test_report_video_check_fail(self, tmp_path):
        rows = [
            _make_result(
                scenario_id="vid_fail",
                overall_pass=False,
                invariants_pass=False,
                video_check_pass=False,
                bundle_dir=str(tmp_path),
            )
        ]
        report_path = tmp_path / "REPORT.md"
        _write_report(
            report_path=report_path,
            suite="full",
            pytest_ok=True,
            pytest_out="ok",
            rows=rows,
            thresholds={"judge_confidence_min": 0.6},
            verify_root=tmp_path,
        )
        content = report_path.read_text()
        assert "Video Generation Issues" in content


# ===================================================================
# _scenario_file_path
# ===================================================================


class TestScenarioFilePath:
    def test_returns_path_under_scenarios_library(self):
        p = _scenario_file_path("hallway_pass.yaml")
        assert p.name == "hallway_pass.yaml"
        assert "scenarios" in str(p)
        assert "library" in str(p)
