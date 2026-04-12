"""Extended tests for navirl/verify/runner.py.

Covers _run_pytest, run_verify (mocked pipeline), and report edge cases
not covered by the base test_verify_runner.py.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from navirl.verify.runner import (
    FAIL,
    NEEDS_HUMAN_REVIEW,
    PASS,
    VerifyResult,
    _run_pytest,
    _write_report,
    run_verify,
)

# ===================================================================
# _run_pytest
# ===================================================================


class TestRunPytest:
    def test_run_pytest_success_when_subprocess_passes(self):
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="5 passed", stderr=""
        )
        with mock.patch("navirl.verify.runner.subprocess.run", return_value=fake_result):
            ok, output = _run_pytest()
            assert ok is True
            assert "5 passed" in output

    def test_run_pytest_failure_when_subprocess_fails(self):
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="1 failed", stderr="ERRORS"
        )
        with mock.patch("navirl.verify.runner.subprocess.run", return_value=fake_result):
            ok, output = _run_pytest()
            assert ok is False
            assert "1 failed" in output
            assert "ERRORS" in output


# ===================================================================
# run_verify (mocked)
# ===================================================================


def _make_fake_log(bundle_dir: Path):
    """Create a mock log object with bundle_dir attribute."""

    class FakeLog:
        pass

    log = FakeLog()
    log.bundle_dir = str(bundle_dir)
    return log


def _setup_verify_scenario(tmp_path, *, invariants_pass=True, judge_pass=True):
    """Set up mocks for run_verify with configurable outcomes."""
    bundle_dir = tmp_path / "suite" / "verify_suite_test" / "bundle"
    bundle_dir.mkdir(parents=True)

    invariants = {
        "overall_pass": invariants_pass,
        "checks": [
            {"name": "collision_free", "pass": invariants_pass},
        ],
    }
    judge_payload = {
        "overall_pass": judge_pass,
        "status": "pass" if judge_pass else "fail",
        "confidence": 0.9 if judge_pass else 0.2,
    }

    fake_log = _make_fake_log(bundle_dir)
    return fake_log, invariants, judge_payload


class TestRunVerify:
    def test_all_pass_returns_pass(self, tmp_path):
        fake_log, invariants, judge_payload = _setup_verify_scenario(tmp_path)

        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(True, "all passed")),
            mock.patch(
                "navirl.verify.runner.run_scenario_file",
                return_value=fake_log,
            ),
            mock.patch(
                "navirl.verify.runner.run_numeric_invariants",
                return_value=invariants,
            ),
            mock.patch("navirl.verify.runner.build_visual_summary", return_value="summary"),
            mock.patch("navirl.verify.runner.sample_key_frames", return_value=[]),
            mock.patch(
                "navirl.verify.runner.run_visual_judge",
                return_value=judge_payload,
            ),
            mock.patch("navirl.verify.runner.write_judge_output"),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            result = run_verify("quick", tmp_path / "out")
            assert result == PASS

    def test_pytest_fail_returns_fail(self, tmp_path):
        fake_log, invariants, judge_payload = _setup_verify_scenario(tmp_path)

        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(False, "1 failed")),
            mock.patch("navirl.verify.runner.run_scenario_file", return_value=fake_log),
            mock.patch("navirl.verify.runner.run_numeric_invariants", return_value=invariants),
            mock.patch("navirl.verify.runner.build_visual_summary", return_value="summary"),
            mock.patch("navirl.verify.runner.sample_key_frames", return_value=[]),
            mock.patch("navirl.verify.runner.run_visual_judge", return_value=judge_payload),
            mock.patch("navirl.verify.runner.write_judge_output"),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            result = run_verify("quick", tmp_path / "out")
            assert result == FAIL

    def test_scenario_exception_records_failure(self, tmp_path):
        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(True, "ok")),
            mock.patch(
                "navirl.verify.runner.run_scenario_file",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            result = run_verify("quick", tmp_path / "out")
            assert result == FAIL

    def test_needs_human_review_returns_review_code(self, tmp_path):
        fake_log, invariants, _jp = _setup_verify_scenario(tmp_path)
        judge_payload = {
            "overall_pass": True,
            "status": "needs_human_review",
            "confidence": 0.5,
        }

        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(True, "ok")),
            mock.patch("navirl.verify.runner.run_scenario_file", return_value=fake_log),
            mock.patch("navirl.verify.runner.run_numeric_invariants", return_value=invariants),
            mock.patch("navirl.verify.runner.build_visual_summary", return_value="s"),
            mock.patch("navirl.verify.runner.sample_key_frames", return_value=[]),
            mock.patch("navirl.verify.runner.run_visual_judge", return_value=judge_payload),
            mock.patch("navirl.verify.runner.write_judge_output"),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            result = run_verify("quick", tmp_path / "out")
            assert result == NEEDS_HUMAN_REVIEW

    def test_full_suite_checks_video(self, tmp_path):
        fake_log, invariants, judge_payload = _setup_verify_scenario(tmp_path)
        video_check = {"pass": True}

        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(True, "ok")),
            mock.patch("navirl.verify.runner.run_scenario_file", return_value=fake_log),
            mock.patch("navirl.verify.runner.run_numeric_invariants", return_value=invariants),
            mock.patch("navirl.verify.runner.build_visual_summary", return_value="s"),
            mock.patch("navirl.verify.runner.sample_key_frames", return_value=[]),
            mock.patch("navirl.verify.runner.run_visual_judge", return_value=judge_payload),
            mock.patch("navirl.verify.runner.write_judge_output"),
            mock.patch("navirl.verify.runner.check_video_artifact", return_value=video_check),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            result = run_verify("full", tmp_path / "out")
            assert result == PASS

    def test_full_suite_video_fail_returns_fail(self, tmp_path):
        fake_log, invariants, judge_payload = _setup_verify_scenario(tmp_path)
        video_check = {"pass": False}

        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(True, "ok")),
            mock.patch("navirl.verify.runner.run_scenario_file", return_value=fake_log),
            mock.patch("navirl.verify.runner.run_numeric_invariants", return_value=invariants),
            mock.patch("navirl.verify.runner.build_visual_summary", return_value="s"),
            mock.patch("navirl.verify.runner.sample_key_frames", return_value=[]),
            mock.patch("navirl.verify.runner.run_visual_judge", return_value=judge_payload),
            mock.patch("navirl.verify.runner.write_judge_output"),
            mock.patch("navirl.verify.runner.check_video_artifact", return_value=video_check),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            result = run_verify("full", tmp_path / "out")
            assert result == FAIL

    def test_creates_report_file(self, tmp_path):
        fake_log, invariants, judge_payload = _setup_verify_scenario(tmp_path)

        with (
            mock.patch("navirl.verify.runner._run_pytest", return_value=(True, "ok")),
            mock.patch("navirl.verify.runner.run_scenario_file", return_value=fake_log),
            mock.patch("navirl.verify.runner.run_numeric_invariants", return_value=invariants),
            mock.patch("navirl.verify.runner.build_visual_summary", return_value="s"),
            mock.patch("navirl.verify.runner.sample_key_frames", return_value=[]),
            mock.patch("navirl.verify.runner.run_visual_judge", return_value=judge_payload),
            mock.patch("navirl.verify.runner.write_judge_output"),
            mock.patch("navirl.verify.runner.prune_old_run_dirs"),
            mock.patch("navirl.verify.runner.resolve_retention_hours", return_value=168.0),
        ):
            run_verify("quick", tmp_path / "out")
            report = tmp_path / "out" / "quick" / "REPORT.md"
            assert report.exists()
            content = report.read_text()
            assert "NavIRL Verification Report" in content


# ===================================================================
# _write_report edge cases
# ===================================================================


class TestWriteReportEdgeCases:
    def test_report_with_feasibility_suggestions(self, tmp_path):
        """Feasibility suggestions appear in notes."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        invariants_data = {
            "overall_pass": False,
            "checks": [
                {
                    "name": "scenario_feasibility",
                    "pass": False,
                    "severity": "critical",
                    "suggestions": ["Reduce agent count", "Widen corridors"],
                },
            ],
        }
        (bundle_dir / "invariants.json").write_text(json.dumps(invariants_data))

        rows = [
            VerifyResult(
                scenario_id="feasibility_test",
                bundle_dir=str(bundle_dir),
                invariants_pass=False,
                judge_status="fail",
                judge_confidence=0.3,
                overall_pass=False,
                notes="failed=scenario_feasibility; fix=Reduce agent count",
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
        assert "Reduce agent count" in content
        assert "Immediate Actions" in content

    def test_report_multiple_failure_categories(self, tmp_path):
        """Failures are categorized by priority (critical, visual, review)."""
        rows = [
            VerifyResult(
                scenario_id="critical_fail",
                bundle_dir=str(tmp_path),
                invariants_pass=False,
                judge_status="fail",
                judge_confidence=0.1,
                overall_pass=False,
            ),
            VerifyResult(
                scenario_id="visual_only",
                bundle_dir=str(tmp_path),
                invariants_pass=True,
                judge_status="fail",
                judge_confidence=0.3,
                overall_pass=False,
            ),
            VerifyResult(
                scenario_id="review_only",
                bundle_dir=str(tmp_path),
                invariants_pass=True,
                judge_status="needs_human_review",
                judge_confidence=0.5,
                overall_pass=False,
            ),
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
        assert "Critical" in content
        assert "Visual" in content
        assert "Review" in content
        # Critical failures listed first
        critical_pos = content.index("critical_fail")
        visual_pos = content.index("visual_only")
        review_pos = content.index("review_only")
        assert critical_pos < visual_pos < review_pos

    def test_report_confidence_descriptions(self, tmp_path):
        """Judge confidence descriptions in failure analysis."""
        bundle_dir = tmp_path / "bundle_conf"
        bundle_dir.mkdir()
        judge_data = {
            "status": "fail",
            "confidence": 0.15,
            "violations": [
                {"type": "test_issue", "severity": "high", "evidence": "test evidence"},
            ],
        }
        (bundle_dir / "judge.json").write_text(json.dumps(judge_data))

        rows = [
            VerifyResult(
                scenario_id="low_conf",
                bundle_dir=str(bundle_dir),
                invariants_pass=True,
                judge_status="fail",
                judge_confidence=0.15,
                overall_pass=False,
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
        assert "Very Low" in content
