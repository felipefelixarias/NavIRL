"""Tests for navirl.repro.checklist — publication readiness verification."""

from __future__ import annotations

import json

import pytest

from navirl.repro.checklist import ChecklistReport, CheckResult, run_checklist


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_passed_result(self):
        r = CheckResult(name="foo", passed=True, message="ok")
        assert r.name == "foo"
        assert r.passed is True
        assert r.message == "ok"

    def test_failed_result(self):
        r = CheckResult(name="bar", passed=False, message="missing")
        assert r.passed is False


# ---------------------------------------------------------------------------
# ChecklistReport
# ---------------------------------------------------------------------------


class TestChecklistReport:
    def test_empty_report_passes(self):
        report = ChecklistReport(package_name="pkg")
        assert report.passed is True
        assert report.total == 0
        assert report.passed_count == 0

    def test_all_pass(self):
        report = ChecklistReport(
            package_name="pkg",
            results=[
                CheckResult("a", True, "ok"),
                CheckResult("b", True, "ok"),
            ],
        )
        assert report.passed is True
        assert report.total == 2
        assert report.passed_count == 2

    def test_one_failure_means_not_passed(self):
        report = ChecklistReport(
            package_name="pkg",
            results=[
                CheckResult("a", True, "ok"),
                CheckResult("b", False, "missing"),
            ],
        )
        assert report.passed is False
        assert report.total == 2
        assert report.passed_count == 1

    def test_to_dict(self):
        report = ChecklistReport(
            package_name="test_pkg",
            results=[CheckResult("a", True, "fine")],
        )
        d = report.to_dict()
        assert d["package_name"] == "test_pkg"
        assert d["passed"] is True
        assert d["total_checks"] == 1
        assert d["passed_checks"] == 1
        assert len(d["results"]) == 1
        assert d["results"][0]["name"] == "a"

    def test_to_markdown_pass(self):
        report = ChecklistReport(
            package_name="test_pkg",
            results=[CheckResult("check_one", True, "all good")],
        )
        md = report.to_markdown()
        assert "test_pkg" in md
        assert "PASS" in md
        assert "1/1" in md
        assert "check_one" in md

    def test_to_markdown_fail(self):
        report = ChecklistReport(
            package_name="pkg",
            results=[CheckResult("x", False, "bad")],
        )
        md = report.to_markdown()
        assert "FAIL" in md
        assert "0/1" in md


# ---------------------------------------------------------------------------
# run_checklist — no MANIFEST.json
# ---------------------------------------------------------------------------


class TestRunChecklistNoManifest:
    def test_missing_manifest(self, tmp_path):
        report = run_checklist(tmp_path)
        assert report.passed is False
        assert any("MANIFEST.json" in r.message for r in report.results)


# ---------------------------------------------------------------------------
# run_checklist — minimal manifest
# ---------------------------------------------------------------------------


class TestRunChecklistMinimal:
    def _write_manifest(self, tmp_path, data):
        (tmp_path / "MANIFEST.json").write_text(json.dumps(data), encoding="utf-8")

    def test_empty_manifest_fails_most_checks(self, tmp_path):
        self._write_manifest(tmp_path, {})
        report = run_checklist(tmp_path)
        assert report.passed is False
        # Should have multiple failing checks
        failed = [r for r in report.results if not r.passed]
        assert len(failed) >= 5

    def test_manifest_exists_check_passes(self, tmp_path):
        self._write_manifest(tmp_path, {})
        report = run_checklist(tmp_path)
        manifest_check = next(r for r in report.results if r.name == "manifest_exists")
        assert manifest_check.passed is True

    def test_identity_requires_name_and_version(self, tmp_path):
        self._write_manifest(tmp_path, {"name": "pkg", "version": "1.0"})
        report = run_checklist(tmp_path)
        identity = next(r for r in report.results if r.name == "identity")
        assert identity.passed is True

    def test_identity_fails_without_version(self, tmp_path):
        self._write_manifest(tmp_path, {"name": "pkg"})
        report = run_checklist(tmp_path)
        identity = next(r for r in report.results if r.name == "identity")
        assert identity.passed is False

    def test_environment_pins(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {
                "environment": {
                    "python_version": "3.12.0",
                    "platform_system": "Linux",
                    "packages": {"numpy": "1.26"},
                }
            },
        )
        report = run_checklist(tmp_path)
        env_check = next(r for r in report.results if r.name == "environment_pins")
        assert env_check.passed is True

    def test_environment_pins_fail_without_python(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {"environment": {"platform_system": "Linux"}},
        )
        report = run_checklist(tmp_path)
        env_check = next(r for r in report.results if r.name == "environment_pins")
        assert env_check.passed is False

    def test_scenarios_included(self, tmp_path):
        self._write_manifest(tmp_path, {})
        (tmp_path / "scenarios").mkdir()
        (tmp_path / "scenarios" / "test.yaml").write_text("scenario: true")
        report = run_checklist(tmp_path)
        sc = next(r for r in report.results if r.name == "scenarios_included")
        assert sc.passed is True

    def test_scenarios_missing(self, tmp_path):
        self._write_manifest(tmp_path, {})
        report = run_checklist(tmp_path)
        sc = next(r for r in report.results if r.name == "scenarios_included")
        assert sc.passed is False

    def test_results_present(self, tmp_path):
        self._write_manifest(tmp_path, {})
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "run1.json").write_text("{}")
        report = run_checklist(tmp_path)
        res = next(r for r in report.results if r.name == "results_present")
        assert res.passed is True

    def test_artifact_checksums(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {"artifacts": [{"name": "model.pt", "sha256": "abc123"}]},
        )
        report = run_checklist(tmp_path)
        art = next(r for r in report.results if r.name == "artifact_checksums")
        assert art.passed is True

    def test_artifact_checksums_fail_missing_hash(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {"artifacts": [{"name": "model.pt"}]},
        )
        report = run_checklist(tmp_path)
        art = next(r for r in report.results if r.name == "artifact_checksums")
        assert art.passed is False

    def test_expected_metrics(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {"expected_metrics": {"success_rate": 0.95}},
        )
        report = run_checklist(tmp_path)
        met = next(r for r in report.results if r.name == "expected_metrics")
        assert met.passed is True

    def test_description_check(self, tmp_path):
        self._write_manifest(tmp_path, {"description": "A test package"})
        report = run_checklist(tmp_path)
        desc = next(r for r in report.results if r.name == "description")
        assert desc.passed is True

    def test_package_pins(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {"environment": {"packages": {"numpy": "1.26"}}},
        )
        report = run_checklist(tmp_path)
        pins = next(r for r in report.results if r.name == "package_pins")
        assert pins.passed is True

    def test_package_checksum(self, tmp_path):
        self._write_manifest(
            tmp_path,
            {"checksum": "sha256:abcdef1234567890"},
        )
        report = run_checklist(tmp_path)
        cs = next(r for r in report.results if r.name == "package_checksum")
        assert cs.passed is True


# ---------------------------------------------------------------------------
# run_checklist — fully passing package
# ---------------------------------------------------------------------------


class TestRunChecklistFullPass:
    def test_complete_package_passes_all_checks(self, tmp_path):
        manifest = {
            "name": "my_study",
            "version": "1.0.0",
            "description": "Reproducibility package for hallway navigation study",
            "environment": {
                "python_version": "3.12.0",
                "platform_system": "Linux",
                "packages": {"numpy": "1.26.4", "navirl": "0.1.0"},
            },
            "artifacts": [
                {"name": "model.pt", "sha256": "abc123def456"},
                {"name": "data.h5", "sha256": "789xyz"},
            ],
            "expected_metrics": {
                "success_rate": 0.95,
                "avg_time_to_goal": 12.5,
            },
            "checksum": "sha256:packagelevelchecksum",
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        (tmp_path / "scenarios").mkdir()
        (tmp_path / "scenarios" / "hallway.yaml").write_text("scenario: hallway")
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "run_001.json").write_text("{}")

        report = run_checklist(tmp_path)
        assert report.passed is True
        assert report.passed_count == report.total
        assert report.total == 10  # all 10 checks
