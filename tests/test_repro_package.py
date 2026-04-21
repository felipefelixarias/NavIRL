"""Tests for the reproducibility package module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navirl.repro import (
    ArtifactEntry,
    ChecklistReport,
    CheckResult,
    ComplianceFinding,
    ComplianceReport,
    EnvironmentPin,
    MetricComparison,
    ReplayReport,
    ReplayResult,
    ReproPackage,
    build_repro_package,
    run_checklist,
    scan_compliance,
    verify_repro_package,
)

# ---------------------------------------------------------------------------
# EnvironmentPin
# ---------------------------------------------------------------------------


class TestEnvironmentPin:
    def test_capture_returns_populated_pin(self):
        pin = EnvironmentPin.capture()
        assert pin.python_version
        assert pin.platform_system
        assert pin.platform_machine

    def test_to_dict_roundtrip(self):
        pin = EnvironmentPin(
            python_version="3.11.0",
            platform_system="Linux",
            platform_machine="x86_64",
            platform_release="6.0",
            packages={"numpy": "1.25.0", "navirl": "0.1.0"},
        )
        data = pin.to_dict()
        restored = EnvironmentPin.from_dict(data)
        assert restored.python_version == pin.python_version
        assert restored.packages == pin.packages

    def test_from_dict_defaults(self):
        pin = EnvironmentPin.from_dict({})
        assert pin.python_version == ""
        assert pin.packages == {}

    def test_capture_includes_packages(self):
        pin = EnvironmentPin.capture()
        # At minimum, pytest should be installed
        assert isinstance(pin.packages, dict)
        assert len(pin.packages) > 0


# ---------------------------------------------------------------------------
# ArtifactEntry
# ---------------------------------------------------------------------------


class TestArtifactEntry:
    def test_to_dict(self):
        entry = ArtifactEntry(relative_path="scenarios/a.yaml", sha256="abc123", size_bytes=100)
        d = entry.to_dict()
        assert d["relative_path"] == "scenarios/a.yaml"
        assert d["sha256"] == "abc123"
        assert d["size_bytes"] == 100

    def test_from_dict(self):
        d = {"relative_path": "results/x.json", "sha256": "def456", "size_bytes": 200}
        entry = ArtifactEntry.from_dict(d)
        assert entry.relative_path == "results/x.json"
        assert entry.sha256 == "def456"


# ---------------------------------------------------------------------------
# ReproPackage
# ---------------------------------------------------------------------------


class TestReproPackage:
    def test_checksum_deterministic(self):
        pkg = ReproPackage(name="test", version="1.0")
        c1 = pkg.checksum()
        c2 = pkg.checksum()
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_checksum_changes_with_content(self):
        pkg1 = ReproPackage(name="a", version="1.0")
        pkg2 = ReproPackage(name="b", version="1.0")
        assert pkg1.checksum() != pkg2.checksum()

    def test_to_dict_includes_all_fields(self):
        pkg = ReproPackage(
            name="study-x",
            version="2.0",
            description="A test study",
            created_at="2024-01-01T00:00:00",
            metadata={"author": "tester"},
        )
        d = pkg.to_dict()
        assert d["name"] == "study-x"
        assert d["version"] == "2.0"
        assert d["description"] == "A test study"
        assert d["checksum"]
        assert d["metadata"]["author"] == "tester"

    def test_from_dict_roundtrip(self):
        pkg = ReproPackage(
            name="rt",
            version="1.0",
            description="roundtrip test",
            expected_metrics={"success_rate": {"mean": 0.9, "std": 0.05}},
        )
        data = pkg.to_dict()
        restored = ReproPackage.from_dict(data)
        assert restored.name == pkg.name
        assert restored.expected_metrics == pkg.expected_metrics

    def test_from_dict_defaults(self):
        pkg = ReproPackage.from_dict({"name": "minimal"})
        assert pkg.version == "1.0"
        assert pkg.artifacts == []


# ---------------------------------------------------------------------------
# build_repro_package
# ---------------------------------------------------------------------------


class TestBuildReproPackage:
    def test_build_from_run_dir(self, tmp_path: Path):
        # Create fake run directory structure
        run_dir = tmp_path / "runs"
        run1 = run_dir / "run_001"
        run1.mkdir(parents=True)
        (run1 / "scenario.yaml").write_text("grid: {rows: 10, cols: 10}\n")
        (run1 / "summary.json").write_text('{"success_rate": 1.0}\n')

        out = tmp_path / "package"
        pkg = build_repro_package(
            name="test-pkg",
            run_dir=run_dir,
            out_dir=out,
            description="test build",
        )

        assert pkg.name == "test-pkg"
        assert len(pkg.artifacts) > 0
        assert (out / "MANIFEST.json").is_file()
        assert (out / "scenarios").is_dir()
        assert (out / "results").is_dir()

    def test_build_with_explicit_scenarios(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir()

        scenario = tmp_path / "my_scenario.yaml"
        scenario.write_text("grid: {rows: 5, cols: 5}\n")

        out = tmp_path / "package"
        pkg = build_repro_package(
            name="explicit-scenarios",
            run_dir=run_dir,
            scenario_paths=[scenario],
            out_dir=out,
        )

        assert (out / "scenarios" / "my_scenario.yaml").is_file()
        # Find the scenario in artifacts
        scenario_artifacts = [a for a in pkg.artifacts if "my_scenario" in a.relative_path]
        assert len(scenario_artifacts) == 1

    def test_build_with_pack_results(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir()

        pack_results = tmp_path / "pack_results.json"
        pack_data = {
            "manifest_name": "test",
            "manifest_version": "1.0",
            "manifest_checksum": "abc",
            "timestamp": "2024-01-01",
            "runs": [
                {
                    "entry_id": "s1",
                    "seed": 42,
                    "metrics": {"success_rate": 1.0, "collisions": 0},
                    "status": "completed",
                    "error": None,
                },
                {
                    "entry_id": "s1",
                    "seed": 43,
                    "metrics": {"success_rate": 0.8, "collisions": 1},
                    "status": "completed",
                    "error": None,
                },
            ],
        }
        pack_results.write_text(json.dumps(pack_data))

        out = tmp_path / "package"
        pkg = build_repro_package(
            name="with-metrics",
            run_dir=run_dir,
            pack_result_path=pack_results,
            out_dir=out,
        )

        assert len(pkg.expected_metrics) > 0
        assert "success_rate" in pkg.expected_metrics

    def test_build_metadata(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir()

        out = tmp_path / "package"
        pkg = build_repro_package(
            name="meta-test",
            run_dir=run_dir,
            out_dir=out,
            metadata={"author": "Alice", "study": "hallway-2024"},
        )

        assert pkg.metadata["author"] == "Alice"
        assert pkg.metadata["study"] == "hallway-2024"

    def test_build_empty_run_dir(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir()

        out = tmp_path / "package"
        pkg = build_repro_package(name="empty", run_dir=run_dir, out_dir=out)

        assert pkg.name == "empty"
        assert (out / "MANIFEST.json").is_file()

    def test_manifest_json_is_valid(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run_dir.mkdir()

        out = tmp_path / "package"
        build_repro_package(name="valid-json", run_dir=run_dir, out_dir=out)

        with (out / "MANIFEST.json").open() as f:
            data = json.load(f)

        assert data["name"] == "valid-json"
        assert "checksum" in data
        assert "environment" in data


# ---------------------------------------------------------------------------
# verify_repro_package
# ---------------------------------------------------------------------------


class TestVerifyReproPackage:
    def test_verify_valid_package(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run1 = run_dir / "run_001"
        run1.mkdir(parents=True)
        (run1 / "scenario.yaml").write_text("grid: {rows: 10}\n")
        (run1 / "summary.json").write_text('{"ok": true}\n')

        out = tmp_path / "package"
        build_repro_package(name="verify-ok", run_dir=run_dir, out_dir=out)

        ok, issues = verify_repro_package(out)
        assert ok
        assert issues == []

    def test_verify_missing_manifest(self, tmp_path: Path):
        ok, issues = verify_repro_package(tmp_path)
        assert not ok
        assert "MANIFEST.json not found" in issues[0]

    def test_verify_detects_missing_artifact(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run1 = run_dir / "run_001"
        run1.mkdir(parents=True)
        (run1 / "scenario.yaml").write_text("data\n")
        (run1 / "summary.json").write_text("{}\n")

        out = tmp_path / "package"
        build_repro_package(name="missing-art", run_dir=run_dir, out_dir=out)

        # Delete an artifact
        scenarios = list((out / "scenarios").glob("*.yaml"))
        assert len(scenarios) > 0
        scenarios[0].unlink()

        ok, issues = verify_repro_package(out)
        assert not ok
        assert any("Missing artifact" in i for i in issues)

    def test_verify_detects_checksum_mismatch(self, tmp_path: Path):
        run_dir = tmp_path / "runs"
        run1 = run_dir / "run_001"
        run1.mkdir(parents=True)
        (run1 / "scenario.yaml").write_text("original\n")
        (run1 / "summary.json").write_text("{}\n")

        out = tmp_path / "package"
        build_repro_package(name="tampered", run_dir=run_dir, out_dir=out)

        # Tamper with a file
        scenarios = list((out / "scenarios").glob("*.yaml"))
        assert len(scenarios) > 0
        scenarios[0].write_text("tampered content\n")

        ok, issues = verify_repro_package(out)
        assert not ok
        assert any("Checksum mismatch" in i for i in issues)


# ---------------------------------------------------------------------------
# CheckResult / ChecklistReport
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_basic_fields(self):
        r = CheckResult("test_check", True, "looks good")
        assert r.name == "test_check"
        assert r.passed
        assert r.message == "looks good"


class TestChecklistReport:
    def test_all_passed(self):
        report = ChecklistReport(
            package_name="test",
            results=[
                CheckResult("a", True, "ok"),
                CheckResult("b", True, "ok"),
            ],
        )
        assert report.passed
        assert report.total == 2
        assert report.passed_count == 2

    def test_some_failed(self):
        report = ChecklistReport(
            package_name="test",
            results=[
                CheckResult("a", True, "ok"),
                CheckResult("b", False, "missing"),
            ],
        )
        assert not report.passed
        assert report.passed_count == 1

    def test_to_dict(self):
        report = ChecklistReport(
            package_name="pkg",
            results=[CheckResult("x", True, "fine")],
        )
        d = report.to_dict()
        assert d["package_name"] == "pkg"
        assert d["passed"] is True
        assert d["total_checks"] == 1

    def test_to_markdown(self):
        report = ChecklistReport(
            package_name="study",
            results=[
                CheckResult("check_a", True, "passed"),
                CheckResult("check_b", False, "failed"),
            ],
        )
        md = report.to_markdown()
        assert "study" in md
        assert "PASS" in md
        assert "FAIL" in md
        assert "check_a" in md

    def test_empty_report(self):
        report = ChecklistReport(package_name="empty")
        assert report.passed  # vacuously true
        assert report.total == 0


# ---------------------------------------------------------------------------
# run_checklist
# ---------------------------------------------------------------------------


class TestRunChecklist:
    def test_full_checklist_on_complete_package(self, tmp_path: Path):
        """Build a complete package and verify checklist passes."""
        run_dir = tmp_path / "runs"
        run1 = run_dir / "run_001"
        run1.mkdir(parents=True)
        (run1 / "scenario.yaml").write_text("grid: {rows: 10}\n")
        (run1 / "summary.json").write_text('{"success_rate": 0.95}\n')

        # Create pack results for expected metrics
        pack_results = tmp_path / "pack_results.json"
        pack_data = {
            "manifest_name": "test",
            "manifest_version": "1.0",
            "manifest_checksum": "abc",
            "timestamp": "2024-01-01",
            "runs": [
                {
                    "entry_id": "s1",
                    "seed": 42,
                    "metrics": {"success_rate": 0.95},
                    "status": "completed",
                    "error": None,
                },
            ],
        }
        pack_results.write_text(json.dumps(pack_data))

        out = tmp_path / "package"
        build_repro_package(
            name="complete-pkg",
            run_dir=run_dir,
            pack_result_path=pack_results,
            out_dir=out,
            description="A complete study",
        )

        report = run_checklist(out)
        assert report.passed
        assert report.passed_count == report.total

    def test_checklist_on_missing_dir(self, tmp_path: Path):
        pkg_dir = tmp_path / "missing"
        pkg_dir.mkdir()
        report = run_checklist(pkg_dir)
        assert not report.passed

    def test_checklist_on_minimal_package(self, tmp_path: Path):
        """A package with no scenarios/results should fail some checks."""
        run_dir = tmp_path / "runs"
        run_dir.mkdir()
        out = tmp_path / "package"
        build_repro_package(name="minimal", run_dir=run_dir, out_dir=out)

        report = run_checklist(out)
        # Should fail on scenarios, results, expected_metrics
        failed = [r for r in report.results if not r.passed]
        assert len(failed) >= 2

    def test_checklist_json_format(self, tmp_path: Path):
        """Verify checklist report serializes correctly."""
        run_dir = tmp_path / "runs"
        run_dir.mkdir()
        out = tmp_path / "package"
        build_repro_package(name="json-test", run_dir=run_dir, out_dir=out)

        report = run_checklist(out)
        d = report.to_dict()
        assert isinstance(d["results"], list)
        assert all("name" in r and "passed" in r for r in d["results"])


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_repro_parser_exists(self):
        from navirl.cli import build_parser

        parser = build_parser()
        # Should not raise
        args = parser.parse_args(["repro", "check", "/some/path"])
        assert args.repro_command == "check"

    def test_repro_build_parser(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["repro", "build", "my-study", "/runs", "--version", "2.0", "--author", "Alice"]
        )
        assert args.name == "my-study"
        assert args.run_dir == "/runs"
        assert args.version == "2.0"
        assert args.author == "Alice"

    def test_repro_verify_parser(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["repro", "verify", "/some/package"])
        assert args.repro_command == "verify"

    def test_repro_replay_parser(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["repro", "replay", "/some/package", "--tolerance", "0.2", "--seed", "99"]
        )
        assert args.repro_command == "replay"
        assert args.tolerance == 0.2
        assert args.seed == 99

    def test_repro_compliance_parser(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["repro", "compliance", "/some/package", "--no-pii"])
        assert args.repro_command == "compliance"
        assert args.no_pii is True


# ---------------------------------------------------------------------------
# MetricComparison
# ---------------------------------------------------------------------------


class TestMetricComparison:
    def test_to_dict(self):
        mc = MetricComparison(
            name="success_rate",
            expected_mean=0.9,
            replayed_value=0.85,
            tolerance=0.1,
            within_tolerance=True,
        )
        d = mc.to_dict()
        assert d["name"] == "success_rate"
        assert d["expected_mean"] == 0.9
        assert d["within_tolerance"] is True

    def test_within_tolerance_true(self):
        mc = MetricComparison(
            name="x", expected_mean=1.0, replayed_value=1.05, tolerance=0.1, within_tolerance=True
        )
        assert mc.within_tolerance

    def test_within_tolerance_false(self):
        mc = MetricComparison(
            name="x", expected_mean=1.0, replayed_value=2.0, tolerance=0.1, within_tolerance=False
        )
        assert not mc.within_tolerance


# ---------------------------------------------------------------------------
# ReplayResult
# ---------------------------------------------------------------------------


class TestReplayResult:
    def test_all_within_tolerance_empty(self):
        rr = ReplayResult(scenario_name="test", seed=42)
        assert rr.all_within_tolerance  # vacuously true

    def test_all_within_tolerance_mixed(self):
        rr = ReplayResult(
            scenario_name="test",
            seed=42,
            comparisons=[
                MetricComparison("a", 1.0, 1.0, 0.1, True),
                MetricComparison("b", 1.0, 5.0, 0.1, False),
            ],
        )
        assert not rr.all_within_tolerance

    def test_to_dict(self):
        rr = ReplayResult(
            scenario_name="hallway",
            seed=42,
            status="completed",
            metrics={"success_rate": 0.9},
        )
        d = rr.to_dict()
        assert d["scenario_name"] == "hallway"
        assert d["seed"] == 42
        assert d["metrics"]["success_rate"] == 0.9


# ---------------------------------------------------------------------------
# ReplayReport
# ---------------------------------------------------------------------------


class TestReplayReport:
    def test_empty_report_fails(self):
        report = ReplayReport(package_name="empty")
        assert not report.passed  # no results means not passed

    def test_passed_with_completed_results(self):
        report = ReplayReport(
            package_name="test",
            results=[
                ReplayResult(
                    scenario_name="s1",
                    seed=42,
                    status="completed",
                    comparisons=[MetricComparison("a", 1.0, 1.0, 0.1, True)],
                ),
            ],
        )
        assert report.passed
        assert report.total == 1
        assert report.completed_count == 1
        assert report.within_tolerance_count == 1

    def test_failed_run(self):
        report = ReplayReport(
            package_name="test",
            results=[
                ReplayResult(scenario_name="s1", seed=42, status="failed", error="boom"),
            ],
        )
        assert not report.passed

    def test_to_dict(self):
        report = ReplayReport(
            package_name="pkg",
            results=[ReplayResult(scenario_name="s1", seed=42)],
        )
        d = report.to_dict()
        assert d["package_name"] == "pkg"
        assert d["total_runs"] == 1
        assert isinstance(d["results"], list)

    def test_to_markdown(self):
        report = ReplayReport(
            package_name="study",
            results=[
                ReplayResult(
                    scenario_name="s1",
                    seed=42,
                    status="completed",
                    comparisons=[MetricComparison("a", 1.0, 1.0, 0.1, True)],
                ),
                ReplayResult(scenario_name="s2", seed=43, status="failed", error="timeout"),
            ],
        )
        md = report.to_markdown()
        assert "study" in md
        assert "s1" in md
        assert "timeout" in md


# ---------------------------------------------------------------------------
# _compare_metrics (internal helper)
# ---------------------------------------------------------------------------


class TestCompareMetrics:
    def test_exact_match(self):
        from navirl.repro.replay import _compare_metrics

        result = _compare_metrics(
            {"success_rate": 0.9},
            {"success_rate": {"mean": 0.9, "std": 0.0}},
            tolerance=0.1,
        )
        assert len(result) == 1
        assert result[0].within_tolerance

    def test_within_std_tolerance(self):
        from navirl.repro.replay import _compare_metrics

        result = _compare_metrics(
            {"success_rate": 0.7},
            {"success_rate": {"mean": 0.9, "std": 0.15}},
            tolerance=0.1,
        )
        # effective tolerance = max(0.1, 0.15*2) = 0.3
        # |0.7 - 0.9| = 0.2 <= 0.3 → within
        assert result[0].within_tolerance

    def test_outside_tolerance(self):
        from navirl.repro.replay import _compare_metrics

        result = _compare_metrics(
            {"success_rate": 0.1},
            {"success_rate": {"mean": 0.9, "std": 0.05}},
            tolerance=0.1,
        )
        assert not result[0].within_tolerance

    def test_missing_replayed_metric(self):
        from navirl.repro.replay import _compare_metrics

        result = _compare_metrics(
            {},
            {"success_rate": {"mean": 0.9, "std": 0.05}},
            tolerance=0.1,
        )
        assert not result[0].within_tolerance

    def test_nan_expected_skipped(self):
        from navirl.repro.replay import _compare_metrics

        result = _compare_metrics(
            {"x": 1.0},
            {"x": {"mean": float("nan"), "std": 0.0}},
            tolerance=0.1,
        )
        assert len(result) == 0

    def test_zero_expected_mean(self):
        from navirl.repro.replay import _compare_metrics

        result = _compare_metrics(
            {"collisions": 0.05},
            {"collisions": {"mean": 0.0, "std": 0.0}},
            tolerance=0.1,
        )
        # |0.05| <= 0.1 → within
        assert result[0].within_tolerance


# ---------------------------------------------------------------------------
# replay_package (unit tests with mocked pipeline)
# ---------------------------------------------------------------------------


class TestReplayPackageMissing:
    def test_missing_manifest(self, tmp_path: Path):
        from navirl.repro.replay import replay_package

        report = replay_package(tmp_path)
        assert not report.passed
        assert "MANIFEST.json not found" in report.results[0].error

    def test_missing_scenarios_dir(self, tmp_path: Path):
        from navirl.repro.replay import replay_package

        manifest = {"name": "test", "expected_metrics": {}}
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))

        report = replay_package(tmp_path)
        assert not report.passed
        assert "No scenarios/" in report.results[0].error

    def test_empty_scenarios_dir(self, tmp_path: Path):
        from navirl.repro.replay import replay_package

        manifest = {"name": "test", "expected_metrics": {}}
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
        (tmp_path / "scenarios").mkdir()

        report = replay_package(tmp_path)
        assert not report.passed
        assert "No scenario YAML" in report.results[0].error


# ---------------------------------------------------------------------------
# ComplianceFinding
# ---------------------------------------------------------------------------


class TestComplianceFinding:
    def test_to_dict(self):
        f = ComplianceFinding(
            category="credential",
            pattern_name="AWS access key",
            file_path="config.yaml",
            line_number=5,
            snippet="AKIA***1234",
        )
        d = f.to_dict()
        assert d["category"] == "credential"
        assert d["line_number"] == 5


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------


class TestComplianceReport:
    def test_empty_report_passes(self):
        report = ComplianceReport(package_name="clean", files_scanned=10)
        assert report.passed
        assert len(report.findings) == 0

    def test_with_findings_fails(self):
        report = ComplianceReport(
            package_name="dirty",
            files_scanned=5,
            findings=[
                ComplianceFinding("credential", "AWS key", "a.yaml", 1, "AKIA***"),
            ],
        )
        assert not report.passed

    def test_category_properties(self):
        report = ComplianceReport(
            package_name="mixed",
            findings=[
                ComplianceFinding("credential", "key", "a.yaml"),
                ComplianceFinding("pii", "email", "b.yaml"),
                ComplianceFinding("sensitive_file", ".env", ".env"),
                ComplianceFinding("pii", "phone", "c.yaml"),
            ],
        )
        assert len(report.credential_findings) == 1
        assert len(report.pii_findings) == 2
        assert len(report.sensitive_file_findings) == 1

    def test_to_dict(self):
        report = ComplianceReport(package_name="test", files_scanned=3)
        d = report.to_dict()
        assert d["passed"] is True
        assert d["files_scanned"] == 3
        assert d["total_findings"] == 0

    def test_to_markdown_clean(self):
        report = ComplianceReport(package_name="clean", files_scanned=5)
        md = report.to_markdown()
        assert "PASS" in md
        assert "No compliance issues" in md

    def test_to_markdown_with_findings(self):
        report = ComplianceReport(
            package_name="dirty",
            files_scanned=5,
            findings=[ComplianceFinding("credential", "API key", "config.yaml", 10, "api_***key")],
        )
        md = report.to_markdown()
        assert "FAIL" in md
        assert "config.yaml" in md


# ---------------------------------------------------------------------------
# scan_compliance
# ---------------------------------------------------------------------------


class TestScanCompliance:
    def test_clean_package(self, tmp_path: Path):
        (tmp_path / "MANIFEST.json").write_text('{"name": "clean"}')
        (tmp_path / "scenarios").mkdir()
        (tmp_path / "scenarios" / "test.yaml").write_text("grid: {rows: 10}\n")

        report = scan_compliance(tmp_path)
        assert report.passed
        assert report.files_scanned >= 2

    def test_detects_aws_key(self, tmp_path: Path):
        # AWS example key from documentation (split to avoid pre-commit secret scanner)
        example_key = "AKIA" + "IOSFODNN7EXAMPLE"
        (tmp_path / "config.yaml").write_text(f"key: {example_key}\n")

        report = scan_compliance(tmp_path)
        assert not report.passed
        creds = report.credential_findings
        assert any("AWS" in f.pattern_name for f in creds)

    def test_detects_private_key_file(self, tmp_path: Path):
        (tmp_path / "server.pem").write_text("dummy content\n")

        report = scan_compliance(tmp_path)
        assert not report.passed
        assert any(f.category == "sensitive_file" for f in report.findings)

    def test_detects_env_file(self, tmp_path: Path):
        (tmp_path / ".env").write_text("DB_PASSWORD=secret\n")

        report = scan_compliance(tmp_path)
        sensitive = report.sensitive_file_findings
        assert len(sensitive) >= 1

    def test_detects_email_pii(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("Contact: alice@example.com for details\n")

        report = scan_compliance(tmp_path)
        pii = report.pii_findings
        assert any("Email" in f.pattern_name for f in pii)

    def test_skip_pii_flag(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("Contact: alice@example.com\n")

        report = scan_compliance(tmp_path, check_pii=False)
        pii = report.pii_findings
        assert len(pii) == 0

    def test_skips_binary_files(self, tmp_path: Path):
        example_key = b"AKIA" + b"IOSFODNN7EXAMPLE"
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n" + example_key)

        report = scan_compliance(tmp_path)
        # Should not find credential in binary file
        creds = report.credential_findings
        assert len(creds) == 0

    def test_detects_private_key_block(self, tmp_path: Path):
        # Split to avoid pre-commit secret scanner
        header = "-----BEGIN RSA " + "PRIV" + "ATE KEY-----"
        (tmp_path / "key.txt").write_text(f"{header}\nMIIEpAIBAAK...\n")

        report = scan_compliance(tmp_path)
        assert not report.passed
        assert any("Private key" in f.pattern_name for f in report.credential_findings)

    def test_large_file_skipped_for_content(self, tmp_path: Path):
        # Create a file just over the limit (split key to avoid pre-commit scanner)
        example_key = "AKIA" + "IOSFODNN7EXAMPLE"
        large_file = tmp_path / "large.txt"
        large_file.write_text(f"{example_key}\n" * 100)

        report = scan_compliance(tmp_path, max_file_size=100)
        # File is scanned by name but not content
        creds = report.credential_findings
        assert len(creds) == 0

    def test_detects_generic_secret(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text('secret: "my_super_secret_token_value"\n')

        report = scan_compliance(tmp_path)
        creds = report.credential_findings
        assert any("secret" in f.pattern_name.lower() for f in creds)
