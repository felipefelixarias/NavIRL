"""Tests for the reproducibility package module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navirl.repro import (
    ArtifactEntry,
    ChecklistReport,
    CheckResult,
    EnvironmentPin,
    ReproPackage,
    build_repro_package,
    run_checklist,
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
