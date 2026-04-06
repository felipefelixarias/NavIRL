"""Tests for navirl/packs/ module: experiment pack schema, loader, reporter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from navirl.packs.loader import load_pack
from navirl.packs.reporter import write_pack_json, write_pack_markdown
from navirl.packs.schema import (
    PackManifest,
    PackResult,
    PackRunResult,
    PackScenarioEntry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "navirl" / "scenarios" / "library"
PACK_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "navirl" / "packs" / "library"


@pytest.fixture
def sample_metrics():
    return {
        "success_rate": 1.0,
        "collisions_agent_agent": 0,
        "collisions_agent_obstacle": 0,
        "intrusion_rate": 0.05,
        "min_dist_robot_human_min": 0.8,
        "min_dist_robot_human_mean": 1.5,
        "oscillation_score": 0.1,
        "path_length_robot": 3.2,
        "time_to_goal_robot": 12.0,
        "deadlock_count": 0,
    }


@pytest.fixture
def sample_manifest():
    return PackManifest(
        name="test-pack",
        version="1.0",
        description="A test pack",
        scenarios=[
            PackScenarioEntry(id="hallway_pass", path="hallway_pass.yaml", seeds=[7, 42]),
            PackScenarioEntry(id="kitchen_congestion", path="kitchen_congestion.yaml", seeds=[7]),
        ],
        metrics=["success_rate", "collisions_agent_agent", "intrusion_rate"],
        metadata={"authors": "Test"},
    )


@pytest.fixture
def sample_result(sample_metrics):
    result = PackResult(
        manifest_name="test-pack",
        manifest_version="1.0",
        manifest_checksum="abc123def456",
        timestamp="2026-04-06T00:00:00+00:00",
    )
    for entry_id, seed in [("hallway_pass", 7), ("hallway_pass", 42), ("kitchen_congestion", 7)]:
        m = dict(sample_metrics)
        m["success_rate"] = 1.0 if seed == 7 else 0.0
        result.runs.append(PackRunResult(entry_id=entry_id, seed=seed, metrics=m))
    return result


# ---------------------------------------------------------------------------
# PackScenarioEntry
# ---------------------------------------------------------------------------


class TestPackScenarioEntry:
    def test_defaults(self):
        entry = PackScenarioEntry(id="test", path="test.yaml")
        assert entry.seeds == [42]

    def test_custom_seeds(self):
        entry = PackScenarioEntry(id="test", path="test.yaml", seeds=[1, 2, 3])
        assert entry.seeds == [1, 2, 3]


# ---------------------------------------------------------------------------
# PackManifest
# ---------------------------------------------------------------------------


class TestPackManifest:
    def test_total_runs(self, sample_manifest):
        assert sample_manifest.total_runs == 3  # 2 + 1

    def test_total_runs_empty(self):
        m = PackManifest(name="empty")
        assert m.total_runs == 0

    def test_checksum_deterministic(self, sample_manifest):
        c1 = sample_manifest.checksum()
        c2 = sample_manifest.checksum()
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_checksum_changes_with_name(self, sample_manifest):
        c1 = sample_manifest.checksum()
        sample_manifest.name = "different-name"
        c2 = sample_manifest.checksum()
        assert c1 != c2

    def test_checksum_changes_with_version(self, sample_manifest):
        c1 = sample_manifest.checksum()
        sample_manifest.version = "2.0"
        c2 = sample_manifest.checksum()
        assert c1 != c2

    def test_checksum_changes_with_seeds(self, sample_manifest):
        c1 = sample_manifest.checksum()
        sample_manifest.scenarios[0].seeds.append(99)
        c2 = sample_manifest.checksum()
        assert c1 != c2

    def test_checksum_changes_with_metrics(self, sample_manifest):
        c1 = sample_manifest.checksum()
        sample_manifest.metrics.append("jerk_proxy")
        c2 = sample_manifest.checksum()
        assert c1 != c2

    def test_checksum_ignores_metadata(self, sample_manifest):
        c1 = sample_manifest.checksum()
        sample_manifest.metadata["extra"] = "ignored"
        c2 = sample_manifest.checksum()
        assert c1 == c2

    def test_checksum_ignores_description(self, sample_manifest):
        c1 = sample_manifest.checksum()
        sample_manifest.description = "totally different"
        c2 = sample_manifest.checksum()
        assert c1 == c2


# ---------------------------------------------------------------------------
# PackResult
# ---------------------------------------------------------------------------


class TestPackResult:
    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["manifest_name"] == "test-pack"
        assert d["total_runs"] == 3
        assert d["completed_runs"] == 3
        assert d["failed_runs"] == 0
        assert len(d["runs"]) == 3

    def test_to_dict_with_failures(self, sample_result):
        sample_result.runs.append(
            PackRunResult(entry_id="bad", seed=1, status="failed", error="boom")
        )
        d = sample_result.to_dict()
        assert d["total_runs"] == 4
        assert d["completed_runs"] == 3
        assert d["failed_runs"] == 1

    def test_aggregate(self, sample_result):
        agg = sample_result.aggregate(["success_rate", "intrusion_rate"])
        assert "success_rate" in agg
        assert "intrusion_rate" in agg
        assert agg["intrusion_rate"]["mean"] == pytest.approx(0.05)
        assert agg["intrusion_rate"]["std"] == pytest.approx(0.0)

    def test_aggregate_with_nan_values(self):
        result = PackResult(
            manifest_name="t", manifest_version="1", manifest_checksum="x"
        )
        result.runs.append(
            PackRunResult(
                entry_id="a", seed=1, metrics={"m": float("inf")}, status="completed"
            )
        )
        agg = result.aggregate(["m"])
        assert math.isnan(agg["m"]["mean"])

    def test_aggregate_missing_metric(self):
        result = PackResult(
            manifest_name="t", manifest_version="1", manifest_checksum="x"
        )
        result.runs.append(
            PackRunResult(entry_id="a", seed=1, metrics={"other": 1.0}, status="completed")
        )
        agg = result.aggregate(["missing_metric"])
        assert math.isnan(agg["missing_metric"]["mean"])

    def test_aggregate_skips_failed_runs(self, sample_metrics):
        result = PackResult(
            manifest_name="t", manifest_version="1", manifest_checksum="x"
        )
        result.runs.append(
            PackRunResult(entry_id="a", seed=1, metrics=sample_metrics, status="completed")
        )
        result.runs.append(
            PackRunResult(entry_id="a", seed=2, status="failed", error="boom")
        )
        agg = result.aggregate(["success_rate"])
        assert agg["success_rate"]["mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_baseline_pack(self):
        pack_path = PACK_LIBRARY_DIR / "social_nav_baseline.yaml"
        manifest = load_pack(pack_path)
        assert manifest.name == "social-nav-baseline"
        assert manifest.version == "1.0"
        assert len(manifest.scenarios) == 5
        assert manifest.total_runs == 15  # 5 scenarios * 3 seeds
        assert "success_rate" in manifest.metrics

    def test_load_resolves_scenario_paths(self):
        pack_path = PACK_LIBRARY_DIR / "social_nav_baseline.yaml"
        manifest = load_pack(pack_path)
        for entry in manifest.scenarios:
            p = Path(entry.path)
            assert p.exists(), f"Scenario not found: {entry.path}"

    def test_load_checksum_stable(self):
        pack_path = PACK_LIBRARY_DIR / "social_nav_baseline.yaml"
        m1 = load_pack(pack_path)
        m2 = load_pack(pack_path)
        assert m1.checksum() == m2.checksum()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Pack manifest not found"):
            load_pack(tmp_path / "nonexistent.yaml")

    def test_load_missing_name(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"scenarios": [{"id": "x", "path": "x.yaml"}]}))
        with pytest.raises(ValueError, match="'name' field"):
            load_pack(p)

    def test_load_no_scenarios(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"name": "empty", "scenarios": []}))
        with pytest.raises(ValueError, match="at least one scenario"):
            load_pack(p)

    def test_load_bad_seeds(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            yaml.dump(
                {
                    "name": "bad",
                    "scenarios": [{"id": "x", "path": "hallway_pass.yaml", "seeds": "not_a_list"}],
                }
            )
        )
        with pytest.raises(ValueError, match="list of integers"):
            load_pack(p)

    def test_load_scenario_entry_without_id(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"name": "bad", "scenarios": [{"path": "x.yaml"}]}))
        with pytest.raises(ValueError, match="'id' field"):
            load_pack(p)

    def test_load_scenario_not_found(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            yaml.dump(
                {
                    "name": "bad",
                    "scenarios": [{"id": "missing", "path": "this_does_not_exist.yaml"}],
                }
            )
        )
        with pytest.raises(FileNotFoundError, match="Cannot resolve"):
            load_pack(p)

    def test_load_default_path_from_id(self, tmp_path):
        """When path is omitted, it defaults to {id}.yaml in the library."""
        p = tmp_path / "pack.yaml"
        p.write_text(
            yaml.dump(
                {
                    "name": "inferred",
                    "scenarios": [{"id": "hallway_pass"}],
                }
            )
        )
        manifest = load_pack(p)
        assert len(manifest.scenarios) == 1
        assert Path(manifest.scenarios[0].path).exists()

    def test_load_default_metrics(self, tmp_path):
        """When metrics are omitted, defaults are used."""
        p = tmp_path / "pack.yaml"
        p.write_text(
            yaml.dump(
                {
                    "name": "no-metrics",
                    "scenarios": [{"id": "hallway_pass"}],
                }
            )
        )
        manifest = load_pack(p)
        assert len(manifest.metrics) > 0
        assert "success_rate" in manifest.metrics

    def test_load_custom_metrics(self, tmp_path):
        p = tmp_path / "pack.yaml"
        p.write_text(
            yaml.dump(
                {
                    "name": "custom",
                    "scenarios": [{"id": "hallway_pass"}],
                    "metrics": ["success_rate"],
                }
            )
        )
        manifest = load_pack(p)
        assert manifest.metrics == ["success_rate"]

    def test_load_not_a_mapping(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_pack(p)

    def test_load_metadata_preserved(self, tmp_path):
        p = tmp_path / "pack.yaml"
        p.write_text(
            yaml.dump(
                {
                    "name": "meta-test",
                    "scenarios": [{"id": "hallway_pass"}],
                    "metadata": {"authors": "Test Team", "purpose": "testing"},
                }
            )
        )
        manifest = load_pack(p)
        assert manifest.metadata["authors"] == "Test Team"


# ---------------------------------------------------------------------------
# Reporter — JSON
# ---------------------------------------------------------------------------


class TestJsonReporter:
    def test_write_json(self, sample_result, tmp_path):
        out = tmp_path / "results.json"
        write_pack_json(sample_result, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["manifest_name"] == "test-pack"
        assert len(data["runs"]) == 3

    def test_json_creates_parent_dirs(self, sample_result, tmp_path):
        out = tmp_path / "nested" / "deep" / "results.json"
        write_pack_json(sample_result, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Reporter — Markdown
# ---------------------------------------------------------------------------


class TestMarkdownReporter:
    def test_write_markdown(self, sample_result, tmp_path):
        out = tmp_path / "report.md"
        write_pack_markdown(sample_result, out, ["success_rate", "intrusion_rate"])
        assert out.exists()
        content = out.read_text()
        assert "test-pack" in content
        assert "Aggregated Metrics" in content
        assert "Per-Scenario Results" in content
        assert "hallway_pass" in content

    def test_markdown_includes_version_and_checksum(self, sample_result, tmp_path):
        out = tmp_path / "report.md"
        write_pack_markdown(sample_result, out)
        content = out.read_text()
        assert "1.0" in content
        assert "abc123def456" in content

    def test_markdown_with_failures(self, sample_result, tmp_path):
        sample_result.runs.append(
            PackRunResult(entry_id="bad_scenario", seed=1, status="failed", error="timeout")
        )
        out = tmp_path / "report.md"
        write_pack_markdown(sample_result, out)
        content = out.read_text()
        assert "Failures" in content
        assert "timeout" in content

    def test_markdown_auto_detects_metrics(self, sample_result, tmp_path):
        out = tmp_path / "report.md"
        write_pack_markdown(sample_result, out)  # No explicit metric_names
        content = out.read_text()
        assert "success_rate" in content

    def test_markdown_creates_parent_dirs(self, sample_result, tmp_path):
        out = tmp_path / "nested" / "report.md"
        write_pack_markdown(sample_result, out)
        assert out.exists()

    def test_markdown_empty_result(self, tmp_path):
        result = PackResult(
            manifest_name="empty", manifest_version="1.0", manifest_checksum="x"
        )
        out = tmp_path / "report.md"
        write_pack_markdown(result, out, ["success_rate"])
        content = out.read_text()
        assert "empty" in content
        assert "Total runs | 0" in content


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_pack_validate_command(self):
        from navirl.cli import build_parser

        parser = build_parser()
        pack_path = str(PACK_LIBRARY_DIR / "social_nav_baseline.yaml")
        args = parser.parse_args(["pack", "validate", pack_path])
        assert args.pack_command == "validate"
        assert args.manifest == pack_path

    def test_pack_run_command_defaults(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["pack", "run", "some_pack.yaml"])
        assert args.pack_command == "run"
        assert args.out == "out/pack"
        assert args.render is False
        assert args.video is False

    def test_pack_run_command_with_options(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["pack", "run", "some_pack.yaml", "--out", "/tmp/test", "--render"]
        )
        assert args.out == "/tmp/test"
        assert args.render is True
