"""Tests for the experiment packs system (ROADMAP #8)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from navirl.packs.loader import load_pack, validate_pack
from navirl.packs.reporter import generate_pack_report
from navirl.packs.runner import PackResult, PackRunResult, _compute_pack_checksum
from navirl.packs.schema import PackManifest, PackScenarioEntry

SCENARIO_LIBRARY = Path(__file__).resolve().parent.parent / "navirl" / "scenarios" / "library"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestPackSchema:
    def test_pack_manifest_defaults(self):
        m = PackManifest(name="test", version="1.0")
        assert m.name == "test"
        assert m.version == "1.0"
        assert m.scenarios == []
        assert len(m.metrics) > 0

    def test_pack_scenario_entry_defaults(self):
        e = PackScenarioEntry(id="foo", path="/tmp/foo.yaml")
        assert e.seeds == [7]

    def test_pack_scenario_entry_custom_seeds(self):
        e = PackScenarioEntry(id="bar", path="/tmp/bar.yaml", seeds=[1, 2, 3])
        assert e.seeds == [1, 2, 3]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_pack(self):
        data = {
            "name": "test",
            "version": "1.0",
            "scenarios": [{"id": "s1", "path": "foo.yaml"}],
        }
        assert validate_pack(data) == []

    def test_missing_required_fields(self):
        errors = validate_pack({})
        assert any("name" in e for e in errors)
        assert any("version" in e for e in errors)
        assert any("scenarios" in e for e in errors)

    def test_scenarios_not_a_list(self):
        data = {"name": "x", "version": "1", "scenarios": "bad"}
        errors = validate_pack(data)
        assert any("list" in e for e in errors)

    def test_scenario_entry_missing_id(self):
        data = {
            "name": "x",
            "version": "1",
            "scenarios": [{"path": "a.yaml"}],
        }
        errors = validate_pack(data)
        assert any("id" in e for e in errors)

    def test_scenario_entry_missing_path(self):
        data = {
            "name": "x",
            "version": "1",
            "scenarios": [{"id": "s1"}],
        }
        errors = validate_pack(data)
        assert any("path" in e for e in errors)

    def test_scenario_entry_bad_seeds(self):
        data = {
            "name": "x",
            "version": "1",
            "scenarios": [{"id": "s1", "path": "a.yaml", "seeds": "bad"}],
        }
        errors = validate_pack(data)
        assert any("seeds" in e for e in errors)

    def test_empty_version(self):
        data = {"name": "x", "version": "", "scenarios": []}
        errors = validate_pack(data)
        assert any("version" in e for e in errors)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_load_valid_pack(self, tmp_path):
        # Create a minimal scenario file.
        scenario = {
            "id": "test_scenario",
            "scene": {"backend": "grid2d", "map": {"source": "builtin", "id": "hallway"}},
            "seed": 7,
            "horizon": {"steps": 10, "dt": 0.1},
            "humans": {
                "controller": {"type": "orca"},
                "count": 1,
                "starts": [[0, 0]],
                "goals": [[1, 0]],
            },
            "robot": {"start": [0, 0.5], "goal": [1, 0.5]},
        }
        scenario_path = tmp_path / "test_scenario.yaml"
        scenario_path.write_text(yaml.safe_dump(scenario))

        pack = {
            "name": "test-pack",
            "version": "0.1",
            "scenarios": [
                {"id": "s1", "path": str(scenario_path), "seeds": [7, 42]},
            ],
        }
        pack_path = tmp_path / "pack.yaml"
        pack_path.write_text(yaml.safe_dump(pack))

        manifest = load_pack(pack_path)
        assert manifest.name == "test-pack"
        assert manifest.version == "0.1"
        assert len(manifest.scenarios) == 1
        assert manifest.scenarios[0].seeds == [7, 42]

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_pack("/nonexistent/pack.yaml")

    def test_load_invalid_manifest(self, tmp_path):
        pack_path = tmp_path / "bad.yaml"
        pack_path.write_text("just a string\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_pack(pack_path)

    def test_load_missing_scenario_file(self, tmp_path):
        pack = {
            "name": "test",
            "version": "1.0",
            "scenarios": [{"id": "s1", "path": "/no/such/file.yaml"}],
        }
        pack_path = tmp_path / "pack.yaml"
        pack_path.write_text(yaml.safe_dump(pack))
        with pytest.raises(FileNotFoundError, match="Scenario not found"):
            load_pack(pack_path)

    def test_load_baseline_pack(self):
        """The shipped baseline pack should load successfully."""
        baseline = (
            Path(__file__).resolve().parent.parent
            / "navirl"
            / "packs"
            / "library"
            / "social_nav_baseline.yaml"
        )
        if not baseline.exists():
            pytest.skip("baseline pack not found")
        manifest = load_pack(baseline)
        assert manifest.name == "social-nav-baseline"
        assert len(manifest.scenarios) == 4


# ---------------------------------------------------------------------------
# Runner helpers (unit tests, no simulation)
# ---------------------------------------------------------------------------


class TestPackResult:
    def test_aggregate_empty(self):
        result = PackResult(pack_name="t", pack_version="1")
        assert result.aggregate() == {}

    def test_aggregate_basic(self):
        runs = [
            PackRunResult(entry_id="a", seed=7, bundle_dir="/tmp/a", metrics={"x": 1.0, "y": 2.0}),
            PackRunResult(entry_id="b", seed=7, bundle_dir="/tmp/b", metrics={"x": 3.0, "y": 4.0}),
        ]
        result = PackResult(pack_name="t", pack_version="1", runs=runs)
        agg = result.aggregate()
        assert agg["avg_x"] == pytest.approx(2.0)
        assert agg["avg_y"] == pytest.approx(3.0)

    def test_aggregate_with_filter(self):
        runs = [
            PackRunResult(entry_id="a", seed=7, bundle_dir="/tmp/a", metrics={"x": 1.0, "y": 2.0}),
        ]
        result = PackResult(pack_name="t", pack_version="1", runs=runs)
        agg = result.aggregate(metric_names=["x"])
        assert "avg_x" in agg
        assert "avg_y" not in agg

    def test_to_dict(self):
        runs = [
            PackRunResult(entry_id="a", seed=7, bundle_dir="/tmp/a", metrics={"x": 1.0}),
        ]
        result = PackResult(pack_name="t", pack_version="1", runs=runs, checksum="abc")
        d = result.to_dict()
        assert d["pack_name"] == "t"
        assert d["num_runs"] == 1
        assert d["runs"][0]["entry_id"] == "a"

    def test_checksum_deterministic(self):
        m = PackManifest(
            name="test",
            version="1.0",
            scenarios=[PackScenarioEntry(id="s", path="/a.yaml", seeds=[7])],
        )
        c1 = _compute_pack_checksum(m)
        c2 = _compute_pack_checksum(m)
        assert c1 == c2
        assert len(c1) == 16


# ---------------------------------------------------------------------------
# Reporter (unit tests)
# ---------------------------------------------------------------------------


class TestReporter:
    def test_generate_report(self, tmp_path):
        runs = [
            PackRunResult(entry_id="a", seed=7, bundle_dir="/tmp/a", metrics={"x": 1.0}),
            PackRunResult(entry_id="a", seed=42, bundle_dir="/tmp/b", metrics={"x": 3.0}),
            PackRunResult(entry_id="b", seed=7, bundle_dir="/tmp/c", metrics={"x": 2.0}),
        ]
        result = PackResult(pack_name="demo", pack_version="0.1", runs=runs, checksum="abc123")

        md_path = generate_pack_report(result, tmp_path)
        assert md_path.exists()
        content = md_path.read_text()
        assert "demo" in content
        assert "avg_x" in content
        assert (tmp_path / "pack_results.json").exists()

    def test_generate_report_empty_runs(self, tmp_path):
        result = PackResult(pack_name="empty", pack_version="0.0")
        md_path = generate_pack_report(result, tmp_path)
        content = md_path.read_text()
        assert "no numeric metrics" in content
