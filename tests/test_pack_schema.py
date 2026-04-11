"""Tests for navirl.packs.schema module."""

from __future__ import annotations

import math

import pytest

from navirl.packs.schema import (
    PackManifest,
    PackResult,
    PackRunResult,
    PackScenarioEntry,
)

# ---------------------------------------------------------------------------
# PackScenarioEntry
# ---------------------------------------------------------------------------


class TestPackScenarioEntry:
    def test_defaults(self):
        e = PackScenarioEntry(id="test", path="test.yaml")
        assert e.id == "test"
        assert e.path == "test.yaml"
        assert e.seeds == [42]

    def test_custom_seeds(self):
        e = PackScenarioEntry(id="test", path="test.yaml", seeds=[1, 2, 3])
        assert e.seeds == [1, 2, 3]


# ---------------------------------------------------------------------------
# PackManifest
# ---------------------------------------------------------------------------


class TestPackManifest:
    def _manifest(self, n_scenarios=2, seeds_per=3):
        entries = [
            PackScenarioEntry(id=f"s{i}", path=f"s{i}.yaml", seeds=list(range(seeds_per)))
            for i in range(n_scenarios)
        ]
        return PackManifest(
            name="test-pack",
            version="1.0",
            description="Test pack",
            scenarios=entries,
            metrics=["success_rate"],
        )

    def test_total_runs(self):
        m = self._manifest(n_scenarios=3, seeds_per=5)
        assert m.total_runs == 15

    def test_checksum_deterministic(self):
        m1 = self._manifest()
        m2 = self._manifest()
        assert m1.checksum() == m2.checksum()

    def test_checksum_changes_with_content(self):
        m1 = self._manifest(n_scenarios=2)
        m2 = self._manifest(n_scenarios=3)
        assert m1.checksum() != m2.checksum()

    def test_defaults(self):
        m = PackManifest(name="test")
        assert m.version == "1.0"
        assert m.description == ""
        assert m.scenarios == []
        assert m.metrics == []
        assert m.metadata == {}

    def test_total_runs_empty(self):
        m = PackManifest(name="empty")
        assert m.total_runs == 0


# ---------------------------------------------------------------------------
# PackRunResult
# ---------------------------------------------------------------------------


class TestPackRunResult:
    def test_defaults(self):
        r = PackRunResult(entry_id="test", seed=42)
        assert r.status == "completed"
        assert r.error is None
        assert r.metrics == {}


# ---------------------------------------------------------------------------
# PackResult
# ---------------------------------------------------------------------------


class TestPackResult:
    def _result(self):
        return PackResult(
            manifest_name="test",
            manifest_version="1.0",
            manifest_checksum="abc123",
            runs=[
                PackRunResult(entry_id="s0", seed=0, metrics={"acc": 0.9, "loss": 0.1}),
                PackRunResult(entry_id="s0", seed=1, metrics={"acc": 0.8, "loss": 0.2}),
                PackRunResult(entry_id="s1", seed=0, metrics={"acc": 0.95}),
                PackRunResult(entry_id="fail", seed=0, status="failed", error="boom"),
            ],
        )

    def test_aggregate_basic(self):
        r = self._result()
        agg = r.aggregate(["acc", "loss"])
        assert agg["acc"]["mean"] == pytest.approx((0.9 + 0.8 + 0.95) / 3)
        assert agg["acc"]["min"] == pytest.approx(0.8)
        assert agg["acc"]["max"] == pytest.approx(0.95)
        # loss only has 2 values (s1/seed0 doesn't have loss)
        assert agg["loss"]["mean"] == pytest.approx(0.15)

    def test_aggregate_missing_metric(self):
        r = self._result()
        agg = r.aggregate(["nonexistent"])
        assert math.isnan(agg["nonexistent"]["mean"])

    def test_aggregate_excludes_failed(self):
        r = self._result()
        agg = r.aggregate(["acc"])
        # Only 3 completed runs
        assert agg["acc"]["mean"] == pytest.approx((0.9 + 0.8 + 0.95) / 3)

    def test_to_dict(self):
        r = self._result()
        d = r.to_dict()
        assert d["manifest_name"] == "test"
        assert d["total_runs"] == 4
        assert d["completed_runs"] == 3
        assert d["failed_runs"] == 1
        assert len(d["runs"]) == 4

    def test_to_dict_run_structure(self):
        r = self._result()
        d = r.to_dict()
        run = d["runs"][0]
        assert "entry_id" in run
        assert "seed" in run
        assert "metrics" in run
        assert "status" in run
        assert "error" in run

    def test_empty_result(self):
        r = PackResult(manifest_name="x", manifest_version="1", manifest_checksum="c")
        d = r.to_dict()
        assert d["total_runs"] == 0
        assert d["completed_runs"] == 0
