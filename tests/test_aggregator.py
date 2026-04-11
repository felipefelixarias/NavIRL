"""Tests for navirl.experiments.aggregator — batch metrics aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from navirl.experiments.aggregator import (
    BatchAggregator,
    BatchSummary,
    RunRecord,
    ScenarioStats,
    _compute_stats,
    write_json_summary,
    write_markdown_summary,
)

# ---------------------------------------------------------------------------
# _compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_single_value(self):
        stats = _compute_stats([5.0])
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["median"] == 5.0

    def test_multiple_values(self):
        stats = _compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0

    def test_empty_returns_nan(self):
        stats = _compute_stats([])
        assert math.isnan(stats["mean"])
        assert math.isnan(stats["min"])

    def test_all_inf_returns_nan(self):
        stats = _compute_stats([float("inf"), float("inf")])
        assert math.isnan(stats["mean"])

    def test_filters_inf_values(self):
        stats = _compute_stats([1.0, float("inf"), 3.0])
        assert stats["mean"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0

    def test_filters_nan_values(self):
        stats = _compute_stats([1.0, float("nan"), 3.0])
        assert stats["mean"] == 2.0


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------


class TestRunRecord:
    def test_defaults(self):
        rec = RunRecord(scenario="hallway", seed=42)
        assert rec.scenario == "hallway"
        assert rec.seed == 42
        assert rec.status == "completed"
        assert rec.error is None
        assert rec.metrics == {}
        assert rec.overrides == {}

    def test_with_metrics(self):
        rec = RunRecord(
            scenario="crossing",
            seed=1,
            metrics={"success_rate": 1.0, "collisions_agent_agent": 0},
        )
        assert rec.metrics["success_rate"] == 1.0


# ---------------------------------------------------------------------------
# BatchAggregator
# ---------------------------------------------------------------------------


class TestBatchAggregator:
    def _make_records(self) -> list[RunRecord]:
        return [
            RunRecord(
                scenario="hallway",
                seed=1,
                metrics={
                    "success_rate": 1.0,
                    "collisions_agent_agent": 0,
                    "path_length_robot": 5.0,
                },
            ),
            RunRecord(
                scenario="hallway",
                seed=2,
                metrics={
                    "success_rate": 1.0,
                    "collisions_agent_agent": 2,
                    "path_length_robot": 6.0,
                },
            ),
            RunRecord(
                scenario="crossing",
                seed=1,
                metrics={
                    "success_rate": 0.0,
                    "collisions_agent_agent": 5,
                    "path_length_robot": 10.0,
                },
            ),
            RunRecord(
                scenario="crossing",
                seed=2,
                status="failed",
                error="timeout",
            ),
        ]

    def test_add_records(self):
        agg = BatchAggregator("test_batch")
        records = self._make_records()
        for r in records:
            agg.add_record(r)
        assert len(agg.records) == 4

    def test_summarize_counts(self):
        agg = BatchAggregator("test_batch")
        for r in self._make_records():
            agg.add_record(r)
        summary = agg.summarize()
        assert summary.total_runs == 4
        assert summary.completed_runs == 3
        assert summary.failed_runs == 1

    def test_summarize_per_scenario(self):
        agg = BatchAggregator("test_batch")
        for r in self._make_records():
            agg.add_record(r)
        summary = agg.summarize()
        assert len(summary.per_scenario) == 2
        scenario_names = [s.scenario for s in summary.per_scenario]
        assert "hallway" in scenario_names
        assert "crossing" in scenario_names

    def test_hallway_successes(self):
        agg = BatchAggregator("test_batch")
        for r in self._make_records():
            agg.add_record(r)
        summary = agg.summarize()
        hallway = next(s for s in summary.per_scenario if s.scenario == "hallway")
        assert hallway.num_runs == 2
        assert hallway.num_successes == 2

    def test_crossing_successes(self):
        agg = BatchAggregator("test_batch")
        for r in self._make_records():
            agg.add_record(r)
        summary = agg.summarize()
        crossing = next(s for s in summary.per_scenario if s.scenario == "crossing")
        assert crossing.num_runs == 1  # only completed runs
        assert crossing.num_successes == 0

    def test_global_metrics_computed(self):
        agg = BatchAggregator("test_batch")
        for r in self._make_records():
            agg.add_record(r)
        summary = agg.summarize()
        assert "success_rate" in summary.global_metrics
        assert "collisions_agent_agent" in summary.global_metrics
        # 3 completed runs: success rates = [1, 1, 0]
        np.testing.assert_allclose(
            summary.global_metrics["success_rate"]["mean"],
            2.0 / 3.0,
            atol=1e-6,
        )

    def test_custom_metric_keys(self):
        agg = BatchAggregator("test_batch")
        for r in self._make_records():
            agg.add_record(r)
        summary = agg.summarize(metric_keys=["path_length_robot"])
        assert "path_length_robot" in summary.global_metrics
        assert "success_rate" not in summary.global_metrics

    def test_empty_aggregator(self):
        agg = BatchAggregator("empty")
        summary = agg.summarize()
        assert summary.total_runs == 0
        assert summary.completed_runs == 0
        assert summary.failed_runs == 0
        assert summary.per_scenario == []

    def test_template_name_propagated(self):
        agg = BatchAggregator("my_template")
        summary = agg.summarize()
        assert summary.template_name == "my_template"

    def test_timestamp_populated(self):
        agg = BatchAggregator("test")
        summary = agg.summarize()
        assert summary.timestamp  # non-empty


# ---------------------------------------------------------------------------
# BatchSummary.to_dict
# ---------------------------------------------------------------------------


class TestBatchSummaryToDict:
    def test_serializable(self):
        summary = BatchSummary(
            template_name="test",
            total_runs=2,
            completed_runs=2,
            failed_runs=0,
            per_scenario=[
                ScenarioStats(
                    scenario="hallway",
                    num_runs=2,
                    num_successes=2,
                    metrics={
                        "success_rate": {
                            "mean": 1.0,
                            "std": 0.0,
                            "min": 1.0,
                            "max": 1.0,
                            "median": 1.0,
                        }
                    },
                )
            ],
            global_metrics={
                "success_rate": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "median": 1.0}
            },
            timestamp="2026-01-01T00:00:00",
        )
        d = summary.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert "hallway" in json_str
        assert d["total_runs"] == 2
        assert len(d["per_scenario"]) == 1


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


class TestWriters:
    def _make_summary(self) -> BatchSummary:
        return BatchSummary(
            template_name="test_exp",
            total_runs=3,
            completed_runs=2,
            failed_runs=1,
            per_scenario=[
                ScenarioStats(
                    scenario="hallway",
                    num_runs=2,
                    num_successes=1,
                    metrics={
                        "success_rate": {
                            "mean": 0.5,
                            "std": 0.5,
                            "min": 0.0,
                            "max": 1.0,
                            "median": 0.5,
                        }
                    },
                )
            ],
            global_metrics={
                "success_rate": {"mean": 0.5, "std": 0.5, "min": 0.0, "max": 1.0, "median": 0.5}
            },
            timestamp="2026-01-01T00:00:00",
        )

    def test_write_json_summary(self, tmp_path: Path):
        summary = self._make_summary()
        out = tmp_path / "results" / "summary.json"
        write_json_summary(summary, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["template_name"] == "test_exp"
        assert data["total_runs"] == 3

    def test_write_markdown_summary(self, tmp_path: Path):
        summary = self._make_summary()
        out = tmp_path / "results" / "report.md"
        write_markdown_summary(summary, out)
        assert out.exists()
        content = out.read_text()
        assert "test_exp" in content
        assert "hallway" in content
        assert "| success_rate" in content

    def test_write_markdown_empty_metrics(self, tmp_path: Path):
        summary = BatchSummary(
            template_name="empty",
            total_runs=0,
            completed_runs=0,
            failed_runs=0,
            per_scenario=[],
            global_metrics={},
        )
        out = tmp_path / "report.md"
        write_markdown_summary(summary, out)
        assert out.exists()
        content = out.read_text()
        assert "empty" in content
