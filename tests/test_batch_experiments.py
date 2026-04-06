"""Tests for navirl/experiments/ module: batch templates, aggregation, runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from navirl.experiments.aggregator import (
    BatchAggregator,
    BatchSummary,
    RunRecord,
    write_json_summary,
    write_markdown_summary,
)
from navirl.experiments.templates import BatchTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "navirl" / "scenarios" / "library"


@pytest.fixture
def simple_template():
    return BatchTemplate(
        name="test_template",
        description="A test template",
        scenarios=["hallway_pass.yaml", "kitchen_congestion.yaml"],
        seeds=[42, 99],
    )


@pytest.fixture
def grid_template():
    return BatchTemplate(
        name="grid_test",
        description="Template with param grid",
        scenarios=["hallway_pass.yaml"],
        seeds=[42],
        param_grid={"scene.orca.neighbor_dist": [2.0, 4.0], "scene.orca.time_horizon": [3.0, 5.0]},
    )


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
        "jerk_proxy": 2.3,
        "path_length_robot": 5.0,
        "time_to_goal_robot": 12.0,
        "deadlock_count": 0,
    }


# ---------------------------------------------------------------------------
# BatchTemplate tests
# ---------------------------------------------------------------------------


class TestBatchTemplate:
    def test_basic_construction(self, simple_template):
        assert simple_template.name == "test_template"
        assert len(simple_template.scenarios) == 2
        assert simple_template.seeds == [42, 99]

    def test_expand_tasks_no_grid(self, simple_template):
        tasks = simple_template.expand_tasks()
        assert len(tasks) == 4  # 2 scenarios * 2 seeds
        seeds_seen = {t["seed"] for t in tasks}
        assert seeds_seen == {42, 99}
        assert all(t["overrides"] == {} for t in tasks)

    def test_expand_tasks_with_grid(self, grid_template):
        tasks = grid_template.expand_tasks()
        # 1 scenario * 1 seed * (2 * 2 = 4 combos) = 4
        assert len(tasks) == 4
        overrides_set = {
            (t["overrides"]["scene.orca.neighbor_dist"], t["overrides"]["scene.orca.time_horizon"])
            for t in tasks
        }
        assert overrides_set == {(2.0, 3.0), (2.0, 5.0), (4.0, 3.0), (4.0, 5.0)}

    def test_total_runs(self, simple_template, grid_template):
        assert simple_template.total_runs == 4
        assert grid_template.total_runs == 4

    def test_resolve_scenarios_library(self):
        tmpl = BatchTemplate(name="lib", scenarios=["library"])
        paths = tmpl.resolve_scenarios()
        assert len(paths) > 0
        assert all(p.suffix == ".yaml" for p in paths)

    def test_from_yaml(self, tmp_path):
        yaml_data = {
            "name": "yaml_test",
            "description": "loaded from yaml",
            "scenarios": ["hallway_pass.yaml"],
            "seeds": [1, 2, 3],
            "tags": ["test"],
        }
        yaml_path = tmp_path / "test.yaml"
        with yaml_path.open("w") as f:
            yaml.safe_dump(yaml_data, f)

        tmpl = BatchTemplate.from_yaml(yaml_path)
        assert tmpl.name == "yaml_test"
        assert tmpl.seeds == [1, 2, 3]
        assert tmpl.tags == ["test"]

    def test_to_yaml_roundtrip(self, tmp_path, grid_template):
        yaml_path = tmp_path / "roundtrip.yaml"
        grid_template.to_yaml(yaml_path)
        loaded = BatchTemplate.from_yaml(yaml_path)
        assert loaded.name == grid_template.name
        assert loaded.param_grid == grid_template.param_grid
        assert loaded.seeds == grid_template.seeds

    def test_empty_scenarios_expand(self):
        tmpl = BatchTemplate(name="empty", scenarios=[])
        assert tmpl.expand_tasks() == []

    def test_expand_param_grid_empty(self):
        tmpl = BatchTemplate(name="no_grid", scenarios=["hallway_pass.yaml"], seeds=[42])
        combos = tmpl._expand_param_grid()
        assert combos == [{}]


# ---------------------------------------------------------------------------
# BatchAggregator tests
# ---------------------------------------------------------------------------


class TestBatchAggregator:
    def test_empty_aggregator(self):
        agg = BatchAggregator(template_name="empty")
        summary = agg.summarize()
        assert summary.total_runs == 0
        assert summary.completed_runs == 0
        assert summary.per_scenario == []

    def test_single_run(self, sample_metrics):
        agg = BatchAggregator(template_name="single")
        agg.add_record(
            RunRecord(scenario="hallway_pass", seed=42, metrics=sample_metrics)
        )
        summary = agg.summarize()
        assert summary.total_runs == 1
        assert summary.completed_runs == 1
        assert len(summary.per_scenario) == 1
        assert summary.per_scenario[0].scenario == "hallway_pass"
        assert summary.per_scenario[0].num_successes == 1

    def test_multiple_seeds(self, sample_metrics):
        agg = BatchAggregator(template_name="multi_seed")
        for seed in [42, 99, 123]:
            metrics = {**sample_metrics, "path_length_robot": float(seed) / 10}
            agg.add_record(RunRecord(scenario="hallway_pass", seed=seed, metrics=metrics))

        summary = agg.summarize()
        assert summary.completed_runs == 3
        stats = summary.per_scenario[0].metrics["path_length_robot"]
        assert stats["mean"] == pytest.approx((4.2 + 9.9 + 12.3) / 3, rel=1e-3)
        assert stats["min"] == pytest.approx(4.2, rel=1e-3)
        assert stats["max"] == pytest.approx(12.3, rel=1e-3)

    def test_multiple_scenarios(self, sample_metrics):
        agg = BatchAggregator(template_name="multi_scenario")
        agg.add_record(RunRecord(scenario="hallway_pass", seed=42, metrics=sample_metrics))
        agg.add_record(
            RunRecord(
                scenario="kitchen_congestion",
                seed=42,
                metrics={**sample_metrics, "success_rate": 0.0},
            )
        )

        summary = agg.summarize()
        assert len(summary.per_scenario) == 2
        names = {s.scenario for s in summary.per_scenario}
        assert names == {"hallway_pass", "kitchen_congestion"}

    def test_failed_runs_tracked(self, sample_metrics):
        agg = BatchAggregator(template_name="with_failures")
        agg.add_record(RunRecord(scenario="hallway_pass", seed=42, metrics=sample_metrics))
        agg.add_record(
            RunRecord(scenario="hallway_pass", seed=99, status="failed", error="boom")
        )

        summary = agg.summarize()
        assert summary.total_runs == 2
        assert summary.completed_runs == 1
        assert summary.failed_runs == 1

    def test_global_metrics_aggregated(self, sample_metrics):
        agg = BatchAggregator(template_name="global")
        agg.add_record(
            RunRecord(scenario="hallway_pass", seed=42, metrics=sample_metrics)
        )
        agg.add_record(
            RunRecord(
                scenario="kitchen_congestion",
                seed=42,
                metrics={**sample_metrics, "intrusion_rate": 0.1},
            )
        )

        summary = agg.summarize()
        assert "intrusion_rate" in summary.global_metrics
        assert summary.global_metrics["intrusion_rate"]["mean"] == pytest.approx(0.075, rel=1e-3)

    def test_summary_to_dict(self, sample_metrics):
        agg = BatchAggregator(template_name="dict_test")
        agg.add_record(RunRecord(scenario="hallway_pass", seed=42, metrics=sample_metrics))
        summary = agg.summarize()
        d = summary.to_dict()
        assert d["template_name"] == "dict_test"
        assert d["total_runs"] == 1
        assert isinstance(d["per_scenario"], list)
        assert isinstance(d["global_metrics"], dict)

    def test_inf_values_handled(self):
        agg = BatchAggregator(template_name="inf_test")
        metrics = {"time_to_goal_robot": float("inf"), "success_rate": 0.0}
        agg.add_record(RunRecord(scenario="test", seed=42, metrics=metrics))
        summary = agg.summarize()
        # inf should be excluded from statistics
        stats = summary.per_scenario[0].metrics.get("time_to_goal_robot", {})
        if stats:
            assert math.isnan(stats["mean"])


# ---------------------------------------------------------------------------
# Output writer tests
# ---------------------------------------------------------------------------


class TestOutputWriters:
    def test_write_json_summary(self, tmp_path, sample_metrics):
        agg = BatchAggregator(template_name="json_test")
        agg.add_record(RunRecord(scenario="hallway_pass", seed=42, metrics=sample_metrics))
        summary = agg.summarize()

        json_path = tmp_path / "summary.json"
        write_json_summary(summary, json_path)

        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["template_name"] == "json_test"
        assert data["total_runs"] == 1

    def test_write_markdown_summary(self, tmp_path, sample_metrics):
        agg = BatchAggregator(template_name="md_test")
        for seed in [42, 99]:
            agg.add_record(RunRecord(scenario="hallway_pass", seed=seed, metrics=sample_metrics))
        summary = agg.summarize()

        md_path = tmp_path / "REPORT.md"
        write_markdown_summary(summary, md_path)

        assert md_path.exists()
        content = md_path.read_text()
        assert "# Batch Experiment Report: md_test" in content
        assert "hallway_pass" in content
        assert "| success_rate" in content

    def test_markdown_contains_overview_table(self, tmp_path, sample_metrics):
        agg = BatchAggregator(template_name="overview")
        agg.add_record(RunRecord(scenario="test", seed=42, metrics=sample_metrics))
        agg.add_record(RunRecord(scenario="test", seed=99, status="failed", error="err"))
        summary = agg.summarize()

        md_path = tmp_path / "REPORT.md"
        write_markdown_summary(summary, md_path)
        content = md_path.read_text()
        assert "| Total runs | 2 |" in content
        assert "| Completed | 1 |" in content
        assert "| Failed | 1 |" in content


# ---------------------------------------------------------------------------
# Template YAML loading from research/templates/
# ---------------------------------------------------------------------------


class TestResearchTemplates:
    def test_seed_sweep_template_loads(self):
        path = Path(__file__).resolve().parent.parent / "research" / "templates" / "seed_sweep.yaml"
        if not path.exists():
            pytest.skip("seed_sweep.yaml not found")
        tmpl = BatchTemplate.from_yaml(path)
        assert tmpl.name == "seed_sweep"
        assert len(tmpl.seeds) == 5
        assert len(tmpl.scenarios) >= 3

    def test_orca_param_study_loads(self):
        path = (
            Path(__file__).resolve().parent.parent / "research" / "templates" / "orca_param_study.yaml"
        )
        if not path.exists():
            pytest.skip("orca_param_study.yaml not found")
        tmpl = BatchTemplate.from_yaml(path)
        assert tmpl.name == "orca_param_study"
        assert "scene.orca.neighbor_dist" in tmpl.param_grid
        # 2 scenarios * 2 seeds * 3*3 grid = 36
        assert tmpl.total_runs == 36

    def test_full_library_resolves(self):
        path = (
            Path(__file__).resolve().parent.parent / "research" / "templates" / "full_library.yaml"
        )
        if not path.exists():
            pytest.skip("full_library.yaml not found")
        tmpl = BatchTemplate.from_yaml(path)
        scenarios = tmpl.resolve_scenarios()
        assert len(scenarios) > 5  # Should find all library scenarios
