"""Tests for navirl/tune/runner.py utility functions.

Covers: _set_dotted_path, _sample_overrides, _apply_overrides,
_score_scenario, _sanitize_json_value, _write_report,
_load_search_space, _resolve_scenarios, TuningConfig validation.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import yaml

from navirl.tune.runner import (
    TuningConfig,
    _apply_overrides,
    _load_search_space,
    _resolve_scenarios,
    _sample_overrides,
    _sanitize_json_value,
    _score_scenario,
    _set_dotted_path,
    _write_report,
    run_tuning,
)

# ---------------------------------------------------------------------------
# _set_dotted_path
# ---------------------------------------------------------------------------


class TestSetDottedPath:
    def test_single_key(self):
        obj = {}
        _set_dotted_path(obj, "foo", 42)
        assert obj == {"foo": 42}

    def test_nested_key(self):
        obj = {}
        _set_dotted_path(obj, "a.b.c", "val")
        assert obj == {"a": {"b": {"c": "val"}}}

    def test_existing_nested_dict(self):
        obj = {"a": {"b": {"c": 1}}}
        _set_dotted_path(obj, "a.b.c", 2)
        assert obj["a"]["b"]["c"] == 2

    def test_overwrites_non_dict_intermediate(self):
        obj = {"a": "string"}
        _set_dotted_path(obj, "a.b", 5)
        assert obj == {"a": {"b": 5}}

    def test_deeply_nested(self):
        obj = {}
        _set_dotted_path(obj, "x.y.z.w.v", 99)
        assert obj["x"]["y"]["z"]["w"]["v"] == 99


# ---------------------------------------------------------------------------
# _sample_overrides
# ---------------------------------------------------------------------------


class TestSampleOverrides:
    def test_returns_one_value_per_key(self):
        rng = random.Random(42)
        space = {"a": [1, 2, 3], "b": [10, 20]}
        result = _sample_overrides(rng, space)
        assert set(result.keys()) == {"a", "b"}
        assert result["a"] in [1, 2, 3]
        assert result["b"] in [10, 20]

    def test_deterministic_with_seed(self):
        space = {"x": [1, 2, 3, 4, 5]}
        r1 = _sample_overrides(random.Random(7), space)
        r2 = _sample_overrides(random.Random(7), space)
        assert r1 == r2

    def test_empty_space(self):
        result = _sample_overrides(random.Random(0), {})
        assert result == {}


# ---------------------------------------------------------------------------
# _apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    def test_applies_orca_human_overrides(self):
        scenario = {
            "humans": {"controller": {"type": "orca", "params": {"lookahead": 2}}},
            "robot": {"controller": {"type": "baseline_astar", "params": {}}},
        }
        overrides = {"humans.controller.params.lookahead": 4}
        result = _apply_overrides(scenario, overrides)
        assert result["humans"]["controller"]["params"]["lookahead"] == 4
        # Original unchanged
        assert scenario["humans"]["controller"]["params"]["lookahead"] == 2

    def test_skips_human_overrides_for_non_orca(self):
        scenario = {
            "humans": {"controller": {"type": "social_force", "params": {"lookahead": 2}}},
            "robot": {"controller": {"type": "baseline_astar", "params": {}}},
        }
        overrides = {"humans.controller.params.lookahead": 4}
        result = _apply_overrides(scenario, overrides)
        # Should not apply the override
        assert result["humans"]["controller"]["params"]["lookahead"] == 2

    def test_applies_orca_plus_human_overrides(self):
        scenario = {
            "humans": {"controller": {"type": "orca_plus", "params": {"lookahead": 2}}},
            "robot": {"controller": {"type": "other"}},
        }
        overrides = {"humans.controller.params.lookahead": 4}
        result = _apply_overrides(scenario, overrides)
        assert result["humans"]["controller"]["params"]["lookahead"] == 4

    def test_skips_robot_overrides_for_non_astar(self):
        scenario = {
            "humans": {"controller": {"type": "orca"}},
            "robot": {"controller": {"type": "learned_policy", "params": {"target_lookahead": 2}}},
        }
        overrides = {"robot.controller.params.target_lookahead": 5}
        result = _apply_overrides(scenario, overrides)
        assert result["robot"]["controller"]["params"]["target_lookahead"] == 2

    def test_geometry_sync_scene_to_eval(self):
        scenario = {
            "humans": {"controller": {"type": "orca"}},
            "robot": {"controller": {"type": "baseline_astar"}},
        }
        overrides = {"scene.orca.wall_clearance_buffer_m": 0.02}
        result = _apply_overrides(scenario, overrides)
        assert result["evaluation"]["wall_clearance_buffer"] == 0.02

    def test_geometry_sync_eval_to_scene(self):
        scenario = {
            "humans": {"controller": {"type": "orca"}},
            "robot": {"controller": {"type": "baseline_astar"}},
        }
        overrides = {"evaluation.wall_clearance_buffer": 0.03}
        result = _apply_overrides(scenario, overrides)
        assert result["scene"]["orca"]["wall_clearance_buffer_m"] == 0.03

    def test_both_geometry_overrides_no_sync(self):
        scenario = {
            "humans": {"controller": {"type": "orca"}},
            "robot": {"controller": {"type": "baseline_astar"}},
        }
        overrides = {
            "scene.orca.wall_clearance_buffer_m": 0.01,
            "evaluation.wall_clearance_buffer": 0.05,
        }
        result = _apply_overrides(scenario, overrides)
        # Both explicitly set — no syncing
        assert result["scene"]["orca"]["wall_clearance_buffer_m"] == 0.01
        assert result["evaluation"]["wall_clearance_buffer"] == 0.05

    def test_applies_general_overrides(self):
        scenario = {
            "humans": {"controller": {"type": "orca"}},
            "robot": {"controller": {"type": "baseline_astar"}},
            "evaluation": {},
        }
        overrides = {"evaluation.deadlock_resample_attempts": 6}
        result = _apply_overrides(scenario, overrides)
        assert result["evaluation"]["deadlock_resample_attempts"] == 6

    def test_missing_controller_keys(self):
        scenario = {}
        overrides = {"humans.controller.params.lookahead": 3}
        result = _apply_overrides(scenario, overrides)
        # Controller type is "" — not orca, so skip
        assert "lookahead" not in str(
            result.get("humans", {}).get("controller", {}).get("params", {})
        )


# ---------------------------------------------------------------------------
# _score_scenario
# ---------------------------------------------------------------------------


class TestScoreScenario:
    def test_perfect_scenario(self):
        metrics = {
            "horizon_steps": 100,
            "collisions_agent_agent": 0,
            "intrusion_rate": 0.0,
            "collisions_agent_obstacle": 0,
            "deadlock_count": 0,
            "_retry_count": 0,
            "oscillation_score": 0.0,
            "jerk_proxy": 0.0,
            "success_rate": 1.0,
        }
        invariants = {
            "overall_pass": True,
            "checks": [
                {"name": "robot_progress", "progress_fraction": 1.0},
                {"name": "wall_proximity_fraction", "near_wall_fraction": 0.0},
                {"name": "motion_jitter", "worst_flip_rate": 0.0},
                {"name": "agent_stop_duration", "top_longest_stops": []},
            ],
        }
        judge = {"confidence": 1.0, "overall_pass": True, "status": "pass"}
        score = _score_scenario(metrics, invariants, judge)
        # Should be positive and high
        assert score > 0

    def test_terrible_scenario(self):
        metrics = {
            "horizon_steps": 100,
            "collisions_agent_agent": 50,
            "intrusion_rate": 0.5,
            "collisions_agent_obstacle": 10,
            "deadlock_count": 5,
            "_retry_count": 3,
            "oscillation_score": 0.8,
            "jerk_proxy": 10.0,
            "success_rate": 0.0,
        }
        invariants = {"overall_pass": False, "checks": []}
        judge = {
            "confidence": 0.1,
            "overall_pass": False,
            "status": "needs_human_review",
        }
        score = _score_scenario(metrics, invariants, judge)
        assert score < 0

    def test_missing_metrics_use_defaults(self):
        metrics = {"horizon_steps": 50}
        invariants = {"overall_pass": True, "checks": []}
        judge = {"confidence": 0.5, "overall_pass": True}
        score = _score_scenario(metrics, invariants, judge)
        assert isinstance(score, float)

    def test_judge_penalties(self):
        base_metrics = {"horizon_steps": 100, "success_rate": 1.0}
        base_invariants = {"overall_pass": True, "checks": []}

        score_ok = _score_scenario(
            base_metrics,
            base_invariants,
            {"confidence": 0.9, "overall_pass": True, "status": "pass"},
        )
        score_fail = _score_scenario(
            base_metrics,
            base_invariants,
            {"confidence": 0.9, "overall_pass": False, "status": "fail"},
        )
        score_review = _score_scenario(
            base_metrics,
            base_invariants,
            {"confidence": 0.9, "overall_pass": False, "status": "needs_human_review"},
        )
        assert score_ok > score_fail
        assert score_fail > score_review

    def test_stop_duration_penalty(self):
        metrics = {"horizon_steps": 100, "success_rate": 0.5}
        invariants = {
            "overall_pass": True,
            "checks": [
                {
                    "name": "agent_stop_duration",
                    "top_longest_stops": [
                        {"max_stopped_seconds": 10.0},
                        {"max_stopped_seconds": 5.0},
                    ],
                },
            ],
        }
        judge = {"confidence": 0.5, "overall_pass": True}
        score = _score_scenario(metrics, invariants, judge)
        # With big stop duration penalty
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# _sanitize_json_value
# ---------------------------------------------------------------------------


class TestSanitizeJsonValue:
    def test_finite_float(self):
        assert _sanitize_json_value(3.14) == 3.14

    def test_inf_becomes_none(self):
        assert _sanitize_json_value(float("inf")) is None
        assert _sanitize_json_value(float("-inf")) is None

    def test_nan_becomes_none(self):
        assert _sanitize_json_value(float("nan")) is None

    def test_nested_dict(self):
        result = _sanitize_json_value({"a": float("inf"), "b": 1.0, "c": "text"})
        assert result == {"a": None, "b": 1.0, "c": "text"}

    def test_nested_list(self):
        result = _sanitize_json_value([float("nan"), 2.0, "ok"])
        assert result == [None, 2.0, "ok"]

    def test_deeply_nested(self):
        result = _sanitize_json_value({"a": [{"b": float("inf")}]})
        assert result == {"a": [{"b": None}]}

    def test_non_float_passthrough(self):
        assert _sanitize_json_value(42) == 42
        assert _sanitize_json_value("hello") == "hello"
        assert _sanitize_json_value(None) is None
        assert _sanitize_json_value(True) is True


# ---------------------------------------------------------------------------
# _load_search_space
# ---------------------------------------------------------------------------


class TestLoadSearchSpace:
    def test_none_returns_default(self):
        result = _load_search_space(None)
        assert isinstance(result, dict)
        assert len(result) > 0
        # Should be a copy
        assert result is not _load_search_space(None)

    def test_loads_json(self, tmp_path):
        space = {"a.b": [1, 2, 3], "c.d": [4, 5]}
        path = tmp_path / "space.json"
        path.write_text(json.dumps(space), encoding="utf-8")
        result = _load_search_space(path)
        assert result == space

    def test_loads_yaml(self, tmp_path):
        space = {"a.b": [1, 2, 3]}
        path = tmp_path / "space.yaml"
        path.write_text(yaml.safe_dump(space), encoding="utf-8")
        result = _load_search_space(path)
        assert result == space

    def test_invalid_not_dict_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="must be an object"):
            _load_search_space(path)

    def test_non_string_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("1: [a, b]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="keys must be strings"):
            _load_search_space(path)

    def test_empty_list_value_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"key": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty list"):
            _load_search_space(path)

    def test_non_list_value_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"key": "scalar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty list"):
            _load_search_space(path)


# ---------------------------------------------------------------------------
# _resolve_scenarios
# ---------------------------------------------------------------------------


class TestResolveScenarios:
    def test_none_returns_defaults(self):
        result = _resolve_scenarios(None, suite="quick")
        assert len(result) == 4
        assert all(isinstance(p, Path) for p in result)

    def test_empty_list_returns_defaults(self):
        result = _resolve_scenarios([], suite="full")
        assert len(result) == 6

    def test_file_path(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("id: test\n", encoding="utf-8")
        result = _resolve_scenarios([str(f)], suite="quick")
        assert len(result) == 1
        assert result[0] == f

    def test_directory_finds_yaml(self, tmp_path):
        sub = tmp_path / "scenarios"
        sub.mkdir()
        (sub / "a.yaml").write_text("id: a\n", encoding="utf-8")
        (sub / "b.yaml").write_text("id: b\n", encoding="utf-8")
        result = _resolve_scenarios([str(sub)], suite="quick")
        assert len(result) == 2

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_scenarios(["/nonexistent/path.yaml"], suite="quick")

    def test_mixed_file_and_dir(self, tmp_path):
        f = tmp_path / "single.yaml"
        f.write_text("id: s\n", encoding="utf-8")
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.yaml").write_text("id: a\n", encoding="utf-8")
        result = _resolve_scenarios([str(f), str(d)], suite="quick")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _write_report
# ---------------------------------------------------------------------------


class TestWriteReport:
    def test_generates_markdown(self, tmp_path):
        report = tmp_path / "report.md"
        _write_report(
            report,
            suite="quick",
            scenarios=[Path("a.yaml"), Path("b.yaml")],
            trials=5,
            seed=42,
            judge_mode="heuristic",
            judge_confidence_min=0.7,
            search_space={"a.b": [1, 2]},
            ranking=[
                {
                    "trial_idx": 0,
                    "aggregate_score": 5.5,
                    "pass_rate": 1.0,
                    "mean_judge_confidence": 0.9,
                    "aegis_realism_score": 0.8,
                    "overrides": {"a.b": 1},
                },
            ],
        )
        content = report.read_text(encoding="utf-8")
        assert "# NavIRL Hyperparameter Tuning Report" in content
        assert "trials: `5`" in content
        assert "seed: `42`" in content
        assert "## Top Trials" in content
        assert "## Best Overrides" in content
        assert "## Reproduction" in content

    def test_empty_ranking(self, tmp_path):
        report = tmp_path / "report.md"
        _write_report(
            report,
            suite="quick",
            scenarios=[],
            trials=1,
            seed=0,
            judge_mode="heuristic",
            judge_confidence_min=0.5,
            search_space={},
            ranking=[],
        )
        content = report.read_text(encoding="utf-8")
        assert "# NavIRL" in content
        # No "Best Overrides" section since ranking is empty
        assert "## Best Overrides" not in content

    def test_aegis_none_renders_dash(self, tmp_path):
        report = tmp_path / "report.md"
        _write_report(
            report,
            suite="quick",
            scenarios=[],
            trials=1,
            seed=0,
            judge_mode="heuristic",
            judge_confidence_min=0.5,
            search_space={},
            ranking=[
                {
                    "trial_idx": 0,
                    "aggregate_score": 1.0,
                    "pass_rate": 0.5,
                    "mean_judge_confidence": 0.7,
                    "overrides": {},
                },
            ],
        )
        content = report.read_text(encoding="utf-8")
        assert "| -" in content or "- |" in content


# ---------------------------------------------------------------------------
# TuningConfig / run_tuning validation
# ---------------------------------------------------------------------------


class TestTuningConfigValidation:
    def test_zero_trials_raises(self, tmp_path):
        config = TuningConfig(out_root=tmp_path, trials=0)
        with pytest.raises(ValueError, match="trials must be positive"):
            run_tuning(config)

    def test_negative_trials_raises(self, tmp_path):
        config = TuningConfig(out_root=tmp_path, trials=-1)
        with pytest.raises(ValueError, match="trials must be positive"):
            run_tuning(config)

    def test_zero_max_frames_raises(self, tmp_path):
        config = TuningConfig(out_root=tmp_path, trials=1, max_frames=0)
        with pytest.raises(ValueError, match="max_frames must be positive"):
            run_tuning(config)
