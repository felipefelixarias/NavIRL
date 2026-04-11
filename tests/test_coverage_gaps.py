"""Tests targeting coverage gaps in artifacts, robots/base, core/env, overseer/rerank,
packs/runner, and experiments/runner modules."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# artifacts.py coverage
# ---------------------------------------------------------------------------
from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours


class TestResolveRetentionHoursEdgeCases:
    """Cover lines 25, 34-36, 39-40 of artifacts.py."""

    def test_negative_requested_hours_raises(self):
        with pytest.raises(ValueError, match="retention hours must be >= 0"):
            resolve_retention_hours(-1.0, env_var="X", default_hours=24.0)

    def test_env_var_non_numeric_raises(self):
        with mock.patch.dict(os.environ, {"TEST_RET": "abc"}):
            with pytest.raises(ValueError, match="must be a number of hours"):
                resolve_retention_hours(None, env_var="TEST_RET", default_hours=24.0)

    def test_env_var_negative_raises(self):
        with mock.patch.dict(os.environ, {"TEST_RET": "-5"}):
            with pytest.raises(ValueError, match="must be >= 0"):
                resolve_retention_hours(None, env_var="TEST_RET", default_hours=24.0)

    def test_env_var_valid(self):
        with mock.patch.dict(os.environ, {"TEST_RET": "48"}):
            assert resolve_retention_hours(None, env_var="TEST_RET", default_hours=24.0) == 48.0

    def test_default_when_no_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MISSING_VAR", None)
            assert resolve_retention_hours(None, env_var="MISSING_VAR", default_hours=12.0) == 12.0


class TestPruneOldRunDirs:
    """Cover lines 53, 62, 67-68, 72, 86-87 of artifacts.py."""

    def test_empty_dir_returns_empty(self, tmp_path):
        result = prune_old_run_dirs(tmp_path, ttl_hours=1.0)
        assert result == []

    def test_symlink_filtered(self, tmp_path):
        real = tmp_path / "run_real"
        real.mkdir()
        link = tmp_path / "run_link"
        link.symlink_to(real)
        # Touch the real dir to make it old
        old_time = time.time() - 7200
        os.utime(real, (old_time, old_time))
        result = prune_old_run_dirs(tmp_path, ttl_hours=0.5, prefixes=("run_",))
        # Only real dir should be pruned, symlink ignored
        assert real in result
        assert link not in result

    def test_stat_oserror_skipped(self, tmp_path):
        d = tmp_path / "run_001"
        d.mkdir()
        original_stat = Path.stat
        call_count = {}

        def patched_stat(self_path, *args, **kwargs):
            if self_path == d:
                call_count.setdefault(d, 0)
                call_count[d] += 1
                # Let is_dir() succeed (first calls), fail on explicit stat()
                if call_count[d] > 2:
                    raise OSError("perm")
            return original_stat(self_path, *args, **kwargs)

        with mock.patch.object(Path, "stat", patched_stat):
            result = prune_old_run_dirs(tmp_path, ttl_hours=0.5)
        assert result == []

    def test_rmtree_oserror_skipped(self, tmp_path):
        d = tmp_path / "run_001"
        d.mkdir()
        old_time = time.time() - 7200
        os.utime(d, (old_time, old_time))
        with mock.patch("navirl.artifacts.shutil.rmtree", side_effect=OSError("perm")):
            result = prune_old_run_dirs(tmp_path, ttl_hours=0.5)
        assert result == []

    def test_nonexistent_root(self):
        result = prune_old_run_dirs("/nonexistent/path/xyz", ttl_hours=1.0)
        assert result == []

    def test_ttl_none_returns_empty(self, tmp_path):
        d = tmp_path / "run_001"
        d.mkdir()
        assert prune_old_run_dirs(tmp_path, ttl_hours=None) == []

    def test_keep_latest(self, tmp_path):
        dirs = []
        for i in range(3):
            d = tmp_path / f"run_{i:03d}"
            d.mkdir()
            t = time.time() - (3 - i) * 3600  # older to newer
            os.utime(d, (t, t))
            dirs.append(d)
        # Keep latest 1, ttl 0.5 hours => prune the 2 oldest
        result = prune_old_run_dirs(tmp_path, ttl_hours=0.5, keep_latest=1)
        # newest dir kept
        assert dirs[2] not in result
        assert len(result) == 2


# ---------------------------------------------------------------------------
# core/env.py coverage — abstract methods and base implementations
# ---------------------------------------------------------------------------
from navirl.core.env import SceneBackend


class ConcreteBackend(SceneBackend):
    """Minimal concrete implementation for testing base class methods."""

    def add_agent(self, agent_id, position, radius, max_speed, kind):
        pass

    def set_preferred_velocity(self, agent_id, velocity):
        pass

    def step(self):
        pass

    def get_position(self, agent_id):
        return (0.0, 0.0)

    def get_velocity(self, agent_id):
        return (0.0, 0.0)

    def shortest_path(self, start, goal):
        return [start, goal]

    def sample_free_point(self):
        return (0.0, 0.0)

    def check_obstacle_collision(self, position, radius):
        return False

    def world_to_map(self, position):
        return (0, 0)

    def map_image(self):
        import numpy as np

        return np.zeros((10, 10), dtype=np.uint8)


class TestSceneBackendBase:
    """Cover lines 85 and 103 of core/env.py."""

    def test_nearest_clear_point_returns_float_tuple(self):
        b = ConcreteBackend()
        result = b.nearest_clear_point((1, 2), 0.5)
        assert result == (1.0, 2.0)
        assert all(isinstance(v, float) for v in result)

    def test_map_metadata_returns_empty_dict(self):
        b = ConcreteBackend()
        assert b.map_metadata() == {}


# ---------------------------------------------------------------------------
# robots/base.py coverage
# ---------------------------------------------------------------------------
from navirl.core.types import Action
from navirl.robots.base import RobotController


class DummyRobot(RobotController):
    """Concrete robot for testing base class validation."""

    def reset(self, robot_id, start, goal, backend):
        super().reset(robot_id, start, goal, backend)

    def step(self, step, time_s, dt, states, emit_event):
        super().step(step, time_s, dt, states, emit_event)
        return Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")


def _noop_emit(event_type, agent_id=None, data=None):
    pass


class TestRobotControllerResetValidation:
    """Cover lines 83, 86, 89, 95-96, 100, 105 of robots/base.py."""

    def test_negative_robot_id(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="non-negative integer"):
            r.reset(-1, (0, 0), (1, 1), None)

    def test_invalid_start_not_tuple(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="Invalid start position"):
            r.reset(0, "bad", (1, 1), None)

    def test_invalid_goal_not_tuple(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="Invalid goal position"):
            r.reset(0, (0, 0), 42, None)

    def test_non_numeric_coordinates(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="must be numeric"):
            r.reset(0, ("a", "b"), (1, 1), None)

    def test_nan_coordinates(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="finite"):
            r.reset(0, (float("nan"), 0), (1, 1), None)

    def test_inf_coordinates(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="finite"):
            r.reset(0, (float("inf"), 0), (1, 1), None)

    def test_too_large_coordinates(self):
        r = DummyRobot()
        with pytest.raises(ValueError, match="too large"):
            r.reset(0, (2e6, 0), (1, 1), None)


class TestRobotControllerStepValidation:
    """Cover lines 138, 141, 144, 148, 151, 154 of robots/base.py."""

    def _setup_robot(self):
        r = DummyRobot()
        r.reset(0, (0, 0), (5, 5), None)
        return r

    def test_negative_step(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="non-negative integer"):
            r.step(-1, 0.0, 0.1, {}, _noop_emit)

    def test_negative_time(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="non-negative"):
            r.step(0, -1.0, 0.1, {}, _noop_emit)

    def test_zero_dt(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="positive"):
            r.step(0, 0.0, 0.0, {}, _noop_emit)

    def test_dt_too_small(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="reasonable bounds"):
            r.step(0, 0.0, 1e-9, {}, _noop_emit)

    def test_dt_too_large(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="reasonable bounds"):
            r.step(0, 0.0, 20.0, {}, _noop_emit)

    def test_states_not_dict(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="dictionary"):
            r.step(0, 0.0, 0.1, [], _noop_emit)

    def test_emit_not_callable(self):
        r = self._setup_robot()
        with pytest.raises(ValueError, match="callable"):
            r.step(0, 0.0, 0.1, {}, "not_callable")


class TestRobotControllerValidateAction:
    """Cover lines 170-171, 179-180, 187 of robots/base.py."""

    def test_non_action_returns_stop(self):
        r = DummyRobot()
        result = r.validate_action("not_an_action")
        assert result.pref_vx == 0.0
        assert result.pref_vy == 0.0

    def test_huge_velocity_returns_stop(self):
        r = DummyRobot()
        action = Action(pref_vx=1e7, pref_vy=1e7, behavior="GO")
        result = r.validate_action(action)
        assert result.pref_vx == 0.0  # STOP action
        assert result.pref_vy == 0.0

    def test_velocity_clamped(self):
        r = DummyRobot()
        action = Action(pref_vx=10.0, pref_vy=-10.0, behavior="GO")
        result = r.validate_action(action)
        max_speed = r.cfg.get("max_speed", 5.0)
        assert result.pref_vx == max_speed
        assert result.pref_vy == -max_speed

    def test_within_limits_unchanged(self):
        r = DummyRobot()
        action = Action(pref_vx=1.0, pref_vy=-1.0, behavior="GO")
        result = r.validate_action(action)
        assert result.pref_vx == 1.0
        assert result.pref_vy == -1.0


class TestRobotControllerPerformance:
    """Cover lines 208-225 of robots/base.py."""

    def test_slow_step_logs_warning(self):
        r = DummyRobot()
        r.reset(0, (0, 0), (5, 5), None)
        r._step_count = 1
        with mock.patch("navirl.robots.base.logger") as mock_logger:
            r.check_computational_performance(0.5)
            mock_logger.warning.assert_called_once()

    def test_consistently_slow_logs_info(self):
        r = DummyRobot()
        r.reset(0, (0, 0), (5, 5), None)
        r._step_count = 20  # multiple of 10, > 10
        r._last_computation_time = r._max_computation_time * 0.9
        with mock.patch("navirl.robots.base.logger") as mock_logger:
            r.check_computational_performance(r._max_computation_time * 0.9)
            # Should get both warning (> max) and info (consistently slow)
            # Actually 0.9 * max < max, so no warning; check info
            # 0.9 * 0.1 = 0.09, which is < 0.1, so no warning
            pass
        # Force consistently slow condition: step_count=20, comp_time > 0.8*max
        r._step_count = 20
        with mock.patch("navirl.robots.base.logger") as mock_logger:
            r.check_computational_performance(r._max_computation_time * 0.85)
            mock_logger.info.assert_called_once()

    def test_config_validation_error(self):
        from navirl.core.plugin_validation import ConfigValidationError

        with mock.patch(
            "navirl.robots.base.validate_controller_config",
            side_effect=ConfigValidationError("bad config"),
        ):
            with pytest.raises(ConfigValidationError):
                DummyRobot()


# ---------------------------------------------------------------------------
# overseer/rerank.py coverage
# ---------------------------------------------------------------------------
from navirl.overseer.rerank import (
    _parse_vlm_ranking,
    _rerank_prompt,
    _scenario_realism_score,
    _trial_realism_score,
    run_aegis_rerank,
)


class TestTrialRealismScoreEdge:
    """Cover line 61 of rerank.py."""

    def test_empty_scenarios_returns_negative(self):
        assert _trial_realism_score({"scenarios": []}) == -100.0

    def test_no_scenarios_key(self):
        assert _trial_realism_score({}) == -100.0


class TestParseVlmRanking:
    """Cover lines 83-94 of rerank.py."""

    def test_valid_ranking(self):
        result = _parse_vlm_ranking({"ranking": [0, 2, 1]}, {0, 1, 2})
        assert result == [0, 2, 1]

    def test_ranking_not_list(self):
        assert _parse_vlm_ranking({"ranking": "bad"}, {0, 1}) == []

    def test_missing_ranking_key(self):
        assert _parse_vlm_ranking({}, {0, 1}) == []

    def test_invalid_values_skipped(self):
        result = _parse_vlm_ranking({"ranking": [0, "bad", None, 1]}, {0, 1})
        assert result == [0, 1]

    def test_out_of_range_skipped(self):
        result = _parse_vlm_ranking({"ranking": [0, 99, 1]}, {0, 1})
        assert result == [0, 1]

    def test_duplicates_removed(self):
        result = _parse_vlm_ranking({"ranking": [0, 0, 1]}, {0, 1})
        assert result == [0, 1]


class TestRunAegisRerankVlm:
    """Cover lines 165-166, 180-186 of rerank.py."""

    def _make_trial(self, idx, score=0.5):
        return {
            "trial_idx": idx,
            "aggregate_score": score,
            "pass_rate": 1.0,
            "mean_judge_confidence": 0.8,
            "scenarios": [
                {
                    "scenario_id": "s1",
                    "judge_confidence": 0.8,
                    "invariants_pass": True,
                    "judge_pass": True,
                    "metrics": {
                        "success_rate": 1.0,
                        "collisions_agent_agent": 0,
                        "collisions_agent_obstacle": 0,
                        "deadlock_count": 0,
                        "intrusion_rate": 0,
                        "jerk_proxy": 0,
                        "oscillation_score": 0,
                    },
                }
            ],
        }

    def test_vlm_mode_with_provider_ranking(self):
        trials = [self._make_trial(i) for i in range(3)]
        mock_config = mock.MagicMock()

        with mock.patch(
            "navirl.overseer.rerank.run_structured_vlm",
            return_value={"ranking": [2, 0, 1]},
        ):
            result = run_aegis_rerank(
                trials,
                mode="vlm",
                provider_config=mock_config,
                top_k=3,
            )

        assert result["provider_used"] is True
        assert result["provider_ranking"] == [2, 0, 1]
        assert result["status"] == "ok"
        # Blended scores should have boost applied
        assert result["blended_scores"][2] > result["heuristic_scores"][2]

    def test_vlm_mode_provider_error_fallback(self):
        from navirl.overseer.provider import ProviderUnavailableError

        trials = [self._make_trial(0)]
        mock_config = mock.MagicMock()

        with mock.patch(
            "navirl.overseer.rerank.run_structured_vlm",
            side_effect=ProviderUnavailableError("no provider"),
        ):
            result = run_aegis_rerank(
                trials,
                mode="vlm",
                provider_config=mock_config,
                allow_fallback=True,
            )

        assert result["provider_used"] is False
        assert "no provider" in result["provider_error"]
        assert result["status"] == "ok"

    def test_vlm_mode_no_fallback(self):
        from navirl.overseer.provider import ProviderCallError

        trials = [self._make_trial(0)]
        mock_config = mock.MagicMock()

        with mock.patch(
            "navirl.overseer.rerank.run_structured_vlm",
            side_effect=ProviderCallError("fail"),
        ):
            result = run_aegis_rerank(
                trials,
                mode="vlm",
                provider_config=mock_config,
                allow_fallback=False,
            )

        assert result["applied"] is False
        assert result["status"] == "needs_human_review"

    def test_vlm_boost_single_ranking(self):
        """When only 1 trial in ranking, boost should be 1.0."""
        trials = [self._make_trial(0)]
        mock_config = mock.MagicMock()

        with mock.patch(
            "navirl.overseer.rerank.run_structured_vlm",
            return_value={"ranking": [0]},
        ):
            result = run_aegis_rerank(
                trials,
                mode="vlm",
                provider_config=mock_config,
                top_k=1,
            )

        expected_boost = 1.0
        assert result["blended_scores"][0] == pytest.approx(
            result["heuristic_scores"][0] + expected_boost
        )


# ---------------------------------------------------------------------------
# packs/runner.py coverage (via mocking)
# ---------------------------------------------------------------------------
from navirl.packs.runner import run_pack
from navirl.packs.schema import PackManifest, PackScenarioEntry


class TestRunPackIntegration:
    """Cover lines 45-105 of packs/runner.py."""

    def test_successful_run(self, tmp_path):
        manifest = PackManifest(
            name="test-pack",
            version="1.0",
            scenarios=[PackScenarioEntry(id="s1", path="fake.yaml", seeds=[42])],
        )

        mock_episode = mock.MagicMock()
        state_dir = tmp_path / "bundle"
        state_dir.mkdir()
        state_file = state_dir / "state.npz"
        state_file.touch()
        scenario_yaml = state_dir / "scenario.yaml"
        scenario_yaml.write_text("name: test\n")
        mock_episode.state_path = str(state_file)

        with (
            mock.patch("navirl.packs.runner.load_scenario", return_value={"name": "test"}),
            mock.patch("navirl.packs.runner.run_scenario_dict", return_value=mock_episode),
            mock.patch(
                "navirl.packs.runner.StandardMetrics.compute",
                return_value={"success_rate": 1.0},
            ),
        ):
            result = run_pack(manifest, tmp_path / "out")

        assert result.manifest_name == "test-pack"
        assert len(result.runs) == 1
        assert result.runs[0].status == "completed"
        assert result.runs[0].metrics == {"success_rate": 1.0}

    def test_failed_run(self, tmp_path):
        manifest = PackManifest(
            name="test-pack",
            version="1.0",
            scenarios=[PackScenarioEntry(id="s1", path="fake.yaml", seeds=[42])],
        )

        with mock.patch(
            "navirl.packs.runner.load_scenario",
            side_effect=FileNotFoundError("not found"),
        ):
            result = run_pack(manifest, tmp_path / "out")

        assert len(result.runs) == 1
        assert result.runs[0].status == "failed"
        assert "not found" in result.runs[0].error


# ---------------------------------------------------------------------------
# experiments/runner.py coverage (via mocking)
# ---------------------------------------------------------------------------
from navirl.experiments.runner import _apply_overrides, run_batch_template
from navirl.experiments.templates import BatchTemplate


class TestRunBatchTemplateIntegration:
    """Cover lines 75-145 of experiments/runner.py."""

    def test_successful_batch(self, tmp_path):
        scenario_path = tmp_path / "scenarios" / "test.yaml"
        scenario_path.parent.mkdir(parents=True)
        scenario_path.write_text("name: test\n")

        template = BatchTemplate(
            name="test-batch",
            scenarios=[str(scenario_path)],
            seeds=[42],
        )

        mock_episode = mock.MagicMock()
        state_dir = tmp_path / "bundle"
        state_dir.mkdir()
        state_file = state_dir / "state.npz"
        state_file.touch()
        scenario_yaml = state_dir / "scenario.yaml"
        scenario_yaml.write_text("name: test\n")
        mock_episode.state_path = str(state_file)

        with (
            mock.patch("navirl.experiments.runner.load_scenario", return_value={"name": "test"}),
            mock.patch("navirl.experiments.runner.run_scenario_dict", return_value=mock_episode),
            mock.patch(
                "navirl.experiments.runner.StandardMetrics.compute",
                return_value={"success_rate": 1.0},
            ),
        ):
            summary = run_batch_template(template, tmp_path / "out")

        assert summary is not None
        assert (tmp_path / "out" / "summary.json").exists()
        assert (tmp_path / "out" / "REPORT.md").exists()

    def test_failed_batch_task(self, tmp_path):
        scenario_path = tmp_path / "scenarios" / "test.yaml"
        scenario_path.parent.mkdir(parents=True)
        scenario_path.write_text("name: test\n")

        template = BatchTemplate(
            name="test-batch",
            scenarios=[str(scenario_path)],
            seeds=[42],
        )

        with mock.patch(
            "navirl.experiments.runner.load_scenario",
            side_effect=RuntimeError("scenario load failed"),
        ):
            summary = run_batch_template(template, tmp_path / "out")

        assert summary is not None
        assert summary.failed_runs == 1
