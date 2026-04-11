"""Extended tests for navirl.pipeline — validation, retry helpers, and expand_state_paths.

Covers uncovered functions: _sanitize_starts_goals, _resample_human_starts_goals_for_retry,
_bump_traversability_offset_for_retry, _human_goal_map, run_scenario_dict validation,
expand_state_paths, run_batch.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from navirl.pipeline import (
    _bump_traversability_offset_for_retry,
    _human_goal_map,
    _resample_human_starts_goals_for_retry,
    _sanitize_starts_goals,
    expand_state_paths,
    run_scenario_dict,
)

# ---------------------------------------------------------------------------
# _sanitize_starts_goals
# ---------------------------------------------------------------------------


class TestSanitizeStartsGoals:
    def _make_backend(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = False
        backend.nearest_clear_point.side_effect = lambda pos, _: pos
        backend.sample_free_point.return_value = (5.0, 5.0)
        return backend

    def test_basic_two_humans_one_robot(self):
        backend = self._make_backend()
        scenario = {
            "humans": {"starts": [], "goals": []},
            "robot": {"start": (0.0, 0.0), "goal": (10.0, 10.0)},
            "_meta": {},
        }
        human_ids = [1, 2]
        human_starts = {1: (1.0, 1.0), 2: (3.0, 3.0)}
        human_goals = {1: (8.0, 8.0), 2: (6.0, 6.0)}

        info = _sanitize_starts_goals(
            scenario=scenario,
            backend=backend,
            human_ids=human_ids,
            human_starts=human_starts,
            human_goals=human_goals,
            human_radius=0.16,
            robot_radius=0.18,
        )

        assert "count" in info
        assert "unresolved_count" in info
        assert info["unresolved_count"] == 0
        # Scenario should be updated with the placed positions
        assert len(scenario["humans"]["starts"]) == 2
        assert len(scenario["humans"]["goals"]) == 2

    def test_updates_scenario_robot_positions(self):
        backend = self._make_backend()
        scenario = {
            "humans": {"starts": [], "goals": []},
            "robot": {"start": (0.0, 0.0), "goal": (10.0, 10.0)},
            "_meta": {},
        }
        _sanitize_starts_goals(
            scenario=scenario,
            backend=backend,
            human_ids=[],
            human_starts={},
            human_goals={},
            human_radius=0.16,
            robot_radius=0.18,
        )
        # Robot positions should still be tuples of floats
        assert isinstance(scenario["robot"]["start"], tuple)
        assert isinstance(scenario["robot"]["goal"], tuple)

    def test_records_adjustments_in_meta(self):
        backend = self._make_backend()
        # Make the backend shift the robot start position
        call_count = [0]

        def shifted_nearest(pos, _):
            call_count[0] += 1
            # Shift the first call slightly
            if call_count[0] == 1:
                return (pos[0] + 0.5, pos[1] + 0.5)
            return pos

        backend.nearest_clear_point.side_effect = shifted_nearest
        scenario = {
            "humans": {"starts": [], "goals": []},
            "robot": {"start": (0.0, 0.0), "goal": (10.0, 10.0)},
            "_meta": {},
        }
        _sanitize_starts_goals(
            scenario=scenario,
            backend=backend,
            human_ids=[],
            human_starts={},
            human_goals={},
            human_radius=0.16,
            robot_radius=0.18,
        )
        assert "anchor_adjustments" in scenario["_meta"]
        assert "anchor_unresolved" in scenario["_meta"]


# ---------------------------------------------------------------------------
# _resample_human_starts_goals_for_retry
# ---------------------------------------------------------------------------


class TestResampleHumanStartsGoalsForRetry:
    def test_resamples_with_valid_backend(self):
        counter = [0]

        def sampler():
            counter[0] += 1
            return (float(counter[0]) * 3.0, float(counter[0]) * 3.0)

        backend = MagicMock()
        backend.sample_free_point = sampler

        scenario = {"humans": {"count": 2, "radius": 0.16, "starts": [], "goals": []}}
        _resample_human_starts_goals_for_retry(scenario, backend)

        assert len(scenario["humans"]["starts"]) == 2
        assert len(scenario["humans"]["goals"]) == 2

    def test_no_humans_is_noop(self):
        backend = MagicMock()
        scenario = {"humans": {"count": 0, "radius": 0.16}}
        _resample_human_starts_goals_for_retry(scenario, backend)
        backend.sample_free_point.assert_not_called()

    def test_negative_count_is_noop(self):
        backend = MagicMock()
        scenario = {"humans": {"count": -1, "radius": 0.16}}
        _resample_human_starts_goals_for_retry(scenario, backend)
        backend.sample_free_point.assert_not_called()

    def test_fallback_when_goal_placement_hard(self):
        """When goals can't satisfy start-goal distance, fallback still produces enough."""
        call_count = [0]

        def sampler():
            call_count[0] += 1
            # Always return same point — goals won't satisfy start-goal distance
            return (0.01 * call_count[0], 0.01 * call_count[0])

        backend = MagicMock()
        backend.sample_free_point = sampler

        scenario = {"humans": {"count": 1, "radius": 0.16, "starts": [], "goals": []}}
        _resample_human_starts_goals_for_retry(scenario, backend)

        assert len(scenario["humans"]["starts"]) == 1
        assert len(scenario["humans"]["goals"]) == 1


# ---------------------------------------------------------------------------
# _bump_traversability_offset_for_retry
# ---------------------------------------------------------------------------


class TestBumpTraversabilityOffset:
    def test_default_bump(self):
        scenario = {}
        result = _bump_traversability_offset_for_retry(scenario)
        assert result >= 0.0
        assert "evaluation" in scenario
        assert "scene" in scenario
        assert scenario["scene"]["orca"]["wall_clearance_buffer_m"] == result
        assert scenario["evaluation"]["wall_clearance_buffer"] == result

    def test_incremental_bumps(self):
        scenario = {
            "evaluation": {
                "traversability_offset_step": 0.01,
                "traversability_offset_max": 0.05,
                "wall_clearance_buffer": 0.01,
            },
            "scene": {"orca": {"wall_clearance_buffer_m": 0.01}},
        }
        result = _bump_traversability_offset_for_retry(scenario)
        assert result == pytest.approx(0.02)

    def test_respects_max_cap(self):
        scenario = {
            "evaluation": {
                "traversability_offset_step": 0.01,
                "traversability_offset_max": 0.02,
                "wall_clearance_buffer": 0.019,
            },
            "scene": {"orca": {"wall_clearance_buffer_m": 0.019}},
        }
        result = _bump_traversability_offset_for_retry(scenario)
        assert result <= 0.02 + 1e-9


# ---------------------------------------------------------------------------
# _human_goal_map
# ---------------------------------------------------------------------------


class TestHumanGoalMap:
    def test_returns_controller_goals_when_dict(self):
        controller = MagicMock()
        controller.goals = {1: (2.0, 3.0), 2: (4.0, 5.0)}
        fallback = {1: (0.0, 0.0), 2: (0.0, 0.0)}
        result = _human_goal_map(controller, fallback)
        assert result[1] == (2.0, 3.0)
        assert result[2] == (4.0, 5.0)

    def test_returns_fallback_when_no_goals_attr(self):
        controller = MagicMock(spec=[])  # No 'goals' attribute
        fallback = {1: (1.0, 2.0)}
        result = _human_goal_map(controller, fallback)
        assert result == fallback

    def test_returns_fallback_when_goals_not_dict(self):
        controller = MagicMock()
        controller.goals = [(1.0, 2.0)]  # List, not dict
        fallback = {1: (1.0, 2.0)}
        result = _human_goal_map(controller, fallback)
        assert result == fallback

    def test_returns_fallback_when_goals_is_none(self):
        controller = MagicMock()
        controller.goals = None
        fallback = {1: (1.0, 2.0)}
        result = _human_goal_map(controller, fallback)
        assert result == fallback


# ---------------------------------------------------------------------------
# run_scenario_dict — validation paths (no full simulation)
# ---------------------------------------------------------------------------


class TestRunScenarioDictValidation:
    def test_rejects_non_dict(self):
        with pytest.raises(TypeError, match="must be a dictionary"):
            run_scenario_dict("not_a_dict", "/tmp/out")

    def test_rejects_empty_dict(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            run_scenario_dict({}, "/tmp/out")

    def test_rejects_missing_required_fields(self):
        with pytest.raises(ValueError, match="missing required fields"):
            run_scenario_dict({"id": "test"}, "/tmp/out")

    def test_rejects_non_string_id(self):
        scenario = {
            "id": 123,
            "seed": 42,
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
        }
        with pytest.raises(ValueError, match="non-empty string"):
            run_scenario_dict(scenario, "/tmp/out")

    def test_rejects_empty_string_id(self):
        scenario = {
            "id": "",
            "seed": 42,
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
        }
        with pytest.raises(ValueError, match="non-empty string"):
            run_scenario_dict(scenario, "/tmp/out")

    def test_rejects_negative_seed(self):
        scenario = {
            "id": "test",
            "seed": -1,
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
            "evaluation": {},
        }
        with pytest.raises(ValueError, match=r"(?i)seed"):
            run_scenario_dict(scenario, "/tmp/out")

    def test_rejects_invalid_seed_type(self):
        scenario = {
            "id": "test",
            "seed": "not_a_number",
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
            "evaluation": {},
        }
        with pytest.raises(ValueError, match=r"(?i)seed"):
            run_scenario_dict(scenario, "/tmp/out")

    def test_rejects_bad_evaluation_type(self):
        scenario = {
            "id": "test",
            "seed": 42,
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
            "evaluation": "not_a_dict",
        }
        with pytest.raises(ValueError, match="evaluation"):
            run_scenario_dict(scenario, "/tmp/out")

    def test_rejects_bad_meta_type(self):
        scenario = {
            "id": "test",
            "seed": 42,
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
            "evaluation": {},
            "_meta": "not_a_dict",
        }
        with pytest.raises(ValueError, match="_meta"):
            run_scenario_dict(scenario, "/tmp/out")

    def test_rejects_negative_deadlock_resample_attempts(self):
        scenario = {
            "id": "test",
            "seed": 42,
            "humans": {},
            "robot": {},
            "scene": {},
            "horizon": {},
            "evaluation": {"deadlock_resample_attempts": -1},
        }
        with pytest.raises(ValueError, match="deadlock_resample_attempts"):
            run_scenario_dict(scenario, "/tmp/out")


# ---------------------------------------------------------------------------
# expand_state_paths
# ---------------------------------------------------------------------------


class TestExpandStatePaths:
    def test_empty_input(self):
        assert expand_state_paths([]) == []

    def test_direct_file(self, tmp_path):
        f = tmp_path / "state.jsonl"
        f.write_text("{}\n")
        result = expand_state_paths([str(f)])
        assert len(result) == 1
        assert result[0] == f.resolve()

    def test_nonexistent_file_skipped(self, tmp_path):
        result = expand_state_paths([str(tmp_path / "nonexistent.jsonl")])
        assert len(result) == 0

    def test_directory_with_state_file(self, tmp_path):
        f = tmp_path / "state.jsonl"
        f.write_text("{}\n")
        result = expand_state_paths([str(tmp_path)])
        assert len(result) == 1
        assert result[0] == f.resolve()

    def test_directory_with_bundle_subdir(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        f = bundle / "state.jsonl"
        f.write_text("{}\n")
        result = expand_state_paths([str(tmp_path)])
        assert len(result) == 1
        assert result[0] == f.resolve()

    def test_directory_recursive_search(self, tmp_path):
        nested = tmp_path / "run1" / "deep"
        nested.mkdir(parents=True)
        f = nested / "state.jsonl"
        f.write_text("{}\n")
        result = expand_state_paths([str(tmp_path)])
        assert len(result) == 1
        assert result[0] == f.resolve()

    def test_deduplication(self, tmp_path):
        f = tmp_path / "state.jsonl"
        f.write_text("{}\n")
        result = expand_state_paths([str(f), str(f)])
        assert len(result) == 1

    def test_glob_pattern(self, tmp_path, monkeypatch):
        # Create state files in tmp_path
        d1 = tmp_path / "run1"
        d1.mkdir()
        f1 = d1 / "state.jsonl"
        f1.write_text("{}\n")
        d2 = tmp_path / "run2"
        d2.mkdir()
        f2 = d2 / "state.jsonl"
        f2.write_text("{}\n")

        # expand_state_paths uses Path().glob() which is relative to cwd
        monkeypatch.chdir(tmp_path)
        result = expand_state_paths(["run*/state.jsonl"])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


class TestRunBatch:
    def test_no_scenarios_raises(self, tmp_path):
        args = argparse.Namespace(
            scenarios=str(tmp_path),
            seeds="42",
            out=str(tmp_path / "out"),
            render=False,
            video=False,
            parallel=1,
        )
        with pytest.raises(FileNotFoundError, match="No scenario YAML"):
            from navirl.pipeline import run_batch

            run_batch(args)

    @patch("navirl.pipeline._run_scenario_worker")
    def test_sequential_execution(self, mock_worker, tmp_path):
        # Create a dummy scenario file
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "test.yaml").write_text("id: test\n")

        mock_log = MagicMock()
        mock_worker.return_value = mock_log

        args = argparse.Namespace(
            scenarios=str(scenario_dir),
            seeds="42,43",
            out=str(tmp_path / "out"),
            render=False,
            video=False,
            parallel=1,
        )
        from navirl.pipeline import run_batch

        logs = run_batch(args)
        assert len(logs) == 2
        assert mock_worker.call_count == 2

    @patch("navirl.pipeline.mp")
    @patch("navirl.pipeline.load_scenario")
    def test_parallel_execution(self, mock_load, mock_mp, tmp_path):
        # Create scenario files
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "test.yaml").write_text("id: test\n")

        mock_pool = MagicMock()
        mock_mp.Pool.return_value.__enter__ = MagicMock(return_value=mock_pool)
        mock_mp.Pool.return_value.__exit__ = MagicMock(return_value=False)
        mock_mp.cpu_count.return_value = 4
        mock_pool.map.return_value = [MagicMock(), MagicMock()]

        args = argparse.Namespace(
            scenarios=str(scenario_dir),
            seeds="42,43",
            out=str(tmp_path / "out"),
            render=False,
            video=False,
            parallel=4,
        )
        from navirl.pipeline import run_batch

        logs = run_batch(args)
        assert len(logs) == 2
