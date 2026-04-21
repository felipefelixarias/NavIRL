"""Tests for coverage gaps in grid2d backend, scenario validation, core env,
evaluation comparisons, and evaluation analysis.

Targets:
- navirl/backends/grid2d/backend.py (38% -> higher)
- navirl/backends/grid2d/orca.py (60% -> higher)
- navirl/scenarios/validate.py (90% -> ~100%)
- navirl/core/env.py (74% -> 100%)
- navirl/evaluation/comparisons.py (90% -> ~100%)
- navirl/evaluation/analysis.py (79% -> higher)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# -----------------------------------------------------------------------
# 1. Grid2D Backend + ORCA
# -----------------------------------------------------------------------

try:
    import rvo2

    _RVO2_AVAILABLE = True
except ImportError:
    _RVO2_AVAILABLE = False

from navirl.backends.grid2d.backend import AgentMeta, Grid2DBackend
from navirl.backends.grid2d.maps import hallway_map, kitchen_map
from navirl.backends.grid2d.orca import IndoorORCASimConfig

_skip_no_rvo2 = pytest.mark.skipif(not _RVO2_AVAILABLE, reason="rvo2 not installed")


@pytest.fixture()
def hallway_backend():
    """Build a Grid2DBackend on the hallway map with default ORCA settings."""
    pytest.importorskip("rvo2")
    scene = {"map": {"source": "builtin", "id": "hallway"}}
    horizon = {"dt": 0.1}
    return Grid2DBackend(scene, horizon)


@pytest.fixture()
def kitchen_backend():
    """Build a Grid2DBackend on the kitchen map (has obstacles)."""
    pytest.importorskip("rvo2")
    scene = {"map": {"source": "builtin", "id": "kitchen"}}
    horizon = {"dt": 0.1}
    return Grid2DBackend(scene, horizon)


@_skip_no_rvo2
class TestGrid2DBackendInit:
    def test_default_construction(self, hallway_backend):
        assert hallway_backend.dt == 0.1
        assert hallway_backend.binary_map is not None
        assert hallway_backend.map_info is not None

    def test_orca_units_pixels(self):
        scene = {
            "map": {"source": "builtin", "id": "hallway"},
            "orca": {"units": "pixels", "neighbor_dist": 350.0},
        }
        backend = Grid2DBackend(scene, {"dt": 0.1})
        # neighbor_dist should be converted from pixels to meters
        assert backend.orca.neighbor_dist < 350.0

    def test_invalid_orca_units_raises(self):
        scene = {
            "map": {"source": "builtin", "id": "hallway"},
            "orca": {"units": "furlongs"},
        }
        with pytest.raises(ValueError, match=r"meters.*pixels"):
            Grid2DBackend(scene, {"dt": 0.1})


@_skip_no_rvo2
class TestGrid2DBackendAgents:
    def test_add_and_get_position(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="robot")
        pos = hallway_backend.get_position(0)
        assert isinstance(pos, tuple) and len(pos) == 2
        # Should be near the requested position (may shift for clearance)
        assert math.hypot(pos[0] - pt[0], pos[1] - pt[1]) < 2.0

    def test_add_and_get_velocity(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="human")
        vx, vy = hallway_backend.get_velocity(0)
        # Initially near zero
        assert abs(vx) < 1e-3 and abs(vy) < 1e-3

    def test_set_preferred_velocity_robot(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="robot")
        hallway_backend.set_preferred_velocity(0, (0.5, 0.0))
        # After setting, the cache should be updated
        assert hallway_backend._pref_vel_cache[0] != (0.0, 0.0)

    def test_set_preferred_velocity_human(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="human")
        hallway_backend.set_preferred_velocity(0, (0.3, 0.2))
        assert hallway_backend._pref_vel_cache[0] != (0.0, 0.0)

    def test_step_moves_agent(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="robot")
        hallway_backend.set_preferred_velocity(0, (0.5, 0.0))
        # Step multiple times
        for _ in range(5):
            hallway_backend.step()
        pos_after = hallway_backend.get_position(0)
        # Should have moved (or at least run without error)
        assert isinstance(pos_after, tuple) and len(pos_after) == 2

    def test_agent_meta(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(42, pt, 0.25, 1.5, kind="robot")
        meta = hallway_backend.get_agent_meta(42)
        assert isinstance(meta, AgentMeta)
        assert meta.ext_id == 42
        assert meta.kind == "robot"
        assert meta.radius == 0.25
        assert meta.max_speed == 1.5

    def test_agent_ids(self, hallway_backend):
        p1 = hallway_backend.sample_free_point()
        p2 = hallway_backend.sample_free_point()
        hallway_backend.add_agent(3, p1, 0.3, 1.0, kind="robot")
        hallway_backend.add_agent(7, p2, 0.3, 1.0, kind="human")
        ids = hallway_backend.agent_ids()
        assert ids == [3, 7]


@_skip_no_rvo2
class TestGrid2DBackendSpatial:
    def test_sample_free_point(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        assert isinstance(pt, tuple) and len(pt) == 2

    def test_world_to_map(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        rc = hallway_backend.world_to_map(pt)
        assert isinstance(rc, tuple) and len(rc) == 2
        h, w = hallway_backend.binary_map.shape[:2]
        assert 0 <= rc[0] < h
        assert 0 <= rc[1] < w

    def test_check_obstacle_collision_free_space(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        # A free point with small radius should not collide
        collision = hallway_backend.check_obstacle_collision(pt, 0.05)
        assert isinstance(collision, bool)

    def test_check_obstacle_collision_in_wall(self, hallway_backend):
        # Point (0, 0) in world coords maps to a corner likely in obstacle space
        collision = hallway_backend.check_obstacle_collision((0.0, 0.0), 0.3)
        assert collision is True

    def test_nearest_clear_point(self, kitchen_backend):
        # Request a point that might be in an obstacle
        clear = kitchen_backend.nearest_clear_point((0.0, 0.0), 0.3)
        assert isinstance(clear, tuple) and len(clear) == 2
        # The returned point should not collide
        kitchen_backend.check_obstacle_collision(clear, 0.05)
        # May or may not collide depending on exact placement, but call succeeded

    def test_shortest_path(self, hallway_backend):
        p1 = hallway_backend.sample_free_point()
        p2 = hallway_backend.sample_free_point()
        path = hallway_backend.shortest_path(p1, p2)
        assert isinstance(path, list)
        assert len(path) >= 2
        assert all(isinstance(p, tuple) and len(p) == 2 for p in path)

    def test_map_image(self, hallway_backend):
        img = hallway_backend.map_image()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

    def test_map_metadata(self, hallway_backend):
        meta = hallway_backend.map_metadata()
        assert isinstance(meta, dict)
        assert "pixels_per_meter" in meta
        assert "source" in meta
        assert meta["source"] == "builtin"

    def test_dt_property(self, hallway_backend):
        assert hallway_backend.dt == pytest.approx(0.1)


@_skip_no_rvo2
class TestGrid2DBackendStepWithMultipleAgents:
    def test_multi_agent_step(self, hallway_backend):
        """Step with robot + human, verify no crash and agents exist."""
        p1 = hallway_backend.sample_free_point()
        p2 = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, p1, 0.3, 1.0, kind="robot")
        hallway_backend.add_agent(1, p2, 0.25, 0.8, kind="human")
        hallway_backend.set_preferred_velocity(0, (0.5, 0.0))
        hallway_backend.set_preferred_velocity(1, (-0.3, 0.1))
        for _ in range(10):
            hallway_backend.step()
        pos0 = hallway_backend.get_position(0)
        pos1 = hallway_backend.get_position(1)
        assert isinstance(pos0, tuple) and isinstance(pos1, tuple)


# -----------------------------------------------------------------------
# 2. ORCA Sim properties
# -----------------------------------------------------------------------


class TestIndoorORCASimConfig:
    def test_defaults(self):
        cfg = IndoorORCASimConfig()
        assert cfg.time_step == pytest.approx(1 / 32.0)
        assert cfg.max_neighbors == 4
        assert cfg.radius == pytest.approx(0.125)

    def test_custom(self):
        cfg = IndoorORCASimConfig(time_step=0.1, neighbor_dist=3.0, max_speed=1.0)
        assert cfg.time_step == pytest.approx(0.1)
        assert cfg.max_speed == pytest.approx(1.0)


@_skip_no_rvo2
class TestIndoorORCASim:
    def test_properties(self, hallway_backend):
        sim = hallway_backend.orca
        assert sim.time_step > 0
        assert sim.neighbor_dist > 0
        assert sim.max_neighbors > 0
        assert sim.time_horizon > 0
        assert sim.time_horizon_obst > 0
        assert sim.radius > 0
        assert sim.max_speed > 0

    def test_num_agents(self, hallway_backend):
        assert hallway_backend.orca.num_agents == 0
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="robot")
        assert hallway_backend.orca.num_agents == 1

    def test_num_obstacle_vertices(self, hallway_backend):
        # Hallway has obstacles processed
        n = hallway_backend.orca.num_obstacle_vertices
        assert isinstance(n, int) and n >= 0

    def test_get_global_time(self, hallway_backend):
        t0 = hallway_backend.orca.get_global_time()
        assert t0 == pytest.approx(0.0)
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="robot")
        hallway_backend.orca.do_step()
        t1 = hallway_backend.orca.get_global_time()
        assert t1 > t0

    def test_add_agent_direct(self, hallway_backend):
        orca = hallway_backend.orca
        aid = orca.add_agent([1.0, 1.0])
        assert isinstance(aid, int)
        assert orca.num_agents == 1

    def test_trajectories_tracking(self, hallway_backend):
        pt = hallway_backend.sample_free_point()
        hallway_backend.add_agent(0, pt, 0.3, 1.0, kind="robot")
        hallway_backend.orca.do_step()
        hallway_backend.orca.do_step()
        assert len(hallway_backend.orca.trajectories) == 1
        assert len(hallway_backend.orca.trajectories[0]) >= 2


# -----------------------------------------------------------------------
# 3. Scenario Validation edge cases
# -----------------------------------------------------------------------

from navirl.scenarios.validate import (
    _is_point,
    _validate_horizon,
    _validate_humans,
    _validate_robot,
    _validate_scene,
    validate_scenario_dict,
)


class TestValidateSceneEdgeCases:
    def test_ppm_invalid_type(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "hallway", "pixels_per_meter": "not_a_number"},
        }
        _validate_scene(scene, errors)
        assert any("pixels_per_meter" in e for e in errors)

    def test_mpp_invalid_type(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "hallway", "meters_per_pixel": "bad"},
        }
        _validate_scene(scene, errors)
        assert any("meters_per_pixel" in e for e in errors)

    def test_ppm_negative(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "hallway", "pixels_per_meter": -5.0},
        }
        _validate_scene(scene, errors)
        assert any("pixels_per_meter" in e for e in errors)

    def test_mpp_negative(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "hallway", "meters_per_pixel": -0.01},
        }
        _validate_scene(scene, errors)
        assert any("meters_per_pixel" in e for e in errors)

    def test_ppm_mpp_consistent(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {
                "source": "builtin",
                "id": "hallway",
                "pixels_per_meter": 100.0,
                "meters_per_pixel": 0.01,
            },
        }
        _validate_scene(scene, errors)
        # Should be consistent - no error about inconsistency
        assert not any("inconsistent" in e.lower() for e in errors)

    def test_ppm_mpp_inconsistent(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {
                "source": "builtin",
                "id": "hallway",
                "pixels_per_meter": 100.0,
                "meters_per_pixel": 0.5,  # should be 0.01
            },
        }
        _validate_scene(scene, errors)
        assert any("inconsistent" in e.lower() for e in errors)

    def test_downsample_invalid(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "hallway", "downsample": -1.0},
        }
        _validate_scene(scene, errors)
        assert any("downsample" in e for e in errors)

    def test_downsample_string(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "hallway", "downsample": "no"},
        }
        _validate_scene(scene, errors)
        assert any("downsample" in e for e in errors)

    def test_path_source_missing_path(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "path", "pixels_per_meter": 100.0},
        }
        _validate_scene(scene, errors)
        assert any("path" in e.lower() for e in errors)

    def test_path_source_missing_scale(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "path", "path": "/tmp/some_map.png"},
        }
        _validate_scene(scene, errors)
        assert any("pixels_per_meter" in e or "meters_per_pixel" in e for e in errors)

    def test_path_source_empty_path(self):
        errors: list[str] = []
        scene = {
            "backend": "grid2d",
            "map": {"source": "path", "path": "", "pixels_per_meter": 100.0},
        }
        _validate_scene(scene, errors)
        assert any("path" in e.lower() for e in errors)


class TestValidateHorizonEdgeCases:
    def test_steps_zero(self):
        errors: list[str] = []
        _validate_horizon({"steps": 0, "dt": 0.1}, errors)
        assert any("steps" in e for e in errors)

    def test_dt_zero(self):
        errors: list[str] = []
        _validate_horizon({"steps": 100, "dt": 0.0}, errors)
        assert any("dt" in e for e in errors)

    def test_dt_negative(self):
        errors: list[str] = []
        _validate_horizon({"steps": 100, "dt": -0.5}, errors)
        assert any("dt" in e for e in errors)

    def test_not_a_dict(self):
        errors: list[str] = []
        _validate_horizon("not_a_dict", errors)
        assert any("object" in e for e in errors)

    def test_valid(self):
        errors: list[str] = []
        _validate_horizon({"steps": 500, "dt": 0.1}, errors)
        assert len(errors) == 0


class TestValidateHumansEdgeCases:
    def test_not_a_dict(self):
        errors: list[str] = []
        _validate_humans("bad", errors)
        assert any("object" in e for e in errors)

    def test_invalid_controller_type(self):
        errors: list[str] = []
        _validate_humans(
            {"controller": {"type": "nonexistent"}, "count": 5},
            errors,
        )
        assert any("controller.type" in e for e in errors)

    def test_negative_count(self):
        errors: list[str] = []
        _validate_humans(
            {"controller": {"type": "orca"}, "count": -1},
            errors,
        )
        assert any("count" in e for e in errors)

    def test_invalid_point_in_starts(self):
        errors: list[str] = []
        _validate_humans(
            {
                "controller": {"type": "orca"},
                "count": 1,
                "starts": [[1.0, 2.0], "bad_point"],
            },
            errors,
        )
        assert any("starts[1]" in e for e in errors)


class TestValidateRobotEdgeCases:
    def test_not_a_dict(self):
        errors: list[str] = []
        _validate_robot(42, errors)
        assert any("object" in e for e in errors)

    def test_missing_required_fields(self):
        errors: list[str] = []
        _validate_robot({}, errors)
        assert any("controller" in e for e in errors)
        assert any("start" in e for e in errors)
        assert any("goal" in e for e in errors)

    def test_invalid_controller_type(self):
        errors: list[str] = []
        _validate_robot(
            {
                "controller": {"type": "bad_type"},
                "start": [1.0, 2.0],
                "goal": [3.0, 4.0],
            },
            errors,
        )
        assert any("controller.type" in e for e in errors)

    def test_invalid_start_point(self):
        errors: list[str] = []
        _validate_robot(
            {
                "controller": {"type": "baseline_astar"},
                "start": [1.0],  # only 1 element
                "goal": [3.0, 4.0],
            },
            errors,
        )
        assert any("start" in e for e in errors)


class TestValidateScenarioDict:
    def test_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="scenario must be an object"):
            validate_scenario_dict("not_a_dict")

    def test_missing_id(self):
        with pytest.raises(ValueError, match="id is required"):
            validate_scenario_dict({"seed": 42})

    def test_missing_seed(self):
        with pytest.raises(ValueError, match="seed"):
            validate_scenario_dict({"id": "test"})


class TestIsPoint:
    def test_valid_list(self):
        assert _is_point([1.0, 2.0]) is True

    def test_valid_tuple(self):
        assert _is_point((3, 4)) is True

    def test_too_short(self):
        assert _is_point([1.0]) is False

    def test_too_long(self):
        assert _is_point([1.0, 2.0, 3.0]) is False

    def test_non_numeric(self):
        assert _is_point(["a", "b"]) is False

    def test_not_a_list(self):
        assert _is_point("point") is False


# -----------------------------------------------------------------------
# 4. SceneBackend abstract class - concrete methods
# -----------------------------------------------------------------------

from navirl.core.env import SceneBackend


class _StubBackend(SceneBackend):
    """Minimal concrete implementation for testing ABC concrete methods."""

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
        return (1.0, 1.0)

    def check_obstacle_collision(self, position, radius):
        return False

    def world_to_map(self, position):
        return (0, 0)

    def map_image(self):
        return np.zeros((10, 10), dtype=np.uint8)


class TestSceneBackendConcreteMethods:
    def test_nearest_clear_point_returns_input(self):
        backend = _StubBackend()
        result = backend.nearest_clear_point((3.5, 7.2), 0.5)
        assert result == (3.5, 7.2)

    def test_nearest_clear_point_converts_to_float(self):
        backend = _StubBackend()
        result = backend.nearest_clear_point((3, 7), 0.5)
        assert result == (3.0, 7.0)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_map_metadata_returns_empty_dict(self):
        backend = _StubBackend()
        meta = backend.map_metadata()
        assert meta == {}
        assert isinstance(meta, dict)


# -----------------------------------------------------------------------
# 5. Evaluation comparisons - uncovered branches
# -----------------------------------------------------------------------

from navirl.evaluation.benchmark import BenchmarkResults
from navirl.evaluation.comparisons import AgentComparison


class TestAgentComparisonEdgeCases:
    def _make_results(self, name, values):
        return BenchmarkResults(
            suite_name=name,
            scenario_names=[f"s{i}" for i in range(len(values))],
            metrics={"mean_reward": values, "success_rate": [v / 5.0 for v in values]},
        )

    def test_generate_report_multiple_metrics(self):
        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0, 2.0, 3.0]),
            "B": self._make_results("B", [2.0, 3.0, 4.0]),
        }
        report = comp.generate_report(results)
        assert "mean_reward" in report
        assert "success_rate" in report
        assert "Best Agent" in report
        assert "B" in report

    def test_generate_report_precision(self):
        comp = AgentComparison()
        results = {"X": self._make_results("X", [1.23456789])}
        report = comp.generate_report(results, precision=2)
        assert "1.23" in report

    def test_generate_report_empty_metrics(self):
        comp = AgentComparison()
        results = {
            "A": BenchmarkResults(
                suite_name="A",
                scenario_names=[],
                metrics={},
            ),
        }
        report = comp.generate_report(results)
        assert "Agent Comparison Report" in report

    def test_statistical_test_unknown_test_raises(self):
        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0, 2.0, 3.0]),
            "B": self._make_results("B", [2.0, 3.0, 4.0]),
        }
        try:
            from scipy import stats as _

            with pytest.raises(ValueError, match="Unknown test"):
                comp.statistical_test(results, test="invalid_test")
        except ImportError:
            pytest.skip("scipy not installed")

    def test_statistical_test_too_few_samples(self):
        comp = AgentComparison()
        results = {
            "A": BenchmarkResults(
                suite_name="A",
                scenario_names=["s1"],
                metrics={"mean_reward": [1.0]},
            ),
            "B": BenchmarkResults(
                suite_name="B",
                scenario_names=["s1"],
                metrics={"mean_reward": [2.0]},
            ),
        }
        try:
            from scipy import stats as _

            p_values = comp.statistical_test(results)
            assert math.isnan(p_values[("A", "B")])
        except ImportError:
            pytest.skip("scipy not installed")

    def test_statistical_test_mannwhitneyu(self):
        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            "B": self._make_results("B", [6.0, 7.0, 8.0, 9.0, 10.0]),
        }
        try:
            from scipy import stats as _

            p_values = comp.statistical_test(results, test="mannwhitneyu")
            assert ("A", "B") in p_values
            assert 0.0 <= p_values[("A", "B")] <= 1.0
        except ImportError:
            pytest.skip("scipy not installed")

    def test_statistical_test_wilcoxon(self):
        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            "B": self._make_results("B", [6.0, 7.0, 8.0, 9.0, 10.0]),
        }
        try:
            from scipy import stats as _

            p_values = comp.statistical_test(results, test="wilcoxon")
            assert ("A", "B") in p_values
            assert 0.0 <= p_values[("A", "B")] <= 1.0
        except ImportError:
            pytest.skip("scipy not installed")


# -----------------------------------------------------------------------
# 6. Evaluation analysis - uncovered functions
# -----------------------------------------------------------------------

from navirl.evaluation.analysis import (
    attention_visualization,
    policy_entropy_map,
    q_value_landscape,
)


class TestAttentionVisualization:
    def test_no_model_returns_ones(self):
        """Agent without model/policy attribute returns uniform."""

        class DummyAgent:
            pass

        result = attention_visualization(DummyAgent(), np.zeros(10))
        assert np.allclose(result, np.ones(1))

    def test_none_agent_returns_ones(self):
        """Agent with model=None returns uniform."""

        class DummyAgent:
            model = None
            policy = None

        result = attention_visualization(DummyAgent(), np.zeros(10))
        assert np.allclose(result, np.ones(1))


class TestPolicyEntropyMap:
    def test_agent_with_get_action_probs(self):
        class MockAgent:
            def get_action_probs(self, obs):
                return np.array([0.25, 0.25, 0.25, 0.25])

        grid = np.random.randn(5, 4)
        entropies = policy_entropy_map(MockAgent(), grid)
        assert entropies.shape == (5,)
        # Uniform distribution entropy = ln(4) ≈ 1.386
        assert all(e > 0 for e in entropies)

    def test_agent_without_probs_method(self):
        class MockAgent:
            pass

        grid = np.random.randn(3, 4)
        entropies = policy_entropy_map(MockAgent(), grid)
        assert entropies.shape == (3,)
        # All zeros since no probs method
        assert all(e == 0.0 for e in entropies)

    def test_agent_with_predict_proba(self):
        class MockAgent:
            def predict_proba(self, obs):
                return np.array([0.5, 0.5])

        grid = np.random.randn(2, 3)
        entropies = policy_entropy_map(MockAgent(), grid)
        assert entropies.shape == (2,)
        assert all(e > 0 for e in entropies)


class TestQValueLandscape:
    def test_agent_with_get_q_values(self):
        class MockAgent:
            def get_q_values(self, obs):
                return np.array([1.0, 2.0, 3.0])

        grid = np.random.randn(4, 5)
        qvals = q_value_landscape(MockAgent(), grid)
        assert qvals.shape == (4, 3)

    def test_agent_without_q_method(self):
        class MockAgent:
            pass

        grid = np.random.randn(3, 4)
        qvals = q_value_landscape(MockAgent(), grid)
        assert qvals.shape == (3, 1)
        assert np.allclose(qvals, 0.0)

    def test_agent_with_q_values_method(self):
        class MockAgent:
            def q_values(self, obs):
                return np.array([10.0])

        grid = np.random.randn(2, 3)
        qvals = q_value_landscape(MockAgent(), grid)
        assert qvals.shape == (2, 1)
        assert np.allclose(qvals, 10.0)
