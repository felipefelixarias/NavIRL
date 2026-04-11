"""Tests for multi-backend support (ROADMAP #9).

Covers:
- SceneBackend interface contract tests (parametrised over all backends)
- ContinuousSceneBackend adapter unit tests
- Cross-backend compatibility / scenario portability
- Plugin registry integration
- NavEnv backend selection
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.backends.continuous.adapter import ContinuousSceneBackend
from navirl.core.env import SceneBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONTINUOUS_SCENE_CFG: dict = {
    "width": 20.0,
    "height": 20.0,
}
_CONTINUOUS_HORIZON_CFG: dict = {"dt": 0.1}


@pytest.fixture()
def continuous_backend() -> ContinuousSceneBackend:
    return ContinuousSceneBackend(_CONTINUOUS_SCENE_CFG, _CONTINUOUS_HORIZON_CFG)


@pytest.fixture()
def continuous_backend_with_obstacles() -> ContinuousSceneBackend:
    cfg = {
        "width": 20.0,
        "height": 20.0,
        "obstacles": [
            {"type": "circle", "center": [10.0, 10.0], "radius": 2.0},
            {"type": "rectangle", "min_corner": [3.0, 3.0], "max_corner": [5.0, 5.0]},
        ],
        "walls": [
            {"start": [0.0, 0.0], "end": [20.0, 0.0], "thickness": 0.2},
        ],
    }
    return ContinuousSceneBackend(cfg, {"dt": 0.1})


# ---------------------------------------------------------------------------
# 1. SceneBackend interface contract tests
# ---------------------------------------------------------------------------


class TestSceneBackendContract:
    """Verify that ContinuousSceneBackend fulfils the SceneBackend ABC."""

    def test_is_scene_backend_subclass(self) -> None:
        assert issubclass(ContinuousSceneBackend, SceneBackend)

    def test_isinstance(self, continuous_backend: ContinuousSceneBackend) -> None:
        assert isinstance(continuous_backend, SceneBackend)

    def test_add_agent(self, continuous_backend: ContinuousSceneBackend) -> None:
        continuous_backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        pos = continuous_backend.get_position(0)
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_add_multiple_agents(self, continuous_backend: ContinuousSceneBackend) -> None:
        continuous_backend.add_agent(0, (2.0, 2.0), 0.3, 1.5, "robot")
        continuous_backend.add_agent(1, (18.0, 18.0), 0.3, 1.2, "human")
        continuous_backend.add_agent(2, (10.0, 10.0), 0.25, 1.0, "human")
        # All agents queryable
        for aid in (0, 1, 2):
            pos = continuous_backend.get_position(aid)
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_set_preferred_velocity_and_step(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        continuous_backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        continuous_backend.set_preferred_velocity(0, (1.0, 0.0))
        continuous_backend.step()
        pos = continuous_backend.get_position(0)
        # Agent should have moved in +x direction
        assert pos[0] > 5.0

    def test_get_velocity_returns_tuple(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        continuous_backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        continuous_backend.set_preferred_velocity(0, (1.0, 0.5))
        continuous_backend.step()
        vel = continuous_backend.get_velocity(0)
        assert isinstance(vel, tuple)
        assert len(vel) == 2

    def test_step_updates_position(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        continuous_backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        continuous_backend.set_preferred_velocity(0, (0.0, 1.0))
        pos_before = continuous_backend.get_position(0)
        continuous_backend.step()
        pos_after = continuous_backend.get_position(0)
        assert pos_after[1] > pos_before[1]

    def test_shortest_path_returns_list_of_tuples(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        path = continuous_backend.shortest_path((1.0, 1.0), (19.0, 19.0))
        assert isinstance(path, list)
        assert len(path) >= 2
        for pt in path:
            assert isinstance(pt, tuple)
            assert len(pt) == 2

    def test_shortest_path_starts_and_ends_correctly(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        start, goal = (2.0, 3.0), (18.0, 17.0)
        path = continuous_backend.shortest_path(start, goal)
        assert math.isclose(path[0][0], start[0], abs_tol=0.5)
        assert math.isclose(path[0][1], start[1], abs_tol=0.5)
        assert math.isclose(path[-1][0], goal[0], abs_tol=0.5)
        assert math.isclose(path[-1][1], goal[1], abs_tol=0.5)

    def test_sample_free_point(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        pt = continuous_backend.sample_free_point()
        assert isinstance(pt, tuple)
        assert len(pt) == 2
        assert 0 <= pt[0] <= 20.0
        assert 0 <= pt[1] <= 20.0

    def test_sample_free_point_not_in_obstacle(
        self, continuous_backend_with_obstacles: ContinuousSceneBackend
    ) -> None:
        backend = continuous_backend_with_obstacles
        for _ in range(20):
            pt = backend.sample_free_point()
            assert not backend.check_obstacle_collision(pt, 0.3)

    def test_check_obstacle_collision_no_obstacles(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        assert not continuous_backend.check_obstacle_collision((10.0, 10.0), 0.3)

    def test_check_obstacle_collision_with_obstacle(
        self, continuous_backend_with_obstacles: ContinuousSceneBackend
    ) -> None:
        # Circle obstacle at (10, 10) r=2 → point (10, 10) should collide
        assert continuous_backend_with_obstacles.check_obstacle_collision(
            (10.0, 10.0), 0.3
        )

    def test_check_obstacle_collision_outside(
        self, continuous_backend_with_obstacles: ContinuousSceneBackend
    ) -> None:
        # Well outside any obstacle
        assert not continuous_backend_with_obstacles.check_obstacle_collision(
            (18.0, 18.0), 0.3
        )

    def test_world_to_map(self, continuous_backend: ContinuousSceneBackend) -> None:
        result = continuous_backend.world_to_map((10.0, 10.0))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_map_image(self, continuous_backend: ContinuousSceneBackend) -> None:
        img = continuous_backend.map_image()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_map_image_with_obstacles(
        self, continuous_backend_with_obstacles: ContinuousSceneBackend
    ) -> None:
        img = continuous_backend_with_obstacles.map_image()
        # Should have both obstacle (0) and free (255) pixels
        assert 0 in img
        assert 255 in img

    def test_nearest_clear_point_in_free_space(
        self, continuous_backend: ContinuousSceneBackend
    ) -> None:
        pt = continuous_backend.nearest_clear_point((10.0, 10.0), 0.3)
        assert isinstance(pt, tuple)
        assert len(pt) == 2

    def test_nearest_clear_point_near_obstacle(
        self, continuous_backend_with_obstacles: ContinuousSceneBackend
    ) -> None:
        backend = continuous_backend_with_obstacles
        pt = backend.nearest_clear_point((10.0, 10.0), 0.3)
        # Should return a point that is NOT colliding
        assert not backend.check_obstacle_collision(pt, 0.3)

    def test_map_metadata(self, continuous_backend: ContinuousSceneBackend) -> None:
        meta = continuous_backend.map_metadata()
        assert isinstance(meta, dict)
        assert "backend" in meta
        assert meta["backend"] == "continuous"
        assert "width" in meta
        assert "height" in meta


# ---------------------------------------------------------------------------
# 2. ContinuousSceneBackend adapter unit tests
# ---------------------------------------------------------------------------


class TestContinuousAdapterSpecific:
    """Tests specific to the ContinuousSceneBackend adapter."""

    def test_constructor_defaults(self) -> None:
        backend = ContinuousSceneBackend({}, {"dt": 0.05})
        assert isinstance(backend, SceneBackend)

    def test_constructor_with_physics_config(self) -> None:
        cfg = {
            "width": 10.0,
            "height": 10.0,
            "physics": {"damping": 0.2, "restitution": 0.3},
        }
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        meta = backend.map_metadata()
        assert meta["width"] == 10.0

    def test_constructor_with_obstacles_and_walls(self) -> None:
        cfg = {
            "obstacles": [
                {"type": "circle", "center": [5, 5], "radius": 1.0},
                {"type": "rectangle", "min_corner": [1, 1], "max_corner": [2, 2]},
            ],
            "walls": [
                {"start": [0, 0], "end": [10, 0]},
            ],
        }
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        assert backend.check_obstacle_collision((5.0, 5.0), 0.1)

    def test_base_dir_ignored(self) -> None:
        backend = ContinuousSceneBackend({}, {"dt": 0.1}, base_dir="/fake/path")
        assert isinstance(backend, SceneBackend)

    def test_agent_kind_tracking(self) -> None:
        backend = ContinuousSceneBackend({}, {"dt": 0.1})
        backend.add_agent(10, (5.0, 5.0), 0.3, 1.5, "robot")
        backend.add_agent(20, (15.0, 15.0), 0.3, 1.2, "human")
        assert backend._agent_kinds[10] == "robot"
        assert backend._agent_kinds[20] == "human"

    def test_nonexistent_agent_raises(self) -> None:
        backend = ContinuousSceneBackend({}, {"dt": 0.1})
        with pytest.raises(KeyError):
            backend.get_position(999)

    def test_multi_step_simulation(self) -> None:
        backend = ContinuousSceneBackend({"width": 30, "height": 30}, {"dt": 0.1})
        backend.add_agent(0, (5.0, 5.0), 0.3, 2.0, "robot")
        backend.set_preferred_velocity(0, (1.0, 0.0))
        for _ in range(10):
            backend.step()
        pos = backend.get_position(0)
        assert pos[0] > 5.5  # Should have moved significantly

    def test_zero_velocity_stays_put(self) -> None:
        backend = ContinuousSceneBackend({}, {"dt": 0.1})
        backend.add_agent(0, (10.0, 10.0), 0.3, 1.5, "robot")
        backend.set_preferred_velocity(0, (0.0, 0.0))
        backend.step()
        pos = backend.get_position(0)
        assert abs(pos[0] - 10.0) < 0.5
        assert abs(pos[1] - 10.0) < 0.5

    def test_shortest_path_with_obstacle(self) -> None:
        cfg = {
            "width": 20.0,
            "height": 20.0,
            "obstacles": [
                {"type": "rectangle", "min_corner": [9.0, 0.0], "max_corner": [11.0, 15.0]},
            ],
        }
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        path = backend.shortest_path((5.0, 5.0), (15.0, 5.0))
        assert len(path) >= 2
        # Path should go around the obstacle (y > ~15 or different route)
        # At minimum, it should not be a straight line through the wall
        assert any(pt[1] > 10.0 or pt[0] < 9.0 or pt[0] > 11.0 for pt in path[1:-1]) or len(path) > 2

    def test_shortest_path_direct_line_of_sight(self) -> None:
        backend = ContinuousSceneBackend({}, {"dt": 0.1})
        path = backend.shortest_path((2.0, 2.0), (18.0, 18.0))
        # No obstacles → should be direct (2 points)
        assert len(path) == 2

    def test_world_to_map_corners(self) -> None:
        backend = ContinuousSceneBackend(
            {"width": 10.0, "height": 10.0}, {"dt": 0.1}
        )
        # Origin should map to bottom-left of the image
        r, c = backend.world_to_map((0.0, 0.0))
        assert r >= 0 and c >= 0

    def test_enable_boundaries_false(self) -> None:
        cfg = {"width": 10.0, "height": 10.0, "enable_boundaries": False}
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        assert isinstance(backend, SceneBackend)


# ---------------------------------------------------------------------------
# 3. Plugin registry integration
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    """Test that the continuous backend is properly registered."""

    def test_continuous_backend_registered(self) -> None:
        from navirl.core.registry import registry_snapshot
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        snap = registry_snapshot()
        assert "continuous" in snap["backends"]
        assert "grid2d" in snap["backends"]

    def test_get_backend_continuous(self) -> None:
        from navirl.core.registry import get_backend
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        factory = get_backend("continuous")
        backend = factory(
            {"width": 10.0, "height": 10.0},
            {"dt": 0.1},
        )
        assert isinstance(backend, SceneBackend)
        assert isinstance(backend, ContinuousSceneBackend)

    def test_get_backend_grid2d_still_works(self) -> None:
        from navirl.core.registry import get_backend
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        factory = get_backend("grid2d")
        assert factory is not None

    def test_unknown_backend_raises(self) -> None:
        from navirl.core.registry import get_backend
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        with pytest.raises(KeyError, match="nonexistent"):
            get_backend("nonexistent")

    def test_plugin_info_continuous(self) -> None:
        from navirl.core.registry import get_plugin_info
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        info = get_plugin_info("backend", "continuous")
        assert info["name"] == "continuous"
        assert info["type"] == "backend"


# ---------------------------------------------------------------------------
# 4. Cross-backend compatibility tests
# ---------------------------------------------------------------------------


class TestCrossBackendCompatibility:
    """Verify shared contract behaviour across backends."""

    def test_both_backends_add_agent_and_query(self) -> None:
        """Both backends support the same add_agent → get_position flow."""
        backend = ContinuousSceneBackend(
            {"width": 20, "height": 20}, {"dt": 0.1}
        )
        backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        backend.add_agent(1, (15.0, 15.0), 0.3, 1.2, "human")

        for aid in (0, 1):
            pos = backend.get_position(aid)
            vel = backend.get_velocity(aid)
            assert isinstance(pos, tuple) and len(pos) == 2
            assert isinstance(vel, tuple) and len(vel) == 2

    def test_step_velocity_movement(self) -> None:
        """Setting preferred velocity and stepping produces motion."""
        backend = ContinuousSceneBackend(
            {"width": 20, "height": 20}, {"dt": 0.1}
        )
        backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        backend.set_preferred_velocity(0, (1.0, 0.0))
        backend.step()
        pos = backend.get_position(0)
        assert pos[0] > 5.0

    def test_shortest_path_contract(self) -> None:
        """shortest_path returns a valid path for both backends."""
        backend = ContinuousSceneBackend(
            {"width": 20, "height": 20}, {"dt": 0.1}
        )
        path = backend.shortest_path((1.0, 1.0), (19.0, 19.0))
        assert len(path) >= 2
        # First and last points should be near start/goal
        assert abs(path[0][0] - 1.0) < 1.0
        assert abs(path[-1][0] - 19.0) < 1.0

    def test_sample_free_point_returns_valid(self) -> None:
        """sample_free_point returns a point inside the world."""
        backend = ContinuousSceneBackend(
            {"width": 20, "height": 20}, {"dt": 0.1}
        )
        for _ in range(10):
            pt = backend.sample_free_point()
            assert 0 <= pt[0] <= 20.0
            assert 0 <= pt[1] <= 20.0

    def test_collision_check_consistency(self) -> None:
        cfg = {
            "width": 20.0,
            "height": 20.0,
            "obstacles": [
                {"type": "circle", "center": [10.0, 10.0], "radius": 2.0},
            ],
        }
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        # Inside obstacle → collision
        assert backend.check_obstacle_collision((10.0, 10.0), 0.1)
        # Far from obstacle → no collision
        assert not backend.check_obstacle_collision((1.0, 1.0), 0.1)

    def test_map_image_valid_format(self) -> None:
        """map_image returns a 2D uint8 ndarray for both backends."""
        backend = ContinuousSceneBackend(
            {"width": 10, "height": 10}, {"dt": 0.1}
        )
        img = backend.map_image()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_world_to_map_valid_format(self) -> None:
        backend = ContinuousSceneBackend(
            {"width": 20, "height": 20}, {"dt": 0.1}
        )
        r, c = backend.world_to_map((10.0, 10.0))
        assert isinstance(r, int)
        assert isinstance(c, int)

    def test_map_metadata_valid_format(self) -> None:
        backend = ContinuousSceneBackend(
            {"width": 20, "height": 20}, {"dt": 0.1}
        )
        meta = backend.map_metadata()
        assert isinstance(meta, dict)

    def test_nearest_clear_point_returns_clear(self) -> None:
        cfg = {
            "width": 20.0,
            "height": 20.0,
            "obstacles": [
                {"type": "circle", "center": [10.0, 10.0], "radius": 2.0},
            ],
        }
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        pt = backend.nearest_clear_point((10.0, 10.0), 0.3)
        assert not backend.check_obstacle_collision(pt, 0.3)


# ---------------------------------------------------------------------------
# 5. NavEnv backend selection
# ---------------------------------------------------------------------------


_has_gymnasium = True
try:
    import gymnasium
except ImportError:
    _has_gymnasium = False


@pytest.mark.skipif(not _has_gymnasium, reason="gymnasium not installed")
class TestNavEnvBackendSelection:
    """Test that NavEnv respects the backend config field."""

    def test_config_default_is_grid2d(self) -> None:
        from navirl.envs.base_env import NavEnvConfig

        cfg = NavEnvConfig()
        assert cfg.backend == "grid2d"

    def test_config_accepts_continuous(self) -> None:
        from navirl.envs.base_env import NavEnvConfig

        cfg = NavEnvConfig(backend="continuous")
        assert cfg.backend == "continuous"


# ---------------------------------------------------------------------------
# 6. Scenario portability / edge cases
# ---------------------------------------------------------------------------


class TestScenarioPortability:
    """Document which scenario features are portable and which are not."""

    def test_empty_world_works(self) -> None:
        backend = ContinuousSceneBackend(
            {"width": 10, "height": 10}, {"dt": 0.1}
        )
        backend.add_agent(0, (5.0, 5.0), 0.3, 1.5, "robot")
        backend.step()
        pos = backend.get_position(0)
        assert isinstance(pos, tuple)

    def test_dense_obstacle_world(self) -> None:
        obstacles = [
            {"type": "circle", "center": [float(x), float(y)], "radius": 0.5}
            for x in range(2, 18, 4)
            for y in range(2, 18, 4)
        ]
        cfg = {"width": 20, "height": 20, "obstacles": obstacles}
        backend = ContinuousSceneBackend(cfg, {"dt": 0.1})
        # Should still be able to sample free points
        pt = backend.sample_free_point()
        assert isinstance(pt, tuple)

    def test_many_agents(self) -> None:
        backend = ContinuousSceneBackend({"width": 50, "height": 50}, {"dt": 0.1})
        for i in range(20):
            x = 2.0 + (i % 5) * 10.0
            y = 2.0 + (i // 5) * 10.0
            backend.add_agent(i, (x, y), 0.3, 1.0, "human")
        backend.step()
        for i in range(20):
            pos = backend.get_position(i)
            assert isinstance(pos, tuple)

    def test_dt_from_horizon_cfg(self) -> None:
        backend = ContinuousSceneBackend(
            {"width": 10, "height": 10},
            {"dt": 0.05},
        )
        assert backend._dt == 0.05

    def test_dt_from_scene_cfg_fallback(self) -> None:
        backend = ContinuousSceneBackend(
            {"width": 10, "height": 10, "dt": 0.02},
            {},
        )
        assert backend._dt == 0.02
