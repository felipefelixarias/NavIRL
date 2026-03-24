"""Tests for navirl/planning/ module: global planners, path dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.planning.base import Path, PlannerConfig
from navirl.planning.global_planners import (
    AStarPlanner,
    DijkstraPlanner,
    ThetaStarPlanner,
    _grid_neighbors,
    _grid_to_pos,
    _is_free,
    _pos_to_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid():
    """10x10 free grid (all zeros = free)."""
    return np.zeros((10, 10), dtype=np.int32)


@pytest.fixture
def obstacle_grid():
    """10x10 grid with a wall in the middle."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 2:8] = 1  # horizontal wall
    return grid


@pytest.fixture
def planner_config():
    return PlannerConfig(max_iterations=50000, time_limit=5.0, resolution=1.0)


# ---------------------------------------------------------------------------
# Path dataclass
# ---------------------------------------------------------------------------

class TestPath:
    def test_path_length_straight(self):
        waypoints = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
        timestamps = np.array([0, 1, 2, 3], dtype=np.float64)
        velocities = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float64)
        p = Path(waypoints=waypoints, timestamps=timestamps, velocities=velocities)
        assert p.length == pytest.approx(3.0)
        assert p.num_waypoints == 4
        assert p.duration == pytest.approx(3.0)

    def test_path_length_single_point(self):
        p = Path(
            waypoints=np.array([[1, 2]]),
            timestamps=np.array([0]),
            velocities=np.array([[0, 0]]),
        )
        assert p.length == 0.0
        assert p.duration == 0.0

    def test_interpolate_at_start(self):
        wp = np.array([[0, 0], [10, 0]], dtype=np.float64)
        ts = np.array([0, 10], dtype=np.float64)
        p = Path(waypoints=wp, timestamps=ts, velocities=np.zeros_like(wp))
        result = p.interpolate(0.0)
        np.testing.assert_allclose(result, [0, 0])

    def test_interpolate_at_end(self):
        wp = np.array([[0, 0], [10, 0]], dtype=np.float64)
        ts = np.array([0, 10], dtype=np.float64)
        p = Path(waypoints=wp, timestamps=ts, velocities=np.zeros_like(wp))
        result = p.interpolate(10.0)
        np.testing.assert_allclose(result, [10, 0])

    def test_interpolate_midpoint(self):
        wp = np.array([[0, 0], [10, 0]], dtype=np.float64)
        ts = np.array([0, 10], dtype=np.float64)
        p = Path(waypoints=wp, timestamps=ts, velocities=np.zeros_like(wp))
        result = p.interpolate(5.0)
        np.testing.assert_allclose(result, [5, 0])

    def test_interpolate_before_start(self):
        wp = np.array([[5, 5], [10, 10]], dtype=np.float64)
        ts = np.array([1, 2], dtype=np.float64)
        p = Path(waypoints=wp, timestamps=ts, velocities=np.zeros_like(wp))
        result = p.interpolate(0.0)
        np.testing.assert_allclose(result, [5, 5])

    def test_interpolate_after_end(self):
        wp = np.array([[0, 0], [10, 10]], dtype=np.float64)
        ts = np.array([0, 1], dtype=np.float64)
        p = Path(waypoints=wp, timestamps=ts, velocities=np.zeros_like(wp))
        result = p.interpolate(100.0)
        np.testing.assert_allclose(result, [10, 10])

    def test_cost_attribute(self):
        p = Path(
            waypoints=np.zeros((2, 2)),
            timestamps=np.array([0, 1]),
            velocities=np.zeros((2, 2)),
            cost=42.0,
        )
        assert p.cost == 42.0

    def test_metadata(self):
        p = Path(
            waypoints=np.zeros((1, 2)),
            timestamps=np.array([0]),
            velocities=np.zeros((1, 2)),
            metadata={"planner": "astar"},
        )
        assert p.metadata["planner"] == "astar"


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

class TestGridHelpers:
    def test_grid_neighbors_corner(self):
        nbrs = _grid_neighbors((0, 0), (10, 10))
        assert (0, 0) not in nbrs
        assert len(nbrs) == 3  # only (0,1), (1,0), (1,1)

    def test_grid_neighbors_center(self):
        nbrs = _grid_neighbors((5, 5), (10, 10))
        assert len(nbrs) == 8

    def test_grid_neighbors_edge(self):
        nbrs = _grid_neighbors((0, 5), (10, 10))
        assert len(nbrs) == 5

    def test_pos_to_grid(self):
        cell = _pos_to_grid(np.array([2.5, 3.5]), np.zeros(2), resolution=1.0)
        assert cell == (2, 3)

    def test_grid_to_pos(self):
        pos = _grid_to_pos((2, 3), np.zeros(2), resolution=1.0)
        np.testing.assert_allclose(pos, [2.0, 3.0])

    def test_pos_grid_round_trip(self):
        origin = np.array([1.0, 2.0])
        pos = np.array([5.0, 7.0])
        cell = _pos_to_grid(pos, origin, resolution=1.0)
        recovered = _grid_to_pos(cell, origin, resolution=1.0)
        np.testing.assert_allclose(recovered, [5.0, 7.0], atol=1.0)

    def test_is_free_on_free_grid(self, simple_grid):
        assert _is_free((3, 3), simple_grid) is True

    def test_is_free_on_obstacle(self, obstacle_grid):
        assert _is_free((5, 4), obstacle_grid) is False

    def test_is_free_none_grid(self):
        assert _is_free((0, 0), None) is True

    def test_is_free_out_of_bounds(self, simple_grid):
        assert _is_free((100, 100), simple_grid) is False


# ---------------------------------------------------------------------------
# A* planner
# ---------------------------------------------------------------------------

class TestAStarPlanner:
    def test_plan_free_grid(self, simple_grid, planner_config):
        planner = AStarPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 5.0]),
            obstacles=simple_grid,
        )
        assert path.num_waypoints >= 2
        assert path.length > 0

    def test_plan_reaches_goal(self, simple_grid, planner_config):
        planner = AStarPlanner(config=planner_config)
        goal = np.array([3.0, 3.0])
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=goal,
            obstacles=simple_grid,
        )
        final = path.waypoints[-1]
        dist = np.linalg.norm(final - goal)
        assert dist < 2.0  # within grid resolution

    def test_plan_with_obstacles(self, obstacle_grid, planner_config):
        planner = AStarPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([2.0, 2.0]),
            goal=np.array([8.0, 8.0]),
            obstacles=obstacle_grid,
        )
        assert path.num_waypoints >= 2

    @pytest.mark.parametrize("heuristic", ["euclidean", "manhattan", "chebyshev"])
    def test_different_heuristics(self, simple_grid, planner_config, heuristic):
        planner = AStarPlanner(config=planner_config, heuristic=heuristic)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 5.0]),
            obstacles=simple_grid,
        )
        assert path.num_waypoints >= 2

    def test_plan_no_obstacles(self, planner_config):
        planner = AStarPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([3.0, 3.0]),
        )
        assert path.num_waypoints >= 2

    def test_plan_same_start_goal(self, simple_grid, planner_config):
        planner = AStarPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([3.0, 3.0]),
            goal=np.array([3.0, 3.0]),
            obstacles=simple_grid,
        )
        assert path.num_waypoints >= 1

    def test_path_has_timestamps(self, simple_grid, planner_config):
        planner = AStarPlanner(config=planner_config, speed=2.0)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 0.0]),
            obstacles=simple_grid,
        )
        assert path.timestamps[0] == 0.0
        assert all(np.diff(path.timestamps) >= 0)


# ---------------------------------------------------------------------------
# Dijkstra planner
# ---------------------------------------------------------------------------

class TestDijkstraPlanner:
    def test_plan_free_grid(self, simple_grid, planner_config):
        planner = DijkstraPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 5.0]),
            obstacles=simple_grid,
        )
        assert path.num_waypoints >= 2

    def test_plan_with_obstacles(self, obstacle_grid, planner_config):
        planner = DijkstraPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([2.0, 2.0]),
            goal=np.array([8.0, 8.0]),
            obstacles=obstacle_grid,
        )
        assert path.num_waypoints >= 2
        assert path.length > 0

    def test_dijkstra_produces_shortest(self, simple_grid, planner_config):
        planner = DijkstraPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 0.0]),
            obstacles=simple_grid,
        )
        # Shortest path on free grid should be roughly the Euclidean distance
        assert path.length <= 6.0  # 5 grid cells + some diagonal


# ---------------------------------------------------------------------------
# Theta* planner
# ---------------------------------------------------------------------------

class TestThetaStarPlanner:
    def test_plan_free_grid(self, simple_grid, planner_config):
        planner = ThetaStarPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 5.0]),
            obstacles=simple_grid,
        )
        assert path.num_waypoints >= 2

    def test_any_angle_shorter(self, simple_grid, planner_config):
        """Theta* should produce paths at most as long as A*."""
        astar = AStarPlanner(config=planner_config)
        theta = ThetaStarPlanner(config=planner_config)
        start = np.array([0.0, 0.0])
        goal = np.array([7.0, 7.0])
        path_a = astar.plan(start, goal, simple_grid)
        path_t = theta.plan(start, goal, simple_grid)
        # Theta* should have at most as many waypoints (it smooths)
        assert path_t.length <= path_a.length + 1e-6


# ---------------------------------------------------------------------------
# PlannerConfig
# ---------------------------------------------------------------------------

class TestPlannerConfig:
    def test_defaults(self):
        cfg = PlannerConfig()
        assert cfg.max_iterations == 10000
        assert cfg.resolution > 0

    def test_custom_config(self):
        cfg = PlannerConfig(max_iterations=100, time_limit=1.0, resolution=0.5)
        assert cfg.max_iterations == 100
        assert cfg.resolution == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPlanningEdgeCases:
    def test_start_on_obstacle(self, obstacle_grid, planner_config):
        """If start is on an obstacle, planner should still return a path (fallback)."""
        planner = AStarPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([5.0, 4.0]),  # on the wall
            goal=np.array([0.0, 0.0]),
            obstacles=obstacle_grid,
        )
        # Should return fallback straight-line path
        assert path.num_waypoints >= 2

    def test_very_close_start_goal(self, simple_grid, planner_config):
        planner = DijkstraPlanner(config=planner_config)
        path = planner.plan(
            start=np.array([3.0, 3.0]),
            goal=np.array([3.1, 3.1]),
            obstacles=simple_grid,
        )
        assert path.num_waypoints >= 1

    def test_speed_affects_timestamps(self, simple_grid, planner_config):
        slow = AStarPlanner(config=planner_config, speed=0.5)
        fast = AStarPlanner(config=planner_config, speed=2.0)
        start, goal = np.array([0.0, 0.0]), np.array([5.0, 0.0])
        path_slow = slow.plan(start, goal, simple_grid)
        path_fast = fast.plan(start, goal, simple_grid)
        if path_slow.duration > 0 and path_fast.duration > 0:
            assert path_slow.duration > path_fast.duration
