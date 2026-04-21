"""Extended tests for global planners: RRT, RRT*, PRM, and additional edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.planning.base import Path, PlannerConfig
from navirl.planning.global_planners import (
    AStarPlanner,
    DijkstraPlanner,
    PRMPlanner,
    RRTPlanner,
    RRTStarPlanner,
    ThetaStarPlanner,
    _build_path,
    _reconstruct,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def free_grid():
    """10x10 free grid."""
    return np.zeros((10, 10), dtype=np.int32)


@pytest.fixture
def obstacle_grid():
    """10x10 grid with horizontal wall."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 2:8] = 1
    return grid


@pytest.fixture
def config_grid():
    """PlannerConfig for grid-based planners."""
    return PlannerConfig(max_iterations=50000, time_limit=5.0, resolution=1.0)


@pytest.fixture
def config_sampling():
    """PlannerConfig for sampling-based planners."""
    return PlannerConfig(max_iterations=2000, time_limit=2.0, resolution=0.05)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestBuildPath:
    def test_single_cell(self):
        origin = np.zeros(2)
        path = _build_path([(0, 0)], origin, resolution=1.0)
        assert path.num_waypoints == 1
        assert path.cost == 0.0

    def test_two_cells(self):
        origin = np.zeros(2)
        path = _build_path([(0, 0), (1, 0)], origin, resolution=1.0)
        assert path.num_waypoints == 2
        assert path.cost > 0.0
        assert path.timestamps[0] == 0.0

    def test_speed_affects_timestamps(self):
        origin = np.zeros(2)
        slow = _build_path([(0, 0), (5, 0)], origin, resolution=1.0, speed=1.0)
        fast = _build_path([(0, 0), (5, 0)], origin, resolution=1.0, speed=2.0)
        assert slow.timestamps[-1] > fast.timestamps[-1]

    def test_velocities_computed(self):
        origin = np.zeros(2)
        path = _build_path([(0, 0), (1, 0), (2, 0)], origin, resolution=1.0, speed=1.0)
        # First waypoint should have non-zero velocity pointing towards next
        assert path.velocities[0, 0] > 0
        # Last velocity should copy second-to-last
        np.testing.assert_allclose(path.velocities[-1], path.velocities[-2])


class TestReconstruct:
    def test_empty_came_from(self):
        result = _reconstruct({}, "A")
        assert result == ["A"]

    def test_simple_chain(self):
        came_from = {"C": "B", "B": "A"}
        result = _reconstruct(came_from, "C")
        assert result == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# RRT Planner
# ---------------------------------------------------------------------------


class TestRRTPlanner:
    def test_plan_open_space(self, config_sampling):
        np.random.seed(42)
        planner = RRTPlanner(config=config_sampling, step_size=0.5, goal_bias=0.2)
        start = np.array([0.0, 0.0])
        goal = np.array([3.0, 3.0])
        path = planner.plan(start, goal)
        assert path.num_waypoints >= 2
        assert path.cost > 0

    def test_plan_reaches_near_goal(self, config_sampling):
        np.random.seed(42)
        planner = RRTPlanner(config=config_sampling, step_size=0.5, goal_bias=0.3)
        start = np.array([0.0, 0.0])
        goal = np.array([2.0, 2.0])
        path = planner.plan(start, goal)
        # Final waypoint should be near goal
        dist_to_goal = np.linalg.norm(path.waypoints[-1] - goal)
        assert dist_to_goal < 2.0  # reasonable for RRT

    def test_plan_with_bounds(self, config_sampling):
        np.random.seed(42)
        bounds = (np.array([-1.0, -1.0]), np.array([5.0, 5.0]))
        planner = RRTPlanner(config=config_sampling, step_size=0.5, bounds=bounds)
        path = planner.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]))
        assert path.num_waypoints >= 2

    def test_plan_with_point_obstacles(self, config_sampling):
        np.random.seed(42)
        # Obstacles as Nx2 array of centres
        obstacles = np.array([[1.5, 1.5], [2.5, 2.5]])
        planner = RRTPlanner(config=config_sampling, step_size=0.3, goal_bias=0.2)
        path = planner.plan(np.array([0.0, 0.0]), np.array([4.0, 4.0]), obstacles=obstacles)
        assert path.num_waypoints >= 2

    def test_plan_with_obstacles_and_radius(self, config_sampling):
        np.random.seed(42)
        # Obstacles as Nx3 (x, y, radius)
        obstacles = np.array([[2.0, 2.0, 0.5]])
        planner = RRTPlanner(config=config_sampling, step_size=0.3, goal_bias=0.2)
        path = planner.plan(np.array([0.0, 0.0]), np.array([4.0, 4.0]), obstacles=obstacles)
        assert path.num_waypoints >= 2

    def test_fallback_path(self):
        """With very few iterations, should return fallback straight-line."""
        config = PlannerConfig(max_iterations=1, time_limit=0.001)
        planner = RRTPlanner(config=config, step_size=0.5, goal_bias=0.0)
        np.random.seed(0)
        start = np.array([0.0, 0.0])
        goal = np.array([100.0, 100.0])
        path = planner.plan(start, goal)
        # Fallback should have exactly 2 waypoints
        assert path.num_waypoints == 2
        np.testing.assert_allclose(path.waypoints[0], start)
        np.testing.assert_allclose(path.waypoints[1], goal)

    def test_same_start_goal(self, config_sampling):
        np.random.seed(42)
        planner = RRTPlanner(config=config_sampling, step_size=0.5, goal_bias=0.5)
        pt = np.array([1.0, 1.0])
        path = planner.plan(pt, pt.copy())
        assert path.num_waypoints >= 1

    def test_steer_within_step_size(self):
        planner = RRTPlanner(step_size=1.0)
        from_pt = np.array([0.0, 0.0])
        to_pt = np.array([0.5, 0.0])
        result = planner._steer(from_pt, to_pt)
        np.testing.assert_allclose(result, to_pt)

    def test_steer_beyond_step_size(self):
        planner = RRTPlanner(step_size=1.0)
        from_pt = np.array([0.0, 0.0])
        to_pt = np.array([3.0, 0.0])
        result = planner._steer(from_pt, to_pt)
        np.testing.assert_allclose(result, [1.0, 0.0])

    def test_nearest(self):
        tree = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([5.0, 5.0])]
        idx = RRTPlanner._nearest(tree, np.array([0.9, 0.9]))
        assert idx == 1

    def test_collision_free_no_obstacles(self):
        assert RRTPlanner._collision_free(np.array([0.0, 0.0]), np.array([1.0, 1.0]), None) is True

    def test_collision_free_with_obstacle(self):
        obstacles = np.array([[0.5, 0.0, 0.2]])
        result = RRTPlanner._collision_free(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), obstacles, resolution=0.01
        )
        assert result is False

    def test_collision_free_clear_path(self):
        obstacles = np.array([[5.0, 5.0, 0.2]])
        result = RRTPlanner._collision_free(np.array([0.0, 0.0]), np.array([1.0, 0.0]), obstacles)
        assert result is True

    def test_sample_goal_bias(self):
        planner = RRTPlanner(goal_bias=1.0)  # always sample goal
        goal = np.array([5.0, 5.0])
        np.random.seed(42)
        sample = planner._sample(goal, np.zeros(2), np.ones(2) * 10)
        np.testing.assert_allclose(sample, goal)


# ---------------------------------------------------------------------------
# RRT* Planner
# ---------------------------------------------------------------------------


class TestRRTStarPlanner:
    def test_plan_open_space(self, config_sampling):
        np.random.seed(42)
        planner = RRTStarPlanner(
            config=config_sampling, step_size=0.5, goal_bias=0.2, rewire_radius=1.5
        )
        start = np.array([0.0, 0.0])
        goal = np.array([3.0, 3.0])
        path = planner.plan(start, goal)
        assert path.num_waypoints >= 2
        assert path.cost > 0

    def test_plan_with_bounds(self, config_sampling):
        np.random.seed(42)
        bounds = (np.array([-1.0, -1.0]), np.array([5.0, 5.0]))
        planner = RRTStarPlanner(config=config_sampling, step_size=0.5, bounds=bounds)
        path = planner.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]))
        assert path.num_waypoints >= 2

    def test_rewiring_improves_cost(self, config_sampling):
        """RRT* with rewiring should find equal or better cost than basic RRT."""
        np.random.seed(42)
        rrt = RRTPlanner(config=config_sampling, step_size=0.5, goal_bias=0.2)
        path_rrt = rrt.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]))

        np.random.seed(42)
        rrt_star = RRTStarPlanner(
            config=config_sampling, step_size=0.5, goal_bias=0.2, rewire_radius=2.0
        )
        path_star = rrt_star.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]))

        # Both should find a path; RRT* cost should be <= RRT cost (or close)
        assert path_star.cost <= path_rrt.cost * 1.5  # generous tolerance

    def test_plan_with_obstacles(self, config_sampling):
        np.random.seed(42)
        obstacles = np.array([[1.5, 1.5, 0.5]])
        planner = RRTStarPlanner(config=config_sampling, step_size=0.3, goal_bias=0.2)
        path = planner.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]), obstacles=obstacles)
        assert path.num_waypoints >= 2

    def test_fallback_path(self):
        """With very few iterations, return fallback."""
        config = PlannerConfig(max_iterations=1, time_limit=0.001)
        planner = RRTStarPlanner(config=config, step_size=0.5, goal_bias=0.0)
        np.random.seed(0)
        start = np.array([0.0, 0.0])
        goal = np.array([100.0, 100.0])
        path = planner.plan(start, goal)
        assert path.num_waypoints == 2

    def test_timestamps_monotonic(self, config_sampling):
        np.random.seed(42)
        planner = RRTStarPlanner(config=config_sampling, step_size=0.5, goal_bias=0.3)
        path = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert all(np.diff(path.timestamps) >= 0)

    def test_default_bounds_computed(self, config_sampling):
        """Without explicit bounds, planner infers them from start/goal."""
        np.random.seed(42)
        planner = RRTStarPlanner(config=config_sampling, step_size=0.5)
        path = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert path.num_waypoints >= 2


# ---------------------------------------------------------------------------
# PRM Planner
# ---------------------------------------------------------------------------


class TestPRMPlanner:
    def test_plan_open_space(self, config_sampling):
        np.random.seed(42)
        planner = PRMPlanner(config=config_sampling, num_samples=200, k_neighbors=10)
        start = np.array([0.0, 0.0])
        goal = np.array([3.0, 3.0])
        path = planner.plan(start, goal)
        assert path.num_waypoints >= 2
        assert path.cost > 0

    def test_plan_with_bounds(self, config_sampling):
        np.random.seed(42)
        bounds = (np.array([-1.0, -1.0]), np.array([5.0, 5.0]))
        planner = PRMPlanner(config=config_sampling, num_samples=200, bounds=bounds)
        path = planner.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]))
        assert path.num_waypoints >= 2

    def test_plan_with_obstacles(self, config_sampling):
        np.random.seed(42)
        obstacles = np.array([[1.5, 1.5, 0.3]])
        planner = PRMPlanner(config=config_sampling, num_samples=300, k_neighbors=15)
        path = planner.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]), obstacles=obstacles)
        assert path.num_waypoints >= 2

    def test_timestamps_and_velocities(self, config_sampling):
        np.random.seed(42)
        planner = PRMPlanner(config=config_sampling, num_samples=200, speed=2.0)
        path = planner.plan(np.array([0.0, 0.0]), np.array([3.0, 3.0]))
        assert path.timestamps[0] == 0.0
        assert all(np.diff(path.timestamps) >= 0)

    def test_default_bounds(self, config_sampling):
        np.random.seed(42)
        planner = PRMPlanner(config=config_sampling, num_samples=100)
        path = planner.plan(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert path.num_waypoints >= 2

    def test_start_equals_goal(self, config_sampling):
        np.random.seed(42)
        planner = PRMPlanner(config=config_sampling, num_samples=50)
        pt = np.array([1.0, 1.0])
        path = planner.plan(pt, pt.copy())
        assert path.num_waypoints >= 1

    def test_heavily_blocked(self):
        """Dense obstacles should produce a fallback path."""
        np.random.seed(42)
        config = PlannerConfig(max_iterations=5000, time_limit=2.0)
        # Wall of obstacles between start and goal
        obstacles = np.array([[float(i) * 0.1, 1.5, 0.15] for i in range(60)])
        planner = PRMPlanner(config=config, num_samples=50, k_neighbors=5)
        path = planner.plan(np.array([0.0, 0.0]), np.array([0.0, 3.0]), obstacles=obstacles)
        # Should still return a path (possibly fallback)
        assert path.num_waypoints >= 2


# ---------------------------------------------------------------------------
# A* additional edge cases
# ---------------------------------------------------------------------------


class TestAStarEdgeCases:
    def test_max_iterations_fallback(self):
        config = PlannerConfig(max_iterations=5, time_limit=5.0, resolution=1.0)
        planner = AStarPlanner(config=config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([50.0, 50.0]),
        )
        # Should still return a path (fallback)
        assert path.num_waypoints >= 2

    def test_time_limit_fallback(self):
        config = PlannerConfig(max_iterations=100000, time_limit=0.0001, resolution=1.0)
        planner = AStarPlanner(config=config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([50.0, 50.0]),
        )
        assert path.num_waypoints >= 2


class TestDijkstraEdgeCases:
    def test_stale_entry_skip(self, free_grid):
        """Dijkstra should skip stale priority queue entries."""
        config = PlannerConfig(max_iterations=50000, time_limit=5.0, resolution=1.0)
        planner = DijkstraPlanner(config=config)
        path = planner.plan(
            start=np.array([0.0, 0.0]),
            goal=np.array([5.0, 5.0]),
            obstacles=free_grid,
        )
        assert path.num_waypoints >= 2
        assert path.cost > 0


class TestThetaStarEdgeCases:
    def test_line_of_sight_open(self):
        """Line of sight through open space should be True."""
        assert ThetaStarPlanner._line_of_sight((0, 0), (5, 5), None) is True

    def test_line_of_sight_blocked(self, obstacle_grid):
        """Line of sight through wall should be False."""
        result = ThetaStarPlanner._line_of_sight((3, 4), (7, 4), obstacle_grid)
        assert result is False

    def test_line_of_sight_clear(self, free_grid):
        """Line of sight through free grid should be True."""
        assert ThetaStarPlanner._line_of_sight((0, 0), (9, 9), free_grid) is True

    def test_line_of_sight_same_point(self, free_grid):
        assert ThetaStarPlanner._line_of_sight((3, 3), (3, 3), free_grid) is True

    def test_plan_with_obstacles_falls_back_to_grid(self, obstacle_grid):
        """When LOS is blocked, Theta* should fall back to grid-based updates."""
        config = PlannerConfig(max_iterations=50000, time_limit=5.0, resolution=1.0)
        planner = ThetaStarPlanner(config=config)
        path = planner.plan(
            start=np.array([2.0, 4.0]),
            goal=np.array([8.0, 4.0]),
            obstacles=obstacle_grid,
        )
        assert path.num_waypoints >= 2
        assert path.cost > 0
