"""Tests for navirl/robots/baselines/prm.py — PRMRobotController.

Covers: initialization, roadmap building, Dijkstra search, path planning,
step behavior, goal reaching, and velocity smoothing.
"""

from __future__ import annotations

import math

import pytest

from navirl.core.types import Action, AgentState
from navirl.robots.baselines.prm import PRMRobotController


class _MockBackend:
    """Minimal mock backend for PRM tests."""

    def __init__(self, obstacles=None, width=20.0, height=20.0):
        self._obstacles = obstacles or set()
        self._width = width
        self._height = height

    def map_metadata(self):
        return {"width": self._width, "height": self._height}

    def check_obstacle_collision(self, pos):
        """Return True if pos is in collision."""
        x, y = pos
        for ox, oy, r in self._obstacles:
            if math.hypot(x - ox, y - oy) < r:
                return True
        return False

    def shortest_path(self, start, goal):
        """Fallback path: just return direct."""
        return [start, goal]


def _make_state(robot_id=0, x=0.0, y=0.0, goal_x=10.0, goal_y=10.0):
    return AgentState(
        agent_id=robot_id,
        kind="robot",
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=0.3,
        max_speed=1.0,
    )


def _noop_event(event_type, agent_id, payload):
    pass


# ============================================================
# Initialization
# ============================================================


class TestPRMInit:
    def test_default_config(self):
        ctrl = PRMRobotController()
        assert ctrl.num_samples == 100
        assert ctrl.connection_radius == 1.5
        assert ctrl.max_connections == 8

    def test_custom_config(self):
        ctrl = PRMRobotController({"num_samples": 50, "max_speed": 1.5})
        assert ctrl.num_samples == 50
        assert ctrl.max_speed == 1.5

    def test_reset(self):
        ctrl = PRMRobotController({"num_samples": 20})
        backend = _MockBackend()
        ctrl.reset(0, (1.0, 1.0), (10.0, 10.0), backend)
        assert ctrl.robot_id == 0
        assert ctrl.start == (1.0, 1.0)
        assert ctrl.goal == (10.0, 10.0)
        assert ctrl.backend is backend
        # Reset should force roadmap rebuild
        assert ctrl.roadmap_built is True  # built during reset->_plan


# ============================================================
# Roadmap building
# ============================================================


class TestRoadmapBuilding:
    def test_build_roadmap_creates_nodes(self):
        ctrl = PRMRobotController({"num_samples": 30})
        ctrl.backend = _MockBackend()
        ctrl._build_roadmap()
        assert ctrl.roadmap_built is True
        assert len(ctrl.roadmap_nodes) > 0
        assert len(ctrl.roadmap_nodes) <= 30

    def test_build_roadmap_with_obstacles(self):
        # Obstacle at center blocks some samples
        obstacles = [(10.0, 10.0, 5.0)]
        ctrl = PRMRobotController({"num_samples": 30})
        ctrl.backend = _MockBackend(obstacles=obstacles)
        ctrl._build_roadmap()
        # Some nodes should still exist in free space
        assert len(ctrl.roadmap_nodes) > 0
        # No node should be inside the obstacle
        for nx, ny in ctrl.roadmap_nodes:
            assert math.hypot(nx - 10.0, ny - 10.0) >= 5.0

    def test_roadmap_edges_symmetric(self):
        ctrl = PRMRobotController({"num_samples": 20, "connection_radius": 5.0})
        ctrl.backend = _MockBackend()
        ctrl._build_roadmap()
        for i, neighbors in ctrl.roadmap_edges.items():
            for j in neighbors:
                assert i in ctrl.roadmap_edges[j], f"Edge {i}->{j} not symmetric"

    def test_no_backend_returns_early(self):
        ctrl = PRMRobotController()
        ctrl.backend = None
        ctrl._build_roadmap()
        assert not ctrl.roadmap_built


# ============================================================
# Dijkstra search
# ============================================================


class TestDijkstraSearch:
    def test_path_found(self):
        ctrl = PRMRobotController()
        ctrl.roadmap_nodes = [(0, 0), (1, 0), (2, 0), (3, 0)]
        ctrl.roadmap_edges = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        path = ctrl._dijkstra_search(0, 3)
        assert path == [0, 1, 2, 3]

    def test_no_path(self):
        ctrl = PRMRobotController()
        ctrl.roadmap_nodes = [(0, 0), (1, 0), (5, 5), (6, 5)]
        ctrl.roadmap_edges = {0: [1], 1: [0], 2: [3], 3: [2]}
        path = ctrl._dijkstra_search(0, 3)
        assert path == []

    def test_invalid_indices(self):
        ctrl = PRMRobotController()
        assert ctrl._dijkstra_search(-1, 0) == []
        assert ctrl._dijkstra_search(0, -1) == []

    def test_same_start_and_goal(self):
        ctrl = PRMRobotController()
        ctrl.roadmap_nodes = [(0, 0), (1, 0)]
        ctrl.roadmap_edges = {0: [1], 1: [0]}
        path = ctrl._dijkstra_search(0, 0)
        assert path == [0]


# ============================================================
# Nearest node
# ============================================================


class TestFindNearestNode:
    def test_basic(self):
        ctrl = PRMRobotController()
        ctrl.roadmap_nodes = [(0, 0), (5, 5), (10, 10)]
        idx = ctrl._find_nearest_roadmap_node((4, 4))
        assert idx == 1

    def test_empty_roadmap(self):
        ctrl = PRMRobotController()
        ctrl.roadmap_nodes = []
        assert ctrl._find_nearest_roadmap_node((0, 0)) == -1


# ============================================================
# Step behavior
# ============================================================


class TestPRMStep:
    def test_goal_reached_returns_done(self):
        ctrl = PRMRobotController({"goal_tolerance": 0.5, "num_samples": 10})
        backend = _MockBackend()
        ctrl.reset(0, (9.9, 9.9), (10.0, 10.0), backend)
        states = {0: _make_state(0, x=9.9, y=9.9, goal_x=10.0, goal_y=10.0)}
        action = ctrl.step(1, 0.1, 0.04, states, _noop_event)
        assert action.behavior == "DONE"
        assert action.pref_vx == 0.0
        assert action.pref_vy == 0.0

    def test_step_returns_action(self):
        ctrl = PRMRobotController({"num_samples": 10, "replan_interval": 100})
        backend = _MockBackend()
        ctrl.reset(0, (0.0, 0.0), (10.0, 10.0), backend)
        states = {0: _make_state(0, x=0.0, y=0.0)}
        action = ctrl.step(1, 0.1, 0.04, states, _noop_event)
        assert isinstance(action, Action)
        speed = math.hypot(action.pref_vx, action.pref_vy)
        assert speed <= ctrl.max_speed + 0.01

    def test_velocity_smoothing(self):
        ctrl = PRMRobotController(
            {
                "num_samples": 10,
                "velocity_smoothing": 0.5,
                "replan_interval": 100,
            }
        )
        backend = _MockBackend()
        ctrl.reset(0, (0.0, 0.0), (10.0, 0.0), backend)
        states = {0: _make_state(0, x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)}
        ctrl.step(1, 0.04, 0.04, states, _noop_event)
        a2 = ctrl.step(2, 0.08, 0.04, states, _noop_event)
        # Second action should be blended with first via smoothing
        assert isinstance(a2, Action)

    def test_replan_trigger(self):
        ctrl = PRMRobotController({"num_samples": 10, "replan_interval": 5})
        backend = _MockBackend()
        ctrl.reset(0, (0.0, 0.0), (10.0, 10.0), backend)
        states = {0: _make_state(0, x=1.0, y=1.0)}
        # Step 5 should trigger replan
        events = []

        def capture_event(etype, aid, payload):
            events.append(etype)

        ctrl.step(5, 0.2, 0.04, states, capture_event)
        assert "robot_prm_replan" in events

    def test_slowdown_near_target(self):
        ctrl = PRMRobotController(
            {
                "num_samples": 10,
                "slowdown_dist": 1.0,
                "max_speed": 2.0,
                "replan_interval": 100,
            }
        )
        backend = _MockBackend()
        ctrl.reset(0, (0.0, 0.0), (0.5, 0.0), backend)
        states = {0: _make_state(0, x=0.0, y=0.0, goal_x=0.5, goal_y=0.0)}
        action = ctrl.step(1, 0.04, 0.04, states, _noop_event)
        speed = math.hypot(action.pref_vx, action.pref_vy)
        # Should be slowed down since we're within slowdown_dist
        assert speed < 2.0


# ============================================================
# Edge validation
# ============================================================


class TestEdgeValidation:
    def test_clear_edge(self):
        ctrl = PRMRobotController()
        ctrl.backend = _MockBackend()
        assert ctrl._is_edge_valid((0, 0), (5, 5)) is True

    def test_blocked_edge(self):
        obstacles = [(2.5, 2.5, 2.0)]
        ctrl = PRMRobotController()
        ctrl.backend = _MockBackend(obstacles=obstacles)
        assert ctrl._is_edge_valid((0, 0), (5, 5)) is False

    def test_valid_position_no_backend(self):
        ctrl = PRMRobotController()
        ctrl.backend = None
        # Should return False when backend is missing
        assert ctrl._is_valid_position((0, 0)) is False


# ============================================================
# Map bounds
# ============================================================


class TestMapBounds:
    def test_from_backend_metadata(self):
        ctrl = PRMRobotController()
        ctrl.backend = _MockBackend(width=30.0, height=25.0)
        bounds = ctrl._get_map_bounds()
        assert bounds == (0.0, 0.0, 30.0, 25.0)

    def test_default_bounds_no_metadata(self):
        ctrl = PRMRobotController()
        ctrl.backend = object()  # no map_metadata
        bounds = ctrl._get_map_bounds()
        assert bounds == (0.0, 0.0, 20.0, 20.0)
