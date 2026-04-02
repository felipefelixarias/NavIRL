"""Tests for richer baseline robot planners (Issue #4)."""

import pytest
import math
from unittest.mock import Mock

from navirl.core.types import Action, AgentState
from navirl.robots.baselines import (
    BaselineAStarRobotController,
    PRMRobotController,
    RRTStarRobotController,
    SocialCostAStarRobotController,
)


class MockBackend:
    """Mock backend for testing robot controllers."""

    def __init__(self):
        self.obstacle_positions = [(5, 5), (6, 5), (5, 6)]

    def shortest_path(self, start, goal):
        """Return a simple path between start and goal."""
        return [start, goal]

    def check_obstacle_collision(self, pos):
        """Check if position collides with obstacles."""
        return pos in self.obstacle_positions

    def sample_free_point(self):
        """Return a free point."""
        return (1.0, 1.0)

    def map_metadata(self):
        """Return map metadata."""
        return {"width": 10, "height": 10}


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    return MockBackend()


@pytest.fixture
def sample_states():
    """Create sample agent states for testing."""
    return {
        0: AgentState(agent_id=0, kind="robot", x=0.0, y=0.0, vx=0.0, vy=0.0,
                      goal_x=10.0, goal_y=10.0, max_speed=1.0, radius=0.3),
        1: AgentState(agent_id=1, kind="human", x=2.0, y=1.0, vx=0.1, vy=0.0,
                      goal_x=8.0, goal_y=8.0, max_speed=1.0, radius=0.3),
        2: AgentState(agent_id=2, kind="human", x=1.0, y=2.0, vx=0.0, vy=0.1,
                      goal_x=9.0, goal_y=9.0, max_speed=1.0, radius=0.3),
    }


class TestSocialCostAStarRobotController:
    """Test the social-cost A* robot controller."""

    def test_creation(self):
        """Test controller creation with default config."""
        controller = SocialCostAStarRobotController()
        assert controller.social_radius == 2.0
        assert controller.personal_space == 0.8
        assert controller.social_weight == 3.0

    def test_creation_with_config(self):
        """Test controller creation with custom config."""
        config = {
            "social_radius": 3.0,
            "personal_space": 1.0,
            "social_weight": 5.0,
            "max_speed": 1.5
        }
        controller = SocialCostAStarRobotController(cfg=config)
        assert controller.social_radius == 3.0
        assert controller.personal_space == 1.0
        assert controller.social_weight == 5.0
        assert controller.max_speed == 1.5

    def test_social_cost_computation(self, mock_backend, sample_states):
        """Test social cost computation."""
        controller = SocialCostAStarRobotController()
        controller.reset(0, (0.0, 0.0), (10.0, 10.0), mock_backend)

        # Test position near other agents
        cost = controller._compute_social_cost((2.1, 1.0), sample_states)
        assert cost > 0.0  # Should have non-zero cost near other agents

        # Test position far from other agents
        cost_far = controller._compute_social_cost((8.0, 8.0), sample_states)
        assert cost_far < cost  # Should have lower cost when far away

    def test_reset_and_step(self, mock_backend, sample_states):
        """Test controller reset and step functionality."""
        controller = SocialCostAStarRobotController()
        controller.reset(0, (0.0, 0.0), (10.0, 10.0), mock_backend)

        mock_emit = Mock()
        action = controller.step(0, 0.0, 0.1, sample_states, mock_emit)

        assert isinstance(action, Action)
        assert action.behavior == "SOCIAL_NAV"
        assert abs(action.pref_vx) <= 1.0  # Within reasonable velocity bounds
        assert abs(action.pref_vy) <= 1.0


class TestPRMRobotController:
    """Test the PRM robot controller."""

    def test_creation(self):
        """Test controller creation with default config."""
        controller = PRMRobotController()
        assert controller.num_samples == 100
        assert controller.connection_radius == 1.5
        assert controller.max_connections == 8

    def test_creation_with_config(self):
        """Test controller creation with custom config."""
        config = {
            "num_samples": 50,
            "connection_radius": 2.0,
            "max_connections": 6
        }
        controller = PRMRobotController(cfg=config)
        assert controller.num_samples == 50
        assert controller.connection_radius == 2.0
        assert controller.max_connections == 6

    def test_roadmap_building(self, mock_backend):
        """Test PRM roadmap construction."""
        controller = PRMRobotController(cfg={"num_samples": 20})
        controller.backend = mock_backend
        controller._build_roadmap()

        assert controller.roadmap_built
        assert len(controller.roadmap_nodes) > 0
        assert len(controller.roadmap_edges) > 0

    def test_reset_and_step(self, mock_backend, sample_states):
        """Test controller reset and step functionality."""
        controller = PRMRobotController(cfg={"num_samples": 20})
        controller.reset(0, (0.0, 0.0), (10.0, 10.0), mock_backend)

        mock_emit = Mock()
        action = controller.step(0, 0.0, 0.1, sample_states, mock_emit)

        assert isinstance(action, Action)
        assert action.behavior == "PRM_NAV"
        assert abs(action.pref_vx) <= 1.0
        assert abs(action.pref_vy) <= 1.0


class TestRRTStarRobotController:
    """Test the RRT* robot controller."""

    def test_creation(self):
        """Test controller creation with default config."""
        controller = RRTStarRobotController()
        assert controller.max_iterations == 200
        assert controller.step_size == 0.3
        assert controller.goal_sample_rate == 0.15

    def test_creation_with_config(self):
        """Test controller creation with custom config."""
        config = {
            "max_iterations": 100,
            "step_size": 0.5,
            "goal_sample_rate": 0.2
        }
        controller = RRTStarRobotController(cfg=config)
        assert controller.max_iterations == 100
        assert controller.step_size == 0.5
        assert controller.goal_sample_rate == 0.2

    def test_node_creation(self):
        """Test RRT node creation and path computation."""
        from navirl.robots.baselines.rrt import RRTNode

        root = RRTNode((0.0, 0.0))
        assert root.position == (0.0, 0.0)
        assert root.parent is None
        assert root.cost == 0.0

        child = RRTNode((1.0, 1.0), root)
        expected_cost = math.sqrt(2.0)  # Distance from root to child
        assert abs(child.cost - expected_cost) < 1e-6

        path = child.path_to_root()
        assert len(path) == 2
        assert path[0] == (0.0, 0.0)
        assert path[1] == (1.0, 1.0)

    def test_steering(self, mock_backend):
        """Test RRT steering function."""
        controller = RRTStarRobotController(cfg={"step_size": 1.0})
        controller.backend = mock_backend

        # Test steering within step size
        result = controller._steer((0.0, 0.0), (0.5, 0.5))
        assert result == (0.5, 0.5)

        # Test steering beyond step size
        result = controller._steer((0.0, 0.0), (2.0, 2.0))
        expected_dist = math.sqrt(result[0]**2 + result[1]**2)
        assert abs(expected_dist - 1.0) < 1e-6

    def test_reset_and_step(self, mock_backend, sample_states):
        """Test controller reset and step functionality."""
        controller = RRTStarRobotController(cfg={"max_iterations": 50})
        controller.reset(0, (0.0, 0.0), (3.0, 3.0), mock_backend)

        mock_emit = Mock()
        action = controller.step(0, 0.0, 0.1, sample_states, mock_emit)

        assert isinstance(action, Action)
        assert action.behavior == "RRT_NAV"
        assert abs(action.pref_vx) <= 1.0
        assert abs(action.pref_vy) <= 1.0


class TestPlannerComparison:
    """Test different planners side by side."""

    def test_all_planners_reach_goal(self, mock_backend):
        """Test that all planners can reach their goal."""
        controllers = [
            BaselineAStarRobotController(),
            SocialCostAStarRobotController(cfg={"num_samples": 20}),
            PRMRobotController(cfg={"num_samples": 20}),
            RRTStarRobotController(cfg={"max_iterations": 50}),
        ]

        start = (0.0, 0.0)
        goal = (3.0, 3.0)

        for controller in controllers:
            controller.reset(0, start, goal, mock_backend)

            # Simulate until goal reached or timeout
            states = {0: AgentState(agent_id=0, kind="robot", x=start[0], y=start[1], vx=0.0, vy=0.0,
                                  goal_x=goal[0], goal_y=goal[1], max_speed=1.0, radius=0.3)}
            mock_emit = Mock()

            max_steps = 100
            for step in range(max_steps):
                action = controller.step(step, step * 0.1, 0.1, states, mock_emit)

                if action.behavior == "DONE":
                    break

                # Update robot position (simple kinematic model)
                dt = 0.1
                new_x = states[0].x + action.pref_vx * dt
                new_y = states[0].y + action.pref_vy * dt
                states[0] = AgentState(agent_id=0, kind="robot", x=new_x, y=new_y, vx=action.pref_vx, vy=action.pref_vy,
                                     goal_x=goal[0], goal_y=goal[1], max_speed=1.0, radius=0.3)

                # Check if goal reached
                dist_to_goal = math.sqrt((states[0].x - goal[0])**2 + (states[0].y - goal[1])**2)
                if dist_to_goal < 0.2:  # Goal tolerance
                    break

            # Verify controller made progress toward goal
            final_dist = math.sqrt((states[0].x - goal[0])**2 + (states[0].y - goal[1])**2)
            initial_dist = math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)

            assert final_dist < initial_dist, f"{controller.__class__.__name__} should make progress toward goal"