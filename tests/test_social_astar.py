from __future__ import annotations

import math
from unittest.mock import MagicMock, call

import pytest

from navirl.core.types import Action, AgentState
from navirl.robots.baselines.social_astar import SocialCostAStarRobotController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(path=None, collision=False):
    """Return a mock backend with configurable shortest_path and collision."""
    backend = MagicMock()
    backend.shortest_path.return_value = path or []
    backend.check_obstacle_collision.return_value = collision
    return backend


def _make_state(
    agent_id=0,
    x=0.0,
    y=0.0,
    vx=0.0,
    vy=0.0,
    max_speed=1.0,
    kind="human",
    radius=0.3,
):
    return AgentState(
        agent_id=agent_id,
        kind=kind,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=0.0,
        goal_y=0.0,
        radius=radius,
        max_speed=max_speed,
    )


def _emit_event(event_type, agent_id, payload):
    """No-op event sink."""


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructorDefaults:
    def test_default_config_values(self):
        ctrl = SocialCostAStarRobotController()
        assert ctrl.goal_tolerance == pytest.approx(0.2)
        assert ctrl.replan_interval == 15
        assert ctrl.max_speed == pytest.approx(0.8)
        assert ctrl.slowdown_dist == pytest.approx(0.7)
        assert ctrl.target_lookahead == 3
        assert ctrl.velocity_smoothing == pytest.approx(0.4)
        assert ctrl.stop_speed == pytest.approx(0.02)
        assert ctrl.social_radius == pytest.approx(2.0)
        assert ctrl.personal_space == pytest.approx(0.8)
        assert ctrl.social_weight == pytest.approx(3.0)
        assert ctrl.crossing_penalty == pytest.approx(2.0)

    def test_custom_config_override(self):
        cfg = {
            "goal_tolerance": 0.5,
            "replan_interval": 10,
            "max_speed": 1.2,
            "slowdown_dist": 1.0,
            "target_lookahead": 5,
            "velocity_smoothing": 0.6,
            "stop_speed": 0.05,
            "social_radius": 3.0,
            "personal_space": 1.0,
            "social_weight": 4.0,
            "crossing_penalty": 3.0,
        }
        ctrl = SocialCostAStarRobotController(cfg)
        assert ctrl.goal_tolerance == pytest.approx(0.5)
        assert ctrl.replan_interval == 10
        assert ctrl.max_speed == pytest.approx(1.2)
        assert ctrl.slowdown_dist == pytest.approx(1.0)
        assert ctrl.target_lookahead == 5
        assert ctrl.velocity_smoothing == pytest.approx(0.6)
        assert ctrl.stop_speed == pytest.approx(0.05)
        assert ctrl.social_radius == pytest.approx(3.0)
        assert ctrl.personal_space == pytest.approx(1.0)
        assert ctrl.social_weight == pytest.approx(4.0)
        assert ctrl.crossing_penalty == pytest.approx(3.0)

    def test_initial_state_attributes(self):
        ctrl = SocialCostAStarRobotController()
        assert ctrl.robot_id == -1
        assert ctrl.start == (0.0, 0.0)
        assert ctrl.goal == (0.0, 0.0)
        assert ctrl.backend is None
        assert ctrl.path == []
        assert ctrl.path_idx == 0
        assert ctrl.last_pref == (0.0, 0.0)


# ---------------------------------------------------------------------------
# _point_to_line_distance tests
# ---------------------------------------------------------------------------


class TestPointToLineDistance:
    def setup_method(self):
        self.ctrl = SocialCostAStarRobotController()

    def test_normal_segment_perpendicular(self):
        # Point (1, 1) to segment from (0, 0) to (2, 0) -> distance = 1.0
        dist = self.ctrl._point_to_line_distance(1.0, 1.0, 0.0, 0.0, 2.0, 0.0)
        assert dist == pytest.approx(1.0)

    def test_degenerate_segment_zero_length(self):
        # Segment is a single point (3, 4), test point at origin
        dist = self.ctrl._point_to_line_distance(0.0, 0.0, 3.0, 4.0, 3.0, 4.0)
        assert dist == pytest.approx(5.0)

    def test_point_on_line(self):
        # Point lies exactly on the segment
        dist = self.ctrl._point_to_line_distance(1.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert dist == pytest.approx(0.0)

    def test_point_at_start_endpoint(self):
        dist = self.ctrl._point_to_line_distance(0.0, 0.0, 0.0, 0.0, 4.0, 0.0)
        assert dist == pytest.approx(0.0)

    def test_point_at_end_endpoint(self):
        dist = self.ctrl._point_to_line_distance(4.0, 0.0, 0.0, 0.0, 4.0, 0.0)
        assert dist == pytest.approx(0.0)

    def test_point_beyond_start(self):
        # Point projects before the start of the segment
        dist = self.ctrl._point_to_line_distance(-1.0, 0.0, 0.0, 0.0, 4.0, 0.0)
        assert dist == pytest.approx(1.0)

    def test_point_beyond_end(self):
        # Point projects past the end of the segment
        dist = self.ctrl._point_to_line_distance(5.0, 3.0, 0.0, 0.0, 4.0, 0.0)
        # Closest is (4, 0), distance = sqrt(1 + 9) = sqrt(10)
        assert dist == pytest.approx(math.sqrt(10.0))

    def test_diagonal_segment(self):
        # Segment from (0,0) to (1,1), point at (1,0)
        # Projection t = 0.5, proj = (0.5, 0.5), dist = sqrt(0.25+0.25)
        dist = self.ctrl._point_to_line_distance(1.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        assert dist == pytest.approx(math.sqrt(0.5))


# ---------------------------------------------------------------------------
# _compute_social_cost tests
# ---------------------------------------------------------------------------


class TestComputeSocialCost:
    def setup_method(self):
        self.ctrl = SocialCostAStarRobotController()
        self.ctrl.robot_id = 0

    def test_empty_states(self):
        cost = self.ctrl._compute_social_cost((0.0, 0.0), {})
        assert cost == 0.0

    def test_skip_self(self):
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0)}
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        assert cost == 0.0

    def test_distant_human_no_cost(self):
        # Human at (10, 10) is far outside social_radius=2.0
        states = {1: _make_state(agent_id=1, x=10.0, y=10.0)}
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        assert cost == 0.0

    def test_human_within_social_radius_but_outside_personal_space(self):
        # Human at (1.5, 0) is within social_radius=2.0 but outside personal_space=0.8
        # No proximity cost, no crossing penalty (stationary)
        states = {1: _make_state(agent_id=1, x=1.5, y=0.0)}
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        assert cost == 0.0

    def test_human_inside_personal_space(self):
        # Human at (0.4, 0) -> dist = 0.4, inside personal_space=0.8
        # proximity_cost = 3.0 * (0.8 - 0.4) / 0.8 = 3.0 * 0.5 = 1.5
        # social_cost += 1.5^2 = 2.25
        states = {1: _make_state(agent_id=1, x=0.4, y=0.0)}
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        expected_prox = 3.0 * (0.8 - 0.4) / 0.8
        assert cost == pytest.approx(expected_prox**2)

    def test_human_at_same_position(self):
        # Human at (0, 0) -> dist = 0, inside personal_space
        # proximity_cost = 3.0 * (0.8 - 0) / 0.8 = 3.0
        # social_cost += 9.0
        states = {1: _make_state(agent_id=1, x=0.0, y=0.0)}
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        assert cost == pytest.approx(9.0)

    def test_moving_human_crossing_penalty(self):
        # Human at (0, 1) moving in +x direction at speed 1.0 (vx=1.0, vy=0.0)
        # Future position: (2.0, 1.0). Path from (0,1) to (2,1).
        # Robot pos (1.0, 0.5): perpendicular distance to that line = 0.5
        # 0.5 < personal_space=0.8, so crossing_cost = 2.0 * (0.8 - 0.5) = 0.6
        # Also dist from robot (1,0.5) to human (0,1) = sqrt(1 + 0.25) ~ 1.118 < 2.0 (social radius)
        # but 1.118 > 0.8 (personal_space), so no proximity cost
        states = {1: _make_state(agent_id=1, x=0.0, y=1.0, vx=1.0, vy=0.0)}
        cost = self.ctrl._compute_social_cost((1.0, 0.5), states)
        expected_crossing = 2.0 * (0.8 - 0.5)
        assert cost == pytest.approx(expected_crossing)

    def test_moving_human_no_crossing_when_far_from_path(self):
        # Human at (0, 0) moving +x at 1.0 m/s. Future: (2, 0). Path along x-axis.
        # Robot at (1, 5): line distance to x-axis segment > personal_space
        states = {1: _make_state(agent_id=1, x=0.0, y=0.0, vx=1.0, vy=0.0)}
        cost = self.ctrl._compute_social_cost((1.0, 5.0), states)
        assert cost == 0.0

    def test_multiple_humans(self):
        # Two humans inside personal space; costs should accumulate
        states = {
            1: _make_state(agent_id=1, x=0.3, y=0.0),
            2: _make_state(agent_id=2, x=0.0, y=0.3),
        }
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        # Each contributes a quadratic proximity cost
        assert cost > 0.0
        # Each individually
        cost1 = self.ctrl._compute_social_cost((0.0, 0.0), {1: states[1]})
        cost2 = self.ctrl._compute_social_cost((0.0, 0.0), {2: states[2]})
        assert cost == pytest.approx(cost1 + cost2)

    def test_slow_human_no_crossing_penalty(self):
        # Human moving at speed < 0.1 should not trigger crossing penalty
        states = {1: _make_state(agent_id=1, x=0.0, y=0.5, vx=0.05, vy=0.0)}
        cost = self.ctrl._compute_social_cost((0.0, 0.0), states)
        # Only proximity cost, dist = 0.5 < personal_space = 0.8
        prox = 3.0 * (0.8 - 0.5) / 0.8
        assert cost == pytest.approx(prox**2)


# ---------------------------------------------------------------------------
# reset tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_sets_state(self):
        ctrl = SocialCostAStarRobotController()
        backend = _make_backend(path=[(0.0, 0.0), (5.0, 5.0)])
        ctrl.reset(1, (0.0, 0.0), (5.0, 5.0), backend)

        assert ctrl.robot_id == 1
        assert ctrl.start == (0.0, 0.0)
        assert ctrl.goal == (5.0, 5.0)
        assert ctrl.backend is backend
        assert ctrl.last_pref == (0.0, 0.0)
        assert ctrl._human_positions == {}
        assert ctrl._human_velocities == {}

    def test_reset_plans_initial_path(self):
        path = [(0.0, 0.0), (2.5, 2.5), (5.0, 5.0)]
        backend = _make_backend(path=path)
        ctrl = SocialCostAStarRobotController()
        ctrl.reset(1, (0.0, 0.0), (5.0, 5.0), backend)

        backend.shortest_path.assert_called_once_with((0.0, 0.0), (5.0, 5.0))
        assert ctrl.path_idx == 0
        assert len(ctrl.path) > 0

    def test_reset_with_empty_path(self):
        backend = _make_backend(path=[])
        ctrl = SocialCostAStarRobotController()
        ctrl.reset(1, (0.0, 0.0), (5.0, 5.0), backend)
        # _plan falls back to [goal] when path is empty
        assert ctrl.path == [(5.0, 5.0)]


# ---------------------------------------------------------------------------
# step tests
# ---------------------------------------------------------------------------


class TestStep:
    def _setup_ctrl(self, cfg=None, path=None):
        """Create a controller and reset it with a mock backend."""
        ctrl = SocialCostAStarRobotController(cfg)
        p = path or [(0.0, 0.0), (2.5, 0.0), (5.0, 0.0)]
        backend = _make_backend(path=p)
        ctrl.reset(0, (0.0, 0.0), (5.0, 0.0), backend)
        return ctrl, backend

    def test_goal_reached_returns_done(self):
        ctrl, _ = self._setup_ctrl()
        # Place robot at goal
        states = {0: _make_state(agent_id=0, x=5.0, y=0.0, kind="robot")}
        action = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert action.behavior == "DONE"
        assert action.pref_vx == pytest.approx(0.0)
        assert action.pref_vy == pytest.approx(0.0)

    def test_goal_within_tolerance_returns_done(self):
        ctrl, _ = self._setup_ctrl()
        # Place robot within goal_tolerance=0.2
        states = {0: _make_state(agent_id=0, x=4.9, y=0.0, kind="robot")}
        action = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert action.behavior == "DONE"

    def test_stationary_target_returns_wait(self):
        # When dist_target < 1e-8, returns WAIT
        ctrl, _ = self._setup_ctrl(
            path=[(0.0, 0.0)],
            cfg={"target_lookahead": 1, "goal_tolerance": 0.01},
        )
        # Robot at the only path point but not at goal (goal is at 5,0)
        # _current_target returns (0,0), robot at (0,0), dist < 1e-8 -> WAIT
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        action = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert action.behavior == "WAIT"

    def test_normal_navigation_returns_social_nav(self):
        ctrl, _ = self._setup_ctrl()
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        action = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert action.behavior == "SOCIAL_NAV"
        # Should have positive x velocity toward goal at (5, 0)
        assert action.pref_vx > 0.0

    def test_velocity_smoothing(self):
        ctrl, _ = self._setup_ctrl(cfg={"velocity_smoothing": 0.5})
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}

        # First step: last_pref = (0, 0), so smoothed = 0.5 * raw
        action1 = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        vx1 = action1.pref_vx

        # Second step from same position: smoothed with previous velocity
        action2 = ctrl.step(2, 0.2, 0.1, states, _emit_event)
        vx2 = action2.pref_vx

        # Second velocity should be larger because it builds on previous
        assert vx2 > vx1

    def test_social_cost_reduces_speed(self):
        ctrl, _ = self._setup_ctrl()
        # Step without humans nearby
        states_no_human = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        action_free = ctrl.step(1, 0.1, 0.1, states_no_human, _emit_event)

        # Reset smoothing state
        ctrl.last_pref = (0.0, 0.0)

        # Step with a human very close (inside personal space)
        states_crowded = {
            0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot"),
            1: _make_state(agent_id=1, x=0.3, y=0.0),
        }
        action_social = ctrl.step(2, 0.2, 0.1, states_crowded, _emit_event)

        # Speed with social cost should be less
        speed_free = math.hypot(action_free.pref_vx, action_free.pref_vy)
        speed_social = math.hypot(action_social.pref_vx, action_social.pref_vy)
        assert speed_social < speed_free

    def test_replanning_at_interval(self):
        ctrl, backend = self._setup_ctrl(cfg={"replan_interval": 5})
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        emit = MagicMock()

        # step=0 triggers replan (0 % 5 == 0)
        ctrl.step(0, 0.0, 0.1, states, emit)
        # shortest_path called once during reset + once during replan
        assert backend.shortest_path.call_count == 2
        emit.assert_called_once()
        assert emit.call_args[0][0] == "robot_social_replan"

    def test_no_replan_off_interval(self):
        ctrl, backend = self._setup_ctrl(cfg={"replan_interval": 5})
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        emit = MagicMock()

        # step=1 does NOT trigger replan (1 % 5 != 0)
        ctrl.step(1, 0.1, 0.1, states, emit)
        # shortest_path only called during reset
        assert backend.shortest_path.call_count == 1
        emit.assert_not_called()

    def test_waypoint_advance(self):
        path = [(0.0, 0.0), (0.1, 0.0), (5.0, 0.0)]
        ctrl, _ = self._setup_ctrl(
            path=path,
            cfg={"goal_tolerance": 0.2, "target_lookahead": 1},
        )
        # Robot very close to second waypoint -> should advance path_idx
        states = {0: _make_state(agent_id=0, x=0.1, y=0.0, kind="robot")}
        ctrl.path_idx = 1  # Currently targeting second point
        ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert ctrl.path_idx >= 2

    def test_human_positions_and_velocities_tracked(self):
        ctrl, _ = self._setup_ctrl()
        states = {
            0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot"),
            1: _make_state(agent_id=1, x=3.0, y=4.0, vx=0.5, vy=-0.5),
        }
        ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert 1 in ctrl._human_positions
        assert ctrl._human_positions[1] == (3.0, 4.0)
        assert ctrl._human_velocities[1] == (0.5, -0.5)
        assert 0 not in ctrl._human_positions

    def test_direction_toward_goal(self):
        ctrl, _ = self._setup_ctrl()
        # Goal at (5, 0), robot at (0, 0) -> velocity should be in +x direction
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        action = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert action.pref_vx > 0.0
        # vy should be zero or near-zero
        assert abs(action.pref_vy) < 0.01

    def test_step_returns_action_type(self):
        ctrl, _ = self._setup_ctrl()
        states = {0: _make_state(agent_id=0, x=0.0, y=0.0, kind="robot")}
        action = ctrl.step(1, 0.1, 0.1, states, _emit_event)
        assert isinstance(action, Action)


# ---------------------------------------------------------------------------
# _social_astar tests
# ---------------------------------------------------------------------------


class TestSocialAstar:
    def setup_method(self):
        self.ctrl = SocialCostAStarRobotController()
        self.ctrl.robot_id = 0
        self.ctrl.backend = _make_backend()

    def test_empty_path_returns_goal(self):
        self.ctrl.backend.shortest_path.return_value = []
        result = self.ctrl._social_astar((0, 0), (5, 5), {})
        assert result == [(5, 5)]

    def test_short_path_returned_unchanged(self):
        path = [(0, 0), (5, 5)]
        self.ctrl.backend.shortest_path.return_value = path
        result = self.ctrl._social_astar((0, 0), (5, 5), {})
        assert result == path

    def test_longer_path_keeps_endpoints(self):
        path = [(0, 0), (2, 2), (4, 4), (5, 5)]
        self.ctrl.backend.shortest_path.return_value = path
        self.ctrl.backend.check_obstacle_collision.return_value = False
        result = self.ctrl._social_astar((0, 0), (5, 5), {})
        assert result[0] == path[0]
        assert result[-1] == path[-1]
        assert len(result) == len(path)

    def test_collision_alternatives_skipped(self):
        path = [(0, 0), (2, 2), (5, 5)]
        self.ctrl.backend.shortest_path.return_value = path
        # All alternatives collide, so original waypoint is kept
        self.ctrl.backend.check_obstacle_collision.return_value = True
        result = self.ctrl._social_astar((0, 0), (5, 5), {})
        assert result[1] == path[1]
