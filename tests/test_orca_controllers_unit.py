"""Unit tests for navirl/humans/orca/controller.py and orca_plus/controller.py.

Tests the ORCA and ORCA+ human controllers with mocked backends, covering
path planning, velocity smoothing, goal swapping, doorway token logic,
anisotropic space scaling, group cohesion, speed profiling, and hesitation.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from navirl.core.types import Action, AgentState
from navirl.humans.orca.controller import ORCAHumanController
from navirl.humans.orca_plus.controller import ORCAPlusHumanController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    agent_id: int,
    x: float,
    y: float,
    goal_x: float = 5.0,
    goal_y: float = 5.0,
    vx: float = 0.0,
    vy: float = 0.0,
    max_speed: float = 1.0,
    kind: str = "human",
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind=kind,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=0.3,
        max_speed=max_speed,
    )


def _noop_emit(event_type, agent_id, payload):
    """No-op event sink."""


def _make_mock_backend(path=None):
    backend = MagicMock()
    if path is not None:
        backend.shortest_path.return_value = path
    else:
        backend.shortest_path.return_value = [(5.0, 5.0)]
    return backend


# ---------------------------------------------------------------------------
# ORCAHumanController tests
# ---------------------------------------------------------------------------


class TestORCAHumanControllerInit:
    def test_default_config(self):
        ctrl = ORCAHumanController()
        assert ctrl.goal_tolerance == pytest.approx(0.22)
        assert ctrl.waypoint_tolerance == pytest.approx(0.2)
        assert ctrl.lookahead == 4
        assert ctrl.velocity_smoothing == pytest.approx(0.25)

    def test_custom_config(self):
        ctrl = ORCAHumanController(cfg={
            "goal_tolerance": 0.5,
            "lookahead": 8,
            "velocity_smoothing": 0.6,
        })
        assert ctrl.goal_tolerance == pytest.approx(0.5)
        assert ctrl.lookahead == 8
        assert ctrl.velocity_smoothing == pytest.approx(0.6)


class TestORCAHumanControllerReset:
    def test_reset_with_backend(self):
        ctrl = ORCAHumanController()
        backend = _make_mock_backend([(2.0, 3.0), (5.0, 5.0)])
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (1.0, 1.0)},
            goals={1: (5.0, 5.0), 2: (6.0, 6.0)},
            backend=backend,
        )
        assert ctrl.human_ids == [1, 2]
        assert 1 in ctrl.paths
        assert 2 in ctrl.paths

    def test_reset_without_backend(self):
        ctrl = ORCAHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=None,
        )
        # Without backend, path is direct to goal
        assert ctrl.paths[1] == [(5.0, 5.0)]

    def test_reset_path_failure_fallback(self):
        """If backend.shortest_path raises, controller falls back to direct path."""
        backend = MagicMock()
        backend.shortest_path.side_effect = RuntimeError("path error")
        ctrl = ORCAHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=backend,
        )
        assert ctrl.paths[1] == [(5.0, 5.0)]

    def test_reset_clears_previous_state(self):
        ctrl = ORCAHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
        )
        ctrl.last_pref[1] = (0.5, 0.5)
        # Reset again with different agents
        ctrl.reset(
            human_ids=[2],
            starts={2: (1.0, 1.0)},
            goals={2: (6.0, 6.0)},
        )
        assert 1 not in ctrl.last_pref
        assert ctrl.human_ids == [2]


class TestORCAPlanPath:
    def test_plan_path_no_backend(self):
        ctrl = ORCAHumanController()
        path = ctrl._plan_path((0.0, 0.0), (5.0, 5.0))
        assert path == [(5.0, 5.0)]

    def test_plan_path_with_backend_empty_result(self):
        backend = _make_mock_backend([])
        ctrl = ORCAHumanController()
        ctrl.backend = backend
        path = ctrl._plan_path((0.0, 0.0), (5.0, 5.0))
        assert path == [(5.0, 5.0)]

    def test_plan_path_with_backend_valid(self):
        backend = _make_mock_backend([(1.0, 1.0), (3.0, 3.0), (5.0, 5.0)])
        ctrl = ORCAHumanController()
        ctrl.backend = backend
        path = ctrl._plan_path((0.0, 0.0), (5.0, 5.0))
        assert len(path) == 3
        assert path[0] == (1.0, 1.0)
        assert path[-1] == (5.0, 5.0)


class TestORCAGoalSwap:
    def test_swap_when_at_goal(self):
        ctrl = ORCAHumanController(cfg={"goal_tolerance": 0.5})
        ctrl.human_ids = [1]
        ctrl.starts = {1: (0.0, 0.0)}
        ctrl.goals = {1: (5.0, 5.0)}
        state = _make_state(1, x=5.0, y=5.0)

        events = []
        def emit(etype, aid, payload):
            events.append((etype, aid, payload))

        swapped = ctrl._maybe_swap_goal(1, state, emit)
        assert swapped
        assert ctrl.goals[1] == (0.0, 0.0)
        assert ctrl.starts[1] == (5.0, 5.0)
        assert len(events) == 1
        assert events[0][0] == "goal_swap"

    def test_no_swap_when_far(self):
        ctrl = ORCAHumanController(cfg={"goal_tolerance": 0.5})
        ctrl.human_ids = [1]
        ctrl.starts = {1: (0.0, 0.0)}
        ctrl.goals = {1: (5.0, 5.0)}
        state = _make_state(1, x=0.0, y=0.0)

        swapped = ctrl._maybe_swap_goal(1, state, _noop_emit)
        assert not swapped


class TestORCAGoalVelocity:
    def test_at_goal_returns_zero(self):
        ctrl = ORCAHumanController(cfg={"goal_tolerance": 0.5})
        state = _make_state(1, x=5.0, y=5.0)
        vx, vy = ctrl._goal_velocity(state, (5.0, 5.0))
        assert vx == 0.0
        assert vy == 0.0

    def test_far_from_goal_returns_max_speed(self):
        ctrl = ORCAHumanController(cfg={"goal_tolerance": 0.1, "slowdown_dist": 0.5})
        state = _make_state(1, x=0.0, y=0.0, max_speed=1.0)
        vx, vy = ctrl._goal_velocity(state, (10.0, 0.0))
        speed = math.hypot(vx, vy)
        assert speed == pytest.approx(1.0, abs=0.01)

    def test_near_goal_slows_down(self):
        ctrl = ORCAHumanController(cfg={"goal_tolerance": 0.1, "slowdown_dist": 2.0})
        state = _make_state(1, x=0.0, y=0.0, max_speed=1.0)
        # Place target 0.5m away — well within slowdown_dist
        vx_near, vy_near = ctrl._goal_velocity(state, (0.5, 0.0))
        # Place target 10m away — beyond slowdown_dist
        vx_far, vy_far = ctrl._goal_velocity(state, (10.0, 0.0))
        speed_near = math.hypot(vx_near, vy_near)
        speed_far = math.hypot(vx_far, vy_far)
        assert speed_near < speed_far


class TestORCASmoothVelocity:
    def test_smoothing_from_zero(self):
        ctrl = ORCAHumanController(cfg={"velocity_smoothing": 0.5})
        ctrl.last_pref = {1: (0.0, 0.0)}
        state = _make_state(1, x=0.0, y=0.0, max_speed=2.0)
        vx, vy = ctrl._smooth_preferred_velocity(1, state, 1.0, 0.0)
        # With alpha=0.5, from (0,0) to (1,0): expect (0.5, 0)
        assert vx == pytest.approx(0.5)
        assert vy == pytest.approx(0.0)

    def test_smoothing_clamps_to_max_speed(self):
        ctrl = ORCAHumanController(cfg={"velocity_smoothing": 1.0})
        ctrl.last_pref = {1: (0.0, 0.0)}
        state = _make_state(1, x=0.0, y=0.0, max_speed=0.5)
        vx, vy = ctrl._smooth_preferred_velocity(1, state, 10.0, 0.0)
        speed = math.hypot(vx, vy)
        assert speed <= state.max_speed + 1e-6

    def test_stop_threshold(self):
        ctrl = ORCAHumanController(cfg={"velocity_smoothing": 1.0, "stop_speed": 0.05})
        ctrl.last_pref = {1: (0.0, 0.0)}
        state = _make_state(1, x=0.0, y=0.0, max_speed=1.0)
        # Very small velocity should be zeroed out
        vx, vy = ctrl._smooth_preferred_velocity(1, state, 0.01, 0.01)
        assert vx == 0.0
        assert vy == 0.0


class TestORCAStep:
    def _setup_ctrl(self):
        ctrl = ORCAHumanController(cfg={"goal_tolerance": 0.1, "velocity_smoothing": 1.0})
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 0.0)},
            backend=None,
        )
        return ctrl

    def test_step_produces_action(self):
        ctrl = self._setup_ctrl()
        state = _make_state(1, x=0.0, y=0.0, goal_x=5.0, goal_y=0.0)
        actions = ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=0, emit_event=_noop_emit)
        assert 1 in actions
        assert isinstance(actions[1], Action)
        # Should move in the +x direction
        assert actions[1].pref_vx > 0

    def test_step_missing_state_returns_stop(self):
        ctrl = self._setup_ctrl()
        actions = ctrl.step(0, 0.0, 0.1, {}, robot_id=0, emit_event=_noop_emit)
        assert actions[1].pref_vx == 0.0
        assert actions[1].pref_vy == 0.0

    def test_step_invalid_position_returns_stop(self):
        ctrl = self._setup_ctrl()
        state = _make_state(1, x=float("nan"), y=0.0)
        actions = ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=0, emit_event=_noop_emit)
        assert actions[1].pref_vx == 0.0
        assert actions[1].pref_vy == 0.0


class TestORCACurrentWaypoint:
    def test_advances_past_close_waypoints(self):
        ctrl = ORCAHumanController(cfg={"waypoint_tolerance": 0.5, "lookahead": 1})
        ctrl.backend = None
        ctrl.goals = {1: (5.0, 0.0)}
        ctrl.paths = {1: [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (5.0, 0.0)]}
        ctrl.path_idx = {1: 0}
        # Agent is at (1.0, 0.0) — within tolerance of first waypoint
        state = _make_state(1, x=1.0, y=0.0)
        wp = ctrl._current_waypoint(1, state)
        # Should advance past first waypoint
        assert wp == (2.0, 0.0)

    def test_replans_when_path_exhausted(self):
        ctrl = ORCAHumanController(cfg={"waypoint_tolerance": 0.5, "lookahead": 1})
        ctrl.backend = None
        ctrl.goals = {1: (5.0, 0.0)}
        ctrl.paths = {1: [(1.0, 0.0)]}
        ctrl.path_idx = {1: 0}
        # Agent at (1.0, 0.0) — within tolerance, exhausts path
        state = _make_state(1, x=1.0, y=0.0)
        wp = ctrl._current_waypoint(1, state)
        assert wp == (5.0, 0.0)  # Replanned to goal

    def test_lookahead_skips_ahead(self):
        ctrl = ORCAHumanController(cfg={"waypoint_tolerance": 0.5, "lookahead": 3})
        ctrl.backend = None
        ctrl.goals = {1: (10.0, 0.0)}
        ctrl.paths = {1: [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0), (10.0, 0.0)]}
        ctrl.path_idx = {1: 0}
        state = _make_state(1, x=0.0, y=0.0)
        wp = ctrl._current_waypoint(1, state)
        # idx=0, lookahead=3, so look_idx = min(4, 0+2) = 2
        assert wp == (6.0, 0.0)


# ---------------------------------------------------------------------------
# ORCAPlusHumanController tests
# ---------------------------------------------------------------------------


class TestORCAPlusInit:
    def test_default_config(self):
        ctrl = ORCAPlusHumanController()
        assert ctrl.enable_doorway_token is True
        assert ctrl.enable_anisotropic_space is True
        assert ctrl.enable_speed_profile is True
        assert ctrl.enable_group_cohesion is False
        assert ctrl.personal_space == pytest.approx(0.9)
        assert ctrl.hesitation_prob == pytest.approx(0.06)

    def test_custom_config(self):
        ctrl = ORCAPlusHumanController(cfg={
            "doorway_token": False,
            "anisotropic_space": False,
            "speed_profile": False,
            "group_cohesion": True,
            "personal_space": 1.2,
            "groups": [[1, 2], [3, 4]],
        })
        assert ctrl.enable_doorway_token is False
        assert ctrl.enable_group_cohesion is True
        assert ctrl.personal_space == pytest.approx(1.2)
        assert ctrl.groups == [[1, 2], [3, 4]]


class TestORCAPlusReset:
    def test_reset_initializes_speeds_and_groups(self):
        ctrl = ORCAPlusHumanController(cfg={"groups": [[1, 2]]})
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (1.0, 0.0)},
            goals={1: (5.0, 5.0), 2: (6.0, 6.0)},
        )
        assert ctrl.current_speed[1] == 0.0
        assert ctrl.current_speed[2] == 0.0
        assert ctrl.token_holder is None
        assert ctrl.group_by_agent[1] == [1, 2]
        assert ctrl.group_by_agent[2] == [1, 2]


class TestORCAPlusDoorway:
    def test_in_doorway(self):
        ctrl = ORCAPlusHumanController(cfg={
            "doorway": {"center": [2.0, 3.0], "half_extents": [0.5, 0.4]},
        })
        assert ctrl._in_doorway(2.0, 3.0) is True
        assert ctrl._in_doorway(2.4, 3.3) is True
        assert ctrl._in_doorway(10.0, 10.0) is False

    def test_in_doorway_with_margin(self):
        ctrl = ORCAPlusHumanController(cfg={
            "doorway": {"center": [0.0, 0.0], "half_extents": [0.25, 0.2]},
        })
        # Outside default extents but within margin
        assert ctrl._in_doorway(0.5, 0.0, margin=0.3) is True
        assert ctrl._in_doorway(0.5, 0.0, margin=0.0) is False

    def test_doorway_candidates(self):
        ctrl = ORCAPlusHumanController(cfg={
            "doorway": {
                "center": [0.0, 0.0],
                "half_extents": [0.5, 0.5],
                "approach_margin": 1.0,
            },
        })
        ctrl.human_ids = [1, 2, 3]
        states = {
            1: _make_state(1, x=0.0, y=0.0),   # in doorway
            2: _make_state(2, x=1.0, y=0.0),    # in approach margin
            3: _make_state(3, x=10.0, y=10.0),  # far away
        }
        cands = ctrl._doorway_candidates(states)
        assert 1 in cands
        assert 2 in cands
        assert 3 not in cands

    def test_token_acquisition_and_release(self):
        ctrl = ORCAPlusHumanController(cfg={
            "doorway_token": True,
            "doorway": {
                "center": [0.0, 0.0],
                "half_extents": [0.5, 0.5],
                "approach_margin": 0.5,
            },
        })
        ctrl.human_ids = [1, 2]

        events = []
        def emit(etype, aid, payload):
            events.append((etype, aid, payload))

        # Agent 1 near doorway, agent 2 far away
        states = {
            1: _make_state(1, x=0.0, y=0.0),
            2: _make_state(2, x=10.0, y=10.0),
        }
        ctrl._update_token(states, emit)
        assert ctrl.token_holder == 1
        assert any(e[0] == "door_token_acquire" for e in events)

        # Move agent 1 far away — token should be released
        events.clear()
        states[1] = _make_state(1, x=10.0, y=10.0)
        ctrl._update_token(states, emit)
        assert ctrl.token_holder is None
        assert any(e[0] == "door_token_release" for e in events)

    def test_token_disabled(self):
        ctrl = ORCAPlusHumanController(cfg={"doorway_token": False})
        ctrl.human_ids = [1]
        states = {1: _make_state(1, x=0.0, y=0.0)}
        ctrl._update_token(states, _noop_emit)
        assert ctrl.token_holder is None


class TestORCAPlusAnisotropicScale:
    def test_returns_one_when_disabled(self):
        ctrl = ORCAPlusHumanController(cfg={"anisotropic_space": False})
        state = _make_state(1, x=0.0, y=0.0)
        scale = ctrl._apply_anisotropic_scale(1, state, (1.0, 0.0), {1: state})
        assert scale == 1.0

    def test_returns_one_when_no_desired_velocity(self):
        ctrl = ORCAPlusHumanController(cfg={"anisotropic_space": True})
        state = _make_state(1, x=0.0, y=0.0)
        scale = ctrl._apply_anisotropic_scale(1, state, (0.0, 0.0), {1: state})
        assert scale == 1.0

    def test_scale_reduces_with_close_frontal_agent(self):
        ctrl = ORCAPlusHumanController(cfg={
            "anisotropic_space": True,
            "personal_space": 2.0,
        })
        state1 = _make_state(1, x=0.0, y=0.0)
        state2 = _make_state(2, x=0.5, y=0.0)  # directly ahead, close
        states = {1: state1, 2: state2}
        scale = ctrl._apply_anisotropic_scale(1, state1, (1.0, 0.0), states)
        assert scale < 1.0
        assert scale >= 0.25  # Minimum clamp

    def test_scale_unaffected_by_agent_behind(self):
        ctrl = ORCAPlusHumanController(cfg={
            "anisotropic_space": True,
            "personal_space": 2.0,
        })
        state1 = _make_state(1, x=0.0, y=0.0)
        state2 = _make_state(2, x=-1.0, y=0.0)  # behind
        states = {1: state1, 2: state2}
        scale = ctrl._apply_anisotropic_scale(1, state1, (1.0, 0.0), states)
        assert scale == 1.0


class TestORCAPlusGroupCohesion:
    def test_no_bias_when_disabled(self):
        ctrl = ORCAPlusHumanController(cfg={"group_cohesion": False})
        state = _make_state(1, x=0.0, y=0.0)
        bx, by = ctrl._group_velocity_bias(1, {1: state})
        assert bx == 0.0
        assert by == 0.0

    def test_no_bias_when_not_in_group(self):
        ctrl = ORCAPlusHumanController(cfg={"group_cohesion": True})
        ctrl.group_by_agent = {}
        state = _make_state(1, x=0.0, y=0.0)
        bx, by = ctrl._group_velocity_bias(1, {1: state})
        assert bx == 0.0
        assert by == 0.0

    def test_bias_toward_group_center(self):
        ctrl = ORCAPlusHumanController(cfg={
            "group_cohesion": True,
            "group_weight": 0.5,
            "groups": [[1, 2, 3]],
        })
        ctrl.group_by_agent = {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]}
        states = {
            1: _make_state(1, x=0.0, y=0.0),
            2: _make_state(2, x=4.0, y=0.0),
            3: _make_state(3, x=0.0, y=4.0),
        }
        bx, by = ctrl._group_velocity_bias(1, states)
        # Group center (excluding 1) is at (2, 2), so bias should point toward +x, +y
        assert bx > 0
        assert by > 0


class TestORCAPlusStep:
    def _setup_ctrl(self, **extra_cfg):
        cfg = {
            "goal_tolerance": 0.1,
            "velocity_smoothing": 1.0,
            "doorway_token": False,
            "anisotropic_space": False,
            "speed_profile": False,
            "group_cohesion": False,
            **extra_cfg,
        }
        ctrl = ORCAPlusHumanController(cfg=cfg)
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 0.0)},
        )
        return ctrl

    def test_basic_step(self):
        ctrl = self._setup_ctrl()
        state = _make_state(1, x=0.0, y=0.0, max_speed=1.0)
        actions = ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=0, emit_event=_noop_emit)
        assert 1 in actions
        assert actions[1].pref_vx > 0  # Moving toward goal

    def test_doorway_yield(self):
        """Non-token-holder near doorway should yield."""
        ctrl = ORCAPlusHumanController(cfg={
            "doorway_token": True,
            "anisotropic_space": False,
            "speed_profile": False,
            "doorway": {
                "center": [0.0, 0.0],
                "half_extents": [1.0, 1.0],
                "approach_margin": 2.0,
            },
        })
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (0.5, 0.0)},
            goals={1: (5.0, 5.0), 2: (6.0, 6.0)},
        )

        states = {
            1: _make_state(1, x=0.0, y=0.0, max_speed=1.0),
            2: _make_state(2, x=0.5, y=0.0, max_speed=1.0),
        }

        events = []
        def emit(etype, aid, payload):
            events.append((etype, aid))

        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=99, emit_event=emit)

        # One agent should be yielding (the non-token-holder)
        yielding = [hid for hid, act in actions.items() if act.behavior == "YIELDING"]
        # Token holder is the first candidate (sorted), so agent 1
        # Agent 2 should yield
        assert 2 in yielding

    def test_speed_profile_acceleration_limit(self):
        ctrl = self._setup_ctrl(speed_profile=True, hesitation_prob=0.0)
        ctrl.current_speed[1] = 0.0
        state = _make_state(1, x=0.0, y=0.0, max_speed=2.0)
        actions = ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=0, emit_event=_noop_emit)
        # With accel_limit=1.6 and dt=0.1, max_delta=0.16
        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        assert speed <= 0.16 + 1e-6

    def test_speed_profile_hesitation(self):
        """With hesitation_prob=1.0, speed should be dramatically reduced."""
        ctrl = self._setup_ctrl(speed_profile=True, hesitation_prob=0.0)
        # Set seed so we can control hesitation
        ctrl.rng.random = lambda: 0.0  # Always triggers hesitation
        ctrl.hesitation_prob = 1.0
        ctrl.current_speed[1] = 1.0

        events = []
        def emit(etype, aid, payload):
            events.append(etype)

        state = _make_state(1, x=0.0, y=0.0, max_speed=2.0)
        actions = ctrl.step(0, 0.0, 0.1, {1: state}, robot_id=0, emit_event=emit)
        assert "hesitation" in events
