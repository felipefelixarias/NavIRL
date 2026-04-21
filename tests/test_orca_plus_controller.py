"""Tests for navirl.humans.orca_plus.controller.ORCAPlusHumanController.

Covers doorway token protocol, anisotropic personal space, group cohesion,
speed profiling with acceleration limits, and hesitation injection.
"""

from __future__ import annotations

import math

import pytest

from navirl.core.types import Action, AgentState
from navirl.humans.orca_plus.controller import ORCAPlusHumanController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    agent_id: int,
    x: float,
    y: float,
    *,
    kind: str = "human",
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 5.0,
    goal_y: float = 0.0,
    radius: float = 0.18,
    max_speed: float = 1.2,
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
        radius=radius,
        max_speed=max_speed,
    )


def _noop_emit(event_type, agent_id, payload):
    """No-op event sink."""


class _CollectingEmit:
    """Event sink that records all emitted events."""

    def __init__(self):
        self.events: list[tuple[str, int | None, dict]] = []

    def __call__(self, event_type, agent_id, payload):
        self.events.append((event_type, agent_id, payload))


def _make_controller(cfg=None, seed=0):
    """Construct controller, bypassing backend path planning."""
    ctrl = ORCAPlusHumanController(cfg=cfg, seed=seed)
    return ctrl


def _reset_simple(ctrl, human_ids, starts, goals):
    """Reset with no backend (direct-to-goal paths)."""
    ctrl.reset(human_ids, starts, goals, backend=None)


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_config(self):
        ctrl = _make_controller()
        assert ctrl.enable_doorway_token is True
        assert ctrl.enable_anisotropic_space is True
        assert ctrl.enable_speed_profile is True
        assert ctrl.enable_group_cohesion is False

    def test_custom_config(self):
        cfg = {
            "doorway_token": False,
            "anisotropic_space": False,
            "speed_profile": False,
            "group_cohesion": True,
            "personal_space": 1.2,
            "accel_limit": 2.0,
            "hesitation_prob": 0.0,
            "groups": [[1, 2]],
        }
        ctrl = _make_controller(cfg=cfg)
        assert ctrl.enable_doorway_token is False
        assert ctrl.enable_group_cohesion is True
        assert ctrl.personal_space == 1.2
        assert ctrl.accel_limit == 2.0

    def test_doorway_config(self):
        cfg = {
            "doorway": {
                "center": [1.0, 2.0],
                "half_extents": [0.5, 0.3],
                "approach_margin": 0.8,
            }
        }
        ctrl = _make_controller(cfg=cfg)
        assert ctrl.door_center == (1.0, 2.0)
        assert ctrl.door_half_extents == (0.5, 0.3)
        assert ctrl.door_approach_margin == 0.8


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_initializes_state(self):
        ctrl = _make_controller()
        _reset_simple(ctrl, [1, 2], {1: (0.0, 0.0), 2: (1.0, 0.0)}, {1: (5.0, 0.0), 2: (5.0, 1.0)})
        assert ctrl.current_speed == {1: 0.0, 2: 0.0}
        assert ctrl.token_holder is None

    def test_reset_builds_group_index(self):
        cfg = {"groups": [[1, 2], [3]]}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(
            ctrl,
            [1, 2, 3],
            {1: (0, 0), 2: (1, 0), 3: (2, 0)},
            {1: (5, 0), 2: (5, 1), 3: (5, 2)},
        )
        assert ctrl.group_by_agent[1] == [1, 2]
        assert ctrl.group_by_agent[2] == [1, 2]
        assert ctrl.group_by_agent[3] == [3]


# ---------------------------------------------------------------------------
# Doorway helpers
# ---------------------------------------------------------------------------


class TestDoorwayHelpers:
    def test_in_doorway_center(self):
        ctrl = _make_controller()
        assert ctrl._in_doorway(0.0, 0.0) is True

    def test_in_doorway_margin(self):
        ctrl = _make_controller()
        # Default half_extents=(0.25, 0.2), approach_margin=0.6
        # With margin=0.6: threshold_x=0.25+0.6=0.85
        assert ctrl._in_doorway(0.8, 0.0, margin=0.6) is True
        assert ctrl._in_doorway(1.0, 0.0, margin=0.6) is False

    def test_in_doorway_outside(self):
        ctrl = _make_controller()
        assert ctrl._in_doorway(10.0, 10.0) is False

    def test_doorway_candidates(self):
        ctrl = _make_controller()
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (10, 10)}, {1: (5, 0), 2: (5, 5)})
        states = {
            1: _make_state(1, 0.0, 0.0),  # near door
            2: _make_state(2, 10.0, 10.0),  # far from door
        }
        cands = ctrl._doorway_candidates(states)
        assert cands == [1]


# ---------------------------------------------------------------------------
# Doorway token protocol
# ---------------------------------------------------------------------------


class TestDoorwayToken:
    def test_token_acquired_on_approach(self):
        ctrl = _make_controller(cfg={"hesitation_prob": 0.0})
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.1, 0)}, {1: (5, 0), 2: (5, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 0.1, 0.0),
        }
        emit = _CollectingEmit()
        ctrl._update_token(states, emit)
        assert ctrl.token_holder == 1
        assert any(e[0] == "door_token_acquire" for e in emit.events)

    def test_token_released_when_leaving(self):
        ctrl = _make_controller(cfg={"hesitation_prob": 0.0})
        _reset_simple(ctrl, [1], {1: (0, 0)}, {1: (5, 0)})
        states_near = {1: _make_state(1, 0.0, 0.0)}
        emit = _CollectingEmit()
        ctrl._update_token(states_near, emit)
        assert ctrl.token_holder == 1

        # Agent moves far away
        states_far = {1: _make_state(1, 10.0, 10.0)}
        ctrl._update_token(states_far, emit)
        assert ctrl.token_holder is None
        assert any(e[0] == "door_token_release" for e in emit.events)

    def test_yielding_non_holder_near_door(self):
        cfg = {"hesitation_prob": 0.0, "speed_profile": False}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.1, 0)}, {1: (5, 0), 2: (5, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 0.1, 0.0),
            99: _make_state(99, -5.0, -5.0, kind="robot"),
        }
        emit = _CollectingEmit()
        actions = ctrl.step(0, 0.0, 0.1, states, 99, emit)
        # Agent 1 gets token (lowest id), agent 2 should yield
        assert actions[2].behavior == "YIELDING"
        assert actions[2].pref_vx == 0.0
        assert actions[2].pref_vy == 0.0

    def test_doorway_disabled(self):
        cfg = {"doorway_token": False, "hesitation_prob": 0.0, "speed_profile": False}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.1, 0)}, {1: (5, 0), 2: (5, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 0.1, 0.0),
            99: _make_state(99, -5.0, -5.0, kind="robot"),
        }
        emit = _CollectingEmit()
        actions = ctrl.step(0, 0.0, 0.1, states, 99, emit)
        # Neither should yield since doorway token is disabled
        assert actions[2].behavior != "YIELDING"


# ---------------------------------------------------------------------------
# Anisotropic personal space
# ---------------------------------------------------------------------------


class TestAnisotropicSpace:
    def test_scale_reduces_when_agent_ahead(self):
        ctrl = _make_controller(cfg={"hesitation_prob": 0.0})
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.5, 0)}, {1: (5, 0), 2: (5, 0)})
        state1 = _make_state(1, 0.0, 0.0)
        states = {
            1: state1,
            2: _make_state(2, 0.5, 0.0),  # directly ahead if moving +x
        }
        scale = ctrl._apply_anisotropic_scale(1, state1, (1.0, 0.0), states)
        # Agent 2 is 0.5m ahead, personal_space=0.9 → scale = 0.5/0.9 ≈ 0.556
        assert 0.25 <= scale < 1.0

    def test_scale_one_when_no_agent_ahead(self):
        ctrl = _make_controller(cfg={"hesitation_prob": 0.0})
        _reset_simple(ctrl, [1], {1: (0, 0)}, {1: (5, 0)})
        state1 = _make_state(1, 0.0, 0.0)
        states = {1: state1}
        scale = ctrl._apply_anisotropic_scale(1, state1, (1.0, 0.0), states)
        assert scale == 1.0

    def test_scale_one_when_zero_velocity(self):
        ctrl = _make_controller()
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.5, 0)}, {1: (5, 0), 2: (5, 0)})
        state1 = _make_state(1, 0.0, 0.0)
        states = {1: state1, 2: _make_state(2, 0.5, 0.0)}
        scale = ctrl._apply_anisotropic_scale(1, state1, (0.0, 0.0), states)
        assert scale == 1.0

    def test_disabled_anisotropic(self):
        ctrl = _make_controller(cfg={"anisotropic_space": False})
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.5, 0)}, {1: (5, 0), 2: (5, 0)})
        state1 = _make_state(1, 0.0, 0.0)
        states = {1: state1, 2: _make_state(2, 0.5, 0.0)}
        scale = ctrl._apply_anisotropic_scale(1, state1, (1.0, 0.0), states)
        assert scale == 1.0


# ---------------------------------------------------------------------------
# Group cohesion
# ---------------------------------------------------------------------------


class TestGroupCohesion:
    def test_no_bias_when_disabled(self):
        ctrl = _make_controller(cfg={"group_cohesion": False, "groups": [[1, 2]]})
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (2, 0)}, {1: (5, 0), 2: (5, 0)})
        states = {1: _make_state(1, 0.0, 0.0), 2: _make_state(2, 2.0, 0.0)}
        bx, by = ctrl._group_velocity_bias(1, states)
        assert bx == 0.0 and by == 0.0

    def test_bias_toward_group_center(self):
        cfg = {"group_cohesion": True, "groups": [[1, 2]], "group_weight": 0.5}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (2, 0)}, {1: (5, 0), 2: (5, 0)})
        states = {1: _make_state(1, 0.0, 0.0), 2: _make_state(2, 2.0, 0.0)}
        bx, by = ctrl._group_velocity_bias(1, states)
        # Group center for agent 1 is at (2.0, 0.0), so bias should be +x
        assert bx > 0.0
        assert abs(by) < 1e-6

    def test_no_bias_solo_agent(self):
        cfg = {"group_cohesion": True, "groups": [[1]]}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1], {1: (0, 0)}, {1: (5, 0)})
        states = {1: _make_state(1, 0.0, 0.0)}
        bx, by = ctrl._group_velocity_bias(1, states)
        assert bx == 0.0 and by == 0.0

    def test_no_bias_ungrouped_agent(self):
        cfg = {"group_cohesion": True, "groups": [[2, 3]]}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1, 2, 3], {1: (0, 0), 2: (1, 0), 3: (2, 0)}, {1: (5, 0), 2: (5, 0), 3: (5, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 1.0, 0.0),
            3: _make_state(3, 2.0, 0.0),
        }
        bx, by = ctrl._group_velocity_bias(1, states)
        assert bx == 0.0 and by == 0.0


# ---------------------------------------------------------------------------
# Speed profiling and hesitation
# ---------------------------------------------------------------------------


class TestSpeedProfile:
    def test_acceleration_limited(self):
        cfg = {"hesitation_prob": 0.0, "accel_limit": 1.0, "doorway_token": False}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1], {1: (0, 0)}, {1: (10, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0, max_speed=2.0),
            99: _make_state(99, -5, -5, kind="robot"),
        }

        # First step: from speed 0, limited by accel_limit * dt
        dt = 0.1
        actions = ctrl.step(0, 0.0, dt, states, 99, _noop_emit)
        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        # Max acceleration = 1.0 * 0.1 = 0.1 m/s from standstill
        assert speed <= 1.0 * dt + 0.01  # small tolerance

    def test_hesitation_reduces_speed(self):
        # Use seed that triggers hesitation (prob=1.0 guarantees it)
        cfg = {"hesitation_prob": 1.0, "hesitation_scale": 0.1, "doorway_token": False, "accel_limit": 100.0}
        ctrl = _make_controller(cfg=cfg, seed=42)
        _reset_simple(ctrl, [1], {1: (0, 0)}, {1: (10, 0)})

        # Give the agent some current speed
        ctrl.current_speed[1] = 1.0
        states = {
            1: _make_state(1, 0.0, 0.0, max_speed=2.0),
            99: _make_state(99, -5, -5, kind="robot"),
        }
        emit = _CollectingEmit()
        actions = ctrl.step(0, 0.0, 0.1, states, 99, emit)
        # With hesitation_prob=1.0, speed should be significantly reduced
        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        assert speed < 0.5  # Should be much less than max_speed
        assert any(e[0] == "hesitation" for e in emit.events)

    def test_speed_profile_disabled(self):
        cfg = {"speed_profile": False, "doorway_token": False, "hesitation_prob": 0.0}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1], {1: (0, 0)}, {1: (10, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0, max_speed=2.0),
            99: _make_state(99, -5, -5, kind="robot"),
        }
        actions = ctrl.step(0, 0.0, 0.1, states, 99, _noop_emit)
        # Without speed profile, no acceleration limiting — velocity from ORCA base
        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        assert speed > 0.0  # Should have some velocity toward goal


# ---------------------------------------------------------------------------
# Full step integration
# ---------------------------------------------------------------------------


class TestStepIntegration:
    def test_step_returns_actions_for_all_humans(self):
        cfg = {"hesitation_prob": 0.0, "doorway_token": False}
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (1, 0)}, {1: (5, 0), 2: (5, 1)})
        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 1.0, 0.0),
            99: _make_state(99, -5, -5, kind="robot"),
        }
        actions = ctrl.step(0, 0.0, 0.1, states, 99, _noop_emit)
        assert 1 in actions
        assert 2 in actions
        assert isinstance(actions[1], Action)
        assert isinstance(actions[2], Action)

    def test_step_with_all_features_enabled(self):
        cfg = {
            "doorway_token": True,
            "anisotropic_space": True,
            "speed_profile": True,
            "group_cohesion": True,
            "groups": [[1, 2]],
            "hesitation_prob": 0.0,
        }
        ctrl = _make_controller(cfg=cfg)
        _reset_simple(ctrl, [1, 2], {1: (0, 0), 2: (0.1, 0)}, {1: (5, 0), 2: (5, 0)})
        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 0.1, 0.0),
            99: _make_state(99, -5, -5, kind="robot"),
        }
        # Should not raise
        actions = ctrl.step(0, 0.0, 0.1, states, 99, _noop_emit)
        assert len(actions) == 2
