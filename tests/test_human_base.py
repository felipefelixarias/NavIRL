"""Tests for navirl.humans.base — HumanController ABC validation logic."""

from __future__ import annotations

import pytest

from navirl.core.types import Action
from navirl.humans.base import HumanController


# ---------------------------------------------------------------------------
# Concrete subclass for testing the ABC
# ---------------------------------------------------------------------------


class _StubController(HumanController):
    """Minimal concrete implementation for testing base class validation."""

    def reset(self, human_ids, starts, goals, backend=None):
        super().reset(human_ids, starts, goals, backend)

    def step(self, step, time_s, dt, states, robot_id, emit_event):
        super().step(step, time_s, dt, states, robot_id, emit_event)
        return {}


def _noop_emit(event_type, agent_id, payload):
    pass


# ---------------------------------------------------------------------------
# Reset validation
# ---------------------------------------------------------------------------


class TestHumanControllerReset:
    def test_valid_reset(self):
        ctrl = _StubController(cfg={})
        ctrl.reset([1, 2], {1: (0.0, 0.0), 2: (1.0, 1.0)}, {1: (5.0, 5.0), 2: (6.0, 6.0)})

    def test_non_list_human_ids_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="human_ids must be a list"):
            ctrl.reset((1, 2), {1: (0.0, 0.0), 2: (1.0, 1.0)}, {1: (5.0, 5.0), 2: (6.0, 6.0)})

    def test_non_int_ids_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="All human_ids must be integers"):
            ctrl.reset(["a"], {"a": (0.0, 0.0)}, {"a": (5.0, 5.0)})

    def test_missing_start_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Missing start position"):
            ctrl.reset([1, 2], {1: (0.0, 0.0)}, {1: (5.0, 5.0), 2: (6.0, 6.0)})

    def test_missing_goal_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Missing goal position"):
            ctrl.reset([1], {1: (0.0, 0.0)}, {})

    def test_invalid_start_too_short(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Invalid start position"):
            ctrl.reset([1], {1: (0.0,)}, {1: (5.0, 5.0)})

    def test_invalid_goal_not_tuple(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Invalid goal position"):
            ctrl.reset([1], {1: (0.0, 0.0)}, {1: 5.0})

    def test_empty_human_ids_ok(self):
        ctrl = _StubController(cfg={})
        ctrl.reset([], {}, {})

    def test_list_start_accepted(self):
        ctrl = _StubController(cfg={})
        ctrl.reset([1], {1: [0.0, 0.0]}, {1: [5.0, 5.0]})


# ---------------------------------------------------------------------------
# Step validation
# ---------------------------------------------------------------------------


class TestHumanControllerStep:
    def test_negative_step_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Step must be non-negative"):
            ctrl.step(-1, 0.0, 0.1, {}, 0, _noop_emit)

    def test_negative_time_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Time must be non-negative"):
            ctrl.step(0, -1.0, 0.1, {}, 0, _noop_emit)

    def test_zero_dt_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Time step must be positive"):
            ctrl.step(0, 0.0, 0.0, {}, 0, _noop_emit)

    def test_negative_dt_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="Time step must be positive"):
            ctrl.step(0, 0.0, -0.1, {}, 0, _noop_emit)

    def test_non_dict_states_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="States must be a dictionary"):
            ctrl.step(0, 0.0, 0.1, [], 0, _noop_emit)

    def test_non_callable_emit_raises(self):
        ctrl = _StubController(cfg={})
        with pytest.raises(ValueError, match="emit_event must be callable"):
            ctrl.step(0, 0.0, 0.1, {}, 0, "not_callable")

    def test_valid_step_returns(self):
        ctrl = _StubController(cfg={})
        result = ctrl.step(0, 0.0, 0.1, {}, 0, _noop_emit)
        assert result == {}

    def test_step_zero_ok(self):
        ctrl = _StubController(cfg={})
        ctrl.step(0, 0.0, 0.01, {}, 0, _noop_emit)

    def test_step_large_values_ok(self):
        ctrl = _StubController(cfg={})
        ctrl.step(100000, 9999.0, 0.01, {}, 0, _noop_emit)


# ---------------------------------------------------------------------------
# validate_action
# ---------------------------------------------------------------------------


class TestValidateAction:
    def test_valid_action_unchanged(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=1.0, pref_vy=0.5, behavior="GO_TO")
        result = ctrl.validate_action(1, action)
        assert result.pref_vx == 1.0
        assert result.pref_vy == 0.5
        assert result.behavior == "GO_TO"

    def test_non_action_returns_stop(self):
        ctrl = _StubController(cfg={})
        result = ctrl.validate_action(1, "not_an_action")
        assert result.pref_vx == 0.0
        assert result.pref_vy == 0.0
        assert result.behavior == "STOP"

    def test_none_returns_stop(self):
        ctrl = _StubController(cfg={})
        result = ctrl.validate_action(1, None)
        assert result.behavior == "STOP"

    def test_dict_returns_stop(self):
        ctrl = _StubController(cfg={})
        result = ctrl.validate_action(1, {"vx": 1.0})
        assert result.behavior == "STOP"

    def test_clamps_excessive_positive_vx(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=10.0, pref_vy=0.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vx == 5.0

    def test_clamps_excessive_negative_vx(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=-10.0, pref_vy=0.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vx == -5.0

    def test_clamps_excessive_positive_vy(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=0.0, pref_vy=7.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vy == 5.0

    def test_clamps_excessive_negative_vy(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=0.0, pref_vy=-8.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vy == -5.0

    def test_within_bounds_not_clamped(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=3.0, pref_vy=-2.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vx == 3.0
        assert result.pref_vy == -2.0

    def test_exactly_at_max_not_clamped(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=5.0, pref_vy=-5.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vx == 5.0
        assert result.pref_vy == -5.0

    def test_zero_velocity_ok(self):
        ctrl = _StubController(cfg={})
        action = Action(pref_vx=0.0, pref_vy=0.0)
        result = ctrl.validate_action(1, action)
        assert result.pref_vx == 0.0
        assert result.pref_vy == 0.0
