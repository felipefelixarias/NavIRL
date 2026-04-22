"""Coverage tests for navirl.humans.orca.controller.ORCAHumanController.

Existing tests exercise the subclass ORCAPlusHumanController. This file
targets the base-class-only paths: plan failure fallbacks, goal swapping,
waypoint advancement, replanning after waypoint exhaustion, and the
defensive error branches in :meth:`step`.
"""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from navirl.core.types import AgentState
from navirl.humans.orca.controller import ORCAHumanController


class _PathBackend:
    def __init__(self, path=None, raise_on_plan: bool = False):
        self.path = path
        self.raise_on_plan = raise_on_plan
        self.plans: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def shortest_path(self, start, goal):
        self.plans.append((tuple(start), tuple(goal)))
        if self.raise_on_plan:
            raise RuntimeError("planner blew up")
        if self.path is None:
            return [(goal[0], goal[1])]
        return list(self.path)


def _human(hid: int, x: float, y: float, max_speed: float = 1.0) -> AgentState:
    return AgentState(
        agent_id=hid,
        kind="human",
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        goal_x=0.0,
        goal_y=0.0,
        max_speed=max_speed,
        radius=0.2,
    )


@pytest.fixture
def emit():
    return Mock()


# ---------------------------------------------------------------------------
# reset() — backend path failure
# ---------------------------------------------------------------------------


class TestResetPlanFailure:
    def test_reset_falls_back_to_direct_path_on_exception(self):
        ctrl = ORCAHumanController()
        backend = _PathBackend(raise_on_plan=True)
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=backend,
        )
        # Despite the exception, controller records direct-to-goal fallback.
        assert ctrl.paths[1] == [(5.0, 5.0)]
        assert ctrl.path_idx[1] == 0
        assert ctrl.last_pref[1] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# _plan_path — direct branches
# ---------------------------------------------------------------------------


class TestPlanPath:
    def test_plan_path_no_backend_returns_goal_only(self):
        ctrl = ORCAHumanController()
        assert ctrl._plan_path((0.0, 0.0), (2.0, 3.0)) == [(2.0, 3.0)]

    def test_plan_path_empty_result_returns_goal_only(self):
        ctrl = ORCAHumanController()
        ctrl.backend = _PathBackend(path=[])
        assert ctrl._plan_path((0.0, 0.0), (1.0, 1.0)) == [(1.0, 1.0)]


# ---------------------------------------------------------------------------
# _maybe_swap_goal — ping-pong behavior
# ---------------------------------------------------------------------------


class TestGoalSwap:
    def test_goal_swap_when_close(self, emit):
        ctrl = ORCAHumanController({"goal_tolerance": 0.25})
        backend = _PathBackend(path=[(0.0, 0.0), (5.0, 5.0)])
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=backend,
        )
        # Human at (5.0, 5.0) — inside tolerance of the goal → swap start/goal.
        state = _human(1, 5.0, 5.0)
        swapped = ctrl._maybe_swap_goal(1, state, emit)
        assert swapped is True
        assert ctrl.goals[1] == (0.0, 0.0)
        assert ctrl.starts[1] == (5.0, 5.0)
        emit.assert_called_once()
        event_name, agent_id, payload = emit.call_args.args
        assert event_name == "goal_swap"
        assert agent_id == 1
        assert payload["new_goal"] == [0.0, 0.0]

    def test_goal_swap_no_op_when_far(self, emit):
        ctrl = ORCAHumanController({"goal_tolerance": 0.25})
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=_PathBackend(path=[(0.0, 0.0), (5.0, 5.0)]),
        )
        state = _human(1, 0.0, 0.0)
        assert ctrl._maybe_swap_goal(1, state, emit) is False
        emit.assert_not_called()


# ---------------------------------------------------------------------------
# _current_waypoint — empty path replan + waypoint advance + exhaustion replan
# ---------------------------------------------------------------------------


class TestCurrentWaypoint:
    def test_empty_path_triggers_replan(self):
        ctrl = ORCAHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=_PathBackend(path=[(5.0, 5.0)]),
        )
        ctrl.paths[1] = []  # force the empty-path branch
        # Supply a backend whose shortest_path returns a real path next time.
        ctrl.backend = _PathBackend(path=[(1.0, 1.0), (5.0, 5.0)])
        wp = ctrl._current_waypoint(1, _human(1, 0.0, 0.0))
        assert ctrl.paths[1] == [(1.0, 1.0), (5.0, 5.0)]
        assert wp in ctrl.paths[1]

    def test_waypoint_advances_past_reached_points(self):
        ctrl = ORCAHumanController({"waypoint_tolerance": 0.5, "lookahead": 1})
        backend = _PathBackend(path=[(0.0, 0.0), (1.0, 0.0), (5.0, 0.0)])
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 0.0)},
            backend=backend,
        )
        # Human at (0.0, 0.0) is within tolerance of wp index 0 only.
        # The advance loop should consume wp 0 and stop at wp 1.
        wp = ctrl._current_waypoint(1, _human(1, 0.0, 0.0))
        assert ctrl.path_idx[1] == 1
        assert wp == (1.0, 0.0)

    def test_exhausted_path_replans_from_current_position(self):
        """When idx advances past the end, _current_waypoint must replan."""
        ctrl = ORCAHumanController({"waypoint_tolerance": 100.0, "lookahead": 1})
        # First planning returns short path; waypoint_tolerance is so large that
        # the advance loop consumes every waypoint, forcing the exhaustion branch.
        backend = _PathBackend(path=[(0.0, 0.0)])
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=backend,
        )
        # Swap to a backend that returns a richer path on replan.
        backend.path = [(2.0, 2.0), (5.0, 5.0)]
        wp = ctrl._current_waypoint(1, _human(1, 1.0, 1.0))
        assert ctrl.path_idx[1] == 0
        assert wp in backend.path


# ---------------------------------------------------------------------------
# _goal_velocity — inside tolerance and clamping
# ---------------------------------------------------------------------------


class TestGoalVelocity:
    def test_zero_velocity_inside_goal_tolerance(self):
        ctrl = ORCAHumanController({"goal_tolerance": 0.5})
        state = _human(1, 0.0, 0.0)
        assert ctrl._goal_velocity(state, (0.1, 0.1)) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# _smooth_preferred_velocity — max speed clamp and stop threshold
# ---------------------------------------------------------------------------


class TestSmoothing:
    def test_clamps_to_max_speed(self):
        ctrl = ORCAHumanController({"velocity_smoothing": 1.0})
        state = _human(1, 0.0, 0.0, max_speed=1.0)
        ctrl.last_pref[1] = (0.0, 0.0)
        svx, svy = ctrl._smooth_preferred_velocity(1, state, 5.0, 0.0)
        assert math.hypot(svx, svy) == pytest.approx(1.0)

    def test_snaps_to_zero_below_stop_speed(self):
        ctrl = ORCAHumanController({"velocity_smoothing": 1.0, "stop_speed": 0.1})
        state = _human(1, 0.0, 0.0, max_speed=1.0)
        ctrl.last_pref[1] = (0.0, 0.0)
        svx, svy = ctrl._smooth_preferred_velocity(1, state, 0.01, 0.01)
        assert svx == 0.0 and svy == 0.0


# ---------------------------------------------------------------------------
# step() — defensive branches
# ---------------------------------------------------------------------------


class TestStepErrorBranches:
    def _ctrl(self, **cfg) -> ORCAHumanController:
        ctrl = ORCAHumanController(cfg or None)
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=_PathBackend(path=[(0.0, 0.0), (5.0, 5.0)]),
        )
        return ctrl

    def test_missing_state_produces_stop(self, emit):
        ctrl = self._ctrl()
        actions = ctrl.step(0, 0.0, 0.1, states={}, robot_id=0, emit_event=emit)
        assert actions[1].behavior == "STOP"
        assert actions[1].pref_vx == 0.0 and actions[1].pref_vy == 0.0

    def test_non_finite_position_produces_stop(self, emit):
        ctrl = self._ctrl()
        states = {1: _human(1, float("nan"), 0.0)}
        actions = ctrl.step(0, 0.0, 0.1, states=states, robot_id=0, emit_event=emit)
        assert actions[1].behavior == "STOP"

    def test_replan_failure_after_goal_swap_does_not_crash(self, emit):
        """If the post-swap replan raises, step() logs and continues."""

        class _OneShotBackend:
            """Planner that succeeds once (reset) then raises on any subsequent call."""

            def __init__(self):
                self.calls = 0

            def shortest_path(self, start, goal):
                self.calls += 1
                if self.calls == 1:
                    return [(0.0, 0.0), (5.0, 5.0)]
                raise RuntimeError("replan failed")

        ctrl = ORCAHumanController({"goal_tolerance": 0.25})
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=_OneShotBackend(),
        )
        # Place human at goal so swap triggers then replan fails.
        state = _human(1, 5.0, 5.0)
        actions = ctrl.step(0, 0.0, 0.1, states={1: state}, robot_id=0, emit_event=emit)
        # step() must still produce an action for the human.
        assert 1 in actions

    def test_outer_exception_handler_returns_stop(self, emit):
        """An exception in the inner pipeline is caught and mapped to STOP."""

        class _ExplodingBackend:
            def shortest_path(self, start, goal):
                return [(0.0, 0.0), (5.0, 5.0)]

        ctrl = ORCAHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (5.0, 5.0)},
            backend=_ExplodingBackend(),
        )

        def _raise(self, *args, **kwargs):  # type: ignore[no-redef]
            raise RuntimeError("boom")

        # Monkey-patch an internal helper so the try/except at step() level fires.
        ctrl._current_waypoint = _raise.__get__(ctrl, ORCAHumanController)
        actions = ctrl.step(
            0, 0.0, 0.1, states={1: _human(1, 0.0, 0.0)}, robot_id=0, emit_event=emit
        )
        assert actions[1].behavior == "STOP"
