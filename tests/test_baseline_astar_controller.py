"""Coverage tests for navirl.robots.baselines.astar.BaselineAStarRobotController.

The existing test_robot_baseline_planners.py only exercises the happy path via
a smoke test. This file targets the stuck-recovery state machine, replanning,
grace period, waypoint advance, goal-tolerance paths, and DONE behavior.
"""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest

from navirl.core.types import Action, AgentState
from navirl.robots.baselines import BaselineAStarRobotController


class _StraightLineBackend:
    """Minimal backend that returns the direct start->goal path."""

    def __init__(self, return_empty: bool = False):
        self.return_empty = return_empty
        self.calls: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def shortest_path(self, start, goal):
        self.calls.append((tuple(start), tuple(goal)))
        if self.return_empty:
            return []
        # Create a short multi-waypoint path so lookahead and advance logic run.
        sx, sy = start
        gx, gy = goal
        return [
            (sx, sy),
            ((sx + gx) / 3, (sy + gy) / 3),
            (2 * (sx + gx) / 3, 2 * (sy + gy) / 3),
            (gx, gy),
        ]


def _robot_state(x: float, y: float, goal=(3.0, 3.0), max_speed: float = 1.0) -> AgentState:
    return AgentState(
        agent_id=0,
        kind="robot",
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        goal_x=goal[0],
        goal_y=goal[1],
        max_speed=max_speed,
        radius=0.2,
    )


@pytest.fixture
def emit():
    return Mock()


class TestPlanFallback:
    def test_empty_path_falls_back_to_goal(self):
        """If backend returns [], the planner stores [goal] as the path."""
        ctrl = BaselineAStarRobotController()
        backend = _StraightLineBackend(return_empty=True)
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        assert ctrl.path == [(3.0, 3.0)]


class TestDoneBehavior:
    def test_step_returns_done_at_goal(self, emit):
        ctrl = BaselineAStarRobotController({"goal_tolerance": 0.25})
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), _StraightLineBackend())
        state = _robot_state(3.0, 3.0)
        action = ctrl.step(0, 0.0, 0.1, {0: state}, emit)
        assert action.behavior == "DONE"
        assert action.pref_vx == 0.0 and action.pref_vy == 0.0


class TestWaypointAdvance:
    def test_close_to_waypoint_advances_path_index(self, emit):
        """Standing on a waypoint makes the controller advance past it."""
        ctrl = BaselineAStarRobotController(
            {"goal_tolerance": 0.1, "target_lookahead": 1, "velocity_smoothing": 1.0}
        )
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), _StraightLineBackend())
        # Place robot at the first intermediate waypoint so the close-to-target
        # branch triggers and the index advances.
        wp0 = ctrl.path[0]
        state = _robot_state(wp0[0], wp0[1])
        action = ctrl.step(1, 0.1, 0.1, {0: state}, emit)
        assert ctrl.path_idx >= 1
        assert action.behavior == "GO_TO"


class TestStuckRecovery:
    def _make_controller(self, **kwargs) -> BaselineAStarRobotController:
        cfg = {
            "stuck_window": 3,  # very small so tests stay fast
            "stuck_dist_threshold": 0.05,
            "wait_duration": 50,  # keep the controller in wait long enough to assert
            "max_wait_cycles": 4,
            "replan_interval": 1000,  # don't let periodic replans confound assertions
            "goal_tolerance": 0.1,
            "target_lookahead": 1,
        }
        cfg.update(kwargs)
        return BaselineAStarRobotController(cfg)

    def test_detect_stuck_false_until_window_full(self):
        ctrl = self._make_controller()
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        # After reset _pos_window has 1 entry; window size 3, so detection stays
        # False until the window fills (lines 91-92).
        assert ctrl._detect_stuck(_robot_state(0.0, 0.0)) is False

    def test_detect_stuck_true_on_full_window_no_progress(self):
        ctrl = self._make_controller()
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        # Fill window with near-identical positions → stuck=True once full.
        ctrl._detect_stuck(_robot_state(0.0, 0.0))
        assert ctrl._detect_stuck(_robot_state(0.005, 0.0)) is True

    def test_detect_stuck_progress_resets_wait_cycles(self):
        ctrl = self._make_controller()
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        ctrl._wait_cycles = 2
        # Fill window then supply a position far from the start.
        ctrl._detect_stuck(_robot_state(0.0, 0.0))
        ctrl._detect_stuck(_robot_state(0.0, 0.0))
        # Large displacement — dist >= threshold → progress branch resets cycles.
        assert ctrl._detect_stuck(_robot_state(1.0, 1.0)) is False
        assert ctrl._wait_cycles == 0

    def test_stuck_triggers_yield_and_wait(self, emit):
        ctrl = self._make_controller(wait_duration=50)
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)

        # Three step() calls with near-stationary position trip the stuck check.
        stuck_pos = _robot_state(0.005, 0.0)
        action = None
        for i in range(3):
            action = ctrl.step(i, 0.1 * i, 0.1, {0: stuck_pos}, emit)

        assert ctrl._is_waiting is True
        assert ctrl._wait_cycles == 1
        assert action is not None and action.behavior == "YIELD"
        yield_events = [c for c in emit.call_args_list if c.args[0] == "robot_yield"]
        assert yield_events, "expected a robot_yield event to be emitted"

    def test_wait_emits_crawl_velocity_toward_goal(self, emit):
        """While waiting, the controller emits a tiny velocity in goal direction."""
        ctrl = self._make_controller(wait_duration=50)
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)

        # Trigger stuck.
        stuck_pos = _robot_state(0.0, 0.0)
        for i in range(3):
            ctrl.step(i, 0.1 * i, 0.1, {0: stuck_pos}, emit)
        assert ctrl._is_waiting is True

        # Next step: still waiting, should emit a small crawl velocity.
        action = ctrl.step(10, 1.0, 0.1, {0: stuck_pos}, emit)
        assert action.behavior == "YIELD"
        speed = math.hypot(action.pref_vx, action.pref_vy)
        assert speed == pytest.approx(0.05, rel=0.1)

    def test_wait_countdown_exits_and_replans(self, emit):
        """After wait_duration ticks, controller exits wait and replans."""
        ctrl = self._make_controller(wait_duration=3)
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)

        stuck_pos = _robot_state(0.0, 0.0)
        for i in range(3):
            ctrl.step(i, 0.1 * i, 0.1, {0: stuck_pos}, emit)
        assert ctrl._is_waiting is True
        backend.calls.clear()

        # wait_duration=3: three more step() calls decrement the counter and exit.
        for i in range(3):
            ctrl.step(100 + i, 10.0 + i * 0.1, 0.1, {0: stuck_pos}, emit)
        assert ctrl._is_waiting is False
        post_wait = [
            c
            for c in emit.call_args_list
            if c.args[0] == "robot_replan" and c.args[2].get("reason") == "post_wait"
        ]
        assert post_wait, "expected a post_wait replan event"
        # Grace period active so stuck detection is skipped temporarily.
        assert ctrl._grace_steps > 0

    def test_grace_period_suppresses_stuck_check(self, emit):
        """Immediately after yielding, _grace_steps prevents immediate re-yield."""
        ctrl = self._make_controller(wait_duration=1, stuck_window=3)
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        # Force waiting state directly, then take a step that exits wait mode.
        ctrl._is_waiting = True
        ctrl._wait_counter = 1
        stuck_pos = _robot_state(0.0, 0.0)
        ctrl.step(100, 10.0, 0.1, {0: stuck_pos}, emit)  # exits wait
        assert ctrl._grace_steps > 0
        before = ctrl._grace_steps
        ctrl.step(101, 10.1, 0.1, {0: stuck_pos}, emit)
        assert ctrl._grace_steps == before - 1


class TestPeriodicReplan:
    def test_replan_interval_triggers_event(self, emit):
        ctrl = BaselineAStarRobotController(
            {"replan_interval": 2, "goal_tolerance": 0.05, "stuck_window": 999}
        )
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        state = _robot_state(0.5, 0.5)
        # step=0: replan (0 % 2 == 0), step=2: replan
        backend.calls.clear()
        ctrl.step(0, 0.0, 0.1, {0: state}, emit)
        ctrl.step(1, 0.1, 0.1, {0: state}, emit)
        ctrl.step(2, 0.2, 0.1, {0: state}, emit)
        replan_events = [c for c in emit.call_args_list if c.args[0] == "robot_replan"]
        assert len(replan_events) == 2
        # Corresponding backend shortest_path calls (plus the one from reset).
        assert len(backend.calls) >= 2


class TestSpeedScaling:
    def test_slowdown_near_target(self, emit):
        """Speed must scale down as dist_target approaches slowdown_dist."""
        ctrl = BaselineAStarRobotController(
            {
                "slowdown_dist": 1.0,
                "max_speed": 1.0,
                "velocity_smoothing": 1.0,
                "stuck_window": 999,
                "goal_tolerance": 0.05,
            }
        )
        backend = _StraightLineBackend()
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), backend)
        # Very close to target (0.1m) -> speed should be ~0.1 * max_speed
        state = _robot_state(ctrl.path[-1][0] - 0.1, ctrl.path[-1][1])
        action = ctrl.step(0, 0.0, 0.1, {0: state}, emit)
        speed = math.hypot(action.pref_vx, action.pref_vy)
        assert speed < 0.2  # scaled well below max


class TestCurrentTargetOverrun:
    def test_current_target_past_path_end_returns_goal(self):
        """_current_target with path_idx beyond path length returns goal."""
        ctrl = BaselineAStarRobotController()
        ctrl.goal = (9.0, 9.0)
        ctrl.path = [(0.0, 0.0), (1.0, 1.0)]
        ctrl.path_idx = 99
        assert ctrl._current_target() == (9.0, 9.0)
