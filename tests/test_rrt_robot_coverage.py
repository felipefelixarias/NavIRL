"""Cover the missing branches in navirl/robots/baselines/rrt.py.

The existing ``test_robot_baseline_planners.py`` covers creation, node
basics, steering and a happy-path step.  This suite targets the edge
paths: map-metadata fallback, collision-check exception handling, the
RRT loop's invalid-sample / invalid-path continues, the planning
fallback to the backend, the early DONE branch, waypoint advance,
wait/stop conditions, and the ``_current_target`` past-end guard.
"""

from __future__ import annotations

import math
from unittest.mock import Mock

from navirl.core.types import Action, AgentState
from navirl.robots.baselines.rrt import RRTNode, RRTStarRobotController


def _state(x: float, y: float, gx: float = 10.0, gy: float = 10.0) -> AgentState:
    return AgentState(
        agent_id=0,
        kind="robot",
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        goal_x=gx,
        goal_y=gy,
        max_speed=1.0,
        radius=0.3,
    )


class _FreeBackend:
    """Backend where every point is free and ``map_metadata`` is present."""

    def __init__(self, width: int = 10, height: int = 10):
        self._w = width
        self._h = height

    def shortest_path(self, start, goal):
        return [start, goal]

    def check_obstacle_collision(self, pos):
        return False

    def sample_free_point(self):
        return (1.0, 1.0)

    def map_metadata(self):
        return {"width": self._w, "height": self._h}


class _BackendNoMetadata:
    """Backend without ``map_metadata`` — forces the default-bounds path."""

    def shortest_path(self, start, goal):
        return [start, goal]

    def check_obstacle_collision(self, pos):
        return False


# ---------------------------------------------------------------------------
# _get_map_bounds fallback (line 76)
# ---------------------------------------------------------------------------


class TestGetMapBoundsFallback:
    def test_backend_without_metadata_uses_defaults(self):
        ctrl = RRTStarRobotController()
        ctrl.backend = _BackendNoMetadata()
        bounds = ctrl._get_map_bounds()
        assert bounds == (0.0, 0.0, 20.0, 20.0)

    def test_backend_with_metadata_uses_its_values(self):
        ctrl = RRTStarRobotController()
        ctrl.backend = _FreeBackend(width=7, height=13)
        bounds = ctrl._get_map_bounds()
        assert bounds == (0.0, 0.0, 7.0, 13.0)


# ---------------------------------------------------------------------------
# _is_valid_position exception path (lines 82-85)
# ---------------------------------------------------------------------------


class TestIsValidPositionErrors:
    def test_attribute_error_returns_false(self):
        """Backend missing ``check_obstacle_collision`` triggers AttributeError
        inside ``_is_valid_position`` and must not propagate.
        """

        class _Bad:
            pass

        ctrl = RRTStarRobotController()
        ctrl.backend = _Bad()
        assert ctrl._is_valid_position((0.0, 0.0)) is False

    def test_type_error_returns_false(self):
        class _Bad:
            def check_obstacle_collision(self, pos):
                return 1 + "nope"  # TypeError

        ctrl = RRTStarRobotController()
        ctrl.backend = _Bad()
        assert ctrl._is_valid_position((0.0, 0.0)) is False

    def test_value_error_returns_false(self):
        class _Bad:
            def check_obstacle_collision(self, pos):
                raise ValueError("out of bounds")

        ctrl = RRTStarRobotController()
        ctrl.backend = _Bad()
        assert ctrl._is_valid_position((0.0, 0.0)) is False


# ---------------------------------------------------------------------------
# _is_path_valid rejects path through an obstacle (line 144)
# + _plan_rrt_star invalid-new-pos continue (line 180)
# ---------------------------------------------------------------------------


class _WallBackend:
    """Backend with an infinite vertical wall at x in [3.5, 4.5] - everywhere
    else is free.  Used to force ``_is_path_valid`` to return False and to
    force ``_plan_rrt_star`` down its ``continue`` branches.
    """

    def shortest_path(self, start, goal):
        return [start, goal]

    def check_obstacle_collision(self, pos):
        return 3.5 <= pos[0] <= 4.5

    def map_metadata(self):
        return {"width": 10, "height": 10}


class TestPathValidityAndRRTContinueBranches:
    def test_is_path_valid_false_when_crosses_wall(self):
        ctrl = RRTStarRobotController()
        ctrl.backend = _WallBackend()
        # Straight line from (0,0) to (8,0) passes through the wall.
        assert ctrl._is_path_valid((0.0, 0.0), (8.0, 0.0)) is False

    def test_is_path_valid_true_when_clear(self):
        ctrl = RRTStarRobotController()
        ctrl.backend = _WallBackend()
        # A path that stays at x < 3.5 is fine.
        assert ctrl._is_path_valid((0.0, 0.0), (3.0, 3.0)) is True

    def test_rrt_star_handles_blocked_map(self):
        """With an unreachable goal (behind a wall), _plan_rrt_star falls
        through the loop without returning early and returns a path to the
        closest node, appending the goal.  This exercises both the 'invalid
        new_pos / invalid path -> continue' branches and the end-of-loop
        fallback.
        """
        ctrl = RRTStarRobotController(cfg={"max_iterations": 50, "step_size": 0.5})
        ctrl.backend = _WallBackend()
        path = ctrl._plan_rrt_star((0.0, 0.0), (8.0, 0.0))
        assert path, "should return a non-empty path"
        assert path[-1] == (8.0, 0.0)


# ---------------------------------------------------------------------------
# _plan fallback when RRT planning raises (lines 257-261)
# ---------------------------------------------------------------------------


class TestPlanFallback:
    def test_exception_falls_back_to_backend_shortest_path(self):
        """If ``_plan_rrt_star`` raises, ``_plan`` must fall back to the
        backend's shortest_path and set the new path with path_idx=0.
        """
        ctrl = RRTStarRobotController(cfg={"max_iterations": 10})

        class _Backend:
            def map_metadata(self):
                return {"width": 10, "height": 10}

            def shortest_path(self, start, goal):
                return [start, (5.0, 5.0), goal]

            def check_obstacle_collision(self, pos):
                return False

        ctrl.backend = _Backend()
        ctrl.goal = (9.0, 9.0)

        # Monkey-patch _plan_rrt_star to raise a RuntimeError.
        def _raise(*args, **kwargs):
            raise RuntimeError("planning failed")

        ctrl._plan_rrt_star = _raise  # type: ignore[assignment]

        ctrl._plan((0.0, 0.0))
        # Fallback path pulled from backend.shortest_path.
        assert ctrl.path == [(0.0, 0.0), (5.0, 5.0), (9.0, 9.0)]
        assert ctrl.path_idx == 0

    def test_empty_plan_falls_back_to_backend(self):
        """If ``_plan_rrt_star`` returns an empty path, fall back to
        ``backend.shortest_path`` via the ``if not self.path`` branch.
        """
        ctrl = RRTStarRobotController(cfg={"max_iterations": 10})

        class _Backend:
            def map_metadata(self):
                return {"width": 10, "height": 10}

            def shortest_path(self, start, goal):
                return [start, goal]

            def check_obstacle_collision(self, pos):
                return False

        ctrl.backend = _Backend()
        ctrl.goal = (9.0, 9.0)

        # Monkey-patch to return empty.
        ctrl._plan_rrt_star = lambda s, g: []  # type: ignore[assignment]

        ctrl._plan((0.0, 0.0))
        assert ctrl.path == [(0.0, 0.0), (9.0, 9.0)]
        assert ctrl.path_idx == 0

    def test_fallback_when_both_planner_and_backend_return_empty(self):
        """If the planner raises and the backend returns falsy, the path
        defaults to ``[self.goal]`` (the ``or [self.goal]`` guard).
        """
        ctrl = RRTStarRobotController()

        class _Backend:
            def map_metadata(self):
                return {"width": 10, "height": 10}

            def shortest_path(self, start, goal):
                return None

            def check_obstacle_collision(self, pos):
                return False

        ctrl.backend = _Backend()
        ctrl.goal = (4.0, 7.0)

        ctrl._plan_rrt_star = lambda s, g: None  # type: ignore[assignment]

        # None is falsy, so the try-block hits ``self.path = None or [self.goal]``.
        ctrl._plan((0.0, 0.0))
        assert ctrl.path == [(4.0, 7.0)]


# ---------------------------------------------------------------------------
# _current_target past-end guard (line 283)
# ---------------------------------------------------------------------------


class TestCurrentTargetGuard:
    def test_returns_goal_when_path_idx_past_end(self):
        ctrl = RRTStarRobotController()
        ctrl.goal = (5.0, 6.0)
        ctrl.path = [(0.0, 0.0), (1.0, 1.0)]
        ctrl.path_idx = 99
        assert ctrl._current_target() == (5.0, 6.0)


# ---------------------------------------------------------------------------
# step: DONE branch (line 300)
# ---------------------------------------------------------------------------


class TestStepAlreadyAtGoal:
    def test_done_when_within_tolerance(self):
        ctrl = RRTStarRobotController(cfg={"goal_tolerance": 0.5})
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), _FreeBackend())
        states = {0: _state(3.0, 3.0, 3.0, 3.0)}
        action = ctrl.step(0, 0.0, 0.1, states, Mock())
        assert action.behavior == "DONE"
        assert action.pref_vx == 0.0
        assert action.pref_vy == 0.0


# ---------------------------------------------------------------------------
# step: waypoint advance (lines 316-318) + WAIT branch (line 321)
# ---------------------------------------------------------------------------


class TestStepWaypointAdvance:
    def test_advance_waypoint_when_close(self):
        """Put the robot at the first waypoint with a second waypoint
        ahead.  The ``dist_target <= goal_tolerance`` branch should advance
        path_idx.
        """
        ctrl = RRTStarRobotController(cfg={"goal_tolerance": 0.2, "replan_interval": 1000})
        ctrl.reset(0, (0.0, 0.0), (9.0, 9.0), _FreeBackend())
        # Install a custom path: (0,0) -> (1,0) -> (9,9)
        ctrl.path = [(0.0, 0.0), (1.0, 0.0), (9.0, 9.0)]
        ctrl.path_idx = 0
        # Disable target_lookahead so current target == path[path_idx]
        ctrl.target_lookahead = 1

        # Robot sitting on top of the first waypoint should advance.
        states = {0: _state(0.0, 0.0, 9.0, 9.0)}
        ctrl.step(1, 0.1, 0.1, states, Mock())
        assert ctrl.path_idx == 1

    def test_wait_when_target_distance_zero(self):
        """When the (post-advance) target coincides with the robot and the
        path is exhausted, the ``dist_target < 1e-8`` branch must return
        WAIT.
        """
        ctrl = RRTStarRobotController(cfg={"goal_tolerance": 0.05, "replan_interval": 1000})
        ctrl.reset(0, (0.0, 0.0), (5.0, 5.0), _FreeBackend())
        # Single-waypoint path at the robot's position.
        ctrl.path = [(1.0, 1.0)]
        ctrl.path_idx = 0
        ctrl.target_lookahead = 1

        # Robot is *at* the path waypoint but not at the goal: advance will
        # not fire (already past end), and _current_target returns the path
        # entry; dist_target == 0.
        states = {0: _state(1.0, 1.0, 5.0, 5.0)}
        action = ctrl.step(1, 0.1, 0.1, states, Mock())
        assert action.behavior == "WAIT"


# ---------------------------------------------------------------------------
# step: stop_speed branch (line 341)
# ---------------------------------------------------------------------------


class TestStepStopSpeed:
    def test_velocity_zeroed_when_below_stop_speed_near_goal(self):
        """Configure a very small computed velocity with the robot closer
        than ``goal_tolerance`` but not quite at the goal.  The final
        ``if math.hypot(vx, vy) < stop_speed and dist_target < goal_tolerance``
        branch should fire and zero out the velocity.
        """
        ctrl = RRTStarRobotController(
            cfg={
                "goal_tolerance": 0.3,
                "slowdown_dist": 10.0,  # so speed_scale = dist/slowdown is tiny
                "max_speed": 1.0,
                "stop_speed": 0.5,  # any velocity magnitude below this triggers stop
                "replan_interval": 1000,
                "velocity_smoothing": 1.0,
            }
        )
        # Goal is 3 away (not within tolerance), so step() won't take the DONE
        # branch. Goal must equal the robot's state goal, since that's what
        # the DONE check uses.
        ctrl.reset(0, (0.0, 0.0), (3.0, 0.0), _FreeBackend())
        # Single-waypoint path close to the robot: path_idx=0 == len-1 means
        # the waypoint-advance branch won't fire.
        ctrl.path = [(0.2, 0.0)]
        ctrl.path_idx = 0
        ctrl.target_lookahead = 1

        states = {0: _state(0.0, 0.0, 3.0, 0.0)}
        action = ctrl.step(1, 0.1, 0.1, states, Mock())
        # dist_target = 0.2 (<goal_tolerance=0.3). speed_scale = 0.2/10 = 0.02.
        # speed = min(1, 1) * 0.02 = 0.02. hypot < stop_speed (0.5) AND
        # dist_target < goal_tolerance -> zeroed.
        assert action.pref_vx == 0.0
        assert action.pref_vy == 0.0


# ---------------------------------------------------------------------------
# Periodic replan event (line 305-309 already covered by happy path, but
# pin it down explicitly so regressions surface).
# ---------------------------------------------------------------------------


class TestPeriodicReplan:
    def test_emit_event_when_step_is_replan_interval_multiple(self):
        ctrl = RRTStarRobotController(cfg={"replan_interval": 5, "max_iterations": 10})
        ctrl.reset(0, (0.0, 0.0), (3.0, 3.0), _FreeBackend())
        emit = Mock()
        states = {0: _state(0.0, 0.0, 3.0, 3.0)}
        # step % 5 == 0 -> replan event fires.
        ctrl.step(5, 0.5, 0.1, states, emit)
        assert any(call.args[0] == "robot_rrt_replan" for call in emit.call_args_list)


# ---------------------------------------------------------------------------
# RRTNode basics (sanity — cheap test that pins the observable behavior
# of ``path_to_root`` and ``_distance_to`` even for a single node).
# ---------------------------------------------------------------------------


class TestRRTNodeExtras:
    def test_path_to_root_single_node(self):
        root = RRTNode((2.0, 3.0))
        assert root.path_to_root() == [(2.0, 3.0)]

    def test_cost_accumulates_along_chain(self):
        a = RRTNode((0.0, 0.0))
        b = RRTNode((3.0, 4.0), a)  # dist 5
        c = RRTNode((3.0, 4.0 + 5.0), b)  # dist 5
        assert math.isclose(b.cost, 5.0)
        assert math.isclose(c.cost, 10.0)
        path = c.path_to_root()
        assert len(path) == 3 and path[0] == (0.0, 0.0)
