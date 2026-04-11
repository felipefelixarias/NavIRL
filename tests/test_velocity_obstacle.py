"""Tests for navirl.models.velocity_obstacle — VO, RVO, HRVO, ORCA."""

from __future__ import annotations

import math

import pytest

from navirl.core.types import AgentState
from navirl.models.velocity_obstacle import (
    HalfPlane,
    HybridReciprocalVO,
    ORCAPurePython,
    ReciprocalVelocityObstacle,
    VelocityObstacle,
    VOCone,
    VOConfig,
    VOHumanController,
    _cross2d,
    _in_vo_cone,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _agent(
    aid=0,
    x=0.0,
    y=0.0,
    vx=0.0,
    vy=0.0,
    radius=0.25,
    max_speed=1.5,
    gx=10.0,
    gy=0.0,
):
    return AgentState(
        agent_id=aid,
        kind="human",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=gx,
        goal_y=gy,
        radius=radius,
        max_speed=max_speed,
    )


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------


class TestCross2D:
    def test_basic(self):
        assert _cross2d(1, 0, 0, 1) == 1.0
        assert _cross2d(0, 1, 1, 0) == -1.0

    def test_parallel(self):
        assert _cross2d(1, 0, 2, 0) == 0.0


class TestInVOCone:
    def test_inside_cone(self):
        # left=(1,0), right=(0,1): cone spans from +x toward +y
        cone = VOCone(0, 0, 1, 0, 0, 1)
        assert _in_vo_cone(0.5, 0.5, cone) is True

    def test_outside_cone(self):
        cone = VOCone(0, 0, 1, 0, 0, 1)
        assert _in_vo_cone(-1, 0, cone) is False


# ---------------------------------------------------------------------------
#  VelocityObstacle
# ---------------------------------------------------------------------------


class TestVelocityObstacle:
    def test_compute_vo_basic(self):
        vo = VelocityObstacle()
        a = _agent(aid=0, x=0, y=0)
        b = _agent(aid=1, x=2, y=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None
        # Apex should be at obstacle's velocity
        assert cone.apex_vx == b.vx
        assert cone.apex_vy == b.vy

    def test_beyond_neighbor_distance(self):
        cfg = VOConfig(neighbor_distance=1.0)
        vo = VelocityObstacle(cfg)
        a = _agent(aid=0, x=0, y=0)
        b = _agent(aid=1, x=5, y=0)
        assert vo.compute_vo(a, b) is None

    def test_overlapping_agents(self):
        vo = VelocityObstacle()
        a = _agent(aid=0, x=0, y=0, radius=0.5)
        b = _agent(aid=1, x=0.3, y=0, radius=0.5)
        cone = vo.compute_vo(a, b)
        assert cone is not None

    def test_coincident_agents(self):
        vo = VelocityObstacle()
        a = _agent(aid=0, x=0, y=0)
        b = _agent(aid=1, x=0, y=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None

    def test_select_velocity_prefers_outside_cone(self):
        vo = VelocityObstacle(VOConfig(num_samples=500))
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b = _agent(aid=1, x=2, y=0, vx=0, vy=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None
        vx, vy = vo.select_velocity(a, [cone], (1.0, 0.0))
        # Should get a velocity (may differ from preferred to avoid collision)
        assert math.hypot(vx, vy) <= a.max_speed + 0.1

    def test_select_velocity_no_cones(self):
        vo = VelocityObstacle()
        a = _agent(aid=0, max_speed=2.0)
        vx, vy = vo.select_velocity(a, [], (1.0, 0.5))
        # No obstacles → should return preferred velocity
        assert abs(vx - 1.0) < 0.01
        assert abs(vy - 0.5) < 0.01


# ---------------------------------------------------------------------------
#  ReciprocalVelocityObstacle
# ---------------------------------------------------------------------------


class TestReciprocalVO:
    def test_apex_is_midpoint(self):
        rvo = ReciprocalVelocityObstacle()
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b = _agent(aid=1, x=2, y=0, vx=-1, vy=0)
        cone = rvo.compute_vo(a, b)
        assert cone is not None
        assert abs(cone.apex_vx - 0.0) < 1e-9  # midpoint of (1, -1) = 0
        assert abs(cone.apex_vy - 0.0) < 1e-9

    def test_returns_none_beyond_range(self):
        cfg = VOConfig(neighbor_distance=1.0)
        rvo = ReciprocalVelocityObstacle(cfg)
        a = _agent(aid=0, x=0, y=0)
        b = _agent(aid=1, x=5, y=0)
        assert rvo.compute_vo(a, b) is None


# ---------------------------------------------------------------------------
#  HybridReciprocalVO
# ---------------------------------------------------------------------------


class TestHybridReciprocalVO:
    def test_returns_cone(self):
        hrvo = HybridReciprocalVO()
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b = _agent(aid=1, x=2, y=0, vx=0, vy=0)
        cone = hrvo.compute_vo(a, b)
        assert cone is not None

    def test_returns_none_beyond_range(self):
        cfg = VOConfig(neighbor_distance=1.0)
        hrvo = HybridReciprocalVO(cfg)
        a = _agent(aid=0, x=0, y=0)
        b = _agent(aid=1, x=5, y=0)
        assert hrvo.compute_vo(a, b) is None


# ---------------------------------------------------------------------------
#  ORCAPurePython
# ---------------------------------------------------------------------------


class TestORCAPurePython:
    def test_compute_orca_lines_basic(self):
        orca = ORCAPurePython()
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b = _agent(aid=1, x=2, y=0, vx=-1, vy=0)
        lines = orca.compute_orca_lines(a, [b], dt=0.1)
        assert len(lines) == 1
        assert isinstance(lines[0], HalfPlane)

    def test_skips_self(self):
        orca = ORCAPurePython()
        a = _agent(aid=0)
        lines = orca.compute_orca_lines(a, [a], dt=0.1)
        assert len(lines) == 0

    def test_colliding_agents(self):
        orca = ORCAPurePython()
        a = _agent(aid=0, x=0, y=0, radius=0.5)
        b = _agent(aid=1, x=0.3, y=0, radius=0.5)
        lines = orca.compute_orca_lines(a, [b], dt=0.1)
        assert len(lines) == 1

    def test_solve_lp_no_constraints(self):
        orca = ORCAPurePython()
        vx, vy = orca.solve_linear_program([], (1.0, 0.5), max_speed=2.0)
        assert abs(vx - 1.0) < 1e-9
        assert abs(vy - 0.5) < 1e-9

    def test_solve_lp_speed_clamp(self):
        orca = ORCAPurePython()
        vx, vy = orca.solve_linear_program([], (10.0, 0.0), max_speed=1.0)
        assert math.hypot(vx, vy) <= 1.0 + 1e-9

    def test_solve_lp_single_constraint(self):
        orca = ORCAPurePython()
        # A half-plane blocking velocity in +x direction
        hp = HalfPlane(point_x=0.5, point_y=0.0, normal_x=-1.0, normal_y=0.0)
        vx, vy = orca.solve_linear_program([hp], (1.0, 0.0), max_speed=2.0)
        # Result should satisfy the constraint: (vx - 0.5) * (-1) >= 0 → vx <= 0.5
        assert vx <= 0.5 + 1e-6

    def test_solve_lp_multiple_constraints(self):
        orca = ORCAPurePython()
        hp1 = HalfPlane(point_x=0.5, point_y=0.0, normal_x=-1.0, normal_y=0.0)
        hp2 = HalfPlane(point_x=0.0, point_y=0.5, normal_x=0.0, normal_y=-1.0)
        vx, vy = orca.solve_linear_program([hp1, hp2], (1.0, 1.0), max_speed=2.0)
        # Should satisfy both constraints
        assert vx <= 0.5 + 1e-6
        assert vy <= 0.5 + 1e-6

    def test_end_to_end_two_agents(self):
        orca = ORCAPurePython()
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b = _agent(aid=1, x=3, y=0, vx=-1, vy=0)
        lines = orca.compute_orca_lines(a, [b], dt=0.1)
        vx, vy = orca.solve_linear_program(lines, (1.0, 0.0), max_speed=1.5)
        assert math.hypot(vx, vy) <= 1.5 + 1e-9


# ---------------------------------------------------------------------------
#  VOHumanController
# ---------------------------------------------------------------------------


class TestVOHumanController:
    @pytest.mark.parametrize("algo", ["orca", "rvo", "hrvo", "vo"])
    def test_instantiation(self, algo):
        ctrl = VOHumanController(algorithm=algo)
        assert ctrl.algorithm == algo

    def test_reset_and_step_orca(self):
        ctrl = VOHumanController(algorithm="orca")
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )
        states = {
            1: _agent(aid=1, x=0, y=0, vx=0, vy=0, gx=10, gy=0),
            99: _agent(aid=99, x=5, y=5),
        }
        events = []
        actions = ctrl.step(0, 0.0, 0.1, states, 99, lambda *a: events.append(a))
        assert 1 in actions
        assert actions[1].behavior == "GO_TO"

    def test_reset_and_step_vo(self):
        ctrl = VOHumanController(algorithm="vo")
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )
        states = {1: _agent(aid=1, x=0, y=0, gx=10, gy=0)}
        actions = ctrl.step(0, 0.0, 0.1, states, 99, lambda *a: None)
        assert 1 in actions

    def test_goal_swap(self):
        ctrl = VOHumanController(algorithm="orca")
        ctrl.goal_tolerance = 1.0
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )
        states = {1: _agent(aid=1, x=10.0, y=0.0, gx=10, gy=0)}
        events = []
        ctrl.step(0, 0.0, 0.1, states, 99, lambda *a: events.append(a))
        assert ctrl.goals[1] == (0.0, 0.0)

    def test_preferred_velocity_at_goal(self):
        ctrl = VOHumanController()
        ctrl.goal_tolerance = 1.0
        a = _agent(aid=0, x=10.0, y=0.0, gx=10, gy=0)
        vx, vy = ctrl._preferred_velocity(a, (10.0, 0.0))
        assert vx == 0.0 and vy == 0.0

    def test_preferred_velocity_toward_goal(self):
        ctrl = VOHumanController()
        ctrl.goal_tolerance = 0.5
        a = _agent(aid=0, x=0, y=0, gx=10, gy=0, max_speed=1.5)
        vx, vy = ctrl._preferred_velocity(a, (10.0, 0.0))
        assert vx > 0
        assert abs(vy) < 1e-9
        assert math.hypot(vx, vy) <= 1.5 + 1e-9
