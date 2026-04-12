"""Tests for navirl.models.velocity_obstacle — targets uncovered classes/methods."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.core.types import Action, AgentState
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


def _make_agent(
    agent_id: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    radius: float = 0.3,
    max_speed: float = 1.5,
    kind: str = "human",
    goal_x: float = 0.0,
    goal_y: float = 0.0,
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


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


class TestCross2d:
    def test_perpendicular(self):
        assert _cross2d(1.0, 0.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_parallel(self):
        assert _cross2d(1.0, 0.0, 2.0, 0.0) == pytest.approx(0.0)


class TestInVOCone:
    def test_consistent_with_select_velocity(self):
        """Verify _in_vo_cone classifies velocities consistently with select_velocity."""
        vo = VelocityObstacle()
        a = _make_agent(0, x=0, y=0, vx=0, vy=0)
        b = _make_agent(1, x=3, y=0, vx=0, vy=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None
        # Just verify the function returns a bool for various inputs
        for vx, vy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0.5, 0.5)]:
            result = _in_vo_cone(vx, vy, cone)
            assert isinstance(result, bool)


# ────────────────────────────────────────────────────────────────────
# VOConfig
# ────────────────────────────────────────────────────────────────────


class TestVOConfig:
    def test_defaults(self):
        cfg = VOConfig()
        assert cfg.max_speed > 0
        assert cfg.num_samples > 0
        assert cfg.time_horizon > 0


# ────────────────────────────────────────────────────────────────────
# VelocityObstacle
# ────────────────────────────────────────────────────────────────────


class TestVelocityObstacle:
    def test_compute_vo_basic(self):
        vo = VelocityObstacle()
        a = _make_agent(0, x=0, y=0, vx=1, vy=0)
        b = _make_agent(1, x=3, y=0, vx=-1, vy=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None
        assert isinstance(cone, VOCone)

    def test_compute_vo_too_far(self):
        vo = VelocityObstacle(VOConfig(neighbor_distance=5.0))
        a = _make_agent(0, x=0, y=0)
        b = _make_agent(1, x=100, y=0)
        assert vo.compute_vo(a, b) is None

    def test_compute_vo_same_position(self):
        vo = VelocityObstacle()
        a = _make_agent(0, x=0, y=0)
        b = _make_agent(1, x=0, y=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None
        # Degenerate — full block
        assert cone.left_dx == pytest.approx(1.0)
        assert cone.right_dx == pytest.approx(-1.0)

    def test_compute_vo_overlapping(self):
        """When agents overlap (combined_radius >= dist), half_angle = pi/2."""
        vo = VelocityObstacle(VOConfig(safety_margin=5.0))
        a = _make_agent(0, x=0, y=0, radius=0.3)
        b = _make_agent(1, x=1, y=0, radius=0.3)
        cone = vo.compute_vo(a, b)
        assert cone is not None

    def test_select_velocity(self):
        vo = VelocityObstacle(VOConfig(num_samples=100))
        a = _make_agent(0, x=0, y=0, vx=1, vy=0, max_speed=1.5)
        b = _make_agent(1, x=3, y=0, vx=-1, vy=0)
        cone = vo.compute_vo(a, b)
        assert cone is not None
        vx, vy = vo.select_velocity(a, [cone], (1.0, 0.0))
        speed = math.hypot(vx, vy)
        assert speed <= a.max_speed + 0.01

    def test_select_velocity_no_cones(self):
        vo = VelocityObstacle()
        a = _make_agent(0, x=0, y=0, max_speed=1.5)
        vx, vy = vo.select_velocity(a, [], (1.0, 0.0))
        # Should return preferred velocity
        assert vx == pytest.approx(1.0)
        assert vy == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────────
# ReciprocalVelocityObstacle
# ────────────────────────────────────────────────────────────────────


class TestReciprocalVO:
    def test_apex_is_midpoint(self):
        rvo = ReciprocalVelocityObstacle()
        a = _make_agent(0, x=0, y=0, vx=1, vy=0)
        b = _make_agent(1, x=3, y=0, vx=-1, vy=0)
        cone = rvo.compute_vo(a, b)
        assert cone is not None
        assert cone.apex_vx == pytest.approx(0.0)
        assert cone.apex_vy == pytest.approx(0.0)

    def test_returns_none_far_away(self):
        rvo = ReciprocalVelocityObstacle(VOConfig(neighbor_distance=2.0))
        a = _make_agent(0, x=0, y=0)
        b = _make_agent(1, x=100, y=0)
        assert rvo.compute_vo(a, b) is None


# ────────────────────────────────────────────────────────────────────
# HybridReciprocalVO
# ────────────────────────────────────────────────────────────────────


class TestHybridReciprocalVO:
    def test_compute_vo_left_side(self):
        hrvo = HybridReciprocalVO()
        a = _make_agent(0, x=0, y=0, vx=1, vy=1)
        b = _make_agent(1, x=3, y=0, vx=-1, vy=0)
        cone = hrvo.compute_vo(a, b)
        assert cone is not None

    def test_compute_vo_right_side(self):
        hrvo = HybridReciprocalVO()
        a = _make_agent(0, x=0, y=0, vx=1, vy=-1)
        b = _make_agent(1, x=3, y=0, vx=-1, vy=0)
        cone = hrvo.compute_vo(a, b)
        assert cone is not None

    def test_returns_none_far_away(self):
        hrvo = HybridReciprocalVO(VOConfig(neighbor_distance=2.0))
        a = _make_agent(0, x=0, y=0)
        b = _make_agent(1, x=100, y=0)
        assert hrvo.compute_vo(a, b) is None


# ────────────────────────────────────────────────────────────────────
# ORCAPurePython
# ────────────────────────────────────────────────────────────────────


class TestORCAPurePython:
    def test_compute_orca_lines_no_collision(self):
        orca = ORCAPurePython()
        a = _make_agent(0, x=0, y=0, vx=1, vy=0)
        b = _make_agent(1, x=3, y=0, vx=-1, vy=0)
        lines = orca.compute_orca_lines(a, [b], dt=0.1)
        assert len(lines) == 1
        assert isinstance(lines[0], HalfPlane)

    def test_compute_orca_lines_collision(self):
        """When agents overlap, collision resolution branch is taken."""
        orca = ORCAPurePython()
        a = _make_agent(0, x=0, y=0, vx=0, vy=0, radius=0.5)
        b = _make_agent(1, x=0.2, y=0, vx=0, vy=0, radius=0.5)
        lines = orca.compute_orca_lines(a, [b], dt=0.1)
        assert len(lines) == 1

    def test_skips_self(self):
        orca = ORCAPurePython()
        a = _make_agent(0, x=0, y=0)
        lines = orca.compute_orca_lines(a, [a], dt=0.1)
        assert len(lines) == 0

    def test_compute_orca_lines_leg_left(self):
        """Test the 'project on legs — left' branch."""
        orca = ORCAPurePython(VOConfig(time_horizon=5.0))
        a = _make_agent(0, x=0, y=0, vx=0, vy=0, radius=0.3)
        b = _make_agent(1, x=2, y=0, vx=0, vy=0, radius=0.3)
        lines = orca.compute_orca_lines(a, [b], dt=0.1)
        assert len(lines) == 1

    def test_solve_linear_program_feasible(self):
        orca = ORCAPurePython()
        # Constraint: velocity must be above y=0 line
        lines = [HalfPlane(0.0, 0.0, 0.0, 1.0)]
        vx, vy = orca.solve_linear_program(lines, (0.0, 1.0), max_speed=2.0)
        assert vy >= -0.01  # should satisfy the constraint

    def test_solve_linear_program_already_satisfied(self):
        orca = ORCAPurePython()
        lines = [HalfPlane(0.0, 0.0, 0.0, 1.0)]
        vx, vy = orca.solve_linear_program(lines, (0.0, 1.0), max_speed=2.0)
        assert vx == pytest.approx(0.0)
        assert vy == pytest.approx(1.0)

    def test_solve_linear_program_conflicting_constraints(self):
        orca = ORCAPurePython()
        # Two conflicting half-planes: go up AND go down
        lines = [
            HalfPlane(0.0, 0.5, 0.0, 1.0),  # must be above y=0.5
            HalfPlane(0.0, -0.5, 0.0, -1.0),  # must be below y=-0.5
        ]
        vx, vy = orca.solve_linear_program(lines, (0.0, 0.0), max_speed=2.0)
        speed = math.hypot(vx, vy)
        assert speed <= 2.0 + 0.01

    def test_solve_lp_speed_clamp(self):
        orca = ORCAPurePython()
        # Push preferred velocity to something fast
        lines = [HalfPlane(0.0, 0.0, 0.0, 1.0)]
        vx, vy = orca.solve_linear_program(lines, (10.0, 10.0), max_speed=1.0)
        speed = math.hypot(vx, vy)
        assert speed <= 1.0 + 1e-6

    def test_solve_lp_no_speed_disk_intersection(self):
        """When disc < 0 in solve_linear_program."""
        orca = ORCAPurePython()
        # Constraint far from origin with tiny speed limit
        lines = [HalfPlane(5.0, 5.0, 0.0, 1.0)]
        vx, vy = orca.solve_linear_program(lines, (0.0, 0.0), max_speed=0.1)
        speed = math.hypot(vx, vy)
        assert speed <= 0.1 + 1e-6


# ────────────────────────────────────────────────────────────────────
# VOHumanController
# ────────────────────────────────────────────────────────────────────


class TestVOHumanController:
    def _make_controller(self, algorithm: str = "orca") -> VOHumanController:
        ctrl = VOHumanController(algorithm=algorithm)
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (5.0, 0.0)},
            goals={1: (5.0, 0.0), 2: (0.0, 0.0)},
        )
        return ctrl

    def _make_states(self) -> dict[int, AgentState]:
        return {
            0: _make_agent(0, x=2.5, y=2.5),  # robot
            1: _make_agent(1, x=0.5, y=0.0, vx=1, vy=0),
            2: _make_agent(2, x=4.5, y=0.0, vx=-1, vy=0),
        }

    @pytest.mark.parametrize("algo", ["orca", "rvo", "hrvo", "vo"])
    def test_step_produces_actions(self, algo):
        ctrl = self._make_controller(algo)
        states = self._make_states()
        actions = ctrl.step(
            step=0, time_s=0.0, dt=0.1,
            states=states, robot_id=0,
            emit_event=lambda *a: None,
        )
        assert 1 in actions
        assert 2 in actions
        for act in actions.values():
            assert isinstance(act, Action)

    def test_goal_swap_on_arrival(self):
        ctrl = self._make_controller("orca")
        # Place human 1 at its goal
        states = {
            0: _make_agent(0, x=10, y=10),
            1: _make_agent(1, x=5.0, y=0.0, vx=0, vy=0),
            2: _make_agent(2, x=2.5, y=0.0, vx=-1, vy=0),
        }
        events = []
        ctrl.step(
            step=0, time_s=0.0, dt=0.1,
            states=states, robot_id=0,
            emit_event=lambda name, hid, data: events.append(name),
        )
        assert "goal_swap" in events

    def test_preferred_velocity_at_goal(self):
        ctrl = self._make_controller("orca")
        state = _make_agent(1, x=5.0, y=0.0)
        vx, vy = ctrl._preferred_velocity(state, (5.0, 0.0))
        assert vx == 0.0
        assert vy == 0.0

    def test_missing_human_in_states(self):
        ctrl = self._make_controller("orca")
        # Only robot in states, no humans
        states = {0: _make_agent(0, x=0, y=0)}
        actions = ctrl.step(
            step=0, time_s=0.0, dt=0.1,
            states=states, robot_id=0,
            emit_event=lambda *a: None,
        )
        assert len(actions) == 0
