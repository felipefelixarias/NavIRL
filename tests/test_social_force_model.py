"""Tests for navirl.models.social_force — Social Force Model implementation."""

from __future__ import annotations

import math

import pytest

from navirl.core.constants import EPSILON
from navirl.core.types import AgentState
from navirl.models.social_force import (
    SocialForceConfig,
    SocialForceHumanController,
    SocialForceModel,
    _anisotropy_weight,
    _point_to_segment_distance,
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
    gx=10.0,
    gy=0.0,
    radius=0.25,
    max_speed=1.5,
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


class TestAnisotropyWeight:
    def test_facing_toward_neighbor(self):
        # heading = (1,0), neighbor direction = (-1,0) → behind agent
        w = _anisotropy_weight(1.0, 0.0, -1.0, 0.0, 0.5)
        # cos_theta = -(1*(-1) + 0*0) = 1 → w = 0.5 + 0.5*(1+1)/2 = 1.0
        assert abs(w - 1.0) < 1e-9

    def test_facing_away_from_neighbor(self):
        # heading = (1,0), neighbor direction = (1,0) → in front
        w = _anisotropy_weight(1.0, 0.0, 1.0, 0.0, 0.5)
        # cos_theta = -(1*1) = -1 → w = 0.5 + 0.5*(1-1)/2 = 0.5
        assert abs(w - 0.5) < 1e-9

    def test_isotropic(self):
        w = _anisotropy_weight(1.0, 0.0, 0.0, 1.0, 0.0)
        # lambda=0: w = 0 + 1*(1 + cos_theta)/2
        # cos_theta = -(0) = 0 → w = (1+0)/2 = 0.5
        assert abs(w - 0.5) < 1e-9


class TestPointToSegmentDistance:
    def test_projection_on_segment(self):
        dist, px, py = _point_to_segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
        assert abs(dist - 3.0) < 1e-9
        assert abs(px - 5.0) < 1e-9
        assert abs(py - 0.0) < 1e-9

    def test_closest_to_endpoint_a(self):
        dist, px, py = _point_to_segment_distance(-1.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        assert abs(px - 0.0) < 1e-9

    def test_closest_to_endpoint_b(self):
        dist, px, py = _point_to_segment_distance(12.0, 0.0, 0.0, 0.0, 10.0, 0.0)
        assert abs(px - 10.0) < 1e-9

    def test_degenerate_segment(self):
        dist, px, py = _point_to_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
        assert abs(dist - 5.0) < 1e-9


# ---------------------------------------------------------------------------
#  SocialForceConfig
# ---------------------------------------------------------------------------


class TestSocialForceConfig:
    def test_default_values(self):
        cfg = SocialForceConfig()
        assert cfg.A > 0
        assert cfg.B > 0
        assert cfg.tau > 0
        assert cfg.interaction_range > 0


# ---------------------------------------------------------------------------
#  SocialForceModel — desired force
# ---------------------------------------------------------------------------


class TestDesiredForce:
    def test_stationary_agent_toward_goal(self):
        model = SocialForceModel()
        a = _agent(x=0, y=0, vx=0, vy=0, gx=10.0, gy=0.0, max_speed=1.5)
        fx, fy = model.compute_desired_force(a, (10.0, 0.0))
        # Should accelerate in +x toward goal
        assert fx > 0
        assert abs(fy) < 1e-9

    def test_at_goal_decelerates(self):
        model = SocialForceModel()
        a = _agent(x=10.0, y=0.0, vx=1.0, vy=0.0, gx=10.0, gy=0.0)
        fx, fy = model.compute_desired_force(a, (10.0, 0.0))
        # At goal with velocity → should decelerate
        assert fx < 0

    def test_desired_force_magnitude(self):
        model = SocialForceModel(SocialForceConfig(tau=0.5))
        a = _agent(vx=0, vy=0, max_speed=1.5, gx=10.0, gy=0.0)
        fx, fy = model.compute_desired_force(a, (10.0, 0.0))
        # f = (v_pref - v_current) / tau = (1.5 - 0) / 0.5 = 3.0
        assert abs(fx - 3.0) < 0.1


# ---------------------------------------------------------------------------
#  SocialForceModel — social force
# ---------------------------------------------------------------------------


class TestSocialForce:
    def test_repulsion_between_agents(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b = _agent(aid=1, x=1.0, y=0, vx=-1, vy=0)
        fx, fy = model.compute_social_force(a, [a, b])
        # Agent b is in front → repulsive force should push a in -x
        assert fx < 0

    def test_no_self_force(self):
        model = SocialForceModel()
        a = _agent(aid=0)
        fx, fy = model.compute_social_force(a, [a])
        assert fx == 0.0 and fy == 0.0

    def test_force_decays_with_distance(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, vx=1, vy=0)
        b_near = _agent(aid=1, x=1.0, y=0)
        b_far = _agent(aid=2, x=3.0, y=0)
        fx_near, _ = model.compute_social_force(a, [a, b_near])
        fx_far, _ = model.compute_social_force(a, [a, b_far])
        assert abs(fx_near) > abs(fx_far)

    def test_beyond_interaction_range(self):
        cfg = SocialForceConfig(interaction_range=2.0)
        model = SocialForceModel(cfg)
        a = _agent(aid=0, x=0, y=0)
        b = _agent(aid=1, x=10.0, y=0)
        fx, fy = model.compute_social_force(a, [a, b])
        assert fx == 0.0 and fy == 0.0


# ---------------------------------------------------------------------------
#  SocialForceModel — wall force
# ---------------------------------------------------------------------------


class TestWallForce:
    def test_wall_repulsion(self):
        model = SocialForceModel()
        a = _agent(x=5.0, y=0.5, radius=0.25)
        walls = [(0.0, 0.0, 10.0, 0.0)]  # wall along x-axis
        fx, fy = model.compute_wall_force(a, walls)
        # Agent close to wall → repulsion in +y
        assert fy > 0

    def test_no_force_far_from_wall(self):
        cfg = SocialForceConfig(wall_interaction_range=3.0)
        model = SocialForceModel(cfg)
        a = _agent(x=5.0, y=10.0)
        walls = [(0.0, 0.0, 10.0, 0.0)]
        fx, fy = model.compute_wall_force(a, walls)
        assert abs(fx) < 1e-6 and abs(fy) < 1e-6

    def test_contact_force_with_wall(self):
        model = SocialForceModel()
        # Agent overlapping wall (y=0.1 < radius=0.25)
        a = _agent(x=5.0, y=0.1, vx=1.0, vy=0.0, radius=0.25)
        walls = [(0.0, 0.0, 10.0, 0.0)]
        fx, fy = model.compute_wall_force(a, walls)
        # Should get strong push in +y (body compression)
        assert fy > 0


# ---------------------------------------------------------------------------
#  SocialForceModel — contact force
# ---------------------------------------------------------------------------


class TestContactForce:
    def test_overlapping_agents(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, radius=0.25)
        b = _agent(aid=1, x=0.3, y=0, radius=0.25)
        # overlap = 0.5 - 0.3 = 0.2 > 0 → contact
        fx, fy = model.compute_contact_force(a, [a, b])
        assert fx < 0  # pushed away from b (in -x)

    def test_no_contact_separated(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, radius=0.25)
        b = _agent(aid=1, x=2.0, y=0, radius=0.25)
        fx, fy = model.compute_contact_force(a, [a, b])
        assert fx == 0.0 and fy == 0.0


# ---------------------------------------------------------------------------
#  SocialForceModel — total force and step
# ---------------------------------------------------------------------------


class TestTotalForceAndStep:
    def test_total_force_is_sum(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, vx=0, vy=0, gx=10, gy=0)
        fx, fy = model.compute_total_force(a, [a])
        # Only desired force active (no neighbors, no walls)
        fd_x, fd_y = model.compute_desired_force(a, (10, 0))
        assert abs(fx - fd_x) < 1e-9
        assert abs(fy - fd_y) < 1e-9

    def test_step_returns_velocities(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, vx=0, vy=0, gx=10, gy=0)
        b = _agent(aid=1, x=5, y=3, vx=0, vy=0, gx=0, gy=0)
        result = model.step([a, b], {0: (10, 0), 1: (0, 0)}, dt=0.04)
        assert 0 in result
        assert 1 in result

    def test_step_respects_max_speed(self):
        model = SocialForceModel()
        a = _agent(aid=0, x=0, y=0, vx=0, vy=0, gx=10, gy=0, max_speed=1.0)
        result = model.step([a], {0: (10, 0)}, dt=10.0)
        vx, vy = result[0]
        assert math.hypot(vx, vy) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
#  SocialForceHumanController
# ---------------------------------------------------------------------------


class TestSocialForceHumanController:
    def test_reset_and_step(self):
        ctrl = SocialForceHumanController()
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (10.0, 0.0)},
            goals={1: (10.0, 0.0), 2: (0.0, 0.0)},
        )
        states = {
            1: _agent(aid=1, x=0, y=0, vx=0, vy=0, gx=10, gy=0),
            2: _agent(aid=2, x=10, y=0, vx=0, vy=0, gx=0, gy=0),
            99: _agent(aid=99, x=5, y=5),  # robot
        }
        events = []
        actions = ctrl.step(0, 0.0, 0.04, states, 99, lambda *a: events.append(a))
        assert 1 in actions
        assert 2 in actions
        assert actions[1].behavior == "GO_TO"

    def test_goal_swap_on_arrival(self):
        ctrl = SocialForceHumanController()
        ctrl.goal_tolerance = 1.0
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )
        # Place agent AT goal
        states = {1: _agent(aid=1, x=10.0, y=0.0)}
        events = []
        ctrl.step(0, 0.0, 0.04, states, 99, lambda *a: events.append(a))
        # Goal should have swapped
        assert ctrl.goals[1] == (0.0, 0.0)
        assert ctrl.starts[1] == (10.0, 0.0)
        assert len(events) == 1
        assert events[0][0] == "goal_swap"

    def test_missing_human_skipped(self):
        ctrl = SocialForceHumanController()
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0, 0), 2: (5, 0)},
            goals={1: (10, 0), 2: (0, 0)},
        )
        # Only human 1 present
        states = {1: _agent(aid=1)}
        actions = ctrl.step(0, 0.0, 0.04, states, 99, lambda *a: None)
        assert 1 in actions
        assert 2 not in actions
