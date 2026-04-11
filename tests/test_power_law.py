"""Tests for navirl.models.power_law module."""

from __future__ import annotations

import math

import pytest

from navirl.core.types import Action, AgentState
from navirl.models.power_law import (
    PowerLawConfig,
    PowerLawHumanController,
    PowerLawModel,
    _time_to_collision,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(
    agent_id=0,
    x=0.0,
    y=0.0,
    vx=0.0,
    vy=0.0,
    goal_x=10.0,
    goal_y=0.0,
    radius=0.18,
    max_speed=1.5,
    kind="human",
):
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


# ---------------------------------------------------------------------------
# PowerLawConfig
# ---------------------------------------------------------------------------


class TestPowerLawConfig:
    def test_defaults(self):
        cfg = PowerLawConfig()
        assert cfg.k == 1.5
        assert cfg.tau_0 == 3.0
        assert cfg.sigma == 0.1
        assert cfg.max_force == 40.0

    def test_custom(self):
        cfg = PowerLawConfig(k=2.0, tau_0=5.0, sigma=0.0)
        assert cfg.k == 2.0
        assert cfg.tau_0 == 5.0
        assert cfg.sigma == 0.0


# ---------------------------------------------------------------------------
# _time_to_collision
# ---------------------------------------------------------------------------


class TestTimeToCollision:
    def test_head_on_collision(self):
        # px,py = other-self, vx,vy = self-other
        # Agents approaching: relative position (5,0), relative velocity
        # that closes the gap. |p + v*t|^2 shrinks when v opposes p.
        # So vx = -2 (closing), px = 5
        tau = _time_to_collision(5.0, 0.0, -2.0, 0.0, 0.36)
        assert tau > 0
        assert tau < 5.0

    def test_no_relative_motion(self):
        tau = _time_to_collision(5.0, 0.0, 0.0, 0.0, 0.36)
        assert tau == float("inf")

    def test_already_overlapping(self):
        tau = _time_to_collision(0.1, 0.0, 1.0, 0.0, 0.5)
        assert tau == 0.0

    def test_moving_apart(self):
        # Relative velocity points same direction as relative position (moving apart)
        tau = _time_to_collision(5.0, 0.0, 2.0, 0.0, 0.36)
        assert tau == float("inf")

    def test_no_collision_miss(self):
        # Moving perpendicular, will miss
        tau = _time_to_collision(5.0, 0.0, 0.0, 1.0, 0.36)
        assert tau == float("inf")


# ---------------------------------------------------------------------------
# PowerLawModel
# ---------------------------------------------------------------------------


class TestPowerLawModel:
    def test_desired_force_at_goal(self):
        model = PowerLawModel(PowerLawConfig(relaxation_time=0.5))
        state = _agent(x=10.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        # At goal, force should decelerate
        assert abs(fx) < 1e-6 and abs(fy) < 1e-6

    def test_desired_force_toward_goal(self):
        model = PowerLawModel(PowerLawConfig(relaxation_time=0.5))
        state = _agent(x=0.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        assert fx > 0  # Should push toward goal (positive x)
        assert abs(fy) < 1e-6  # No y-force

    def test_anticipatory_force_no_neighbors(self):
        model = PowerLawModel(PowerLawConfig(sigma=0.0))
        state = _agent(agent_id=0)
        fx, fy = model.compute_anticipatory_force(state, [state])
        # Only self in the list, should skip
        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_anticipatory_force_with_approaching_agent(self):
        model = PowerLawModel(PowerLawConfig(sigma=0.0))
        # rel_px = other.x - state.x = 3, rel_vx = state.vx - other.vx
        # For collision: need vx such that rel_vx is negative (closing)
        # a1 stationary, a2 approaching from right
        a1 = _agent(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0)
        a2 = _agent(agent_id=1, x=3.0, y=0.0, vx=-2.0, vy=0.0)
        # rel_vx = 0 - (-2) = 2, rel_px = 3 → |3 + 2t| shrinks? No, grows.
        # Actually in the code: vx = self - other, and eq is |p + v*t|^2
        # p=(3,0), v=(2,0) → grows. Need v negative.
        # a1 moving toward a2: vx=2, a2 stationary
        a1 = _agent(agent_id=0, x=0.0, y=0.0, vx=2.0, vy=0.0)
        a2 = _agent(agent_id=1, x=3.0, y=0.0, vx=2.0, vy=0.0)
        # rel_vx = 2-2 = 0, no relative motion. Try differently:
        # a1 fast toward a2, a2 slower
        a1 = _agent(agent_id=0, x=0.0, y=0.0, vx=3.0, vy=0.0)
        a2 = _agent(agent_id=1, x=3.0, y=0.0, vx=0.0, vy=0.0)
        # rel_vx = 3-0 = 3, rel_px = 3, |3+3t| = 3(1+t) → grows → no collision
        # The math: solve (3+3t)^2 = r^2 → t = (r-3)/3 or (-r-3)/3 both negative
        # So we need rel_vx negative. That means other.vx > state.vx
        a1 = _agent(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0)
        a2 = _agent(agent_id=1, x=3.0, y=0.0, vx=3.0, vy=0.0)
        # rel_vx = 0-3 = -3, rel_px = 3, |3-3t| → hits 0 at t=1
        fx, fy = model.compute_anticipatory_force(a1, [a1, a2])
        # Force should push agent 0 away from predicted collision (negative x)
        assert fx != 0.0

    def test_anticipatory_force_far_away_ignored(self):
        model = PowerLawModel(PowerLawConfig(sigma=0.0, neighbor_distance=5.0))
        a1 = _agent(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0)
        a2 = _agent(agent_id=1, x=100.0, y=0.0, vx=0.0, vy=0.0)
        fx, fy = model.compute_anticipatory_force(a1, [a1, a2])
        assert abs(fx) < 1e-6
        assert abs(fy) < 1e-6

    def test_total_force_combines(self):
        model = PowerLawModel(PowerLawConfig(sigma=0.0))
        state = _agent(x=0.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_total_force(state, [state])
        # Should have desired force toward goal
        assert fx > 0

    def test_step_produces_velocities(self):
        model = PowerLawModel(PowerLawConfig(sigma=0.0))
        a1 = _agent(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0)
        a2 = _agent(agent_id=1, x=5.0, y=5.0, vx=0.0, vy=0.0, goal_x=0.0, goal_y=0.0)
        goals = {0: (10.0, 0.0), 1: (0.0, 0.0)}
        vels = model.step([a1, a2], goals, dt=0.04)
        assert 0 in vels
        assert 1 in vels
        assert len(vels[0]) == 2
        assert len(vels[1]) == 2

    def test_step_clamps_speed(self):
        cfg = PowerLawConfig(sigma=0.0, relaxation_time=0.01)
        model = PowerLawModel(cfg)
        state = _agent(x=0.0, y=0.0, vx=0.0, vy=0.0, max_speed=1.0)
        vels = model.step([state], {0: (100.0, 0.0)}, dt=1.0)
        vx, vy = vels[0]
        speed = math.hypot(vx, vy)
        assert speed <= 1.0 + 1e-6

    def test_noise_adds_variation(self):
        model = PowerLawModel(PowerLawConfig(sigma=1.0))
        state = _agent(agent_id=0)
        results = set()
        for _ in range(10):
            fx, fy = model.compute_anticipatory_force(state, [state])
            results.add(round(fx, 4))
        # With noise, we should get different values
        assert len(results) > 1


# ---------------------------------------------------------------------------
# PowerLawHumanController
# ---------------------------------------------------------------------------


class TestPowerLawHumanController:
    def _make_controller(self):
        ctrl = PowerLawHumanController(PowerLawConfig(sigma=0.0))
        starts = {0: (0.0, 0.0), 1: (10.0, 0.0)}
        goals = {0: (10.0, 0.0), 1: (0.0, 0.0)}
        ctrl.reset([0, 1], starts, goals)
        return ctrl

    def test_reset_stores_ids(self):
        ctrl = self._make_controller()
        assert ctrl.human_ids == [0, 1]
        assert ctrl.goals[0] == (10.0, 0.0)

    def test_step_returns_actions(self):
        ctrl = self._make_controller()
        states = {
            0: _agent(agent_id=0, x=0, y=0, vx=0, vy=0),
            1: _agent(agent_id=1, x=10, y=0, vx=0, vy=0),
        }

        def emit(event_type, agent_id, data):
            pass

        actions = ctrl.step(0, 0.0, 0.04, states, robot_id=-1, emit_event=emit)
        assert 0 in actions
        assert 1 in actions
        assert isinstance(actions[0], Action)
        assert actions[0].behavior == "GO_TO"
        assert actions[0].metadata["model"] == "power_law"

    def test_goal_swap_on_arrival(self):
        ctrl = PowerLawHumanController(PowerLawConfig(sigma=0.0))
        ctrl.reset([0], {0: (0.0, 0.0)}, {0: (1.0, 0.0)})
        ctrl.goal_tolerance = 0.5

        states = {
            0: _agent(agent_id=0, x=0.8, y=0.0, vx=0.5, vy=0.0, goal_x=1.0, goal_y=0.0),
        }

        events = []

        def emit(event_type, agent_id, data):
            events.append((event_type, agent_id, data))

        ctrl.step(0, 0.0, 0.04, states, robot_id=-1, emit_event=emit)
        # Should have swapped goals
        assert len(events) == 1
        assert events[0][0] == "goal_swap"
        assert ctrl.goals[0] == (0.0, 0.0)

    def test_missing_state_skipped(self):
        ctrl = PowerLawHumanController(PowerLawConfig(sigma=0.0))
        ctrl.reset([0, 1], {0: (0, 0), 1: (10, 0)}, {0: (10, 0), 1: (0, 0)})
        states = {0: _agent(agent_id=0)}  # Agent 1 missing
        actions = ctrl.step(0, 0.0, 0.04, states, robot_id=-1, emit_event=lambda *a: None)
        assert 0 in actions
        assert 1 not in actions

    def test_speed_clamping(self):
        ctrl = PowerLawHumanController(PowerLawConfig(sigma=0.0, relaxation_time=0.001))
        ctrl.reset([0], {0: (0, 0)}, {0: (100, 0)})
        states = {0: _agent(agent_id=0, x=0, y=0, vx=0, vy=0, max_speed=1.0)}
        actions = ctrl.step(0, 0.0, 1.0, states, robot_id=-1, emit_event=lambda *a: None)
        speed = math.hypot(actions[0].pref_vx, actions[0].pref_vy)
        assert speed <= 1.0 + 1e-6
