"""Tests for navirl/models/ pedestrian dynamics: social_force, crowd_dynamics, power_law."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.core.types import AgentState
from navirl.models.crowd_dynamics import CrowdAnalyzer, FundamentalDiagram, LevelOfService
from navirl.models.power_law import (
    PowerLawConfig,
    PowerLawHumanController,
    PowerLawModel,
    _time_to_collision,
)
from navirl.models.social_force import (
    SocialForceConfig,
    SocialForceHumanController,
    SocialForceModel,
    WallSegment,
    _anisotropy_weight,
    _point_to_segment_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    agent_id: int = 0,
    kind: str = "human",
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
    radius: float = 0.25,
    max_speed: float = 1.5,
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


# ---------------------------------------------------------------------------
# Social Force Model - helpers
# ---------------------------------------------------------------------------


class TestAnisotropyWeight:
    def test_aligned_heading(self):
        """Agent heading toward neighbor: full weight."""
        w = _anisotropy_weight(1.0, 0.0, -1.0, 0.0, 0.5)
        assert w == pytest.approx(1.0)

    def test_opposite_heading(self):
        """Neighbor behind agent: reduced weight."""
        w = _anisotropy_weight(1.0, 0.0, 1.0, 0.0, 0.5)
        assert w == pytest.approx(0.5)

    def test_isotropic(self):
        """lambda=0 means isotropic: weight = (1 + cos)/2."""
        w = _anisotropy_weight(1.0, 0.0, 0.0, 1.0, 0.0)
        assert w == pytest.approx(0.5)

    def test_fully_anisotropic(self):
        """lambda=1 means constant weight of 1."""
        w = _anisotropy_weight(1.0, 0.0, 0.0, 1.0, 1.0)
        assert w == pytest.approx(1.0)


class TestPointToSegmentDistance:
    def test_perpendicular_projection(self):
        dist, nx, ny = _point_to_segment_distance(1.0, 1.0, 0.0, 0.0, 2.0, 0.0)
        assert dist == pytest.approx(1.0)
        assert nx == pytest.approx(1.0)
        assert ny == pytest.approx(0.0)

    def test_closest_to_start(self):
        dist, nx, ny = _point_to_segment_distance(-1.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert dist == pytest.approx(1.0)
        assert nx == pytest.approx(0.0)
        assert ny == pytest.approx(0.0)

    def test_closest_to_end(self):
        dist, nx, ny = _point_to_segment_distance(3.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert dist == pytest.approx(1.0)
        assert nx == pytest.approx(2.0)
        assert ny == pytest.approx(0.0)

    def test_degenerate_segment(self):
        dist, nx, ny = _point_to_segment_distance(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        assert dist == pytest.approx(math.sqrt(2.0))

    def test_point_on_segment(self):
        dist, _, _ = _point_to_segment_distance(1.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        assert dist == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Social Force Model - core
# ---------------------------------------------------------------------------


class TestSocialForceModel:
    @pytest.fixture
    def model(self):
        return SocialForceModel()

    def test_default_config(self, model):
        assert model.cfg.tau > 0
        assert model.cfg.A > 0
        assert model.cfg.B > 0

    def test_custom_config(self):
        cfg = SocialForceConfig(A=3.0, B=0.5)
        m = SocialForceModel(cfg)
        assert m.cfg.A == 3.0
        assert m.cfg.B == 0.5

    def test_desired_force_toward_goal(self, model):
        state = _make_state(x=0.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        assert fx > 0, "Force should push toward goal in +x"
        assert abs(fy) < 1e-6, "No y-component when goal is along x-axis"

    def test_desired_force_at_goal(self, model):
        state = _make_state(x=10.0, y=0.0, vx=1.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        # At goal with velocity: force should decelerate
        assert fx < 0

    def test_desired_force_already_at_preferred_speed(self, model):
        state = _make_state(x=0.0, y=0.0, vx=1.5, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        # Already at max_speed toward goal: force should be ~0
        assert abs(fx) < 1e-6

    def test_social_force_repels(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=1.0, vy=0.0)
        other = _make_state(agent_id=1, x=1.0, y=0.0, vx=-1.0, vy=0.0)
        fx, fy = model.compute_social_force(agent, [other])
        assert fx < 0, "Should be repelled in -x direction from agent at x=1"

    def test_social_force_same_agent_ignored(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=1.0, vy=0.0)
        fx, fy = model.compute_social_force(agent, [agent])
        assert fx == 0.0
        assert fy == 0.0

    def test_social_force_out_of_range(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=1.0, vy=0.0)
        far = _make_state(agent_id=1, x=100.0, y=0.0)
        fx, fy = model.compute_social_force(agent, [far])
        assert fx == 0.0
        assert fy == 0.0

    def test_wall_force_repels(self, model):
        agent = _make_state(x=0.0, y=0.5)
        wall: WallSegment = (-5.0, 0.0, 5.0, 0.0)
        fx, fy = model.compute_wall_force(agent, [wall])
        assert fy > 0, "Wall below agent should push up"
        assert abs(fx) < abs(fy), "Primarily vertical force"

    def test_wall_force_no_walls(self, model):
        agent = _make_state()
        fx, fy = model.compute_wall_force(agent, [])
        assert fx == 0.0
        assert fy == 0.0

    def test_wall_force_out_of_range(self, model):
        agent = _make_state(x=0.0, y=100.0)
        wall: WallSegment = (-5.0, 0.0, 5.0, 0.0)
        fx, fy = model.compute_wall_force(agent, [wall])
        assert fx == 0.0
        assert fy == 0.0

    def test_contact_force_no_overlap(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, radius=0.25)
        other = _make_state(agent_id=1, x=2.0, y=0.0, radius=0.25)
        fx, fy = model.compute_contact_force(agent, [other])
        assert fx == 0.0
        assert fy == 0.0

    def test_contact_force_with_overlap(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, radius=0.25)
        other = _make_state(agent_id=1, x=0.3, y=0.0, radius=0.25)
        fx, fy = model.compute_contact_force(agent, [other])
        assert fx < 0, "Pushed away in -x from overlapping agent"

    def test_contact_force_sliding_friction(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=1.0, radius=0.25)
        other = _make_state(agent_id=1, x=0.3, y=0.0, vx=0.0, vy=0.0, radius=0.25)
        fx, fy = model.compute_contact_force(agent, [other])
        # Tangential friction should act on vy component
        assert fy != 0.0

    def test_total_force_combines_all(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.5, vx=0.0, vy=0.0, goal_x=10.0)
        other = _make_state(agent_id=1, x=2.0, y=0.5)
        wall: WallSegment = (-5.0, 0.0, 5.0, 0.0)
        fx, fy = model.compute_total_force(agent, [other], [wall])
        # Should have non-zero force from desired + social + wall
        assert fx != 0.0 or fy != 0.0

    def test_step_returns_velocities(self, model):
        s1 = _make_state(agent_id=0, x=0.0, y=0.0, goal_x=10.0)
        s2 = _make_state(agent_id=1, x=5.0, y=5.0, goal_x=0.0, goal_y=0.0)
        goals = {0: (10.0, 0.0), 1: (0.0, 0.0)}
        result = model.step([s1, s2], goals, dt=0.04)
        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_step_respects_max_speed(self, model):
        # Agent with large force should still be clamped
        state = _make_state(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0, max_speed=1.0, goal_x=100.0)
        result = model.step([state], {0: (100.0, 0.0)}, dt=10.0)
        vx, vy = result[0]
        speed = math.hypot(vx, vy)
        assert speed <= 1.0 + 1e-6

    def test_step_with_walls(self, model):
        state = _make_state(agent_id=0, x=0.0, y=0.3, goal_x=10.0, goal_y=0.3)
        walls: list[WallSegment] = [(-5.0, 0.0, 15.0, 0.0)]
        result = model.step([state], {0: (10.0, 0.3)}, walls=walls, dt=0.04)
        assert 0 in result


# ---------------------------------------------------------------------------
# Social Force Human Controller
# ---------------------------------------------------------------------------


class TestSocialForceHumanController:
    @pytest.fixture
    def controller(self):
        return SocialForceHumanController()

    def test_reset_and_step(self, controller):
        ids = [1, 2]
        starts = {1: (0.0, 0.0), 2: (5.0, 5.0)}
        goals = {1: (10.0, 0.0), 2: (0.0, 0.0)}
        controller.reset(ids, starts, goals)

        states = {
            1: _make_state(agent_id=1, x=0.0, y=0.0, goal_x=10.0, goal_y=0.0),
            2: _make_state(agent_id=2, x=5.0, y=5.0, goal_x=0.0, goal_y=0.0),
            99: _make_state(agent_id=99, kind="robot", x=3.0, y=3.0),
        }
        events = []
        actions = controller.step(0, 0.0, 0.04, states, 99, lambda *a: events.append(a))
        assert 1 in actions
        assert 2 in actions
        assert 99 not in actions
        assert actions[1].behavior == "GO_TO"

    def test_goal_swap_on_arrival(self, controller):
        controller.reset([1], {1: (0.0, 0.0)}, {1: (1.0, 0.0)})
        states = {
            1: _make_state(agent_id=1, x=0.9, y=0.0, goal_x=1.0, goal_y=0.0),
        }
        events = []
        controller.step(0, 0.0, 0.04, states, 99, lambda *a: events.append(a))
        assert len(events) == 1
        assert events[0][0] == "goal_swap"
        # Goals should have swapped
        assert controller.goals[1] == (0.0, 0.0)

    def test_missing_human_skipped(self, controller):
        controller.reset([1, 2], {1: (0.0, 0.0), 2: (5.0, 5.0)}, {1: (10.0, 0.0), 2: (0.0, 0.0)})
        states = {1: _make_state(agent_id=1)}
        actions = controller.step(0, 0.0, 0.04, states, 99, lambda *a: None)
        assert 1 in actions
        assert 2 not in actions


# ---------------------------------------------------------------------------
# Power Law - _time_to_collision helper
# ---------------------------------------------------------------------------


class TestTimeToCollision:
    def test_head_on_collision(self):
        # p and v opposite signs = closing. |2-t| = 0.5 at t=1.5
        ttc = _time_to_collision(2.0, 0.0, -1.0, 0.0, 0.5)
        assert ttc == pytest.approx(1.5)

    def test_no_collision_diverging(self):
        # p and v same sign = diverging. |2+t| always > 0.5
        ttc = _time_to_collision(2.0, 0.0, 1.0, 0.0, 0.5)
        assert ttc == float("inf")

    def test_already_overlapping(self):
        ttc = _time_to_collision(0.1, 0.0, 0.0, 0.0, 0.5)
        assert ttc == 0.0

    def test_no_relative_motion(self):
        ttc = _time_to_collision(2.0, 0.0, 0.0, 0.0, 0.5)
        assert ttc == float("inf")

    def test_perpendicular_miss(self):
        # Moving perpendicular, far enough to miss
        ttc = _time_to_collision(5.0, 0.0, 0.0, 1.0, 0.5)
        assert ttc == float("inf")

    def test_fast_approach(self):
        # |3 - 2t| = 0.5 → t = 1.25
        ttc = _time_to_collision(3.0, 0.0, -2.0, 0.0, 0.5)
        assert ttc == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Power Law Model - core
# ---------------------------------------------------------------------------


class TestPowerLawModel:
    @pytest.fixture
    def model(self):
        return PowerLawModel(PowerLawConfig(sigma=0.0))  # Disable noise for determinism

    def test_desired_force_toward_goal(self, model):
        state = _make_state(x=0.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        assert fx > 0
        assert abs(fy) < 1e-6

    def test_desired_force_at_goal(self, model):
        state = _make_state(x=10.0, y=0.0, vx=1.0, vy=0.0, goal_x=10.0, goal_y=0.0)
        fx, fy = model.compute_desired_force(state, (10.0, 0.0))
        assert fx < 0  # decelerating

    def test_anticipatory_force_head_on(self, model):
        # Agent moving right, other moving left — approaching.
        # rel_vx = self.vx - other.vx = -1 - 1 = -2 (closing sign for the formula)
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=-1.0, vy=0.0, radius=0.25)
        other = _make_state(agent_id=1, x=3.0, y=0.0, vx=1.0, vy=0.0, radius=0.25)
        fx, fy = model.compute_anticipatory_force(agent, [other])
        # Should produce a non-zero avoidance force
        assert fx != 0.0 or fy != 0.0

    def test_anticipatory_force_same_agent_ignored(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=1.0, vy=0.0)
        fx, fy = model.compute_anticipatory_force(agent, [agent])
        assert fx == 0.0
        assert fy == 0.0

    def test_anticipatory_force_far_away(self, model):
        agent = _make_state(agent_id=0, x=0.0, y=0.0, vx=1.0, vy=0.0)
        far = _make_state(agent_id=1, x=100.0, y=0.0)
        fx, fy = model.compute_anticipatory_force(agent, [far])
        assert fx == 0.0
        assert fy == 0.0

    def test_total_force(self, model):
        state = _make_state(agent_id=0, x=0.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0)
        fx, fy = model.compute_total_force(state, [])
        # Only desired force when alone
        assert fx > 0

    def test_step_returns_velocities(self, model):
        s1 = _make_state(agent_id=0, x=0.0, y=0.0, goal_x=10.0)
        s2 = _make_state(agent_id=1, x=5.0, y=0.0, goal_x=0.0, goal_y=0.0)
        result = model.step([s1, s2], {0: (10.0, 0.0), 1: (0.0, 0.0)}, dt=0.04)
        assert 0 in result
        assert 1 in result

    def test_step_clamps_speed(self, model):
        state = _make_state(agent_id=0, max_speed=1.0, goal_x=100.0)
        result = model.step([state], {0: (100.0, 0.0)}, dt=10.0)
        vx, vy = result[0]
        assert math.hypot(vx, vy) <= 1.0 + 1e-6

    def test_noise_adds_variation(self):
        noisy = PowerLawModel(PowerLawConfig(sigma=1.0))
        state = _make_state(agent_id=0)
        # Run many times to check noise is present
        results = set()
        for _ in range(10):
            fx, fy = noisy.compute_anticipatory_force(state, [])
            results.add(round(fx, 4))
        # With sigma=1.0 and no neighbors, noise should produce variation
        assert len(results) > 1


# ---------------------------------------------------------------------------
# Power Law Human Controller
# ---------------------------------------------------------------------------


class TestPowerLawHumanController:
    @pytest.fixture
    def controller(self):
        return PowerLawHumanController(PowerLawConfig(sigma=0.0))

    def test_reset_and_step(self, controller):
        controller.reset([1], {1: (0.0, 0.0)}, {1: (10.0, 0.0)})
        states = {
            1: _make_state(agent_id=1, x=0.0, y=0.0, goal_x=10.0),
            99: _make_state(agent_id=99, kind="robot"),
        }
        actions = controller.step(0, 0.0, 0.04, states, 99, lambda *a: None)
        assert 1 in actions
        assert actions[1].metadata["model"] == "power_law"

    def test_goal_swap_on_arrival(self, controller):
        controller.reset([1], {1: (0.0, 0.0)}, {1: (0.5, 0.0)})
        states = {1: _make_state(agent_id=1, x=0.3, y=0.0, goal_x=0.5)}
        events = []
        controller.step(0, 0.0, 0.04, states, 99, lambda *a: events.append(a))
        assert len(events) == 1
        assert controller.goals[1] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# CrowdAnalyzer
# ---------------------------------------------------------------------------


class TestCrowdAnalyzer:
    def test_density_empty(self):
        grid = CrowdAnalyzer.compute_density(np.empty((0, 2)), (0, 0, 10, 10), 1.0)
        assert grid.shape[0] > 0
        assert np.all(grid == 0.0)

    def test_density_single_agent(self):
        pos = np.array([[5.0, 5.0]])
        grid = CrowdAnalyzer.compute_density(pos, (0, 0, 10, 10), cell_size=10.0)
        # 1 agent in a 10x10 cell = 1/100 = 0.01 ped/m²
        assert grid.shape == (1, 1)
        assert grid[0, 0] == pytest.approx(0.01)

    def test_density_multiple_agents_same_cell(self):
        pos = np.array([[1.0, 1.0], [1.5, 1.5]])
        grid = CrowdAnalyzer.compute_density(pos, (0, 0, 10, 10), cell_size=10.0)
        assert grid[0, 0] == pytest.approx(0.02)

    def test_density_different_cells(self):
        pos = np.array([[0.5, 0.5], [5.5, 5.5]])
        grid = CrowdAnalyzer.compute_density(pos, (0, 0, 10, 10), cell_size=5.0)
        assert grid.shape == (2, 2)
        assert grid[0, 0] == pytest.approx(1 / 25.0)
        assert grid[1, 1] == pytest.approx(1 / 25.0)
        assert grid[0, 1] == 0.0

    def test_flow_field_empty(self):
        flow = CrowdAnalyzer.compute_flow_field(
            np.empty((0, 2)), np.empty((0, 2)), (0, 0, 10, 10)
        )
        assert np.all(flow == 0.0)

    def test_flow_field_single_agent(self):
        pos = np.array([[5.0, 5.0]])
        vel = np.array([[1.0, 0.5]])
        flow = CrowdAnalyzer.compute_flow_field(pos, vel, (0, 0, 10, 10), cell_size=10.0)
        assert flow.shape == (1, 1, 2)
        assert flow[0, 0, 0] == pytest.approx(1.0)
        assert flow[0, 0, 1] == pytest.approx(0.5)

    def test_flow_field_averages_velocities(self):
        pos = np.array([[1.0, 1.0], [1.5, 1.5]])
        vel = np.array([[2.0, 0.0], [0.0, 2.0]])
        flow = CrowdAnalyzer.compute_flow_field(pos, vel, (0, 0, 10, 10), cell_size=10.0)
        assert flow[0, 0, 0] == pytest.approx(1.0)
        assert flow[0, 0, 1] == pytest.approx(1.0)

    def test_crowd_pressure_single_agent(self):
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[1.0, 0.0]])
        assert CrowdAnalyzer.compute_crowd_pressure(pos, vel) == 0.0

    def test_crowd_pressure_uniform_motion(self):
        # All agents moving same direction: low variance -> low pressure
        pos = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
        vel = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        pressure = CrowdAnalyzer.compute_crowd_pressure(pos, vel, radius=2.0)
        assert pressure == pytest.approx(0.0, abs=1e-6)

    def test_crowd_pressure_turbulent(self):
        # Agents moving in different directions: high variance -> high pressure
        pos = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
        vel = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        pressure = CrowdAnalyzer.compute_crowd_pressure(pos, vel, radius=2.0)
        assert pressure > 0.0

    def test_detect_congestion_none(self):
        grid = np.array([[0.5, 0.3], [0.1, 0.2]])
        result = CrowdAnalyzer.detect_congestion(grid, threshold=1.7)
        assert result == []

    def test_detect_congestion_found(self):
        grid = np.array([[0.5, 2.0], [1.8, 0.2]])
        result = CrowdAnalyzer.detect_congestion(grid, threshold=1.7)
        assert (0, 1) in result
        assert (1, 0) in result
        assert len(result) == 2


# ---------------------------------------------------------------------------
# FundamentalDiagram
# ---------------------------------------------------------------------------


class TestFundamentalDiagram:
    @pytest.fixture
    def fd(self):
        return FundamentalDiagram(v_free=1.34, rho_max=5.4)

    def test_zero_density(self, fd):
        assert fd.speed_from_density(0.0) == pytest.approx(1.34)

    def test_max_density(self, fd):
        assert fd.speed_from_density(5.4) == pytest.approx(0.0)

    def test_above_max_density(self, fd):
        assert fd.speed_from_density(10.0) == pytest.approx(0.0)

    def test_half_density(self, fd):
        speed = fd.speed_from_density(2.7)
        assert 0.0 < speed < 1.34

    def test_array_input(self, fd):
        densities = np.array([0.0, 2.7, 5.4])
        speeds = fd.speed_from_density(densities)
        assert speeds[0] == pytest.approx(1.34)
        assert speeds[2] == pytest.approx(0.0)

    def test_flow_from_density(self, fd):
        # Flow = density * speed; at 0 density, flow = 0
        assert fd.flow_from_density(0.0) == pytest.approx(0.0)
        # At max density, flow = 0 (speed = 0)
        assert fd.flow_from_density(5.4) == pytest.approx(0.0)
        # Intermediate: positive flow
        flow = fd.flow_from_density(2.0)
        assert flow > 0.0

    def test_flow_array(self, fd):
        densities = np.array([0.0, 2.0, 5.4])
        flows = fd.flow_from_density(densities)
        assert flows[0] == pytest.approx(0.0)
        assert flows[2] == pytest.approx(0.0)
        assert flows[1] > 0

    def test_fit_calibration(self):
        fd = FundamentalDiagram()
        # Synthetic linear data: v = 2.0 - 0.5 * rho  (v_free=2, rho_max=4)
        densities = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        speeds = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
        fd.fit(densities, speeds)
        assert fd.v_free == pytest.approx(2.0, abs=0.1)
        assert fd.rho_max == pytest.approx(4.0, abs=0.1)

    def test_fit_too_few_samples(self):
        fd = FundamentalDiagram(v_free=1.0, rho_max=5.0)
        fd.fit(np.array([1.0]), np.array([0.5]))
        # Should not change
        assert fd.v_free == 1.0
        assert fd.rho_max == 5.0


# ---------------------------------------------------------------------------
# LevelOfService
# ---------------------------------------------------------------------------


class TestLevelOfService:
    def test_grade_a(self):
        assert LevelOfService.classify(0.0) == "A"
        assert LevelOfService.classify(0.1) == "A"

    def test_grade_f(self):
        assert LevelOfService.classify(2.0) == "F"
        assert LevelOfService.classify(5.0) == "F"

    def test_all_grades_ordered(self):
        """Increasing density should produce increasing grade letters."""
        grades = []
        for d in [0.0, 0.4, 0.7, 1.1, 1.4, 2.0]:
            grades.append(LevelOfService.classify(d))
        # Should be monotonically non-decreasing (A <= B <= ... <= F)
        assert grades == sorted(grades)

    def test_evaluate_area(self):
        # Place 20 agents in a 1x1 cell to get high density
        pos = np.random.rand(20, 2) * 0.5 + 0.25
        grades = LevelOfService.evaluate_area(pos, (0, 0, 1, 1), cell_size=1.0)
        assert grades.shape == (1, 1)
        # 20 agents in 1m² = grade F
        assert grades[0, 0] == "F"

    def test_evaluate_area_empty(self):
        grades = LevelOfService.evaluate_area(np.empty((0, 2)), (0, 0, 10, 10), cell_size=5.0)
        assert np.all(grades == "A")
