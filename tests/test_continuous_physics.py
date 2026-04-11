"""Tests for navirl.backends.continuous.physics — physics engine, integration, forces, collisions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.backends.continuous.obstacles import CircleObstacle, ObstacleCollection
from navirl.backends.continuous.physics import (
    AgentState,
    IntegrationMethod,
    PhysicsConfig,
    PhysicsEngine,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _agent(px=0.0, py=0.0, vx=0.0, vy=0.0, heading=0.0, radius=0.3, mass=80.0, **kwargs):
    return AgentState(
        position=np.array([px, py]),
        velocity=np.array([vx, vy]),
        heading=heading,
        radius=radius,
        mass=mass,
        **kwargs,
    )


def _engine(**kwargs):
    return PhysicsEngine(config=PhysicsConfig(**kwargs))


# ===========================================================================
#  AgentState
# ===========================================================================


class TestAgentState:
    def test_speed_stationary(self):
        a = _agent()
        assert a.speed == pytest.approx(0.0)

    def test_speed_moving(self):
        a = _agent(vx=3.0, vy=4.0)
        assert a.speed == pytest.approx(5.0)

    def test_copy_independence(self):
        a = _agent(px=1.0, py=2.0, vx=3.0, vy=4.0)
        c = a.copy()
        c.position[0] = 99.0
        c.velocity[1] = 99.0
        assert a.position[0] == pytest.approx(1.0)
        assert a.velocity[1] == pytest.approx(4.0)

    def test_copy_preserves_scalars(self):
        a = _agent(heading=1.5, radius=0.5, mass=70.0)
        c = a.copy()
        assert c.heading == pytest.approx(1.5)
        assert c.radius == pytest.approx(0.5)
        assert c.mass == pytest.approx(70.0)

    def test_post_init_coercion(self):
        a = AgentState(position=[1, 2], velocity=[3, 4])
        assert a.position.dtype == np.float64
        assert a.velocity.dtype == np.float64

    def test_defaults(self):
        a = AgentState()
        np.testing.assert_array_equal(a.position, [0.0, 0.0])
        np.testing.assert_array_equal(a.velocity, [0.0, 0.0])
        assert a.heading == 0.0


# ===========================================================================
#  PhysicsConfig
# ===========================================================================


class TestPhysicsConfig:
    def test_defaults(self):
        c = PhysicsConfig()
        assert c.integration_method == IntegrationMethod.SEMI_IMPLICIT_EULER
        assert c.damping == pytest.approx(0.1)
        assert c.restitution == pytest.approx(0.5)


# ===========================================================================
#  Integration methods
# ===========================================================================


class TestIntegration:
    """Test each integration method in isolation."""

    @pytest.fixture
    def engine_euler(self):
        return _engine(integration_method=IntegrationMethod.EULER, damping=0.0)

    @pytest.fixture
    def engine_semi_implicit(self):
        return _engine(integration_method=IntegrationMethod.SEMI_IMPLICIT_EULER, damping=0.0)

    @pytest.fixture
    def engine_verlet(self):
        return _engine(integration_method=IntegrationMethod.VELOCITY_VERLET, damping=0.0)

    @pytest.fixture
    def engine_rk4(self):
        return _engine(integration_method=IntegrationMethod.RK4, damping=0.0)

    def test_euler_constant_velocity(self, engine_euler):
        agents = {0: _agent(px=0.0, py=0.0, vx=1.0, vy=0.0)}
        actions = {0: np.array([1.0, 0.0])}
        result = engine_euler.step(agents, actions, dt=1.0)
        # With action == current velocity, no acceleration → position += velocity * dt
        assert result[0].position[0] == pytest.approx(1.0, abs=0.1)

    def test_semi_implicit_constant_velocity(self, engine_semi_implicit):
        agents = {0: _agent(px=0.0, py=0.0, vx=1.0, vy=0.0)}
        actions = {0: np.array([1.0, 0.0])}
        result = engine_semi_implicit.step(agents, actions, dt=1.0)
        assert result[0].position[0] == pytest.approx(1.0, abs=0.1)

    def test_verlet_constant_velocity(self, engine_verlet):
        agents = {0: _agent(px=0.0, py=0.0, vx=1.0, vy=0.0)}
        actions = {0: np.array([1.0, 0.0])}
        result = engine_verlet.step(agents, actions, dt=1.0)
        assert result[0].position[0] == pytest.approx(1.0, abs=0.1)

    def test_rk4_constant_velocity(self, engine_rk4):
        agents = {0: _agent(px=0.0, py=0.0, vx=1.0, vy=0.0)}
        actions = {0: np.array([1.0, 0.0])}
        result = engine_rk4.step(agents, actions, dt=1.0)
        assert result[0].position[0] == pytest.approx(1.0, abs=0.1)

    def test_euler_acceleration(self, engine_euler):
        """Agent starts at rest, action pushes to desired velocity."""
        agents = {0: _agent(px=0.0, py=0.0, vx=0.0, vy=0.0)}
        actions = {0: np.array([1.0, 0.0])}
        result = engine_euler.step(agents, actions, dt=0.1)
        # Should start moving
        assert result[0].velocity[0] > 0

    def test_all_methods_conserve_agent_count(self):
        for method in IntegrationMethod:
            engine = _engine(integration_method=method, damping=0.0)
            agents = {0: _agent(), 1: _agent(px=5.0)}
            actions = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
            result = engine.step(agents, actions, dt=0.1)
            assert len(result) == 2, f"Method {method} lost agents"


# ===========================================================================
#  Force computation
# ===========================================================================


class TestForces:
    def test_damping_force(self):
        engine = _engine(damping=1.0, friction_coefficient=0.0)
        agents = {0: _agent(vx=1.0)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        # Damping should oppose velocity
        assert forces[0][0] < 0

    def test_no_damping_when_zero(self):
        engine = _engine(damping=0.0, friction_coefficient=0.0)
        agents = {0: _agent(vx=1.0)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        # No damping and no friction → forces only from repulsion/boundary
        # No other agents or obstacles → zero
        np.testing.assert_allclose(forces[0], [0.0, 0.0])

    def test_friction_force(self):
        engine = _engine(damping=0.0, friction_coefficient=0.5)
        agents = {0: _agent(vx=2.0)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        assert forces[0][0] < 0  # friction opposes motion

    def test_friction_zero_speed(self):
        engine = _engine(damping=0.0, friction_coefficient=0.5)
        agents = {0: _agent(vx=0.0, vy=0.0)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        np.testing.assert_allclose(forces[0], [0.0, 0.0])

    def test_agent_repulsion_force(self):
        engine = _engine(damping=0.0, friction_coefficient=0.0)
        # Two agents overlapping
        agents = {
            0: _agent(px=0.0, radius=0.3),
            1: _agent(px=0.4, radius=0.3),
        }
        forces = engine._compute_forces(agents, {0: np.zeros(2), 1: np.zeros(2)}, dt=0.1)
        # Agent 0 should be pushed left, agent 1 pushed right
        assert forces[0][0] < 0
        assert forces[1][0] > 0

    def test_obstacle_repulsion_force(self):
        obs_col = ObstacleCollection()
        obs_col.add(CircleObstacle(center=np.array([1.0, 0.0]), radius=0.5))
        engine = PhysicsEngine(
            config=PhysicsConfig(damping=0.0, friction_coefficient=0.0),
            obstacles=obs_col,
        )
        # Agent very close to obstacle
        agents = {0: _agent(px=0.2, radius=0.3)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        # Should be pushed away from the obstacle (leftward)
        assert forces[0][0] < 0

    def test_boundary_force(self):
        engine = PhysicsEngine(
            config=PhysicsConfig(damping=0.0, friction_coefficient=0.0),
            world_bounds=(0.0, 0.0, 10.0, 10.0),
        )
        # Agent near left boundary
        agents = {0: _agent(px=0.1, py=5.0, radius=0.3)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        assert forces[0][0] > 0  # pushed right, away from left wall

    def test_no_boundary_force_without_bounds(self):
        engine = _engine(damping=0.0, friction_coefficient=0.0)
        agents = {0: _agent(px=-100.0)}
        forces = engine._compute_forces(agents, {0: np.zeros(2)}, dt=0.1)
        np.testing.assert_allclose(forces[0], [0.0, 0.0])


# ===========================================================================
#  Collision resolution
# ===========================================================================


class TestCollisionResolution:
    def test_agent_collision_pushes_apart(self):
        engine = _engine(damping=0.0, enable_collision_response=True)
        # Overlapping agents
        agents = {
            0: _agent(px=0.0, radius=0.5),
            1: _agent(px=0.5, radius=0.5),
        }
        actions = {0: np.zeros(2), 1: np.zeros(2)}
        result = engine.step(agents, actions, dt=0.01)
        # After step, they should be further apart
        sep = np.linalg.norm(result[0].position - result[1].position)
        orig_sep = 0.5
        assert sep > orig_sep

    def test_obstacle_collision_resolution(self):
        obs_col = ObstacleCollection()
        obs_col.add(CircleObstacle(center=np.array([1.0, 0.0]), radius=0.5))
        engine = PhysicsEngine(
            config=PhysicsConfig(damping=0.0, enable_collision_response=True),
            obstacles=obs_col,
        )
        # Agent overlapping with obstacle
        agents = {0: _agent(px=0.3, radius=0.3, vx=1.0)}
        actions = {0: np.zeros(2)}
        result = engine.step(agents, actions, dt=0.01)
        # Agent should be pushed away from obstacle
        assert result[0].position[0] < 0.3  # pushed leftward

    def test_boundary_collision_resolution(self):
        engine = PhysicsEngine(
            config=PhysicsConfig(damping=0.0, restitution=0.5, enable_collision_response=True),
            world_bounds=(0.0, 0.0, 10.0, 10.0),
        )
        # Agent past left boundary
        agents = {0: _agent(px=-0.5, py=5.0, vx=-1.0, radius=0.3)}
        actions = {0: np.zeros(2)}
        result = engine.step(agents, actions, dt=0.01)
        # Should be clamped to boundary
        assert result[0].position[0] >= 0.3  # radius from boundary

    def test_boundary_collision_restitution(self):
        engine = PhysicsEngine(
            config=PhysicsConfig(
                damping=0.0,
                restitution=1.0,
                enable_collision_response=True,
                integration_method=IntegrationMethod.EULER,
            ),
            world_bounds=(0.0, 0.0, 10.0, 10.0),
        )
        # Agent hitting left wall
        agents = {0: _agent(px=0.1, py=5.0, vx=-2.0, radius=0.3)}
        actions = {0: np.array([-2.0, 0.0])}
        result = engine.step(agents, actions, dt=0.01)
        # Velocity should have been reflected (or at least not be negative anymore after resolution)
        assert result[0].velocity[0] >= 0


# ===========================================================================
#  Speed limiting
# ===========================================================================


class TestSpeedLimiting:
    def test_speed_clamped(self):
        engine = _engine(damping=0.0)
        agents = {0: _agent(vx=0.0, max_speed=1.0)}
        actions = {0: np.array([100.0, 0.0])}  # huge desired velocity
        result = engine.step(agents, actions, dt=1.0)
        assert result[0].speed <= 1.0 + 1e-6


# ===========================================================================
#  Action to acceleration
# ===========================================================================


class TestActionToAcceleration:
    def test_zero_dt(self):
        engine = _engine()
        state = _agent(vx=1.0)
        acc = engine._action_to_acceleration(state, np.array([2.0, 0.0]), dt=0.0)
        np.testing.assert_array_equal(acc, [0.0, 0.0])

    def test_desired_speed_clamped(self):
        engine = _engine()
        state = _agent(max_speed=1.0)
        acc = engine._action_to_acceleration(state, np.array([100.0, 0.0]), dt=1.0)
        # The desired velocity should be clamped to max_speed before computing acceleration
        # acc = (clamped_desired - current_vel) / dt
        assert acc[0] == pytest.approx(1.0)

    def test_match_current_velocity(self):
        engine = _engine()
        state = _agent(vx=1.0)
        acc = engine._action_to_acceleration(state, np.array([1.0, 0.0]), dt=1.0)
        np.testing.assert_allclose(acc, [0.0, 0.0], atol=1e-12)


# ===========================================================================
#  Heading update
# ===========================================================================


class TestHeadingUpdate:
    def test_stationary_preserves_heading(self):
        engine = _engine()
        state = _agent(heading=1.0)
        h = engine._update_heading(state, np.array([0.0, 0.0]), dt=0.1)
        assert h == pytest.approx(1.0)

    def test_heading_follows_velocity(self):
        engine = _engine()
        state = _agent(heading=0.0)
        h = engine._update_heading(state, np.array([0.0, 1.0]), dt=10.0)
        # With long dt, heading should reach pi/2
        assert h == pytest.approx(math.pi / 2, abs=0.1)

    def test_heading_wraps(self):
        engine = _engine()
        state = _agent(heading=math.pi - 0.1, max_angular_velocity=100.0)
        h = engine._update_heading(state, np.array([-1.0, 0.1]), dt=1.0)
        # Should be near pi
        assert -math.pi <= h <= math.pi

    def test_angular_velocity_clamping(self):
        engine = _engine()
        state = _agent(heading=0.0, max_angular_velocity=0.1)
        h = engine._update_heading(state, np.array([0.0, 1.0]), dt=0.01)
        # Max change = 0.1 * 0.01 = 0.001
        assert abs(h - 0.0) <= 0.1 * 0.01 + 1e-12


# ===========================================================================
#  Utility methods
# ===========================================================================


class TestUtility:
    def test_get_collision_pairs_empty(self):
        engine = _engine()
        assert engine.get_collision_pairs() == []

    def test_get_collision_pairs_after_step(self):
        engine = _engine(damping=0.0)
        agents = {
            0: _agent(px=0.0, radius=0.3),
            1: _agent(px=0.4, radius=0.3),
        }
        engine.step(agents, {0: np.zeros(2), 1: np.zeros(2)}, dt=0.1)
        pairs = engine.get_collision_pairs()
        assert len(pairs) == 1
        assert set(pairs[0]) == {0, 1}

    def test_check_line_of_sight_clear(self):
        engine = _engine()
        assert engine.check_line_of_sight(np.array([0.0, 0.0]), np.array([5.0, 0.0]))

    def test_check_line_of_sight_blocked(self):
        obs_col = ObstacleCollection()
        obs_col.add(CircleObstacle(center=np.array([2.5, 0.0]), radius=1.0))
        engine = PhysicsEngine(obstacles=obs_col)
        assert not engine.check_line_of_sight(np.array([0.0, 0.0]), np.array([5.0, 0.0]))

    def test_check_line_of_sight_same_point(self):
        engine = _engine()
        assert engine.check_line_of_sight(np.array([1.0, 1.0]), np.array([1.0, 1.0]))

    def test_compute_energy_empty(self):
        engine = _engine()
        e = engine.compute_energy({})
        assert e["total"] == 0.0

    def test_compute_energy_stationary(self):
        engine = _engine()
        agents = {0: _agent(vx=0.0, vy=0.0, mass=80.0)}
        e = engine.compute_energy(agents)
        assert e["total"] == pytest.approx(0.0)

    def test_compute_energy_moving(self):
        engine = _engine()
        agents = {
            0: _agent(vx=1.0, vy=0.0, mass=80.0),
            1: _agent(vx=0.0, vy=2.0, mass=60.0),
        }
        e = engine.compute_energy(agents)
        ke_0 = 0.5 * 80.0 * 1.0
        ke_1 = 0.5 * 60.0 * 4.0
        assert e["total"] == pytest.approx(ke_0 + ke_1)
        assert e["mean"] == pytest.approx((ke_0 + ke_1) / 2)
        assert e["max"] == pytest.approx(max(ke_0, ke_1))
        assert e["min"] == pytest.approx(min(ke_0, ke_1))


# ===========================================================================
#  Full step integration tests
# ===========================================================================


class TestFullStep:
    def test_step_no_agents(self):
        engine = _engine()
        result = engine.step({}, {}, dt=0.1)
        assert result == {}

    def test_step_preserves_state_structure(self):
        engine = _engine()
        agents = {0: _agent(px=1.0, py=2.0, vx=0.5, vy=-0.5)}
        result = engine.step(agents, {0: np.array([0.5, -0.5])}, dt=0.1)
        assert 0 in result
        assert isinstance(result[0], AgentState)

    def test_step_does_not_mutate_input(self):
        engine = _engine()
        a = _agent(px=1.0, py=2.0, vx=1.0)
        agents = {0: a}
        original_pos = a.position.copy()
        engine.step(agents, {0: np.array([1.0, 0.0])}, dt=0.1)
        np.testing.assert_array_equal(a.position, original_pos)

    def test_step_with_collision_disabled(self):
        engine = _engine(enable_collision_response=False, damping=0.0)
        agents = {
            0: _agent(px=0.0, radius=0.5),
            1: _agent(px=0.2, radius=0.5),
        }
        actions = {0: np.zeros(2), 1: np.zeros(2)}
        result = engine.step(agents, actions, dt=0.01)
        # With collisions disabled, agents can overlap
        assert len(result) == 2

    def test_multiple_steps_motion(self):
        engine = _engine(damping=0.0, friction_coefficient=0.0)
        agents = {0: _agent(vx=1.0, max_speed=2.0)}
        for _ in range(10):
            agents = engine.step(agents, {0: np.array([1.0, 0.0])}, dt=0.1)
        # After 10 steps of 0.1s at ~1 m/s, should have moved ~1m
        assert agents[0].position[0] > 0.5
