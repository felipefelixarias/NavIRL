"""Tests for navirl.simulation.physics — integration, motion models, constraints, and engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.simulation.physics import (
    DynamicModel,
    ForceRecord,
    KinematicModel,
    MaterialProperties,
    PhysicsState,
    SimplePhysics,
    VelocityConstraint,
    WallConstraint,
    _euler_step,
    _rk4_step,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _state(px=0.0, py=0.0, vx=0.0, vy=0.0, orient=0.0):
    return PhysicsState(
        position=np.array([px, py]),
        velocity=np.array([vx, vy]),
        orientation=orient,
    )


# ---------------------------------------------------------------------------
#  PhysicsState
# ---------------------------------------------------------------------------


class TestPhysicsState:
    def test_copy_independence(self):
        s = _state(1.0, 2.0, 3.0, 4.0)
        c = s.copy()
        c.position[0] = 99.0
        assert s.position[0] == 1.0, "copy must not alias position"

    def test_defaults(self):
        s = _state()
        np.testing.assert_array_equal(s.acceleration, np.zeros(2))
        assert s.orientation == 0.0
        assert s.angular_velocity == 0.0


# ---------------------------------------------------------------------------
#  MaterialProperties
# ---------------------------------------------------------------------------


class TestMaterialProperties:
    def test_defaults(self):
        m = MaterialProperties()
        assert m.static_friction == 0.6
        assert m.kinetic_friction == 0.4
        assert m.restitution == 0.3
        assert m.drag == 0.0


# ---------------------------------------------------------------------------
#  Integration helpers
# ---------------------------------------------------------------------------


class TestEulerStep:
    def test_stationary(self):
        pos, vel = _euler_step(np.zeros(2), np.zeros(2), np.zeros(2), 0.1)
        np.testing.assert_array_equal(pos, np.zeros(2))
        np.testing.assert_array_equal(vel, np.zeros(2))

    def test_constant_velocity(self):
        pos, vel = _euler_step(np.zeros(2), np.array([1.0, 0.0]), np.zeros(2), 0.5)
        np.testing.assert_allclose(pos, [0.5, 0.0])
        np.testing.assert_allclose(vel, [1.0, 0.0])

    def test_constant_acceleration(self):
        pos, vel = _euler_step(
            np.zeros(2), np.zeros(2), np.array([2.0, 0.0]), 1.0
        )
        # v_new = 0 + 2*1 = 2; x_new = 0 + 2*1 = 2
        np.testing.assert_allclose(vel, [2.0, 0.0])
        np.testing.assert_allclose(pos, [2.0, 0.0])


class TestRK4Step:
    def test_constant_acceleration(self):
        """RK4 with constant acceleration should be exact."""

        def constant_acc(p, v):
            return np.array([1.0, 0.0])

        pos, vel = _rk4_step(np.zeros(2), np.zeros(2), constant_acc, 1.0)
        # Exact: v(1) = 1, x(1) = 0.5
        np.testing.assert_allclose(vel, [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(pos, [0.5, 0.0], atol=1e-12)

    def test_harmonic_oscillator(self):
        """RK4 on a simple harmonic oscillator for one step."""

        def sho_acc(p, v):
            return -p  # F = -x (spring constant k=1, m=1)

        # Start at x=1, v=0 (pure position)
        pos, vel = _rk4_step(np.array([1.0, 0.0]), np.zeros(2), sho_acc, 0.01)
        # After a tiny step the position barely changes
        assert pos[0] < 1.0, "should move toward origin"
        assert vel[0] < 0.0, "velocity should be negative (restoring)"


# ---------------------------------------------------------------------------
#  KinematicModel
# ---------------------------------------------------------------------------


class TestKinematicModel:
    def test_zero_force_deceleration(self):
        """With zero force a moving agent should decelerate toward zero."""
        model = KinematicModel(max_speed=2.0, response_time=0.5)
        s = _state(vx=1.0)
        acc = model.compute_acceleration(s, np.zeros(2), mass=1.0, dt=0.1)
        # desired_vel = 0, acc = (0 - 1)/0.5 = -2
        np.testing.assert_allclose(acc, [-2.0, 0.0])

    def test_speed_limit(self):
        """Desired velocity should be clamped to max_speed."""
        model = KinematicModel(max_speed=1.0)
        s = _state()
        # Large force → large desired velocity → should be clamped
        acc = model.compute_acceleration(s, np.array([100.0, 0.0]), mass=1.0, dt=0.1)
        # desired_vel clamped to [1,0], acc = (1-0)/response_time
        assert acc[0] > 0
        # After integration the speed must not exceed max_speed
        new_state = model.integrate(s, np.array([100.0, 0.0]), 1.0, 0.1)
        speed = np.linalg.norm(new_state.velocity)
        assert speed <= 1.0 + 1e-9

    def test_integrate_euler(self):
        model = KinematicModel(max_speed=5.0)
        s = _state()
        ns = model.integrate(s, np.array([3.0, 0.0]), 1.0, 0.1, method="euler")
        assert ns.position[0] > 0 or ns.velocity[0] > 0

    def test_integrate_rk4(self):
        model = KinematicModel(max_speed=5.0)
        s = _state()
        ns = model.integrate(s, np.array([3.0, 0.0]), 1.0, 0.1, method="rk4")
        assert ns.velocity[0] > 0

    def test_orientation_updates_with_velocity(self):
        model = KinematicModel(max_speed=5.0)
        s = _state()
        ns = model.integrate(s, np.array([0.0, 5.0]), 1.0, 0.5)
        # Moving in +y → orientation ≈ π/2
        assert abs(ns.orientation - math.pi / 2) < 0.5

    def test_accel_limit(self):
        """Acceleration magnitude should be clamped to max_accel."""
        model = KinematicModel(max_speed=10.0, max_accel=1.0, response_time=0.01)
        s = _state()
        acc = model.compute_acceleration(s, np.array([100.0, 0.0]), 1.0, 0.1)
        assert np.linalg.norm(acc) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
#  DynamicModel
# ---------------------------------------------------------------------------


class TestDynamicModel:
    def test_force_clamping(self):
        model = DynamicModel(max_force=5.0)
        s = _state()
        acc = model.compute_acceleration(s, np.array([100.0, 0.0]), 1.0, 0.1)
        # Net force clamped to 5 → acc = 5/1 = 5 (minus friction)
        assert np.linalg.norm(acc) <= 5.0 + 1e-6

    def test_static_friction_prevents_creep(self):
        """Small force below static friction threshold → zero acceleration."""
        mat = MaterialProperties(static_friction=0.6)
        model = DynamicModel(max_force=100.0, material=mat)
        s = _state()  # stationary
        # static friction threshold = 0.6 * 1.0 * 9.81 ≈ 5.886
        acc = model.compute_acceleration(s, np.array([1.0, 0.0]), mass=1.0, dt=0.1)
        np.testing.assert_array_equal(acc, np.zeros(2))

    def test_kinetic_friction_opposes_motion(self):
        mat = MaterialProperties(kinetic_friction=0.4, static_friction=0.0, drag=0.0)
        model = DynamicModel(max_force=100.0, material=mat)
        s = _state(vx=2.0)  # moving in +x
        acc = model.compute_acceleration(s, np.array([10.0, 0.0]), 1.0, 0.1)
        # Friction = -mu * m * g ≈ -3.924 in x
        # Net acc = (10 - 3.924)/1 ≈ 6.076
        assert acc[0] < 10.0, "friction should reduce forward acceleration"
        assert acc[0] > 0, "net force should still be forward"

    def test_drag_force(self):
        mat = MaterialProperties(drag=1.0, kinetic_friction=0.0, static_friction=0.0)
        model = DynamicModel(max_force=100.0, material=mat)
        s = _state(vx=3.0)
        acc = model.compute_acceleration(s, np.array([20.0, 0.0]), 1.0, 0.1)
        # drag_force = -1.0 * 3 * 3 = -9 in x
        # acc = (20 - 9) / 1 = 11
        np.testing.assert_allclose(acc[0], 11.0, atol=0.1)

    def test_integrate_euler(self):
        model = DynamicModel(max_speed=10.0, max_force=50.0)
        s = _state()
        ns = model.integrate(s, np.array([10.0, 0.0]), 1.0, 0.1, method="euler")
        assert ns.velocity[0] > 0

    def test_integrate_rk4(self):
        model = DynamicModel(max_speed=10.0, max_force=50.0)
        s = _state()
        ns = model.integrate(s, np.array([10.0, 0.0]), 1.0, 0.1, method="rk4")
        assert ns.velocity[0] > 0

    def test_speed_cap(self):
        model = DynamicModel(max_speed=1.0, max_force=1000.0)
        mat = MaterialProperties(static_friction=0.0, kinetic_friction=0.0)
        model.material = mat
        s = _state()
        ns = model.integrate(s, np.array([1000.0, 0.0]), 1.0, 1.0)
        assert np.linalg.norm(ns.velocity) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
#  VelocityConstraint
# ---------------------------------------------------------------------------


class TestVelocityConstraint:
    def test_within_limits(self):
        vc = VelocityConstraint(entity_id=0, min_speed=0.5, max_speed=2.0)
        v = np.array([1.0, 0.0])
        result = vc.apply(v)
        np.testing.assert_array_equal(result, v)

    def test_above_max(self):
        vc = VelocityConstraint(entity_id=0, max_speed=1.0)
        v = np.array([3.0, 4.0])  # speed = 5
        result = vc.apply(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-9)

    def test_below_min(self):
        vc = VelocityConstraint(entity_id=0, min_speed=2.0, max_speed=5.0)
        v = np.array([0.5, 0.0])  # speed = 0.5
        result = vc.apply(v)
        np.testing.assert_allclose(np.linalg.norm(result), 2.0, atol=1e-9)

    def test_zero_velocity_unchanged(self):
        vc = VelocityConstraint(entity_id=0, min_speed=1.0, max_speed=2.0)
        v = np.array([0.0, 0.0])
        result = vc.apply(v)
        np.testing.assert_array_equal(result, v)


# ---------------------------------------------------------------------------
#  WallConstraint
# ---------------------------------------------------------------------------


class TestWallConstraint:
    def test_no_penetration(self):
        wc = WallConstraint(seg_a=np.array([0.0, 0.0]), seg_b=np.array([10.0, 0.0]))
        pos = np.array([5.0, 2.0])  # well above wall
        vel = np.array([0.0, 0.0])
        new_pos, new_vel = wc.apply(pos, vel, radius=0.5)
        np.testing.assert_array_equal(new_pos, pos)

    def test_penetration_pushout(self):
        wc = WallConstraint(
            seg_a=np.array([0.0, 0.0]),
            seg_b=np.array([10.0, 0.0]),
            restitution=0.0,
        )
        # Agent at y=0.2, radius=0.5 → penetrating wall by 0.3
        pos = np.array([5.0, 0.2])
        vel = np.array([0.0, -1.0])
        new_pos, new_vel = wc.apply(pos, vel, radius=0.5)
        # Should be pushed up to y=0.5
        assert new_pos[1] >= 0.5 - 1e-6
        # Velocity toward wall should be reflected
        assert new_vel[1] >= 0.0

    def test_degenerate_segment(self):
        wc = WallConstraint(seg_a=np.array([5.0, 5.0]), seg_b=np.array([5.0, 5.0]))
        pos = np.array([5.0, 5.1])
        vel = np.array([0.0, 0.0])
        new_pos, new_vel = wc.apply(pos, vel, radius=0.5)
        # degenerate segment, should still handle gracefully
        assert new_pos is not None

    def test_bouncy_wall(self):
        wc = WallConstraint(
            seg_a=np.array([0.0, 0.0]),
            seg_b=np.array([10.0, 0.0]),
            restitution=1.0,
        )
        pos = np.array([5.0, 0.2])
        vel = np.array([1.0, -2.0])
        new_pos, new_vel = wc.apply(pos, vel, radius=0.5)
        # With restitution=1.0, velocity normal component should fully reverse
        assert new_vel[1] > 0


# ---------------------------------------------------------------------------
#  SimplePhysics engine
# ---------------------------------------------------------------------------


class TestSimplePhysics:
    def test_default_model_kinematic(self):
        sp = SimplePhysics()
        model = sp.get_model(0)
        assert isinstance(model, KinematicModel)

    def test_default_model_dynamic(self):
        sp = SimplePhysics(default_model="dynamic")
        model = sp.get_model(0)
        assert isinstance(model, DynamicModel)

    def test_set_model(self):
        sp = SimplePhysics()
        custom = DynamicModel()
        sp.set_model(42, custom)
        assert sp.get_model(42) is custom
        # Other entities still use default
        assert isinstance(sp.get_model(0), KinematicModel)

    def test_force_accumulation(self):
        sp = SimplePhysics()
        sp.apply_force(1, [3.0, 0.0], "drive")
        sp.apply_force(1, [0.0, 2.0], "social")
        net = sp.net_force(1)
        np.testing.assert_allclose(net, [3.0, 2.0])

    def test_net_force_empty(self):
        sp = SimplePhysics()
        net = sp.net_force(99)
        np.testing.assert_array_equal(net, np.zeros(2))

    def test_force_records(self):
        sp = SimplePhysics()
        sp.apply_force(1, [1.0, 0.0], "a")
        sp.apply_force(2, [0.0, 1.0], "b")
        records = sp.force_records()
        assert len(records) == 2
        assert all(isinstance(r, ForceRecord) for r in records)

    def test_clear_forces(self):
        sp = SimplePhysics()
        sp.apply_force(1, [1.0, 0.0])
        sp.clear_forces()
        np.testing.assert_array_equal(sp.net_force(1), np.zeros(2))

    def test_velocity_constraint_management(self):
        sp = SimplePhysics()
        sp.add_velocity_constraint(1, min_speed=0.0, max_speed=2.0)
        sp.remove_velocity_constraint(1)
        # No error on removing non-existent
        sp.remove_velocity_constraint(999)

    def test_wall_constraint_add(self):
        sp = SimplePhysics()
        sp.add_wall_constraint([0, 0], [10, 0], restitution=0.5)
        assert len(sp._wall_constraints) == 1

    def test_reset(self):
        sp = SimplePhysics()
        sp.apply_force(1, [1, 0])
        sp.total_collisions = 10
        sp.step_count = 5
        sp.reset()
        assert sp.total_collisions == 0
        assert sp.step_count == 0
        assert sp.net_force(1).sum() == 0

    def test_stats(self):
        sp = SimplePhysics()
        sp.apply_force(1, [1, 0])
        sp.add_velocity_constraint(1)
        sp.add_wall_constraint([0, 0], [1, 0])
        stats = sp.stats()
        assert stats["num_force_entries"] == 1
        assert stats["num_velocity_constraints"] == 1
        assert stats["num_wall_constraints"] == 1

    def test_repr(self):
        sp = SimplePhysics(integration_method="rk4")
        r = repr(sp)
        assert "rk4" in r
        assert "SimplePhysics" in r

    def test_set_default_kinematic(self):
        sp = SimplePhysics()
        sp.set_default_kinematic(max_speed=5.0)
        model = sp.get_model(0)
        assert isinstance(model, KinematicModel)
        assert model.max_speed == 5.0

    def test_set_default_dynamic(self):
        sp = SimplePhysics()
        sp.set_default_dynamic(max_force=30.0)
        sp._default_model_name = "dynamic"
        model = sp.get_model(0)
        assert isinstance(model, DynamicModel)
        assert model.max_force == 30.0
