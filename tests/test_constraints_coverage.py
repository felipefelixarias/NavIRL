"""Tests for uncovered paths in navirl.safety.constraints."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.safety.constraints import (
    AccelerationConstraint,
    BoundaryConstraint,
    CollisionConstraint,
    ConstraintSet,
    ProxemicsConstraint,
    SpeedConstraint,
)

# ---------------------------------------------------------------------------
# CollisionConstraint.project – bisection branch
# ---------------------------------------------------------------------------


class TestCollisionConstraintProject:
    """Cover the bisection loop in CollisionConstraint.project (line 93+)."""

    def test_project_reduces_speed_until_safe(self):
        """When heading toward an obstacle, project should scale velocity down."""
        obs_pos = np.array([[1.0, 0.0]])
        cc = CollisionConstraint(
            obstacle_positions=obs_pos,
            obstacle_radii=0.3,
            agent_radius=0.25,
            time_horizon=2.0,
            dt=0.1,
        )
        state = np.array([0.0, 0.0])
        action = np.array([2.0, 0.0])  # heading straight at obstacle

        assert not cc.is_safe(state, action)
        projected = cc.project(state, action)
        # Projected action should have reduced speed
        assert np.linalg.norm(projected[:2]) < np.linalg.norm(action[:2])

    def test_project_finds_safe_scale(self):
        """Bisection should find a reduced velocity that avoids collision."""
        obs_pos = np.array([[0.8, 0.0]])
        cc = CollisionConstraint(
            obstacle_positions=obs_pos,
            obstacle_radii=0.2,
            agent_radius=0.2,
            time_horizon=2.0,
            dt=0.1,
        )
        state = np.array([0.0, 0.0])
        action = np.array([2.0, 0.0])  # will reach obstacle within horizon

        assert not cc.is_safe(state, action)
        projected = cc.project(state, action)
        # Bisection found a safe scale
        assert cc.is_safe(state, projected)
        assert np.linalg.norm(projected[:2]) <= np.linalg.norm(action[:2])

    def test_project_safe_action_unchanged(self):
        """A safe action should be returned as-is (copy)."""
        cc = CollisionConstraint(
            obstacle_positions=np.array([[10.0, 10.0]]),
            obstacle_radii=0.3,
            agent_radius=0.25,
        )
        state = np.array([0.0, 0.0])
        action = np.array([0.1, 0.0])
        projected = cc.project(state, action)
        np.testing.assert_array_almost_equal(projected, action)


# ---------------------------------------------------------------------------
# AccelerationConstraint – jerk checking and clamping
# ---------------------------------------------------------------------------


class TestAccelerationConstraintJerk:
    """Cover jerk limit paths in AccelerationConstraint (lines 168-171, 183-187)."""

    def test_is_safe_jerk_violation(self):
        """Large jerk should be flagged unsafe."""
        ac = AccelerationConstraint(max_acceleration=100.0, max_jerk=1.0, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        # Set a previous acceleration to create a jerk reference
        ac._prev_acceleration = np.array([0.0, 0.0])

        # Action that creates huge acceleration (and therefore huge jerk)
        action = np.array([5.0, 0.0])
        assert not ac.is_safe(state, action)

    def test_is_safe_jerk_within_limit(self):
        """Small jerk should pass."""
        ac = AccelerationConstraint(max_acceleration=100.0, max_jerk=1000.0, dt=0.1)
        ac._prev_acceleration = np.array([0.0, 0.0])
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([0.01, 0.0])
        assert ac.is_safe(state, action)

    def test_is_safe_no_prev_acceleration_skips_jerk(self):
        """Without previous acceleration, jerk check is skipped."""
        ac = AccelerationConstraint(max_acceleration=100.0, max_jerk=0.001, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([0.5, 0.0])
        # No _prev_acceleration set, jerk check skipped
        assert ac.is_safe(state, action)

    def test_project_clamps_jerk(self):
        """Project should clamp jerk when it exceeds max_jerk."""
        ac = AccelerationConstraint(max_acceleration=100.0, max_jerk=1.0, dt=0.1)
        ac._prev_acceleration = np.array([0.0, 0.0])
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([10.0, 0.0])  # huge acceleration → huge jerk

        projected = ac.project(state, action)
        # Projected action should be reduced due to jerk clamping
        assert np.linalg.norm(projected[:2]) < np.linalg.norm(action[:2])

    def test_acceleration_with_short_state(self):
        """State with fewer than 4 elements should default velocity to zero."""
        ac = AccelerationConstraint(max_acceleration=5.0, dt=0.1)
        state = np.array([1.0, 2.0])  # only position, no velocity
        action = np.array([0.3, 0.0])
        # Should not raise
        assert isinstance(ac.is_safe(state, action), bool)
        projected = ac.project(state, action)
        assert projected.shape == action.shape


# ---------------------------------------------------------------------------
# ProxemicsConstraint
# ---------------------------------------------------------------------------


class TestProxemicsConstraint:
    """Cover ProxemicsConstraint (lines 231-261)."""

    def test_default_min_distance(self):
        """min_distance should default to personal_radius."""
        pc = ProxemicsConstraint(personal_radius=1.5)
        assert pc.min_distance == 1.5

    def test_explicit_min_distance(self):
        """Explicit min_distance overrides default."""
        pc = ProxemicsConstraint(personal_radius=1.5, min_distance=0.8)
        assert pc.min_distance == 0.8

    def test_is_safe_no_pedestrians(self):
        """No pedestrians should always be safe."""
        pc = ProxemicsConstraint()
        state = np.array([0.0, 0.0])
        action = np.array([1.0, 0.0])
        assert pc.is_safe(state, action)

    def test_is_safe_far_pedestrian(self):
        """Pedestrian far away should be safe."""
        pc = ProxemicsConstraint(
            pedestrian_positions=np.array([[100.0, 100.0]]),
            min_distance=1.2,
        )
        state = np.array([0.0, 0.0])
        action = np.array([0.5, 0.0])
        assert pc.is_safe(state, action)

    def test_is_safe_close_pedestrian(self):
        """Moving toward a nearby pedestrian should be unsafe."""
        pc = ProxemicsConstraint(
            pedestrian_positions=np.array([[0.2, 0.0]]),
            min_distance=1.2,
            dt=0.1,
        )
        state = np.array([0.0, 0.0])
        action = np.array([1.0, 0.0])  # moves to (0.1, 0) — within 1.2 of (0.2, 0)
        assert not pc.is_safe(state, action)

    def test_project_pushes_away(self):
        """Project should redirect velocity away from pedestrian."""
        pc = ProxemicsConstraint(
            pedestrian_positions=np.array([[0.5, 0.0]]),
            min_distance=1.2,
            dt=0.1,
        )
        state = np.array([0.0, 0.0])
        action = np.array([5.0, 0.0])

        projected = pc.project(state, action)
        # Should not be the same as original
        assert not np.allclose(projected, action)

    def test_project_safe_action_unchanged(self):
        """Safe action should be returned as a copy."""
        pc = ProxemicsConstraint(
            pedestrian_positions=np.array([[100.0, 100.0]]),
            min_distance=1.2,
        )
        state = np.array([0.0, 0.0])
        action = np.array([0.1, 0.0])
        projected = pc.project(state, action)
        np.testing.assert_array_almost_equal(projected, action)

    def test_project_zero_repulsion_norm(self):
        """When agent is exactly on top of pedestrian, should return zero action."""
        pc = ProxemicsConstraint(
            pedestrian_positions=np.array([[0.0, 0.0]]),
            min_distance=1.2,
            dt=0.1,
        )
        # Agent at origin, pedestrian at origin → next_pos is at action*dt
        # next_pos - ped = action*dt, so diffs will be small
        state = np.array([0.0, 0.0])
        # Action that puts us exactly on the pedestrian: (0,0)*dt = (0,0)
        action = np.array([0.0, 0.0])
        # is_safe: next_pos = (0,0), dist = 0 < 1.2 → unsafe
        assert not pc.is_safe(state, action)
        projected = pc.project(state, action)
        # With zero action the repulsion norm will be zero, so we get zero action
        np.testing.assert_array_almost_equal(projected, np.zeros(2))

    def test_project_multiple_pedestrians(self):
        """Project with multiple pedestrians violating."""
        pc = ProxemicsConstraint(
            pedestrian_positions=np.array([[0.3, 0.0], [0.0, 0.3]]),
            min_distance=1.0,
            dt=0.1,
        )
        state = np.array([0.0, 0.0])
        action = np.array([1.0, 1.0])
        projected = pc.project(state, action)
        assert projected.shape == action.shape


# ---------------------------------------------------------------------------
# BoundaryConstraint
# ---------------------------------------------------------------------------


class TestBoundaryConstraint:
    """Cover BoundaryConstraint (lines 288-306)."""

    def test_is_safe_inside(self):
        bc = BoundaryConstraint(x_min=-5, x_max=5, y_min=-5, y_max=5, dt=0.1)
        state = np.array([0.0, 0.0])
        action = np.array([1.0, 1.0])
        assert bc.is_safe(state, action)

    def test_is_safe_outside_x(self):
        bc = BoundaryConstraint(x_min=-5, x_max=5, y_min=-5, y_max=5, dt=0.1)
        state = np.array([4.9, 0.0])
        action = np.array([10.0, 0.0])  # next pos: 4.9 + 10*0.1 = 5.9 > 5
        assert not bc.is_safe(state, action)

    def test_is_safe_outside_y(self):
        bc = BoundaryConstraint(x_min=-5, x_max=5, y_min=-5, y_max=5, dt=0.1)
        state = np.array([0.0, -4.9])
        action = np.array([0.0, -10.0])  # next pos: -4.9 - 1.0 = -5.9 < -5
        assert not bc.is_safe(state, action)

    def test_project_clamps_to_boundary(self):
        bc = BoundaryConstraint(x_min=-5, x_max=5, y_min=-5, y_max=5, dt=0.1)
        state = np.array([4.5, 0.0])
        action = np.array([20.0, 0.0])  # next pos: 4.5 + 2.0 = 6.5 > 5

        projected = bc.project(state, action)
        # Verify next position after projection is within bounds
        next_pos = state[:2] + projected[:2] * bc.dt
        assert next_pos[0] <= 5.0 + 1e-8
        assert next_pos[0] >= -5.0 - 1e-8

    def test_project_safe_unchanged(self):
        bc = BoundaryConstraint(x_min=-5, x_max=5, y_min=-5, y_max=5, dt=0.1)
        state = np.array([0.0, 0.0])
        action = np.array([1.0, 1.0])
        projected = bc.project(state, action)
        np.testing.assert_array_almost_equal(projected, action)

    def test_project_corner_case(self):
        """Both x and y exceed bounds."""
        bc = BoundaryConstraint(x_min=-1, x_max=1, y_min=-1, y_max=1, dt=0.1)
        state = np.array([0.9, 0.9])
        action = np.array([20.0, 20.0])  # next: (2.9, 2.9) way out of bounds

        projected = bc.project(state, action)
        next_pos = state[:2] + projected[:2] * bc.dt
        assert next_pos[0] <= 1.0 + 1e-8
        assert next_pos[1] <= 1.0 + 1e-8


# ---------------------------------------------------------------------------
# ConstraintSet
# ---------------------------------------------------------------------------


class TestConstraintSet:
    """Cover ConstraintSet (lines 341, 348-360)."""

    def test_add(self):
        cs = ConstraintSet()
        sc = SpeedConstraint(max_speed=1.0)
        cs.add(sc)
        assert len(cs.constraints) == 1
        assert cs.constraints[0] is sc

    def test_is_safe_all_pass(self):
        cs = ConstraintSet(constraints=[
            SpeedConstraint(max_speed=2.0),
            BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10),
        ])
        state = np.array([0.0, 0.0])
        action = np.array([0.5, 0.0])
        assert cs.is_safe(state, action)

    def test_is_safe_one_fails(self):
        cs = ConstraintSet(constraints=[
            SpeedConstraint(max_speed=0.1),  # will fail
            BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10),
        ])
        state = np.array([0.0, 0.0])
        action = np.array([1.0, 0.0])
        assert not cs.is_safe(state, action)

    def test_project_satisfies_all(self):
        cs = ConstraintSet(constraints=[
            SpeedConstraint(max_speed=0.5),
            BoundaryConstraint(x_min=-1, x_max=1, y_min=-1, y_max=1, dt=0.1),
        ])
        state = np.array([0.0, 0.0])
        action = np.array([10.0, 10.0])

        projected = cs.project(state, action)
        # Speed should be within limit
        assert np.linalg.norm(projected[:2]) <= 0.5 + 1e-6

    def test_project_empty_set(self):
        """Empty constraint set should return action unchanged."""
        cs = ConstraintSet()
        state = np.array([0.0, 0.0])
        action = np.array([5.0, 5.0])
        projected = cs.project(state, action)
        np.testing.assert_array_almost_equal(projected, action)

    def test_project_converges(self):
        """Fixed-point iteration should converge."""
        cs = ConstraintSet(
            constraints=[
                SpeedConstraint(max_speed=1.0),
                BoundaryConstraint(x_min=-2, x_max=2, y_min=-2, y_max=2, dt=0.1),
            ],
            max_iters=20,
        )
        state = np.array([1.9, 1.9])
        action = np.array([50.0, 50.0])
        projected = cs.project(state, action)
        # After projection, all constraints should be satisfied
        assert cs.is_safe(state, projected)

    def test_project_already_safe(self):
        """Already safe action is returned as-is."""
        cs = ConstraintSet(constraints=[SpeedConstraint(max_speed=5.0)])
        state = np.array([0.0, 0.0])
        action = np.array([0.1, 0.0])
        projected = cs.project(state, action)
        np.testing.assert_array_almost_equal(projected, action)
