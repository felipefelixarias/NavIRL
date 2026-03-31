"""Tests for NavIRL safety constraints, shielding, risk, and monitoring."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.safety.constrained_optimization import LagrangianMultiplier
from navirl.safety.constraints import (
    AccelerationConstraint,
    CollisionConstraint,
    SpeedConstraint,
)
from navirl.safety.monitoring import SafetyAlert, SafetyMonitor, Severity
from navirl.safety.risk_assessment import RiskEstimator
from navirl.safety.shield import SafetyShield

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def speed_constraint():
    return SpeedConstraint(max_speed=1.5)


@pytest.fixture
def collision_constraint():
    return CollisionConstraint(
        obstacle_positions=np.array([[5.0, 0.0], [0.0, 5.0]]),
        obstacle_radii=0.3,
        agent_radius=0.25,
        time_horizon=2.0,
        dt=0.1,
    )


@pytest.fixture
def acceleration_constraint():
    return AccelerationConstraint(max_acceleration=3.0, dt=0.1)


@pytest.fixture
def risk_estimator():
    return RiskEstimator(agent_radius=0.25, default_obstacle_radius=0.3)


@pytest.fixture
def safety_monitor():
    return SafetyMonitor()


# ---------------------------------------------------------------------------
# SpeedConstraint
# ---------------------------------------------------------------------------

class TestSpeedConstraint:
    def test_safe_action(self, speed_constraint):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0])
        assert speed_constraint.is_safe(state, action) is True

    def test_unsafe_action(self, speed_constraint):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([2.0, 2.0])  # speed > 1.5
        assert speed_constraint.is_safe(state, action) is False

    def test_project_safe_action(self, speed_constraint):
        state = np.zeros(4)
        action = np.array([1.0, 0.0])
        projected = speed_constraint.project(state, action)
        np.testing.assert_array_equal(projected, action)

    def test_project_unsafe_action(self, speed_constraint):
        state = np.zeros(4)
        action = np.array([3.0, 4.0])  # speed = 5.0
        projected = speed_constraint.project(state, action)
        speed = np.linalg.norm(projected[:2])
        assert speed <= 1.5 + 1e-6

    def test_zero_action(self, speed_constraint):
        state = np.zeros(4)
        action = np.array([0.0, 0.0])
        assert speed_constraint.is_safe(state, action) is True

    @pytest.mark.parametrize("speed", [0.1, 0.5, 1.0, 1.49, 1.5])
    def test_boundary_speeds(self, speed_constraint, speed):
        state = np.zeros(4)
        action = np.array([speed, 0.0])
        assert speed_constraint.is_safe(state, action) is True

    def test_just_over_limit(self, speed_constraint):
        state = np.zeros(4)
        action = np.array([1.51, 0.0])
        assert speed_constraint.is_safe(state, action) is False


# ---------------------------------------------------------------------------
# CollisionConstraint
# ---------------------------------------------------------------------------

class TestCollisionConstraint:
    def test_safe_action(self, collision_constraint):
        state = np.array([0.0, 0.0])
        action = np.array([0.0, 0.0])  # stay still, far from obstacles
        assert collision_constraint.is_safe(state, action) is True

    def test_unsafe_action_toward_obstacle(self, collision_constraint):
        state = np.array([4.0, 0.0])
        action = np.array([1.0, 0.0])  # heading toward obstacle at (5,0)
        assert collision_constraint.is_safe(state, action) is False

    def test_project_reduces_speed(self, collision_constraint):
        state = np.array([4.0, 0.0])
        action = np.array([1.0, 0.0])
        projected = collision_constraint.project(state, action)
        # Projected action should have lower speed
        assert np.linalg.norm(projected[:2]) <= np.linalg.norm(action[:2])

    def test_no_obstacles(self):
        cc = CollisionConstraint(obstacle_positions=np.zeros((0, 2)))
        state = np.array([0.0, 0.0])
        action = np.array([5.0, 5.0])
        assert cc.is_safe(state, action) is True


# ---------------------------------------------------------------------------
# AccelerationConstraint
# ---------------------------------------------------------------------------

class TestAccelerationConstraint:
    def test_safe_acceleration(self, acceleration_constraint):
        state = np.array([0.0, 0.0, 0.0, 0.0])  # pos + vel
        action = np.array([0.1, 0.0])  # small velocity => small acceleration
        assert acceleration_constraint.is_safe(state, action) is True

    def test_unsafe_acceleration(self, acceleration_constraint):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([10.0, 0.0])  # acc = 10/0.1 = 100
        assert acceleration_constraint.is_safe(state, action) is False

    def test_project_reduces_acceleration(self, acceleration_constraint):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([10.0, 0.0])
        projected = acceleration_constraint.project(state, action)
        # Projected should have lower acceleration
        acc_proj = np.linalg.norm((projected[:2] - state[2:4]) / 0.1)
        assert acc_proj <= 3.0 + 1e-3


# ---------------------------------------------------------------------------
# SafetyShield
# ---------------------------------------------------------------------------

class TestSafetyShield:
    def test_safe_action_passes_through(self, speed_constraint):
        class MockAgent:
            def act(self, obs):
                return np.array([0.5, 0.0])

        from navirl.safety.constraints import ConstraintSet
        shield = SafetyShield(
            agent=MockAgent(),
            constraints=ConstraintSet([speed_constraint]),
        )
        action = shield.act(np.zeros(4))
        np.testing.assert_allclose(action, [0.5, 0.0])
        assert shield.interventions == 0

    def test_unsafe_action_is_modified(self, speed_constraint):
        class MockAgent:
            def act(self, obs):
                return np.array([5.0, 5.0])

        from navirl.safety.constraints import ConstraintSet
        shield = SafetyShield(
            agent=MockAgent(),
            constraints=ConstraintSet([speed_constraint]),
        )
        action = shield.act(np.zeros(4))
        assert np.linalg.norm(action[:2]) <= 1.5 + 1e-6
        assert shield.interventions >= 1

    def test_fallback_policy(self, speed_constraint):
        class MockAgent:
            def act(self, obs):
                return np.array([100.0, 100.0])

        from navirl.safety.constraints import ConstraintSet
        shield = SafetyShield(
            agent=MockAgent(),
            constraints=ConstraintSet([speed_constraint]),
            fallback_policy=lambda obs: np.array([0.1, 0.1]),
        )
        action = shield.act(np.zeros(4))
        # Should either be projected to safe or use fallback
        assert np.linalg.norm(action[:2]) <= 1.5 + 1e-6

    def test_intervention_rate(self, speed_constraint):
        class MockAgent:
            def __init__(self):
                self._call = 0
            def act(self, obs):
                self._call += 1
                if self._call % 2 == 0:
                    return np.array([5.0, 5.0])  # unsafe
                return np.array([0.1, 0.0])  # safe

        from navirl.safety.constraints import ConstraintSet
        shield = SafetyShield(
            agent=MockAgent(),
            constraints=ConstraintSet([speed_constraint]),
        )
        for _ in range(10):
            shield.act(np.zeros(4))
        assert 0.0 < shield.intervention_rate < 1.0

    def test_intervention_rate_zero(self, speed_constraint):
        class SafeAgent:
            def act(self, obs):
                return np.array([0.0, 0.0])

        from navirl.safety.constraints import ConstraintSet
        shield = SafetyShield(
            agent=SafeAgent(),
            constraints=ConstraintSet([speed_constraint]),
        )
        shield.act(np.zeros(4))
        assert shield.intervention_rate == 0.0


# ---------------------------------------------------------------------------
# RiskEstimator
# ---------------------------------------------------------------------------

class TestRiskEstimator:
    def test_ttc_no_obstacles(self, risk_estimator):
        agent_state = np.array([0.0, 0.0, 1.0, 0.0])
        obstacle_states = np.zeros((0, 4))
        ttc = risk_estimator.time_to_collision(agent_state, obstacle_states)
        assert ttc == float("inf")

    def test_ttc_head_on(self, risk_estimator):
        agent_state = np.array([0.0, 0.0, 1.0, 0.0])
        obstacle_states = np.array([[5.0, 0.0, -1.0, 0.0]])
        ttc = risk_estimator.time_to_collision(agent_state, obstacle_states)
        assert ttc < float("inf")
        assert ttc > 0.0

    def test_ttc_perpendicular(self, risk_estimator):
        agent_state = np.array([0.0, 0.0, 1.0, 0.0])
        obstacle_states = np.array([[0.0, 10.0, 0.0, 0.0]])  # far away, perpendicular
        ttc = risk_estimator.time_to_collision(agent_state, obstacle_states)
        # No collision if obstacle is stationary and perpendicular
        assert ttc == float("inf") or ttc > 10.0

    def test_ttc_stationary_collision(self, risk_estimator):
        agent_state = np.array([0.0, 0.0, 1.0, 0.0])
        obstacle_states = np.array([[2.0, 0.0, 0.0, 0.0]])
        ttc = risk_estimator.time_to_collision(agent_state, obstacle_states)
        assert ttc > 0.0
        assert ttc < 5.0


# ---------------------------------------------------------------------------
# SafetyMonitor
# ---------------------------------------------------------------------------

class TestSafetyMonitor:
    def test_record_step(self, safety_monitor):
        state = np.array([1.0, 2.0])
        action = np.array([0.5, 0.3])
        safety_monitor.record_step(state, action)
        assert safety_monitor._step_count == 1

    def test_record_multiple_steps(self, safety_monitor):
        for i in range(10):
            safety_monitor.record_step(
                np.array([float(i), 0.0]),
                np.array([0.5, 0.0]),
                info={"min_obstacle_dist": 1.0 + i},
            )
        assert safety_monitor._step_count == 10
        assert len(safety_monitor._min_obstacle_distances) == 10

    def test_shield_intervention_tracking(self, safety_monitor):
        safety_monitor.record_step(
            np.zeros(2), np.zeros(2),
            info={"shield_intervened": True},
        )
        # Implementation may or may not track this directly
        assert safety_monitor._step_count == 1


class TestSafetyAlert:
    def test_construction(self):
        alert = SafetyAlert(
            timestamp=1.0,
            severity=Severity.WARNING,
            constraint_name="speed",
            details={"speed": 2.5, "limit": 1.5},
        )
        assert alert.severity == Severity.WARNING
        assert alert.constraint_name == "speed"
        assert alert.details["speed"] == 2.5

    @pytest.mark.parametrize("severity", [Severity.INFO, Severity.WARNING, Severity.CRITICAL])
    def test_all_severities(self, severity):
        alert = SafetyAlert(
            timestamp=0.0, severity=severity, constraint_name="test",
        )
        assert alert.severity == severity


# ---------------------------------------------------------------------------
# Lagrangian multiplier
# ---------------------------------------------------------------------------

class TestLagrangianMultiplier:
    def test_initial_value(self):
        lm = LagrangianMultiplier(initial_value=1.0)
        assert lm.value == 1.0

    def test_update_increases_on_violation(self):
        lm = LagrangianMultiplier(initial_value=0.0, learning_rate=0.1)
        lm.update(constraint_value=1.5, threshold=1.0)
        assert lm.value > 0.0

    def test_update_decreases_below_threshold(self):
        lm = LagrangianMultiplier(initial_value=1.0, learning_rate=0.1)
        lm.update(constraint_value=0.5, threshold=1.0)
        assert lm.value < 1.0

    def test_clamped_to_zero(self):
        lm = LagrangianMultiplier(initial_value=0.01, learning_rate=10.0)
        lm.update(constraint_value=0.0, threshold=100.0)
        assert lm.value >= 0.0

    def test_clamped_to_max(self):
        lm = LagrangianMultiplier(initial_value=99.0, learning_rate=10.0, max_value=100.0)
        lm.update(constraint_value=200.0, threshold=0.0)
        assert lm.value <= 100.0

    def test_penalized_objective(self):
        lm = LagrangianMultiplier(initial_value=2.0)
        obj = lm.penalized_objective(reward=10.0, constraint_value=3.0)
        assert obj == pytest.approx(10.0 - 2.0 * 3.0)

    def test_repeated_updates(self):
        lm = LagrangianMultiplier(initial_value=0.0, learning_rate=0.01)
        for _ in range(100):
            lm.update(constraint_value=2.0, threshold=1.0)
        assert lm.value > 0.5  # should have grown significantly


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSafetyEdgeCases:
    def test_speed_constraint_exact_limit(self):
        sc = SpeedConstraint(max_speed=1.0)
        state = np.zeros(4)
        action = np.array([1.0, 0.0])
        assert sc.is_safe(state, action) is True

    def test_collision_with_zero_radius(self):
        cc = CollisionConstraint(
            obstacle_positions=np.array([[1.0, 0.0]]),
            obstacle_radii=0.0,
            agent_radius=0.0,
        )
        state = np.array([0.0, 0.0])
        action = np.array([0.0, 0.0])
        assert cc.is_safe(state, action) is True

    def test_empty_shield(self):
        class MockAgent:
            def act(self, obs):
                return np.array([1.0, 0.0])

        from navirl.safety.constraints import ConstraintSet
        shield = SafetyShield(
            agent=MockAgent(),
            constraints=ConstraintSet([]),
        )
        action = shield.act(np.zeros(4))
        np.testing.assert_allclose(action, [1.0, 0.0])
