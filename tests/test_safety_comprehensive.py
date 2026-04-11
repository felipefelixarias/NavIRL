"""Comprehensive tests for navirl.safety — constraints, optimization, monitoring, risk, shield."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.safety.constrained_optimization import (
    CPOUpdate,
    LagrangianMultiplier,
    PIDLagrangian,
)
from navirl.safety.constraints import (
    AccelerationConstraint,
    BoundaryConstraint,
    CollisionConstraint,
    ConstraintSet,
    ProxemicsConstraint,
    SpeedConstraint,
)
from navirl.safety.monitoring import SafetyAlert, SafetyMonitor, Severity
from navirl.safety.risk_assessment import PredictiveRiskModel, RiskEstimator
from navirl.safety.shield import CBFShield, ReachabilityShield, SafetyShield

# ---------------------------------------------------------------------------
# SpeedConstraint
# ---------------------------------------------------------------------------


class TestSpeedConstraintComprehensive:
    def test_safe_within_limit(self):
        c = SpeedConstraint(max_speed=1.5)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0])
        assert c.is_safe(state, action)

    def test_unsafe_over_limit(self):
        c = SpeedConstraint(max_speed=1.5)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([2.0, 2.0])
        assert not c.is_safe(state, action)

    def test_project_scales_down(self):
        c = SpeedConstraint(max_speed=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([3.0, 4.0])
        projected = c.project(state, action)
        speed = np.linalg.norm(projected[:2])
        assert speed <= 1.0 + 1e-6

    def test_project_no_change_when_safe(self):
        c = SpeedConstraint(max_speed=5.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0])
        projected = c.project(state, action)
        np.testing.assert_allclose(projected[:2], action[:2], atol=1e-10)

    def test_zero_speed(self):
        c = SpeedConstraint(max_speed=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0])
        assert c.is_safe(state, action)

    def test_exactly_at_limit(self):
        c = SpeedConstraint(max_speed=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0])
        assert c.is_safe(state, action)


# ---------------------------------------------------------------------------
# BoundaryConstraint
# ---------------------------------------------------------------------------


class TestBoundaryConstraintComprehensive:
    def test_safe_inside(self):
        c = BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0])
        assert c.is_safe(state, action)

    def test_unsafe_x_max(self):
        c = BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10, dt=0.1)
        state = np.array([9.9, 0.0, 0.0, 0.0])
        action = np.array([5.0, 0.0])
        assert not c.is_safe(state, action)

    def test_unsafe_y_max(self):
        c = BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10, dt=0.1)
        state = np.array([0.0, 9.9, 0.0, 0.0])
        action = np.array([0.0, 5.0])
        assert not c.is_safe(state, action)

    def test_unsafe_negative(self):
        c = BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10, dt=0.1)
        state = np.array([-9.9, 0.0, 0.0, 0.0])
        action = np.array([-5.0, 0.0])
        assert not c.is_safe(state, action)

    def test_project_clamps(self):
        c = BoundaryConstraint(x_min=-10, x_max=10, y_min=-10, y_max=10, dt=0.1)
        state = np.array([9.5, 0.0, 0.0, 0.0])
        action = np.array([10.0, 0.0])
        projected = c.project(state, action)
        next_pos = state[:2] + projected[:2] * 0.1
        assert next_pos[0] <= 10.0 + 1e-6


# ---------------------------------------------------------------------------
# CollisionConstraint
# ---------------------------------------------------------------------------


class TestCollisionConstraintComprehensive:
    def test_safe_no_obstacles(self):
        c = CollisionConstraint(
            obstacle_positions=np.zeros((0, 2)),
            obstacle_radii=0.3,
            agent_radius=0.25,
            time_horizon=1.0,
            dt=0.1,
        )
        assert c.is_safe(np.zeros(4), np.array([1.0, 0.0]))

    def test_unsafe_collision_path(self):
        c = CollisionConstraint(
            obstacle_positions=np.array([[2.0, 0.0]]),
            obstacle_radii=0.5,
            agent_radius=0.25,
            time_horizon=3.0,
            dt=0.1,
        )
        state = np.array([0.0, 0.0, 1.0, 0.0])
        assert not c.is_safe(state, np.array([1.0, 0.0]))

    def test_safe_perpendicular(self):
        c = CollisionConstraint(
            obstacle_positions=np.array([[5.0, 0.0]]),
            obstacle_radii=0.3,
            agent_radius=0.25,
            time_horizon=2.0,
            dt=0.1,
        )
        assert c.is_safe(np.zeros(4), np.array([0.0, 1.0]))

    def test_multiple_obstacles(self):
        c = CollisionConstraint(
            obstacle_positions=np.array([[3.0, 0.0], [0.0, 3.0]]),
            obstacle_radii=np.array([0.5, 0.5]),
            agent_radius=0.25,
            time_horizon=5.0,
            dt=0.1,
        )
        result = c.is_safe(np.zeros(4), np.array([1.0, 1.0]))
        assert isinstance(result, bool)

    def test_project(self):
        c = CollisionConstraint(
            obstacle_positions=np.array([[1.0, 0.0]]),
            obstacle_radii=0.3,
            agent_radius=0.25,
            time_horizon=2.0,
            dt=0.1,
        )
        projected = c.project(np.array([0.0, 0.0, 1.0, 0.0]), np.array([2.0, 0.0]))
        assert np.linalg.norm(projected[:2]) <= np.linalg.norm([2.0, 0.0]) + 1e-6


# ---------------------------------------------------------------------------
# AccelerationConstraint
# ---------------------------------------------------------------------------


class TestAccelerationConstraintComprehensive:
    def test_safe_low_accel(self):
        c = AccelerationConstraint(max_acceleration=3.0, dt=0.1)
        assert c.is_safe(np.zeros(4), np.array([0.1, 0.0]))

    def test_unsafe_high_accel(self):
        c = AccelerationConstraint(max_acceleration=1.0, dt=0.1)
        assert not c.is_safe(np.zeros(4), np.array([5.0, 0.0]))

    def test_project(self):
        c = AccelerationConstraint(max_acceleration=1.0, dt=0.1)
        projected = c.project(np.zeros(4), np.array([10.0, 0.0]))
        assert isinstance(projected, np.ndarray)

    def test_with_jerk(self):
        c = AccelerationConstraint(max_acceleration=3.0, max_jerk=5.0, dt=0.1)
        result = c.is_safe(np.zeros(4), np.array([0.1, 0.0]))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# ProxemicsConstraint
# ---------------------------------------------------------------------------


class TestProxemicsConstraintComprehensive:
    def test_safe_far(self):
        c = ProxemicsConstraint(
            pedestrian_positions=np.array([[10.0, 0.0]]),
            personal_radius=1.2,
            dt=0.1,
        )
        assert c.is_safe(np.zeros(4), np.array([0.0, 1.0]))

    def test_unsafe_close(self):
        c = ProxemicsConstraint(
            pedestrian_positions=np.array([[0.5, 0.0]]),
            personal_radius=1.2,
            dt=0.1,
        )
        assert not c.is_safe(np.zeros(4), np.array([1.0, 0.0]))

    def test_no_pedestrians(self):
        c = ProxemicsConstraint(
            pedestrian_positions=np.zeros((0, 2)),
            dt=0.1,
        )
        assert c.is_safe(np.zeros(4), np.array([1.0, 0.0]))

    def test_project(self):
        c = ProxemicsConstraint(
            pedestrian_positions=np.array([[0.5, 0.0]]),
            personal_radius=1.2,
            dt=0.1,
        )
        projected = c.project(np.zeros(4), np.array([2.0, 0.0]))
        assert isinstance(projected, np.ndarray)

    def test_multiple_pedestrians(self):
        c = ProxemicsConstraint(
            pedestrian_positions=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            personal_radius=1.5,
            dt=0.1,
        )
        result = c.is_safe(np.zeros(4), np.array([0.5, 0.0]))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# ConstraintSet
# ---------------------------------------------------------------------------


class TestConstraintSetComprehensive:
    def test_empty_safe(self):
        cs = ConstraintSet()
        assert cs.is_safe(np.zeros(4), np.array([1.0, 0.0]))

    def test_single_constraint(self):
        cs = ConstraintSet()
        cs.add(SpeedConstraint(max_speed=1.0))
        assert not cs.is_safe(np.zeros(4), np.array([5.0, 5.0]))
        assert cs.is_safe(np.zeros(4), np.array([0.5, 0.0]))

    def test_project(self):
        cs = ConstraintSet(constraints=[SpeedConstraint(max_speed=1.0)], max_iters=10)
        projected = cs.project(np.zeros(4), np.array([5.0, 5.0]))
        assert np.linalg.norm(projected[:2]) <= 1.0 + 1e-3

    def test_combined(self):
        cs = ConstraintSet(
            constraints=[
                SpeedConstraint(max_speed=2.0),
                BoundaryConstraint(x_min=-5, x_max=5, y_min=-5, y_max=5, dt=0.1),
            ]
        )
        state = np.array([4.9, 0.0, 0.0, 0.0])
        assert not cs.is_safe(state, np.array([2.0, 0.0]))


# ---------------------------------------------------------------------------
# LagrangianMultiplier
# ---------------------------------------------------------------------------


class TestLagrangianMultiplierComprehensive:
    def test_initial_value(self):
        lm = LagrangianMultiplier(initial_value=0.5)
        assert lm.value == pytest.approx(0.5)

    def test_increases_on_violation(self):
        lm = LagrangianMultiplier(initial_value=0.0, learning_rate=0.1)
        lm.update(constraint_value=2.0, threshold=1.0)
        assert lm.value > 0.0

    def test_decreases_on_satisfaction(self):
        lm = LagrangianMultiplier(initial_value=1.0, learning_rate=0.1)
        lm.update(constraint_value=0.0, threshold=1.0)
        assert lm.value < 1.0

    def test_clamped_zero(self):
        lm = LagrangianMultiplier(initial_value=0.01, learning_rate=10.0)
        lm.update(constraint_value=0.0, threshold=100.0)
        assert lm.value >= 0.0

    def test_clamped_max(self):
        lm = LagrangianMultiplier(initial_value=0.0, learning_rate=1000.0, max_value=5.0)
        lm.update(constraint_value=100.0, threshold=0.0)
        assert lm.value <= 5.0

    def test_penalized_objective(self):
        lm = LagrangianMultiplier(initial_value=2.0)
        assert lm.penalized_objective(10.0, 3.0) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# PIDLagrangian
# ---------------------------------------------------------------------------


class TestPIDLagrangianComprehensive:
    def test_initial(self):
        pid = PIDLagrangian()
        assert pid.value >= 0.0

    def test_violation_increases(self):
        pid = PIDLagrangian(kp=1.0, ki=0.01, kd=0.1)
        pid.update(constraint_value=5.0, threshold=1.0)
        assert pid.value > 0.0

    def test_penalized_objective(self):
        pid = PIDLagrangian(kp=1.0)
        pid.update(constraint_value=2.0, threshold=1.0)
        assert pid.penalized_objective(10.0, 2.0) < 10.0

    def test_reset(self):
        pid = PIDLagrangian()
        pid.update(constraint_value=5.0, threshold=1.0)
        pid.reset()
        assert pid.value == pytest.approx(0.0)

    def test_convergence(self):
        pid = PIDLagrangian(kp=0.5, ki=0.01, kd=0.1)
        for _ in range(20):
            pid.update(constraint_value=2.0, threshold=1.0)
        v1 = pid.value
        for _ in range(20):
            pid.update(constraint_value=0.5, threshold=1.0)
        assert pid.value < v1


# ---------------------------------------------------------------------------
# CPOUpdate
# ---------------------------------------------------------------------------


class TestCPOUpdateComprehensive:
    def test_cg_identity(self):
        b = np.array([1.0, 2.0, 3.0])
        x = CPOUpdate._conjugate_gradient(lambda v: v, b, 10, 1e-10)
        np.testing.assert_allclose(x, b, atol=1e-6)

    def test_cg_spd(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x = CPOUpdate._conjugate_gradient(lambda v: A @ v, b, 20, 1e-10)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-4)

    def test_step(self):
        cpo = CPOUpdate(max_kl=0.01, cost_limit=25.0)
        params = np.array([1.0, 2.0, 3.0])
        updated = cpo.step(
            params,
            reward_grad=np.array([0.1, -0.1, 0.2]),
            cost_grad=np.array([0.01, 0.01, 0.01]),
            fisher_mvp_fn=lambda v: v * 1.1,
            current_cost=0.0,
        )
        assert updated.shape == params.shape

    def test_step_high_cost(self):
        cpo = CPOUpdate(max_kl=0.01, cost_limit=1.0)
        updated = cpo.step(
            np.array([1.0, 2.0]),
            reward_grad=np.array([0.5, 0.5]),
            cost_grad=np.array([1.0, 1.0]),
            fisher_mvp_fn=lambda v: v * 2.0,
            current_cost=5.0,
        )
        assert updated.shape == (2,)


# ---------------------------------------------------------------------------
# SafetyMonitor
# ---------------------------------------------------------------------------


class TestSafetyMonitorComprehensive:
    def test_initial(self):
        mon = SafetyMonitor()
        s = mon.get_statistics()
        assert s["total_steps"] == 0

    def test_record_step(self):
        mon = SafetyMonitor()
        mon.record_step(
            np.array([0.0, 0.0, 1.0, 0.0]), np.zeros(2), info={"min_obstacle_dist": 2.0}
        )
        s = mon.get_statistics()
        assert s["total_steps"] == 1
        assert s["min_obstacle_distance"] == pytest.approx(2.0)

    def test_record_violation_alert(self):
        mon = SafetyMonitor()
        alert = SafetyAlert(timestamp=1.0, severity=Severity.WARNING, constraint_name="speed")
        mon.record_step(np.zeros(4), np.zeros(2), info={"violation": alert})
        assert len(mon.get_violations()) == 1

    def test_record_violation_dict(self):
        mon = SafetyMonitor()
        mon.record_step(
            np.zeros(4),
            np.zeros(2),
            info={
                "violation": {
                    "timestamp": 1.0,
                    "severity": "warning",
                    "constraint_name": "speed",
                }
            },
        )
        assert len(mon.get_violations()) == 1

    def test_shield_intervention(self):
        mon = SafetyMonitor()
        mon.record_step(np.zeros(4), np.zeros(2), info={"shield_intervened": True})
        assert mon.get_statistics()["shield_interventions"] == 1

    def test_speed_stats(self):
        mon = SafetyMonitor()
        for v in [1.0, 2.0, 3.0]:
            # Speed is computed from action[:2], not state
            mon.record_step(np.zeros(4), np.array([v, 0.0]), info={})
        s = mon.get_statistics()
        assert s["mean_speed"] == pytest.approx(2.0)
        assert s["max_speed"] == pytest.approx(3.0)

    def test_reset(self):
        mon = SafetyMonitor()
        mon.record_step(np.zeros(4), np.zeros(2), info={})
        mon.reset()
        assert mon.get_statistics()["total_steps"] == 0

    def test_severity_breakdown(self):
        mon = SafetyMonitor()
        for sev in [Severity.INFO, Severity.WARNING, Severity.CRITICAL]:
            alert = SafetyAlert(timestamp=0.0, severity=sev, constraint_name="test")
            mon.record_step(np.zeros(4), np.zeros(2), info={"violation": alert})
        s = mon.get_statistics()
        assert s["violations_info"] == 1
        assert s["violations_warning"] == 1
        assert s["violations_critical"] == 1

    def test_no_info(self):
        mon = SafetyMonitor()
        mon.record_step(np.zeros(4), np.zeros(2), info=None)
        assert mon.get_statistics()["total_steps"] == 1

    def test_violation_rate(self):
        mon = SafetyMonitor()
        for i in range(10):
            info = {}
            if i < 3:
                info["violation"] = SafetyAlert(
                    timestamp=float(i), severity=Severity.WARNING, constraint_name="t"
                )
            mon.record_step(np.zeros(4), np.zeros(2), info=info)
        assert mon.get_statistics()["violation_rate"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Severity & SafetyAlert
# ---------------------------------------------------------------------------


class TestSeverityAndAlert:
    def test_severity_values(self):
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.CRITICAL == "critical"

    def test_alert_creation(self):
        a = SafetyAlert(
            timestamp=1.5,
            severity=Severity.WARNING,
            constraint_name="speed",
            details={"speed": 2.5},
        )
        assert a.severity == Severity.WARNING
        assert a.constraint_name == "speed"


# ---------------------------------------------------------------------------
# RiskEstimator
# ---------------------------------------------------------------------------


class TestRiskEstimatorComprehensive:
    def test_ttc_no_obstacles(self):
        re = RiskEstimator()
        assert re.time_to_collision(np.array([0, 0, 1, 0]), np.zeros((0, 4))) == float("inf")

    def test_ttc_head_on(self):
        re = RiskEstimator(agent_radius=0.25, default_obstacle_radius=0.25)
        ttc = re.time_to_collision(np.array([0, 0, 1, 0]), np.array([[5, 0, -1, 0]]))
        assert 0 < ttc < 5

    def test_ttc_no_collision(self):
        re = RiskEstimator()
        ttc = re.time_to_collision(np.array([0, 0, 1, 0]), np.array([[0, 10, 0, 1]]))
        assert ttc == float("inf") or ttc > 100

    def test_collision_prob(self):
        re = RiskEstimator()
        prob = re.collision_probability(
            np.array([0, 0, 1, 0]), np.array([[2, 0, 0, 0]]), dt=0.1, horizon=3.0, n_samples=50
        )
        assert 0.0 <= prob <= 1.0

    def test_collision_prob_no_obs(self):
        re = RiskEstimator()
        prob = re.collision_probability(
            np.array([0, 0, 1, 0]), np.zeros((0, 4)), dt=0.1, horizon=3.0, n_samples=10
        )
        assert prob == pytest.approx(0.0)

    def test_risk_field(self):
        re = RiskEstimator()
        field = re.risk_field(
            np.array([0, 0]),
            np.array([[3, 0, 0, 0], [0, 3, 0, 0]]),
            resolution=1.0,
            field_size=10.0,
        )
        assert field.ndim == 2
        assert np.all(field >= 0.0)

    def test_risk_field_no_obs(self):
        re = RiskEstimator()
        field = re.risk_field(np.array([0, 0]), np.zeros((0, 4)), resolution=1.0, field_size=6.0)
        assert np.allclose(field, 0.0)


# ---------------------------------------------------------------------------
# PredictiveRiskModel
# ---------------------------------------------------------------------------


class TestPredictiveRiskModelComprehensive:
    def test_const_vel(self):
        model = PredictiveRiskModel(prediction_fn=None)
        trajs = model.predict_trajectories(np.array([[0, 0, 1, 0], [5, 0, -1, 0]]), horizon=10)
        assert trajs.shape == (2, 10, 2)
        assert trajs[0, -1, 0] > trajs[0, 0, 0]

    def test_assess_risk(self):
        model = PredictiveRiskModel(prediction_fn=None)
        agent_traj = np.array([[i * 0.1, 0] for i in range(10)])
        obstacle_trajs = np.array([[[5 - i * 0.1, 0] for i in range(10)]])
        risk = model.assess_risk(agent_traj, obstacle_trajs)
        assert risk.shape == (10,)
        assert np.all((risk >= 0) & (risk <= 1))

    def test_risk_increases_converging(self):
        model = PredictiveRiskModel(prediction_fn=None)
        agent_traj = np.array([[i * 0.5, 0] for i in range(10)])
        obstacle_trajs = np.array([[[5 - i * 0.5, 0] for i in range(10)]])
        risk = model.assess_risk(agent_traj, obstacle_trajs)
        assert risk[-1] >= risk[0]

    def test_custom_prediction(self):
        model = PredictiveRiskModel(prediction_fn=lambda s, h: np.zeros((s.shape[0], h, 2)))
        trajs = model.predict_trajectories(np.array([[0, 0, 1, 0]]), horizon=5)
        assert trajs.shape == (1, 5, 2)
        np.testing.assert_allclose(trajs, 0.0)


# ---------------------------------------------------------------------------
# SafetyShield
# ---------------------------------------------------------------------------


class TestSafetyShieldComprehensive:
    class _Fast:
        def act(self, obs):
            return np.array([5.0, 5.0])

    class _Slow:
        def act(self, obs):
            return np.array([0.5, 0.0])

    def test_intercepts_unsafe(self):
        shield = SafetyShield(
            self._Fast(), ConstraintSet(constraints=[SpeedConstraint(max_speed=1.0)])
        )
        assert np.linalg.norm(shield.act(np.zeros(4))[:2]) <= 1.0 + 1e-3

    def test_intervention_rate(self):
        shield = SafetyShield(
            self._Fast(), ConstraintSet(constraints=[SpeedConstraint(max_speed=1.0)])
        )
        for _ in range(10):
            shield.act(np.zeros(4))
        assert shield.intervention_rate > 0

    def test_no_intervention(self):
        shield = SafetyShield(
            self._Slow(), ConstraintSet(constraints=[SpeedConstraint(max_speed=2.0)])
        )
        for _ in range(5):
            shield.act(np.zeros(4))
        assert shield.intervention_rate == pytest.approx(0.0)

    def test_reset(self):
        shield = SafetyShield(
            self._Fast(), ConstraintSet(constraints=[SpeedConstraint(max_speed=1.0)])
        )
        shield.act(np.zeros(4))
        shield.reset_stats()
        assert shield.interventions == 0

    def test_fallback(self):
        shield = SafetyShield(
            self._Fast(),
            ConstraintSet(constraints=[SpeedConstraint(max_speed=0.001)]),
            fallback_policy=lambda obs: np.array([0.0, 0.0]),
        )
        action = shield.act(np.zeros(4))
        assert np.linalg.norm(action) < 1.0


# ---------------------------------------------------------------------------
# CBFShield
# ---------------------------------------------------------------------------


class TestCBFShieldComprehensive:
    def _make(self):
        return CBFShield(
            barrier_fn=lambda x: 10 - np.linalg.norm(x[:2]),
            barrier_grad_fn=lambda x: np.concatenate(
                [-x[:2] / (np.linalg.norm(x[:2]) + 1e-10), np.zeros(max(0, len(x) - 2))]
            ),
            dynamics_fn=lambda x, u: x[:2] + u * 0.1,
            alpha=1.0,
            action_dim=2,
        )

    def test_safe_state(self):
        s = self._make()
        assert s.is_safe(np.array([0.0, 0.0]))

    def test_cbf_value(self):
        s = self._make()
        assert s.cbf_value(np.array([0.0, 0.0])) == pytest.approx(10.0)

    def test_filter_safe_action(self):
        s = self._make()
        a = s.filter_action(np.array([0.0, 0.0]), np.array([0.1, 0.0]))
        assert isinstance(a, np.ndarray)

    def test_unsafe_state(self):
        s = CBFShield(
            barrier_fn=lambda x: 1 - np.linalg.norm(x[:2]),
            barrier_grad_fn=lambda x: -x[:2] / (np.linalg.norm(x[:2]) + 1e-10),
            dynamics_fn=lambda x, u: x[:2] + u * 0.1,
            action_dim=2,
        )
        assert not s.is_safe(np.array([2.0, 0.0]))


# ---------------------------------------------------------------------------
# ReachabilityShield
# ---------------------------------------------------------------------------


class TestReachabilityShieldComprehensive:
    def _make(self):
        safe = np.ones((10, 10), dtype=bool)
        safe[0, :] = safe[-1, :] = safe[:, 0] = safe[:, -1] = False
        return ReachabilityShield(
            safe_set=safe,
            state_bounds=np.array([[0, 10], [0, 10]], dtype=float),
            dynamics_fn=lambda s, a: s + a * 0.1,
            fallback_action=np.array([0.0, 0.0]),
        )

    def test_safe_state(self):
        assert self._make().is_state_safe(np.array([5.0, 5.0]))

    def test_unsafe_boundary(self):
        assert not self._make().is_state_safe(np.array([0.0, 0.0]))

    def test_filter_safe(self):
        a = self._make().filter_action(np.array([5, 5.0]), np.array([0.1, 0.1]))
        np.testing.assert_allclose(a, [0.1, 0.1])

    def test_filter_unsafe_fallback(self):
        a = self._make().filter_action(np.array([0.5, 0.5]), np.array([-100, -100.0]))
        np.testing.assert_allclose(a, [0.0, 0.0])
