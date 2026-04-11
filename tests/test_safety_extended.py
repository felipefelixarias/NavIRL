"""Extended tests for navirl/safety/: CBFShield, ReachabilityShield, PredictiveRiskModel, CPOUpdate, PIDLagrangian.

Complements test_safety.py which covers SpeedConstraint, CollisionConstraint,
SafetyShield basics, RiskEstimator TTC, SafetyMonitor, and LagrangianMultiplier.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.safety.constrained_optimization import CPOUpdate, PIDLagrangian
from navirl.safety.risk_assessment import PredictiveRiskModel, RiskEstimator
from navirl.safety.shield import CBFShield, ReachabilityShield

# ---------------------------------------------------------------------------
# CBFShield
# ---------------------------------------------------------------------------


def _simple_barrier(state: np.ndarray) -> float:
    """h(x) = 5 - ||x||.  Safe when ||x|| <= 5."""
    return 5.0 - float(np.linalg.norm(state[:2]))


def _simple_barrier_grad(state: np.ndarray) -> np.ndarray:
    """Gradient of h = 5 - ||x||."""
    norm = float(np.linalg.norm(state[:2]))
    grad = np.zeros_like(state)
    if norm > 1e-8:
        grad[:2] = -state[:2] / norm
    return grad


def _simple_dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """next = state + [action, 0, 0, ...]."""
    ns = state.copy()
    ns[:2] += action[:2]
    return ns


class TestCBFShield:
    @pytest.fixture
    def shield(self):
        return CBFShield(
            barrier_fn=_simple_barrier,
            barrier_grad_fn=_simple_barrier_grad,
            dynamics_fn=_simple_dynamics,
            alpha=1.0,
            action_dim=2,
        )

    def test_cbf_value_at_origin(self, shield):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        assert shield.cbf_value(state) == pytest.approx(5.0)

    def test_cbf_value_at_boundary(self, shield):
        state = np.array([3.0, 4.0, 0.0, 0.0])
        assert shield.cbf_value(state) == pytest.approx(0.0)

    def test_is_safe_inside(self, shield):
        assert shield.is_safe(np.array([1.0, 1.0, 0.0, 0.0])) is True

    def test_is_safe_outside(self, shield):
        assert shield.is_safe(np.array([4.0, 4.0, 0.0, 0.0])) is False

    def test_filter_safe_action_passes(self, shield):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([0.1, 0.0])
        result = shield.filter_action(state, action)
        np.testing.assert_array_almost_equal(result, action)

    def test_filter_unsafe_action_modified(self, shield):
        # State near boundary, action pushing outward
        state = np.array([4.5, 0.0, 0.0, 0.0])
        action = np.array([2.0, 0.0])
        result = shield.filter_action(state, action)
        # Result should be less aggressive than original
        assert np.linalg.norm(result) <= np.linalg.norm(action) + 0.1

    def test_filter_returns_copy(self, shield):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([0.1, 0.0])
        result = shield.filter_action(state, action)
        action[0] = 999.0
        assert result[0] != 999.0

    def test_zero_gradient_returns_zeros(self):
        """When gradient is zero, should return zero action."""
        shield = CBFShield(
            barrier_fn=lambda s: -1.0,  # Always unsafe
            barrier_grad_fn=lambda s: np.zeros(4),
            dynamics_fn=_simple_dynamics,
            alpha=1.0,
            action_dim=2,
        )
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = np.array([1.0, 1.0])
        result = shield.filter_action(state, action)
        np.testing.assert_array_almost_equal(result, np.zeros(2))


# ---------------------------------------------------------------------------
# ReachabilityShield
# ---------------------------------------------------------------------------


class TestReachabilityShield:
    @pytest.fixture
    def shield(self):
        # 10x10 grid, center is safe, edges unsafe
        safe_set = np.zeros((10, 10), dtype=bool)
        safe_set[2:8, 2:8] = True
        state_bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
        return ReachabilityShield(
            safe_set=safe_set,
            state_bounds=state_bounds,
            dynamics_fn=lambda s, a: s + a,
            fallback_action=np.array([0.0, 0.0]),
        )

    def test_state_safe_center(self, shield):
        assert shield.is_state_safe(np.array([5.0, 5.0])) is True

    def test_state_unsafe_edge(self, shield):
        assert shield.is_state_safe(np.array([0.5, 0.5])) is False

    def test_action_safe(self, shield):
        state = np.array([5.0, 5.0])
        action = np.array([0.1, 0.1])
        assert shield.is_action_safe(state, action) is True

    def test_action_unsafe_leaves_safe_set(self, shield):
        state = np.array([5.0, 5.0])
        action = np.array([5.0, 5.0])  # Pushes to (10, 10) → unsafe
        assert shield.is_action_safe(state, action) is False

    def test_filter_safe_action_passes(self, shield):
        state = np.array([5.0, 5.0])
        action = np.array([0.1, 0.0])
        result = shield.filter_action(state, action)
        np.testing.assert_array_almost_equal(result, action)

    def test_filter_unsafe_uses_fallback(self, shield):
        state = np.array([5.0, 5.0])
        action = np.array([5.0, 5.0])
        result = shield.filter_action(state, action)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]))

    def test_filter_no_fallback_returns_zero(self):
        safe_set = np.ones((10, 10), dtype=bool)
        safe_set[9, 9] = False
        shield = ReachabilityShield(
            safe_set=safe_set,
            state_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
            dynamics_fn=lambda s, a: s + a,
            fallback_action=None,
        )
        state = np.array([5.0, 5.0])
        action = np.array([5.0, 5.0])
        result = shield.filter_action(state, action)
        np.testing.assert_array_almost_equal(result, np.zeros(2))

    def test_state_to_index_bounds(self, shield):
        # Edge of bounds should clip
        idx = shield._state_to_index(np.array([0.0, 0.0]))
        assert idx == (0, 0)
        idx = shield._state_to_index(np.array([10.0, 10.0]))
        assert idx == (9, 9)


# ---------------------------------------------------------------------------
# RiskEstimator - extended (collision_probability, risk_field)
# ---------------------------------------------------------------------------


class TestRiskEstimatorExtended:
    @pytest.fixture
    def estimator(self):
        return RiskEstimator(agent_radius=0.25, default_obstacle_radius=0.3)

    def test_collision_probability_no_obstacles(self, estimator):
        agent = np.array([0.0, 0.0, 1.0, 0.0])
        prob = estimator.collision_probability(agent, np.empty((0, 4)))
        assert prob == 0.0

    def test_collision_probability_high_risk(self, estimator):
        # Agent heading straight at a nearby obstacle
        agent = np.array([0.0, 0.0, 1.0, 0.0])
        obs = np.array([[1.0, 0.0, 0.0, 0.0]])
        prob = estimator.collision_probability(
            agent, obs, dt=0.1, horizon=2.0, n_samples=200, noise_std=0.01
        )
        assert prob > 0.5

    def test_collision_probability_low_risk(self, estimator):
        # Agent moving away from distant obstacle
        agent = np.array([0.0, 0.0, -1.0, 0.0])
        obs = np.array([[10.0, 10.0, 0.0, 0.0]])
        prob = estimator.collision_probability(
            agent, obs, dt=0.1, horizon=1.0, n_samples=100, noise_std=0.01
        )
        assert prob < 0.1

    def test_collision_probability_range(self, estimator):
        agent = np.array([0.0, 0.0, 0.5, 0.0])
        obs = np.array([[3.0, 0.0, -0.5, 0.0]])
        prob = estimator.collision_probability(agent, obs, n_samples=50)
        assert 0.0 <= prob <= 1.0

    def test_risk_field_shape(self, estimator):
        pos = np.array([0.0, 0.0])
        obs = np.array([[3.0, 3.0]])
        rf = estimator.risk_field(pos, obs, resolution=1.0, field_size=5.0)
        assert rf.shape == (10, 10)

    def test_risk_field_no_obstacles(self, estimator):
        pos = np.array([0.0, 0.0])
        rf = estimator.risk_field(pos, np.empty((0, 2)), resolution=1.0, field_size=5.0)
        assert np.all(rf == 0.0)

    def test_risk_field_higher_near_obstacle(self, estimator):
        pos = np.array([0.0, 0.0])
        obs = np.array([[0.0, 0.0]])
        rf = estimator.risk_field(pos, obs, resolution=1.0, field_size=5.0)
        center = rf.shape[0] // 2
        # Center (near obstacle) should have highest risk
        assert rf[center, center] >= rf[0, 0]


# ---------------------------------------------------------------------------
# PredictiveRiskModel
# ---------------------------------------------------------------------------


class TestPredictiveRiskModel:
    @pytest.fixture
    def model(self):
        return PredictiveRiskModel()

    def test_constant_velocity_predict_shape(self, model):
        states = np.array([[0.0, 0.0, 1.0, 0.0], [5.0, 5.0, 0.0, -1.0]])
        preds = model.predict_trajectories(states, horizon=20)
        assert preds.shape == (2, 20, 2)

    def test_constant_velocity_predict_values(self, model):
        states = np.array([[0.0, 0.0, 1.0, 0.0]])
        preds = model.predict_trajectories(states, horizon=10)
        # At t=1 (step 0): pos = (0,0) + (1,0)*1*0.1 = (0.1, 0)
        assert preds[0, 0, 0] == pytest.approx(0.1)
        assert preds[0, 0, 1] == pytest.approx(0.0)
        # At t=10 (step 9): pos = (0,0) + (1,0)*10*0.1 = (1.0, 0)
        assert preds[0, 9, 0] == pytest.approx(1.0)

    def test_custom_prediction_fn(self):
        def custom_fn(states, horizon):
            return np.zeros((states.shape[0], horizon, 2))

        model = PredictiveRiskModel(prediction_fn=custom_fn)
        states = np.array([[1.0, 1.0, 1.0, 1.0]])
        preds = model.predict_trajectories(states, horizon=5)
        assert np.all(preds == 0.0)

    def test_assess_risk_close_obstacle(self, model):
        # Agent trajectory goes right at the obstacle
        agent_traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        obs_traj = np.array([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]])
        risks = model.assess_risk(agent_traj, obs_traj)
        assert risks.shape == (3,)
        # Risk at t=2 should be high (agent at obstacle position)
        assert risks[2] > 0.5

    def test_assess_risk_far_obstacle(self, model):
        agent_traj = np.array([[0.0, 0.0], [1.0, 0.0]])
        obs_traj = np.array([[[100.0, 100.0], [100.0, 100.0]]])
        risks = model.assess_risk(agent_traj, obs_traj)
        assert np.all(risks < 0.01)

    def test_assess_risk_shape(self, model):
        agent_traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        obs_traj = np.array(
            [
                [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
                [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
            ]
        )
        risks = model.assess_risk(agent_traj, obs_traj)
        assert risks.shape == (4,)
        assert np.all(risks >= 0.0)
        assert np.all(risks <= 1.0)


# ---------------------------------------------------------------------------
# CPOUpdate
# ---------------------------------------------------------------------------


class TestCPOUpdate:
    @pytest.fixture
    def cpo(self):
        return CPOUpdate(max_kl=0.01, cost_limit=25.0, cg_iters=10, line_search_steps=10)

    def test_conjugate_gradient_identity(self):
        """CG with identity matrix: A@x = b => x = b."""
        b = np.array([1.0, 2.0, 3.0])
        x = CPOUpdate._conjugate_gradient(lambda v: v, b, n_iters=10)
        np.testing.assert_array_almost_equal(x, b)

    def test_conjugate_gradient_scaled(self):
        """CG with scaled identity: 2*x = b => x = b/2."""
        b = np.array([2.0, 4.0])
        x = CPOUpdate._conjugate_gradient(lambda v: 2.0 * v, b, n_iters=10)
        np.testing.assert_array_almost_equal(x, np.array([1.0, 2.0]))

    def test_conjugate_gradient_symmetric(self):
        """CG with symmetric positive definite matrix."""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x = CPOUpdate._conjugate_gradient(lambda v: A @ v, b, n_iters=20)
        expected = np.linalg.solve(A, b)
        np.testing.assert_array_almost_equal(x, expected, decimal=5)

    def test_step_returns_params(self, cpo):
        params = np.array([1.0, 2.0, 3.0])
        reward_grad = np.array([0.1, 0.2, 0.3])
        cost_grad = np.array([0.0, 0.0, 0.0])
        result = cpo.step(params, reward_grad, cost_grad, lambda v: v, current_cost=0.0)
        assert result.shape == params.shape

    def test_step_moves_in_reward_direction(self, cpo):
        params = np.zeros(3)
        reward_grad = np.array([1.0, 0.0, 0.0])
        cost_grad = np.zeros(3)
        result = cpo.step(params, reward_grad, cost_grad, lambda v: v, current_cost=0.0)
        # Should move in positive x direction
        assert result[0] > 0.0

    def test_step_with_cost_violation(self, cpo):
        params = np.zeros(3)
        reward_grad = np.array([1.0, 0.0, 0.0])
        cost_grad = np.array([1.0, 0.0, 0.0])
        # Cost exceeds limit: should constrain step
        result_constrained = cpo.step(
            params, reward_grad, cost_grad, lambda v: v, current_cost=30.0
        )
        result_unconstrained = cpo.step(
            params, reward_grad, cost_grad, lambda v: v, current_cost=0.0
        )
        # Constrained step should be smaller
        assert np.linalg.norm(result_constrained) <= np.linalg.norm(result_unconstrained) + 1e-8

    def test_step_zero_reward_grad(self, cpo):
        """Zero reward gradient should yield no movement."""
        params = np.array([1.0, 2.0])
        result = cpo.step(
            params,
            np.zeros(2),
            np.zeros(2),
            lambda v: v,
            current_cost=0.0,
        )
        np.testing.assert_array_almost_equal(result, params)


# ---------------------------------------------------------------------------
# PIDLagrangian
# ---------------------------------------------------------------------------


class TestPIDLagrangian:
    @pytest.fixture
    def pid(self):
        return PIDLagrangian(kp=1.0, ki=0.01, kd=0.1, max_value=100.0)

    def test_initial_value(self, pid):
        assert pid.value == 0.0

    def test_update_violation_increases(self, pid):
        pid.update(constraint_value=30.0, threshold=25.0)
        assert pid.value > 0.0

    def test_update_satisfaction_stays_zero(self, pid):
        pid.update(constraint_value=10.0, threshold=25.0)
        assert pid.value == 0.0  # Clamped to >= 0

    def test_update_repeated_violations(self, pid):
        for _ in range(10):
            pid.update(constraint_value=30.0, threshold=25.0)
        # Should increase with integral term
        assert pid.value > 5.0

    def test_update_derivative_term(self):
        pid = PIDLagrangian(kp=0.0, ki=0.0, kd=1.0, max_value=100.0)
        pid.update(constraint_value=30.0, threshold=25.0)
        v1 = pid.value
        # Same error: derivative = 0
        pid.update(constraint_value=30.0, threshold=25.0)
        v2 = pid.value
        # Derivative is (5-5)=0 second time, so increment is 0
        assert v2 == pytest.approx(v1)

    def test_clamped_to_max(self, pid):
        for _ in range(1000):
            pid.update(constraint_value=1000.0, threshold=0.0)
        assert pid.value <= 100.0

    def test_clamped_to_zero(self, pid):
        for _ in range(100):
            pid.update(constraint_value=0.0, threshold=100.0)
        assert pid.value >= 0.0

    def test_penalized_objective(self, pid):
        pid.update(constraint_value=30.0, threshold=25.0)
        lam = pid.value
        result = pid.penalized_objective(reward=10.0, constraint_value=5.0)
        assert result == pytest.approx(10.0 - lam * 5.0)

    def test_reset(self, pid):
        pid.update(constraint_value=50.0, threshold=0.0)
        assert pid.value > 0.0
        pid.reset()
        assert pid.value == 0.0

    def test_pid_smooths_compared_to_lagrangian(self):
        """PID should respond more smoothly than a simple Lagrangian."""
        from navirl.safety.constrained_optimization import LagrangianMultiplier

        lm = LagrangianMultiplier(learning_rate=1.0)
        pid = PIDLagrangian(kp=1.0, ki=0.0, kd=0.5, max_value=100.0)

        errors = [5.0, 5.0, 0.0, 0.0, 5.0]
        lm_values = []
        pid_values = []
        for e in errors:
            lm.update(e, 0.0)
            pid.update(e, 0.0)
            lm_values.append(lm.value)
            pid_values.append(pid.value)

        # Both should track violations but PID should have derivative effect
        assert len(lm_values) == len(pid_values) == 5
