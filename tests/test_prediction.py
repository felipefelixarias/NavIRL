"""Tests for navirl/prediction/ module: predictors, prediction result."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.prediction.base import PredictionResult, TrajectoryPredictor
from navirl.prediction.constant_velocity import (
    ConstantVelocityPredictor,
    KalmanPredictor,
    LinearPredictor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_obs():
    """Observed straight-line trajectory: 8 points along x-axis."""
    return np.column_stack([np.arange(8, dtype=np.float64), np.zeros(8)])


@pytest.fixture
def curved_obs():
    """Observed curved trajectory: circular arc."""
    t = np.linspace(0, np.pi / 2, 10)
    return np.column_stack([np.cos(t), np.sin(t)])


@pytest.fixture
def short_obs():
    """Minimal 2-point observation."""
    return np.array([[0.0, 0.0], [1.0, 0.0]])


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_basic_construction(self):
        trajs = np.random.randn(5, 12, 2)
        probs = np.ones(5) / 5
        ts = np.arange(12) * 0.4
        result = PredictionResult(trajectories=trajs, probabilities=probs, timestamps=ts)
        assert result.num_samples == 5
        assert result.horizon == 12

    def test_best_trajectory(self):
        trajs = np.zeros((3, 5, 2))
        trajs[1] = 1.0  # make sample 1 distinct
        probs = np.array([0.1, 0.8, 0.1])
        ts = np.arange(5) * 0.4
        result = PredictionResult(trajectories=trajs, probabilities=probs, timestamps=ts)
        best = result.best_trajectory()
        np.testing.assert_allclose(best, 1.0)

    def test_mean_trajectory(self):
        trajs = np.zeros((2, 4, 2))
        trajs[0] = 2.0
        trajs[1] = 4.0
        probs = np.array([0.5, 0.5])
        ts = np.arange(4) * 0.4
        result = PredictionResult(trajectories=trajs, probabilities=probs, timestamps=ts)
        mean = result.mean_trajectory()
        np.testing.assert_allclose(mean, 3.0)

    def test_single_sample(self):
        trajs = np.random.randn(1, 10, 2)
        probs = np.array([1.0])
        ts = np.arange(10) * 0.4
        result = PredictionResult(trajectories=trajs, probabilities=probs, timestamps=ts)
        assert result.num_samples == 1
        np.testing.assert_array_equal(result.best_trajectory(), trajs[0])

    def test_metadata(self):
        result = PredictionResult(
            trajectories=np.zeros((1, 1, 2)),
            probabilities=np.array([1.0]),
            timestamps=np.array([0.4]),
            metadata={"model": "cv"},
        )
        assert result.metadata["model"] == "cv"


# ---------------------------------------------------------------------------
# ConstantVelocityPredictor
# ---------------------------------------------------------------------------

class TestConstantVelocityPredictor:
    def test_basic_prediction(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=5, dt=0.4, num_samples=1)
        result = pred.predict(straight_obs)
        assert result.trajectories.shape == (1, 5, 2)
        assert result.probabilities.shape == (1,)
        assert result.timestamps.shape == (5,)

    def test_straight_line_extrapolation(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=3, dt=1.0, num_samples=1)
        result = pred.predict(straight_obs)
        # velocity = [1, 0], so predictions should extend along x
        expected_x = straight_obs[-1, 0] + np.arange(1, 4) * 1.0
        np.testing.assert_allclose(result.trajectories[0, :, 0], expected_x, atol=1e-10)
        np.testing.assert_allclose(result.trajectories[0, :, 1], 0.0, atol=1e-10)

    def test_multiple_samples(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=5, dt=0.4, num_samples=10)
        result = pred.predict(straight_obs)
        assert result.trajectories.shape == (10, 5, 2)
        assert np.allclose(result.probabilities, 0.1)

    def test_with_noise(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=5, dt=0.4, num_samples=10, noise_std=0.5)
        result = pred.predict(straight_obs)
        # With noise, samples should differ
        assert not np.allclose(result.trajectories[0], result.trajectories[1])

    def test_too_short_trajectory(self):
        pred = ConstantVelocityPredictor(horizon=5)
        obs = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="at least 2"):
            pred.predict(obs)

    @pytest.mark.parametrize("horizon", [1, 5, 20, 50])
    def test_various_horizons(self, straight_obs, horizon):
        pred = ConstantVelocityPredictor(horizon=horizon, dt=0.4, num_samples=1)
        result = pred.predict(straight_obs)
        assert result.horizon == horizon

    def test_timestamps(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=5, dt=0.2)
        result = pred.predict(straight_obs)
        expected_ts = np.arange(1, 6) * 0.2
        np.testing.assert_allclose(result.timestamps, expected_ts)


# ---------------------------------------------------------------------------
# LinearPredictor
# ---------------------------------------------------------------------------

class TestLinearPredictor:
    def test_basic_prediction(self, straight_obs):
        pred = LinearPredictor(horizon=5, dt=0.4, num_samples=1)
        result = pred.predict(straight_obs)
        assert result.trajectories.shape == (1, 5, 2)

    def test_linear_extrapolation(self, straight_obs):
        pred = LinearPredictor(horizon=3, dt=1.0, fit_window=5, num_samples=1)
        result = pred.predict(straight_obs)
        # Should extrapolate approximately linearly
        diffs = np.diff(result.trajectories[0, :, 0])
        np.testing.assert_allclose(diffs, diffs[0], atol=0.5)

    def test_fit_window(self, straight_obs):
        pred = LinearPredictor(horizon=5, dt=0.4, fit_window=3, num_samples=1)
        result = pred.predict(straight_obs)
        assert result.trajectories.shape == (1, 5, 2)

    def test_too_short_trajectory(self):
        pred = LinearPredictor(horizon=5)
        obs = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError, match="at least 2"):
            pred.predict(obs)

    def test_with_noise(self, straight_obs):
        pred = LinearPredictor(horizon=5, dt=0.4, num_samples=5, noise_std=1.0)
        result = pred.predict(straight_obs)
        assert result.trajectories.shape == (5, 5, 2)


# ---------------------------------------------------------------------------
# KalmanPredictor
# ---------------------------------------------------------------------------

class TestKalmanPredictor:
    def test_basic_prediction(self, straight_obs):
        pred = KalmanPredictor(horizon=5, dt=0.4, num_samples=10)
        result = pred.predict(straight_obs)
        assert result.trajectories.shape == (10, 5, 2)

    def test_constant_velocity_input(self, straight_obs):
        pred = KalmanPredictor(horizon=3, dt=1.0, num_samples=1)
        result = pred.predict(straight_obs)
        # Mean trajectory should roughly continue linearly
        mean = result.mean_trajectory()
        # x should be increasing
        assert mean[-1, 0] > straight_obs[-1, 0]

    def test_short_input(self, short_obs):
        pred = KalmanPredictor(horizon=5, dt=0.4, num_samples=5)
        result = pred.predict(short_obs)
        assert result.trajectories.shape == (5, 5, 2)

    def test_too_short_trajectory(self):
        pred = KalmanPredictor(horizon=5)
        with pytest.raises(ValueError, match="at least 2"):
            pred.predict(np.array([[0.0, 0.0]]))

    def test_multiple_samples_differ(self, straight_obs):
        pred = KalmanPredictor(horizon=5, dt=0.4, num_samples=20, process_noise=1.0)
        result = pred.predict(straight_obs)
        # Samples should not be identical when process noise is high
        assert not np.allclose(result.trajectories[0], result.trajectories[1])

    def test_probabilities_sum_to_one(self, straight_obs):
        pred = KalmanPredictor(horizon=5, dt=0.4, num_samples=10)
        result = pred.predict(straight_obs)
        assert result.probabilities.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class TestTrajectoryPredictorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TrajectoryPredictor()

    def test_subclass_implementation(self, straight_obs):
        class DummyPredictor(TrajectoryPredictor):
            def predict(self, observed_trajectory, context=None):
                n = 5
                trajs = np.zeros((1, n, 2))
                return PredictionResult(
                    trajectories=trajs,
                    probabilities=np.array([1.0]),
                    timestamps=np.arange(n) * 0.4,
                )
        pred = DummyPredictor()
        result = pred.predict(straight_obs)
        assert result.num_samples == 1


# ---------------------------------------------------------------------------
# Trajectory sampling and consistency
# ---------------------------------------------------------------------------

class TestTrajectorySampling:
    def test_cv_deterministic_without_noise(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=5, dt=0.4, num_samples=5, noise_std=0.0)
        result = pred.predict(straight_obs)
        # All samples should be identical without noise
        for i in range(1, 5):
            np.testing.assert_array_equal(
                result.trajectories[0], result.trajectories[i]
            )

    def test_increasing_noise_with_horizon(self, straight_obs):
        pred = ConstantVelocityPredictor(horizon=20, dt=0.4, num_samples=100, noise_std=0.1)
        result = pred.predict(straight_obs)
        # Variance should increase with prediction horizon
        var_early = np.var(result.trajectories[:, 0, :], axis=0)
        var_late = np.var(result.trajectories[:, -1, :], axis=0)
        assert np.sum(var_late) > np.sum(var_early)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPredictionEdgeCases:
    def test_two_point_observation(self, short_obs):
        pred = ConstantVelocityPredictor(horizon=1, dt=0.4, num_samples=1)
        result = pred.predict(short_obs)
        assert result.trajectories.shape == (1, 1, 2)

    def test_identical_observed_points(self):
        obs = np.array([[5.0, 5.0], [5.0, 5.0]])
        pred = ConstantVelocityPredictor(horizon=3, dt=0.4, num_samples=1, noise_std=0.0)
        result = pred.predict(obs)
        # Zero velocity => predictions should stay at same point
        np.testing.assert_allclose(result.trajectories[0], [[5, 5], [5, 5], [5, 5]])

    def test_negative_positions(self):
        obs = np.array([[-10.0, -20.0], [-9.0, -19.0]])
        pred = ConstantVelocityPredictor(horizon=2, dt=1.0, num_samples=1, noise_std=0.0)
        result = pred.predict(obs)
        np.testing.assert_allclose(result.trajectories[0, 0], [-8.0, -18.0])

    def test_large_velocity(self):
        obs = np.array([[0.0, 0.0], [1000.0, 0.0]])
        pred = ConstantVelocityPredictor(horizon=1, dt=1.0, num_samples=1, noise_std=0.0)
        result = pred.predict(obs)
        np.testing.assert_allclose(result.trajectories[0, 0], [2000.0, 0.0])
