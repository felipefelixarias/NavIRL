from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from navirl.prediction.base import PredictionResult, TrajectoryPredictor


class ConstantVelocityPredictor(TrajectoryPredictor):
    """Predict future positions assuming constant velocity.

    The velocity is estimated from the last two observed positions and
    propagated forward for *horizon* steps.
    """

    def __init__(
        self,
        horizon: int = 12,
        dt: float = 0.4,
        num_samples: int = 1,
        noise_std: float = 0.0,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.noise_std = noise_std

    def predict(
        self,
        observed_trajectory: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        if observed_trajectory.shape[0] < 2:
            raise ValueError("Need at least 2 observed positions for constant velocity.")

        last_pos = observed_trajectory[-1]  # (2,)
        velocity = observed_trajectory[-1] - observed_trajectory[-2]  # (2,)

        timestamps = np.arange(1, self.horizon + 1) * self.dt

        trajectories = np.zeros((self.num_samples, self.horizon, 2))
        for s in range(self.num_samples):
            for t in range(self.horizon):
                noise = (
                    np.random.randn(2) * self.noise_std * (t + 1)
                    if self.noise_std > 0
                    else 0.0
                )
                trajectories[s, t] = last_pos + velocity * (t + 1) + noise

        probabilities = np.ones(self.num_samples) / self.num_samples

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )


class LinearPredictor(TrajectoryPredictor):
    """Linear extrapolation from recent positions.

    Fits a least-squares line through the last *fit_window* observed
    positions and extrapolates forward.
    """

    def __init__(
        self,
        horizon: int = 12,
        dt: float = 0.4,
        fit_window: int = 5,
        num_samples: int = 1,
        noise_std: float = 0.0,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.fit_window = fit_window
        self.num_samples = num_samples
        self.noise_std = noise_std

    def predict(
        self,
        observed_trajectory: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        obs = observed_trajectory[-self.fit_window :]
        T_obs = obs.shape[0]
        if T_obs < 2:
            raise ValueError("Need at least 2 observed positions for linear prediction.")

        t_obs = np.arange(T_obs, dtype=np.float64)

        # Fit line for x and y independently.
        coeffs_x = np.polyfit(t_obs, obs[:, 0], deg=1)
        coeffs_y = np.polyfit(t_obs, obs[:, 1], deg=1)

        t_pred = np.arange(T_obs, T_obs + self.horizon, dtype=np.float64)
        timestamps = np.arange(1, self.horizon + 1) * self.dt

        trajectories = np.zeros((self.num_samples, self.horizon, 2))
        for s in range(self.num_samples):
            x_pred = np.polyval(coeffs_x, t_pred)
            y_pred = np.polyval(coeffs_y, t_pred)
            if self.noise_std > 0:
                scale = self.noise_std * np.arange(1, self.horizon + 1)
                x_pred += np.random.randn(self.horizon) * scale
                y_pred += np.random.randn(self.horizon) * scale
            trajectories[s, :, 0] = x_pred
            trajectories[s, :, 1] = y_pred

        probabilities = np.ones(self.num_samples) / self.num_samples

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )


@dataclass
class _KalmanState:
    """Internal Kalman filter state: [x, y, vx, vy, ax, ay]."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(6))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6))


class KalmanPredictor(TrajectoryPredictor):
    """Kalman filter-based prediction with a constant-acceleration model.

    State vector: ``[x, y, vx, vy, ax, ay]``.
    """

    def __init__(
        self,
        horizon: int = 12,
        dt: float = 0.4,
        num_samples: int = 20,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Transition matrix for constant-acceleration model.
        self.F = np.eye(6)
        self.F[0, 2] = dt
        self.F[0, 4] = 0.5 * dt**2
        self.F[1, 3] = dt
        self.F[1, 5] = 0.5 * dt**2
        self.F[2, 4] = dt
        self.F[3, 5] = dt

        # Observation matrix (we only observe position).
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Process noise covariance.
        self.Q = np.eye(6) * process_noise

        # Measurement noise covariance.
        self.R = np.eye(2) * measurement_noise

    def _initialize_state(self, observed: np.ndarray) -> _KalmanState:
        state = _KalmanState()
        state.mean[:2] = observed[-1]
        if observed.shape[0] >= 2:
            vel = (observed[-1] - observed[-2]) / self.dt
            state.mean[2:4] = vel
        if observed.shape[0] >= 3:
            vel_prev = (observed[-2] - observed[-3]) / self.dt
            acc = (state.mean[2:4] - vel_prev) / self.dt
            state.mean[4:6] = acc
        state.covariance = np.eye(6) * 1.0
        return state

    def _kalman_update(self, state: _KalmanState, measurement: np.ndarray) -> _KalmanState:
        y = measurement - self.H @ state.mean
        S = self.H @ state.covariance @ self.H.T + self.R
        K = state.covariance @ self.H.T @ np.linalg.inv(S)
        state.mean = state.mean + K @ y
        state.covariance = (np.eye(6) - K @ self.H) @ state.covariance
        return state

    def _kalman_predict_step(self, state: _KalmanState) -> _KalmanState:
        state.mean = self.F @ state.mean
        state.covariance = self.F @ state.covariance @ self.F.T + self.Q
        return state

    def predict(
        self,
        observed_trajectory: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        if observed_trajectory.shape[0] < 2:
            raise ValueError("Need at least 2 observed positions for Kalman prediction.")

        # Run filter over observed data.
        state = self._initialize_state(observed_trajectory[:3])
        for i in range(3, observed_trajectory.shape[0]):
            state = self._kalman_predict_step(state)
            state = self._kalman_update(state, observed_trajectory[i])

        # Propagate forward and sample from the predictive distribution.
        timestamps = np.arange(1, self.horizon + 1) * self.dt
        mean_traj = np.zeros((self.horizon, 6))
        cov_traj = np.zeros((self.horizon, 6, 6))

        pred_state = _KalmanState(
            mean=state.mean.copy(), covariance=state.covariance.copy()
        )
        for t in range(self.horizon):
            pred_state = self._kalman_predict_step(pred_state)
            mean_traj[t] = pred_state.mean
            cov_traj[t] = pred_state.covariance

        # Sample trajectories from the Gaussian predictions.
        trajectories = np.zeros((self.num_samples, self.horizon, 2))
        for t in range(self.horizon):
            pos_mean = mean_traj[t, :2]
            pos_cov = cov_traj[t, :2, :2]
            # Ensure symmetry / positive-definiteness.
            pos_cov = 0.5 * (pos_cov + pos_cov.T) + 1e-6 * np.eye(2)
            samples = np.random.multivariate_normal(pos_mean, pos_cov, size=self.num_samples)
            trajectories[:, t, :] = samples

        probabilities = np.ones(self.num_samples) / self.num_samples

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )
