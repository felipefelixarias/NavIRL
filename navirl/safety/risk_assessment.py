"""Risk assessment for navigation.

Provides collision-risk estimation via time-to-collision, probabilistic
collision models, and trajectory-prediction-based risk scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# RiskEstimator
# ---------------------------------------------------------------------------

class RiskEstimator:
    """Estimates collision risk using geometric and probabilistic methods.

    Parameters
    ----------
    agent_radius : float
        Collision radius of the controlled agent.
    default_obstacle_radius : float
        Default radius assumed for obstacles if not provided explicitly.
    """

    def __init__(
        self,
        agent_radius: float = 0.25,
        default_obstacle_radius: float = 0.3,
    ) -> None:
        self.agent_radius = agent_radius
        self.default_obstacle_radius = default_obstacle_radius

    # ------------------------------------------------------------------
    # Time-to-collision
    # ------------------------------------------------------------------

    def time_to_collision(
        self,
        agent_state: np.ndarray,
        obstacle_states: np.ndarray,
    ) -> float:
        """Compute the minimum time-to-collision with any obstacle.

        Assumes constant-velocity motion for both agent and obstacles.

        Parameters
        ----------
        agent_state : np.ndarray
            ``[x, y, vx, vy]`` of the agent.
        obstacle_states : np.ndarray
            Shape ``(N, 4)`` – ``[x, y, vx, vy]`` per obstacle.

        Returns
        -------
        float
            Minimum TTC in seconds.  ``np.inf`` when no collision is predicted.
        """
        if obstacle_states.shape[0] == 0:
            return float("inf")

        combined_radius = self.agent_radius + self.default_obstacle_radius
        min_ttc = float("inf")

        agent_pos = agent_state[:2]
        agent_vel = agent_state[2:4]

        for obs in obstacle_states:
            dp = obs[:2] - agent_pos
            dv = obs[2:4] - agent_vel

            a = float(dv @ dv)
            b = 2.0 * float(dp @ dv)
            c = float(dp @ dp) - combined_radius ** 2

            if a < 1e-12:
                # No relative motion; skip or already colliding.
                if c <= 0:
                    return 0.0
                continue

            disc = b * b - 4.0 * a * c
            if disc < 0:
                continue

            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            for t in (t1, t2):
                if 0.0 <= t < min_ttc:
                    min_ttc = t

        return min_ttc

    # ------------------------------------------------------------------
    # Collision probability (Monte-Carlo)
    # ------------------------------------------------------------------

    def collision_probability(
        self,
        agent_state: np.ndarray,
        obstacle_states: np.ndarray,
        dt: float = 0.1,
        horizon: float = 3.0,
        n_samples: int = 100,
        noise_std: float = 0.1,
    ) -> float:
        """Estimate collision probability via Monte-Carlo forward simulation.

        Both agent and obstacles are forward-simulated with additive Gaussian
        noise on their velocities.

        Parameters
        ----------
        agent_state : np.ndarray
            ``[x, y, vx, vy]``.
        obstacle_states : np.ndarray
            ``(N, 4)`` with ``[x, y, vx, vy]``.
        dt : float
            Simulation timestep.
        horizon : float
            Look-ahead horizon (seconds).
        n_samples : int
            Number of Monte-Carlo samples.
        noise_std : float
            Standard deviation of velocity noise.

        Returns
        -------
        float
            Estimated probability in ``[0, 1]``.
        """
        if obstacle_states.shape[0] == 0:
            return 0.0

        combined_radius = self.agent_radius + self.default_obstacle_radius
        steps = max(1, int(horizon / dt))
        collisions = 0

        for _ in range(n_samples):
            a_pos = agent_state[:2].copy()
            a_vel = agent_state[2:4].copy()
            o_pos = obstacle_states[:, :2].copy()
            o_vel = obstacle_states[:, 2:4].copy()
            hit = False
            for _t in range(steps):
                a_vel_noisy = a_vel + np.random.randn(2) * noise_std
                o_vel_noisy = o_vel + np.random.randn(*o_vel.shape) * noise_std
                a_pos = a_pos + a_vel_noisy * dt
                o_pos = o_pos + o_vel_noisy * dt
                dists = np.linalg.norm(o_pos - a_pos, axis=1)
                if np.any(dists < combined_radius):
                    hit = True
                    break
            if hit:
                collisions += 1

        return collisions / n_samples

    # ------------------------------------------------------------------
    # Risk field
    # ------------------------------------------------------------------

    def risk_field(
        self,
        position: np.ndarray,
        obstacles: np.ndarray,
        resolution: float = 0.5,
        field_size: float = 10.0,
    ) -> np.ndarray:
        """Compute a 2-D risk map centred on *position*.

        Risk is modelled as an inverse-distance potential from each obstacle.

        Parameters
        ----------
        position : np.ndarray
            ``[x, y]`` centre of the risk map.
        obstacles : np.ndarray
            ``(N, 2)`` obstacle positions.
        resolution : float
            Grid cell size (metres).
        field_size : float
            Half-extent of the square risk map (metres).

        Returns
        -------
        np.ndarray
            2-D array of risk values (higher = more dangerous).
        """
        n_cells = int(2 * field_size / resolution)
        risk = np.zeros((n_cells, n_cells), dtype=np.float64)

        xs = np.linspace(
            position[0] - field_size, position[0] + field_size, n_cells
        )
        ys = np.linspace(
            position[1] - field_size, position[1] + field_size, n_cells
        )
        xx, yy = np.meshgrid(xs, ys, indexing="ij")

        for obs in obstacles:
            dist = np.sqrt((xx - obs[0]) ** 2 + (yy - obs[1]) ** 2)
            dist = np.clip(dist, 1e-4, None)
            risk += 1.0 / dist

        return risk


# ---------------------------------------------------------------------------
# PredictiveRiskModel
# ---------------------------------------------------------------------------

class PredictiveRiskModel:
    """Uses trajectory prediction to estimate future risk.

    Parameters
    ----------
    prediction_fn : callable, optional
        ``(current_states, horizon) -> np.ndarray`` that returns predicted
        positions of shape ``(N, T, 2)``.  If ``None``, a constant-velocity
        model is used.
    agent_radius : float
        Collision radius of the agent.
    obstacle_radius : float
        Default obstacle radius.
    """

    def __init__(
        self,
        prediction_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
        agent_radius: float = 0.25,
        obstacle_radius: float = 0.3,
        dt: float = 0.1,
    ) -> None:
        self.prediction_fn = prediction_fn or self._constant_velocity_predict
        self.agent_radius = agent_radius
        self.obstacle_radius = obstacle_radius
        self.dt = dt

    # -- default predictor --------------------------------------------------

    @staticmethod
    def _constant_velocity_predict(
        current_states: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Constant-velocity trajectory prediction.

        Parameters
        ----------
        current_states : np.ndarray
            ``(N, 4)`` with ``[x, y, vx, vy]``.
        horizon : int
            Number of future steps.

        Returns
        -------
        np.ndarray
            ``(N, horizon, 2)`` predicted positions.
        """
        n = current_states.shape[0]
        predictions = np.empty((n, horizon, 2))
        for t in range(horizon):
            predictions[:, t, :] = (
                current_states[:, :2] + current_states[:, 2:4] * (t + 1) * 0.1
            )
        return predictions

    # -- public API ---------------------------------------------------------

    def predict_trajectories(
        self, current_states: np.ndarray, horizon: int = 30
    ) -> np.ndarray:
        """Predict future trajectories for the given states.

        Returns
        -------
        np.ndarray
            ``(N, horizon, 2)`` predicted positions.
        """
        return self.prediction_fn(current_states, horizon)

    def assess_risk(
        self,
        agent_trajectory: np.ndarray,
        predicted_trajectories: np.ndarray,
    ) -> np.ndarray:
        """Score per-timestep collision risk between agent and obstacles.

        Parameters
        ----------
        agent_trajectory : np.ndarray
            ``(T, 2)`` planned positions of the agent.
        predicted_trajectories : np.ndarray
            ``(N, T, 2)`` predicted obstacle positions.

        Returns
        -------
        np.ndarray
            ``(T,)`` risk scores in ``[0, 1]`` per timestep.
        """
        combined_r = self.agent_radius + self.obstacle_radius
        T = agent_trajectory.shape[0]
        risks = np.zeros(T)

        for t in range(T):
            dists = np.linalg.norm(
                predicted_trajectories[:, t, :] - agent_trajectory[t], axis=1
            )
            # Sigmoid-style risk: 1 when dist=0, ~0 when dist >> combined_r
            per_obs = np.exp(-((dists / combined_r) ** 2))
            risks[t] = float(np.max(per_obs)) if per_obs.size > 0 else 0.0

        return risks
