"""Sensor fusion and state estimation.

Provides :class:`SensorFusion` (combines observations from multiple sensors
into a single dictionary) and :class:`KalmanStateEstimator` (an Extended
Kalman Filter for fusing heterogeneous sensor data into a coherent state
estimate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.constants import SIM
from navirl.sensors.base import SensorBase

# ---------------------------------------------------------------------------
#  SensorFusion
# ---------------------------------------------------------------------------

class SensorFusion:
    """Combines observations from multiple named sensors.

    Parameters
    ----------
    sensors : dict[str, SensorBase]
        Mapping from sensor name to sensor instance.
    dt : float
        Nominal time step.  Used to decide whether sensors fire on a given
        tick when operating at different rates.
    rates : dict[str, float] | None
        Optional per-sensor rate (Hz).  If a sensor's rate is lower than
        ``1/dt`` it will only be queried on the appropriate ticks.  Omitted
        sensors default to every tick.
    """

    def __init__(
        self,
        sensors: dict[str, SensorBase],
        dt: float = SIM.dt,
        rates: dict[str, float] | None = None,
    ) -> None:
        self.sensors = sensors
        self.dt = dt
        self.rates = rates or {}
        self._tick: int = 0
        self._last_obs: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset all sensors and the tick counter."""
        self._tick = 0
        self._last_obs.clear()
        for sensor in self.sensors.values():
            sensor.reset()

    def observe(self, world_state: dict[str, Any]) -> dict[str, Any]:
        """Query all sensors and return a combined observation dictionary.

        Sensors that are not scheduled to fire on the current tick return
        their most recent observation (sample-and-hold).

        Returns
        -------
        dict[str, Any]
            Keys are sensor names; values are the corresponding observations.
            An additional key ``"_tick"`` records the current step counter.
        """
        obs: dict[str, Any] = {"_tick": self._tick}

        for name, sensor in self.sensors.items():
            if self._should_fire(name):
                reading = sensor.observe(world_state)
                self._last_obs[name] = reading
                obs[name] = reading
            else:
                # Sample-and-hold: return last observation
                obs[name] = self._last_obs.get(name, None)

        self._tick += 1
        return obs

    def seed(self, seed: int) -> None:
        """Seed all constituent sensors."""
        for i, sensor in enumerate(self.sensors.values()):
            sensor.seed(seed + i)

    def get_observation_space(self) -> dict[str, Any]:
        """Return the combined observation space as a nested dict."""
        return {
            name: sensor.get_observation_space()
            for name, sensor in self.sensors.items()
        }

    def _should_fire(self, name: str) -> bool:
        """Determine whether sensor *name* should produce an observation on
        the current tick."""
        if name not in self.rates:
            return True
        rate = self.rates[name]
        if rate <= 0:
            return True
        period_ticks = max(1, int(round(1.0 / (rate * self.dt))))
        return (self._tick % period_ticks) == 0


# ---------------------------------------------------------------------------
#  Extended Kalman Filter state estimator
# ---------------------------------------------------------------------------

@dataclass
class EKFConfig:
    """Configuration for the Extended Kalman Filter.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector.  Default 6 corresponds to
        ``[x, y, heading, vx, vy, omega]``.
    process_noise : np.ndarray | None
        Process noise covariance Q.  If None, a sensible default is used.
    initial_covariance : float
        Diagonal value for the initial state covariance P0.
    dt : float
        Nominal prediction time step.
    """

    state_dim: int = 6
    process_noise: np.ndarray | None = None
    initial_covariance: float = 1.0
    dt: float = SIM.dt


class KalmanStateEstimator:
    """Extended Kalman Filter for robot state estimation from multiple sensors.

    State vector: ``[x, y, heading, vx, vy, omega]``

    Supports two update modalities:

    * **position update** -- from e.g. GPS or odometry (observes x, y, heading).
    * **velocity update** -- from e.g. IMU (observes vx, vy, omega).

    Both updates can be called independently at any rate.
    """

    def __init__(self, config: EKFConfig | None = None) -> None:
        cfg = config or EKFConfig()
        self.dt = cfg.dt
        n = cfg.state_dim

        self.state = np.zeros(n, dtype=np.float64)
        self.P = np.eye(n, dtype=np.float64) * cfg.initial_covariance

        if cfg.process_noise is not None:
            self.Q = np.asarray(cfg.process_noise, dtype=np.float64)
        else:
            self.Q = np.diag([0.01, 0.01, 0.005, 0.05, 0.05, 0.02]).astype(
                np.float64
            )

    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset the filter state and covariance."""
        n = self.state.shape[0]
        self.state = state.copy() if state is not None else np.zeros(n, dtype=np.float64)
        self.P = np.eye(n, dtype=np.float64) * 1.0

    # -- Predict -------------------------------------------------------------

    def predict(self, dt: float | None = None) -> np.ndarray:
        """Constant-velocity prediction step.

        Parameters
        ----------
        dt : float | None
            Time step override (defaults to ``self.dt``).

        Returns
        -------
        (6,) ndarray
            Predicted state.
        """
        dt = dt if dt is not None else self.dt
        x, y, theta, vx, vy, omega = self.state

        # Non-linear state transition
        self.state[0] = x + vx * dt
        self.state[1] = y + vy * dt
        self.state[2] = theta + omega * dt
        # Wrap heading
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        # vx, vy, omega assumed constant (no acceleration model)

        # Jacobian of f w.r.t. state
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.P = F @ self.P @ F.T + self.Q * dt
        return self.state.copy()

    # -- Update: position (x, y, heading) ------------------------------------

    def update_position(
        self,
        z: np.ndarray,
        R: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update with a position measurement ``[x, y, heading]``.

        Parameters
        ----------
        z : (3,) array
            Measured ``[x, y, heading]``.
        R : (3, 3) array | None
            Measurement noise covariance.  Default ``diag(0.05, 0.05, 0.02)``.

        Returns
        -------
        (6,) ndarray
            Updated state.
        """
        if R is None:
            R = np.diag([0.05, 0.05, 0.02]).astype(np.float64)
        z = np.asarray(z, dtype=np.float64)

        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # heading

        y = z - H @ self.state
        # Wrap heading innovation
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = (np.eye(6) - K @ H) @ self.P

        return self.state.copy()

    # -- Update: velocity (vx, vy, omega) ------------------------------------

    def update_velocity(
        self,
        z: np.ndarray,
        R: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update with a velocity measurement ``[vx, vy, omega]``.

        Parameters
        ----------
        z : (3,) array
            Measured ``[vx, vy, omega]``.
        R : (3, 3) array | None
            Measurement noise covariance.  Default ``diag(0.1, 0.1, 0.05)``.

        Returns
        -------
        (6,) ndarray
            Updated state.
        """
        if R is None:
            R = np.diag([0.1, 0.1, 0.05]).astype(np.float64)
        z = np.asarray(z, dtype=np.float64)

        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 3] = 1.0  # vx
        H[1, 4] = 1.0  # vy
        H[2, 5] = 1.0  # omega

        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        return self.state.copy()

    # -- Full update with arbitrary observation matrix -----------------------

    def update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """Generic Kalman update with arbitrary observation model.

        Parameters
        ----------
        z : (m,) array
            Measurement vector.
        H : (m, 6) array
            Observation matrix mapping state to measurement space.
        R : (m, m) array
            Measurement noise covariance.

        Returns
        -------
        (6,) ndarray
            Updated state.
        """
        z = np.asarray(z, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.state.copy()

    @property
    def position(self) -> np.ndarray:
        """Current position estimate ``[x, y]``."""
        return self.state[:2].copy()

    @property
    def heading(self) -> float:
        """Current heading estimate (radians)."""
        return float(self.state[2])

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate ``[vx, vy]``."""
        return self.state[3:5].copy()
