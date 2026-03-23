"""Simulated Inertial Measurement Unit (IMU).

Provides :class:`IMUSensor` which produces 3-D linear acceleration and 3-D
angular velocity readings from the robot's kinematic state.  Supports
configurable Gaussian noise, bias random walk (drift), and dead-reckoning
orientation integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from navirl.core.constants import GRAVITY, SIM
from navirl.sensors.base import NoiseModel, SensorBase


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class IMUConfig:
    """Configuration for the simulated IMU.

    The robot is assumed to move in 2-D (x-y plane) but the IMU reports full
    3-D vectors so that the interface matches real hardware.

    Parameters
    ----------
    accel_noise_std : float
        Standard deviation of additive noise on linear acceleration (m/s^2).
    gyro_noise_std : float
        Standard deviation of additive noise on angular velocity (rad/s).
    accel_bias_std : float
        Standard deviation of the bias random walk for accelerometer.
    gyro_bias_std : float
        Standard deviation of the bias random walk for gyroscope.
    dt : float
        Expected time step for integration (seconds).
    gravity_compensation : bool
        If True, the z-acceleration includes +g (sensor at rest reads +9.81).
    """

    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.01
    accel_bias_std: float = 0.001
    gyro_bias_std: float = 0.0002
    dt: float = SIM.dt
    gravity_compensation: bool = True


# ---------------------------------------------------------------------------
#  IMUSensor
# ---------------------------------------------------------------------------

class IMUSensor(SensorBase):
    """Simulated 6-axis IMU (accelerometer + gyroscope).

    World state keys
    ----------------
    * ``robot_pos`` : (2,) ndarray -- position [x, y] in metres.
    * ``robot_vel`` : (2,) ndarray -- velocity [vx, vy] in m/s.
    * ``robot_heading`` : float -- heading in radians.
    * ``robot_angular_vel`` : float -- angular velocity in rad/s.
    * ``robot_accel`` : (2,) ndarray -- linear acceleration [ax, ay] in m/s^2
      (optional; estimated from velocity if absent).

    Returns
    -------
    dict
        ``linear_acceleration`` : (3,) ndarray -- [ax, ay, az] in m/s^2.
        ``angular_velocity`` : (3,) ndarray -- [wx, wy, wz] in rad/s.
        ``orientation`` : (3,) ndarray -- integrated [roll, pitch, yaw] in rad.
    """

    def __init__(
        self,
        config: Optional[IMUConfig] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        self._cfg = config or IMUConfig()
        super().__init__(config=self._cfg, noise_model=noise_model)

        # Internal state for bias drift and orientation integration
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._orientation = np.zeros(3, dtype=np.float64)  # roll, pitch, yaw
        self._prev_vel: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset biases, orientation estimate, and velocity cache."""
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._orientation = np.zeros(3, dtype=np.float64)
        self._prev_vel = None

    # -- SensorBase interface ------------------------------------------------

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "linear_acceleration": {
                "shape": (3,),
                "dtype": np.float64,
                "low": -50.0,
                "high": 50.0,
            },
            "angular_velocity": {
                "shape": (3,),
                "dtype": np.float64,
                "low": -10.0,
                "high": 10.0,
            },
            "orientation": {
                "shape": (3,),
                "dtype": np.float64,
                "low": -np.pi,
                "high": np.pi,
            },
        }

    def _raw_observe(self, world_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        dt = self._cfg.dt

        # --- Linear acceleration ---
        if "robot_accel" in world_state:
            accel_2d = np.asarray(world_state["robot_accel"], dtype=np.float64)
        else:
            vel_2d = np.asarray(world_state.get("robot_vel", [0.0, 0.0]),
                                dtype=np.float64)
            if self._prev_vel is not None:
                accel_2d = (vel_2d - self._prev_vel) / dt
            else:
                accel_2d = np.zeros(2, dtype=np.float64)
            self._prev_vel = vel_2d.copy()

        accel_3d = np.array([accel_2d[0], accel_2d[1], 0.0], dtype=np.float64)
        if self._cfg.gravity_compensation:
            accel_3d[2] += GRAVITY  # sensor at rest reads +g on z

        # --- Angular velocity ---
        omega_z = float(world_state.get("robot_angular_vel", 0.0))
        gyro_3d = np.array([0.0, 0.0, omega_z], dtype=np.float64)

        # --- Bias random walk ---
        self._accel_bias += self._rng.normal(
            0, self._cfg.accel_bias_std, size=3) * np.sqrt(dt)
        self._gyro_bias += self._rng.normal(
            0, self._cfg.gyro_bias_std, size=3) * np.sqrt(dt)

        # --- Apply bias and noise ---
        accel_noisy = (
            accel_3d
            + self._accel_bias
            + self._rng.normal(0, self._cfg.accel_noise_std, size=3)
        )
        gyro_noisy = (
            gyro_3d
            + self._gyro_bias
            + self._rng.normal(0, self._cfg.gyro_noise_std, size=3)
        )

        # --- Integrate orientation ---
        self._orientation += gyro_noisy * dt
        # Wrap yaw to [-pi, pi]
        self._orientation[2] = (
            (self._orientation[2] + np.pi) % (2 * np.pi) - np.pi
        )

        return {
            "linear_acceleration": accel_noisy,
            "angular_velocity": gyro_noisy,
            "orientation": self._orientation.copy(),
        }

    def observe(self, world_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Override to skip the generic noise model (noise is applied internally)."""
        return self._raw_observe(world_state)
