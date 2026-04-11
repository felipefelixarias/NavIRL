"""Tests for navirl/sensors/imu.py.

Covers: IMUConfig defaults, IMUSensor construction, reset, observation space,
raw observation with explicit accel, velocity-derived accel, gravity
compensation, bias drift, noise application, orientation integration,
and yaw wrapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.core.constants import GRAVITY, SIM
from navirl.sensors.imu import IMUConfig, IMUSensor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def imu() -> IMUSensor:
    """IMU with default config (noise enabled)."""
    return IMUSensor()


@pytest.fixture()
def quiet_imu() -> IMUSensor:
    """IMU with zero noise/bias for deterministic tests."""
    cfg = IMUConfig(
        accel_noise_std=0.0,
        gyro_noise_std=0.0,
        accel_bias_std=0.0,
        gyro_bias_std=0.0,
    )
    return IMUSensor(config=cfg)


@pytest.fixture()
def no_gravity_imu() -> IMUSensor:
    """Deterministic IMU without gravity compensation."""
    cfg = IMUConfig(
        accel_noise_std=0.0,
        gyro_noise_std=0.0,
        accel_bias_std=0.0,
        gyro_bias_std=0.0,
        gravity_compensation=False,
    )
    return IMUSensor(config=cfg)


def _world(
    *,
    pos=(0.0, 0.0),
    vel=(0.0, 0.0),
    heading=0.0,
    angular_vel=0.0,
    accel=None,
) -> dict:
    ws: dict = {
        "robot_pos": np.array(pos, dtype=np.float64),
        "robot_vel": np.array(vel, dtype=np.float64),
        "robot_heading": heading,
        "robot_angular_vel": angular_vel,
    }
    if accel is not None:
        ws["robot_accel"] = np.array(accel, dtype=np.float64)
    return ws


# ---------------------------------------------------------------------------
# IMUConfig
# ---------------------------------------------------------------------------


class TestIMUConfig:
    def test_defaults(self):
        cfg = IMUConfig()
        assert cfg.accel_noise_std == 0.05
        assert cfg.gyro_noise_std == 0.01
        assert cfg.accel_bias_std == 0.001
        assert cfg.gyro_bias_std == 0.0002
        assert cfg.dt == SIM.dt
        assert cfg.gravity_compensation is True

    def test_custom_values(self):
        cfg = IMUConfig(accel_noise_std=0.1, dt=0.05, gravity_compensation=False)
        assert cfg.accel_noise_std == 0.1
        assert cfg.dt == 0.05
        assert cfg.gravity_compensation is False


# ---------------------------------------------------------------------------
# Construction and reset
# ---------------------------------------------------------------------------


class TestIMUSensorInit:
    def test_default_construction(self, imu):
        assert imu._cfg.accel_noise_std == 0.05
        np.testing.assert_array_equal(imu._accel_bias, np.zeros(3))
        np.testing.assert_array_equal(imu._gyro_bias, np.zeros(3))
        np.testing.assert_array_equal(imu._orientation, np.zeros(3))
        assert imu._prev_vel is None

    def test_custom_config(self):
        cfg = IMUConfig(accel_noise_std=0.5)
        sensor = IMUSensor(config=cfg)
        assert sensor._cfg.accel_noise_std == 0.5

    def test_reset_clears_state(self, quiet_imu):
        # Observe a few steps to build up state
        quiet_imu.observe(_world(vel=(1.0, 0.0), angular_vel=0.5))
        quiet_imu.observe(_world(vel=(2.0, 0.0), angular_vel=0.5))

        quiet_imu.reset()
        np.testing.assert_array_equal(quiet_imu._accel_bias, np.zeros(3))
        np.testing.assert_array_equal(quiet_imu._gyro_bias, np.zeros(3))
        np.testing.assert_array_equal(quiet_imu._orientation, np.zeros(3))
        assert quiet_imu._prev_vel is None


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------


class TestObservationSpace:
    def test_keys(self, imu):
        space = imu.get_observation_space()
        assert set(space.keys()) == {"linear_acceleration", "angular_velocity", "orientation"}

    def test_shapes(self, imu):
        space = imu.get_observation_space()
        for key in space:
            assert space[key]["shape"] == (3,)

    def test_dtypes(self, imu):
        space = imu.get_observation_space()
        for key in space:
            assert space[key]["dtype"] == np.float64


# ---------------------------------------------------------------------------
# Observation outputs
# ---------------------------------------------------------------------------


class TestObserveOutput:
    def test_output_keys(self, imu):
        obs = imu.observe(_world())
        assert set(obs.keys()) == {"linear_acceleration", "angular_velocity", "orientation"}

    def test_output_shapes(self, imu):
        obs = imu.observe(_world())
        for key in obs:
            assert obs[key].shape == (3,)

    def test_output_dtypes(self, imu):
        obs = imu.observe(_world())
        for key in obs:
            assert obs[key].dtype == np.float64


# ---------------------------------------------------------------------------
# Gravity compensation
# ---------------------------------------------------------------------------


class TestGravityCompensation:
    def test_gravity_at_rest(self, quiet_imu):
        obs = quiet_imu.observe(_world())
        # At rest with gravity compensation: z-accel should be +GRAVITY
        assert obs["linear_acceleration"][2] == pytest.approx(GRAVITY, abs=1e-10)

    def test_no_gravity_at_rest(self, no_gravity_imu):
        obs = no_gravity_imu.observe(_world())
        # Without gravity compensation: z-accel should be 0
        assert obs["linear_acceleration"][2] == pytest.approx(0.0, abs=1e-10)

    def test_xy_accel_zero_at_rest(self, quiet_imu):
        obs = quiet_imu.observe(_world())
        assert obs["linear_acceleration"][0] == pytest.approx(0.0, abs=1e-10)
        assert obs["linear_acceleration"][1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Explicit acceleration
# ---------------------------------------------------------------------------


class TestExplicitAccel:
    def test_explicit_accel_used(self, quiet_imu):
        obs = quiet_imu.observe(_world(accel=(3.0, 4.0)))
        assert obs["linear_acceleration"][0] == pytest.approx(3.0, abs=1e-10)
        assert obs["linear_acceleration"][1] == pytest.approx(4.0, abs=1e-10)

    def test_explicit_accel_overrides_velocity(self, quiet_imu):
        # First call to set prev_vel
        quiet_imu.observe(_world(vel=(0.0, 0.0)))
        # Velocity suggests accel, but explicit accel should override
        obs = quiet_imu.observe(_world(vel=(10.0, 0.0), accel=(1.0, 0.0)))
        assert obs["linear_acceleration"][0] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Velocity-derived acceleration
# ---------------------------------------------------------------------------


class TestVelocityDerivedAccel:
    def test_first_step_zero_accel(self, quiet_imu):
        """First step has no previous velocity -> zero accel."""
        obs = quiet_imu.observe(_world(vel=(5.0, 0.0)))
        assert obs["linear_acceleration"][0] == pytest.approx(0.0, abs=1e-10)
        assert obs["linear_acceleration"][1] == pytest.approx(0.0, abs=1e-10)

    def test_second_step_finite_difference(self, quiet_imu):
        """Second step estimates accel from velocity change."""
        dt = quiet_imu._cfg.dt
        quiet_imu.observe(_world(vel=(0.0, 0.0)))
        obs = quiet_imu.observe(_world(vel=(dt, 0.0)))
        # Expected accel = (dt - 0) / dt = 1.0
        assert obs["linear_acceleration"][0] == pytest.approx(1.0, abs=1e-10)

    def test_prev_vel_updated(self, quiet_imu):
        quiet_imu.observe(_world(vel=(1.0, 2.0)))
        np.testing.assert_array_equal(quiet_imu._prev_vel, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Angular velocity
# ---------------------------------------------------------------------------


class TestAngularVelocity:
    def test_zero_angular_vel(self, quiet_imu):
        obs = quiet_imu.observe(_world())
        np.testing.assert_allclose(obs["angular_velocity"], [0.0, 0.0, 0.0], atol=1e-10)

    def test_nonzero_angular_vel(self, quiet_imu):
        obs = quiet_imu.observe(_world(angular_vel=1.5))
        assert obs["angular_velocity"][2] == pytest.approx(1.5, abs=1e-10)
        # x and y angular velocity are always 0 for 2D motion
        assert obs["angular_velocity"][0] == pytest.approx(0.0, abs=1e-10)
        assert obs["angular_velocity"][1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Orientation integration
# ---------------------------------------------------------------------------


class TestOrientationIntegration:
    def test_zero_rotation_stays_zero(self, quiet_imu):
        obs = quiet_imu.observe(_world())
        np.testing.assert_allclose(obs["orientation"], [0.0, 0.0, 0.0], atol=1e-10)

    def test_yaw_integrates(self, quiet_imu):
        dt = quiet_imu._cfg.dt
        obs = quiet_imu.observe(_world(angular_vel=1.0))
        # yaw should be approximately 1.0 * dt
        assert obs["orientation"][2] == pytest.approx(1.0 * dt, abs=1e-10)

    def test_yaw_accumulates(self, quiet_imu):
        dt = quiet_imu._cfg.dt
        quiet_imu.observe(_world(angular_vel=1.0))
        obs = quiet_imu.observe(_world(angular_vel=1.0))
        assert obs["orientation"][2] == pytest.approx(2.0 * dt, abs=1e-10)

    def test_yaw_wraps_to_pi(self, quiet_imu):
        """Yaw should wrap to [-pi, pi]."""
        dt = quiet_imu._cfg.dt
        # Large angular velocity to force wrapping
        n_steps = int(np.ceil(np.pi / dt)) + 10
        for _ in range(n_steps):
            obs = quiet_imu.observe(_world(angular_vel=1.0))
        yaw = obs["orientation"][2]
        assert -np.pi <= yaw <= np.pi

    def test_orientation_is_copy(self, quiet_imu):
        """Returned orientation should be a copy, not a reference."""
        obs1 = quiet_imu.observe(_world(angular_vel=1.0))
        obs2 = quiet_imu.observe(_world(angular_vel=1.0))
        # Modifying obs1 should not affect obs2
        assert obs1["orientation"] is not obs2["orientation"]


# ---------------------------------------------------------------------------
# Noise and bias
# ---------------------------------------------------------------------------


class TestNoiseAndBias:
    def test_noise_adds_variability(self):
        """With noise enabled, repeated identical observations should vary."""
        imu = IMUSensor(config=IMUConfig(accel_noise_std=1.0, gyro_noise_std=1.0))
        ws = _world()
        obs1 = imu.observe(ws)
        imu.reset()
        obs2 = imu.observe(ws)
        # Extremely unlikely to be identical with noise_std=1.0
        assert not np.allclose(obs1["linear_acceleration"], obs2["linear_acceleration"])

    def test_bias_drifts_over_time(self):
        """Bias should accumulate over multiple steps."""
        cfg = IMUConfig(
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
            accel_bias_std=0.1,
            gyro_bias_std=0.1,
        )
        imu = IMUSensor(config=cfg)
        ws = _world()
        for _ in range(100):
            imu.observe(ws)
        # After 100 steps, bias should have drifted away from zero
        assert np.linalg.norm(imu._accel_bias) > 0.0
        assert np.linalg.norm(imu._gyro_bias) > 0.0

    def test_zero_noise_deterministic(self, quiet_imu):
        """With zero noise and bias, observations should be deterministic."""
        ws = _world(accel=(1.0, 2.0), angular_vel=0.5)
        obs = quiet_imu.observe(ws)
        assert obs["linear_acceleration"][0] == pytest.approx(1.0, abs=1e-10)
        assert obs["linear_acceleration"][1] == pytest.approx(2.0, abs=1e-10)
        assert obs["angular_velocity"][2] == pytest.approx(0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_vel_defaults_zero(self, quiet_imu):
        ws = {"robot_pos": np.array([0.0, 0.0]), "robot_heading": 0.0}
        obs = quiet_imu.observe(ws)
        assert obs["linear_acceleration"][0] == pytest.approx(0.0, abs=1e-10)

    def test_missing_angular_vel_defaults_zero(self, quiet_imu):
        ws = {
            "robot_pos": np.array([0.0, 0.0]),
            "robot_vel": np.array([0.0, 0.0]),
            "robot_heading": 0.0,
        }
        obs = quiet_imu.observe(ws)
        np.testing.assert_allclose(obs["angular_velocity"], [0.0, 0.0, 0.0], atol=1e-10)

    def test_negative_angular_vel(self, quiet_imu):
        obs = quiet_imu.observe(_world(angular_vel=-2.0))
        assert obs["angular_velocity"][2] == pytest.approx(-2.0, abs=1e-10)

    def test_large_accel(self, quiet_imu):
        obs = quiet_imu.observe(_world(accel=(100.0, -50.0)))
        assert obs["linear_acceleration"][0] == pytest.approx(100.0, abs=1e-10)
        assert obs["linear_acceleration"][1] == pytest.approx(-50.0, abs=1e-10)
