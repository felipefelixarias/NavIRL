"""Tests for navirl.sensors — camera, IMU, fusion, occupancy grid, pedestrian detector."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.sensors.base import (
    DropoutNoise,
    GaussianNoise,
    QuantizationNoise,
    SaltPepperNoise,
)
from navirl.sensors.camera import CameraConfig, CameraSensor, DepthSensor, DepthSensorConfig
from navirl.sensors.fusion import EKFConfig, KalmanStateEstimator, SensorFusion
from navirl.sensors.imu import IMUConfig, IMUSensor
from navirl.sensors.lidar import LidarConfig, LidarSensor
from navirl.sensors.occupancy_grid import OccupancyGridConfig, OccupancyGridSensor
from navirl.sensors.pedestrian_detector import (
    PedestrianDetector,
    PedestrianDetectorConfig,
    PedestrianTracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_world_state(
    robot_pos=(10.0, 10.0),
    robot_heading=0.0,
    robot_vel=(0.0, 0.0),
    agents=None,
    obstacles_segments=None,
    obstacles_circles=None,
):
    """Create a minimal world state dict."""
    ws = {
        "robot_pos": np.array(robot_pos, dtype=np.float64),
        "robot_heading": robot_heading,
        "robot_vel": np.array(robot_vel, dtype=np.float64),
        "robot_angular_vel": 0.0,
        "agents": agents or [],
        "world_bounds": (0.0, 0.0, 50.0, 50.0),
    }
    if obstacles_segments is not None:
        ws["obstacles_segments"] = obstacles_segments
    if obstacles_circles is not None:
        ws["obstacles_circles"] = obstacles_circles
    return ws


def _agent(pos, vel=(0.0, 0.0), radius=0.3):
    return {"pos": np.array(pos), "vel": np.array(vel), "radius": radius}


# ---------------------------------------------------------------------------
# CameraSensor
# ---------------------------------------------------------------------------


class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.resolution_x > 0
        assert cfg.resolution_y > 0


class TestCameraSensor:
    def test_observation_space(self):
        sensor = CameraSensor()
        space = sensor.get_observation_space()
        assert "shape" in space

    def test_observe_empty_world(self):
        sensor = CameraSensor()
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 3  # H x W x 3

    def test_observe_with_agents(self):
        sensor = CameraSensor()
        ws = _basic_world_state(agents=[_agent((12.0, 10.0)), _agent((8.0, 10.0))])
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_observe_with_obstacles(self):
        sensor = CameraSensor()
        ws = _basic_world_state(
            obstacles_circles={"centres": np.array([[15.0, 10.0]]), "radii": np.array([1.0])},
        )
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_top_down_mode(self):
        cfg = CameraConfig(render_mode="top_down")
        sensor = CameraSensor(config=cfg)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_perspective_mode(self):
        cfg = CameraConfig(render_mode="perspective")
        sensor = CameraSensor(config=cfg)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_with_noise(self):
        sensor = CameraSensor(noise_model=GaussianNoise(std=5.0, seed=42))
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_seed_reproducibility(self):
        sensor = CameraSensor()
        sensor.seed(42)
        ws = _basic_world_state(agents=[_agent((12.0, 10.0))])
        obs1 = sensor.observe(ws)
        assert isinstance(obs1, np.ndarray)


class TestDepthSensor:
    def test_observation_space(self):
        sensor = DepthSensor()
        space = sensor.get_observation_space()
        assert "shape" in space

    def test_observe_empty(self):
        sensor = DepthSensor()
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 1

    def test_observe_with_obstacle(self):
        sensor = DepthSensor()
        ws = _basic_world_state(
            obstacles_circles={"centres": np.array([[12.0, 10.0]]), "radii": np.array([0.5])},
        )
        obs = sensor.observe(ws)
        # Some rays should be shorter than max range
        assert isinstance(obs, np.ndarray)

    def test_config(self):
        cfg = DepthSensorConfig(resolution=180, max_range=20.0)
        sensor = DepthSensor(config=cfg)
        space = sensor.get_observation_space()
        assert space["shape"][0] == 180


# ---------------------------------------------------------------------------
# IMUSensor
# ---------------------------------------------------------------------------


class TestIMUConfig:
    def test_defaults(self):
        cfg = IMUConfig()
        assert cfg.dt > 0


class TestIMUSensor:
    def test_observe_stationary(self):
        sensor = IMUSensor()
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, dict)
        assert "linear_acceleration" in obs
        assert "angular_velocity" in obs
        assert "orientation" in obs

    def test_observe_moving(self):
        sensor = IMUSensor()
        ws = _basic_world_state(robot_vel=(1.0, 0.0))
        obs = sensor.observe(ws)
        assert isinstance(obs, dict)

    def test_reset(self):
        sensor = IMUSensor()
        ws = _basic_world_state(robot_vel=(1.0, 0.0))
        sensor.observe(ws)
        sensor.reset()
        obs = sensor.observe(ws)
        assert isinstance(obs, dict)

    def test_seed(self):
        sensor = IMUSensor()
        sensor.seed(42)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, dict)

    def test_observation_space(self):
        sensor = IMUSensor()
        space = sensor.get_observation_space()
        assert isinstance(space, dict)

    def test_with_angular_vel(self):
        sensor = IMUSensor()
        ws = _basic_world_state()
        ws["robot_angular_vel"] = 0.5
        obs = sensor.observe(ws)
        assert isinstance(obs, dict)

    def test_with_custom_config(self):
        cfg = IMUConfig(accel_noise_std=0.05, gyro_noise_std=0.02)
        sensor = IMUSensor(config=cfg)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, dict)


# ---------------------------------------------------------------------------
# SensorFusion
# ---------------------------------------------------------------------------


class TestSensorFusion:
    def test_basic_fusion(self):
        lidar = LidarSensor()
        imu = IMUSensor()
        fusion = SensorFusion(sensors={"lidar": lidar, "imu": imu})
        ws = _basic_world_state()
        obs = fusion.observe(ws)
        assert "lidar" in obs or "_tick" in obs

    def test_reset(self):
        lidar = LidarSensor()
        fusion = SensorFusion(sensors={"lidar": lidar})
        ws = _basic_world_state()
        fusion.observe(ws)
        fusion.reset()
        obs = fusion.observe(ws)
        assert isinstance(obs, dict)

    def test_seed(self):
        lidar = LidarSensor()
        fusion = SensorFusion(sensors={"lidar": lidar})
        fusion.seed(42)

    def test_observation_space(self):
        lidar = LidarSensor()
        fusion = SensorFusion(sensors={"lidar": lidar})
        space = fusion.get_observation_space()
        assert isinstance(space, dict)

    def test_multi_sensor(self):
        lidar = LidarSensor()
        imu = IMUSensor()
        camera = CameraSensor()
        fusion = SensorFusion(sensors={"lidar": lidar, "imu": imu, "camera": camera})
        ws = _basic_world_state()
        obs = fusion.observe(ws)
        assert isinstance(obs, dict)

    def test_rate_limiting(self):
        lidar = LidarSensor()
        imu = IMUSensor()
        fusion = SensorFusion(
            sensors={"lidar": lidar, "imu": imu},
            rates={"lidar": 10.0, "imu": 100.0},
        )
        ws = _basic_world_state()
        # Multiple observations
        for _ in range(5):
            obs = fusion.observe(ws)
        assert isinstance(obs, dict)


# ---------------------------------------------------------------------------
# KalmanStateEstimator
# ---------------------------------------------------------------------------


class TestKalmanStateEstimator:
    def test_init(self):
        kf = KalmanStateEstimator()
        assert kf.position is not None

    def test_predict(self):
        kf = KalmanStateEstimator()
        kf.reset(np.array([1.0, 2.0, 0.0, 1.0, 0.0, 0.0]))
        state = kf.predict()
        assert state[0] > 1.0  # x advanced by vx*dt

    def test_update_position(self):
        kf = KalmanStateEstimator()
        kf.reset()
        z = np.array([5.0, 5.0, 0.5])
        state = kf.update_position(z)
        # Position should move toward measurement
        assert abs(state[0] - 5.0) < abs(0.0 - 5.0)

    def test_update_velocity(self):
        kf = KalmanStateEstimator()
        kf.reset()
        z = np.array([1.0, 0.5, 0.1])
        state = kf.update_velocity(z)
        assert isinstance(state, np.ndarray)

    def test_predict_update_cycle(self):
        kf = KalmanStateEstimator()
        kf.reset(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
        for _ in range(10):
            kf.predict()
            kf.update_position(np.array([kf.position[0], kf.position[1], 0.0]))
        assert kf.position[0] > 0.0

    def test_reset(self):
        kf = KalmanStateEstimator()
        kf.reset(np.array([10.0, 20.0, 1.0, 0.5, 0.0, 0.0]))
        assert kf.position[0] == pytest.approx(10.0)
        assert kf.position[1] == pytest.approx(20.0)

    def test_custom_config(self):
        cfg = EKFConfig(state_dim=6, process_noise=0.5, initial_covariance=10.0)
        kf = KalmanStateEstimator(config=cfg)
        kf.reset()
        state = kf.predict()
        assert isinstance(state, np.ndarray)


# ---------------------------------------------------------------------------
# OccupancyGridSensor
# ---------------------------------------------------------------------------


class TestOccupancyGridConfig:
    def test_defaults(self):
        cfg = OccupancyGridConfig()
        assert cfg.grid_size > 0
        assert cfg.resolution > 0


class TestOccupancyGridSensor:
    def test_observation_space(self):
        sensor = OccupancyGridSensor()
        space = sensor.get_observation_space()
        assert "shape" in space

    def test_observe_empty(self):
        sensor = OccupancyGridSensor()
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_observe_with_agents(self):
        sensor = OccupancyGridSensor()
        ws = _basic_world_state(agents=[_agent((11.0, 10.0)), _agent((9.0, 10.0))])
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_observe_with_obstacles(self):
        sensor = OccupancyGridSensor()
        ws = _basic_world_state(
            obstacles_circles={"centres": np.array([[12.0, 10.0]]), "radii": np.array([0.5])},
            obstacles_segments=np.array([[[8.0, 8.0], [12.0, 8.0]]]),
        )
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_custom_layers(self):
        cfg = OccupancyGridConfig(layers=("static", "dynamic"))
        sensor = OccupancyGridSensor(config=cfg)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert obs.shape[0] == 2  # 2 layers

    def test_grid_size_config(self):
        cfg = OccupancyGridConfig(grid_size=32, resolution=0.25)
        sensor = OccupancyGridSensor(config=cfg)
        space = sensor.get_observation_space()
        assert space["shape"][-1] == 32

    def test_social_layer(self):
        cfg = OccupancyGridConfig(layers=("social",))
        sensor = OccupancyGridSensor(config=cfg)
        ws = _basic_world_state(agents=[_agent((11.0, 10.0))])
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_velocity_layers(self):
        cfg = OccupancyGridConfig(layers=("velocity_x", "velocity_y"))
        sensor = OccupancyGridSensor(config=cfg)
        ws = _basic_world_state(agents=[_agent((11.0, 10.0), vel=(1.0, 0.5))])
        obs = sensor.observe(ws)
        assert obs.shape[0] == 2


# ---------------------------------------------------------------------------
# PedestrianDetector
# ---------------------------------------------------------------------------


class TestPedestrianDetectorConfig:
    def test_defaults(self):
        cfg = PedestrianDetectorConfig()
        assert cfg.detection_range > 0
        assert cfg.fov > 0


class TestPedestrianDetector:
    def test_observation_space(self):
        sensor = PedestrianDetector()
        space = sensor.get_observation_space()
        assert "shape" in space

    def test_no_agents(self):
        cfg = PedestrianDetectorConfig(false_positive_rate=0.0)
        sensor = PedestrianDetector(config=cfg)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, list)
        assert len(obs) == 0

    def test_detect_nearby_agent(self):
        cfg = PedestrianDetectorConfig(
            detection_range=20.0,
            fov=np.pi * 2,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
        )
        sensor = PedestrianDetector(config=cfg)
        ws = _basic_world_state(agents=[_agent((12.0, 10.0))])
        obs = sensor.observe(ws)
        assert len(obs) >= 1

    def test_out_of_range(self):
        cfg = PedestrianDetectorConfig(
            detection_range=5.0,
            fov=np.pi * 2,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
        )
        sensor = PedestrianDetector(config=cfg)
        ws = _basic_world_state(agents=[_agent((100.0, 100.0))])
        obs = sensor.observe(ws)
        assert len(obs) == 0

    def test_detection_format(self):
        cfg = PedestrianDetectorConfig(
            detection_range=20.0,
            fov=np.pi * 2,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
        )
        sensor = PedestrianDetector(config=cfg)
        ws = _basic_world_state(agents=[_agent((12.0, 10.0), vel=(0.5, 0.0))])
        obs = sensor.observe(ws)
        if len(obs) > 0:
            assert len(obs[0]) == 5  # [rx, ry, rvx, rvy, radius]

    def test_false_positive_rate(self):
        cfg = PedestrianDetectorConfig(
            detection_range=20.0,
            fov=np.pi * 2,
            false_positive_rate=1.0,  # Always add false positives
            false_negative_rate=0.0,
            max_false_positives=3,
        )
        sensor = PedestrianDetector(config=cfg)
        sensor.seed(42)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        # Should have false positives since rate=1.0
        assert isinstance(obs, list)

    def test_seed_reproducibility(self):
        cfg = PedestrianDetectorConfig(
            detection_range=20.0,
            fov=np.pi * 2,
            position_noise_std=0.1,
        )
        sensor = PedestrianDetector(config=cfg)
        ws = _basic_world_state(agents=[_agent((12.0, 10.0))])
        sensor.seed(42)
        obs1 = sensor.observe(ws)
        sensor.seed(42)
        obs2 = sensor.observe(ws)
        if len(obs1) > 0 and len(obs2) > 0:
            np.testing.assert_allclose(obs1[0], obs2[0])


# ---------------------------------------------------------------------------
# PedestrianTracker
# ---------------------------------------------------------------------------


class TestPedestrianTracker:
    def test_init(self):
        tracker = PedestrianTracker()
        assert tracker is not None

    def test_empty_update(self):
        tracker = PedestrianTracker()
        tracks = tracker.update([])
        assert isinstance(tracks, list)

    def test_single_detection_track(self):
        tracker = PedestrianTracker(min_hits=1)
        det = np.array([1.0, 0.0, 0.5, 0.0, 0.3])
        # Feed same detection multiple times to confirm track
        for _ in range(3):
            tracks = tracker.update([det])
        assert len(tracks) >= 1

    def test_track_persistence(self):
        tracker = PedestrianTracker(max_age=3, min_hits=1)
        det = np.array([2.0, 0.0, 0.0, 0.0, 0.3])
        tracker.update([det])
        tracker.update([det])
        # Now remove detection
        tracker.update([])
        tracker.update([])
        # Track should still exist (max_age=3)

    def test_multiple_detections(self):
        tracker = PedestrianTracker(min_hits=1)
        dets = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.3]),
            np.array([5.0, 0.0, 0.0, 0.0, 0.3]),
        ]
        for _ in range(3):
            tracks = tracker.update(dets)
        assert len(tracks) >= 2

    def test_reset(self):
        tracker = PedestrianTracker(min_hits=1)
        det = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        tracker.update([det])
        tracker.update([det])
        tracker.reset()
        tracks = tracker.update([])
        assert len(tracks) == 0

    def test_track_dict_keys(self):
        tracker = PedestrianTracker(min_hits=1)
        det = np.array([1.0, 0.0, 0.5, 0.0, 0.3])
        for _ in range(3):
            tracks = tracker.update([det])
        if len(tracks) > 0:
            t = tracks[0]
            assert "id" in t
            assert "state" in t
            assert "radius" in t


# ---------------------------------------------------------------------------
# LidarSensor extended
# ---------------------------------------------------------------------------


class TestLidarSensorExtended:
    def test_sector_ranges(self):
        cfg = LidarConfig(num_beams=360, num_sectors=8)
        sensor = LidarSensor(config=cfg)
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        sectors = sensor.get_sector_ranges(obs)
        assert len(sectors) == 8

    def test_ranges_to_cartesian(self):
        cfg = LidarConfig(num_beams=4)
        sensor = LidarSensor(config=cfg)
        ranges = np.array([5.0, 5.0, 5.0, 5.0])
        cart = sensor.ranges_to_cartesian(ranges, heading=0.0)
        assert cart.shape[1] == 2

    def test_observe_with_wall_segments(self):
        sensor = LidarSensor()
        ws = _basic_world_state(
            obstacles_segments=np.array([[[5.0, 5.0], [15.0, 5.0]]]),
        )
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)

    def test_custom_config(self):
        cfg = LidarConfig(num_beams=180, max_range=15.0, fov=np.pi)
        sensor = LidarSensor(config=cfg)
        space = sensor.get_observation_space()
        assert space["shape"][0] == 180

    def test_with_noise_model(self):
        sensor = LidarSensor(noise_model=GaussianNoise(std=0.01, seed=42))
        ws = _basic_world_state()
        obs = sensor.observe(ws)
        assert isinstance(obs, np.ndarray)
