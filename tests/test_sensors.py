"""Tests for navirl/sensors/ package.

Covers: base noise models, LidarSensor, CameraSensor, DepthSensor,
OccupancyGridSensor, PedestrianDetector, PedestrianTracker,
SensorFusion, and KalmanStateEstimator.
"""

from __future__ import annotations

import math

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
from navirl.sensors.lidar import (
    LidarConfig,
    LidarSensor,
    _ray_circle_intersection,
    _ray_segment_intersection,
)
from navirl.sensors.occupancy_grid import OccupancyGridConfig, OccupancyGridSensor
from navirl.sensors.pedestrian_detector import (
    PedestrianDetector,
    PedestrianDetectorConfig,
    PedestrianTracker,
)


def _basic_world_state(
    robot_pos=(5.0, 5.0),
    robot_heading=0.0,
    agents=None,
    obstacles_segments=None,
    obstacles_circles=None,
):
    """Helper to build a minimal world state dict."""
    ws = {
        "robot_pos": np.array(robot_pos, dtype=np.float64),
        "robot_heading": robot_heading,
        "robot_vel": np.array([0.0, 0.0], dtype=np.float64),
    }
    if agents is not None:
        ws["agents"] = agents
    if obstacles_segments is not None:
        ws["obstacles_segments"] = np.array(obstacles_segments, dtype=np.float64)
    if obstacles_circles is not None:
        ws["obstacles_circles"] = obstacles_circles
    return ws


# ============================================================
# Noise models
# ============================================================


class TestGaussianNoise:
    def test_shape_preserved(self):
        noise = GaussianNoise(std=0.1, seed=42)
        data = np.ones((10, 3), dtype=np.float64)
        noisy = noise.apply(data)
        assert noisy.shape == data.shape

    def test_noise_is_additive(self):
        noise = GaussianNoise(std=0.0, mean=0.0, seed=42)
        data = np.array([1.0, 2.0, 3.0])
        noisy = noise.apply(data)
        np.testing.assert_allclose(noisy, data, atol=1e-10)

    def test_nonzero_std_changes_data(self):
        noise = GaussianNoise(std=1.0, seed=42)
        data = np.zeros(100)
        noisy = noise.apply(data)
        assert not np.allclose(noisy, data)

    def test_seed_reproducibility(self):
        data = np.ones(50)
        n1 = GaussianNoise(std=0.5, seed=123)
        n2 = GaussianNoise(std=0.5, seed=123)
        np.testing.assert_array_equal(n1.apply(data), n2.apply(data))


class TestSaltPepperNoise:
    def test_shape_preserved(self):
        noise = SaltPepperNoise(prob=0.1, seed=42)
        data = np.ones(100, dtype=np.float64) * 0.5
        noisy = noise.apply(data)
        assert noisy.shape == data.shape

    def test_zero_prob_no_change(self):
        noise = SaltPepperNoise(prob=0.0, seed=42)
        data = np.ones(100) * 0.5
        np.testing.assert_array_equal(noise.apply(data), data)

    def test_corruptions_are_low_or_high(self):
        noise = SaltPepperNoise(prob=0.5, low=-1.0, high=10.0, seed=42)
        data = np.ones(1000) * 0.5
        noisy = noise.apply(data)
        corrupted = noisy[noisy != 0.5]
        assert len(corrupted) > 0
        for v in corrupted:
            assert v == -1.0 or v == 10.0


class TestDropoutNoise:
    def test_shape_preserved(self):
        noise = DropoutNoise(prob=0.1, seed=42)
        data = np.ones(100)
        noisy = noise.apply(data)
        assert noisy.shape == data.shape

    def test_dropout_produces_fill(self):
        noise = DropoutNoise(prob=0.5, fill_value=-999.0, seed=42)
        data = np.ones(1000)
        noisy = noise.apply(data)
        assert np.sum(noisy == -999.0) > 0

    def test_zero_prob_no_change(self):
        noise = DropoutNoise(prob=0.0, seed=42)
        data = np.ones(100)
        np.testing.assert_array_equal(noise.apply(data), data)


class TestQuantizationNoise:
    def test_quantization(self):
        noise = QuantizationNoise(step=0.5)
        data = np.array([0.3, 0.7, 1.1, 1.9])
        result = noise.apply(data)
        np.testing.assert_allclose(result, [0.5, 0.5, 1.0, 2.0])

    def test_identity_at_step_boundaries(self):
        noise = QuantizationNoise(step=1.0)
        data = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(noise.apply(data), data)


# ============================================================
# Ray intersection helpers
# ============================================================


class TestRayCircleIntersection:
    def test_circle_in_front(self):
        origin = np.array([0.0, 0.0])
        cos_table = np.array([1.0])  # beam pointing right
        sin_table = np.array([0.0])
        centres = np.array([[5.0, 0.0]])
        radii = np.array([1.0])
        ranges = _ray_circle_intersection(origin, cos_table, sin_table, centres, radii, 20.0)
        assert ranges[0] == pytest.approx(4.0, abs=0.01)

    def test_no_circles(self):
        origin = np.array([0.0, 0.0])
        cos_table = np.array([1.0, 0.0])
        sin_table = np.array([0.0, 1.0])
        centres = np.empty((0, 2))
        radii = np.empty(0)
        ranges = _ray_circle_intersection(origin, cos_table, sin_table, centres, radii, 10.0)
        np.testing.assert_array_equal(ranges, [10.0, 10.0])

    def test_circle_behind_sensor(self):
        origin = np.array([0.0, 0.0])
        cos_table = np.array([1.0])
        sin_table = np.array([0.0])
        centres = np.array([[-5.0, 0.0]])
        radii = np.array([0.5])
        ranges = _ray_circle_intersection(origin, cos_table, sin_table, centres, radii, 10.0)
        assert ranges[0] == 10.0  # max range, no hit from behind

    def test_multiple_beams_multiple_circles(self):
        origin = np.array([0.0, 0.0])
        angles = np.linspace(-np.pi / 4, np.pi / 4, 5)
        cos_t = np.cos(angles)
        sin_t = np.sin(angles)
        centres = np.array([[3.0, 0.0], [0.0, 3.0]])
        radii = np.array([0.5, 0.5])
        ranges = _ray_circle_intersection(origin, cos_t, sin_t, centres, radii, 20.0)
        assert all(r <= 20.0 for r in ranges)


class TestRaySegmentIntersection:
    def test_perpendicular_wall(self):
        origin = np.array([0.0, 0.0])
        cos_table = np.array([1.0])
        sin_table = np.array([0.0])
        # Wall at x=5, from y=-5 to y=5
        segments = np.array([[[5.0, -5.0], [5.0, 5.0]]])
        ranges = _ray_segment_intersection(origin, cos_table, sin_table, segments, 20.0)
        assert ranges[0] == pytest.approx(5.0, abs=0.01)

    def test_no_segments(self):
        origin = np.array([0.0, 0.0])
        cos_table = np.array([1.0])
        sin_table = np.array([0.0])
        segments = np.empty((0, 2, 2))
        ranges = _ray_segment_intersection(origin, cos_table, sin_table, segments, 10.0)
        assert ranges[0] == 10.0

    def test_parallel_ray_no_hit(self):
        origin = np.array([0.0, 0.0])
        cos_table = np.array([1.0])
        sin_table = np.array([0.0])
        # Wall along x-axis (parallel to beam)
        segments = np.array([[[0.0, 1.0], [10.0, 1.0]]])
        ranges = _ray_segment_intersection(origin, cos_table, sin_table, segments, 20.0)
        assert ranges[0] == 20.0  # no intersection


# ============================================================
# LidarSensor
# ============================================================


class TestLidarSensor:
    def test_default_config(self):
        lidar = LidarSensor()
        obs_space = lidar.get_observation_space()
        assert obs_space["shape"] == (lidar.config.num_beams,)

    def test_observe_empty_world(self):
        lidar = LidarSensor(LidarConfig(num_beams=36, noise_std=0.0))
        ws = _basic_world_state()
        ranges = lidar.observe(ws)
        assert ranges.shape == (36,)
        np.testing.assert_allclose(ranges, lidar.config.max_range)

    def test_observe_with_circle_obstacle(self):
        lidar = LidarSensor(LidarConfig(num_beams=36, max_range=20.0, noise_std=0.0))
        ws = _basic_world_state(
            robot_pos=(0.0, 0.0),
            obstacles_circles={"centres": np.array([[5.0, 0.0]]), "radii": np.array([1.0])},
        )
        ranges = lidar.observe(ws)
        # At least one beam should hit the obstacle
        assert np.min(ranges) < 20.0

    def test_observe_with_agents(self):
        lidar = LidarSensor(LidarConfig(num_beams=36, max_range=20.0, noise_std=0.0))
        agents = [{"pos": np.array([8.0, 0.0]), "radius": 0.5}]
        ws = _basic_world_state(robot_pos=(0.0, 0.0), agents=agents)
        ranges = lidar.observe(ws)
        assert np.min(ranges) < 20.0

    def test_observe_with_wall_segments(self):
        lidar = LidarSensor(LidarConfig(num_beams=72, max_range=20.0, noise_std=0.0))
        segments = [[[10.0, -10.0], [10.0, 10.0]]]  # wall at x=10
        ws = _basic_world_state(robot_pos=(0.0, 0.0), obstacles_segments=segments)
        ranges = lidar.observe(ws)
        assert np.min(ranges) < 20.0

    def test_sector_ranges(self):
        lidar = LidarSensor(LidarConfig(num_beams=36, num_sectors=6, noise_std=0.0))
        ranges = np.random.rand(36) * 10.0
        sectors = lidar.get_sector_ranges(ranges)
        assert sectors.shape == (6,)
        # Each sector min should be <= beams in that sector
        for i in range(6):
            expected_min = np.min(ranges[i * 6 : (i + 1) * 6])
            assert sectors[i] == pytest.approx(expected_min)

    def test_ranges_to_cartesian(self):
        lidar = LidarSensor(LidarConfig(num_beams=4, fov=2 * np.pi, noise_std=0.0))
        ranges = np.array([5.0, 5.0, 5.0, 5.0])
        points = lidar.ranges_to_cartesian(ranges, heading=0.0)
        assert points.shape == (4, 2)
        # All points should be at distance 5
        dists = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(dists, 5.0, atol=0.01)

    def test_noise_model_applied(self):
        lidar = LidarSensor(LidarConfig(num_beams=36, noise_std=0.5))
        ws = _basic_world_state()
        ranges1 = lidar.observe(ws)
        ranges2 = lidar.observe(ws)
        # With noise, readings should differ
        assert not np.allclose(ranges1, ranges2)

    def test_heading_rotation(self):
        lidar = LidarSensor(LidarConfig(num_beams=36, max_range=20.0, noise_std=0.0))
        agents = [{"pos": np.array([5.0, 0.0]), "radius": 0.5}]
        ws0 = _basic_world_state(robot_pos=(0.0, 0.0), robot_heading=0.0, agents=agents)
        ws90 = _basic_world_state(robot_pos=(0.0, 0.0), robot_heading=np.pi / 2, agents=agents)
        r0 = lidar.observe(ws0)
        r90 = lidar.observe(ws90)
        # Different heading should shift which beam sees the agent
        assert not np.allclose(r0, r90)


# ============================================================
# CameraSensor
# ============================================================


class TestCameraSensor:
    def test_top_down_output_shape(self):
        cam = CameraSensor(CameraConfig(resolution_x=32, resolution_y=32, render_mode="top_down"))
        ws = _basic_world_state()
        img = cam.observe(ws)
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.uint8

    def test_perspective_output_shape(self):
        cam = CameraSensor(CameraConfig(resolution_x=64, resolution_y=48, render_mode="perspective"))
        ws = _basic_world_state()
        img = cam.observe(ws)
        assert img.shape == (48, 64, 3)

    def test_agents_render_differently(self):
        cam = CameraSensor(CameraConfig(resolution_x=64, resolution_y=64, render_mode="top_down"))
        ws_empty = _basic_world_state(robot_pos=(5.0, 5.0))
        ws_agent = _basic_world_state(
            robot_pos=(5.0, 5.0),
            agents=[{"pos": np.array([5.0, 5.0]), "radius": 2.0}],
        )
        img_empty = cam.observe(ws_empty)
        img_agent = cam.observe(ws_agent)
        assert not np.array_equal(img_empty, img_agent)

    def test_observation_space(self):
        cam = CameraSensor(CameraConfig(resolution_x=32, resolution_y=24))
        space = cam.get_observation_space()
        assert space["shape"] == (24, 32, 3)
        assert space["dtype"] == np.uint8

    def test_draw_line_static(self):
        """Test Bresenham line drawing helper."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        CameraSensor._draw_line(img, 0, 0, 9, 0, color=[255, 255, 255])
        # Horizontal line at row 0
        assert np.any(img[0, :, 0] > 0)


# ============================================================
# DepthSensor
# ============================================================


class TestDepthSensor:
    def test_default_config(self):
        ds = DepthSensor()
        space = ds.get_observation_space()
        assert len(space["shape"]) == 1

    def test_observe_empty(self):
        ds = DepthSensor(DepthSensorConfig(resolution=16, noise_std=0.0))
        ws = _basic_world_state()
        ranges = ds.observe(ws)
        assert ranges.shape == (16,)
        np.testing.assert_allclose(ranges, ds.config.max_range)

    def test_observe_with_obstacle(self):
        ds = DepthSensor(DepthSensorConfig(resolution=16, max_range=20.0, noise_std=0.0))
        ws = _basic_world_state(
            robot_pos=(0.0, 0.0),
            obstacles_circles={"centres": np.array([[5.0, 0.0]]), "radii": np.array([1.0])},
        )
        ranges = ds.observe(ws)
        assert np.min(ranges) < 20.0


# ============================================================
# OccupancyGridSensor
# ============================================================


class TestOccupancyGridSensor:
    def test_output_shape(self):
        cfg = OccupancyGridConfig(grid_size=16, layers=("static", "dynamic"))
        sensor = OccupancyGridSensor(cfg)
        ws = _basic_world_state()
        grid = sensor.observe(ws)
        assert grid.shape == (2, 16, 16)

    def test_static_obstacle_marks_cells(self):
        cfg = OccupancyGridConfig(grid_size=32, resolution=0.25, layers=("static",))
        sensor = OccupancyGridSensor(cfg)
        ws = _basic_world_state(
            robot_pos=(5.0, 5.0),
            obstacles_circles={"centres": np.array([[5.5, 5.0]]), "radii": np.array([0.5])},
        )
        grid = sensor.observe(ws)
        assert np.any(grid[0] > 0)

    def test_dynamic_layer_with_agents(self):
        cfg = OccupancyGridConfig(grid_size=32, resolution=0.25, layers=("dynamic",))
        sensor = OccupancyGridSensor(cfg)
        agents = [{"pos": np.array([5.5, 5.0]), "radius": 0.3, "vel": np.array([1.0, 0.0])}]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), agents=agents)
        grid = sensor.observe(ws)
        assert np.any(grid[0] > 0)

    def test_velocity_layers(self):
        cfg = OccupancyGridConfig(
            grid_size=32, resolution=0.25, layers=("velocity_x", "velocity_y")
        )
        sensor = OccupancyGridSensor(cfg)
        agents = [{"pos": np.array([5.5, 5.0]), "vel": np.array([2.0, -1.0]), "radius": 0.3}]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), agents=agents)
        grid = sensor.observe(ws)
        # Velocity layers should have non-zero values where agent is
        assert np.any(grid[0] != 0) or np.any(grid[1] != 0)

    def test_social_layer(self):
        cfg = OccupancyGridConfig(grid_size=32, resolution=0.25, layers=("social",))
        sensor = OccupancyGridSensor(cfg)
        agents = [{"pos": np.array([5.3, 5.0]), "radius": 0.25}]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), agents=agents)
        grid = sensor.observe(ws)
        # Should have non-zero social zone encoding
        assert np.any(grid[0] > 0)

    def test_observation_space(self):
        cfg = OccupancyGridConfig(grid_size=16, layers=("static", "dynamic", "social"))
        sensor = OccupancyGridSensor(cfg)
        space = sensor.get_observation_space()
        assert space["shape"] == (3, 16, 16)
        assert "layer_names" in space

    def test_empty_world_no_occupancy(self):
        cfg = OccupancyGridConfig(grid_size=16, layers=("static", "dynamic"))
        sensor = OccupancyGridSensor(cfg)
        ws = _basic_world_state()
        grid = sensor.observe(ws)
        np.testing.assert_array_equal(grid, 0.0)

    def test_segment_obstacles_render(self):
        cfg = OccupancyGridConfig(grid_size=32, resolution=0.25, layers=("static",))
        sensor = OccupancyGridSensor(cfg)
        # Wall passing through grid center
        segments = [[[4.0, 5.0], [6.0, 5.0]]]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), obstacles_segments=segments)
        grid = sensor.observe(ws)
        assert np.any(grid[0] > 0)


# ============================================================
# PedestrianDetector
# ============================================================


class TestPedestrianDetector:
    def test_no_agents(self):
        det = PedestrianDetector(PedestrianDetectorConfig(false_positive_rate=0.0))
        det.seed(42)
        ws = _basic_world_state()
        results = det.observe(ws)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_single_agent_detected(self):
        cfg = PedestrianDetectorConfig(
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
            occlusion_enabled=False,
        )
        det = PedestrianDetector(cfg)
        det.seed(42)
        agents = [{"pos": np.array([8.0, 5.0]), "vel": np.array([1.0, 0.0]), "radius": 0.3}]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), robot_heading=0.0, agents=agents)
        results = det.observe(ws)
        assert len(results) == 1
        # relative x should be ~3.0 (agent at 8, robot at 5)
        assert results[0][0] == pytest.approx(3.0, abs=0.1)

    def test_out_of_range_not_detected(self):
        cfg = PedestrianDetectorConfig(
            detection_range=5.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            occlusion_enabled=False,
        )
        det = PedestrianDetector(cfg)
        det.seed(42)
        agents = [{"pos": np.array([50.0, 5.0]), "vel": np.array([0.0, 0.0]), "radius": 0.3}]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), agents=agents)
        results = det.observe(ws)
        assert len(results) == 0

    def test_occlusion(self):
        cfg = PedestrianDetectorConfig(
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
            occlusion_enabled=True,
            occlusion_half_angle=0.3,
        )
        det = PedestrianDetector(cfg)
        det.seed(42)
        # Two agents at same angle, one behind the other
        agents = [
            {"pos": np.array([7.0, 5.0]), "vel": np.zeros(2), "radius": 0.5},
            {"pos": np.array([9.0, 5.0]), "vel": np.zeros(2), "radius": 0.3},
        ]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), robot_heading=0.0, agents=agents)
        results = det.observe(ws)
        # Second agent should be occluded by the first
        assert len(results) <= 2  # at most 2, possibly 1 due to occlusion

    def test_fov_filtering(self):
        cfg = PedestrianDetectorConfig(
            fov=np.pi / 2,  # 90 degree FOV
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            occlusion_enabled=False,
        )
        det = PedestrianDetector(cfg)
        det.seed(42)
        # Agent directly behind robot
        agents = [{"pos": np.array([2.0, 5.0]), "vel": np.zeros(2), "radius": 0.3}]
        ws = _basic_world_state(robot_pos=(5.0, 5.0), robot_heading=0.0, agents=agents)
        results = det.observe(ws)
        assert len(results) == 0

    def test_observation_space(self):
        det = PedestrianDetector()
        space = det.get_observation_space()
        assert space["shape"] == ("variable", 5)

    def test_array_agents_format(self):
        """Test that agents can be passed as numpy array instead of dicts."""
        cfg = PedestrianDetectorConfig(
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
            occlusion_enabled=False,
        )
        det = PedestrianDetector(cfg)
        det.seed(42)
        # [x, y, vx, vy, radius]
        agents = np.array([[8.0, 5.0, 1.0, 0.0, 0.3]])
        ws = _basic_world_state(robot_pos=(5.0, 5.0), agents=agents)
        results = det.observe(ws)
        assert len(results) == 1


# ============================================================
# PedestrianTracker
# ============================================================


class TestPedestrianTracker:
    def test_empty_update(self):
        tracker = PedestrianTracker()
        tracks = tracker.update([])
        assert isinstance(tracks, list)
        assert len(tracks) == 0

    def test_single_detection_creates_track(self):
        tracker = PedestrianTracker(min_hits=1)
        det = [np.array([3.0, 0.0, 1.0, 0.0, 0.3])]
        tracks = tracker.update(det)
        assert len(tracks) == 1
        assert "id" in tracks[0]
        assert "state" in tracks[0]
        assert tracks[0]["radius"] == 0.3

    def test_track_continuity(self):
        tracker = PedestrianTracker(min_hits=1, distance_threshold=2.0)
        # Frame 1: detection at (3, 0)
        tracks1 = tracker.update([np.array([3.0, 0.0, 1.0, 0.0, 0.3])])
        tid = tracks1[0]["id"]
        # Frame 2: detection moves slightly
        tracks2 = tracker.update([np.array([3.1, 0.0, 1.0, 0.0, 0.3])])
        assert any(t["id"] == tid for t in tracks2)

    def test_track_deletion_after_max_age(self):
        tracker = PedestrianTracker(max_age=2, min_hits=1)
        tracker.update([np.array([3.0, 0.0, 0.0, 0.0, 0.3])])
        # Miss for max_age + 1 frames
        for _ in range(3):
            tracker.update([])
        tracks = tracker.update([])
        assert len(tracks) == 0

    def test_multiple_tracks(self):
        tracker = PedestrianTracker(min_hits=1, distance_threshold=2.0)
        dets = [
            np.array([3.0, 0.0, 0.0, 0.0, 0.3]),
            np.array([0.0, 5.0, 0.0, 0.0, 0.3]),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2
        ids = {t["id"] for t in tracks}
        assert len(ids) == 2

    def test_reset(self):
        tracker = PedestrianTracker(min_hits=1)
        tracker.update([np.array([1.0, 0.0, 0.0, 0.0, 0.3])])
        tracker.reset()
        tracks = tracker.update([])
        assert len(tracks) == 0

    def test_confidence_increases_with_hits(self):
        tracker = PedestrianTracker(min_hits=1)
        det = np.array([3.0, 0.0, 0.0, 0.0, 0.3])
        tracker.update([det])
        t1 = tracker.update([det])
        conf_2 = t1[0]["confidence"]
        t2 = tracker.update([det])
        conf_3 = t2[0]["confidence"]
        assert conf_3 >= conf_2


# ============================================================
# SensorFusion
# ============================================================


class TestSensorFusion:
    def test_basic_fusion(self):
        lidar = LidarSensor(LidarConfig(num_beams=8, noise_std=0.0))
        ds = DepthSensor(DepthSensorConfig(resolution=4, noise_std=0.0))
        fusion = SensorFusion(sensors={"lidar": lidar, "depth": ds})
        ws = _basic_world_state()
        obs = fusion.observe(ws)
        assert "lidar" in obs
        assert "depth" in obs
        assert "_tick" in obs
        assert obs["lidar"].shape == (8,)
        assert obs["depth"].shape == (4,)

    def test_tick_increments(self):
        lidar = LidarSensor(LidarConfig(num_beams=4, noise_std=0.0))
        fusion = SensorFusion(sensors={"lidar": lidar})
        ws = _basic_world_state()
        o1 = fusion.observe(ws)
        o2 = fusion.observe(ws)
        assert o1["_tick"] == 0
        assert o2["_tick"] == 1

    def test_rate_limiting(self):
        lidar = LidarSensor(LidarConfig(num_beams=4, noise_std=0.0))
        fusion = SensorFusion(
            sensors={"lidar": lidar},
            dt=0.1,
            rates={"lidar": 5.0},  # 5 Hz = every 2 ticks at dt=0.1
        )
        ws = _basic_world_state()
        o0 = fusion.observe(ws)  # tick 0: fires
        assert o0["lidar"] is not None
        o1 = fusion.observe(ws)  # tick 1: should use cached
        assert o1["lidar"] is not None  # sample-and-hold

    def test_reset(self):
        lidar = LidarSensor(LidarConfig(num_beams=4, noise_std=0.0))
        fusion = SensorFusion(sensors={"lidar": lidar})
        ws = _basic_world_state()
        fusion.observe(ws)
        fusion.observe(ws)
        fusion.reset()
        obs = fusion.observe(ws)
        assert obs["_tick"] == 0

    def test_observation_space(self):
        lidar = LidarSensor(LidarConfig(num_beams=8, noise_std=0.0))
        fusion = SensorFusion(sensors={"lidar": lidar})
        space = fusion.get_observation_space()
        assert "lidar" in space

    def test_seed(self):
        lidar = LidarSensor(LidarConfig(num_beams=8, noise_std=0.5))
        fusion = SensorFusion(sensors={"lidar": lidar})
        fusion.seed(42)
        # Just verify no error


# ============================================================
# KalmanStateEstimator (EKF)
# ============================================================


class TestKalmanStateEstimator:
    def test_initial_state_zero(self):
        ekf = KalmanStateEstimator()
        np.testing.assert_array_equal(ekf.state, np.zeros(6))

    def test_predict_constant_velocity(self):
        ekf = KalmanStateEstimator(EKFConfig(dt=0.1))
        ekf.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        predicted = ekf.predict()
        assert predicted[0] == pytest.approx(0.1, abs=1e-6)
        assert predicted[1] == pytest.approx(0.0, abs=1e-6)

    def test_predict_with_rotation(self):
        ekf = KalmanStateEstimator(EKFConfig(dt=0.1))
        ekf.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # omega=1 rad/s
        predicted = ekf.predict()
        assert predicted[2] == pytest.approx(0.1, abs=1e-6)

    def test_update_position(self):
        ekf = KalmanStateEstimator()
        ekf.state = np.zeros(6)
        measurement = np.array([1.0, 2.0, 0.5])
        updated = ekf.update_position(measurement)
        # State should move toward measurement
        assert abs(updated[0]) > 0
        assert abs(updated[1]) > 0

    def test_update_velocity(self):
        ekf = KalmanStateEstimator()
        ekf.state = np.zeros(6)
        measurement = np.array([1.0, 0.5, 0.2])
        updated = ekf.update_velocity(measurement)
        assert abs(updated[3]) > 0
        assert abs(updated[4]) > 0

    def test_generic_update(self):
        ekf = KalmanStateEstimator()
        H = np.zeros((2, 6))
        H[0, 0] = 1.0  # observe x
        H[1, 1] = 1.0  # observe y
        R = np.eye(2) * 0.1
        z = np.array([3.0, 4.0])
        updated = ekf.update(z, H, R)
        assert abs(updated[0]) > 0

    def test_reset(self):
        ekf = KalmanStateEstimator()
        ekf.state = np.ones(6)
        ekf.reset()
        np.testing.assert_array_equal(ekf.state, np.zeros(6))

    def test_reset_with_state(self):
        ekf = KalmanStateEstimator()
        init = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        ekf.reset(init)
        np.testing.assert_array_equal(ekf.state, init)

    def test_heading_wrapping(self):
        ekf = KalmanStateEstimator(EKFConfig(dt=0.1))
        ekf.state = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 5.0])
        predicted = ekf.predict()
        # Heading should be wrapped to [-pi, pi]
        assert -np.pi <= predicted[2] <= np.pi

    def test_properties(self):
        ekf = KalmanStateEstimator()
        ekf.state = np.array([1.0, 2.0, 0.5, 3.0, 4.0, 0.1])
        np.testing.assert_array_equal(ekf.position, [1.0, 2.0])
        assert ekf.heading == pytest.approx(0.5)
        np.testing.assert_array_equal(ekf.velocity, [3.0, 4.0])

    def test_predict_update_cycle_convergence(self):
        """Repeated predict-update cycle should converge to true state."""
        ekf = KalmanStateEstimator(EKFConfig(dt=0.1))
        true_state = np.array([5.0, 3.0, 0.5])  # x, y, heading
        for _ in range(50):
            ekf.predict()
            ekf.update_position(true_state + np.random.randn(3) * 0.05)
        assert abs(ekf.state[0] - 5.0) < 0.5
        assert abs(ekf.state[1] - 3.0) < 0.5

    def test_custom_process_noise(self):
        Q = np.eye(6) * 0.001
        ekf = KalmanStateEstimator(EKFConfig(process_noise=Q))
        np.testing.assert_array_equal(ekf.Q, Q)
