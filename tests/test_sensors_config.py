"""Tests for navirl/robots/sensors_config.py.

Covers angle wrapping, 2D/3D sensor poses, FOV computation, ray generation,
point visibility, occlusion checking, range scan simulation, sensor suite,
sensor fusion, and preset configurations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.robots.sensors_config import (
    FusionWeight,
    GaussianNoise,
    RangeProportionalNoise,
    SaltPepperNoise,
    SensorFusionConfig,
    SensorMount,
    SensorSuite,
    SensorType,
    UniformNoise,
    _wrap_angle,
    autonomous_vehicle_suite,
    check_point_visibility,
    compute_fov_polygon,
    compute_fov_rays,
    compute_visible_points,
    default_mobile_robot_suite,
    fuse_position_estimates,
    fuse_with_covariance,
    raytrace_occlusion,
    sensor_world_pose_2d,
    sensor_world_pose_3d,
    simulate_range_scan,
)

# ===================================================================
# _wrap_angle
# ===================================================================


class TestWrapAngle:
    def test_zero(self):
        assert _wrap_angle(0.0) == pytest.approx(0.0)

    def test_pi(self):
        # pi wraps to -pi
        assert abs(_wrap_angle(math.pi)) == pytest.approx(math.pi)

    def test_positive_overflow(self):
        result = _wrap_angle(3 * math.pi)
        assert abs(result) == pytest.approx(math.pi, abs=1e-10)

    def test_negative(self):
        assert _wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)

    def test_large_positive(self):
        result = _wrap_angle(10 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_large_negative(self):
        result = _wrap_angle(-7 * math.pi)
        assert -math.pi <= result <= math.pi


# ===================================================================
# sensor_world_pose_2d
# ===================================================================


class TestSensorWorldPose2D:
    def test_no_offset(self):
        mount = SensorMount(offset_x=0.0, offset_y=0.0, yaw=0.0)
        sx, sy, syaw = sensor_world_pose_2d(1.0, 2.0, 0.0, mount)
        assert sx == pytest.approx(1.0)
        assert sy == pytest.approx(2.0)
        assert syaw == pytest.approx(0.0)

    def test_forward_offset(self):
        mount = SensorMount(offset_x=0.5, offset_y=0.0, yaw=0.0)
        sx, sy, syaw = sensor_world_pose_2d(0.0, 0.0, 0.0, mount)
        assert sx == pytest.approx(0.5)
        assert sy == pytest.approx(0.0)

    def test_lateral_offset(self):
        mount = SensorMount(offset_x=0.0, offset_y=0.5, yaw=0.0)
        sx, sy, syaw = sensor_world_pose_2d(0.0, 0.0, 0.0, mount)
        assert sx == pytest.approx(0.0)
        assert sy == pytest.approx(0.5)

    def test_rotated_robot(self):
        mount = SensorMount(offset_x=1.0, offset_y=0.0, yaw=0.0)
        sx, sy, syaw = sensor_world_pose_2d(0.0, 0.0, math.pi / 2, mount)
        assert sx == pytest.approx(0.0, abs=1e-10)
        assert sy == pytest.approx(1.0)

    def test_sensor_yaw_offset(self):
        mount = SensorMount(offset_x=0.0, offset_y=0.0, yaw=math.pi / 4)
        sx, sy, syaw = sensor_world_pose_2d(0.0, 0.0, math.pi / 4, mount)
        assert syaw == pytest.approx(math.pi / 2)


# ===================================================================
# sensor_world_pose_3d
# ===================================================================


class TestSensorWorldPose3D:
    def test_no_offset_3d(self):
        mount = SensorMount(offset_x=0, offset_y=0, offset_z=0, roll=0, pitch=0, yaw=0)
        pos, rpy = sensor_world_pose_3d(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, mount)
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-10)
        np.testing.assert_allclose(rpy, [0.0, 0.0, 0.0], atol=1e-10)

    def test_vertical_offset(self):
        mount = SensorMount(offset_x=0, offset_y=0, offset_z=1.5)
        pos, rpy = sensor_world_pose_3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mount)
        assert pos[2] == pytest.approx(1.5)

    def test_forward_offset_rotated(self):
        mount = SensorMount(offset_x=1.0, offset_y=0, offset_z=0)
        pos, rpy = sensor_world_pose_3d(0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2, mount)
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(1.0)

    def test_rpy_addition(self):
        mount = SensorMount(roll=0.1, pitch=0.2, yaw=0.3)
        pos, rpy = sensor_world_pose_3d(0, 0, 0, 0.1, 0.2, 0.3, mount)
        assert rpy[0] == pytest.approx(_wrap_angle(0.2))
        assert rpy[1] == pytest.approx(_wrap_angle(0.4))
        assert rpy[2] == pytest.approx(_wrap_angle(0.6))


# ===================================================================
# compute_fov_polygon
# ===================================================================


class TestComputeFovPolygon:
    def test_shape(self):
        poly = compute_fov_polygon(0, 0, 0, math.pi / 2, 10.0, num_points=32)
        assert poly.shape == (34, 2)

    def test_starts_and_ends_at_sensor(self):
        poly = compute_fov_polygon(5.0, 3.0, 0, math.pi, 5.0, num_points=16)
        np.testing.assert_allclose(poly[0], [5.0, 3.0])
        np.testing.assert_allclose(poly[-1], [5.0, 3.0])

    def test_arc_within_range(self):
        poly = compute_fov_polygon(0, 0, 0, 2 * math.pi, 10.0, num_points=64)
        dists = np.linalg.norm(poly[1:-1], axis=1)
        np.testing.assert_allclose(dists, 10.0, atol=1e-10)


# ===================================================================
# compute_fov_rays
# ===================================================================


class TestComputeFovRays:
    def test_ray_count(self):
        mount = SensorMount(
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(1.0),
        )
        rays = compute_fov_rays(0, 0, 0, mount)
        expected_n = int(math.pi / math.radians(1.0))
        assert rays.shape[0] == expected_n
        assert rays.shape[1] == 2

    def test_rays_are_unit_vectors(self):
        mount = SensorMount(
            fov_horizontal=math.pi / 2,
            resolution_horizontal=math.radians(5.0),
        )
        rays = compute_fov_rays(0, 0, 0, mount)
        norms = np.linalg.norm(rays, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_ray_direction_centered(self):
        mount = SensorMount(
            fov_horizontal=math.radians(10.0),
            resolution_horizontal=math.radians(10.0),
        )
        rays = compute_fov_rays(0, 0, 0, mount)
        # Single ray should point forward (along x-axis)
        assert rays[0, 0] == pytest.approx(1.0, abs=0.1)


# ===================================================================
# check_point_visibility
# ===================================================================


class TestCheckPointVisibility:
    def test_visible_point(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=math.pi)
        point = np.array([5.0, 0.0])
        assert check_point_visibility(0, 0, 0, mount, point) is True

    def test_out_of_range(self):
        mount = SensorMount(max_range=5.0, min_range=0.1, fov_horizontal=2 * math.pi)
        point = np.array([10.0, 0.0])
        assert check_point_visibility(0, 0, 0, mount, point) is False

    def test_too_close(self):
        mount = SensorMount(max_range=10.0, min_range=1.0, fov_horizontal=2 * math.pi)
        point = np.array([0.5, 0.0])
        assert check_point_visibility(0, 0, 0, mount, point) is False

    def test_outside_fov(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=math.pi / 4)
        # Point directly behind
        point = np.array([-5.0, 0.0])
        assert check_point_visibility(0, 0, 0, mount, point) is False

    def test_edge_of_fov(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=math.pi)
        # Point at exactly half-FOV angle
        angle = math.pi / 2 - 0.01
        point = np.array([5.0 * math.cos(angle), 5.0 * math.sin(angle)])
        assert check_point_visibility(0, 0, 0, mount, point) is True


# ===================================================================
# raytrace_occlusion
# ===================================================================


class TestRaytraceOcclusion:
    def test_no_obstacles(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=2 * math.pi)
        target = np.array([5.0, 0.0])
        obstacles = np.empty((0, 2))
        assert raytrace_occlusion(0, 0, 0, mount, target, obstacles) is False

    def test_obstacle_blocks(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=2 * math.pi)
        target = np.array([5.0, 0.0])
        # Obstacle directly in front at distance 3
        obstacles = np.array([[3.0, 0.0]])
        assert raytrace_occlusion(0, 0, 0, mount, target, obstacles, obstacle_radius=0.5) is True

    def test_obstacle_beside_ray(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=2 * math.pi)
        target = np.array([5.0, 0.0])
        # Obstacle far off to the side
        obstacles = np.array([[3.0, 5.0]])
        assert raytrace_occlusion(0, 0, 0, mount, target, obstacles, obstacle_radius=0.3) is False

    def test_obstacle_behind_target(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=2 * math.pi)
        target = np.array([3.0, 0.0])
        obstacles = np.array([[6.0, 0.0]])
        assert raytrace_occlusion(0, 0, 0, mount, target, obstacles, obstacle_radius=0.3) is False

    def test_target_outside_fov(self):
        mount = SensorMount(max_range=10.0, min_range=0.1, fov_horizontal=math.pi / 4)
        target = np.array([-5.0, 0.0])
        obstacles = np.empty((0, 2))
        # Outside FOV is treated as occluded
        assert raytrace_occlusion(0, 0, 0, mount, target, obstacles) is True

    def test_coincident_point(self):
        mount = SensorMount(max_range=10.0, min_range=0.0, fov_horizontal=2 * math.pi)
        target = np.array([0.0, 0.0])
        obstacles = np.empty((0, 2))
        # Target at sensor position
        assert raytrace_occlusion(0, 0, 0, mount, target, obstacles) is False


# ===================================================================
# compute_visible_points
# ===================================================================


class TestComputeVisiblePoints:
    def test_all_visible(self):
        mount = SensorMount(max_range=20.0, min_range=0.1, fov_horizontal=2 * math.pi)
        points = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0]])
        vis = compute_visible_points(0, 0, 0, mount, points)
        assert vis.all()

    def test_some_occluded(self):
        mount = SensorMount(max_range=20.0, min_range=0.1, fov_horizontal=2 * math.pi)
        points = np.array([[5.0, 0.0], [0.0, 5.0]])
        obstacles = np.array([[3.0, 0.0]])
        vis = compute_visible_points(0, 0, 0, mount, points, obstacles, obstacle_radius=0.5)
        assert vis[0] is np.False_  # Occluded
        assert vis[1] is np.True_  # Not occluded

    def test_no_obstacles(self):
        mount = SensorMount(max_range=20.0, min_range=0.1, fov_horizontal=2 * math.pi)
        points = np.array([[5.0, 0.0]])
        vis = compute_visible_points(0, 0, 0, mount, points, obstacles=None)
        assert vis[0] is np.True_


# ===================================================================
# Noise models
# ===================================================================


class TestNoiseModels:
    def test_gaussian_noise_shape(self):
        noise = GaussianNoise(std_dev=0.1)
        sample = noise.sample((10,))
        assert sample.shape == (10,)

    def test_gaussian_noise_scalar(self):
        noise = GaussianNoise(std_dev=0.01)
        sample = noise.sample()
        assert isinstance(sample, np.ndarray)

    def test_uniform_noise_range(self):
        noise = UniformNoise(half_range=1.0)
        np.random.seed(42)
        samples = noise.sample((10000,))
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_range_proportional_noise(self):
        noise = RangeProportionalNoise(coefficient=0.01)
        ranges = np.array([10.0, 20.0, 30.0])
        result = noise.sample_at_range(ranges)
        assert result.shape == ranges.shape

    def test_salt_pepper_noise_preserves_shape(self):
        noise = SaltPepperNoise(p_salt=0.1, p_pepper=0.05)
        ranges = np.full(100, 5.0)
        result = noise.apply(ranges, max_range=30.0)
        assert result.shape == ranges.shape

    def test_salt_pepper_applies_corruption(self):
        np.random.seed(42)
        noise = SaltPepperNoise(p_salt=0.5, p_pepper=0.0)
        ranges = np.full(1000, 5.0)
        result = noise.apply(ranges, max_range=30.0)
        # With 50% salt probability, many should be max_range
        assert (result == 30.0).sum() > 100


# ===================================================================
# simulate_range_scan
# ===================================================================


class TestSimulateRangeScan:
    def test_no_obstacles(self):
        mount = SensorMount(
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(10.0),
            max_range=10.0,
            min_range=0.1,
        )
        obstacles = np.empty((0, 2))
        ranges, angles = simulate_range_scan(0, 0, 0, mount, obstacles)
        assert ranges.shape == angles.shape
        np.testing.assert_allclose(ranges, mount.max_range)

    def test_with_obstacle(self):
        mount = SensorMount(
            fov_horizontal=math.radians(10.0),
            resolution_horizontal=math.radians(10.0),
            max_range=20.0,
            min_range=0.1,
        )
        obstacles = np.array([[5.0, 0.0]])
        ranges, angles = simulate_range_scan(0, 0, 0, mount, obstacles, obstacle_radius=0.5)
        # At least one ray should detect the obstacle
        assert ranges.min() < mount.max_range

    def test_gaussian_noise_applied(self):
        mount = SensorMount(
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(5.0),
            max_range=10.0,
            min_range=0.1,
        )
        obstacles = np.empty((0, 2))
        noise = GaussianNoise(std_dev=0.5)
        ranges, _ = simulate_range_scan(0, 0, 0, mount, obstacles, noise=noise)
        # With noise, not all ranges will be exactly max_range
        # (clipped to [min, max] though)
        assert ranges.min() >= mount.min_range
        assert ranges.max() <= mount.max_range

    def test_range_proportional_noise_applied(self):
        mount = SensorMount(
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(5.0),
            max_range=10.0,
            min_range=0.1,
        )
        obstacles = np.empty((0, 2))
        noise = RangeProportionalNoise(coefficient=0.1)
        ranges, _ = simulate_range_scan(0, 0, 0, mount, obstacles, noise=noise)
        assert ranges.min() >= mount.min_range

    def test_salt_pepper_noise_applied(self):
        np.random.seed(42)
        mount = SensorMount(
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(1.0),
            max_range=10.0,
            min_range=0.1,
        )
        obstacles = np.array([[5.0, 0.0]])
        sp = SaltPepperNoise(p_salt=0.3, p_pepper=0.3)
        ranges, _ = simulate_range_scan(
            0, 0, 0, mount, obstacles, obstacle_radius=0.3, salt_pepper=sp
        )
        # Some readings should be corrupted to 0 or max_range
        assert (ranges == 0.0).sum() > 0 or (ranges == mount.max_range).sum() > 0


# ===================================================================
# SensorSuite
# ===================================================================


class TestSensorSuite:
    def test_empty_suite(self):
        suite = SensorSuite()
        assert suite.num_sensors == 0
        assert suite.mounts == []

    def test_add_and_get(self):
        suite = SensorSuite()
        mount = SensorMount(name="test_lidar")
        suite.add(mount)
        assert suite.num_sensors == 1
        assert suite.get("test_lidar") is not None
        assert suite.get("nonexistent") is None

    def test_remove(self):
        suite = SensorSuite([SensorMount(name="a"), SensorMount(name="b")])
        suite.remove("a")
        assert suite.num_sensors == 1
        assert suite.get("a") is None
        assert suite.get("b") is not None

    def test_enabled_mounts(self):
        suite = SensorSuite(
            [
                SensorMount(name="on", enabled=True),
                SensorMount(name="off", enabled=False),
            ]
        )
        enabled = suite.enabled_mounts()
        assert len(enabled) == 1
        assert enabled[0].name == "on"

    def test_world_poses_2d(self):
        mount = SensorMount(name="front", offset_x=0.5, enabled=True)
        suite = SensorSuite([mount])
        poses = suite.world_poses_2d(0.0, 0.0, 0.0)
        assert len(poses) == 1
        name, sx, sy, syaw = poses[0]
        assert name == "front"
        assert sx == pytest.approx(0.5)

    def test_world_poses_2d_skips_disabled(self):
        suite = SensorSuite(
            [
                SensorMount(name="on", enabled=True),
                SensorMount(name="off", enabled=False),
            ]
        )
        poses = suite.world_poses_2d(0, 0, 0)
        assert len(poses) == 1

    def test_combined_fov_polygon(self):
        mount = SensorMount(
            name="front",
            fov_horizontal=math.pi / 2,
            max_range=5.0,
            enabled=True,
        )
        suite = SensorSuite([mount])
        polys = suite.combined_fov_polygon(0, 0, 0)
        assert len(polys) == 1
        assert polys[0].shape[1] == 2

    def test_any_sensor_sees(self):
        mount = SensorMount(
            name="front",
            max_range=10.0,
            min_range=0.1,
            fov_horizontal=math.pi,
            enabled=True,
        )
        suite = SensorSuite([mount])
        assert suite.any_sensor_sees(0, 0, 0, np.array([5.0, 0.0])) is True
        assert suite.any_sensor_sees(0, 0, 0, np.array([-5.0, 0.0])) is False

    def test_visible_from_all(self):
        mount = SensorMount(
            name="front",
            max_range=10.0,
            min_range=0.1,
            fov_horizontal=2 * math.pi,
            enabled=True,
        )
        suite = SensorSuite([mount])
        points = np.array([[3.0, 0.0], [0.0, 3.0]])
        vis = suite.visible_from_all(0, 0, 0, points)
        assert vis.all()

    def test_visible_from_all_with_obstacles(self):
        mount = SensorMount(
            name="front",
            max_range=10.0,
            min_range=0.1,
            fov_horizontal=2 * math.pi,
            enabled=True,
        )
        suite = SensorSuite([mount])
        points = np.array([[5.0, 0.0]])
        obstacles = np.array([[3.0, 0.0]])
        vis = suite.visible_from_all(0, 0, 0, points, obstacles, obstacle_radius=0.5)
        assert not vis[0]

    def test_scan_all(self):
        lidar = SensorMount(
            name="lidar",
            sensor_type=SensorType.LIDAR_2D,
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(10.0),
            max_range=10.0,
            min_range=0.1,
            enabled=True,
        )
        camera = SensorMount(
            name="camera",
            sensor_type=SensorType.CAMERA_RGB,
            enabled=True,
        )
        suite = SensorSuite([lidar, camera])
        obstacles = np.array([[5.0, 0.0]])
        results = suite.scan_all(0, 0, 0, obstacles)
        # Only range sensors (lidar) produce scans
        assert "lidar" in results
        assert "camera" not in results
        ranges, angles = results["lidar"]
        assert ranges.shape == angles.shape

    def test_scan_all_with_noise(self):
        lidar = SensorMount(
            name="lidar",
            sensor_type=SensorType.LIDAR_2D,
            fov_horizontal=math.pi,
            resolution_horizontal=math.radians(5.0),
            max_range=10.0,
            min_range=0.1,
            enabled=True,
        )
        suite = SensorSuite([lidar])
        obstacles = np.empty((0, 2))
        noise = GaussianNoise(std_dev=0.1)
        results = suite.scan_all(0, 0, 0, obstacles, noise=noise)
        assert "lidar" in results


# ===================================================================
# Sensor fusion
# ===================================================================


class TestFusePositionEstimates:
    def test_empty_estimates(self):
        config = SensorFusionConfig()
        result = fuse_position_estimates({}, config)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_single_estimate(self):
        config = SensorFusionConfig(weights=[FusionWeight(sensor_name="lidar", weight=1.0)])
        estimates = {"lidar": np.array([5.0, 3.0])}
        result = fuse_position_estimates(estimates, config)
        np.testing.assert_allclose(result, [5.0, 3.0])

    def test_weighted_average(self):
        config = SensorFusionConfig(
            weights=[
                FusionWeight(sensor_name="lidar", weight=3.0),
                FusionWeight(sensor_name="gps", weight=1.0),
            ]
        )
        estimates = {
            "lidar": np.array([4.0, 0.0]),
            "gps": np.array([8.0, 0.0]),
        }
        result = fuse_position_estimates(estimates, config)
        # Weighted: (3*4 + 1*8) / 4 = 5.0
        assert result[0] == pytest.approx(5.0)

    def test_default_weight(self):
        """Sensors without configured weight get default weight 1.0."""
        config = SensorFusionConfig(weights=[])
        estimates = {
            "a": np.array([2.0, 0.0]),
            "b": np.array([4.0, 0.0]),
        }
        result = fuse_position_estimates(estimates, config)
        np.testing.assert_allclose(result, [3.0, 0.0])

    def test_3d_estimates(self):
        config = SensorFusionConfig(weights=[FusionWeight(sensor_name="s1", weight=1.0)])
        estimates = {"s1": np.array([1.0, 2.0, 3.0])}
        result = fuse_position_estimates(estimates, config)
        assert result.shape == (3,)


class TestFuseWithCovariance:
    def test_empty_estimates(self):
        mean, cov = fuse_with_covariance({}, {})
        np.testing.assert_array_equal(mean, np.zeros(2))
        assert cov[0, 0] == pytest.approx(1e6)

    def test_single_estimate(self):
        estimates = {"lidar": np.array([5.0, 3.0])}
        covariances = {"lidar": np.eye(2) * 0.1}
        mean, cov = fuse_with_covariance(estimates, covariances)
        np.testing.assert_allclose(mean, [5.0, 3.0], atol=0.01)

    def test_two_estimates(self):
        estimates = {
            "a": np.array([4.0, 0.0]),
            "b": np.array([6.0, 0.0]),
        }
        covariances = {
            "a": np.eye(2) * 1.0,
            "b": np.eye(2) * 1.0,
        }
        mean, cov = fuse_with_covariance(estimates, covariances)
        # Equal covariance => average
        np.testing.assert_allclose(mean, [5.0, 0.0], atol=0.01)

    def test_lower_covariance_gets_more_weight(self):
        estimates = {
            "precise": np.array([10.0, 0.0]),
            "noisy": np.array([0.0, 0.0]),
        }
        covariances = {
            "precise": np.eye(2) * 0.01,
            "noisy": np.eye(2) * 100.0,
        }
        mean, cov = fuse_with_covariance(estimates, covariances)
        # Should be much closer to the precise estimate
        assert mean[0] > 9.0

    def test_missing_covariance_uses_identity(self):
        estimates = {"s1": np.array([5.0, 5.0])}
        mean, cov = fuse_with_covariance(estimates, {})
        np.testing.assert_allclose(mean, [5.0, 5.0], atol=0.1)


# ===================================================================
# Preset suites
# ===================================================================


class TestPresetSuites:
    def test_default_mobile_robot_suite(self):
        suite = default_mobile_robot_suite()
        assert suite.num_sensors == 3
        assert suite.get("front_lidar") is not None
        assert suite.get("rear_ultrasonic") is not None
        assert suite.get("wheel_encoders") is not None

    def test_autonomous_vehicle_suite(self):
        suite = autonomous_vehicle_suite()
        assert suite.num_sensors == 3
        assert suite.get("roof_lidar") is not None
        assert suite.get("front_camera") is not None
        assert suite.get("gps") is not None

    def test_mobile_suite_scans(self):
        """Mobile suite can run scans."""
        suite = default_mobile_robot_suite()
        obstacles = np.array([[3.0, 0.0]])
        results = suite.scan_all(0, 0, 0, obstacles)
        # front_lidar and rear_ultrasonic are range sensors
        assert len(results) >= 1

    def test_vehicle_suite_visibility(self):
        suite = autonomous_vehicle_suite()
        point = np.array([5.0, 0.0])
        # Roof lidar has 360° FOV
        assert suite.any_sensor_sees(0, 0, 0, point) is True
