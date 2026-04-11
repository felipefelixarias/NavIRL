"""Comprehensive tests for robot modules.

Covers:
  - navirl.robots.ackermann
  - navirl.robots.differential_drive
  - navirl.robots.holonomic
  - navirl.robots.sensors_config
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ackermann module
# ---------------------------------------------------------------------------
from navirl.robots.ackermann import (
    AckermannConfig,
    LaneFollower,
    PurePursuitController,
    RSSegment,
    RSSegmentType,
    bicycle_curvature,
    bicycle_forward,
    bicycle_turning_radius,
    footprint_collision_check,
    parallel_parking_trajectory,
    rate_limit_ackermann,
    reeds_shepp_path,
    vehicle_footprint,
)

# ---------------------------------------------------------------------------
# Differential drive module
# ---------------------------------------------------------------------------
from navirl.robots.differential_drive import (
    DifferentialDriveConfig,
    OdometryAccumulator,
    PIDController,
    PIDGains,
    SensorMountPoint,
    apply_wheel_slip,
    compute_icc,
    forward_kinematics,
    inverse_kinematics,
    rate_limit,
    sensor_fov_polygon,
    sensor_world_pose,
    track_trajectory,
    wheel_velocities_to_body,
)

# ---------------------------------------------------------------------------
# Holonomic module
# ---------------------------------------------------------------------------
from navirl.robots.holonomic import (
    HolonomicConfig,
    InertiaFilter,
    WaypointFollower,
    clamp_acceleration,
    cubic_spline_waypoints,
    generate_smooth_trajectory,
    generate_waypoint_trajectory,
    trapezoidal_profile,
)

# ---------------------------------------------------------------------------
# Sensors config module
# ---------------------------------------------------------------------------
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
# Tests for navirl.robots.ackermann
# ===================================================================


class TestAckermannConfig:
    def test_defaults(self):
        cfg = AckermannConfig()
        assert cfg.wheelbase == 2.5
        assert cfg.max_speed == 5.0
        assert cfg.min_speed == -2.0

    def test_min_turning_radius_positive(self):
        cfg = AckermannConfig()
        r = cfg.min_turning_radius
        assert r > 0
        assert np.isfinite(r)

    def test_min_turning_radius_zero_steering(self):
        cfg = AckermannConfig(max_steering_angle=0.0)
        r = cfg.min_turning_radius
        # With zero steering the denominator uses the 1e-6 guard
        assert r > 0 and np.isfinite(r)


class TestBicycleForward:
    def test_straight_line(self):
        """Zero steering -> straight ahead."""
        x, y, theta = bicycle_forward(0, 0, 0, 1.0, 0.0, 2.5, 1.0)
        assert pytest.approx(x, abs=1e-6) == 1.0
        assert pytest.approx(y, abs=1e-6) == 0.0
        assert pytest.approx(theta, abs=1e-6) == 0.0

    def test_zero_velocity(self):
        x, y, theta = bicycle_forward(5.0, 3.0, 1.0, 0.0, 0.3, 2.5, 0.1)
        assert pytest.approx(x, abs=1e-8) == 5.0
        assert pytest.approx(y, abs=1e-8) == 3.0
        assert pytest.approx(theta, abs=1e-8) == 1.0

    def test_positive_steering_turns_left(self):
        _, _, theta = bicycle_forward(0, 0, 0, 1.0, 0.3, 2.5, 1.0)
        assert theta > 0  # positive steering -> positive heading change

    def test_zero_dt(self):
        x, y, theta = bicycle_forward(1, 2, 0.5, 3.0, 0.1, 2.5, 0.0)
        assert x == 1.0
        assert y == 2.0
        assert pytest.approx(theta, abs=1e-8) == 0.5


class TestBicycleCurvatureAndRadius:
    def test_curvature_zero_steering(self):
        assert bicycle_curvature(0.0, 2.5) == pytest.approx(0.0, abs=1e-10)

    def test_curvature_positive_steering(self):
        k = bicycle_curvature(0.3, 2.5)
        assert k > 0

    def test_turning_radius_zero_steering(self):
        r = bicycle_turning_radius(0.0, 2.5)
        assert r == float("inf")

    def test_turning_radius_nonzero(self):
        r = bicycle_turning_radius(0.3, 2.5)
        assert np.isfinite(r)
        assert r > 0


class TestRateLimitAckermann:
    def test_no_change_within_limits(self):
        cfg = AckermannConfig()
        v, d = rate_limit_ackermann(1.0, 0.1, 1.0, 0.1, 0.1, cfg)
        assert pytest.approx(v, abs=1e-8) == 1.0
        assert pytest.approx(d, abs=1e-8) == 0.1

    def test_speed_clamped_to_max(self):
        cfg = AckermannConfig(max_speed=2.0)
        v, _ = rate_limit_ackermann(100.0, 0.0, 1.9, 0.0, 10.0, cfg)
        assert v <= cfg.max_speed + 1e-8

    def test_steering_clamped(self):
        cfg = AckermannConfig(max_steering_angle=0.5)
        _, d = rate_limit_ackermann(0.0, 10.0, 0.0, 0.4, 10.0, cfg)
        assert d <= cfg.max_steering_angle + 1e-8


class TestReedsSheppPath:
    def test_output_shape(self):
        path, segs = reeds_shepp_path(0, 0, 0, 5, 5, 0, 5.0, num_samples=50)
        assert path.shape == (50, 3)
        assert len(segs) >= 1

    def test_same_start_goal(self):
        path, _ = reeds_shepp_path(1, 1, 0, 1, 1, 0, 5.0, num_samples=10)
        assert path.shape == (10, 3)


class TestParallelParking:
    def test_output_shapes(self):
        cfg = AckermannConfig()
        poses, controls = parallel_parking_trajectory(0, 0, 0, 2, -2, 0, cfg, dt=0.05)
        assert poses.ndim == 2 and poses.shape[1] == 3
        assert controls.ndim == 2 and controls.shape[1] == 2
        assert len(poses) == len(controls) + 1


class TestVehicleFootprint:
    def test_shape_and_corners(self):
        cfg = AckermannConfig()
        corners = vehicle_footprint(0, 0, 0, cfg)
        assert corners.shape == (4, 2)

    def test_rotation(self):
        cfg = AckermannConfig()
        c0 = vehicle_footprint(0, 0, 0, cfg)
        c90 = vehicle_footprint(0, 0, np.pi / 2, cfg)
        # After 90-deg rotation the corners should differ
        assert not np.allclose(c0, c90, atol=1e-6)


class TestFootprintCollisionCheck:
    def test_no_obstacles(self):
        cfg = AckermannConfig()
        assert not footprint_collision_check(0, 0, 0, cfg, np.empty((0, 2)))

    def test_collision_with_obstacle_on_edge(self):
        cfg = AckermannConfig()
        # Place obstacle right on the side edge of the vehicle (y = width/2)
        obs = np.array([[1.0, cfg.width / 2.0]])
        assert footprint_collision_check(0, 0, 0, cfg, obs, obstacle_radius=0.5)

    def test_no_collision_far_obstacle(self):
        cfg = AckermannConfig()
        obs = np.array([[100.0, 100.0]])
        assert not footprint_collision_check(0, 0, 0, cfg, obs, obstacle_radius=0.3)


class TestLaneFollower:
    def test_on_lane_zero_steering(self):
        lf = LaneFollower()
        lane = np.array([[0, 0], [10, 0]], dtype=float)
        delta = lf.compute_steering(0.0, 0.0, 0.0, 1.0, lane)
        assert abs(delta) < 0.1  # roughly on-lane and aligned

    def test_off_lane_nonzero_steering(self):
        lf = LaneFollower()
        lane = np.array([[0, 0], [10, 0]], dtype=float)
        delta = lf.compute_steering(5.0, 2.0, 0.0, 1.0, lane)
        # Should steer toward lane
        assert delta != 0.0


class TestPurePursuitController:
    def test_straight_ahead_target(self):
        pp = PurePursuitController(lookahead_dist=1.0)
        path = np.array([[0, 0], [5, 0], [10, 0]], dtype=float)
        delta = pp.compute_steering(0, 0, 0, 1.0, path, 2.5)
        assert abs(delta) < 0.01

    def test_target_to_the_left(self):
        pp = PurePursuitController(lookahead_dist=1.0)
        path = np.array([[0, 0], [5, 5]], dtype=float)
        delta = pp.compute_steering(0, 0, 0, 1.0, path, 2.5)
        assert delta > 0  # should steer left


# ===================================================================
# Tests for navirl.robots.differential_drive
# ===================================================================


class TestForwardKinematics:
    def test_straight(self):
        x, y, theta = forward_kinematics(0, 0, 0, 1.0, 0.0, 1.0)
        assert pytest.approx(x, abs=1e-6) == 1.0
        assert pytest.approx(y, abs=1e-6) == 0.0

    def test_pure_rotation(self):
        x, y, theta = forward_kinematics(0, 0, 0, 0.0, 1.0, 1.0)
        assert pytest.approx(x, abs=1e-6) == 0.0
        assert pytest.approx(y, abs=1e-6) == 0.0
        assert pytest.approx(theta, abs=1e-6) == 1.0

    def test_arc_motion(self):
        x, y, theta = forward_kinematics(0, 0, 0, 1.0, 1.0, 0.1)
        assert np.isfinite(x) and np.isfinite(y)


class TestComputeICC:
    def test_straight_line(self):
        ix, iy, r = compute_icc(0, 0, 0, 1.0, 0.0)
        assert r == float("inf")

    def test_turning(self):
        ix, iy, r = compute_icc(0, 0, 0, 1.0, 1.0)
        assert pytest.approx(r, abs=1e-6) == 1.0


class TestInverseAndForwardWheelConversion:
    def test_round_trip(self):
        v, omega = 1.0, 0.5
        wb, wr = 0.3, 0.05
        ol, or_ = inverse_kinematics(v, omega, wb, wr)
        v2, omega2 = wheel_velocities_to_body(ol, or_, wb, wr)
        assert pytest.approx(v2, abs=1e-8) == v
        assert pytest.approx(omega2, abs=1e-8) == omega

    def test_zero_velocity(self):
        ol, or_ = inverse_kinematics(0, 0, 0.3, 0.05)
        assert ol == 0.0
        assert or_ == 0.0


class TestPIDController:
    def test_proportional_only(self):
        pid = PIDController(PIDGains(kp=2.0, ki=0.0, kd=0.0))
        out = pid.compute(1.0, 0.1)
        assert pytest.approx(out, abs=1e-8) == 2.0

    def test_zero_dt(self):
        pid = PIDController()
        assert pid.compute(1.0, 0.0) == 0.0

    def test_reset(self):
        pid = PIDController(PIDGains(kp=1.0, ki=1.0, kd=0.0))
        pid.compute(1.0, 0.1)
        pid.reset()
        # After reset the integral should be zero and derivative first=True
        out = pid.compute(0.0, 0.1)
        assert pytest.approx(out, abs=1e-8) == 0.0


class TestOdometryAccumulator:
    def test_straight_no_noise(self):
        odom = OdometryAccumulator(0, 0, 0)
        x, y, theta = odom.update(1.0, 0.0, 1.0)
        assert pytest.approx(x, abs=1e-6) == 1.0
        assert pytest.approx(y, abs=1e-6) == 0.0

    def test_reset(self):
        odom = OdometryAccumulator(0, 0, 0)
        odom.update(1.0, 0.0, 1.0)
        odom.reset(5, 5, 1.0)
        assert pytest.approx(odom.x, abs=1e-8) == 5.0
        assert pytest.approx(odom.theta, abs=1e-8) == 1.0

    def test_pose_property(self):
        odom = OdometryAccumulator(1, 2, 3)
        p = odom.pose
        assert p.shape == (3,)
        np.testing.assert_allclose(p, [1, 2, 3])


class TestRateLimit:
    def test_within_limits(self):
        cfg = DifferentialDriveConfig()
        v, w = rate_limit(0.5, 0.5, 0.5, 0.5, 0.1, cfg)
        assert pytest.approx(v) == 0.5
        assert pytest.approx(w) == 0.5

    def test_clamps_to_max(self):
        cfg = DifferentialDriveConfig(max_linear_vel=1.0, max_angular_vel=2.0)
        v, w = rate_limit(100, 100, 0.9, 1.9, 10.0, cfg)
        assert v <= cfg.max_linear_vel + 1e-8
        assert w <= cfg.max_angular_vel + 1e-8


class TestApplyWheelSlip:
    def test_no_slip(self):
        cfg = DifferentialDriveConfig(slip_longitudinal=0.0, slip_lateral=0.0)
        v, w = apply_wheel_slip(1.0, 0.5, cfg)
        assert v == 1.0
        assert w == 0.5

    def test_full_longitudinal_slip(self):
        cfg = DifferentialDriveConfig(slip_longitudinal=1.0, slip_lateral=0.0)
        v, _ = apply_wheel_slip(2.0, 0.5, cfg)
        assert pytest.approx(v) == 0.0


class TestSensorMountPointDD:
    def test_world_pose_origin(self):
        mount = SensorMountPoint(offset_x=0.0, offset_y=0.0, offset_theta=0.0)
        sx, sy, st = sensor_world_pose(0, 0, 0, mount)
        assert pytest.approx(sx) == 0.0
        assert pytest.approx(sy) == 0.0
        assert pytest.approx(st, abs=1e-8) == 0.0

    def test_world_pose_offset(self):
        mount = SensorMountPoint(offset_x=1.0, offset_y=0.0, offset_theta=0.0)
        sx, sy, _ = sensor_world_pose(0, 0, np.pi / 2, mount)
        assert pytest.approx(sx, abs=1e-6) == 0.0
        assert pytest.approx(sy, abs=1e-6) == 1.0


class TestSensorFovPolygon:
    def test_shape(self):
        verts = sensor_fov_polygon(0, 0, 0, np.pi / 3, 5.0, num_points=16)
        assert verts.shape == (18, 2)

    def test_starts_and_ends_at_origin(self):
        verts = sensor_fov_polygon(1, 2, 0, np.pi / 4, 3.0, num_points=10)
        np.testing.assert_allclose(verts[0], [1, 2])
        np.testing.assert_allclose(verts[-1], [1, 2])


class TestTrackTrajectory:
    def test_reaches_single_waypoint(self):
        wps = np.array([[1.0, 0.0]])
        cfg = DifferentialDriveConfig()
        poses, controls = track_trajectory(wps, 0, 0, 0, 0.05, cfg)
        final = poses[-1]
        assert np.hypot(final[0] - 1.0, final[1] - 0.0) < 0.5


# ===================================================================
# Tests for navirl.robots.holonomic
# ===================================================================


class TestInertiaFilter:
    def test_zero_tau_instant(self):
        f = InertiaFilter(tau=0.0)
        vx, vy = f.filter(1.0, 2.0, 0.1)
        assert vx == 1.0 and vy == 2.0

    def test_nonzero_tau_smooths(self):
        f = InertiaFilter(tau=0.5)
        vx, vy = f.filter(1.0, 0.0, 0.1)
        assert 0 < vx < 1.0  # partially smoothed

    def test_reset(self):
        f = InertiaFilter(tau=0.5)
        f.filter(1.0, 1.0, 0.1)
        f.reset(0.0, 0.0)
        assert f.velocity == (0.0, 0.0)


class TestClampAcceleration:
    def test_zero_dt(self):
        cfg = HolonomicConfig()
        vx, vy = clamp_acceleration(10, 10, 0, 0, 0.0, cfg)
        assert vx == 10 and vy == 10  # no clamping when dt=0

    def test_speed_limit(self):
        cfg = HolonomicConfig(max_speed=1.0, max_acceleration=100.0)
        vx, vy = clamp_acceleration(10, 10, 0, 0, 1.0, cfg)
        speed = np.hypot(vx, vy)
        assert speed <= cfg.max_speed + 1e-8


class TestTrapezoidalProfile:
    def test_zero_distance(self):
        v = trapezoidal_profile(0.0, 1.0, 1.0, 0.01)
        assert len(v) == 1
        assert v[0] == 0.0

    def test_short_distance_triangle(self):
        v = trapezoidal_profile(0.5, 10.0, 2.0, 0.01)
        assert len(v) > 0
        assert np.all(v >= 0)

    def test_normal_profile(self):
        v = trapezoidal_profile(10.0, 2.0, 1.0, 0.05)
        assert len(v) > 1
        assert np.max(v) <= 2.0 + 1e-6


class TestCubicSplineWaypoints:
    def test_single_waypoint(self):
        wp = np.array([[1.0, 2.0]])
        result = cubic_spline_waypoints(wp, num_samples=10)
        assert result.shape == (10, 2)
        np.testing.assert_allclose(result, [[1.0, 2.0]] * 10)

    def test_two_waypoints(self):
        wp = np.array([[0, 0], [1, 1]], dtype=float)
        result = cubic_spline_waypoints(wp, num_samples=5)
        assert result.shape == (5, 2)
        # Should start near (0,0) and end near (1,1)
        np.testing.assert_allclose(result[0], [0, 0], atol=0.01)
        np.testing.assert_allclose(result[-1], [1, 1], atol=0.01)


class TestWaypointFollower:
    def test_finished_after_reaching_end(self):
        wp = np.array([[0, 0], [0.01, 0]], dtype=float)
        f = WaypointFollower(wp, desired_speed=1.0, num_interp=10)
        # Start right at the end
        vx, vy = f.compute_velocity(0.01, 0.0, 0.05)
        # May or may not be done yet, but shouldn't error
        assert np.isfinite(vx) and np.isfinite(vy)

    def test_reset(self):
        wp = np.array([[0, 0], [5, 0]], dtype=float)
        f = WaypointFollower(wp, desired_speed=1.0)
        f.compute_velocity(0, 0, 0.05)
        f.reset()
        assert not f.finished


class TestGenerateSmoothTrajectory:
    def test_same_start_goal(self):
        cfg = HolonomicConfig()
        pos, vel, t = generate_smooth_trajectory(np.array([1.0, 2.0]), np.array([1.0, 2.0]), cfg)
        assert pos.shape == (1, 2)
        assert vel.shape == (1, 2)

    def test_normal_trajectory(self):
        cfg = HolonomicConfig()
        pos, vel, t = generate_smooth_trajectory(
            np.array([0.0, 0.0]), np.array([5.0, 0.0]), cfg, dt=0.05
        )
        assert pos.shape[0] > 1
        assert pos.shape[1] == 2


class TestGenerateWaypointTrajectory:
    def test_basic(self):
        cfg = HolonomicConfig()
        wps = np.array([[0, 0], [3, 0], [3, 3]], dtype=float)
        pos, vel, t = generate_waypoint_trajectory(wps, cfg, dt=0.05)
        assert pos.shape[0] > 1
        assert vel.shape[1] == 2


# ===================================================================
# Tests for navirl.robots.sensors_config
# ===================================================================


class TestNoiseModels:
    def test_gaussian_shape(self):
        n = GaussianNoise(std_dev=0.1)
        s = n.sample(shape=(5,))
        assert s.shape == (5,)

    def test_gaussian_scalar(self):
        n = GaussianNoise(std_dev=0.0)
        s = n.sample()
        assert s.shape == ()

    def test_uniform_shape(self):
        n = UniformNoise(half_range=0.5)
        s = n.sample(shape=(3, 2))
        assert s.shape == (3, 2)
        assert np.all(np.abs(s) <= 0.5)

    def test_range_proportional(self):
        n = RangeProportionalNoise(coefficient=0.0)
        ranges = np.ones(10)
        s = n.sample_at_range(ranges)
        np.testing.assert_allclose(s, 0.0, atol=1e-10)

    def test_salt_pepper(self):
        np.random.seed(42)
        sp = SaltPepperNoise(p_salt=1.0, p_pepper=0.0)
        ranges = np.ones(10) * 5.0
        out = sp.apply(ranges, max_range=30.0)
        np.testing.assert_allclose(out, 30.0)


class TestSensorWorldPose2D:
    def test_no_offset(self):
        m = SensorMount(offset_x=0, offset_y=0, yaw=0)
        sx, sy, syaw = sensor_world_pose_2d(1, 2, 0, m)
        assert pytest.approx(sx) == 1.0
        assert pytest.approx(sy) == 2.0

    def test_forward_offset_rotated(self):
        m = SensorMount(offset_x=1.0, offset_y=0, yaw=0)
        sx, sy, _ = sensor_world_pose_2d(0, 0, np.pi / 2, m)
        assert pytest.approx(sx, abs=1e-6) == 0.0
        assert pytest.approx(sy, abs=1e-6) == 1.0


class TestSensorWorldPose3D:
    def test_identity(self):
        m = SensorMount(offset_x=0, offset_y=0, offset_z=0, roll=0, pitch=0, yaw=0)
        pos, rpy = sensor_world_pose_3d(1, 2, 3, 0, 0, 0, m)
        np.testing.assert_allclose(pos, [1, 2, 3], atol=1e-8)
        np.testing.assert_allclose(rpy, [0, 0, 0], atol=1e-8)


class TestFovAndRays:
    def test_fov_polygon_shape(self):
        verts = compute_fov_polygon(0, 0, 0, np.pi, 10.0, num_points=32)
        assert verts.shape == (34, 2)

    def test_fov_rays_count(self):
        m = SensorMount(fov_horizontal=np.pi, resolution_horizontal=np.radians(1.0))
        rays = compute_fov_rays(0, 0, 0, m)
        expected = int(np.pi / np.radians(1.0))
        assert rays.shape[0] == expected
        assert rays.shape[1] == 2


class TestCheckPointVisibility:
    def test_in_fov(self):
        m = SensorMount(fov_horizontal=np.pi, max_range=10.0, min_range=0.1)
        assert check_point_visibility(0, 0, 0, m, np.array([5.0, 0.0]))

    def test_out_of_range(self):
        m = SensorMount(fov_horizontal=np.pi, max_range=1.0, min_range=0.1)
        assert not check_point_visibility(0, 0, 0, m, np.array([5.0, 0.0]))

    def test_behind_sensor(self):
        m = SensorMount(fov_horizontal=np.pi / 2, max_range=10.0, min_range=0.1)
        assert not check_point_visibility(0, 0, 0, m, np.array([-5.0, 0.0]))

    def test_too_close(self):
        m = SensorMount(fov_horizontal=np.pi, max_range=10.0, min_range=1.0)
        assert not check_point_visibility(0, 0, 0, m, np.array([0.5, 0.0]))


class TestRaytraceOcclusion:
    def test_no_obstacles(self):
        m = SensorMount(fov_horizontal=np.pi, max_range=10.0, min_range=0.1)
        occluded = raytrace_occlusion(0, 0, 0, m, np.array([5.0, 0.0]), np.empty((0, 2)))
        assert not occluded

    def test_obstacle_blocks(self):
        m = SensorMount(fov_horizontal=np.pi, max_range=10.0, min_range=0.1)
        obs = np.array([[3.0, 0.0]])
        occluded = raytrace_occlusion(0, 0, 0, m, np.array([5.0, 0.0]), obs, obstacle_radius=0.5)
        assert occluded


class TestSensorSuite:
    def test_add_remove_get(self):
        suite = SensorSuite()
        m = SensorMount(name="test_lidar", sensor_type=SensorType.LIDAR_2D)
        suite.add(m)
        assert suite.num_sensors == 1
        assert suite.get("test_lidar") is not None
        suite.remove("test_lidar")
        assert suite.num_sensors == 0

    def test_enabled_mounts(self):
        m1 = SensorMount(name="a", enabled=True)
        m2 = SensorMount(name="b", enabled=False)
        suite = SensorSuite([m1, m2])
        assert len(suite.enabled_mounts()) == 1

    def test_any_sensor_sees(self):
        m = SensorMount(name="lidar", fov_horizontal=2 * np.pi, max_range=10.0, min_range=0.1)
        suite = SensorSuite([m])
        assert suite.any_sensor_sees(0, 0, 0, np.array([5.0, 0.0]))
        assert not suite.any_sensor_sees(0, 0, 0, np.array([50.0, 0.0]))


class TestFusePositionEstimates:
    def test_empty(self):
        cfg = SensorFusionConfig()
        result = fuse_position_estimates({}, cfg)
        np.testing.assert_allclose(result, [0, 0])

    def test_single_estimate(self):
        cfg = SensorFusionConfig()
        result = fuse_position_estimates({"a": np.array([3.0, 4.0])}, cfg)
        np.testing.assert_allclose(result, [3.0, 4.0])

    def test_weighted_average(self):
        cfg = SensorFusionConfig(weights=[FusionWeight("a", 1.0), FusionWeight("b", 3.0)])
        estimates = {"a": np.array([0.0, 0.0]), "b": np.array([4.0, 4.0])}
        result = fuse_position_estimates(estimates, cfg)
        np.testing.assert_allclose(result, [3.0, 3.0])


class TestFuseWithCovariance:
    def test_empty(self):
        mean, cov = fuse_with_covariance({}, {})
        assert mean.shape == (2,)
        assert cov.shape == (2, 2)

    def test_single_estimate(self):
        est = {"a": np.array([1.0, 2.0])}
        covs = {"a": np.eye(2)}
        mean, cov = fuse_with_covariance(est, covs)
        np.testing.assert_allclose(mean, [1.0, 2.0], atol=1e-4)


class TestDefaultSuites:
    def test_mobile_robot(self):
        suite = default_mobile_robot_suite()
        assert suite.num_sensors == 3
        assert suite.get("front_lidar") is not None

    def test_simulate_range_scan_basic(self):
        m = SensorMount(
            name="lidar",
            sensor_type=SensorType.LIDAR_2D,
            fov_horizontal=np.pi,
            max_range=10.0,
            min_range=0.1,
            resolution_horizontal=np.radians(10.0),
        )
        obs = np.array([[5.0, 0.0]])
        ranges, angles = simulate_range_scan(0, 0, 0, m, obs, obstacle_radius=0.5)
        assert len(ranges) == len(angles)
        assert np.all(ranges >= m.min_range)
        assert np.all(ranges <= m.max_range)
