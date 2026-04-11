from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from navirl.ros.conversions import (
    action_to_twist,
    image_to_numpy,
    laser_scan_to_lidar_obs,
    odometry_to_state,
    person_array_to_social_obs,
    pose_to_goal,
)

# ---------------------------------------------------------------------------
# Helper factories for duck-typed message objects
# ---------------------------------------------------------------------------


def _make_laser_scan(
    ranges: list[float],
    range_min: float = 0.0,
    range_max: float = 30.0,
) -> SimpleNamespace:
    return SimpleNamespace(ranges=ranges, range_min=range_min, range_max=range_max)


def _make_odometry(
    x: float = 0.0,
    y: float = 0.0,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    vx: float = 0.0,
    vy: float = 0.0,
    omega: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
            ),
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=vx, y=vy, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=omega),
            ),
        ),
    )


def _make_track(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    track_id: int = 0,
    detection_score: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
            ),
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=vx, y=vy, z=0.0),
                angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            ),
        ),
        track_id=track_id,
        detection_score=detection_score,
    )


def _make_person_array(tracks: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(tracks=tracks)


def _make_image(
    height: int,
    width: int,
    data: bytes,
    encoding: str = "rgb8",
) -> SimpleNamespace:
    return SimpleNamespace(height=height, width=width, data=data, encoding=encoding)


# ===================================================================
# laser_scan_to_lidar_obs
# ===================================================================


class TestLaserScanToLidarObs:
    def test_normal_ranges(self):
        msg = _make_laser_scan([1.0, 2.0, 3.0], range_min=0.0, range_max=30.0)
        result = laser_scan_to_lidar_obs(msg)
        assert result.dtype == np.float32
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_inf_replaced_with_range_max(self):
        msg = _make_laser_scan([1.0, float("inf"), float("-inf")], range_max=10.0)
        result = laser_scan_to_lidar_obs(msg)
        np.testing.assert_array_almost_equal(result, [1.0, 10.0, 10.0])

    def test_nan_replaced_with_range_max(self):
        msg = _make_laser_scan([float("nan"), 5.0], range_max=20.0)
        result = laser_scan_to_lidar_obs(msg)
        np.testing.assert_array_almost_equal(result, [20.0, 5.0])

    def test_clipping_to_range(self):
        msg = _make_laser_scan([0.5, 50.0, -1.0], range_min=1.0, range_max=10.0)
        result = laser_scan_to_lidar_obs(msg)
        # 0.5 clipped up to 1.0, 50.0 clipped down to 10.0, -1.0 clipped up to 1.0
        np.testing.assert_array_almost_equal(result, [1.0, 10.0, 1.0])

    def test_empty_ranges(self):
        msg = _make_laser_scan([], range_max=10.0)
        result = laser_scan_to_lidar_obs(msg)
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_single_element(self):
        msg = _make_laser_scan([7.5], range_max=30.0)
        result = laser_scan_to_lidar_obs(msg)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(7.5)

    def test_all_invalid(self):
        msg = _make_laser_scan(
            [float("inf"), float("nan"), float("-inf")], range_max=5.0
        )
        result = laser_scan_to_lidar_obs(msg)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0, 5.0])

    def test_default_range_max_when_missing(self):
        # range_max defaults to 30.0 via getattr
        msg = SimpleNamespace(ranges=[float("inf")])
        result = laser_scan_to_lidar_obs(msg)
        assert result[0] == pytest.approx(30.0)


# ===================================================================
# odometry_to_state
# ===================================================================


class TestOdometryToState:
    def test_identity_orientation(self):
        msg = _make_odometry(x=1.0, y=2.0, vx=0.5, vy=0.1, omega=0.3)
        state = odometry_to_state(msg)
        assert state["x"] == pytest.approx(1.0)
        assert state["y"] == pytest.approx(2.0)
        assert state["yaw"] == pytest.approx(0.0)  # identity quaternion -> yaw 0
        assert state["vx"] == pytest.approx(0.5)
        assert state["vy"] == pytest.approx(0.1)
        assert state["omega"] == pytest.approx(0.3)

    def test_90_degree_yaw(self):
        # Quaternion for 90-degree rotation about z: qz=sin(pi/4), qw=cos(pi/4)
        qz = math.sin(math.pi / 4)
        qw = math.cos(math.pi / 4)
        msg = _make_odometry(qz=qz, qw=qw)
        state = odometry_to_state(msg)
        assert state["yaw"] == pytest.approx(math.pi / 2, abs=1e-6)

    def test_180_degree_yaw(self):
        # Quaternion for 180-degree rotation: qz=1, qw=0
        msg = _make_odometry(qz=1.0, qw=0.0)
        state = odometry_to_state(msg)
        assert abs(state["yaw"]) == pytest.approx(math.pi, abs=1e-6)

    def test_zero_state(self):
        msg = _make_odometry()
        state = odometry_to_state(msg)
        assert set(state.keys()) == {"x", "y", "yaw", "vx", "vy", "omega"}
        for key in state:
            assert state[key] == pytest.approx(0.0)

    def test_negative_velocities(self):
        msg = _make_odometry(vx=-1.0, vy=-2.0, omega=-0.5)
        state = odometry_to_state(msg)
        assert state["vx"] == pytest.approx(-1.0)
        assert state["vy"] == pytest.approx(-2.0)
        assert state["omega"] == pytest.approx(-0.5)


# ===================================================================
# person_array_to_social_obs
# ===================================================================


class TestPersonArrayToSocialObs:
    def test_empty_tracks(self):
        msg = _make_person_array([])
        result = person_array_to_social_obs(msg)
        assert result.shape == (0, 7)
        assert result.dtype == np.float64

    def test_single_track(self):
        track = _make_track(x=1.0, y=2.0, vx=0.5, vy=-0.3, track_id=42, detection_score=0.9)
        msg = _make_person_array([track])
        result = person_array_to_social_obs(msg)
        assert result.shape == (1, 7)
        assert result[0, 0] == pytest.approx(1.0)  # x
        assert result[0, 1] == pytest.approx(2.0)  # y
        assert result[0, 2] == pytest.approx(0.5)  # vx
        assert result[0, 3] == pytest.approx(-0.3)  # vy
        assert result[0, 4] == pytest.approx(0.0)  # theta (identity quat)
        assert result[0, 5] == pytest.approx(42.0)  # track_id
        assert result[0, 6] == pytest.approx(0.9)  # detection_score

    def test_multiple_tracks(self):
        tracks = [
            _make_track(x=1.0, y=2.0, track_id=1),
            _make_track(x=3.0, y=4.0, track_id=2),
            _make_track(x=5.0, y=6.0, track_id=3),
        ]
        msg = _make_person_array(tracks)
        result = person_array_to_social_obs(msg)
        assert result.shape == (3, 7)
        np.testing.assert_array_almost_equal(result[:, 0], [1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(result[:, 1], [2.0, 4.0, 6.0])

    def test_no_tracks_attribute_uses_persons(self):
        track = _make_track(x=10.0, y=20.0, track_id=99)
        msg = SimpleNamespace(persons=[track])
        result = person_array_to_social_obs(msg)
        assert result.shape == (1, 7)
        assert result[0, 0] == pytest.approx(10.0)

    def test_no_tracks_or_persons_returns_empty(self):
        msg = SimpleNamespace()
        result = person_array_to_social_obs(msg)
        assert result.shape == (0, 7)

    def test_fallback_detection_id(self):
        # Track without track_id but with detection_id
        track = _make_track(x=1.0, y=2.0)
        del track.track_id
        track.detection_id = 77
        msg = _make_person_array([track])
        result = person_array_to_social_obs(msg)
        assert result[0, 5] == pytest.approx(77.0)

    def test_fallback_is_matched(self):
        # Track without detection_score but with is_matched
        track = _make_track(x=1.0, y=2.0)
        del track.detection_score
        track.is_matched = 0.5
        msg = _make_person_array([track])
        result = person_array_to_social_obs(msg)
        assert result[0, 6] == pytest.approx(0.5)

    def test_orientation_in_track(self):
        qz = math.sin(math.pi / 4)
        qw = math.cos(math.pi / 4)
        track = _make_track(qz=qz, qw=qw)
        msg = _make_person_array([track])
        result = person_array_to_social_obs(msg)
        assert result[0, 4] == pytest.approx(math.pi / 2, abs=1e-6)


# ===================================================================
# action_to_twist
# ===================================================================


class TestActionToTwist:
    def test_continuous_action(self):
        result = action_to_twist(np.array([1.5, 0.3]), action_type="continuous")
        assert result["linear_x"] == pytest.approx(1.5)
        assert result["angular_z"] == pytest.approx(0.3)
        assert result["linear_y"] == 0.0
        assert result["linear_z"] == 0.0
        assert result["angular_x"] == 0.0
        assert result["angular_y"] == 0.0

    def test_continuous_single_element(self):
        result = action_to_twist(np.array([2.0]), action_type="continuous")
        assert result["linear_x"] == pytest.approx(2.0)
        assert result["angular_z"] == pytest.approx(0.0)

    def test_continuous_empty(self):
        result = action_to_twist(np.array([]), action_type="continuous")
        assert result["linear_x"] == pytest.approx(0.0)
        assert result["angular_z"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "idx, expected_linear, expected_angular",
        [
            (0, 0.0, 0.0),    # stop
            (1, 0.5, 0.0),    # forward
            (2, -0.3, 0.0),   # backward
            (3, 0.2, 0.5),    # turn left
            (4, 0.2, -0.5),   # turn right
            (5, 0.5, 0.3),    # forward-left
            (6, 0.5, -0.3),   # forward-right
        ],
    )
    def test_discrete_actions(self, idx, expected_linear, expected_angular):
        result = action_to_twist(np.array([idx]), action_type="discrete")
        assert result["linear_x"] == pytest.approx(expected_linear)
        assert result["angular_z"] == pytest.approx(expected_angular)

    def test_discrete_unknown_index(self):
        result = action_to_twist(np.array([99]), action_type="discrete")
        assert result["linear_x"] == pytest.approx(0.0)
        assert result["angular_z"] == pytest.approx(0.0)

    def test_discrete_negative_index(self):
        result = action_to_twist(np.array([-1]), action_type="discrete")
        assert result["linear_x"] == pytest.approx(0.0)
        assert result["angular_z"] == pytest.approx(0.0)

    def test_discrete_empty(self):
        result = action_to_twist(np.array([]), action_type="discrete")
        assert result["linear_x"] == pytest.approx(0.0)
        assert result["angular_z"] == pytest.approx(0.0)

    def test_accepts_python_list(self):
        result = action_to_twist([1.0, -0.5], action_type="continuous")
        assert result["linear_x"] == pytest.approx(1.0)
        assert result["angular_z"] == pytest.approx(-0.5)

    def test_accepts_scalar(self):
        result = action_to_twist(np.float64(3.0), action_type="discrete")
        assert result["linear_x"] == pytest.approx(0.2)
        assert result["angular_z"] == pytest.approx(0.5)

    def test_continuous_negative_values(self):
        result = action_to_twist(np.array([-1.0, -2.0]), action_type="continuous")
        assert result["linear_x"] == pytest.approx(-1.0)
        assert result["angular_z"] == pytest.approx(-2.0)


# ===================================================================
# pose_to_goal
# ===================================================================


class TestPoseToGoal:
    def test_pose_stamped_style(self):
        # PoseStamped: msg.pose.position.x
        msg = SimpleNamespace(
            pose=SimpleNamespace(position=SimpleNamespace(x=3.0, y=4.0))
        )
        result = pose_to_goal(msg)
        assert result == (pytest.approx(3.0), pytest.approx(4.0))

    def test_plain_pose_style(self):
        # Pose: msg.position.x
        msg = SimpleNamespace(position=SimpleNamespace(x=5.0, y=6.0))
        result = pose_to_goal(msg)
        assert result == (pytest.approx(5.0), pytest.approx(6.0))

    def test_bare_position(self):
        # Bare object with x, y directly (no .pose, no .position)
        msg = SimpleNamespace(x=7.0, y=8.0)
        result = pose_to_goal(msg)
        assert result == (pytest.approx(7.0), pytest.approx(8.0))

    def test_negative_coordinates(self):
        msg = SimpleNamespace(position=SimpleNamespace(x=-1.5, y=-2.5))
        result = pose_to_goal(msg)
        assert result == (pytest.approx(-1.5), pytest.approx(-2.5))

    def test_zero_coordinates(self):
        msg = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
        result = pose_to_goal(msg)
        assert result == (0.0, 0.0)


# ===================================================================
# image_to_numpy
# ===================================================================


class TestImageToNumpy:
    def test_rgb8(self):
        height, width = 2, 3
        data = bytes(range(height * width * 3))
        msg = _make_image(height, width, data, encoding="rgb8")
        result = image_to_numpy(msg)
        assert result.shape == (2, 3, 3)
        assert result.dtype == np.uint8

    def test_bgr8_converted_to_rgb(self):
        height, width = 1, 1
        # BGR pixel: B=10, G=20, R=30
        data = bytes([10, 20, 30])
        msg = _make_image(height, width, data, encoding="bgr8")
        result = image_to_numpy(msg)
        assert result.shape == (1, 1, 3)
        # After BGR->RGB reversal: R=30, G=20, B=10
        np.testing.assert_array_equal(result[0, 0], [30, 20, 10])

    def test_bgra8_converted_to_argb(self):
        height, width = 1, 1
        # BGRA pixel: B=10, G=20, R=30, A=255
        data = bytes([10, 20, 30, 255])
        msg = _make_image(height, width, data, encoding="bgra8")
        result = image_to_numpy(msg)
        assert result.shape == (1, 1, 4)
        # After full channel reversal: A=255, R=30, G=20, B=10
        np.testing.assert_array_equal(result[0, 0], [255, 30, 20, 10])

    def test_mono8(self):
        height, width = 2, 2
        data = bytes([0, 64, 128, 255])
        msg = _make_image(height, width, data, encoding="mono8")
        result = image_to_numpy(msg)
        assert result.shape == (2, 2)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [[0, 64], [128, 255]])

    def test_32fc1(self):
        height, width = 1, 2
        arr = np.array([1.5, 2.5], dtype=np.float32)
        data = arr.tobytes()
        msg = _make_image(height, width, data, encoding="32FC1")
        result = image_to_numpy(msg)
        assert result.shape == (1, 2)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [[1.5, 2.5]])

    def test_rgba8(self):
        height, width = 1, 1
        data = bytes([100, 150, 200, 255])
        msg = _make_image(height, width, data, encoding="rgba8")
        result = image_to_numpy(msg)
        assert result.shape == (1, 1, 4)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[0, 0], [100, 150, 200, 255])

    def test_8uc1(self):
        height, width = 1, 3
        data = bytes([10, 20, 30])
        msg = _make_image(height, width, data, encoding="8UC1")
        result = image_to_numpy(msg)
        assert result.shape == (1, 3)
        assert result.dtype == np.uint8

    def test_8uc3(self):
        height, width = 1, 1
        data = bytes([10, 20, 30])
        msg = _make_image(height, width, data, encoding="8UC3")
        result = image_to_numpy(msg)
        assert result.shape == (1, 1, 3)
        assert result.dtype == np.uint8

    def test_16uc1(self):
        height, width = 1, 2
        arr = np.array([1000, 2000], dtype=np.uint16)
        data = arr.tobytes()
        msg = _make_image(height, width, data, encoding="16UC1")
        result = image_to_numpy(msg)
        assert result.shape == (1, 2)
        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, [[1000, 2000]])

    def test_unknown_encoding_defaults_to_uint8_3ch(self):
        height, width = 1, 1
        data = bytes([10, 20, 30])
        msg = _make_image(height, width, data, encoding="unknown_enc")
        result = image_to_numpy(msg)
        assert result.shape == (1, 1, 3)
        assert result.dtype == np.uint8

    def test_bgr8_larger_image(self):
        height, width = 2, 2
        # 4 BGR pixels
        data = bytes(
            [
                10, 20, 30,  # (0,0)
                40, 50, 60,  # (0,1)
                70, 80, 90,  # (1,0)
                100, 110, 120,  # (1,1)
            ]
        )
        msg = _make_image(height, width, data, encoding="bgr8")
        result = image_to_numpy(msg)
        assert result.shape == (2, 2, 3)
        # First pixel BGR(10,20,30) -> RGB(30,20,10)
        np.testing.assert_array_equal(result[0, 0], [30, 20, 10])
        np.testing.assert_array_equal(result[1, 1], [120, 110, 100])

    def test_data_as_list(self):
        # data passed as a list of ints (not bytes) -- converted via bytes()
        height, width = 1, 2
        data = [10, 20, 30, 40, 50, 60]
        msg = _make_image(height, width, data, encoding="rgb8")
        result = image_to_numpy(msg)
        assert result.shape == (1, 2, 3)

    def test_default_encoding(self):
        # When encoding attribute missing, defaults to rgb8
        height, width = 1, 1
        data = bytes([1, 2, 3])
        msg = SimpleNamespace(height=height, width=width, data=data)
        result = image_to_numpy(msg)
        assert result.shape == (1, 1, 3)
