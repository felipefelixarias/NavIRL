from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.ros.tf_utils import (
    TransformManager,
    _invert_2d,
    _yaw_to_rotation_matrix,
    robot_to_world,
    world_to_robot,
)

# ---- _yaw_to_rotation_matrix ------------------------------------------------


class TestYawToRotationMatrix:
    """Tests for the 2x2 rotation matrix helper."""

    def test_identity_at_zero(self):
        R = _yaw_to_rotation_matrix(0.0)
        np.testing.assert_allclose(R, np.eye(2), atol=1e-15)

    def test_shape(self):
        R = _yaw_to_rotation_matrix(1.0)
        assert R.shape == (2, 2)

    def test_determinant_is_one(self):
        for yaw in [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3, 2.5]:
            R = _yaw_to_rotation_matrix(yaw)
            assert pytest.approx(np.linalg.det(R), abs=1e-14) == 1.0

    def test_orthogonality(self):
        for yaw in [0.3, math.pi / 6, -1.2, math.pi]:
            R = _yaw_to_rotation_matrix(yaw)
            np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-14)
            np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-14)

    def test_90_degrees(self):
        R = _yaw_to_rotation_matrix(math.pi / 2)
        expected = np.array([[0, -1], [1, 0]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_180_degrees(self):
        R = _yaw_to_rotation_matrix(math.pi)
        expected = np.array([[-1, 0], [0, -1]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_negative_90_degrees(self):
        R = _yaw_to_rotation_matrix(-math.pi / 2)
        expected = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_inverse_equals_transpose(self):
        yaw = 0.77
        R = _yaw_to_rotation_matrix(yaw)
        np.testing.assert_allclose(np.linalg.inv(R), R.T, atol=1e-14)

    def test_composition(self):
        """R(a) @ R(b) == R(a + b)."""
        a, b = 0.5, 1.1
        np.testing.assert_allclose(
            _yaw_to_rotation_matrix(a) @ _yaw_to_rotation_matrix(b),
            _yaw_to_rotation_matrix(a + b),
            atol=1e-14,
        )


# ---- world_to_robot / robot_to_world ----------------------------------------


class TestWorldToRobot:
    """Tests for world_to_robot conversion."""

    def test_identity_pose_dict(self):
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        result = world_to_robot((3.0, 4.0), pose)
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-14)

    def test_identity_pose_tuple(self):
        pose = (0.0, 0.0, 0.0)
        result = world_to_robot(np.array([3.0, 4.0]), pose)
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-14)

    def test_pure_translation(self):
        pose = {"x": 1.0, "y": 2.0, "yaw": 0.0}
        result = world_to_robot((4.0, 5.0), pose)
        np.testing.assert_allclose(result, [3.0, 3.0], atol=1e-14)

    def test_pure_rotation_90(self):
        pose = (0.0, 0.0, math.pi / 2)
        result = world_to_robot((1.0, 0.0), pose)
        # Rotating by -pi/2: (1,0) -> (0, -1)
        np.testing.assert_allclose(result, [0.0, -1.0], atol=1e-14)

    def test_translation_and_rotation(self):
        pose = {"x": 1.0, "y": 0.0, "yaw": math.pi / 2}
        # delta = (2-1, 3-0) = (1, 3), rotate by -pi/2 -> (3, -1)
        result = world_to_robot((2.0, 3.0), pose)
        np.testing.assert_allclose(result, [3.0, -1.0], atol=1e-14)

    def test_yaw_180(self):
        pose = (0.0, 0.0, math.pi)
        result = world_to_robot((1.0, 0.0), pose)
        np.testing.assert_allclose(result, [-1.0, 0.0], atol=1e-14)

    def test_accepts_numpy_pos(self):
        pose = (0.0, 0.0, 0.0)
        result = world_to_robot(np.array([5.0, 6.0]), pose)
        np.testing.assert_allclose(result, [5.0, 6.0], atol=1e-14)


class TestRobotToWorld:
    """Tests for robot_to_world conversion."""

    def test_identity_pose_dict(self):
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        result = robot_to_world((3.0, 4.0), pose)
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-14)

    def test_identity_pose_tuple(self):
        pose = (0.0, 0.0, 0.0)
        result = robot_to_world(np.array([1.0, 2.0]), pose)
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-14)

    def test_pure_translation(self):
        pose = {"x": 1.0, "y": 2.0, "yaw": 0.0}
        result = robot_to_world((3.0, 3.0), pose)
        np.testing.assert_allclose(result, [4.0, 5.0], atol=1e-14)

    def test_pure_rotation_90(self):
        pose = (0.0, 0.0, math.pi / 2)
        result = robot_to_world((1.0, 0.0), pose)
        # Rotate (1,0) by pi/2 -> (0, 1)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-14)

    def test_yaw_negative_90(self):
        pose = (0.0, 0.0, -math.pi / 2)
        result = robot_to_world((1.0, 0.0), pose)
        np.testing.assert_allclose(result, [0.0, -1.0], atol=1e-14)


class TestRoundTrip:
    """world_to_robot and robot_to_world should be inverses."""

    @pytest.mark.parametrize(
        "world_pt,pose",
        [
            ((5.0, 3.0), {"x": 1.0, "y": 2.0, "yaw": 0.5}),
            ((0.0, 0.0), (3.0, 4.0, math.pi / 3)),
            ((-2.0, 7.0), {"x": -1.0, "y": -1.0, "yaw": -math.pi / 4}),
            ((10.0, -3.0), (0.0, 0.0, math.pi)),
        ],
    )
    def test_world_robot_world(self, world_pt, pose):
        robot_pt = world_to_robot(world_pt, pose)
        recovered = robot_to_world(robot_pt, pose)
        np.testing.assert_allclose(recovered, world_pt, atol=1e-12)

    @pytest.mark.parametrize(
        "robot_pt,pose",
        [
            ((1.0, 0.0), (2.0, 3.0, 0.7)),
            ((0.0, 0.0), {"x": 0.0, "y": 0.0, "yaw": 1.0}),
            ((-1.0, 2.0), (5.0, -5.0, -1.5)),
        ],
    )
    def test_robot_world_robot(self, robot_pt, pose):
        world_pt = robot_to_world(robot_pt, pose)
        recovered = world_to_robot(world_pt, pose)
        np.testing.assert_allclose(recovered, robot_pt, atol=1e-12)


# ---- _invert_2d --------------------------------------------------------------


class TestInvert2D:
    """Tests for the internal rigid-transform inversion helper."""

    def test_identity(self):
        ix, iy = _invert_2d(0.0, 0.0, 0.0)
        assert pytest.approx(ix, abs=1e-14) == 0.0
        assert pytest.approx(iy, abs=1e-14) == 0.0

    def test_pure_translation(self):
        ix, iy = _invert_2d(3.0, 4.0, 0.0)
        assert pytest.approx(ix, abs=1e-14) == -3.0
        assert pytest.approx(iy, abs=1e-14) == -4.0

    def test_pure_rotation(self):
        ix, iy = _invert_2d(0.0, 0.0, math.pi / 2)
        assert pytest.approx(ix, abs=1e-14) == 0.0
        assert pytest.approx(iy, abs=1e-14) == 0.0

    def test_double_invert_returns_to_original(self):
        x, y, yaw = 2.0, -1.0, 0.8
        ix, iy = _invert_2d(x, y, yaw)
        rx, ry = _invert_2d(ix, iy, -yaw)
        assert pytest.approx(rx, abs=1e-12) == x
        assert pytest.approx(ry, abs=1e-12) == y


# ---- TransformManager -------------------------------------------------------


class TestTransformManager:
    """Tests for TransformManager in manual-cache mode (no ROS2)."""

    def test_creation(self):
        tm = TransformManager()
        assert tm is not None

    def test_set_and_lookup(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 1.0, 2.0, 0.5)
        x, y, yaw = tm.lookup("map", "base_link")
        assert pytest.approx(x) == 1.0
        assert pytest.approx(y) == 2.0
        assert pytest.approx(yaw) == 0.5

    def test_lookup_missing_raises(self):
        tm = TransformManager()
        with pytest.raises(Exception, match="not available"):
            tm.lookup("map", "nonexistent")

    def test_inverse_lookup(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 3.0, 0.0, 0.0)
        # Inverse: base_link -> map should give (-3, 0, 0)
        x, y, yaw = tm.lookup("base_link", "map")
        assert pytest.approx(x, abs=1e-12) == -3.0
        assert pytest.approx(y, abs=1e-12) == 0.0
        assert pytest.approx(yaw, abs=1e-12) == 0.0

    def test_inverse_lookup_with_rotation(self):
        tm = TransformManager()
        tm.set_transform("map", "odom", 1.0, 0.0, math.pi / 2)
        x, y, yaw = tm.lookup("odom", "map")
        # invert_2d(1, 0, pi/2): inv_x = -(cos(pi/2)*1 + sin(pi/2)*0) = 0
        #                         inv_y = -(- sin(pi/2)*1 + cos(pi/2)*0) = 1
        # yaw = -pi/2
        assert pytest.approx(x, abs=1e-12) == 0.0
        assert pytest.approx(y, abs=1e-12) == 1.0
        assert pytest.approx(yaw, abs=1e-12) == -math.pi / 2

    def test_overwrite_transform(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 1.0, 2.0, 0.0)
        tm.set_transform("map", "base_link", 5.0, 6.0, 1.0)
        x, y, yaw = tm.lookup("map", "base_link")
        assert pytest.approx(x) == 5.0
        assert pytest.approx(y) == 6.0
        assert pytest.approx(yaw) == 1.0

    def test_transform_point_identity(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 0.0, 0.0, 0.0)
        result = tm.transform_point((3.0, 4.0), "base_link", "map")
        np.testing.assert_allclose(result, [3.0, 4.0], atol=1e-12)

    def test_transform_point_translation(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 1.0, 2.0, 0.0)
        result = tm.transform_point((0.0, 0.0), "base_link", "map")
        np.testing.assert_allclose(result, [1.0, 2.0], atol=1e-12)

    def test_transform_point_rotation(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 0.0, 0.0, math.pi / 2)
        result = tm.transform_point((1.0, 0.0), "base_link", "map")
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)

    def test_transform_point_combined(self):
        tm = TransformManager()
        tm.set_transform("map", "base_link", 1.0, 2.0, math.pi / 2)
        result = tm.transform_point((1.0, 0.0), "base_link", "map")
        # R(pi/2) @ [1,0] + [1,2] = [0,1] + [1,2] = [1,3]
        np.testing.assert_allclose(result, [1.0, 3.0], atol=1e-12)

    def test_transform_point_accepts_numpy(self):
        tm = TransformManager()
        tm.set_transform("a", "b", 0.0, 0.0, 0.0)
        result = tm.transform_point(np.array([2.0, 3.0]), "b", "a")
        np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-12)

    def test_spin_once_no_crash(self):
        tm = TransformManager()
        tm.spin_once()
        tm.spin_once(timeout_sec=0.1)

    def test_custom_cache_duration(self):
        tm = TransformManager(cache_duration=5.0)
        assert tm._cache_duration == 5.0

    def test_multiple_frames(self):
        tm = TransformManager()
        tm.set_transform("map", "odom", 1.0, 0.0, 0.0)
        tm.set_transform("odom", "base_link", 0.0, 1.0, 0.0)
        x1, y1, _ = tm.lookup("map", "odom")
        x2, y2, _ = tm.lookup("odom", "base_link")
        assert pytest.approx(x1) == 1.0
        assert pytest.approx(x2) == 0.0
        assert pytest.approx(y2) == 1.0

    def test_stale_transform_still_returned_with_warning(self):
        """Even stale transforms are returned (with a warning)."""
        tm = TransformManager(cache_duration=0.0)
        tm.set_transform("map", "base_link", 1.0, 2.0, 0.3)
        # cache_duration=0 means it's immediately stale, but still returned
        x, y, yaw = tm.lookup("map", "base_link")
        assert pytest.approx(x) == 1.0
        assert pytest.approx(y) == 2.0
        assert pytest.approx(yaw) == 0.3
