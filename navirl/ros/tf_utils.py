"""TF2 transform utilities for NavIRL.

Provides a lightweight :class:`TransformManager` that caches frame
transforms and convenience functions for converting positions between
the world and robot frames.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from navirl.utils.geometry import quat_to_yaw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded ROS2 / TF2 imports
# ---------------------------------------------------------------------------
try:
    import rclpy
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time as RosTime
    from tf2_ros import TransformException

    _TF2_AVAILABLE = True
except ImportError:
    _TF2_AVAILABLE = False
    TransformException = Exception  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Pure-numpy helpers (no ROS2 dependency)
# ---------------------------------------------------------------------------


def _yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for the given yaw angle."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def world_to_robot(
    pos: np.ndarray | tuple[float, float],
    robot_pose: dict[str, float] | tuple[float, float, float],
) -> np.ndarray:
    """Transform a world-frame position into the robot's local frame.

    Parameters
    ----------
    pos : array-like, shape ``(2,)``
        ``(x, y)`` in the world frame.
    robot_pose : dict or tuple
        Either ``{"x": ..., "y": ..., "yaw": ...}`` or ``(x, y, yaw)``.

    Returns
    -------
    np.ndarray
        ``(x, y)`` relative to the robot (forward = +x, left = +y).
    """
    pos = np.asarray(pos, dtype=np.float64).ravel()[:2]

    if isinstance(robot_pose, dict):
        rx, ry, ryaw = robot_pose["x"], robot_pose["y"], robot_pose["yaw"]
    else:
        rx, ry, ryaw = robot_pose[0], robot_pose[1], robot_pose[2]

    # Translate then rotate by -yaw
    delta = pos - np.array([rx, ry], dtype=np.float64)
    rot = _yaw_to_rotation_matrix(-ryaw)
    return rot @ delta


def robot_to_world(
    pos: np.ndarray | tuple[float, float],
    robot_pose: dict[str, float] | tuple[float, float, float],
) -> np.ndarray:
    """Transform a robot-local position into the world frame.

    Parameters
    ----------
    pos : array-like, shape ``(2,)``
        ``(x, y)`` in the robot frame.
    robot_pose : dict or tuple
        Either ``{"x": ..., "y": ..., "yaw": ...}`` or ``(x, y, yaw)``.

    Returns
    -------
    np.ndarray
        ``(x, y)`` in the world frame.
    """
    pos = np.asarray(pos, dtype=np.float64).ravel()[:2]

    if isinstance(robot_pose, dict):
        rx, ry, ryaw = robot_pose["x"], robot_pose["y"], robot_pose["yaw"]
    else:
        rx, ry, ryaw = robot_pose[0], robot_pose[1], robot_pose[2]

    rot = _yaw_to_rotation_matrix(ryaw)
    world = rot @ pos + np.array([rx, ry], dtype=np.float64)
    return world


# ---------------------------------------------------------------------------
# TransformManager
# ---------------------------------------------------------------------------


class TransformManager:
    """Cache and look up TF2 transforms between coordinate frames.

    When ROS2 / tf2 is available, the manager wraps a
    ``tf2_ros.Buffer`` + ``TransformListener``.  Otherwise it falls back
    to a simple in-memory cache that must be populated manually via
    :meth:`set_transform`.

    Parameters
    ----------
    node : rclpy.node.Node or None
        An existing ROS2 node to attach the TF listener to.  If *None*,
        a standalone node is created (requires ``rclpy.init``).
    cache_duration : float
        How long (seconds) to keep transforms in the buffer.
    """

    def __init__(
        self,
        node: Any = None,
        cache_duration: float = 10.0,
    ) -> None:
        self._cache_duration = cache_duration
        self._node: Any = node
        self._tf_buffer: Any = None
        self._tf_listener: Any = None

        # Fallback manual cache: (parent, child) -> (x, y, yaw, timestamp)
        self._manual_cache: dict[tuple[str, str], tuple[float, float, float, float]] = {}

        if _TF2_AVAILABLE:
            self._init_tf2()
        else:
            logger.info("TransformManager: tf2_ros not available -- using manual cache mode.")

    def _init_tf2(self) -> None:
        if self._node is None:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("navirl_tf_manager")

        self._tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=self._cache_duration)
        )
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        logger.info("TransformManager: TF2 listener initialised.")

    # -- Public API ---------------------------------------------------------

    def set_transform(
        self,
        parent_frame: str,
        child_frame: str,
        x: float,
        y: float,
        yaw: float,
    ) -> None:
        """Manually cache a 2-D transform (useful when TF2 is unavailable)."""
        self._manual_cache[(parent_frame, child_frame)] = (x, y, yaw, time.time())

    def lookup(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 0.5,
    ) -> tuple[float, float, float]:
        """Look up the ``(x, y, yaw)`` of *source_frame* in *target_frame*.

        Raises ``TransformException`` if the transform is unavailable.
        """
        # Try TF2 first
        if self._tf_buffer is not None:
            try:
                ts: TransformStamped = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    RosTime(),
                    timeout=rclpy.duration.Duration(seconds=timeout_sec),
                )
                t = ts.transform.translation
                r = ts.transform.rotation
                yaw = quat_to_yaw(r.x, r.y, r.z, r.w)
                return (float(t.x), float(t.y), yaw)
            except TransformException:
                pass  # fall through to manual cache

        # Fallback: manual cache
        key = (target_frame, source_frame)
        if key in self._manual_cache:
            x, y, yaw, ts_time = self._manual_cache[key]
            age = time.time() - ts_time
            if age <= self._cache_duration:
                return (x, y, yaw)
            else:
                logger.warning(
                    "TransformManager: cached transform %s -> %s is stale (%.1fs old).",
                    target_frame,
                    source_frame,
                    age,
                )
                return (x, y, yaw)

        # Try the inverse
        inv_key = (source_frame, target_frame)
        if inv_key in self._manual_cache:
            x, y, yaw, _ = self._manual_cache[inv_key]
            # Invert the 2-D transform
            inv_x, inv_y = _invert_2d(x, y, yaw)
            return (inv_x, inv_y, -yaw)

        raise TransformException(
            f"Transform from '{source_frame}' to '{target_frame}' not available."
        )

    def transform_point(
        self,
        point: np.ndarray | tuple[float, float],
        source_frame: str,
        target_frame: str,
    ) -> np.ndarray:
        """Transform a 2-D point from *source_frame* to *target_frame*."""
        x, y, yaw = self.lookup(target_frame, source_frame)
        # The lookup gives source in target -> we apply the inverse
        # to transform a point expressed in source into target.
        p = np.asarray(point, dtype=np.float64).ravel()[:2]
        rot = _yaw_to_rotation_matrix(yaw)
        return rot @ p + np.array([x, y], dtype=np.float64)

    def spin_once(self, timeout_sec: float = 0.01) -> None:
        """Spin the underlying ROS2 node to receive new TF data."""
        if self._node is not None and _TF2_AVAILABLE:
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _invert_2d(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Invert a 2-D rigid transform ``(x, y, yaw)``."""
    c, s = math.cos(yaw), math.sin(yaw)
    inv_x = -(c * x + s * y)
    inv_y = -(-s * x + c * y)
    return inv_x, inv_y
