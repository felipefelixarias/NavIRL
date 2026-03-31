"""Message conversion utilities between ROS2 messages and NavIRL formats.

Every function in this module accepts a ROS2 message (or a plain dict that
mirrors its fields) and returns a NumPy array or Python dict consumable by
NavIRL agents.  ROS2 message types are imported lazily so this module can
be loaded even when ROS2 is not installed -- callers that only use the
numpy-level helpers will not need rclpy at all.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Guarded ROS2 imports (only needed when real ROS msgs are passed)
# ---------------------------------------------------------------------------
try:
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, LaserScan

    _ROS2_MSG_AVAILABLE = True
except ImportError:
    _ROS2_MSG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion to a yaw angle (radians)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Public conversion functions
# ---------------------------------------------------------------------------


def laser_scan_to_lidar_obs(msg: Any) -> np.ndarray:
    """Convert a ``sensor_msgs/LaserScan`` to a 1-D numpy float32 array.

    Invalid (inf / NaN) ranges are clipped to ``range_max``.

    Parameters
    ----------
    msg : LaserScan or dict-like
        A ROS2 LaserScan message (or any object with ``ranges``,
        ``range_min``, and ``range_max`` attributes).

    Returns
    -------
    np.ndarray
        Shape ``(N,)`` where *N* is the number of scan beams.
    """
    ranges = np.array(msg.ranges, dtype=np.float32)
    range_max = float(getattr(msg, "range_max", 30.0))
    range_min = float(getattr(msg, "range_min", 0.0))

    # Replace inf / NaN with range_max
    invalid = ~np.isfinite(ranges)
    ranges[invalid] = range_max

    # Clip to valid range
    np.clip(ranges, range_min, range_max, out=ranges)

    return ranges


def odometry_to_state(msg: Any) -> dict[str, Any]:
    """Convert a ``nav_msgs/Odometry`` to a state dictionary.

    Returns
    -------
    dict
        Keys: ``x``, ``y``, ``yaw`` (radians), ``vx``, ``vy``, ``omega``.
    """
    pose = msg.pose.pose
    twist = msg.twist.twist

    x = float(pose.position.x)
    y = float(pose.position.y)
    yaw = _quat_to_yaw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    vx = float(twist.linear.x)
    vy = float(twist.linear.y)
    omega = float(twist.angular.z)

    return {
        "x": x,
        "y": y,
        "yaw": yaw,
        "vx": vx,
        "vy": vy,
        "omega": omega,
    }


def person_array_to_social_obs(msg: Any) -> np.ndarray:
    """Convert a tracked-persons message to a ``(N, 7)`` numpy array.

    Each row: ``[x, y, vx, vy, theta, id, detection_score]``.

    Accepts any message with a ``tracks`` iterable whose elements expose
    ``pose.pose.position``, ``twist.twist.linear``, and optionally
    ``track_id`` / ``detection_id`` and ``is_matched``.
    """
    tracks = getattr(msg, "tracks", getattr(msg, "persons", []))
    if not tracks:
        return np.zeros((0, 7), dtype=np.float64)

    rows = []
    for t in tracks:
        # Position
        pos = t.pose.pose.position
        px, py = float(pos.x), float(pos.y)

        # Velocity
        vel = t.twist.twist.linear
        vx, vy = float(vel.x), float(vel.y)

        # Orientation
        ori = t.pose.pose.orientation
        theta = _quat_to_yaw(float(ori.x), float(ori.y), float(ori.z), float(ori.w))

        track_id = float(getattr(t, "track_id", getattr(t, "detection_id", 0)))
        score = float(getattr(t, "detection_score", getattr(t, "is_matched", 1.0)))

        rows.append([px, py, vx, vy, theta, track_id, score])

    return np.array(rows, dtype=np.float64)


def action_to_twist(
    action: np.ndarray,
    action_type: str = "continuous",
) -> dict[str, float]:
    """Convert a NavIRL action array to a Twist-like dictionary.

    Parameters
    ----------
    action : np.ndarray
        For *continuous* actions: ``[linear_x, angular_z]``.
        For *discrete* actions: scalar index mapped to preset velocities.
    action_type : str
        ``"continuous"`` or ``"discrete"``.

    Returns
    -------
    dict
        ``{"linear_x": ..., "linear_y": 0.0, "linear_z": 0.0,
          "angular_x": 0.0, "angular_y": 0.0, "angular_z": ...}``
    """
    action = np.asarray(action, dtype=np.float64).ravel()

    if action_type == "discrete":
        # Map discrete indices to (linear_x, angular_z) pairs
        _DISCRETE_MAP = {
            0: (0.0, 0.0),  # stop
            1: (0.5, 0.0),  # forward
            2: (-0.3, 0.0),  # backward
            3: (0.2, 0.5),  # turn left
            4: (0.2, -0.5),  # turn right
            5: (0.5, 0.3),  # forward-left
            6: (0.5, -0.3),  # forward-right
        }
        idx = int(action[0]) if action.size > 0 else 0
        linear_x, angular_z = _DISCRETE_MAP.get(idx, (0.0, 0.0))
    else:
        linear_x = float(action[0]) if action.size > 0 else 0.0
        angular_z = float(action[1]) if action.size > 1 else 0.0

    return {
        "linear_x": linear_x,
        "linear_y": 0.0,
        "linear_z": 0.0,
        "angular_x": 0.0,
        "angular_y": 0.0,
        "angular_z": angular_z,
    }


def pose_to_goal(msg: Any) -> tuple[float, float]:
    """Extract an ``(x, y)`` goal from a Pose/PoseStamped message.

    Also accepts plain objects with ``position.x`` / ``position.y`` or
    a ``pose.position`` hierarchy.
    """
    # PoseStamped wraps a Pose inside .pose
    pose = getattr(msg, "pose", msg)
    position = getattr(pose, "position", pose)
    return (float(position.x), float(position.y))


def image_to_numpy(msg: Any) -> np.ndarray:
    """Convert a ``sensor_msgs/Image`` message to a numpy array.

    Supports common encodings: ``rgb8``, ``bgr8``, ``mono8``, ``32FC1``.
    Falls back to a best-effort reshape for unknown encodings.

    Parameters
    ----------
    msg : Image or dict-like
        A ROS2 Image message.

    Returns
    -------
    np.ndarray
        ``(H, W, C)`` for colour images, ``(H, W)`` for mono/depth.
    """
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(getattr(msg, "encoding", "rgb8")).lower()
    data = bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray)) else msg.data

    _ENCODING_MAP = {
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "8uc3": (np.uint8, 3),
        "16uc1": (np.uint16, 1),
        "32fc1": (np.float32, 1),
    }

    dtype, channels = _ENCODING_MAP.get(encoding, (np.uint8, 3))
    arr = np.frombuffer(data, dtype=dtype)

    if channels == 1:
        arr = arr.reshape((height, width))
    else:
        arr = arr.reshape((height, width, channels))

    # Convert BGR -> RGB for convenience
    if encoding in ("bgr8", "bgra8") and arr.ndim == 3:
        arr = arr[..., ::-1].copy()

    return arr
