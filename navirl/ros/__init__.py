"""NavIRL ROS2 integration package.

Provides bridges between NavIRL agents and ROS2 robotics middleware.
All ROS2 dependencies are lazily imported so that the rest of NavIRL
remains usable even when ROS2 is not installed.
"""

from __future__ import annotations

import importlib

_ROS2_AVAILABLE: bool | None = None


def ros2_available() -> bool:
    """Return *True* if a usable ROS2 Python environment is detected."""
    global _ROS2_AVAILABLE
    if _ROS2_AVAILABLE is None:
        try:
            importlib.import_module("rclpy")
            _ROS2_AVAILABLE = True
        except ImportError:
            _ROS2_AVAILABLE = False
    return _ROS2_AVAILABLE


def _require_ros2(name: str) -> None:
    if not ros2_available():
        raise ImportError(
            f"Cannot use navirl.ros.{name} because ROS2 (rclpy) is not installed. "
            "Please install ROS2 Humble or later and source the setup script before "
            "importing this module.  See https://docs.ros.org/en/humble/Installation.html"
        )


# Lazy attribute access so ``from navirl.ros import NavIRLNode`` works
# without eagerly importing rclpy at package-init time.
_LAZY_SUBMODULES = {
    "NavIRLNode": "node",
    "GazeboBridge": "bridges",
    "IsaacBridge": "bridges",
    "HabitatBridge": "bridges",
    "CostmapManager": "costmap",
    "SocialCostmapLayer": "costmap",
    "PredictiveCostmapLayer": "costmap",
    "TransformManager": "tf_utils",
    "world_to_robot": "tf_utils",
    "robot_to_world": "tf_utils",
    "laser_scan_to_lidar_obs": "conversions",
    "odometry_to_state": "conversions",
    "person_array_to_social_obs": "conversions",
    "action_to_twist": "conversions",
    "pose_to_goal": "conversions",
    "image_to_numpy": "conversions",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        submodule_name = _LAZY_SUBMODULES[name]
        module = importlib.import_module(f".{submodule_name}", __name__)
        attr = getattr(module, name)
        # Cache on the package so future lookups are fast.
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ros2_available",
    *_LAZY_SUBMODULES.keys(),
]
