"""Coverage tests for navirl/ros/__init__.py.

The package defers rclpy imports until needed. These tests exercise both the
``ros2_available()`` availability probe and the lazy ``__getattr__`` loader
without requiring ROS2 to be installed on the test host.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def fresh_ros_pkg(monkeypatch):
    """Reload navirl.ros in isolation so the module-level cache state is reset."""
    for name in list(sys.modules):
        if name == "navirl.ros" or name.startswith("navirl.ros."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    pkg = importlib.import_module("navirl.ros")
    yield pkg
    for name in list(sys.modules):
        if name == "navirl.ros" or name.startswith("navirl.ros."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_ros2_available_returns_bool(fresh_ros_pkg):
    """``ros2_available`` returns a bool regardless of environment."""
    result = fresh_ros_pkg.ros2_available()
    assert isinstance(result, bool)


def test_ros2_available_caches_result(fresh_ros_pkg):
    """A second call reuses the cached value without re-importing."""
    first = fresh_ros_pkg.ros2_available()
    # Prevent a second importlib.import_module("rclpy") from succeeding.
    import navirl.ros as pkg_module

    original_import = importlib.import_module

    def _fail(name, *args, **kwargs):
        if name == "rclpy":
            raise AssertionError("rclpy should not be imported on cached call")
        return original_import(name, *args, **kwargs)

    pkg_module.importlib.import_module = _fail
    try:
        assert fresh_ros_pkg.ros2_available() is first
    finally:
        pkg_module.importlib.import_module = original_import


def test_ros2_available_false_when_rclpy_missing(monkeypatch, fresh_ros_pkg):
    """If rclpy import fails, the probe returns False."""
    import navirl.ros as pkg_module

    # Reset the cache to force recomputation.
    pkg_module._ROS2_AVAILABLE = None

    original_import = importlib.import_module

    def _fake(name, *args, **kwargs):
        if name == "rclpy":
            raise ImportError("synthetic: rclpy not available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(pkg_module.importlib, "import_module", _fake)
    assert pkg_module.ros2_available() is False


def test_lazy_getattr_loads_tf_utils_symbol(fresh_ros_pkg):
    """Accessing a lazily-declared symbol imports its submodule on demand."""
    # ``world_to_robot`` maps to the tf_utils submodule.
    attr = fresh_ros_pkg.world_to_robot
    assert callable(attr)

    from navirl.ros.tf_utils import world_to_robot as direct

    assert attr is direct


def test_lazy_getattr_caches_on_package(fresh_ros_pkg):
    """Once resolved, the attribute is cached on the package's globals."""
    first = fresh_ros_pkg.world_to_robot  # first access populates the cache
    # On a second access the import machinery is not consulted.
    cached = fresh_ros_pkg.__dict__["world_to_robot"]
    assert cached is first


def test_lazy_getattr_unknown_raises_attribute_error(fresh_ros_pkg):
    with pytest.raises(AttributeError, match="no attribute 'not_a_real_symbol'"):
        _ = fresh_ros_pkg.not_a_real_symbol


def test_all_exports_include_lazy_entries(fresh_ros_pkg):
    exported = set(fresh_ros_pkg.__all__)
    assert "ros2_available" in exported
    # Representative entries from each submodule group.
    for name in ("NavIRLNode", "TransformManager", "laser_scan_to_lidar_obs"):
        assert name in exported
