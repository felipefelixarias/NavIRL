"""Tests for high-level robot controller classes.

Covers the ``__init__``, ``reset``, ``step``, and accessor methods of:
  - HolonomicRobot
  - DifferentialDriveRobot
  - AckermannRobot

These classes were previously untested despite being the primary
RobotController implementations used by the simulation pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.core.types import Action, AgentState
from navirl.robots.ackermann import AckermannConfig, AckermannRobot
from navirl.robots.differential_drive import (
    DifferentialDriveConfig,
    DifferentialDriveRobot,
    SensorMountPoint,
)
from navirl.robots.holonomic import HolonomicConfig, HolonomicRobot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    agent_id: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 5.0,
    goal_y: float = 0.0,
    max_speed: float = 2.0,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind="robot",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=0.15,
        max_speed=max_speed,
        metadata={},
    )


class _EventCollector:
    """Simple event sink for testing."""

    def __init__(self):
        self.events: list[tuple] = []

    def __call__(self, event_type: str, agent_id: int, data: dict) -> None:
        self.events.append((event_type, agent_id, data))


# ===================================================================
# HolonomicRobot
# ===================================================================


class TestHolonomicRobot:
    def test_init_defaults(self):
        robot = HolonomicRobot()
        assert robot.config is not None
        assert robot.speed == 0.0
        np.testing.assert_array_equal(robot.position, [0.0, 0.0])
        np.testing.assert_array_equal(robot.velocity, [0.0, 0.0])

    def test_init_with_waypoints(self):
        wps = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        robot = HolonomicRobot(waypoints=wps, desired_speed=1.5)
        assert robot._follower is not None

    def test_reset(self):
        robot = HolonomicRobot()
        robot.reset(robot_id=7, start=(1.0, 2.0), goal=(5.0, 5.0), backend=None)
        assert robot._robot_id == 7
        np.testing.assert_allclose(robot.position, [1.0, 2.0])
        assert robot._goal == (5.0, 5.0)
        assert robot.speed == 0.0

    def test_step_moves_toward_goal(self):
        robot = HolonomicRobot(desired_speed=1.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0, goal_x=5.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert isinstance(action, Action)
        assert action.pref_vx > 0.0  # Should move toward goal (positive x)
        assert action.behavior == "GO_TO"
        assert len(sink.events) == 1
        assert sink.events[0][0] == "holonomic_step"

    def test_step_returns_done_at_goal(self):
        robot = HolonomicRobot()
        robot.reset(robot_id=0, start=(5.0, 0.0), goal=(5.0, 0.0), backend=None)

        states = {0: _make_state(x=5.0, y=0.0, goal_x=5.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert action.behavior == "DONE"
        assert action.pref_vx == 0.0
        assert action.pref_vy == 0.0

    def test_step_with_waypoints(self):
        wps = np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        robot = HolonomicRobot(waypoints=wps, desired_speed=1.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert action.pref_vx > 0.0
        assert action.behavior == "GO_TO"

    def test_multi_step_progress(self):
        robot = HolonomicRobot(desired_speed=1.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        sink = _EventCollector()

        x = 0.0
        for i in range(20):
            states = {0: _make_state(x=x, y=0.0)}
            action = robot.step(step=i, time_s=i * 0.1, dt=0.1, states=states, emit_event=sink)
            x += action.pref_vx * 0.1

        # Should have made progress toward goal
        assert x > 0.5

    def test_set_waypoints(self):
        robot = HolonomicRobot()
        assert robot._follower is None

        wps = np.array([[0.0, 0.0], [1.0, 1.0]])
        robot.set_waypoints(wps, desired_speed=2.0)
        assert robot._follower is not None

    def test_set_waypoints_default_speed(self):
        robot = HolonomicRobot(desired_speed=3.0)
        wps = np.array([[0.0, 0.0], [1.0, 1.0]])
        robot.set_waypoints(wps)
        assert robot._follower is not None

    def test_accessors_after_step(self):
        robot = HolonomicRobot(desired_speed=1.0)
        robot.reset(robot_id=0, start=(1.0, 2.0), goal=(5.0, 2.0), backend=None)

        states = {0: _make_state(x=1.0, y=2.0)}
        sink = _EventCollector()
        robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        pos = robot.position
        vel = robot.velocity
        spd = robot.speed

        assert pos.shape == (2,)
        assert vel.shape == (2,)
        assert spd >= 0.0
        assert np.isclose(spd, float(np.hypot(vel[0], vel[1])))


# ===================================================================
# DifferentialDriveRobot
# ===================================================================


class TestDifferentialDriveRobot:
    def test_init_defaults(self):
        robot = DifferentialDriveRobot()
        assert robot.config is not None
        assert robot._v == 0.0
        assert robot._omega == 0.0

    def test_init_with_path(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        robot = DifferentialDriveRobot(path=path)
        assert robot._path is not None

    def test_init_with_sensor_mounts(self):
        mounts = [SensorMountPoint(offset_x=0.1, offset_y=0.0, offset_theta=0.0)]
        robot = DifferentialDriveRobot(sensor_mounts=mounts)
        assert len(robot.sensor_mounts) == 1

    def test_reset(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=3, start=(1.0, 2.0), goal=(5.0, 5.0), backend=None)
        assert robot._robot_id == 3
        np.testing.assert_allclose(robot.pose, [1.0, 2.0, 0.0])
        assert robot.velocity == (0.0, 0.0)
        assert robot._goal == (5.0, 5.0)

    def test_step_moves_toward_goal(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert isinstance(action, Action)
        assert action.behavior == "GO_TO"
        assert len(sink.events) == 1
        assert sink.events[0][0] == "diffdrive_step"

    def test_step_returns_done_at_goal(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(5.0, 0.0), goal=(5.0, 0.0), backend=None)

        states = {0: _make_state(x=5.0, y=0.0, goal_x=5.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert action.behavior == "DONE"

    def test_step_with_path(self):
        path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        robot = DifferentialDriveRobot(path=path)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(4.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0, goal_x=4.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert action.behavior == "GO_TO"

    def test_step_advances_waypoints(self):
        """When close to a waypoint, the robot should advance to the next one."""
        path = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]])
        robot = DifferentialDriveRobot(path=path)
        robot.reset(robot_id=0, start=(0.05, 0.0), goal=(5.0, 0.0), backend=None)

        # Place agent very close to first waypoint
        states = {0: _make_state(x=0.05, y=0.0, goal_x=5.0)}
        sink = _EventCollector()
        robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        # Should have advanced past waypoint 0
        assert robot._wp_idx >= 1

    def test_pose_property(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(1.5, 2.5), goal=(5.0, 0.0), backend=None)
        pose = robot.pose
        assert pose.shape == (3,)
        np.testing.assert_allclose(pose[:2], [1.5, 2.5])

    def test_velocity_property(self):
        robot = DifferentialDriveRobot()
        v, omega = robot.velocity
        assert v == 0.0
        assert omega == 0.0

    def test_odometry_pose(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(1.0, 2.0), goal=(5.0, 0.0), backend=None)
        odo = robot.odometry_pose
        assert odo.shape == (3,)

    def test_get_sensor_poses_empty(self):
        robot = DifferentialDriveRobot()
        poses = robot.get_sensor_poses()
        assert poses == []

    def test_get_sensor_poses(self):
        mounts = [
            SensorMountPoint(offset_x=0.1, offset_y=0.0, offset_theta=0.0),
            SensorMountPoint(offset_x=0.0, offset_y=0.1, offset_theta=math.pi / 2),
        ]
        robot = DifferentialDriveRobot(sensor_mounts=mounts)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        poses = robot.get_sensor_poses()
        assert len(poses) == 2

    def test_get_wheel_speeds(self):
        robot = DifferentialDriveRobot()
        wl, wr = robot.get_wheel_speeds()
        assert wl == 0.0
        assert wr == 0.0

    def test_multi_step_progress(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        sink = _EventCollector()

        x, y = 0.0, 0.0
        for i in range(50):
            states = {0: _make_state(x=x, y=y)}
            action = robot.step(step=i, time_s=i * 0.1, dt=0.1, states=states, emit_event=sink)
            x += action.pref_vx * 0.1
            y += action.pref_vy * 0.1

        # Should have made progress toward goal
        assert x > 0.5

    def test_wheel_speeds_after_motion(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        sink = _EventCollector()

        states = {0: _make_state(x=0.0, y=0.0)}
        robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        wl, wr = robot.get_wheel_speeds()
        # After a step toward goal, wheels should be spinning
        assert abs(wl) > 0.0 or abs(wr) > 0.0


# ===================================================================
# AckermannRobot
# ===================================================================


class TestAckermannRobot:
    def test_init_defaults(self):
        robot = AckermannRobot()
        assert robot.config is not None
        assert robot.steering_angle == 0.0
        np.testing.assert_array_equal(robot.pose, [0.0, 0.0, 0.0])

    def test_init_pure_pursuit(self):
        robot = AckermannRobot(controller="pure_pursuit")
        assert robot._pursuit is not None

    def test_init_stanley(self):
        robot = AckermannRobot(controller="stanley")
        assert robot._stanley is not None

    def test_init_with_lane(self):
        lane = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        robot = AckermannRobot(lane_points=lane)
        assert robot._lane is not None

    def test_reset(self):
        robot = AckermannRobot()
        robot.reset(robot_id=5, start=(1.0, 2.0), goal=(10.0, 2.0), backend=None)
        assert robot._robot_id == 5
        np.testing.assert_allclose(robot.pose[:2], [1.0, 2.0])
        assert robot._goal == (10.0, 2.0)
        assert robot.steering_angle == 0.0

    def test_step_pure_pursuit(self):
        robot = AckermannRobot(controller="pure_pursuit", desired_speed=2.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(10.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0, goal_x=10.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert isinstance(action, Action)
        assert action.behavior == "GO_TO"
        assert len(sink.events) == 1
        assert sink.events[0][0] == "ackermann_step"

    def test_step_stanley(self):
        lane = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        robot = AckermannRobot(controller="stanley", lane_points=lane, desired_speed=2.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(10.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0, goal_x=10.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert isinstance(action, Action)
        assert action.behavior == "GO_TO"

    def test_step_returns_done_at_goal(self):
        robot = AckermannRobot()
        robot.reset(robot_id=0, start=(10.0, 0.0), goal=(10.0, 0.0), backend=None)

        states = {0: _make_state(x=10.0, y=0.0, goal_x=10.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        assert action.behavior == "DONE"

    def test_step_creates_lane_when_none(self):
        """When no lane is provided, the robot should create one from start to goal."""
        robot = AckermannRobot(controller="pure_pursuit")
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(10.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0, goal_x=10.0)}
        sink = _EventCollector()
        robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        # After step, a lane should have been auto-generated
        assert robot._lane is not None
        assert len(robot._lane) == 2

    def test_pose_property(self):
        robot = AckermannRobot()
        robot.reset(robot_id=0, start=(3.0, 4.0), goal=(10.0, 0.0), backend=None)
        pose = robot.pose
        assert pose.shape == (3,)
        np.testing.assert_allclose(pose[:2], [3.0, 4.0])

    def test_steering_angle_property(self):
        robot = AckermannRobot()
        assert robot.steering_angle == 0.0

    def test_compute_reeds_shepp(self):
        robot = AckermannRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        path = robot.compute_reeds_shepp(5.0, 0.0, 0.0, num_samples=50)
        assert path.shape == (50, 3)

    def test_plan_parallel_park(self):
        robot = AckermannRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        poses, controls = robot.plan_parallel_park(3.0, 1.0, 0.0, dt=0.05)
        assert poses.ndim == 2
        assert controls.ndim == 2

    def test_get_footprint(self):
        robot = AckermannRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        fp = robot.get_footprint()
        assert fp.shape == (4, 2)

    def test_set_lane(self):
        robot = AckermannRobot()
        lane = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        robot.set_lane(lane)
        assert robot._lane is not None
        np.testing.assert_array_equal(robot._lane, lane)

    def test_set_lane_makes_copy(self):
        robot = AckermannRobot()
        lane = np.array([[0.0, 0.0], [1.0, 0.0]])
        robot.set_lane(lane)
        lane[0, 0] = 999.0
        assert robot._lane[0, 0] != 999.0

    def test_multi_step_pure_pursuit(self):
        robot = AckermannRobot(controller="pure_pursuit", desired_speed=2.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(10.0, 0.0), backend=None)
        sink = _EventCollector()

        x, y = 0.0, 0.0
        for i in range(30):
            states = {0: _make_state(x=x, y=y, goal_x=10.0)}
            action = robot.step(step=i, time_s=i * 0.1, dt=0.1, states=states, emit_event=sink)
            x += action.pref_vx * 0.1
            y += action.pref_vy * 0.1

        assert x > 1.0  # Should have made forward progress

    def test_multi_step_stanley(self):
        lane = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        robot = AckermannRobot(controller="stanley", lane_points=lane, desired_speed=2.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(10.0, 0.0), backend=None)
        sink = _EventCollector()

        x, y = 0.0, 0.0
        for i in range(30):
            states = {0: _make_state(x=x, y=y, goal_x=10.0)}
            action = robot.step(step=i, time_s=i * 0.1, dt=0.1, states=states, emit_event=sink)
            x += action.pref_vx * 0.1
            y += action.pref_vy * 0.1

        assert x > 1.0


# ===================================================================
# Edge cases across all robots
# ===================================================================


class TestRobotEdgeCases:
    def test_holonomic_reset_clears_velocity(self):
        robot = HolonomicRobot(desired_speed=1.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)

        # Step to get some velocity
        states = {0: _make_state(x=0.0, y=0.0)}
        sink = _EventCollector()
        robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)
        assert robot.speed > 0.0

        # Reset should clear it
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)
        assert robot.speed == 0.0

    def test_diffdrive_reset_clears_pid(self):
        robot = DifferentialDriveRobot()
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(5.0, 0.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0)}
        sink = _EventCollector()
        robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        # Reset and verify clean state
        robot.reset(robot_id=1, start=(1.0, 1.0), goal=(3.0, 3.0), backend=None)
        assert robot._robot_id == 1
        assert robot.velocity == (0.0, 0.0)

    def test_holonomic_diagonal_goal(self):
        robot = HolonomicRobot(desired_speed=1.0)
        robot.reset(robot_id=0, start=(0.0, 0.0), goal=(3.0, 4.0), backend=None)

        states = {0: _make_state(x=0.0, y=0.0, goal_x=3.0, goal_y=4.0)}
        sink = _EventCollector()
        action = robot.step(step=0, time_s=0.0, dt=0.1, states=states, emit_event=sink)

        # Should move in both x and y directions
        assert action.pref_vx > 0.0
        assert action.pref_vy > 0.0

    def test_holonomic_with_custom_config(self):
        cfg = HolonomicConfig(max_speed=3.0, max_acceleration=5.0)
        robot = HolonomicRobot(config=cfg, desired_speed=2.0)
        assert robot.config.max_speed == 3.0
        assert robot.config.max_acceleration == 5.0

    def test_diffdrive_with_custom_config(self):
        cfg = DifferentialDriveConfig(wheel_base=0.5, wheel_radius=0.05)
        robot = DifferentialDriveRobot(config=cfg)
        assert robot.config.wheel_base == 0.5

    def test_ackermann_with_custom_config(self):
        cfg = AckermannConfig(wheelbase=3.0, max_speed=8.0)
        robot = AckermannRobot(config=cfg, desired_speed=4.0)
        assert robot.config.wheelbase == 3.0
