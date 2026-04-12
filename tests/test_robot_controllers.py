"""Tests for robot controller classes (DifferentialDriveRobot, HolonomicRobot, AckermannRobot).

These tests cover the .reset()/.step() lifecycle and accessor properties that
were previously untested. The helper functions are tested in test_robots.py.
"""

from __future__ import annotations

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
    agent_id: int,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
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
        radius=0.3,
        max_speed=2.0,
    )


class EventCollector:
    """Collects events emitted by robot controllers."""

    def __init__(self):
        self.events: list[tuple] = []

    def __call__(self, event_type: str, agent_id: int | None, payload: dict) -> None:
        self.events.append((event_type, agent_id, payload))


# ===================================================================
# DifferentialDriveRobot
# ===================================================================


class TestDifferentialDriveRobot:
    def test_reset_sets_state(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (1.0, 2.0), (10.0, 5.0), backend=None)
        pose = robot.pose
        assert pose[0] == pytest.approx(1.0)
        assert pose[1] == pytest.approx(2.0)
        assert pose[2] == pytest.approx(0.0)  # heading starts at 0

    def test_velocity_starts_zero(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        v, omega = robot.velocity
        assert v == pytest.approx(0.0)
        assert omega == pytest.approx(0.0)

    def test_odometry_pose_at_start(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (3.0, 4.0), (10.0, 0.0), backend=None)
        odom = robot.odometry_pose
        assert odom[0] == pytest.approx(3.0)
        assert odom[1] == pytest.approx(4.0)

    def test_step_moves_toward_goal(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        events = EventCollector()

        # Run several steps toward a goal straight ahead
        for i in range(5):
            states = {0: _make_state(0, robot.pose[0], robot.pose[1], goal_x=5.0)}
            action = robot.step(i, i * 0.1, 0.1, states, events)
            assert isinstance(action, Action)

        # Should have moved in the positive x direction
        assert robot.pose[0] > 0.0
        assert len(events.events) == 5
        assert events.events[0][0] == "diffdrive_step"

    def test_step_returns_done_at_goal(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (5.0, 0.0), (5.0, 0.0), backend=None)
        events = EventCollector()
        states = {0: _make_state(0, 5.0, 0.0, goal_x=5.0)}
        action = robot.step(0, 0.0, 0.1, states, events)
        assert action.behavior == "DONE"
        assert action.pref_vx == pytest.approx(0.0)
        assert action.pref_vy == pytest.approx(0.0)

    def test_step_with_path(self):
        waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        robot = DifferentialDriveRobot(path=waypoints)
        robot.reset(0, (0.0, 0.0), (3.0, 0.0), backend=None)
        events = EventCollector()

        for i in range(10):
            states = {0: _make_state(0, robot.pose[0], robot.pose[1], goal_x=3.0)}
            action = robot.step(i, i * 0.1, 0.1, states, events)
            assert isinstance(action, Action)

        # Should have progressed forward
        assert robot.pose[0] > 0.0

    def test_get_wheel_speeds(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        omega_l, omega_r = robot.get_wheel_speeds()
        # At rest both wheels should be stopped
        assert omega_l == pytest.approx(0.0)
        assert omega_r == pytest.approx(0.0)

    def test_get_sensor_poses_empty(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        poses = robot.get_sensor_poses()
        assert poses == []

    def test_get_sensor_poses_with_mount(self):
        mount = SensorMountPoint(name="lidar", offset_x=0.2, offset_y=0.0, offset_theta=0.0)
        robot = DifferentialDriveRobot(sensor_mounts=[mount])
        robot.reset(0, (1.0, 0.0), (5.0, 0.0), backend=None)
        poses = robot.get_sensor_poses()
        assert len(poses) == 1
        sx, sy, stheta = poses[0]
        assert sx == pytest.approx(1.2, abs=1e-6)
        assert sy == pytest.approx(0.0, abs=1e-6)

    def test_multiple_resets(self):
        robot = DifferentialDriveRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        events = EventCollector()
        states = {0: _make_state(0, 0.5, 0.0, goal_x=5.0)}
        robot.step(0, 0.0, 0.1, states, events)

        # Reset to a different position
        robot.reset(1, (10.0, 10.0), (20.0, 10.0), backend=None)
        assert robot.pose[0] == pytest.approx(10.0)
        assert robot.pose[1] == pytest.approx(10.0)
        v, omega = robot.velocity
        assert v == pytest.approx(0.0)

    def test_custom_config(self):
        cfg = DifferentialDriveConfig(
            wheel_radius=0.1,
            wheel_base=0.5,
            max_linear_vel=2.0,
            max_angular_vel=3.0,
        )
        robot = DifferentialDriveRobot(config=cfg)
        assert robot.config.wheel_radius == 0.1
        assert robot.config.wheel_base == 0.5


# ===================================================================
# HolonomicRobot
# ===================================================================


class TestHolonomicRobot:
    def test_reset_sets_position(self):
        robot = HolonomicRobot()
        robot.reset(0, (2.0, 3.0), (10.0, 5.0), backend=None)
        pos = robot.position
        assert pos[0] == pytest.approx(2.0)
        assert pos[1] == pytest.approx(3.0)

    def test_velocity_starts_zero(self):
        robot = HolonomicRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        vel = robot.velocity
        assert vel[0] == pytest.approx(0.0)
        assert vel[1] == pytest.approx(0.0)
        assert robot.speed == pytest.approx(0.0)

    def test_step_moves_toward_goal(self):
        robot = HolonomicRobot(desired_speed=1.0)
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        events = EventCollector()

        for i in range(5):
            states = {0: _make_state(0, robot.position[0], robot.position[1], goal_x=5.0)}
            action = robot.step(i, i * 0.1, 0.1, states, events)
            assert isinstance(action, Action)
            assert action.behavior == "GO_TO"

        assert len(events.events) == 5
        assert events.events[0][0] == "holonomic_step"

    def test_step_returns_done_at_goal(self):
        robot = HolonomicRobot()
        robot.reset(0, (5.0, 0.0), (5.0, 0.0), backend=None)
        events = EventCollector()
        states = {0: _make_state(0, 5.0, 0.0, goal_x=5.0)}
        action = robot.step(0, 0.0, 0.1, states, events)
        assert action.behavior == "DONE"

    def test_step_with_waypoints(self):
        wps = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        robot = HolonomicRobot(waypoints=wps, desired_speed=1.0)
        robot.reset(0, (0.0, 0.0), (3.0, 0.0), backend=None)
        events = EventCollector()

        actions = []
        for i in range(5):
            states = {0: _make_state(0, robot.position[0], robot.position[1], goal_x=3.0)}
            action = robot.step(i, i * 0.1, 0.1, states, events)
            actions.append(action)

        # All actions should be valid
        assert all(isinstance(a, Action) for a in actions)

    def test_set_waypoints(self):
        robot = HolonomicRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 0.0), backend=None)
        robot.set_waypoints(np.array([[1.0, 0.0], [2.0, 0.0]]), desired_speed=2.0)
        assert robot._follower is not None

    def test_custom_config(self):
        cfg = HolonomicConfig(max_speed=3.0, max_acceleration=5.0, inertia_tau=0.2)
        robot = HolonomicRobot(config=cfg)
        assert robot.config.max_speed == 3.0
        assert robot.config.max_acceleration == 5.0


# ===================================================================
# AckermannRobot
# ===================================================================


class TestAckermannRobot:
    def test_reset_sets_pose(self):
        robot = AckermannRobot()
        robot.reset(0, (1.0, 2.0), (10.0, 2.0), backend=None)
        pose = robot.pose
        assert pose[0] == pytest.approx(1.0)
        assert pose[1] == pytest.approx(2.0)
        assert pose[2] == pytest.approx(0.0)

    def test_steering_angle_starts_zero(self):
        robot = AckermannRobot()
        robot.reset(0, (0.0, 0.0), (10.0, 0.0), backend=None)
        assert robot.steering_angle == pytest.approx(0.0)

    def test_step_pure_pursuit(self):
        robot = AckermannRobot(controller="pure_pursuit", desired_speed=2.0)
        robot.reset(0, (0.0, 0.0), (10.0, 0.0), backend=None)
        events = EventCollector()

        for i in range(5):
            states = {0: _make_state(0, robot.pose[0], robot.pose[1], goal_x=10.0)}
            action = robot.step(i, i * 0.1, 0.1, states, events)
            assert isinstance(action, Action)

        assert len(events.events) == 5
        assert events.events[0][0] == "ackermann_step"
        # Should move forward
        assert robot.pose[0] > 0.0

    def test_step_stanley(self):
        lane = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        robot = AckermannRobot(controller="stanley", lane_points=lane, desired_speed=2.0)
        robot.reset(0, (0.0, 0.0), (10.0, 0.0), backend=None)
        events = EventCollector()

        for i in range(5):
            states = {0: _make_state(0, robot.pose[0], robot.pose[1], goal_x=10.0)}
            action = robot.step(i, i * 0.1, 0.1, states, events)
            assert isinstance(action, Action)

        assert robot.pose[0] > 0.0

    def test_step_returns_done_at_goal(self):
        robot = AckermannRobot()
        robot.reset(0, (10.0, 0.0), (10.0, 0.0), backend=None)
        events = EventCollector()
        states = {0: _make_state(0, 10.0, 0.0, goal_x=10.0)}
        action = robot.step(0, 0.0, 0.1, states, events)
        assert action.behavior == "DONE"

    def test_step_builds_lane_when_none(self):
        """When no lane_points given, step() should auto-create a 2-point lane."""
        robot = AckermannRobot(controller="pure_pursuit", lane_points=None)
        robot.reset(0, (0.0, 0.0), (10.0, 0.0), backend=None)
        events = EventCollector()
        states = {0: _make_state(0, 0.0, 0.0, goal_x=10.0)}
        action = robot.step(0, 0.0, 0.1, states, events)
        assert isinstance(action, Action)
        assert robot._lane is not None
        assert robot._lane.shape == (2, 2)

    def test_custom_config(self):
        cfg = AckermannConfig(wheelbase=3.0, max_speed=8.0)
        robot = AckermannRobot(config=cfg)
        assert robot.config.wheelbase == 3.0
        assert robot.config.max_speed == 8.0

    def test_compute_reeds_shepp(self):
        robot = AckermannRobot()
        robot.reset(0, (0.0, 0.0), (5.0, 5.0), backend=None)
        path = robot.compute_reeds_shepp(5.0, 5.0, np.pi / 2, num_samples=50)
        assert path.shape[0] > 0
        assert path.shape[1] == 3  # x, y, theta
