"""Tests for navirl/robots/fleet.py — fleet coordination, formations, and communication."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from navirl.core.types import Action, AgentState
from navirl.robots.fleet import (
    CommunicationModel,
    FleetPlanner,
    FormationConfig,
    FormationType,
    Message,
    RobotFleet,
    compute_formation_offsets,
    fleet_collision_avoidance,
    greedy_task_assignment,
    hungarian_task_assignment,
    rotate_offsets,
)

# ---------------------------------------------------------------------------
# Helper to create AgentState instances
# ---------------------------------------------------------------------------


def _state(
    agent_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0, radius: float = 0.3
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind="robot",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=0.0,
        goal_y=0.0,
        radius=radius,
        max_speed=1.0,
    )


# ---------------------------------------------------------------------------
# Formation offsets
# ---------------------------------------------------------------------------


class TestFormationOffsets:
    def test_line_formation_symmetric(self):
        cfg = FormationConfig(formation_type=FormationType.LINE, spacing=2.0)
        offsets = compute_formation_offsets(3, cfg)
        assert offsets.shape == (3, 2)
        # x-coords should all be 0
        np.testing.assert_allclose(offsets[:, 0], 0.0)
        # y-coords should be symmetric around 0
        assert offsets[0, 1] < 0
        assert offsets[2, 1] > 0
        np.testing.assert_allclose(offsets[1, 1], 0.0)

    def test_line_formation_spacing(self):
        cfg = FormationConfig(formation_type=FormationType.LINE, spacing=1.0)
        offsets = compute_formation_offsets(2, cfg)
        dist = np.linalg.norm(offsets[1] - offsets[0])
        np.testing.assert_allclose(dist, 1.0)

    def test_circle_formation_single_robot(self):
        cfg = FormationConfig(formation_type=FormationType.CIRCLE, spacing=2.0)
        offsets = compute_formation_offsets(1, cfg)
        np.testing.assert_allclose(offsets, [[0.0, 0.0]])

    def test_circle_formation_equidistant(self):
        cfg = FormationConfig(formation_type=FormationType.CIRCLE, spacing=2.0)
        offsets = compute_formation_offsets(4, cfg)
        assert offsets.shape == (4, 2)
        # All robots should be at the same distance from origin
        radii = np.linalg.norm(offsets, axis=1)
        np.testing.assert_allclose(radii, radii[0])

    def test_v_shape_leader_at_origin(self):
        cfg = FormationConfig(formation_type=FormationType.V_SHAPE, spacing=2.0)
        offsets = compute_formation_offsets(5, cfg)
        np.testing.assert_allclose(offsets[0], [0.0, 0.0])
        # Follower robots should be behind the leader (negative x)
        for i in range(1, 5):
            assert offsets[i, 0] < 0

    def test_wedge_formation_same_as_v(self):
        cfg_v = FormationConfig(formation_type=FormationType.V_SHAPE, spacing=2.0)
        cfg_w = FormationConfig(formation_type=FormationType.WEDGE, spacing=2.0)
        offsets_v = compute_formation_offsets(3, cfg_v)
        offsets_w = compute_formation_offsets(3, cfg_w)
        np.testing.assert_allclose(offsets_v, offsets_w)

    def test_custom_formation(self):
        custom = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cfg = FormationConfig(formation_type=FormationType.CUSTOM, custom_offsets=custom)
        offsets = compute_formation_offsets(3, cfg)
        np.testing.assert_allclose(offsets, custom)

    def test_custom_formation_padding(self):
        custom = np.array([[0.0, 0.0], [1.0, 0.0]])
        cfg = FormationConfig(formation_type=FormationType.CUSTOM, custom_offsets=custom)
        offsets = compute_formation_offsets(4, cfg)
        assert offsets.shape == (4, 2)
        # Extra robots padded with zeros
        np.testing.assert_allclose(offsets[2], [0.0, 0.0])
        np.testing.assert_allclose(offsets[3], [0.0, 0.0])

    def test_custom_formation_truncation(self):
        custom = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cfg = FormationConfig(formation_type=FormationType.CUSTOM, custom_offsets=custom)
        offsets = compute_formation_offsets(2, cfg)
        assert offsets.shape == (2, 2)


class TestRotateOffsets:
    def test_zero_rotation(self):
        offsets = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = rotate_offsets(offsets, 0.0)
        np.testing.assert_allclose(rotated, offsets, atol=1e-10)

    def test_90_degree_rotation(self):
        offsets = np.array([[1.0, 0.0]])
        rotated = rotate_offsets(offsets, np.pi / 2)
        np.testing.assert_allclose(rotated, [[0.0, 1.0]], atol=1e-10)

    def test_180_degree_rotation(self):
        offsets = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = rotate_offsets(offsets, np.pi)
        np.testing.assert_allclose(rotated, [[-1.0, 0.0], [0.0, -1.0]], atol=1e-10)


# ---------------------------------------------------------------------------
# Collision avoidance
# ---------------------------------------------------------------------------


class TestFleetCollisionAvoidance:
    def test_no_collision_no_change(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        velocities = np.array([[0.0, 0.0], [0.0, 0.0]])
        desired = np.array([[1.0, 0.0], [-1.0, 0.0]])
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, 0.1)
        # Far apart, no adjustment needed — should remain close to desired
        np.testing.assert_allclose(adjusted, desired, atol=0.1)

    def test_overlapping_robots_push_apart(self):
        positions = np.array([[0.0, 0.0], [0.0, 0.0]])  # Same position
        velocities = np.zeros((2, 2))
        desired = np.zeros((2, 2))
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, 0.1)
        # Should push them apart
        assert not np.allclose(adjusted[0], adjusted[1])

    def test_close_robots_repulsed(self):
        positions = np.array([[0.0, 0.0], [0.3, 0.0]])  # Within min_dist
        velocities = np.zeros((2, 2))
        desired = np.zeros((2, 2))
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, 0.1)
        # Robot 0 pushed left, robot 1 pushed right
        assert adjusted[0, 0] < 0
        assert adjusted[1, 0] > 0

    def test_separating_robots_no_intervention(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0]])
        velocities = np.zeros((2, 2))
        desired = np.array([[-1.0, 0.0], [1.0, 0.0]])  # Moving apart
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, 0.1)
        np.testing.assert_allclose(adjusted, desired, atol=1e-10)

    def test_head_on_collision_course(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0]])
        velocities = np.zeros((2, 2))
        desired = np.array([[5.0, 0.0], [-5.0, 0.0]])  # Closing fast
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, 0.1)
        # Closing speed should be reduced
        rel_speed_before = abs(desired[0, 0] - desired[1, 0])
        rel_speed_after = abs(adjusted[0, 0] - adjusted[1, 0])
        assert rel_speed_after < rel_speed_before


# ---------------------------------------------------------------------------
# Task assignment
# ---------------------------------------------------------------------------


class TestGreedyTaskAssignment:
    def test_one_to_one(self):
        robots = np.array([[0.0, 0.0]])
        tasks = np.array([[1.0, 0.0]])
        result = greedy_task_assignment(robots, tasks)
        assert result == [0]

    def test_nearest_neighbour(self):
        robots = np.array([[0.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        result = greedy_task_assignment(robots, tasks)
        assert result[0] == 0  # Robot 0 closer to task 0
        assert result[1] == 1  # Robot 1 closer to task 1

    def test_no_tasks(self):
        robots = np.array([[0.0, 0.0], [1.0, 0.0]])
        tasks = np.zeros((0, 2))
        result = greedy_task_assignment(robots, tasks)
        assert result == [-1, -1]

    def test_more_robots_than_tasks(self):
        robots = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        tasks = np.array([[0.5, 0.0]])
        result = greedy_task_assignment(robots, tasks)
        assert result.count(0) == 1  # Exactly one robot gets the task
        assert result.count(-1) == 2  # Two unassigned


class TestHungarianTaskAssignment:
    def test_basic_assignment(self):
        robots = np.array([[0.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        result = hungarian_task_assignment(robots, tasks)
        assert result[0] == 0
        assert result[1] == 1

    def test_no_tasks(self):
        robots = np.array([[0.0, 0.0]])
        tasks = np.zeros((0, 2))
        result = hungarian_task_assignment(robots, tasks)
        assert result == [-1]

    def test_no_robots(self):
        robots = np.zeros((0, 2))
        tasks = np.array([[1.0, 0.0]])
        result = hungarian_task_assignment(robots, tasks)
        assert result == []


# ---------------------------------------------------------------------------
# Communication model
# ---------------------------------------------------------------------------


class TestCommunicationModel:
    def test_send_in_range(self):
        comm = CommunicationModel(max_range=10.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={"data": "hello"})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([5.0, 0.0])}
        assert comm.send(msg, positions) is True
        received = comm.receive(1)
        assert len(received) == 1
        assert received[0].payload["data"] == "hello"

    def test_send_out_of_range(self):
        comm = CommunicationModel(max_range=5.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
        assert comm.send(msg, positions) is False

    def test_broadcast(self):
        comm = CommunicationModel(max_range=20.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=-1, payload={"alert": True})
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, 0.0]),
            2: np.array([8.0, 0.0]),
        }
        assert comm.send(msg, positions) is True
        assert len(comm.receive(1)) == 1
        assert len(comm.receive(2)) == 1

    def test_broadcast_partial_range(self):
        comm = CommunicationModel(max_range=6.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=-1, payload={})
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, 0.0]),  # In range
            2: np.array([10.0, 0.0]),  # Out of range
        }
        assert comm.send(msg, positions) is True
        assert len(comm.receive(1)) == 1
        assert len(comm.receive(2)) == 0

    def test_missing_sender(self):
        comm = CommunicationModel()
        msg = Message(sender_id=99, receiver_id=1, payload={})
        positions = {1: np.array([0.0, 0.0])}
        assert comm.send(msg, positions) is False

    def test_missing_receiver(self):
        comm = CommunicationModel()
        msg = Message(sender_id=0, receiver_id=99, payload={})
        positions = {0: np.array([0.0, 0.0])}
        assert comm.send(msg, positions) is False

    def test_receive_clears_inbox(self):
        comm = CommunicationModel(max_range=20.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
        comm.send(msg, positions)
        assert len(comm.receive(1)) == 1
        assert len(comm.receive(1)) == 0  # Already cleared

    def test_clear(self):
        comm = CommunicationModel(max_range=20.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
        comm.send(msg, positions)
        comm.clear()
        assert len(comm.receive(1)) == 0

    def test_packet_loss(self):
        """With 100% loss, no messages delivered."""
        comm = CommunicationModel(max_range=100.0, loss_probability=1.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
        assert comm.send(msg, positions) is False


# ---------------------------------------------------------------------------
# FleetPlanner
# ---------------------------------------------------------------------------


class TestFleetPlanner:
    def test_assign_tasks_greedy(self):
        planner = FleetPlanner()
        robots = np.array([[0.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        result = planner.assign_tasks(robots, tasks, method="greedy")
        assert result[0] == 0
        assert result[1] == 1

    def test_assign_tasks_auction(self):
        planner = FleetPlanner()
        robots = np.array([[0.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        result = planner.assign_tasks(robots, tasks, method="auction")
        assert result[0] == 0
        assert result[1] == 1

    def test_compute_formation_targets(self):
        planner = FleetPlanner(
            FormationConfig(
                formation_type=FormationType.LINE,
                spacing=2.0,
            )
        )
        centroid = np.array([5.0, 5.0])
        targets = planner.compute_formation_targets(centroid, 0.0, 3)
        assert targets.shape == (3, 2)
        # Centroid of targets should be at the given centroid
        np.testing.assert_allclose(np.mean(targets, axis=0), centroid, atol=1e-10)

    def test_avoid_collisions_delegates(self):
        planner = FleetPlanner(safety_margin=0.5)
        pos = np.array([[0.0, 0.0], [10.0, 0.0]])
        vel = np.zeros((2, 2))
        des = np.array([[1.0, 0.0], [-1.0, 0.0]])
        radii = np.array([0.3, 0.3])
        result = planner.avoid_collisions(pos, vel, des, radii, 0.1)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# RobotFleet
# ---------------------------------------------------------------------------


class TestRobotFleet:
    def _make_controller(self):
        ctrl = MagicMock()
        ctrl.reset = MagicMock()
        ctrl.step = MagicMock(return_value=Action(pref_vx=1.0, pref_vy=0.0))
        return ctrl

    def test_add_remove_robot(self):
        fleet = RobotFleet()
        ctrl = self._make_controller()
        fleet.add_robot(1, ctrl)
        assert fleet.size == 1
        assert fleet.robot_ids == [1]
        fleet.remove_robot(1)
        assert fleet.size == 0

    def test_set_leader(self):
        fleet = RobotFleet()
        fleet.set_leader(1)
        assert fleet._leader_id == 1

    def test_reset_calls_controllers(self):
        ctrl1 = self._make_controller()
        ctrl2 = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl1, 1: ctrl2})
        starts = {0: (0.0, 0.0), 1: (1.0, 0.0)}
        goals = {0: (5.0, 0.0), 1: (6.0, 0.0)}
        fleet.reset(starts, goals, backend=None)
        ctrl1.reset.assert_called_once_with(0, (0.0, 0.0), (5.0, 0.0), None)
        ctrl2.reset.assert_called_once_with(1, (1.0, 0.0), (6.0, 0.0), None)

    def test_step_empty_fleet(self):
        fleet = RobotFleet()
        result = fleet.step(0, 0.0, 0.1, {}, lambda *a: None)
        assert result == {}

    def test_step_returns_actions(self):
        ctrl = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl})
        states = {0: _state(0, 0.0, 0.0)}
        emit = MagicMock()
        actions = fleet.step(0, 0.0, 0.1, states, emit)
        assert 0 in actions
        assert isinstance(actions[0], Action)
        emit.assert_called()

    def test_step_with_leader(self):
        ctrl0 = self._make_controller()
        ctrl1 = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl0, 1: ctrl1})
        fleet.set_leader(0)
        states = {
            0: _state(0, 0.0, 0.0, vx=1.0),
            1: _state(1, 2.0, 0.0),
        }
        actions = fleet.step(0, 0.0, 0.1, states, lambda *a: None)
        assert len(actions) == 2

    def test_broadcast_and_receive(self):
        ctrl0 = self._make_controller()
        ctrl1 = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl0, 1: ctrl1}, comm_range=50.0)
        states = {
            0: _state(0, 0.0, 0.0),
            1: _state(1, 5.0, 0.0),
        }
        result = fleet.broadcast(0, {"hello": True}, 1.0, states)
        assert result is True
        msgs = fleet.get_messages(1)
        assert len(msgs) == 1
        assert msgs[0].payload["hello"] is True

    def test_assign_tasks(self):
        ctrl0 = self._make_controller()
        ctrl1 = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl0, 1: ctrl1})
        states = {
            0: _state(0, 0.0, 0.0),
            1: _state(1, 10.0, 0.0),
        }
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        assignments = fleet.assign_tasks(tasks, states)
        assert assignments[0] == 0
        assert assignments[1] == 1

    def test_fleet_centroid(self):
        ctrl0 = self._make_controller()
        ctrl1 = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl0, 1: ctrl1})
        states = {
            0: _state(0, 0.0, 0.0),
            1: _state(1, 4.0, 2.0),
        }
        centroid = fleet.fleet_centroid(states)
        np.testing.assert_allclose(centroid, [2.0, 1.0])

    def test_fleet_centroid_empty(self):
        fleet = RobotFleet()
        centroid = fleet.fleet_centroid({})
        np.testing.assert_allclose(centroid, [0.0, 0.0])

    def test_fleet_spread(self):
        ctrl0 = self._make_controller()
        ctrl1 = self._make_controller()
        fleet = RobotFleet(controllers={0: ctrl0, 1: ctrl1})
        states = {
            0: _state(0, 0.0, 0.0),
            1: _state(1, 3.0, 4.0),
        }
        spread = fleet.fleet_spread(states)
        np.testing.assert_allclose(spread, 5.0)  # 3-4-5 triangle

    def test_fleet_spread_single_robot(self):
        fleet = RobotFleet(controllers={0: self._make_controller()})
        states = {0: _state(0, 0.0, 0.0)}
        assert fleet.fleet_spread(states) == 0.0

    def test_formation_error_empty(self):
        fleet = RobotFleet()
        assert fleet.formation_error({}) == 0.0

    def test_formation_error_in_formation(self):
        ctrl0 = self._make_controller()
        ctrl1 = self._make_controller()
        fleet = RobotFleet(
            controllers={0: ctrl0, 1: ctrl1},
            formation_config=FormationConfig(
                formation_type=FormationType.LINE,
                spacing=2.0,
            ),
        )
        # Place robots at ideal formation positions
        offsets = compute_formation_offsets(2, fleet._formation_config)
        centroid = np.array([5.0, 5.0])
        targets = offsets + centroid
        states = {
            0: _state(0, targets[0, 0], targets[0, 1]),
            1: _state(1, targets[1, 0], targets[1, 1]),
        }
        error = fleet.formation_error(states, heading=0.0)
        np.testing.assert_allclose(error, 0.0, atol=1e-10)
