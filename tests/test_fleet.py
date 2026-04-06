"""Tests for navirl/robots/fleet.py — formations, collision avoidance, task assignment, communication."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.robots.fleet import (
    CommunicationModel,
    FleetPlanner,
    FormationConfig,
    FormationType,
    Message,
    compute_formation_offsets,
    fleet_collision_avoidance,
    greedy_task_assignment,
    hungarian_task_assignment,
    rotate_offsets,
)

# ---------------------------------------------------------------------------
# Formation offsets
# ---------------------------------------------------------------------------


class TestFormationOffsets:
    def test_line_single_robot(self):
        cfg = FormationConfig(formation_type=FormationType.LINE, spacing=1.5)
        offsets = compute_formation_offsets(1, cfg)
        assert offsets.shape == (1, 2)
        np.testing.assert_allclose(offsets, [[0.0, 0.0]])

    def test_line_three_robots(self):
        cfg = FormationConfig(formation_type=FormationType.LINE, spacing=2.0)
        offsets = compute_formation_offsets(3, cfg)
        assert offsets.shape == (3, 2)
        # x should all be 0, y should be evenly spaced
        np.testing.assert_allclose(offsets[:, 0], [0.0, 0.0, 0.0])
        # Total spread = (3-1)*2 = 4.0, centered at 0
        np.testing.assert_allclose(offsets[:, 1], [-2.0, 0.0, 2.0])

    def test_circle_single_robot(self):
        cfg = FormationConfig(formation_type=FormationType.CIRCLE, spacing=1.5)
        offsets = compute_formation_offsets(1, cfg)
        np.testing.assert_allclose(offsets, [[0.0, 0.0]])

    def test_circle_four_robots(self):
        cfg = FormationConfig(formation_type=FormationType.CIRCLE, spacing=2.0)
        offsets = compute_formation_offsets(4, cfg)
        assert offsets.shape == (4, 2)
        # All should be equidistant from origin
        distances = np.linalg.norm(offsets, axis=1)
        np.testing.assert_allclose(distances, distances[0] * np.ones(4), atol=1e-10)

    def test_circle_spacing_preserved(self):
        """Adjacent robots should be approximately `spacing` apart."""
        spacing = 3.0
        cfg = FormationConfig(formation_type=FormationType.CIRCLE, spacing=spacing)
        offsets = compute_formation_offsets(6, cfg)
        for i in range(6):
            j = (i + 1) % 6
            dist = np.linalg.norm(offsets[i] - offsets[j])
            assert dist == pytest.approx(spacing, abs=1e-6)

    def test_v_shape_leader_at_front(self):
        cfg = FormationConfig(formation_type=FormationType.V_SHAPE, spacing=2.0)
        offsets = compute_formation_offsets(5, cfg)
        assert offsets.shape == (5, 2)
        # Leader (index 0) is at origin
        np.testing.assert_allclose(offsets[0], [0.0, 0.0])
        # Followers are behind (negative x)
        for i in range(1, 5):
            assert offsets[i, 0] < 0

    def test_v_shape_symmetric(self):
        cfg = FormationConfig(formation_type=FormationType.V_SHAPE, spacing=2.0)
        offsets = compute_formation_offsets(5, cfg)
        # Robots 1 and 2 should be at same x, symmetric y
        assert offsets[1, 0] == pytest.approx(offsets[2, 0])
        assert offsets[1, 1] == pytest.approx(-offsets[2, 1])

    def test_wedge_same_as_v(self):
        cfg_v = FormationConfig(formation_type=FormationType.V_SHAPE, spacing=2.0, v_angle=0.5)
        cfg_w = FormationConfig(formation_type=FormationType.WEDGE, spacing=2.0, v_angle=0.5)
        offsets_v = compute_formation_offsets(4, cfg_v)
        offsets_w = compute_formation_offsets(4, cfg_w)
        np.testing.assert_allclose(offsets_v, offsets_w)

    def test_custom_offsets(self):
        custom = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cfg = FormationConfig(formation_type=FormationType.CUSTOM, custom_offsets=custom)
        offsets = compute_formation_offsets(3, cfg)
        np.testing.assert_allclose(offsets, custom)

    def test_custom_offsets_padding(self):
        """If fewer custom offsets than robots, pad with zeros."""
        custom = np.array([[1.0, 2.0]])
        cfg = FormationConfig(formation_type=FormationType.CUSTOM, custom_offsets=custom)
        offsets = compute_formation_offsets(3, cfg)
        assert offsets.shape == (3, 2)
        np.testing.assert_allclose(offsets[0], [1.0, 2.0])
        np.testing.assert_allclose(offsets[1], [0.0, 0.0])
        np.testing.assert_allclose(offsets[2], [0.0, 0.0])

    def test_custom_offsets_truncation(self):
        """If more custom offsets than robots, truncate."""
        custom = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        cfg = FormationConfig(formation_type=FormationType.CUSTOM, custom_offsets=custom)
        offsets = compute_formation_offsets(2, cfg)
        assert offsets.shape == (2, 2)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


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
        offsets = np.array([[1.0, 1.0]])
        rotated = rotate_offsets(offsets, np.pi)
        np.testing.assert_allclose(rotated, [[-1.0, -1.0]], atol=1e-10)

    def test_preserves_distances(self):
        offsets = np.array([[3.0, 4.0], [1.0, 0.0]])
        rotated = rotate_offsets(offsets, 1.23)
        orig_dists = np.linalg.norm(offsets, axis=1)
        rot_dists = np.linalg.norm(rotated, axis=1)
        np.testing.assert_allclose(orig_dists, rot_dists, atol=1e-10)


# ---------------------------------------------------------------------------
# Collision avoidance
# ---------------------------------------------------------------------------


class TestFleetCollisionAvoidance:
    def test_no_collision_no_change(self):
        """Robots far apart should keep their desired velocities."""
        positions = np.array([[0.0, 0.0], [100.0, 0.0]])
        velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])
        desired = velocities.copy()
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, dt=0.1)
        np.testing.assert_allclose(adjusted, desired, atol=1e-10)

    def test_overlapping_pushes_apart(self):
        """Overlapping robots should be pushed apart."""
        positions = np.array([[0.0, 0.0], [0.5, 0.0]])
        velocities = np.zeros((2, 2))
        desired = np.zeros((2, 2))
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, dt=0.1)
        # Robot 0 should be pushed left, robot 1 right
        assert adjusted[0, 0] < 0
        assert adjusted[1, 0] > 0

    def test_separating_robots_unchanged(self):
        """Robots moving apart should not be adjusted."""
        positions = np.array([[0.0, 0.0], [2.0, 0.0]])
        velocities = np.array([[-1.0, 0.0], [1.0, 0.0]])
        desired = velocities.copy()
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, dt=0.1)
        np.testing.assert_allclose(adjusted, desired, atol=1e-10)

    def test_head_on_collision_slows(self):
        """Robots heading toward each other should have speeds reduced."""
        positions = np.array([[0.0, 0.0], [3.0, 0.0]])
        desired = np.array([[5.0, 0.0], [-5.0, 0.0]])
        velocities = desired.copy()
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, dt=0.5)
        # Closing speed should be reduced
        closing_before = desired[0, 0] - desired[1, 0]
        closing_after = adjusted[0, 0] - adjusted[1, 0]
        assert closing_after < closing_before

    def test_coincident_positions(self):
        """Exactly coincident robots should be pushed apart without NaN."""
        positions = np.array([[1.0, 1.0], [1.0, 1.0]])
        velocities = np.zeros((2, 2))
        desired = np.zeros((2, 2))
        radii = np.array([0.3, 0.3])
        adjusted = fleet_collision_avoidance(positions, velocities, desired, radii, dt=0.1)
        assert not np.any(np.isnan(adjusted))
        # Should push apart
        assert np.linalg.norm(adjusted[0] - adjusted[1]) > 0


# ---------------------------------------------------------------------------
# Task assignment (greedy)
# ---------------------------------------------------------------------------


class TestGreedyTaskAssignment:
    def test_equal_robots_tasks(self):
        robots = np.array([[0.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        assignments = greedy_task_assignment(robots, tasks)
        # Robot 0 closest to task 0, robot 1 closest to task 1
        assert assignments[0] == 0
        assert assignments[1] == 1

    def test_more_robots_than_tasks(self):
        robots = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0]])
        assignments = greedy_task_assignment(robots, tasks)
        assert assignments.count(0) == 1
        assert assignments.count(-1) == 2

    def test_no_tasks(self):
        robots = np.array([[0.0, 0.0], [1.0, 0.0]])
        tasks = np.empty((0, 2))
        assignments = greedy_task_assignment(robots, tasks)
        assert assignments == [-1, -1]

    def test_single_robot_single_task(self):
        robots = np.array([[0.0, 0.0]])
        tasks = np.array([[3.0, 4.0]])
        assignments = greedy_task_assignment(robots, tasks)
        assert assignments == [0]

    def test_more_tasks_than_robots(self):
        robots = np.array([[0.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assignments = greedy_task_assignment(robots, tasks)
        assert assignments == [0]  # Closest task


# ---------------------------------------------------------------------------
# Task assignment (hungarian/auction)
# ---------------------------------------------------------------------------


class TestHungarianTaskAssignment:
    def test_equal_robots_tasks(self):
        robots = np.array([[0.0, 0.0], [10.0, 0.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 0.0]])
        assignments = hungarian_task_assignment(robots, tasks)
        # Each robot should get exactly one task
        assert len(assignments) == 2
        assigned_tasks = [a for a in assignments if a != -1]
        assert len(set(assigned_tasks)) == 2

    def test_empty_robots(self):
        robots = np.empty((0, 2))
        tasks = np.array([[1.0, 0.0]])
        assignments = hungarian_task_assignment(robots, tasks)
        assert assignments == []

    def test_empty_tasks(self):
        robots = np.array([[0.0, 0.0]])
        tasks = np.empty((0, 2))
        assignments = hungarian_task_assignment(robots, tasks)
        assert assignments == [-1]

    def test_optimal_assignment(self):
        """The auction should find a reasonable (not necessarily globally optimal) assignment."""
        robots = np.array([[0.0, 0.0], [10.0, 10.0]])
        tasks = np.array([[1.0, 0.0], [9.0, 10.0]])
        assignments = hungarian_task_assignment(robots, tasks)
        # Robot 0 should get task 0 (close), robot 1 should get task 1 (close)
        assert assignments[0] == 0
        assert assignments[1] == 1


# ---------------------------------------------------------------------------
# Communication model
# ---------------------------------------------------------------------------


class TestCommunicationModel:
    def test_send_within_range(self):
        comm = CommunicationModel(max_range=10.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={"data": "hello"})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([5.0, 0.0])}
        assert comm.send(msg, positions) is True

    def test_send_out_of_range(self):
        comm = CommunicationModel(max_range=5.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
        assert comm.send(msg, positions) is False

    def test_receive_messages(self):
        comm = CommunicationModel(max_range=100.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={"x": 1})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
        comm.send(msg, positions)
        received = comm.receive(1)
        assert len(received) == 1
        assert received[0].payload == {"x": 1}

    def test_receive_clears_inbox(self):
        comm = CommunicationModel(max_range=100.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
        comm.send(msg, positions)
        comm.receive(1)
        assert comm.receive(1) == []

    def test_broadcast(self):
        comm = CommunicationModel(max_range=100.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=-1, payload={"alert": True})
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, 0.0]),
            2: np.array([10.0, 0.0]),
        }
        assert comm.send(msg, positions) is True
        assert len(comm.receive(1)) == 1
        assert len(comm.receive(2)) == 1
        assert comm.receive(0) == []  # Sender doesn't get own broadcast

    def test_broadcast_range_limited(self):
        comm = CommunicationModel(max_range=7.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=-1, payload={})
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([5.0, 0.0]),  # In range
            2: np.array([10.0, 0.0]),  # Out of range
        }
        comm.send(msg, positions)
        assert len(comm.receive(1)) == 1
        assert len(comm.receive(2)) == 0

    def test_unknown_sender(self):
        comm = CommunicationModel(max_range=100.0, loss_probability=0.0)
        msg = Message(sender_id=99, receiver_id=1, payload={})
        positions = {1: np.array([0.0, 0.0])}
        assert comm.send(msg, positions) is False

    def test_unknown_receiver(self):
        comm = CommunicationModel(max_range=100.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=99, payload={})
        positions = {0: np.array([0.0, 0.0])}
        assert comm.send(msg, positions) is False

    def test_clear(self):
        comm = CommunicationModel(max_range=100.0, loss_probability=0.0)
        msg = Message(sender_id=0, receiver_id=1, payload={})
        positions = {0: np.array([0.0, 0.0]), 1: np.array([1.0, 0.0])}
        comm.send(msg, positions)
        comm.clear()
        assert comm.receive(1) == []

    def test_message_timestamp(self):
        msg = Message(sender_id=0, receiver_id=1, payload={}, timestamp=1.5)
        assert msg.timestamp == 1.5

    def test_message_default_timestamp(self):
        msg = Message(sender_id=0, receiver_id=1, payload={})
        assert msg.timestamp == 0.0


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
        assert len(result) == 2

    def test_compute_formation_targets(self):
        planner = FleetPlanner(FormationConfig(formation_type=FormationType.LINE, spacing=2.0))
        centroid = np.array([5.0, 5.0])
        targets = planner.compute_formation_targets(centroid, heading=0.0, n_robots=3)
        assert targets.shape == (3, 2)
        # Centroid of targets should be approximately the given centroid
        np.testing.assert_allclose(np.mean(targets, axis=0), centroid, atol=1e-10)

    def test_avoid_collisions_delegates(self):
        planner = FleetPlanner(safety_margin=0.5)
        positions = np.array([[0.0, 0.0], [100.0, 0.0]])
        velocities = np.array([[1.0, 0.0], [0.0, 0.0]])
        desired = velocities.copy()
        radii = np.array([0.3, 0.3])
        adjusted = planner.avoid_collisions(positions, velocities, desired, radii, dt=0.1)
        # Far apart, should be unchanged
        np.testing.assert_allclose(adjusted, desired, atol=1e-10)
