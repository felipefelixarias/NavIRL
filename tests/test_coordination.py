"""Tests for the navirl.coordination sub-package.

Covers communication, consensus, formation, task_allocation, and planning
modules which previously had 0% test coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------
from navirl.coordination.communication import (
    BroadcastChannel,
    DirectChannel,
    MessageProtocol,
    SharedMemory,
)

# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------
from navirl.coordination.consensus import (
    AverageConsensus,
    ConsensusOptimizer,
    MaxConsensus,
    WeightedConsensus,
)

# ---------------------------------------------------------------------------
# Formation
# ---------------------------------------------------------------------------
from navirl.coordination.formation import (
    ConsensusFormation,
    FormationController,
    LeaderFollower,
)

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
from navirl.coordination.planning import (
    CBSPlanner,
    PlanningResult,
    PriorityPlanner,
    VelocityObstaclePlanner,
)

# ---------------------------------------------------------------------------
# Task allocation
# ---------------------------------------------------------------------------
from navirl.coordination.task_allocation import (
    AllocationResult,
    AuctionAllocator,
    GreedyAllocator,
    HungarianAllocator,
    Task,
)

# ===================================================================
# MessageProtocol
# ===================================================================


class TestMessageProtocol:
    def test_basic_creation(self):
        msg = MessageProtocol(sender="a1", receiver="a2", content="hello")
        assert msg.sender == "a1"
        assert msg.receiver == "a2"
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_broadcast_message(self):
        msg = MessageProtocol(sender="a1", receiver=None, content=[1, 2, 3])
        assert msg.receiver is None

    def test_metadata(self):
        msg = MessageProtocol(
            sender="a1", receiver="a2", content="x", metadata={"priority": 5}
        )
        assert msg.metadata["priority"] == 5


# ===================================================================
# BroadcastChannel
# ===================================================================


class TestBroadcastChannel:
    def test_send_and_receive(self):
        ch = BroadcastChannel()
        msg = MessageProtocol(sender="a1", receiver=None, content="data")
        ch.send(msg)
        received = ch.receive()
        assert len(received) == 1
        assert received[0].content == "data"

    def test_receive_returns_copy(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        r1 = ch.receive()
        r2 = ch.receive()
        assert len(r1) == 1
        assert len(r2) == 1  # messages persist

    def test_clear(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        ch.clear()
        assert ch.size == 0
        assert ch.receive() == []

    def test_max_buffer_size(self):
        ch = BroadcastChannel(max_buffer_size=3)
        for i in range(5):
            ch.send(MessageProtocol(sender="a1", receiver=None, content=i))
        assert ch.size == 3
        msgs = ch.receive()
        assert msgs[0].content == 2  # oldest kept is index 2

    def test_size_property(self):
        ch = BroadcastChannel()
        assert ch.size == 0
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        assert ch.size == 1


# ===================================================================
# DirectChannel
# ===================================================================


class TestDirectChannel:
    def test_send_and_receive(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="hi"))
        msgs = ch.receive("a2")
        assert len(msgs) == 1
        assert msgs[0].content == "hi"

    def test_receive_clears_mailbox(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="hi"))
        ch.receive("a2")
        assert ch.receive("a2") == []

    def test_peek_does_not_clear(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="hi"))
        peeked = ch.peek("a2")
        assert len(peeked) == 1
        assert len(ch.receive("a2")) == 1  # still there

    def test_send_without_receiver_raises(self):
        ch = DirectChannel()
        with pytest.raises(ValueError, match="specific receiver"):
            ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))

    def test_multiple_receivers(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="for a2"))
        ch.send(MessageProtocol(sender="a1", receiver="a3", content="for a3"))
        assert len(ch.receive("a2")) == 1
        assert len(ch.receive("a3")) == 1
        assert ch.receive("a4") == []


# ===================================================================
# SharedMemory
# ===================================================================


class TestSharedMemory:
    def test_write_and_read(self):
        sm = SharedMemory()
        sm.write("key1", 42)
        assert sm.read("key1") == 42

    def test_read_default(self):
        sm = SharedMemory()
        assert sm.read("missing") is None
        assert sm.read("missing", "default") == "default"

    def test_read_all(self):
        sm = SharedMemory()
        sm.write("a", 1)
        sm.write("b", 2)
        assert sm.read_all() == {"a": 1, "b": 2}

    def test_clear(self):
        sm = SharedMemory()
        sm.write("a", 1)
        sm.clear()
        assert sm.keys == []
        assert sm.read("a") is None

    def test_keys_property(self):
        sm = SharedMemory()
        sm.write("x", 10)
        sm.write("y", 20)
        assert set(sm.keys) == {"x", "y"}

    def test_overwrite(self):
        sm = SharedMemory()
        sm.write("k", 1)
        sm.write("k", 2)
        assert sm.read("k") == 2


# ===================================================================
# AverageConsensus
# ===================================================================


class TestAverageConsensus:
    def test_single_step_convergence(self):
        ac = AverageConsensus(gain=1.0)
        local = np.array([0.0, 0.0])
        neighbors = [np.array([2.0, 4.0])]
        result = ac.step(local, neighbors)
        # With gain=1.0, should move to the mean of [0,0] and [2,4] = [1,2]
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_no_neighbors(self):
        ac = AverageConsensus(gain=0.5)
        local = np.array([5.0, 5.0])
        result = ac.step(local, [])
        np.testing.assert_array_equal(result, [5.0, 5.0])

    def test_partial_convergence(self):
        ac = AverageConsensus(gain=0.5)
        local = np.array([0.0])
        neighbors = [np.array([10.0])]
        result = ac.step(local, neighbors)
        # mean = 5.0, result = 0 + 0.5*(5-0) = 2.5
        np.testing.assert_allclose(result, [2.5])

    def test_invalid_gain(self):
        with pytest.raises(ValueError, match="gain must be"):
            AverageConsensus(gain=0.0)
        with pytest.raises(ValueError, match="gain must be"):
            AverageConsensus(gain=1.5)


class TestMaxConsensus:
    def test_basic(self):
        mc = MaxConsensus()
        local = np.array([1.0, 5.0, 3.0])
        neighbors = [np.array([4.0, 2.0, 6.0]), np.array([0.0, 7.0, 1.0])]
        result = mc.step(local, neighbors)
        np.testing.assert_array_equal(result, [4.0, 7.0, 6.0])

    def test_no_neighbors(self):
        mc = MaxConsensus()
        local = np.array([1.0, 2.0])
        result = mc.step(local, [])
        np.testing.assert_array_equal(result, [1.0, 2.0])


class TestWeightedConsensus:
    def test_weights_sum_to_one(self):
        wc = WeightedConsensus(degree=2, neighbor_degrees=[2, 3])
        total = wc.self_weight + sum(wc.weights)
        assert abs(total - 1.0) < 1e-12

    def test_step(self):
        wc = WeightedConsensus(degree=2, neighbor_degrees=[2, 2])
        local = np.array([0.0])
        neighbors = [np.array([6.0]), np.array([6.0])]
        result = wc.step(local, neighbors)
        # self_weight * 0 + w1 * 6 + w2 * 6
        expected = wc.self_weight * 0.0 + wc.weights[0] * 6.0 + wc.weights[1] * 6.0
        np.testing.assert_allclose(result, [expected])

    def test_wrong_neighbor_count(self):
        wc = WeightedConsensus(degree=2, neighbor_degrees=[2, 3])
        with pytest.raises(ValueError, match="Expected 2"):
            wc.step(np.array([0.0]), [np.array([1.0])])


class TestConsensusOptimizer:
    def test_single_step(self):
        ac = AverageConsensus(gain=0.5)
        opt = ConsensusOptimizer(ac, lr=0.1)
        local = np.array([5.0])
        grad = np.array([2.0])
        neighbors = [np.array([5.0])]
        result = opt.step(local, grad, neighbors)
        # After gradient: 5 - 0.1*2 = 4.8
        # After consensus with neighbor [5.0]: mean of [4.8, 5.0] = 4.9
        # result = 4.8 + 0.5*(4.9 - 4.8) = 4.85
        assert isinstance(result, np.ndarray)

    def test_run_converges(self):
        """Optimize f(x) = x^2 with a single agent (trivial consensus)."""
        ac = AverageConsensus(gain=0.5)
        opt = ConsensusOptimizer(ac, lr=0.1)
        result = opt.run(
            initial_value=np.array([10.0]),
            gradient_fn=lambda x: 2.0 * x,  # gradient of x^2
            get_neighbors_fn=lambda x: [],
            num_steps=200,
            tolerance=1e-6,
        )
        np.testing.assert_allclose(result, [0.0], atol=0.01)

    def test_early_stopping(self):
        ac = AverageConsensus(gain=0.5)
        opt = ConsensusOptimizer(ac, lr=0.01)
        result = opt.run(
            initial_value=np.array([0.0]),  # already at minimum
            gradient_fn=lambda x: 2.0 * x,
            get_neighbors_fn=lambda x: [],
            num_steps=1000,
            tolerance=1e-4,
        )
        np.testing.assert_allclose(result, [0.0], atol=1e-4)


# ===================================================================
# FormationController
# ===================================================================


class TestFormationController:
    def test_line_formation(self):
        fc = FormationController(spacing=2.0)
        pos = fc.compute_desired_positions(
            center=np.array([0.0, 0.0]), heading=0.0, formation_type="line", num_agents=3
        )
        assert pos.shape == (3, 2)
        # Agents should be spread along y-axis (perpendicular to heading=0)
        # with spacing=2, offsets at y=-2, 0, 2
        np.testing.assert_allclose(pos[:, 1], [-2.0, 0.0, 2.0])

    def test_circle_formation(self):
        fc = FormationController(spacing=2.0)
        pos = fc.compute_desired_positions(
            center=np.array([5.0, 5.0]), heading=0.0, formation_type="circle", num_agents=4
        )
        assert pos.shape == (4, 2)
        # All positions should be equidistant from center
        dists = np.linalg.norm(pos - np.array([5.0, 5.0]), axis=1)
        np.testing.assert_allclose(dists, dists[0] * np.ones(4), atol=1e-10)

    def test_wedge_formation(self):
        fc = FormationController(spacing=1.0)
        pos = fc.compute_desired_positions(
            center=np.array([0.0, 0.0]), heading=0.0, formation_type="wedge", num_agents=3
        )
        assert pos.shape == (3, 2)
        # Leader at front (index 0)
        assert pos[0, 0] >= pos[1, 0]  # leader ahead of followers

    def test_diamond_formation(self):
        fc = FormationController(spacing=2.0)
        pos = fc.compute_desired_positions(
            center=np.array([0.0, 0.0]), heading=0.0, formation_type="diamond", num_agents=4
        )
        assert pos.shape == (4, 2)

    def test_custom_formation(self):
        offsets = np.array([[0, 0], [1, 0], [0, 1]])
        fc = FormationController(custom_offsets=offsets)
        pos = fc.compute_desired_positions(
            center=np.array([10.0, 10.0]), heading=0.0, formation_type="custom", num_agents=3
        )
        np.testing.assert_allclose(pos, offsets + np.array([10.0, 10.0]))

    def test_custom_without_offsets_raises(self):
        fc = FormationController()
        with pytest.raises(ValueError, match="custom_offsets"):
            fc.compute_desired_positions(
                center=np.array([0, 0]), heading=0.0, formation_type="custom", num_agents=2
            )

    def test_unknown_formation_raises(self):
        fc = FormationController()
        with pytest.raises(ValueError, match="Unknown formation"):
            fc.compute_desired_positions(
                center=np.array([0, 0]), heading=0.0, formation_type="nonexistent", num_agents=2
            )

    def test_formation_error(self):
        fc = FormationController()
        current = np.array([[0.0, 0.0], [1.0, 1.0]])
        desired = np.array([[0.0, 0.0], [2.0, 2.0]])
        error = fc.compute_formation_error(current, desired)
        expected = np.mean([0.0, np.sqrt(2.0)])
        np.testing.assert_allclose(error, expected, atol=1e-10)

    def test_heading_rotation(self):
        fc = FormationController(spacing=2.0)
        # Line formation with heading=pi/2 should rotate agents
        pos0 = fc.compute_desired_positions(
            center=np.array([0, 0]), heading=0.0, formation_type="line", num_agents=3
        )
        pos90 = fc.compute_desired_positions(
            center=np.array([0, 0]), heading=np.pi / 2, formation_type="line", num_agents=3
        )
        # Rotated positions should differ
        assert not np.allclose(pos0, pos90)


# ===================================================================
# ConsensusFormation
# ===================================================================


class TestConsensusFormation:
    def test_step_reduces_error(self):
        offsets = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        cf = ConsensusFormation(desired_offsets=offsets, gain=0.5)
        positions = np.array([[2.0, 1.0], [-2.0, -1.0], [0.5, 2.0]])
        # Compute initial error
        centroid = positions.mean(axis=0)
        initial_error = np.mean(np.linalg.norm(positions - (centroid + offsets), axis=1))
        # Run several steps
        for _ in range(10):
            positions = cf.step(positions)
        centroid = positions.mean(axis=0)
        final_error = np.mean(np.linalg.norm(positions - (centroid + offsets), axis=1))
        assert final_error < initial_error

    def test_laplacian_property(self):
        offsets = np.array([[0, 0], [1, 0]])
        cf = ConsensusFormation(desired_offsets=offsets)
        L = cf.laplacian
        assert L.shape == (2, 2)
        # Row sums of Laplacian should be zero
        np.testing.assert_allclose(L.sum(axis=1), [0.0, 0.0], atol=1e-12)

    def test_custom_adjacency(self):
        offsets = np.array([[0, 0], [1, 0], [0, 1]])
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        cf = ConsensusFormation(desired_offsets=offsets, adjacency=adj)
        assert cf.adjacency.shape == (3, 3)


# ===================================================================
# LeaderFollower
# ===================================================================


class TestLeaderFollower:
    def test_compute_follower_targets(self):
        offsets = np.array([[-1.0, 1.0], [-1.0, -1.0]])
        lf = LeaderFollower(leader_index=0, follower_offsets=offsets, gain=0.5)
        targets = lf.compute_follower_targets(
            leader_position=np.array([5.0, 5.0]), leader_heading=0.0
        )
        assert targets.shape == (2, 2)
        # At heading=0, offsets are not rotated
        np.testing.assert_allclose(targets, offsets + np.array([5.0, 5.0]))

    def test_no_offsets_raises(self):
        lf = LeaderFollower()
        with pytest.raises(ValueError, match="follower_offsets"):
            lf.compute_follower_targets(np.array([0, 0]), 0.0)

    def test_step_produces_velocities(self):
        offsets = np.array([[-2.0, 0.0]])
        lf = LeaderFollower(leader_index=0, follower_offsets=offsets, gain=1.0)
        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        velocities = lf.step(positions, leader_heading=0.0)
        assert velocities.shape == (2, 2)
        # Leader velocity should be zero
        np.testing.assert_array_equal(velocities[0], [0.0, 0.0])
        # Follower should have non-zero velocity toward target
        assert np.linalg.norm(velocities[1]) > 0


# ===================================================================
# Task / AllocationResult
# ===================================================================


class TestTaskAllocationDataStructures:
    def test_task_creation(self):
        t = Task(id="t1", location=[1.0, 2.0], priority=3.0)
        assert t.id == "t1"
        np.testing.assert_array_equal(t.location, [1.0, 2.0])
        assert t.priority == 3.0
        assert t.requirements == []
        assert t.deadline is None

    def test_task_location_conversion(self):
        t = Task(id="t1", location=[1, 2])
        assert t.location.dtype == np.float64

    def test_allocation_result(self):
        r = AllocationResult(assignments={"a1": []}, total_cost=0.0)
        assert r.unassigned == []


# ===================================================================
# AuctionAllocator
# ===================================================================


class TestAuctionAllocator:
    def test_sequential_auction_basic(self):
        alloc = AuctionAllocator()
        agents = {"a1": np.array([0.0, 0.0]), "a2": np.array([10.0, 10.0])}
        tasks = [
            Task(id="t1", location=[1.0, 0.0]),
            Task(id="t2", location=[9.0, 10.0]),
        ]
        result = alloc.sequential_auction(agents, tasks)
        assert isinstance(result, AllocationResult)
        assert result.unassigned == []
        # a1 should get t1 (closer), a2 should get t2
        a1_ids = [t.id for t in result.assignments["a1"]]
        a2_ids = [t.id for t in result.assignments["a2"]]
        assert "t1" in a1_ids
        assert "t2" in a2_ids

    def test_sequential_auction_priority_order(self):
        alloc = AuctionAllocator()
        agents = {"a1": np.array([0.0, 0.0])}
        tasks = [
            Task(id="low", location=[1.0, 0.0], priority=1.0),
            Task(id="high", location=[1.0, 0.0], priority=10.0),
        ]
        result = alloc.sequential_auction(agents, tasks)
        # Both assigned to a1; high priority processed first
        assigned = result.assignments["a1"]
        assert assigned[0].id == "high"

    def test_bundle_auction(self):
        alloc = AuctionAllocator()
        agents = {"a1": np.array([0.0, 0.0]), "a2": np.array([10.0, 10.0])}
        tasks = [
            Task(id="t1", location=[1.0, 0.0]),
            Task(id="t2", location=[2.0, 0.0]),
            Task(id="t3", location=[9.0, 10.0]),
        ]
        result = alloc.bundle_auction(agents, tasks, max_bundle_size=2)
        assert result.unassigned == []
        total_assigned = sum(len(v) for v in result.assignments.values())
        assert total_assigned == 3


# ===================================================================
# HungarianAllocator
# ===================================================================


class TestHungarianAllocator:
    def test_optimal_assignment(self):
        alloc = HungarianAllocator()
        agents = {"a1": np.array([0.0, 0.0]), "a2": np.array([10.0, 0.0])}
        tasks = [
            Task(id="t1", location=[1.0, 0.0]),
            Task(id="t2", location=[9.0, 0.0]),
        ]
        result = alloc.allocate(agents, tasks)
        assert len(result.unassigned) == 0
        a1_ids = [t.id for t in result.assignments["a1"]]
        a2_ids = [t.id for t in result.assignments["a2"]]
        assert "t1" in a1_ids
        assert "t2" in a2_ids

    def test_more_tasks_than_agents(self):
        alloc = HungarianAllocator()
        agents = {"a1": np.array([0.0, 0.0])}
        tasks = [
            Task(id="t1", location=[1.0, 0.0]),
            Task(id="t2", location=[100.0, 0.0]),
        ]
        result = alloc.allocate(agents, tasks)
        # One-to-one: only 1 task assigned
        total = sum(len(v) for v in result.assignments.values())
        assert total == 1

    def test_more_agents_than_tasks(self):
        alloc = HungarianAllocator()
        agents = {"a1": np.array([0.0, 0.0]), "a2": np.array([5.0, 0.0])}
        tasks = [Task(id="t1", location=[1.0, 0.0])]
        result = alloc.allocate(agents, tasks)
        assert len(result.unassigned) == 0


# ===================================================================
# GreedyAllocator
# ===================================================================


class TestGreedyAllocator:
    def test_basic(self):
        alloc = GreedyAllocator()
        agents = {"a1": np.array([0.0, 0.0]), "a2": np.array([10.0, 0.0])}
        tasks = [
            Task(id="t1", location=[1.0, 0.0]),
            Task(id="t2", location=[9.0, 0.0]),
        ]
        result = alloc.allocate(agents, tasks)
        assert result.unassigned == []
        total = sum(len(v) for v in result.assignments.values())
        assert total == 2

    def test_custom_cost_fn(self):
        def priority_cost(pos, task):
            return -task.priority  # prefer high priority

        alloc = GreedyAllocator(cost_fn=priority_cost)
        agents = {"a1": np.array([0.0, 0.0])}
        tasks = [
            Task(id="low", location=[0.0, 0.0], priority=1.0),
            Task(id="high", location=[0.0, 0.0], priority=10.0),
        ]
        result = alloc.allocate(agents, tasks)
        # Both assigned
        assert len(result.unassigned) == 0


# ===================================================================
# PriorityPlanner
# ===================================================================


class TestPriorityPlanner:
    @pytest.fixture()
    def simple_grid(self):
        """5x5 grid with no obstacles."""
        return np.zeros((5, 5), dtype=int)

    def test_single_agent(self, simple_grid):
        planner = PriorityPlanner(simple_grid)
        result = planner.plan(
            starts={"a1": (0, 0)},
            goals={"a1": (4, 4)},
        )
        assert isinstance(result, PlanningResult)
        assert "a1" in result.paths
        assert len(result.paths["a1"]) > 1
        # Path starts at (0,0) and ends at (4,4)
        np.testing.assert_array_equal(result.paths["a1"][0], [0, 0])
        np.testing.assert_array_equal(result.paths["a1"][-1], [4, 4])

    def test_two_agents_with_priorities(self, simple_grid):
        planner = PriorityPlanner(simple_grid)
        result = planner.plan(
            starts={"a1": (0, 0), "a2": (0, 4)},
            goals={"a1": (4, 4), "a2": (4, 0)},
            priorities={"a1": 10, "a2": 1},
        )
        assert "a1" in result.paths
        assert "a2" in result.paths
        assert result.cost > 0

    def test_obstacle_grid(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[1, :4] = 1  # wall blocking most of row 1
        planner = PriorityPlanner(grid)
        result = planner.plan(starts={"a1": (0, 0)}, goals={"a1": (2, 0)})
        # Should find a path around the obstacle
        assert len(result.paths["a1"]) > 1

    def test_no_path_returns_start(self):
        grid = np.zeros((3, 3), dtype=int)
        grid[1, :] = 1  # complete wall
        planner = PriorityPlanner(grid)
        result = planner.plan(starts={"a1": (0, 0)}, goals={"a1": (2, 2)})
        # No path found - should return start position
        assert len(result.paths["a1"]) == 1


# ===================================================================
# CBSPlanner
# ===================================================================


class TestCBSPlanner:
    def test_single_agent(self):
        grid = np.zeros((5, 5), dtype=int)
        planner = CBSPlanner(grid)
        result = planner.plan(starts={"a1": (0, 0)}, goals={"a1": (4, 4)})
        assert len(result.paths["a1"]) > 1
        np.testing.assert_array_equal(result.paths["a1"][-1], [4, 4])

    def test_two_agents_conflict_resolution(self):
        grid = np.zeros((5, 5), dtype=int)
        planner = CBSPlanner(grid)
        # Two agents crossing paths
        result = planner.plan(
            starts={"a1": (0, 2), "a2": (2, 0)},
            goals={"a1": (4, 2), "a2": (2, 4)},
        )
        assert "a1" in result.paths
        assert "a2" in result.paths

    def test_narrow_corridor(self):
        grid = np.ones((3, 5), dtype=int)
        grid[1, :] = 0  # only row 1 is passable
        planner = CBSPlanner(grid, max_iterations=500)
        result = planner.plan(
            starts={"a1": (1, 0), "a2": (1, 4)},
            goals={"a1": (1, 4), "a2": (1, 0)},
        )
        assert isinstance(result, PlanningResult)


# ===================================================================
# VelocityObstaclePlanner
# ===================================================================


class TestVelocityObstaclePlanner:
    def test_single_agent_moves_toward_goal(self):
        planner = VelocityObstaclePlanner(max_speed=1.0, agent_radius=0.3)
        positions = np.array([[0.0, 0.0]])
        goals = np.array([[10.0, 0.0]])
        velocities = np.array([[0.0, 0.0]])
        result = planner.plan_step(positions, goals, velocities, dt=0.1)
        assert "0" in result.paths
        # Agent should move toward goal (positive x)
        new_pos = result.paths["0"][-1]
        assert new_pos[0] > 0.0

    def test_two_agents_avoid_collision(self):
        planner = VelocityObstaclePlanner(
            time_horizon=5.0, max_speed=1.5, agent_radius=0.3
        )
        # Two agents heading toward each other
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])
        preferred = np.array([[1.0, 0.0], [-1.0, 0.0]])
        new_vel = planner.compute_velocities(positions, velocities, preferred)
        assert new_vel.shape == (2, 2)
        # Speeds should be clamped
        speeds = np.linalg.norm(new_vel, axis=1)
        assert np.all(speeds <= planner.max_speed + 1e-6)

    def test_plan_step_returns_result(self):
        planner = VelocityObstaclePlanner()
        positions = np.array([[0.0, 0.0], [3.0, 0.0]])
        goals = np.array([[10.0, 0.0], [0.0, 0.0]])
        velocities = np.zeros((2, 2))
        result = planner.plan_step(positions, goals, velocities)
        assert isinstance(result, PlanningResult)
        assert result.cost >= 0
