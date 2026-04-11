"""Tests for navirl.models.behavior_tree — composite, decorator, condition,
and action nodes plus the BehaviorTreeHumanController.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from navirl.core.constants import EPSILON
from navirl.core.types import Action, AgentState
from navirl.models.behavior_tree import (
    ActionNode,
    AvoidCollision,
    BehaviorTree,
    BehaviorTreeHumanController,
    Blackboard,
    Condition,
    FollowGroup,
    GoToGoal,
    Inverter,
    IsNearGoal,
    IsObstacleAhead,
    MaintainDistance,
    Repeater,
    Selector,
    Sequence,
    Status,
    WaitInQueue,
    YieldAtDoorway,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
    agent_id: int = 0,
    max_speed: float = 1.5,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind="human",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=0.3,
        max_speed=max_speed,
    )


def _make_bb(
    agent: AgentState | None = None,
    neighbours: list[AgentState] | None = None,
    goal: tuple[float, float] = (10.0, 0.0),
) -> Blackboard:
    return Blackboard(
        agent=agent or _make_agent(),
        neighbours=neighbours or [],
        goal=goal,
    )


# ===================================================================
# Composite nodes
# ===================================================================


class _AlwaysSuccess(ActionNode):
    def tick(self, bb: Blackboard) -> Status:
        return Status.SUCCESS

    def reset_state(self) -> None:
        pass


class _AlwaysFailure(ActionNode):
    def tick(self, bb: Blackboard) -> Status:
        return Status.FAILURE

    def reset_state(self) -> None:
        pass


class _AlwaysRunning(ActionNode):
    def tick(self, bb: Blackboard) -> Status:
        return Status.RUNNING

    def reset_state(self) -> None:
        pass


class TestSequence:
    def test_all_success(self):
        seq = Sequence([_AlwaysSuccess(), _AlwaysSuccess()])
        assert seq.tick(_make_bb()) == Status.SUCCESS

    def test_first_fails(self):
        seq = Sequence([_AlwaysFailure(), _AlwaysSuccess()])
        assert seq.tick(_make_bb()) == Status.FAILURE

    def test_running_stops_sequence(self):
        seq = Sequence([_AlwaysRunning(), _AlwaysSuccess()])
        assert seq.tick(_make_bb()) == Status.RUNNING

    def test_reset_propagates(self):
        child = Repeater(_AlwaysSuccess(), max_repeats=3)
        child.tick(_make_bb())  # increment count
        seq = Sequence([child])
        seq.reset_state()
        assert child._count == 0


class TestSelector:
    def test_first_succeeds(self):
        sel = Selector([_AlwaysSuccess(), _AlwaysFailure()])
        assert sel.tick(_make_bb()) == Status.SUCCESS

    def test_all_fail(self):
        sel = Selector([_AlwaysFailure(), _AlwaysFailure()])
        assert sel.tick(_make_bb()) == Status.FAILURE

    def test_running_stops(self):
        sel = Selector([_AlwaysRunning(), _AlwaysSuccess()])
        assert sel.tick(_make_bb()) == Status.RUNNING

    def test_fallback_to_second(self):
        sel = Selector([_AlwaysFailure(), _AlwaysSuccess()])
        assert sel.tick(_make_bb()) == Status.SUCCESS


# ===================================================================
# Decorator nodes
# ===================================================================


class TestInverter:
    def test_inverts_success(self):
        inv = Inverter(_AlwaysSuccess())
        assert inv.tick(_make_bb()) == Status.FAILURE

    def test_inverts_failure(self):
        inv = Inverter(_AlwaysFailure())
        assert inv.tick(_make_bb()) == Status.SUCCESS

    def test_passes_running(self):
        inv = Inverter(_AlwaysRunning())
        assert inv.tick(_make_bb()) == Status.RUNNING

    def test_reset(self):
        child = Repeater(_AlwaysSuccess(), max_repeats=5)
        child.tick(_make_bb())
        inv = Inverter(child)
        inv.reset_state()
        assert child._count == 0


class TestRepeater:
    def test_counts_up(self):
        rep = Repeater(_AlwaysSuccess(), max_repeats=3)
        bb = _make_bb()
        assert rep.tick(bb) == Status.RUNNING
        assert rep._count == 1
        assert rep.tick(bb) == Status.RUNNING
        assert rep._count == 2
        assert rep.tick(bb) == Status.SUCCESS
        assert rep._count == 3

    def test_failure_stops(self):
        rep = Repeater(_AlwaysFailure(), max_repeats=5)
        assert rep.tick(_make_bb()) == Status.FAILURE

    def test_reset(self):
        rep = Repeater(_AlwaysSuccess(), max_repeats=5)
        rep.tick(_make_bb())
        rep.tick(_make_bb())
        rep.reset_state()
        assert rep._count == 0

    def test_already_maxed(self):
        rep = Repeater(_AlwaysSuccess(), max_repeats=1)
        rep.tick(_make_bb())
        assert rep.tick(_make_bb()) == Status.SUCCESS


# ===================================================================
# Condition nodes
# ===================================================================


class TestCondition:
    def test_predicate_true(self):
        cond = Condition(predicate=lambda bb: True)
        assert cond.tick(_make_bb()) == Status.SUCCESS

    def test_predicate_false(self):
        cond = Condition(predicate=lambda bb: False)
        assert cond.tick(_make_bb()) == Status.FAILURE

    def test_no_predicate(self):
        cond = Condition()
        assert cond.tick(_make_bb()) == Status.FAILURE


class TestIsNearGoal:
    def test_near(self):
        agent = _make_agent(x=9.8, y=0.0)
        bb = _make_bb(agent=agent, goal=(10.0, 0.0))
        assert IsNearGoal(radius=0.5).tick(bb) == Status.SUCCESS

    def test_far(self):
        agent = _make_agent(x=0.0, y=0.0)
        bb = _make_bb(agent=agent, goal=(10.0, 0.0))
        assert IsNearGoal(radius=0.5).tick(bb) == Status.FAILURE


class TestIsObstacleAhead:
    def test_obstacle_in_cone(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        neighbour = _make_agent(x=1.0, y=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[neighbour])
        assert IsObstacleAhead(distance=2.0).tick(bb) == Status.SUCCESS

    def test_no_obstacle(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        bb = _make_bb(agent=agent, neighbours=[])
        assert IsObstacleAhead().tick(bb) == Status.FAILURE

    def test_stationary_agent(self):
        agent = _make_agent(vx=0.0, vy=0.0)
        bb = _make_bb(agent=agent)
        assert IsObstacleAhead().tick(bb) == Status.FAILURE

    def test_obstacle_behind(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        behind = _make_agent(x=-1.0, y=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[behind])
        assert IsObstacleAhead(distance=2.0).tick(bb) == Status.FAILURE


# ===================================================================
# Action nodes
# ===================================================================


class TestGoToGoal:
    def test_sets_velocity_toward_goal(self):
        agent = _make_agent(x=0.0, y=0.0)
        bb = _make_bb(agent=agent, goal=(10.0, 0.0))
        status = GoToGoal().tick(bb)
        assert status == Status.RUNNING
        assert bb.pref_vx > 0
        assert bb.pref_vy == pytest.approx(0.0, abs=1e-6)

    def test_at_goal(self):
        agent = _make_agent(x=10.0, y=0.0)
        bb = _make_bb(agent=agent, goal=(10.0, 0.0))
        status = GoToGoal().tick(bb)
        assert status == Status.SUCCESS
        assert bb.pref_vx == 0.0
        assert bb.pref_vy == 0.0

    def test_reset(self):
        node = GoToGoal()
        node.reset_state()  # Should not raise


class TestAvoidCollision:
    def test_no_neighbours(self):
        bb = _make_bb(neighbours=[])
        status = AvoidCollision().tick(bb)
        assert status == Status.SUCCESS

    def test_close_neighbour_adds_avoidance(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        neighbour = _make_agent(x=0.5, y=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[neighbour])
        AvoidCollision().tick(bb)
        # Should push away from neighbour (negative x velocity component)
        assert bb.pref_vx < 0

    def test_overlapping_agents(self):
        agent = _make_agent(x=0.0, y=0.0)
        neighbour = _make_agent(x=0.0, y=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[neighbour])
        AvoidCollision().tick(bb)
        # Should still produce some avoidance direction
        assert bb.behavior == "AVOID"


class TestMaintainDistance:
    def test_no_neighbours(self):
        bb = _make_bb()
        assert MaintainDistance().tick(bb) == Status.SUCCESS

    def test_too_close(self):
        agent = _make_agent(x=0.0, y=0.0)
        close = _make_agent(x=0.3, y=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[close])
        MaintainDistance(min_distance=1.0).tick(bb)
        # Should push away (positive x, since agent is at origin and neighbour is right)
        # The force is agent.x - n.x = -0.3, pushing agent left
        assert bb.pref_vx < 0  # pushed away from neighbour

    def test_far_enough(self):
        agent = _make_agent(x=0.0, y=0.0)
        far = _make_agent(x=5.0, y=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[far])
        result = MaintainDistance(min_distance=1.0).tick(bb)
        assert result == Status.SUCCESS
        assert bb.pref_vx == pytest.approx(0.0)


class TestYieldAtDoorway:
    def test_oncoming_agent_yields(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        oncoming = _make_agent(x=1.0, y=0.0, vx=-1.0, vy=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[oncoming])
        bb.pref_vx = 1.0
        bb.pref_vy = 0.0
        status = YieldAtDoorway(yield_distance=2.0).tick(bb)
        assert status == Status.SUCCESS
        assert bb.behavior == "YIELD"
        assert bb.pref_vx < 1.0  # reduced speed

    def test_no_oncoming(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        same_dir = _make_agent(x=1.0, y=0.0, vx=1.0, vy=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[same_dir])
        status = YieldAtDoorway(yield_distance=2.0).tick(bb)
        assert status == Status.FAILURE

    def test_stationary_agent(self):
        agent = _make_agent(vx=0.0, vy=0.0)
        bb = _make_bb(agent=agent)
        assert YieldAtDoorway().tick(bb) == Status.FAILURE


class TestFollowGroup:
    def test_aligned_group(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        n1 = _make_agent(x=1.0, y=0.5, vx=1.0, vy=0.0, agent_id=1)
        n2 = _make_agent(x=1.0, y=-0.5, vx=1.0, vy=0.0, agent_id=2)
        bb = _make_bb(agent=agent, neighbours=[n1, n2])
        status = FollowGroup(group_radius=3.0).tick(bb)
        assert status == Status.SUCCESS
        assert bb.behavior == "GROUP_FOLLOW"

    def test_no_group(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        bb = _make_bb(agent=agent, neighbours=[])
        status = FollowGroup().tick(bb)
        assert status == Status.FAILURE

    def test_opposing_directions(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        opp = _make_agent(x=1.0, y=0.0, vx=-1.0, vy=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[opp])
        status = FollowGroup(group_radius=3.0).tick(bb)
        assert status == Status.FAILURE


class TestWaitInQueue:
    def test_stationary_ahead(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        ahead = _make_agent(x=0.5, y=0.0, vx=0.0, vy=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[ahead])
        status = WaitInQueue(queue_distance=1.0).tick(bb)
        assert status == Status.SUCCESS
        assert bb.pref_vx == 0.0
        assert bb.behavior == "QUEUE"

    def test_nobody_ahead(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        bb = _make_bb(agent=agent, neighbours=[])
        assert WaitInQueue().tick(bb) == Status.FAILURE

    def test_agent_ahead_moving(self):
        agent = _make_agent(x=0.0, y=0.0, vx=1.0, vy=0.0)
        ahead = _make_agent(x=0.5, y=0.0, vx=1.0, vy=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[ahead])
        # Moving agent ahead → don't wait
        assert WaitInQueue(queue_distance=1.0).tick(bb) == Status.FAILURE

    def test_stationary_agent_looks_toward_goal(self):
        agent = _make_agent(x=0.0, y=0.0, vx=0.0, vy=0.0)
        ahead = _make_agent(x=5.0, y=0.0, vx=0.0, vy=0.0, agent_id=1)
        bb = _make_bb(agent=agent, neighbours=[ahead], goal=(10.0, 0.0))
        # Neighbour too far
        assert WaitInQueue(queue_distance=1.0).tick(bb) == Status.FAILURE


# ===================================================================
# BehaviorTree wrapper
# ===================================================================


class TestBehaviorTree:
    def test_default_tree(self):
        tree = BehaviorTree.default_pedestrian_tree()
        agent = _make_agent(x=0.0, y=0.0, vx=0.5, vy=0.0)
        bb = _make_bb(agent=agent, goal=(10.0, 0.0))
        status = tree.tick(bb)
        assert isinstance(status, Status)

    def test_reset(self):
        tree = BehaviorTree.default_pedestrian_tree()
        tree.reset()  # Should not raise


# ===================================================================
# BehaviorTreeHumanController
# ===================================================================


class TestBehaviorTreeHumanController:
    def test_reset_and_step(self):
        ctrl = BehaviorTreeHumanController()
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (0.0, 5.0)},
            goals={1: (10.0, 0.0), 2: (10.0, 5.0)},
        )

        states = {
            0: _make_agent(agent_id=0, x=5.0, y=2.5),  # robot
            1: _make_agent(agent_id=1, x=0.0, y=0.0),
            2: _make_agent(agent_id=2, x=0.0, y=5.0),
        }
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.04, states=states, robot_id=0, emit_event=emit)

        assert 1 in actions
        assert 2 in actions
        assert isinstance(actions[1], Action)
        assert isinstance(actions[2], Action)

    def test_goal_swap_on_arrival(self):
        ctrl = BehaviorTreeHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )

        # Place agent at goal
        states = {
            0: _make_agent(agent_id=0, x=5.0, y=0.0),
            1: _make_agent(agent_id=1, x=10.0, y=0.0),
        }
        emit = MagicMock()
        ctrl.step(step=0, time_s=0.0, dt=0.04, states=states, robot_id=0, emit_event=emit)

        # Goal should have swapped
        assert ctrl.goals[1] == (0.0, 0.0)
        assert ctrl.starts[1] == (10.0, 0.0)
        emit.assert_called_once()

    def test_speed_clamping(self):
        ctrl = BehaviorTreeHumanController()
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (100.0, 0.0)},
        )

        states = {
            0: _make_agent(agent_id=0, x=50.0, y=0.0),
            1: _make_agent(agent_id=1, x=0.0, y=0.0, max_speed=1.0),
        }
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.04, states=states, robot_id=0, emit_event=emit)

        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        assert speed <= 1.0 + 1e-6
