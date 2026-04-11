from __future__ import annotations

"""Tests covering routine module coverage gaps.

Targets uncovered lines in compiler.py, behavior_integration.py, and schema.py
including: task type handlers, condition nodes, temporal constraints, looping,
repetition, fallback behaviors, event emission, factory methods, and YAML
serialization edge cases.
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from navirl.core.types import Action, AgentState
from navirl.models.behavior_tree import Blackboard, Selector, Sequence, Status
from navirl.routines.behavior_integration import (
    CompiledRoutineController,
    RoutineControllerFactory,
    _validate_file_path,
)
from navirl.routines.compiler import (
    AgentNearbyCondition,
    AvoidArea,
    DelayedStartDecorator,
    GoToTarget,
    InteractAtLocation,
    LocationReachedCondition,
    LoopingSequence,
    ProbabilisticCondition,
    RepeatingSequence,
    RoutineCompiler,
    TimeElapsedCondition,
    TimeoutDecorator,
    WaitForDuration,
)
from navirl.routines.schema import (
    Branch,
    Condition,
    ConditionType,
    RoutineSpec,
    Task,
    TaskType,
    TemporalConstraint,
    validate_routine_spec,
)


def _make_agent(agent_id=1, x=0.0, y=0.0, **kw):
    defaults = dict(
        kind="human", vx=0.0, vy=0.0, goal_x=5.0, goal_y=5.0, radius=0.2, max_speed=1.0
    )
    defaults.update(kw)
    return AgentState(agent_id=agent_id, x=x, y=y, **defaults)


def _make_bb(agent=None, **kw):
    if agent is None:
        agent = _make_agent()
    defaults = dict(dt=0.1, metadata={"sim_time": 0.0})
    defaults.update(kw)
    return Blackboard(agent=agent, **defaults)


# ---------------------------------------------------------------------------
# Compiler: task type handlers
# ---------------------------------------------------------------------------


class TestCompilerTaskTypes:
    """Tests for all task type compilation paths."""

    def setup_method(self):
        self.compiler = RoutineCompiler()

    def test_compile_interact_task(self):
        """Test INTERACT task compiles to InteractAtLocation."""
        spec = RoutineSpec(
            id="interact",
            description="Interact",
            tasks=[Task(type=TaskType.INTERACT, params={"location": (1.0, 2.0), "interaction_type": "vending"})],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, InteractAtLocation)
        assert plan.root_node.interaction_type == "vending"

    def test_compile_follow_task(self):
        """Test FOLLOW task compilation."""
        from navirl.models.behavior_tree import FollowGroup

        spec = RoutineSpec(
            id="follow",
            description="Follow",
            tasks=[Task(type=TaskType.FOLLOW, params={"group_radius": 4.0})],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, FollowGroup)

    def test_compile_avoid_task(self):
        """Test AVOID task compiles to AvoidArea."""
        spec = RoutineSpec(
            id="avoid",
            description="Avoid",
            tasks=[Task(type=TaskType.AVOID, params={"location": (3.0, 3.0), "radius": 2.0})],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, AvoidArea)

    def test_compile_queue_task(self):
        """Test QUEUE task compilation."""
        from navirl.models.behavior_tree import WaitInQueue

        spec = RoutineSpec(
            id="queue",
            description="Queue",
            tasks=[Task(type=TaskType.QUEUE, params={"queue_distance": 1.0})],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, WaitInQueue)

    def test_compile_group_task(self):
        """Test GROUP task compilation."""
        from navirl.models.behavior_tree import FollowGroup

        spec = RoutineSpec(
            id="group",
            description="Group",
            tasks=[Task(type=TaskType.GROUP, params={"max_separation": 3.0})],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, FollowGroup)

    def test_compile_custom_task(self):
        """Test CUSTOM task with registered handler."""
        handler_called_with = []

        def handler(task):
            handler_called_with.append(task)
            return GoToTarget(99.0, 99.0)

        self.compiler.register_custom_task_handler("my_handler", handler)
        spec = RoutineSpec(
            id="custom",
            description="Custom",
            tasks=[Task(type=TaskType.CUSTOM, params={"handler": "my_handler"})],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, GoToTarget)
        assert len(handler_called_with) == 1

    def test_compile_custom_task_missing_handler_raises(self):
        """CUSTOM task without registered handler raises ValueError."""
        spec = RoutineSpec(
            id="bad_custom",
            description="Bad custom",
            tasks=[Task(type=TaskType.CUSTOM, params={"handler": "nonexistent"})],
        )
        with pytest.raises(ValueError, match="must have a registered handler"):
            self.compiler.compile(spec)

    def test_compile_wait_without_duration_raises(self):
        """WAIT task without duration raises ValueError."""
        spec = RoutineSpec(
            id="bad_wait",
            description="Bad wait",
            tasks=[Task(type=TaskType.WAIT, params={})],
        )
        with pytest.raises(ValueError, match="must have 'duration'"):
            self.compiler.compile(spec)

    def test_empty_task_list_raises(self):
        """Empty tasks in _compile_tasks raises ValueError."""
        spec = RoutineSpec(id="empty", description="Empty", tasks=[])
        with pytest.raises(ValueError, match="must have at least one task"):
            self.compiler.compile(spec)

    def test_branch_with_empty_tasks_raises(self):
        """Branch with empty task list raises ValueError."""
        spec = RoutineSpec(
            id="bad_branch",
            description="Bad branch",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[Branch(condition=Condition.time_elapsed(5.0), tasks=[])],
        )
        with pytest.raises(ValueError, match="must have at least one task"):
            self.compiler.compile(spec)


# ---------------------------------------------------------------------------
# Compiler: condition compilation
# ---------------------------------------------------------------------------


class TestCompilerConditions:
    """Tests for condition compilation paths."""

    def setup_method(self):
        self.compiler = RoutineCompiler()

    def test_compile_time_elapsed_condition(self):
        """TIME_ELAPSED condition compiles correctly."""
        spec = RoutineSpec(
            id="time_branch",
            description="Time branch",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[
                Branch(
                    condition=Condition.time_elapsed(10.0),
                    tasks=[Task.go_to(5.0, 5.0)],
                )
            ],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, Selector)

    def test_compile_location_reached_condition(self):
        """LOCATION_REACHED condition compiles correctly."""
        spec = RoutineSpec(
            id="loc_branch",
            description="Location branch",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[
                Branch(
                    condition=Condition.location_reached(2.0, 3.0, 0.5),
                    tasks=[Task.go_to(5.0, 5.0)],
                )
            ],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, Selector)

    def test_compile_agent_nearby_condition(self):
        """AGENT_NEARBY condition compiles correctly."""
        spec = RoutineSpec(
            id="agent_branch",
            description="Agent branch",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[
                Branch(
                    condition=Condition.agent_nearby(42, 2.0),
                    tasks=[Task.go_to(5.0, 5.0)],
                )
            ],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, Selector)

    def test_compile_custom_condition(self):
        """CUSTOM condition with registered handler compiles correctly."""

        def handler(condition):
            return lambda bb: True

        self.compiler.register_custom_condition_handler("my_cond", handler)
        spec = RoutineSpec(
            id="custom_cond",
            description="Custom condition",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[
                Branch(
                    condition=Condition(ConditionType.CUSTOM, {"handler": "my_cond"}),
                    tasks=[Task.go_to(5.0, 5.0)],
                )
            ],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, Selector)

    def test_compile_probabilistic_branch(self):
        """Branch with probability < 1.0 wraps in ProbabilisticCondition."""
        spec = RoutineSpec(
            id="prob_branch",
            description="Probabilistic branch",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[
                Branch(
                    condition=Condition.time_elapsed(5.0),
                    tasks=[Task.go_to(5.0, 5.0)],
                    probability=0.5,
                )
            ],
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, Selector)


# ---------------------------------------------------------------------------
# Compiler: temporal constraints
# ---------------------------------------------------------------------------


class TestTemporalConstraints:
    """Tests for temporal constraint decorators."""

    def setup_method(self):
        self.compiler = RoutineCompiler()

    def test_compile_with_max_duration(self):
        """Temporal constraint with max_duration wraps in TimeoutDecorator."""
        spec = RoutineSpec(
            id="timeout",
            description="Timeout",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(max_duration=30.0),
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, TimeoutDecorator)

    def test_compile_with_start_time(self):
        """Temporal constraint with start_time wraps in DelayedStartDecorator."""
        spec = RoutineSpec(
            id="delayed",
            description="Delayed start",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(start_time=10.0),
        )
        plan = self.compiler.compile(spec)
        assert isinstance(plan.root_node, DelayedStartDecorator)

    def test_compile_with_both_constraints(self):
        """Both start_time and max_duration are applied."""
        spec = RoutineSpec(
            id="both",
            description="Both constraints",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(start_time=5.0, max_duration=30.0),
        )
        plan = self.compiler.compile(spec)
        # Outer node is DelayedStart (applied second), inner is Timeout
        assert isinstance(plan.root_node, DelayedStartDecorator)
        assert isinstance(plan.root_node.child, TimeoutDecorator)

    def test_timeout_decorator_tick(self):
        """TimeoutDecorator returns FAILURE when time exceeded."""
        child = GoToTarget(5.0, 5.0)
        decorator = TimeoutDecorator(child, max_duration=10.0)
        bb = _make_bb()

        # First tick, within time
        status = decorator.tick(bb)
        assert status == Status.RUNNING

        # After timeout
        bb.metadata["sim_time"] = 15.0
        status = decorator.tick(bb)
        assert status == Status.FAILURE

    def test_timeout_decorator_reset(self):
        """TimeoutDecorator resets start time and child."""
        child = WaitForDuration(5.0)
        decorator = TimeoutDecorator(child, max_duration=10.0)
        bb = _make_bb()
        decorator.tick(bb)
        assert decorator._start_time is not None

        decorator.reset_state()
        assert decorator._start_time is None

    def test_delayed_start_decorator_tick(self):
        """DelayedStartDecorator returns RUNNING before start_time."""
        child = GoToTarget(5.0, 5.0)
        decorator = DelayedStartDecorator(child, start_time=10.0)
        bb = _make_bb()

        # Before start time
        bb.metadata["sim_time"] = 5.0
        status = decorator.tick(bb)
        assert status == Status.RUNNING

        # After start time
        bb.metadata["sim_time"] = 15.0
        status = decorator.tick(bb)
        assert status == Status.RUNNING  # child runs, agent not at target

    def test_delayed_start_decorator_reset(self):
        """DelayedStartDecorator resets child."""
        child = WaitForDuration(5.0)
        decorator = DelayedStartDecorator(child, start_time=10.0)
        bb = _make_bb(metadata={"sim_time": 15.0})
        decorator.tick(bb)

        decorator.reset_state()
        assert child._start_time is None


# ---------------------------------------------------------------------------
# Compiler: repeating and looping sequences
# ---------------------------------------------------------------------------


class TestRepeatingAndLooping:
    """Tests for RepeatingSequence and LoopingSequence nodes."""

    def test_repeating_sequence_counts(self):
        """RepeatingSequence repeats child N times."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = RepeatingSequence(child, repetitions=3)
        bb = _make_bb()

        # First success -> repetition 1, returns RUNNING (more to go)
        assert node.tick(bb) == Status.RUNNING
        assert node._current_repetition == 1

        # Second success -> repetition 2
        assert node.tick(bb) == Status.RUNNING
        assert node._current_repetition == 2

        # Third success -> all done
        assert node.tick(bb) == Status.SUCCESS
        assert node._current_repetition == 3

    def test_repeating_sequence_running_child(self):
        """RepeatingSequence returns RUNNING when child is RUNNING."""
        child = Mock()
        child.tick = Mock(return_value=Status.RUNNING)

        node = RepeatingSequence(child, repetitions=3)
        bb = _make_bb()

        assert node.tick(bb) == Status.RUNNING
        assert node._current_repetition == 0

    def test_repeating_sequence_failure(self):
        """RepeatingSequence returns FAILURE when child fails."""
        child = Mock()
        child.tick = Mock(return_value=Status.FAILURE)

        node = RepeatingSequence(child, repetitions=3)
        bb = _make_bb()

        assert node.tick(bb) == Status.FAILURE

    def test_repeating_sequence_reset(self):
        """RepeatingSequence reset clears counter."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = RepeatingSequence(child, repetitions=2)
        bb = _make_bb()
        node.tick(bb)

        node.reset_state()
        assert node._current_repetition == 0
        child.reset_state.assert_called()

    def test_repeating_sequence_already_done(self):
        """RepeatingSequence returns SUCCESS if already completed."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = RepeatingSequence(child, repetitions=1)
        bb = _make_bb()

        assert node.tick(bb) == Status.SUCCESS
        # Second tick after completion
        assert node.tick(bb) == Status.SUCCESS

    def test_looping_sequence_infinite(self):
        """LoopingSequence loops indefinitely with max_loops=-1."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = LoopingSequence(child, max_loops=-1)
        bb = _make_bb()

        for _ in range(10):
            assert node.tick(bb) == Status.RUNNING

        assert node._current_loop == 10

    def test_looping_sequence_max_loops(self):
        """LoopingSequence finishes after max_loops."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = LoopingSequence(child, max_loops=2)
        bb = _make_bb()

        # First loop completion
        assert node.tick(bb) == Status.RUNNING
        # Second loop completion
        assert node.tick(bb) == Status.SUCCESS

    def test_looping_sequence_reset(self):
        """LoopingSequence reset clears loop counter."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = LoopingSequence(child, max_loops=5)
        bb = _make_bb()
        node.tick(bb)

        node.reset_state()
        assert node._current_loop == 0
        child.reset_state.assert_called()

    def test_looping_sequence_already_done(self):
        """LoopingSequence returns SUCCESS if max_loops reached."""
        child = Mock()
        child.tick = Mock(return_value=Status.SUCCESS)
        child.reset_state = Mock()

        node = LoopingSequence(child, max_loops=1)
        bb = _make_bb()

        # Complete the one loop
        node.tick(bb)
        # After completion, should return SUCCESS
        assert node.tick(bb) == Status.SUCCESS


# ---------------------------------------------------------------------------
# Condition nodes: reset and edge cases
# ---------------------------------------------------------------------------


class TestConditionNodeEdgeCases:
    """Tests for condition node reset methods and edge cases."""

    def test_time_elapsed_reset(self):
        """TimeElapsedCondition reset clears start time."""
        node = TimeElapsedCondition(5.0)
        bb = _make_bb()
        node.tick(bb)
        assert node._start_time is not None

        node.reset_state()
        assert node._start_time is None

    def test_location_reached_reset(self):
        """LocationReachedCondition reset is a no-op."""
        node = LocationReachedCondition(1.0, 2.0)
        node.reset_state()  # should not raise

    def test_agent_nearby_reset(self):
        """AgentNearbyCondition reset is a no-op."""
        node = AgentNearbyCondition(agent_id=1, distance=2.0)
        node.reset_state()  # should not raise

    def test_agent_nearby_any_agent(self):
        """AgentNearbyCondition with agent_id=None matches any agent."""
        node = AgentNearbyCondition(agent_id=None, distance=5.0)
        neighbor = _make_agent(agent_id=99, x=1.0, y=1.0)
        bb = _make_bb(neighbours=[neighbor])
        assert node.tick(bb) == Status.SUCCESS

    def test_agent_nearby_wrong_id(self):
        """AgentNearbyCondition fails when agent_id doesn't match."""
        node = AgentNearbyCondition(agent_id=42, distance=5.0)
        neighbor = _make_agent(agent_id=99, x=1.0, y=1.0)
        bb = _make_bb(neighbours=[neighbor])
        assert node.tick(bb) == Status.FAILURE

    def test_probabilistic_condition_always_succeeds(self):
        """ProbabilisticCondition with probability=1.0 delegates to wrapped."""
        wrapped = TimeElapsedCondition(0.0)  # always succeeds at time 0
        node = ProbabilisticCondition(wrapped, probability=1.0)
        bb = _make_bb(metadata={"sim_time": 1.0})
        # With probability 1.0 it should always evaluate the wrapped condition
        status = node.tick(bb)
        assert status == Status.SUCCESS

    def test_probabilistic_condition_caches_result(self):
        """ProbabilisticCondition evaluates only once per reset."""
        wrapped = Mock()
        wrapped.tick = Mock(return_value=Status.SUCCESS)

        node = ProbabilisticCondition(wrapped, probability=1.0)
        bb = _make_bb()

        node.tick(bb)
        node.tick(bb)  # Second call should use cached result
        wrapped.tick.assert_called_once()

    def test_probabilistic_condition_reset(self):
        """ProbabilisticCondition reset clears evaluation state."""
        wrapped = Mock()
        wrapped.tick = Mock(return_value=Status.SUCCESS)
        wrapped.reset_state = Mock()

        node = ProbabilisticCondition(wrapped, probability=1.0)
        bb = _make_bb()
        node.tick(bb)

        node.reset_state()
        assert node._evaluated is False
        assert node._result == Status.FAILURE
        wrapped.reset_state.assert_called_once()


# ---------------------------------------------------------------------------
# AvoidArea edge cases
# ---------------------------------------------------------------------------


class TestAvoidAreaEdgeCases:
    """Tests for AvoidArea edge cases."""

    def test_avoid_agent_not_found(self):
        """AvoidArea returns SUCCESS when target agent is not in neighbors."""
        node = AvoidArea({"agent_id": 42})
        bb = _make_bb(neighbours=[])
        assert node.tick(bb) == Status.SUCCESS

    def test_avoid_area_far_away(self):
        """AvoidArea returns SUCCESS when agent is far from avoid area."""
        node = AvoidArea({"location": (100.0, 100.0), "radius": 2.0})
        bb = _make_bb()
        assert node.tick(bb) == Status.SUCCESS

    def test_avoid_area_very_close(self):
        """AvoidArea picks random direction when too close (distance < 0.1)."""
        node = AvoidArea({"location": (0.0, 0.0), "radius": 5.0})
        agent = _make_agent(x=0.0, y=0.0)
        bb = _make_bb(agent=agent)

        status = node.tick(bb)
        assert status == Status.RUNNING
        # Should have set some velocity
        speed = math.hypot(bb.pref_vx, bb.pref_vy)
        assert speed > 0

    def test_avoid_by_agent_id(self):
        """AvoidArea avoids a specific agent by ID."""
        node = AvoidArea({"agent_id": 42, "radius": 5.0})
        neighbor = _make_agent(agent_id=42, x=0.5, y=0.0)
        agent = _make_agent(x=0.0, y=0.0)
        bb = _make_bb(agent=agent, neighbours=[neighbor])

        status = node.tick(bb)
        assert status == Status.RUNNING
        assert bb.behavior == "AVOID"

    def test_avoid_area_reset(self):
        """AvoidArea reset is a no-op (stateless node)."""
        node = AvoidArea({"location": (1.0, 1.0)})
        node.reset_state()  # should not raise


# ---------------------------------------------------------------------------
# GoToTarget and WaitForDuration resets
# ---------------------------------------------------------------------------


class TestNodeResets:
    """Tests for node reset methods."""

    def test_goto_target_reset(self):
        """GoToTarget reset resets internal GoToGoal."""
        node = GoToTarget(5.0, 5.0)
        bb = _make_bb()
        node.tick(bb)
        node.reset_state()  # should not raise

    def test_wait_for_duration_reset(self):
        """WaitForDuration reset clears start time."""
        node = WaitForDuration(5.0)
        bb = _make_bb()
        node.tick(bb)
        assert node._start_time is not None

        node.reset_state()
        assert node._start_time is None

    def test_interact_at_location_reset(self):
        """InteractAtLocation reset clears interaction state."""
        node = InteractAtLocation((1.0, 1.0), "test")
        agent = _make_agent(x=1.0, y=1.0)
        bb = _make_bb(agent=agent)
        node.tick(bb)
        assert node._interaction_started is True

        node.reset_state()
        assert node._interaction_started is False


# ---------------------------------------------------------------------------
# behavior_integration.py: event emission and fallback
# ---------------------------------------------------------------------------


class TestControllerEventEmission:
    """Tests for event emission during routine execution."""

    def setup_method(self):
        self.agent = _make_agent(agent_id=1, x=5.0, y=3.0)  # at target for GoToTarget
        self.robot = _make_agent(agent_id=100, x=10.0, y=10.0, kind="robot")
        self.states = {1: self.agent, 100: self.robot}

    def test_routine_completed_event(self):
        """Controller emits routine_completed when routine finishes."""
        # Use a wait task with 0 duration so it completes immediately via sim_time
        spec = RoutineSpec(
            id="quick_wait",
            description="Quick wait",
            tasks=[Task.wait(0.0)],
        )
        controller = CompiledRoutineController({1: spec})
        controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 3.0)}, backend=None
        )

        emit = Mock()
        # At time_s=1.0, the WaitForDuration with duration 0.0 should complete
        controller.step(step=0, time_s=1.0, dt=0.1, states=self.states, robot_id=100, emit_event=emit)

        # Check routine_completed event was emitted
        event_names = [call[0][0] for call in emit.call_args_list]
        assert "routine_completed" in event_names

    def test_routine_error_event_and_fallback(self):
        """Controller emits routine_error and falls back on exception."""
        spec = RoutineSpec(
            id="error_routine",
            description="Error routine",
            tasks=[Task.go_to(1.0, 1.0)],
        )
        controller = CompiledRoutineController({1: spec})
        controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 3.0)}, backend=None
        )

        # Sabotage the behavior plan to cause an error
        plan = controller.compiled_plans[1]
        plan.root_node = Mock()
        plan.root_node.tick = Mock(side_effect=RuntimeError("test error"))

        emit = Mock()
        actions = controller.step(
            step=0, time_s=1.0, dt=0.1, states=self.states, robot_id=100, emit_event=emit
        )

        event_names = [call[0][0] for call in emit.call_args_list]
        assert "routine_error" in event_names
        # Should have returned a fallback action
        assert actions[1].metadata.get("fallback") is True


class TestFallbackBehaviors:
    """Tests for fallback behavior modes."""

    def setup_method(self):
        self.agent = _make_agent(agent_id=1)
        self.robot = _make_agent(agent_id=100, x=10.0, y=10.0, kind="robot")
        self.states = {1: self.agent, 100: self.robot}

    def test_static_fallback(self):
        """Static fallback keeps agent in place."""
        controller = CompiledRoutineController({}, fallback_behavior="static")
        controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 5.0)}, backend=None
        )

        emit = Mock()
        actions = controller.step(
            step=0, time_s=0.0, dt=0.1, states=self.states, robot_id=100, emit_event=emit
        )

        assert actions[1].pref_vx == 0.0
        assert actions[1].pref_vy == 0.0
        assert actions[1].behavior == "STATIC"

    def test_goal_swap_fallback(self):
        """Goal swap fallback moves agent toward goal."""
        controller = CompiledRoutineController({}, fallback_behavior="goal_swap")
        controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 5.0)}, backend=None
        )

        emit = Mock()
        actions = controller.step(
            step=0, time_s=0.0, dt=0.1, states=self.states, robot_id=100, emit_event=emit
        )

        assert actions[1].pref_vx > 0
        assert actions[1].pref_vy > 0
        assert actions[1].metadata.get("fallback") is True

    def test_goal_swap_at_goal_swaps(self):
        """Goal swap switches start and goal when agent reaches goal."""
        agent_at_goal = _make_agent(agent_id=1, x=5.0, y=5.0)
        states = {1: agent_at_goal, 100: self.robot}

        controller = CompiledRoutineController({}, fallback_behavior="goal_swap")
        controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 5.0)}, backend=None
        )

        emit = Mock()
        controller.step(
            step=0, time_s=0.0, dt=0.1, states=states, robot_id=100, emit_event=emit
        )

        # Should have emitted goal_swap event
        event_names = [call[0][0] for call in emit.call_args_list]
        assert "goal_swap" in event_names

        # Goals and starts should be swapped
        assert controller.goals[1] == (0.0, 0.0)
        assert controller.starts[1] == (5.0, 5.0)

    def test_goal_swap_agent_at_exact_goal(self):
        """Goal swap handles agent exactly at goal position (distance=0)."""
        agent_at_goal = _make_agent(agent_id=1, x=0.0, y=0.0)
        states = {1: agent_at_goal, 100: self.robot}

        controller = CompiledRoutineController({}, fallback_behavior="goal_swap")
        controller.reset(
            human_ids=[1],
            starts={1: (5.0, 5.0)},
            goals={1: (0.0, 0.0)},
            backend=None,
        )

        emit = Mock()
        actions = controller.step(
            step=0, time_s=0.0, dt=0.1, states=states, robot_id=100, emit_event=emit
        )

        # After swap, agent should now move toward the new goal
        assert isinstance(actions[1], Action)

    def test_unknown_fallback_raises(self):
        """Unknown fallback behavior raises ValueError."""
        controller = CompiledRoutineController({}, fallback_behavior="unknown")
        controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 5.0)}, backend=None
        )

        emit = Mock()
        with pytest.raises(ValueError, match="Unknown fallback behavior"):
            controller.step(
                step=0, time_s=0.0, dt=0.1, states=self.states, robot_id=100, emit_event=emit
            )


# ---------------------------------------------------------------------------
# behavior_integration.py: add_routine compile failure
# ---------------------------------------------------------------------------


class TestControllerRoutineManagement:
    """Tests for add/remove routine edge cases."""

    def test_add_routine_compile_failure(self):
        """add_routine handles compile failure gracefully."""
        controller = CompiledRoutineController({})
        # Create an invalid spec (empty tasks) - this should fail during compilation
        bad_spec = RoutineSpec(id="bad", description="Bad", tasks=[])

        # Should not raise, just print warning
        controller.add_routine(1, bad_spec)
        assert 1 not in controller.compiled_plans

    def test_compile_routines_failure(self):
        """_compile_routines handles individual routine compile failures."""
        bad_spec = RoutineSpec(id="bad", description="Bad", tasks=[])
        good_spec = RoutineSpec(
            id="good", description="Good", tasks=[Task.go_to(1.0, 1.0)]
        )

        controller = CompiledRoutineController({1: bad_spec, 2: good_spec})
        # Good spec should compile, bad should be skipped
        assert 1 not in controller.compiled_plans
        assert 2 in controller.compiled_plans


# ---------------------------------------------------------------------------
# behavior_integration.py: factory methods
# ---------------------------------------------------------------------------


class TestRoutineControllerFactoryEdgeCases:
    """Tests for factory edge cases."""

    def test_from_yaml_files_valid(self):
        """from_yaml_files loads valid YAML routine files."""
        yaml_content = """
id: test_routine
description: Test routine
tasks:
  - type: go_to
    params:
      x: 1.0
      y: 2.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            controller = RoutineControllerFactory.from_yaml_files({1: path})
            assert 1 in controller.routines
        finally:
            Path(path).unlink()

    def test_from_yaml_files_invalid_path(self):
        """from_yaml_files handles invalid file paths gracefully."""
        controller = RoutineControllerFactory.from_yaml_files({1: "/nonexistent/file.yaml"})
        assert 1 not in controller.routines

    def test_from_scenario_config_invalid_routine(self):
        """from_scenario_config parses routine but compile fails gracefully."""
        config = {
            "routines": {
                "1": {"id": "bad", "description": "Missing tasks field"},
            }
        }
        controller = RoutineControllerFactory.from_scenario_config(config)
        # The routine is parsed into routines dict, but compilation fails
        assert 1 in controller.routines
        assert 1 not in controller.compiled_plans

    def test_create_simple_routines_empty_positions(self):
        """create_simple_routines skips agents with empty positions."""
        controller = RoutineControllerFactory.create_simple_routines({1: [], 2: [(1.0, 1.0)]})
        assert 1 not in controller.routines
        assert 2 in controller.routines


# ---------------------------------------------------------------------------
# behavior_integration.py: path validation
# ---------------------------------------------------------------------------


class TestPathValidation:
    """Tests for _validate_file_path security function."""

    def test_valid_yaml_path(self):
        """Valid YAML file path passes validation."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name

        try:
            result = _validate_file_path(path)
            assert result == Path(path).resolve()
        finally:
            Path(path).unlink()

    def test_nonexistent_path_raises(self):
        """Nonexistent file raises ValueError."""
        with pytest.raises(ValueError, match="File does not exist"):
            _validate_file_path("/nonexistent/file.yaml")

    def test_directory_path_raises(self):
        """Directory path raises ValueError."""
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="not a file"):
                _validate_file_path(d)

    def test_path_traversal_raises(self):
        """Path traversal attempt raises ValueError."""
        with pytest.raises(ValueError, match="Invalid file path"):
            _validate_file_path("../../../etc/passwd")

    def test_non_yaml_raises(self):
        """Non-YAML file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Only YAML files"):
                _validate_file_path(path)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# schema.py: completion conditions and YAML edge cases
# ---------------------------------------------------------------------------


class TestSchemaEdgeCases:
    """Tests for schema serialization and completion conditions."""

    def test_task_with_completion_condition_roundtrip(self):
        """Task with completion_condition survives YAML roundtrip."""
        spec = RoutineSpec(
            id="cc_test",
            description="Completion condition test",
            tasks=[
                Task(
                    type=TaskType.GO_TO,
                    params={"x": 1.0, "y": 2.0},
                    completion_condition=Condition.location_reached(1.0, 2.0, 0.3),
                )
            ],
        )

        yaml_str = spec.to_yaml()
        parsed = RoutineSpec.from_yaml(yaml_str)

        assert parsed.tasks[0].completion_condition is not None
        assert parsed.tasks[0].completion_condition.type == ConditionType.LOCATION_REACHED
        assert parsed.tasks[0].completion_condition.params["x"] == 1.0

    def test_to_dict_with_all_temporal_constraints(self):
        """to_dict serializes all temporal constraint fields."""
        spec = RoutineSpec(
            id="tc_test",
            description="TC test",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(
                start_time=0.0, end_time=100.0, max_duration=50.0, min_duration=5.0
            ),
        )

        d = spec.to_dict()
        tc = d["temporal_constraints"]
        assert tc["start_time"] == 0.0
        assert tc["end_time"] == 100.0
        assert tc["max_duration"] == 50.0
        assert tc["min_duration"] == 5.0

    def test_to_dict_without_temporal_constraints(self):
        """to_dict omits temporal_constraints when None."""
        spec = RoutineSpec(
            id="no_tc",
            description="No TC",
            tasks=[Task.go_to(1.0, 1.0)],
        )

        d = spec.to_dict()
        assert "temporal_constraints" not in d

    def test_to_dict_with_branches(self):
        """to_dict serializes branches correctly."""
        spec = RoutineSpec(
            id="branch_test",
            description="Branch test",
            tasks=[Task.go_to(1.0, 1.0)],
            branches=[
                Branch(
                    condition=Condition.time_elapsed(10.0),
                    tasks=[Task.wait(5.0)],
                    probability=0.8,
                )
            ],
        )

        d = spec.to_dict()
        assert "branches" in d
        assert len(d["branches"]) == 1
        assert d["branches"][0]["probability"] == 0.8
        assert d["branches"][0]["condition"]["type"] == "time_elapsed"

    def test_from_dict_with_completion_condition(self):
        """from_dict parses tasks with completion conditions."""
        data = {
            "id": "cc_from_dict",
            "description": "CC from dict",
            "tasks": [
                {
                    "type": "go_to",
                    "params": {"x": 1.0, "y": 2.0},
                    "completion_condition": {
                        "type": "location_reached",
                        "params": {"x": 1.0, "y": 2.0, "radius": 0.5},
                    },
                }
            ],
        }

        spec = RoutineSpec.from_dict(data)
        assert spec.tasks[0].completion_condition is not None
        assert spec.tasks[0].completion_condition.type == ConditionType.LOCATION_REACHED

    def test_to_dict_completion_condition_serialization(self):
        """to_dict serializes task completion conditions."""
        spec = RoutineSpec(
            id="cc_serial",
            description="CC serialization",
            tasks=[
                Task(
                    type=TaskType.WAIT,
                    params={"duration": 5.0},
                    duration=5.0,
                    completion_condition=Condition.time_elapsed(5.0),
                )
            ],
        )

        d = spec.to_dict()
        task = d["tasks"][0]
        assert "completion_condition" in task
        assert task["completion_condition"]["type"] == "time_elapsed"

    def test_convert_for_yaml_nested(self):
        """_convert_for_yaml handles nested tuples and dicts."""
        spec = RoutineSpec(
            id="nested",
            description="Nested",
            tasks=[Task.go_to(1.0, 1.0)],
            metadata={"locations": [(1.0, 2.0), (3.0, 4.0)], "nested": {"pos": (5.0, 6.0)}},
        )

        d = spec.to_dict()
        assert d["metadata"]["locations"] == [[1.0, 2.0], [3.0, 4.0]]
        assert d["metadata"]["nested"]["pos"] == [5.0, 6.0]
