from __future__ import annotations

"""Tests for the routine compiler system.

This module contains comprehensive tests for routine schema, compilation,
and behavior tree integration.
"""

from unittest.mock import Mock

import pytest
from jsonschema import ValidationError

from navirl.core.types import Action, AgentState
from navirl.models.behavior_tree import Blackboard, Status
from navirl.routines.behavior_integration import CompiledRoutineController, RoutineControllerFactory
from navirl.routines.compiler import (
    AgentNearbyCondition,
    AvoidArea,
    EndTimeDecorator,
    GoToTarget,
    InteractAtLocation,
    LocationReachedCondition,
    MinDurationDecorator,
    RoutineCompiler,
    TimeElapsedCondition,
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


class TestRoutineSchema:
    """Tests for routine schema and validation."""

    def test_task_creation(self):
        """Test creating various task types."""
        # GO_TO task
        go_task = Task.go_to(1.0, 2.0, speed=1.5)
        assert go_task.type == TaskType.GO_TO
        assert go_task.params["x"] == 1.0
        assert go_task.params["y"] == 2.0
        assert go_task.params["speed"] == 1.5

        # WAIT task
        wait_task = Task.wait(5.0)
        assert wait_task.type == TaskType.WAIT
        assert wait_task.params["duration"] == 5.0
        assert wait_task.duration == 5.0

        # INTERACT task
        interact_task = Task.interact((3.0, 4.0), "coffee_machine")
        assert interact_task.type == TaskType.INTERACT
        assert interact_task.params["location"] == (3.0, 4.0)
        assert interact_task.params["interaction_type"] == "coffee_machine"

        # QUEUE task
        queue_task = Task.queue((5.0, 6.0), 120.0)
        assert queue_task.type == TaskType.QUEUE
        assert queue_task.params["queue_location"] == (5.0, 6.0)
        assert queue_task.params["max_wait_time"] == 120.0

    def test_condition_creation(self):
        """Test creating various condition types."""
        # Time elapsed condition
        time_cond = Condition.time_elapsed(10.0)
        assert time_cond.type == ConditionType.TIME_ELAPSED
        assert time_cond.params["seconds"] == 10.0

        # Location reached condition
        loc_cond = Condition.location_reached(1.0, 2.0, 0.3)
        assert loc_cond.type == ConditionType.LOCATION_REACHED
        assert loc_cond.params["x"] == 1.0
        assert loc_cond.params["y"] == 2.0
        assert loc_cond.params["radius"] == 0.3

        # Agent nearby condition
        agent_cond = Condition.agent_nearby(42, 1.5)
        assert agent_cond.type == ConditionType.AGENT_NEARBY
        assert agent_cond.params["agent_id"] == 42
        assert agent_cond.params["distance"] == 1.5

    def _normalize_for_yaml_comparison(self, obj):
        """Normalize objects for YAML comparison (tuples become lists)."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._normalize_for_yaml_comparison(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_for_yaml_comparison(item) for item in obj]
        else:
            return obj

    def test_routine_spec_yaml_roundtrip(self):
        """Test converting routine spec to/from YAML."""
        # Create a routine with various task types
        tasks = [Task.go_to(1.0, 2.0), Task.wait(5.0), Task.interact((3.0, 4.0), "printer")]

        branches = [Branch(condition=Condition.time_elapsed(30.0), tasks=[Task.go_to(10.0, 10.0)])]

        temporal_constraints = TemporalConstraint(start_time=0.0, max_duration=120.0)

        original = RoutineSpec(
            id="test_routine",
            description="Test routine for validation",
            tasks=tasks,
            branches=branches,
            temporal_constraints=temporal_constraints,
            repetitions=2,
            loop=True,
            metadata={"priority": "high"},
        )

        # Convert to YAML and back
        yaml_content = original.to_yaml()
        parsed = RoutineSpec.from_yaml(yaml_content)

        # Verify core properties
        assert parsed.id == original.id
        assert parsed.description == original.description
        assert parsed.repetitions == original.repetitions
        assert parsed.loop == original.loop
        assert parsed.metadata == original.metadata

        # Verify tasks (normalize for YAML comparison - tuples become lists)
        assert len(parsed.tasks) == len(original.tasks)
        for orig_task, parsed_task in zip(original.tasks, parsed.tasks, strict=True):
            assert parsed_task.type == orig_task.type
            assert parsed_task.params == self._normalize_for_yaml_comparison(orig_task.params)
            assert parsed_task.priority == orig_task.priority

        # Verify branches
        assert len(parsed.branches) == len(original.branches)
        orig_branch = original.branches[0]
        parsed_branch = parsed.branches[0]
        assert parsed_branch.condition.type == orig_branch.condition.type
        assert parsed_branch.condition.params == self._normalize_for_yaml_comparison(
            orig_branch.condition.params
        )
        assert len(parsed_branch.tasks) == len(orig_branch.tasks)

        # Verify temporal constraints
        assert parsed.temporal_constraints.start_time == original.temporal_constraints.start_time
        assert (
            parsed.temporal_constraints.max_duration == original.temporal_constraints.max_duration
        )

    def test_schema_validation(self):
        """Test JSON schema validation."""
        # Valid routine
        valid_data = {
            "id": "valid_routine",
            "description": "A valid routine",
            "tasks": [{"type": "go_to", "params": {"x": 1.0, "y": 2.0}, "priority": 1}],
        }

        # Should not raise
        validate_routine_spec(valid_data)

        # Invalid routine (missing required field)
        invalid_data = {"description": "Missing ID field", "tasks": []}

        with pytest.raises(ValidationError):
            validate_routine_spec(invalid_data)

        # Invalid task type
        invalid_task_type = {
            "id": "invalid_task",
            "description": "Has invalid task type",
            "tasks": [{"type": "invalid_type", "params": {}}],
        }

        with pytest.raises(ValidationError):
            validate_routine_spec(invalid_task_type)


class TestRoutineCompiler:
    """Tests for the routine compiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compiler = RoutineCompiler()
        self.mock_agent = AgentState(
            agent_id=1,
            kind="human",
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            goal_x=5.0,
            goal_y=5.0,
            radius=0.2,
            max_speed=1.0,
        )

    def test_simple_goto_compilation(self):
        """Test compiling a simple go-to routine."""
        routine = RoutineSpec(
            id="simple_goto", description="Go to a target location", tasks=[Task.go_to(5.0, 3.0)]
        )

        plan = self.compiler.compile(routine)

        assert plan.routine_id == "simple_goto"
        assert plan.original_spec == routine
        assert isinstance(plan.root_node, GoToTarget)

        # Test execution
        blackboard = Blackboard(agent=self.mock_agent, goal=(5.0, 3.0), dt=0.1)

        status = plan.root_node.tick(blackboard)
        assert status == Status.RUNNING

        # Should set preferred velocity toward target
        assert blackboard.pref_vx > 0  # Moving right
        assert blackboard.pref_vy > 0  # Moving up

    def test_wait_task_compilation(self):
        """Test compiling a wait task."""
        routine = RoutineSpec(
            id="wait_routine", description="Wait for 5 seconds", tasks=[Task.wait(5.0)]
        )

        plan = self.compiler.compile(routine)
        assert isinstance(plan.root_node, WaitForDuration)

        # Test execution
        blackboard = Blackboard(agent=self.mock_agent, dt=0.1, metadata={"sim_time": 0.0})

        # First tick - should start waiting
        status = plan.root_node.tick(blackboard)
        assert status == Status.RUNNING
        assert blackboard.pref_vx == 0.0
        assert blackboard.pref_vy == 0.0

        # Simulate time passing
        blackboard.metadata["sim_time"] = 6.0
        status = plan.root_node.tick(blackboard)
        assert status == Status.SUCCESS

    def test_sequence_compilation(self):
        """Test compiling multiple tasks into a sequence."""
        routine = RoutineSpec(
            id="sequence_routine",
            description="Go somewhere then wait",
            tasks=[Task.go_to(2.0, 3.0), Task.wait(2.0)],
        )

        plan = self.compiler.compile(routine)

        # Should create a sequence
        from navirl.models.behavior_tree import Sequence

        assert isinstance(plan.root_node, Sequence)

    def test_conditional_branch_compilation(self):
        """Test compiling routines with conditional branches."""
        main_tasks = [Task.go_to(1.0, 1.0)]
        branch_tasks = [Task.go_to(5.0, 5.0)]

        routine = RoutineSpec(
            id="branching_routine",
            description="Routine with conditional branch",
            tasks=main_tasks,
            branches=[Branch(condition=Condition.time_elapsed(10.0), tasks=branch_tasks)],
        )

        plan = self.compiler.compile(routine)

        # Should create a selector with branches
        from navirl.models.behavior_tree import Selector

        assert isinstance(plan.root_node, Selector)

    def test_repetition_compilation(self):
        """Test compiling routines with repetitions."""
        routine = RoutineSpec(
            id="repeating_routine",
            description="Repeat a task",
            tasks=[Task.go_to(1.0, 1.0)],
            repetitions=3,
        )

        plan = self.compiler.compile(routine)

        # Should wrap in a repeating sequence
        from navirl.routines.compiler import RepeatingSequence

        assert isinstance(plan.root_node, RepeatingSequence)

    def test_looping_compilation(self):
        """Test compiling looping routines."""
        routine = RoutineSpec(
            id="looping_routine",
            description="Loop a task indefinitely",
            tasks=[Task.go_to(1.0, 1.0)],
            loop=True,
        )

        plan = self.compiler.compile(routine)

        # Should wrap in a looping sequence
        from navirl.routines.compiler import LoopingSequence

        assert isinstance(plan.root_node, LoopingSequence)

    def test_custom_task_handler(self):
        """Test registering and using custom task handlers."""

        def custom_handler(task):
            return GoToTarget(10.0, 10.0)  # Fixed target regardless of params

        self.compiler.register_custom_task_handler("my_custom_task", custom_handler)

        # Create a routine with a custom task
        custom_task = Task(
            type=TaskType.CUSTOM, params={"handler": "my_custom_task", "data": "test"}
        )

        routine = RoutineSpec(
            id="custom_routine", description="Uses custom task", tasks=[custom_task]
        )

        plan = self.compiler.compile(routine)
        assert isinstance(plan.root_node, GoToTarget)

    def test_validation_errors(self):
        """Test compilation validation errors."""
        # Empty task list
        empty_routine = RoutineSpec(id="empty", description="No tasks", tasks=[])

        with pytest.raises(ValueError, match="must have at least one task"):
            self.compiler.compile(empty_routine)

        # GO_TO task missing parameters
        invalid_goto = RoutineSpec(
            id="invalid_goto",
            description="Invalid go-to task",
            tasks=[Task(type=TaskType.GO_TO, params={})],
        )

        with pytest.raises(ValueError, match="must have 'x' and 'y' parameters"):
            self.compiler.compile(invalid_goto)


class TestBehaviorNodes:
    """Tests for individual behavior tree nodes created by the compiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = AgentState(
            agent_id=1,
            kind="human",
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            goal_x=5.0,
            goal_y=5.0,
            radius=0.2,
            max_speed=1.0,
        )

    def test_goto_target_node(self):
        """Test GoToTarget behavior node."""
        node = GoToTarget(5.0, 3.0)

        blackboard = Blackboard(
            agent=self.mock_agent,
            goal=(0.0, 0.0),  # Will be overridden
            dt=0.1,
        )

        status = node.tick(blackboard)
        assert status == Status.RUNNING
        assert blackboard.goal == (5.0, 3.0)
        assert blackboard.pref_vx > 0
        assert blackboard.pref_vy > 0

    def test_wait_for_duration_node(self):
        """Test WaitForDuration behavior node."""
        node = WaitForDuration(2.0)

        blackboard = Blackboard(agent=self.mock_agent, dt=0.1, metadata={"sim_time": 10.0})

        # First tick - start waiting
        status = node.tick(blackboard)
        assert status == Status.RUNNING
        assert blackboard.pref_vx == 0.0
        assert blackboard.pref_vy == 0.0

        # Time passes
        blackboard.metadata["sim_time"] = 12.5
        status = node.tick(blackboard)
        assert status == Status.SUCCESS

    def test_interact_at_location_node(self):
        """Test InteractAtLocation behavior node."""
        node = InteractAtLocation((1.0, 1.0), "coffee_machine")

        # Agent far from location
        blackboard = Blackboard(agent=self.mock_agent, dt=0.1)

        status = node.tick(blackboard)
        assert status == Status.RUNNING
        assert blackboard.behavior.startswith("INTERACT_APPROACH")

        # Agent at location
        close_agent = AgentState(
            agent_id=1,
            kind="human",
            x=1.0,
            y=1.0,
            vx=0.0,
            vy=0.0,
            goal_x=5.0,
            goal_y=5.0,
            radius=0.2,
            max_speed=1.0,
        )
        blackboard.agent = close_agent

        status = node.tick(blackboard)
        assert status == Status.SUCCESS
        assert blackboard.behavior.startswith("INTERACTING")

    def test_avoid_area_node(self):
        """Test AvoidArea behavior node."""
        node = AvoidArea({"location": (2.0, 2.0), "radius": 1.0})

        # Agent inside avoid radius
        close_agent = AgentState(
            agent_id=1,
            kind="human",
            x=2.0,
            y=2.5,
            vx=0.0,
            vy=0.0,
            goal_x=5.0,
            goal_y=5.0,
            radius=0.2,
            max_speed=1.0,
        )

        blackboard = Blackboard(agent=close_agent, dt=0.1)

        status = node.tick(blackboard)
        assert status == Status.RUNNING
        assert blackboard.behavior == "AVOID"
        # Should be moving away from avoid location
        assert blackboard.pref_vy > 0  # Moving away (up)

    def test_time_elapsed_condition(self):
        """Test TimeElapsedCondition node."""
        node = TimeElapsedCondition(5.0)

        blackboard = Blackboard(agent=self.mock_agent, dt=0.1, metadata={"sim_time": 0.0})

        # Not enough time elapsed
        status = node.tick(blackboard)
        assert status == Status.FAILURE

        # Enough time elapsed
        blackboard.metadata["sim_time"] = 6.0
        status = node.tick(blackboard)
        assert status == Status.SUCCESS

    def test_location_reached_condition(self):
        """Test LocationReachedCondition node."""
        node = LocationReachedCondition(2.0, 3.0, 0.5)

        # Agent far from location
        blackboard = Blackboard(agent=self.mock_agent, dt=0.1)
        status = node.tick(blackboard)
        assert status == Status.FAILURE

        # Agent at location
        close_agent = AgentState(
            agent_id=1,
            kind="human",
            x=1.9,
            y=3.1,
            vx=0.0,
            vy=0.0,
            goal_x=5.0,
            goal_y=5.0,
            radius=0.2,
            max_speed=1.0,
        )
        blackboard.agent = close_agent

        status = node.tick(blackboard)
        assert status == Status.SUCCESS

    def test_agent_nearby_condition(self):
        """Test AgentNearbyCondition node."""
        node = AgentNearbyCondition(agent_id=42, distance=2.0)

        # No neighbors
        blackboard = Blackboard(agent=self.mock_agent, dt=0.1, neighbours=[])
        status = node.tick(blackboard)
        assert status == Status.FAILURE

        # Neighbor too far
        far_neighbor = AgentState(
            agent_id=42,
            kind="human",
            x=10.0,
            y=10.0,
            vx=0.0,
            vy=0.0,
            goal_x=15.0,
            goal_y=15.0,
            radius=0.2,
            max_speed=1.0,
        )
        blackboard.neighbours = [far_neighbor]
        status = node.tick(blackboard)
        assert status == Status.FAILURE

        # Neighbor close enough
        close_neighbor = AgentState(
            agent_id=42,
            kind="human",
            x=1.0,
            y=1.0,
            vx=0.0,
            vy=0.0,
            goal_x=5.0,
            goal_y=5.0,
            radius=0.2,
            max_speed=1.0,
        )
        blackboard.neighbours = [close_neighbor]
        status = node.tick(blackboard)
        assert status == Status.SUCCESS


class TestCompiledRoutineController:
    """Tests for the compiled routine controller integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.routine_spec = RoutineSpec(
            id="test_routine",
            description="Test routine for controller",
            tasks=[Task.go_to(5.0, 3.0)],
        )

        self.controller = CompiledRoutineController({1: self.routine_spec})

        self.mock_states = {
            1: AgentState(
                agent_id=1,
                kind="human",
                x=0.0,
                y=0.0,
                vx=0.0,
                vy=0.0,
                goal_x=5.0,
                goal_y=5.0,
                radius=0.2,
                max_speed=1.0,
            ),
            100: AgentState(  # Robot
                agent_id=100,
                kind="robot",
                x=10.0,
                y=10.0,
                vx=0.0,
                vy=0.0,
                goal_x=15.0,
                goal_y=15.0,
                radius=0.3,
                max_speed=1.5,
            ),
        }

    def test_controller_reset(self):
        """Test controller reset functionality."""
        self.controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (10.0, 10.0)}, backend=None
        )

        assert self.controller.human_ids == [1]
        assert self.controller.starts == {1: (0.0, 0.0)}
        assert self.controller.goals == {1: (10.0, 10.0)}

    def test_routine_execution(self):
        """Test executing a routine through the controller."""
        self.controller.reset(
            human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (10.0, 10.0)}, backend=None
        )

        mock_emit = Mock()
        actions = self.controller.step(
            step=0, time_s=0.0, dt=0.1, states=self.mock_states, robot_id=100, emit_event=mock_emit
        )

        assert 1 in actions
        action = actions[1]
        assert isinstance(action, Action)
        assert action.pref_vx > 0  # Should be moving toward (5, 3)
        assert action.pref_vy > 0

        # Should have routine metadata
        assert "routine_id" in action.metadata
        assert action.metadata["routine_id"] == "test_routine"

    def test_fallback_behavior(self):
        """Test fallback behavior when no routine is assigned."""
        controller = CompiledRoutineController({})  # No routines

        controller.reset(human_ids=[1], starts={1: (0.0, 0.0)}, goals={1: (5.0, 5.0)}, backend=None)

        mock_emit = Mock()
        actions = controller.step(
            step=0, time_s=0.0, dt=0.1, states=self.mock_states, robot_id=100, emit_event=mock_emit
        )

        action = actions[1]
        assert action.metadata.get("fallback") is True

    def test_add_remove_routines(self):
        """Test adding and removing routines dynamically."""
        controller = CompiledRoutineController({})

        # Add routine
        controller.add_routine(1, self.routine_spec)
        assert 1 in controller.compiled_plans

        # Remove routine
        controller.remove_routine(1)
        assert 1 not in controller.compiled_plans


class TestRoutineControllerFactory:
    """Tests for the routine controller factory."""

    def test_from_scenario_config(self):
        """Test creating controller from scenario configuration."""
        config = {
            "routines": {
                "1": {
                    "id": "test_routine",
                    "description": "Test routine",
                    "tasks": [{"type": "go_to", "params": {"x": 5.0, "y": 3.0}}],
                }
            },
            "fallback_behavior": "static",
        }

        controller = RoutineControllerFactory.from_scenario_config(config)
        assert isinstance(controller, CompiledRoutineController)
        assert 1 in controller.routines
        assert controller.fallback_behavior == "static"

    def test_create_simple_routines(self):
        """Test creating simple go-to routines from position sequences."""
        agent_positions = {1: [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], 2: [(5.0, 5.0), (6.0, 6.0)]}

        controller = RoutineControllerFactory.create_simple_routines(agent_positions)
        assert isinstance(controller, CompiledRoutineController)
        assert len(controller.routines) == 2
        assert all(spec.loop for spec in controller.routines.values())


class TestTemporalConstraintValidation:
    """Tests for semantic validation of temporal constraints."""

    def test_valid_constraints(self):
        """Valid constraints should not raise."""
        TemporalConstraint(start_time=0.0, end_time=10.0)
        TemporalConstraint(min_duration=1.0, max_duration=5.0)
        TemporalConstraint(start_time=0.0, end_time=10.0, min_duration=1.0, max_duration=5.0)

    def test_start_time_after_end_time_raises(self):
        with pytest.raises(ValueError, match="start_time.*must be less than end_time"):
            TemporalConstraint(start_time=10.0, end_time=5.0)

    def test_start_time_equals_end_time_raises(self):
        with pytest.raises(ValueError, match="start_time.*must be less than end_time"):
            TemporalConstraint(start_time=5.0, end_time=5.0)

    def test_min_duration_exceeds_max_duration_raises(self):
        with pytest.raises(ValueError, match="min_duration.*must not exceed max_duration"):
            TemporalConstraint(min_duration=10.0, max_duration=5.0)

    def test_equal_min_max_duration_allowed(self):
        tc = TemporalConstraint(min_duration=5.0, max_duration=5.0)
        assert tc.min_duration == 5.0
        assert tc.max_duration == 5.0


class TestEndTimeDecorator:
    """Tests for the EndTimeDecorator."""

    def _make_bb(self, sim_time: float) -> Blackboard:
        agent = Mock()
        agent.x, agent.y = 0.0, 0.0
        agent.max_speed = 1.0
        bb = Blackboard(agent=agent)
        bb.neighbours = []
        bb.metadata = {"sim_time": sim_time}
        return bb

    def test_before_end_time_delegates_to_child(self):
        child = Mock()
        child.tick.return_value = Status.RUNNING
        node = EndTimeDecorator(child, end_time=10.0)
        bb = self._make_bb(5.0)
        assert node.tick(bb) == Status.RUNNING
        child.tick.assert_called_once_with(bb)

    def test_at_end_time_returns_failure(self):
        child = Mock()
        node = EndTimeDecorator(child, end_time=10.0)
        bb = self._make_bb(10.0)
        assert node.tick(bb) == Status.FAILURE
        child.tick.assert_not_called()

    def test_past_end_time_returns_failure(self):
        child = Mock()
        node = EndTimeDecorator(child, end_time=10.0)
        bb = self._make_bb(15.0)
        assert node.tick(bb) == Status.FAILURE
        child.tick.assert_not_called()

    def test_reset_resets_child(self):
        child = Mock()
        node = EndTimeDecorator(child, end_time=10.0)
        node.reset_state()
        child.reset_state.assert_called_once()


class TestMinDurationDecorator:
    """Tests for the MinDurationDecorator."""

    def _make_bb(self, sim_time: float) -> Blackboard:
        agent = Mock()
        agent.x, agent.y = 0.0, 0.0
        agent.max_speed = 1.0
        bb = Blackboard(agent=agent)
        bb.neighbours = []
        bb.metadata = {"sim_time": sim_time}
        bb.pref_vx = 1.0
        bb.pref_vy = 1.0
        return bb

    def test_child_running_within_min_duration(self):
        child = Mock()
        child.tick.return_value = Status.RUNNING
        node = MinDurationDecorator(child, min_duration=5.0)
        bb = self._make_bb(0.0)
        assert node.tick(bb) == Status.RUNNING

    def test_child_succeeds_before_min_duration_holds(self):
        child = Mock()
        child.tick.return_value = Status.SUCCESS
        node = MinDurationDecorator(child, min_duration=5.0)

        bb = self._make_bb(0.0)
        result = node.tick(bb)
        assert result == Status.RUNNING  # Must wait for min_duration

        # Now advance past min_duration
        bb2 = self._make_bb(6.0)
        result2 = node.tick(bb2)
        assert result2 == Status.SUCCESS

    def test_child_failure_propagates_immediately(self):
        child = Mock()
        child.tick.return_value = Status.FAILURE
        node = MinDurationDecorator(child, min_duration=5.0)
        bb = self._make_bb(0.0)
        assert node.tick(bb) == Status.FAILURE

    def test_child_succeeds_after_min_duration(self):
        child = Mock()
        child.tick.return_value = Status.SUCCESS
        node = MinDurationDecorator(child, min_duration=1.0)
        bb = self._make_bb(0.0)
        node.tick(bb)  # Start timer at t=0, child succeeds

        bb2 = self._make_bb(2.0)
        assert node.tick(bb2) == Status.SUCCESS

    def test_reset_clears_state(self):
        child = Mock()
        child.tick.return_value = Status.SUCCESS
        node = MinDurationDecorator(child, min_duration=5.0)

        bb = self._make_bb(0.0)
        node.tick(bb)
        node.reset_state()

        assert node._start_time is None
        assert node._child_done is False
        child.reset_state.assert_called_once()


class TestCustomConditionValidation:
    """Tests for custom condition parameter validation in the compiler."""

    def test_custom_condition_missing_handler_raises(self):
        compiler = RoutineCompiler()
        condition = Condition(ConditionType.CUSTOM, params={})
        with pytest.raises(ValueError, match="must have a 'handler' parameter"):
            compiler._compile_condition(condition)

    def test_custom_condition_unregistered_handler_raises(self):
        compiler = RoutineCompiler()
        condition = Condition(ConditionType.CUSTOM, params={"handler": "nonexistent"})
        with pytest.raises(ValueError, match="not registered"):
            compiler._compile_condition(condition)


class TestTemporalConstraintCompilation:
    """Tests for full compilation with end_time and min_duration constraints."""

    def test_compile_with_end_time(self):
        spec = RoutineSpec(
            id="test",
            description="test",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(end_time=100.0),
        )
        compiler = RoutineCompiler()
        plan = compiler.compile(spec)
        assert isinstance(plan.root_node, EndTimeDecorator)
        assert plan.root_node.end_time == 100.0

    def test_compile_with_min_duration(self):
        spec = RoutineSpec(
            id="test",
            description="test",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(min_duration=5.0),
        )
        compiler = RoutineCompiler()
        plan = compiler.compile(spec)
        assert isinstance(plan.root_node, MinDurationDecorator)
        assert plan.root_node.min_duration == 5.0

    def test_compile_with_all_temporal_constraints(self):
        spec = RoutineSpec(
            id="test",
            description="test",
            tasks=[Task.go_to(1.0, 1.0)],
            temporal_constraints=TemporalConstraint(
                start_time=1.0, end_time=100.0, min_duration=2.0, max_duration=50.0
            ),
        )
        compiler = RoutineCompiler()
        plan = compiler.compile(spec)
        # Outermost should be DelayedStart (applied last)
        from navirl.routines.compiler import DelayedStartDecorator, TimeoutDecorator

        assert isinstance(plan.root_node, DelayedStartDecorator)
        assert isinstance(plan.root_node.child, EndTimeDecorator)
        assert isinstance(plan.root_node.child.child, TimeoutDecorator)
        assert isinstance(plan.root_node.child.child.child, MinDurationDecorator)
