"""Routine compiler for converting structured specs to executable behaviors.

This module implements the core compiler that translates structured routine
specifications into behavior trees and executable plans that can be consumed
by human controllers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from navirl.models.behavior_tree import (
    ActionNode,
    Blackboard,
    Condition,
    FollowGroup,
    GoToGoal,
    Node,
    Selector,
    Sequence,
    Status,
    WaitInQueue,
)
from navirl.routines.schema import (
    Branch,
    ConditionType,
    RoutineSpec,
    Task,
    TaskType,
    TemporalConstraint,
)
from navirl.routines.schema import Condition as RoutineCondition


@dataclass
class CompilationContext:
    """Context information for the compilation process."""

    routine_id: str
    current_step: int = 0
    start_time: float = 0.0
    elapsed_time: float = 0.0
    loop_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledBehaviorPlan:
    """A compiled behavior plan from a routine specification."""

    routine_id: str
    root_node: Node
    context: CompilationContext
    original_spec: RoutineSpec

    def reset(self) -> None:
        """Reset the behavior plan state."""
        self.root_node.reset_state()
        self.context.current_step = 0
        self.context.elapsed_time = 0.0
        self.context.loop_count = 0


class RoutineCompiler:
    """Compiler for converting routine specifications to executable behavior plans."""

    def __init__(self) -> None:
        self._custom_task_handlers: dict[str, Callable[[Task], Node]] = {}
        self._custom_condition_handlers: dict[str, Callable[[RoutineCondition], Callable]] = {}

    def register_custom_task_handler(self, task_name: str, handler: Callable[[Task], Node]) -> None:
        """Register a custom handler for task compilation.

        Args:
            task_name: Name of the custom task type.
            handler: Function that takes a Task and returns a behavior tree Node.
        """
        self._custom_task_handlers[task_name] = handler

    def register_custom_condition_handler(
        self, condition_name: str, handler: Callable[[RoutineCondition], Callable]
    ) -> None:
        """Register a custom handler for condition compilation.

        Args:
            condition_name: Name of the custom condition type.
            handler: Function that takes a Condition and returns a predicate function.
        """
        self._custom_condition_handlers[condition_name] = handler

    def compile(self, spec: RoutineSpec) -> CompiledBehaviorPlan:
        """Compile a routine specification into an executable behavior plan.

        Args:
            spec: The routine specification to compile.

        Returns:
            A compiled behavior plan that can be executed by controllers.

        Raises:
            ValueError: If the specification contains invalid or unsupported elements.
        """
        context = CompilationContext(routine_id=spec.id, metadata=spec.metadata.copy())

        # Validate the specification
        self._validate_spec(spec)

        # Compile main task sequence
        main_tasks = self._compile_tasks(spec.tasks, context)

        # Compile conditional branches
        branch_nodes = self._compile_branches(spec.branches, context)

        # Create the root behavior tree structure
        if branch_nodes:
            # If we have branches, create a selector that tries branches first, then main tasks
            root_node = Selector([*branch_nodes, main_tasks])
        else:
            root_node = main_tasks

        # Handle repetitions and looping
        if spec.loop:
            root_node = LoopingSequence(root_node, spec.repetitions)
        elif spec.repetitions > 1:
            root_node = RepeatingSequence(root_node, spec.repetitions)

        # Apply temporal constraints
        if spec.temporal_constraints:
            root_node = self._apply_temporal_constraints(
                root_node, spec.temporal_constraints, context
            )

        return CompiledBehaviorPlan(
            routine_id=spec.id, root_node=root_node, context=context, original_spec=spec
        )

    def _validate_spec(self, spec: RoutineSpec) -> None:
        """Validate a routine specification for compilation.

        Args:
            spec: The routine specification to validate.

        Raises:
            ValueError: If the specification is invalid.
        """
        if not spec.tasks:
            raise ValueError(f"Routine '{spec.id}' must have at least one task")

        for i, task in enumerate(spec.tasks):
            self._validate_task(task, f"task {i}")

        for i, branch in enumerate(spec.branches):
            if not branch.tasks:
                raise ValueError(f"Branch {i} in routine '{spec.id}' must have at least one task")
            for j, task in enumerate(branch.tasks):
                self._validate_task(task, f"branch {i}, task {j}")

    def _validate_task(self, task: Task, context: str) -> None:
        """Validate a single task.

        Args:
            task: The task to validate.
            context: Context string for error messages.

        Raises:
            ValueError: If the task is invalid.
        """
        if task.type == TaskType.GO_TO:
            if "x" not in task.params or "y" not in task.params:
                raise ValueError(f"GO_TO {context} must have 'x' and 'y' parameters")

        elif task.type == TaskType.WAIT:
            if "duration" not in task.params and task.duration is None:
                raise ValueError(f"WAIT {context} must have 'duration' parameter or duration field")

        elif task.type == TaskType.CUSTOM and (
            "handler" not in task.params or task.params["handler"] not in self._custom_task_handlers
        ):
            raise ValueError(f"CUSTOM {context} must have a registered handler")

    def _compile_tasks(self, tasks: list[Task], context: CompilationContext) -> Node:
        """Compile a list of tasks into a behavior tree node.

        Args:
            tasks: List of tasks to compile.
            context: Compilation context.

        Returns:
            A behavior tree node representing the compiled tasks.
        """
        if not tasks:
            raise ValueError("Cannot compile empty task list")

        if len(tasks) == 1:
            return self._compile_single_task(tasks[0], context)

        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        # Compile each task
        compiled_nodes = [self._compile_single_task(task, context) for task in sorted_tasks]

        # Create a sequence for multiple tasks
        return Sequence(compiled_nodes)

    def _compile_single_task(self, task: Task, context: CompilationContext) -> Node:
        """Compile a single task into a behavior tree node.

        Args:
            task: The task to compile.
            context: Compilation context.

        Returns:
            A behavior tree node representing the compiled task.
        """
        if task.type == TaskType.GO_TO:
            return self._compile_goto_task(task)

        elif task.type == TaskType.WAIT:
            return self._compile_wait_task(task)

        elif task.type == TaskType.INTERACT:
            return self._compile_interact_task(task)

        elif task.type == TaskType.FOLLOW:
            return FollowGroup(task.params.get("group_radius", 3.0))

        elif task.type == TaskType.AVOID:
            return AvoidArea(task.params)

        elif task.type == TaskType.QUEUE:
            return WaitInQueue(task.params.get("queue_distance", 0.8))

        elif task.type == TaskType.GROUP:
            return FollowGroup(task.params.get("max_separation", 2.0))

        elif task.type == TaskType.CUSTOM:
            handler_name = task.params["handler"]
            handler = self._custom_task_handlers[handler_name]
            return handler(task)

        else:
            raise ValueError(f"Unsupported task type: {task.type}")

    def _compile_goto_task(self, task: Task) -> Node:
        """Compile a GO_TO task."""
        target_x = task.params["x"]
        target_y = task.params["y"]
        return GoToTarget(target_x, target_y)

    def _compile_wait_task(self, task: Task) -> Node:
        """Compile a WAIT task."""
        duration = task.params.get("duration", task.duration)
        if duration is None:
            raise ValueError("WAIT task must have a duration")
        return WaitForDuration(duration)

    def _compile_interact_task(self, task: Task) -> Node:
        """Compile an INTERACT task."""
        location = task.params.get("location", (0.0, 0.0))
        interaction_type = task.params.get("interaction_type", "default")
        return InteractAtLocation(location, interaction_type)

    def _compile_branches(self, branches: list[Branch], context: CompilationContext) -> list[Node]:
        """Compile conditional branches into behavior tree nodes.

        Args:
            branches: List of branches to compile.
            context: Compilation context.

        Returns:
            List of behavior tree nodes representing compiled branches.
        """
        compiled_branches = []

        for branch in branches:
            # Compile the condition
            condition_node = self._compile_condition(branch.condition)

            # Compile the branch tasks
            task_node = self._compile_tasks(branch.tasks, context)

            # Create a conditional sequence: condition must succeed for tasks to execute
            if branch.probability < 1.0:
                # Probabilistic branch
                probabilistic_condition = ProbabilisticCondition(condition_node, branch.probability)
                branch_node = Sequence([probabilistic_condition, task_node])
            else:
                branch_node = Sequence([condition_node, task_node])

            compiled_branches.append(branch_node)

        return compiled_branches

    def _compile_condition(self, condition: RoutineCondition) -> Node:
        """Compile a condition into a behavior tree condition node.

        Args:
            condition: The condition to compile.

        Returns:
            A condition node that evaluates the condition.
        """
        if condition.type == ConditionType.TIME_ELAPSED:
            seconds = condition.params["seconds"]
            return TimeElapsedCondition(seconds)

        elif condition.type == ConditionType.LOCATION_REACHED:
            x = condition.params["x"]
            y = condition.params["y"]
            radius = condition.params.get("radius", 0.5)
            return LocationReachedCondition(x, y, radius)

        elif condition.type == ConditionType.AGENT_NEARBY:
            agent_id = condition.params.get("agent_id")
            distance = condition.params.get("distance", 2.0)
            return AgentNearbyCondition(agent_id, distance)

        elif condition.type == ConditionType.CUSTOM:
            if "handler" not in condition.params:
                raise ValueError("CUSTOM condition must have a 'handler' parameter")
            handler_name = condition.params["handler"]
            if handler_name not in self._custom_condition_handlers:
                raise ValueError(f"CUSTOM condition handler '{handler_name}' is not registered")
            handler = self._custom_condition_handlers[handler_name]
            predicate = handler(condition)
            return Condition(predicate)

        else:
            raise ValueError(f"Unsupported condition type: {condition.type}")

    def _apply_temporal_constraints(
        self, node: Node, constraints: TemporalConstraint, context: CompilationContext
    ) -> Node:
        """Apply temporal constraints to a node.

        Args:
            node: The node to constrain.
            constraints: The temporal constraints to apply.
            context: Compilation context.

        Returns:
            A node with temporal constraints applied.
        """
        if constraints.min_duration is not None:
            node = MinDurationDecorator(node, constraints.min_duration)

        if constraints.max_duration is not None:
            node = TimeoutDecorator(node, constraints.max_duration)

        if constraints.end_time is not None:
            node = EndTimeDecorator(node, constraints.end_time)

        if constraints.start_time is not None:
            node = DelayedStartDecorator(node, constraints.start_time)

        return node


# Custom behavior tree nodes for routine compilation


class GoToTarget(ActionNode):
    """Go to a specific target location."""

    def __init__(self, target_x: float, target_y: float) -> None:
        super().__init__()
        self.target_x = target_x
        self.target_y = target_y
        # Cache the GoToGoal instance to improve performance
        self._go_to_goal = GoToGoal()

    def tick(self, bb: Blackboard) -> Status:
        # Override the goal in the blackboard
        bb.goal = (self.target_x, self.target_y)

        # Use the cached go-to-goal behavior
        return self._go_to_goal.tick(bb)

    def reset_state(self) -> None:
        # Reset the cached go-to-goal behavior when this node is reset
        self._go_to_goal.reset_state()


class WaitForDuration(ActionNode):
    """Wait for a specific duration."""

    def __init__(self, duration: float) -> None:
        super().__init__()
        self.duration = duration
        self._start_time: float | None = None

    def tick(self, bb: Blackboard) -> Status:
        if self._start_time is None:
            self._start_time = bb.metadata.get("sim_time", 0.0)

        current_time = bb.metadata.get("sim_time", 0.0)
        elapsed = current_time - self._start_time

        if elapsed >= self.duration:
            return Status.SUCCESS

        # Set velocity to zero (wait)
        bb.pref_vx = 0.0
        bb.pref_vy = 0.0
        bb.behavior = "WAIT"
        bb.metadata["wait_remaining"] = self.duration - elapsed

        return Status.RUNNING

    def reset_state(self) -> None:
        self._start_time = None


class InteractAtLocation(ActionNode):
    """Interact with an object at a specific location."""

    def __init__(self, location: tuple[float, float], interaction_type: str = "default") -> None:
        super().__init__()
        self.location = location
        self.interaction_type = interaction_type
        self._interaction_started = False

    def tick(self, bb: Blackboard) -> Status:
        # First, get to the location
        dx = self.location[0] - bb.agent.x
        dy = self.location[1] - bb.agent.y
        distance = math.hypot(dx, dy)

        if distance > 0.3:  # Not close enough, move there first
            # Move to interaction location
            speed = min(bb.agent.max_speed, distance / 0.5)
            bb.pref_vx = speed * dx / distance if distance > 0 else 0
            bb.pref_vy = speed * dy / distance if distance > 0 else 0
            bb.behavior = f"INTERACT_APPROACH_{self.interaction_type.upper()}"
            return Status.RUNNING

        # At location, perform interaction
        if not self._interaction_started:
            self._interaction_started = True
            bb.metadata["interaction_type"] = self.interaction_type
            bb.metadata["interaction_location"] = self.location

        # Simulate interaction time (could be parameterized)
        bb.pref_vx = 0.0
        bb.pref_vy = 0.0
        bb.behavior = f"INTERACTING_{self.interaction_type.upper()}"

        # For simplicity, interaction completes immediately
        # In a real system, this might depend on simulation state
        return Status.SUCCESS

    def reset_state(self) -> None:
        self._interaction_started = False


class AvoidArea(ActionNode):
    """Avoid a specific area or agent."""

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__()
        self.avoid_location = params.get("location")
        self.avoid_radius = params.get("radius", 2.0)
        self.avoid_agent_id = params.get("agent_id")

    def tick(self, bb: Blackboard) -> Status:
        avoid_x, avoid_y = 0.0, 0.0

        if self.avoid_location:
            avoid_x, avoid_y = self.avoid_location
        elif self.avoid_agent_id:
            # Find the agent to avoid
            for neighbor in bb.neighbours:
                if neighbor.agent_id == self.avoid_agent_id:
                    avoid_x, avoid_y = neighbor.x, neighbor.y
                    break
            else:
                return Status.SUCCESS  # Agent not found, nothing to avoid

        # Calculate avoidance vector
        dx = bb.agent.x - avoid_x
        dy = bb.agent.y - avoid_y
        distance = math.hypot(dx, dy)

        if distance > self.avoid_radius:
            return Status.SUCCESS  # Far enough away

        if distance < 0.1:
            # Too close, pick a random direction
            angle = random.uniform(0, 2 * math.pi)
            avoid_vx = math.cos(angle) * bb.agent.max_speed
            avoid_vy = math.sin(angle) * bb.agent.max_speed
        else:
            # Move away from the avoid location
            strength = max(0.0, 1.0 - distance / self.avoid_radius)
            avoid_vx = strength * (dx / distance) * bb.agent.max_speed
            avoid_vy = strength * (dy / distance) * bb.agent.max_speed

        bb.pref_vx += avoid_vx
        bb.pref_vy += avoid_vy
        bb.behavior = "AVOID"
        return Status.RUNNING

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


# Condition nodes


class TimeElapsedCondition(Condition):
    """Condition that succeeds after a specific time has elapsed."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds
        self._start_time: float | None = None

    def tick(self, bb: Blackboard) -> Status:
        if self._start_time is None:
            self._start_time = bb.metadata.get("sim_time", 0.0)

        current_time = bb.metadata.get("sim_time", 0.0)
        elapsed = current_time - self._start_time

        return Status.SUCCESS if elapsed >= self.seconds else Status.FAILURE

    def reset_state(self) -> None:
        self._start_time = None


class LocationReachedCondition(Condition):
    """Condition that succeeds when the agent reaches a specific location."""

    def __init__(self, x: float, y: float, radius: float = 0.5) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.radius = radius

    def tick(self, bb: Blackboard) -> Status:
        dx = self.x - bb.agent.x
        dy = self.y - bb.agent.y
        distance = math.hypot(dx, dy)

        return Status.SUCCESS if distance <= self.radius else Status.FAILURE

    def reset_state(self) -> None:
        pass


class AgentNearbyCondition(Condition):
    """Condition that succeeds when an agent is nearby."""

    def __init__(self, agent_id: int | None = None, distance: float = 2.0) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.distance = distance

    def tick(self, bb: Blackboard) -> Status:
        for neighbor in bb.neighbours:
            if self.agent_id is not None and neighbor.agent_id != self.agent_id:
                continue

            dx = neighbor.x - bb.agent.x
            dy = neighbor.y - bb.agent.y
            dist = math.hypot(dx, dy)

            if dist <= self.distance:
                return Status.SUCCESS

        return Status.FAILURE

    def reset_state(self) -> None:
        pass


class ProbabilisticCondition(Condition):
    """Condition that wraps another condition with probability."""

    def __init__(self, wrapped_condition: Node, probability: float) -> None:
        super().__init__()
        self.wrapped_condition = wrapped_condition
        self.probability = probability
        self._evaluated = False
        self._result = Status.FAILURE

    def tick(self, bb: Blackboard) -> Status:
        if not self._evaluated:
            if random.random() < self.probability:
                self._result = self.wrapped_condition.tick(bb)
            else:
                self._result = Status.FAILURE
            self._evaluated = True

        return self._result

    def reset_state(self) -> None:
        self._evaluated = False
        self._result = Status.FAILURE
        self.wrapped_condition.reset_state()


# Decorator nodes for temporal constraints


class TimeoutDecorator(Node):
    """Decorator that enforces a maximum execution time."""

    def __init__(self, child: Node, max_duration: float) -> None:
        self.child = child
        self.max_duration = max_duration
        self._start_time: float | None = None

    def tick(self, bb: Blackboard) -> Status:
        if self._start_time is None:
            self._start_time = bb.metadata.get("sim_time", 0.0)

        current_time = bb.metadata.get("sim_time", 0.0)
        elapsed = current_time - self._start_time

        if elapsed >= self.max_duration:
            return Status.FAILURE  # Timeout

        return self.child.tick(bb)

    def reset_state(self) -> None:
        self._start_time = None
        self.child.reset_state()


class DelayedStartDecorator(Node):
    """Decorator that delays execution until a start time."""

    def __init__(self, child: Node, start_time: float) -> None:
        self.child = child
        self.start_time = start_time

    def tick(self, bb: Blackboard) -> Status:
        current_time = bb.metadata.get("sim_time", 0.0)

        if current_time < self.start_time:
            return Status.RUNNING  # Not time yet

        return self.child.tick(bb)

    def reset_state(self) -> None:
        self.child.reset_state()


class EndTimeDecorator(Node):
    """Decorator that fails execution after a wall-clock end time."""

    def __init__(self, child: Node, end_time: float) -> None:
        self.child = child
        self.end_time = end_time

    def tick(self, bb: Blackboard) -> Status:
        current_time = bb.metadata.get("sim_time", 0.0)

        if current_time >= self.end_time:
            return Status.FAILURE  # Past end time

        return self.child.tick(bb)

    def reset_state(self) -> None:
        self.child.reset_state()


class MinDurationDecorator(Node):
    """Decorator that enforces a minimum execution time.

    If the child succeeds before min_duration has elapsed, this decorator
    keeps returning RUNNING until the minimum time is met, then returns SUCCESS.
    """

    def __init__(self, child: Node, min_duration: float) -> None:
        self.child = child
        self.min_duration = min_duration
        self._start_time: float | None = None
        self._child_done = False

    def tick(self, bb: Blackboard) -> Status:
        if self._start_time is None:
            self._start_time = bb.metadata.get("sim_time", 0.0)

        current_time = bb.metadata.get("sim_time", 0.0)
        elapsed = current_time - self._start_time

        if not self._child_done:
            result = self.child.tick(bb)
            if result == Status.FAILURE:
                return Status.FAILURE  # Propagate failure immediately
            if result == Status.SUCCESS:
                self._child_done = True

        if elapsed >= self.min_duration:
            if self._child_done:
                return Status.SUCCESS
            # min_duration met but child still running — delegate to child
            return self.child.tick(bb)

        # min_duration not yet met
        if self._child_done:
            # Child done but we need to wait — hold position
            bb.pref_vx = 0.0
            bb.pref_vy = 0.0
            return Status.RUNNING

        return Status.RUNNING

    def reset_state(self) -> None:
        self._start_time = None
        self._child_done = False
        self.child.reset_state()


class RepeatingSequence(Node):
    """Node that repeats a child sequence a specific number of times."""

    def __init__(self, child: Node, repetitions: int) -> None:
        self.child = child
        self.repetitions = repetitions
        self._current_repetition = 0

    def tick(self, bb: Blackboard) -> Status:
        if self._current_repetition >= self.repetitions:
            return Status.SUCCESS

        result = self.child.tick(bb)

        if result == Status.SUCCESS:
            self._current_repetition += 1
            self.child.reset_state()

            if self._current_repetition >= self.repetitions:
                return Status.SUCCESS
            else:
                return Status.RUNNING

        return result

    def reset_state(self) -> None:
        self._current_repetition = 0
        self.child.reset_state()


class LoopingSequence(Node):
    """Node that loops a child sequence indefinitely or for a set number of iterations."""

    def __init__(self, child: Node, max_loops: int = -1) -> None:
        self.child = child
        self.max_loops = max_loops
        self._current_loop = 0

    def tick(self, bb: Blackboard) -> Status:
        if self.max_loops > 0 and self._current_loop >= self.max_loops:
            return Status.SUCCESS

        result = self.child.tick(bb)

        if result == Status.SUCCESS:
            self._current_loop += 1
            self.child.reset_state()

            if self.max_loops > 0 and self._current_loop >= self.max_loops:
                return Status.SUCCESS

        return Status.RUNNING  # Always running (looping)

    def reset_state(self) -> None:
        self._current_loop = 0
        self.child.reset_state()
