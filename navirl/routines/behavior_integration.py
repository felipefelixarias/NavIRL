"""Integration layer for using compiled routines with behavior trees and controllers.

This module provides the interface between compiled routine behavior plans
and the existing human controller system, allowing routines to be seamlessly
used in scenarios.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController
from navirl.models.behavior_tree import Blackboard, Status
from navirl.routines.compiler import CompiledBehaviorPlan, RoutineCompiler
from navirl.routines.schema import RoutineSpec


class CompiledRoutineController(HumanController):
    """Human controller that executes compiled routine behavior plans.

    This controller wraps compiled routine behavior plans and provides the
    HumanController interface for use in scenarios. It manages multiple
    agents, each potentially following different routines.

    Parameters
    ----------
    routines : Dict[int, RoutineSpec]
        Map from human agent ID to routine specification.
    compiler : RoutineCompiler, optional
        Compiler instance to use. If None, a default compiler is created.
    fallback_behavior : str, optional
        Behavior to fall back to if no routine is specified for an agent.
        Can be "static" (stay in place) or "goal_swap" (standard goal swapping).
    """

    def __init__(
        self,
        routines: dict[int, RoutineSpec],
        compiler: RoutineCompiler | None = None,
        fallback_behavior: str = "goal_swap",
    ) -> None:
        self.routines = dict(routines)
        self.compiler = compiler or RoutineCompiler()
        self.fallback_behavior = fallback_behavior

        # Compiled behavior plans for each agent
        self.compiled_plans: dict[int, CompiledBehaviorPlan] = {}

        # Controller state
        self.human_ids: list[int] = []
        self.starts: dict[int, tuple[float, float]] = {}
        self.goals: dict[int, tuple[float, float]] = {}
        self.backend: Any = None

        # Simulation state for time tracking
        self._sim_start_time: float = 0.0

        # Compile routines
        self._compile_routines()

    def _compile_routines(self) -> None:
        """Compile all routine specifications into behavior plans."""
        self.compiled_plans.clear()

        for agent_id, routine_spec in self.routines.items():
            try:
                plan = self.compiler.compile(routine_spec)
                self.compiled_plans[agent_id] = plan
            except Exception as e:
                logger.warning("Failed to compile routine for agent %d: %s", agent_id, e)
                # Agent will fall back to default behavior

    def add_routine(self, agent_id: int, routine_spec: RoutineSpec) -> None:
        """Add or update a routine for a specific agent.

        Args:
            agent_id: ID of the agent.
            routine_spec: Routine specification to compile and assign.
        """
        self.routines[agent_id] = routine_spec

        try:
            plan = self.compiler.compile(routine_spec)
            self.compiled_plans[agent_id] = plan
        except Exception as e:
            print(f"Warning: Failed to compile routine for agent {agent_id}: {e}")
            # Remove any existing plan
            self.compiled_plans.pop(agent_id, None)

    def remove_routine(self, agent_id: int) -> None:
        """Remove a routine for a specific agent.

        Args:
            agent_id: ID of the agent.
        """
        self.routines.pop(agent_id, None)
        self.compiled_plans.pop(agent_id, None)

    # HumanController interface implementation

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        """Reset the controller for a new episode.

        Args:
            human_ids: List of human agent IDs.
            starts: Starting positions for each agent.
            goals: Goal positions for each agent.
            backend: Simulation backend instance.
        """
        self.human_ids = list(human_ids)
        self.starts = dict(starts)
        self.goals = dict(goals)
        self.backend = backend
        self._sim_start_time = 0.0

        # Reset all compiled behavior plans
        for plan in self.compiled_plans.values():
            plan.reset()

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        """Execute one step of behavior for all agents.

        Args:
            step: Current simulation step.
            time_s: Current simulation time in seconds.
            dt: Time step duration.
            states: Current states of all agents.
            robot_id: ID of the robot agent.
            emit_event: Function to emit simulation events.

        Returns:
            Dictionary mapping agent ID to desired action.
        """
        if self._sim_start_time == 0.0:
            self._sim_start_time = time_s

        actions: dict[int, Action] = {}

        for agent_id in self.human_ids:
            agent_state = states[agent_id]

            if agent_id in self.compiled_plans:
                # Execute compiled routine
                action = self._execute_routine(
                    agent_id, agent_state, states, robot_id, time_s, dt, emit_event
                )
            else:
                # Fall back to default behavior
                action = self._execute_fallback_behavior(
                    agent_id, agent_state, states, robot_id, time_s, dt, emit_event
                )

            actions[agent_id] = action

        return actions

    def _execute_routine(
        self,
        agent_id: int,
        agent_state: AgentState,
        all_states: dict[int, AgentState],
        robot_id: int,
        time_s: float,
        dt: float,
        emit_event: EventSink,
    ) -> Action:
        """Execute a compiled routine for a specific agent.

        Args:
            agent_id: ID of the agent.
            agent_state: Current state of the agent.
            all_states: States of all agents.
            robot_id: ID of the robot agent.
            time_s: Current simulation time.
            dt: Time step duration.
            emit_event: Function to emit events.

        Returns:
            Desired action for the agent.
        """
        plan = self.compiled_plans[agent_id]

        # Build neighbor list (excluding self)
        neighbors = [state for aid, state in all_states.items() if aid != agent_id]

        # Get current goal (may be modified by routine)
        current_goal = self.goals.get(agent_id, (agent_state.x, agent_state.y))

        # Create blackboard with simulation context
        blackboard = Blackboard(
            agent=agent_state,
            neighbours=neighbors,
            robot=all_states.get(robot_id),
            goal=current_goal,
            dt=dt,
            metadata={
                "sim_time": time_s - self._sim_start_time,
                "sim_step": plan.context.current_step,
                "agent_id": agent_id,
                "routine_id": plan.routine_id,
            },
        )

        # Execute the behavior tree
        try:
            status = plan.root_node.tick(blackboard)
            plan.context.current_step += 1
            plan.context.elapsed_time = time_s - self._sim_start_time

            # Emit routine events
            if status == Status.SUCCESS:
                emit_event(
                    "routine_completed",
                    agent_id,
                    {
                        "routine_id": plan.routine_id,
                        "elapsed_time": plan.context.elapsed_time,
                        "steps": plan.context.current_step,
                    },
                )

                # Handle looping or repetitions
                if (
                    plan.original_spec.loop
                    or plan.context.loop_count < plan.original_spec.repetitions - 1
                ):
                    plan.reset()
                    plan.context.loop_count += 1
                    emit_event(
                        "routine_loop",
                        agent_id,
                        {"routine_id": plan.routine_id, "loop_count": plan.context.loop_count},
                    )

            # Clamp velocity to agent's max speed
            pref_vx, pref_vy = blackboard.pref_vx, blackboard.pref_vy
            speed = math.hypot(pref_vx, pref_vy)
            if speed > agent_state.max_speed and speed > 1e-6:
                scale = agent_state.max_speed / speed
                pref_vx *= scale
                pref_vy *= scale

            return Action(
                pref_vx=pref_vx,
                pref_vy=pref_vy,
                behavior=blackboard.behavior,
                metadata={
                    **blackboard.metadata,
                    "routine_status": status.name.lower(),
                    "routine_id": plan.routine_id,
                },
            )

        except Exception as e:
            # Routine execution failed, emit error and fall back
            emit_event(
                "routine_error",
                agent_id,
                {"routine_id": plan.routine_id, "error": str(e), "step": plan.context.current_step},
            )

            return self._execute_fallback_behavior(
                agent_id, agent_state, all_states, robot_id, time_s, dt, emit_event
            )

    def _execute_fallback_behavior(
        self,
        agent_id: int,
        agent_state: AgentState,
        all_states: dict[int, AgentState],
        robot_id: int,
        time_s: float,
        dt: float,
        emit_event: EventSink,
    ) -> Action:
        """Execute fallback behavior when no routine is available.

        Args:
            agent_id: ID of the agent.
            agent_state: Current state of the agent.
            all_states: States of all agents.
            robot_id: ID of the robot agent.
            time_s: Current simulation time.
            dt: Time step duration.
            emit_event: Function to emit events.

        Returns:
            Desired action for the agent.
        """
        if self.fallback_behavior == "static":
            # Stay in place
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="STATIC", metadata={"fallback": True})

        elif self.fallback_behavior == "goal_swap":
            # Standard goal swapping behavior
            gx, gy = self.goals.get(agent_id, (agent_state.x, agent_state.y))

            # Check if at goal (swap if so)
            if math.hypot(gx - agent_state.x, gy - agent_state.y) < 0.5:
                prev_goal = self.goals[agent_id]
                prev_start = self.starts[agent_id]
                self.goals[agent_id] = prev_start
                self.starts[agent_id] = prev_goal

                emit_event(
                    "goal_swap",
                    agent_id,
                    {
                        "new_goal": list(self.goals[agent_id]),
                        "new_start": list(self.starts[agent_id]),
                    },
                )

                gx, gy = self.goals[agent_id]

            # Move toward goal
            dx = gx - agent_state.x
            dy = gy - agent_state.y
            distance = math.hypot(dx, dy)

            if distance < 1e-6:
                pref_vx, pref_vy = 0.0, 0.0
            else:
                # Simple proportional control
                speed = min(agent_state.max_speed, distance / 0.5)
                pref_vx = speed * dx / distance
                pref_vy = speed * dy / distance

            return Action(
                pref_vx=pref_vx, pref_vy=pref_vy, behavior="GO_TO", metadata={"fallback": True}
            )

        else:
            raise ValueError(f"Unknown fallback behavior: {self.fallback_behavior}")


def _validate_file_path(file_path: str) -> Path:
    """Validate file path to prevent path traversal attacks.

    Args:
        file_path: File path to validate.

    Returns:
        Validated Path object.

    Raises:
        ValueError: If path is unsafe or doesn't exist.
    """
    try:
        # Convert to Path object and resolve to absolute path
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        # Check if it's actually a file (not a directory)
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Prevent path traversal by checking the original path contains ".."
        # This prevents attacks like "../../../etc/passwd" without being overly restrictive
        if ".." in os.path.normpath(file_path):
            raise ValueError(f"Path traversal detected: {file_path}")

        # Additional security: ensure it's a YAML file
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Only YAML files are allowed: {file_path}")

        return path

    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid file path: {file_path} - {e}") from e


class RoutineControllerFactory:
    """Factory for creating routine controllers with different configurations."""

    @staticmethod
    def from_yaml_files(routine_files: dict[int, str]) -> CompiledRoutineController:
        """Create a routine controller from YAML files.

        Args:
            routine_files: Map from agent ID to YAML file path.

        Returns:
            Configured routine controller.
        """
        routines = {}

        for agent_id, file_path in routine_files.items():
            try:
                # Validate file path to prevent path traversal attacks
                validated_path = _validate_file_path(file_path)

                with open(validated_path) as f:
                    content = f.read()
                routine_spec = RoutineSpec.from_yaml(content)
                routines[agent_id] = routine_spec
            except Exception as e:
                logger.warning(
                    "Failed to load routine for agent %d from %s: %s", agent_id, file_path, e
                )

        return CompiledRoutineController(routines)

    @staticmethod
    def from_scenario_config(config: dict[str, Any]) -> CompiledRoutineController:
        """Create a routine controller from scenario configuration.

        Args:
            config: Scenario configuration containing routine specifications.

        Returns:
            Configured routine controller.

        Example config format:
        ```yaml
        routines:
          1:  # agent ID
            id: "morning_coffee"
            description: "Get morning coffee"
            tasks:
              - type: "go_to"
                params: {"x": -2.0, "y": 1.0}
              - type: "wait"
                params: {"duration": 5.0}
          2:
            id: "check_email"
            # ... more routine spec
        fallback_behavior: "goal_swap"
        ```
        """
        routines = {}
        routine_configs = config.get("routines", {})

        for agent_id_str, routine_config in routine_configs.items():
            try:
                agent_id = int(agent_id_str)
                routine_spec = RoutineSpec.from_dict(routine_config)
                routines[agent_id] = routine_spec
            except Exception as e:
                logger.warning("Failed to parse routine for agent %s: %s", agent_id_str, e)

        fallback_behavior = config.get("fallback_behavior", "goal_swap")

        return CompiledRoutineController(routines, fallback_behavior=fallback_behavior)

    @staticmethod
    def create_simple_routines(
        agent_positions: dict[int, list[tuple[float, float]]],
    ) -> CompiledRoutineController:
        """Create simple go-to routines from position sequences.

        Args:
            agent_positions: Map from agent ID to list of positions to visit.

        Returns:
            Routine controller with simple go-to sequences.
        """
        routines = {}

        for agent_id, positions in agent_positions.items():
            if not positions:
                continue

            # Create a simple sequence of go-to tasks
            tasks = []
            for i, (x, y) in enumerate(positions):
                tasks.append(
                    {
                        "type": "go_to",
                        "params": {"x": x, "y": y},
                        "priority": len(positions) - i,  # Earlier tasks have higher priority
                    }
                )

            routine_spec = RoutineSpec.from_dict(
                {
                    "id": f"agent_{agent_id}_tour",
                    "description": f"Tour sequence for agent {agent_id}",
                    "tasks": tasks,
                    "loop": True,
                }
            )

            routines[agent_id] = routine_spec

        return CompiledRoutineController(routines)
