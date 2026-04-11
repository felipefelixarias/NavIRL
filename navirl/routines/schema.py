"""Schema definitions for structured routine specifications.

This module defines the data structures for representing human daily-life routines
that can be compiled into executable behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import jsonschema
import yaml


class TaskType(Enum):
    """Types of tasks that can be included in a routine."""

    GO_TO = auto()  # Move to a specific location
    WAIT = auto()  # Wait for a duration or condition
    INTERACT = auto()  # Interact with an object/location
    FOLLOW = auto()  # Follow another agent
    AVOID = auto()  # Avoid a specific area or agent
    QUEUE = auto()  # Join and wait in a queue
    GROUP = auto()  # Maintain group cohesion
    CUSTOM = auto()  # Custom behavior node


class ConditionType(Enum):
    """Types of conditions for branching and temporal constraints."""

    TIME_ELAPSED = auto()  # Time-based condition
    LOCATION_REACHED = auto()  # Spatial condition
    AGENT_NEARBY = auto()  # Proximity to other agents
    CUSTOM = auto()  # Custom predicate function


@dataclass
class Condition:
    """A condition that can trigger branching or task completion."""

    type: ConditionType
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def time_elapsed(cls, seconds: float) -> Condition:
        """Create a time-based condition."""
        return cls(ConditionType.TIME_ELAPSED, {"seconds": seconds})

    @classmethod
    def location_reached(cls, x: float, y: float, radius: float = 0.5) -> Condition:
        """Create a location-based condition."""
        return cls(ConditionType.LOCATION_REACHED, {"x": x, "y": y, "radius": radius})

    @classmethod
    def agent_nearby(cls, agent_id: int | None = None, distance: float = 2.0) -> Condition:
        """Create an agent proximity condition."""
        return cls(ConditionType.AGENT_NEARBY, {"agent_id": agent_id, "distance": distance})


@dataclass
class Task:
    """A single task within a routine."""

    type: TaskType
    params: dict[str, Any] = field(default_factory=dict)
    duration: float | None = None
    completion_condition: Condition | None = None
    priority: int = 1

    @classmethod
    def go_to(cls, x: float, y: float, speed: float = 1.0) -> Task:
        """Create a go-to task."""
        return cls(TaskType.GO_TO, {"x": x, "y": y, "speed": speed})

    @classmethod
    def wait(cls, duration: float) -> Task:
        """Create a wait task."""
        return cls(TaskType.WAIT, {"duration": duration}, duration=duration)

    @classmethod
    def interact(cls, location: tuple[float, float], interaction_type: str = "default") -> Task:
        """Create an interaction task."""
        return cls(TaskType.INTERACT, {"location": location, "interaction_type": interaction_type})

    @classmethod
    def queue(cls, queue_location: tuple[float, float], max_wait_time: float = 300.0) -> Task:
        """Create a queue task."""
        return cls(
            TaskType.QUEUE, {"queue_location": queue_location, "max_wait_time": max_wait_time}
        )


@dataclass
class Branch:
    """A conditional branch in a routine."""

    condition: Condition
    tasks: list[Task]
    probability: float = 1.0  # For probabilistic branching


@dataclass
class TemporalConstraint:
    """Temporal constraints on task execution."""

    start_time: float | None = None
    end_time: float | None = None
    max_duration: float | None = None
    min_duration: float | None = None

    def __post_init__(self) -> None:
        """Validate semantic consistency of temporal constraints."""
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            raise ValueError(
                f"start_time ({self.start_time}) must be less than end_time ({self.end_time})"
            )
        if (
            self.min_duration is not None
            and self.max_duration is not None
            and self.min_duration > self.max_duration
        ):
            raise ValueError(
                f"min_duration ({self.min_duration}) must not exceed max_duration ({self.max_duration})"
            )


@dataclass
class RoutineSpec:
    """Complete specification for a structured routine."""

    id: str
    description: str
    tasks: list[Task]
    branches: list[Branch] = field(default_factory=list)
    temporal_constraints: TemporalConstraint | None = None
    repetitions: int = 1
    loop: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> RoutineSpec:
        """Load a routine specification from YAML content."""
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutineSpec:
        """Create a routine specification from a dictionary."""
        # Convert tasks
        tasks = []
        for task_data in data.get("tasks", []):
            task_type = TaskType[task_data["type"].upper()]
            params = task_data.get("params", {})
            duration = task_data.get("duration")
            priority = task_data.get("priority", 1)

            # Handle completion condition
            completion_condition = None
            if "completion_condition" in task_data:
                cc_data = task_data["completion_condition"]
                condition_type = ConditionType[cc_data["type"].upper()]
                completion_condition = Condition(condition_type, cc_data.get("params", {}))

            tasks.append(
                Task(
                    type=task_type,
                    params=params,
                    duration=duration,
                    completion_condition=completion_condition,
                    priority=priority,
                )
            )

        # Convert branches
        branches = []
        for branch_data in data.get("branches", []):
            condition_data = branch_data["condition"]
            condition_type = ConditionType[condition_data["type"].upper()]
            condition = Condition(condition_type, condition_data.get("params", {}))

            branch_tasks = []
            for task_data in branch_data.get("tasks", []):
                task_type = TaskType[task_data["type"].upper()]
                branch_tasks.append(
                    Task(
                        type=task_type,
                        params=task_data.get("params", {}),
                        duration=task_data.get("duration"),
                        priority=task_data.get("priority", 1),
                    )
                )

            branches.append(
                Branch(
                    condition=condition,
                    tasks=branch_tasks,
                    probability=branch_data.get("probability", 1.0),
                )
            )

        # Convert temporal constraints
        temporal_constraints = None
        if "temporal_constraints" in data:
            tc_data = data["temporal_constraints"]
            temporal_constraints = TemporalConstraint(
                start_time=tc_data.get("start_time"),
                end_time=tc_data.get("end_time"),
                max_duration=tc_data.get("max_duration"),
                min_duration=tc_data.get("min_duration"),
            )

        return cls(
            id=data["id"],
            description=data["description"],
            tasks=tasks,
            branches=branches,
            temporal_constraints=temporal_constraints,
            repetitions=data.get("repetitions", 1),
            loop=data.get("loop", False),
            metadata=data.get("metadata", {}),
        )

    def _convert_for_yaml(self, obj: Any) -> Any:
        """Convert objects to YAML-safe representations."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_yaml(item) for item in obj]
        else:
            return obj

    def to_dict(self) -> dict[str, Any]:
        """Convert routine specification to dictionary."""
        result = {
            "id": self.id,
            "description": self.description,
            "tasks": [],
            "repetitions": self.repetitions,
            "loop": self.loop,
            "metadata": self._convert_for_yaml(self.metadata),
        }

        # Convert tasks
        for task in self.tasks:
            task_dict = {
                "type": task.type.name.lower(),
                "params": self._convert_for_yaml(task.params),
                "priority": task.priority,
            }
            if task.duration is not None:
                task_dict["duration"] = task.duration
            if task.completion_condition is not None:
                task_dict["completion_condition"] = {
                    "type": task.completion_condition.type.name.lower(),
                    "params": self._convert_for_yaml(task.completion_condition.params),
                }
            result["tasks"].append(task_dict)

        # Convert branches
        if self.branches:
            result["branches"] = []
            for branch in self.branches:
                branch_dict = {
                    "condition": {
                        "type": branch.condition.type.name.lower(),
                        "params": self._convert_for_yaml(branch.condition.params),
                    },
                    "tasks": [],
                    "probability": branch.probability,
                }
                for task in branch.tasks:
                    branch_dict["tasks"].append(
                        {
                            "type": task.type.name.lower(),
                            "params": self._convert_for_yaml(task.params),
                            "priority": task.priority,
                        }
                    )
                result["branches"].append(branch_dict)

        # Convert temporal constraints
        if self.temporal_constraints is not None:
            tc = self.temporal_constraints
            result["temporal_constraints"] = {}
            if tc.start_time is not None:
                result["temporal_constraints"]["start_time"] = tc.start_time
            if tc.end_time is not None:
                result["temporal_constraints"]["end_time"] = tc.end_time
            if tc.max_duration is not None:
                result["temporal_constraints"]["max_duration"] = tc.max_duration
            if tc.min_duration is not None:
                result["temporal_constraints"]["min_duration"] = tc.min_duration

        return result

    def to_yaml(self) -> str:
        """Convert routine specification to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)


# JSON Schema for validation
ROUTINE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://navirl.dev/schema/routine-v1.json",
    "title": "NavIRL RoutineSpec v1",
    "type": "object",
    "required": ["id", "description", "tasks"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "go_to",
                            "wait",
                            "interact",
                            "follow",
                            "avoid",
                            "queue",
                            "group",
                            "custom",
                        ],
                    },
                    "params": {"type": "object"},
                    "duration": {"type": "number", "minimum": 0},
                    "priority": {"type": "integer", "minimum": 1},
                    "completion_condition": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "time_elapsed",
                                    "location_reached",
                                    "agent_nearby",
                                    "custom",
                                ],
                            },
                            "params": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "branches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["condition", "tasks"],
                "properties": {
                    "condition": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "time_elapsed",
                                    "location_reached",
                                    "agent_nearby",
                                    "custom",
                                ],
                            },
                            "params": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                    "tasks": {"type": "array", "items": {"$ref": "#/properties/tasks/items"}},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "additionalProperties": False,
            },
        },
        "temporal_constraints": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "minimum": 0},
                "end_time": {"type": "number", "minimum": 0},
                "max_duration": {"type": "number", "minimum": 0},
                "min_duration": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "repetitions": {"type": "integer", "minimum": 1},
        "loop": {"type": "boolean"},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}


def validate_routine_spec(data: dict[str, Any]) -> None:
    """Validate a routine specification against the schema.

    Args:
        data: The routine specification data to validate.

    Raises:
        jsonschema.ValidationError: If the data is invalid.
    """
    jsonschema.validate(data, ROUTINE_SCHEMA)
