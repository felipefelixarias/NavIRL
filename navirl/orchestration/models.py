"""Data models for distributed simulation orchestration.

Defines the core task and result types used across all executor backends.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


class TaskStatus(enum.Enum):
    """Lifecycle states for a simulation task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SimulationTask:
    """A single simulation run to execute.

    Parameters
    ----------
    task_id:
        Unique identifier for this task within a job.
    scenario_path:
        Path to the scenario YAML file.
    seed:
        Random seed for reproducibility.
    overrides:
        Dotted-path parameter overrides to apply to the scenario.
    """

    task_id: str
    scenario_path: str
    seed: int
    overrides: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Deterministic hash of task inputs for deduplication."""
        blob = json.dumps(
            {
                "scenario_path": self.scenario_path,
                "seed": self.seed,
                "overrides": self.overrides,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class TaskResult:
    """Result of executing a single simulation task.

    Parameters
    ----------
    task_id:
        ID of the task that produced this result.
    status:
        Final status of the task.
    metrics:
        Computed metrics from the simulation run (empty on failure).
    bundle_dir:
        Path to the output bundle directory (empty on failure).
    error:
        Error message if the task failed.
    wall_time_s:
        Wall-clock time for the task in seconds.
    """

    task_id: str
    status: TaskStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    bundle_dir: str = ""
    error: str = ""
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "metrics": self.metrics,
            "bundle_dir": self.bundle_dir,
            "error": self.error,
            "wall_time_s": self.wall_time_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        """Deserialize from a dictionary."""
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            metrics=data.get("metrics", {}),
            bundle_dir=data.get("bundle_dir", ""),
            error=data.get("error", ""),
            wall_time_s=data.get("wall_time_s", 0.0),
        )
