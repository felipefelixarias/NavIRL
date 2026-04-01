from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Path:
    """Represents a planned path through the environment.

    Attributes:
        waypoints: Sequence of 2-D positions, shape ``(N, 2)``.
        timestamps: Time at each waypoint, shape ``(N,)``.
        velocities: Velocity at each waypoint, shape ``(N, 2)``.
        cost: Total cost of the path (lower is better).
    """

    waypoints: np.ndarray  # (N, 2)
    timestamps: np.ndarray  # (N,)
    velocities: np.ndarray  # (N, 2)
    cost: float = 0.0

    # Optional extra metadata.
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Total Euclidean length of the path."""
        if self.waypoints.shape[0] < 2:
            return 0.0
        diffs = np.diff(self.waypoints, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def duration(self) -> float:
        """Total duration of the path in seconds."""
        if self.timestamps.size == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def num_waypoints(self) -> int:
        return int(self.waypoints.shape[0])

    def interpolate(self, t: float) -> np.ndarray:
        """Linearly interpolate the path at time *t*.

        Returns the interpolated 2-D position.
        """
        if t <= self.timestamps[0]:
            return self.waypoints[0].copy()
        if t >= self.timestamps[-1]:
            return self.waypoints[-1].copy()
        idx = int(np.searchsorted(self.timestamps, t)) - 1
        idx = max(0, min(idx, self.num_waypoints - 2))
        dt = self.timestamps[idx + 1] - self.timestamps[idx]
        if dt < 1e-12:
            return self.waypoints[idx].copy()
        alpha = (t - self.timestamps[idx]) / dt
        return (1 - alpha) * self.waypoints[idx] + alpha * self.waypoints[idx + 1]


@dataclass
class PlannerConfig:
    """Common configuration shared across planners.

    Attributes:
        max_iterations: Maximum iterations / expansion steps.
        time_limit: Maximum wall-clock planning time in seconds.
        resolution: Grid or sampling resolution (metres).
    """

    max_iterations: int = 10000
    time_limit: float = 5.0
    resolution: float = 0.1


class Planner(ABC):
    """Abstract base class for motion planners.

    All planners share the same interface: given a start, goal, static
    obstacles, and dynamic agents, produce a :class:`Path`.
    """

    @abstractmethod
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        """Plan a path from *start* to *goal*.

        Args:
            start: Start position ``(2,)``.
            goal: Goal position ``(2,)``.
            obstacles: Static obstacle map or list of obstacle positions.
                Interpretation is planner-specific (e.g. occupancy grid or
                Nx2 array of obstacle centres).
            dynamic_agents: List of predicted agent trajectories, each with
                shape ``(T, 2)``.

        Returns:
            A :class:`Path` from *start* toward *goal*.
        """
        del start, goal, obstacles, dynamic_agents
        raise NotImplementedError
