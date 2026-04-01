"""Episode logging with structured data, export, statistics, and trajectory recording.

This module provides comprehensive episode logging for the NavIRL simulation
framework. It records agent states, events, trajectories, and computes
episode-level statistics. Supports JSON, CSV, and JSONL export formats.
"""

from __future__ import annotations

import csv
import json
import math
import time
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from navirl.core.types import AgentState, EventRecord

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryPoint:
    """A single point in an agent's trajectory.

    Attributes:
        step: Simulation step index.
        time_s: Wall-clock simulation time in seconds.
        x: X coordinate of the agent.
        y: Y coordinate of the agent.
        vx: X component of velocity.
        vy: Y component of velocity.
        speed: Scalar speed (computed from vx, vy).
        heading: Heading angle in radians.
    """

    step: int
    time_s: float
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    heading: float

    @classmethod
    def from_agent_state(cls, step: int, time_s: float, state: AgentState) -> TrajectoryPoint:
        """Create a trajectory point from an agent state.

        Args:
            step: Current simulation step.
            time_s: Current simulation time in seconds.
            state: Agent state to extract position/velocity from.

        Returns:
            A new TrajectoryPoint instance.
        """
        speed = math.sqrt(state.vx**2 + state.vy**2)
        heading = math.atan2(state.vy, state.vx)
        return cls(
            step=step,
            time_s=time_s,
            x=state.x,
            y=state.y,
            vx=state.vx,
            vy=state.vy,
            speed=speed,
            heading=heading,
        )


@dataclass
class AgentTrajectory:
    """Full trajectory for a single agent across an episode.

    Attributes:
        agent_id: Unique agent identifier.
        kind: Agent kind (``"robot"`` or ``"human"``).
        points: Ordered list of trajectory points.
    """

    agent_id: int
    kind: str
    points: list[TrajectoryPoint] = field(default_factory=list)

    def add_point(self, point: TrajectoryPoint) -> None:
        """Append a trajectory point.

        Args:
            point: The trajectory point to append.
        """
        self.points.append(point)

    @property
    def total_distance(self) -> float:
        """Compute total path length traversed by the agent.

        Returns:
            Sum of Euclidean distances between consecutive points.
        """
        if len(self.points) < 2:
            return 0.0
        dist = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i - 1].x
            dy = self.points[i].y - self.points[i - 1].y
            dist += math.sqrt(dx * dx + dy * dy)
        return dist

    @property
    def duration(self) -> float:
        """Total duration of the trajectory in seconds.

        Returns:
            Time difference between last and first points.
        """
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].time_s - self.points[0].time_s

    @property
    def average_speed(self) -> float:
        """Mean speed across all trajectory points.

        Returns:
            Average scalar speed, or 0.0 if no points recorded.
        """
        if not self.points:
            return 0.0
        return float(np.mean([p.speed for p in self.points]))

    @property
    def max_speed(self) -> float:
        """Peak speed observed in the trajectory.

        Returns:
            Maximum scalar speed, or 0.0 if no points recorded.
        """
        if not self.points:
            return 0.0
        return float(np.max([p.speed for p in self.points]))

    @property
    def displacement(self) -> float:
        """Straight-line displacement from start to end.

        Returns:
            Euclidean distance between first and last positions.
        """
        if len(self.points) < 2:
            return 0.0
        dx = self.points[-1].x - self.points[0].x
        dy = self.points[-1].y - self.points[0].y
        return math.sqrt(dx * dx + dy * dy)

    @property
    def path_efficiency(self) -> float:
        """Ratio of displacement to total distance (1.0 is perfectly straight).

        Returns:
            Value between 0.0 and 1.0, or 0.0 if total distance is zero.
        """
        td = self.total_distance
        if td < 1e-9:
            return 0.0
        return self.displacement / td

    def positions_array(self) -> np.ndarray:
        """Return positions as a numpy array of shape ``(N, 2)``.

        Returns:
            Array of (x, y) coordinates for every trajectory point.
        """
        if not self.points:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float64)

    def velocities_array(self) -> np.ndarray:
        """Return velocities as a numpy array of shape ``(N, 2)``.

        Returns:
            Array of (vx, vy) values for every trajectory point.
        """
        if not self.points:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([[p.vx, p.vy] for p in self.points], dtype=np.float64)

    def speeds_array(self) -> np.ndarray:
        """Return scalar speeds as a 1-D numpy array.

        Returns:
            Array of speed values for every trajectory point.
        """
        if not self.points:
            return np.empty(0, dtype=np.float64)
        return np.array([p.speed for p in self.points], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trajectory to a dictionary.

        Returns:
            Dictionary with agent_id, kind, and list of point dicts.
        """
        return {
            "agent_id": self.agent_id,
            "kind": self.kind,
            "num_points": len(self.points),
            "total_distance": self.total_distance,
            "duration": self.duration,
            "average_speed": self.average_speed,
            "max_speed": self.max_speed,
            "displacement": self.displacement,
            "path_efficiency": self.path_efficiency,
            "points": [
                {
                    "step": p.step,
                    "time_s": p.time_s,
                    "x": p.x,
                    "y": p.y,
                    "vx": p.vx,
                    "vy": p.vy,
                    "speed": p.speed,
                    "heading": p.heading,
                }
                for p in self.points
            ],
        }


@dataclass
class EpisodeEvent:
    """An enriched event record with additional metadata.

    Attributes:
        step: Simulation step when the event occurred.
        time_s: Simulation time in seconds.
        event_type: Category of the event (e.g. ``"collision"``).
        agent_id: ID of the agent involved, or ``None`` for global events.
        payload: Arbitrary additional data.
        wall_time: Wall-clock timestamp when the event was recorded.
    """

    step: int
    time_s: float
    event_type: str
    agent_id: int | None
    payload: dict[str, Any] = field(default_factory=dict)
    wall_time: float = field(default_factory=time.time)

    @classmethod
    def from_event_record(cls, record: EventRecord) -> EpisodeEvent:
        """Create an EpisodeEvent from a core EventRecord.

        Args:
            record: The core event record.

        Returns:
            A new EpisodeEvent with the current wall-clock time.
        """
        return cls(
            step=record.step,
            time_s=record.time_s,
            event_type=record.event_type,
            agent_id=record.agent_id,
            payload=dict(record.payload),
            wall_time=time.time(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Flat dictionary representation of the event.
        """
        return {
            "step": self.step,
            "time_s": self.time_s,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "wall_time": self.wall_time,
        }


@dataclass
class EpisodeStatistics:
    """Aggregated statistics for a completed episode.

    Attributes:
        episode_id: Unique identifier for the episode.
        num_steps: Total number of simulation steps.
        duration_s: Total simulation duration in seconds.
        wall_duration_s: Total wall-clock duration in seconds.
        num_agents: Number of agents in the episode.
        num_robots: Number of robot agents.
        num_humans: Number of human agents.
        num_events: Total events recorded.
        num_collisions: Number of collision events.
        mean_speed: Mean scalar speed across all agents.
        max_speed: Maximum scalar speed observed.
        mean_path_efficiency: Mean path efficiency across all agents.
        total_distance_all: Sum of total distances for all agents.
        event_counts: Counts per event type.
        per_agent_stats: Per-agent summary statistics.
    """

    episode_id: str
    num_steps: int = 0
    duration_s: float = 0.0
    wall_duration_s: float = 0.0
    num_agents: int = 0
    num_robots: int = 0
    num_humans: int = 0
    num_events: int = 0
    num_collisions: int = 0
    mean_speed: float = 0.0
    max_speed: float = 0.0
    mean_path_efficiency: float = 0.0
    total_distance_all: float = 0.0
    event_counts: dict[str, int] = field(default_factory=dict)
    per_agent_stats: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistics to a dictionary.

        Returns:
            Dictionary containing all statistics fields.
        """
        return {
            "episode_id": self.episode_id,
            "num_steps": self.num_steps,
            "duration_s": self.duration_s,
            "wall_duration_s": self.wall_duration_s,
            "num_agents": self.num_agents,
            "num_robots": self.num_robots,
            "num_humans": self.num_humans,
            "num_events": self.num_events,
            "num_collisions": self.num_collisions,
            "mean_speed": self.mean_speed,
            "max_speed": self.max_speed,
            "mean_path_efficiency": self.mean_path_efficiency,
            "total_distance_all": self.total_distance_all,
            "event_counts": dict(self.event_counts),
            "per_agent_stats": {str(k): dict(v) for k, v in self.per_agent_stats.items()},
        }


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------


class EpisodeLogger:
    """Comprehensive episode logger for NavIRL simulations.

    Records agent states, events, and trajectories during an episode. Provides
    methods for computing statistics, exporting data in multiple formats, and
    managing episode lifecycle.

    Args:
        bundle_dir: Directory where episode artifacts are stored.
        episode_id: Optional episode identifier (auto-generated if omitted).
        buffer_size: Number of state lines to buffer before flushing.
        record_trajectories: Whether to maintain in-memory trajectories.

    Example::

        logger = EpisodeLogger(Path("/tmp/ep_001"))
        logger.write_state(0, 0.0, agents)
        logger.write_event(event_record)
        stats = logger.compute_statistics()
        logger.export_csv(Path("/tmp/ep_001/states.csv"))
        logger.close()
    """

    def __init__(
        self,
        bundle_dir: Path,
        episode_id: str | None = None,
        buffer_size: int = 100,
        record_trajectories: bool = True,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        self.episode_id = episode_id or uuid.uuid4().hex[:12]
        self._buffer_size = buffer_size
        self._record_trajectories = record_trajectories

        # File paths
        self.state_path = self.bundle_dir / "state.jsonl"
        self.events_path = self.bundle_dir / "events.jsonl"
        self.scenario_path = self.bundle_dir / "scenario.yaml"
        self.summary_path = self.bundle_dir / "summary.json"
        self.trajectories_path = self.bundle_dir / "trajectories.json"
        self.statistics_path = self.bundle_dir / "statistics.json"

        # Open file handles
        self._state_f = self.state_path.open("w", encoding="utf-8")
        self._events_f = self.events_path.open("w", encoding="utf-8")

        # In-memory storage
        self._trajectories: dict[int, AgentTrajectory] = {}
        self._events: list[EpisodeEvent] = []
        self._state_buffer: list[str] = []
        self._step_count: int = 0
        self._last_time_s: float = 0.0
        self._first_time_s: float | None = None
        self._wall_start: float = time.time()
        self._closed: bool = False
        self._agent_kinds: dict[int, str] = {}
        self._metadata: dict[str, Any] = {}
        self._reward_history: list[dict[str, Any]] = []

    # -- Context manager support ------------------------------------------------

    def __enter__(self) -> EpisodeLogger:
        """Enter context manager.

        Returns:
            This logger instance.
        """
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit context manager and close the logger."""
        self.close()

    # -- Scenario ---------------------------------------------------------------

    def write_resolved_scenario(self, scenario: dict[str, Any]) -> None:
        """Write the resolved scenario configuration to a YAML file.

        Args:
            scenario: Dictionary containing the scenario configuration.
        """
        with self.scenario_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(scenario, f, sort_keys=False)

    # -- State recording --------------------------------------------------------

    def write_state(self, step: int, time_s: float, agents: list[AgentState]) -> None:
        """Record the state of all agents at a given simulation step.

        Writes the state as a JSON line and optionally updates in-memory
        trajectory records.

        Args:
            step: Current simulation step index.
            time_s: Current simulation time in seconds.
            agents: List of agent states at this step.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed EpisodeLogger.")

        if self._first_time_s is None:
            self._first_time_s = time_s

        self._step_count = max(self._step_count, step + 1)
        self._last_time_s = time_s

        row = {
            "step": int(step),
            "time_s": float(time_s),
            "agents": [
                {
                    "id": int(a.agent_id),
                    "kind": a.kind,
                    "x": float(a.x),
                    "y": float(a.y),
                    "vx": float(a.vx),
                    "vy": float(a.vy),
                    "goal_x": float(a.goal_x),
                    "goal_y": float(a.goal_y),
                    "radius": float(a.radius),
                    "max_speed": float(a.max_speed),
                    "behavior": a.behavior,
                    "metadata": a.metadata,
                }
                for a in agents
            ],
        }
        line = json.dumps(row, sort_keys=True) + "\n"
        self._state_buffer.append(line)

        if len(self._state_buffer) >= self._buffer_size:
            self._flush_state_buffer()

        # Update trajectories
        if self._record_trajectories:
            for a in agents:
                self._agent_kinds[a.agent_id] = a.kind
                if a.agent_id not in self._trajectories:
                    self._trajectories[a.agent_id] = AgentTrajectory(
                        agent_id=a.agent_id, kind=a.kind
                    )
                point = TrajectoryPoint.from_agent_state(step, time_s, a)
                self._trajectories[a.agent_id].add_point(point)

    def _flush_state_buffer(self) -> None:
        """Flush the in-memory state buffer to disk."""
        if self._state_buffer and not self._state_f.closed:
            self._state_f.writelines(self._state_buffer)
            self._state_f.flush()
            self._state_buffer.clear()

    # -- Event recording --------------------------------------------------------

    def write_event(self, event: EventRecord) -> None:
        """Record a simulation event.

        Args:
            event: The event record to log.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed EpisodeLogger.")

        self._events_f.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        self._events_f.flush()

        ep_event = EpisodeEvent.from_event_record(event)
        self._events.append(ep_event)

    def write_custom_event(
        self,
        step: int,
        time_s: float,
        event_type: str,
        agent_id: int | None = None,
        **payload: Any,
    ) -> None:
        """Record a custom event with arbitrary keyword payload.

        This is a convenience method that constructs an ``EventRecord``
        internally.

        Args:
            step: Simulation step.
            time_s: Simulation time in seconds.
            event_type: Category string for the event.
            agent_id: Optional agent ID.
            **payload: Additional key-value pairs stored in the event payload.
        """
        record = EventRecord(
            step=step,
            time_s=time_s,
            event_type=event_type,
            agent_id=agent_id,
            payload=dict(payload),
        )
        self.write_event(record)

    # -- Reward recording -------------------------------------------------------

    def write_reward(
        self,
        step: int,
        time_s: float,
        agent_id: int,
        reward: float,
        components: dict[str, float] | None = None,
    ) -> None:
        """Record a reward signal for an agent.

        Args:
            step: Simulation step.
            time_s: Simulation time in seconds.
            agent_id: The agent receiving the reward.
            reward: Scalar reward value.
            components: Optional breakdown of reward into named components.
        """
        entry = {
            "step": step,
            "time_s": time_s,
            "agent_id": agent_id,
            "reward": reward,
            "components": components or {},
        }
        self._reward_history.append(entry)

    # -- Metadata ---------------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to the episode.

        Args:
            key: Metadata key.
            value: Metadata value (must be JSON-serializable).
        """
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve episode metadata.

        Args:
            key: Metadata key.
            default: Value returned when key is not found.

        Returns:
            The metadata value, or *default* if not present.
        """
        return self._metadata.get(key, default)

    # -- Summary ----------------------------------------------------------------

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write an episode summary JSON file.

        Args:
            summary: Dictionary to serialize as the summary.
        """
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    # -- Statistics -------------------------------------------------------------

    def compute_statistics(self) -> EpisodeStatistics:
        """Compute aggregate statistics for the episode.

        Analyses trajectories, events, and reward history to produce an
        ``EpisodeStatistics`` instance.

        Returns:
            Aggregated episode statistics.
        """
        wall_duration = time.time() - self._wall_start
        duration_s = self._last_time_s - (self._first_time_s or 0.0)

        num_robots = sum(1 for k in self._agent_kinds.values() if k == "robot")
        num_humans = sum(1 for k in self._agent_kinds.values() if k == "human")

        # Event counts
        event_counts: dict[str, int] = {}
        for ev in self._events:
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1
        num_collisions = event_counts.get("collision", 0)

        # Per-agent stats
        all_speeds: list[float] = []
        efficiencies: list[float] = []
        total_dist = 0.0
        per_agent: dict[int, dict[str, float]] = {}

        for aid, traj in self._trajectories.items():
            td = traj.total_distance
            total_dist += td
            avg_spd = traj.average_speed
            mx_spd = traj.max_speed
            eff = traj.path_efficiency
            all_speeds.extend(p.speed for p in traj.points)
            efficiencies.append(eff)
            per_agent[aid] = {
                "total_distance": td,
                "average_speed": avg_spd,
                "max_speed": mx_spd,
                "path_efficiency": eff,
                "displacement": traj.displacement,
                "duration": traj.duration,
                "num_points": len(traj.points),
            }

        mean_speed = float(np.mean(all_speeds)) if all_speeds else 0.0
        max_speed_all = float(np.max(all_speeds)) if all_speeds else 0.0
        mean_eff = float(np.mean(efficiencies)) if efficiencies else 0.0

        stats = EpisodeStatistics(
            episode_id=self.episode_id,
            num_steps=self._step_count,
            duration_s=duration_s,
            wall_duration_s=wall_duration,
            num_agents=len(self._agent_kinds),
            num_robots=num_robots,
            num_humans=num_humans,
            num_events=len(self._events),
            num_collisions=num_collisions,
            mean_speed=mean_speed,
            max_speed=max_speed_all,
            mean_path_efficiency=mean_eff,
            total_distance_all=total_dist,
            event_counts=event_counts,
            per_agent_stats=per_agent,
        )
        return stats

    def save_statistics(self) -> EpisodeStatistics:
        """Compute statistics and write them to a JSON file.

        Returns:
            The computed statistics.
        """
        stats = self.compute_statistics()
        with self.statistics_path.open("w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, sort_keys=True)
        return stats

    # -- Trajectory access ------------------------------------------------------

    def get_trajectory(self, agent_id: int) -> AgentTrajectory | None:
        """Retrieve the trajectory for a specific agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            The trajectory object, or ``None`` if not found.
        """
        return self._trajectories.get(agent_id)

    def get_all_trajectories(self) -> dict[int, AgentTrajectory]:
        """Return all recorded trajectories.

        Returns:
            Dictionary mapping agent IDs to their trajectories.
        """
        return dict(self._trajectories)

    def save_trajectories(self) -> None:
        """Write all trajectories to a JSON file."""
        data = {str(aid): traj.to_dict() for aid, traj in self._trajectories.items()}
        with self.trajectories_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # -- Event access -----------------------------------------------------------

    def get_events(self, event_type: str | None = None) -> list[EpisodeEvent]:
        """Return recorded events, optionally filtered by type.

        Args:
            event_type: If provided, only return events of this type.

        Returns:
            List of matching episode events.
        """
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.event_type == event_type]

    def get_event_counts(self) -> dict[str, int]:
        """Count events by type.

        Returns:
            Dictionary mapping event types to their counts.
        """
        counts: dict[str, int] = {}
        for ev in self._events:
            counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
        return counts

    # -- Reward access ----------------------------------------------------------

    def get_reward_history(self, agent_id: int | None = None) -> list[dict[str, Any]]:
        """Return reward history, optionally filtered by agent.

        Args:
            agent_id: If provided, only return rewards for this agent.

        Returns:
            List of reward entry dictionaries.
        """
        if agent_id is None:
            return list(self._reward_history)
        return [r for r in self._reward_history if r["agent_id"] == agent_id]

    def get_cumulative_reward(self, agent_id: int) -> float:
        """Compute cumulative reward for a given agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            Sum of all rewards received by the agent.
        """
        return sum(r["reward"] for r in self._reward_history if r["agent_id"] == agent_id)

    # -- Export -----------------------------------------------------------------

    def export_csv(self, output_path: Path | None = None) -> Path:
        """Export state data to a CSV file.

        Each row represents one agent at one step.

        Args:
            output_path: Destination path. Defaults to ``states.csv`` in the
                bundle directory.

        Returns:
            Path to the written CSV file.
        """
        output_path = output_path or (self.bundle_dir / "states.csv")
        self._flush_state_buffer()

        fieldnames = [
            "step",
            "time_s",
            "agent_id",
            "kind",
            "x",
            "y",
            "vx",
            "vy",
            "goal_x",
            "goal_y",
            "radius",
            "max_speed",
            "behavior",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            with self.state_path.open("r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    for agent in row.get("agents", []):
                        csv_row = {
                            "step": row["step"],
                            "time_s": row["time_s"],
                            "agent_id": agent["id"],
                            "kind": agent["kind"],
                            "x": agent["x"],
                            "y": agent["y"],
                            "vx": agent["vx"],
                            "vy": agent["vy"],
                            "goal_x": agent["goal_x"],
                            "goal_y": agent["goal_y"],
                            "radius": agent["radius"],
                            "max_speed": agent["max_speed"],
                            "behavior": agent["behavior"],
                        }
                        writer.writerow(csv_row)
        return output_path

    def export_events_csv(self, output_path: Path | None = None) -> Path:
        """Export events to a CSV file.

        Args:
            output_path: Destination path. Defaults to ``events.csv`` in the
                bundle directory.

        Returns:
            Path to the written CSV file.
        """
        output_path = output_path or (self.bundle_dir / "events.csv")
        fieldnames = ["step", "time_s", "event_type", "agent_id", "wall_time", "payload"]

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ev in self._events:
                writer.writerow(
                    {
                        "step": ev.step,
                        "time_s": ev.time_s,
                        "event_type": ev.event_type,
                        "agent_id": ev.agent_id,
                        "wall_time": ev.wall_time,
                        "payload": json.dumps(ev.payload),
                    }
                )
        return output_path

    def export_json(self, output_path: Path | None = None) -> Path:
        """Export the full episode data to a single JSON file.

        Includes states, events, trajectories, rewards, metadata,
        and statistics.

        Args:
            output_path: Destination path. Defaults to ``episode.json`` in the
                bundle directory.

        Returns:
            Path to the written JSON file.
        """
        output_path = output_path or (self.bundle_dir / "episode.json")
        self._flush_state_buffer()

        # Re-read states from disk
        states = []
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        states.append(json.loads(line))

        data = {
            "episode_id": self.episode_id,
            "metadata": self._metadata,
            "states": states,
            "events": [ev.to_dict() for ev in self._events],
            "trajectories": {str(aid): traj.to_dict() for aid, traj in self._trajectories.items()},
            "rewards": self._reward_history,
            "statistics": self.compute_statistics().to_dict(),
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path

    def export_rewards_csv(self, output_path: Path | None = None) -> Path:
        """Export reward history to a CSV file.

        Args:
            output_path: Destination path. Defaults to ``rewards.csv`` in the
                bundle directory.

        Returns:
            Path to the written CSV file.
        """
        output_path = output_path or (self.bundle_dir / "rewards.csv")
        fieldnames = ["step", "time_s", "agent_id", "reward"]

        # Collect component keys
        all_component_keys: set[str] = set()
        for entry in self._reward_history:
            all_component_keys.update(entry.get("components", {}).keys())
        sorted_keys = sorted(all_component_keys)
        fieldnames.extend(sorted_keys)

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._reward_history:
                row: dict[str, Any] = {
                    "step": entry["step"],
                    "time_s": entry["time_s"],
                    "agent_id": entry["agent_id"],
                    "reward": entry["reward"],
                }
                for k in sorted_keys:
                    row[k] = entry.get("components", {}).get(k, "")
                writer.writerow(row)
        return output_path

    # -- Iteration & replay -----------------------------------------------------

    def iter_states(self) -> Iterator[dict[str, Any]]:
        """Iterate over recorded state lines from disk.

        Yields:
            Parsed state dictionaries, one per simulation step.
        """
        self._flush_state_buffer()
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def pairwise_distances(self, step: int) -> np.ndarray | None:
        """Compute pairwise distances between all agents at a given step.

        Args:
            step: The simulation step to query.

        Returns:
            Square numpy array of pairwise distances, or ``None`` if the
            step has no trajectory data.
        """
        agent_ids = sorted(self._trajectories.keys())
        n = len(agent_ids)
        if n == 0:
            return None

        positions: list[tuple[float, float]] = []
        for aid in agent_ids:
            traj = self._trajectories[aid]
            found = False
            for p in traj.points:
                if p.step == step:
                    positions.append((p.x, p.y))
                    found = True
                    break
            if not found:
                return None

        pos_arr = np.array(positions, dtype=np.float64)
        diff = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def minimum_separation(self) -> float:
        """Find the minimum separation between any two agents across all steps.

        Returns:
            Minimum distance observed, or ``float('inf')`` if fewer than
            two agents are tracked.
        """
        agent_ids = sorted(self._trajectories.keys())
        if len(agent_ids) < 2:
            return float("inf")

        min_dist = float("inf")
        steps_set: set[int] = set()
        for traj in self._trajectories.values():
            for p in traj.points:
                steps_set.add(p.step)

        for step in steps_set:
            dists = self.pairwise_distances(step)
            if dists is not None:
                np.fill_diagonal(dists, float("inf"))
                step_min = float(np.min(dists))
                min_dist = min(min_dist, step_min)

        return min_dist

    # -- Close ------------------------------------------------------------------

    def close(self) -> None:
        """Flush buffers and close all file handles.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._flush_state_buffer()
        if not self._state_f.closed:
            self._state_f.close()
        if not self._events_f.closed:
            self._events_f.close()
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Whether this logger has been closed.

        Returns:
            ``True`` if :meth:`close` has been called.
        """
        return self._closed


@contextmanager
def episode_context(
    bundle_dir: Path,
    episode_id: str | None = None,
    **kwargs: Any,
) -> Generator[EpisodeLogger, None, None]:
    """Context manager that creates, yields, and closes an EpisodeLogger.

    Args:
        bundle_dir: Directory for episode artifacts.
        episode_id: Optional episode identifier.
        **kwargs: Additional keyword arguments forwarded to ``EpisodeLogger``.

    Yields:
        An open ``EpisodeLogger`` instance.

    Example::

        with episode_context(Path("/tmp/ep")) as logger:
            logger.write_state(0, 0.0, agents)
    """
    logger = EpisodeLogger(bundle_dir, episode_id=episode_id, **kwargs)
    try:
        yield logger
    finally:
        logger.close()
