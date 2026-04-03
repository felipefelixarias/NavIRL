"""Data loaders for NavIRL logs, ROS bags, and generic CSVs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from navirl.data.trajectory import Trajectory, TrajectoryCollection


class NavIRLLogLoader:
    """Load NavIRL episode logs produced by the simulator.

    Expects a directory (or single file) containing ``state.jsonl`` and
    optionally ``events.jsonl``.

    Attributes:
        states: Parsed state rows.
        events: Parsed event rows (empty list if file not found).
    """

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def load(self, path: str | Path) -> NavIRLLogLoader:
        """Load state and event logs from *path*.

        Parameters:
            path: Path to an episode directory or a ``state.jsonl`` file.

        Returns:
            ``self`` for chaining.
        """
        path = Path(path)
        if path.is_file():
            state_file = path
            event_file = path.parent / "events.jsonl"
        else:
            state_file = path / "state.jsonl"
            event_file = path / "events.jsonl"

        if not state_file.is_file():
            raise FileNotFoundError(f"State log not found: {state_file}")

        self.states = self._read_jsonl(state_file)
        self.events = self._read_jsonl(event_file) if event_file.is_file() else []
        return self

    def to_trajectories(self) -> TrajectoryCollection:
        """Convert loaded states to a :class:`TrajectoryCollection`.

        Each unique ``agent_id`` found in the state rows produces one trajectory.
        State rows are expected to have ``t``, ``x``, ``y`` (and optionally
        ``vx``, ``vy``, ``agent_id``) keys.

        Returns:
            A :class:`TrajectoryCollection` with one trajectory per agent.
        """
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.states:
            aid = str(row.get("agent_id", "robot"))
            grouped[aid].append(row)

        tc = TrajectoryCollection()
        for aid, rows in sorted(grouped.items()):
            rows.sort(key=lambda r: r.get("t", 0.0))
            ts = np.array([r.get("t", 0.0) for r in rows], dtype=np.float64)
            pos = np.array([[r.get("x", 0.0), r.get("y", 0.0)] for r in rows], dtype=np.float64)
            vel = None
            if "vx" in rows[0] and "vy" in rows[0]:
                vel = np.array(
                    [[r.get("vx", 0.0), r.get("vy", 0.0)] for r in rows],
                    dtype=np.float64,
                )
            tc.add(Trajectory(timestamps=ts, positions=pos, velocities=vel, agent_id=aid))
        return tc

    @staticmethod
    def _read_jsonl(filepath: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    rows.append(json.loads(stripped_line))
        return rows


class ROSBagLoader:
    """Stub loader for ROS bag files.

    Raises :class:`ImportError` with a clear message if ``rosbag`` is not
    installed.
    """

    def __init__(self) -> None:
        try:
            import rosbag  # noqa: F401

            self._rosbag = rosbag
        except ImportError:
            self._rosbag = None

    def load(self, path: str | Path, topic: str = "/trajectory") -> TrajectoryCollection:
        """Load trajectories from a ROS bag file.

        Parameters:
            path: Path to the ``.bag`` file.
            topic: ROS topic name to read trajectory messages from.

        Returns:
            A :class:`TrajectoryCollection`.

        Raises:
            ImportError: If the ``rosbag`` package is not installed.
        """
        if self._rosbag is None:
            raise ImportError(
                "ROSBagLoader requires the 'rosbag' package. "
                "Install it with: pip install rosbag  (or install ROS)"
            )
        bag = self._rosbag.Bag(str(path), "r")
        timestamps: list[float] = []
        positions: list[list[float]] = []
        try:
            for _topic, msg, t in bag.read_messages(topics=[topic]):
                timestamps.append(t.to_sec())
                positions.append([msg.pose.position.x, msg.pose.position.y])
        finally:
            bag.close()

        tc = TrajectoryCollection()
        if timestamps:
            tc.add(
                Trajectory(
                    timestamps=np.array(timestamps),
                    positions=np.array(positions),
                    agent_id="ros_agent",
                )
            )
        return tc


class GenericCSVLoader:
    """Load trajectory data from generic CSV files.

    Parameters:
        timestamp_col: Column name or index for timestamps.
        x_col: Column name or index for x-coordinates.
        y_col: Column name or index for y-coordinates.
        agent_col: Column name or index for agent ID (optional).
        delimiter: CSV delimiter character.
    """

    def __init__(
        self,
        timestamp_col: str | int = 0,
        x_col: str | int = 1,
        y_col: str | int = 2,
        agent_col: str | int | None = None,
        delimiter: str = ",",
    ) -> None:
        self.timestamp_col = timestamp_col
        self.x_col = x_col
        self.y_col = y_col
        self.agent_col = agent_col
        self.delimiter = delimiter

    def load(self, path: str | Path) -> TrajectoryCollection:
        """Load trajectories from a CSV file or directory of CSV files.

        Parameters:
            path: Path to a CSV file or directory containing CSV files.

        Returns:
            A :class:`TrajectoryCollection`.
        """
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
        else:
            files = [path]

        tc = TrajectoryCollection()
        for f in files:
            tc_part = self._load_file(f)
            for t in tc_part:
                tc.add(t)
        return tc

    def _load_file(self, filepath: Path) -> TrajectoryCollection:
        grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            header: list[str] | None = None
            # Determine if columns are specified by name (str) or index (int).
            use_names = isinstance(self.timestamp_col, str)
            if use_names:
                header = next(reader, None)
                if header is None:
                    return TrajectoryCollection()
                col_map = {name.strip(): idx for idx, name in enumerate(header)}
                t_idx = col_map.get(str(self.timestamp_col), 0)
                x_idx = col_map.get(str(self.x_col), 1)
                y_idx = col_map.get(str(self.y_col), 2)
                a_idx = col_map.get(str(self.agent_col)) if self.agent_col is not None else None
            else:
                t_idx = int(self.timestamp_col)
                x_idx = int(self.x_col)
                y_idx = int(self.y_col)
                a_idx = int(self.agent_col) if self.agent_col is not None else None

            for row in reader:
                try:
                    ts = float(row[t_idx])
                    x = float(row[x_idx])
                    y = float(row[y_idx])
                    aid = row[a_idx].strip() if a_idx is not None else "agent_0"
                    grouped[aid].append((ts, x, y))
                except (ValueError, IndexError):
                    continue

        tc = TrajectoryCollection()
        for aid, data in sorted(grouped.items()):
            data.sort(key=lambda r: r[0])
            arr = np.array(data, dtype=np.float64)
            tc.add(
                Trajectory(
                    timestamps=arr[:, 0],
                    positions=arr[:, 1:3],
                    agent_id=aid,
                )
            )
        return tc


class BatchLoader:
    """Batch trajectories for training loops.

    Parameters:
        collection: Source trajectory collection.
        batch_size: Number of trajectories per batch.
        shuffle: Whether to shuffle before batching.
        seed: Random seed when shuffle is True.
    """

    def __init__(
        self,
        collection: TrajectoryCollection,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.collection = collection
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.collection)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[Trajectory]]:
        """Yield batches of trajectories."""
        indices = np.arange(len(self.collection))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield [self.collection[int(i)] for i in batch_idx]
