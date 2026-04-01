"""Trajectory processing utilities: dataclasses, interpolation, smoothing, resampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    """A single agent trajectory with timestamps, positions, and optional velocities.

    Attributes:
        timestamps: 1-D array of shape ``(T,)`` with monotonically increasing times.
        positions: 2-D array of shape ``(T, 2)`` with (x, y) positions.
        velocities: Optional 2-D array of shape ``(T, 2)`` with (vx, vy).
        agent_id: Identifier for the agent this trajectory belongs to.
    """

    timestamps: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None = None
    agent_id: str | int = ""

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.velocities is not None:
            self.velocities = np.asarray(self.velocities, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Total duration of the trajectory in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass
class TrajectoryCollection:
    """A queryable collection of :class:`Trajectory` objects.

    Attributes:
        trajectories: List of trajectory instances.
    """

    trajectories: list[Trajectory] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]

    def __iter__(self):
        return iter(self.trajectories)

    def add(self, traj: Trajectory) -> None:
        """Append a trajectory to the collection."""
        self.trajectories.append(traj)

    def filter_by_agent(self, agent_id: str | int) -> TrajectoryCollection:
        """Return a new collection containing only trajectories for *agent_id*."""
        return TrajectoryCollection(
            [t for t in self.trajectories if t.agent_id == agent_id]
        )

    def filter_by_duration(self, min_duration: float) -> TrajectoryCollection:
        """Return trajectories whose duration is at least *min_duration* seconds."""
        return TrajectoryCollection(
            [t for t in self.trajectories if t.duration >= min_duration]
        )

    def to_numpy(self) -> np.ndarray:
        """Stack all positions into a single ``(N, 2)`` array."""
        if not self.trajectories:
            return np.empty((0, 2), dtype=np.float64)
        return np.concatenate([t.positions for t in self.trajectories], axis=0)

    @property
    def agent_ids(self) -> list[str | int]:
        """Unique agent IDs present in the collection."""
        seen: dict[str | int, None] = {}
        for t in self.trajectories:
            seen.setdefault(t.agent_id, None)
        return list(seen.keys())


# ---------------------------------------------------------------------------
# Functional utilities
# ---------------------------------------------------------------------------


def interpolate(trajectory: Trajectory, dt: float) -> Trajectory:
    """Resample *trajectory* to uniform timestep *dt* via linear interpolation.

    Parameters:
        trajectory: Source trajectory.
        dt: Desired uniform time interval in seconds.

    Returns:
        A new :class:`Trajectory` with uniformly spaced timestamps.
    """
    if len(trajectory) < 2:
        return trajectory
    t_old = trajectory.timestamps
    t_new = np.arange(t_old[0], t_old[-1], dt)
    if len(t_new) == 0:
        t_new = t_old[:1]
    pos_new = np.column_stack(
        [np.interp(t_new, t_old, trajectory.positions[:, i]) for i in range(2)]
    )
    vel_new = None
    if trajectory.velocities is not None:
        vel_new = np.column_stack(
            [np.interp(t_new, t_old, trajectory.velocities[:, i]) for i in range(2)]
        )
    return Trajectory(
        timestamps=t_new,
        positions=pos_new,
        velocities=vel_new,
        agent_id=trajectory.agent_id,
    )


def smooth(trajectory: Trajectory, window: int = 5) -> Trajectory:
    """Apply a moving-average filter with *window* size to positions.

    Parameters:
        trajectory: Source trajectory.
        window: Kernel size (must be odd and positive).

    Returns:
        A new :class:`Trajectory` with smoothed positions.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if window % 2 == 0:
        window += 1
    if len(trajectory) <= window:
        return trajectory
    kernel = np.ones(window) / window
    smoothed = np.column_stack(
        [np.convolve(trajectory.positions[:, i], kernel, mode="same") for i in range(2)]
    )
    return Trajectory(
        timestamps=trajectory.timestamps.copy(),
        positions=smoothed,
        velocities=trajectory.velocities,
        agent_id=trajectory.agent_id,
    )


def compute_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocities from positions using finite differences.

    Parameters:
        positions: Array of shape ``(T, 2)``.
        dt: Time interval between consecutive positions.

    Returns:
        Velocity array of shape ``(T, 2)`` (forward difference, last entry repeated).
    """
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return np.zeros_like(positions)
    vel = np.diff(positions, axis=0) / dt
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    return vel


def compute_accelerations(velocities: np.ndarray, dt: float) -> np.ndarray:
    """Compute accelerations from velocities using finite differences.

    Parameters:
        velocities: Array of shape ``(T, 2)``.
        dt: Time interval between consecutive velocity samples.

    Returns:
        Acceleration array of shape ``(T, 2)``.
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    if len(velocities) < 2:
        return np.zeros_like(velocities)
    acc = np.diff(velocities, axis=0) / dt
    acc = np.concatenate([acc, acc[-1:]], axis=0)
    return acc


def resample(trajectory: Trajectory, new_dt: float) -> Trajectory:
    """Alias for :func:`interpolate` with a different timestep."""
    return interpolate(trajectory, new_dt)


def align_trajectories(traj_list: Sequence[Trajectory]) -> list[Trajectory]:
    """Temporally align a list of trajectories to a common time window.

    Clips all trajectories to the overlapping time interval and resamples them
    to the finest common resolution.

    Parameters:
        traj_list: Trajectories to align.

    Returns:
        List of aligned trajectories covering the same time range.
    """
    if not traj_list:
        return []
    t_start = max(t.timestamps[0] for t in traj_list)
    t_end = min(t.timestamps[-1] for t in traj_list)
    if t_start >= t_end:
        return [
            Trajectory(
                timestamps=np.array([t_start]),
                positions=np.zeros((1, 2)),
                agent_id=t.agent_id,
            )
            for t in traj_list
        ]
    # Use the finest dt among all trajectories.
    dts = []
    for t in traj_list:
        if len(t.timestamps) >= 2:
            dts.append(float(np.min(np.diff(t.timestamps))))
    dt = min(dts) if dts else 0.1

    aligned: list[Trajectory] = []
    for traj in traj_list:
        cropped = crop_to_region(
            TrajectoryCollection([traj]),
            bounds=None,
            time_bounds=(t_start, t_end),
        )
        if cropped.trajectories:
            aligned.append(interpolate(cropped.trajectories[0], dt))
        else:
            aligned.append(
                Trajectory(
                    timestamps=np.array([t_start]),
                    positions=np.zeros((1, 2)),
                    agent_id=traj.agent_id,
                )
            )
    return aligned


def crop_to_region(
    trajectories: TrajectoryCollection,
    bounds: tuple[float, float, float, float] | None = None,
    *,
    time_bounds: tuple[float, float] | None = None,
) -> TrajectoryCollection:
    """Crop trajectories to a spatial and/or temporal bounding box.

    Parameters:
        trajectories: Input collection.
        bounds: Optional ``(x_min, y_min, x_max, y_max)`` spatial bounds.
        time_bounds: Optional ``(t_min, t_max)`` temporal bounds.

    Returns:
        A new :class:`TrajectoryCollection` with cropped trajectories.
    """
    result: list[Trajectory] = []
    for traj in trajectories:
        mask = np.ones(len(traj), dtype=bool)
        if bounds is not None:
            x_min, y_min, x_max, y_max = bounds
            mask &= (
                (traj.positions[:, 0] >= x_min)
                & (traj.positions[:, 0] <= x_max)
                & (traj.positions[:, 1] >= y_min)
                & (traj.positions[:, 1] <= y_max)
            )
        if time_bounds is not None:
            t_min, t_max = time_bounds
            mask &= (traj.timestamps >= t_min) & (traj.timestamps <= t_max)
        if np.any(mask):
            vel = traj.velocities[mask] if traj.velocities is not None else None
            result.append(
                Trajectory(
                    timestamps=traj.timestamps[mask],
                    positions=traj.positions[mask],
                    velocities=vel,
                    agent_id=traj.agent_id,
                )
            )
    return TrajectoryCollection(result)
