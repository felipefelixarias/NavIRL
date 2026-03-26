"""Data augmentation for pedestrian trajectories."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from navirl.data.trajectory import Trajectory


def rotate_trajectory(traj: Trajectory, angle: float) -> Trajectory:
    """Rotate trajectory positions around the origin by *angle* radians.

    Parameters:
        traj: Input trajectory.
        angle: Rotation angle in radians (counter-clockwise positive).

    Returns:
        A new trajectory with rotated positions and velocities.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    new_pos = traj.positions @ rot.T
    new_vel = traj.velocities @ rot.T if traj.velocities is not None else None
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=new_pos,
        velocities=new_vel,
        agent_id=traj.agent_id,
    )


def mirror_trajectory(traj: Trajectory, axis: str = "x") -> Trajectory:
    """Mirror trajectory positions along an axis.

    Parameters:
        traj: Input trajectory.
        axis: ``"x"`` to flip the y-coordinates, ``"y"`` to flip the x-coordinates.

    Returns:
        A new mirrored trajectory.
    """
    new_pos = traj.positions.copy()
    new_vel = traj.velocities.copy() if traj.velocities is not None else None
    if axis == "x":
        new_pos[:, 1] *= -1
        if new_vel is not None:
            new_vel[:, 1] *= -1
    elif axis == "y":
        new_pos[:, 0] *= -1
        if new_vel is not None:
            new_vel[:, 0] *= -1
    else:
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=new_pos,
        velocities=new_vel,
        agent_id=traj.agent_id,
    )


def scale_trajectory(traj: Trajectory, factor: float) -> Trajectory:
    """Scale trajectory positions by *factor*.

    Parameters:
        traj: Input trajectory.
        factor: Multiplicative scale factor applied to positions.

    Returns:
        A new scaled trajectory.
    """
    new_pos = traj.positions * factor
    new_vel = traj.velocities * factor if traj.velocities is not None else None
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=new_pos,
        velocities=new_vel,
        agent_id=traj.agent_id,
    )


def add_noise(traj: Trajectory, std: float, seed: int | None = None) -> Trajectory:
    """Add Gaussian noise to trajectory positions.

    Parameters:
        traj: Input trajectory.
        std: Standard deviation of the noise.
        seed: Optional random seed for reproducibility.

    Returns:
        A new trajectory with noise added to positions.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=traj.positions.shape)
    return Trajectory(
        timestamps=traj.timestamps.copy(),
        positions=traj.positions + noise,
        velocities=traj.velocities,
        agent_id=traj.agent_id,
    )


def time_warp(traj: Trajectory, factor: float) -> Trajectory:
    """Warp the time axis of a trajectory by *factor*.

    A factor > 1 stretches (slows), < 1 compresses (speeds up).

    Parameters:
        traj: Input trajectory.
        factor: Multiplicative factor applied to timestamps relative to start.

    Returns:
        A new trajectory with warped timestamps.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    t0 = traj.timestamps[0]
    new_ts = t0 + (traj.timestamps - t0) * factor
    new_vel = None
    if traj.velocities is not None:
        new_vel = traj.velocities / factor
    return Trajectory(
        timestamps=new_ts,
        positions=traj.positions.copy(),
        velocities=new_vel,
        agent_id=traj.agent_id,
    )


def crop_trajectory(traj: Trajectory, start: int, end: int) -> Trajectory:
    """Crop a trajectory to indices ``[start, end)``.

    Parameters:
        traj: Input trajectory.
        start: Start index (inclusive).
        end: End index (exclusive).

    Returns:
        A new cropped trajectory.
    """
    vel = traj.velocities[start:end] if traj.velocities is not None else None
    return Trajectory(
        timestamps=traj.timestamps[start:end].copy(),
        positions=traj.positions[start:end].copy(),
        velocities=vel.copy() if vel is not None else None,
        agent_id=traj.agent_id,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Type alias for an augmentation function: Trajectory -> Trajectory
AugmentFn = Callable[[Trajectory], Trajectory]


class AugmentationPipeline:
    """Chain multiple augmentation transforms and apply them sequentially.

    Example::

        pipe = AugmentationPipeline()
        pipe.add(lambda t: rotate_trajectory(t, np.pi / 4))
        pipe.add(lambda t: add_noise(t, 0.01))
        augmented = pipe(trajectory)
    """

    def __init__(self, transforms: Sequence[AugmentFn] | None = None) -> None:
        self._transforms: list[AugmentFn] = list(transforms) if transforms else []

    def add(self, fn: AugmentFn) -> AugmentationPipeline:
        """Append an augmentation function to the pipeline.

        Returns ``self`` for chaining.
        """
        self._transforms.append(fn)
        return self

    def __call__(self, traj: Trajectory) -> Trajectory:
        """Apply all transforms sequentially to *traj*."""
        result = traj
        for fn in self._transforms:
            result = fn(result)
        return result

    def __len__(self) -> int:
        return len(self._transforms)
