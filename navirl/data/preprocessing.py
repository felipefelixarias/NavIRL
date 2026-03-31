"""Data preprocessing: normalization, feature computation, observation building."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from navirl.data.trajectory import Trajectory, TrajectoryCollection


def normalize_positions(
    trajectories: TrajectoryCollection,
    method: Literal["minmax", "standard"] = "minmax",
) -> tuple[TrajectoryCollection, dict[str, np.ndarray]]:
    """Normalize positions across all trajectories in-place.

    Parameters:
        trajectories: Collection of trajectories to normalize.
        method: ``"minmax"`` scales to [0, 1]; ``"standard"`` zero-mean unit-variance.

    Returns:
        Tuple of (normalized collection, stats dict).
        The stats dict contains the parameters needed to reverse the normalization
        (``"min"``/``"max"`` for minmax, ``"mean"``/``"std"`` for standard).
    """
    all_pos = trajectories.to_numpy()
    if len(all_pos) == 0:
        return trajectories, {}

    stats: dict[str, np.ndarray] = {}
    if method == "minmax":
        p_min = all_pos.min(axis=0)
        p_max = all_pos.max(axis=0)
        denom = p_max - p_min
        denom[denom == 0] = 1.0
        stats["min"] = p_min
        stats["max"] = p_max
        normed = [
            Trajectory(
                timestamps=t.timestamps.copy(),
                positions=(t.positions - p_min) / denom,
                velocities=t.velocities,
                agent_id=t.agent_id,
            )
            for t in trajectories
        ]
    elif method == "standard":
        mean = all_pos.mean(axis=0)
        std = all_pos.std(axis=0)
        std[std == 0] = 1.0
        stats["mean"] = mean
        stats["std"] = std
        normed = [
            Trajectory(
                timestamps=t.timestamps.copy(),
                positions=(t.positions - mean) / std,
                velocities=t.velocities,
                agent_id=t.agent_id,
            )
            for t in trajectories
        ]
    else:
        raise ValueError(f"Unknown normalization method '{method}'")

    return TrajectoryCollection(normed), stats


def compute_social_features(
    ego_traj: Trajectory,
    neighbor_trajs: Sequence[Trajectory],
    *,
    max_neighbors: int = 6,
) -> np.ndarray:
    """Compute social interaction features between the ego agent and its neighbours.

    For each timestep in *ego_traj* the features encode the relative positions and
    velocities of the nearest *max_neighbors* neighbours (zero-padded if fewer exist).

    Parameters:
        ego_traj: Ego agent trajectory.
        neighbor_trajs: Trajectories of neighbouring agents.
        max_neighbors: Maximum number of neighbours to encode.

    Returns:
        Feature array of shape ``(T, max_neighbors * 4)`` where each neighbour
        contributes ``(dx, dy, dvx, dvy)``.
    """
    T = len(ego_traj)
    feat_dim = max_neighbors * 4
    features = np.zeros((T, feat_dim), dtype=np.float64)

    ego_vel = ego_traj.velocities if ego_traj.velocities is not None else np.zeros((T, 2))

    for t_idx in range(T):
        ego_pos = ego_traj.positions[t_idx]
        ego_v = ego_vel[t_idx]
        ts = ego_traj.timestamps[t_idx]

        # Gather neighbour states at this timestamp (nearest timestamp match).
        neighbours: list[tuple[float, np.ndarray, np.ndarray]] = []
        for ntraj in neighbor_trajs:
            if len(ntraj) == 0:
                continue
            closest_idx = int(np.argmin(np.abs(ntraj.timestamps - ts)))
            n_pos = ntraj.positions[closest_idx]
            n_vel = ntraj.velocities[closest_idx] if ntraj.velocities is not None else np.zeros(2)
            dist = float(np.linalg.norm(n_pos - ego_pos))
            neighbours.append((dist, n_pos, n_vel))

        # Sort by distance and take closest.
        neighbours.sort(key=lambda x: x[0])
        for n_idx, (_, n_pos, n_vel) in enumerate(neighbours[:max_neighbors]):
            offset = n_idx * 4
            features[t_idx, offset : offset + 2] = n_pos - ego_pos
            features[t_idx, offset + 2 : offset + 4] = n_vel - ego_v

    return features


def compute_map_features(
    position: np.ndarray,
    occupancy_grid: np.ndarray,
    *,
    patch_size: int = 16,
    resolution: float = 0.1,
) -> np.ndarray:
    """Extract a local occupancy patch around *position*.

    Parameters:
        position: ``(x, y)`` world coordinates of the query point.
        occupancy_grid: 2-D binary occupancy grid (1 = occupied).
        patch_size: Side length (in grid cells) of the local patch to extract.
        resolution: Meters per grid cell.

    Returns:
        Flattened occupancy patch of shape ``(patch_size * patch_size,)``.
    """
    position = np.asarray(position, dtype=np.float64)
    h, w = occupancy_grid.shape[:2]
    cx = int(round(position[0] / resolution)) + w // 2
    cy = int(round(position[1] / resolution)) + h // 2
    half = patch_size // 2

    # Pad grid for boundary safety
    padded = np.zeros((h + patch_size, w + patch_size), dtype=occupancy_grid.dtype)
    padded[half : half + h, half : half + w] = occupancy_grid
    cx += half
    cy += half

    patch = padded[cy - half : cy + half, cx - half : cx + half]
    return patch.flatten().astype(np.float64)


def encode_goal(goal_pos: np.ndarray, current_pos: np.ndarray) -> np.ndarray:
    """Encode the goal as a relative vector and distance from the current position.

    Parameters:
        goal_pos: ``(x, y)`` goal position.
        current_pos: ``(x, y)`` current agent position.

    Returns:
        Array of shape ``(3,)``: ``(dx, dy, distance)``.
    """
    goal_pos = np.asarray(goal_pos, dtype=np.float64)
    current_pos = np.asarray(current_pos, dtype=np.float64)
    delta = goal_pos - current_pos
    dist = float(np.linalg.norm(delta))
    return np.array([delta[0], delta[1], dist], dtype=np.float64)


def build_observation(
    ego_state: np.ndarray,
    neighbors: np.ndarray,
    goal: np.ndarray,
    map_data: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate sub-observations into a single flat feature vector.

    Parameters:
        ego_state: Ego agent state vector (e.g. position + velocity).
        neighbors: Social features from :func:`compute_social_features` (single timestep).
        goal: Goal encoding from :func:`encode_goal`.
        map_data: Optional local map features from :func:`compute_map_features`.

    Returns:
        1-D concatenated observation vector.
    """
    parts = [
        np.asarray(ego_state, dtype=np.float64).ravel(),
        np.asarray(neighbors, dtype=np.float64).ravel(),
        np.asarray(goal, dtype=np.float64).ravel(),
    ]
    if map_data is not None:
        parts.append(np.asarray(map_data, dtype=np.float64).ravel())
    return np.concatenate(parts)
