"""Trajectory dataset handling: loading, windowing, splitting, augmentation, and context.

Provides a comprehensive pipeline for loading pedestrian trajectory data from
CSV, JSON, and protobuf formats, creating sliding-window sequences for
training, splitting into train/val/test partitions, augmenting trajectories,
finding neighboring agents, and extracting scene context.
"""

from __future__ import annotations

import csv
import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from navirl.data.trajectory import Trajectory, TrajectoryCollection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryWindow:
    """A fixed-length window extracted from a trajectory.

    Attributes:
        observed: Positions for the observation horizon, shape ``(obs_len, 2)``.
        future: Positions for the prediction horizon, shape ``(pred_len, 2)``.
        observed_vel: Velocities for the observation horizon, shape ``(obs_len, 2)``.
        future_vel: Velocities for the prediction horizon, shape ``(pred_len, 2)``.
        timestamps_obs: Timestamps for the observation horizon, shape ``(obs_len,)``.
        timestamps_fut: Timestamps for the prediction horizon, shape ``(pred_len,)``.
        agent_id: The agent identifier.
        neighbor_positions: Positions of neighboring agents at each observed
            timestep, list of arrays each ``(N_t, 2)``.
        scene_context: Optional scene context array (e.g., semantic map patch).
    """

    observed: np.ndarray
    future: np.ndarray
    observed_vel: np.ndarray
    future_vel: np.ndarray
    timestamps_obs: np.ndarray
    timestamps_fut: np.ndarray
    agent_id: Union[str, int]
    neighbor_positions: List[np.ndarray] = field(default_factory=list)
    scene_context: Optional[np.ndarray] = None


@dataclass
class SplitConfig:
    """Configuration for dataset splitting.

    Attributes:
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.
        by_scene: If True, split by scene rather than by window.
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = True
    seed: int = 42
    by_scene: bool = False

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f}"
            )


@dataclass
class AugmentationConfig:
    """Configuration for trajectory data augmentation.

    Attributes:
        rotation_range: Range of rotation angles in radians ``(min, max)``.
        flip_x: Whether to apply horizontal flipping.
        flip_y: Whether to apply vertical flipping.
        scale_range: Range of scaling factors ``(min, max)``.
        noise_std: Standard deviation of Gaussian positional noise.
        speed_perturbation_range: Range of speed scaling factors ``(min, max)``.
        augmentation_factor: Number of augmented copies per original sample.
        seed: Random seed for reproducibility.
    """

    rotation_range: Tuple[float, float] = (-np.pi, np.pi)
    flip_x: bool = True
    flip_y: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.05
    speed_perturbation_range: Tuple[float, float] = (0.8, 1.2)
    augmentation_factor: int = 4
    seed: int = 42


# ---------------------------------------------------------------------------
# CSV / JSON / Protobuf loaders
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> List[Trajectory]:
    """Load trajectories from a CSV file.

    Expected columns: ``frame_id, agent_id, x, y [, vx, vy]``.
    The file may use tab or comma delimiters; both are auto-detected.

    Parameters:
        path: Path to the CSV file.

    Returns:
        List of :class:`Trajectory` objects, one per unique agent.
    """
    text = path.read_text()
    delimiter = "\t" if "\t" in text.split("\n")[0] else ","

    rows: List[List[str]] = []
    reader = csv.reader(text.strip().splitlines(), delimiter=delimiter)
    for row in reader:
        # Skip header rows that contain non-numeric first fields
        try:
            float(row[0])
        except (ValueError, IndexError):
            continue
        rows.append(row)

    if not rows:
        logger.warning("No data rows found in %s", path)
        return []

    # Parse into structured arrays
    agent_data: Dict[str, Dict[str, list]] = {}
    has_vel = len(rows[0]) >= 6

    for row in rows:
        frame_id = float(row[0])
        agent_id = row[1].strip()
        x, y = float(row[2]), float(row[3])

        if agent_id not in agent_data:
            agent_data[agent_id] = {
                "timestamps": [],
                "positions": [],
                "velocities": [],
            }
        agent_data[agent_id]["timestamps"].append(frame_id)
        agent_data[agent_id]["positions"].append([x, y])
        if has_vel:
            vx, vy = float(row[4]), float(row[5])
            agent_data[agent_id]["velocities"].append([vx, vy])

    trajectories: List[Trajectory] = []
    for aid, data in agent_data.items():
        ts = np.array(data["timestamps"], dtype=np.float64)
        pos = np.array(data["positions"], dtype=np.float64)
        vel = (
            np.array(data["velocities"], dtype=np.float64)
            if data["velocities"]
            else None
        )
        # Sort by timestamp
        order = np.argsort(ts)
        trajectories.append(
            Trajectory(
                timestamps=ts[order],
                positions=pos[order],
                velocities=vel[order] if vel is not None else None,
                agent_id=aid,
            )
        )
    return trajectories


def _load_json(path: Path) -> List[Trajectory]:
    """Load trajectories from a JSON file.

    Expected schema::

        {
            "trajectories": [
                {
                    "agent_id": "ped_1",
                    "timestamps": [0.0, 0.4, 0.8, ...],
                    "positions": [[x0, y0], [x1, y1], ...],
                    "velocities": [[vx0, vy0], ...]  // optional
                },
                ...
            ]
        }

    Parameters:
        path: Path to the JSON file.

    Returns:
        List of :class:`Trajectory` objects.
    """
    with open(path, "r") as fp:
        data = json.load(fp)

    traj_list = data if isinstance(data, list) else data.get("trajectories", [])

    trajectories: List[Trajectory] = []
    for entry in traj_list:
        ts = np.array(entry["timestamps"], dtype=np.float64)
        pos = np.array(entry["positions"], dtype=np.float64)
        vel = (
            np.array(entry["velocities"], dtype=np.float64)
            if "velocities" in entry
            else None
        )
        agent_id = entry.get("agent_id", entry.get("id", len(trajectories)))
        order = np.argsort(ts)
        trajectories.append(
            Trajectory(
                timestamps=ts[order],
                positions=pos[order],
                velocities=vel[order] if vel is not None else None,
                agent_id=agent_id,
            )
        )
    return trajectories


def _load_protobuf(path: Path) -> List[Trajectory]:
    """Load trajectories from a lightweight binary protobuf-like format.

    Binary layout per trajectory record:
        - 4 bytes: agent_id length (uint32, little-endian)
        - N bytes: agent_id string (UTF-8)
        - 4 bytes: number of timesteps T (uint32, little-endian)
        - T * 8 bytes: timestamps (float64, little-endian)
        - T * 16 bytes: positions (float64 pairs x, y)
        - 1 byte: has_velocities flag (0 or 1)
        - If has_velocities: T * 16 bytes: velocities (float64 pairs vx, vy)

    Parameters:
        path: Path to the binary file.

    Returns:
        List of :class:`Trajectory` objects.
    """
    raw = path.read_bytes()
    offset = 0
    trajectories: List[Trajectory] = []

    while offset < len(raw):
        # Agent id
        if offset + 4 > len(raw):
            break
        (id_len,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        agent_id = raw[offset : offset + id_len].decode("utf-8")
        offset += id_len

        # Number of timesteps
        (num_t,) = struct.unpack_from("<I", raw, offset)
        offset += 4

        # Timestamps
        ts = np.frombuffer(raw, dtype="<f8", count=num_t, offset=offset).copy()
        offset += num_t * 8

        # Positions
        pos = np.frombuffer(
            raw, dtype="<f8", count=num_t * 2, offset=offset
        ).reshape(num_t, 2).copy()
        offset += num_t * 16

        # Velocities flag
        if offset < len(raw):
            has_vel = raw[offset]
            offset += 1
        else:
            has_vel = 0

        vel = None
        if has_vel:
            vel = np.frombuffer(
                raw, dtype="<f8", count=num_t * 2, offset=offset
            ).reshape(num_t, 2).copy()
            offset += num_t * 16

        trajectories.append(
            Trajectory(timestamps=ts, positions=pos, velocities=vel, agent_id=agent_id)
        )

    return trajectories


# ---------------------------------------------------------------------------
# Velocity computation helper
# ---------------------------------------------------------------------------


def _compute_velocities(traj: Trajectory) -> np.ndarray:
    """Compute velocities via finite differences if not already available.

    Parameters:
        traj: Input trajectory.

    Returns:
        Velocity array of shape ``(T, 2)``.
    """
    if traj.velocities is not None:
        return traj.velocities

    T = len(traj.timestamps)
    if T < 2:
        return np.zeros((T, 2), dtype=np.float64)

    dt = np.diff(traj.timestamps)
    dt = np.where(dt == 0, 1e-6, dt)  # avoid division by zero
    dp = np.diff(traj.positions, axis=0)
    vel = dp / dt[:, None]
    # Pad first velocity to match length
    vel = np.vstack([vel[:1], vel])
    return vel


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------


def create_windows(
    trajectories: List[Trajectory],
    obs_len: int = 8,
    pred_len: int = 12,
    stride: int = 1,
    min_length: Optional[int] = None,
) -> List[TrajectoryWindow]:
    """Create sliding-window samples from a list of trajectories.

    For each trajectory of length ``T``, a window of total length
    ``obs_len + pred_len`` is slid with the given stride.

    Parameters:
        trajectories: Input trajectories.
        obs_len: Number of observation timesteps.
        pred_len: Number of future (prediction) timesteps.
        stride: Step size for sliding the window.
        min_length: Minimum trajectory length to consider; defaults to
            ``obs_len + pred_len``.

    Returns:
        List of :class:`TrajectoryWindow` instances.
    """
    total_len = obs_len + pred_len
    if min_length is None:
        min_length = total_len

    windows: List[TrajectoryWindow] = []
    for traj in trajectories:
        T = len(traj.timestamps)
        if T < min_length:
            continue
        vel = _compute_velocities(traj)
        for start in range(0, T - total_len + 1, stride):
            end_obs = start + obs_len
            end_pred = end_obs + pred_len
            windows.append(
                TrajectoryWindow(
                    observed=traj.positions[start:end_obs].copy(),
                    future=traj.positions[end_obs:end_pred].copy(),
                    observed_vel=vel[start:end_obs].copy(),
                    future_vel=vel[end_obs:end_pred].copy(),
                    timestamps_obs=traj.timestamps[start:end_obs].copy(),
                    timestamps_fut=traj.timestamps[end_obs:end_pred].copy(),
                    agent_id=traj.agent_id,
                )
            )
    logger.info(
        "Created %d windows from %d trajectories (obs=%d, pred=%d, stride=%d)",
        len(windows),
        len(trajectories),
        obs_len,
        pred_len,
        stride,
    )
    return windows


# ---------------------------------------------------------------------------
# Train / Val / Test splitting
# ---------------------------------------------------------------------------


def split_windows(
    windows: List[TrajectoryWindow],
    config: Optional[SplitConfig] = None,
) -> Tuple[List[TrajectoryWindow], List[TrajectoryWindow], List[TrajectoryWindow]]:
    """Split trajectory windows into train, validation, and test sets.

    Parameters:
        windows: Full list of trajectory windows.
        config: Split configuration; uses defaults if ``None``.

    Returns:
        Tuple of ``(train, val, test)`` window lists.
    """
    if config is None:
        config = SplitConfig()

    rng = np.random.RandomState(config.seed)
    n = len(windows)

    if config.by_scene:
        # Group windows by agent_id (proxy for scene)
        scene_ids: List[Union[str, int]] = []
        seen: Dict[Union[str, int], int] = {}
        indices_by_scene: Dict[int, List[int]] = {}
        for i, w in enumerate(windows):
            if w.agent_id not in seen:
                seen[w.agent_id] = len(seen)
                scene_ids.append(w.agent_id)
            sid = seen[w.agent_id]
            indices_by_scene.setdefault(sid, []).append(i)

        scene_order = np.arange(len(scene_ids))
        if config.shuffle:
            rng.shuffle(scene_order)

        n_scenes = len(scene_ids)
        n_train = int(n_scenes * config.train_ratio)
        n_val = int(n_scenes * config.val_ratio)

        train_scenes = set(scene_order[:n_train].tolist())
        val_scenes = set(scene_order[n_train : n_train + n_val].tolist())
        test_scenes = set(scene_order[n_train + n_val :].tolist())

        train = [windows[i] for s in train_scenes for i in indices_by_scene[s]]
        val = [windows[i] for s in val_scenes for i in indices_by_scene[s]]
        test = [windows[i] for s in test_scenes for i in indices_by_scene[s]]
    else:
        indices = np.arange(n)
        if config.shuffle:
            rng.shuffle(indices)

        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        train = [windows[i] for i in indices[:n_train]]
        val = [windows[i] for i in indices[n_train : n_train + n_val]]
        test = [windows[i] for i in indices[n_train + n_val :]]

    logger.info(
        "Split %d windows -> train=%d, val=%d, test=%d",
        n,
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def split_trajectories(
    trajectories: List[Trajectory],
    config: Optional[SplitConfig] = None,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """Split raw trajectories into train, validation, and test sets.

    Parameters:
        trajectories: Full list of trajectories.
        config: Split configuration; uses defaults if ``None``.

    Returns:
        Tuple of ``(train, val, test)`` trajectory lists.
    """
    if config is None:
        config = SplitConfig()

    rng = np.random.RandomState(config.seed)
    n = len(trajectories)
    indices = np.arange(n)
    if config.shuffle:
        rng.shuffle(indices)

    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train = [trajectories[i] for i in indices[:n_train]]
    val = [trajectories[i] for i in indices[n_train : n_train + n_val]]
    test = [trajectories[i] for i in indices[n_train + n_val :]]
    return train, val, test


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def _rotate(positions: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2-D positions by *angle* radians around the origin."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return positions @ rot.T


def _flip(positions: np.ndarray, axis: str) -> np.ndarray:
    """Flip positions along 'x' (negate y) or 'y' (negate x)."""
    out = positions.copy()
    if axis == "x":
        out[:, 1] *= -1
    else:
        out[:, 0] *= -1
    return out


def _scale(positions: np.ndarray, factor: float) -> np.ndarray:
    """Scale positions by a uniform factor."""
    return positions * factor


def _add_noise(positions: np.ndarray, std: float, rng: np.random.RandomState) -> np.ndarray:
    """Add Gaussian noise to positions."""
    return positions + rng.normal(0.0, std, size=positions.shape)


def _perturb_speed(
    positions: np.ndarray,
    timestamps: np.ndarray,
    factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perturb effective speed by interpolating positions.

    Scaling factor > 1 increases speed, < 1 decreases speed.  This is
    achieved by stretching/compressing the time axis and re-sampling
    positions at the original timestamps via linear interpolation.

    Parameters:
        positions: Shape ``(T, 2)`` positions.
        timestamps: Shape ``(T,)`` timestamps.
        factor: Speed scaling factor.

    Returns:
        Tuple of (new_positions, timestamps).
    """
    if len(timestamps) < 2 or factor == 1.0:
        return positions.copy(), timestamps.copy()

    t0 = timestamps[0]
    shifted = t0 + (timestamps - t0) / factor
    new_x = np.interp(timestamps, shifted, positions[:, 0])
    new_y = np.interp(timestamps, shifted, positions[:, 1])
    return np.column_stack([new_x, new_y]), timestamps.copy()


def augment_window(
    window: TrajectoryWindow,
    config: Optional[AugmentationConfig] = None,
    rng: Optional[np.random.RandomState] = None,
) -> TrajectoryWindow:
    """Apply a single random augmentation to a trajectory window.

    The augmentation randomly selects and applies one or more of:
    rotation, flipping, scaling, noise injection, and speed perturbation.

    Parameters:
        window: Input trajectory window.
        config: Augmentation configuration.
        rng: Random state for reproducibility.

    Returns:
        A new augmented :class:`TrajectoryWindow`.
    """
    if config is None:
        config = AugmentationConfig()
    if rng is None:
        rng = np.random.RandomState(config.seed)

    obs = window.observed.copy()
    fut = window.future.copy()
    full = np.vstack([obs, fut])
    ts_full = np.concatenate([window.timestamps_obs, window.timestamps_fut])

    # Random rotation
    angle = rng.uniform(*config.rotation_range)
    full = _rotate(full, angle)

    # Random flip
    if config.flip_x and rng.rand() < 0.5:
        full = _flip(full, "x")
    if config.flip_y and rng.rand() < 0.5:
        full = _flip(full, "y")

    # Random scale
    scale_factor = rng.uniform(*config.scale_range)
    full = _scale(full, scale_factor)

    # Noise injection
    if config.noise_std > 0:
        full = _add_noise(full, config.noise_std, rng)

    # Speed perturbation
    speed_factor = rng.uniform(*config.speed_perturbation_range)
    full, ts_full = _perturb_speed(full, ts_full, speed_factor)

    obs_len = len(window.observed)
    new_obs = full[:obs_len]
    new_fut = full[obs_len:]

    # Recompute velocities
    dt_obs = np.diff(ts_full[:obs_len])
    dt_obs = np.where(dt_obs == 0, 1e-6, dt_obs)
    new_obs_vel = np.diff(new_obs, axis=0) / dt_obs[:, None]
    new_obs_vel = np.vstack([new_obs_vel[:1], new_obs_vel])

    dt_fut = np.diff(ts_full[obs_len:])
    dt_fut = np.where(dt_fut == 0, 1e-6, dt_fut)
    new_fut_vel = np.diff(new_fut, axis=0) / dt_fut[:, None]
    new_fut_vel = np.vstack([new_fut_vel[:1], new_fut_vel])

    return TrajectoryWindow(
        observed=new_obs,
        future=new_fut,
        observed_vel=new_obs_vel,
        future_vel=new_fut_vel,
        timestamps_obs=ts_full[:obs_len],
        timestamps_fut=ts_full[obs_len:],
        agent_id=window.agent_id,
        neighbor_positions=window.neighbor_positions,
        scene_context=window.scene_context,
    )


def augment_dataset(
    windows: List[TrajectoryWindow],
    config: Optional[AugmentationConfig] = None,
) -> List[TrajectoryWindow]:
    """Augment a full dataset of windows, producing multiple augmented copies.

    Parameters:
        windows: Original trajectory windows.
        config: Augmentation configuration.

    Returns:
        Combined list of original plus augmented windows.
    """
    if config is None:
        config = AugmentationConfig()

    rng = np.random.RandomState(config.seed)
    augmented: List[TrajectoryWindow] = list(windows)

    for _ in range(config.augmentation_factor):
        for w in windows:
            augmented.append(augment_window(w, config, rng))

    logger.info(
        "Augmented %d windows by factor %d -> %d total",
        len(windows),
        config.augmentation_factor,
        len(augmented),
    )
    return augmented


# ---------------------------------------------------------------------------
# Neighbor finding
# ---------------------------------------------------------------------------


def find_neighbors(
    windows: List[TrajectoryWindow],
    radius: float = 5.0,
    max_neighbors: int = 10,
    timestamp_tolerance: float = 0.1,
) -> None:
    """Populate neighbor positions for each window based on spatial proximity.

    For each window, finds other windows whose agent was within *radius*
    meters at the same timesteps and stores their positions in
    ``window.neighbor_positions``.

    Parameters:
        windows: List of trajectory windows to update in-place.
        radius: Search radius in meters.
        max_neighbors: Maximum number of neighbors per timestep.
        timestamp_tolerance: Maximum allowed difference in timestamps for
            two agents to be considered contemporaneous.
    """
    # Build lookup: timestamp -> list of (agent_id, window_idx, position)
    ts_lookup: Dict[float, List[Tuple[Union[str, int], int, np.ndarray]]] = {}

    for w_idx, w in enumerate(windows):
        for t_idx, t in enumerate(w.timestamps_obs):
            rounded_t = round(t / timestamp_tolerance) * timestamp_tolerance
            ts_lookup.setdefault(rounded_t, []).append(
                (w.agent_id, w_idx, w.observed[t_idx])
            )

    for w_idx, w in enumerate(windows):
        neighbor_pos_per_step: List[np.ndarray] = []
        for t_idx, t in enumerate(w.timestamps_obs):
            rounded_t = round(t / timestamp_tolerance) * timestamp_tolerance
            candidates = ts_lookup.get(rounded_t, [])
            pos_self = w.observed[t_idx]
            neighbors: List[Tuple[float, np.ndarray]] = []
            for aid, cidx, cpos in candidates:
                if cidx == w_idx:
                    continue
                if aid == w.agent_id:
                    continue
                dist = np.linalg.norm(cpos - pos_self)
                if dist <= radius:
                    neighbors.append((dist, cpos))
            # Sort by distance and limit
            neighbors.sort(key=lambda x: x[0])
            if neighbors:
                nbr_arr = np.array(
                    [n[1] for n in neighbors[:max_neighbors]], dtype=np.float64
                )
            else:
                nbr_arr = np.zeros((0, 2), dtype=np.float64)
            neighbor_pos_per_step.append(nbr_arr)
        w.neighbor_positions = neighbor_pos_per_step

    logger.info("Computed neighbors for %d windows (radius=%.1f)", len(windows), radius)


# ---------------------------------------------------------------------------
# Scene context extraction
# ---------------------------------------------------------------------------


def extract_scene_context(
    windows: List[TrajectoryWindow],
    scene_map: Optional[np.ndarray] = None,
    patch_size: int = 64,
    resolution: float = 0.1,
    map_origin: Tuple[float, float] = (0.0, 0.0),
) -> None:
    """Extract local scene context patches for each trajectory window.

    If a semantic scene map is provided, a patch centered on the agent's
    last observed position is extracted and stored in
    ``window.scene_context``.  If no map is provided, a blank patch is
    stored.

    Parameters:
        windows: List of trajectory windows to update in-place.
        scene_map: 2-D or 3-D semantic map array.  For a 2-D map the
            shape is ``(H, W)``; for a 3-D map ``(H, W, C)``.
        patch_size: Side length of the extracted patch in pixels.
        resolution: Meters per pixel.
        map_origin: ``(x, y)`` world coordinates of the map's top-left
            corner.
    """
    half = patch_size // 2

    for w in windows:
        if scene_map is None:
            w.scene_context = np.zeros((patch_size, patch_size), dtype=np.float32)
            continue

        # Agent position at end of observation
        pos = w.observed[-1]
        px = int((pos[0] - map_origin[0]) / resolution)
        py = int((pos[1] - map_origin[1]) / resolution)

        h, ww = scene_map.shape[:2]
        y_start = max(0, py - half)
        y_end = min(h, py + half)
        x_start = max(0, px - half)
        x_end = min(ww, px + half)

        if scene_map.ndim == 3:
            patch = np.zeros((patch_size, patch_size, scene_map.shape[2]), dtype=np.float32)
        else:
            patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        # Compute offsets into the patch
        py_off = half - (py - y_start)
        px_off = half - (px - x_start)
        crop = scene_map[y_start:y_end, x_start:x_end]

        if crop.size > 0:
            patch[
                py_off : py_off + crop.shape[0],
                px_off : px_off + crop.shape[1],
            ] = crop.astype(np.float32)

        w.scene_context = patch

    logger.info(
        "Extracted scene context patches (%dx%d) for %d windows",
        patch_size,
        patch_size,
        len(windows),
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TrajectoryDatasetPipeline:
    """End-to-end pipeline for loading, processing, and preparing trajectory data.

    Example::

        pipeline = TrajectoryDatasetPipeline(
            obs_len=8,
            pred_len=12,
            stride=1,
            augmentation=AugmentationConfig(augmentation_factor=3),
            split=SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        )
        pipeline.load("path/to/data.csv")
        train, val, test = pipeline.get_splits()

    Parameters:
        obs_len: Observation window length.
        pred_len: Prediction window length.
        stride: Sliding window stride.
        augmentation: Augmentation configuration (``None`` to disable).
        split: Split configuration.
        neighbor_radius: Radius for neighbor search.
        max_neighbors: Maximum neighbors per timestep.
        scene_map: Optional semantic map for context extraction.
        patch_size: Scene context patch size.
        map_resolution: Scene map resolution in meters/pixel.
    """

    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        stride: int = 1,
        augmentation: Optional[AugmentationConfig] = None,
        split: Optional[SplitConfig] = None,
        neighbor_radius: float = 5.0,
        max_neighbors: int = 10,
        scene_map: Optional[np.ndarray] = None,
        patch_size: int = 64,
        map_resolution: float = 0.1,
    ) -> None:
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.augmentation = augmentation
        self.split_config = split if split is not None else SplitConfig()
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.scene_map = scene_map
        self.patch_size = patch_size
        self.map_resolution = map_resolution

        self._trajectories: List[Trajectory] = []
        self._windows: List[TrajectoryWindow] = []
        self._train: List[TrajectoryWindow] = []
        self._val: List[TrajectoryWindow] = []
        self._test: List[TrajectoryWindow] = []

    # ---- Loading ---------------------------------------------------------

    def load(self, path: Union[str, Path]) -> "TrajectoryDatasetPipeline":
        """Load trajectories from a file.

        The format is inferred from the file extension:
        ``.csv`` / ``.tsv`` -> CSV, ``.json`` -> JSON, ``.pb`` / ``.bin``
        -> protobuf.

        Parameters:
            path: Path to the data file.

        Returns:
            ``self`` for method chaining.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext in (".csv", ".tsv", ".txt"):
            self._trajectories = _load_csv(path)
        elif ext == ".json":
            self._trajectories = _load_json(path)
        elif ext in (".pb", ".bin", ".proto"):
            self._trajectories = _load_protobuf(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info("Loaded %d trajectories from %s", len(self._trajectories), path)
        return self

    def load_multiple(self, paths: Sequence[Union[str, Path]]) -> "TrajectoryDatasetPipeline":
        """Load and merge trajectories from multiple files.

        Parameters:
            paths: Sequence of file paths.

        Returns:
            ``self`` for method chaining.
        """
        all_trajs: List[Trajectory] = []
        for p in paths:
            p = Path(p)
            ext = p.suffix.lower()
            if ext in (".csv", ".tsv", ".txt"):
                all_trajs.extend(_load_csv(p))
            elif ext == ".json":
                all_trajs.extend(_load_json(p))
            elif ext in (".pb", ".bin", ".proto"):
                all_trajs.extend(_load_protobuf(p))
            else:
                logger.warning("Skipping unsupported file: %s", p)
        self._trajectories = all_trajs
        logger.info("Loaded %d trajectories from %d files", len(all_trajs), len(paths))
        return self

    # ---- Processing ------------------------------------------------------

    def process(self) -> "TrajectoryDatasetPipeline":
        """Run the full processing pipeline: windowing, neighbors, context, augment, split.

        Returns:
            ``self`` for method chaining.
        """
        # Windowing
        self._windows = create_windows(
            self._trajectories, self.obs_len, self.pred_len, self.stride
        )

        # Neighbor finding
        find_neighbors(
            self._windows, self.neighbor_radius, self.max_neighbors
        )

        # Scene context
        if self.scene_map is not None:
            extract_scene_context(
                self._windows,
                self.scene_map,
                self.patch_size,
                self.map_resolution,
            )

        # Augmentation
        if self.augmentation is not None:
            self._windows = augment_dataset(self._windows, self.augmentation)

        # Splitting
        self._train, self._val, self._test = split_windows(
            self._windows, self.split_config
        )
        return self

    # ---- Accessors -------------------------------------------------------

    def get_splits(
        self,
    ) -> Tuple[List[TrajectoryWindow], List[TrajectoryWindow], List[TrajectoryWindow]]:
        """Return the train, val, test splits.

        Returns:
            Tuple of ``(train_windows, val_windows, test_windows)``.
        """
        return self._train, self._val, self._test

    @property
    def trajectories(self) -> List[Trajectory]:
        """Raw loaded trajectories."""
        return self._trajectories

    @property
    def windows(self) -> List[TrajectoryWindow]:
        """All processed trajectory windows (before splitting)."""
        return self._windows

    def get_numpy_arrays(
        self, split: str = "train"
    ) -> Dict[str, np.ndarray]:
        """Return numpy arrays for a given split suitable for model training.

        Parameters:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            Dictionary with keys ``"obs"``, ``"fut"``, ``"obs_vel"``,
            ``"fut_vel"`` containing stacked numpy arrays.
        """
        mapping = {"train": self._train, "val": self._val, "test": self._test}
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}', expected train/val/test")

        windows = mapping[split]
        if not windows:
            return {
                "obs": np.zeros((0, self.obs_len, 2)),
                "fut": np.zeros((0, self.pred_len, 2)),
                "obs_vel": np.zeros((0, self.obs_len, 2)),
                "fut_vel": np.zeros((0, self.pred_len, 2)),
            }

        return {
            "obs": np.array([w.observed for w in windows]),
            "fut": np.array([w.future for w in windows]),
            "obs_vel": np.array([w.observed_vel for w in windows]),
            "fut_vel": np.array([w.future_vel for w in windows]),
        }

    def iterate_batches(
        self, split: str = "train", batch_size: int = 64, shuffle: bool = True
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over mini-batches of trajectory windows.

        Parameters:
            split: One of ``"train"``, ``"val"``, ``"test"``.
            batch_size: Number of windows per batch.
            shuffle: Whether to shuffle each epoch.

        Yields:
            Dictionary with numpy arrays for the batch.
        """
        arrays = self.get_numpy_arrays(split)
        n = arrays["obs"].shape[0]
        if n == 0:
            return

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield {k: v[batch_idx] for k, v in arrays.items()}

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of the pipeline state.

        Returns:
            Dictionary with counts and configuration info.
        """
        return {
            "num_trajectories": len(self._trajectories),
            "num_windows": len(self._windows),
            "num_train": len(self._train),
            "num_val": len(self._val),
            "num_test": len(self._test),
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "stride": self.stride,
            "augmentation_enabled": self.augmentation is not None,
        }
