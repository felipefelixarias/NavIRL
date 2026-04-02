"""Standard pedestrian trajectory datasets: ETH/UCY, generic social navigation."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from navirl.data.trajectory import Trajectory, TrajectoryCollection


class TrajectoryDataset(ABC):
    """Base class for trajectory datasets.

    Subclasses must implement :meth:`_load_raw` which populates
    ``self._scenes`` from the data source.
    """

    def __init__(self) -> None:
        self._scenes: list[TrajectoryCollection] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Load data from *path* (file or directory).

        Parameters:
            path: Filesystem path to the dataset root.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        self._scenes = self._load_raw(path)
        self._loaded = True

    def get_scene(self, idx: int) -> TrajectoryCollection:
        """Return the trajectory collection for scene *idx*.

        Parameters:
            idx: Zero-based scene index.
        """
        self._check_loaded()
        return self._scenes[idx]

    def trajectories(self) -> TrajectoryCollection:
        """Return all trajectories across all scenes as a single collection."""
        self._check_loaded()
        all_trajs: list[Trajectory] = []
        for scene in self._scenes:
            all_trajs.extend(scene.trajectories)
        return TrajectoryCollection(all_trajs)

    def to_numpy(self) -> np.ndarray:
        """Stack all positions from all scenes into a single ``(N, 2)`` array."""
        return self.trajectories().to_numpy()

    def train_test_split(
        self, test_ratio: float = 0.2, seed: int = 42
    ) -> tuple[list[TrajectoryCollection], list[TrajectoryCollection]]:
        """Split scenes into train / test sets.

        Parameters:
            test_ratio: Fraction of scenes for the test set.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_scenes, test_scenes).
        """
        self._check_loaded()
        rng = np.random.default_rng(seed)
        n = len(self._scenes)
        n_test = max(1, int(n * test_ratio))
        indices = rng.permutation(n)
        test_idx = set(indices[:n_test].tolist())
        train = [s for i, s in enumerate(self._scenes) if i not in test_idx]
        test = [s for i, s in enumerate(self._scenes) if i in test_idx]
        return train, test

    @property
    def num_scenes(self) -> int:
        """Number of scenes loaded."""
        self._check_loaded()
        return len(self._scenes)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call .load(path) first.")

    @abstractmethod
    def _load_raw(self, path: Path) -> list[TrajectoryCollection]:
        """Parse raw data from *path* and return a list of scene collections."""


# ======================================================================
# Concrete datasets
# ======================================================================

_ETH_UCY_SCENES = ("eth", "hotel", "univ", "zara1", "zara2")


class ETHUCYDataset(TrajectoryDataset):
    """Loader for the ETH/UCY pedestrian trajectory datasets.

    Expects the standard directory layout where each scene has a
    ``<scene>.txt`` file with whitespace-separated columns:
    ``frame_id  pedestrian_id  x  y``.

    Supported scenes: ``eth``, ``hotel``, ``univ``, ``zara1``, ``zara2``.

    Parameters:
        scenes: Subset of scenes to load.  Defaults to all five.
        delim: Column delimiter in the text files.  ``"auto"`` tries tab then space.
    """

    def __init__(
        self,
        scenes: Sequence[str] = _ETH_UCY_SCENES,
        delim: str = "auto",
    ) -> None:
        super().__init__()
        for s in scenes:
            if s not in _ETH_UCY_SCENES:
                raise ValueError(f"Unknown ETH/UCY scene '{s}'. Choose from {_ETH_UCY_SCENES}")
        self.scene_names = list(scenes)
        self.delim = delim

    def _load_raw(self, path: Path) -> list[TrajectoryCollection]:
        collections: list[TrajectoryCollection] = []
        for scene_name in self.scene_names:
            scene_file = self._find_scene_file(path, scene_name)
            if scene_file is None:
                raise FileNotFoundError(
                    f"Could not find data file for scene '{scene_name}' under {path}"
                )
            collections.append(self._parse_scene_file(scene_file))
        return collections

    def _find_scene_file(self, root: Path, scene: str) -> Path | None:
        """Locate the trajectory text file for a given scene name."""
        candidates = [
            root / f"{scene}.txt",
            root / scene / f"{scene}.txt",
            root / scene / "true_pos_.csv",
            root / f"{scene}_true_pos_.csv",
        ]
        for c in candidates:
            if c.is_file():
                return c
        # Fallback: search recursively
        for p in root.rglob(f"{scene}*"):
            if p.is_file() and p.suffix in (".txt", ".csv"):
                return p
        return None

    def _parse_scene_file(self, filepath: Path) -> TrajectoryCollection:
        """Parse a single ETH/UCY scene file into a :class:`TrajectoryCollection`."""
        rows: list[tuple[float, int, float, float]] = []
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if self.delim == "auto" else line.split(self.delim)
                if len(parts) < 4 and self.delim == "auto":
                    parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    frame = float(parts[0])
                    ped_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    rows.append((frame, ped_id, x, y))
                except (ValueError, IndexError):
                    continue

        # Group by pedestrian id
        grouped: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for frame, ped_id, x, y in rows:
            grouped[ped_id].append((frame, x, y))

        tc = TrajectoryCollection()
        for ped_id, data in sorted(grouped.items()):
            data.sort(key=lambda r: r[0])
            arr = np.array(data, dtype=np.float64)
            tc.add(
                Trajectory(
                    timestamps=arr[:, 0],
                    positions=arr[:, 1:3],
                    agent_id=ped_id,
                )
            )
        return tc


class SocialDataset(TrajectoryDataset):
    """Generic social navigation dataset loader.

    Reads CSV files with columns: ``timestamp, agent_id, x, y [, vx, vy]``.

    Parameters:
        has_header: Whether the CSV files have a header row.
    """

    def __init__(self, has_header: bool = True) -> None:
        super().__init__()
        self.has_header = has_header

    def _load_raw(self, path: Path) -> list[TrajectoryCollection]:
        path = Path(path)
        if path.is_file():
            files = [path]
        else:
            files = sorted(path.glob("*.csv"))
            if not files:
                files = sorted(path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No CSV/TXT files found under {path}")
        return [self._parse_csv(f) for f in files]

    def _parse_csv(self, filepath: Path) -> TrajectoryCollection:
        grouped: dict[Any, list[tuple[float, float, float, float | None, float | None]]] = (
            defaultdict(list)
        )
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if self.has_header:
                next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    ts = float(row[0])
                    agent_id = row[1].strip()
                    x = float(row[2])
                    y = float(row[3])
                    vx = float(row[4]) if len(row) > 4 and row[4].strip() else None
                    vy = float(row[5]) if len(row) > 5 and row[5].strip() else None
                    grouped[agent_id].append((ts, x, y, vx, vy))
                except (ValueError, IndexError):
                    continue

        tc = TrajectoryCollection()
        for agent_id, data in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            data.sort(key=lambda r: r[0])
            arr = np.array([(t, x, y) for t, x, y, _, _ in data], dtype=np.float64)
            has_vel = all(vx is not None and vy is not None for _, _, _, vx, vy in data)
            vel = None
            if has_vel:
                vel = np.array([(vx, vy) for _, _, _, vx, vy in data], dtype=np.float64)
            tc.add(
                Trajectory(
                    timestamps=arr[:, 0],
                    positions=arr[:, 1:3],
                    velocities=vel,
                    agent_id=agent_id,
                )
            )
        return tc
