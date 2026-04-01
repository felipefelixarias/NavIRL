"""
Demonstration Dataset
=====================

Utilities for loading, preprocessing, and serving expert demonstrations in
various formats (NPZ, HDF5, NavIRL logs).  Provides a PyTorch
:class:`~torch.utils.data.Dataset` interface when Torch is available, and
pure-NumPy access otherwise.
"""

from __future__ import annotations

import glob
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    import h5py  # type: ignore[import-untyped]

    _HDF5_AVAILABLE = True
except ImportError:
    _HDF5_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["DemonstrationDataset"]


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------


@dataclass
class FeatureStatistics:
    """Per-feature mean and standard deviation computed over the dataset.

    Attributes:
        obs_mean: Observation mean.
        obs_std: Observation standard deviation.
        action_mean: Action mean.
        action_std: Action standard deviation.
    """

    obs_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    obs_std: np.ndarray = field(default_factory=lambda: np.array([]))
    action_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    action_std: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# DemonstrationDataset
# ---------------------------------------------------------------------------


class DemonstrationDataset:
    """Dataset for loading and processing expert demonstrations.

    Supports multiple file formats and provides train/val/test splitting,
    normalisation, augmentation, and an optional PyTorch
    :class:`~torch.utils.data.Dataset` interface.

    Parameters
    ----------
    observations : np.ndarray, optional
        Pre-loaded observation array ``(N, *obs_shape)``.
    actions : np.ndarray, optional
        Pre-loaded action array ``(N, *action_shape)``.
    rewards : np.ndarray, optional
        Pre-loaded reward array ``(N,)``.
    next_observations : np.ndarray, optional
        Pre-loaded next-observation array ``(N, *obs_shape)``.
    dones : np.ndarray, optional
        Pre-loaded done-flag array ``(N,)``.
    """

    def __init__(
        self,
        observations: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        rewards: np.ndarray | None = None,
        next_observations: np.ndarray | None = None,
        dones: np.ndarray | None = None,
    ) -> None:
        self.observations = (
            observations if observations is not None else np.empty((0,), dtype=np.float32)
        )
        self.actions = actions if actions is not None else np.empty((0,), dtype=np.float32)
        n = len(self.observations)
        self.rewards = rewards if rewards is not None else np.zeros(n, dtype=np.float32)
        self.next_observations = (
            next_observations if next_observations is not None else np.zeros_like(self.observations)
        )
        self.dones = dones if dones is not None else np.zeros(n, dtype=np.float32)

        self._stats: FeatureStatistics | None = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @classmethod
    def load_from_npz(cls, path: str | pathlib.Path) -> DemonstrationDataset:
        """Load demonstrations from a ``.npz`` file.

        Expected keys: ``obs``, ``actions``, and optionally ``rewards``,
        ``next_obs``, ``dones``.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npz`` file.

        Returns
        -------
        DemonstrationDataset
        """
        path = pathlib.Path(path)
        data = np.load(str(path), allow_pickle=False)
        logger.info("Loaded %d transitions from %s", len(data["obs"]), path)
        return cls(
            observations=data["obs"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            rewards=data.get("rewards", np.zeros(len(data["obs"]))).astype(np.float32),
            next_observations=data.get("next_obs", np.zeros_like(data["obs"])).astype(
                np.float32
            ),
            dones=data.get("dones", np.zeros(len(data["obs"]))).astype(np.float32),
        )

    @classmethod
    def load_from_hdf5(cls, path: str | pathlib.Path) -> DemonstrationDataset:
        """Load demonstrations from an HDF5 file.

        Expected datasets: ``obs``, ``actions``, and optionally ``rewards``,
        ``next_obs``, ``dones``.

        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file.

        Returns
        -------
        DemonstrationDataset

        Raises
        ------
        ImportError
            If ``h5py`` is not installed.
        """
        if not _HDF5_AVAILABLE:
            raise ImportError("h5py is required to load HDF5 files: pip install h5py")
        path = pathlib.Path(path)
        with h5py.File(str(path), "r") as f:
            obs = np.asarray(f["obs"], dtype=np.float32)
            actions = np.asarray(f["actions"], dtype=np.float32)
            rewards = (
                np.asarray(f["rewards"], dtype=np.float32)
                if "rewards" in f
                else np.zeros(len(obs), dtype=np.float32)
            )
            next_obs = (
                np.asarray(f["next_obs"], dtype=np.float32)
                if "next_obs" in f
                else np.zeros_like(obs)
            )
            dones = (
                np.asarray(f["dones"], dtype=np.float32)
                if "dones" in f
                else np.zeros(len(obs), dtype=np.float32)
            )
        logger.info("Loaded %d transitions from %s (HDF5)", len(obs), path)
        return cls(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            dones=dones,
        )

    @classmethod
    def load_from_navirl_logs(
        cls, log_dir: str | pathlib.Path
    ) -> DemonstrationDataset:
        """Load demonstrations from NavIRL log directory.

        Scans *log_dir* for ``.npz`` files named ``episode_*.npz`` and
        concatenates them into a single dataset.

        Parameters
        ----------
        log_dir : str or Path
            Directory containing NavIRL episode logs.

        Returns
        -------
        DemonstrationDataset
        """
        log_dir = pathlib.Path(log_dir)
        pattern = str(log_dir / "episode_*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No episode_*.npz files found in {log_dir}"
            )

        all_obs: list[np.ndarray] = []
        all_act: list[np.ndarray] = []
        all_rew: list[np.ndarray] = []
        all_next: list[np.ndarray] = []
        all_done: list[np.ndarray] = []

        for f in files:
            data = np.load(f, allow_pickle=False)
            all_obs.append(data["obs"].astype(np.float32))
            all_act.append(data["actions"].astype(np.float32))
            if "rewards" in data:
                all_rew.append(data["rewards"].astype(np.float32))
            else:
                all_rew.append(np.zeros(len(data["obs"]), dtype=np.float32))
            if "next_obs" in data:
                all_next.append(data["next_obs"].astype(np.float32))
            else:
                all_next.append(np.zeros_like(data["obs"], dtype=np.float32))
            if "dones" in data:
                all_done.append(data["dones"].astype(np.float32))
            else:
                all_done.append(np.zeros(len(data["obs"]), dtype=np.float32))

        logger.info(
            "Loaded %d episodes (%d transitions) from %s",
            len(files),
            sum(len(o) for o in all_obs),
            log_dir,
        )
        return cls(
            observations=np.concatenate(all_obs, axis=0),
            actions=np.concatenate(all_act, axis=0),
            rewards=np.concatenate(all_rew, axis=0),
            next_observations=np.concatenate(all_next, axis=0),
            dones=np.concatenate(all_done, axis=0),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def compute_statistics(self) -> FeatureStatistics:
        """Compute per-feature mean and standard deviation.

        Returns
        -------
        FeatureStatistics
        """
        eps = 1e-8
        self._stats = FeatureStatistics(
            obs_mean=self.observations.mean(axis=0),
            obs_std=self.observations.std(axis=0) + eps,
            action_mean=self.actions.mean(axis=0),
            action_std=self.actions.std(axis=0) + eps,
        )
        return self._stats

    def normalize(
        self,
        stats: FeatureStatistics | None = None,
        *,
        normalize_actions: bool = False,
    ) -> None:
        """Normalise observations (and optionally actions) in place.

        Parameters
        ----------
        stats : FeatureStatistics, optional
            Pre-computed statistics.  If *None*, statistics are computed from
            the current data.
        normalize_actions : bool
            Whether to normalise actions as well.
        """
        if stats is None:
            stats = self.compute_statistics()
        self.observations = (self.observations - stats.obs_mean) / stats.obs_std
        if self.next_observations.shape == self.observations.shape:
            self.next_observations = (
                self.next_observations - stats.obs_mean
            ) / stats.obs_std
        if normalize_actions:
            self.actions = (self.actions - stats.action_mean) / stats.action_std

    def augment_gaussian_noise(
        self,
        obs_noise_std: float = 0.01,
        action_noise_std: float = 0.0,
    ) -> None:
        """Add Gaussian noise to observations (and optionally actions).

        Parameters
        ----------
        obs_noise_std : float
            Standard deviation of noise added to observations.
        action_noise_std : float
            Standard deviation of noise added to actions.
        """
        self.observations = self.observations + np.random.normal(
            0, obs_noise_std, size=self.observations.shape
        ).astype(np.float32)
        if action_noise_std > 0.0:
            self.actions = self.actions + np.random.normal(
                0, action_noise_std, size=self.actions.shape
            ).astype(np.float32)

    def filter_by_reward(self, min_return: float) -> DemonstrationDataset:
        """Filter trajectories whose cumulative reward is below a threshold.

        Trajectories are identified by episode boundaries in ``self.dones``.

        Parameters
        ----------
        min_return : float
            Minimum cumulative reward to keep a trajectory.

        Returns
        -------
        DemonstrationDataset
            A new dataset with only the high-reward trajectories.
        """
        # Find episode boundaries
        done_indices = np.where(self.dones > 0.5)[0]
        if len(done_indices) == 0:
            return DemonstrationDataset(
                observations=self.observations.copy(),
                actions=self.actions.copy(),
                rewards=self.rewards.copy(),
                next_observations=self.next_observations.copy(),
                dones=self.dones.copy(),
            )

        keep_indices: list[np.ndarray] = []
        start = 0
        for end_idx in done_indices:
            ep_return = self.rewards[start : end_idx + 1].sum()
            if ep_return >= min_return:
                keep_indices.append(np.arange(start, end_idx + 1))
            start = end_idx + 1

        # Handle trailing data after last done
        if start < len(self.observations):
            ep_return = self.rewards[start:].sum()
            if ep_return >= min_return:
                keep_indices.append(np.arange(start, len(self.observations)))

        if not keep_indices:
            logger.warning("No trajectories met the min_return=%.2f threshold.", min_return)
            return DemonstrationDataset()

        idx = np.concatenate(keep_indices)
        return DemonstrationDataset(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            dones=self.dones[idx],
        )

    # ------------------------------------------------------------------
    # Train / val / test splitting
    # ------------------------------------------------------------------

    def split(
        self,
        train_ratio: float | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> tuple[DemonstrationDataset, DemonstrationDataset, DemonstrationDataset | None]:
        """Split the dataset into train, validation, and optional test sets.

        Parameters
        ----------
        val_ratio : float
            Fraction of data reserved for validation.
        test_ratio : float
            Fraction of data reserved for testing.
        shuffle : bool
            Whether to shuffle before splitting.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple
            ``(train_ds, val_ds, test_ds)`` where ``test_ds`` is *None*
            when ``test_ratio == 0``.
        """
        n = len(self)
        if train_ratio is not None:
            val_ratio = max(0.0, 1.0 - train_ratio - test_ratio)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :] if n_test > 0 else None

        def _subset(idx: np.ndarray) -> DemonstrationDataset:
            return DemonstrationDataset(
                observations=self.observations[idx],
                actions=self.actions[idx],
                rewards=self.rewards[idx],
                next_observations=self.next_observations[idx],
                dones=self.dones[idx],
            )

        train_ds = _subset(train_idx)
        val_ds = _subset(val_idx)
        test_ds = _subset(test_idx) if test_idx is not None else None

        logger.info(
            "Split dataset: train=%d  val=%d  test=%s",
            len(train_ds),
            len(val_ds),
            len(test_ds) if test_ds is not None else "N/A",
        )
        if train_ratio is not None and test_ratio == 0.0:
            return train_ds, val_ds
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def to_torch_dataset(self) -> Any:
        """Convert to a PyTorch :class:`~torch.utils.data.Dataset`.

        Returns
        -------
        torch.utils.data.TensorDataset

        Raises
        ------
        RuntimeError
            If PyTorch is not installed.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for to_torch_dataset(). "
                "Install with: pip install torch"
            )
        return torch.utils.data.TensorDataset(
            torch.as_tensor(self.observations, dtype=torch.float32),
            torch.as_tensor(self.actions, dtype=torch.float32),
            torch.as_tensor(self.rewards, dtype=torch.float32),
            torch.as_tensor(self.next_observations, dtype=torch.float32),
            torch.as_tensor(self.dones, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        dict
            Keys: ``"obs"``, ``"actions"``, ``"rewards"``, ``"next_obs"``,
            ``"dones"``.
        """
        indices = np.random.randint(0, len(self), size=batch_size)
        return {
            "obs": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "obs": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_observations[idx],
            "dones": self.dones[idx],
        }

    def __repr__(self) -> str:
        obs_shape = self.observations.shape[1:] if len(self.observations) > 0 else "?"
        act_shape = self.actions.shape[1:] if len(self.actions) > 0 else "?"
        return (
            f"DemonstrationDataset(n={len(self)}, "
            f"obs_shape={obs_shape}, action_shape={act_shape})"
        )

    def save(self, path: str | pathlib.Path) -> None:
        np.savez(
            path,
            obs=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_obs=self.next_observations,
            dones=self.dones,
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> DemonstrationDataset:
        path = pathlib.Path(path)
        if path.suffix == ".npz":
            return cls.load_from_npz(path)
        if path.suffix in {".h5", ".hdf5"}:
            return cls.load_from_hdf5(path)
        raise ValueError(f"Unsupported demonstration dataset format: {path.suffix}")
