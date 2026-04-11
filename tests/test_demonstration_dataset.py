"""Tests for navirl/imitation/dataset.py — DemonstrationDataset (NumPy paths)."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

import importlib.util
import sys

# Import dataset module directly to avoid navirl.imitation.__init__ pulling in
# torch-dependent siblings (AIRL, BC, GAIL).
_spec = importlib.util.spec_from_file_location(
    "navirl.imitation.dataset",
    pathlib.Path(__file__).resolve().parent.parent / "navirl" / "imitation" / "dataset.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
DemonstrationDataset = _mod.DemonstrationDataset
FeatureStatistics = _mod.FeatureStatistics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Return arrays for 50 transitions with obs_dim=4, act_dim=2."""
    rng = np.random.default_rng(42)
    n = 50
    obs = rng.standard_normal((n, 4)).astype(np.float32)
    actions = rng.standard_normal((n, 2)).astype(np.float32)
    rewards = rng.standard_normal(n).astype(np.float32)
    next_obs = obs + 0.1 * rng.standard_normal((n, 4)).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[24] = 1.0  # episode boundary at index 24
    dones[49] = 1.0  # episode boundary at end
    return obs, actions, rewards, next_obs, dones


@pytest.fixture
def dataset(sample_data):
    obs, actions, rewards, next_obs, dones = sample_data
    return DemonstrationDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        dones=dones,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_arrays(self, dataset):
        assert len(dataset) == 50
        assert dataset.observations.shape == (50, 4)
        assert dataset.actions.shape == (50, 2)

    def test_empty_dataset(self):
        ds = DemonstrationDataset()
        assert len(ds) == 0
        assert ds.observations.shape == (0,)

    def test_obs_only(self):
        obs = np.ones((10, 3), dtype=np.float32)
        ds = DemonstrationDataset(observations=obs)
        assert len(ds) == 10
        assert ds.rewards.shape == (10,)
        assert ds.next_observations.shape == (10, 3)
        assert ds.dones.shape == (10,)
        np.testing.assert_array_equal(ds.rewards, 0.0)

    def test_dtype_conversion(self):
        obs = np.ones((5, 2), dtype=np.float64)
        ds = DemonstrationDataset(observations=obs)
        assert ds.observations.dtype == np.float32


# ---------------------------------------------------------------------------
# __getitem__ and __repr__
# ---------------------------------------------------------------------------


class TestDunderMethods:
    def test_getitem(self, dataset):
        item = dataset[0]
        assert set(item.keys()) == {"obs", "actions", "rewards", "next_obs", "dones"}
        assert item["obs"].shape == (4,)

    def test_repr(self, dataset):
        r = repr(dataset)
        assert "DemonstrationDataset" in r
        assert "n=50" in r

    def test_repr_empty(self):
        r = repr(DemonstrationDataset())
        assert "n=0" in r


# ---------------------------------------------------------------------------
# Save / Load NPZ
# ---------------------------------------------------------------------------


class TestSaveLoadNpz:
    def test_round_trip(self, dataset):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "demo.npz"
            dataset.save(path)
            loaded = DemonstrationDataset.load_from_npz(path)

        assert len(loaded) == len(dataset)
        np.testing.assert_allclose(loaded.observations, dataset.observations)
        np.testing.assert_allclose(loaded.actions, dataset.actions)
        np.testing.assert_allclose(loaded.rewards, dataset.rewards)
        np.testing.assert_allclose(loaded.next_observations, dataset.next_observations)
        np.testing.assert_allclose(loaded.dones, dataset.dones)

    def test_load_dispatches_npz(self, dataset):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "demo.npz"
            dataset.save(path)
            loaded = DemonstrationDataset.load(path)
        assert len(loaded) == 50

    def test_load_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DemonstrationDataset.load("/tmp/demo.csv")


# ---------------------------------------------------------------------------
# Load from NavIRL logs
# ---------------------------------------------------------------------------


class TestLoadFromNavirlLogs:
    def test_loads_multiple_episodes(self):
        with tempfile.TemporaryDirectory() as td:
            for i in range(3):
                rng = np.random.default_rng(i)
                obs = rng.standard_normal((10, 4)).astype(np.float32)
                actions = rng.standard_normal((10, 2)).astype(np.float32)
                np.savez(
                    pathlib.Path(td) / f"episode_{i:03d}.npz",
                    obs=obs,
                    actions=actions,
                )
            ds = DemonstrationDataset.load_from_navirl_logs(td)
        assert len(ds) == 30  # 3 episodes * 10

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError, match="No episode"):
                DemonstrationDataset.load_from_navirl_logs(td)

    def test_with_optional_fields(self):
        with tempfile.TemporaryDirectory() as td:
            obs = np.ones((5, 2), dtype=np.float32)
            actions = np.ones((5, 2), dtype=np.float32)
            rewards = np.ones(5, dtype=np.float32) * 2.0
            np.savez(
                pathlib.Path(td) / "episode_000.npz",
                obs=obs,
                actions=actions,
                rewards=rewards,
            )
            ds = DemonstrationDataset.load_from_navirl_logs(td)
        np.testing.assert_allclose(ds.rewards, 2.0)


# ---------------------------------------------------------------------------
# Statistics and Normalization
# ---------------------------------------------------------------------------


class TestStatisticsAndNorm:
    def test_compute_statistics(self, dataset):
        stats = dataset.compute_statistics()
        assert isinstance(stats, FeatureStatistics)
        assert stats.obs_mean.shape == (4,)
        assert stats.obs_std.shape == (4,)
        assert stats.action_mean.shape == (2,)
        assert stats.action_std.shape == (2,)
        # std should be positive (epsilon added)
        assert np.all(stats.obs_std > 0)

    def test_normalize_obs(self, dataset):
        original_obs = dataset.observations.copy()
        dataset.normalize()
        # After normalization, mean should be ~0 and std ~1
        assert np.abs(dataset.observations.mean(axis=0)).max() < 0.1
        # Original should differ
        assert not np.allclose(dataset.observations, original_obs)

    def test_normalize_with_actions(self, dataset):
        original_act = dataset.actions.copy()
        dataset.normalize(normalize_actions=True)
        assert not np.allclose(dataset.actions, original_act)

    def test_normalize_with_provided_stats(self, dataset):
        stats = FeatureStatistics(
            obs_mean=np.zeros(4, dtype=np.float32),
            obs_std=np.ones(4, dtype=np.float32),
            action_mean=np.zeros(2, dtype=np.float32),
            action_std=np.ones(2, dtype=np.float32),
        )
        original = dataset.observations.copy()
        dataset.normalize(stats=stats)
        # With mean=0, std=1: (x - 0) / 1 = x, so should be unchanged
        np.testing.assert_allclose(dataset.observations, original)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:
    def test_gaussian_noise_obs(self, dataset):
        original = dataset.observations.copy()
        dataset.augment_gaussian_noise(obs_noise_std=0.1)
        assert not np.allclose(dataset.observations, original)
        # Noise should be small
        diff = np.abs(dataset.observations - original)
        assert diff.mean() < 0.5

    def test_gaussian_noise_actions(self, dataset):
        original = dataset.actions.copy()
        dataset.augment_gaussian_noise(obs_noise_std=0.0, action_noise_std=0.1)
        assert not np.allclose(dataset.actions, original)

    def test_zero_noise_no_change(self, dataset):
        original = dataset.observations.copy()
        dataset.augment_gaussian_noise(obs_noise_std=0.0, action_noise_std=0.0)
        np.testing.assert_allclose(dataset.observations, original)


# ---------------------------------------------------------------------------
# Filter by reward
# ---------------------------------------------------------------------------


class TestFilterByReward:
    def test_filters_low_reward_episodes(self, dataset):
        # Set first episode (indices 0-24) to low reward, second (25-49) to high
        dataset.rewards[:25] = -1.0
        dataset.rewards[25:] = 10.0
        filtered = dataset.filter_by_reward(min_return=0.0)
        assert len(filtered) == 25  # only the second episode

    def test_no_dones_returns_copy(self):
        ds = DemonstrationDataset(
            observations=np.ones((10, 2), dtype=np.float32),
            actions=np.ones((10, 2), dtype=np.float32),
            rewards=np.ones(10, dtype=np.float32),
            dones=np.zeros(10, dtype=np.float32),  # no episode boundaries
        )
        filtered = ds.filter_by_reward(min_return=0.0)
        assert len(filtered) == 10

    def test_all_filtered_returns_empty(self, dataset):
        dataset.rewards[:] = -100.0
        filtered = dataset.filter_by_reward(min_return=0.0)
        assert len(filtered) == 0

    def test_trailing_data_after_last_done(self):
        n = 20
        obs = np.ones((n, 2), dtype=np.float32)
        actions = np.ones((n, 2), dtype=np.float32)
        rewards = np.ones(n, dtype=np.float32) * 5.0
        dones = np.zeros(n, dtype=np.float32)
        dones[9] = 1.0  # episode boundary at 9, then 10-19 is trailing
        ds = DemonstrationDataset(
            observations=obs, actions=actions, rewards=rewards, dones=dones,
        )
        filtered = ds.filter_by_reward(min_return=1.0)
        assert len(filtered) == 20  # both chunks pass


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_train_val_split(self, dataset):
        train, val = dataset.split(train_ratio=0.8, seed=0)
        assert len(train) + len(val) == len(dataset)

    def test_train_val_test_split(self, dataset):
        train, val, test = dataset.split(val_ratio=0.2, test_ratio=0.1, seed=0)
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_test_none_when_zero(self, dataset):
        train, val, test = dataset.split(val_ratio=0.2, test_ratio=0.0, seed=0)
        assert test is None

    def test_invalid_ratios(self, dataset):
        with pytest.raises(ValueError, match="<= 1.0"):
            dataset.split(train_ratio=0.9, test_ratio=0.2)

    def test_no_shuffle(self, dataset):
        train, val = dataset.split(train_ratio=0.8, shuffle=False)
        n_train = len(train)
        np.testing.assert_allclose(train.observations, dataset.observations[:n_train])

    def test_reproducibility(self, dataset):
        t1, v1 = dataset.split(train_ratio=0.8, seed=42)
        t2, v2 = dataset.split(train_ratio=0.8, seed=42)
        np.testing.assert_array_equal(t1.observations, t2.observations)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_batch(self, dataset):
        batch = dataset.sample(batch_size=8)
        assert set(batch.keys()) == {"obs", "actions", "rewards", "next_obs", "dones"}
        assert batch["obs"].shape == (8, 4)
        assert batch["actions"].shape == (8, 2)
        assert batch["rewards"].shape == (8,)

    def test_sample_size_one(self, dataset):
        batch = dataset.sample(batch_size=1)
        assert batch["obs"].shape == (1, 4)


# ---------------------------------------------------------------------------
# PyTorch interface
# ---------------------------------------------------------------------------


class TestTorchInterface:
    def test_torch_not_available_raises(self, dataset, monkeypatch):
        monkeypatch.setattr(_mod, "_TORCH_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="PyTorch"):
            dataset.to_torch_dataset()
