"""Tests for navirl/data/ module: trajectory, augmentation, datasets, preprocessing, loaders."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from navirl.data.augmentation import (
    AugmentationPipeline,
    add_noise,
    crop_trajectory,
    mirror_trajectory,
    rotate_trajectory,
    scale_trajectory,
    time_warp,
)
from navirl.data.datasets import ETHUCYDataset, SocialDataset
from navirl.data.loaders import BatchLoader, GenericCSVLoader, NavIRLLogLoader
from navirl.data.preprocessing import (
    build_observation,
    compute_map_features,
    compute_social_features,
    encode_goal,
    normalize_positions,
)
from navirl.data.trajectory import (
    Trajectory,
    TrajectoryCollection,
    align_trajectories,
    compute_accelerations,
    compute_velocities,
    crop_to_region,
    interpolate,
    resample,
    smooth,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_trajectory():
    """Straight-line trajectory with 10 timesteps."""
    ts = np.arange(10, dtype=np.float64) * 0.1
    pos = np.column_stack([np.arange(10, dtype=np.float64), np.zeros(10)])
    vel = np.column_stack([np.ones(10), np.zeros(10)])
    return Trajectory(timestamps=ts, positions=pos, velocities=vel, agent_id="a1")


@pytest.fixture
def curved_trajectory():
    """Circular arc trajectory."""
    t = np.linspace(0, np.pi, 20)
    pos = np.column_stack([np.cos(t), np.sin(t)])
    ts = np.arange(20, dtype=np.float64) * 0.2
    return Trajectory(timestamps=ts, positions=pos, agent_id="c1")


@pytest.fixture
def collection(simple_trajectory, curved_trajectory):
    tc = TrajectoryCollection()
    tc.add(simple_trajectory)
    tc.add(curved_trajectory)
    return tc


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_construction_basic(self):
        t = Trajectory(timestamps=[0, 1, 2], positions=[[0, 0], [1, 0], [2, 0]])
        assert len(t) == 3
        assert t.timestamps.dtype == np.float64
        assert t.positions.shape == (3, 2)
        assert t.velocities is None

    def test_construction_with_velocities(self):
        t = Trajectory(
            timestamps=[0, 1],
            positions=[[0, 0], [1, 1]],
            velocities=[[1, 1], [1, 1]],
        )
        assert t.velocities is not None
        assert t.velocities.shape == (2, 2)

    def test_duration_single_point(self):
        t = Trajectory(timestamps=[5.0], positions=[[0, 0]])
        assert t.duration == 0.0

    def test_duration_multiple_points(self, simple_trajectory):
        assert simple_trajectory.duration == pytest.approx(0.9, abs=1e-10)

    def test_agent_id_default(self):
        t = Trajectory(timestamps=[0], positions=[[0, 0]])
        assert t.agent_id == ""


# ---------------------------------------------------------------------------
# TrajectoryCollection
# ---------------------------------------------------------------------------

class TestTrajectoryCollection:
    def test_empty_collection(self):
        tc = TrajectoryCollection()
        assert len(tc) == 0
        arr = tc.to_numpy()
        assert arr.shape == (0, 2)

    def test_add_and_len(self, collection):
        assert len(collection) == 2

    def test_getitem(self, collection, simple_trajectory):
        assert collection[0].agent_id == simple_trajectory.agent_id

    def test_iter(self, collection):
        ids = [t.agent_id for t in collection]
        assert len(ids) == 2

    def test_filter_by_agent(self, collection):
        filtered = collection.filter_by_agent("a1")
        assert len(filtered) == 1
        assert filtered[0].agent_id == "a1"

    def test_filter_by_agent_missing(self, collection):
        filtered = collection.filter_by_agent("nonexistent")
        assert len(filtered) == 0

    def test_filter_by_duration(self, collection):
        # curved traj has duration ~3.8s; simple has ~0.9s
        filtered = collection.filter_by_duration(1.0)
        assert len(filtered) == 1

    def test_to_numpy(self, collection):
        arr = collection.to_numpy()
        assert arr.shape[1] == 2
        assert arr.shape[0] == 30  # 10 + 20

    def test_agent_ids(self, collection):
        ids = collection.agent_ids
        assert set(ids) == {"a1", "c1"}


# ---------------------------------------------------------------------------
# Interpolation / resampling
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_interpolate_uniform(self, simple_trajectory):
        result = interpolate(simple_trajectory, dt=0.05)
        assert result.positions.shape[1] == 2
        dts = np.diff(result.timestamps)
        np.testing.assert_allclose(dts, 0.05, atol=1e-10)

    def test_interpolate_preserves_agent_id(self, simple_trajectory):
        result = interpolate(simple_trajectory, dt=0.05)
        assert result.agent_id == "a1"

    def test_interpolate_with_velocities(self, simple_trajectory):
        result = interpolate(simple_trajectory, dt=0.05)
        assert result.velocities is not None

    def test_interpolate_single_point(self):
        t = Trajectory(timestamps=[1.0], positions=[[5, 5]])
        result = interpolate(t, dt=0.1)
        assert len(result) == 1

    def test_resample_is_alias(self, simple_trajectory):
        r1 = interpolate(simple_trajectory, 0.05)
        r2 = resample(simple_trajectory, 0.05)
        np.testing.assert_array_equal(r1.positions, r2.positions)

    @pytest.mark.parametrize("dt", [0.01, 0.05, 0.1, 0.5])
    def test_interpolate_various_dt(self, simple_trajectory, dt):
        result = interpolate(simple_trajectory, dt)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_smooth_reduces_noise(self):
        rng = np.random.default_rng(42)
        noisy = np.column_stack([np.arange(50, dtype=np.float64), np.zeros(50)])
        noisy += rng.normal(0, 0.5, noisy.shape)
        t = Trajectory(timestamps=np.arange(50) * 0.1, positions=noisy)
        smoothed = smooth(t, window=5)
        # Smoothed should have lower variance in y
        assert np.std(smoothed.positions[:, 1]) < np.std(noisy[:, 1])

    def test_smooth_window_even_becomes_odd(self):
        t = Trajectory(timestamps=np.arange(20) * 0.1, positions=np.zeros((20, 2)))
        result = smooth(t, window=4)
        assert len(result) == 20

    def test_smooth_window_too_large(self, simple_trajectory):
        # window > len -> returns original
        result = smooth(simple_trajectory, window=99)
        np.testing.assert_array_equal(result.positions, simple_trajectory.positions)

    def test_smooth_invalid_window(self):
        t = Trajectory(timestamps=[0], positions=[[0, 0]])
        with pytest.raises(ValueError):
            smooth(t, window=0)


# ---------------------------------------------------------------------------
# Velocity / acceleration computation
# ---------------------------------------------------------------------------

class TestVelocityAcceleration:
    def test_compute_velocities_shape(self):
        pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
        vel = compute_velocities(pos, dt=0.1)
        assert vel.shape == (4, 2)

    def test_compute_velocities_values(self):
        pos = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        vel = compute_velocities(pos, dt=1.0)
        np.testing.assert_allclose(vel[0], [1.0, 0.0])
        np.testing.assert_allclose(vel[-1], vel[-2])

    def test_compute_velocities_single_point(self):
        pos = np.array([[5, 5]], dtype=np.float64)
        vel = compute_velocities(pos, dt=0.1)
        np.testing.assert_array_equal(vel, [[0, 0]])

    def test_compute_accelerations_shape(self):
        vel = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        acc = compute_accelerations(vel, dt=0.1)
        assert acc.shape == (3, 2)

    def test_compute_accelerations_constant_velocity(self):
        vel = np.ones((5, 2), dtype=np.float64)
        acc = compute_accelerations(vel, dt=0.1)
        np.testing.assert_allclose(acc, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Alignment and cropping
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_align_trajectories_overlap(self):
        t1 = Trajectory(timestamps=np.arange(10) * 0.1, positions=np.zeros((10, 2)), agent_id="a")
        t2 = Trajectory(timestamps=np.arange(5, 15) * 0.1, positions=np.ones((10, 2)), agent_id="b")
        aligned = align_trajectories([t1, t2])
        assert len(aligned) == 2
        for t in aligned:
            assert t.timestamps[0] >= 0.5 - 1e-9
            assert t.timestamps[-1] <= 0.9 + 1e-9

    def test_align_empty_list(self):
        assert align_trajectories([]) == []

    def test_align_no_overlap(self):
        t1 = Trajectory(timestamps=np.array([0, 1.0]), positions=np.zeros((2, 2)), agent_id="a")
        t2 = Trajectory(timestamps=np.array([2, 3.0]), positions=np.ones((2, 2)), agent_id="b")
        aligned = align_trajectories([t1, t2])
        assert len(aligned) == 2

    def test_crop_to_region_spatial(self, collection):
        cropped = crop_to_region(collection, bounds=(0, -1, 5, 1))
        assert len(cropped) > 0

    def test_crop_to_region_temporal(self, collection):
        cropped = crop_to_region(collection, bounds=None, time_bounds=(0.0, 0.5))
        for t in cropped:
            assert t.timestamps[-1] <= 0.5 + 1e-9


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class TestAugmentation:
    def test_rotate_trajectory(self, simple_trajectory):
        rotated = rotate_trajectory(simple_trajectory, np.pi / 2)
        # x should become ~0, y should become ~original x
        np.testing.assert_allclose(rotated.positions[5, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(rotated.positions[5, 1], 5.0, atol=1e-10)

    def test_rotate_preserves_length(self, simple_trajectory):
        rotated = rotate_trajectory(simple_trajectory, np.pi / 4)
        assert len(rotated) == len(simple_trajectory)

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_mirror_trajectory(self, simple_trajectory, axis):
        mirrored = mirror_trajectory(simple_trajectory, axis=axis)
        assert mirrored.positions.shape == simple_trajectory.positions.shape

    def test_mirror_invalid_axis(self, simple_trajectory):
        with pytest.raises(ValueError, match="axis must be"):
            mirror_trajectory(simple_trajectory, axis="z")

    def test_scale_trajectory(self, simple_trajectory):
        scaled = scale_trajectory(simple_trajectory, factor=2.0)
        np.testing.assert_allclose(scaled.positions, simple_trajectory.positions * 2.0)

    def test_add_noise_reproducible(self, simple_trajectory):
        n1 = add_noise(simple_trajectory, std=0.1, seed=0)
        n2 = add_noise(simple_trajectory, std=0.1, seed=0)
        np.testing.assert_array_equal(n1.positions, n2.positions)

    def test_add_noise_changes_positions(self, simple_trajectory):
        noisy = add_noise(simple_trajectory, std=1.0, seed=42)
        assert not np.allclose(noisy.positions, simple_trajectory.positions)

    def test_time_warp_stretch(self, simple_trajectory):
        warped = time_warp(simple_trajectory, factor=2.0)
        assert warped.duration == pytest.approx(simple_trajectory.duration * 2.0, rel=1e-9)

    def test_time_warp_invalid(self, simple_trajectory):
        with pytest.raises(ValueError):
            time_warp(simple_trajectory, factor=-1.0)

    def test_crop_trajectory_basic(self, simple_trajectory):
        cropped = crop_trajectory(simple_trajectory, 2, 5)
        assert len(cropped) == 3

    def test_pipeline_empty(self, simple_trajectory):
        pipe = AugmentationPipeline()
        assert len(pipe) == 0
        result = pipe(simple_trajectory)
        np.testing.assert_array_equal(result.positions, simple_trajectory.positions)

    def test_pipeline_chaining(self, simple_trajectory):
        pipe = AugmentationPipeline()
        pipe.add(lambda t: scale_trajectory(t, 2.0))
        pipe.add(lambda t: rotate_trajectory(t, np.pi))
        result = pipe(simple_trajectory)
        assert result.positions.shape == simple_trajectory.positions.shape

    def test_pipeline_returns_self(self):
        pipe = AugmentationPipeline()
        ret = pipe.add(lambda t: t)
        assert ret is pipe


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_normalize_minmax(self, collection):
        normed, stats = normalize_positions(collection, method="minmax")
        arr = normed.to_numpy()
        assert arr.min() >= -1e-9
        assert arr.max() <= 1.0 + 1e-9
        assert "min" in stats
        assert "max" in stats

    def test_normalize_standard(self, collection):
        normed, stats = normalize_positions(collection, method="standard")
        assert "mean" in stats
        assert "std" in stats

    def test_normalize_empty(self):
        tc = TrajectoryCollection()
        normed, stats = normalize_positions(tc)
        assert stats == {}

    def test_normalize_invalid_method(self, collection):
        with pytest.raises(ValueError):
            normalize_positions(collection, method="invalid")

    def test_encode_goal(self):
        result = encode_goal(np.array([10, 0]), np.array([0, 0]))
        np.testing.assert_allclose(result, [10, 0, 10])

    def test_encode_goal_same_position(self):
        result = encode_goal(np.array([5, 5]), np.array([5, 5]))
        np.testing.assert_allclose(result, [0, 0, 0])

    def test_build_observation_no_map(self):
        ego = np.array([1, 2, 0.5, 0.5])
        neighbors = np.zeros(24)
        goal = np.array([10, 0, 10])
        obs = build_observation(ego, neighbors, goal)
        assert obs.shape == (4 + 24 + 3,)

    def test_build_observation_with_map(self):
        ego = np.array([1, 2])
        neighbors = np.zeros(8)
        goal = np.array([5, 5, 7.07])
        map_data = np.zeros(16)
        obs = build_observation(ego, neighbors, goal, map_data)
        assert obs.shape == (2 + 8 + 3 + 16,)

    def test_compute_social_features_shape(self, simple_trajectory, curved_trajectory):
        feat = compute_social_features(simple_trajectory, [curved_trajectory], max_neighbors=3)
        assert feat.shape == (len(simple_trajectory), 3 * 4)

    def test_compute_social_features_no_neighbors(self, simple_trajectory):
        feat = compute_social_features(simple_trajectory, [], max_neighbors=6)
        np.testing.assert_array_equal(feat, 0.0)

    def test_compute_map_features_shape(self):
        grid = np.zeros((50, 50), dtype=np.int32)
        result = compute_map_features(np.array([0, 0]), grid, patch_size=8)
        assert result.shape == (64,)


# ---------------------------------------------------------------------------
# Dataset loading with mock data
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_ethucy_invalid_scene(self):
        with pytest.raises(ValueError, match="Unknown ETH/UCY scene"):
            ETHUCYDataset(scenes=["invalid_scene"])

    def test_ethucy_file_not_found(self, tmp_path):
        ds = ETHUCYDataset(scenes=["eth"])
        with pytest.raises(FileNotFoundError):
            ds.load(tmp_path)

    def test_social_dataset_load(self, tmp_path):
        csv_file = tmp_path / "scene.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "agent_id", "x", "y"])
            for i in range(20):
                writer.writerow([i * 0.1, "ped1", i * 0.5, 0.0])
                writer.writerow([i * 0.1, "ped2", 0.0, i * 0.3])
        ds = SocialDataset(has_header=True)
        ds.load(tmp_path)
        assert ds.num_scenes == 1
        tc = ds.trajectories()
        assert len(tc) == 2

    def test_social_dataset_with_velocities(self, tmp_path):
        csv_file = tmp_path / "scene.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "agent_id", "x", "y", "vx", "vy"])
            for i in range(10):
                writer.writerow([i * 0.1, "a1", i, 0, 1.0, 0.0])
        ds = SocialDataset(has_header=True)
        ds.load(csv_file)
        tc = ds.trajectories()
        assert tc[0].velocities is not None

    def test_dataset_not_loaded_error(self):
        ds = SocialDataset()
        with pytest.raises(RuntimeError, match="not loaded"):
            ds.trajectories()

    def test_train_test_split(self, tmp_path):
        # Create multiple scene files
        for scene_idx in range(5):
            f = tmp_path / f"scene_{scene_idx}.csv"
            with open(f, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["timestamp", "agent_id", "x", "y"])
                for i in range(10):
                    writer.writerow([i * 0.1, "a1", i, scene_idx])
        ds = SocialDataset(has_header=True)
        ds.load(tmp_path)
        train, test = ds.train_test_split(test_ratio=0.4, seed=0)
        assert len(train) + len(test) == 5
        assert len(test) >= 1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_navirl_log_loader(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        with open(state_file, "w") as f:
            for i in range(10):
                row = {"t": i * 0.1, "x": float(i), "y": 0.0, "agent_id": "robot"}
                f.write(json.dumps(row) + "\n")
        loader = NavIRLLogLoader()
        loader.load(tmp_path)
        tc = loader.to_trajectories()
        assert len(tc) == 1
        assert tc[0].agent_id == "robot"

    def test_navirl_log_loader_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NavIRLLogLoader().load(tmp_path)

    def test_generic_csv_loader(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(15):
                writer.writerow([i * 0.1, float(i), 0.0])
        loader = GenericCSVLoader(timestamp_col=0, x_col=1, y_col=2)
        tc = loader.load(csv_file)
        assert len(tc) == 1
        assert len(tc[0]) == 15

    def test_generic_csv_loader_with_agent_col(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(10):
                writer.writerow([i * 0.1, float(i), 0.0, "agentA"])
                writer.writerow([i * 0.1, 0.0, float(i), "agentB"])
        loader = GenericCSVLoader(timestamp_col=0, x_col=1, y_col=2, agent_col=3)
        tc = loader.load(csv_file)
        assert len(tc) == 2

    def test_batch_loader(self, collection):
        loader = BatchLoader(collection, batch_size=1, shuffle=False)
        assert len(loader) == 2
        batches = list(loader)
        assert len(batches) == 2
        assert len(batches[0]) == 1

    def test_batch_loader_shuffle(self, collection):
        loader = BatchLoader(collection, batch_size=2, shuffle=True, seed=0)
        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0]) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_trajectory_collection_to_numpy(self):
        tc = TrajectoryCollection()
        arr = tc.to_numpy()
        assert arr.shape == (0, 2)

    def test_trajectory_with_lists(self):
        t = Trajectory(timestamps=[0, 1], positions=[[0, 0], [1, 1]])
        assert isinstance(t.timestamps, np.ndarray)

    def test_compute_velocities_two_points(self):
        pos = np.array([[0, 0], [1, 1]], dtype=np.float64)
        vel = compute_velocities(pos, dt=1.0)
        assert vel.shape == (2, 2)
        np.testing.assert_allclose(vel[0], [1, 1])

    def test_rotate_trajectory_zero_angle(self, simple_trajectory):
        rotated = rotate_trajectory(simple_trajectory, 0.0)
        np.testing.assert_allclose(rotated.positions, simple_trajectory.positions, atol=1e-14)

    def test_scale_trajectory_zero(self, simple_trajectory):
        scaled = scale_trajectory(simple_trajectory, factor=0.0)
        np.testing.assert_array_equal(scaled.positions, 0.0)
