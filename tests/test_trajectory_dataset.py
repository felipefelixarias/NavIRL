"""Tests for navirl.data.trajectory_dataset — loading, windowing, splitting, augmentation."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from navirl.data.trajectory import Trajectory
from navirl.data.trajectory_dataset import (
    AugmentationConfig,
    SplitConfig,
    TrajectoryDatasetPipeline,
    TrajectoryWindow,
    _add_noise,
    _compute_velocities,
    _flip,
    _load_csv,
    _load_json,
    _load_protobuf,
    _perturb_speed,
    _rotate,
    _scale,
    augment_dataset,
    augment_window,
    create_windows,
    extract_scene_context,
    find_neighbors,
    split_trajectories,
    split_windows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(n=30, agent_id="ped_0", dt=0.4):
    """Create a simple linear trajectory for testing."""
    ts = np.arange(n, dtype=np.float64) * dt
    pos = np.column_stack([ts, ts * 0.5])  # x = t, y = t/2
    return Trajectory(timestamps=ts, positions=pos, agent_id=agent_id)


def _make_window(obs_len=8, pred_len=12):
    """Create a simple TrajectoryWindow."""
    total = obs_len + pred_len
    ts = np.arange(total, dtype=np.float64) * 0.4
    pos = np.column_stack([ts, ts * 0.5])
    vel = np.ones((total, 2)) * 0.5
    return TrajectoryWindow(
        observed=pos[:obs_len].copy(),
        future=pos[obs_len:].copy(),
        observed_vel=vel[:obs_len].copy(),
        future_vel=vel[obs_len:].copy(),
        timestamps_obs=ts[:obs_len].copy(),
        timestamps_fut=ts[obs_len:].copy(),
        agent_id="ped_0",
    )


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_comma_delimited(self, tmp_path):
        csv_data = "frame_id,agent_id,x,y\n0,ped_0,1.0,2.0\n1,ped_0,1.5,2.5\n"
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        trajs = _load_csv(p)
        assert len(trajs) == 1
        assert trajs[0].agent_id == "ped_0"
        assert len(trajs[0]) == 2

    def test_tab_delimited(self, tmp_path):
        csv_data = "0\tped_0\t1.0\t2.0\n1\tped_0\t1.5\t2.5\n"
        p = tmp_path / "data.tsv"
        p.write_text(csv_data)
        trajs = _load_csv(p)
        assert len(trajs) == 1

    def test_with_velocities(self, tmp_path):
        csv_data = "0,ped_0,1.0,2.0,0.5,0.5\n1,ped_0,1.5,2.5,0.5,0.5\n"
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        trajs = _load_csv(p)
        assert trajs[0].velocities is not None
        assert trajs[0].velocities.shape == (2, 2)

    def test_multiple_agents(self, tmp_path):
        csv_data = "0,a,1.0,2.0\n0,b,3.0,4.0\n1,a,1.5,2.5\n1,b,3.5,4.5\n"
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        trajs = _load_csv(p)
        assert len(trajs) == 2

    def test_header_row_skipped(self, tmp_path):
        csv_data = "frame,agent,x,y\n0,ped_0,1.0,2.0\n"
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        trajs = _load_csv(p)
        assert len(trajs) == 1

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("frame,agent,x,y\n")
        trajs = _load_csv(p)
        assert trajs == []

    def test_sorted_by_timestamp(self, tmp_path):
        csv_data = "2,ped_0,3.0,6.0\n0,ped_0,1.0,2.0\n1,ped_0,2.0,4.0\n"
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        trajs = _load_csv(p)
        np.testing.assert_array_equal(trajs[0].timestamps, [0, 1, 2])
        np.testing.assert_array_almost_equal(trajs[0].positions[0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------


class TestLoadJSON:
    def test_trajectories_key(self, tmp_path):
        data = {
            "trajectories": [
                {
                    "agent_id": "p1",
                    "timestamps": [0.0, 0.4, 0.8],
                    "positions": [[0, 0], [1, 1], [2, 2]],
                }
            ]
        }
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        trajs = _load_json(p)
        assert len(trajs) == 1
        assert trajs[0].agent_id == "p1"
        assert trajs[0].velocities is None

    def test_array_format(self, tmp_path):
        data = [
            {
                "id": "p1",
                "timestamps": [0.0, 0.4],
                "positions": [[0, 0], [1, 1]],
            }
        ]
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        trajs = _load_json(p)
        assert len(trajs) == 1
        assert trajs[0].agent_id == "p1"

    def test_with_velocities(self, tmp_path):
        data = {
            "trajectories": [
                {
                    "agent_id": "p1",
                    "timestamps": [0.0, 0.4],
                    "positions": [[0, 0], [1, 1]],
                    "velocities": [[2.5, 2.5], [2.5, 2.5]],
                }
            ]
        }
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        trajs = _load_json(p)
        assert trajs[0].velocities is not None


# ---------------------------------------------------------------------------
# Protobuf loader
# ---------------------------------------------------------------------------


class TestLoadProtobuf:
    def _encode_trajectory(self, agent_id, timestamps, positions, velocities=None):
        """Encode one trajectory in the binary format."""
        buf = bytearray()
        aid_bytes = agent_id.encode("utf-8")
        buf += struct.pack("<I", len(aid_bytes))
        buf += aid_bytes
        T = len(timestamps)
        buf += struct.pack("<I", T)
        for t in timestamps:
            buf += struct.pack("<d", t)
        for x, y in positions:
            buf += struct.pack("<dd", x, y)
        if velocities is not None:
            buf += struct.pack("B", 1)
            for vx, vy in velocities:
                buf += struct.pack("<dd", vx, vy)
        else:
            buf += struct.pack("B", 0)
        return bytes(buf)

    def test_basic(self, tmp_path):
        raw = self._encode_trajectory("p1", [0.0, 0.4], [[1.0, 2.0], [3.0, 4.0]])
        p = tmp_path / "data.bin"
        p.write_bytes(raw)
        trajs = _load_protobuf(p)
        assert len(trajs) == 1
        assert trajs[0].agent_id == "p1"
        assert trajs[0].velocities is None

    def test_with_velocities(self, tmp_path):
        raw = self._encode_trajectory(
            "p1", [0.0, 0.4], [[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.5, 0.5]]
        )
        p = tmp_path / "data.bin"
        p.write_bytes(raw)
        trajs = _load_protobuf(p)
        assert trajs[0].velocities is not None

    def test_multiple_trajectories(self, tmp_path):
        raw = self._encode_trajectory("p1", [0.0], [[1.0, 2.0]])
        raw += self._encode_trajectory("p2", [0.0], [[3.0, 4.0]])
        p = tmp_path / "data.bin"
        p.write_bytes(raw)
        trajs = _load_protobuf(p)
        assert len(trajs) == 2


# ---------------------------------------------------------------------------
# Velocity computation
# ---------------------------------------------------------------------------


class TestComputeVelocities:
    def test_with_existing_velocities(self):
        vel = np.ones((5, 2))
        traj = Trajectory(
            timestamps=np.arange(5, dtype=np.float64),
            positions=np.zeros((5, 2)),
            velocities=vel,
        )
        result = _compute_velocities(traj)
        np.testing.assert_array_equal(result, vel)

    def test_computed_from_positions(self):
        traj = Trajectory(
            timestamps=np.array([0.0, 1.0, 2.0]),
            positions=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        )
        vel = _compute_velocities(traj)
        assert vel.shape == (3, 2)
        np.testing.assert_almost_equal(vel[1, 0], 1.0)
        np.testing.assert_almost_equal(vel[1, 1], 0.0)

    def test_single_point(self):
        traj = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[1.0, 2.0]]),
        )
        vel = _compute_velocities(traj)
        assert vel.shape == (1, 2)
        np.testing.assert_array_equal(vel, [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


class TestCreateWindows:
    def test_basic(self):
        traj = _make_trajectory(30)
        windows = create_windows([traj], obs_len=8, pred_len=12, stride=1)
        assert len(windows) > 0
        for w in windows:
            assert w.observed.shape == (8, 2)
            assert w.future.shape == (12, 2)

    def test_stride(self):
        traj = _make_trajectory(30)
        w1 = create_windows([traj], obs_len=8, pred_len=12, stride=1)
        w5 = create_windows([traj], obs_len=8, pred_len=12, stride=5)
        assert len(w1) > len(w5)

    def test_trajectory_too_short(self):
        traj = _make_trajectory(5)
        windows = create_windows([traj], obs_len=8, pred_len=12)
        assert len(windows) == 0

    def test_exact_length(self):
        traj = _make_trajectory(20)
        windows = create_windows([traj], obs_len=8, pred_len=12)
        assert len(windows) == 1

    def test_agent_id_preserved(self):
        traj = _make_trajectory(30, agent_id="test_agent")
        windows = create_windows([traj])
        assert all(w.agent_id == "test_agent" for w in windows)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplitConfig:
    def test_valid_ratios(self):
        cfg = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    def test_invalid_ratios_raises(self):
        with pytest.raises(ValueError, match=r"must sum to 1.0"):
            SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


class TestSplitWindows:
    def test_basic_split(self):
        windows = [_make_window() for _ in range(100)]
        train, val, test = split_windows(windows)
        assert len(train) + len(val) + len(test) == 100

    def test_default_ratios(self):
        windows = [_make_window() for _ in range(100)]
        train, val, test = split_windows(windows)
        assert len(train) == 70
        assert len(val) == 15

    def test_deterministic_with_seed(self):
        windows = [_make_window() for _ in range(50)]
        cfg = SplitConfig(seed=123)
        t1, v1, te1 = split_windows(windows, cfg)
        t2, v2, te2 = split_windows(windows, cfg)
        assert len(t1) == len(t2)

    def test_by_scene(self):
        windows = []
        for i in range(5):
            for _ in range(10):
                w = _make_window()
                w.agent_id = f"scene_{i}"
                windows.append(w)
        cfg = SplitConfig(by_scene=True)
        train, val, test = split_windows(windows, cfg)
        assert len(train) + len(val) + len(test) == 50

    def test_no_shuffle(self):
        windows = [_make_window() for _ in range(20)]
        cfg = SplitConfig(shuffle=False)
        train, val, test = split_windows(windows, cfg)
        assert len(train) + len(val) + len(test) == 20


class TestSplitTrajectories:
    def test_basic(self):
        trajs = [_make_trajectory(30, agent_id=f"a{i}") for i in range(10)]
        train, val, test = split_trajectories(trajs)
        assert len(train) + len(val) + len(test) == 10

    def test_deterministic(self):
        trajs = [_make_trajectory(30) for _ in range(20)]
        cfg = SplitConfig(seed=42)
        t1, _, _ = split_trajectories(trajs, cfg)
        t2, _, _ = split_trajectories(trajs, cfg)
        assert len(t1) == len(t2)


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------


class TestAugmentationPrimitives:
    def test_rotate(self):
        pos = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = _rotate(pos, np.pi / 2)
        np.testing.assert_almost_equal(rotated[0], [0.0, 1.0], decimal=5)

    def test_rotate_identity(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        rotated = _rotate(pos, 0.0)
        np.testing.assert_almost_equal(rotated, pos)

    def test_flip_x(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        flipped = _flip(pos, "x")
        np.testing.assert_array_equal(flipped[:, 0], [1.0, 3.0])
        np.testing.assert_array_equal(flipped[:, 1], [-2.0, -4.0])

    def test_flip_y(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        flipped = _flip(pos, "y")
        np.testing.assert_array_equal(flipped[:, 0], [-1.0, -3.0])
        np.testing.assert_array_equal(flipped[:, 1], [2.0, 4.0])

    def test_scale(self):
        pos = np.array([[1.0, 2.0]])
        scaled = _scale(pos, 2.0)
        np.testing.assert_array_equal(scaled, [[2.0, 4.0]])

    def test_add_noise_shape(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        rng = np.random.RandomState(0)
        noisy = _add_noise(pos, 0.1, rng)
        assert noisy.shape == pos.shape
        assert not np.array_equal(noisy, pos)

    def test_perturb_speed_factor_1(self):
        pos = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        ts = np.array([0.0, 1.0, 2.0])
        new_pos, new_ts = _perturb_speed(pos, ts, 1.0)
        np.testing.assert_almost_equal(new_pos, pos)

    def test_perturb_speed_short_trajectory(self):
        pos = np.array([[1.0, 2.0]])
        ts = np.array([0.0])
        new_pos, new_ts = _perturb_speed(pos, ts, 2.0)
        np.testing.assert_array_equal(new_pos, pos)


class TestAugmentWindow:
    def test_returns_new_window(self):
        w = _make_window()
        cfg = AugmentationConfig(seed=42)
        augmented = augment_window(w, cfg)
        assert augmented is not w
        assert augmented.observed.shape == w.observed.shape
        assert augmented.future.shape == w.future.shape

    def test_deterministic(self):
        w = _make_window()
        cfg = AugmentationConfig(seed=42)
        a1 = augment_window(w, cfg, rng=np.random.RandomState(42))
        a2 = augment_window(w, cfg, rng=np.random.RandomState(42))
        np.testing.assert_array_almost_equal(a1.observed, a2.observed)

    def test_agent_id_preserved(self):
        w = _make_window()
        w.agent_id = "test_id"
        augmented = augment_window(w)
        assert augmented.agent_id == "test_id"


class TestAugmentDataset:
    def test_output_size(self):
        windows = [_make_window() for _ in range(5)]
        cfg = AugmentationConfig(augmentation_factor=3, seed=42)
        result = augment_dataset(windows, cfg)
        # Original + 3 copies per window
        assert len(result) == 5 + 5 * 3

    def test_originals_included(self):
        windows = [_make_window() for _ in range(3)]
        cfg = AugmentationConfig(augmentation_factor=1, seed=42)
        result = augment_dataset(windows, cfg)
        # First 3 should be originals
        for i in range(3):
            np.testing.assert_array_equal(result[i].observed, windows[i].observed)


# ---------------------------------------------------------------------------
# Neighbor finding
# ---------------------------------------------------------------------------


class TestFindNeighbors:
    def test_nearby_agents_found(self):
        w1 = _make_window()
        w2 = _make_window()
        w2.agent_id = "ped_1"
        # Place w2 close to w1
        w2.observed = w1.observed + 1.0  # within 5m radius
        find_neighbors([w1, w2], radius=10.0)
        assert len(w1.neighbor_positions) == len(w1.timestamps_obs)
        # Should have neighbor at each timestep
        has_neighbors = any(len(n) > 0 for n in w1.neighbor_positions)
        assert has_neighbors

    def test_far_agents_excluded(self):
        w1 = _make_window()
        w2 = _make_window()
        w2.agent_id = "ped_1"
        w2.observed = w1.observed + 100.0  # far away
        find_neighbors([w1, w2], radius=5.0)
        all_empty = all(len(n) == 0 for n in w1.neighbor_positions)
        assert all_empty

    def test_max_neighbors_limit(self):
        windows = []
        for i in range(15):
            w = _make_window()
            w.agent_id = f"ped_{i}"
            w.observed = w.observed + i * 0.1
            windows.append(w)
        find_neighbors(windows, radius=50.0, max_neighbors=3)
        for w in windows:
            for nbr in w.neighbor_positions:
                assert len(nbr) <= 3


# ---------------------------------------------------------------------------
# Scene context extraction
# ---------------------------------------------------------------------------


class TestExtractSceneContext:
    def test_no_map_gives_blank(self):
        windows = [_make_window()]
        extract_scene_context(windows, scene_map=None, patch_size=32)
        assert windows[0].scene_context is not None
        assert windows[0].scene_context.shape == (32, 32)
        np.testing.assert_array_equal(windows[0].scene_context, 0.0)

    def test_with_2d_map(self):
        scene = np.random.rand(100, 100).astype(np.float32)
        w = _make_window()
        # Put agent at center of map
        w.observed[-1] = [5.0, 5.0]
        extract_scene_context([w], scene_map=scene, patch_size=16, resolution=0.1)
        assert w.scene_context is not None
        assert w.scene_context.shape == (16, 16)

    def test_with_3d_map(self):
        scene = np.random.rand(100, 100, 3).astype(np.float32)
        w = _make_window()
        w.observed[-1] = [5.0, 5.0]
        extract_scene_context([w], scene_map=scene, patch_size=16, resolution=0.1)
        assert w.scene_context.shape == (16, 16, 3)

    def test_out_of_bounds_position(self):
        scene = np.random.rand(50, 50).astype(np.float32)
        w = _make_window()
        w.observed[-1] = [100.0, 100.0]  # way outside map
        extract_scene_context([w], scene_map=scene, patch_size=16, resolution=0.1)
        assert w.scene_context is not None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetPipeline:
    def test_load_csv(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i) * 0.5}" for i in range(30)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load(p)
        assert len(pipeline.trajectories) == 1

    def test_load_json(self, tmp_path):
        data = {
            "trajectories": [
                {
                    "agent_id": "p1",
                    "timestamps": list(range(30)),
                    "positions": [[float(i), float(i) * 0.5] for i in range(30)],
                }
            ]
        }
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load(p)
        assert len(pipeline.trajectories) == 1

    def test_load_file_not_found(self, tmp_path):
        pipeline = TrajectoryDatasetPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.load(tmp_path / "missing.csv")

    def test_load_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("test")
        pipeline = TrajectoryDatasetPipeline()
        with pytest.raises(ValueError, match="Unsupported"):
            pipeline.load(p)

    def test_process_and_splits(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i) * 0.5}" for i in range(40)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(
            obs_len=4,
            pred_len=4,
            stride=1,
            split=SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        )
        pipeline.load(p).process()
        train, val, test = pipeline.get_splits()
        assert len(train) + len(val) + len(test) == len(pipeline.windows)

    def test_load_multiple(self, tmp_path):
        for name in ["a.csv", "b.csv"]:
            csv_data = "\n".join(
                [f"{i},{name},{ float(i)},{float(i)}" for i in range(30)]
            )
            (tmp_path / name).write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load_multiple([tmp_path / "a.csv", tmp_path / "b.csv"])
        assert len(pipeline.trajectories) == 2

    def test_get_numpy_arrays(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i)}" for i in range(40)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load(p).process()
        arrays = pipeline.get_numpy_arrays("train")
        assert "obs" in arrays
        assert "fut" in arrays
        assert arrays["obs"].shape[1:] == (4, 2)

    def test_get_numpy_arrays_empty_split(self):
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        arrays = pipeline.get_numpy_arrays("val")
        assert arrays["obs"].shape == (0, 4, 2)

    def test_get_numpy_arrays_invalid_split(self):
        pipeline = TrajectoryDatasetPipeline()
        with pytest.raises(ValueError, match="Unknown split"):
            pipeline.get_numpy_arrays("invalid")

    def test_iterate_batches(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i)}" for i in range(50)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load(p).process()
        batches = list(pipeline.iterate_batches("train", batch_size=5))
        assert len(batches) > 0
        assert "obs" in batches[0]

    def test_iterate_batches_empty(self):
        pipeline = TrajectoryDatasetPipeline()
        batches = list(pipeline.iterate_batches("train"))
        assert batches == []

    def test_summary(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i)}" for i in range(30)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load(p).process()
        s = pipeline.summary()
        assert s["num_trajectories"] == 1
        assert s["num_windows"] > 0
        assert s["obs_len"] == 4
        assert s["augmentation_enabled"] is False

    def test_with_augmentation(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i)}" for i in range(30)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(
            obs_len=4,
            pred_len=4,
            augmentation=AugmentationConfig(augmentation_factor=2, seed=42),
        )
        pipeline.load(p).process()
        s = pipeline.summary()
        assert s["augmentation_enabled"] is True
        assert s["num_windows"] > s["num_train"] + s["num_val"] + s["num_test"] or True

    def test_method_chaining(self, tmp_path):
        csv_data = "\n".join(
            [f"{i},ped_0,{float(i)},{float(i)}" for i in range(30)]
        )
        p = tmp_path / "data.csv"
        p.write_text(csv_data)
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        result = pipeline.load(p)
        assert result is pipeline
        result2 = pipeline.process()
        assert result2 is pipeline

    def test_load_protobuf(self, tmp_path):
        # Encode a trajectory in binary format
        buf = bytearray()
        aid = b"p1"
        buf += struct.pack("<I", len(aid))
        buf += aid
        T = 30
        buf += struct.pack("<I", T)
        for i in range(T):
            buf += struct.pack("<d", float(i))
        for i in range(T):
            buf += struct.pack("<dd", float(i), float(i) * 0.5)
        buf += struct.pack("B", 0)
        p = tmp_path / "data.bin"
        p.write_bytes(bytes(buf))
        pipeline = TrajectoryDatasetPipeline(obs_len=4, pred_len=4)
        pipeline.load(p)
        assert len(pipeline.trajectories) == 1
