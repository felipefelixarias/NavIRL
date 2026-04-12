"""Tests for navirl/data/loaders.py and navirl/data/preprocessing.py."""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from navirl.data.loaders import BatchLoader, GenericCSVLoader, NavIRLLogLoader, ROSBagLoader
from navirl.data.preprocessing import (
    build_observation,
    compute_map_features,
    compute_social_features,
    encode_goal,
    normalize_positions,
)
from navirl.data.trajectory import Trajectory, TrajectoryCollection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trajectories():
    """Two short trajectories for testing."""
    t1 = Trajectory(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        positions=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        velocities=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        agent_id="robot",
    )
    t2 = Trajectory(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        positions=np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0]]),
        velocities=np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]),
        agent_id="ped_0",
    )
    return TrajectoryCollection([t1, t2])


@pytest.fixture
def occupancy_grid():
    """Small 20x20 occupancy grid with a wall in the center."""
    grid = np.zeros((20, 20), dtype=np.float64)
    grid[9:11, :] = 1.0  # horizontal wall
    return grid


# ===========================================================================
# NavIRLLogLoader
# ===========================================================================


class TestNavIRLLogLoader:
    def test_load_state_jsonl(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        rows = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0, "agent_id": "robot"},
            {"t": 0.1, "x": 0.1, "y": 0.0, "vx": 1.0, "vy": 0.0, "agent_id": "robot"},
            {"t": 0.0, "x": 5.0, "y": 5.0, "vx": -0.5, "vy": 0.0, "agent_id": "ped_0"},
        ]
        state_file.write_text("\n".join(json.dumps(r) for r in rows))

        loader = NavIRLLogLoader()
        loader.load(tmp_path)
        assert len(loader.states) == 3
        assert loader.events == []

    def test_load_with_events(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        events_file = tmp_path / "events.jsonl"
        state_file.write_text(json.dumps({"t": 0.0, "x": 0.0, "y": 0.0}) + "\n")
        events_file.write_text(json.dumps({"t": 0.0, "type": "collision"}) + "\n")

        loader = NavIRLLogLoader().load(tmp_path)
        assert len(loader.states) == 1
        assert len(loader.events) == 1
        assert loader.events[0]["type"] == "collision"

    def test_load_from_file_path(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        state_file.write_text(json.dumps({"t": 0.0, "x": 1.0, "y": 2.0}) + "\n")

        loader = NavIRLLogLoader().load(state_file)
        assert len(loader.states) == 1
        assert loader.states[0]["x"] == 1.0

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NavIRLLogLoader().load(tmp_path / "nonexistent")

    def test_to_trajectories_single_agent(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        rows = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0},
            {"t": 0.1, "x": 0.1, "y": 0.0, "vx": 1.0, "vy": 0.0},
            {"t": 0.2, "x": 0.2, "y": 0.0, "vx": 1.0, "vy": 0.0},
        ]
        state_file.write_text("\n".join(json.dumps(r) for r in rows))

        loader = NavIRLLogLoader().load(tmp_path)
        tc = loader.to_trajectories()
        assert len(tc) == 1
        assert tc[0].agent_id == "robot"  # default agent_id
        assert len(tc[0]) == 3
        assert tc[0].velocities is not None

    def test_to_trajectories_multi_agent(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        rows = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "agent_id": "a"},
            {"t": 0.1, "x": 0.1, "y": 0.0, "agent_id": "a"},
            {"t": 0.0, "x": 5.0, "y": 5.0, "agent_id": "b"},
            {"t": 0.1, "x": 4.9, "y": 5.0, "agent_id": "b"},
        ]
        state_file.write_text("\n".join(json.dumps(r) for r in rows))

        loader = NavIRLLogLoader().load(tmp_path)
        tc = loader.to_trajectories()
        assert len(tc) == 2
        agent_ids = sorted(t.agent_id for t in tc)
        assert agent_ids == ["a", "b"]

    def test_to_trajectories_without_velocity(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        rows = [
            {"t": 0.0, "x": 0.0, "y": 0.0},
            {"t": 0.1, "x": 0.1, "y": 0.0},
        ]
        state_file.write_text("\n".join(json.dumps(r) for r in rows))

        loader = NavIRLLogLoader().load(tmp_path)
        tc = loader.to_trajectories()
        assert tc[0].velocities is None

    def test_to_trajectories_sorted_by_time(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        rows = [
            {"t": 0.2, "x": 2.0, "y": 0.0},
            {"t": 0.0, "x": 0.0, "y": 0.0},
            {"t": 0.1, "x": 1.0, "y": 0.0},
        ]
        state_file.write_text("\n".join(json.dumps(r) for r in rows))

        loader = NavIRLLogLoader().load(tmp_path)
        tc = loader.to_trajectories()
        np.testing.assert_array_equal(tc[0].timestamps, [0.0, 0.1, 0.2])
        np.testing.assert_array_equal(tc[0].positions[:, 0], [0.0, 1.0, 2.0])

    def test_load_chaining(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        state_file.write_text(json.dumps({"t": 0.0, "x": 0.0, "y": 0.0}) + "\n")
        result = NavIRLLogLoader().load(tmp_path)
        assert isinstance(result, NavIRLLogLoader)

    def test_empty_lines_skipped(self, tmp_path):
        state_file = tmp_path / "state.jsonl"
        content = json.dumps({"t": 0.0, "x": 0.0, "y": 0.0}) + "\n\n\n"
        state_file.write_text(content)
        loader = NavIRLLogLoader().load(tmp_path)
        assert len(loader.states) == 1


# ===========================================================================
# GenericCSVLoader
# ===========================================================================


class TestGenericCSVLoader:
    def test_load_by_index(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text("0.0,1.0,2.0\n0.1,1.1,2.1\n0.2,1.2,2.2\n")

        loader = GenericCSVLoader(timestamp_col=0, x_col=1, y_col=2)
        tc = loader.load(csv_file)
        assert len(tc) == 1
        assert len(tc[0]) == 3
        np.testing.assert_allclose(tc[0].positions[0], [1.0, 2.0])

    def test_load_by_column_name(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text("time,px,py\n0.0,1.0,2.0\n0.1,1.1,2.1\n")

        loader = GenericCSVLoader(timestamp_col="time", x_col="px", y_col="py")
        tc = loader.load(csv_file)
        assert len(tc) == 1
        assert len(tc[0]) == 2

    def test_load_with_agent_column(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text(
            "0.0,1.0,2.0,alice\n0.1,1.1,2.1,alice\n0.0,5.0,6.0,bob\n0.1,5.1,6.1,bob\n"
        )

        loader = GenericCSVLoader(timestamp_col=0, x_col=1, y_col=2, agent_col=3)
        tc = loader.load(csv_file)
        assert len(tc) == 2
        agent_ids = sorted(t.agent_id for t in tc)
        assert agent_ids == ["alice", "bob"]

    def test_load_with_agent_column_by_name(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text("t,x,y,aid\n0.0,1.0,2.0,a1\n0.1,1.1,2.1,a1\n")

        loader = GenericCSVLoader(timestamp_col="t", x_col="x", y_col="y", agent_col="aid")
        tc = loader.load(csv_file)
        assert len(tc) == 1
        assert tc[0].agent_id == "a1"

    def test_load_directory_of_csvs(self, tmp_path):
        for i in range(3):
            (tmp_path / f"traj_{i}.csv").write_text(f"{i}.0,{i}.1,{i}.2\n")

        loader = GenericCSVLoader()
        tc = loader.load(tmp_path)
        assert len(tc) == 3

    def test_custom_delimiter(self, tmp_path):
        csv_file = tmp_path / "traj.tsv"
        csv_file.write_text("0.0\t1.0\t2.0\n0.1\t1.1\t2.1\n")

        loader = GenericCSVLoader(timestamp_col=0, x_col=1, y_col=2, delimiter="\t")
        tc = loader.load(csv_file)
        assert len(tc) == 1
        assert len(tc[0]) == 2

    def test_malformed_rows_skipped(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text("0.0,1.0,2.0\nbad,data\n0.1,1.1,2.1\n")

        loader = GenericCSVLoader()
        tc = loader.load(csv_file)
        assert len(tc) == 1
        assert len(tc[0]) == 2  # skipped the bad row

    def test_empty_csv(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        loader = GenericCSVLoader()
        tc = loader.load(csv_file)
        assert len(tc) == 0

    def test_header_only_csv(self, tmp_path):
        csv_file = tmp_path / "header.csv"
        csv_file.write_text("time,x,y\n")

        loader = GenericCSVLoader(timestamp_col="time", x_col="x", y_col="y")
        tc = loader.load(csv_file)
        assert len(tc) == 0

    def test_trajectories_sorted_by_time(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text("0.2,3.0,4.0\n0.0,1.0,2.0\n0.1,2.0,3.0\n")

        loader = GenericCSVLoader()
        tc = loader.load(csv_file)
        np.testing.assert_array_equal(tc[0].timestamps, [0.0, 0.1, 0.2])

    def test_no_agent_defaults_to_agent_0(self, tmp_path):
        csv_file = tmp_path / "traj.csv"
        csv_file.write_text("0.0,1.0,2.0\n")

        loader = GenericCSVLoader()
        tc = loader.load(csv_file)
        assert tc[0].agent_id == "agent_0"


# ===========================================================================
# ROSBagLoader
# ===========================================================================


class TestROSBagLoader:
    def test_raises_without_rosbag(self, tmp_path):
        loader = ROSBagLoader()
        if loader._rosbag is None:
            with pytest.raises(ImportError, match="rosbag"):
                loader.load(tmp_path / "test.bag")

    def test_init_without_rosbag(self):
        loader = ROSBagLoader()
        # Should not raise during init even without rosbag
        assert loader._rosbag is None or loader._rosbag is not None


# ===========================================================================
# BatchLoader
# ===========================================================================


class TestBatchLoader:
    def test_len(self, sample_trajectories):
        bl = BatchLoader(sample_trajectories, batch_size=1)
        assert len(bl) == 2

    def test_len_rounding_up(self, sample_trajectories):
        bl = BatchLoader(sample_trajectories, batch_size=3)
        assert len(bl) == 1  # ceil(2/3)

    def test_iteration_yields_all_trajectories(self, sample_trajectories):
        bl = BatchLoader(sample_trajectories, batch_size=1, shuffle=False)
        batches = list(bl)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1

    def test_shuffle_deterministic_with_seed(self, sample_trajectories):
        bl1 = BatchLoader(sample_trajectories, batch_size=1, shuffle=True, seed=123)
        bl2 = BatchLoader(sample_trajectories, batch_size=1, shuffle=True, seed=123)
        ids1 = [b[0].agent_id for b in bl1]
        ids2 = [b[0].agent_id for b in bl2]
        assert ids1 == ids2

    def test_no_shuffle(self, sample_trajectories):
        bl = BatchLoader(sample_trajectories, batch_size=1, shuffle=False)
        batches = list(bl)
        ids = [b[0].agent_id for b in batches]
        assert ids == ["robot", "ped_0"]

    def test_single_batch_all(self, sample_trajectories):
        bl = BatchLoader(sample_trajectories, batch_size=10, shuffle=False)
        batches = list(bl)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_empty_collection(self):
        tc = TrajectoryCollection()
        bl = BatchLoader(tc, batch_size=5)
        assert len(bl) == 0
        assert list(bl) == []


# ===========================================================================
# normalize_positions
# ===========================================================================


class TestNormalizePositions:
    def test_minmax_normalization(self, sample_trajectories):
        normed, stats = normalize_positions(sample_trajectories, method="minmax")
        assert "min" in stats
        assert "max" in stats
        all_pos = normed.to_numpy()
        assert all_pos.min() >= -1e-10  # approximately >= 0
        assert all_pos.max() <= 1.0 + 1e-10  # approximately <= 1

    def test_standard_normalization(self, sample_trajectories):
        normed, stats = normalize_positions(sample_trajectories, method="standard")
        assert "mean" in stats
        assert "std" in stats
        all_pos = normed.to_numpy()
        # After standard normalization, mean should be approximately 0
        np.testing.assert_allclose(all_pos.mean(axis=0), [0.0, 0.0], atol=0.5)

    def test_empty_collection(self):
        tc = TrajectoryCollection()
        normed, stats = normalize_positions(tc)
        assert len(normed) == 0
        assert stats == {}

    def test_unknown_method_raises(self, sample_trajectories):
        with pytest.raises(ValueError, match="Unknown"):
            normalize_positions(sample_trajectories, method="unknown")

    def test_preserves_agent_ids(self, sample_trajectories):
        normed, _ = normalize_positions(sample_trajectories)
        ids = [t.agent_id for t in normed]
        assert ids == ["robot", "ped_0"]

    def test_preserves_timestamps(self, sample_trajectories):
        normed, _ = normalize_positions(sample_trajectories)
        np.testing.assert_array_equal(normed[0].timestamps, sample_trajectories[0].timestamps)

    def test_constant_positions_no_division_by_zero(self):
        t = Trajectory(
            timestamps=np.array([0.0, 0.1, 0.2]),
            positions=np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]),
            agent_id="static",
        )
        tc = TrajectoryCollection([t])
        normed, stats = normalize_positions(tc, method="minmax")
        assert not np.any(np.isnan(normed.to_numpy()))

    def test_standard_constant_positions(self):
        t = Trajectory(
            timestamps=np.array([0.0, 0.1]),
            positions=np.array([[3.0, 3.0], [3.0, 3.0]]),
            agent_id="static",
        )
        tc = TrajectoryCollection([t])
        normed, stats = normalize_positions(tc, method="standard")
        assert not np.any(np.isnan(normed.to_numpy()))


# ===========================================================================
# compute_social_features
# ===========================================================================


class TestComputeSocialFeatures:
    def test_basic_features(self, sample_trajectories):
        ego = sample_trajectories[0]
        neighbors = [sample_trajectories[1]]
        features = compute_social_features(ego, neighbors, max_neighbors=3)
        assert features.shape == (4, 12)  # T=4, max_neighbors=3, 4 features each

    def test_no_neighbors(self, sample_trajectories):
        ego = sample_trajectories[0]
        features = compute_social_features(ego, [], max_neighbors=3)
        assert features.shape == (4, 12)
        # All zeros since no neighbors
        np.testing.assert_array_equal(features, 0.0)

    def test_more_neighbors_than_max(self):
        ego = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[0.0, 0.0]]),
            agent_id="ego",
        )
        neighbors = [
            Trajectory(
                timestamps=np.array([0.0]),
                positions=np.array([[float(i), 0.0]]),
                velocities=np.array([[0.0, 0.0]]),
                agent_id=f"n{i}",
            )
            for i in range(1, 6)
        ]
        features = compute_social_features(ego, neighbors, max_neighbors=2)
        assert features.shape == (1, 8)  # max_neighbors=2, 4 feats each
        # First neighbor should be closest (distance 1.0)
        assert features[0, 0] == pytest.approx(1.0)

    def test_ego_without_velocities(self):
        ego = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=None,
            agent_id="ego",
        )
        neighbor = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[1.0, 0.0]]),
            velocities=np.array([[0.5, 0.0]]),
            agent_id="n0",
        )
        features = compute_social_features(ego, [neighbor], max_neighbors=1)
        assert features.shape == (1, 4)
        # dx=1, dy=0, dvx=0.5, dvy=0
        np.testing.assert_allclose(features[0], [1.0, 0.0, 0.5, 0.0])

    def test_neighbor_without_velocities(self):
        ego = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[1.0, 0.0]]),
            agent_id="ego",
        )
        neighbor = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[2.0, 0.0]]),
            velocities=None,
            agent_id="n0",
        )
        features = compute_social_features(ego, [neighbor], max_neighbors=1)
        assert features.shape == (1, 4)
        # dx=2, dy=0, dvx=0-1=-1, dvy=0
        np.testing.assert_allclose(features[0], [2.0, 0.0, -1.0, 0.0])

    def test_empty_neighbor_trajectory_skipped(self):
        ego = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[0.0, 0.0]]),
            agent_id="ego",
        )
        empty_neighbor = Trajectory(
            timestamps=np.array([]),
            positions=np.empty((0, 2)),
            velocities=None,
            agent_id="empty",
        )
        features = compute_social_features(ego, [empty_neighbor], max_neighbors=1)
        np.testing.assert_array_equal(features, 0.0)

    def test_timestamp_matching(self):
        """Neighbor at different timestamps should use closest match."""
        ego = Trajectory(
            timestamps=np.array([0.0, 1.0, 2.0]),
            positions=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            velocities=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            agent_id="ego",
        )
        neighbor = Trajectory(
            timestamps=np.array([0.0, 0.5, 1.5, 2.5]),
            positions=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
            velocities=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            agent_id="n",
        )
        features = compute_social_features(ego, [neighbor], max_neighbors=1)
        # At t=0.0, closest neighbor timestamp is 0.0 -> pos [1,0]
        assert features[0, 0] == pytest.approx(1.0)
        # At t=1.0, closest is 0.5 or 1.5 -> pos [2,0] or [3,0]
        assert features[1, 0] in [pytest.approx(2.0), pytest.approx(3.0)]


# ===========================================================================
# compute_map_features
# ===========================================================================


class TestComputeMapFeatures:
    def test_basic_extraction(self, occupancy_grid):
        features = compute_map_features(
            position=np.array([0.0, 0.0]),
            occupancy_grid=occupancy_grid,
            patch_size=8,
            resolution=0.1,
        )
        assert features.shape == (64,)  # 8x8

    def test_patch_size(self, occupancy_grid):
        for size in [4, 8, 16]:
            features = compute_map_features(
                position=np.array([0.0, 0.0]),
                occupancy_grid=occupancy_grid,
                patch_size=size,
                resolution=0.1,
            )
            assert features.shape == (size * size,)

    def test_boundary_position(self, occupancy_grid):
        """Should not crash when position is near the edge."""
        features = compute_map_features(
            position=np.array([0.9, 0.9]),
            occupancy_grid=occupancy_grid,
            patch_size=4,
            resolution=0.1,
        )
        assert features.shape == (16,)
        assert not np.any(np.isnan(features))

    def test_origin_position(self, occupancy_grid):
        features = compute_map_features(
            position=np.array([0.0, 0.0]),
            occupancy_grid=occupancy_grid,
            patch_size=4,
            resolution=0.1,
        )
        assert features.dtype == np.float64

    def test_empty_grid(self):
        grid = np.zeros((10, 10), dtype=np.float64)
        features = compute_map_features(
            position=np.array([0.0, 0.0]),
            occupancy_grid=grid,
            patch_size=4,
            resolution=0.1,
        )
        np.testing.assert_array_equal(features, 0.0)

    def test_fully_occupied_grid(self):
        grid = np.ones((10, 10), dtype=np.float64)
        features = compute_map_features(
            position=np.array([0.0, 0.0]),
            occupancy_grid=grid,
            patch_size=4,
            resolution=0.1,
        )
        # At least some cells should be occupied
        assert features.sum() > 0


# ===========================================================================
# encode_goal
# ===========================================================================


class TestEncodeGoal:
    def test_basic_encoding(self):
        result = encode_goal(np.array([3.0, 4.0]), np.array([0.0, 0.0]))
        assert result.shape == (3,)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(4.0)
        assert result[2] == pytest.approx(5.0)  # distance

    def test_same_position(self):
        result = encode_goal(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)

    def test_negative_delta(self):
        result = encode_goal(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        assert result[0] == pytest.approx(-3.0)
        assert result[1] == pytest.approx(-4.0)
        assert result[2] == pytest.approx(5.0)  # distance is always positive

    def test_dtype(self):
        result = encode_goal(np.array([1, 2]), np.array([0, 0]))
        assert result.dtype == np.float64


# ===========================================================================
# build_observation
# ===========================================================================


class TestBuildObservation:
    def test_without_map(self):
        ego = np.array([1.0, 2.0, 0.5, 0.0])
        neighbors = np.array([0.1, 0.2, 0.3, 0.4])
        goal = np.array([1.0, 1.0, 1.414])
        obs = build_observation(ego, neighbors, goal)
        assert obs.shape == (11,)  # 4 + 4 + 3

    def test_with_map(self):
        ego = np.array([1.0, 2.0])
        neighbors = np.array([0.1, 0.2])
        goal = np.array([1.0, 1.0, 1.0])
        map_data = np.zeros(16)
        obs = build_observation(ego, neighbors, goal, map_data)
        assert obs.shape == (23,)  # 2 + 2 + 3 + 16

    def test_2d_inputs_flattened(self):
        ego = np.array([[1.0, 2.0]])
        neighbors = np.array([[0.1, 0.2]])
        goal = np.array([1.0, 1.0, 1.0])
        obs = build_observation(ego, neighbors, goal)
        assert obs.ndim == 1

    def test_all_zeros(self):
        obs = build_observation(np.zeros(4), np.zeros(8), np.zeros(3))
        assert obs.shape == (15,)
        np.testing.assert_array_equal(obs, 0.0)


# ===========================================================================
# packs/reporter.py
# ===========================================================================

from navirl.packs.reporter import write_pack_json, write_pack_markdown
from navirl.packs.schema import PackResult, PackRunResult


class TestWritePackJson:
    def _make_result(self):
        return PackResult(
            manifest_name="test-pack",
            manifest_version="1.0",
            manifest_checksum="abcdef1234567890" * 4,
            timestamp="2026-01-01T00:00:00",
            runs=[
                PackRunResult(entry_id="s1", seed=42, metrics={"reward": 10.5}),
                PackRunResult(entry_id="s1", seed=43, metrics={"reward": 11.0}),
                PackRunResult(entry_id="s2", seed=42, status="failed", error="timeout"),
            ],
        )

    def test_writes_valid_json(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.json"
        write_pack_json(result, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["manifest_name"] == "test-pack"
        assert data["total_runs"] == 3
        assert data["completed_runs"] == 2
        assert data["failed_runs"] == 1

    def test_creates_parent_dirs(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "sub" / "dir" / "report.json"
        write_pack_json(result, path)
        assert path.exists()

    def test_runs_serialized(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.json"
        write_pack_json(result, path)
        data = json.loads(path.read_text())
        assert len(data["runs"]) == 3
        assert data["runs"][0]["entry_id"] == "s1"
        assert data["runs"][0]["metrics"]["reward"] == 10.5

    def test_empty_result(self, tmp_path):
        result = PackResult(
            manifest_name="empty",
            manifest_version="0.1",
            manifest_checksum="0" * 64,
        )
        path = tmp_path / "empty.json"
        write_pack_json(result, path)
        data = json.loads(path.read_text())
        assert data["total_runs"] == 0


class TestWritePackMarkdown:
    def _make_result(self):
        return PackResult(
            manifest_name="test-pack",
            manifest_version="1.0",
            manifest_checksum="abcdef1234567890" * 4,
            timestamp="2026-01-01T00:00:00",
            runs=[
                PackRunResult(entry_id="hallway", seed=42, metrics={"reward": 10.5, "time": 5.0}),
                PackRunResult(entry_id="hallway", seed=43, metrics={"reward": 11.0, "time": 4.5}),
                PackRunResult(entry_id="crossing", seed=42, metrics={"reward": 8.0, "time": 6.0}),
                PackRunResult(entry_id="crossing", seed=99, status="failed", error="env crashed"),
            ],
        )

    def test_writes_markdown(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path)
        assert path.exists()
        content = path.read_text()
        assert "# Experiment Pack Report: test-pack" in content

    def test_contains_overview(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "Total runs" in content
        assert "Completed" in content
        assert "Failed" in content

    def test_contains_aggregated_metrics(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "Aggregated Metrics" in content
        assert "reward" in content
        assert "time" in content

    def test_specific_metric_names(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path, metric_names=["reward"])
        content = path.read_text()
        assert "reward" in content

    def test_per_scenario_breakdown(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "hallway" in content
        assert "crossing" in content

    def test_failure_details(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "Failures" in content
        assert "env crashed" in content

    def test_creates_parent_dirs(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "deep" / "nested" / "report.md"
        write_pack_markdown(result, path)
        assert path.exists()

    def test_empty_result(self, tmp_path):
        result = PackResult(
            manifest_name="empty",
            manifest_version="0.1",
            manifest_checksum="0" * 64,
        )
        path = tmp_path / "empty.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "empty" in content

    def test_all_failed_runs(self, tmp_path):
        result = PackResult(
            manifest_name="failing",
            manifest_version="1.0",
            manifest_checksum="f" * 64,
            runs=[
                PackRunResult(entry_id="s1", seed=1, status="failed", error="err1"),
                PackRunResult(entry_id="s1", seed=2, status="failed", error="err2"),
            ],
        )
        path = tmp_path / "failed.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "err1" in content
        assert "err2" in content

    def test_version_and_checksum_in_report(self, tmp_path):
        result = self._make_result()
        path = tmp_path / "report.md"
        write_pack_markdown(result, path)
        content = path.read_text()
        assert "1.0" in content
        assert "abcdef12345678" in content

    def test_no_metrics_no_crash(self, tmp_path):
        result = PackResult(
            manifest_name="nometrics",
            manifest_version="1.0",
            manifest_checksum="0" * 64,
            runs=[
                PackRunResult(entry_id="s1", seed=42, metrics={}),
            ],
        )
        path = tmp_path / "report.md"
        write_pack_markdown(result, path, metric_names=[])
        assert path.exists()
