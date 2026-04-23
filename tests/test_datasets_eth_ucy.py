"""Tests for navirl.data.datasets — ETHUCYDataset parsing and SocialDataset edge paths.

Covers ETH/UCY scene file parsing, ``_find_scene_file`` fallback behavior,
``TrajectoryDataset.load`` error handling, and the small public-API helpers
(``get_scene``, ``to_numpy``) that aren't exercised by ``tests/test_data.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from navirl.data.datasets import ETHUCYDataset, SocialDataset

# ---------------------------------------------------------------------------
# ETHUCYDataset — file discovery
# ---------------------------------------------------------------------------


def _write_eth_file(path: Path, rows: list[tuple[float, int, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for frame, ped, x, y in rows:
            f.write(f"{frame}\t{ped}\t{x}\t{y}\n")


class TestETHUCYFileDiscovery:
    def test_finds_top_level_txt(self, tmp_path):
        _write_eth_file(tmp_path / "eth.txt", [(0.0, 1, 1.0, 2.0)])
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_finds_scene_subdir_txt(self, tmp_path):
        scene_dir = tmp_path / "hotel"
        scene_dir.mkdir()
        _write_eth_file(scene_dir / "hotel.txt", [(0.0, 1, 1.0, 2.0)])
        ds = ETHUCYDataset(scenes=["hotel"])
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_finds_scene_subdir_csv_true_pos(self, tmp_path):
        scene_dir = tmp_path / "univ"
        scene_dir.mkdir()
        # The candidate list looks for `true_pos_.csv` directly
        (scene_dir / "true_pos_.csv").write_text("0.0\t1\t1.0\t2.0\n", encoding="utf-8")
        ds = ETHUCYDataset(scenes=["univ"])
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_recursive_fallback(self, tmp_path):
        """When no candidate path matches, a recursive ``rglob`` search runs."""
        deep = tmp_path / "raw" / "extracted"
        deep.mkdir(parents=True)
        # File named with the scene prefix and a recognized suffix
        _write_eth_file(deep / "zara1_data.txt", [(0.0, 1, 0.0, 0.0)])
        ds = ETHUCYDataset(scenes=["zara1"])
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_recursive_fallback_skips_unknown_suffix(self, tmp_path):
        """Files that match the scene name but have a non-recognized suffix are ignored."""
        (tmp_path / "zara2.dat").write_text("garbage")
        ds = ETHUCYDataset(scenes=["zara2"])
        with pytest.raises(FileNotFoundError):
            ds.load(tmp_path)

    def test_missing_scene_raises(self, tmp_path):
        # Provide eth but request hotel — the missing one should raise
        _write_eth_file(tmp_path / "eth.txt", [(0.0, 1, 1.0, 2.0)])
        ds = ETHUCYDataset(scenes=["eth", "hotel"])
        with pytest.raises(FileNotFoundError, match="hotel"):
            ds.load(tmp_path)

    def test_load_nonexistent_path_raises(self, tmp_path):
        ds = ETHUCYDataset(scenes=["eth"])
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ds.load(tmp_path / "missing_dir")


# ---------------------------------------------------------------------------
# ETHUCYDataset — file parsing
# ---------------------------------------------------------------------------


class TestETHUCYParsing:
    def test_parses_tab_separated(self, tmp_path):
        rows = [
            (0.0, 1, 1.0, 2.0),
            (1.0, 1, 1.5, 2.5),
            (2.0, 1, 2.0, 3.0),
            (0.0, 2, 5.0, 5.0),
            (1.0, 2, 5.5, 5.5),
        ]
        _write_eth_file(tmp_path / "eth.txt", rows)
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        scene = ds.get_scene(0)
        assert len(scene) == 2  # two pedestrians
        # Trajectory for ped 1 should have 3 frames sorted
        ped1 = next(t for t in scene if t.agent_id == 1)
        assert len(ped1) == 3
        np.testing.assert_allclose(ped1.timestamps, [0.0, 1.0, 2.0])

    def test_falls_back_to_whitespace_split(self, tmp_path):
        """When ``delim='auto'`` and tab-split yields too few fields, retries on whitespace."""
        path = tmp_path / "eth.txt"
        # Space-separated rather than tab-separated
        path.write_text("0.0 1 1.0 2.0\n1.0 1 1.5 2.5\n", encoding="utf-8")
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        scene = ds.get_scene(0)
        assert len(scene) == 1

    def test_explicit_delimiter(self, tmp_path):
        path = tmp_path / "eth.txt"
        path.write_text("0.0,1,1.0,2.0\n1.0,1,1.5,2.5\n", encoding="utf-8")
        ds = ETHUCYDataset(scenes=["eth"], delim=",")
        ds.load(tmp_path)
        scene = ds.get_scene(0)
        assert len(scene) == 1

    def test_skips_blank_and_short_lines(self, tmp_path):
        path = tmp_path / "eth.txt"
        path.write_text(
            "0.0\t1\t1.0\t2.0\n"
            "\n"  # blank line
            "1.0\t1\n"  # too few fields
            "2.0\t1\t3.0\t4.0\n",
            encoding="utf-8",
        )
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        scene = ds.get_scene(0)
        assert len(scene) == 1
        ped1 = scene[0]
        assert len(ped1) == 2  # only the two valid rows

    def test_skips_unparseable_rows(self, tmp_path):
        path = tmp_path / "eth.txt"
        path.write_text(
            "0.0\t1\t1.0\t2.0\n"
            "bad\trow\twithtext\there\n"
            "1.0\t1\t3.0\t4.0\n",
            encoding="utf-8",
        )
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        scene = ds.get_scene(0)
        assert len(scene[0]) == 2

    def test_sorts_frames_per_pedestrian(self, tmp_path):
        rows = [
            (5.0, 1, 5.0, 5.0),
            (1.0, 1, 1.0, 1.0),
            (3.0, 1, 3.0, 3.0),
        ]
        _write_eth_file(tmp_path / "eth.txt", rows)
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        ped = ds.get_scene(0)[0]
        # Should be sorted ascending by timestamp
        assert list(ped.timestamps) == [1.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# TrajectoryDataset — public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_get_scene_before_load_raises(self):
        ds = ETHUCYDataset(scenes=["eth"])
        with pytest.raises(RuntimeError, match="not loaded"):
            ds.get_scene(0)

    def test_to_numpy_aggregates_all_scenes(self, tmp_path):
        # Two scenes under the same root
        _write_eth_file(tmp_path / "eth.txt", [(0.0, 1, 1.0, 2.0), (1.0, 1, 1.5, 2.5)])
        scene_dir = tmp_path / "hotel"
        scene_dir.mkdir()
        _write_eth_file(scene_dir / "hotel.txt", [(0.0, 9, 9.0, 9.0)])
        ds = ETHUCYDataset(scenes=["eth", "hotel"])
        ds.load(tmp_path)
        arr = ds.to_numpy()
        # 2 + 1 = 3 positions
        assert arr.shape == (3, 2)

    def test_num_scenes_before_load_raises(self):
        ds = ETHUCYDataset(scenes=["eth"])
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = ds.num_scenes


# ---------------------------------------------------------------------------
# SocialDataset — edge paths
# ---------------------------------------------------------------------------


class TestSocialDatasetEdges:
    def test_load_single_file_path(self, tmp_path):
        """Passing a file path directly works (vs a directory)."""
        f = tmp_path / "scene.csv"
        with f.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "agent_id", "x", "y"])
            writer.writerow([0.0, "a", 1.0, 2.0])
            writer.writerow([1.0, "a", 1.5, 2.5])
        ds = SocialDataset(has_header=True)
        ds.load(f)
        assert ds.num_scenes == 1

    def test_load_directory_no_csv_falls_back_to_txt(self, tmp_path):
        f = tmp_path / "scene.txt"
        with f.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([0.0, "a", 1.0, 2.0])
            writer.writerow([1.0, "a", 1.5, 2.5])
        ds = SocialDataset(has_header=False)
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_directory_with_no_data_files_raises(self, tmp_path):
        # Empty directory
        (tmp_path / "notes.md").write_text("nothing here")
        ds = SocialDataset()
        with pytest.raises(FileNotFoundError, match="No CSV/TXT"):
            ds.load(tmp_path)

    def test_skips_short_rows(self, tmp_path):
        f = tmp_path / "scene.csv"
        with f.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "agent_id", "x", "y"])
            writer.writerow([0.0, "a", 1.0, 2.0])
            writer.writerow([1.0, "a"])  # too few columns
            writer.writerow([2.0, "a", 1.5, 2.5])
        ds = SocialDataset(has_header=True)
        ds.load(f)
        scene = ds.get_scene(0)
        assert len(scene[0]) == 2  # only the two valid rows

    def test_skips_unparseable_rows(self, tmp_path):
        f = tmp_path / "scene.csv"
        with f.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "agent_id", "x", "y"])
            writer.writerow([0.0, "a", 1.0, 2.0])
            writer.writerow(["bad", "a", "x", "y"])  # non-numeric
            writer.writerow([1.0, "a", 2.0, 3.0])
        ds = SocialDataset(has_header=True)
        ds.load(f)
        scene = ds.get_scene(0)
        assert len(scene[0]) == 2
