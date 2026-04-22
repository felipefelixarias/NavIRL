"""Coverage tests for navirl.data.datasets — parsers and fallback paths.

Existing ``tests/test_data.py`` covers the happy-path SocialDataset loader and
the obvious error cases, but leaves the actual ETHUCY parser, recursive scene
discovery, txt fallbacks, and ``to_numpy`` / ``get_scene`` uncovered. These
tests exercise those paths with synthesised on-disk data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from navirl.data.datasets import ETHUCYDataset, SocialDataset

# ---------------------------------------------------------------------------
# ETHUCY text parser — full loader path
# ---------------------------------------------------------------------------


def _write_ethucy(path: Path, rows: list[tuple[float, int, float, float]], *, sep: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for frame, ped, x, y in rows:
            f.write(f"{frame}{sep}{ped}{sep}{x}{sep}{y}\n")


class TestETHUCYParser:
    def test_parse_tab_delimited(self, tmp_path):
        rows = [
            (10.0, 1, 0.0, 0.0),
            (20.0, 1, 1.0, 0.0),
            (30.0, 1, 2.0, 0.0),
            (10.0, 2, 5.0, 5.0),
            (20.0, 2, 5.5, 5.0),
        ]
        _write_ethucy(tmp_path / "eth.txt", rows, sep="\t")
        ds = ETHUCYDataset(scenes=["eth"], delim="auto")
        ds.load(tmp_path)
        assert ds.num_scenes == 1
        tc = ds.get_scene(0)
        assert len(tc) == 2  # two pedestrians
        ped1 = next(t for t in tc.trajectories if t.agent_id == 1)
        assert len(ped1) == 3
        np.testing.assert_allclose(ped1.positions[0], [0.0, 0.0])
        np.testing.assert_allclose(ped1.timestamps, [10.0, 20.0, 30.0])

    def test_parse_space_delimited_auto(self, tmp_path):
        """``delim='auto'`` must fall back to whitespace split when tab split fails."""
        rows = [
            (1.0, 7, 0.0, 0.0),
            (2.0, 7, 0.1, 0.2),
            (3.0, 7, 0.2, 0.4),
        ]
        _write_ethucy(tmp_path / "hotel.txt", rows, sep=" ")
        ds = ETHUCYDataset(scenes=["hotel"])
        ds.load(tmp_path)
        tc = ds.get_scene(0)
        assert len(tc) == 1
        assert tc.trajectories[0].agent_id == 7
        assert len(tc.trajectories[0]) == 3

    def test_parse_explicit_delim(self, tmp_path):
        _write_ethucy(tmp_path / "zara1.txt", [(1.0, 3, 0.0, 0.0), (2.0, 3, 1.0, 0.0)], sep=",")
        ds = ETHUCYDataset(scenes=["zara1"], delim=",")
        ds.load(tmp_path)
        tc = ds.get_scene(0)
        assert len(tc) == 1

    def test_skips_blank_and_malformed_lines(self, tmp_path):
        """Blank lines and rows with too few / non-numeric fields are dropped."""
        path = tmp_path / "univ.txt"
        path.write_text(
            "\n"  # blank
            "1.0\t1\t0.0\t0.0\n"  # valid
            "bad\tnot\ta\trow\n"  # non-numeric -> ValueError branch
            "2.0\t1\t1.0\n"  # too few fields
            "3.0\t1\t2.0\t0.0\n"  # valid
            "   \n"  # whitespace-only
        )
        ds = ETHUCYDataset(scenes=["univ"])
        ds.load(tmp_path)
        tc = ds.get_scene(0)
        assert len(tc) == 1
        assert len(tc.trajectories[0]) == 2  # only two valid rows survived

    def test_sorts_rows_by_frame_per_pedestrian(self, tmp_path):
        """Out-of-order input must emerge sorted by timestamp."""
        rows = [
            (30.0, 1, 3.0, 0.0),
            (10.0, 1, 1.0, 0.0),
            (20.0, 1, 2.0, 0.0),
        ]
        _write_ethucy(tmp_path / "eth.txt", rows, sep="\t")
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        traj = ds.get_scene(0).trajectories[0]
        np.testing.assert_allclose(traj.timestamps, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(traj.positions[:, 0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# _find_scene_file — discovery fallbacks
# ---------------------------------------------------------------------------


class TestSceneFileDiscovery:
    def test_nested_scene_directory(self, tmp_path):
        """``root/eth/eth.txt`` layout (one of the tried candidates)."""
        (tmp_path / "eth").mkdir()
        _write_ethucy(tmp_path / "eth" / "eth.txt", [(1.0, 1, 0.0, 0.0)], sep="\t")
        ds = ETHUCYDataset(scenes=["eth"])
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_recursive_fallback_picks_txt(self, tmp_path):
        """File outside the standard candidate paths is found by rglob fallback."""
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        _write_ethucy(deep / "zara2_session.txt", [(1.0, 1, 0.0, 0.0)], sep="\t")
        ds = ETHUCYDataset(scenes=["zara2"])
        ds.load(tmp_path)
        assert ds.num_scenes == 1

    def test_missing_scene_raises(self, tmp_path):
        # Create an unrelated file so the directory is not empty.
        (tmp_path / "something_else.txt").write_text("x\n")
        ds = ETHUCYDataset(scenes=["eth"])
        with pytest.raises(FileNotFoundError, match="scene 'eth'"):
            ds.load(tmp_path)


# ---------------------------------------------------------------------------
# Base class helpers — load path validation, to_numpy, get_scene
# ---------------------------------------------------------------------------


class TestBaseDatasetBehavior:
    def test_load_nonexistent_path_raises(self, tmp_path):
        ds = SocialDataset()
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
            ds.load(missing)

    def test_get_scene_before_load_raises(self):
        ds = SocialDataset()
        with pytest.raises(RuntimeError, match="not loaded"):
            ds.get_scene(0)

    def test_to_numpy_before_load_raises(self):
        ds = SocialDataset()
        with pytest.raises(RuntimeError, match="not loaded"):
            ds.to_numpy()

    def test_to_numpy_stacks_all_scenes(self, tmp_path):
        # Two scenes with 3 rows each => 6 x 2 positions
        for i in range(2):
            p = tmp_path / f"scene_{i}.csv"
            with p.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t", "id", "x", "y"])
                for k in range(3):
                    w.writerow([k * 0.1, f"a{i}", float(k), float(i)])
        ds = SocialDataset(has_header=True)
        ds.load(tmp_path)
        arr = ds.to_numpy()
        assert arr.shape == (6, 2)


# ---------------------------------------------------------------------------
# SocialDataset parsing — txt fallback and malformed-row tolerance
# ---------------------------------------------------------------------------


class TestSocialDatasetEdges:
    def test_falls_back_to_txt_when_no_csv(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("0.0,a,1.0,1.0\n0.1,a,1.1,1.2\n0.2,a,1.2,1.4\n")
        ds = SocialDataset(has_header=False)
        ds.load(tmp_path)
        assert ds.num_scenes == 1
        tc = ds.get_scene(0)
        assert len(tc) == 1
        assert len(tc.trajectories[0]) == 3

    def test_no_files_raises(self, tmp_path):
        # Directory with only an unsupported extension.
        (tmp_path / "README.md").write_text("no data here\n")
        ds = SocialDataset()
        with pytest.raises(FileNotFoundError, match="No CSV/TXT"):
            ds.load(tmp_path)

    def test_skips_short_and_bad_rows(self, tmp_path):
        csv_file = tmp_path / "scene.csv"
        with csv_file.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "id", "x", "y"])
            w.writerow(["0.0", "a", "1.0", "1.0"])
            w.writerow(["short", "row"])  # len < 4 -> skipped
            w.writerow(["notnum", "a", "bad", "bad"])  # ValueError -> skipped
            w.writerow(["0.1", "a", "1.1", "1.2"])
        ds = SocialDataset(has_header=True)
        ds.load(csv_file)
        tc = ds.get_scene(0)
        assert len(tc) == 1
        assert len(tc.trajectories[0]) == 2

    def test_mixed_velocity_column_not_applied(self, tmp_path):
        """If only some rows carry velocities, the parser drops velocity arrays
        rather than failing mid-parse."""
        csv_file = tmp_path / "mixed.csv"
        with csv_file.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "id", "x", "y", "vx", "vy"])
            w.writerow(["0.0", "a", "0.0", "0.0", "1.0", "0.0"])
            w.writerow(["0.1", "a", "0.1", "0.0", "", ""])  # missing vel -> None
        ds = SocialDataset(has_header=True)
        ds.load(csv_file)
        traj = ds.get_scene(0).trajectories[0]
        assert traj.velocities is None
