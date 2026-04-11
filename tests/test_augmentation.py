"""Tests for navirl.data.augmentation — trajectory augmentation functions."""

from __future__ import annotations

import math

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
from navirl.data.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(n: int = 10, with_velocities: bool = True) -> Trajectory:
    """Create a simple straight-line trajectory for testing."""
    ts = np.linspace(0.0, 1.0, n)
    positions = np.column_stack([ts, ts * 2.0])  # diagonal line
    velocities = np.ones((n, 2)) * np.array([1.0, 2.0]) if with_velocities else None
    return Trajectory(timestamps=ts, positions=positions, velocities=velocities, agent_id="test")


# ---------------------------------------------------------------------------
# rotate_trajectory
# ---------------------------------------------------------------------------


class TestRotateTrajectory:
    def test_identity_rotation(self):
        traj = _make_trajectory()
        result = rotate_trajectory(traj, 0.0)
        np.testing.assert_allclose(result.positions, traj.positions, atol=1e-12)

    def test_90_degree_rotation(self):
        traj = _make_trajectory()
        result = rotate_trajectory(traj, math.pi / 2)
        # After 90° CCW: (x, y) -> (-y, x)
        np.testing.assert_allclose(result.positions[:, 0], -traj.positions[:, 1], atol=1e-12)
        np.testing.assert_allclose(result.positions[:, 1], traj.positions[:, 0], atol=1e-12)

    def test_180_degree_rotation(self):
        traj = _make_trajectory()
        result = rotate_trajectory(traj, math.pi)
        np.testing.assert_allclose(result.positions, -traj.positions, atol=1e-12)

    def test_rotates_velocities(self):
        traj = _make_trajectory(with_velocities=True)
        result = rotate_trajectory(traj, math.pi / 2)
        np.testing.assert_allclose(result.velocities[:, 0], -traj.velocities[:, 1], atol=1e-12)
        np.testing.assert_allclose(result.velocities[:, 1], traj.velocities[:, 0], atol=1e-12)

    def test_no_velocities(self):
        traj = _make_trajectory(with_velocities=False)
        result = rotate_trajectory(traj, math.pi / 4)
        assert result.velocities is None

    def test_preserves_timestamps(self):
        traj = _make_trajectory()
        result = rotate_trajectory(traj, 0.5)
        np.testing.assert_array_equal(result.timestamps, traj.timestamps)

    def test_preserves_agent_id(self):
        traj = _make_trajectory()
        result = rotate_trajectory(traj, 0.5)
        assert result.agent_id == traj.agent_id

    def test_full_rotation_returns_to_original(self):
        traj = _make_trajectory()
        result = rotate_trajectory(traj, 2 * math.pi)
        np.testing.assert_allclose(result.positions, traj.positions, atol=1e-12)


# ---------------------------------------------------------------------------
# mirror_trajectory
# ---------------------------------------------------------------------------


class TestMirrorTrajectory:
    def test_mirror_x_flips_y(self):
        traj = _make_trajectory()
        result = mirror_trajectory(traj, axis="x")
        np.testing.assert_allclose(result.positions[:, 0], traj.positions[:, 0])
        np.testing.assert_allclose(result.positions[:, 1], -traj.positions[:, 1])

    def test_mirror_y_flips_x(self):
        traj = _make_trajectory()
        result = mirror_trajectory(traj, axis="y")
        np.testing.assert_allclose(result.positions[:, 0], -traj.positions[:, 0])
        np.testing.assert_allclose(result.positions[:, 1], traj.positions[:, 1])

    def test_mirror_velocities_x(self):
        traj = _make_trajectory(with_velocities=True)
        result = mirror_trajectory(traj, axis="x")
        np.testing.assert_allclose(result.velocities[:, 0], traj.velocities[:, 0])
        np.testing.assert_allclose(result.velocities[:, 1], -traj.velocities[:, 1])

    def test_mirror_velocities_y(self):
        traj = _make_trajectory(with_velocities=True)
        result = mirror_trajectory(traj, axis="y")
        np.testing.assert_allclose(result.velocities[:, 0], -traj.velocities[:, 0])
        np.testing.assert_allclose(result.velocities[:, 1], traj.velocities[:, 1])

    def test_no_velocities(self):
        traj = _make_trajectory(with_velocities=False)
        result = mirror_trajectory(traj, axis="x")
        assert result.velocities is None

    def test_invalid_axis_raises(self):
        traj = _make_trajectory()
        with pytest.raises(ValueError, match="axis must be"):
            mirror_trajectory(traj, axis="z")

    def test_double_mirror_returns_original(self):
        traj = _make_trajectory()
        result = mirror_trajectory(mirror_trajectory(traj, "x"), "x")
        np.testing.assert_allclose(result.positions, traj.positions)


# ---------------------------------------------------------------------------
# scale_trajectory
# ---------------------------------------------------------------------------


class TestScaleTrajectory:
    def test_scale_by_one(self):
        traj = _make_trajectory()
        result = scale_trajectory(traj, 1.0)
        np.testing.assert_allclose(result.positions, traj.positions)

    def test_scale_doubles_positions(self):
        traj = _make_trajectory()
        result = scale_trajectory(traj, 2.0)
        np.testing.assert_allclose(result.positions, traj.positions * 2.0)

    def test_scale_velocities(self):
        traj = _make_trajectory(with_velocities=True)
        result = scale_trajectory(traj, 3.0)
        np.testing.assert_allclose(result.velocities, traj.velocities * 3.0)

    def test_scale_no_velocities(self):
        traj = _make_trajectory(with_velocities=False)
        result = scale_trajectory(traj, 2.0)
        assert result.velocities is None

    def test_scale_by_zero(self):
        traj = _make_trajectory()
        result = scale_trajectory(traj, 0.0)
        np.testing.assert_allclose(result.positions, np.zeros_like(traj.positions))


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------


class TestAddNoise:
    def test_zero_noise_preserves(self):
        traj = _make_trajectory()
        result = add_noise(traj, std=0.0, seed=42)
        np.testing.assert_allclose(result.positions, traj.positions, atol=1e-12)

    def test_noise_changes_positions(self):
        traj = _make_trajectory()
        result = add_noise(traj, std=1.0, seed=42)
        assert not np.allclose(result.positions, traj.positions)

    def test_noise_is_reproducible(self):
        traj = _make_trajectory()
        r1 = add_noise(traj, std=0.5, seed=123)
        r2 = add_noise(traj, std=0.5, seed=123)
        np.testing.assert_array_equal(r1.positions, r2.positions)

    def test_noise_preserves_velocities(self):
        traj = _make_trajectory(with_velocities=True)
        result = add_noise(traj, std=0.1, seed=0)
        np.testing.assert_array_equal(result.velocities, traj.velocities)

    def test_noise_preserves_timestamps(self):
        traj = _make_trajectory()
        result = add_noise(traj, std=0.1, seed=0)
        np.testing.assert_array_equal(result.timestamps, traj.timestamps)


# ---------------------------------------------------------------------------
# time_warp
# ---------------------------------------------------------------------------


class TestTimeWarp:
    def test_identity_warp(self):
        traj = _make_trajectory()
        result = time_warp(traj, factor=1.0)
        np.testing.assert_allclose(result.timestamps, traj.timestamps)

    def test_slowdown(self):
        traj = _make_trajectory()
        result = time_warp(traj, factor=2.0)
        # Duration should double
        original_duration = traj.timestamps[-1] - traj.timestamps[0]
        new_duration = result.timestamps[-1] - result.timestamps[0]
        np.testing.assert_allclose(new_duration, original_duration * 2.0)

    def test_speedup(self):
        traj = _make_trajectory()
        result = time_warp(traj, factor=0.5)
        original_duration = traj.timestamps[-1] - traj.timestamps[0]
        new_duration = result.timestamps[-1] - result.timestamps[0]
        np.testing.assert_allclose(new_duration, original_duration * 0.5)

    def test_velocities_scaled_inversely(self):
        traj = _make_trajectory(with_velocities=True)
        factor = 2.0
        result = time_warp(traj, factor=factor)
        np.testing.assert_allclose(result.velocities, traj.velocities / factor)

    def test_no_velocities(self):
        traj = _make_trajectory(with_velocities=False)
        result = time_warp(traj, factor=2.0)
        assert result.velocities is None

    def test_positions_preserved(self):
        traj = _make_trajectory()
        result = time_warp(traj, factor=2.0)
        np.testing.assert_allclose(result.positions, traj.positions)

    def test_zero_factor_raises(self):
        traj = _make_trajectory()
        with pytest.raises(ValueError, match="factor must be positive"):
            time_warp(traj, factor=0.0)

    def test_negative_factor_raises(self):
        traj = _make_trajectory()
        with pytest.raises(ValueError, match="factor must be positive"):
            time_warp(traj, factor=-1.0)


# ---------------------------------------------------------------------------
# crop_trajectory
# ---------------------------------------------------------------------------


class TestCropTrajectory:
    def test_crop_middle(self):
        traj = _make_trajectory(n=10)
        result = crop_trajectory(traj, 2, 7)
        assert len(result) == 5
        np.testing.assert_allclose(result.timestamps, traj.timestamps[2:7])
        np.testing.assert_allclose(result.positions, traj.positions[2:7])

    def test_crop_full(self):
        traj = _make_trajectory(n=10)
        result = crop_trajectory(traj, 0, 10)
        assert len(result) == 10
        np.testing.assert_allclose(result.positions, traj.positions)

    def test_crop_velocities(self):
        traj = _make_trajectory(n=10, with_velocities=True)
        result = crop_trajectory(traj, 3, 8)
        np.testing.assert_allclose(result.velocities, traj.velocities[3:8])

    def test_crop_no_velocities(self):
        traj = _make_trajectory(n=10, with_velocities=False)
        result = crop_trajectory(traj, 1, 5)
        assert result.velocities is None

    def test_crop_preserves_agent_id(self):
        traj = _make_trajectory(n=10)
        result = crop_trajectory(traj, 0, 5)
        assert result.agent_id == traj.agent_id


# ---------------------------------------------------------------------------
# AugmentationPipeline
# ---------------------------------------------------------------------------


class TestAugmentationPipeline:
    def test_empty_pipeline(self):
        pipe = AugmentationPipeline()
        traj = _make_trajectory()
        result = pipe(traj)
        np.testing.assert_allclose(result.positions, traj.positions)

    def test_single_transform(self):
        pipe = AugmentationPipeline()
        pipe.add(lambda t: scale_trajectory(t, 2.0))
        traj = _make_trajectory()
        result = pipe(traj)
        np.testing.assert_allclose(result.positions, traj.positions * 2.0)

    def test_chained_transforms(self):
        traj = _make_trajectory()
        pipe = AugmentationPipeline()
        pipe.add(lambda t: scale_trajectory(t, 2.0)).add(lambda t: scale_trajectory(t, 3.0))
        result = pipe(traj)
        np.testing.assert_allclose(result.positions, traj.positions * 6.0)

    def test_len(self):
        pipe = AugmentationPipeline()
        assert len(pipe) == 0
        pipe.add(lambda t: t)
        assert len(pipe) == 1
        pipe.add(lambda t: t)
        assert len(pipe) == 2

    def test_init_with_transforms(self):
        transforms = [
            lambda t: scale_trajectory(t, 2.0),
            lambda t: mirror_trajectory(t, "x"),
        ]
        pipe = AugmentationPipeline(transforms=transforms)
        assert len(pipe) == 2

    def test_compose_rotate_and_mirror(self):
        traj = _make_trajectory()
        pipe = AugmentationPipeline()
        pipe.add(lambda t: rotate_trajectory(t, math.pi))
        pipe.add(lambda t: mirror_trajectory(t, "x"))
        result = pipe(traj)
        # pi rotation: (x,y)->(-x,-y), then mirror x: (-x,-y)->(-x,y)
        np.testing.assert_allclose(result.positions[:, 0], -traj.positions[:, 0], atol=1e-12)
        np.testing.assert_allclose(result.positions[:, 1], traj.positions[:, 1], atol=1e-12)
