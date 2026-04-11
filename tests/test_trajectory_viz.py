"""Tests for navirl/viz/trajectory_viz.py — trajectory visualization utilities."""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from navirl.viz.trajectory_viz import (
    _default_colors,
    _ensure_array,
    _finite_diff,
    animate_trajectory,
    plot_acceleration_profile,
    plot_curvature,
    plot_heading_profile,
    plot_social_distances_over_time,
    plot_trajectories_comparison,
    plot_trajectory,
    plot_trajectory_3d,
    plot_trajectory_heatmap,
    plot_trajectory_uncertainty,
    plot_velocity_profile,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def straight_xy():
    """Simple straight-line trajectory along x-axis."""
    n = 20
    return np.arange(n, dtype=np.float64), np.zeros(n, dtype=np.float64)


@pytest.fixture
def circle_xy():
    """Circular trajectory for curvature/heading tests."""
    t = np.linspace(0, 2 * math.pi, 100)
    return np.cos(t), np.sin(t)


@pytest.fixture
def zigzag_xy():
    """Zigzag trajectory with velocity variation."""
    n = 40
    x = np.arange(n, dtype=np.float64) * 0.5
    y = np.sin(np.arange(n) * 0.5) * 2.0
    return x, y


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestEnsureArray:
    def test_ndarray_passthrough(self):
        a = np.array([1.0, 2.0])
        result = _ensure_array(a)
        assert result is a

    def test_list_conversion(self):
        result = _ensure_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_tuple_conversion(self):
        result = _ensure_array((4.0, 5.0))
        assert isinstance(result, np.ndarray)

    def test_scalar_conversion(self):
        result = _ensure_array(7.0)
        assert isinstance(result, np.ndarray)


class TestFiniteDiff:
    def test_single_element(self):
        arr = np.array([5.0])
        result = _finite_diff(arr)
        np.testing.assert_array_equal(result, [0.0])

    def test_two_elements(self):
        arr = np.array([0.0, 2.0])
        result = _finite_diff(arr, dt=1.0)
        # forward diff at start, backward diff at end
        np.testing.assert_allclose(result, [2.0, 2.0])

    def test_constant_array(self):
        arr = np.ones(10) * 3.0
        result = _finite_diff(arr)
        np.testing.assert_allclose(result, np.zeros(10), atol=1e-12)

    def test_linear_ramp(self):
        arr = np.arange(5, dtype=np.float64)  # 0, 1, 2, 3, 4
        result = _finite_diff(arr, dt=1.0)
        # central diff for interior: (arr[i+1]-arr[i-1])/2 = 1.0
        np.testing.assert_allclose(result, np.ones(5), atol=1e-12)

    def test_custom_dt(self):
        arr = np.array([0.0, 1.0, 4.0])
        result = _finite_diff(arr, dt=0.5)
        # forward: (1-0)/0.5=2, central: (4-0)/1.0=4, backward: (4-1)/0.5=6
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0])


class TestDefaultColors:
    def test_small_n(self):
        colors = _default_colors(5)
        assert len(colors) == 5
        assert all(c.startswith("#") for c in colors)

    def test_large_n(self):
        colors = _default_colors(15)
        assert len(colors) == 15

    def test_zero(self):
        assert _default_colors(0) == []


# ---------------------------------------------------------------------------
# plot_trajectory
# ---------------------------------------------------------------------------


class TestPlotTrajectory:
    def test_basic_plot(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory(x, y)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_with_existing_axes(self, straight_xy):
        x, y = straight_xy
        fig0, ax0 = plt.subplots()
        fig, ax = plot_trajectory(x, y, ax=ax0)
        assert ax is ax0
        assert fig is fig0

    def test_colorby_speed(self, zigzag_xy):
        x, y = zigzag_xy
        fig, ax = plot_trajectory(x, y, colorby_speed=True, dt=0.5)
        assert isinstance(fig, plt.Figure)

    def test_arrow_interval(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory(x, y, arrow_interval=3)
        assert isinstance(fig, plt.Figure)

    def test_no_markers(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory(x, y, marker_start=False, marker_end=False)
        assert isinstance(fig, plt.Figure)

    def test_with_title_and_label(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory(
            x, y, title="Test", label="robot", xlabel="X", ylabel="Y"
        )
        assert ax.get_title() == "Test"

    def test_no_equal_aspect(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory(x, y, equal_aspect=False)
        assert isinstance(fig, plt.Figure)

    def test_empty_arrays(self):
        fig, ax = plot_trajectory([], [])
        assert isinstance(fig, plt.Figure)

    def test_single_point(self):
        fig, ax = plot_trajectory([1.0], [2.0])
        assert isinstance(fig, plt.Figure)

    def test_list_input(self):
        fig, ax = plot_trajectory([0, 1, 2, 3], [0, 1, 0, 1])
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_trajectories_comparison
# ---------------------------------------------------------------------------


class TestPlotTrajectoriesComparison:
    def test_basic_comparison(self):
        trajs = [
            {"x": [0, 1, 2], "y": [0, 0, 0], "label": "A"},
            {"x": [0, 1, 2], "y": [1, 1, 1], "label": "B"},
        ]
        fig, ax = plot_trajectories_comparison(trajs)
        assert isinstance(fig, plt.Figure)

    def test_custom_colors(self):
        trajs = [{"x": [0, 1], "y": [0, 1]}]
        fig, ax = plot_trajectories_comparison(trajs, colors=["red"])
        assert isinstance(fig, plt.Figure)

    def test_no_legend(self):
        trajs = [{"x": [0, 1], "y": [0, 1]}]
        fig, ax = plot_trajectories_comparison(trajs, legend=False)
        assert isinstance(fig, plt.Figure)

    def test_with_linestyle(self):
        trajs = [
            {"x": [0, 1, 2], "y": [0, 1, 0], "linestyle": "--", "color": "blue"},
        ]
        fig, ax = plot_trajectories_comparison(trajs)
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self):
        _, ax0 = plt.subplots()
        trajs = [{"x": [0, 1], "y": [0, 1]}]
        fig, ax = plot_trajectories_comparison(trajs, ax=ax0)
        assert ax is ax0


# ---------------------------------------------------------------------------
# plot_trajectory_heatmap
# ---------------------------------------------------------------------------


class TestPlotTrajectoryHeatmap:
    def test_basic_heatmap(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory_heatmap(x, y)
        assert isinstance(fig, plt.Figure)

    def test_log_scale(self, zigzag_xy):
        x, y = zigzag_xy
        fig, ax = plot_trajectory_heatmap(x, y, log_scale=True)
        assert isinstance(fig, plt.Figure)

    def test_custom_bins(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory_heatmap(x, y, bins=10)
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self, straight_xy):
        x, y = straight_xy
        _, ax0 = plt.subplots()
        fig, ax = plot_trajectory_heatmap(x, y, ax=ax0)
        assert ax is ax0

    def test_gaussian_smoothing(self, zigzag_xy):
        """Test heatmap with Gaussian smoothing (scipy available)."""
        x, y = zigzag_xy
        fig, ax = plot_trajectory_heatmap(x, y, sigma=1.5)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_velocity_profile
# ---------------------------------------------------------------------------


class TestPlotVelocityProfile:
    def test_basic(self):
        vx = np.ones(20)
        vy = np.zeros(20)
        fig, ax = plot_velocity_profile(vx, vy)
        assert isinstance(fig, plt.Figure)

    def test_no_components(self):
        vx = np.ones(10)
        vy = np.ones(10)
        fig, ax = plot_velocity_profile(vx, vy, show_components=False)
        assert isinstance(fig, plt.Figure)

    def test_with_dt(self):
        vx = np.linspace(0, 5, 30)
        vy = np.zeros(30)
        fig, ax = plot_velocity_profile(vx, vy, dt=0.1)
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self):
        _, ax0 = plt.subplots()
        fig, ax = plot_velocity_profile([1, 2], [0, 0], ax=ax0)
        assert ax is ax0


# ---------------------------------------------------------------------------
# plot_acceleration_profile
# ---------------------------------------------------------------------------


class TestPlotAccelerationProfile:
    def test_basic(self, zigzag_xy):
        # Use zigzag positions as velocity proxies
        vx = np.diff(zigzag_xy[0])
        vy = np.diff(zigzag_xy[1])
        fig, ax = plot_acceleration_profile(vx, vy)
        assert isinstance(fig, plt.Figure)

    def test_constant_velocity(self):
        vx = np.ones(20)
        vy = np.zeros(20)
        fig, ax = plot_acceleration_profile(vx, vy)
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self):
        _, ax0 = plt.subplots()
        fig, ax = plot_acceleration_profile([1, 2, 3], [0, 0, 0], ax=ax0)
        assert ax is ax0


# ---------------------------------------------------------------------------
# plot_heading_profile
# ---------------------------------------------------------------------------


class TestPlotHeadingProfile:
    def test_basic(self, circle_xy):
        vx = np.diff(circle_xy[0])
        vy = np.diff(circle_xy[1])
        fig, ax = plot_heading_profile(vx, vy)
        assert isinstance(fig, plt.Figure)

    def test_no_unwrap(self):
        vx = [1, 0, -1, 0, 1]
        vy = [0, 1, 0, -1, 0]
        fig, ax = plot_heading_profile(vx, vy, unwrap=False)
        assert isinstance(fig, plt.Figure)

    def test_radians(self):
        vx = [1, 0, -1]
        vy = [0, 1, 0]
        fig, ax = plot_heading_profile(vx, vy, degrees=False)
        assert "rad" in ax.get_ylabel()

    def test_degrees(self):
        vx = [1, 0, -1]
        vy = [0, 1, 0]
        fig, ax = plot_heading_profile(vx, vy, degrees=True)
        assert "deg" in ax.get_ylabel()

    def test_with_existing_axes(self):
        _, ax0 = plt.subplots()
        fig, ax = plot_heading_profile([1, 2], [0, 1], ax=ax0)
        assert ax is ax0


# ---------------------------------------------------------------------------
# plot_curvature
# ---------------------------------------------------------------------------


class TestPlotCurvature:
    def test_straight_line(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_curvature(x, y)
        assert isinstance(fig, plt.Figure)

    def test_circle_has_curvature(self, circle_xy):
        x, y = circle_xy
        fig, ax = plot_curvature(x, y, dt=0.01)
        assert isinstance(fig, plt.Figure)

    def test_no_clip(self, circle_xy):
        x, y = circle_xy
        fig, ax = plot_curvature(x, y, clip=None)
        assert isinstance(fig, plt.Figure)

    def test_custom_clip(self, circle_xy):
        x, y = circle_xy
        fig, ax = plot_curvature(x, y, clip=5.0)
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self, straight_xy):
        x, y = straight_xy
        _, ax0 = plt.subplots()
        fig, ax = plot_curvature(x, y, ax=ax0)
        assert ax is ax0


# ---------------------------------------------------------------------------
# animate_trajectory
# ---------------------------------------------------------------------------


class TestAnimateTrajectory:
    def test_basic_animation(self, straight_xy):
        x, y = straight_xy
        fig, anim = animate_trajectory(x, y, interval=100)
        assert isinstance(fig, plt.Figure)
        assert anim is not None

    def test_with_background(self, straight_xy):
        x, y = straight_xy
        bg = np.random.rand(10, 10)
        fig, anim = animate_trajectory(x, y, background_img=bg)
        assert isinstance(fig, plt.Figure)

    def test_with_custom_limits(self, straight_xy):
        x, y = straight_xy
        fig, anim = animate_trajectory(x, y, xlim=(-5, 25), ylim=(-5, 5))
        assert isinstance(fig, plt.Figure)

    def test_no_repeat(self, straight_xy):
        x, y = straight_xy
        fig, anim = animate_trajectory(x, y, repeat=False)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_trajectory_3d
# ---------------------------------------------------------------------------


class TestPlotTrajectory3D:
    def test_basic_3d(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory_3d(x, y)
        assert isinstance(fig, plt.Figure)

    def test_custom_time(self, straight_xy):
        x, y = straight_xy
        t = np.arange(len(x)) * 0.25
        fig, ax = plot_trajectory_3d(x, y, t=t)
        assert isinstance(fig, plt.Figure)

    def test_colorby_speed(self, zigzag_xy):
        x, y = zigzag_xy
        fig, ax = plot_trajectory_3d(x, y, colorby_speed=True, dt=0.5)
        assert isinstance(fig, plt.Figure)

    def test_custom_view(self, straight_xy):
        x, y = straight_xy
        fig, ax = plot_trajectory_3d(x, y, elevation=45, azimuth=-30)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_trajectory_uncertainty
# ---------------------------------------------------------------------------


class TestPlotTrajectoryUncertainty:
    def test_basic_uncertainty(self, straight_xy):
        x, y = straight_xy
        n = len(x)
        sigma_x = np.ones(n) * 0.1
        sigma_y = np.ones(n) * 0.1
        fig, ax = plot_trajectory_uncertainty(x, y, sigma_x, sigma_y)
        assert isinstance(fig, plt.Figure)

    def test_with_correlation(self, straight_xy):
        x, y = straight_xy
        n = len(x)
        sigma_x = np.ones(n) * 0.2
        sigma_y = np.ones(n) * 0.15
        rho = np.ones(n) * 0.5
        fig, ax = plot_trajectory_uncertainty(
            x, y, sigma_x, sigma_y, correlation=rho
        )
        assert isinstance(fig, plt.Figure)

    def test_few_ellipses(self, straight_xy):
        x, y = straight_xy
        n = len(x)
        fig, ax = plot_trajectory_uncertainty(
            x, y, np.ones(n) * 0.1, np.ones(n) * 0.1, n_ellipses=3
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_confidence(self, straight_xy):
        x, y = straight_xy
        n = len(x)
        fig, ax = plot_trajectory_uncertainty(
            x, y, np.ones(n) * 0.1, np.ones(n) * 0.1, confidence=0.99
        )
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self, straight_xy):
        x, y = straight_xy
        n = len(x)
        _, ax0 = plt.subplots()
        fig, ax = plot_trajectory_uncertainty(
            x, y, np.ones(n) * 0.1, np.ones(n) * 0.1, ax=ax0
        )
        assert ax is ax0


# ---------------------------------------------------------------------------
# plot_social_distances_over_time
# ---------------------------------------------------------------------------


class TestPlotSocialDistances:
    def test_basic(self):
        positions = [
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (0.5, 0.0), 1: (1.5, 0.0)},
            {0: (1.0, 0.0), 1: (2.0, 0.0)},
        ]
        fig, ax = plot_social_distances_over_time(positions)
        assert isinstance(fig, plt.Figure)

    def test_three_agents(self):
        positions = [
            {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)},
            {0: (0.5, 0.0), 1: (1.5, 0.0), 2: (0.5, 1.0)},
        ]
        fig, ax = plot_social_distances_over_time(positions)
        assert isinstance(fig, plt.Figure)

    def test_no_min_distance_line(self):
        positions = [{0: (0.0, 0.0), 1: (1.0, 0.0)}]
        fig, ax = plot_social_distances_over_time(
            positions, min_distance_line=None
        )
        assert isinstance(fig, plt.Figure)

    def test_no_min_envelope(self):
        positions = [
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (0.5, 0.0), 1: (1.5, 0.0)},
        ]
        fig, ax = plot_social_distances_over_time(
            positions, show_min_envelope=False
        )
        assert isinstance(fig, plt.Figure)

    def test_single_agent_no_pairs(self):
        positions = [{0: (0.0, 0.0)}, {0: (1.0, 0.0)}]
        fig, ax = plot_social_distances_over_time(positions)
        assert isinstance(fig, plt.Figure)

    def test_custom_dt(self):
        positions = [
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (0.5, 0.0), 1: (1.5, 0.0)},
        ]
        fig, ax = plot_social_distances_over_time(positions, dt=0.5)
        assert isinstance(fig, plt.Figure)

    def test_with_existing_axes(self):
        _, ax0 = plt.subplots()
        positions = [
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (0.5, 0.0), 1: (1.5, 0.0)},
        ]
        fig, ax = plot_social_distances_over_time(positions, ax=ax0)
        assert ax is ax0

    def test_agents_appearing_disappearing(self):
        """Some agents not present at every step."""
        positions = [
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (0.5, 0.0)},  # agent 1 missing
            {0: (1.0, 0.0), 1: (2.0, 0.0)},
        ]
        fig, ax = plot_social_distances_over_time(positions)
        assert isinstance(fig, plt.Figure)
