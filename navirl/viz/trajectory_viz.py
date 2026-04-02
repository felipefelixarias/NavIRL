"""Trajectory visualization utilities for NavIRL.

Provides functions for plotting single and multi-agent trajectories,
velocity/acceleration/heading/curvature profiles, heatmaps, 3D views,
uncertainty envelopes, social distance timeseries, and trajectory animations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception as e:
    logger.debug(f"Failed to set matplotlib backend to Agg: {e}")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401 - registers 3-D projection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_array(data: Any) -> np.ndarray:
    """Convert list / tuple / scalar to numpy array."""
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(data, dtype=np.float64)


def _finite_diff(arr: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """First-order finite differences with same-length output (forward/backward at edges)."""
    out = np.empty_like(arr)
    if len(arr) < 2:
        out[:] = 0.0
        return out
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    out[0] = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out


def _default_colors(n: int) -> list[str]:
    """Return *n* distinguishable colour hex codes."""
    cmap = cm.get_cmap("tab10") if n <= 10 else cm.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i % cmap.N)) for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_trajectory(
    x: Sequence[float],
    y: Sequence[float],
    *,
    ax: plt.Axes | None = None,
    color: str = "#3498db",
    linewidth: float = 1.8,
    marker_start: bool = True,
    marker_end: bool = True,
    marker_size: float = 8.0,
    arrow_interval: int = 0,
    label: str | None = None,
    alpha: float = 1.0,
    colorby_speed: bool = False,
    dt: float = 1.0,
    cmap_name: str = "viridis",
    title: str | None = None,
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    equal_aspect: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single 2-D trajectory.

    Parameters
    ----------
    x, y : array-like
        Position sequences.
    ax : matplotlib Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    color : str
        Line colour (ignored when *colorby_speed* is True).
    linewidth : float
        Line width in points.
    marker_start, marker_end : bool
        Whether to draw start / end markers.
    marker_size : float
        Size of start/end markers.
    arrow_interval : int
        If > 0, draw direction arrows every *arrow_interval* steps.
    label : str, optional
        Legend label.
    alpha : float
        Line transparency.
    colorby_speed : bool
        Colour the trajectory by instantaneous speed.
    dt : float
        Timestep between samples (used for speed colouring).
    cmap_name : str
        Colourmap name when *colorby_speed* is True.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    equal_aspect : bool
        If True, set equal aspect ratio.
    figsize : tuple
        Figure size in inches (only used when creating a new figure).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    xa = _ensure_array(x)
    ya = _ensure_array(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if colorby_speed and len(xa) > 1:
        vx = _finite_diff(xa, dt)
        vy = _finite_diff(ya, dt)
        speed = np.hypot(vx, vy)
        points = np.column_stack([xa, ya]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(speed.min(), speed.max())
        lc = LineCollection(segments, cmap=cmap_name, norm=norm, linewidth=linewidth, alpha=alpha)
        lc.set_array(speed[:-1])
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, pad=0.02)
        cbar.set_label("Speed (m/s)")
        ax.set_xlim(xa.min() - 0.5, xa.max() + 0.5)
        ax.set_ylim(ya.min() - 0.5, ya.max() + 0.5)
    else:
        ax.plot(xa, ya, color=color, linewidth=linewidth, alpha=alpha, label=label)

    if marker_start and len(xa) > 0:
        ax.plot(
            xa[0],
            ya[0],
            "o",
            color="green",
            markersize=marker_size,
            zorder=5,
            label="Start" if label is None else None,
        )
    if marker_end and len(xa) > 0:
        ax.plot(
            xa[-1],
            ya[-1],
            "s",
            color="red",
            markersize=marker_size,
            zorder=5,
            label="End" if label is None else None,
        )

    if arrow_interval > 0 and len(xa) > 1:
        for i in range(0, len(xa) - 1, arrow_interval):
            dx = xa[min(i + 1, len(xa) - 1)] - xa[i]
            dy = ya[min(i + 1, len(ya) - 1)] - ya[i]
            length = math.hypot(dx, dy)
            if length > 1e-8:
                ax.annotate(
                    "",
                    xy=(xa[i] + dx * 0.5, ya[i] + dy * 0.5),
                    xytext=(xa[i], ya[i]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
                )

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_trajectories_comparison(
    trajectories: Sequence[dict[str, Any]],
    *,
    ax: plt.Axes | None = None,
    colors: Sequence[str] | None = None,
    linewidth: float = 1.5,
    title: str = "Trajectory Comparison",
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    equal_aspect: bool = True,
    legend: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Compare multiple trajectories side-by-side.

    Parameters
    ----------
    trajectories : list of dicts
        Each dict must contain ``'x'`` and ``'y'`` keys (array-like).
        Optionally ``'label'``, ``'color'``, ``'linestyle'``.
    colors : list of str, optional
        Override colours for each trajectory.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = _default_colors(len(trajectories))

    for idx, traj in enumerate(trajectories):
        xa = _ensure_array(traj["x"])
        ya = _ensure_array(traj["y"])
        lbl = traj.get("label", f"Trajectory {idx}")
        ls = traj.get("linestyle", "-")
        c = traj.get("color", colors[idx % len(colors)])
        ax.plot(xa, ya, color=c, linewidth=linewidth, linestyle=ls, label=lbl)
        ax.plot(xa[0], ya[0], "o", color=c, markersize=6, zorder=5)
        if len(xa) > 0:
            ax.plot(xa[-1], ya[-1], "s", color=c, markersize=6, zorder=5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    if legend:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_trajectory_heatmap(
    x: Sequence[float],
    y: Sequence[float],
    *,
    bins: int = 50,
    ax: plt.Axes | None = None,
    cmap: str = "hot",
    title: str = "Trajectory Heatmap",
    xlabel: str = "x (m)",
    ylabel: str = "y (m)",
    figsize: tuple[float, float] = (8, 7),
    log_scale: bool = False,
    sigma: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2-D histogram heatmap of positions visited.

    Parameters
    ----------
    x, y : array-like
        Position data (can be concatenated from multiple agents).
    bins : int
        Number of histogram bins per axis.
    log_scale : bool
        If True, use logarithmic colour scale.
    sigma : float
        Gaussian smoothing sigma (pixels).  0 means no smoothing.
    """
    xa = _ensure_array(x)
    ya = _ensure_array(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    hist, xedges, yedges = np.histogram2d(xa, ya, bins=bins)

    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            hist = gaussian_filter(hist, sigma=sigma)
        except ImportError:
            pass  # gracefully skip smoothing

    norm = (
        mcolors.LogNorm(vmin=max(hist[hist > 0].min(), 1e-1), vmax=hist.max())
        if log_scale and hist.max() > 0
        else None
    )

    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        aspect="auto",
        norm=norm,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Visit count")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    return fig, ax


def plot_velocity_profile(
    vx: Sequence[float],
    vy: Sequence[float],
    *,
    dt: float = 1.0,
    ax: plt.Axes | None = None,
    title: str = "Velocity Profile",
    figsize: tuple[float, float] = (10, 4),
    show_components: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot speed and optionally vx/vy over time.

    Parameters
    ----------
    vx, vy : array-like
        Velocity components.
    dt : float
        Timestep for constructing the time axis.
    show_components : bool
        If True, also plot vx and vy individually.
    """
    vxa = _ensure_array(vx)
    vya = _ensure_array(vy)
    speed = np.hypot(vxa, vya)
    t = np.arange(len(speed)) * dt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(t, speed, color="#2c3e50", linewidth=1.6, label="Speed")
    if show_components:
        ax.plot(t, vxa, "--", color="#3498db", linewidth=1.0, alpha=0.7, label="vx")
        ax.plot(t, vya, "--", color="#e74c3c", linewidth=1.0, alpha=0.7, label="vy")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.fill_between(t, 0, speed, alpha=0.12, color="#2c3e50")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_acceleration_profile(
    vx: Sequence[float],
    vy: Sequence[float],
    *,
    dt: float = 1.0,
    ax: plt.Axes | None = None,
    title: str = "Acceleration Profile",
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot acceleration magnitude and components over time.

    Parameters
    ----------
    vx, vy : array-like
        Velocity components.
    dt : float
        Timestep between samples.
    """
    vxa = _ensure_array(vx)
    vya = _ensure_array(vy)
    ax_vals = _finite_diff(vxa, dt)
    ay_vals = _finite_diff(vya, dt)
    amag = np.hypot(ax_vals, ay_vals)
    t = np.arange(len(amag)) * dt

    if ax is None:
        fig, ax_plt = plt.subplots(figsize=figsize)
    else:
        ax_plt = ax
        fig = ax.get_figure()

    ax_plt.plot(t, amag, color="#8e44ad", linewidth=1.6, label="|a|")
    ax_plt.plot(t, ax_vals, "--", color="#2980b9", linewidth=0.9, alpha=0.6, label="ax")
    ax_plt.plot(t, ay_vals, "--", color="#c0392b", linewidth=0.9, alpha=0.6, label="ay")
    ax_plt.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax_plt.fill_between(t, 0, amag, alpha=0.1, color="#8e44ad")
    ax_plt.set_title(title)
    ax_plt.set_xlabel("Time (s)")
    ax_plt.set_ylabel("Acceleration (m/s^2)")
    ax_plt.legend(loc="best", fontsize=8)
    ax_plt.grid(True, alpha=0.3)
    return fig, ax_plt


def plot_heading_profile(
    vx: Sequence[float],
    vy: Sequence[float],
    *,
    dt: float = 1.0,
    unwrap: bool = True,
    ax: plt.Axes | None = None,
    title: str = "Heading Profile",
    figsize: tuple[float, float] = (10, 4),
    degrees: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot heading angle over time derived from velocity.

    Parameters
    ----------
    vx, vy : array-like
        Velocity components.
    dt : float
        Timestep.
    unwrap : bool
        If True, unwrap the angle to remove discontinuities.
    degrees : bool
        If True, plot in degrees; otherwise radians.
    """
    vxa = _ensure_array(vx)
    vya = _ensure_array(vy)
    heading = np.arctan2(vya, vxa)
    if unwrap:
        heading = np.unwrap(heading)
    t = np.arange(len(heading)) * dt

    if degrees:
        heading = np.degrees(heading)
        unit = "deg"
    else:
        unit = "rad"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(t, heading, color="#16a085", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Heading ({unit})")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_curvature(
    x: Sequence[float],
    y: Sequence[float],
    *,
    dt: float = 1.0,
    ax: plt.Axes | None = None,
    title: str = "Path Curvature",
    figsize: tuple[float, float] = (10, 4),
    clip: float | None = 10.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot curvature (1/R) along the trajectory over time.

    Uses the parametric curvature formula:
        kappa = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)

    Parameters
    ----------
    x, y : array-like
        Position data.
    dt : float
        Timestep.
    clip : float or None
        Maximum curvature value to display.  None for no clipping.
    """
    xa = _ensure_array(x)
    ya = _ensure_array(y)
    xd = _finite_diff(xa, dt)
    yd = _finite_diff(ya, dt)
    xdd = _finite_diff(xd, dt)
    ydd = _finite_diff(yd, dt)

    denom = (xd**2 + yd**2) ** 1.5
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    kappa = np.abs(xd * ydd - yd * xdd) / safe_denom
    kappa[denom < 1e-12] = 0.0
    if clip is not None:
        kappa = np.clip(kappa, 0, clip)

    t = np.arange(len(kappa)) * dt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(t, kappa, color="#d35400", linewidth=1.4)
    ax.fill_between(t, 0, kappa, alpha=0.15, color="#d35400")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Curvature (1/m)")
    ax.grid(True, alpha=0.3)
    return fig, ax


def animate_trajectory(
    x: Sequence[float],
    y: Sequence[float],
    *,
    dt: float = 0.1,
    trail_length: int = 30,
    interval: int = 50,
    figsize: tuple[float, float] = (8, 6),
    color: str = "#3498db",
    title: str = "Trajectory Animation",
    repeat: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    background_img: np.ndarray | None = None,
    background_extent: Sequence[float] | None = None,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Create an animated trajectory visualisation.

    Parameters
    ----------
    x, y : array-like
        Position data.
    dt : float
        Timestep for the time label.
    trail_length : int
        Number of past positions to show as a fading trail.
    interval : int
        Milliseconds between frames.
    background_img : ndarray, optional
        Background image (displayed with ``imshow``).
    background_extent : sequence, optional
        Extent ``[xmin, xmax, ymin, ymax]`` for the background image.

    Returns
    -------
    fig : matplotlib Figure
    anim : matplotlib FuncAnimation
    """
    xa = _ensure_array(x)
    ya = _ensure_array(y)
    n = len(xa)

    fig, ax = plt.subplots(figsize=figsize)

    if background_img is not None:
        ext = (
            background_extent
            if background_extent is not None
            else [xa.min() - 1, xa.max() + 1, ya.min() - 1, ya.max() + 1]
        )
        ax.imshow(background_img, extent=ext, origin="lower", alpha=0.5, aspect="auto")

    x_margin = max(1.0, (xa.max() - xa.min()) * 0.1)
    y_margin = max(1.0, (ya.max() - ya.min()) * 0.1)
    ax.set_xlim(xlim if xlim else (xa.min() - x_margin, xa.max() + x_margin))
    ax.set_ylim(ylim if ylim else (ya.min() - y_margin, ya.max() + y_margin))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)

    (trail_line,) = ax.plot([], [], color=color, linewidth=1.2, alpha=0.5)
    (agent_dot,) = ax.plot([], [], "o", color=color, markersize=8, zorder=5)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment="top"
    )
    (start_marker,) = ax.plot([xa[0]], [ya[0]], "o", color="green", markersize=7, zorder=4)
    (end_marker,) = ax.plot([xa[-1]], [ya[-1]], "s", color="red", markersize=7, zorder=4, alpha=0.3)

    def _init():
        trail_line.set_data([], [])
        agent_dot.set_data([], [])
        time_text.set_text("")
        return trail_line, agent_dot, time_text

    def _update(frame: int):
        lo = max(0, frame - trail_length)
        trail_line.set_data(xa[lo : frame + 1], ya[lo : frame + 1])
        agent_dot.set_data([xa[frame]], [ya[frame]])
        time_text.set_text(f"t = {frame * dt:.2f} s")
        return trail_line, agent_dot, time_text

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=n,
        interval=interval,
        blit=True,
        repeat=repeat,
    )
    return fig, anim


def plot_trajectory_3d(
    x: Sequence[float],
    y: Sequence[float],
    t: Sequence[float] | None = None,
    *,
    dt: float = 1.0,
    color: str = "#2980b9",
    linewidth: float = 1.5,
    title: str = "Trajectory (x, y, t)",
    figsize: tuple[float, float] = (10, 7),
    elevation: float = 25.0,
    azimuth: float = -60.0,
    colorby_speed: bool = False,
    cmap_name: str = "plasma",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot trajectory in 3-D space (x, y, time).

    Parameters
    ----------
    x, y : array-like
        Position data.
    t : array-like, optional
        Time values.  If None, constructed from *dt*.
    colorby_speed : bool
        Colour segments by speed.
    """
    xa = _ensure_array(x)
    ya = _ensure_array(y)
    ta = _ensure_array(t) if t is not None else np.arange(len(xa)) * dt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if colorby_speed and len(xa) > 1:
        vx_arr = _finite_diff(xa, dt)
        vy_arr = _finite_diff(ya, dt)
        speed = np.hypot(vx_arr, vy_arr)
        cmap = cm.get_cmap(cmap_name)
        norm = plt.Normalize(speed.min(), speed.max())
        for i in range(len(xa) - 1):
            ax.plot(
                xa[i : i + 2],
                ya[i : i + 2],
                ta[i : i + 2],
                color=cmap(norm(speed[i])),
                linewidth=linewidth,
            )
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Speed (m/s)")
    else:
        ax.plot(xa, ya, ta, color=color, linewidth=linewidth)

    ax.scatter([xa[0]], [ya[0]], [ta[0]], color="green", s=50, zorder=5, label="Start")
    ax.scatter([xa[-1]], [ya[-1]], [ta[-1]], color="red", s=50, zorder=5, marker="s", label="End")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("Time (s)")
    ax.set_title(title)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.legend(fontsize=8)
    return fig, ax


def plot_trajectory_uncertainty(
    x: Sequence[float],
    y: Sequence[float],
    sigma_x: Sequence[float],
    sigma_y: Sequence[float],
    *,
    ax: plt.Axes | None = None,
    n_ellipses: int = 20,
    confidence: float = 0.95,
    color: str = "#2980b9",
    ellipse_color: str = "#3498db",
    ellipse_alpha: float = 0.15,
    linewidth: float = 1.5,
    title: str = "Trajectory with Uncertainty",
    figsize: tuple[float, float] = (10, 8),
    correlation: Sequence[float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a trajectory with position uncertainty ellipses.

    Parameters
    ----------
    x, y : array-like
        Mean positions.
    sigma_x, sigma_y : array-like
        Standard deviations in x and y.
    n_ellipses : int
        Number of evenly-spaced ellipses to draw.
    confidence : float
        Confidence level (0-1) for ellipse scaling.  Uses chi-squared
        quantile with 2 DOF.
    correlation : array-like, optional
        Correlation coefficient between x and y at each step.  If None,
        ellipses are axis-aligned.
    """
    xa = _ensure_array(x)
    ya = _ensure_array(y)
    sx = _ensure_array(sigma_x)
    sy = _ensure_array(sigma_y)
    rho = _ensure_array(correlation) if correlation is not None else np.zeros(len(xa))

    # chi-squared quantile for 2 DOF
    chi2_val = -2.0 * math.log(1.0 - confidence)
    scale = math.sqrt(chi2_val)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(xa, ya, color=color, linewidth=linewidth, zorder=3)

    indices = np.linspace(0, len(xa) - 1, n_ellipses, dtype=int)
    for idx in indices:
        cov = np.array(
            [
                [sx[idx] ** 2, rho[idx] * sx[idx] * sy[idx]],
                [rho[idx] * sx[idx] * sy[idx], sy[idx] ** 2],
            ]
        )
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
        w = 2.0 * scale * math.sqrt(eigvals[0])
        h = 2.0 * scale * math.sqrt(eigvals[1])

        ell = Ellipse(
            xy=(xa[idx], ya[idx]),
            width=w,
            height=h,
            angle=angle,
            facecolor=ellipse_color,
            edgecolor=color,
            alpha=ellipse_alpha,
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(ell)

    ax.plot(xa[0], ya[0], "o", color="green", markersize=7, zorder=5)
    ax.plot(xa[-1], ya[-1], "s", color="red", markersize=7, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_social_distances_over_time(
    positions: Sequence[dict[int, tuple[float, float]]],
    *,
    dt: float = 1.0,
    ax: plt.Axes | None = None,
    title: str = "Pairwise Social Distances Over Time",
    figsize: tuple[float, float] = (12, 5),
    min_distance_line: float | None = 0.5,
    show_min_envelope: bool = True,
    cmap_name: str = "tab10",
    max_pairs: int = 15,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot pairwise inter-agent distances over time.

    Parameters
    ----------
    positions : list of dict
        Each element maps agent_id -> (x, y) at that timestep.
    dt : float
        Timestep between frames.
    min_distance_line : float or None
        Draw a horizontal line at this distance (e.g. collision threshold).
    show_min_envelope : bool
        If True, plot the minimum pairwise distance at each step as a
        thick line.
    max_pairs : int
        Maximum number of agent pairs to plot (to avoid visual clutter).
    """
    n_steps = len(positions)
    t = np.arange(n_steps) * dt

    # Collect all agent ids across all steps
    all_ids: set[int] = set()
    for pos_dict in positions:
        all_ids.update(pos_dict.keys())
    sorted_ids = sorted(all_ids)

    # Build pairwise distance time series
    pairs: list[tuple[int, int]] = []
    pair_distances: list[np.ndarray] = []
    for i_idx, id_a in enumerate(sorted_ids):
        for id_b in sorted_ids[i_idx + 1 :]:
            dists = np.full(n_steps, np.nan)
            for step_idx, pos_dict in enumerate(positions):
                if id_a in pos_dict and id_b in pos_dict:
                    xa, ya = pos_dict[id_a]
                    xb, yb = pos_dict[id_b]
                    dists[step_idx] = math.hypot(xa - xb, ya - yb)
            if not np.all(np.isnan(dists)):
                pairs.append((id_a, id_b))
                pair_distances.append(dists)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cmap = cm.get_cmap(cmap_name)
    n_plot = min(len(pairs), max_pairs)
    for pidx in range(n_plot):
        c = cmap(pidx % 10)
        id_a, id_b = pairs[pidx]
        ax.plot(t, pair_distances[pidx], color=c, linewidth=0.8, alpha=0.6, label=f"{id_a}-{id_b}")

    if show_min_envelope and pair_distances:
        all_dists = np.stack(pair_distances, axis=0)
        min_dist = np.nanmin(all_dists, axis=0)
        ax.plot(t, min_dist, color="black", linewidth=2.0, label="Min distance", zorder=5)

    if min_distance_line is not None:
        ax.axhline(
            min_distance_line,
            color="red",
            linewidth=1.2,
            linestyle="--",
            label=f"Threshold ({min_distance_line} m)",
        )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.legend(loc="best", fontsize=7, ncol=max(1, n_plot // 5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return fig, ax
