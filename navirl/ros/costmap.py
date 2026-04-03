"""Costmap integration -- layered costmaps with social and predictive layers.

Provides NumPy-backed costmap operations that can be converted to / from
ROS2 ``nav_msgs/OccupancyGrid`` messages.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded ROS2 imports
# ---------------------------------------------------------------------------
try:
    from nav_msgs.msg import OccupancyGrid

    _ROS2_MSG_AVAILABLE = True
except ImportError:
    _ROS2_MSG_AVAILABLE = False

# Cost values following the ROS2 Nav2 convention
FREE_SPACE = 0
INSCRIBED_COST = 99
LETHAL_COST = 100
NO_INFORMATION = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _world_to_grid(
    wx: float,
    wy: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> tuple[int, int]:
    """Convert world coordinates to grid cell indices."""
    gx = int(round((wx - origin_x) / resolution))
    gy = int(round((wy - origin_y) / resolution))
    return gx, gy


def _grid_to_world(
    gx: int,
    gy: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> tuple[float, float]:
    """Convert grid cell indices to world coordinates (cell centre)."""
    wx = origin_x + gx * resolution
    wy = origin_y + gy * resolution
    return wx, wy


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a normalized 2-D Gaussian kernel."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


# ---------------------------------------------------------------------------
# CostmapManager
# ---------------------------------------------------------------------------


class CostmapManager:
    """Manages a stack of costmap layers and merges them into a single grid.

    Parameters
    ----------
    width : int
        Grid width in cells.
    height : int
        Grid height in cells.
    resolution : float
        Metres per cell.
    origin : tuple of float
        ``(x, y)`` world position of the grid origin (lower-left corner).
    """

    def __init__(
        self,
        width: int = 200,
        height: int = 200,
        resolution: float = 0.05,
        origin: tuple[float, float] = (-5.0, -5.0),
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x, self.origin_y = origin

        # Master costmap (int8 following OccupancyGrid convention)
        self._master: np.ndarray = np.zeros((height, width), dtype=np.int8)

        # Registered layers: name -> layer object
        self._layers: dict[str, _CostmapLayerBase] = {}

    # -- Layer management ---------------------------------------------------

    def add_layer(self, name: str, layer: _CostmapLayerBase) -> None:
        layer.resize(self.width, self.height, self.resolution, self.origin_x, self.origin_y)
        self._layers[name] = layer
        logger.info("CostmapManager: added layer '%s'.", name)

    def remove_layer(self, name: str) -> None:
        self._layers.pop(name, None)

    def get_layer(self, name: str) -> _CostmapLayerBase | None:
        return self._layers.get(name)

    # -- Update / merge -----------------------------------------------------

    def update(self, **kwargs: Any) -> np.ndarray:
        """Update all layers and merge into the master costmap.

        Extra *kwargs* are forwarded to each layer's ``update`` method
        (e.g. ``pedestrians=...`` for social layers).

        Returns the merged master costmap.
        """
        self._master[:] = FREE_SPACE

        for layer in self._layers.values():
            layer.update(**kwargs)
            grid = layer.grid
            # Take element-wise maximum (most conservative cost)
            self._master = np.maximum(self._master, grid).astype(np.int8)

        return self._master.copy()

    @property
    def master(self) -> np.ndarray:
        return self._master.copy()

    # -- ROS2 message conversion --------------------------------------------

    def to_occupancy_grid(self, frame_id: str = "map", stamp: Any = None) -> Any:
        """Convert the master costmap to a ``nav_msgs/OccupancyGrid``.

        Returns a dict when ROS2 messages are unavailable.
        """
        data = self._master.ravel().tolist()

        if _ROS2_MSG_AVAILABLE:
            msg = OccupancyGrid()
            msg.header.frame_id = frame_id
            if stamp is not None:
                msg.header.stamp = stamp
            msg.info.resolution = self.resolution
            msg.info.width = self.width
            msg.info.height = self.height
            msg.info.origin.position.x = self.origin_x
            msg.info.origin.position.y = self.origin_y
            msg.data = data
            return msg

        return {
            "frame_id": frame_id,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "origin": (self.origin_x, self.origin_y),
            "data": data,
        }

    @classmethod
    def from_occupancy_grid(cls, msg: Any) -> CostmapManager:
        """Construct a CostmapManager from a ``nav_msgs/OccupancyGrid``."""
        width = int(msg.info.width)
        height = int(msg.info.height)
        resolution = float(msg.info.resolution)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)

        mgr = cls(width=width, height=height, resolution=resolution, origin=(ox, oy))
        mgr._master = np.array(msg.data, dtype=np.int8).reshape((height, width))
        return mgr

    # -- Convenience --------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return _world_to_grid(x, y, self.origin_x, self.origin_y, self.resolution)

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return _grid_to_world(gx, gy, self.origin_x, self.origin_y, self.resolution)

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height


# ---------------------------------------------------------------------------
# Layer base class
# ---------------------------------------------------------------------------


class _CostmapLayerBase:
    """Abstract base for costmap layers."""

    def __init__(self) -> None:
        self._grid: np.ndarray = np.zeros((1, 1), dtype=np.int8)
        self._width = 1
        self._height = 1
        self._resolution = 0.05
        self._origin_x = 0.0
        self._origin_y = 0.0

    def resize(
        self,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        self._width = width
        self._height = height
        self._resolution = resolution
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._grid = np.zeros((height, width), dtype=np.int8)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def update(self, **kwargs: Any) -> None:
        """Override in subclasses to recompute ``self._grid``."""

    def clear(self) -> None:
        self._grid[:] = FREE_SPACE


# ---------------------------------------------------------------------------
# Static / inflation helpers
# ---------------------------------------------------------------------------


class StaticCostmapLayer(_CostmapLayerBase):
    """Layer loaded from a static map (e.g. SLAM output)."""

    def load_from_array(self, arr: np.ndarray) -> None:
        """Set the grid directly from a numpy array."""
        self._grid = np.clip(arr, -1, 100).astype(np.int8)

    def update(self, **kwargs: Any) -> None:
        pass  # static -- nothing to recompute


class InflationCostmapLayer(_CostmapLayerBase):
    """Inflates obstacles by a configurable radius."""

    def __init__(self, inflation_radius: float = 0.3) -> None:
        super().__init__()
        self.inflation_radius = inflation_radius
        self._source_grid: np.ndarray | None = None

    def set_source(self, grid: np.ndarray) -> None:
        self._source_grid = grid

    def update(self, **kwargs: Any) -> None:
        src = kwargs.get("obstacles", self._source_grid)
        if src is None:
            return
        radius_cells = max(int(round(self.inflation_radius / self._resolution)), 1)
        kernel_size = 2 * radius_cells + 1
        sigma = radius_cells / 2.0
        _gaussian_kernel(kernel_size, sigma)

        from scipy.ndimage import maximum_filter  # type: ignore[import-untyped]

        inflated = maximum_filter(src.astype(np.float64), size=kernel_size)
        # Scale to 0..INSCRIBED_COST for inflated cells, preserve lethal
        lethal_mask = src >= LETHAL_COST
        inflated = np.clip(inflated, 0, INSCRIBED_COST)
        inflated[lethal_mask] = LETHAL_COST
        self._grid = inflated.astype(np.int8)


# ---------------------------------------------------------------------------
# Social costmap layer
# ---------------------------------------------------------------------------


class SocialCostmapLayer(_CostmapLayerBase):
    """Encodes pedestrian proxemics zones into the costmap.

    Uses an asymmetric Gaussian model (larger in front of the person
    than behind) following the Social Force Model conventions.

    Parameters
    ----------
    personal_radius : float
        Radius of the high-cost personal space (metres).
    social_radius : float
        Radius of the softer social-awareness zone (metres).
    front_scale : float
        Multiplier for the *front* extent relative to the rear.
    """

    def __init__(
        self,
        personal_radius: float = 0.5,
        social_radius: float = 2.0,
        front_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.personal_radius = personal_radius
        self.social_radius = social_radius
        self.front_scale = front_scale

    def update(self, **kwargs: Any) -> None:
        """Recompute the layer from current pedestrian positions.

        Expects ``kwargs["pedestrians"]`` as an ``(N, 7)`` array
        with columns ``[x, y, vx, vy, theta, id, score]`` --
        the format produced by :func:`conversions.person_array_to_social_obs`.
        """
        self.clear()
        peds: np.ndarray | None = kwargs.get("pedestrians")
        if peds is None or peds.size == 0:
            return

        if peds.ndim == 1:
            peds = peds.reshape(1, -1)

        for row in peds:
            px, py = float(row[0]), float(row[1])
            theta = float(row[4]) if row.shape[0] > 4 else 0.0
            self._stamp_proxemics(px, py, theta)

    def _stamp_proxemics(self, wx: float, wy: float, theta: float) -> None:
        """Stamp an asymmetric Gaussian cost blob at world (wx, wy)."""
        cx, cy = _world_to_grid(wx, wy, self._origin_x, self._origin_y, self._resolution)
        radius_cells = int(round(self.social_radius / self._resolution))

        y_lo = max(cy - radius_cells, 0)
        y_hi = min(cy + radius_cells + 1, self._height)
        x_lo = max(cx - radius_cells, 0)
        x_hi = min(cx + radius_cells + 1, self._width)

        if y_lo >= y_hi or x_lo >= x_hi:
            return

        # Build local coordinate arrays
        ys = np.arange(y_lo, y_hi) - cy
        xs = np.arange(x_lo, x_hi) - cx
        xx, yy = np.meshgrid(xs, ys)

        # Rotate into pedestrian frame
        cos_t, sin_t = math.cos(-theta), math.sin(-theta)
        rx = xx * cos_t - yy * sin_t
        ry = xx * sin_t + yy * cos_t

        # Asymmetric sigma: larger in front
        sigma_front = self.social_radius / self._resolution
        sigma_rear = sigma_front / self.front_scale
        sigma_x = np.where(rx >= 0, sigma_front, sigma_rear)
        sigma_y = sigma_front * 0.8  # slightly narrower laterally

        dist2 = (rx / sigma_x) ** 2 + (ry / sigma_y) ** 2
        cost = np.exp(-0.5 * dist2)

        # Personal space -> high cost, social zone -> moderate
        personal_r2 = (self.personal_radius / self._resolution) ** 2
        normalized = dist2 * (sigma_front**2)  # back to cell-distance scale
        blob = np.where(
            normalized < personal_r2,
            INSCRIBED_COST,
            (cost * INSCRIBED_COST * 0.6).astype(np.int8),
        ).astype(np.int8)

        existing = self._grid[y_lo:y_hi, x_lo:x_hi]
        self._grid[y_lo:y_hi, x_lo:x_hi] = np.maximum(existing, blob)


# ---------------------------------------------------------------------------
# Predictive costmap layer
# ---------------------------------------------------------------------------


class PredictiveCostmapLayer(_CostmapLayerBase):
    """Encodes *predicted* pedestrian trajectories into the costmap.

    Given a prediction horizon (list of future positions per person),
    stamps decaying cost along each trajectory so the robot avoids
    regions where pedestrians are expected to be.

    Parameters
    ----------
    decay_factor : float
        Cost multiplier per prediction step (0-1; lower = faster decay).
    prediction_radius : float
        Radius (metres) of the cost blob at each predicted position.
    """

    def __init__(
        self,
        decay_factor: float = 0.85,
        prediction_radius: float = 0.4,
    ) -> None:
        super().__init__()
        self.decay_factor = decay_factor
        self.prediction_radius = prediction_radius

    def update(self, **kwargs: Any) -> None:
        """Recompute from predicted trajectories.

        Expects ``kwargs["predicted_trajectories"]`` as a list of
        ``(T, 2)`` arrays (one per pedestrian), where *T* is the
        prediction horizon length and columns are ``[x, y]``.
        """
        self.clear()
        trajectories: Sequence[np.ndarray] | None = kwargs.get("predicted_trajectories")
        if not trajectories:
            return

        radius_cells = max(int(round(self.prediction_radius / self._resolution)), 1)

        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if traj.ndim != 2 or traj.shape[1] < 2:
                continue
            cost_scale = 1.0
            for t in range(traj.shape[0]):
                wx, wy = float(traj[t, 0]), float(traj[t, 1])
                cx, cy = _world_to_grid(wx, wy, self._origin_x, self._origin_y, self._resolution)
                self._stamp_circle(cx, cy, radius_cells, cost_scale)
                cost_scale *= self.decay_factor

    def _stamp_circle(self, cx: int, cy: int, radius: int, scale: float) -> None:
        """Stamp a filled circle of cost onto the grid."""
        y_lo = max(cy - radius, 0)
        y_hi = min(cy + radius + 1, self._height)
        x_lo = max(cx - radius, 0)
        x_hi = min(cx + radius + 1, self._width)

        if y_lo >= y_hi or x_lo >= x_hi:
            return

        ys = np.arange(y_lo, y_hi) - cy
        xs = np.arange(x_lo, x_hi) - cx
        xx, yy = np.meshgrid(xs, ys)
        dist2 = xx**2 + yy**2
        mask = dist2 <= radius**2

        cost = np.exp(-dist2 / (2.0 * (radius * 0.5) ** 2)) * INSCRIBED_COST * scale
        cost_int = cost.astype(np.int8)

        region = self._grid[y_lo:y_hi, x_lo:x_hi]
        self._grid[y_lo:y_hi, x_lo:x_hi] = np.where(mask, np.maximum(region, cost_int), region)
