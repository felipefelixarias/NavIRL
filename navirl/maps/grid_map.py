"""2-D occupancy grid map for NavIRL.

Provides :class:`GridMap` with ray casting, Bresenham line drawing,
flood fill, distance transform, C-space inflation, map merging,
submap extraction, and world-to-grid coordinate transforms.
"""

from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREE: int = 0
OCCUPIED: int = 1
UNKNOWN: int = -1


# ---------------------------------------------------------------------------
# GridMap
# ---------------------------------------------------------------------------

class GridMap:
    """2-D occupancy grid map.

    The grid uses integer cell values:
    * ``0`` – free
    * ``1`` – occupied
    * ``-1`` – unknown

    Parameters
    ----------
    width : int
        Number of columns.
    height : int
        Number of rows.
    resolution : float
        Metres per cell.
    origin : tuple of float
        World coordinates ``(x, y)`` of the grid origin (lower-left).
    default_value : int
        Initial cell value.
    """

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        resolution: float = 0.1,
        origin: tuple[float, float] = (0.0, 0.0),
        default_value: int = FREE,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin, dtype=np.float64)
        self.data: np.ndarray = np.full(
            (height, width), default_value, dtype=np.int8
        )

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coords to grid (row, col).

        Returns clipped indices that are always valid.
        """
        col = int(np.floor((x - self.origin[0]) / self.resolution))
        row = int(np.floor((y - self.origin[1]) / self.resolution))
        col = max(0, min(col, self.width - 1))
        row = max(0, min(row, self.height - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid (row, col) to world coords (centre of cell)."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (row + 0.5) * self.resolution
        return x, y

    def world_to_grid_float(self, x: float, y: float) -> tuple[float, float]:
        """Convert world coords to continuous grid coordinates."""
        col = (x - self.origin[0]) / self.resolution
        row = (y - self.origin[1]) / self.resolution
        return row, col

    def in_bounds(self, row: int, col: int) -> bool:
        """Check if (row, col) is within the grid."""
        return 0 <= row < self.height and 0 <= col < self.width

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def get(self, row: int, col: int) -> int:
        """Return cell value; UNKNOWN if out-of-bounds."""
        if not self.in_bounds(row, col):
            return UNKNOWN
        return int(self.data[row, col])

    def set(self, row: int, col: int, value: int) -> None:
        """Set a single cell value."""
        if self.in_bounds(row, col):
            self.data[row, col] = value

    def is_free(self, row: int, col: int) -> bool:
        return self.get(row, col) == FREE

    def is_occupied(self, row: int, col: int) -> bool:
        return self.get(row, col) == OCCUPIED

    def set_world(self, x: float, y: float, value: int) -> None:
        """Set cell at world coordinates."""
        r, c = self.world_to_grid(x, y)
        self.set(r, c, value)

    def get_world(self, x: float, y: float) -> int:
        """Get cell value at world coordinates."""
        r, c = self.world_to_grid(x, y)
        return self.get(r, c)

    # ------------------------------------------------------------------
    # Bresenham line
    # ------------------------------------------------------------------

    @staticmethod
    def bresenham(
        r0: int, c0: int, r1: int, c1: int
    ) -> list[tuple[int, int]]:
        """Bresenham's line algorithm between two grid cells.

        Returns list of ``(row, col)`` along the line.
        """
        cells: list[tuple[int, int]] = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr
        r, c = r0, c0
        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
        return cells

    def line_cells(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> list[tuple[int, int]]:
        """Return grid cells along a world-coordinate line segment."""
        r0, c0 = self.world_to_grid(x0, y0)
        r1, c1 = self.world_to_grid(x1, y1)
        return self.bresenham(r0, c0, r1, c1)

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def ray_cast(
        self,
        x: float,
        y: float,
        angle: float,
        max_range: float = 50.0,
    ) -> tuple[float, tuple[int, int]]:
        """Cast a ray from (x, y) at *angle* (radians).

        Returns ``(distance, (hit_row, hit_col))`` of the first occupied
        cell.  If no hit, returns ``(max_range, (-1, -1))``.
        """
        dx = np.cos(angle) * self.resolution
        dy = np.sin(angle) * self.resolution
        cx, cy = x, y
        steps = int(max_range / self.resolution)
        for _ in range(steps):
            cx += dx
            cy += dy
            r, c = self.world_to_grid(cx, cy)
            if not self.in_bounds(r, c):
                break
            if self.data[r, c] == OCCUPIED:
                dist = float(np.hypot(cx - x, cy - y))
                return dist, (r, c)
        return max_range, (-1, -1)

    def ray_cast_fan(
        self,
        x: float,
        y: float,
        start_angle: float,
        end_angle: float,
        n_rays: int = 36,
        max_range: float = 50.0,
    ) -> list[tuple[float, float, float]]:
        """Cast multiple rays in a fan.

        Returns list of ``(angle, distance, hit_x, hit_y)`` tuples.
        Actually returns ``(angle, distance, hit_x)`` simplified to
        ``(angle, distance, endpoint_x)``.  Full form below.
        """
        angles = np.linspace(start_angle, end_angle, n_rays)
        results: list[tuple[float, float, float]] = []
        for a in angles:
            dist, _ = self.ray_cast(x, y, a, max_range)
            hit_x = x + dist * np.cos(a)
            y + dist * np.sin(a)
            results.append((float(a), dist, float(hit_x)))
        return results

    # ------------------------------------------------------------------
    # Flood fill
    # ------------------------------------------------------------------

    def flood_fill(
        self, row: int, col: int, new_value: int
    ) -> int:
        """Flood-fill from (row, col) replacing connected cells of the
        same value with *new_value*.

        Returns the number of cells filled.
        """
        if not self.in_bounds(row, col):
            return 0
        old_value = int(self.data[row, col])
        if old_value == new_value:
            return 0
        queue: deque[tuple[int, int]] = deque()
        queue.append((row, col))
        filled = 0
        while queue:
            r, c = queue.popleft()
            if not self.in_bounds(r, c):
                continue
            if self.data[r, c] != old_value:
                continue
            self.data[r, c] = new_value
            filled += 1
            queue.append((r - 1, c))
            queue.append((r + 1, c))
            queue.append((r, c - 1))
            queue.append((r, c + 1))
        return filled

    def connected_component(
        self, row: int, col: int, value: int | None = None
    ) -> np.ndarray:
        """Return a boolean mask of the connected component at (row, col).

        Parameters
        ----------
        value : int or None
            Cell value to match.  Defaults to the cell's current value.
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        if not self.in_bounds(row, col):
            return mask
        if value is None:
            value = int(self.data[row, col])
        queue: deque[tuple[int, int]] = deque()
        queue.append((row, col))
        while queue:
            r, c = queue.popleft()
            if not self.in_bounds(r, c):
                continue
            if mask[r, c]:
                continue
            if self.data[r, c] != value:
                continue
            mask[r, c] = True
            queue.append((r - 1, c))
            queue.append((r + 1, c))
            queue.append((r, c - 1))
            queue.append((r, c + 1))
        return mask

    # ------------------------------------------------------------------
    # Distance transform
    # ------------------------------------------------------------------

    def distance_transform(self) -> np.ndarray:
        """Compute the distance transform (in cells) from occupied cells.

        Uses a BFS-based approach.  Returns a float array where each
        cell's value is the Chebyshev distance to the nearest occupied
        cell.
        """
        dist = np.full((self.height, self.width), np.inf, dtype=np.float64)
        queue: deque[tuple[int, int]] = deque()
        for r in range(self.height):
            for c in range(self.width):
                if self.data[r, c] == OCCUPIED:
                    dist[r, c] = 0.0
                    queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    nd = dist[r, c] + 1.0
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        queue.append((nr, nc))
        return dist

    def distance_transform_world(self) -> np.ndarray:
        """Distance transform in world-coordinate units (metres)."""
        return self.distance_transform() * self.resolution

    # ------------------------------------------------------------------
    # Inflation (C-space expansion)
    # ------------------------------------------------------------------

    def inflate(self, radius: float) -> GridMap:
        """Return a new :class:`GridMap` with occupied cells inflated.

        Parameters
        ----------
        radius : float
            Inflation radius in world units.
        """
        cells = max(1, int(np.ceil(radius / self.resolution)))
        new_map = GridMap(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin=tuple(self.origin),  # type: ignore[arg-type]
        )
        new_map.data = self.data.copy()
        occupied = np.argwhere(self.data == OCCUPIED)
        for r, c in occupied:
            r_min = max(0, r - cells)
            r_max = min(self.height, r + cells + 1)
            c_min = max(0, c - cells)
            c_max = min(self.width, c + cells + 1)
            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    if (rr - r) ** 2 + (cc - c) ** 2 <= cells ** 2:
                        new_map.data[rr, cc] = OCCUPIED
        return new_map

    def inflate_inplace(self, radius: float) -> None:
        """Inflate occupied cells in place."""
        inflated = self.inflate(radius)
        self.data = inflated.data

    # ------------------------------------------------------------------
    # Map merging
    # ------------------------------------------------------------------

    def merge(self, other: GridMap, mode: str = "overwrite") -> None:
        """Merge *other* into this map.

        Parameters
        ----------
        mode : str
            ``"overwrite"`` – other's non-unknown cells replace ours.
            ``"max"`` – take the max (most restrictive) of both.
            ``"min"`` – take the min (most permissive) of both.
        """
        for r in range(min(self.height, other.height)):
            for c in range(min(self.width, other.width)):
                o_val = other.data[r, c]
                if mode == "overwrite":
                    if o_val != UNKNOWN:
                        self.data[r, c] = o_val
                elif mode == "max":
                    self.data[r, c] = max(int(self.data[r, c]), int(o_val))
                elif mode == "min":
                    self.data[r, c] = min(int(self.data[r, c]), int(o_val))

    # ------------------------------------------------------------------
    # Submap extraction
    # ------------------------------------------------------------------

    def submap(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> GridMap:
        """Extract a rectangular submap.

        Returns a new :class:`GridMap` whose origin is adjusted.
        """
        r0 = max(0, row_start)
        c0 = max(0, col_start)
        r1 = min(self.height, row_end)
        c1 = min(self.width, col_end)
        sub_h = r1 - r0
        sub_w = c1 - c0
        if sub_h <= 0 or sub_w <= 0:
            return GridMap(0, 0, self.resolution, tuple(self.origin))  # type: ignore[arg-type]
        ox = self.origin[0] + c0 * self.resolution
        oy = self.origin[1] + r0 * self.resolution
        gm = GridMap(sub_w, sub_h, self.resolution, (ox, oy))
        gm.data = self.data[r0:r1, c0:c1].copy()
        return gm

    def submap_world(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> GridMap:
        """Extract submap by world-coordinate bounds."""
        r0, c0 = self.world_to_grid(x_min, y_min)
        r1, c1 = self.world_to_grid(x_max, y_max)
        return self.submap(r0, c0, r1 + 1, c1 + 1)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def draw_line(
        self,
        x0: float, y0: float, x1: float, y1: float,
        value: int = OCCUPIED,
    ) -> None:
        """Draw a line on the grid in world coordinates."""
        for r, c in self.line_cells(x0, y0, x1, y1):
            self.set(r, c, value)

    def draw_rect(
        self,
        x: float, y: float, w: float, h: float,
        value: int = OCCUPIED,
        filled: bool = True,
    ) -> None:
        """Draw a rectangle on the grid (world coords)."""
        r0, c0 = self.world_to_grid(x, y)
        r1, c1 = self.world_to_grid(x + w, y + h)
        r_lo, r_hi = min(r0, r1), max(r0, r1)
        c_lo, c_hi = min(c0, c1), max(c0, c1)
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                if filled or r in (r_lo, r_hi) or c in (c_lo, c_hi):
                    self.set(r, c, value)

    def draw_circle(
        self,
        cx: float, cy: float, radius: float,
        value: int = OCCUPIED,
        filled: bool = True,
    ) -> None:
        """Draw a circle on the grid (world coords)."""
        cells = int(np.ceil(radius / self.resolution))
        cr, cc = self.world_to_grid(cx, cy)
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                dist_sq = dr * dr + dc * dc
                if filled:
                    if dist_sq <= cells * cells:
                        self.set(cr + dr, cc + dc, value)
                else:
                    if abs(dist_sq - cells * cells) <= cells:
                        self.set(cr + dr, cc + dc, value)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def count_free(self) -> int:
        return int(np.sum(self.data == FREE))

    def count_occupied(self) -> int:
        return int(np.sum(self.data == OCCUPIED))

    def count_unknown(self) -> int:
        return int(np.sum(self.data == UNKNOWN))

    def occupancy_ratio(self) -> float:
        """Fraction of known cells that are occupied."""
        known = self.count_free() + self.count_occupied()
        if known == 0:
            return 0.0
        return self.count_occupied() / known

    @property
    def world_width(self) -> float:
        return self.width * self.resolution

    @property
    def world_height(self) -> float:
        return self.height * self.resolution

    # ------------------------------------------------------------------
    # Copy / clear
    # ------------------------------------------------------------------

    def copy(self) -> GridMap:
        """Return a deep copy."""
        gm = GridMap(
            self.width, self.height, self.resolution,
            tuple(self.origin),  # type: ignore[arg-type]
        )
        gm.data = self.data.copy()
        return gm

    def clear(self, value: int = FREE) -> None:
        """Set all cells to *value*."""
        self.data[:] = value

    # ------------------------------------------------------------------
    # NumPy interop
    # ------------------------------------------------------------------

    def as_binary(self) -> np.ndarray:
        """Return a boolean array (True = occupied)."""
        return self.data == OCCUPIED

    def as_float(self) -> np.ndarray:
        """Return float array: 1.0 occupied, 0.0 free, 0.5 unknown."""
        out = np.where(self.data == OCCUPIED, 1.0, 0.0)
        out = np.where(self.data == UNKNOWN, 0.5, out)
        return out

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        resolution: float = 0.1,
        origin: tuple[float, float] = (0.0, 0.0),
        threshold: float = 0.5,
    ) -> GridMap:
        """Create a GridMap from a 2-D numpy array.

        Values >= *threshold* are occupied, else free.
        """
        h, w = arr.shape[:2]
        gm = cls(w, h, resolution, origin)
        gm.data = np.where(arr >= threshold, OCCUPIED, FREE).astype(np.int8)
        return gm

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GridMap({self.width}x{self.height}, "
            f"res={self.resolution}, "
            f"occ={self.occupancy_ratio():.1%})"
        )
