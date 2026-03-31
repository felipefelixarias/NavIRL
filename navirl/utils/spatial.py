"""Spatial data structures for efficient neighbor queries.

Provides spatial indexing structures optimized for 2-D pedestrian
simulation, including grid-based spatial hashing and a simple
k-d tree implementation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Spatial hash grid
# ---------------------------------------------------------------------------


class SpatialHashGrid:
    """Grid-based spatial hashing for fast neighbor queries in 2-D.

    Divides space into a uniform grid of cells.  Each entity is
    inserted into the cell that contains its position.  Neighbor
    queries check only cells within the query radius.

    This is well-suited for scenarios where entities have roughly
    uniform density and the query radius is known in advance.

    Parameters
    ----------
    cell_size : float
        Size of each grid cell.  Should be at least as large as the
        typical query radius for best performance.

    Examples
    --------
    >>> grid = SpatialHashGrid(cell_size=2.0)
    >>> grid.insert(0, np.array([1.0, 1.0]))
    >>> grid.insert(1, np.array([1.5, 1.5]))
    >>> grid.insert(2, np.array([10.0, 10.0]))
    >>> neighbors = grid.query(np.array([1.0, 1.0]), radius=2.0)
    >>> sorted(neighbors)
    [0, 1]
    """

    def __init__(self, cell_size: float = 1.0) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        self._cells: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
        self._positions: dict[int, np.ndarray] = {}

    def _cell_key(self, position: np.ndarray) -> tuple[int, int]:
        """Compute the grid cell key for a position."""
        return (
            int(math.floor(position[0] / self.cell_size)),
            int(math.floor(position[1] / self.cell_size)),
        )

    def clear(self) -> None:
        """Remove all entities."""
        self._cells.clear()
        self._positions.clear()

    def insert(self, entity_id: int, position: np.ndarray) -> None:
        """Insert an entity into the grid.

        Parameters
        ----------
        entity_id : int
            Unique identifier for the entity.
        position : np.ndarray
            2-D position, shape (2,).
        """
        position = np.asarray(position, dtype=np.float64)
        key = self._cell_key(position)
        self._cells[key].append((entity_id, position))
        self._positions[entity_id] = position

    def remove(self, entity_id: int) -> None:
        """Remove an entity from the grid.

        Parameters
        ----------
        entity_id : int
            Entity to remove.
        """
        if entity_id not in self._positions:
            return
        pos = self._positions.pop(entity_id)
        key = self._cell_key(pos)
        self._cells[key] = [(eid, p) for eid, p in self._cells[key] if eid != entity_id]
        if not self._cells[key]:
            del self._cells[key]

    def update(self, entity_id: int, new_position: np.ndarray) -> None:
        """Update an entity's position.

        Parameters
        ----------
        entity_id : int
            Entity to update.
        new_position : np.ndarray
            New 2-D position.
        """
        self.remove(entity_id)
        self.insert(entity_id, new_position)

    def bulk_insert(
        self,
        entity_ids: Sequence[int],
        positions: np.ndarray,
    ) -> None:
        """Insert multiple entities at once.

        Parameters
        ----------
        entity_ids : sequence of int
            Entity identifiers.
        positions : np.ndarray
            Positions, shape (N, 2).
        """
        positions = np.asarray(positions, dtype=np.float64)
        for eid, pos in zip(entity_ids, positions):
            self.insert(eid, pos)

    def rebuild(
        self,
        entity_ids: Sequence[int],
        positions: np.ndarray,
    ) -> None:
        """Clear and rebuild the grid from scratch.

        Parameters
        ----------
        entity_ids : sequence of int
            Entity identifiers.
        positions : np.ndarray
            Positions, shape (N, 2).
        """
        self.clear()
        self.bulk_insert(entity_ids, positions)

    def query(
        self,
        position: np.ndarray,
        radius: float,
        exclude_id: int | None = None,
    ) -> list[int]:
        """Find all entities within a radius of a query position.

        Parameters
        ----------
        position : np.ndarray
            Query position, shape (2,).
        radius : float
            Search radius.
        exclude_id : int, optional
            Entity ID to exclude from results.

        Returns
        -------
        list of int
            Entity IDs within the radius.
        """
        position = np.asarray(position, dtype=np.float64)
        radius_sq = radius * radius

        cx, cy = self._cell_key(position)
        cell_range = int(math.ceil(radius / self.cell_size))

        results = []
        for dx in range(-cell_range, cell_range + 1):
            for dy in range(-cell_range, cell_range + 1):
                key = (cx + dx, cy + dy)
                if key not in self._cells:
                    continue
                for entity_id, entity_pos in self._cells[key]:
                    if entity_id == exclude_id:
                        continue
                    diff = entity_pos - position
                    if diff[0] * diff[0] + diff[1] * diff[1] <= radius_sq:
                        results.append(entity_id)

        return results

    def query_with_distances(
        self,
        position: np.ndarray,
        radius: float,
        exclude_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Find entities within radius and return with distances.

        Parameters
        ----------
        position : np.ndarray
            Query position, shape (2,).
        radius : float
            Search radius.
        exclude_id : int, optional
            Entity to exclude.

        Returns
        -------
        list of (int, float)
            Pairs of (entity_id, distance), sorted by distance.
        """
        position = np.asarray(position, dtype=np.float64)
        radius_sq = radius * radius

        cx, cy = self._cell_key(position)
        cell_range = int(math.ceil(radius / self.cell_size))

        results = []
        for dx in range(-cell_range, cell_range + 1):
            for dy in range(-cell_range, cell_range + 1):
                key = (cx + dx, cy + dy)
                if key not in self._cells:
                    continue
                for entity_id, entity_pos in self._cells[key]:
                    if entity_id == exclude_id:
                        continue
                    diff = entity_pos - position
                    dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
                    if dist_sq <= radius_sq:
                        results.append((entity_id, math.sqrt(dist_sq)))

        results.sort(key=lambda x: x[1])
        return results

    def query_k_nearest(
        self,
        position: np.ndarray,
        k: int,
        max_radius: float = float("inf"),
        exclude_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Find k-nearest neighbors.

        Uses an expanding search if max_radius is not provided.

        Parameters
        ----------
        position : np.ndarray
            Query position, shape (2,).
        k : int
            Number of neighbors to find.
        max_radius : float
            Maximum search radius.
        exclude_id : int, optional
            Entity to exclude.

        Returns
        -------
        list of (int, float)
            Up to k nearest (entity_id, distance) pairs, sorted by distance.
        """
        if max_radius < float("inf"):
            candidates = self.query_with_distances(position, max_radius, exclude_id)
            return candidates[:k]

        # Expanding search
        radius = self.cell_size * 2
        while radius < 1e6:
            candidates = self.query_with_distances(position, radius, exclude_id)
            if len(candidates) >= k:
                return candidates[:k]
            radius *= 2.0

        return candidates[:k] if candidates else []

    def count_in_region(
        self,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
    ) -> int:
        """Count entities in an axis-aligned rectangular region.

        Parameters
        ----------
        min_corner : np.ndarray
            Lower-left corner, shape (2,).
        max_corner : np.ndarray
            Upper-right corner, shape (2,).

        Returns
        -------
        int
            Number of entities in the region.
        """
        min_corner = np.asarray(min_corner, dtype=np.float64)
        max_corner = np.asarray(max_corner, dtype=np.float64)

        min_cx = int(math.floor(min_corner[0] / self.cell_size))
        min_cy = int(math.floor(min_corner[1] / self.cell_size))
        max_cx = int(math.floor(max_corner[0] / self.cell_size))
        max_cy = int(math.floor(max_corner[1] / self.cell_size))

        count = 0
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                key = (cx, cy)
                if key not in self._cells:
                    continue
                for _, pos in self._cells[key]:
                    if (
                        min_corner[0] <= pos[0] <= max_corner[0]
                        and min_corner[1] <= pos[1] <= max_corner[1]
                    ):
                        count += 1

        return count

    @property
    def num_entities(self) -> int:
        """Number of entities in the grid."""
        return len(self._positions)

    @property
    def num_cells(self) -> int:
        """Number of non-empty cells."""
        return len(self._cells)

    def density_map(
        self,
        bounds: tuple[float, float, float, float],
        resolution: float | None = None,
    ) -> np.ndarray:
        """Compute a density map over a bounded region.

        Parameters
        ----------
        bounds : tuple
            (x_min, y_min, x_max, y_max).
        resolution : float, optional
            Grid resolution for the density map.  Defaults to cell_size.

        Returns
        -------
        np.ndarray
            2-D density map, shape (rows, cols).
        """
        x_min, y_min, x_max, y_max = bounds
        if resolution is None:
            resolution = self.cell_size

        cols = max(1, int(math.ceil((x_max - x_min) / resolution)))
        rows = max(1, int(math.ceil((y_max - y_min) / resolution)))
        density = np.zeros((rows, cols))

        for pos in self._positions.values():
            c = int((pos[0] - x_min) / resolution)
            r = int((pos[1] - y_min) / resolution)
            if 0 <= r < rows and 0 <= c < cols:
                density[r, c] += 1.0

        return density


# ---------------------------------------------------------------------------
# KD-Tree (simple 2-D implementation)
# ---------------------------------------------------------------------------


@dataclass
class _KDNode:
    """Internal node for KDTree2D."""

    point: np.ndarray
    entity_id: int
    left: _KDNode | None = None
    right: _KDNode | None = None
    axis: int = 0


class KDTree2D:
    """Simple 2-D k-d tree for nearest neighbor queries.

    This is a lightweight implementation suitable for moderate-sized
    point sets (up to ~100k points).  For larger sets, consider
    scipy.spatial.KDTree.

    Parameters
    ----------
    points : np.ndarray
        Points, shape (N, 2).
    entity_ids : sequence of int, optional
        Entity identifiers.  Defaults to range(N).

    Examples
    --------
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    >>> tree = KDTree2D(points)
    >>> tree.query_nearest(np.array([0.1, 0.1]))
    (0, 0.1414...)
    """

    def __init__(
        self,
        points: np.ndarray,
        entity_ids: Sequence[int] | None = None,
    ) -> None:
        points = np.asarray(points, dtype=np.float64)
        if entity_ids is None:
            entity_ids = list(range(len(points)))

        self._size = len(points)
        self._root = self._build(points, list(entity_ids), depth=0)

    def _build(
        self,
        points: np.ndarray,
        entity_ids: list[int],
        depth: int,
    ) -> _KDNode | None:
        """Recursively build the k-d tree."""
        if len(points) == 0:
            return None

        axis = depth % 2
        sorted_indices = np.argsort(points[:, axis])
        median = len(sorted_indices) // 2

        idx = sorted_indices[median]
        node = _KDNode(
            point=points[idx].copy(),
            entity_id=entity_ids[idx],
            axis=axis,
        )

        left_indices = sorted_indices[:median]
        right_indices = sorted_indices[median + 1 :]

        if len(left_indices) > 0:
            node.left = self._build(
                points[left_indices],
                [entity_ids[i] for i in left_indices],
                depth + 1,
            )
        if len(right_indices) > 0:
            node.right = self._build(
                points[right_indices],
                [entity_ids[i] for i in right_indices],
                depth + 1,
            )

        return node

    @property
    def size(self) -> int:
        """Number of points in the tree."""
        return self._size

    def query_nearest(
        self,
        point: np.ndarray,
        exclude_id: int | None = None,
    ) -> tuple[int, float]:
        """Find the nearest neighbor to a query point.

        Parameters
        ----------
        point : np.ndarray
            Query point, shape (2,).
        exclude_id : int, optional
            Entity ID to exclude.

        Returns
        -------
        tuple of (int, float)
            (entity_id, distance) of nearest neighbor.
        """
        point = np.asarray(point, dtype=np.float64)
        best: list[tuple[int, float]] = [(-1, float("inf"))]

        def _search(node: _KDNode | None) -> None:
            if node is None:
                return

            if node.entity_id != exclude_id:
                dist = float(np.linalg.norm(node.point - point))
                if dist < best[0][1]:
                    best[0] = (node.entity_id, dist)

            diff = point[node.axis] - node.point[node.axis]

            if diff <= 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            _search(first)

            if abs(diff) < best[0][1]:
                _search(second)

        _search(self._root)
        return best[0]

    def query_k_nearest(
        self,
        point: np.ndarray,
        k: int,
        exclude_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Find k-nearest neighbors.

        Parameters
        ----------
        point : np.ndarray
            Query point, shape (2,).
        k : int
            Number of neighbors.
        exclude_id : int, optional
            Entity ID to exclude.

        Returns
        -------
        list of (int, float)
            Up to k nearest (entity_id, distance) pairs, sorted by distance.
        """
        point = np.asarray(point, dtype=np.float64)
        # Use a simple list-based max-heap substitute
        heap: list[tuple[float, int]] = []  # (neg_dist, entity_id)

        def _search(node: _KDNode | None) -> None:
            if node is None:
                return

            if node.entity_id != exclude_id:
                dist = float(np.linalg.norm(node.point - point))
                if len(heap) < k:
                    heap.append((-dist, node.entity_id))
                    heap.sort(reverse=True)  # Keep sorted by neg_dist (desc)
                elif dist < -heap[0][0]:
                    heap[0] = (-dist, node.entity_id)
                    heap.sort(reverse=True)

            diff = point[node.axis] - node.point[node.axis]

            if diff <= 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            _search(first)

            max_dist = -heap[0][0] if len(heap) == k else float("inf")
            if abs(diff) < max_dist:
                _search(second)

        _search(self._root)

        results = [(eid, -neg_d) for neg_d, eid in sorted(heap, key=lambda x: -x[0])]
        return results

    def query_radius(
        self,
        point: np.ndarray,
        radius: float,
        exclude_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Find all points within a radius.

        Parameters
        ----------
        point : np.ndarray
            Query point, shape (2,).
        radius : float
            Search radius.
        exclude_id : int, optional
            Entity ID to exclude.

        Returns
        -------
        list of (int, float)
            (entity_id, distance) pairs within radius, sorted by distance.
        """
        point = np.asarray(point, dtype=np.float64)
        results: list[tuple[int, float]] = []

        def _search(node: _KDNode | None) -> None:
            if node is None:
                return

            if node.entity_id != exclude_id:
                dist = float(np.linalg.norm(node.point - point))
                if dist <= radius:
                    results.append((node.entity_id, dist))

            diff = point[node.axis] - node.point[node.axis]

            if diff <= 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            _search(first)

            if abs(diff) <= radius:
                _search(second)

        _search(self._root)
        results.sort(key=lambda x: x[1])
        return results

    def query_rectangle(
        self,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
    ) -> list[int]:
        """Find all points in an axis-aligned rectangle.

        Parameters
        ----------
        min_corner : np.ndarray
            Lower-left corner, shape (2,).
        max_corner : np.ndarray
            Upper-right corner, shape (2,).

        Returns
        -------
        list of int
            Entity IDs within the rectangle.
        """
        min_corner = np.asarray(min_corner, dtype=np.float64)
        max_corner = np.asarray(max_corner, dtype=np.float64)
        results: list[int] = []

        def _search(node: _KDNode | None) -> None:
            if node is None:
                return

            if (
                min_corner[0] <= node.point[0] <= max_corner[0]
                and min_corner[1] <= node.point[1] <= max_corner[1]
            ):
                results.append(node.entity_id)

            axis = node.axis
            if min_corner[axis] <= node.point[axis]:
                _search(node.left)
            if max_corner[axis] >= node.point[axis]:
                _search(node.right)

        _search(self._root)
        return results


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def find_neighbors_in_radius(
    positions: np.ndarray,
    query_idx: int,
    radius: float,
) -> list[int]:
    """Find indices of all points within radius of positions[query_idx].

    Simple brute-force implementation for small point sets.

    Parameters
    ----------
    positions : np.ndarray
        All positions, shape (N, 2).
    query_idx : int
        Index of the query point.
    radius : float
        Search radius.

    Returns
    -------
    list of int
        Indices of neighbors (excluding query_idx).
    """
    positions = np.asarray(positions, dtype=np.float64)
    query = positions[query_idx]
    diffs = positions - query
    dists = np.linalg.norm(diffs, axis=1)
    mask = (dists <= radius) & (np.arange(len(positions)) != query_idx)
    return list(np.where(mask)[0])


def find_k_nearest(
    positions: np.ndarray,
    query_idx: int,
    k: int,
) -> list[int]:
    """Find k-nearest neighbor indices (brute force).

    Parameters
    ----------
    positions : np.ndarray
        All positions, shape (N, 2).
    query_idx : int
        Index of the query point.
    k : int
        Number of neighbors.

    Returns
    -------
    list of int
        Indices of k nearest neighbors (excluding query_idx).
    """
    positions = np.asarray(positions, dtype=np.float64)
    query = positions[query_idx]
    dists = np.linalg.norm(positions - query, axis=1)
    dists[query_idx] = float("inf")
    indices = np.argsort(dists)[:k]
    return list(indices)


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances.

    Parameters
    ----------
    positions : np.ndarray
        Positions, shape (N, 2).

    Returns
    -------
    np.ndarray
        Distance matrix, shape (N, N).
    """
    positions = np.asarray(positions, dtype=np.float64)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def minimum_distances(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> np.ndarray:
    """Compute minimum distance from each point in A to any point in B.

    Parameters
    ----------
    group_a : np.ndarray
        First group of positions, shape (N, 2).
    group_b : np.ndarray
        Second group of positions, shape (M, 2).

    Returns
    -------
    np.ndarray
        Minimum distances for each point in A, shape (N,).
    """
    group_a = np.asarray(group_a, dtype=np.float64)
    group_b = np.asarray(group_b, dtype=np.float64)
    diff = group_a[:, np.newaxis, :] - group_b[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return np.min(dists, axis=1)


def compute_voronoi_neighbors(
    positions: np.ndarray,
) -> dict[int, list[int]]:
    """Compute approximate Voronoi neighbors using Delaunay-like connectivity.

    Uses a simple approach based on k-nearest neighbors and angular
    coverage to approximate Voronoi adjacency without requiring
    scipy.

    Parameters
    ----------
    positions : np.ndarray
        Positions, shape (N, 2).

    Returns
    -------
    dict
        Mapping from point index to list of neighbor indices.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}

    if n <= 1:
        return neighbors

    # For each point, find neighbors by angular sectors
    num_sectors = 6
    sector_angle = 2.0 * math.pi / num_sectors

    for i in range(n):
        # Find distances to all other points
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = float("inf")

        # For each angular sector, find the nearest point
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])

        for s in range(num_sectors):
            sector_start = -math.pi + s * sector_angle
            sector_end = sector_start + sector_angle

            # Find points in this sector
            in_sector = np.zeros(n, dtype=bool)
            for j in range(n):
                if j == i:
                    continue
                a = angles[j]
                # Normalize to sector range
                while a < sector_start:
                    a += 2.0 * math.pi
                while a > sector_start + 2.0 * math.pi:
                    a -= 2.0 * math.pi
                if sector_start <= a < sector_end:
                    in_sector[j] = True

            sector_indices = np.where(in_sector)[0]
            if len(sector_indices) > 0:
                nearest = sector_indices[np.argmin(dists[sector_indices])]
                if nearest not in neighbors[i]:
                    neighbors[i].append(nearest)
                if i not in neighbors[nearest]:
                    neighbors[nearest].append(i)

    return neighbors
