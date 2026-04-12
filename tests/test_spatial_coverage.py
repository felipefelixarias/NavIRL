"""Tests for navirl.utils.spatial — targets uncovered functions and branches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.utils.spatial import (
    KDTree2D,
    SpatialHashGrid,
    compute_voronoi_neighbors,
    find_k_nearest,
    find_neighbors_in_radius,
    minimum_distances,
    pairwise_distances,
)

# ────────────────────────────────────────────────────────────────────
# SpatialHashGrid
# ────────────────────────────────────────────────────────────────────


class TestSpatialHashGrid:
    def test_insert_and_query(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.insert(1, np.array([1.5, 1.5]))
        grid.insert(2, np.array([10.0, 10.0]))
        neighbors = grid.query(np.array([1.0, 1.0]), radius=2.0)
        assert sorted(neighbors) == [0, 1]

    def test_invalid_cell_size(self):
        with pytest.raises(ValueError):
            SpatialHashGrid(cell_size=0.0)
        with pytest.raises(ValueError):
            SpatialHashGrid(cell_size=-1.0)

    def test_remove(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.5, 0.5]))
        grid.insert(1, np.array([0.6, 0.6]))
        grid.remove(0)
        neighbors = grid.query(np.array([0.5, 0.5]), radius=1.0)
        assert 0 not in neighbors
        assert 1 in neighbors

    def test_remove_nonexistent(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.remove(999)  # should not raise

    def test_remove_cleans_empty_cell(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.5, 0.5]))
        grid.remove(0)
        assert grid.num_cells == 0

    def test_update(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.update(0, np.array([10.0, 10.0]))
        neighbors = grid.query(np.array([10.0, 10.0]), radius=1.0)
        assert 0 in neighbors
        neighbors_old = grid.query(np.array([0.0, 0.0]), radius=1.0)
        assert 0 not in neighbors_old

    def test_bulk_insert(self):
        grid = SpatialHashGrid(cell_size=1.0)
        ids = [0, 1, 2]
        positions = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        grid.bulk_insert(ids, positions)
        assert grid.num_entities == 3

    def test_rebuild(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.rebuild([1, 2], np.array([[5, 5], [6, 6]], dtype=float))
        assert grid.num_entities == 2
        assert 0 not in grid.query(np.array([0.0, 0.0]), radius=1.0)

    def test_query_with_exclude(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([0.1, 0.1]))
        result = grid.query(np.array([0.0, 0.0]), radius=1.0, exclude_id=0)
        assert 0 not in result
        assert 1 in result

    def test_query_with_distances(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([1.0, 0.0]))
        grid.insert(2, np.array([0.5, 0.0]))
        results = grid.query_with_distances(np.array([0.0, 0.0]), radius=2.0)
        # Should be sorted by distance
        assert results[0][0] == 0
        assert results[1][0] == 2
        assert results[2][0] == 1

    def test_query_with_distances_exclude(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([1.0, 0.0]))
        results = grid.query_with_distances(np.array([0.0, 0.0]), radius=2.0, exclude_id=0)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_query_k_nearest_with_max_radius(self):
        grid = SpatialHashGrid(cell_size=2.0)
        for i in range(5):
            grid.insert(i, np.array([float(i), 0.0]))
        results = grid.query_k_nearest(np.array([0.0, 0.0]), k=2, max_radius=3.0)
        assert len(results) == 2
        assert results[0][0] == 0  # closest

    def test_query_k_nearest_expanding(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([3.0, 0.0]))
        grid.insert(2, np.array([5.0, 0.0]))
        results = grid.query_k_nearest(np.array([0.0, 0.0]), k=2)
        assert len(results) == 2

    def test_count_in_region(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.insert(1, np.array([2.0, 2.0]))
        grid.insert(2, np.array([5.0, 5.0]))
        count = grid.count_in_region(np.array([0.0, 0.0]), np.array([3.0, 3.0]))
        assert count == 2

    def test_num_entities_and_cells(self):
        grid = SpatialHashGrid(cell_size=10.0)
        assert grid.num_entities == 0
        assert grid.num_cells == 0
        grid.insert(0, np.array([0.0, 0.0]))
        assert grid.num_entities == 1
        assert grid.num_cells >= 1

    def test_density_map(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.5, 0.5]))
        grid.insert(1, np.array([0.6, 0.6]))
        grid.insert(2, np.array([3.0, 3.0]))
        dmap = grid.density_map((0.0, 0.0, 4.0, 4.0), resolution=2.0)
        assert dmap.shape == (2, 2)
        assert dmap[0, 0] == 2.0  # two entities in first cell
        assert dmap[1, 1] == 1.0

    def test_density_map_default_resolution(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        dmap = grid.density_map((0.0, 0.0, 4.0, 4.0))
        assert dmap.shape == (2, 2)


# ────────────────────────────────────────────────────────────────────
# KDTree2D
# ────────────────────────────────────────────────────────────────────


class TestKDTree2D:
    @pytest.fixture()
    def tree_and_points(self):
        pts = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1], [5, 5]],
            dtype=float,
        )
        return KDTree2D(pts), pts

    def test_size(self, tree_and_points):
        tree, pts = tree_and_points
        assert tree.size == len(pts)

    def test_query_nearest(self, tree_and_points):
        tree, _ = tree_and_points
        eid, dist = tree.query_nearest(np.array([0.1, 0.1]))
        assert eid == 0
        assert dist == pytest.approx(math.hypot(0.1, 0.1))

    def test_query_nearest_with_exclude(self, tree_and_points):
        tree, _ = tree_and_points
        eid, dist = tree.query_nearest(np.array([0.0, 0.0]), exclude_id=0)
        assert eid != 0
        assert eid in (1, 2)

    def test_query_k_nearest(self, tree_and_points):
        tree, _ = tree_and_points
        results = tree.query_k_nearest(np.array([0.0, 0.0]), k=3)
        assert len(results) == 3
        # First should be self (entity 0)
        assert results[0][0] == 0

    def test_query_k_nearest_with_exclude(self, tree_and_points):
        tree, _ = tree_and_points
        results = tree.query_k_nearest(np.array([0.0, 0.0]), k=2, exclude_id=0)
        assert len(results) == 2
        assert all(eid != 0 for eid, _ in results)

    def test_query_radius(self, tree_and_points):
        tree, _ = tree_and_points
        results = tree.query_radius(np.array([0.0, 0.0]), radius=1.5)
        ids = [eid for eid, _ in results]
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids
        assert 4 not in ids  # (5,5) is far

    def test_query_radius_with_exclude(self, tree_and_points):
        tree, _ = tree_and_points
        results = tree.query_radius(np.array([0.0, 0.0]), radius=1.5, exclude_id=0)
        ids = [eid for eid, _ in results]
        assert 0 not in ids

    def test_query_rectangle(self, tree_and_points):
        tree, _ = tree_and_points
        results = tree.query_rectangle(np.array([0.0, 0.0]), np.array([1.5, 1.5]))
        assert set(results) == {0, 1, 2, 3}

    def test_query_rectangle_partial(self, tree_and_points):
        tree, _ = tree_and_points
        results = tree.query_rectangle(np.array([0.5, 0.5]), np.array([1.5, 1.5]))
        assert 3 in results  # (1,1)
        assert 0 not in results  # (0,0)

    def test_custom_entity_ids(self):
        pts = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        tree = KDTree2D(pts, entity_ids=[10, 20, 30])
        eid, _ = tree.query_nearest(np.array([0.0, 0.0]))
        assert eid == 10

    def test_empty_tree(self):
        pts = np.array([], dtype=float).reshape(0, 2)
        tree = KDTree2D(pts)
        assert tree.size == 0


# ────────────────────────────────────────────────────────────────────
# Convenience functions
# ────────────────────────────────────────────────────────────────────


class TestFindNeighborsInRadius:
    def test_basic(self):
        positions = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=float)
        neighbors = find_neighbors_in_radius(positions, 0, radius=1.5)
        assert set(neighbors) == {1, 2}

    def test_excludes_self(self):
        positions = np.array([[0, 0], [0, 0]], dtype=float)
        neighbors = find_neighbors_in_radius(positions, 0, radius=1.0)
        assert neighbors == [1]


class TestFindKNearest:
    def test_basic(self):
        positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        result = find_k_nearest(positions, 0, k=2)
        assert list(result) == [1, 2]


class TestPairwiseDistances:
    def test_basic(self):
        positions = np.array([[0, 0], [3, 4]], dtype=float)
        d = pairwise_distances(positions)
        assert d.shape == (2, 2)
        assert d[0, 0] == pytest.approx(0.0)
        assert d[0, 1] == pytest.approx(5.0)
        assert d[1, 0] == pytest.approx(5.0)


class TestMinimumDistances:
    def test_basic(self):
        a = np.array([[0, 0], [10, 10]], dtype=float)
        b = np.array([[1, 0], [2, 0]], dtype=float)
        result = minimum_distances(a, b)
        assert result[0] == pytest.approx(1.0)
        assert result.shape == (2,)


class TestComputeVoronoiNeighbors:
    def test_triangle(self):
        positions = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        neighbors = compute_voronoi_neighbors(positions)
        # Each should be neighbor of each other in a triangle
        for i in range(3):
            assert len(neighbors[i]) >= 1

    def test_single_point(self):
        positions = np.array([[0, 0]], dtype=float)
        neighbors = compute_voronoi_neighbors(positions)
        assert neighbors == {0: []}

    def test_two_points(self):
        positions = np.array([[0, 0], [1, 0]], dtype=float)
        neighbors = compute_voronoi_neighbors(positions)
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]

    def test_grid(self):
        positions = np.array(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            dtype=float,
        )
        neighbors = compute_voronoi_neighbors(positions)
        # Center point (1,0) = index 1 should have multiple neighbors
        assert len(neighbors[1]) >= 2
