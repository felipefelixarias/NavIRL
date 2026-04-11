"""Extended tests for navirl.utils.spatial — functions not covered by test_utils.py.

Covers: SpatialHashGrid.density_map, SpatialHashGrid edge cases,
KDTree2D edge cases, and compute_voronoi_neighbors.
"""

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

# ===================================================================
# SpatialHashGrid — density_map
# ===================================================================


class TestSpatialHashGridDensityMap:
    def test_empty_grid(self):
        grid = SpatialHashGrid(cell_size=1.0)
        density = grid.density_map((0, 0, 10, 10), resolution=2.0)
        assert density.shape == (5, 5)
        assert density.sum() == 0.0

    def test_single_entity(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([1.5, 1.5]))
        density = grid.density_map((0, 0, 4, 4), resolution=2.0)
        assert density.sum() == 1.0

    def test_multiple_entities_same_cell(self):
        grid = SpatialHashGrid(cell_size=5.0)
        grid.insert(0, np.array([0.5, 0.5]))
        grid.insert(1, np.array([0.6, 0.6]))
        density = grid.density_map((0, 0, 2, 2), resolution=1.0)
        assert density.sum() == 2.0

    def test_default_resolution(self):
        grid = SpatialHashGrid(cell_size=3.0)
        grid.insert(0, np.array([1.0, 1.0]))
        density = grid.density_map((0, 0, 9, 9))
        # Default resolution = cell_size = 3.0, so 3x3 grid
        assert density.shape == (3, 3)

    def test_entity_outside_bounds(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([100.0, 100.0]))
        density = grid.density_map((0, 0, 10, 10), resolution=2.0)
        assert density.sum() == 0.0


# ===================================================================
# SpatialHashGrid — additional edge cases
# ===================================================================


class TestSpatialHashGridEdgeCases:
    def test_remove_nonexistent(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.remove(999)  # Should not raise

    def test_remove_cleans_empty_cell(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([1.0, 1.0]))
        assert grid.num_cells == 1
        grid.remove(0)
        assert grid.num_cells == 0

    def test_query_k_nearest_with_max_radius(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([1.0, 0.0]))
        grid.insert(2, np.array([100.0, 0.0]))
        results = grid.query_k_nearest(
            np.array([0.0, 0.0]),
            k=2,
            max_radius=5.0,
        )
        ids = [r[0] for r in results]
        assert 0 in ids
        assert 1 in ids
        assert 2 not in ids

    def test_query_k_nearest_expanding_search(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([5.0, 0.0]))
        # No max_radius -> expanding search
        results = grid.query_k_nearest(np.array([0.0, 0.0]), k=2)
        assert len(results) == 2

    def test_query_k_nearest_fewer_than_k(self):
        grid = SpatialHashGrid(cell_size=1.0)
        grid.insert(0, np.array([0.0, 0.0]))
        results = grid.query_k_nearest(
            np.array([0.0, 0.0]),
            k=5,
            max_radius=100.0,
        )
        assert len(results) == 1

    def test_negative_coordinates(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([-3.0, -4.0]))
        grid.insert(1, np.array([-3.5, -4.5]))
        neighbors = grid.query(np.array([-3.0, -4.0]), radius=2.0)
        assert 0 in neighbors
        assert 1 in neighbors

    def test_count_in_region_empty(self):
        grid = SpatialHashGrid(cell_size=1.0)
        count = grid.count_in_region(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert count == 0


# ===================================================================
# KDTree2D — additional edge cases
# ===================================================================


class TestKDTree2DEdgeCases:
    def test_query_nearest_single_point(self):
        tree = KDTree2D(np.array([[5.0, 5.0]]))
        eid, dist = tree.query_nearest(np.array([0.0, 0.0]))
        assert eid == 0
        assert dist == pytest.approx(math.sqrt(50))

    def test_query_k_nearest_with_exclude(self):
        points = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        tree = KDTree2D(points, entity_ids=[10, 20, 30])
        results = tree.query_k_nearest(np.array([0.0, 0.0]), k=2, exclude_id=10)
        ids = [r[0] for r in results]
        assert 10 not in ids
        assert 20 in ids
        assert 30 in ids

    def test_query_radius_with_exclude(self):
        points = np.array([[0, 0], [1, 0], [5, 0]], dtype=float)
        tree = KDTree2D(points, entity_ids=[10, 20, 30])
        results = tree.query_radius(np.array([0.0, 0.0]), radius=2.0, exclude_id=10)
        ids = [r[0] for r in results]
        assert 10 not in ids
        assert 20 in ids

    def test_query_radius_empty_result(self):
        points = np.array([[10, 10], [20, 20]], dtype=float)
        tree = KDTree2D(points)
        results = tree.query_radius(np.array([0.0, 0.0]), radius=1.0)
        assert results == []

    def test_query_rectangle_empty(self):
        points = np.array([[10, 10], [20, 20]], dtype=float)
        tree = KDTree2D(points)
        ids = tree.query_rectangle(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert ids == []

    def test_default_entity_ids(self):
        points = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        tree = KDTree2D(points)  # No entity_ids provided
        eid, dist = tree.query_nearest(np.array([0.0, 0.0]))
        assert eid == 0
        assert dist == pytest.approx(0.0)

    def test_large_point_set(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(-100, 100, size=(500, 2))
        tree = KDTree2D(points)
        assert tree.size == 500
        # Nearest to origin should be found
        eid, dist = tree.query_nearest(np.array([0.0, 0.0]))
        assert eid >= 0
        assert dist >= 0

    def test_query_k_nearest_k_larger_than_size(self):
        points = np.array([[0, 0], [1, 0]], dtype=float)
        tree = KDTree2D(points)
        results = tree.query_k_nearest(np.array([0.0, 0.0]), k=10)
        assert len(results) == 2


# ===================================================================
# compute_voronoi_neighbors
# ===================================================================


class TestComputeVoronoiNeighbors:
    def test_single_point(self):
        positions = np.array([[1.0, 1.0]])
        neighbors = compute_voronoi_neighbors(positions)
        assert neighbors == {0: []}

    def test_two_points(self):
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
        neighbors = compute_voronoi_neighbors(positions)
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]

    def test_triangle(self):
        positions = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [1.0, 2.0],
            ]
        )
        neighbors = compute_voronoi_neighbors(positions)
        # Each point should neighbor the other two
        for i in range(3):
            assert len(neighbors[i]) == 2

    def test_grid_pattern(self):
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ],
            dtype=float,
        )
        neighbors = compute_voronoi_neighbors(positions)
        # Center point (index 4) should have many neighbors
        assert len(neighbors[4]) >= 4

    def test_empty(self):
        positions = np.empty((0, 2))
        neighbors = compute_voronoi_neighbors(positions)
        assert neighbors == {}

    def test_symmetric(self):
        rng = np.random.default_rng(123)
        positions = rng.uniform(-10, 10, size=(20, 2))
        neighbors = compute_voronoi_neighbors(positions)
        # Neighborhood should be symmetric
        for i, nbrs in neighbors.items():
            for j in nbrs:
                assert i in neighbors[j], f"Asymmetric: {i} -> {j} but not {j} -> {i}"


# ===================================================================
# Convenience functions — additional edge cases
# ===================================================================


class TestConvenienceEdgeCases:
    def test_find_neighbors_all_within_radius(self):
        positions = np.array([[0, 0], [0.1, 0], [0.2, 0]], dtype=float)
        nbrs = find_neighbors_in_radius(positions, query_idx=0, radius=1.0)
        assert 1 in nbrs
        assert 2 in nbrs

    def test_find_k_nearest_k_equals_n(self):
        positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        nbrs = find_k_nearest(positions, query_idx=0, k=2)
        assert len(nbrs) == 2

    def test_pairwise_distances_single(self):
        positions = np.array([[3.0, 4.0]])
        D = pairwise_distances(positions)
        assert D.shape == (1, 1)
        assert D[0, 0] == pytest.approx(0.0)

    def test_minimum_distances_single_point_each(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result = minimum_distances(a, b)
        assert result[0] == pytest.approx(5.0)
