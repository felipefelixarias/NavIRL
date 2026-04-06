"""Tests for navirl.utils modules (geometry, math_utils, spatial).

These modules previously had 12-16% test coverage. This file adds
tests for the uncovered utility functions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.utils.geometry import (
    angle_between,
    angle_diff,
    angular_velocity,
    closest_point_on_line,
    cross2d,
    distance,
    dot2d,
    heading_from_velocity,
    line_segment_intersection,
    normalize_angle,
    point_in_polygon,
    point_to_line_distance,
    polygon_area,
    polygon_centroid,
    quat_to_yaw,
    ray_segment_intersection,
    rotate_point,
    rotate_points,
    wrap_angle,
)
from navirl.utils.math_utils import (
    cosine_similarity,
    entropy,
    exponential_moving_average,
    finite_difference,
    gaussian_kernel,
    inverse_lerp,
    kl_divergence,
    lerp,
    remap,
    running_mean,
    running_std,
    sigmoid,
    smooth_step,
    smoother_step,
    softmax,
)
from navirl.utils.spatial import (
    KDTree2D,
    SpatialHashGrid,
    find_k_nearest,
    find_neighbors_in_radius,
    minimum_distances,
    pairwise_distances,
)


# ===================================================================
# Geometry — Angle utilities
# ===================================================================


class TestAngleUtilities:
    def test_normalize_angle_zero(self):
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_normalize_angle_pi(self):
        result = normalize_angle(math.pi)
        assert abs(result) == pytest.approx(math.pi)

    def test_normalize_angle_large(self):
        result = normalize_angle(3 * math.pi)
        assert -math.pi <= result <= math.pi
        assert abs(result) == pytest.approx(math.pi, abs=1e-10)

    def test_normalize_angle_negative(self):
        result = normalize_angle(-3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_wrap_angle_positive(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)
        result = wrap_angle(3 * math.pi)
        assert 0 <= result < 2 * math.pi
        assert result == pytest.approx(math.pi)

    def test_wrap_angle_negative(self):
        result = wrap_angle(-math.pi / 2)
        assert 0 <= result < 2 * math.pi
        assert result == pytest.approx(3 * math.pi / 2)

    def test_angle_diff_small(self):
        assert angle_diff(0.1, 0.0) == pytest.approx(0.1)
        assert angle_diff(0.0, 0.1) == pytest.approx(-0.1)

    def test_angle_diff_wrapping(self):
        # From -170° to 170° should be -20° not 340°
        a = math.radians(170)
        b = math.radians(-170)
        result = angle_diff(a, b)
        assert result == pytest.approx(math.radians(-20), abs=1e-10)

    def test_angle_between_perpendicular(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert angle_between(v1, v2) == pytest.approx(math.pi / 2)

    def test_angle_between_parallel(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([2.0, 0.0])
        assert angle_between(v1, v2) == pytest.approx(0.0)

    def test_angle_between_antiparallel(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert angle_between(v1, v2) == pytest.approx(math.pi)

    def test_angle_between_zero_vector(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        assert angle_between(v1, v2) == pytest.approx(0.0)

    def test_heading_from_velocity(self):
        assert heading_from_velocity(1.0, 0.0) == pytest.approx(0.0)
        assert heading_from_velocity(0.0, 1.0) == pytest.approx(math.pi / 2)
        assert heading_from_velocity(-1.0, 0.0) == pytest.approx(math.pi)
        assert heading_from_velocity(0.0, 0.0) == pytest.approx(0.0)

    def test_angular_velocity(self):
        result = angular_velocity(0.0, math.pi / 2, 1.0)
        assert result == pytest.approx(math.pi / 2)

    def test_angular_velocity_zero_dt(self):
        assert angular_velocity(0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_quat_to_yaw_identity(self):
        # Identity quaternion (0, 0, 0, 1) -> 0 yaw
        assert quat_to_yaw(0, 0, 0, 1) == pytest.approx(0.0)

    def test_quat_to_yaw_90deg(self):
        # 90° rotation about z: qw=cos(45°), qz=sin(45°)
        qw = math.cos(math.pi / 4)
        qz = math.sin(math.pi / 4)
        assert quat_to_yaw(0, 0, qz, qw) == pytest.approx(math.pi / 2)


# ===================================================================
# Geometry — 2D Operations
# ===================================================================


class TestBasic2D:
    def test_cross2d(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert cross2d(u, v) == pytest.approx(1.0)
        assert cross2d(v, u) == pytest.approx(-1.0)

    def test_dot2d(self):
        u = np.array([3.0, 4.0])
        v = np.array([1.0, 2.0])
        assert dot2d(u, v) == pytest.approx(11.0)

    def test_distance_single(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        assert distance(p1, p2) == pytest.approx(5.0)

    def test_distance_batch(self):
        p1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        p2 = np.array([[3.0, 4.0], [1.0, 1.0]])
        result = distance(p1, p2)
        np.testing.assert_allclose(result, [5.0, 0.0])

    def test_rotate_point_90deg(self):
        point = np.array([1.0, 0.0])
        rotated = rotate_point(point, math.pi / 2)
        np.testing.assert_allclose(rotated, [0.0, 1.0], atol=1e-10)

    def test_rotate_point_with_center(self):
        point = np.array([2.0, 0.0])
        center = np.array([1.0, 0.0])
        rotated = rotate_point(point, math.pi / 2, center)
        np.testing.assert_allclose(rotated, [1.0, 1.0], atol=1e-10)

    def test_rotate_points(self):
        points = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = rotate_points(points, math.pi / 2)
        np.testing.assert_allclose(rotated[0], [0.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(rotated[1], [-1.0, 0.0], atol=1e-10)

    def test_rotate_points_with_center(self):
        points = np.array([[2.0, 0.0], [0.0, 0.0]])
        center = np.array([1.0, 0.0])
        rotated = rotate_points(points, math.pi, center)
        np.testing.assert_allclose(rotated[0], [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(rotated[1], [2.0, 0.0], atol=1e-10)


# ===================================================================
# Geometry — Line / Segment Operations
# ===================================================================


class TestLineOperations:
    def test_closest_point_on_segment(self):
        cp = closest_point_on_line(
            np.array([5.0, 3.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
        )
        np.testing.assert_allclose(cp, [5.0, 0.0])

    def test_closest_point_clamped_before_start(self):
        cp = closest_point_on_line(
            np.array([-5.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
        )
        np.testing.assert_allclose(cp, [0.0, 0.0])

    def test_closest_point_unclamped(self):
        cp = closest_point_on_line(
            np.array([-5.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
            clamp_to_segment=False,
        )
        np.testing.assert_allclose(cp, [-5.0, 0.0])

    def test_point_to_line_distance(self):
        d = point_to_line_distance(
            np.array([5.0, 3.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
        )
        assert d == pytest.approx(3.0)

    def test_line_segment_intersection(self):
        result = line_segment_intersection(
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            np.array([10.0, 0.0]),
            np.array([0.0, 10.0]),
        )
        assert result is not None
        np.testing.assert_allclose(result, [5.0, 5.0])

    def test_line_segment_no_intersection(self):
        result = line_segment_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        )
        assert result is None

    def test_line_segment_parallel(self):
        result = line_segment_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        )
        assert result is None

    def test_ray_segment_intersection_hit(self):
        t = ray_segment_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([5.0, -1.0]),
            np.array([5.0, 1.0]),
        )
        assert t is not None
        assert t == pytest.approx(5.0)

    def test_ray_segment_intersection_miss(self):
        t = ray_segment_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([5.0, 2.0]),
            np.array([5.0, 3.0]),
        )
        assert t is None


# ===================================================================
# Geometry — Polygon Operations
# ===================================================================


class TestPolygonOperations:
    def _square(self):
        """CCW unit square centered at origin."""
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)

    def test_point_in_polygon_inside(self):
        assert point_in_polygon(np.array([0.0, 0.0]), self._square())

    def test_point_in_polygon_outside(self):
        assert not point_in_polygon(np.array([5.0, 5.0]), self._square())

    def test_polygon_area_square(self):
        area = polygon_area(self._square())
        assert abs(area) == pytest.approx(4.0)

    def test_polygon_area_triangle(self):
        tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
        area = polygon_area(tri)
        assert abs(area) == pytest.approx(6.0)

    def test_polygon_centroid_square(self):
        centroid = polygon_centroid(self._square())
        np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1e-10)


# ===================================================================
# Math Utils — Basic scalar
# ===================================================================


class TestMathScalar:
    def test_lerp(self):
        assert lerp(0, 10, 0.0) == pytest.approx(0.0)
        assert lerp(0, 10, 0.5) == pytest.approx(5.0)
        assert lerp(0, 10, 1.0) == pytest.approx(10.0)

    def test_inverse_lerp(self):
        assert inverse_lerp(0, 10, 5) == pytest.approx(0.5)
        assert inverse_lerp(0, 10, 0) == pytest.approx(0.0)
        assert inverse_lerp(5, 5, 5) == pytest.approx(0.0)  # equal endpoints

    def test_remap(self):
        assert remap(5, 0, 10, 0, 100) == pytest.approx(50.0)
        assert remap(0, 0, 10, 100, 200) == pytest.approx(100.0)

    def test_smooth_step(self):
        assert smooth_step(0, 1, -1) == pytest.approx(0.0)
        assert smooth_step(0, 1, 0.5) == pytest.approx(0.5)
        assert smooth_step(0, 1, 2) == pytest.approx(1.0)

    def test_smoother_step(self):
        assert smoother_step(0, 1, -1) == pytest.approx(0.0)
        assert smoother_step(0, 1, 0.5) == pytest.approx(0.5)
        assert smoother_step(0, 1, 2) == pytest.approx(1.0)


# ===================================================================
# Math Utils — Activation / probability
# ===================================================================


class TestActivationFunctions:
    def test_sigmoid_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self):
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_array(self):
        x = np.array([-10.0, 0.0, 10.0])
        result = sigmoid(x)
        assert result[1] == pytest.approx(0.5)
        assert result[0] < 0.01
        assert result[2] > 0.99

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result.sum() == pytest.approx(1.0)
        assert result[2] > result[1] > result[0]

    def test_softmax_temperature(self):
        x = np.array([1.0, 2.0, 3.0])
        # High temperature -> more uniform
        hot = softmax(x, temperature=10.0)
        cold = softmax(x, temperature=0.1)
        assert hot[2] - hot[0] < cold[2] - cold[0]


# ===================================================================
# Math Utils — Statistical
# ===================================================================


class TestStatistical:
    def test_running_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = running_mean(values, window=3)
        assert len(result) == 5
        # Middle element should be mean of surrounding window
        assert result[2] == pytest.approx(3.0)  # mean of [2,3,4] or similar

    def test_running_std(self):
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = running_std(values, window=3)
        assert result[2] == pytest.approx(0.0)

    def test_exponential_moving_average(self):
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = exponential_moving_average(values, alpha=0.5)
        # Constant input -> constant output
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 1.0])


# ===================================================================
# Math Utils — Kernels
# ===================================================================


class TestKernels:
    def test_gaussian_kernel_sums_to_one(self):
        k = gaussian_kernel(size=5, sigma=1.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-6)

    def test_gaussian_kernel_symmetric(self):
        k = gaussian_kernel(size=7, sigma=1.0)
        np.testing.assert_allclose(k, k[::-1])


# ===================================================================
# Math Utils — Finite differences
# ===================================================================


class TestFiniteDifference:
    def test_first_order(self):
        # Linear function: derivative should be constant
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = finite_difference(values, dt=1.0, order=1)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-10)


# ===================================================================
# Math Utils — Distance metrics
# ===================================================================


class TestDistanceMetrics:
    def test_cosine_similarity_parallel(self):
        a = np.array([1.0, 0.0])
        b = np.array([2.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_antiparallel(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)


# ===================================================================
# Math Utils — Entropy
# ===================================================================


class TestEntropy:
    def test_entropy_uniform(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        result = entropy(probs)
        assert result == pytest.approx(math.log(4), abs=1e-10)

    def test_entropy_certain(self):
        probs = np.array([1.0, 0.0, 0.0])
        result = entropy(probs)
        assert result == pytest.approx(0.0)

    def test_kl_divergence_same(self):
        p = np.array([0.5, 0.5])
        result = kl_divergence(p, p)
        assert result == pytest.approx(0.0, abs=1e-10)


# ===================================================================
# Spatial — SpatialHashGrid
# ===================================================================


class TestSpatialHashGrid:
    def test_insert_and_query(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.insert(1, np.array([1.5, 1.5]))
        grid.insert(2, np.array([10.0, 10.0]))
        neighbors = grid.query(np.array([1.0, 1.0]), radius=2.0)
        assert 0 in neighbors
        assert 1 in neighbors
        assert 2 not in neighbors

    def test_query_exclude(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.insert(1, np.array([1.5, 1.5]))
        neighbors = grid.query(np.array([1.0, 1.0]), radius=2.0, exclude_id=0)
        assert 0 not in neighbors
        assert 1 in neighbors

    def test_remove(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.remove(0)
        neighbors = grid.query(np.array([1.0, 1.0]), radius=2.0)
        assert 0 not in neighbors

    def test_update(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.update(0, np.array([100.0, 100.0]))
        nearby_old = grid.query(np.array([1.0, 1.0]), radius=2.0)
        assert 0 not in nearby_old
        nearby_new = grid.query(np.array([100.0, 100.0]), radius=2.0)
        assert 0 in nearby_new

    def test_bulk_insert(self):
        grid = SpatialHashGrid(cell_size=2.0)
        ids = [0, 1, 2]
        positions = np.array([[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]])
        grid.bulk_insert(ids, positions)
        assert grid.num_entities == 3

    def test_rebuild(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(99, np.array([50.0, 50.0]))
        ids = [0, 1]
        positions = np.array([[1.0, 1.0], [2.0, 2.0]])
        grid.rebuild(ids, positions)
        assert grid.num_entities == 2

    def test_clear(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.clear()
        assert grid.num_entities == 0

    def test_num_cells(self):
        grid = SpatialHashGrid(cell_size=5.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.insert(1, np.array([100.0, 100.0]))
        assert grid.num_cells == 2

    def test_invalid_cell_size(self):
        with pytest.raises(ValueError):
            SpatialHashGrid(cell_size=0)

    def test_query_with_distances(self):
        grid = SpatialHashGrid(cell_size=5.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([3.0, 0.0]))
        grid.insert(2, np.array([1.0, 0.0]))
        results = grid.query_with_distances(np.array([0.0, 0.0]), radius=5.0)
        # Should be sorted by distance
        ids = [r[0] for r in results]
        assert ids[0] == 0
        assert ids[1] == 2
        assert ids[2] == 1

    def test_query_k_nearest(self):
        grid = SpatialHashGrid(cell_size=5.0)
        grid.insert(0, np.array([0.0, 0.0]))
        grid.insert(1, np.array([1.0, 0.0]))
        grid.insert(2, np.array([10.0, 0.0]))
        results = grid.query_k_nearest(np.array([0.0, 0.0]), k=2)
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert 0 in ids
        assert 1 in ids

    def test_count_in_region(self):
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(0, np.array([1.0, 1.0]))
        grid.insert(1, np.array([5.0, 5.0]))
        grid.insert(2, np.array([100.0, 100.0]))
        count = grid.count_in_region(
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
        )
        assert count == 2


# ===================================================================
# Spatial — KDTree2D
# ===================================================================


class TestKDTree2D:
    def _make_tree(self):
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
        ])
        return KDTree2D(points, entity_ids=[10, 20, 30, 40])

    def test_size(self):
        tree = self._make_tree()
        assert tree.size == 4

    def test_query_nearest(self):
        tree = self._make_tree()
        eid, dist = tree.query_nearest(np.array([0.5, 0.0]))
        assert eid in (10, 20)
        assert dist == pytest.approx(0.5)

    def test_query_nearest_exclude(self):
        tree = self._make_tree()
        eid, dist = tree.query_nearest(np.array([0.0, 0.0]), exclude_id=10)
        assert eid == 20
        assert dist == pytest.approx(1.0)

    def test_query_k_nearest(self):
        tree = self._make_tree()
        results = tree.query_k_nearest(np.array([0.0, 0.0]), k=2)
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert 10 in ids
        assert 20 in ids

    def test_query_radius(self):
        tree = self._make_tree()
        results = tree.query_radius(np.array([0.0, 0.0]), radius=2.0)
        ids = [r[0] for r in results]
        assert 10 in ids
        assert 20 in ids
        assert 30 not in ids

    def test_query_rectangle(self):
        tree = self._make_tree()
        ids = tree.query_rectangle(
            np.array([0.0, 0.0]),
            np.array([2.0, 2.0]),
        )
        assert 10 in ids
        assert 20 in ids
        assert 30 not in ids

    def test_empty_tree(self):
        tree = KDTree2D(np.empty((0, 2)))
        assert tree.size == 0


# ===================================================================
# Spatial — Convenience functions
# ===================================================================


class TestSpatialConvenience:
    def test_find_neighbors_in_radius(self):
        positions = np.array([[0, 0], [1, 0], [5, 0], [10, 0]], dtype=float)
        nbrs = find_neighbors_in_radius(positions, query_idx=0, radius=2.0)
        assert 1 in nbrs
        assert 2 not in nbrs
        assert 0 not in nbrs  # excludes self

    def test_find_k_nearest(self):
        positions = np.array([[0, 0], [1, 0], [5, 0], [10, 0]], dtype=float)
        nbrs = find_k_nearest(positions, query_idx=0, k=2)
        assert len(nbrs) == 2
        assert 1 in nbrs

    def test_pairwise_distances(self):
        positions = np.array([[0, 0], [3, 4]], dtype=float)
        D = pairwise_distances(positions)
        assert D.shape == (2, 2)
        assert D[0, 0] == pytest.approx(0.0)
        assert D[0, 1] == pytest.approx(5.0)
        assert D[1, 0] == pytest.approx(5.0)

    def test_minimum_distances(self):
        a = np.array([[0, 0], [10, 10]], dtype=float)
        b = np.array([[1, 0], [2, 0]], dtype=float)
        result = minimum_distances(a, b)
        assert result[0] == pytest.approx(1.0)
        assert result[1] > 1.0
