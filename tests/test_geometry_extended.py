"""Extended tests for navirl.utils.geometry — functions not covered by test_utils.py.

Covers: normalize_vector, circle_circle_intersection, circle_line_intersection,
convex_hull, minimum_bounding_rectangle, transform_2d, build_transform_matrix,
apply_transform_matrix, compute_curvature, compute_arc_length,
simplify_trajectory, and edge cases for polygon/line functions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.utils.geometry import (
    apply_transform_matrix,
    build_transform_matrix,
    circle_circle_intersection,
    circle_line_intersection,
    closest_point_on_line,
    compute_arc_length,
    compute_curvature,
    convex_hull,
    minimum_bounding_rectangle,
    normalize_vector,
    point_in_polygon,
    point_to_line_distance,
    polygon_area,
    polygon_centroid,
    simplify_trajectory,
    transform_2d,
)

# ===================================================================
# normalize_vector
# ===================================================================


class TestNormalizeVector:
    def test_unit_x(self):
        ux, uy, mag = normalize_vector(3.0, 0.0)
        assert ux == pytest.approx(1.0)
        assert uy == pytest.approx(0.0)
        assert mag == pytest.approx(3.0)

    def test_unit_y(self):
        ux, uy, mag = normalize_vector(0.0, -4.0)
        assert ux == pytest.approx(0.0)
        assert uy == pytest.approx(-1.0)
        assert mag == pytest.approx(4.0)

    def test_diagonal(self):
        ux, uy, mag = normalize_vector(3.0, 4.0)
        assert mag == pytest.approx(5.0)
        assert ux == pytest.approx(0.6)
        assert uy == pytest.approx(0.8)

    def test_zero_vector(self):
        ux, uy, mag = normalize_vector(0.0, 0.0)
        assert ux == 0.0
        assert uy == 0.0
        assert mag == 0.0

    def test_tiny_vector(self):
        ux, uy, mag = normalize_vector(1e-12, 1e-12)
        assert ux == 0.0
        assert uy == 0.0
        assert mag == 0.0


# ===================================================================
# circle_circle_intersection
# ===================================================================


class TestCircleCircleIntersection:
    def test_no_intersection_far_apart(self):
        pts = circle_circle_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([10.0, 0.0]),
            1.0,
        )
        assert pts == []

    def test_no_intersection_concentric(self):
        pts = circle_circle_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([0.0, 0.0]),
            2.0,
        )
        assert pts == []

    def test_no_intersection_one_inside_other(self):
        pts = circle_circle_intersection(
            np.array([0.0, 0.0]),
            5.0,
            np.array([1.0, 0.0]),
            1.0,
        )
        assert pts == []

    def test_tangent_externally(self):
        pts = circle_circle_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([2.0, 0.0]),
            1.0,
        )
        assert len(pts) == 1
        np.testing.assert_allclose(pts[0], [1.0, 0.0], atol=1e-10)

    def test_two_intersections(self):
        pts = circle_circle_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([1.0, 0.0]),
            1.0,
        )
        assert len(pts) == 2
        # Both points should be at distance 1 from each center
        for p in pts:
            assert np.linalg.norm(p) == pytest.approx(1.0, abs=1e-10)
            assert np.linalg.norm(p - np.array([1.0, 0.0])) == pytest.approx(1.0, abs=1e-10)

    def test_identical_circles(self):
        # Same center, same radius -> concentric check returns empty
        pts = circle_circle_intersection(
            np.array([1.0, 1.0]),
            2.0,
            np.array([1.0, 1.0]),
            2.0,
        )
        assert pts == []


# ===================================================================
# circle_line_intersection
# ===================================================================


class TestCircleLineIntersection:
    def test_line_through_center(self):
        pts = circle_line_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([-2.0, 0.0]),
            np.array([2.0, 0.0]),
        )
        assert len(pts) == 2
        xs = sorted([p[0] for p in pts])
        assert xs[0] == pytest.approx(-1.0, abs=1e-10)
        assert xs[1] == pytest.approx(1.0, abs=1e-10)

    def test_line_tangent(self):
        pts = circle_line_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([-2.0, 1.0]),
            np.array([2.0, 1.0]),
        )
        # Tangent may return 1 or 2 nearly-coincident points depending on
        # floating point precision of the discriminant
        assert len(pts) >= 1
        for p in pts:
            assert p[1] == pytest.approx(1.0, abs=1e-10)

    def test_line_misses(self):
        pts = circle_line_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([-2.0, 5.0]),
            np.array([2.0, 5.0]),
        )
        assert pts == []

    def test_segment_partially_inside(self):
        # Segment from center to outside — one intersection
        pts = circle_line_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
        )
        assert len(pts) == 1
        np.testing.assert_allclose(pts[0], [1.0, 0.0], atol=1e-10)

    def test_degenerate_segment(self):
        # Zero-length segment
        pts = circle_line_intersection(
            np.array([0.0, 0.0]),
            1.0,
            np.array([0.5, 0.0]),
            np.array([0.5, 0.0]),
        )
        assert pts == []


# ===================================================================
# convex_hull
# ===================================================================


class TestConvexHull:
    def test_square(self):
        points = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        hull = convex_hull(points)
        # Interior point should be excluded
        assert len(hull) == 4

    def test_triangle(self):
        points = np.array([[0, 0], [1, 0], [0.5, 1.0]], dtype=float)
        hull = convex_hull(points)
        assert len(hull) == 3

    def test_collinear(self):
        points = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        hull = convex_hull(points)
        assert len(hull) >= 2

    def test_single_point(self):
        points = np.array([[3.0, 4.0]])
        hull = convex_hull(points)
        assert len(hull) == 1

    def test_two_points(self):
        points = np.array([[0, 0], [1, 1]], dtype=float)
        hull = convex_hull(points)
        assert len(hull) == 2

    def test_many_points_circle(self):
        # Points on a circle
        angles = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = convex_hull(points)
        assert len(hull) == 20  # All points are on the hull


# ===================================================================
# minimum_bounding_rectangle
# ===================================================================


class TestMinimumBoundingRectangle:
    def test_square_points(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        corners, width, height, angle = minimum_bounding_rectangle(points)
        assert corners.shape == (4, 2)
        assert width * height == pytest.approx(1.0, abs=1e-6)

    def test_single_point(self):
        points = np.array([[5.0, 5.0]])
        corners, width, height, angle = minimum_bounding_rectangle(points)
        assert width == 0.0
        assert height == 0.0

    def test_rectangle_axis_aligned(self):
        points = np.array(
            [
                [0, 0],
                [4, 0],
                [4, 2],
                [0, 2],
            ],
            dtype=float,
        )
        corners, width, height, angle = minimum_bounding_rectangle(points)
        dims = sorted([width, height])
        assert dims[0] == pytest.approx(2.0, abs=1e-6)
        assert dims[1] == pytest.approx(4.0, abs=1e-6)


# ===================================================================
# transform_2d
# ===================================================================


class TestTransform2D:
    def test_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = transform_2d(pts)
        np.testing.assert_allclose(result, pts)

    def test_translation(self):
        pt = np.array([1.0, 2.0])
        result = transform_2d(pt, translation=np.array([10.0, 20.0]))
        np.testing.assert_allclose(result, [11.0, 22.0])

    def test_rotation(self):
        pt = np.array([1.0, 0.0])
        result = transform_2d(pt, rotation=math.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_scale(self):
        pt = np.array([1.0, 2.0])
        result = transform_2d(pt, scale=3.0)
        np.testing.assert_allclose(result, [3.0, 6.0])

    def test_combined(self):
        pt = np.array([1.0, 0.0])
        result = transform_2d(
            pt,
            translation=np.array([5.0, 0.0]),
            rotation=math.pi / 2,
            scale=2.0,
        )
        # scale(1,0)*2 = (2,0), rotate 90° -> (0,2), translate -> (5,2)
        np.testing.assert_allclose(result, [5.0, 2.0], atol=1e-10)

    def test_batch(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = transform_2d(pts, scale=2.0)
        np.testing.assert_allclose(result, [[2.0, 0.0], [0.0, 2.0]])


# ===================================================================
# build_transform_matrix / apply_transform_matrix
# ===================================================================


class TestTransformMatrix:
    def test_identity_matrix(self):
        M = build_transform_matrix()
        np.testing.assert_allclose(M, np.eye(3), atol=1e-12)

    def test_translation_matrix(self):
        M = build_transform_matrix(tx=5.0, ty=3.0)
        pt = np.array([1.0, 2.0])
        result = apply_transform_matrix(pt, M)
        np.testing.assert_allclose(result, [6.0, 5.0])

    def test_rotation_matrix(self):
        M = build_transform_matrix(rotation=math.pi / 2)
        pt = np.array([1.0, 0.0])
        result = apply_transform_matrix(pt, M)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_scale_matrix(self):
        M = build_transform_matrix(scale=3.0)
        pt = np.array([1.0, 2.0])
        result = apply_transform_matrix(pt, M)
        np.testing.assert_allclose(result, [3.0, 6.0], atol=1e-10)

    def test_batch_transform(self):
        M = build_transform_matrix(tx=1.0, ty=2.0)
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = apply_transform_matrix(pts, M)
        np.testing.assert_allclose(result, [[1.0, 2.0], [2.0, 3.0]])


# ===================================================================
# compute_curvature
# ===================================================================


class TestComputeCurvature:
    def test_straight_line(self):
        positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        curv = compute_curvature(positions)
        assert len(curv) == 4
        assert curv[0] == 0.0
        assert curv[-1] == 0.0
        np.testing.assert_allclose(curv[1:-1], [0.0, 0.0], atol=1e-10)

    def test_circle_arc(self):
        # Points on a unit circle should have curvature ~1
        angles = np.linspace(0, math.pi / 2, 10)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
        curv = compute_curvature(positions)
        # Interior curvature should be approximately 1 (unit circle)
        for c in curv[1:-1]:
            assert c == pytest.approx(1.0, abs=0.1)

    def test_single_point(self):
        curv = compute_curvature(np.array([[0.0, 0.0]]))
        assert len(curv) == 1
        assert curv[0] == 0.0

    def test_two_points(self):
        curv = compute_curvature(np.array([[0.0, 0.0], [1.0, 0.0]]))
        np.testing.assert_allclose(curv, [0.0, 0.0])

    def test_duplicate_points(self):
        positions = np.array([[0, 0], [0, 0], [1, 0]], dtype=float)
        curv = compute_curvature(positions)
        # Should handle degenerate case gracefully
        assert len(curv) == 3
        assert curv[1] == 0.0  # degenerate


# ===================================================================
# compute_arc_length
# ===================================================================


class TestComputeArcLength:
    def test_straight_line(self):
        positions = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)
        arc = compute_arc_length(positions)
        np.testing.assert_allclose(arc, [0.0, 1.0, 3.0])

    def test_single_point(self):
        arc = compute_arc_length(np.array([[5.0, 5.0]]))
        np.testing.assert_allclose(arc, [0.0])

    def test_empty(self):
        arc = compute_arc_length(np.empty((0, 2)))
        assert len(arc) == 0

    def test_diagonal(self):
        positions = np.array([[0, 0], [1, 1]], dtype=float)
        arc = compute_arc_length(positions)
        assert arc[0] == 0.0
        assert arc[1] == pytest.approx(math.sqrt(2))


# ===================================================================
# simplify_trajectory
# ===================================================================


class TestSimplifyTrajectory:
    def test_straight_line(self):
        # Points on a line should simplify to just endpoints
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [4, 0],
            ],
            dtype=float,
        )
        simplified = simplify_trajectory(positions, epsilon=0.1)
        assert len(simplified) == 2
        np.testing.assert_allclose(simplified[0], [0, 0])
        np.testing.assert_allclose(simplified[-1], [4, 0])

    def test_l_shape(self):
        positions = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [2, 1],
                [2, 2],
            ],
            dtype=float,
        )
        simplified = simplify_trajectory(positions, epsilon=0.1)
        # Should keep the corner point
        assert len(simplified) >= 3

    def test_single_point(self):
        result = simplify_trajectory(np.array([[1.0, 2.0]]))
        assert len(result) == 1

    def test_two_points(self):
        pts = np.array([[0, 0], [1, 1]], dtype=float)
        result = simplify_trajectory(pts, epsilon=0.1)
        assert len(result) == 2

    def test_large_epsilon(self):
        # Very large epsilon should reduce to endpoints
        positions = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 0],
                [3, 1],
                [4, 0],
            ],
            dtype=float,
        )
        simplified = simplify_trajectory(positions, epsilon=100.0)
        assert len(simplified) == 2


# ===================================================================
# Edge cases for existing functions
# ===================================================================


class TestGeometryEdgeCases:
    def test_closest_point_degenerate_segment(self):
        # Zero-length segment
        cp = closest_point_on_line(
            np.array([5.0, 3.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(cp, [1.0, 1.0])

    def test_polygon_area_fewer_than_3(self):
        assert polygon_area(np.array([[0, 0], [1, 1]], dtype=float)) == 0.0
        assert polygon_area(np.array([[0, 0]], dtype=float)) == 0.0

    def test_polygon_centroid_empty(self):
        c = polygon_centroid(np.empty((0, 2)))
        np.testing.assert_allclose(c, [0.0, 0.0])

    def test_polygon_centroid_single_point(self):
        c = polygon_centroid(np.array([[3.0, 4.0]]))
        np.testing.assert_allclose(c, [3.0, 4.0])

    def test_polygon_centroid_two_points(self):
        c = polygon_centroid(np.array([[0.0, 0.0], [4.0, 0.0]]))
        np.testing.assert_allclose(c, [2.0, 0.0])

    def test_polygon_centroid_degenerate(self):
        # Collinear points -> zero area -> falls back to mean
        c = polygon_centroid(np.array([[0, 0], [1, 0], [2, 0]], dtype=float))
        np.testing.assert_allclose(c, [1.0, 0.0])

    def test_point_in_polygon_on_edge(self):
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        # On edge - implementation-dependent, just verify it doesn't crash
        result = point_in_polygon(np.array([1.0, 0.0]), square)
        assert isinstance(result, bool)

    def test_point_to_line_distance_on_line(self):
        d = point_to_line_distance(
            np.array([5.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 0.0]),
        )
        assert d == pytest.approx(0.0)
