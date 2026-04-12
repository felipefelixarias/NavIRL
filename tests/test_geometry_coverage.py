"""Tests for navirl.utils.geometry — targets uncovered functions and branches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.utils.geometry import (
    angle_between,
    angle_diff,
    angular_velocity,
    apply_transform_matrix,
    build_transform_matrix,
    circle_circle_intersection,
    circle_line_intersection,
    closest_point_on_line,
    compute_arc_length,
    compute_curvature,
    convex_hull,
    cross2d,
    distance,
    dot2d,
    heading_from_velocity,
    line_segment_intersection,
    minimum_bounding_rectangle,
    normalize_angle,
    normalize_vector,
    point_in_polygon,
    point_to_line_distance,
    polygon_area,
    polygon_centroid,
    quat_to_yaw,
    ray_segment_intersection,
    rotate_point,
    rotate_points,
    simplify_trajectory,
    transform_2d,
    wrap_angle,
)

# ────────────────────────────────────────────────────────────────────
# Angle utilities
# ────────────────────────────────────────────────────────────────────


class TestNormalizeAngle:
    def test_already_normalised(self):
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_positive_wrap(self):
        result = normalize_angle(3 * math.pi)
        assert abs(abs(result) - math.pi) < 1e-10

    def test_negative_wrap(self):
        result = normalize_angle(-3 * math.pi)
        assert abs(abs(result) - math.pi) < 1e-10

    def test_exact_pi(self):
        result = normalize_angle(math.pi)
        assert -math.pi <= result <= math.pi


class TestWrapAngle:
    def test_zero(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)

    def test_negative(self):
        result = wrap_angle(-math.pi / 2)
        assert 0.0 <= result < 2 * math.pi

    def test_large_positive(self):
        result = wrap_angle(5 * math.pi)
        assert 0.0 <= result < 2 * math.pi


class TestAngleDiff:
    def test_same(self):
        assert angle_diff(1.0, 1.0) == pytest.approx(0.0)

    def test_opposite(self):
        d = angle_diff(math.pi, 0.0)
        assert abs(d) == pytest.approx(math.pi, abs=1e-12)

    def test_wraps_shortest_path(self):
        d = angle_diff(0.1, 2 * math.pi - 0.1)
        assert d == pytest.approx(0.2, abs=1e-12)


class TestAngleBetween:
    def test_parallel(self):
        assert angle_between([1, 0], [2, 0]) == pytest.approx(0.0)

    def test_perpendicular(self):
        assert angle_between([1, 0], [0, 1]) == pytest.approx(math.pi / 2)

    def test_opposite(self):
        assert angle_between([1, 0], [-1, 0]) == pytest.approx(math.pi)

    def test_zero_vector(self):
        assert angle_between([0, 0], [1, 0]) == 0.0


class TestHeadingFromVelocity:
    def test_east(self):
        assert heading_from_velocity(1.0, 0.0) == pytest.approx(0.0)

    def test_north(self):
        assert heading_from_velocity(0.0, 1.0) == pytest.approx(math.pi / 2)

    def test_zero_velocity(self):
        assert heading_from_velocity(0.0, 0.0) == 0.0


class TestAngularVelocity:
    def test_basic(self):
        result = angular_velocity(0.0, math.pi / 2, 1.0)
        assert result == pytest.approx(math.pi / 2)

    def test_zero_dt(self):
        assert angular_velocity(0.0, 1.0, 0.0) == 0.0

    def test_negative_dt(self):
        assert angular_velocity(0.0, 1.0, -1.0) == 0.0


class TestQuatToYaw:
    def test_identity(self):
        assert quat_to_yaw(0, 0, 0, 1) == pytest.approx(0.0)

    def test_90_deg_yaw(self):
        # Quaternion for 90-deg rotation about z: (0, 0, sin(45°), cos(45°))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        assert quat_to_yaw(0, 0, s, c) == pytest.approx(math.pi / 2)


# ────────────────────────────────────────────────────────────────────
# Basic 2-D operations
# ────────────────────────────────────────────────────────────────────


class TestNormalizeVector:
    def test_unit(self):
        ux, uy, mag = normalize_vector(3.0, 4.0)
        assert mag == pytest.approx(5.0)
        assert ux == pytest.approx(0.6)
        assert uy == pytest.approx(0.8)

    def test_zero(self):
        ux, uy, mag = normalize_vector(0.0, 0.0)
        assert (ux, uy, mag) == (0.0, 0.0, 0.0)


class TestCross2dDot2d:
    def test_cross(self):
        assert cross2d(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)

    def test_dot(self):
        assert dot2d(np.array([1, 2]), np.array([3, 4])) == pytest.approx(11.0)


class TestDistance:
    def test_simple(self):
        assert distance(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)

    def test_batch(self):
        p1 = np.array([[0, 0], [1, 1]])
        p2 = np.array([[3, 4], [1, 1]])
        result = distance(p1, p2)
        np.testing.assert_allclose(result, [5.0, 0.0])


class TestRotatePoint:
    def test_90_degrees(self):
        result = rotate_point(np.array([1.0, 0.0]), math.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)

    def test_around_center(self):
        result = rotate_point(
            np.array([2.0, 0.0]), math.pi, center=np.array([1.0, 0.0])
        )
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)


class TestRotatePoints:
    def test_multiple(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = rotate_points(pts, math.pi / 2)
        np.testing.assert_allclose(result[0], [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(result[1], [-1.0, 0.0], atol=1e-12)

    def test_with_center(self):
        pts = np.array([[2.0, 0.0]])
        result = rotate_points(pts, math.pi, center=np.array([1.0, 0.0]))
        np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-12)


# ────────────────────────────────────────────────────────────────────
# Line / segment operations
# ────────────────────────────────────────────────────────────────────


class TestClosestPointOnLine:
    def test_midpoint(self):
        result = closest_point_on_line(
            np.array([0.5, 1.0]), np.array([0, 0]), np.array([1, 0])
        )
        np.testing.assert_allclose(result, [0.5, 0.0])

    def test_clamped_before_start(self):
        result = closest_point_on_line(
            np.array([-1.0, 0.0]), np.array([0, 0]), np.array([1, 0])
        )
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_unclamped(self):
        result = closest_point_on_line(
            np.array([-1.0, 0.0]),
            np.array([0, 0]),
            np.array([1, 0]),
            clamp_to_segment=False,
        )
        np.testing.assert_allclose(result, [-1.0, 0.0])

    def test_degenerate_segment(self):
        result = closest_point_on_line(
            np.array([5.0, 5.0]), np.array([1, 1]), np.array([1, 1])
        )
        np.testing.assert_allclose(result, [1.0, 1.0])


class TestPointToLineDistance:
    def test_distance_to_segment(self):
        d = point_to_line_distance(
            np.array([0.5, 1.0]), np.array([0, 0]), np.array([1, 0])
        )
        assert d == pytest.approx(1.0)

    def test_distance_to_infinite_line(self):
        d = point_to_line_distance(
            np.array([5.0, 1.0]),
            np.array([0, 0]),
            np.array([1, 0]),
            segment=False,
        )
        assert d == pytest.approx(1.0)


class TestLineSegmentIntersection:
    def test_crossing(self):
        result = line_segment_intersection(
            np.array([0, 0]),
            np.array([2, 2]),
            np.array([2, 0]),
            np.array([0, 2]),
        )
        assert result is not None
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_parallel(self):
        result = line_segment_intersection(
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
        )
        assert result is None

    def test_no_intersection(self):
        result = line_segment_intersection(
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([2, 1]),
            np.array([3, 1]),
        )
        assert result is None


class TestRaySegmentIntersection:
    def test_hit(self):
        t = ray_segment_intersection(
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([5, -1]),
            np.array([5, 1]),
        )
        assert t is not None
        assert t == pytest.approx(5.0)

    def test_miss_behind(self):
        t = ray_segment_intersection(
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([-5, -1]),
            np.array([-5, 1]),
        )
        assert t is None

    def test_parallel(self):
        t = ray_segment_intersection(
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([5, 1]),
        )
        assert t is None


class TestCircleCircleIntersection:
    def test_two_points(self):
        pts = circle_circle_intersection(
            np.array([0, 0]), 1.0, np.array([1, 0]), 1.0
        )
        assert len(pts) == 2

    def test_tangent(self):
        pts = circle_circle_intersection(
            np.array([0, 0]), 1.0, np.array([2, 0]), 1.0
        )
        assert len(pts) == 1
        np.testing.assert_allclose(pts[0], [1.0, 0.0], atol=1e-10)

    def test_too_far(self):
        pts = circle_circle_intersection(
            np.array([0, 0]), 1.0, np.array([5, 0]), 1.0
        )
        assert len(pts) == 0

    def test_concentric(self):
        pts = circle_circle_intersection(
            np.array([0, 0]), 1.0, np.array([0, 0]), 2.0
        )
        assert len(pts) == 0

    def test_one_inside_other(self):
        pts = circle_circle_intersection(
            np.array([0, 0]), 3.0, np.array([0.5, 0]), 1.0
        )
        assert len(pts) == 0


class TestCircleLineIntersection:
    def test_through_center(self):
        pts = circle_line_intersection(
            np.array([0, 0]), 1.0, np.array([-2, 0]), np.array([2, 0])
        )
        assert len(pts) == 2

    def test_tangent(self):
        # Line y=1 tangent to unit circle at (0,1)
        pts = circle_line_intersection(
            np.array([0, 0]), 1.0, np.array([-2, 1]), np.array([2, 1])
        )
        # Tangent line may produce 1 or 2 very close points depending on numerics
        assert len(pts) >= 1
        for p in pts:
            assert p[1] == pytest.approx(1.0, abs=1e-10)

    def test_miss(self):
        pts = circle_line_intersection(
            np.array([0, 0]), 1.0, np.array([-2, 3]), np.array([2, 3])
        )
        assert len(pts) == 0

    def test_degenerate_segment(self):
        pts = circle_line_intersection(
            np.array([0, 0]), 1.0, np.array([0, 0]), np.array([0, 0])
        )
        assert len(pts) == 0


# ────────────────────────────────────────────────────────────────────
# Polygon operations
# ────────────────────────────────────────────────────────────────────


class TestPointInPolygon:
    @pytest.fixture()
    def square(self):
        return np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)

    def test_inside(self, square):
        assert point_in_polygon(np.array([2, 2]), square) is True

    def test_outside(self, square):
        assert point_in_polygon(np.array([5, 5]), square) is False

    def test_near_edge(self, square):
        # Just inside
        assert point_in_polygon(np.array([0.01, 0.01]), square) is True


class TestPolygonArea:
    def test_unit_square(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert polygon_area(sq) == pytest.approx(1.0)

    def test_triangle(self):
        tri = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        assert polygon_area(tri) == pytest.approx(2.0)

    def test_degenerate(self):
        assert polygon_area(np.array([[0, 0], [1, 1]])) == 0.0


class TestPolygonCentroid:
    def test_unit_square(self):
        sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        c = polygon_centroid(sq)
        np.testing.assert_allclose(c, [1.0, 1.0], atol=1e-10)

    def test_empty(self):
        c = polygon_centroid(np.array([], dtype=float).reshape(0, 2))
        np.testing.assert_allclose(c, [0.0, 0.0])

    def test_single_point(self):
        c = polygon_centroid(np.array([[3.0, 4.0]]))
        np.testing.assert_allclose(c, [3.0, 4.0])

    def test_two_points(self):
        c = polygon_centroid(np.array([[0, 0], [2, 2]], dtype=float))
        np.testing.assert_allclose(c, [1.0, 1.0])


class TestConvexHull:
    def test_square_with_interior_point(self):
        pts = np.array(
            [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], dtype=float
        )
        hull = convex_hull(pts)
        assert len(hull) == 4

    def test_collinear(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        hull = convex_hull(pts)
        assert len(hull) >= 2

    def test_two_points(self):
        pts = np.array([[0, 0], [1, 1]], dtype=float)
        hull = convex_hull(pts)
        assert len(hull) == 2


class TestMinimumBoundingRectangle:
    def test_square_points(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        corners, w, h, angle = minimum_bounding_rectangle(pts)
        assert corners.shape == (4, 2)
        assert w * h == pytest.approx(1.0, abs=1e-6)

    def test_single_point(self):
        pts = np.array([[3.0, 4.0]])
        corners, w, h, angle = minimum_bounding_rectangle(pts)
        assert w == 0.0
        assert h == 0.0


# ────────────────────────────────────────────────────────────────────
# Transformation utilities
# ────────────────────────────────────────────────────────────────────


class TestTransform2d:
    def test_translate(self):
        result = transform_2d(np.array([1.0, 0.0]), translation=np.array([2, 3]))
        np.testing.assert_allclose(result, [3.0, 3.0])

    def test_scale(self):
        result = transform_2d(np.array([1.0, 2.0]), scale=2.0)
        np.testing.assert_allclose(result, [2.0, 4.0])

    def test_rotate(self):
        result = transform_2d(np.array([1.0, 0.0]), rotation=math.pi / 2)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)

    def test_batch(self):
        pts = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = transform_2d(pts, translation=np.array([1, 1]))
        np.testing.assert_allclose(result, [[2.0, 1.0], [1.0, 2.0]])

    def test_no_rotation_skips(self):
        result = transform_2d(np.array([1.0, 2.0]), rotation=0.0)
        np.testing.assert_allclose(result, [1.0, 2.0])


class TestBuildAndApplyTransformMatrix:
    def test_identity(self):
        m = build_transform_matrix()
        np.testing.assert_allclose(m, np.eye(3), atol=1e-12)

    def test_translate(self):
        m = build_transform_matrix(tx=3, ty=4)
        result = apply_transform_matrix(np.array([0.0, 0.0]), m)
        np.testing.assert_allclose(result, [3.0, 4.0])

    def test_rotate_90(self):
        m = build_transform_matrix(rotation=math.pi / 2)
        result = apply_transform_matrix(np.array([1.0, 0.0]), m)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)

    def test_scale(self):
        m = build_transform_matrix(scale=2.0)
        result = apply_transform_matrix(np.array([1.0, 1.0]), m)
        np.testing.assert_allclose(result, [2.0, 2.0], atol=1e-12)

    def test_batch_points(self):
        m = build_transform_matrix(tx=1, ty=2)
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = apply_transform_matrix(pts, m)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.0, 2.0])


# ────────────────────────────────────────────────────────────────────
# Trajectory utilities
# ────────────────────────────────────────────────────────────────────


class TestComputeCurvature:
    def test_straight_line(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        k = compute_curvature(pts)
        assert k[0] == 0.0
        assert k[-1] == 0.0
        np.testing.assert_allclose(k[1:-1], [0.0, 0.0])

    def test_circular_arc(self):
        angles = np.linspace(0, math.pi / 2, 20)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        k = compute_curvature(pts)
        # Curvature of unit circle ≈ 1
        assert np.mean(k[1:-1]) == pytest.approx(1.0, abs=0.05)

    def test_single_point(self):
        k = compute_curvature(np.array([[0, 0]], dtype=float))
        assert len(k) == 1
        assert k[0] == 0.0

    def test_duplicate_points(self):
        pts = np.array([[0, 0], [0, 0], [0, 0]], dtype=float)
        k = compute_curvature(pts)
        np.testing.assert_allclose(k, [0.0, 0.0, 0.0])


class TestComputeArcLength:
    def test_straight(self):
        pts = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)
        arc = compute_arc_length(pts)
        np.testing.assert_allclose(arc, [0.0, 1.0, 3.0])

    def test_single_point(self):
        arc = compute_arc_length(np.array([[0, 0]], dtype=float))
        assert len(arc) == 1
        assert arc[0] == 0.0


class TestSimplifyTrajectory:
    def test_straight_line_simplified(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        simplified = simplify_trajectory(pts, epsilon=0.01)
        # Should reduce to endpoints
        assert len(simplified) == 2

    def test_keeps_sharp_turn(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=float)
        simplified = simplify_trajectory(pts, epsilon=0.01)
        assert len(simplified) >= 3

    def test_short_trajectory(self):
        pts = np.array([[0, 0], [1, 1]], dtype=float)
        simplified = simplify_trajectory(pts, epsilon=1.0)
        assert len(simplified) == 2
