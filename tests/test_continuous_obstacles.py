"""Tests for navirl.backends.continuous.obstacles — geometric primitives and spatial indexing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.backends.continuous.obstacles import (
    CircleObstacle,
    LineObstacle,
    Obstacle,
    ObstacleCollection,
    PolygonObstacle,
    RectangleObstacle,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

SQRT2 = math.sqrt(2.0)


def _unit_square() -> RectangleObstacle:
    """Axis-aligned unit square centred at the origin."""
    return RectangleObstacle(min_corner=np.array([-1, -1]), max_corner=np.array([1, 1]))


def _unit_circle() -> CircleObstacle:
    return CircleObstacle(center=np.array([0.0, 0.0]), radius=1.0)


def _triangle() -> PolygonObstacle:
    """CCW triangle centred near origin."""
    return PolygonObstacle(vertices=np.array([[0.0, 2.0], [-1.0, 0.0], [1.0, 0.0]]))


def _horizontal_wall() -> LineObstacle:
    """Horizontal wall from (-5, 0) to (5, 0), thickness=0.2."""
    return LineObstacle(start=np.array([-5.0, 0.0]), end=np.array([5.0, 0.0]), thickness=0.2)


# ===========================================================================
#  CircleObstacle
# ===========================================================================


class TestCircleObstacle:
    def test_contains_point_inside(self):
        c = _unit_circle()
        assert c.contains_point(np.array([0.0, 0.0]))
        assert c.contains_point(np.array([0.5, 0.0]))

    def test_contains_point_boundary(self):
        c = _unit_circle()
        assert c.contains_point(np.array([1.0, 0.0]))

    def test_contains_point_outside(self):
        c = _unit_circle()
        assert not c.contains_point(np.array([2.0, 0.0]))

    def test_distance_to_point_outside(self):
        c = _unit_circle()
        d = c.distance_to_point(np.array([3.0, 0.0]))
        assert d == pytest.approx(2.0)

    def test_distance_to_point_inside_negative(self):
        c = _unit_circle()
        d = c.distance_to_point(np.array([0.5, 0.0]))
        assert d == pytest.approx(-0.5)

    def test_distance_to_point_on_boundary(self):
        c = _unit_circle()
        d = c.distance_to_point(np.array([0.0, 1.0]))
        assert d == pytest.approx(0.0, abs=1e-12)

    def test_intersects_circle_overlap(self):
        c = _unit_circle()
        assert c.intersects_circle(np.array([1.5, 0.0]), 0.6)

    def test_intersects_circle_touching(self):
        c = _unit_circle()
        assert c.intersects_circle(np.array([2.0, 0.0]), 1.0)

    def test_intersects_circle_separate(self):
        c = _unit_circle()
        assert not c.intersects_circle(np.array([5.0, 0.0]), 0.5)

    def test_ray_cast_hit(self):
        c = _unit_circle()
        t = c.ray_cast(np.array([-5.0, 0.0]), np.array([1.0, 0.0]))
        assert t is not None
        assert t == pytest.approx(4.0)

    def test_ray_cast_miss(self):
        c = _unit_circle()
        t = c.ray_cast(np.array([-5.0, 5.0]), np.array([1.0, 0.0]))
        assert t is None

    def test_ray_cast_from_inside(self):
        c = _unit_circle()
        t = c.ray_cast(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert t is not None
        assert t == pytest.approx(1.0)

    def test_ray_cast_away_from_circle(self):
        c = _unit_circle()
        t = c.ray_cast(np.array([5.0, 0.0]), np.array([1.0, 0.0]))
        assert t is None

    def test_closest_point_outside(self):
        c = _unit_circle()
        p = c.closest_point(np.array([3.0, 0.0]))
        np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-12)

    def test_closest_point_at_center(self):
        c = _unit_circle()
        p = c.closest_point(np.array([0.0, 0.0]))
        np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-12)

    def test_normal_at_outside(self):
        c = _unit_circle()
        n = c.normal_at(np.array([3.0, 0.0]))
        np.testing.assert_allclose(n, [1.0, 0.0], atol=1e-12)

    def test_normal_at_center(self):
        c = _unit_circle()
        n = c.normal_at(np.array([0.0, 0.0]))
        np.testing.assert_allclose(n, [1.0, 0.0], atol=1e-12)

    def test_normal_at_diagonal(self):
        c = _unit_circle()
        n = c.normal_at(np.array([1.0, 1.0]))
        expected = np.array([1.0, 1.0]) / SQRT2
        np.testing.assert_allclose(n, expected, atol=1e-12)

    def test_bounding_box(self):
        c = CircleObstacle(center=np.array([2.0, 3.0]), radius=0.5)
        bb_min, bb_max = c.get_bounding_box()
        np.testing.assert_allclose(bb_min, [1.5, 2.5])
        np.testing.assert_allclose(bb_max, [2.5, 3.5])

    def test_get_vertices_approximation(self):
        c = _unit_circle()
        verts = c.get_vertices()
        assert verts is not None
        assert verts.shape[1] == 2
        # All vertices should be on the circle
        dists = np.linalg.norm(verts - c.center, axis=1)
        np.testing.assert_allclose(dists, 1.0, atol=1e-12)

    def test_inflate(self):
        c = _unit_circle()
        inflated = c.inflate(0.5)
        assert inflated.radius == pytest.approx(1.5)
        np.testing.assert_array_equal(inflated.center, c.center)

    def test_post_init_coercion(self):
        c = CircleObstacle(center=[1, 2], radius=3.0)
        assert c.center.dtype == np.float64


# ===========================================================================
#  RectangleObstacle
# ===========================================================================


class TestRectangleObstacle:
    def test_center(self):
        r = _unit_square()
        np.testing.assert_allclose(r.center, [0.0, 0.0])

    def test_width_height(self):
        r = _unit_square()
        assert r.width == pytest.approx(2.0)
        assert r.height == pytest.approx(2.0)

    def test_contains_point_inside(self):
        r = _unit_square()
        assert r.contains_point(np.array([0.0, 0.0]))

    def test_contains_point_boundary(self):
        r = _unit_square()
        assert r.contains_point(np.array([1.0, 0.0]))

    def test_contains_point_outside(self):
        r = _unit_square()
        assert not r.contains_point(np.array([2.0, 0.0]))

    def test_distance_to_point_outside(self):
        r = _unit_square()
        assert r.distance_to_point(np.array([3.0, 0.0])) == pytest.approx(2.0)

    def test_distance_to_point_outside_corner(self):
        r = _unit_square()
        d = r.distance_to_point(np.array([2.0, 2.0]))
        assert d == pytest.approx(SQRT2)

    def test_distance_to_point_inside_negative(self):
        r = _unit_square()
        d = r.distance_to_point(np.array([0.0, 0.0]))
        assert d == pytest.approx(-1.0)

    def test_intersects_circle(self):
        r = _unit_square()
        assert r.intersects_circle(np.array([1.5, 0.0]), 0.6)
        assert not r.intersects_circle(np.array([5.0, 0.0]), 0.1)

    def test_ray_cast_hit(self):
        r = _unit_square()
        t = r.ray_cast(np.array([-5.0, 0.0]), np.array([1.0, 0.0]))
        assert t is not None
        assert t == pytest.approx(4.0)

    def test_ray_cast_miss(self):
        r = _unit_square()
        t = r.ray_cast(np.array([-5.0, 0.0]), np.array([0.0, 1.0]))
        assert t is None

    def test_ray_cast_parallel_outside(self):
        r = _unit_square()
        t = r.ray_cast(np.array([-5.0, 5.0]), np.array([1.0, 0.0]))
        assert t is None

    def test_closest_point_outside(self):
        r = _unit_square()
        p = r.closest_point(np.array([3.0, 0.0]))
        np.testing.assert_allclose(p, [1.0, 0.0])

    def test_closest_point_inside(self):
        r = _unit_square()
        p = r.closest_point(np.array([0.8, 0.0]))
        # Nearest edge is at x=1
        np.testing.assert_allclose(p, [1.0, 0.0])

    def test_normal_at_outside(self):
        r = _unit_square()
        n = r.normal_at(np.array([3.0, 0.0]))
        np.testing.assert_allclose(n, [1.0, 0.0], atol=1e-12)

    def test_normal_at_on_boundary_right(self):
        r = _unit_square()
        n = r.normal_at(np.array([1.0, 0.0]))
        assert n[0] == pytest.approx(1.0)

    def test_normal_at_on_boundary_top(self):
        r = _unit_square()
        n = r.normal_at(np.array([0.0, 1.0]))
        assert n[1] == pytest.approx(1.0)

    def test_bounding_box(self):
        r = _unit_square()
        bb_min, bb_max = r.get_bounding_box()
        np.testing.assert_allclose(bb_min, [-1.0, -1.0])
        np.testing.assert_allclose(bb_max, [1.0, 1.0])

    def test_get_vertices(self):
        r = _unit_square()
        v = r.get_vertices()
        assert v.shape == (4, 2)

    def test_inflate(self):
        r = _unit_square()
        inflated = r.inflate(0.5)
        assert inflated.width == pytest.approx(3.0)
        assert inflated.height == pytest.approx(3.0)


# ===========================================================================
#  LineObstacle
# ===========================================================================


class TestLineObstacle:
    def test_length(self):
        w = _horizontal_wall()
        assert w.length == pytest.approx(10.0)

    def test_direction(self):
        w = _horizontal_wall()
        np.testing.assert_allclose(w.direction, [1.0, 0.0], atol=1e-12)

    def test_normal_vec(self):
        w = _horizontal_wall()
        np.testing.assert_allclose(w.normal_vec, [0.0, 1.0], atol=1e-12)

    def test_direction_zero_length(self):
        w = LineObstacle(start=np.array([0.0, 0.0]), end=np.array([0.0, 0.0]))
        np.testing.assert_allclose(w.direction, [1.0, 0.0])

    def test_contains_point_near_wall(self):
        w = _horizontal_wall()
        assert w.contains_point(np.array([0.0, 0.05]))  # within thickness/2
        assert not w.contains_point(np.array([0.0, 0.2]))  # outside thickness/2

    def test_distance_to_point(self):
        w = _horizontal_wall()
        d = w.distance_to_point(np.array([0.0, 1.0]))
        assert d == pytest.approx(0.9)  # 1.0 - thickness/2

    def test_distance_to_point_beyond_endpoint(self):
        w = _horizontal_wall()
        d = w.distance_to_point(np.array([6.0, 0.0]))
        assert d == pytest.approx(1.0 - 0.1)  # dist to end minus half thickness

    def test_intersects_circle(self):
        w = _horizontal_wall()
        assert w.intersects_circle(np.array([0.0, 0.5]), 0.5)
        assert not w.intersects_circle(np.array([0.0, 5.0]), 0.1)

    def test_ray_cast_perpendicular_hit(self):
        w = _horizontal_wall()
        t = w.ray_cast(np.array([0.0, -5.0]), np.array([0.0, 1.0]))
        assert t is not None
        assert t == pytest.approx(5.0)

    def test_ray_cast_parallel_miss(self):
        w = _horizontal_wall()
        t = w.ray_cast(np.array([0.0, 5.0]), np.array([1.0, 0.0]))
        assert t is None

    def test_ray_cast_behind(self):
        w = _horizontal_wall()
        t = w.ray_cast(np.array([0.0, 5.0]), np.array([0.0, 1.0]))
        assert t is None

    def test_closest_point_on_segment(self):
        w = _horizontal_wall()
        p = w.closest_point(np.array([0.0, 5.0]))
        np.testing.assert_allclose(p, [0.0, 0.0], atol=1e-12)

    def test_closest_point_beyond_start(self):
        w = _horizontal_wall()
        p = w.closest_point(np.array([-10.0, 0.0]))
        np.testing.assert_allclose(p, [-5.0, 0.0], atol=1e-12)

    def test_closest_point_zero_length(self):
        w = LineObstacle(start=np.array([1.0, 1.0]), end=np.array([1.0, 1.0]))
        p = w._closest_point_on_segment(np.array([5.0, 5.0]))
        np.testing.assert_allclose(p, [1.0, 1.0])

    def test_normal_at(self):
        w = _horizontal_wall()
        n = w.normal_at(np.array([0.0, 5.0]))
        np.testing.assert_allclose(n, [0.0, 1.0], atol=1e-12)

    def test_bounding_box(self):
        w = _horizontal_wall()
        bb_min, bb_max = w.get_bounding_box()
        assert bb_min[1] == pytest.approx(-0.1)
        assert bb_max[1] == pytest.approx(0.1)

    def test_get_vertices(self):
        w = _horizontal_wall()
        v = w.get_vertices()
        assert v.shape == (4, 2)

    def test_inflate(self):
        w = _horizontal_wall()
        inflated = w.inflate(0.1)
        assert inflated.thickness == pytest.approx(0.4)


# ===========================================================================
#  PolygonObstacle
# ===========================================================================


class TestPolygonObstacle:
    def test_center(self):
        tri = _triangle()
        c = tri.center
        np.testing.assert_allclose(c, [0.0, 2.0 / 3.0], atol=1e-12)

    def test_num_vertices(self):
        tri = _triangle()
        assert tri.num_vertices == 3

    def test_contains_point_inside(self):
        tri = _triangle()
        assert tri.contains_point(np.array([0.0, 0.5]))

    def test_contains_point_outside(self):
        tri = _triangle()
        assert not tri.contains_point(np.array([5.0, 5.0]))

    def test_contains_point_vertex_level(self):
        # Points clearly outside should fail
        tri = _triangle()
        assert not tri.contains_point(np.array([0.0, 3.0]))

    def test_distance_to_point_outside(self):
        tri = _triangle()
        d = tri.distance_to_point(np.array([0.0, -1.0]))
        assert d == pytest.approx(1.0)

    def test_distance_to_point_inside_negative(self):
        tri = _triangle()
        d = tri.distance_to_point(np.array([0.0, 0.5]))
        assert d < 0

    def test_intersects_circle_overlap(self):
        tri = _triangle()
        assert tri.intersects_circle(np.array([0.0, 0.5]), 0.5)

    def test_intersects_circle_separate(self):
        tri = _triangle()
        assert not tri.intersects_circle(np.array([10.0, 10.0]), 0.1)

    def test_ray_cast_hit(self):
        tri = _triangle()
        t = tri.ray_cast(np.array([0.0, -5.0]), np.array([0.0, 1.0]))
        assert t is not None
        assert t == pytest.approx(5.0)  # hits bottom edge at y=0

    def test_ray_cast_miss(self):
        tri = _triangle()
        t = tri.ray_cast(np.array([10.0, 0.0]), np.array([0.0, 1.0]))
        assert t is None

    def test_closest_point(self):
        tri = _triangle()
        p = tri.closest_point(np.array([0.0, -1.0]))
        # Should be on the bottom edge
        assert p[1] == pytest.approx(0.0, abs=1e-6)

    def test_normal_at(self):
        tri = _triangle()
        n = tri.normal_at(np.array([0.0, -1.0]))
        # Normal of bottom edge should point downward (outward)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-6)

    def test_bounding_box(self):
        tri = _triangle()
        bb_min, bb_max = tri.get_bounding_box()
        np.testing.assert_allclose(bb_min, [-1.0, 0.0])
        np.testing.assert_allclose(bb_max, [1.0, 2.0])

    def test_get_vertices(self):
        tri = _triangle()
        v = tri.get_vertices()
        assert v.shape == (3, 2)

    def test_inflate(self):
        tri = _triangle()
        inflated = tri.inflate(0.5)
        # All vertices should be further from center
        orig_dists = np.linalg.norm(tri.vertices - tri._center, axis=1)
        new_dists = np.linalg.norm(inflated.vertices - inflated._center, axis=1)
        assert np.all(new_dists > orig_dists)

    def test_ray_cast_parallel_to_edge(self):
        tri = _triangle()
        t = tri.ray_cast(np.array([-5.0, 0.0]), np.array([1.0, 0.0]))
        assert t is not None  # should hit left edge of triangle

    def test_polygon_square(self):
        """Test polygon with a square shape for more predictable geometry."""
        sq = PolygonObstacle(vertices=np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float))
        assert sq.contains_point(np.array([1.0, 1.0]))
        assert not sq.contains_point(np.array([3.0, 1.0]))


# ===========================================================================
#  Obstacle ABC
# ===========================================================================


class TestObstacleABC:
    def test_inflate_raises(self):
        """Default inflate raises NotImplementedError on a subclass that doesn't override it."""

        # Create a minimal concrete subclass without inflate
        class _BareObstacle(Obstacle):
            def contains_point(self, p):
                return False

            def distance_to_point(self, p):
                return 0.0

            def intersects_circle(self, c, r):
                return False

            def ray_cast(self, o, d):
                return None

            def closest_point(self, p):
                return np.zeros(2)

            def normal_at(self, p):
                return np.array([1.0, 0.0])

            def get_bounding_box(self):
                return np.zeros(2), np.ones(2)

            def get_vertices(self):
                return None

        with pytest.raises(NotImplementedError):
            _BareObstacle().inflate(1.0)


# ===========================================================================
#  ObstacleCollection
# ===========================================================================


class TestObstacleCollection:
    def test_add_and_len(self):
        col = ObstacleCollection()
        idx = col.add(_unit_circle())
        assert idx == 0
        assert len(col) == 1

    def test_getitem(self):
        col = ObstacleCollection()
        c = _unit_circle()
        col.add(c)
        assert col[0] is c

    def test_get_all_obstacles(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        col.add(_unit_square())
        assert len(col.get_all_obstacles()) == 2

    def test_clear(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        col.clear()
        assert len(col) == 0

    def test_check_collision_hit(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        assert col.check_collision(np.array([0.5, 0.0]), 0.3)

    def test_check_collision_miss(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        assert not col.check_collision(np.array([10.0, 0.0]), 0.1)

    def test_nearest_obstacle_distance_empty(self):
        col = ObstacleCollection()
        assert col.nearest_obstacle_distance(np.array([0.0, 0.0])) == float("inf")

    def test_nearest_obstacle_distance(self):
        col = ObstacleCollection()
        col.add(CircleObstacle(center=np.array([5.0, 0.0]), radius=1.0))
        d = col.nearest_obstacle_distance(np.array([0.0, 0.0]))
        assert d == pytest.approx(4.0)

    def test_ray_cast_hit(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        result = col.ray_cast(np.array([-5.0, 0.0]), np.array([1.0, 0.0]))
        assert result is not None
        t, idx = result
        assert t == pytest.approx(4.0)
        assert idx == 0

    def test_ray_cast_miss(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        result = col.ray_cast(np.array([-5.0, 5.0]), np.array([1.0, 0.0]))
        assert result is None

    def test_ray_cast_max_distance(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        result = col.ray_cast(np.array([-5.0, 0.0]), np.array([1.0, 0.0]), max_distance=2.0)
        assert result is None

    def test_ray_cast_multiple_obstacles_returns_closest(self):
        col = ObstacleCollection()
        col.add(CircleObstacle(center=np.array([3.0, 0.0]), radius=0.5))
        col.add(CircleObstacle(center=np.array([10.0, 0.0]), radius=0.5))
        result = col.ray_cast(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert result is not None
        t, idx = result
        assert idx == 0
        assert t == pytest.approx(2.5)

    def test_multi_ray_cast(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        distances = col.multi_ray_cast(np.array([-5.0, 0.0]), num_rays=4, max_distance=20.0)
        assert len(distances) == 4
        # The ray going east (angle=0) should hit the circle
        assert distances[0] == pytest.approx(4.0)

    def test_multi_ray_cast_no_obstacles(self):
        col = ObstacleCollection()
        distances = col.multi_ray_cast(np.array([0.0, 0.0]), num_rays=8, max_distance=10.0)
        np.testing.assert_array_equal(distances, np.full(8, 10.0))

    def test_inflate_all(self):
        col = ObstacleCollection()
        col.add(_unit_circle())
        col.add(_unit_square())
        inflated = col.inflate_all(0.5)
        assert len(inflated) == 2
        assert isinstance(inflated[0], CircleObstacle)
        assert inflated[0].radius == pytest.approx(1.5)

    def test_inflate_all_with_uninflatable(self):
        """inflate_all should keep obstacles that don't support inflation."""
        col = ObstacleCollection()
        col.add(_unit_circle())
        # Even after inflate_all, collection should have same length
        inflated = col.inflate_all(0.1)
        assert len(inflated) == 1

    def test_spatial_index_grid(self):
        """Verify spatial index correctly finds obstacles in nearby cells."""
        col = ObstacleCollection(cell_size=2.0)
        col.add(CircleObstacle(center=np.array([0.0, 0.0]), radius=0.5))
        col.add(CircleObstacle(center=np.array([10.0, 10.0]), radius=0.5))
        # Check collision near first obstacle
        assert col.check_collision(np.array([0.5, 0.0]), 0.5)
        # No collision near empty space
        assert not col.check_collision(np.array([5.0, 5.0]), 0.1)
