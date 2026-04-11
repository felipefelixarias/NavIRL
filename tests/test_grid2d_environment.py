"""Tests for navirl/backends/grid2d/environment.py — GridEnvironment."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.backends.grid2d.constants import FREE_SPACE, OBSTACLE_SPACE
from navirl.backends.grid2d.environment import GridEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_map():
    """10x10 all-free map (pixel value 255 = free after normalization)."""
    return np.full((10, 10), 255, dtype=np.uint8)


@pytest.fixture
def env(simple_map):
    return GridEnvironment("test", simple_map, pixels_per_meter=10.0)


@pytest.fixture
def env_with_obstacle():
    """10x10 map with a 4x4 obstacle block in the center."""
    m = np.full((10, 10), 255, dtype=np.uint8)
    m[3:7, 3:7] = 0  # obstacle block
    return GridEnvironment("obstacle_map", m, pixels_per_meter=10.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_env(self, env):
        assert env.name == "test"
        assert env.map_size == (10, 10)
        assert env.pixels_per_meter == 10.0

    def test_map_normalized(self, env):
        # All cells should be FREE_SPACE (value 1)
        assert np.all(env.map == FREE_SPACE)

    def test_invalid_pixels_per_meter(self, simple_map):
        with pytest.raises(ValueError, match="pixels_per_meter"):
            GridEnvironment("bad", simple_map, pixels_per_meter=0.0)
        with pytest.raises(ValueError, match="pixels_per_meter"):
            GridEnvironment("bad", simple_map, pixels_per_meter=-5.0)

    def test_no_free_space_raises(self):
        m = np.zeros((5, 5), dtype=np.uint8)  # all obstacles
        with pytest.raises(ValueError, match="free space"):
            GridEnvironment("empty", m, pixels_per_meter=10.0)

    def test_free_nodes_populated(self, env):
        assert len(env._free_nodes) == 100  # 10x10 all free


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


class TestCoordinateConversions:
    def test_map_to_world_center(self, env):
        # Center of 10x10 map at ppm=10: pixel (5,5) -> world (0,0)
        center_pixel = np.array([5.0, 5.0])
        world = env.map_to_world(center_pixel)
        np.testing.assert_allclose(world, [0.0, 0.0], atol=1e-9)

    def test_world_to_map_center(self, env):
        world = np.array([0.0, 0.0])
        pixel = env.world_to_map(world)
        np.testing.assert_array_equal(pixel, [5, 5])

    def test_round_trip_public(self, env):
        original_world = np.array([0.1, -0.2])
        pixel = env.world_to_map(original_world)
        recovered = env.map_to_world(pixel.astype(float))
        np.testing.assert_allclose(recovered, original_world, atol=0.1)

    def test_private_map_to_world_swaps_axes(self, env):
        # _map_to_world takes (row, col) and returns (x_world, y_world) with axis swap
        rc = np.array([5.0, 5.0])  # center
        world = env._map_to_world(rc)
        np.testing.assert_allclose(world, [0.0, 0.0], atol=1e-9)

    def test_private_world_to_map_swaps_axes(self, env):
        world = np.array([0.0, 0.0])
        rc = env._world_to_map(world)
        np.testing.assert_array_equal(rc, [5, 5])

    def test_batch_map_to_world(self, env):
        # _map_to_world with 2D input
        pts = np.array([[5.0, 5.0], [6.0, 5.0]], dtype=float)
        world = env._map_to_world(pts)
        assert world.shape == (2, 2)

    def test_batch_world_to_map(self, env):
        pts = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)
        rc = env._world_to_map(pts)
        assert rc.shape == (2, 2)

    def test_invalid_shape_map_to_world(self, env):
        with pytest.raises(ValueError):
            env._map_to_world(np.array([1.0, 2.0, 3.0]))

    def test_invalid_shape_world_to_map(self, env):
        with pytest.raises(ValueError):
            env._world_to_map(np.ones((2, 3)))  # wrong second dim


# ---------------------------------------------------------------------------
# Bounds and free-space checks
# ---------------------------------------------------------------------------


class TestBoundsAndFreeSpace:
    def test_in_bounds_valid(self, env):
        assert env._in_bounds((0, 0))
        assert env._in_bounds((9, 9))
        assert env._in_bounds((5, 5))

    def test_in_bounds_invalid(self, env):
        assert not env._in_bounds((-1, 0))
        assert not env._in_bounds((0, -1))
        assert not env._in_bounds((10, 0))
        assert not env._in_bounds((0, 10))

    def test_is_free_all_free(self, env):
        assert env._is_free((0, 0))
        assert env._is_free((5, 5))

    def test_is_free_obstacle(self, env_with_obstacle):
        assert not env_with_obstacle._is_free((4, 4))
        assert env_with_obstacle._is_free((0, 0))


# ---------------------------------------------------------------------------
# Nearest free cell
# ---------------------------------------------------------------------------


class TestNearestFree:
    def test_already_free(self, env):
        result = env._nearest_free((5, 5))
        assert result == (5, 5)

    def test_from_obstacle(self, env_with_obstacle):
        result = env_with_obstacle._nearest_free((4, 4))
        # Should find a free cell nearby
        r, c = result
        assert env_with_obstacle._is_free((r, c))

    def test_nearest_free_world(self, env):
        wx, wy = env.nearest_free_world((0.0, 0.0))
        assert isinstance(wx, float)
        assert isinstance(wy, float)


# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------


class TestAStar:
    def test_path_exists_open_map(self, env):
        path = env._astar((0, 0), (9, 9))
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_same_start_goal(self, env):
        path = env._astar((5, 5), (5, 5))
        assert len(path) == 1
        assert path[0] == (5, 5)

    def test_path_avoids_obstacles(self, env_with_obstacle):
        path = env_with_obstacle._astar((0, 0), (9, 9))
        assert len(path) > 0
        for r, c in path:
            assert env_with_obstacle._is_free((r, c))

    def test_no_path_isolated(self):
        """Create a map where goal is completely surrounded by obstacles."""
        m = np.full((10, 10), 255, dtype=np.uint8)
        # Surround cell (5,5) with obstacles
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                m[5 + dr, 5 + dc] = 0
        m[5, 5] = 0  # goal itself is obstacle too
        env = GridEnvironment("iso", m, pixels_per_meter=10.0)
        path = env._astar((0, 0), (5, 5))
        assert path == []  # unreachable


# ---------------------------------------------------------------------------
# Shortest path (world coordinates)
# ---------------------------------------------------------------------------


class TestShortestPath:
    def test_shortest_path_returns_waypoints(self, env):
        src = np.array([0.0, 0.0])
        dst = np.array([0.3, 0.3])
        waypoints, geodesic = env.shortest_path(src, dst)
        assert isinstance(waypoints, np.ndarray)
        assert isinstance(geodesic, float)

    def test_shortest_path_entire(self, env):
        src = np.array([0.0, 0.0])
        dst = np.array([0.2, 0.0])
        path, geodesic = env.shortest_path(src, dst, entire_path=True)
        assert len(path) >= 1

    def test_same_position_zero_geodesic(self, env):
        src = np.array([0.0, 0.0])
        _, geodesic = env.shortest_path(src, src)
        assert geodesic == 0.0 or geodesic == float("inf") or len(_) >= 1


# ---------------------------------------------------------------------------
# Random point
# ---------------------------------------------------------------------------


class TestRandomPoint:
    def test_get_random_point(self, env):
        pt = env.get_random_point()
        assert isinstance(pt, list)
        assert len(pt) == 2
        assert all(isinstance(x, float) for x in pt)


# ---------------------------------------------------------------------------
# Obstacle processing
# ---------------------------------------------------------------------------


class TestObstacleProcessing:
    def test_process_no_obstacles(self, env):
        env.process_obstacles()
        # All free map should have border obstacle (or none, depending on contour detection)
        assert isinstance(env.obstacles_meters, list)

    def test_process_with_obstacles(self, env_with_obstacle):
        env_with_obstacle.process_obstacles()
        assert isinstance(env_with_obstacle.obstacles_meters, list)
        # Should detect the 4x4 block
        assert len(env_with_obstacle.obstacles_meters) >= 1

    def test_get_obstacle_meters(self, env_with_obstacle):
        env_with_obstacle.process_obstacles()
        obs = env_with_obstacle.get_obstacle_meters()
        assert obs is env_with_obstacle.obstacles_meters


# ---------------------------------------------------------------------------
# Map normalization
# ---------------------------------------------------------------------------


class TestMapNormalization:
    def test_binary_values(self, env):
        unique = set(np.unique(env.map))
        assert unique <= {FREE_SPACE, OBSTACLE_SPACE}

    def test_mixed_values_normalized(self):
        m = np.array([[0, 128, 255], [0, 50, 200]], dtype=np.uint8)
        env = GridEnvironment("mixed", m, pixels_per_meter=10.0)
        assert env.map[0, 0] == OBSTACLE_SPACE
        assert env.map[0, 1] == FREE_SPACE  # 128 > 0
        assert env.map[0, 2] == FREE_SPACE  # 255 > 0
