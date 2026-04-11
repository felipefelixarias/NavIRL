"""Tests for navirl.maps.grid_map module."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# The navirl.maps package __init__ imports map_utils which may not exist;
# load the submodule directly from its file path to avoid the cascade.
_spec = importlib.util.spec_from_file_location(
    "navirl.maps.grid_map",
    Path(__file__).resolve().parent.parent / "navirl" / "maps" / "grid_map.py",
)
_grid_map = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _grid_map
_spec.loader.exec_module(_grid_map)
GridMap = _grid_map.GridMap
FREE = _grid_map.FREE
OCCUPIED = _grid_map.OCCUPIED
UNKNOWN = _grid_map.UNKNOWN


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestGridMapConstruction:
    def test_default_values(self):
        gm = GridMap()
        assert gm.width == 100
        assert gm.height == 100
        assert gm.resolution == 0.1
        assert np.array_equal(gm.origin, [0.0, 0.0])
        assert gm.data.shape == (100, 100)
        assert np.all(gm.data == FREE)

    def test_custom_values(self):
        gm = GridMap(50, 30, 0.05, (1.0, 2.0), OCCUPIED)
        assert gm.width == 50
        assert gm.height == 30
        assert gm.resolution == 0.05
        assert np.allclose(gm.origin, [1.0, 2.0])
        assert np.all(gm.data == OCCUPIED)

    def test_world_dimensions(self):
        gm = GridMap(200, 100, 0.05)
        assert gm.world_width == pytest.approx(10.0)
        assert gm.world_height == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


class TestCoordinateTransforms:
    def test_world_to_grid_origin(self):
        gm = GridMap(100, 100, 0.1)
        r, c = gm.world_to_grid(0.0, 0.0)
        assert r == 0
        assert c == 0

    def test_world_to_grid_interior(self):
        gm = GridMap(100, 100, 0.1)
        r, c = gm.world_to_grid(5.0, 3.0)
        assert c == 50
        assert r == 30

    def test_world_to_grid_clips_negative(self):
        gm = GridMap(100, 100, 0.1)
        r, c = gm.world_to_grid(-10.0, -10.0)
        assert r == 0
        assert c == 0

    def test_world_to_grid_clips_large(self):
        gm = GridMap(100, 100, 0.1)
        r, c = gm.world_to_grid(100.0, 100.0)
        assert r == 99
        assert c == 99

    def test_grid_to_world_center(self):
        gm = GridMap(100, 100, 0.1)
        x, y = gm.grid_to_world(0, 0)
        assert x == pytest.approx(0.05)
        assert y == pytest.approx(0.05)

    def test_world_to_grid_float(self):
        gm = GridMap(100, 100, 0.1)
        r, c = gm.world_to_grid_float(5.0, 3.0)
        assert c == pytest.approx(50.0)
        assert r == pytest.approx(30.0)

    def test_roundtrip_transform(self):
        gm = GridMap(100, 100, 0.1)
        x_orig, y_orig = 3.25, 7.85
        r, c = gm.world_to_grid(x_orig, y_orig)
        x_back, y_back = gm.grid_to_world(r, c)
        # Should be within one cell of original
        assert abs(x_back - x_orig) < gm.resolution
        assert abs(y_back - y_orig) < gm.resolution

    def test_world_to_grid_with_offset_origin(self):
        gm = GridMap(100, 100, 0.1, (-5.0, -5.0))
        r, c = gm.world_to_grid(0.0, 0.0)
        assert c == 50
        assert r == 50


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------


class TestBoundsChecking:
    def test_in_bounds_valid(self):
        gm = GridMap(10, 10)
        assert gm.in_bounds(0, 0)
        assert gm.in_bounds(9, 9)
        assert gm.in_bounds(5, 5)

    def test_in_bounds_invalid(self):
        gm = GridMap(10, 10)
        assert not gm.in_bounds(-1, 0)
        assert not gm.in_bounds(0, -1)
        assert not gm.in_bounds(10, 0)
        assert not gm.in_bounds(0, 10)


# ---------------------------------------------------------------------------
# Cell access
# ---------------------------------------------------------------------------


class TestCellAccess:
    def test_get_set(self):
        gm = GridMap(10, 10)
        gm.set(5, 5, OCCUPIED)
        assert gm.get(5, 5) == OCCUPIED
        assert gm.is_occupied(5, 5)
        assert not gm.is_free(5, 5)

    def test_get_out_of_bounds_returns_unknown(self):
        gm = GridMap(10, 10)
        assert gm.get(-1, 0) == UNKNOWN
        assert gm.get(100, 100) == UNKNOWN

    def test_set_out_of_bounds_no_op(self):
        gm = GridMap(10, 10)
        gm.set(-1, 0, OCCUPIED)  # Should not crash

    def test_set_world_get_world(self):
        gm = GridMap(100, 100, 0.1)
        gm.set_world(3.0, 4.0, OCCUPIED)
        assert gm.get_world(3.0, 4.0) == OCCUPIED

    def test_is_free_default(self):
        gm = GridMap(10, 10)
        assert gm.is_free(0, 0)


# ---------------------------------------------------------------------------
# Bresenham line
# ---------------------------------------------------------------------------


class TestBresenham:
    def test_horizontal_line(self):
        cells = GridMap.bresenham(0, 0, 0, 5)
        assert len(cells) == 6
        for r, _c in cells:
            assert r == 0

    def test_vertical_line(self):
        cells = GridMap.bresenham(0, 0, 5, 0)
        assert len(cells) == 6
        for _r, c in cells:
            assert c == 0

    def test_diagonal_line(self):
        cells = GridMap.bresenham(0, 0, 5, 5)
        assert (0, 0) in cells
        assert (5, 5) in cells
        assert len(cells) >= 6

    def test_single_point(self):
        cells = GridMap.bresenham(3, 3, 3, 3)
        assert cells == [(3, 3)]

    def test_line_cells_world(self):
        gm = GridMap(100, 100, 0.1)
        cells = gm.line_cells(0.0, 0.0, 1.0, 0.0)
        assert len(cells) > 1
        # All on the same row
        rows = {r for r, c in cells}
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------


class TestRayCasting:
    def test_ray_hits_wall(self):
        gm = GridMap(100, 100, 0.1)
        # Place a wall at column 50 (x = 5.0)
        for r in range(100):
            gm.set(r, 50, OCCUPIED)

        dist, (hr, hc) = gm.ray_cast(0.0, 5.0, 0.0, max_range=10.0)  # East
        assert dist < 10.0
        assert hc == 50

    def test_ray_misses(self):
        gm = GridMap(100, 100, 0.1)
        dist, hit = gm.ray_cast(5.0, 5.0, 0.0, max_range=2.0)
        assert dist == 2.0
        assert hit == (-1, -1)

    def test_ray_cast_fan(self):
        gm = GridMap(100, 100, 0.1)
        results = gm.ray_cast_fan(5.0, 5.0, 0.0, math.pi, n_rays=5, max_range=3.0)
        assert len(results) == 5
        for angle, _dist, _hit_x in results:
            assert 0.0 <= angle <= math.pi + 1e-6


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------


class TestFloodFill:
    def test_basic_fill(self):
        gm = GridMap(10, 10)
        filled = gm.flood_fill(0, 0, OCCUPIED)
        assert filled == 100  # Entire 10x10 grid
        assert np.all(gm.data == OCCUPIED)

    def test_fill_blocked_by_wall(self):
        gm = GridMap(10, 10)
        # Create a wall dividing the grid
        for r in range(10):
            gm.set(r, 5, OCCUPIED)
        filled = gm.flood_fill(0, 0, 2)
        # Should fill left half minus the wall
        assert filled == 50

    def test_fill_same_value_no_op(self):
        gm = GridMap(10, 10)
        filled = gm.flood_fill(0, 0, FREE)
        assert filled == 0

    def test_fill_out_of_bounds(self):
        gm = GridMap(10, 10)
        filled = gm.flood_fill(-1, -1, OCCUPIED)
        assert filled == 0

    def test_connected_component(self):
        gm = GridMap(10, 10)
        for r in range(10):
            gm.set(r, 5, OCCUPIED)
        mask = gm.connected_component(0, 0)
        assert mask[0, 0]
        assert mask[0, 4]
        assert not mask[0, 5]  # Wall
        assert not mask[0, 6]  # Other side

    def test_connected_component_out_of_bounds(self):
        gm = GridMap(10, 10)
        mask = gm.connected_component(-1, -1)
        assert not mask.any()


# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------


class TestDistanceTransform:
    def test_no_obstacles(self):
        gm = GridMap(5, 5)
        dt = gm.distance_transform()
        assert np.all(dt == np.inf)

    def test_single_obstacle(self):
        gm = GridMap(5, 5)
        gm.set(2, 2, OCCUPIED)
        dt = gm.distance_transform()
        assert dt[2, 2] == 0.0
        assert dt[2, 3] == 1.0
        assert dt[3, 2] == 1.0
        assert dt[0, 2] == 2.0

    def test_world_distance_transform(self):
        gm = GridMap(5, 5, 0.5)
        gm.set(2, 2, OCCUPIED)
        dt = gm.distance_transform_world()
        assert dt[2, 2] == 0.0
        assert dt[2, 3] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------


class TestInflation:
    def test_inflate_single_cell(self):
        gm = GridMap(20, 20, 0.1)
        gm.set(10, 10, OCCUPIED)
        inflated = gm.inflate(0.2)  # 2 cells radius
        assert inflated.is_occupied(10, 10)
        assert inflated.is_occupied(10, 11)
        assert inflated.is_occupied(11, 10)
        assert inflated.count_occupied() > 1

    def test_inflate_preserves_original(self):
        gm = GridMap(20, 20, 0.1)
        gm.set(10, 10, OCCUPIED)
        original_count = gm.count_occupied()
        _ = gm.inflate(0.2)
        assert gm.count_occupied() == original_count

    def test_inflate_inplace(self):
        gm = GridMap(20, 20, 0.1)
        gm.set(10, 10, OCCUPIED)
        gm.inflate_inplace(0.2)
        assert gm.count_occupied() > 1


# ---------------------------------------------------------------------------
# Map merging
# ---------------------------------------------------------------------------


class TestMapMerging:
    def test_merge_overwrite(self):
        gm1 = GridMap(10, 10)
        gm2 = GridMap(10, 10, default_value=UNKNOWN)
        gm2.set(5, 5, OCCUPIED)
        gm1.merge(gm2, mode="overwrite")
        assert gm1.is_occupied(5, 5)
        # Unknown cells shouldn't overwrite
        assert gm1.is_free(0, 0)

    def test_merge_max(self):
        gm1 = GridMap(10, 10)
        gm2 = GridMap(10, 10)
        gm2.set(5, 5, OCCUPIED)
        gm1.merge(gm2, mode="max")
        assert gm1.is_occupied(5, 5)

    def test_merge_min(self):
        gm1 = GridMap(10, 10, default_value=OCCUPIED)
        gm2 = GridMap(10, 10)
        gm1.merge(gm2, mode="min")
        assert gm1.is_free(0, 0)


# ---------------------------------------------------------------------------
# Submap
# ---------------------------------------------------------------------------


class TestSubmap:
    def test_basic_submap(self):
        gm = GridMap(20, 20, 0.1)
        gm.set(5, 5, OCCUPIED)
        sub = gm.submap(0, 0, 10, 10)
        assert sub.width == 10
        assert sub.height == 10
        assert sub.is_occupied(5, 5)

    def test_submap_empty(self):
        gm = GridMap(20, 20, 0.1)
        sub = gm.submap(5, 5, 3, 3)  # end < start
        assert sub.width == 0
        assert sub.height == 0

    def test_submap_world(self):
        gm = GridMap(100, 100, 0.1)
        gm.set_world(3.0, 4.0, OCCUPIED)
        sub = gm.submap_world(2.0, 3.0, 4.0, 5.0)
        assert sub.width > 0
        assert sub.height > 0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class TestDrawing:
    def test_draw_line(self):
        gm = GridMap(100, 100, 0.1)
        gm.draw_line(0.0, 0.0, 5.0, 0.0)
        assert gm.count_occupied() > 0

    def test_draw_rect_filled(self):
        gm = GridMap(100, 100, 0.1)
        gm.draw_rect(1.0, 1.0, 2.0, 2.0, filled=True)
        assert gm.count_occupied() > 0

    def test_draw_rect_outline(self):
        gm = GridMap(100, 100, 0.1)
        gm.draw_rect(1.0, 1.0, 2.0, 2.0, filled=False)
        assert gm.count_occupied() > 0

    def test_draw_circle_filled(self):
        gm = GridMap(100, 100, 0.1)
        gm.draw_circle(5.0, 5.0, 1.0, filled=True)
        occ = gm.count_occupied()
        assert occ > 0
        # Approximate area check: pi * r^2 / resolution^2
        expected_cells = math.pi * (1.0 / 0.1) ** 2
        assert abs(occ - expected_cells) / expected_cells < 0.3

    def test_draw_circle_outline(self):
        gm = GridMap(100, 100, 0.1)
        gm.draw_circle(5.0, 5.0, 1.0, filled=False)
        assert gm.count_occupied() > 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_counts_default(self):
        gm = GridMap(10, 10)
        assert gm.count_free() == 100
        assert gm.count_occupied() == 0
        assert gm.count_unknown() == 0

    def test_occupancy_ratio(self):
        gm = GridMap(10, 10)
        for c in range(5):
            gm.set(0, c, OCCUPIED)
        ratio = gm.occupancy_ratio()
        assert ratio == pytest.approx(5.0 / 100.0)

    def test_occupancy_ratio_all_unknown(self):
        gm = GridMap(10, 10, default_value=UNKNOWN)
        assert gm.occupancy_ratio() == 0.0


# ---------------------------------------------------------------------------
# Copy, clear, interop
# ---------------------------------------------------------------------------


class TestCopyAndInterop:
    def test_copy_is_independent(self):
        gm = GridMap(10, 10)
        gm.set(0, 0, OCCUPIED)
        gm2 = gm.copy()
        gm2.set(0, 0, FREE)
        assert gm.is_occupied(0, 0)
        assert gm2.is_free(0, 0)

    def test_clear(self):
        gm = GridMap(10, 10)
        gm.set(5, 5, OCCUPIED)
        gm.clear()
        assert np.all(gm.data == FREE)

    def test_clear_with_value(self):
        gm = GridMap(10, 10)
        gm.clear(UNKNOWN)
        assert np.all(gm.data == UNKNOWN)

    def test_as_binary(self):
        gm = GridMap(10, 10)
        gm.set(5, 5, OCCUPIED)
        binary = gm.as_binary()
        assert binary[5, 5]
        assert not binary[0, 0]

    def test_as_float(self):
        gm = GridMap(10, 10)
        gm.set(0, 0, OCCUPIED)
        gm.set(1, 1, UNKNOWN)
        flt = gm.as_float()
        assert flt[0, 0] == pytest.approx(1.0)
        assert flt[1, 1] == pytest.approx(0.5)
        assert flt[2, 2] == pytest.approx(0.0)

    def test_from_array(self):
        arr = np.zeros((10, 10))
        arr[5, 5] = 1.0
        gm = GridMap.from_array(arr, resolution=0.2)
        assert gm.width == 10
        assert gm.height == 10
        assert gm.is_occupied(5, 5)
        assert gm.is_free(0, 0)

    def test_repr(self):
        gm = GridMap(10, 10)
        s = repr(gm)
        assert "GridMap" in s
        assert "10x10" in s
