from __future__ import annotations

import math
import types
from typing import Any

import numpy as np
import pytest

from navirl.ros.costmap import (
    FREE_SPACE,
    INSCRIBED_COST,
    LETHAL_COST,
    NO_INFORMATION,
    CostmapManager,
    InflationCostmapLayer,
    PredictiveCostmapLayer,
    SocialCostmapLayer,
    StaticCostmapLayer,
    _gaussian_kernel,
    _grid_to_world,
    _world_to_grid,
)

# =====================================================================
# Helper functions
# =====================================================================


class TestWorldToGrid:
    """Tests for _world_to_grid."""

    def test_origin_maps_to_zero(self) -> None:
        gx, gy = _world_to_grid(0.0, 0.0, 0.0, 0.0, 0.05)
        assert (gx, gy) == (0, 0)

    def test_positive_offset(self) -> None:
        gx, gy = _world_to_grid(1.0, 2.0, 0.0, 0.0, 0.05)
        assert gx == 20
        assert gy == 40

    def test_negative_origin(self) -> None:
        gx, gy = _world_to_grid(0.0, 0.0, -5.0, -5.0, 0.05)
        assert gx == 100
        assert gy == 100

    def test_returns_int(self) -> None:
        gx, gy = _world_to_grid(0.123, 0.456, 0.0, 0.0, 0.05)
        assert isinstance(gx, int)
        assert isinstance(gy, int)


class TestGridToWorld:
    """Tests for _grid_to_world."""

    def test_origin_maps_to_origin(self) -> None:
        wx, wy = _grid_to_world(0, 0, -5.0, -5.0, 0.05)
        assert wx == pytest.approx(-5.0)
        assert wy == pytest.approx(-5.0)

    def test_positive_cells(self) -> None:
        wx, wy = _grid_to_world(100, 100, -5.0, -5.0, 0.05)
        assert wx == pytest.approx(0.0)
        assert wy == pytest.approx(0.0)


class TestRoundTrip:
    """World -> grid -> world should return ~original coordinates."""

    @pytest.mark.parametrize(
        "wx, wy",
        [(0.0, 0.0), (1.5, -2.3), (-4.9, -4.9), (4.95, 4.95)],
    )
    def test_round_trip(self, wx: float, wy: float) -> None:
        origin_x, origin_y, res = -5.0, -5.0, 0.05
        gx, gy = _world_to_grid(wx, wy, origin_x, origin_y, res)
        wx2, wy2 = _grid_to_world(gx, gy, origin_x, origin_y, res)
        assert wx2 == pytest.approx(wx, abs=res)
        assert wy2 == pytest.approx(wy, abs=res)


class TestGaussianKernel:
    """Tests for _gaussian_kernel."""

    def test_sums_to_one(self) -> None:
        k = _gaussian_kernel(5, 1.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-12)

    def test_shape(self) -> None:
        k = _gaussian_kernel(7, 2.0)
        assert k.shape == (7, 7)

    def test_center_is_max(self) -> None:
        k = _gaussian_kernel(5, 1.0)
        center = k[2, 2]
        assert center == k.max()

    def test_symmetric(self) -> None:
        k = _gaussian_kernel(9, 3.0)
        np.testing.assert_allclose(k, k[::-1, :], atol=1e-15)
        np.testing.assert_allclose(k, k[:, ::-1], atol=1e-15)

    def test_all_positive(self) -> None:
        k = _gaussian_kernel(5, 1.0)
        assert (k > 0).all()


# =====================================================================
# Constants
# =====================================================================


class TestConstants:
    def test_values(self) -> None:
        assert FREE_SPACE == 0
        assert INSCRIBED_COST == 99
        assert LETHAL_COST == 100
        assert NO_INFORMATION == -1


# =====================================================================
# CostmapManager
# =====================================================================


class TestCostmapManagerInit:
    def test_default_dimensions(self) -> None:
        mgr = CostmapManager()
        assert mgr.width == 200
        assert mgr.height == 200
        assert mgr.resolution == 0.05
        assert mgr.origin_x == -5.0
        assert mgr.origin_y == -5.0

    def test_custom_dimensions(self) -> None:
        mgr = CostmapManager(width=50, height=60, resolution=0.1, origin=(0.0, 0.0))
        assert mgr.width == 50
        assert mgr.height == 60
        assert mgr.resolution == 0.1

    def test_master_starts_free(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        assert (mgr.master == FREE_SPACE).all()

    def test_master_returns_copy(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        m = mgr.master
        m[0, 0] = 50
        assert mgr.master[0, 0] == FREE_SPACE


class TestCostmapManagerLayers:
    def test_add_and_get_layer(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        layer = StaticCostmapLayer()
        mgr.add_layer("static", layer)
        assert mgr.get_layer("static") is layer

    def test_get_missing_layer_returns_none(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        assert mgr.get_layer("nonexistent") is None

    def test_remove_layer(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        layer = StaticCostmapLayer()
        mgr.add_layer("static", layer)
        mgr.remove_layer("static")
        assert mgr.get_layer("static") is None

    def test_remove_nonexistent_is_noop(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        mgr.remove_layer("nope")  # should not raise

    def test_add_layer_resizes(self) -> None:
        mgr = CostmapManager(width=30, height=40)
        layer = StaticCostmapLayer()
        mgr.add_layer("s", layer)
        assert layer.grid.shape == (40, 30)


class TestCostmapManagerUpdate:
    def test_update_empty_returns_free(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        result = mgr.update()
        assert (result == FREE_SPACE).all()

    def test_update_merges_via_max(self) -> None:
        mgr = CostmapManager(width=10, height=10)

        layer_a = StaticCostmapLayer()
        mgr.add_layer("a", layer_a)
        arr_a = np.zeros((10, 10), dtype=np.int8)
        arr_a[5, 5] = 50
        layer_a.load_from_array(arr_a)

        layer_b = StaticCostmapLayer()
        mgr.add_layer("b", layer_b)
        arr_b = np.zeros((10, 10), dtype=np.int8)
        arr_b[5, 5] = 80
        layer_b.load_from_array(arr_b)

        merged = mgr.update()
        assert merged[5, 5] == 80  # max of 50 and 80

    def test_update_returns_copy(self) -> None:
        mgr = CostmapManager(width=10, height=10)
        result = mgr.update()
        result[0, 0] = 99
        assert mgr.master[0, 0] == FREE_SPACE


class TestCostmapManagerConvenience:
    def test_world_to_grid(self) -> None:
        mgr = CostmapManager(width=200, height=200, resolution=0.05, origin=(-5.0, -5.0))
        gx, gy = mgr.world_to_grid(0.0, 0.0)
        assert gx == 100
        assert gy == 100

    def test_grid_to_world(self) -> None:
        mgr = CostmapManager(resolution=0.05, origin=(-5.0, -5.0))
        wx, wy = mgr.grid_to_world(100, 100)
        assert wx == pytest.approx(0.0)
        assert wy == pytest.approx(0.0)

    def test_in_bounds_true(self) -> None:
        mgr = CostmapManager(width=200, height=200)
        assert mgr.in_bounds(0, 0) is True
        assert mgr.in_bounds(199, 199) is True

    def test_in_bounds_false(self) -> None:
        mgr = CostmapManager(width=200, height=200)
        assert mgr.in_bounds(-1, 0) is False
        assert mgr.in_bounds(0, -1) is False
        assert mgr.in_bounds(200, 0) is False
        assert mgr.in_bounds(0, 200) is False

    def test_in_bounds_edge_cases(self) -> None:
        mgr = CostmapManager(width=1, height=1)
        assert mgr.in_bounds(0, 0) is True
        assert mgr.in_bounds(1, 0) is False


# =====================================================================
# OccupancyGrid conversion (dict fallback, no ROS2)
# =====================================================================


class TestOccupancyGridConversion:
    def test_to_occupancy_grid_dict_keys(self) -> None:
        mgr = CostmapManager(width=10, height=10, resolution=0.1, origin=(1.0, 2.0))
        d = mgr.to_occupancy_grid(frame_id="odom")
        # When ROS2 is unavailable the result is a plain dict
        if isinstance(d, dict):
            assert d["frame_id"] == "odom"
            assert d["resolution"] == 0.1
            assert d["width"] == 10
            assert d["height"] == 10
            assert d["origin"] == (1.0, 2.0)
            assert len(d["data"]) == 100

    def test_to_occupancy_grid_data_matches_master(self) -> None:
        mgr = CostmapManager(width=5, height=5)
        layer = StaticCostmapLayer()
        mgr.add_layer("s", layer)
        arr = np.zeros((5, 5), dtype=np.int8)
        arr[2, 3] = 42
        layer.load_from_array(arr)
        mgr.update()

        d = mgr.to_occupancy_grid()
        if isinstance(d, dict):
            assert d["data"][2 * 5 + 3] == 42

    def test_to_occupancy_grid_with_stamp(self) -> None:
        mgr = CostmapManager(width=5, height=5)
        result = mgr.to_occupancy_grid(frame_id="map", stamp=12345)
        # Should not raise regardless of stamp type
        assert result is not None

    def test_from_occupancy_grid_round_trip(self) -> None:
        """Build a msg-like object and reconstruct a CostmapManager."""
        original = CostmapManager(width=8, height=6, resolution=0.1, origin=(1.0, 2.0))
        layer = StaticCostmapLayer()
        original.add_layer("s", layer)
        arr = np.zeros((6, 8), dtype=np.int8)
        arr[3, 4] = 77
        layer.load_from_array(arr)
        original.update()

        # Build a fake ROS-like msg object
        msg = types.SimpleNamespace(
            info=types.SimpleNamespace(
                width=8,
                height=6,
                resolution=0.1,
                origin=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=1.0, y=2.0),
                ),
            ),
            data=original.master.ravel().tolist(),
        )

        restored = CostmapManager.from_occupancy_grid(msg)
        assert restored.width == 8
        assert restored.height == 6
        assert restored.resolution == pytest.approx(0.1)
        assert restored.origin_x == pytest.approx(1.0)
        assert restored.origin_y == pytest.approx(2.0)
        np.testing.assert_array_equal(restored.master, original.master)


# =====================================================================
# StaticCostmapLayer
# =====================================================================


class TestStaticCostmapLayer:
    def test_load_from_array(self) -> None:
        mgr = CostmapManager(width=5, height=5)
        layer = StaticCostmapLayer()
        mgr.add_layer("static", layer)

        arr = np.full((5, 5), 42, dtype=np.int8)
        layer.load_from_array(arr)
        np.testing.assert_array_equal(layer.grid, arr)

    def test_load_clips_values(self) -> None:
        layer = StaticCostmapLayer()
        arr = np.array([[200, -50]], dtype=np.int16)
        layer.load_from_array(arr)
        assert layer.grid[0, 0] == 100
        assert layer.grid[0, 1] == -1

    def test_update_is_noop(self) -> None:
        layer = StaticCostmapLayer()
        arr = np.array([[10, 20]], dtype=np.int8)
        layer.load_from_array(arr)
        layer.update()
        assert layer.grid[0, 0] == 10

    def test_clear(self) -> None:
        mgr = CostmapManager(width=5, height=5)
        layer = StaticCostmapLayer()
        mgr.add_layer("s", layer)
        layer.load_from_array(np.full((5, 5), 50, dtype=np.int8))
        layer.clear()
        assert (layer.grid == FREE_SPACE).all()

    def test_persistence_through_updates(self) -> None:
        """Static layer should keep its data across manager updates."""
        mgr = CostmapManager(width=5, height=5)
        layer = StaticCostmapLayer()
        mgr.add_layer("s", layer)
        arr = np.zeros((5, 5), dtype=np.int8)
        arr[1, 1] = 88
        layer.load_from_array(arr)

        mgr.update()
        mgr.update()
        assert mgr.master[1, 1] == 88


# =====================================================================
# SocialCostmapLayer
# =====================================================================


class TestSocialCostmapLayer:
    def _make_manager_with_social(
        self, width: int = 200, height: int = 200
    ) -> tuple[CostmapManager, SocialCostmapLayer]:
        mgr = CostmapManager(width=width, height=height, resolution=0.05, origin=(-5.0, -5.0))
        layer = SocialCostmapLayer(personal_radius=0.5, social_radius=2.0, front_scale=1.5)
        mgr.add_layer("social", layer)
        return mgr, layer

    def test_no_pedestrians_stays_free(self) -> None:
        mgr, layer = self._make_manager_with_social()
        mgr.update(pedestrians=None)
        assert (mgr.master == FREE_SPACE).all()

    def test_empty_pedestrians_stays_free(self) -> None:
        mgr, layer = self._make_manager_with_social()
        mgr.update(pedestrians=np.array([]))
        assert (mgr.master == FREE_SPACE).all()

    def test_single_pedestrian_creates_cost(self) -> None:
        mgr, layer = self._make_manager_with_social()
        # Pedestrian at world (0, 0), heading 0
        ped = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        mgr.update(pedestrians=ped)

        # Grid center for (0,0) with origin (-5,-5) and res 0.05 => (100,100)
        master = mgr.master
        assert master[100, 100] > FREE_SPACE

    def test_cost_near_pedestrian_is_high(self) -> None:
        mgr, layer = self._make_manager_with_social()
        ped = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        mgr.update(pedestrians=ped)
        master = mgr.master
        # The cell right at the pedestrian should be at inscribed cost
        assert master[100, 100] >= INSCRIBED_COST

    def test_cost_decays_with_distance(self) -> None:
        mgr, layer = self._make_manager_with_social()
        ped = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        mgr.update(pedestrians=ped)
        master = mgr.master
        # Far from pedestrian should be less
        assert master[100, 100] > master[100, 130]

    def test_multiple_pedestrians(self) -> None:
        mgr, layer = self._make_manager_with_social()
        peds = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0],
        ])
        mgr.update(pedestrians=peds)
        master = mgr.master
        # Both pedestrian locations should have cost
        assert master[100, 100] > FREE_SPACE  # (0,0) -> (100,100)
        gx2, gy2 = mgr.world_to_grid(2.0, 2.0)
        assert master[gy2, gx2] > FREE_SPACE

    def test_1d_pedestrian_reshaped(self) -> None:
        """A single pedestrian passed as 1D should still work."""
        mgr, layer = self._make_manager_with_social()
        ped = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        mgr.update(pedestrians=ped)
        assert mgr.master[100, 100] > FREE_SPACE

    def test_pedestrian_out_of_bounds(self) -> None:
        """Pedestrian far outside the grid should not crash."""
        mgr, layer = self._make_manager_with_social()
        ped = np.array([[100.0, 100.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        mgr.update(pedestrians=ped)  # should not raise
        assert (mgr.master == FREE_SPACE).all()


# =====================================================================
# PredictiveCostmapLayer
# =====================================================================


class TestPredictiveCostmapLayer:
    def _make_manager_with_predictive(
        self,
    ) -> tuple[CostmapManager, PredictiveCostmapLayer]:
        mgr = CostmapManager(width=200, height=200, resolution=0.05, origin=(-5.0, -5.0))
        layer = PredictiveCostmapLayer(decay_factor=0.85, prediction_radius=0.4)
        mgr.add_layer("predictive", layer)
        return mgr, layer

    def test_no_trajectories_stays_free(self) -> None:
        mgr, layer = self._make_manager_with_predictive()
        mgr.update(predicted_trajectories=None)
        assert (mgr.master == FREE_SPACE).all()

    def test_empty_trajectories_stays_free(self) -> None:
        mgr, layer = self._make_manager_with_predictive()
        mgr.update(predicted_trajectories=[])
        assert (mgr.master == FREE_SPACE).all()

    def test_single_trajectory_creates_cost(self) -> None:
        mgr, layer = self._make_manager_with_predictive()
        traj = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        mgr.update(predicted_trajectories=[traj])
        master = mgr.master
        # First point at grid (100, 100) should have cost
        assert master[100, 100] > FREE_SPACE

    def test_cost_decays_along_trajectory(self) -> None:
        mgr, layer = self._make_manager_with_predictive()
        # Trajectory along x-axis, spaced far enough to not overlap
        traj = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        mgr.update(predicted_trajectories=[traj])
        master = mgr.master

        gx0, gy0 = mgr.world_to_grid(0.0, 0.0)
        gx1, gy1 = mgr.world_to_grid(2.0, 0.0)
        gx2, gy2 = mgr.world_to_grid(4.0, 0.0)

        cost_first = master[gy0, gx0]
        cost_second = master[gy1, gx1]
        cost_third = master[gy2, gx2]

        assert cost_first > FREE_SPACE
        assert cost_second > FREE_SPACE
        # Due to decay, later steps should have equal or lower cost at center
        assert cost_first >= cost_second
        assert cost_second >= cost_third

    def test_multiple_trajectories(self) -> None:
        mgr, layer = self._make_manager_with_predictive()
        traj1 = np.array([[0.0, 0.0], [0.5, 0.0]])
        traj2 = np.array([[0.0, 2.0], [0.5, 2.0]])
        mgr.update(predicted_trajectories=[traj1, traj2])
        master = mgr.master

        gx1, gy1 = mgr.world_to_grid(0.0, 0.0)
        gx2, gy2 = mgr.world_to_grid(0.0, 2.0)
        assert master[gy1, gx1] > FREE_SPACE
        assert master[gy2, gx2] > FREE_SPACE

    def test_bad_trajectory_shape_skipped(self) -> None:
        """A 1D trajectory should be skipped without error."""
        mgr, layer = self._make_manager_with_predictive()
        bad_traj = np.array([0.0, 1.0])
        mgr.update(predicted_trajectories=[bad_traj])
        assert (mgr.master == FREE_SPACE).all()

    def test_trajectory_out_of_bounds(self) -> None:
        """Trajectory points outside the grid should not crash."""
        mgr, layer = self._make_manager_with_predictive()
        traj = np.array([[100.0, 100.0], [200.0, 200.0]])
        mgr.update(predicted_trajectories=[traj])  # should not raise


# =====================================================================
# InflationCostmapLayer (requires scipy)
# =====================================================================


class TestInflationCostmapLayer:
    @pytest.fixture(autouse=True)
    def _require_scipy(self) -> None:
        pytest.importorskip("scipy")

    def _make_manager_with_inflation(
        self,
    ) -> tuple[CostmapManager, InflationCostmapLayer]:
        mgr = CostmapManager(width=50, height=50, resolution=0.05, origin=(0.0, 0.0))
        layer = InflationCostmapLayer(inflation_radius=0.3)
        mgr.add_layer("inflation", layer)
        return mgr, layer

    def test_no_source_is_noop(self) -> None:
        mgr, layer = self._make_manager_with_inflation()
        mgr.update()
        assert (mgr.master == FREE_SPACE).all()

    def test_set_source_inflates(self) -> None:
        mgr, layer = self._make_manager_with_inflation()
        src = np.zeros((50, 50), dtype=np.int8)
        src[25, 25] = LETHAL_COST
        layer.set_source(src)
        mgr.update()

        master = mgr.master
        # The lethal cell should remain lethal
        assert master[25, 25] == LETHAL_COST
        # Nearby cells should have been inflated (non-zero)
        assert master[25, 26] > FREE_SPACE
        assert master[26, 25] > FREE_SPACE

    def test_inflation_does_not_exceed_inscribed(self) -> None:
        """Non-lethal inflated cells should not exceed INSCRIBED_COST."""
        mgr, layer = self._make_manager_with_inflation()
        src = np.zeros((50, 50), dtype=np.int8)
        src[25, 25] = LETHAL_COST
        layer.set_source(src)
        mgr.update()

        master = mgr.master
        non_lethal = master[master < LETHAL_COST]
        assert (non_lethal <= INSCRIBED_COST).all()

    def test_obstacles_kwarg(self) -> None:
        """Passing obstacles via kwargs should also work."""
        mgr, layer = self._make_manager_with_inflation()
        src = np.zeros((50, 50), dtype=np.int8)
        src[10, 10] = LETHAL_COST
        mgr.update(obstacles=src)
        assert mgr.master[10, 10] == LETHAL_COST


# =====================================================================
# _CostmapLayerBase
# =====================================================================


class TestCostmapLayerBase:
    def test_resize(self) -> None:
        layer = StaticCostmapLayer()  # concrete subclass
        layer.resize(30, 40, 0.1, 1.0, 2.0)
        assert layer.grid.shape == (40, 30)

    def test_clear_resets_grid(self) -> None:
        layer = StaticCostmapLayer()
        layer.resize(5, 5, 0.05, 0.0, 0.0)
        layer.load_from_array(np.full((5, 5), 50, dtype=np.int8))
        layer.clear()
        assert (layer.grid == FREE_SPACE).all()

    def test_grid_property(self) -> None:
        layer = StaticCostmapLayer()
        assert isinstance(layer.grid, np.ndarray)


# =====================================================================
# Integration: multiple layers combined
# =====================================================================


class TestMultiLayerIntegration:
    def test_static_plus_social(self) -> None:
        mgr = CostmapManager(width=200, height=200, resolution=0.05, origin=(-5.0, -5.0))

        static = StaticCostmapLayer()
        mgr.add_layer("static", static)
        arr = np.zeros((200, 200), dtype=np.int8)
        arr[50, 50] = 80
        static.load_from_array(arr)

        social = SocialCostmapLayer()
        mgr.add_layer("social", social)

        ped = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        merged = mgr.update(pedestrians=ped)

        # Static obstacle should be present
        assert merged[50, 50] == 80
        # Social cost should be present near pedestrian
        assert merged[100, 100] > FREE_SPACE

    def test_all_layers_cleared_on_update(self) -> None:
        """Social layer should clear before re-stamping on each update."""
        mgr = CostmapManager(width=200, height=200, resolution=0.05, origin=(-5.0, -5.0))
        social = SocialCostmapLayer()
        mgr.add_layer("social", social)

        ped = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
        mgr.update(pedestrians=ped)

        # Update with no pedestrians should clear social cost
        mgr.update(pedestrians=np.array([]))
        second = mgr.master
        assert (second == FREE_SPACE).all()
