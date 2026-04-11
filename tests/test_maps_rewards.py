"""Tests for navirl.maps.grid_map, navirl.rewards.base, and navirl.rewards.navigation."""

from __future__ import annotations

import importlib
import importlib.util
import math
import pathlib as _pathlib
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: navirl.maps and navirl.rewards __init__.py files import
# submodules that may not exist in this checkout. We register stub packages
# and load the specific modules we need directly from their file paths.
# ---------------------------------------------------------------------------

_root = _pathlib.Path(__file__).resolve().parent.parent / "navirl"


def _load_module(fqn: str, filepath: _pathlib.Path) -> types.ModuleType:
    """Load a single Python file as *fqn* into sys.modules."""
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _ensure_stub_package(fqn: str, path: _pathlib.Path) -> None:
    """Register a lightweight stub package so child modules can be imported."""
    if fqn not in sys.modules:
        stub = types.ModuleType(fqn)
        stub.__path__ = [str(path)]  # type: ignore[attr-defined]
        stub.__package__ = fqn
        sys.modules[fqn] = stub


_ensure_stub_package("navirl.maps", _root / "maps")
_ensure_stub_package("navirl.rewards", _root / "rewards")

_grid_map = _load_module("navirl.maps.grid_map", _root / "maps" / "grid_map.py")
_rewards_base = _load_module("navirl.rewards.base", _root / "rewards" / "base.py")
_rewards_nav = _load_module("navirl.rewards.navigation", _root / "rewards" / "navigation.py")

GridMap = _grid_map.GridMap
FREE = _grid_map.FREE
OCCUPIED = _grid_map.OCCUPIED
UNKNOWN = _grid_map.UNKNOWN

CompositeReward = _rewards_base.CompositeReward
RewardClipper = _rewards_base.RewardClipper
RewardComponent = _rewards_base.RewardComponent
RewardFunction = _rewards_base.RewardFunction
RewardNormalizer = _rewards_base.RewardNormalizer
RewardShaper = _rewards_base.RewardShaper

BoundaryPenalty = _rewards_nav.BoundaryPenalty
CollisionPenalty = _rewards_nav.CollisionPenalty
GoalReward = _rewards_nav.GoalReward
PathFollowingReward = _rewards_nav.PathFollowingReward
ProgressReward = _rewards_nav.ProgressReward
SmoothnessReward = _rewards_nav.SmoothnessReward
TimePenaltyReward = _rewards_nav.TimePenaltyReward
VelocityReward = _rewards_nav.VelocityReward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConstantReward(RewardFunction):
    """Simple concrete reward for testing base wrappers."""

    def __init__(self, value: float = 1.0, name: str | None = None) -> None:
        super().__init__(name=name or "ConstantReward")
        self._value = value

    def compute(self, state, action, next_state, *, info=None) -> float:
        return self._value


def _make_state(
    position=(0.0, 0.0),
    goal=(5.0, 5.0),
    velocity=(0.0, 0.0),
    heading=0.0,
    dt=0.1,
    **kwargs,
):
    s = {
        "position": np.array(position, dtype=np.float64),
        "goal": np.array(goal, dtype=np.float64),
        "velocity": np.array(velocity, dtype=np.float64),
        "heading": heading,
        "dt": dt,
    }
    s.update(kwargs)
    return s


# ===================================================================
# GridMap tests (18 tests)
# ===================================================================


class TestGridMapConstruction:
    def test_default_and_custom_construction(self):
        gm = GridMap()
        assert gm.width == 100 and gm.height == 100
        assert gm.resolution == 0.1
        assert np.all(gm.data == FREE)
        gm2 = GridMap(50, 30, 0.5, (1.0, 2.0), default_value=UNKNOWN)
        assert gm2.width == 50 and gm2.height == 30
        assert np.all(gm2.data == UNKNOWN)

    def test_from_array(self):
        arr = np.array([[0.0, 0.8], [0.3, 1.0]])
        gm = GridMap.from_array(arr, resolution=0.5, threshold=0.5)
        assert gm.get(0, 0) == FREE
        assert gm.get(0, 1) == OCCUPIED
        assert gm.get(1, 1) == OCCUPIED


class TestGridMapCoordinates:
    def test_world_to_grid_and_back(self):
        gm = GridMap(10, 10, 1.0, (0.0, 0.0))
        row, col = gm.world_to_grid(2.5, 3.5)
        assert col == 2 and row == 3
        x, y = gm.grid_to_world(row, col)
        assert x == pytest.approx(2.5) and y == pytest.approx(3.5)
        # Clipping and bounds
        row2, col2 = gm.world_to_grid(-5.0, 100.0)
        assert col2 == 0 and row2 == 9
        assert gm.in_bounds(0, 0) and gm.in_bounds(9, 9)
        assert not gm.in_bounds(-1, 0) and not gm.in_bounds(10, 0)


class TestGridMapCellAccess:
    def test_get_set_and_out_of_bounds(self):
        gm = GridMap(5, 5, 1.0)
        gm.set(2, 2, OCCUPIED)
        assert gm.get(2, 2) == OCCUPIED
        assert gm.get(-1, 0) == UNKNOWN  # Out of bounds
        gm.set(100, 100, OCCUPIED)  # Noop, should not raise
        # is_free / is_occupied
        assert gm.is_free(0, 0) and not gm.is_occupied(0, 0)
        gm.set(0, 0, OCCUPIED)
        assert gm.is_occupied(0, 0) and not gm.is_free(0, 0)

    def test_set_world_get_world(self):
        gm = GridMap(10, 10, 1.0)
        gm.set_world(3.5, 4.5, OCCUPIED)
        assert gm.get_world(3.5, 4.5) == OCCUPIED


class TestGridMapBresenhamAndRayCast:
    def test_bresenham(self):
        assert GridMap.bresenham(3, 3, 3, 3) == [(3, 3)]
        cells = GridMap.bresenham(0, 0, 0, 5)
        assert len(cells) == 6 and cells[-1] == (0, 5)

    def test_ray_cast_no_hit(self):
        gm = GridMap(10, 10, 1.0)
        dist, hit = gm.ray_cast(5.5, 5.5, 0.0, max_range=3.0)
        assert dist == 3.0 and hit == (-1, -1)

    def test_ray_cast_hit(self):
        gm = GridMap(20, 20, 1.0)
        gm.set(0, 15, OCCUPIED)
        dist, hit = gm.ray_cast(10.5, 0.5, 0.0, max_range=30.0)
        assert hit == (0, 15) and dist < 30.0


class TestGridMapFloodAndDistance:
    def test_flood_fill(self):
        gm = GridMap(5, 5, 1.0)
        assert gm.flood_fill(0, 0, OCCUPIED) == 25
        gm2 = GridMap(5, 5, 1.0)
        assert gm2.flood_fill(-1, -1, OCCUPIED) == 0  # Out of bounds
        assert gm2.flood_fill(0, 0, FREE) == 0  # Same value

    def test_connected_component_with_barrier(self):
        gm = GridMap(5, 5, 1.0)
        for r in range(5):
            gm.set(r, 2, OCCUPIED)
        mask = gm.connected_component(0, 0)
        assert mask[0, 0] and mask[0, 1] and not mask[0, 3]

    def test_distance_transform(self):
        gm = GridMap(5, 5, 1.0)
        gm.set(2, 2, OCCUPIED)
        dt = gm.distance_transform()
        assert dt[2, 2] == 0.0 and dt[2, 3] == 1.0
        # World scaling
        gm2 = GridMap(5, 5, 0.5)
        gm2.set(2, 2, OCCUPIED)
        assert gm2.distance_transform_world()[2, 3] == pytest.approx(0.5)


class TestGridMapOperations:
    def test_inflate(self):
        gm = GridMap(10, 10, 1.0)
        gm.set(5, 5, OCCUPIED)
        inflated = gm.inflate(2.0)
        assert inflated.count_occupied() > 1
        assert gm.count_occupied() == 1  # Original unchanged

    def test_merge_overwrite(self):
        gm1 = GridMap(5, 5, 1.0)
        gm2 = GridMap(5, 5, 1.0)
        gm2.set(0, 0, OCCUPIED)
        gm1.merge(gm2, mode="overwrite")
        assert gm1.get(0, 0) == OCCUPIED

    def test_submap(self):
        gm = GridMap(10, 10, 1.0)
        gm.set(3, 3, OCCUPIED)
        sub = gm.submap(2, 2, 5, 5)
        assert sub.width == 3 and sub.height == 3
        assert sub.get(1, 1) == OCCUPIED

    def test_statistics_and_properties(self):
        gm = GridMap(10, 10, 0.5)
        assert gm.count_free() == 100
        gm.set(0, 0, OCCUPIED)
        assert gm.occupancy_ratio() == pytest.approx(1 / 100)
        assert gm.world_width == pytest.approx(5.0)
        assert gm.world_height == pytest.approx(5.0)

    def test_copy_and_clear(self):
        gm = GridMap(5, 5, 1.0)
        gm.set(0, 0, OCCUPIED)
        copy = gm.copy()
        copy.set(0, 0, FREE)
        assert gm.get(0, 0) == OCCUPIED  # Independent
        gm.clear(UNKNOWN)
        assert gm.count_unknown() == 25

    def test_as_binary_and_as_float(self):
        gm = GridMap(3, 3, 1.0)
        gm.set(0, 0, OCCUPIED)
        gm.set(1, 1, UNKNOWN)
        assert gm.as_binary()[0, 0] and not gm.as_binary()[2, 2]
        f = gm.as_float()
        assert f[0, 0] == pytest.approx(1.0) and f[1, 1] == pytest.approx(0.5)


# ===================================================================
# rewards/base tests (16 tests)
# ===================================================================


class TestRewardFunctionBase:
    def test_compute_and_call(self):
        r = ConstantReward(3.0)
        assert r.compute({}, None, {}) == 3.0
        assert r({}, None, {}) == 3.0  # __call__

    def test_name_defaults_and_custom(self):
        assert ConstantReward().name == "ConstantReward"
        assert ConstantReward(name="custom").name == "custom"

    def test_reset_and_get_info(self):
        r = ConstantReward()
        assert r.reset() is None
        assert r.get_info() == {}


class TestRewardComponent:
    def test_weighted_and_disabled(self):
        comp = RewardComponent(reward_fn=ConstantReward(5.0), weight=0.5)
        assert comp.compute({}, None, {}) == pytest.approx(2.5)
        comp.enabled = False
        assert comp.compute({}, None, {}) == 0.0

    def test_name_and_info(self):
        comp = RewardComponent(reward_fn=ConstantReward(name="foo"), weight=2.0)
        assert comp.name == "foo"
        info = comp.get_info()
        assert info["weight"] == 2.0 and info["enabled"] is True


class TestCompositeReward:
    def test_sum_of_components(self):
        c1 = RewardComponent(ConstantReward(1.0), weight=2.0)
        c2 = RewardComponent(ConstantReward(3.0), weight=1.0)
        cr = CompositeReward([c1, c2])
        assert cr.compute({}, None, {}) == pytest.approx(5.0)

    def test_empty_composite(self):
        cr = CompositeReward()
        assert cr.compute({}, None, {}) == 0.0 and len(cr) == 0

    def test_add_remove_get(self):
        cr = CompositeReward()
        comp = RewardComponent(ConstantReward(1.0, name="a"))
        cr.add_component(comp)
        assert cr.get_component("a") is comp
        assert cr.remove_component("a") is comp
        assert cr.remove_component("missing") is None

    def test_set_weight_and_enable_disable(self):
        comp = RewardComponent(ConstantReward(1.0, name="x"))
        cr = CompositeReward([comp])
        cr.set_weight("x", 5.0)
        assert comp.weight == 5.0
        with pytest.raises(KeyError):
            cr.set_weight("missing", 1.0)
        cr.disable("x")
        assert cr.compute({}, None, {}) == 0.0
        cr.enable("x")
        assert cr.compute({}, None, {}) == pytest.approx(5.0)

    def test_filter_by_tag(self):
        c1 = RewardComponent(ConstantReward(name="a"), tags=["social"])
        c2 = RewardComponent(ConstantReward(name="b"), tags=["nav"])
        cr = CompositeReward([c1, c2])
        assert len(cr.filter_by_tag("social")) == 1

    def test_decomposition_info(self):
        c1 = RewardComponent(ConstantReward(2.0, name="a"))
        cr = CompositeReward([c1], track_decomposition=True)
        cr.compute({}, None, {})
        assert cr.get_info()["decomposition"]["a"] == pytest.approx(2.0)


class TestRewardClipper:
    def test_clips_and_passthrough(self):
        clipper = RewardClipper(ConstantReward(100.0), low=-5.0, high=5.0)
        assert clipper.compute({}, None, {}) == 5.0
        clipper2 = RewardClipper(ConstantReward(3.0), low=-5.0, high=5.0)
        assert clipper2.compute({}, None, {}) == 3.0

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            RewardClipper(ConstantReward(), low=5.0, high=5.0)

    def test_clip_fraction_and_info(self):
        clipper = RewardClipper(ConstantReward(100.0), low=-1.0, high=1.0)
        assert clipper.clip_fraction == 0.0
        clipper.compute({}, None, {})
        assert clipper.clip_fraction == 1.0
        assert clipper.get_info()["raw"] == 100.0


class TestRewardNormalizerAndShaper:
    def test_warmup_returns_raw(self):
        norm = RewardNormalizer(ConstantReward(5.0), warmup=10)
        assert norm.compute({}, None, {}) == 5.0

    def test_reset_stats(self):
        norm = RewardNormalizer(ConstantReward(1.0), warmup=0)
        for _ in range(20):
            norm.compute({}, None, {})
        norm.reset_stats()
        assert norm.mean == 0.0 and norm.std == 1.0

    def test_shaping_with_goal_potential(self):
        fn = ConstantReward(0.0)
        s = _make_state(position=(3.0, 0.0), goal=(0.0, 0.0))
        ns = _make_state(position=(2.0, 0.0), goal=(0.0, 0.0))
        shaper = RewardShaper.with_goal_potential(fn, gamma=1.0, scale=1.0)
        # phi(s) = -3, phi(ns) = -2, shaping = 1*(-2) - (-3) = 1.0
        assert shaper.compute(s, None, ns) == pytest.approx(1.0)

    def test_shaper_invalid_gamma(self):
        with pytest.raises(ValueError):
            RewardShaper(ConstantReward(), lambda s: 0.0, gamma=1.5)

    def test_goal_distance_potential(self):
        s = {"position": [0.0, 0.0], "goal": [3.0, 4.0]}
        assert RewardShaper.goal_distance_potential(s) == pytest.approx(-5.0)


# ===================================================================
# rewards/navigation tests (16 tests)
# ===================================================================


class TestGoalReward:
    def test_sparse_at_goal_and_far(self):
        r = GoalReward(mode="sparse", threshold=1.0, success_reward=10.0)
        assert r.compute({}, None, _make_state(position=(0.0, 0.0), goal=(0.5, 0.0))) == 10.0
        r2 = GoalReward(mode="sparse", threshold=1.0)
        assert r2.compute({}, None, _make_state(position=(0.0, 0.0), goal=(5.0, 5.0))) == 0.0

    def test_dense_mode(self):
        r = GoalReward(mode="dense", max_distance=10.0, dense_scale=1.0)
        val = r.compute({}, None, _make_state(position=(0.0, 0.0), goal=(5.0, 0.0)))
        assert val == pytest.approx(0.5)

    def test_shaped_mode_progress(self):
        r = GoalReward(mode="shaped", shaped_scale=1.0)
        s1 = _make_state(position=(5.0, 0.0), goal=(0.0, 0.0))
        s2 = _make_state(position=(3.0, 0.0), goal=(0.0, 0.0))
        r.compute(s1, None, s1)
        assert r.compute(s1, None, s2) == pytest.approx(2.0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            GoalReward(mode="invalid")


class TestPathFollowingReward:
    def test_on_and_off_path(self):
        r = PathFollowingReward(path=[(0, 0), (10, 0)], tolerance=1.0, scale=1.0)
        assert r.compute({}, None, _make_state(position=(5.0, 0.0))) == pytest.approx(1.0)
        r2 = PathFollowingReward(path=[(0, 0), (10, 0)], tolerance=0.0, falloff=2.0, scale=1.0)
        assert r2.compute({}, None, _make_state(position=(5.0, 3.0))) < 1.0

    def test_no_path_and_validation(self):
        assert PathFollowingReward().compute({}, None, _make_state()) == 0.0
        with pytest.raises(ValueError):
            PathFollowingReward().set_path([(0, 0)])


class TestTimePenaltyReward:
    def test_penalty_and_negation(self):
        assert TimePenaltyReward(penalty=-0.1).compute({}, None, {}) == pytest.approx(-0.1)
        assert TimePenaltyReward(penalty=0.5).compute({}, None, {}) == pytest.approx(-0.5)

    def test_max_cumulative(self):
        r = TimePenaltyReward(penalty=-1.0, max_cumulative=2.0)
        r.compute({}, None, {})
        r.compute({}, None, {})
        assert r.compute({}, None, {}) == 0.0  # Capped


class TestCollisionPenalty:
    def test_no_collision(self):
        r = CollisionPenalty(agent_radius=0.2)
        ns = _make_state(position=(0.0, 0.0), pedestrians=[], obstacles=[])
        assert r.compute({}, None, ns) == 0.0

    def test_pedestrian_collision(self):
        r = CollisionPenalty(agent_radius=0.2, pedestrian_penalty=-10.0)
        ns = _make_state(position=(0.0, 0.0), pedestrians=[{"position": [0.1, 0.0]}])
        assert r.compute({}, None, ns) == pytest.approx(-10.0)

    def test_obstacle_collision_point_array(self):
        r = CollisionPenalty(agent_radius=0.5, obstacle_penalty=-5.0)
        ns = _make_state(position=(0.0, 0.0), obstacles=np.array([[0.1, 0.0]]))
        assert r.compute({}, None, ns) == pytest.approx(-5.0)


class TestProgressReward:
    def test_first_call_zero_then_progress(self):
        r = ProgressReward(scale=1.0)
        s1 = _make_state(position=(5.0, 0.0), goal=(0.0, 0.0))
        s2 = _make_state(position=(3.0, 0.0), goal=(0.0, 0.0))
        assert r.compute({}, None, s1) == 0.0
        assert r.compute({}, None, s2) == pytest.approx(2.0)

    def test_regression(self):
        r = ProgressReward(scale=1.0, regression_scale=2.0)
        r.compute({}, None, _make_state(position=(3.0, 0.0), goal=(0.0, 0.0)))
        val = r.compute({}, None, _make_state(position=(5.0, 0.0), goal=(0.0, 0.0)))
        assert val == pytest.approx(-4.0)


class TestVelocityReward:
    def test_stopped_and_at_target(self):
        r = VelocityReward(stop_threshold=0.1, stop_penalty=-1.0, target_speed=1.0, tolerance=0.1)
        assert r.compute({}, None, _make_state(velocity=(0.0, 0.0))) == pytest.approx(-1.0)
        assert r.compute({}, None, _make_state(velocity=(1.0, 0.0))) == 0.0

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            VelocityReward(mode="bad")


class TestSmoothnessReward:
    def test_first_step_no_penalty(self):
        r = SmoothnessReward()
        assert r.compute({}, None, _make_state(velocity=(1.0, 0.0))) == 0.0

    def test_abrupt_acceleration(self):
        r = SmoothnessReward(accel_weight=1.0, max_accel=0.0, linear=True)
        s = _make_state(velocity=(0.0, 0.0), dt=0.1)
        r.compute(s, None, s)
        val = r.compute(s, None, _make_state(velocity=(1.0, 0.0), dt=0.1))
        assert val < 0.0


class TestBoundaryPenalty:
    def test_inside_no_penalty(self):
        r = BoundaryPenalty(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, margin=1.0)
        assert r.compute({}, None, _make_state(position=(5.0, 5.0))) == 0.0

    def test_near_boundary_linear(self):
        r = BoundaryPenalty(x_min=0.0, margin=1.0, penalty_scale=1.0, mode="linear")
        assert r.compute({}, None, _make_state(position=(0.5, 5.0))) == pytest.approx(-0.5)

    def test_outside_hard_penalty(self):
        r = BoundaryPenalty(x_min=0.0, hard_penalty=-10.0)
        assert r.compute({}, None, _make_state(position=(-1.0, 5.0))) == pytest.approx(-10.0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            BoundaryPenalty(mode="cubic")
