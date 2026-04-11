"""Tests for navirl.rewards.navigation module."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the submodule directly from file to avoid the package __init__
# which tries to import navirl.rewards.learned (may not exist).
# First ensure the base module is loadable.
_base_path = Path(__file__).resolve().parent.parent / "navirl" / "rewards" / "base.py"
_base_spec = importlib.util.spec_from_file_location("navirl.rewards.base", _base_path)
if "navirl.rewards.base" not in sys.modules:
    _base_mod = importlib.util.module_from_spec(_base_spec)
    sys.modules[_base_spec.name] = _base_mod
    _base_spec.loader.exec_module(_base_mod)

_nav_path = Path(__file__).resolve().parent.parent / "navirl" / "rewards" / "navigation.py"
_nav_spec = importlib.util.spec_from_file_location("navirl.rewards.navigation", _nav_path)
_nav = importlib.util.module_from_spec(_nav_spec)
sys.modules[_nav_spec.name] = _nav
_nav_spec.loader.exec_module(_nav)
BoundaryPenalty = _nav.BoundaryPenalty
CollisionPenalty = _nav.CollisionPenalty
GoalReward = _nav.GoalReward
PathFollowingReward = _nav.PathFollowingReward
ProgressReward = _nav.ProgressReward
SmoothnessReward = _nav.SmoothnessReward
TimePenaltyReward = _nav.TimePenaltyReward
VelocityReward = _nav.VelocityReward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(pos=(0.0, 0.0), goal=(10.0, 0.0), vel=(1.0, 0.0), **kw):
    """Build a minimal state dict."""
    s = {"position": np.array(pos), "goal": np.array(goal), "velocity": np.array(vel)}
    s.update(kw)
    return s


# ---------------------------------------------------------------------------
# GoalReward
# ---------------------------------------------------------------------------


class TestGoalReward:
    def test_sparse_not_at_goal(self):
        r = GoalReward(mode="sparse", threshold=0.3)
        s1 = _state(pos=(0, 0))
        s2 = _state(pos=(5, 0))
        assert r.compute(s1, None, s2) == 0.0

    def test_sparse_at_goal(self):
        r = GoalReward(mode="sparse", threshold=0.5, success_reward=10.0)
        s1 = _state(pos=(0, 0))
        s2 = _state(pos=(9.8, 0), goal=(10, 0))
        assert r.compute(s1, None, s2) == 10.0

    def test_dense_reward_closer_is_better(self):
        r = GoalReward(mode="dense", max_distance=10.0, dense_scale=1.0)
        close = r.compute(_state(), None, _state(pos=(8, 0)))
        far = r.compute(_state(), None, _state(pos=(2, 0)))
        assert close > far

    def test_shaped_first_step_no_progress(self):
        r = GoalReward(mode="shaped")
        r.reset()
        # First step: no previous distance cached
        v = r.compute(_state(), None, _state(pos=(5, 0)))
        # _prev_distance was None, so only arrival bonus if within threshold
        assert isinstance(v, float)

    def test_shaped_progress(self):
        r = GoalReward(mode="shaped", shaped_scale=1.0, success_reward=0.0)
        r.reset()
        r.compute(_state(), None, _state(pos=(0, 0)))  # prime
        v = r.compute(_state(), None, _state(pos=(5, 0)))  # closer to goal at (10,0)
        assert v > 0  # Made progress

    def test_shaped_regression(self):
        r = GoalReward(mode="shaped", shaped_scale=1.0, success_reward=0.0)
        r.reset()
        r.compute(_state(), None, _state(pos=(5, 0)))  # prime at d=5
        v = r.compute(_state(), None, _state(pos=(0, 0)))  # farther from goal at d=10
        assert v < 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            GoalReward(mode="invalid")

    def test_reset_clears_state(self):
        r = GoalReward(mode="shaped")
        r.compute(_state(), None, _state())
        r.reset()
        assert r._prev_distance is None


# ---------------------------------------------------------------------------
# PathFollowingReward
# ---------------------------------------------------------------------------


class TestPathFollowingReward:
    def test_no_path_returns_zero(self):
        r = PathFollowingReward()
        assert r.compute(_state(), None, _state()) == 0.0

    def test_on_path_high_reward(self):
        path = [(0, 0), (10, 0)]
        r = PathFollowingReward(path=path, tolerance=1.0, scale=1.0)
        v = r.compute(_state(), None, _state(pos=(5, 0)))
        assert v > 0.5

    def test_far_from_path_low_reward(self):
        path = [(0, 0), (10, 0)]
        r = PathFollowingReward(path=path, tolerance=0.5, falloff=2.0, scale=1.0)
        v = r.compute(_state(), None, _state(pos=(5, 10)))
        assert v < 0.1

    def test_advance_bonus(self):
        path = [(0, 0), (5, 0), (10, 0)]
        r = PathFollowingReward(path=path, advance_bonus=1.0)
        r.compute(_state(), None, _state(pos=(0, 0)))  # seg 0
        v2 = r.compute(_state(), None, _state(pos=(7.5, 0)))  # seg 1
        assert v2 > r.compute(_state(), None, _state(pos=(2.5, 0)))

    def test_invalid_path_shape(self):
        with pytest.raises(ValueError, match="shape"):
            PathFollowingReward(path=[(0,), (1,)])

    def test_too_few_waypoints(self):
        with pytest.raises(ValueError, match="at least 2"):
            PathFollowingReward(path=[(0, 0)])

    def test_set_path(self):
        r = PathFollowingReward()
        r.set_path([(0, 0), (10, 0)])
        assert r._path is not None

    def test_reset(self):
        r = PathFollowingReward(path=[(0, 0), (10, 0)])
        r._furthest_segment = 5
        r.reset()
        assert r._furthest_segment == 0

    def test_point_segment_distance(self):
        p = np.array([1.0, 1.0])
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 0.0])
        d = PathFollowingReward._point_segment_distance(p, a, b)
        assert d == pytest.approx(1.0)

    def test_point_segment_distance_degenerate(self):
        p = np.array([1.0, 1.0])
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])  # zero-length segment
        d = PathFollowingReward._point_segment_distance(p, a, b)
        assert d == pytest.approx(math.sqrt(2))


# ---------------------------------------------------------------------------
# TimePenaltyReward
# ---------------------------------------------------------------------------


class TestTimePenaltyReward:
    def test_constant_penalty(self):
        r = TimePenaltyReward(penalty=-0.1)
        assert r.compute(_state(), None, _state()) == pytest.approx(-0.1)

    def test_positive_penalty_is_negated(self):
        r = TimePenaltyReward(penalty=0.1)
        assert r.compute(_state(), None, _state()) == pytest.approx(-0.1)

    def test_dt_scaling(self):
        r = TimePenaltyReward(penalty=-1.0, use_dt=True)
        s = _state(dt=0.04)
        v = r.compute(s, None, _state())
        assert v == pytest.approx(-0.04)

    def test_max_cumulative(self):
        r = TimePenaltyReward(penalty=-1.0, max_cumulative=2.0)
        r.compute(_state(), None, _state())  # -1.0
        r.compute(_state(), None, _state())  # -2.0, at limit
        v = r.compute(_state(), None, _state())
        assert v == 0.0  # Capped

    def test_reset(self):
        r = TimePenaltyReward(penalty=-1.0, max_cumulative=1.0)
        r.compute(_state(), None, _state())
        r.reset()
        # Should be able to accumulate again
        v = r.compute(_state(), None, _state())
        assert v == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# CollisionPenalty
# ---------------------------------------------------------------------------


class TestCollisionPenalty:
    def test_no_collisions(self):
        r = CollisionPenalty()
        s = _state(pos=(5, 5), pedestrians=[], obstacles=[])
        assert r.compute(s, None, s) == 0.0

    def test_pedestrian_collision(self):
        r = CollisionPenalty(agent_radius=0.2, pedestrian_penalty=-10.0)
        s = _state(
            pos=(0, 0),
            pedestrians=[{"position": [0.1, 0.0], "radius": 0.18}],
        )
        v = r.compute(_state(), None, s)
        assert v == pytest.approx(-10.0)

    def test_no_pedestrian_collision_when_far(self):
        r = CollisionPenalty(agent_radius=0.2, pedestrian_penalty=-10.0)
        s = _state(
            pos=(0, 0),
            pedestrians=[{"position": [5.0, 5.0], "radius": 0.18}],
        )
        assert r.compute(_state(), None, s) == 0.0

    def test_obstacle_collision_dict(self):
        r = CollisionPenalty(agent_radius=0.2, obstacle_penalty=-5.0)
        s = _state(
            pos=(0, 0),
            obstacles=[{"position": [0.1, 0.0], "radius": 0.1}],
        )
        v = r.compute(_state(), None, s)
        assert v == pytest.approx(-5.0)

    def test_obstacle_collision_ndarray_points(self):
        r = CollisionPenalty(agent_radius=0.5, obstacle_penalty=-5.0)
        s = _state(
            pos=(0, 0),
            obstacles=np.array([[0.1, 0.0], [10.0, 10.0]]),
        )
        v = r.compute(_state(), None, s)
        assert v == pytest.approx(-5.0)  # One collision

    def test_obstacle_collision_ndarray_segments(self):
        r = CollisionPenalty(agent_radius=0.5, obstacle_penalty=-5.0)
        s = _state(
            pos=(0, 0),
            obstacles=np.array([[0.0, -1.0, 0.0, 1.0]]),  # line segment through origin
        )
        v = r.compute(_state(), None, s)
        assert v == pytest.approx(-5.0)

    def test_cumulative_mode(self):
        r = CollisionPenalty(agent_radius=0.5, pedestrian_penalty=-10.0, cumulative=True)
        s = _state(
            pos=(0, 0),
            pedestrians=[
                {"position": [0.1, 0.0], "radius": 0.18},
                {"position": [-0.1, 0.0], "radius": 0.18},
            ],
        )
        v = r.compute(_state(), None, s)
        assert v == pytest.approx(-20.0)

    def test_non_cumulative_mode(self):
        r = CollisionPenalty(agent_radius=0.5, pedestrian_penalty=-10.0, cumulative=False)
        s = _state(
            pos=(0, 0),
            pedestrians=[
                {"position": [0.1, 0.0], "radius": 0.18},
                {"position": [-0.1, 0.0], "radius": 0.18},
            ],
        )
        v = r.compute(_state(), None, s)
        assert v == pytest.approx(-10.0)

    def test_get_info(self):
        r = CollisionPenalty(agent_radius=0.5, pedestrian_penalty=-10.0)
        s = _state(
            pos=(0, 0),
            pedestrians=[{"position": [0.1, 0.0], "radius": 0.18}],
        )
        r.compute(_state(), None, s)
        info = r.get_info()
        assert info["n_collisions"] == 1


# ---------------------------------------------------------------------------
# ProgressReward
# ---------------------------------------------------------------------------


class TestProgressReward:
    def test_first_step_returns_zero(self):
        r = ProgressReward()
        v = r.compute(_state(), None, _state(pos=(5, 0)))
        assert v == 0.0

    def test_progress_positive(self):
        r = ProgressReward(scale=1.0, max_reward=10.0)
        r.reset()
        r.compute(_state(), None, _state(pos=(0, 0)))  # d=10
        v = r.compute(_state(), None, _state(pos=(5, 0)))  # d=5, progress=5
        assert v > 0

    def test_regression_negative(self):
        r = ProgressReward(scale=1.0, regression_scale=2.0)
        r.reset()
        r.compute(_state(), None, _state(pos=(5, 0)))  # d=5
        v = r.compute(_state(), None, _state(pos=(0, 0)))  # d=10, regressed
        assert v < 0

    def test_distance_gain(self):
        r = ProgressReward(scale=1.0, distance_gain=True)
        r.reset()
        r.compute(_state(), None, _state(pos=(9, 0), goal=(10, 0)))
        v_near = r.compute(_state(), None, _state(pos=(9.5, 0), goal=(10, 0)))
        r.reset()
        r.compute(_state(), None, _state(pos=(0, 0), goal=(10, 0)))
        v_far = r.compute(_state(), None, _state(pos=(0.5, 0), goal=(10, 0)))
        # Progress near goal should be worth more
        assert v_near > v_far

    def test_max_reward_clamp(self):
        r = ProgressReward(scale=100.0, max_reward=5.0)
        r.reset()
        r.compute(_state(), None, _state(pos=(0, 0)))
        v = r.compute(_state(), None, _state(pos=(9, 0)))
        assert abs(v) <= 5.0

    def test_reset(self):
        r = ProgressReward()
        r.compute(_state(), None, _state())
        r.reset()
        assert r._prev_dist is None


# ---------------------------------------------------------------------------
# VelocityReward
# ---------------------------------------------------------------------------


class TestVelocityReward:
    def test_at_target_speed(self):
        r = VelocityReward(target_speed=1.0, tolerance=0.1, mode="penalty_only")
        v = r.compute(_state(), None, _state(vel=(1.0, 0.0)))
        assert v == pytest.approx(0.0)

    def test_stopped_penalty(self):
        r = VelocityReward(stop_threshold=0.05, stop_penalty=-0.5)
        v = r.compute(_state(), None, _state(vel=(0.0, 0.0)))
        assert v == pytest.approx(-0.5)

    def test_too_fast_penalty(self):
        r = VelocityReward(target_speed=1.0, tolerance=0.1, penalty_weight=1.0, mode="penalty_only")
        v = r.compute(_state(), None, _state(vel=(3.0, 0.0)))
        assert v < 0

    def test_reward_match_mode(self):
        r = VelocityReward(target_speed=1.0, mode="reward_match")
        v_good = r.compute(_state(), None, _state(vel=(1.0, 0.0)))
        v_bad = r.compute(_state(), None, _state(vel=(3.0, 0.0)))
        assert v_good > v_bad

    def test_speed_key(self):
        r = VelocityReward(target_speed=1.0, tolerance=0.1, mode="penalty_only")
        v = r.compute(_state(), None, {"speed": 1.0})
        assert v == pytest.approx(0.0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            VelocityReward(mode="bad")


# ---------------------------------------------------------------------------
# SmoothnessReward
# ---------------------------------------------------------------------------


class TestSmoothnessReward:
    def test_constant_velocity_no_penalty(self):
        r = SmoothnessReward(max_accel=2.0)
        s1 = _state(vel=(1, 0), dt=0.1)
        s2 = _state(vel=(1, 0), dt=0.1)
        r.compute(s1, None, s1)  # prime
        v = r.compute(s1, None, s2)
        assert v == pytest.approx(0.0)

    def test_high_accel_penalty(self):
        r = SmoothnessReward(accel_weight=1.0, max_accel=1.0)
        s1 = _state(vel=(0, 0), dt=0.1)
        s2 = _state(vel=(10, 0), dt=0.1)
        r.compute(s1, None, s1)  # prime
        v = r.compute(s1, None, s2)
        assert v < 0

    def test_heading_penalty(self):
        r = SmoothnessReward(heading_weight=1.0, max_heading_change=0.1)
        s1 = _state(vel=(1, 0), heading=0.0, dt=0.1)
        s2 = _state(vel=(1, 0), heading=math.pi, dt=0.1)
        r.compute(_state(), None, s1)  # prime
        v = r.compute(s1, None, s2)
        assert v < 0

    def test_linear_mode(self):
        r = SmoothnessReward(accel_weight=1.0, max_accel=1.0, linear=True)
        s1 = _state(vel=(0, 0), dt=0.1)
        s2 = _state(vel=(5, 0), dt=0.1)
        r.compute(s1, None, s1)
        v = r.compute(s1, None, s2)
        assert v < 0

    def test_reset(self):
        r = SmoothnessReward()
        r.compute(_state(), None, _state(vel=(1, 0)))
        r.reset()
        assert r._prev_velocity is None
        assert r._prev_heading is None

    def test_angle_diff_wrapping(self):
        d = SmoothnessReward._angle_diff(0.1, 2 * math.pi - 0.1)
        assert abs(d) < 0.3  # Should wrap to ~0.2


# ---------------------------------------------------------------------------
# BoundaryPenalty
# ---------------------------------------------------------------------------


class TestBoundaryPenalty:
    def test_well_inside_no_penalty(self):
        r = BoundaryPenalty(x_min=0, x_max=10, y_min=0, y_max=10, margin=1.0)
        v = r.compute(_state(), None, _state(pos=(5, 5)))
        assert v == 0.0

    def test_near_boundary_penalty(self):
        r = BoundaryPenalty(x_min=0, x_max=10, y_min=0, y_max=10, margin=1.0, penalty_scale=1.0)
        v = r.compute(_state(), None, _state(pos=(0.5, 5)))
        assert v < 0

    def test_outside_boundary_hard_penalty(self):
        r = BoundaryPenalty(x_min=0, x_max=10, y_min=0, y_max=10, hard_penalty=-5.0)
        v = r.compute(_state(), None, _state(pos=(-1, 5)))
        assert v == pytest.approx(-5.0)

    def test_circular_boundary(self):
        r = BoundaryPenalty(center=(5, 5), radius=3.0, margin=1.0, hard_penalty=-5.0)
        # Inside
        v_in = r.compute(_state(), None, _state(pos=(5, 5)))
        assert v_in == 0.0
        # Outside
        v_out = r.compute(_state(), None, _state(pos=(10, 10)))
        assert v_out == pytest.approx(-5.0)

    def test_quadratic_mode(self):
        r = BoundaryPenalty(
            x_min=0,
            x_max=10,
            y_min=0,
            y_max=10,
            margin=2.0,
            mode="quadratic",
            penalty_scale=1.0,
        )
        v = r.compute(_state(), None, _state(pos=(1.0, 5)))
        assert v < 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            BoundaryPenalty(mode="cubic")

    def test_no_bounds_no_penalty(self):
        r = BoundaryPenalty()
        v = r.compute(_state(), None, _state(pos=(100, 100)))
        assert v == 0.0
