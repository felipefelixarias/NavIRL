"""Extended tests for navirl/rewards/base.py — RewardNormalizer, RewardShaper, CompositeReward.

Covers code paths not exercised by test_maps_rewards.py:
  - RewardNormalizer: Welford accuracy, EMA mode, center/scale toggles, clip, post-warmup
  - RewardShaper: potential caching, error handling, scale, reset, compute diagnostics
  - CompositeReward: summary(), __iter__, __repr__, reset, decomposition toggle off
  - RewardClipper: reset forwarding, repr, get_info bounds
  - RewardComponent: reset forwarding, repr, tags
"""

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
# Bootstrap: navirl.rewards __init__ imports submodules that may not exist.
# Load the specific module directly from its file path.
# ---------------------------------------------------------------------------

_root = _pathlib.Path(__file__).resolve().parent.parent / "navirl"


def _load_module(fqn: str, filepath: _pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(fqn, filepath)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _ensure_stub_package(fqn: str, path: _pathlib.Path) -> None:
    if fqn not in sys.modules:
        stub = types.ModuleType(fqn)
        stub.__path__ = [str(path)]  # type: ignore[attr-defined]
        stub.__package__ = fqn
        sys.modules[fqn] = stub


_ensure_stub_package("navirl.rewards", _root / "rewards")
_rewards_base = _load_module("navirl.rewards.base", _root / "rewards" / "base.py")

CompositeReward = _rewards_base.CompositeReward
RewardClipper = _rewards_base.RewardClipper
RewardComponent = _rewards_base.RewardComponent
RewardFunction = _rewards_base.RewardFunction
RewardNormalizer = _rewards_base.RewardNormalizer
RewardShaper = _rewards_base.RewardShaper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConstantReward(RewardFunction):
    """Returns a fixed value."""

    def __init__(self, value: float = 1.0, name: str | None = None) -> None:
        super().__init__(name=name or "Const")
        self._value = value
        self._reset_count = 0

    def compute(self, state, action, next_state, *, info=None) -> float:
        return self._value

    def reset(self) -> None:
        self._reset_count += 1


class SequenceReward(RewardFunction):
    """Returns successive values from a sequence, cycling."""

    def __init__(self, values: list[float]) -> None:
        super().__init__(name="Seq")
        self._values = values
        self._idx = 0

    def compute(self, state, action, next_state, *, info=None) -> float:
        v = self._values[self._idx % len(self._values)]
        self._idx += 1
        return v


S, A, NS = {}, None, {}  # dummy transition


def _make_state(position=(0.0, 0.0), goal=(5.0, 5.0)):
    return {
        "position": np.array(position, dtype=np.float64),
        "goal": np.array(goal, dtype=np.float64),
    }


# ===================================================================
# RewardNormalizer — Welford mode
# ===================================================================


class TestRewardNormalizerWelford:
    """Welford online normalisation (default mode, gamma=None)."""

    def test_warmup_phase_returns_raw(self):
        norm = RewardNormalizer(ConstantReward(7.0), warmup=5)
        for _ in range(5):
            assert norm.compute(S, A, NS) == 7.0
        # After warmup, normalisation kicks in — constant input → zero mean
        val = norm.compute(S, A, NS)
        assert val != 7.0

    def test_welford_mean_std_accuracy(self):
        """Feed known values and verify mean/std converge."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        seq = SequenceReward(values)
        norm = RewardNormalizer(seq, warmup=0, center=True, scale=True, clip=None)
        results = []
        for _ in range(5):
            results.append(norm.compute(S, A, NS))
        assert norm.mean == pytest.approx(np.mean(values))
        assert norm.std == pytest.approx(np.std(values, ddof=0), abs=1e-6)

    def test_center_only(self):
        """scale=False means divide by 1, just subtract mean."""
        norm = RewardNormalizer(
            ConstantReward(10.0), warmup=0, center=True, scale=False, clip=None
        )
        # After enough calls the mean converges to 10 → normalised ≈ 0
        for _ in range(200):
            norm.compute(S, A, NS)
        val = norm.compute(S, A, NS)
        assert abs(val) < 0.1

    def test_scale_only(self):
        """center=False means don't subtract mean, just divide by std."""
        norm = RewardNormalizer(
            ConstantReward(5.0), warmup=0, center=False, scale=True, clip=None
        )
        # Constant input → std stays 1.0 (from initial), so value stays ~5
        val = norm.compute(S, A, NS)
        # First call: count=1, std=1.0 (welford needs 2+ samples)
        assert val == pytest.approx(5.0)

    def test_neither_center_nor_scale(self):
        """center=False, scale=False → raw value returned (post-warmup too)."""
        norm = RewardNormalizer(
            ConstantReward(42.0), warmup=0, center=False, scale=False, clip=None
        )
        for _ in range(10):
            assert norm.compute(S, A, NS) == pytest.approx(42.0)

    def test_clip_value(self):
        """Post-warmup values are clipped to [-clip, clip]."""
        seq = SequenceReward([0.0] * 110 + [1000.0])
        norm = RewardNormalizer(seq, warmup=100, clip=5.0)
        # Consume warmup
        for _ in range(110):
            norm.compute(S, A, NS)
        # 111th value = 1000 — after normalisation should be clipped
        val = norm.compute(S, A, NS)
        assert -5.0 <= val <= 5.0

    def test_std_returns_1_with_single_sample(self):
        norm = RewardNormalizer(ConstantReward(3.0), warmup=0)
        norm.compute(S, A, NS)
        assert norm.std == 1.0  # count < 2

    def test_reset_resets_wrapped_not_stats(self):
        inner = ConstantReward(1.0)
        norm = RewardNormalizer(inner, warmup=0)
        norm.compute(S, A, NS)
        norm.compute(S, A, NS)
        assert norm._count == 2
        norm.reset()
        assert inner._reset_count == 1
        assert norm._count == 2  # stats preserved

    def test_reset_stats_clears_everything(self):
        norm = RewardNormalizer(ConstantReward(3.0), warmup=0)
        for _ in range(10):
            norm.compute(S, A, NS)
        norm.reset_stats()
        assert norm._count == 0
        assert norm.mean == 0.0
        assert norm.std == 1.0
        assert norm._ema_mean == 0.0
        assert norm._ema_var == 1.0

    def test_get_info_keys(self):
        norm = RewardNormalizer(ConstantReward(2.0), warmup=0)
        norm.compute(S, A, NS)
        info = norm.get_info()
        assert set(info.keys()) == {"raw", "normalised", "running_mean", "running_std", "count"}
        assert info["raw"] == 2.0
        assert info["count"] == 1

    def test_name_default_and_custom(self):
        n1 = RewardNormalizer(ConstantReward(name="foo"))
        assert n1.name == "Normalized(foo)"
        n2 = RewardNormalizer(ConstantReward(), name="custom_name")
        assert n2.name == "custom_name"


# ===================================================================
# RewardNormalizer — EMA mode
# ===================================================================


class TestRewardNormalizerEMA:
    """Exponential moving average normalisation (gamma != None)."""

    def test_ema_mean_tracks_input(self):
        norm = RewardNormalizer(ConstantReward(10.0), gamma=0.9, warmup=0, clip=None)
        for _ in range(200):
            norm.compute(S, A, NS)
        # EMA mean should converge close to 10
        assert norm.mean == pytest.approx(10.0, abs=0.5)

    def test_ema_std_is_nonnegative(self):
        norm = RewardNormalizer(ConstantReward(5.0), gamma=0.95, warmup=0)
        for _ in range(50):
            norm.compute(S, A, NS)
        assert norm.std >= 0.0

    def test_ema_low_gamma_fast_tracking(self):
        """gamma=0 → EMA mean = latest value exactly."""
        norm = RewardNormalizer(SequenceReward([1.0, 2.0, 3.0]), gamma=0.0, warmup=0, clip=None)
        norm.compute(S, A, NS)
        assert norm.mean == pytest.approx(1.0)
        norm.compute(S, A, NS)
        assert norm.mean == pytest.approx(2.0)

    def test_ema_updates_use_gamma_branch(self):
        norm = RewardNormalizer(ConstantReward(5.0), gamma=0.9, warmup=0)
        norm.compute(S, A, NS)
        # With gamma set, _update_ema should have been called
        assert norm._ema_mean != 0.0


# ===================================================================
# RewardShaper
# ===================================================================


def _val_potential(s):
    """Extract 'val' key as potential."""
    return float(s.get("val", 0.0))


def _const_potential(s):
    """Always return 1.0."""
    return 1.0


def _zero_potential(s):
    """Always return 0.0."""
    return 0.0


class TestRewardShaperCompute:
    def test_basic_shaping_formula(self):
        """r' = r + scale * (gamma * Phi(s') - Phi(s))"""
        base = ConstantReward(1.0)
        shaper = RewardShaper(base, _val_potential, gamma=0.99, scale=2.0)
        s = {"val": 3.0}
        ns = {"val": 5.0}
        # shaping = 2.0 * (0.99 * 5.0 - 3.0) = 2.0 * (4.95 - 3.0) = 2.0 * 1.95 = 3.9
        expected = 1.0 + 3.9
        assert shaper.compute(s, A, ns) == pytest.approx(expected)

    def test_gamma_zero_cancels_next_potential(self):
        base = ConstantReward(0.0)
        shaper = RewardShaper(base, _val_potential, gamma=0.0, scale=1.0)
        s = {"val": 10.0}
        ns = {"val": 999.0}
        # shaping = 1.0 * (0.0 * 999 - 10) = -10
        assert shaper.compute(s, A, ns) == pytest.approx(-10.0)

    def test_gamma_one_exact_difference(self):
        base = ConstantReward(0.0)
        shaper = RewardShaper(base, _val_potential, gamma=1.0, scale=1.0)
        s = {"val": 2.0}
        ns = {"val": 7.0}
        # shaping = 1.0 * (1.0 * 7 - 2) = 5.0
        assert shaper.compute(s, A, ns) == pytest.approx(5.0)

    def test_scale_parameter(self):
        base = ConstantReward(0.0)
        shaper1 = RewardShaper(base, _const_potential, gamma=1.0, scale=1.0)
        shaper2 = RewardShaper(base, _const_potential, gamma=1.0, scale=3.0)
        val1 = shaper1.compute(S, A, NS)
        val2 = shaper2.compute(S, A, NS)
        # Both give 0 since potential is constant, but scale multiplies shaping
        assert val1 == pytest.approx(0.0)
        assert val2 == pytest.approx(0.0)

    def test_potential_error_returns_zero(self):
        """If potential_fn raises, _compute_potential returns 0."""
        base = ConstantReward(5.0)

        def bad_potential(s):
            raise RuntimeError("boom")

        shaper = RewardShaper(base, bad_potential, gamma=0.99)
        # phi(s) = 0, phi(s') = 0 → shaping = 0
        assert shaper.compute(S, A, NS) == pytest.approx(5.0)

    def test_reset_clears_cached_potential(self):
        base = ConstantReward(0.0)
        shaper = RewardShaper(base, _const_potential, gamma=0.99)
        shaper.compute(S, A, NS)
        assert shaper._prev_potential is not None
        shaper.reset()
        assert shaper._prev_potential is None
        assert base._reset_count == 1

    def test_get_info_diagnostics(self):
        base = ConstantReward(2.0)
        shaper = RewardShaper(base, _zero_potential, gamma=0.95, scale=1.5)
        shaper.compute(S, A, NS)
        info = shaper.get_info()
        assert info["base_reward"] == 2.0
        assert info["gamma"] == 0.95
        assert info["scale"] == 1.5
        assert "shaping_bonus" in info
        assert "total" in info

    def test_invalid_gamma_low(self):
        with pytest.raises(ValueError, match="gamma"):
            RewardShaper(ConstantReward(), _zero_potential, gamma=-0.1)

    def test_invalid_gamma_high(self):
        with pytest.raises(ValueError, match="gamma"):
            RewardShaper(ConstantReward(), _zero_potential, gamma=1.01)

    def test_name_default_and_custom(self):
        s1 = RewardShaper(ConstantReward(name="base"), _zero_potential, gamma=0.99)
        assert s1.name == "Shaped(base)"
        s2 = RewardShaper(ConstantReward(), _zero_potential, gamma=0.99, name="my_shaper")
        assert s2.name == "my_shaper"


class TestRewardShaperFactory:
    def test_with_goal_potential_basic(self):
        base = ConstantReward(0.0)
        shaper = RewardShaper.with_goal_potential(base, gamma=1.0, scale=1.0)
        s = _make_state(position=(3.0, 0.0), goal=(0.0, 0.0))
        ns = _make_state(position=(2.0, 0.0), goal=(0.0, 0.0))
        # phi(s) = -3, phi(ns) = -2, shaping = -2 - (-3) = 1.0
        assert shaper.compute(s, A, ns) == pytest.approx(1.0)

    def test_with_goal_potential_custom_name(self):
        shaper = RewardShaper.with_goal_potential(
            ConstantReward(), gamma=0.9, name="goal_shaped"
        )
        assert shaper.name == "goal_shaped"

    def test_with_goal_potential_scale(self):
        base = ConstantReward(0.0)
        shaper = RewardShaper.with_goal_potential(base, gamma=1.0, scale=2.0)
        s = _make_state(position=(3.0, 0.0), goal=(0.0, 0.0))
        ns = _make_state(position=(2.0, 0.0), goal=(0.0, 0.0))
        # shaping = 2.0 * (1.0 * (-2) - (-3)) = 2.0 * 1.0 = 2.0
        assert shaper.compute(s, A, ns) == pytest.approx(2.0)

    def test_goal_distance_potential_static(self):
        s = _make_state(position=(3.0, 4.0), goal=(0.0, 0.0))
        assert RewardShaper.goal_distance_potential(s) == pytest.approx(-5.0)

    def test_goal_distance_potential_at_goal(self):
        s = _make_state(position=(2.0, 3.0), goal=(2.0, 3.0))
        assert RewardShaper.goal_distance_potential(s) == pytest.approx(0.0)


# ===================================================================
# CompositeReward — extended
# ===================================================================


class TestCompositeRewardExtended:
    def test_summary_format(self):
        c1 = RewardComponent(ConstantReward(1.0, name="goal"), weight=2.0, tags=["nav"])
        c2 = RewardComponent(ConstantReward(1.0, name="time"), weight=-0.5, enabled=False)
        cr = CompositeReward([c1, c2], name="TestComposite")
        summary = cr.summary()
        assert "TestComposite" in summary
        assert "goal" in summary
        assert "time" in summary
        assert "ON" in summary
        assert "OFF" in summary
        assert "nav" in summary

    def test_iter_yields_components(self):
        c1 = RewardComponent(ConstantReward(name="a"))
        c2 = RewardComponent(ConstantReward(name="b"))
        cr = CompositeReward([c1, c2])
        names = [c.name for c in cr]
        assert names == ["a", "b"]

    def test_repr(self):
        cr = CompositeReward([RewardComponent(ConstantReward())])
        assert "n_components=1" in repr(cr)

    def test_len(self):
        cr = CompositeReward()
        assert len(cr) == 0
        cr.add_component(RewardComponent(ConstantReward()))
        assert len(cr) == 1

    def test_reset_clears_decomposition_and_forwards(self):
        inner = ConstantReward(3.0)
        comp = RewardComponent(inner)
        cr = CompositeReward([comp])
        cr.compute(S, A, NS)
        assert cr._last_decomposition != {}
        cr.reset()
        assert cr._last_decomposition == {}
        assert inner._reset_count == 1

    def test_decomposition_tracking_off(self):
        c1 = RewardComponent(ConstantReward(2.0, name="x"))
        cr = CompositeReward([c1], track_decomposition=False)
        result = cr.compute(S, A, NS)
        assert result == pytest.approx(2.0)
        assert cr.get_info()["decomposition"] == {}
        assert cr.get_info()["total"] == 0.0

    def test_get_info_total_matches_compute(self):
        c1 = RewardComponent(ConstantReward(3.0, name="a"), weight=2.0)
        c2 = RewardComponent(ConstantReward(1.0, name="b"), weight=1.0)
        cr = CompositeReward([c1, c2])
        val = cr.compute(S, A, NS)
        info = cr.get_info()
        assert info["total"] == pytest.approx(val)
        assert info["decomposition"]["a"] == pytest.approx(6.0)
        assert info["decomposition"]["b"] == pytest.approx(1.0)

    def test_enable_disable_nonexistent_is_noop(self):
        cr = CompositeReward()
        cr.enable("missing")  # should not raise
        cr.disable("missing")  # should not raise

    def test_multiple_components_same_tag(self):
        c1 = RewardComponent(ConstantReward(name="a"), tags=["social", "nav"])
        c2 = RewardComponent(ConstantReward(name="b"), tags=["social"])
        c3 = RewardComponent(ConstantReward(name="c"), tags=["nav"])
        cr = CompositeReward([c1, c2, c3])
        assert len(cr.filter_by_tag("social")) == 2
        assert len(cr.filter_by_tag("nav")) == 2
        assert len(cr.filter_by_tag("missing")) == 0


# ===================================================================
# RewardClipper — extended
# ===================================================================


class TestRewardClipperExtended:
    def test_reset_forwards_to_inner(self):
        inner = ConstantReward(5.0)
        clipper = RewardClipper(inner, low=-1.0, high=1.0)
        clipper.reset()
        assert inner._reset_count == 1

    def test_get_info_bounds(self):
        clipper = RewardClipper(ConstantReward(0.0), low=-2.0, high=3.0)
        clipper.compute(S, A, NS)
        info = clipper.get_info()
        assert info["bounds"] == (-2.0, 3.0)

    def test_clips_negative(self):
        clipper = RewardClipper(ConstantReward(-100.0), low=-5.0, high=5.0)
        assert clipper.compute(S, A, NS) == -5.0

    def test_exact_boundary_no_clip(self):
        clipper = RewardClipper(ConstantReward(5.0), low=-5.0, high=5.0)
        clipper.compute(S, A, NS)
        assert clipper.clip_fraction == 0.0

    def test_asymmetric_bounds(self):
        clipper = RewardClipper(ConstantReward(100.0), low=-1.0, high=50.0)
        assert clipper.compute(S, A, NS) == 50.0

    def test_name_default_and_custom(self):
        c1 = RewardClipper(ConstantReward(name="inner"), low=-1.0, high=1.0)
        assert c1.name == "Clipped(inner)"
        c2 = RewardClipper(ConstantReward(), low=-1.0, high=1.0, name="custom")
        assert c2.name == "custom"

    def test_multiple_calls_fraction(self):
        # 3 values: -100 (clipped), 0 (not), 100 (clipped) → fraction 2/3
        seq = SequenceReward([-100.0, 0.0, 100.0])
        clipper = RewardClipper(seq, low=-10.0, high=10.0)
        for _ in range(3):
            clipper.compute(S, A, NS)
        assert clipper.clip_fraction == pytest.approx(2 / 3)


# ===================================================================
# RewardComponent — extended
# ===================================================================


class TestRewardComponentExtended:
    def test_repr_format(self):
        comp = RewardComponent(ConstantReward(name="test"), weight=0.5, enabled=True)
        r = repr(comp)
        assert "test" in r
        assert "0.5" in r

    def test_reset_forwards(self):
        inner = ConstantReward()
        comp = RewardComponent(inner)
        comp.reset()
        assert inner._reset_count == 1

    def test_tags_in_constructor(self):
        comp = RewardComponent(ConstantReward(), tags=["a", "b"])
        assert comp.tags == ["a", "b"]

    def test_info_augmented_with_metadata(self):
        comp = RewardComponent(ConstantReward(), weight=3.0, enabled=False)
        info = comp.get_info()
        assert info["weight"] == 3.0
        assert info["enabled"] is False


# ===================================================================
# RewardFunction — abstract base edge cases
# ===================================================================


class TestRewardFunctionBase:
    def test_repr(self):
        r = ConstantReward(name="test_fn")
        assert "ConstantReward" in repr(r)
        assert "test_fn" in repr(r)

    def test_call_delegates_to_compute(self):
        r = ConstantReward(99.0)
        assert r(S, A, NS) == 99.0
        assert r(S, A, NS, info={"key": "val"}) == 99.0
