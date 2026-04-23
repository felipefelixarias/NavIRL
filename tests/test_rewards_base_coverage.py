"""Fill coverage gaps in navirl/rewards/base.py.

The ``test_maps_rewards.py`` suite already exercises the common happy
paths (weighted sum, clip, warmup). This suite targets the uncovered
branches: reset/get_info methods, the EMA normaliser path, and the
RewardShaper exception fallback.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

# Load navirl.rewards.base directly to avoid the rewards/__init__ import
# chain (which pulls in optional submodules that may not exist).
_base_path = Path(__file__).resolve().parent.parent / "navirl" / "rewards" / "base.py"
_base_spec = importlib.util.spec_from_file_location("navirl.rewards.base", _base_path)
if "navirl.rewards.base" not in sys.modules:
    _base_mod = importlib.util.module_from_spec(_base_spec)
    sys.modules[_base_spec.name] = _base_mod
    _base_spec.loader.exec_module(_base_mod)
else:
    _base_mod = sys.modules["navirl.rewards.base"]

RewardFunction = _base_mod.RewardFunction
RewardComponent = _base_mod.RewardComponent
CompositeReward = _base_mod.CompositeReward
RewardNormalizer = _base_mod.RewardNormalizer
RewardClipper = _base_mod.RewardClipper
RewardShaper = _base_mod.RewardShaper


class _Counting(RewardFunction):
    """Reward that counts compute/reset invocations and returns configured values."""

    def __init__(self, values: list[float] | None = None, name: str | None = None) -> None:
        super().__init__(name=name)
        self._values = list(values) if values is not None else [0.0]
        self.compute_calls = 0
        self.reset_calls = 0

    def compute(self, state, action, next_state, *, info=None):  # type: ignore[override]
        v = self._values[min(self.compute_calls, len(self._values) - 1)]
        self.compute_calls += 1
        return v

    def reset(self) -> None:
        self.reset_calls += 1


class _Constant(RewardFunction):
    def __init__(self, value: float = 0.0, name: str | None = None) -> None:
        super().__init__(name=name)
        self._value = value

    def compute(self, state, action, next_state, *, info=None):  # type: ignore[override]
        return self._value


# ---------------------------------------------------------------------------
# RewardFunction dunder methods
# ---------------------------------------------------------------------------


class TestRewardFunctionDunders:
    """Coverage for __call__ and __repr__ on the base class."""

    def test_call_forwards_to_compute(self):
        r = _Constant(value=7.0, name="foo")
        assert r({}, None, {}) == 7.0

    def test_call_forwards_info_kwarg(self):
        recorded = {}

        class _Inspect(RewardFunction):
            def compute(self, state, action, next_state, *, info=None):  # type: ignore[override]
                recorded["info"] = info
                return 0.0

        _Inspect()({}, None, {}, info={"k": 1})
        assert recorded["info"] == {"k": 1}

    def test_repr_includes_class_and_name(self):
        r = _Constant(name="my_reward")
        out = repr(r)
        assert "_Constant" in out
        assert "my_reward" in out


# ---------------------------------------------------------------------------
# RewardComponent reset & __repr__ (lines 204, 214)
# ---------------------------------------------------------------------------


class TestRewardComponentResetAndRepr:
    def test_reset_forwards_to_inner(self):
        inner = _Counting()
        comp = RewardComponent(reward_fn=inner, weight=3.0)
        comp.reset()
        assert inner.reset_calls == 1

    def test_repr_contains_name_weight_enabled(self):
        comp = RewardComponent(reward_fn=_Constant(name="x"), weight=2.5, enabled=False)
        s = repr(comp)
        assert "x" in s
        assert "2.5" in s
        assert "False" in s


# ---------------------------------------------------------------------------
# CompositeReward reset / summary / iter / components property (344-346, 363-368, 374, 377)
# ---------------------------------------------------------------------------


class TestCompositeRewardExtras:
    def test_reset_forwards_and_clears_decomposition(self):
        inner1 = _Counting()
        inner2 = _Counting()
        cr = CompositeReward([RewardComponent(inner1), RewardComponent(inner2)])
        cr.compute({}, None, {})
        # Pre-reset: last_decomposition populated
        assert cr._last_decomposition  # type: ignore[attr-defined]
        cr.reset()
        assert inner1.reset_calls == 1 and inner2.reset_calls == 1
        assert cr._last_decomposition == {}  # type: ignore[attr-defined]

    def test_components_property_returns_internal_list(self):
        c1 = RewardComponent(_Constant(1.0, name="a"))
        cr = CompositeReward([c1])
        # Exposed for external mutation / inspection.
        assert cr.components is cr._components  # type: ignore[attr-defined]
        assert cr.components == [c1]

    def test_summary_includes_name_count_and_component_lines(self):
        c_on = RewardComponent(_Constant(1.0, name="alpha"), weight=2.0, tags=["nav", "x"])
        c_off = RewardComponent(_Constant(name="beta"), weight=-1.0, enabled=False)
        cr = CompositeReward([c_on, c_off], name="mix")
        out = cr.summary()
        assert "CompositeReward 'mix'" in out
        assert "2 components" in out
        assert "alpha" in out and "beta" in out
        assert "ON " in out and "OFF" in out
        assert "nav" in out  # tag listed

    def test_summary_empty_tag_list_still_renders(self):
        c = RewardComponent(_Constant(1.0, name="only"))
        cr = CompositeReward([c])
        out = cr.summary()
        # Empty tag block should still produce a bracketed line.
        assert "only" in out and "tags=[]" in out

    def test_iter_yields_components_in_order(self):
        comps = [RewardComponent(_Constant(float(i), name=f"c{i}")) for i in range(3)]
        cr = CompositeReward(comps)
        assert list(iter(cr)) == comps

    def test_repr_has_n_components(self):
        cr = CompositeReward([RewardComponent(_Constant()) for _ in range(4)])
        assert "n_components=4" in repr(cr)

    def test_untracked_composite_returns_zero_totals(self):
        c = RewardComponent(_Constant(2.0, name="z"))
        cr = CompositeReward([c], track_decomposition=False)
        assert cr.compute({}, None, {}) == pytest.approx(2.0)
        # get_info always queries _last_decomposition which stays empty here.
        info = cr.get_info()
        assert info["decomposition"] == {}
        assert info["total"] == 0.0


# ---------------------------------------------------------------------------
# RewardNormalizer EMA branch (454, 461, 476-480, 484) + reset + get_info
# ---------------------------------------------------------------------------


class TestRewardNormalizerEma:
    """Exercise the ``gamma`` exponential-moving-average branch."""

    def test_ema_mean_tracks_series_via_public_property(self):
        # Values constant => EMA mean converges to that value; with warmup=0
        # the update is applied from call 1.
        norm = RewardNormalizer(_Constant(5.0), warmup=0, gamma=0.5, clip=None)
        # One compute advances EMA: mean = 0.5*0 + 0.5*5 = 2.5
        norm.compute({}, None, {})
        assert norm.mean == pytest.approx(2.5)
        # Second compute: mean = 0.5*2.5 + 0.5*5 = 3.75
        norm.compute({}, None, {})
        assert norm.mean == pytest.approx(3.75)

    def test_ema_std_is_sqrt_of_ema_var(self):
        norm = RewardNormalizer(_Counting([0.0, 2.0, 2.0]), warmup=0, gamma=0.5, clip=None)
        for _ in range(3):
            norm.compute({}, None, {})
        # std should be sqrt(max(_ema_var, 0))
        assert norm.std == pytest.approx(math.sqrt(max(norm._ema_var, 0.0)))  # type: ignore[attr-defined]

    def test_ema_normalisation_applied_when_warmup_skipped(self):
        # With gamma set, ``_count`` is never incremented (EMA path), so
        # ``warmup`` must be < 0 to escape the warmup guard and actually
        # apply centering.  This exercises the ``if self._center`` branch
        # of ``compute`` on top of the EMA update.
        inner = _Counting([10.0, 10.0, 10.0])
        norm = RewardNormalizer(inner, warmup=-1, gamma=0.5, clip=None, center=True, scale=False)
        # First compute: EMA mean -> 0.5*0 + 0.5*10 = 5, centered = 10 - 5 = 5.
        out = norm.compute({}, None, {})
        assert out == pytest.approx(5.0)

    def test_reset_forwards_but_keeps_stats(self):
        inner = _Counting([1.0, 1.0])
        norm = RewardNormalizer(inner, warmup=0, gamma=None)
        norm.compute({}, None, {})
        count_before = norm._count  # type: ignore[attr-defined]
        norm.reset()
        # reset() does NOT clear stats, only the inner function's state.
        assert inner.reset_calls == 1
        assert norm._count == count_before  # type: ignore[attr-defined]

    def test_get_info_shape_and_values(self):
        norm = RewardNormalizer(_Constant(3.0), warmup=0, gamma=None, clip=None)
        norm.compute({}, None, {})
        info = norm.get_info()
        assert set(info) == {"raw", "normalised", "running_mean", "running_std", "count"}
        assert info["raw"] == 3.0
        assert info["count"] >= 1


# ---------------------------------------------------------------------------
# RewardClipper.reset forwards (line 633)
# ---------------------------------------------------------------------------


class TestRewardClipperReset:
    def test_reset_forwards_to_inner(self):
        inner = _Counting()
        clip = RewardClipper(inner, low=-1.0, high=1.0)
        clip.reset()
        assert inner.reset_calls == 1


# ---------------------------------------------------------------------------
# RewardShaper exception fallback + reset + get_info (763-767, 771-772, 776)
# ---------------------------------------------------------------------------


class TestRewardShaperErrorAndLifecycle:
    def test_potential_exception_returns_zero_and_logs(self, caplog):
        def bad_phi(state):
            raise RuntimeError("boom")

        shaper = RewardShaper(_Constant(5.0), bad_phi, gamma=1.0)
        with caplog.at_level("WARNING"):
            total = shaper.compute({}, None, {})
        # phi raised -> both s and next_s potentials fall back to 0 -> shaping=0 -> total=base=5.
        assert total == pytest.approx(5.0)
        assert any("Potential function raised" in rec.message for rec in caplog.records)

    def test_potential_returning_non_float_coerced_via_float(self):
        # Returning an int goes through float(...) cleanly; string would raise -> 0.
        called = {"n": 0}

        def phi_non_float(state):
            called["n"] += 1
            return "not-a-number"

        shaper = RewardShaper(_Constant(0.0), phi_non_float, gamma=0.5)
        # Should still produce a finite number (0) because float("not-a-number") raises.
        out = shaper.compute(
            {"position": [0, 0], "goal": [0, 0]}, None, {"position": [0, 0], "goal": [0, 0]}
        )
        assert math.isfinite(out)
        # phi was attempted on both s and next_s.
        assert called["n"] == 2

    def test_reset_clears_prev_potential_and_forwards(self):
        inner = _Counting()

        def phi(state):
            return 1.0

        shaper = RewardShaper(inner, phi, gamma=1.0)
        shaper.compute({}, None, {})
        assert shaper._prev_potential == 1.0  # type: ignore[attr-defined]
        shaper.reset()
        assert inner.reset_calls == 1
        assert shaper._prev_potential is None  # type: ignore[attr-defined]

    def test_get_info_exposes_base_shaping_total_gamma_scale(self):
        # phi(s) = 0, phi(ns) = 2, gamma=0.5, scale=2 -> shaping = 2*(0.5*2-0) = 2
        phis = iter([0.0, 2.0])

        def phi(_):
            return next(phis)

        shaper = RewardShaper(_Constant(4.0), phi, gamma=0.5, scale=2.0)
        shaper.compute({}, None, {})
        info = shaper.get_info()
        assert info["gamma"] == 0.5
        assert info["scale"] == 2.0
        assert info["base_reward"] == pytest.approx(4.0)
        assert info["shaping_bonus"] == pytest.approx(2.0)
        assert info["total"] == pytest.approx(6.0)

    def test_with_goal_potential_factory_uses_goal_distance(self):
        inner = _Constant(0.0)
        shaper = RewardShaper.with_goal_potential(inner, gamma=1.0, scale=1.0)
        s = {"position": [4.0, 0.0], "goal": [0.0, 0.0]}
        ns = {"position": [3.0, 0.0], "goal": [0.0, 0.0]}
        # phi(s) = -4, phi(ns) = -3, shaping = -3 - (-4) = 1.
        assert shaper.compute(s, None, ns) == pytest.approx(1.0)
