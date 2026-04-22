"""Tests for under-covered decision methods of navirl.humans.behavior_model.

Complements the broad smoke tests in test_humans.py by targeting branches that
were previously uncovered:

- AttentionModel.step: enters distraction, returns to focus when the timer expires.
- AttentionModel.can_perceive: co-located observer/target short-circuit.
- BehaviorModel.should_yield: assertiveness vs. compliance vs. speed-ratio paths.
- BehaviorModel.adapt_speed: blending toward slower neighbours and the
  assertiveness-weighted preferred-speed mix.
- BehaviorModel.choose_side: non-compliant random branch.
- BehaviorModel.compute_comfort: personal-space intrusion and distraction penalty.
- BehaviorModel.compute_stress: very-close neighbour stress term.
- BehaviorModel.reset: propagates to the attention model.
- Personality-specific attention defaults and factory classmethods.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.humans.behavior_model import (
    AttentionModel,
    BehaviorModel,
    PersonalityParams,
)
from navirl.humans.pedestrian_state import PedestrianState, PersonalityTag

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _ped(
    pid: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
    heading: float = 0.0,
    radius: float = 0.3,
) -> PedestrianState:
    return PedestrianState(
        pid=pid,
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        heading=heading,
        goal=np.array([goal_x, goal_y], dtype=np.float64),
        radius=radius,
    )


# ---------------------------------------------------------------------------
# AttentionModel.step -- distraction lifecycle
# ---------------------------------------------------------------------------


class _AlwaysEnterRng:
    """rng.random() returns 0.0 (always < any positive probability)."""

    def random(self) -> float:
        return 0.0

    def exponential(self, mean: float) -> float:
        return float(mean)


class _NeverEnterRng:
    """rng.random() returns 1.0 (never < a probability in [0, 1])."""

    def random(self) -> float:
        return 1.0


class TestAttentionModelStep:
    def test_enters_distracted_when_probability_triggers(self):
        attn = AttentionModel(
            distraction_probability=1.0,
            distraction_duration_mean=2.0,
        )
        assert not attn.is_distracted
        attn.step(dt=0.5, rng=_AlwaysEnterRng())
        # random()=0 < 1.0*0.5=0.5, so we entered distraction.
        assert attn.is_distracted
        # Remaining time comes from exponential(mean) stub.
        assert attn._distraction_remaining == pytest.approx(2.0)

    def test_skips_entering_when_rng_rejects(self):
        attn = AttentionModel(distraction_probability=0.01)
        attn.step(dt=0.05, rng=_NeverEnterRng())
        assert not attn.is_distracted

    def test_distraction_timer_counts_down(self):
        attn = AttentionModel(distraction_probability=0.0)
        attn._distracted = True
        attn._distraction_remaining = 1.0
        attn.step(dt=0.3, rng=_NeverEnterRng())
        assert attn.is_distracted
        assert attn._distraction_remaining == pytest.approx(0.7)

    def test_distraction_clears_when_timer_expires(self):
        attn = AttentionModel(distraction_probability=0.0)
        attn._distracted = True
        attn._distraction_remaining = 0.1
        attn.step(dt=0.5, rng=_NeverEnterRng())
        assert not attn.is_distracted
        assert attn._distraction_remaining == 0.0

    def test_distraction_clamps_remaining_to_zero(self):
        attn = AttentionModel(distraction_probability=0.0)
        attn._distracted = True
        attn._distraction_remaining = 0.2
        attn.step(dt=5.0, rng=_NeverEnterRng())  # Overshoot.
        # Remaining is reset to 0 rather than being negative.
        assert attn._distraction_remaining == 0.0


class TestAttentionCanPerceiveCollocated:
    def test_colocated_returns_true(self):
        """When observer and target coincide, perception is always granted."""
        attn = AttentionModel(awareness_radius=10.0, field_of_view=math.pi / 4.0)
        observer = _ped(x=0.0, y=0.0, heading=0.0)
        # Target within < 1e-9 of observer -> short-circuit.
        assert attn.can_perceive(observer, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# _default_attention -- personality-specific paths
# ---------------------------------------------------------------------------


class TestDefaultAttentionPerPersonality:
    def test_distracted_attention_defaults(self):
        bm = BehaviorModel(personality=PersonalityTag.DISTRACTED, rng_seed=0)
        # DISTRACTED branch of _default_attention.
        assert bm.attention.awareness_radius == pytest.approx(5.0)
        assert bm.attention.field_of_view == pytest.approx(math.pi / 3.0)
        assert bm.attention.distraction_probability > 0.0

    def test_child_attention_defaults(self):
        bm = BehaviorModel(personality=PersonalityTag.CHILD, rng_seed=0)
        # CHILD branch of _default_attention — 180 deg FOV and small distraction.
        assert bm.attention.awareness_radius == pytest.approx(5.0)
        assert bm.attention.field_of_view == pytest.approx(math.pi)
        assert bm.attention.reaction_time == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# create_child classmethod
# ---------------------------------------------------------------------------


class TestCreateChild:
    def test_create_child_sets_personality(self):
        bm = BehaviorModel.create_child(base_preferred_speed=1.0, rng_seed=0)
        assert bm.personality == PersonalityTag.CHILD
        # Child personality applies 0.9 speed factor.
        assert bm.preferred_speed == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# BehaviorModel.should_yield
# ---------------------------------------------------------------------------


class _YieldingCoinRng:
    """random() returns a configurable constant for deterministic yielding."""

    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


class TestShouldYield:
    def test_passive_yields_for_equal_speed_neighbour(self):
        """assertiveness=0 -> base p_yield=1.0; equal-speed neighbour gives 0.8."""
        params = PersonalityParams(assertiveness=0.0, compliance=0.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _YieldingCoinRng(0.5)  # < p_yield=0.8 -> yields.

        ego = _ped(x=0.0, y=0.0, vx=1.0, heading=0.0)
        other = _ped(x=5.0, y=0.0, vx=-1.0, heading=math.pi)
        assert bm.should_yield(ego, other) is True

    def test_equal_speed_passive_declines_when_rng_above_threshold(self):
        """With p_yield=0.8 an rng value above 0.8 still declines to yield."""
        params = PersonalityParams(assertiveness=0.0, compliance=0.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _YieldingCoinRng(0.95)  # 0.95 >= 0.8 -> no yield.

        ego = _ped(x=0.0, y=0.0, vx=1.0, heading=0.0)
        other = _ped(x=5.0, y=0.0, vx=-1.0, heading=math.pi)
        assert bm.should_yield(ego, other) is False

    def test_fully_assertive_never_yields(self):
        """assertiveness=1 -> base p_yield=0, compliance=0 keeps it near 0 so rng>=p_yield."""
        params = PersonalityParams(assertiveness=1.0, compliance=0.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _YieldingCoinRng(0.001)

        ego = _ped(x=0.0, y=0.0, vx=1.0, heading=0.0)
        # Other on the left (positive y -> positive bearing) so no +0.15 bump even if
        # compliance were nonzero.
        other = _ped(x=0.0, y=5.0, vx=0.0, heading=0.0)
        assert bm.should_yield(ego, other) is False

    def test_faster_other_increases_yield_probability(self):
        """speed_ratio > 1 multiplies p_yield by 1.2; slower divides by 0.8."""
        params = PersonalityParams(assertiveness=0.5, compliance=0.0)

        bm_fast = BehaviorModel(params=params, rng_seed=0)
        bm_fast.rng = _YieldingCoinRng(0.55)  # Just above 0.5
        ego = _ped(x=0.0, y=0.0, vx=1.0, heading=0.0)
        fast_other = _ped(x=5.0, y=0.0, vx=-2.0, heading=math.pi)
        # p_yield = 0.5 * 1.2 = 0.6; rng=0.55 < 0.6 -> yields
        assert bm_fast.should_yield(ego, fast_other) is True

        bm_slow = BehaviorModel(params=params, rng_seed=0)
        bm_slow.rng = _YieldingCoinRng(0.45)  # Just below 0.5 but above 0.4
        slow_other = _ped(x=5.0, y=0.0, vx=-0.1, heading=math.pi)
        # p_yield = 0.5 * 0.8 = 0.4; rng=0.45 >= 0.4 -> does not yield
        assert bm_slow.should_yield(ego, slow_other) is False

    def test_compliance_bonus_for_right_hand_neighbour(self):
        """A neighbour on the ego's right adds 0.15 * compliance to p_yield."""
        # With assertiveness=1.0 the base is 0, and slower neighbour drops to 0.0.
        # The compliance bump should be the only source of p_yield.
        params = PersonalityParams(assertiveness=1.0, compliance=1.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _YieldingCoinRng(0.1)

        ego = _ped(x=0.0, y=0.0, vx=1.0, heading=0.0)
        # Other on the ego's right: heading is +x, so +y is left, -y is right.
        right_side_other = _ped(x=1.0, y=-1.0, vx=0.1, heading=0.0)
        # p_yield = 0.0 * 0.8 + 0.15 (compliance) = 0.15; rng=0.1 < 0.15 -> yields
        assert bm.should_yield(ego, right_side_other) is True

    def test_yield_probability_clamped_to_unit_interval(self):
        """Aggregated p_yield beyond 1 is clamped, so rng=0.999 still triggers yield."""
        # High compliance bump combined with faster neighbour pushes p_yield > 1.
        params = PersonalityParams(assertiveness=0.0, compliance=1.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _YieldingCoinRng(0.999)

        ego = _ped(x=0.0, y=0.0, vx=1.0, heading=0.0)
        # Other on the right moving fast -> ratio>1, +0.15 compliance bonus.
        right_fast = _ped(x=1.0, y=-1.0, vx=-3.0, heading=math.pi)
        assert bm.should_yield(ego, right_fast) is True


# ---------------------------------------------------------------------------
# BehaviorModel.adapt_speed
# ---------------------------------------------------------------------------


class TestAdaptSpeedWithNeighbours:
    def test_all_neighbours_outside_awareness(self):
        """Neighbours beyond awareness_radius don't affect adaptation."""
        bm = BehaviorModel(rng_seed=42)
        # Force awareness radius tiny so no one is "perceived" inside.
        bm.attention = AttentionModel(awareness_radius=0.1)

        ego = _ped(vx=1.0)
        far_neighbour = _ped(x=10.0, y=0.0, vx=0.5)
        assert bm.adapt_speed(ego, [far_neighbour]) == pytest.approx(bm.preferred_speed)

    def test_slow_neighbour_drags_adapted_speed_down(self):
        """Nearby slow neighbours blend the adapted speed toward their mean."""
        # assertiveness=0 -> fully influenced by blended neighbour speed.
        params = PersonalityParams(assertiveness=0.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.attention = AttentionModel(awareness_radius=5.0)

        ego = _ped(vx=1.2)
        slow_close = _ped(x=1.0, y=0.0, vx=0.2)
        adapted = bm.adapt_speed(ego, [slow_close])

        # Adapted must be strictly between neighbour speed and preferred speed.
        assert adapted < bm.preferred_speed
        assert adapted >= 0.0

    def test_result_clamped_to_max_speed(self):
        """Adapted speed cannot exceed max_speed regardless of blending."""
        bm = BehaviorModel(rng_seed=0)
        bm.attention = AttentionModel(awareness_radius=5.0)

        ego = _ped(vx=0.1)
        # Neighbour much faster than max_speed -> blending pulls up, but capped.
        fast_close = _ped(x=0.5, y=0.0, vx=50.0)
        adapted = bm.adapt_speed(ego, [fast_close])
        assert adapted <= bm.max_speed + 1e-9


# ---------------------------------------------------------------------------
# BehaviorModel.choose_side
# ---------------------------------------------------------------------------


class _SeqRng:
    """rng.random() returns successive values from a sequence."""

    def __init__(self, values):
        self._iter = iter(values)

    def random(self) -> float:
        return float(next(self._iter))


class TestChooseSide:
    def test_compliant_path_passes_right(self):
        params = PersonalityParams(compliance=1.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _SeqRng([0.0])  # < compliance -> returns -1 immediately.
        side = bm.choose_side(_ped(), _ped(x=3.0))
        assert side == -1

    def test_non_compliant_random_left(self):
        """compliance=0 -> skip right, then rng()<0.5 -> pass on the left (+1)."""
        params = PersonalityParams(compliance=0.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _SeqRng([0.9, 0.1])  # First rejects right, second picks left.
        side = bm.choose_side(_ped(), _ped(x=3.0))
        assert side == 1

    def test_non_compliant_random_right(self):
        """compliance=0 + second rng>=0.5 -> pass on the right (-1)."""
        params = PersonalityParams(compliance=0.0)
        bm = BehaviorModel(params=params, rng_seed=0)
        bm.rng = _SeqRng([0.9, 0.9])
        side = bm.choose_side(_ped(), _ped(x=3.0))
        assert side == -1


# ---------------------------------------------------------------------------
# BehaviorModel.compute_comfort
# ---------------------------------------------------------------------------


class TestComputeComfort:
    def test_close_neighbour_reduces_comfort(self):
        bm = BehaviorModel(rng_seed=0)
        ego = _ped(vx=bm.preferred_speed)
        intrusive = _ped(x=0.1, y=0.0)  # Well inside personal_space*1.5.
        alone = bm.compute_comfort(ego, [])
        crowded = bm.compute_comfort(ego, [intrusive])
        assert crowded < alone
        assert 0.0 <= crowded <= 1.0

    def test_distraction_penalty_applied(self):
        bm = BehaviorModel(rng_seed=0)
        bm.attention._distracted = True
        ego = _ped(vx=bm.preferred_speed)
        comfort_distracted = bm.compute_comfort(ego, [])

        bm.attention._distracted = False
        comfort_focused = bm.compute_comfort(ego, [])
        assert comfort_distracted < comfort_focused

    def test_comfort_clamped_to_zero(self):
        """Many close neighbours should not drive comfort below 0."""
        bm = BehaviorModel(rng_seed=0)
        bm.attention._distracted = True
        ego = _ped(vx=0.0)  # Also triggers large speed deviation penalty.
        neighbours = [_ped(pid=i, x=0.05 * (i + 1), y=0.0) for i in range(10)]
        comfort = bm.compute_comfort(ego, neighbours)
        assert comfort == 0.0


# ---------------------------------------------------------------------------
# BehaviorModel.compute_stress
# ---------------------------------------------------------------------------


class TestComputeStress:
    def test_very_close_neighbour_increases_stress(self):
        bm = BehaviorModel(rng_seed=0)
        ego = _ped(vx=bm.preferred_speed, radius=0.3)
        # Distance < radius*3 = 0.9 triggers stress term.
        close = _ped(x=0.4, y=0.0, radius=0.3)
        stress_alone = bm.compute_stress(ego, [])
        stress_close = bm.compute_stress(ego, [close])
        assert stress_close > stress_alone


# ---------------------------------------------------------------------------
# BehaviorModel.reset
# ---------------------------------------------------------------------------


class TestBehaviorModelReset:
    def test_reset_clears_attention_distraction(self):
        bm = BehaviorModel(rng_seed=0)
        bm.attention._distracted = True
        bm.attention._distraction_remaining = 3.0
        bm.reset()
        assert not bm.attention.is_distracted
        assert bm.attention._distraction_remaining == 0.0
