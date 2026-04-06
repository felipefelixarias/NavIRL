"""Tests for navirl.evaluation.trajectory_metrics module.

Covers displacement errors, collision metrics, path quality, comfort,
goal achievement, social compliance, and the TrajectoryEvaluator class.
Previously at 0% coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.data.trajectory import Trajectory
from navirl.evaluation.trajectory_metrics import (
    MetricResult,
    MetricSummary,
    TrajectoryEvaluator,
    acceleration_profile,
    ade_fde_batch,
    average_displacement_error,
    collision_count,
    collision_rate,
    comfort_score,
    displacement_error_at_horizon,
    energy_expenditure,
    final_displacement_error,
    goal_achievement_rate,
    jerk_metric,
    mean_minimum_distance,
    minimum_separation_distance,
    paired_permutation_test,
    path_curvature,
    path_efficiency_ratio,
    path_irregularity,
    social_compliance_score,
    speed_profile,
    time_to_collision,
    time_to_goal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(
    positions: list[list[float]],
    dt: float = 0.1,
    t0: float = 0.0,
) -> Trajectory:
    """Create a trajectory from a list of [x, y] positions with uniform dt."""
    pos = np.array(positions, dtype=np.float64)
    ts = np.arange(len(pos), dtype=np.float64) * dt + t0
    return Trajectory(timestamps=ts, positions=pos)


def _straight_line(length: int = 10, speed: float = 1.0, dt: float = 0.1) -> Trajectory:
    """Trajectory moving along +x at constant speed."""
    positions = [[i * speed * dt, 0.0] for i in range(length)]
    return _make_trajectory(positions, dt=dt)


# ===================================================================
# MetricResult and MetricSummary
# ===================================================================


class TestMetricResult:
    def test_repr(self):
        r = MetricResult(name="ADE", value=0.5, unit="m", sample_size=10)
        s = repr(r)
        assert "ADE" in s
        assert "0.5" in s

    def test_repr_with_ci(self):
        r = MetricResult(name="ADE", value=0.5, unit="m", ci_lower=0.3, ci_upper=0.7)
        s = repr(r)
        assert "CI=" in s


class TestMetricSummary:
    def test_add_and_to_dict(self):
        ms = MetricSummary(model_name="model_a")
        ms.add(MetricResult(name="ADE", value=0.5))
        ms.add(MetricResult(name="FDE", value=1.0))
        d = ms.to_dict()
        assert d == {"ADE": 0.5, "FDE": 1.0}

    def test_to_table_row(self):
        ms = MetricSummary(model_name="model_a")
        ms.add(MetricResult(name="ADE", value=0.1234))
        row = ms.to_table_row(precision=3)
        assert "model_a" in row
        assert "0.123" in row


# ===================================================================
# Displacement errors
# ===================================================================


class TestDisplacementErrors:
    def test_ade_identical(self):
        t = _straight_line()
        assert average_displacement_error(t, t) == 0.0

    def test_ade_offset(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 1], [1, 1], [2, 1]])
        assert average_displacement_error(t1, t2) == pytest.approx(1.0)

    def test_fde_identical(self):
        t = _straight_line()
        assert final_displacement_error(t, t) == 0.0

    def test_fde_offset(self):
        t1 = _make_trajectory([[0, 0], [1, 0]])
        t2 = _make_trajectory([[0, 0], [1, 2]])
        assert final_displacement_error(t1, t2) == pytest.approx(2.0)

    def test_ade_empty(self):
        empty = Trajectory(timestamps=np.array([]), positions=np.empty((0, 2)))
        t = _straight_line()
        assert np.isnan(average_displacement_error(empty, t))

    def test_fde_empty(self):
        empty = Trajectory(timestamps=np.array([]), positions=np.empty((0, 2)))
        t = _straight_line()
        assert np.isnan(final_displacement_error(empty, t))

    def test_ade_different_lengths(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 1], [1, 1]])
        # Should use min length = 2
        ade = average_displacement_error(t1, t2)
        assert ade == pytest.approx(1.0)


class TestAdeFdeBatch:
    def test_batch(self):
        pred = [_make_trajectory([[i, 0], [i + 1, 0]]) for i in range(5)]
        gt = [_make_trajectory([[i, 1], [i + 1, 1]]) for i in range(5)]
        ade_res, fde_res = ade_fde_batch(pred, gt, n_bootstrap=50)
        assert ade_res.name == "ADE"
        assert fde_res.name == "FDE"
        assert ade_res.value == pytest.approx(1.0)
        assert fde_res.value == pytest.approx(1.0)
        assert ade_res.sample_size == 5


class TestDisplacementAtHorizon:
    def test_horizons(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]], dt=1.0)
        t2 = _make_trajectory([[0, 1], [1, 1], [2, 1], [3, 1]], dt=1.0)
        result = displacement_error_at_horizon(t1, t2, horizons=[1.0, 2.0])
        assert 1.0 in result
        assert result[1.0] == pytest.approx(1.0)

    def test_horizon_beyond_trajectory(self):
        t1 = _make_trajectory([[0, 0], [1, 0]], dt=1.0)
        t2 = _make_trajectory([[0, 0], [1, 0]], dt=1.0)
        result = displacement_error_at_horizon(t1, t2, horizons=[10.0])
        assert np.isnan(result[10.0])


# ===================================================================
# Collision and safety metrics
# ===================================================================


class TestCollisionMetrics:
    def test_collision_rate_no_collision(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 10], [1, 10], [2, 10]])
        rate = collision_rate(t1, [t2], collision_radius=0.5)
        assert rate == 0.0

    def test_collision_rate_full_collision(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 0.1], [1, 0.1], [2, 0.1]])
        rate = collision_rate(t1, [t2], collision_radius=0.5)
        assert rate == pytest.approx(1.0)

    def test_collision_rate_empty(self):
        t1 = _make_trajectory([[0, 0]])
        assert collision_rate(t1, [], collision_radius=0.5) == 0.0

    def test_collision_count_distinct_events(self):
        # Create trajectory with two separate collision events
        positions = [[i, 0] for i in range(20)]
        t1 = _make_trajectory(positions)
        # Other agent overlaps at steps 2-4 and 15-17
        other_pos = [[i, 0] if (2 <= i <= 4 or 15 <= i <= 17) else [i, 100] for i in range(20)]
        t2 = _make_trajectory(other_pos)
        count = collision_count(t1, [t2], collision_radius=0.5, min_gap_steps=5)
        assert count == 2

    def test_collision_count_empty(self):
        empty = Trajectory(timestamps=np.array([]), positions=np.empty((0, 2)))
        assert collision_count(empty, [], collision_radius=0.5) == 0

    def test_minimum_separation_distance(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 3], [1, 3], [2, 3]])
        assert minimum_separation_distance(t1, [t2]) == pytest.approx(3.0)

    def test_minimum_separation_no_neighbors(self):
        t = _straight_line()
        assert minimum_separation_distance(t, []) == np.inf

    def test_mean_minimum_distance(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 5], [1, 5], [2, 5]])
        mmd = mean_minimum_distance(t1, [t2])
        assert mmd == pytest.approx(5.0)

    def test_mean_minimum_distance_empty(self):
        t = _straight_line()
        assert mean_minimum_distance(t, []) == np.inf


class TestTimeToCollision:
    def test_agents_approaching(self):
        # Two agents on a collision course
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]], dt=1.0)
        t2 = _make_trajectory([[10, 0], [9, 0], [8, 0], [7, 0]], dt=1.0)
        ttc = time_to_collision(t1, [t2], collision_radius=0.5)
        assert ttc.shape == (4,)
        # TTC should be finite for at least some steps
        assert np.any(np.isfinite(ttc))

    def test_short_trajectory(self):
        t = _make_trajectory([[0, 0]])
        ttc = time_to_collision(t, [], collision_radius=0.5)
        assert len(ttc) == 1
        assert ttc[0] == np.inf


# ===================================================================
# Path quality
# ===================================================================


class TestPathQuality:
    def test_efficiency_straight_line(self):
        t = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]])
        eff = path_efficiency_ratio(t)
        assert eff == pytest.approx(1.0)

    def test_efficiency_with_detour(self):
        t = _make_trajectory([[0, 0], [0, 5], [5, 5], [5, 0]])
        eff = path_efficiency_ratio(t)
        assert 0.0 < eff < 1.0

    def test_efficiency_short_trajectory(self):
        t = _make_trajectory([[0, 0]])
        assert path_efficiency_ratio(t) == 1.0

    def test_efficiency_with_explicit_goal(self):
        t = _make_trajectory([[0, 0], [5, 0]])
        eff = path_efficiency_ratio(t, goal=np.array([5, 0]))
        assert eff == pytest.approx(1.0)

    def test_irregularity_straight_line(self):
        t = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]])
        assert path_irregularity(t) == pytest.approx(0.0, abs=1e-10)

    def test_irregularity_zigzag(self):
        t = _make_trajectory([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
        irr = path_irregularity(t)
        assert irr > 0

    def test_irregularity_short(self):
        t = _make_trajectory([[0, 0], [1, 0]])
        assert path_irregularity(t) == 0.0

    def test_curvature_straight(self):
        t = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]])
        curv = path_curvature(t)
        np.testing.assert_allclose(curv, [0.0, 0.0], atol=1e-10)

    def test_curvature_short(self):
        t = _make_trajectory([[0, 0], [1, 0]])
        curv = path_curvature(t)
        assert len(curv) == 0

    def test_curvature_circle(self):
        # Quarter circle should have non-zero curvature
        angles = np.linspace(0, np.pi / 2, 20)
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        t = Trajectory(timestamps=np.arange(20) * 0.1, positions=positions)
        curv = path_curvature(t)
        assert len(curv) > 0
        assert np.all(curv > 0)


# ===================================================================
# Comfort metrics
# ===================================================================


class TestComfortMetrics:
    def test_speed_profile(self):
        t = _make_trajectory([[0, 0], [1, 0], [2, 0]], dt=1.0)
        speeds = speed_profile(t)
        assert len(speeds) == 3
        np.testing.assert_allclose(speeds, [1.0, 1.0, 1.0])

    def test_speed_profile_single_point(self):
        t = _make_trajectory([[0, 0]])
        speeds = speed_profile(t)
        assert speeds[0] == 0.0

    def test_acceleration_profile(self):
        # Accelerating trajectory: speeds 0, 1, 2 m/s
        t = _make_trajectory([[0, 0], [0, 0], [1, 0], [3, 0]], dt=1.0)
        accels = acceleration_profile(t)
        assert len(accels) == 4

    def test_jerk_metric(self):
        t = _straight_line(length=10, speed=1.0, dt=0.1)
        j = jerk_metric(t)
        assert isinstance(j, float)

    def test_jerk_short_trajectory(self):
        t = _make_trajectory([[0, 0]])
        assert jerk_metric(t) == 0.0

    def test_comfort_score_range(self):
        t = _straight_line(length=20, speed=0.5, dt=0.1)
        score = comfort_score(t)
        assert 0.0 <= score <= 1.0

    def test_energy_expenditure(self):
        t = _make_trajectory([[0, 0], [1, 0], [3, 0], [6, 0]], dt=1.0)
        e = energy_expenditure(t)
        assert isinstance(e, float)
        assert e >= 0.0

    def test_energy_short_trajectory(self):
        t = _make_trajectory([[0, 0], [1, 0]])
        assert energy_expenditure(t) == 0.0


# ===================================================================
# Goal achievement
# ===================================================================


class TestGoalMetrics:
    def test_goal_achievement_rate(self):
        t1 = _make_trajectory([[0, 0], [5, 0]])  # reaches goal at [5, 0]
        t2 = _make_trajectory([[0, 0], [1, 0]])  # doesn't reach [5, 0]
        goals = [np.array([5.0, 0.0]), np.array([5.0, 0.0])]
        result = goal_achievement_rate([t1, t2], goals, threshold=1.0)
        assert result.name == "goal_achievement_rate"
        assert result.value == pytest.approx(0.5)

    def test_time_to_goal_reached(self):
        t = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]], dt=1.0)
        ttg = time_to_goal(t, goal=np.array([2.0, 0.0]), threshold=0.5)
        assert ttg == pytest.approx(2.0)

    def test_time_to_goal_not_reached(self):
        t = _make_trajectory([[0, 0], [1, 0]])
        ttg = time_to_goal(t, goal=np.array([100.0, 0.0]), threshold=0.5)
        assert ttg == np.inf


# ===================================================================
# Social compliance
# ===================================================================


class TestSocialCompliance:
    def test_score_no_violations(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 100], [1, 100], [2, 100]])
        score = social_compliance_score(t1, [t2])
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # should be high with no violations

    def test_score_with_violations(self):
        t1 = _make_trajectory([[0, 0], [1, 0], [2, 0]])
        t2 = _make_trajectory([[0, 0.1], [1, 0.1], [2, 0.1]])
        score = social_compliance_score(t1, [t2], personal_space=0.8, collision_radius=0.5)
        assert score < 0.5  # should be low with violations


# ===================================================================
# Paired permutation test
# ===================================================================


class TestPairedPermutationTest:
    def test_identical_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        p = paired_permutation_test(a, a, n_permutations=500, rng=rng)
        assert p >= 0.0  # no significant difference

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, size=50)
        b = rng.normal(5, 1, size=50)
        p = paired_permutation_test(a, b, n_permutations=500, rng=rng)
        assert p < 0.05  # should detect significant difference


# ===================================================================
# TrajectoryEvaluator
# ===================================================================


class TestTrajectoryEvaluator:
    @pytest.fixture()
    def evaluator(self):
        return TrajectoryEvaluator(
            collision_radius=0.5,
            personal_space=0.8,
            goal_threshold=1.0,
            n_bootstrap=50,
        )

    def test_evaluate_single_basic(self, evaluator):
        pred = _make_trajectory([[0, 0], [1, 0], [2, 0], [3, 0]], dt=0.5)
        gt = _make_trajectory([[0, 0.5], [1, 0.5], [2, 0.5], [3, 0.5]], dt=0.5)
        results = evaluator.evaluate_single(pred, gt)
        assert "ADE" in results
        assert "FDE" in results
        assert "path_efficiency" in results
        assert "comfort_score" in results
        assert results["ADE"] == pytest.approx(0.5)

    def test_evaluate_single_with_neighbors(self, evaluator):
        pred = _make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5)
        gt = _make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5)
        nbr = _make_trajectory([[0, 5], [1, 5], [2, 5]], dt=0.5)
        results = evaluator.evaluate_single(pred, gt, neighbours=[nbr])
        assert "collision_rate" in results
        assert "social_compliance" in results
        assert "min_separation" in results

    def test_evaluate_single_with_goal(self, evaluator):
        pred = _make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5)
        gt = _make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5)
        results = evaluator.evaluate_single(pred, gt, goal=np.array([2.0, 0.0]))
        assert "time_to_goal" in results
        assert "goal_reached" in results

    def test_evaluate_batch(self, evaluator):
        preds = [_make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5) for _ in range(5)]
        gts = [_make_trajectory([[0, 0.5], [1, 0.5], [2, 0.5]], dt=0.5) for _ in range(5)]
        summary = evaluator.evaluate_batch(preds, gts, model_name="test_model")
        assert summary.model_name == "test_model"
        assert "ADE" in summary.results
        assert summary.results["ADE"].sample_size == 5

    def test_compare(self, evaluator):
        preds_a = [_make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5) for _ in range(10)]
        preds_b = [_make_trajectory([[0, 0], [1, 0.5], [2, 1]], dt=0.5) for _ in range(10)]
        gts = [_make_trajectory([[0, 0], [1, 0], [2, 0]], dt=0.5) for _ in range(10)]
        summary_a = evaluator.evaluate_batch(preds_a, gts)
        summary_b = evaluator.evaluate_batch(preds_b, gts)
        p_values = evaluator.compare(summary_a, summary_b, n_permutations=100)
        assert isinstance(p_values, dict)
        assert len(p_values) > 0
