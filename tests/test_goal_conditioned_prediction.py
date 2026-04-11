"""Tests for navirl.prediction.goal_conditioned module.

Covers GoalConditionedPredictor (goal estimation, path planning, predict)
and IntentPredictor (intent classification from trajectory history).
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.prediction.base import PredictionResult
from navirl.prediction.goal_conditioned import (
    DECELERATION_THRESHOLD,
    HIGH_CONFIDENCE_SCORE,
    GoalConditionedPredictor,
    IntentPredictor,
    PedestrianIntent,
    _CandidateGoal,
)

# ---------------------------------------------------------------------------
# GoalConditionedPredictor — construction
# ---------------------------------------------------------------------------

class TestGoalConditionedPredictorConstruction:
    def test_default_params(self):
        p = GoalConditionedPredictor()
        assert p.horizon == 12
        assert p.dt == pytest.approx(0.4)
        assert p.num_goals == 5
        assert p.num_samples_per_goal == 4

    def test_custom_params(self):
        p = GoalConditionedPredictor(
            horizon=20,
            dt=0.2,
            num_goals=3,
            num_samples_per_goal=2,
            goal_noise_std=0.1,
            velocity_smoothing=0.5,
        )
        assert p.horizon == 20
        assert p.dt == pytest.approx(0.2)
        assert p.num_goals == 3


# ---------------------------------------------------------------------------
# GoalConditionedPredictor — _estimate_goals
# ---------------------------------------------------------------------------

class TestEstimateGoals:
    def test_velocity_extrapolation(self):
        """With 2+ observations, goals should be extrapolated from velocity."""
        p = GoalConditionedPredictor(num_goals=5)
        observed = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        goals = p._estimate_goals(observed)
        assert len(goals) >= 1
        # Goals should be ahead in the x direction
        for g in goals:
            assert g.position[0] > 2.0
            assert g.probability > 0

    def test_probabilities_sum_to_one(self):
        p = GoalConditionedPredictor(num_goals=5)
        observed = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        goals = p._estimate_goals(observed)
        total_prob = sum(g.probability for g in goals)
        assert total_prob == pytest.approx(1.0, abs=1e-6)

    def test_single_observation_fallback(self):
        """With only 1 observation, should create a fallback goal at current pos."""
        p = GoalConditionedPredictor(num_goals=3)
        observed = np.array([[5.0, 5.0]])
        goals = p._estimate_goals(observed)
        assert len(goals) >= 1
        # With uniform probs since no velocity info
        assert goals[0].probability == pytest.approx(1.0 / len(goals), abs=1e-6)

    def test_stationary_agent(self):
        """Agent not moving — velocity is near zero, should still produce goals."""
        p = GoalConditionedPredictor(num_goals=5)
        observed = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        goals = p._estimate_goals(observed)
        assert len(goals) >= 1

    def test_scene_candidate_goals(self):
        """When context provides candidate goals, they should be included."""
        p = GoalConditionedPredictor(num_goals=10)
        observed = np.array([[0.0, 0.0], [1.0, 0.0]])
        context = {"candidate_goals": [[20.0, 0.0], [0.0, 20.0]]}
        goals = p._estimate_goals(observed, context)
        # Should include both velocity-extrapolated and scene goals
        positions = [g.position for g in goals]
        # At least one goal near [20, 0]
        has_scene_goal = any(np.linalg.norm(pos - [20, 0]) < 1.0 for pos in positions)
        assert has_scene_goal

    def test_max_num_goals_trimmed(self):
        p = GoalConditionedPredictor(num_goals=2)
        observed = np.array([[0.0, 0.0], [1.0, 0.0]])
        context = {"candidate_goals": [[20, 0], [0, 20], [10, 10]]}
        goals = p._estimate_goals(observed, context)
        assert len(goals) <= 2


# ---------------------------------------------------------------------------
# GoalConditionedPredictor — _plan_path_to_goal
# ---------------------------------------------------------------------------

class TestPlanPathToGoal:
    def test_output_shape(self):
        p = GoalConditionedPredictor(horizon=12, dt=0.4)
        observed = np.array([[0.0, 0.0], [1.0, 0.0]])
        traj = p._plan_path_to_goal(
            start=observed[-1],
            goal=np.array([10.0, 0.0]),
            observed=observed,
        )
        assert traj.shape == (12, 2)

    def test_starts_near_start_position(self):
        p = GoalConditionedPredictor(horizon=12)
        start = np.array([3.0, 4.0])
        observed = np.array([[2.0, 4.0], [3.0, 4.0]])
        traj = p._plan_path_to_goal(start, np.array([10.0, 4.0]), observed)
        # First point should be near start
        assert np.linalg.norm(traj[0] - start) < 2.0

    def test_ends_near_goal(self):
        p = GoalConditionedPredictor(horizon=20, velocity_smoothing=0.5)
        start = np.array([0.0, 0.0])
        goal = np.array([10.0, 0.0])
        observed = np.array([[0.0, 0.0], [0.0, 0.0]])  # Stationary
        traj = p._plan_path_to_goal(start, goal, observed)
        # Last point should be near goal (Hermite interpolation with t=1 goes to p1)
        np.testing.assert_allclose(traj[-1], goal, atol=0.5)

    def test_single_observation(self):
        """With only 1 observation, velocity is zero — still produces valid path."""
        p = GoalConditionedPredictor(horizon=8)
        observed = np.array([[5.0, 5.0]])
        traj = p._plan_path_to_goal(observed[-1], np.array([10.0, 10.0]), observed)
        assert traj.shape == (8, 2)


# ---------------------------------------------------------------------------
# GoalConditionedPredictor — predict
# ---------------------------------------------------------------------------

class TestGoalConditionedPredict:
    def test_predict_output_type(self):
        p = GoalConditionedPredictor(horizon=8, num_goals=3, num_samples_per_goal=2)
        observed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = p.predict(observed)
        assert isinstance(result, PredictionResult)

    def test_predict_trajectory_shape(self):
        horizon = 10
        num_goals = 3
        samples_per = 2
        p = GoalConditionedPredictor(
            horizon=horizon, num_goals=num_goals, num_samples_per_goal=samples_per,
        )
        observed = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
        result = p.predict(observed)
        n_total = result.trajectories.shape[0]
        assert n_total <= num_goals * samples_per
        assert result.trajectories.shape[1] == horizon
        assert result.trajectories.shape[2] == 2

    def test_predict_probabilities_sum_to_one(self):
        p = GoalConditionedPredictor(horizon=8, num_goals=3, num_samples_per_goal=2)
        observed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = p.predict(observed)
        assert result.probabilities.sum() == pytest.approx(1.0, abs=1e-6)

    def test_predict_timestamps(self):
        p = GoalConditionedPredictor(horizon=5, dt=0.3)
        observed = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = p.predict(observed)
        expected_ts = np.arange(1, 6) * 0.3
        np.testing.assert_allclose(result.timestamps, expected_ts)

    def test_predict_with_context(self):
        p = GoalConditionedPredictor(horizon=8, num_goals=5, num_samples_per_goal=2)
        observed = np.array([[0.0, 0.0], [1.0, 0.0]])
        context = {"candidate_goals": [[10.0, 0.0], [0.0, 10.0]]}
        result = p.predict(observed, context)
        assert result.num_samples > 0

    def test_predict_best_trajectory(self):
        p = GoalConditionedPredictor(horizon=8, num_goals=3, num_samples_per_goal=2)
        observed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = p.predict(observed)
        best = result.best_trajectory()
        assert best.shape == (8, 2)

    def test_predict_mean_trajectory(self):
        p = GoalConditionedPredictor(horizon=8, num_goals=3, num_samples_per_goal=2)
        observed = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = p.predict(observed)
        mean = result.mean_trajectory()
        assert mean.shape == (8, 2)


# ---------------------------------------------------------------------------
# IntentPredictor — construction
# ---------------------------------------------------------------------------

class TestIntentPredictorConstruction:
    def test_default_params(self):
        ip = IntentPredictor()
        assert ip.stop_speed_threshold == pytest.approx(0.1)
        assert ip.turn_curvature_threshold == pytest.approx(0.3)
        assert ip.dt == pytest.approx(0.4)

    def test_custom_params(self):
        ip = IntentPredictor(
            stop_speed_threshold=0.2,
            turn_curvature_threshold=0.5,
            crossing_lateral_threshold=0.8,
            dt=0.2,
        )
        assert ip.stop_speed_threshold == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# IntentPredictor — classify
# ---------------------------------------------------------------------------

class TestIntentClassify:
    def test_too_few_observations_returns_unknown(self):
        ip = IntentPredictor()
        obs = np.array([[0.0, 0.0], [1.0, 0.0]])  # Only 2 points
        intent, probs = ip.classify(obs)
        assert intent == PedestrianIntent.UNKNOWN
        assert PedestrianIntent.UNKNOWN.value in probs

    def test_straight_walking(self):
        """Agent moving in a straight line at constant speed."""
        ip = IntentPredictor(dt=0.1)
        # Constant velocity to the right
        obs = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
        ])
        intent, probs = ip.classify(obs)
        assert intent == PedestrianIntent.WALKING_STRAIGHT
        assert probs[PedestrianIntent.WALKING_STRAIGHT.value] > 0

    def test_stopping(self):
        """Agent decelerating to a stop."""
        ip = IntentPredictor(dt=0.1, stop_speed_threshold=0.5)
        # Decelerating
        obs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [1.7, 0.0],
            [1.72, 0.0],  # Nearly stopped, decelerating
        ])
        intent, probs = ip.classify(obs)
        assert intent in (PedestrianIntent.STOPPING, PedestrianIntent.WAITING)

    def test_waiting(self):
        """Agent that is stationary (not decelerating, just standing)."""
        ip = IntentPredictor(dt=0.1, stop_speed_threshold=0.5)
        obs = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
        ])
        intent, probs = ip.classify(obs)
        assert intent == PedestrianIntent.WAITING

    def test_turning_left(self):
        """Agent turning left (positive curvature)."""
        ip = IntentPredictor(dt=0.1, turn_curvature_threshold=0.2)
        # Curving left: moving right then curving upward
        t = np.linspace(0, np.pi / 2, 8)
        r = 3.0
        obs = np.column_stack([r * np.cos(t), r * np.sin(t)])
        intent, probs = ip.classify(obs)
        assert intent == PedestrianIntent.TURNING_LEFT

    def test_turning_right(self):
        """Agent turning right (negative curvature)."""
        ip = IntentPredictor(dt=0.1, turn_curvature_threshold=0.2)
        # Curving right: moving right then curving downward
        t = np.linspace(0, np.pi / 2, 8)
        r = 3.0
        obs = np.column_stack([r * np.cos(t), -r * np.sin(t)])
        intent, probs = ip.classify(obs)
        assert intent == PedestrianIntent.TURNING_RIGHT

    def test_crossing(self):
        """Agent crossing laterally relative to road direction."""
        ip = IntentPredictor(dt=0.1, crossing_lateral_threshold=0.3)
        # Road goes in x direction, agent crosses in y direction
        obs = np.array([
            [5.0, 0.0],
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
        ])
        context = {"road_direction": np.array([1.0, 0.0])}
        intent, probs = ip.classify(obs, context)
        assert intent == PedestrianIntent.CROSSING

    def test_classify_probabilities_sum_to_one(self):
        ip = IntentPredictor(dt=0.1)
        obs = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        _, probs = ip.classify(obs)
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_classify_with_default_road_direction(self):
        """When no road direction in context, should use default [1, 0]."""
        ip = IntentPredictor(dt=0.1)
        obs = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        intent, probs = ip.classify(obs)
        assert isinstance(intent, PedestrianIntent)

    def test_classify_unnormalized_road_direction(self):
        """Road direction should be normalized internally."""
        ip = IntentPredictor(dt=0.1)
        obs = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        context = {"road_direction": np.array([100.0, 0.0])}
        intent, probs = ip.classify(obs, context)
        assert isinstance(intent, PedestrianIntent)


# ---------------------------------------------------------------------------
# PedestrianIntent enum
# ---------------------------------------------------------------------------

class TestPedestrianIntentEnum:
    def test_all_intents(self):
        expected = {
            "crossing", "waiting", "turning_left", "turning_right",
            "stopping", "walking_straight", "unknown",
        }
        actual = {i.value for i in PedestrianIntent}
        assert actual == expected

    def test_intent_from_value(self):
        assert PedestrianIntent("crossing") == PedestrianIntent.CROSSING
        assert PedestrianIntent("unknown") == PedestrianIntent.UNKNOWN


# ---------------------------------------------------------------------------
# _CandidateGoal
# ---------------------------------------------------------------------------

class TestCandidateGoal:
    def test_default_probability(self):
        g = _CandidateGoal(position=np.array([1.0, 2.0]))
        assert g.probability == 0.0

    def test_custom_probability(self):
        g = _CandidateGoal(position=np.array([1.0, 2.0]), probability=0.8)
        assert g.probability == pytest.approx(0.8)
