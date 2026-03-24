"""Tests for navirl/evaluation/ module: metrics, benchmark, comparisons, analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from navirl.data.trajectory import Trajectory
from navirl.evaluation.metrics_extended import (
    collision_rate,
    comfort_score,
    heading_change_rate,
    jerk_metric,
    minimum_separation_distance,
    path_efficiency,
    path_length,
    personal_space_violations,
    social_force_integral,
    success_rate,
    time_to_goal,
    timeout_rate,
    topological_complexity,
    velocity_smoothness,
)
from navirl.evaluation.benchmark import (
    BenchmarkResults,
    BenchmarkSuite,
)
from navirl.evaluation.comparisons import AgentComparison
from navirl.evaluation.analysis import failure_analysis, trajectory_clustering


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_trajectory():
    """Agent moving along x-axis at constant speed."""
    n = 20
    ts = np.arange(n, dtype=np.float64) * 0.5
    pos = np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n)])
    return Trajectory(timestamps=ts, positions=pos, agent_id="robot")


@pytest.fixture
def zigzag_trajectory():
    """Agent zigzagging to introduce jerk."""
    n = 40
    ts = np.arange(n, dtype=np.float64) * 0.25
    x = np.arange(n, dtype=np.float64) * 0.5
    y = np.sin(np.arange(n) * 0.5) * 2.0
    pos = np.column_stack([x, y])
    return Trajectory(timestamps=ts, positions=pos, agent_id="robot")


@pytest.fixture
def pedestrian_trajectory():
    """A pedestrian trajectory near the robot."""
    n = 20
    ts = np.arange(n, dtype=np.float64) * 0.5
    pos = np.column_stack([np.arange(n, dtype=np.float64), np.ones(n) * 0.3])
    return Trajectory(timestamps=ts, positions=pos, agent_id="ped1")


# ---------------------------------------------------------------------------
# time_to_goal
# ---------------------------------------------------------------------------

class TestTimeToGoal:
    def test_goal_reached(self, straight_trajectory):
        ttg = time_to_goal(straight_trajectory, goal=np.array([5.0, 0.0]), threshold=0.5)
        assert ttg < float("inf")
        assert ttg >= 0.0

    def test_goal_not_reached(self, straight_trajectory):
        ttg = time_to_goal(straight_trajectory, goal=np.array([100.0, 100.0]), threshold=0.1)
        assert ttg == float("inf")

    def test_goal_at_start(self, straight_trajectory):
        ttg = time_to_goal(straight_trajectory, goal=np.array([0.0, 0.0]), threshold=0.5)
        assert ttg == pytest.approx(0.0)

    @pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0, 5.0])
    def test_varying_thresholds(self, straight_trajectory, threshold):
        ttg = time_to_goal(straight_trajectory, goal=np.array([10.0, 0.0]), threshold=threshold)
        assert isinstance(ttg, float)


# ---------------------------------------------------------------------------
# path_length / path_efficiency
# ---------------------------------------------------------------------------

class TestPathMetrics:
    def test_path_length_straight(self, straight_trajectory):
        pl = path_length(straight_trajectory)
        assert pl == pytest.approx(19.0, abs=1e-9)

    def test_path_length_single_point(self):
        t = Trajectory(timestamps=[0], positions=[[0, 0]])
        assert path_length(t) == 0.0

    def test_path_efficiency_perfect(self, straight_trajectory):
        optimal = path_length(straight_trajectory)
        eff = path_efficiency(straight_trajectory, optimal_length=optimal)
        assert eff == pytest.approx(1.0)

    def test_path_efficiency_longer_path(self, zigzag_trajectory):
        eff = path_efficiency(zigzag_trajectory, optimal_length=1.0)
        assert 0.0 < eff <= 1.0

    def test_path_efficiency_zero_length(self):
        t = Trajectory(timestamps=[0], positions=[[0, 0]])
        assert path_efficiency(t, optimal_length=5.0) == 0.0


# ---------------------------------------------------------------------------
# Jerk and smoothness
# ---------------------------------------------------------------------------

class TestSmoothness:
    def test_jerk_straight_line(self, straight_trajectory):
        j = jerk_metric(straight_trajectory)
        assert j == pytest.approx(0.0, abs=1e-6)

    def test_jerk_zigzag(self, zigzag_trajectory):
        j = jerk_metric(zigzag_trajectory)
        assert j > 0.0

    def test_jerk_short_trajectory(self):
        t = Trajectory(timestamps=[0, 1, 2], positions=[[0, 0], [1, 0], [2, 0]])
        assert jerk_metric(t) == 0.0  # < 4 points

    def test_velocity_smoothness_constant(self, straight_trajectory):
        vs = velocity_smoothness(straight_trajectory)
        assert vs == pytest.approx(0.0, abs=1e-9)

    def test_velocity_smoothness_varying(self, zigzag_trajectory):
        vs = velocity_smoothness(zigzag_trajectory)
        assert vs > 0.0

    def test_heading_change_rate_straight(self, straight_trajectory):
        hcr = heading_change_rate(straight_trajectory)
        assert hcr == pytest.approx(0.0, abs=1e-9)

    def test_heading_change_rate_zigzag(self, zigzag_trajectory):
        hcr = heading_change_rate(zigzag_trajectory)
        assert hcr > 0.0

    def test_heading_change_short(self):
        t = Trajectory(timestamps=[0, 1], positions=[[0, 0], [1, 0]])
        assert heading_change_rate(t) == 0.0


# ---------------------------------------------------------------------------
# Social metrics
# ---------------------------------------------------------------------------

class TestSocialMetrics:
    def test_personal_space_violations(self, straight_trajectory, pedestrian_trajectory):
        violations = personal_space_violations(
            straight_trajectory, [pedestrian_trajectory], threshold=0.5
        )
        assert violations > 0

    def test_personal_space_no_violations(self, straight_trajectory):
        far_ped = Trajectory(
            timestamps=np.arange(20) * 0.5,
            positions=np.column_stack([np.arange(20.0), np.ones(20) * 100]),
        )
        violations = personal_space_violations(straight_trajectory, [far_ped], threshold=0.5)
        assert violations == 0

    def test_minimum_separation_distance(self, straight_trajectory, pedestrian_trajectory):
        min_dist = minimum_separation_distance(straight_trajectory, [pedestrian_trajectory])
        assert min_dist < float("inf")
        assert min_dist >= 0.0

    def test_minimum_separation_no_peds(self, straight_trajectory):
        assert minimum_separation_distance(straight_trajectory, []) == float("inf")

    def test_social_force_integral(self, straight_trajectory, pedestrian_trajectory):
        sfi = social_force_integral(straight_trajectory, [pedestrian_trajectory])
        assert sfi > 0.0

    def test_social_force_no_peds(self, straight_trajectory):
        assert social_force_integral(straight_trajectory, []) == 0.0

    def test_topological_complexity(self, straight_trajectory):
        obstacles = np.array([[5.0, 0.5], [10.0, 0.5]])
        tc = topological_complexity(straight_trajectory, obstacles, proximity_threshold=1.0)
        assert tc >= 0

    def test_topological_complexity_no_obstacles(self, straight_trajectory):
        assert topological_complexity(straight_trajectory, np.empty((0, 2))) == 0


# ---------------------------------------------------------------------------
# Event-based metrics
# ---------------------------------------------------------------------------

class TestEventMetrics:
    def test_collision_rate_none(self):
        events = [{"type": "info"}, {"type": "info"}]
        assert collision_rate(events) == 0.0

    def test_collision_rate_half(self):
        events = [{"type": "collision"}, {"type": "info"}]
        assert collision_rate(events) == pytest.approx(0.5)

    def test_collision_rate_empty(self):
        assert collision_rate([]) == 0.0

    def test_success_rate_all(self):
        episodes = [{"success": True}, {"success": True}]
        assert success_rate(episodes) == 1.0

    def test_success_rate_none(self):
        episodes = [{"success": False}, {"success": False}]
        assert success_rate(episodes) == 0.0

    def test_success_rate_empty(self):
        assert success_rate([]) == 0.0

    def test_timeout_rate(self):
        episodes = [{"timeout": True}, {"timeout": False}, {"timeout": True}]
        assert timeout_rate(episodes) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Comfort score
# ---------------------------------------------------------------------------

class TestComfortScore:
    def test_comfort_score_range(self, straight_trajectory, pedestrian_trajectory):
        score = comfort_score(straight_trajectory, [pedestrian_trajectory])
        assert 0.0 <= score <= 1.0

    def test_comfort_score_no_peds(self, straight_trajectory):
        score = comfort_score(straight_trajectory, [])
        assert 0.0 <= score <= 1.0

    def test_comfort_score_zero_weights(self, straight_trajectory):
        score = comfort_score(
            straight_trajectory, [],
            jerk_weight=0, space_weight=0, heading_weight=0,
        )
        assert score == 1.0


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    def test_from_predefined(self):
        suite = BenchmarkSuite.from_predefined("basic")
        assert len(suite.scenarios) == 3

    def test_from_predefined_invalid(self):
        with pytest.raises(ValueError, match="Unknown suite"):
            BenchmarkSuite.from_predefined("nonexistent")

    def test_available_suites(self):
        suites = BenchmarkSuite.available_suites()
        assert "basic" in suites
        assert "crowd" in suites

    def test_run_suite(self):
        """Run a suite with mock agent and scenario."""

        class MockAgent:
            def reset(self):
                pass
            def act(self, obs):
                return 0

        class MockScenario:
            def __init__(self):
                self._step = 0
            def reset(self):
                self._step = 0
                return np.zeros(4)
            def step(self, action):
                self._step += 1
                done = self._step >= 3
                return np.zeros(4), 1.0, done, {"success": done}

        suite = BenchmarkSuite(scenarios=[{"name": "test_scenario"}])
        results = suite.run(
            agent=MockAgent(),
            scenario_factory=lambda cfg: MockScenario(),
            n_episodes=2,
            max_steps=10,
        )
        assert "success_rate" in results.metrics
        assert len(results.scenario_names) == 1


# ---------------------------------------------------------------------------
# BenchmarkResults
# ---------------------------------------------------------------------------

class TestBenchmarkResults:
    def test_to_table(self):
        results = BenchmarkResults(
            suite_name="test",
            scenario_names=["s1", "s2"],
            metrics={"success_rate": [0.8, 0.9], "reward": [10.0, 15.0]},
        )
        table = results.to_table()
        assert "s1" in table
        assert "MEAN" in table

    def test_to_table_empty(self):
        results = BenchmarkResults()
        assert results.to_table() == "(no results)"

    def test_to_latex(self):
        results = BenchmarkResults(
            suite_name="test",
            scenario_names=["s1"],
            metrics={"success_rate": [0.5]},
        )
        latex = results.to_latex()
        assert "tabular" in latex
        assert "s1" in latex

    def test_to_latex_empty(self):
        results = BenchmarkResults()
        assert "No results" in results.to_latex()


# ---------------------------------------------------------------------------
# AgentComparison
# ---------------------------------------------------------------------------

class TestAgentComparison:
    def _make_results(self, name, values):
        return BenchmarkResults(
            suite_name=name,
            scenario_names=["s1", "s2", "s3"],
            metrics={"mean_reward": values},
        )

    def test_generate_report(self):
        comp = AgentComparison()
        results = {
            "DQN": self._make_results("DQN", [1.0, 2.0, 3.0]),
            "PPO": self._make_results("PPO", [2.0, 3.0, 4.0]),
        }
        report = comp.generate_report(results)
        assert "DQN" in report
        assert "PPO" in report
        assert "Best Agent" in report


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

class TestFailureAnalysis:
    def test_all_success(self):
        episodes = [{"success": True}, {"success": True}]
        cats = failure_analysis(episodes)
        assert len(cats["success"]) == 2
        assert len(cats["collision"]) == 0

    def test_mixed_failures(self):
        episodes = [
            {"success": False, "collision": True},
            {"success": False, "timeout": True},
            {"success": False, "info": {"stuck": True}},
            {"success": False},
            {"success": True},
        ]
        cats = failure_analysis(episodes)
        assert len(cats["collision"]) == 1
        assert len(cats["timeout"]) == 1
        assert len(cats["stuck"]) == 1
        assert len(cats["other"]) == 1
        assert len(cats["success"]) == 1

    def test_empty_episodes(self):
        cats = failure_analysis([])
        assert all(len(v) == 0 for v in cats.values())


# ---------------------------------------------------------------------------
# Trajectory clustering
# ---------------------------------------------------------------------------

class TestTrajectoryClustering:
    def test_basic_clustering(self):
        trajs = []
        for _ in range(10):
            pos = np.random.randn(20, 2) + np.array([0, 0])
            trajs.append(Trajectory(timestamps=np.arange(20) * 0.1, positions=pos))
        for _ in range(10):
            pos = np.random.randn(20, 2) + np.array([50, 50])
            trajs.append(Trajectory(timestamps=np.arange(20) * 0.1, positions=pos))
        labels = trajectory_clustering(trajs, n_clusters=2, seed=42)
        assert labels.shape == (20,)
        assert len(set(labels)) == 2

    def test_empty_trajectories(self):
        labels = trajectory_clustering([], n_clusters=3)
        assert len(labels) == 0

    def test_single_trajectory(self):
        t = Trajectory(timestamps=np.arange(5) * 0.1, positions=np.zeros((5, 2)))
        labels = trajectory_clustering([t], n_clusters=1)
        assert labels.shape == (1,)
