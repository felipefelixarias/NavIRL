"""Extended tests for navirl/evaluation/ — comparisons and analysis coverage gaps."""

from __future__ import annotations

import numpy as np
import pytest

from navirl.data.trajectory import Trajectory
from navirl.evaluation.analysis import (
    _get_probs,
    _get_q_values,
    _kmeans,
    attention_visualization,
    failure_analysis,
    policy_entropy_map,
    q_value_landscape,
    trajectory_clustering,
)
from navirl.evaluation.benchmark import BenchmarkResults
from navirl.evaluation.comparisons import AgentComparison


# ---------------------------------------------------------------------------
# _kmeans internal
# ---------------------------------------------------------------------------


class TestKMeans:
    def test_basic_two_clusters(self):
        # Two well-separated clusters
        rng = np.random.default_rng(0)
        c1 = rng.normal(loc=0, scale=0.1, size=(20, 2))
        c2 = rng.normal(loc=10, scale=0.1, size=(20, 2))
        X = np.vstack([c1, c2])
        labels = _kmeans(X, k=2, seed=42)
        assert labels.shape == (40,)
        assert len(set(labels)) == 2
        # All cluster-1 points should have same label
        assert len(set(labels[:20])) == 1
        assert len(set(labels[20:])) == 1
        assert labels[0] != labels[20]

    def test_k_larger_than_n(self):
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
        labels = _kmeans(X, k=10, seed=0)
        assert labels.shape == (3,)
        assert len(set(labels)) == 3  # k clamped to n

    def test_k_zero(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        labels = _kmeans(X, k=0)
        np.testing.assert_array_equal(labels, [0, 0])

    def test_single_point(self):
        X = np.array([[5.0, 5.0]])
        labels = _kmeans(X, k=1)
        assert labels.shape == (1,)

    def test_convergence(self):
        """Identical points should converge in 1 iteration."""
        X = np.ones((10, 3))
        labels = _kmeans(X, k=2, max_iter=5)
        assert labels.shape == (10,)


# ---------------------------------------------------------------------------
# _get_probs / _get_q_values helpers
# ---------------------------------------------------------------------------


class TestGetProbs:
    def test_with_get_action_probs(self):
        class Agent:
            def get_action_probs(self, obs):
                return np.array([0.5, 0.5])

        result = _get_probs(Agent(), np.zeros(4))
        np.testing.assert_array_equal(result, [0.5, 0.5])

    def test_with_predict_proba(self):
        class Agent:
            def predict_proba(self, obs):
                return np.array([0.3, 0.7])

        result = _get_probs(Agent(), np.zeros(4))
        np.testing.assert_array_equal(result, [0.3, 0.7])

    def test_no_method(self):
        class Agent:
            pass

        assert _get_probs(Agent(), np.zeros(4)) is None


class TestGetQValues:
    def test_with_get_q_values(self):
        class Agent:
            def get_q_values(self, obs):
                return np.array([1.0, 2.0, 3.0])

        result = _get_q_values(Agent(), np.zeros(4))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_fallback(self):
        class Agent:
            pass

        result = _get_q_values(Agent(), np.zeros(4))
        np.testing.assert_array_equal(result, [0.0])


# ---------------------------------------------------------------------------
# policy_entropy_map
# ---------------------------------------------------------------------------


class TestPolicyEntropyMap:
    def test_uniform_distribution(self):
        class Agent:
            def get_action_probs(self, obs):
                return np.array([0.25, 0.25, 0.25, 0.25])

        grid = np.zeros((5, 4))
        entropies = policy_entropy_map(Agent(), grid)
        assert entropies.shape == (5,)
        # Uniform distribution over 4 actions => entropy = ln(4) ≈ 1.386
        expected = -4 * (0.25 * np.log(0.25))
        np.testing.assert_allclose(entropies, expected, atol=1e-6)

    def test_deterministic_distribution(self):
        class Agent:
            def get_action_probs(self, obs):
                return np.array([1.0, 0.0, 0.0])

        grid = np.zeros((3, 2))
        entropies = policy_entropy_map(Agent(), grid)
        assert entropies.shape == (3,)
        # Near-zero entropy for deterministic policy
        assert all(e < 0.01 for e in entropies)

    def test_no_probs_method(self):
        class Agent:
            pass

        grid = np.zeros((2, 3))
        entropies = policy_entropy_map(Agent(), grid)
        np.testing.assert_array_equal(entropies, [0.0, 0.0])


# ---------------------------------------------------------------------------
# q_value_landscape
# ---------------------------------------------------------------------------


class TestQValueLandscape:
    def test_basic(self):
        class Agent:
            def get_q_values(self, obs):
                return obs * 2.0

        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = q_value_landscape(Agent(), grid)
        np.testing.assert_array_equal(result, [[2.0, 4.0], [6.0, 8.0]])

    def test_fallback_agent(self):
        class Agent:
            pass

        grid = np.zeros((3, 2))
        result = q_value_landscape(Agent(), grid)
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# attention_visualization (without PyTorch)
# ---------------------------------------------------------------------------


class TestAttentionVisualization:
    def test_no_model(self):
        class Agent:
            pass

        result = attention_visualization(Agent(), np.zeros(4))
        # Should return dummy uniform
        np.testing.assert_array_equal(result, [1.0])

    def test_model_no_attention_layer(self):
        """Agent has a model attr but no attention layer."""

        class FakeModel:
            def named_modules(self):
                return iter([("fc1", None), ("fc2", None)])

        class Agent:
            model = FakeModel()

        try:
            import torch  # noqa: F401

            result = attention_visualization(Agent(), np.zeros(4))
            np.testing.assert_array_equal(result, [1.0])
        except ImportError:
            result = attention_visualization(Agent(), np.zeros(4))
            np.testing.assert_array_equal(result, [1.0])


# ---------------------------------------------------------------------------
# trajectory_clustering (extended)
# ---------------------------------------------------------------------------


class TestTrajectoryClustering:
    def test_short_trajectories(self):
        """Single-point trajectories should still cluster."""
        trajs = [
            Trajectory(timestamps=[0], positions=[[0, 0]]),
            Trajectory(timestamps=[0], positions=[[100, 100]]),
        ]
        labels = trajectory_clustering(trajs, n_clusters=2)
        assert labels.shape == (2,)

    def test_many_clusters(self):
        rng = np.random.default_rng(42)
        trajs = []
        for i in range(15):
            pos = rng.normal(loc=i * 10, scale=0.1, size=(10, 2))
            trajs.append(Trajectory(timestamps=np.arange(10) * 0.1, positions=pos))
        labels = trajectory_clustering(trajs, n_clusters=5)
        assert labels.shape == (15,)
        assert len(set(labels)) <= 5


# ---------------------------------------------------------------------------
# AgentComparison (extended)
# ---------------------------------------------------------------------------


class TestAgentComparisonExtended:
    def _make_results(self, name, reward_vals, sr_vals=None):
        metrics = {"mean_reward": list(reward_vals)}
        if sr_vals is not None:
            metrics["success_rate"] = list(sr_vals)
        return BenchmarkResults(
            suite_name=name,
            scenario_names=[f"s{i}" for i in range(len(reward_vals))],
            metrics=metrics,
        )

    def test_plot_comparison(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0, 2.0], [0.8, 0.9]),
            "B": self._make_results("B", [3.0, 4.0], [0.7, 0.6]),
        }
        fig = comp.plot_comparison(results)
        assert fig is not None
        plt.close("all")

    def test_plot_comparison_subset_metrics(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0], [0.5]),
            "B": self._make_results("B", [2.0], [0.6]),
        }
        fig = comp.plot_comparison(results, metrics=["mean_reward"])
        assert fig is not None
        plt.close("all")

    def test_generate_report_multiple_metrics(self):
        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0, 2.0], [0.8, 1.0]),
            "B": self._make_results("B", [3.0, 4.0], [0.5, 0.6]),
        }
        report = comp.generate_report(results)
        assert "mean_reward" in report
        assert "success_rate" in report
        assert "Best Agent" in report

    def test_generate_report_precision(self):
        comp = AgentComparison()
        results = {
            "X": self._make_results("X", [1.23456789]),
        }
        report = comp.generate_report(results, precision=2)
        assert "1.23" in report

    def test_run_comparison(self):
        """Integration test for run_comparison with mock agents."""
        from navirl.evaluation.benchmark import BenchmarkSuite

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
                done = self._step >= 2
                return np.zeros(4), 1.0, done, {"success": done}

        suite = BenchmarkSuite(scenarios=[{"name": "test"}])
        comp = AgentComparison()
        results = comp.run_comparison(
            agents={"agent1": MockAgent(), "agent2": MockAgent()},
            suite=suite,
            scenario_factory=lambda cfg: MockScenario(),
            n_episodes=2,
            max_steps=5,
        )
        assert "agent1" in results
        assert "agent2" in results

    def test_statistical_test_short_data(self):
        """Insufficient data should return nan."""
        comp = AgentComparison()
        results = {
            "A": self._make_results("A", [1.0]),
            "B": self._make_results("B", [2.0]),
        }
        try:
            pvals = comp.statistical_test(results)
            assert np.isnan(pvals[("A", "B")])
        except ImportError:
            pytest.skip("scipy not installed")


# ---------------------------------------------------------------------------
# metrics/base aggregate_reports
# ---------------------------------------------------------------------------


class TestAggregateReports:
    def test_empty_reports(self):
        from navirl.metrics.base import aggregate_reports

        result = aggregate_reports([])
        assert result == {"num_reports": 0}

    def test_single_report(self):
        from navirl.metrics.base import aggregate_reports

        reports = [
            {
                "success_rate": 1.0,
                "intrusion_rate": 0.05,
                "collisions_agent_agent": 0,
                "path_length_robot": 10.5,
                "time_to_goal_robot": 8.2,
            }
        ]
        result = aggregate_reports(reports)
        assert result["num_reports"] == 1
        assert result["avg_success_rate"] == pytest.approx(1.0)
        assert result["avg_path_length_robot"] == pytest.approx(10.5)
        assert result["pass_count"] == 1

    def test_multiple_reports(self):
        from navirl.metrics.base import aggregate_reports

        reports = [
            {"success_rate": 1.0, "jerk_proxy": 0.1, "oscillation_score": 0.5},
            {"success_rate": 0.5, "jerk_proxy": 0.3, "oscillation_score": 0.2},
            {"success_rate": 0.0, "jerk_proxy": 0.2},
        ]
        result = aggregate_reports(reports)
        assert result["num_reports"] == 3
        assert result["avg_success_rate"] == pytest.approx(0.5)
        assert result["avg_jerk_proxy"] == pytest.approx(0.2)
        assert result["pass_count"] == 1  # only first has sr >= 1.0

    def test_non_numeric_values_excluded(self):
        from navirl.metrics.base import aggregate_reports

        # Non-numeric values for scalar keys are properly excluded from averages
        reports = [
            {"jerk_proxy": "high"},  # non-numeric should be skipped
        ]
        result = aggregate_reports(reports)
        assert result["num_reports"] == 1
        assert "avg_jerk_proxy" not in result

    def test_missing_keys(self):
        from navirl.metrics.base import aggregate_reports

        reports = [
            {"collisions_agent_obstacle": 2},
            {},  # empty dict
        ]
        result = aggregate_reports(reports)
        assert result["num_reports"] == 2
        assert result["avg_collisions_agent_obstacle"] == pytest.approx(2.0)
        assert result["pass_count"] == 0
