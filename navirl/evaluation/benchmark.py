"""Benchmarking framework: predefined suites and aggregated results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class BenchmarkAgent(Protocol):
    """Minimal protocol an agent must satisfy to be benchmarked."""

    def reset(self) -> None: ...
    def act(self, observation: Any) -> Any: ...


class BenchmarkScenario(Protocol):
    """Minimal protocol for a scenario used in benchmarking."""

    def reset(self) -> Any: ...
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]: ...


# -----------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------


@dataclass
class BenchmarkResults:
    """Aggregated results from a benchmark run.

    Attributes:
        suite_name: Name of the benchmark suite that produced these results.
        scenario_names: List of scenario identifiers.
        metrics: Mapping from metric name to a list of per-scenario values.
        raw_episodes: Optional list of per-episode result dicts.
    """

    suite_name: str = ""
    scenario_names: list[str] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    raw_episodes: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def to_table(self, precision: int = 3) -> str:
        """Render results as a human-readable ASCII table.

        Parameters:
            precision: Number of decimal places.

        Returns:
            Formatted table string.
        """
        if not self.metrics:
            return "(no results)"
        metric_names = sorted(self.metrics.keys())
        col_widths = [max(12, len(n) + 2) for n in metric_names]
        header = "Metric".ljust(20) + "".join(
            n.ljust(w) for n, w in zip(metric_names, col_widths, strict=False)
        )
        sep = "-" * len(header)
        rows: list[str] = [header, sep]

        # Per-scenario rows
        for s_idx, s_name in enumerate(self.scenario_names):
            row = s_name[:18].ljust(20)
            for m_name, w in zip(metric_names, col_widths, strict=False):
                vals = self.metrics.get(m_name, [])
                val = vals[s_idx] if s_idx < len(vals) else float("nan")
                row += f"{val:.{precision}f}".ljust(w)
            rows.append(row)

        # Summary row
        rows.append(sep)
        summary = "MEAN".ljust(20)
        for m_name, w in zip(metric_names, col_widths, strict=False):
            vals = self.metrics.get(m_name, [])
            mean_val = float(np.mean(vals)) if vals else float("nan")
            summary += f"{mean_val:.{precision}f}".ljust(w)
        rows.append(summary)
        return "\n".join(rows)

    def to_latex(self, precision: int = 3) -> str:
        """Render results as a LaTeX tabular environment.

        Parameters:
            precision: Number of decimal places.

        Returns:
            LaTeX table string.
        """
        if not self.metrics:
            return "% No results"
        metric_names = sorted(self.metrics.keys())
        len(metric_names) + 1
        lines = [
            r"\begin{tabular}{" + "l" + "r" * len(metric_names) + "}",
            r"\toprule",
            "Scenario & " + " & ".join(metric_names) + r" \\",
            r"\midrule",
        ]
        for s_idx, s_name in enumerate(self.scenario_names):
            vals_str = []
            for m_name in metric_names:
                vals = self.metrics.get(m_name, [])
                v = vals[s_idx] if s_idx < len(vals) else float("nan")
                vals_str.append(f"{v:.{precision}f}")
            lines.append(s_name + " & " + " & ".join(vals_str) + r" \\")
        lines.append(r"\midrule")
        mean_strs = []
        for m_name in metric_names:
            vals = self.metrics.get(m_name, [])
            mean_strs.append(f"{float(np.mean(vals)):.{precision}f}" if vals else "---")
        lines.append("Mean & " + " & ".join(mean_strs) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def plot_comparison(self, other: BenchmarkResults | None = None) -> Any:
        """Plot a bar chart comparing metrics.  Returns a matplotlib Figure.

        Parameters:
            other: Optional second result set for side-by-side comparison.

        Returns:
            A :class:`matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        metric_names = sorted(self.metrics.keys())
        means = [float(np.mean(self.metrics[m])) for m in metric_names]
        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 1.2), 5))
        ax.bar(x - width / 2, means, width, label=self.suite_name or "A")
        if other is not None:
            other_means = [float(np.mean(other.metrics.get(m, [0.0]))) for m in metric_names]
            ax.bar(x + width / 2, other_means, width, label=other.suite_name or "B")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("Benchmark Comparison")
        ax.legend()
        fig.tight_layout()
        return fig


# -----------------------------------------------------------------------
# Predefined suite definitions
# -----------------------------------------------------------------------

_PREDEFINED_SUITES: dict[str, list[dict[str, Any]]] = {
    "basic": [
        {"name": "straight_line", "n_pedestrians": 0, "goal_distance": 10.0},
        {"name": "single_obstacle", "n_pedestrians": 0, "goal_distance": 10.0},
        {"name": "corridor", "n_pedestrians": 0, "goal_distance": 15.0},
    ],
    "crowd": [
        {"name": "sparse_crowd", "n_pedestrians": 5, "goal_distance": 10.0},
        {"name": "dense_crowd", "n_pedestrians": 20, "goal_distance": 10.0},
        {"name": "crossing_flow", "n_pedestrians": 10, "goal_distance": 12.0},
    ],
    "social": [
        {"name": "pass_on_right", "n_pedestrians": 1, "goal_distance": 8.0},
        {"name": "group_avoidance", "n_pedestrians": 4, "goal_distance": 10.0},
        {"name": "doorway", "n_pedestrians": 3, "goal_distance": 6.0},
        {"name": "narrow_hallway", "n_pedestrians": 2, "goal_distance": 12.0},
    ],
    "adversarial": [
        {"name": "head_on", "n_pedestrians": 1, "goal_distance": 8.0},
        {"name": "sudden_stop", "n_pedestrians": 1, "goal_distance": 8.0},
        {"name": "squeeze", "n_pedestrians": 6, "goal_distance": 6.0},
        {"name": "random_walk", "n_pedestrians": 10, "goal_distance": 10.0},
    ],
}


class BenchmarkSuite:
    """Collection of scenarios for standardised agent evaluation.

    Provides predefined suites (``"basic"``, ``"crowd"``, ``"social"``,
    ``"adversarial"``) and supports custom scenario lists.

    Parameters:
        scenarios: Explicit list of scenario config dicts.  If ``None``,
            use :meth:`from_predefined` to load a named suite.
    """

    def __init__(
        self,
        scenarios: list[dict[str, Any]] | None = None,
    ) -> None:
        self.scenarios: list[dict[str, Any]] = scenarios or []

    @classmethod
    def from_predefined(cls, suite_name: str) -> BenchmarkSuite:
        """Load a predefined benchmark suite by name.

        Parameters:
            suite_name: One of ``"basic"``, ``"crowd"``, ``"social"``, ``"adversarial"``.

        Returns:
            A :class:`BenchmarkSuite` instance.
        """
        if suite_name not in _PREDEFINED_SUITES:
            available = ", ".join(sorted(_PREDEFINED_SUITES.keys()))
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")
        return cls(scenarios=list(_PREDEFINED_SUITES[suite_name]))

    @classmethod
    def available_suites(cls) -> list[str]:
        """Return names of all predefined suites."""
        return sorted(_PREDEFINED_SUITES.keys())

    def run(
        self,
        agent: BenchmarkAgent,
        scenario_factory: Any,
        *,
        n_episodes: int = 10,
        max_steps: int = 500,
    ) -> BenchmarkResults:
        """Run all scenarios in the suite and collect metrics.

        Parameters:
            agent: The agent to evaluate (must satisfy :class:`BenchmarkAgent` protocol).
            scenario_factory: Callable that takes a scenario config dict and returns
                a :class:`BenchmarkScenario`.
            n_episodes: Number of episodes per scenario.
            max_steps: Maximum steps per episode.

        Returns:
            Aggregated :class:`BenchmarkResults`.
        """

        results = BenchmarkResults(suite_name="custom")
        all_metrics: dict[str, list[float]] = {
            "success_rate": [],
            "mean_reward": [],
            "mean_path_length": [],
            "mean_steps": [],
        }

        for sc_cfg in self.scenarios:
            sc_name = sc_cfg.get("name", "unnamed")
            results.scenario_names.append(sc_name)
            successes = 0
            rewards: list[float] = []
            steps_list: list[int] = []

            for _ep in range(n_episodes):
                env = scenario_factory(sc_cfg)
                obs = env.reset()
                agent.reset()
                total_reward = 0.0
                done = False
                step_count = 0
                for _ in range(max_steps):
                    action = agent.act(obs)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    step_count += 1
                    if done:
                        break
                if info.get("success", done):
                    successes += 1
                rewards.append(total_reward)
                steps_list.append(step_count)
                results.raw_episodes.append(
                    {
                        "scenario": sc_name,
                        "success": info.get("success", done),
                        "reward": total_reward,
                        "steps": step_count,
                    }
                )

            all_metrics["success_rate"].append(successes / max(n_episodes, 1))
            all_metrics["mean_reward"].append(float(np.mean(rewards)))
            all_metrics["mean_steps"].append(float(np.mean(steps_list)))

        results.metrics = all_metrics
        return results
