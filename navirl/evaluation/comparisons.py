"""Agent comparison tools: run multiple agents on the same scenarios and compare."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from navirl.evaluation.benchmark import (
    BenchmarkAgent,
    BenchmarkResults,
    BenchmarkSuite,
)


def _try_import_scipy():
    """Lazy import for scipy.stats to keep it optional."""
    try:
        from scipy import stats

        return stats
    except ImportError:
        raise ImportError(
            "scipy is required for statistical tests. Install with: pip install scipy"
        )


class AgentComparison:
    """Compare multiple agents on the same set of scenarios.

    Example::

        comp = AgentComparison()
        results = comp.run_comparison(
            agents={"DQN": dqn_agent, "PPO": ppo_agent},
            suite=suite,
            scenario_factory=factory,
        )
        print(comp.generate_report(results))
    """

    def run_comparison(
        self,
        agents: dict[str, BenchmarkAgent],
        suite: BenchmarkSuite,
        scenario_factory: Any,
        *,
        n_episodes: int = 10,
        max_steps: int = 500,
    ) -> dict[str, BenchmarkResults]:
        """Run each agent through the benchmark suite.

        Parameters:
            agents: Mapping from agent name to agent instance.
            suite: The benchmark suite to run.
            scenario_factory: Callable that creates a scenario from a config dict.
            n_episodes: Number of episodes per scenario per agent.
            max_steps: Maximum steps per episode.

        Returns:
            Mapping from agent name to :class:`BenchmarkResults`.
        """
        all_results: dict[str, BenchmarkResults] = {}
        for name, agent in agents.items():
            res = suite.run(
                agent,
                scenario_factory,
                n_episodes=n_episodes,
                max_steps=max_steps,
            )
            res.suite_name = name
            all_results[name] = res
        return all_results

    def statistical_test(
        self,
        results: dict[str, BenchmarkResults],
        test: str = "wilcoxon",
        metric: str = "mean_reward",
    ) -> dict[tuple[str, str], float]:
        """Perform pairwise statistical tests between agents.

        Parameters:
            results: Mapping from agent name to results (from :meth:`run_comparison`).
            test: Statistical test to use.  Currently supports ``"wilcoxon"``
                (Wilcoxon signed-rank) and ``"mannwhitneyu"`` (Mann-Whitney U).
            metric: Which metric to compare.

        Returns:
            Mapping from agent-name pairs ``(a, b)`` to p-values.
        """
        stats = _try_import_scipy()
        names = sorted(results.keys())
        p_values: dict[tuple[str, str], float] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                vals_a = np.array(results[a].metrics.get(metric, []))
                vals_b = np.array(results[b].metrics.get(metric, []))
                min_len = min(len(vals_a), len(vals_b))
                if min_len < 2:
                    p_values[(a, b)] = float("nan")
                    continue
                vals_a = vals_a[:min_len]
                vals_b = vals_b[:min_len]
                if test == "wilcoxon":
                    _, p = stats.wilcoxon(vals_a, vals_b)
                elif test == "mannwhitneyu":
                    _, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                else:
                    raise ValueError(f"Unknown test '{test}'. Use 'wilcoxon' or 'mannwhitneyu'.")
                p_values[(a, b)] = float(p)
        return p_values

    def plot_comparison(
        self,
        results: dict[str, BenchmarkResults],
        metrics: Sequence[str] | None = None,
    ) -> Any:
        """Create grouped bar charts comparing agents across metrics.

        Parameters:
            results: Agent results from :meth:`run_comparison`.
            metrics: Subset of metrics to plot.  Defaults to all available.

        Returns:
            A :class:`matplotlib.figure.Figure`.
        """
        import matplotlib.pyplot as plt

        agent_names = sorted(results.keys())
        if metrics is None:
            all_m: set[str] = set()
            for res in results.values():
                all_m.update(res.metrics.keys())
            metrics = sorted(all_m)

        n_agents = len(agent_names)
        n_metrics = len(metrics)
        x = np.arange(n_metrics)
        width = 0.8 / max(n_agents, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))
        for idx, agent_name in enumerate(agent_names):
            res = results[agent_name]
            means = [
                float(np.mean(res.metrics.get(m, [0.0]))) for m in metrics
            ]
            offset = (idx - n_agents / 2 + 0.5) * width
            ax.bar(x + offset, means, width, label=agent_name)

        ax.set_xticks(x)
        ax.set_xticklabels(list(metrics), rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("Agent Comparison")
        ax.legend()
        fig.tight_layout()
        return fig

    def generate_report(
        self,
        results: dict[str, BenchmarkResults],
        precision: int = 3,
    ) -> str:
        """Generate a Markdown-formatted comparison report.

        Parameters:
            results: Agent results from :meth:`run_comparison`.
            precision: Number of decimal places.

        Returns:
            Markdown report string.
        """
        lines: list[str] = ["# Agent Comparison Report", ""]
        agent_names = sorted(results.keys())

        # Gather all metric names.
        all_metrics: set[str] = set()
        for res in results.values():
            all_metrics.update(res.metrics.keys())
        metric_names = sorted(all_metrics)

        # Header row
        header = "| Metric | " + " | ".join(agent_names) + " |"
        sep = "|" + "---|" * (len(agent_names) + 1)
        lines.extend([header, sep])

        for m in metric_names:
            row = f"| {m} "
            for name in agent_names:
                vals = results[name].metrics.get(m, [])
                mean = float(np.mean(vals)) if vals else float("nan")
                std = float(np.std(vals)) if vals else float("nan")
                row += f"| {mean:.{precision}f} +/- {std:.{precision}f} "
            row += "|"
            lines.append(row)

        lines.append("")

        # Best agent per metric
        lines.append("## Best Agent per Metric")
        lines.append("")
        for m in metric_names:
            best_name = ""
            best_val = -float("inf")
            for name in agent_names:
                vals = results[name].metrics.get(m, [])
                mean = float(np.mean(vals)) if vals else -float("inf")
                if mean > best_val:
                    best_val = mean
                    best_name = name
            lines.append(f"- **{m}**: {best_name} ({best_val:.{precision}f})")

        return "\n".join(lines)
