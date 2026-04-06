"""Aggregation of metrics across batch experiment runs.

Collects per-run metric dictionaries, groups them by scenario and seed,
and produces machine-readable JSON plus a publication-friendly Markdown
summary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Metric keys that are always aggregated when present.
DEFAULT_METRIC_KEYS: list[str] = [
    "success_rate",
    "collisions_agent_agent",
    "collisions_agent_obstacle",
    "intrusion_rate",
    "min_dist_robot_human_min",
    "min_dist_robot_human_mean",
    "oscillation_score",
    "jerk_proxy",
    "path_length_robot",
    "time_to_goal_robot",
    "deadlock_count",
]


@dataclass
class RunRecord:
    """A single run's metadata and metrics."""

    scenario: str
    seed: int
    overrides: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None


@dataclass
class ScenarioStats:
    """Aggregated statistics for a single scenario across seeds."""

    scenario: str
    num_runs: int
    num_successes: int
    metrics: dict[str, dict[str, float]]  # metric_name -> {mean, std, min, max, median}


@dataclass
class BatchSummary:
    """Top-level summary of a batch experiment."""

    template_name: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    per_scenario: list[ScenarioStats]
    global_metrics: dict[str, dict[str, float]]
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "template_name": self.template_name,
            "timestamp": self.timestamp,
            "total_runs": self.total_runs,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "per_scenario": [
                {
                    "scenario": s.scenario,
                    "num_runs": s.num_runs,
                    "num_successes": s.num_successes,
                    "metrics": s.metrics,
                }
                for s in self.per_scenario
            ],
            "global_metrics": self.global_metrics,
        }


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of finite values."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "median": float("nan")}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


class BatchAggregator:
    """Collects run records and produces aggregated summaries."""

    def __init__(self, template_name: str = "") -> None:
        self.template_name = template_name
        self.records: list[RunRecord] = []

    def add_record(self, record: RunRecord) -> None:
        """Add a single run record."""
        self.records.append(record)

    def summarize(self, metric_keys: list[str] | None = None) -> BatchSummary:
        """Aggregate all collected records into a :class:`BatchSummary`.

        Parameters
        ----------
        metric_keys:
            Which metric keys to aggregate.  Defaults to
            :data:`DEFAULT_METRIC_KEYS`.
        """
        keys = metric_keys or DEFAULT_METRIC_KEYS
        completed = [r for r in self.records if r.status == "completed"]
        failed = [r for r in self.records if r.status == "failed"]

        # Group by scenario name
        by_scenario: dict[str, list[RunRecord]] = {}
        for rec in completed:
            by_scenario.setdefault(rec.scenario, []).append(rec)

        per_scenario: list[ScenarioStats] = []
        all_values: dict[str, list[float]] = {k: [] for k in keys}

        for scenario_name in sorted(by_scenario):
            runs = by_scenario[scenario_name]
            n_success = sum(1 for r in runs if r.metrics.get("success_rate", 0.0) >= 1.0)
            scenario_metrics: dict[str, dict[str, float]] = {}
            for key in keys:
                vals = [
                    float(r.metrics[key])
                    for r in runs
                    if key in r.metrics and isinstance(r.metrics[key], (int, float))
                ]
                if vals:
                    scenario_metrics[key] = _compute_stats(vals)
                    all_values[key].extend(vals)
            per_scenario.append(
                ScenarioStats(
                    scenario=scenario_name,
                    num_runs=len(runs),
                    num_successes=n_success,
                    metrics=scenario_metrics,
                )
            )

        global_metrics = {k: _compute_stats(v) for k, v in all_values.items() if v}

        return BatchSummary(
            template_name=self.template_name,
            total_runs=len(self.records),
            completed_runs=len(completed),
            failed_runs=len(failed),
            per_scenario=per_scenario,
            global_metrics=global_metrics,
            timestamp=datetime.now(UTC).isoformat(),
        )


# -----------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------


def write_json_summary(summary: BatchSummary, path: str | Path) -> None:
    """Write the batch summary as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)


def write_markdown_summary(summary: BatchSummary, path: str | Path) -> None:
    """Write a publication-friendly Markdown report of the batch results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Batch Experiment Report: {summary.template_name}")
    lines.append("")
    lines.append(f"**Generated:** {summary.timestamp}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total runs | {summary.total_runs} |")
    lines.append(f"| Completed | {summary.completed_runs} |")
    lines.append(f"| Failed | {summary.failed_runs} |")
    lines.append("")

    # Global metrics table
    if summary.global_metrics:
        lines.append("## Aggregated Metrics (all scenarios)")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | Max | Median |")
        lines.append("|--------|------|-----|-----|-----|--------|")
        for key, stats in sorted(summary.global_metrics.items()):
            lines.append(
                f"| {key} | {stats['mean']:.4f} | {stats['std']:.4f} "
                f"| {stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} |"
            )
        lines.append("")

    # Per-scenario breakdown
    if summary.per_scenario:
        lines.append("## Per-Scenario Results")
        lines.append("")
        for sc in summary.per_scenario:
            lines.append(f"### {sc.scenario}")
            lines.append("")
            lines.append(
                f"Runs: {sc.num_runs} | Successes: {sc.num_successes} "
                f"| Success rate: {sc.num_successes / max(sc.num_runs, 1):.1%}"
            )
            lines.append("")
            if sc.metrics:
                lines.append("| Metric | Mean | Std | Min | Max |")
                lines.append("|--------|------|-----|-----|-----|")
                for key, stats in sorted(sc.metrics.items()):
                    lines.append(
                        f"| {key} | {stats['mean']:.4f} | {stats['std']:.4f} "
                        f"| {stats['min']:.4f} | {stats['max']:.4f} |"
                    )
                lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
