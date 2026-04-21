"""Replay a reproducibility package and diff results against expected outputs.

Loads scenario configs from a built reproducibility package, re-runs them
through the simulation pipeline, and compares the resulting metrics against
the expected values stored in the package manifest.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison of a single metric between expected and replayed values."""

    name: str
    expected_mean: float
    replayed_value: float
    tolerance: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_mean": self.expected_mean,
            "replayed_value": self.replayed_value,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class ReplayResult:
    """Result of replaying a single scenario from a reproducibility package."""

    scenario_name: str
    seed: int
    status: str = "completed"
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    comparisons: list[MetricComparison] = field(default_factory=list)

    @property
    def all_within_tolerance(self) -> bool:
        return all(c.within_tolerance for c in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "metrics": self.metrics,
            "all_within_tolerance": self.all_within_tolerance,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


@dataclass
class ReplayReport:
    """Complete report from replaying a reproducibility package."""

    package_name: str
    results: list[ReplayResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            len(self.results) > 0
            and all(r.status == "completed" for r in self.results)
            and all(r.all_within_tolerance for r in self.results)
        )

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.results if r.status == "completed")

    @property
    def within_tolerance_count(self) -> int:
        return sum(
            1
            for r in self.results
            if r.status == "completed" and r.all_within_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "passed": self.passed,
            "total_runs": self.total,
            "completed_runs": self.completed_count,
            "within_tolerance": self.within_tolerance_count,
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Replay Report: {self.package_name}",
            "",
            f"**Status**: {status} "
            f"({self.within_tolerance_count}/{self.total} within tolerance)",
            "",
            "| Scenario | Seed | Status | Within Tolerance | Details |",
            "|----------|------|--------|-----------------|---------|",
        ]
        for r in self.results:
            tol_str = "YES" if r.all_within_tolerance else "NO"
            if r.status != "completed":
                detail = r.error or "unknown error"
                lines.append(
                    f"| {r.scenario_name} | {r.seed} | {r.status} | - | {detail} |"
                )
            else:
                n_pass = sum(1 for c in r.comparisons if c.within_tolerance)
                detail = f"{n_pass}/{len(r.comparisons)} metrics match"
                lines.append(
                    f"| {r.scenario_name} | {r.seed} | completed | {tol_str} | {detail} |"
                )
        lines.append("")
        return "\n".join(lines)


def _compare_metrics(
    replayed: dict[str, float],
    expected: dict[str, dict[str, float]],
    tolerance: float,
) -> list[MetricComparison]:
    """Compare replayed metrics against expected values."""
    comparisons: list[MetricComparison] = []
    for name, stats in expected.items():
        expected_mean = stats.get("mean", float("nan"))
        if math.isnan(expected_mean):
            continue
        replayed_val = replayed.get(name, float("nan"))

        expected_std = stats.get("std", 0.0)
        effective_tol = max(tolerance, expected_std * 2.0)

        if math.isnan(replayed_val):
            within = False
        elif expected_mean == 0.0:
            within = abs(replayed_val) <= effective_tol
        else:
            within = abs(replayed_val - expected_mean) <= effective_tol

        comparisons.append(
            MetricComparison(
                name=name,
                expected_mean=expected_mean,
                replayed_value=replayed_val,
                tolerance=effective_tol,
                within_tolerance=within,
            )
        )
    return comparisons


def replay_package(
    package_dir: Path,
    out_dir: Path | None = None,
    *,
    tolerance: float = 0.1,
    seed_override: int | None = None,
) -> ReplayReport:
    """Replay scenarios from a reproducibility package and compare results.

    Loads the package manifest, re-runs each scenario through the simulation
    pipeline, and compares resulting metrics against the expected values.

    Parameters
    ----------
    package_dir:
        Root directory of the reproducibility package (contains MANIFEST.json).
    out_dir:
        Output directory for replay run artifacts. Defaults to a ``replay/``
        subdirectory under *package_dir*.
    tolerance:
        Base tolerance for metric comparisons. The effective tolerance is
        ``max(tolerance, 2 * expected_std)`` per metric.
    seed_override:
        If provided, override the seed for all replayed scenarios.

    Returns
    -------
    ReplayReport
        Full replay report with per-scenario comparisons.
    """
    manifest_path = package_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return ReplayReport(
            package_name=package_dir.name,
            results=[
                ReplayResult(
                    scenario_name="<none>",
                    seed=0,
                    status="failed",
                    error="MANIFEST.json not found",
                )
            ],
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    package_name = data.get("name", package_dir.name)
    expected_metrics = data.get("expected_metrics", {})

    if out_dir is None:
        out_dir = package_dir / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find scenarios in the package
    scenarios_dir = package_dir / "scenarios"
    if not scenarios_dir.is_dir():
        return ReplayReport(
            package_name=package_name,
            results=[
                ReplayResult(
                    scenario_name="<none>",
                    seed=0,
                    status="failed",
                    error="No scenarios/ directory found in package",
                )
            ],
        )

    scenario_files = sorted(scenarios_dir.glob("*.yaml"))
    if not scenario_files:
        return ReplayReport(
            package_name=package_name,
            results=[
                ReplayResult(
                    scenario_name="<none>",
                    seed=0,
                    status="failed",
                    error="No scenario YAML files found",
                )
            ],
        )

    # Lazy imports to avoid circular dependencies and heavy imports at module level
    import yaml

    from navirl.core.seeds import set_global_seed
    from navirl.metrics.standard import StandardMetrics
    from navirl.pipeline import run_scenario_dict

    metrics_collector = StandardMetrics()
    results: list[ReplayResult] = []

    for scenario_path in scenario_files:
        scenario_name = scenario_path.stem

        try:
            with scenario_path.open("r", encoding="utf-8") as f:
                scenario = yaml.safe_load(f)

            seed = seed_override if seed_override is not None else scenario.get("seed", 42)
            scenario["seed"] = seed
            set_global_seed(seed)

            logger.info("Replaying scenario %s (seed=%d)", scenario_name, seed)

            episode_log = run_scenario_dict(
                scenario,
                out_root=str(out_dir),
                render_override=False,
                video_override=False,
            )

            state_path = Path(episode_log.state_path)
            bundle_dir = state_path.parent
            scenario_yaml = bundle_dir / "scenario.yaml"
            with scenario_yaml.open("r", encoding="utf-8") as f:
                run_scenario = yaml.safe_load(f)

            run_metrics = metrics_collector.compute(state_path, run_scenario)

            comparisons = _compare_metrics(run_metrics, expected_metrics, tolerance)

            results.append(
                ReplayResult(
                    scenario_name=scenario_name,
                    seed=seed,
                    status="completed",
                    metrics=run_metrics,
                    comparisons=comparisons,
                )
            )

        except Exception as exc:
            logger.warning("Replay of %s failed: %s", scenario_name, exc)
            results.append(
                ReplayResult(
                    scenario_name=scenario_name,
                    seed=seed_override or 42,
                    status="failed",
                    error=str(exc),
                )
            )

    return ReplayReport(package_name=package_name, results=results)
