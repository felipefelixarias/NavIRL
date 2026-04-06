"""Batch template runner.

Executes all tasks defined by a :class:`~navirl.experiments.templates.BatchTemplate`,
collects metrics, and produces aggregated summaries.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from navirl.core.seeds import set_global_seed
from navirl.experiments.aggregator import (
    BatchAggregator,
    BatchSummary,
    RunRecord,
    write_json_summary,
    write_markdown_summary,
)
from navirl.experiments.templates import BatchTemplate
from navirl.metrics.standard import StandardMetrics
from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario

logger = logging.getLogger(__name__)


def _apply_overrides(scenario: dict, overrides: dict[str, Any]) -> dict:
    """Apply dotted-path overrides to a scenario dict.

    Example: ``{"scene.orca.neighbor_dist": 4.0}`` sets
    ``scenario["scene"]["orca"]["neighbor_dist"] = 4.0``.
    """
    scenario = copy.deepcopy(scenario)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = scenario
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return scenario


def run_batch_template(
    template: BatchTemplate,
    out_root: str | Path,
    *,
    render: bool = False,
    video: bool = False,
) -> BatchSummary:
    """Execute a batch template and return an aggregated summary.

    Parameters
    ----------
    template:
        The batch template defining scenarios, seeds, and parameter grids.
    out_root:
        Root output directory for all run bundles.
    render:
        Whether to render visuals during simulation.
    video:
        Whether to record video output.

    Returns
    -------
    BatchSummary
        Aggregated results across all runs.  JSON and Markdown reports are
        also written to ``{out_root}/summary.json`` and
        ``{out_root}/REPORT.md``.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = template.expand_tasks()
    aggregator = BatchAggregator(template_name=template.name)
    metrics_collector = StandardMetrics()

    for i, task in enumerate(tasks):
        scenario_path: Path = task["scenario"]
        seed: int = task["seed"]
        overrides: dict[str, Any] = task["overrides"]
        scenario_name = scenario_path.stem

        logger.info(
            "Running task %d/%d: %s seed=%d overrides=%s",
            i + 1,
            len(tasks),
            scenario_name,
            seed,
            overrides or "{}",
        )

        try:
            scenario = load_scenario(str(scenario_path))
            scenario["seed"] = seed
            if overrides:
                scenario = _apply_overrides(scenario, overrides)

            set_global_seed(seed)
            episode_log = run_scenario_dict(
                scenario,
                out_root=str(out_root),
                render_override=render,
                video_override=video,
            )

            state_path = Path(episode_log.state_path)
            bundle_dir = state_path.parent
            scenario_yaml = bundle_dir / "scenario.yaml"
            with scenario_yaml.open("r", encoding="utf-8") as f:
                run_scenario = yaml.safe_load(f)

            run_metrics = metrics_collector.compute(state_path, run_scenario)

            aggregator.add_record(
                RunRecord(
                    scenario=scenario_name,
                    seed=seed,
                    overrides=overrides,
                    metrics=run_metrics,
                    status="completed",
                )
            )
        except Exception as exc:
            logger.warning("Task %d failed: %s", i + 1, exc)
            aggregator.add_record(
                RunRecord(
                    scenario=scenario_name,
                    seed=seed,
                    overrides=overrides,
                    status="failed",
                    error=str(exc),
                )
            )

    summary = aggregator.summarize()

    write_json_summary(summary, out_root / "summary.json")
    write_markdown_summary(summary, out_root / "REPORT.md")

    return summary
