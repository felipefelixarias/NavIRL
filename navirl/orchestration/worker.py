"""Shard worker for executing simulation tasks.

A :class:`ShardWorker` takes a single :class:`TaskShard` and runs all
its tasks sequentially, producing a :class:`ShardResult`.  Workers are
designed to run independently, potentially on different machines, with
results persisted via a :class:`ResultStore`.
"""

from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from navirl.core.seeds import set_global_seed
from navirl.experiments.aggregator import RunRecord
from navirl.experiments.runner import _apply_overrides
from navirl.metrics.standard import StandardMetrics
from navirl.orchestration.manifest import TaskShard
from navirl.orchestration.result_store import ResultStore, ShardResult
from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario

logger = logging.getLogger(__name__)


class ShardWorker:
    """Executes all tasks in a single shard.

    Parameters
    ----------
    shard:
        The task shard to execute.
    out_root:
        Root output directory for run artifacts.
    manifest_id:
        Manifest identifier for result traceability.
    result_store:
        Optional result store for persisting results.
    render:
        Whether to enable rendering during simulation.
    video:
        Whether to record video output.
    """

    def __init__(
        self,
        shard: TaskShard,
        out_root: str | Path,
        *,
        manifest_id: str = "",
        result_store: ResultStore | None = None,
        render: bool = False,
        video: bool = False,
    ) -> None:
        self.shard = shard
        self.out_root = Path(out_root)
        self.manifest_id = manifest_id
        self.result_store = result_store
        self.render = render
        self.video = video

    def run(self) -> ShardResult:
        """Execute all tasks in the shard and return the result.

        Each task is run independently; a failing task does not abort
        the shard.  The overall shard status is ``"completed"`` if all
        tasks succeed, ``"partial"`` if some fail, or ``"failed"`` if
        all tasks fail.
        """
        result = ShardResult(
            shard_id=self.shard.shard_id,
            manifest_id=self.manifest_id,
            started_at=datetime.now(UTC).isoformat(),
        )

        metrics_collector = StandardMetrics()
        successes = 0
        failures = 0

        for i, task in enumerate(self.shard.tasks):
            scenario_path: Path = task["scenario"]
            seed: int = task["seed"]
            overrides: dict[str, Any] = task.get("overrides", {})
            scenario_name = scenario_path.stem

            logger.info(
                "Shard %d task %d/%d: %s seed=%d",
                self.shard.shard_id,
                i + 1,
                self.shard.num_tasks,
                scenario_name,
                seed,
            )

            try:
                scenario = load_scenario(str(scenario_path))
                scenario["seed"] = seed
                if overrides:
                    scenario = _apply_overrides(scenario, overrides)

                set_global_seed(seed)
                episode_log = run_scenario_dict(
                    scenario,
                    out_root=str(self.out_root),
                    render_override=self.render,
                    video_override=self.video,
                )

                state_path = Path(episode_log.state_path)
                bundle_dir = state_path.parent
                scenario_yaml = bundle_dir / "scenario.yaml"
                with scenario_yaml.open("r", encoding="utf-8") as f:
                    run_scenario = yaml.safe_load(f)

                run_metrics = metrics_collector.compute(state_path, run_scenario)

                result.records.append(
                    RunRecord(
                        scenario=scenario_name,
                        seed=seed,
                        overrides=overrides,
                        metrics=run_metrics,
                        status="completed",
                    )
                )
                successes += 1

            except Exception as exc:
                logger.warning(
                    "Shard %d task %d failed: %s",
                    self.shard.shard_id,
                    i + 1,
                    exc,
                )
                result.records.append(
                    RunRecord(
                        scenario=scenario_name,
                        seed=seed,
                        overrides=overrides,
                        status="failed",
                        error=str(exc),
                    )
                )
                failures += 1

        result.finished_at = datetime.now(UTC).isoformat()
        result.attempts = 1

        if failures == 0:
            result.status = "completed"
        elif successes == 0:
            result.status = "failed"
        else:
            result.status = "partial"

        if self.result_store is not None:
            self.result_store.save(result)

        return result
