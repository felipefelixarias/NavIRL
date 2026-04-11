"""Task executor backends for simulation orchestration.

Provides an abstract executor interface and a local multiprocessing
implementation.  Additional backends (e.g. Ray, Dask, Kubernetes) can
be added by subclassing :class:`TaskExecutor`.
"""

from __future__ import annotations

import abc
import copy
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import yaml

from navirl.core.seeds import set_global_seed
from navirl.experiments.runner import _apply_overrides
from navirl.metrics.standard import StandardMetrics
from navirl.orchestration.models import SimulationTask, TaskResult, TaskStatus
from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario

logger = logging.getLogger(__name__)


class TaskExecutor(abc.ABC):
    """Abstract interface for executing simulation tasks.

    Subclasses implement the actual execution strategy (local processes,
    remote workers, cluster submission, etc.).
    """

    @abc.abstractmethod
    def execute_batch(
        self,
        tasks: list[SimulationTask],
        out_root: str,
        *,
        progress_callback: Any | None = None,
    ) -> list[TaskResult]:
        """Execute a batch of tasks and return results.

        Parameters
        ----------
        tasks:
            List of simulation tasks to execute.
        out_root:
            Root output directory for run bundles.
        progress_callback:
            Optional callable invoked as ``callback(completed, total)``
            after each task finishes.

        Returns
        -------
        list[TaskResult]
            One result per input task, in the same order.
        """


def _execute_single_task(
    task: SimulationTask,
    out_root: str,
) -> TaskResult:
    """Execute a single simulation task and return its result.

    This function is designed to be called in a subprocess via
    multiprocessing, so it avoids referencing any shared mutable state.
    """
    t0 = time.monotonic()
    try:
        scenario = load_scenario(task.scenario_path)
        scenario["seed"] = task.seed
        if task.overrides:
            scenario = _apply_overrides(scenario, task.overrides)

        set_global_seed(task.seed)
        episode_log = run_scenario_dict(
            scenario,
            out_root=out_root,
            render_override=False,
            video_override=False,
        )

        state_path = Path(episode_log.state_path)
        bundle_dir = state_path.parent
        scenario_yaml = bundle_dir / "scenario.yaml"
        with scenario_yaml.open("r", encoding="utf-8") as f:
            run_scenario = yaml.safe_load(f)

        metrics_collector = StandardMetrics()
        run_metrics = metrics_collector.compute(state_path, run_scenario)

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            metrics=run_metrics,
            bundle_dir=str(bundle_dir),
            wall_time_s=time.monotonic() - t0,
        )
    except Exception as exc:
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=str(exc),
            wall_time_s=time.monotonic() - t0,
        )


def _worker_entry(args: tuple[dict[str, Any], str]) -> dict[str, Any]:
    """Multiprocessing worker entry point.

    Accepts and returns plain dicts to avoid pickling issues with
    dataclasses across process boundaries.
    """
    task_data, out_root = args
    task = SimulationTask(
        task_id=task_data["task_id"],
        scenario_path=task_data["scenario_path"],
        seed=task_data["seed"],
        overrides=task_data.get("overrides", {}),
    )
    result = _execute_single_task(task, out_root)
    return result.to_dict()


class LocalExecutor(TaskExecutor):
    """Execute simulation tasks using local multiprocessing.

    Parameters
    ----------
    max_workers:
        Maximum number of parallel worker processes.  Defaults to the
        number of CPU cores.  Use ``1`` for sequential execution.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        if max_workers is None:
            max_workers = mp.cpu_count() or 1
        self.max_workers = max(1, max_workers)

    def execute_batch(
        self,
        tasks: list[SimulationTask],
        out_root: str,
        *,
        progress_callback: Any | None = None,
    ) -> list[TaskResult]:
        if not tasks:
            return []

        Path(out_root).mkdir(parents=True, exist_ok=True)

        # Serialize tasks to plain dicts for multiprocessing
        work_items = [
            (
                {
                    "task_id": t.task_id,
                    "scenario_path": t.scenario_path,
                    "seed": t.seed,
                    "overrides": copy.deepcopy(t.overrides),
                },
                out_root,
            )
            for t in tasks
        ]

        results: list[TaskResult] = []

        if self.max_workers <= 1:
            # Sequential execution
            for i, item in enumerate(work_items):
                result_dict = _worker_entry(item)
                results.append(TaskResult.from_dict(result_dict))
                if progress_callback is not None:
                    progress_callback(i + 1, len(tasks))
        else:
            # Parallel execution
            num_procs = min(self.max_workers, len(tasks))
            logger.info("Launching %d workers for %d tasks", num_procs, len(tasks))
            with mp.Pool(processes=num_procs) as pool:
                for i, result_dict in enumerate(
                    pool.imap(  # preserves order
                        _worker_entry,
                        work_items,
                    )
                ):
                    results.append(TaskResult.from_dict(result_dict))
                    if progress_callback is not None:
                        progress_callback(i + 1, len(tasks))

        return results
