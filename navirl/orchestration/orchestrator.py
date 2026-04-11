"""High-level orchestrator for distributed simulation sweeps.

The :class:`Orchestrator` manages the full lifecycle of a distributed
batch experiment: manifest creation, worker dispatch, retry logic,
progress tracking, and deterministic result merging.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from navirl.experiments.aggregator import (
    BatchSummary,
    write_json_summary,
    write_markdown_summary,
)
from navirl.experiments.templates import BatchTemplate
from navirl.orchestration.manifest import ShardManifest
from navirl.orchestration.result_store import ResultStore, ShardResult
from navirl.orchestration.worker import ShardWorker

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator.

    Attributes
    ----------
    num_shards:
        Number of shards to partition work into.
    max_retries:
        Maximum retry attempts per shard before giving up.
    max_workers:
        Maximum concurrent workers (``None`` = one per shard).
    render:
        Whether to enable rendering during simulation.
    video:
        Whether to record video output.
    """

    num_shards: int = 4
    max_retries: int = 2
    max_workers: int | None = None
    render: bool = False
    video: bool = False


class Orchestrator:
    """Manages distributed execution of batch experiments.

    Parameters
    ----------
    template:
        The batch template defining the experiment.
    out_root:
        Root output directory.
    config:
        Orchestrator configuration.
    """

    def __init__(
        self,
        template: BatchTemplate,
        out_root: str | Path,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.template = template
        self.out_root = Path(out_root)
        self.config = config or OrchestratorConfig()
        self.result_store = ResultStore(self.out_root / "results")
        self._manifest: ShardManifest | None = None

    @property
    def manifest(self) -> ShardManifest:
        """The shard manifest for this experiment (created on first access)."""
        if self._manifest is None:
            tasks = self.template.expand_tasks()
            self._manifest = ShardManifest.from_tasks(
                tasks,
                num_shards=self.config.num_shards,
                template_name=self.template.name,
            )
        return self._manifest

    def save_manifest(self) -> Path:
        """Save the manifest to disk and return the file path."""
        path = self.out_root / "manifest.yaml"
        self.manifest.save(path)
        return path

    def _run_shard(self, shard_id: int) -> ShardResult:
        """Execute a single shard with retry logic."""
        shard = self.manifest.shards[shard_id]
        last_result: ShardResult | None = None

        for attempt in range(1, self.config.max_retries + 1):
            logger.info(
                "Shard %d attempt %d/%d",
                shard_id,
                attempt,
                self.config.max_retries,
            )

            worker = ShardWorker(
                shard=shard,
                out_root=self.out_root / f"shard_{shard_id:04d}",
                manifest_id=self.manifest.manifest_id,
                result_store=self.result_store,
                render=self.config.render,
                video=self.config.video,
            )
            result = worker.run()
            result.attempts = attempt

            if result.status == "completed":
                self.result_store.save(result)
                return result

            last_result = result
            logger.warning(
                "Shard %d attempt %d status: %s",
                shard_id,
                attempt,
                result.status,
            )

        if last_result is not None:
            self.result_store.save(last_result)
            return last_result

        # Should not reach here, but handle gracefully.
        empty = ShardResult(shard_id=shard_id, status="failed", error="No attempts made")
        self.result_store.save(empty)
        return empty

    def run(self) -> BatchSummary:
        """Execute all shards and return a merged summary.

        Shards are executed in parallel using a thread pool.  Failed shards
        are retried up to ``config.max_retries`` times.  Results are merged
        deterministically by shard order.

        Returns
        -------
        BatchSummary
            Aggregated results.  Also written to ``{out_root}/summary.json``
            and ``{out_root}/REPORT.md``.
        """
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.save_manifest()

        pending = list(range(self.manifest.num_shards))
        max_workers = self.config.max_workers or self.manifest.num_shards

        logger.info(
            "Starting orchestrated run: %d shards, %d total tasks, %d max workers",
            self.manifest.num_shards,
            self.manifest.total_tasks,
            max_workers,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_shard, shard_id): shard_id
                for shard_id in pending
            }
            for future in concurrent.futures.as_completed(futures):
                shard_id = futures[future]
                try:
                    result = future.result()
                    logger.info(
                        "Shard %d finished: status=%s (%d records)",
                        shard_id,
                        result.status,
                        len(result.records),
                    )
                except Exception as exc:
                    logger.error("Shard %d raised: %s", shard_id, exc)

        summary = self.result_store.merge_results(
            num_shards=self.manifest.num_shards,
            template_name=self.template.name,
        )

        write_json_summary(summary, self.out_root / "summary.json")
        write_markdown_summary(summary, self.out_root / "REPORT.md")

        logger.info(
            "Orchestration complete: %d/%d runs completed",
            summary.completed_runs,
            summary.total_runs,
        )

        return summary

    def resume(self) -> BatchSummary:
        """Resume a previously interrupted orchestrated run.

        Only re-executes shards that have not yet completed successfully.
        """
        pending = self.result_store.pending_shards(self.manifest.num_shards)
        if not pending:
            logger.info("All shards already completed, nothing to resume")
            return self.result_store.merge_results(
                num_shards=self.manifest.num_shards,
                template_name=self.template.name,
            )

        logger.info("Resuming: %d shards pending", len(pending))
        max_workers = self.config.max_workers or len(pending)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_shard, shard_id): shard_id
                for shard_id in pending
            }
            for future in concurrent.futures.as_completed(futures):
                shard_id = futures[future]
                try:
                    result = future.result()
                    logger.info(
                        "Shard %d resumed: status=%s",
                        shard_id,
                        result.status,
                    )
                except Exception as exc:
                    logger.error("Shard %d raised: %s", shard_id, exc)

        summary = self.result_store.merge_results(
            num_shards=self.manifest.num_shards,
            template_name=self.template.name,
        )

        write_json_summary(summary, self.out_root / "summary.json")
        write_markdown_summary(summary, self.out_root / "REPORT.md")

        return summary

    def status(self) -> dict[str, Any]:
        """Return current orchestration status."""
        completed = self.result_store.completed_shards(self.manifest.num_shards)
        pending = self.result_store.pending_shards(self.manifest.num_shards)
        return {
            "manifest_id": self.manifest.manifest_id,
            "template_name": self.template.name,
            "total_shards": self.manifest.num_shards,
            "total_tasks": self.manifest.total_tasks,
            "completed_shards": len(completed),
            "pending_shards": len(pending),
            "completed_shard_ids": completed,
            "pending_shard_ids": pending,
        }
