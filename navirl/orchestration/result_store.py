"""Persistent result storage for distributed orchestration.

Provides :class:`ShardResult` for recording per-shard outcomes and
:class:`ResultStore` for filesystem-based persistence with deterministic
merge ordering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from navirl.experiments.aggregator import BatchAggregator, BatchSummary, RunRecord

logger = logging.getLogger(__name__)


@dataclass
class ShardResult:
    """Result of executing a single shard.

    Attributes
    ----------
    shard_id:
        Index of the shard that produced this result.
    manifest_id:
        Manifest identifier for traceability.
    records:
        Per-task run records.
    status:
        Overall shard status (``"completed"``, ``"partial"``, ``"failed"``).
    attempts:
        Number of execution attempts for this shard.
    started_at:
        ISO timestamp when execution began.
    finished_at:
        ISO timestamp when execution ended.
    error:
        Error message if the shard failed entirely.
    """

    shard_id: int
    manifest_id: str = ""
    records: list[RunRecord] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    started_at: str = ""
    finished_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "manifest_id": self.manifest_id,
            "status": self.status,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "records": [
                {
                    "scenario": r.scenario,
                    "seed": r.seed,
                    "overrides": r.overrides,
                    "metrics": r.metrics,
                    "status": r.status,
                    "error": r.error,
                }
                for r in self.records
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardResult:
        records = [
            RunRecord(
                scenario=r["scenario"],
                seed=r["seed"],
                overrides=r.get("overrides", {}),
                metrics=r.get("metrics", {}),
                status=r.get("status", "completed"),
                error=r.get("error"),
            )
            for r in data.get("records", [])
        ]
        return cls(
            shard_id=data["shard_id"],
            manifest_id=data.get("manifest_id", ""),
            records=records,
            status=data.get("status", "pending"),
            attempts=data.get("attempts", 0),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            error=data.get("error"),
        )


class ResultStore:
    """Filesystem-backed store for shard results.

    Results are persisted as individual JSON files under a root directory,
    one file per shard.  This allows independent workers to write results
    without coordination, and the orchestrator to merge them deterministically.

    Parameters
    ----------
    root:
        Root directory for result files.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _shard_path(self, shard_id: int) -> Path:
        return self.root / f"shard_{shard_id:04d}.json"

    def save(self, result: ShardResult) -> Path:
        """Persist a shard result to disk."""
        path = self._shard_path(result.shard_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        return path

    def load(self, shard_id: int) -> ShardResult | None:
        """Load a shard result from disk, or ``None`` if not found."""
        path = self._shard_path(shard_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return ShardResult.from_dict(json.load(f))

    def load_all(self, num_shards: int) -> list[ShardResult | None]:
        """Load results for all shard IDs in order."""
        return [self.load(i) for i in range(num_shards)]

    def completed_shards(self, num_shards: int) -> list[int]:
        """Return shard IDs with ``status == 'completed'``."""
        completed = []
        for i in range(num_shards):
            result = self.load(i)
            if result is not None and result.status == "completed":
                completed.append(i)
        return completed

    def pending_shards(self, num_shards: int) -> list[int]:
        """Return shard IDs that have not completed successfully."""
        completed = set(self.completed_shards(num_shards))
        return [i for i in range(num_shards) if i not in completed]

    def merge_results(
        self,
        num_shards: int,
        template_name: str = "",
    ) -> BatchSummary:
        """Merge all shard results into a single :class:`BatchSummary`.

        Shard records are merged in deterministic order (by shard_id,
        then by record order within each shard).
        """
        aggregator = BatchAggregator(template_name=template_name)

        for shard_id in range(num_shards):
            result = self.load(shard_id)
            if result is None:
                logger.warning("Shard %d has no result file", shard_id)
                continue
            for record in result.records:
                aggregator.add_record(record)

        return aggregator.summarize()
