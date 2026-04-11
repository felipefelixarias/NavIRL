"""Shard manifest for splitting batch tasks across workers.

A :class:`ShardManifest` takes a flat list of expanded tasks from a
:class:`~navirl.experiments.templates.BatchTemplate` and partitions
them into numbered shards that can be executed independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskShard:
    """A numbered subset of tasks to be executed by one worker.

    Attributes
    ----------
    shard_id:
        Zero-based shard index.
    tasks:
        List of task dicts (scenario, seed, overrides) assigned to this shard.
    """

    shard_id: int
    tasks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        serializable_tasks = []
        for t in self.tasks:
            st = dict(t)
            if "scenario" in st:
                st["scenario"] = str(st["scenario"])
            serializable_tasks.append(st)
        return {"shard_id": self.shard_id, "tasks": serializable_tasks}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskShard:
        tasks = []
        for t in data["tasks"]:
            task = dict(t)
            if "scenario" in task:
                task["scenario"] = Path(task["scenario"])
            tasks.append(task)
        return cls(shard_id=data["shard_id"], tasks=tasks)


@dataclass
class ShardManifest:
    """Partitions expanded tasks into numbered shards.

    Parameters
    ----------
    template_name:
        Name of the originating batch template.
    shards:
        List of task shards.
    manifest_id:
        Unique identifier for this manifest (derived from content hash).
    """

    template_name: str
    shards: list[TaskShard] = field(default_factory=list)
    manifest_id: str = ""

    @classmethod
    def from_tasks(
        cls,
        tasks: list[dict[str, Any]],
        num_shards: int,
        template_name: str = "",
    ) -> ShardManifest:
        """Create a manifest by evenly distributing tasks across shards.

        Tasks are assigned round-robin to ensure balanced shard sizes.
        The assignment is deterministic for the same input.

        Parameters
        ----------
        tasks:
            Flat list of task dicts from ``BatchTemplate.expand_tasks()``.
        num_shards:
            Number of shards to create.  Clamped to ``len(tasks)`` if larger.
        template_name:
            Name of the originating template.
        """
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")

        num_shards = min(num_shards, max(len(tasks), 1))
        shard_lists: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]

        for i, task in enumerate(tasks):
            shard_lists[i % num_shards].append(task)

        shards = [
            TaskShard(shard_id=i, tasks=shard_lists[i])
            for i in range(num_shards)
        ]

        manifest = cls(template_name=template_name, shards=shards)
        manifest.manifest_id = manifest._compute_id()
        return manifest

    @property
    def total_tasks(self) -> int:
        return sum(s.num_tasks for s in self.shards)

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    def _compute_id(self) -> str:
        """Deterministic content hash for deduplication."""
        content = json.dumps(
            {
                "template_name": self.template_name,
                "shards": [s.to_dict() for s in self.shards],
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "template_name": self.template_name,
            "num_shards": self.num_shards,
            "total_tasks": self.total_tasks,
            "shards": [s.to_dict() for s in self.shards],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardManifest:
        shards = [TaskShard.from_dict(s) for s in data["shards"]]
        return cls(
            template_name=data["template_name"],
            shards=shards,
            manifest_id=data.get("manifest_id", ""),
        )

    def save(self, path: str | Path) -> None:
        """Serialize manifest to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def load(cls, path: str | Path) -> ShardManifest:
        """Load manifest from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
