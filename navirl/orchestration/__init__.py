"""Distributed simulation orchestration for large parameter sweeps.

This package provides sharded task distribution, worker execution,
retry semantics, and deterministic result merging for running
large-scale NavIRL experiments across multiple processes or machines.
"""

from __future__ import annotations

from navirl.orchestration.manifest import ShardManifest, TaskShard
from navirl.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from navirl.orchestration.result_store import ResultStore, ShardResult
from navirl.orchestration.worker import ShardWorker

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "ResultStore",
    "ShardManifest",
    "ShardResult",
    "ShardWorker",
    "TaskShard",
]
