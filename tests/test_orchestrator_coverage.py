"""Tests for navirl/orchestration/orchestrator.py — run, resume, status methods.

Covers the Orchestrator.run(), resume(), status(), _run_shard() methods
that are currently at 41% coverage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from navirl.experiments.aggregator import BatchSummary, RunRecord
from navirl.experiments.templates import BatchTemplate
from navirl.orchestration.manifest import ShardManifest, TaskShard
from navirl.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from navirl.orchestration.result_store import ResultStore, ShardResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_template(name: str = "test_template", num_tasks: int = 4) -> MagicMock:
    template = MagicMock(spec=BatchTemplate)
    template.name = name
    template.expand_tasks.return_value = [MagicMock(task_id=f"task_{i}") for i in range(num_tasks)]
    return template


def _make_shard_result(shard_id: int, status: str = "completed", records: int = 2) -> ShardResult:
    result = ShardResult(shard_id=shard_id, status=status)
    result.records = [MagicMock() for _ in range(records)]
    return result


def _make_batch_summary(completed: int = 4, total: int = 4) -> BatchSummary:
    summary = MagicMock(spec=BatchSummary)
    summary.completed_runs = completed
    summary.total_runs = total
    return summary


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.num_shards == 4
        assert cfg.max_retries == 2
        assert cfg.max_workers is None
        assert cfg.render is False
        assert cfg.video is False

    def test_custom_values(self):
        cfg = OrchestratorConfig(num_shards=8, max_retries=5, max_workers=2)
        assert cfg.num_shards == 8
        assert cfg.max_retries == 5
        assert cfg.max_workers == 2


# ---------------------------------------------------------------------------
# Orchestrator.__init__ and properties
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_default_config(self, tmp_path):
        template = _make_template()
        orch = Orchestrator(template, tmp_path)
        assert orch.config.num_shards == 4
        assert orch.out_root == tmp_path

    def test_custom_config(self, tmp_path):
        template = _make_template()
        cfg = OrchestratorConfig(num_shards=2)
        orch = Orchestrator(template, tmp_path, config=cfg)
        assert orch.config.num_shards == 2

    def test_manifest_lazy_creation(self, tmp_path):
        template = _make_template(num_tasks=8)
        orch = Orchestrator(template, tmp_path, config=OrchestratorConfig(num_shards=2))
        assert orch._manifest is None
        _ = orch.manifest  # Triggers creation
        assert orch._manifest is not None
        template.expand_tasks.assert_called_once()

    def test_manifest_caching(self, tmp_path):
        template = _make_template(num_tasks=4)
        orch = Orchestrator(template, tmp_path)
        m1 = orch.manifest
        m2 = orch.manifest
        assert m1 is m2
        template.expand_tasks.assert_called_once()


# ---------------------------------------------------------------------------
# Orchestrator.save_manifest
# ---------------------------------------------------------------------------


class TestSaveManifest:
    def test_saves_to_disk(self, tmp_path):
        template = _make_template(num_tasks=4)
        orch = Orchestrator(template, tmp_path)

        # Mock the manifest's save method
        with patch.object(ShardManifest, "save"):
            # Need to create a real manifest first
            orch._manifest = MagicMock(spec=ShardManifest)
            path = orch.save_manifest()
            assert path == tmp_path / "manifest.yaml"
            orch._manifest.save.assert_called_once_with(path)


# ---------------------------------------------------------------------------
# Orchestrator._run_shard
# ---------------------------------------------------------------------------


class TestRunShard:
    def test_successful_first_attempt(self, tmp_path):
        template = _make_template(num_tasks=4)
        cfg = OrchestratorConfig(num_shards=2, max_retries=3)
        orch = Orchestrator(template, tmp_path, config=cfg)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.shards = {0: MagicMock(), 1: MagicMock()}
        mock_manifest.manifest_id = "test_id"
        orch._manifest = mock_manifest

        good_result = _make_shard_result(0, "completed")

        with (
            patch("navirl.orchestration.orchestrator.ShardWorker") as MockWorker,
            patch.object(orch.result_store, "save"),
        ):
            mock_worker_instance = MagicMock()
            mock_worker_instance.run.return_value = good_result
            MockWorker.return_value = mock_worker_instance

            result = orch._run_shard(0)
            assert result.status == "completed"
            assert result.attempts == 1

    def test_retry_then_success(self, tmp_path):
        template = _make_template(num_tasks=4)
        cfg = OrchestratorConfig(num_shards=2, max_retries=3)
        orch = Orchestrator(template, tmp_path, config=cfg)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.shards = {0: MagicMock()}
        mock_manifest.manifest_id = "test_id"
        orch._manifest = mock_manifest

        fail_result = _make_shard_result(0, "failed", records=0)
        good_result = _make_shard_result(0, "completed")

        with (
            patch("navirl.orchestration.orchestrator.ShardWorker") as MockWorker,
            patch.object(orch.result_store, "save"),
        ):
            mock_worker = MagicMock()
            mock_worker.run.side_effect = [fail_result, good_result]
            MockWorker.return_value = mock_worker

            result = orch._run_shard(0)
            assert result.status == "completed"
            assert result.attempts == 2

    def test_all_retries_exhausted(self, tmp_path):
        template = _make_template(num_tasks=4)
        cfg = OrchestratorConfig(num_shards=2, max_retries=2)
        orch = Orchestrator(template, tmp_path, config=cfg)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.shards = {0: MagicMock()}
        mock_manifest.manifest_id = "test_id"
        orch._manifest = mock_manifest

        fail1 = _make_shard_result(0, "failed", records=0)
        fail2 = _make_shard_result(0, "failed", records=0)

        with (
            patch("navirl.orchestration.orchestrator.ShardWorker") as MockWorker,
            patch.object(orch.result_store, "save"),
        ):
            mock_worker = MagicMock()
            mock_worker.run.side_effect = [fail1, fail2]
            MockWorker.return_value = mock_worker

            result = orch._run_shard(0)
            assert result.status == "failed"


# ---------------------------------------------------------------------------
# Orchestrator.run
# ---------------------------------------------------------------------------


class TestOrchestratorRun:
    def test_full_run(self, tmp_path):
        template = _make_template(num_tasks=2)
        cfg = OrchestratorConfig(num_shards=2, max_retries=1)
        orch = Orchestrator(template, tmp_path, config=cfg)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.shards = {0: MagicMock(), 1: MagicMock()}
        mock_manifest.manifest_id = "test_id"
        mock_manifest.num_shards = 2
        mock_manifest.total_tasks = 2
        mock_manifest.save = MagicMock()
        orch._manifest = mock_manifest

        good_result = _make_shard_result(0, "completed")
        summary = _make_batch_summary()

        with (
            patch("navirl.orchestration.orchestrator.ShardWorker") as MockWorker,
            patch.object(orch.result_store, "save"),
            patch.object(orch.result_store, "merge_results", return_value=summary),
            patch("navirl.orchestration.orchestrator.write_json_summary"),
            patch("navirl.orchestration.orchestrator.write_markdown_summary"),
        ):
            mock_worker = MagicMock()
            mock_worker.run.return_value = good_result
            MockWorker.return_value = mock_worker

            result = orch.run()
            assert result.completed_runs == 4

    def test_run_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "output"
        template = _make_template(num_tasks=1)
        cfg = OrchestratorConfig(num_shards=1, max_retries=1)
        orch = Orchestrator(template, out, config=cfg)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.shards = {0: MagicMock()}
        mock_manifest.manifest_id = "test_id"
        mock_manifest.num_shards = 1
        mock_manifest.total_tasks = 1
        mock_manifest.save = MagicMock()
        orch._manifest = mock_manifest

        good_result = _make_shard_result(0, "completed")
        summary = _make_batch_summary(1, 1)

        with (
            patch("navirl.orchestration.orchestrator.ShardWorker") as MockWorker,
            patch.object(orch.result_store, "save"),
            patch.object(orch.result_store, "merge_results", return_value=summary),
            patch("navirl.orchestration.orchestrator.write_json_summary"),
            patch("navirl.orchestration.orchestrator.write_markdown_summary"),
        ):
            mock_worker = MagicMock()
            mock_worker.run.return_value = good_result
            MockWorker.return_value = mock_worker

            orch.run()
            assert out.exists()


# ---------------------------------------------------------------------------
# Orchestrator.resume
# ---------------------------------------------------------------------------


class TestOrchestratorResume:
    def test_nothing_to_resume(self, tmp_path):
        template = _make_template(num_tasks=2)
        orch = Orchestrator(template, tmp_path)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.num_shards = 2
        orch._manifest = mock_manifest

        summary = _make_batch_summary()

        with (
            patch.object(orch.result_store, "pending_shards", return_value=[]),
            patch.object(orch.result_store, "merge_results", return_value=summary),
        ):
            result = orch.resume()
            assert result.completed_runs == 4

    def test_resumes_pending_shards(self, tmp_path):
        template = _make_template(num_tasks=4)
        cfg = OrchestratorConfig(num_shards=4, max_retries=1)
        orch = Orchestrator(template, tmp_path, config=cfg)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.shards = {0: MagicMock(), 1: MagicMock(), 2: MagicMock(), 3: MagicMock()}
        mock_manifest.manifest_id = "test_id"
        mock_manifest.num_shards = 4
        orch._manifest = mock_manifest

        good_result = _make_shard_result(2, "completed")
        summary = _make_batch_summary()

        with (
            patch.object(orch.result_store, "pending_shards", return_value=[2, 3]),
            patch.object(orch.result_store, "merge_results", return_value=summary),
            patch("navirl.orchestration.orchestrator.ShardWorker") as MockWorker,
            patch.object(orch.result_store, "save"),
            patch("navirl.orchestration.orchestrator.write_json_summary"),
            patch("navirl.orchestration.orchestrator.write_markdown_summary"),
        ):
            mock_worker = MagicMock()
            mock_worker.run.return_value = good_result
            MockWorker.return_value = mock_worker

            result = orch.resume()
            assert result.completed_runs == 4


# ---------------------------------------------------------------------------
# Orchestrator.status
# ---------------------------------------------------------------------------


class TestOrchestratorStatus:
    def test_returns_status_dict(self, tmp_path):
        template = _make_template(num_tasks=4)
        orch = Orchestrator(template, tmp_path)

        mock_manifest = MagicMock(spec=ShardManifest)
        mock_manifest.manifest_id = "test_123"
        mock_manifest.num_shards = 4
        mock_manifest.total_tasks = 4
        orch._manifest = mock_manifest

        with (
            patch.object(orch.result_store, "completed_shards", return_value=[0, 1]),
            patch.object(orch.result_store, "pending_shards", return_value=[2, 3]),
        ):
            status = orch.status()
            assert status["manifest_id"] == "test_123"
            assert status["template_name"] == "test_template"
            assert status["total_shards"] == 4
            assert status["total_tasks"] == 4
            assert status["completed_shards"] == 2
            assert status["pending_shards"] == 2
            assert status["completed_shard_ids"] == [0, 1]
            assert status["pending_shard_ids"] == [2, 3]
