"""Tests for navirl.orchestration — distributed simulation orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from navirl.orchestration.manifest import ShardManifest, TaskShard
from navirl.orchestration.result_store import ResultStore, ShardResult
from navirl.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from navirl.orchestration.worker import ShardWorker
from navirl.experiments.aggregator import RunRecord
from navirl.experiments.templates import BatchTemplate


# ============================================================================
# TaskShard
# ============================================================================


class TestTaskShard:
    def test_empty_shard(self):
        shard = TaskShard(shard_id=0)
        assert shard.num_tasks == 0
        assert shard.shard_id == 0

    def test_num_tasks(self):
        tasks = [
            {"scenario": Path("a.yaml"), "seed": 1, "overrides": {}},
            {"scenario": Path("b.yaml"), "seed": 2, "overrides": {}},
        ]
        shard = TaskShard(shard_id=3, tasks=tasks)
        assert shard.num_tasks == 2
        assert shard.shard_id == 3

    def test_to_dict_serializes_paths(self):
        shard = TaskShard(
            shard_id=0,
            tasks=[{"scenario": Path("/foo/bar.yaml"), "seed": 42, "overrides": {}}],
        )
        d = shard.to_dict()
        assert d["shard_id"] == 0
        assert d["tasks"][0]["scenario"] == "/foo/bar.yaml"
        assert isinstance(d["tasks"][0]["scenario"], str)

    def test_from_dict_restores_paths(self):
        data = {
            "shard_id": 1,
            "tasks": [{"scenario": "/foo/bar.yaml", "seed": 42, "overrides": {}}],
        }
        shard = TaskShard.from_dict(data)
        assert shard.shard_id == 1
        assert isinstance(shard.tasks[0]["scenario"], Path)

    def test_roundtrip(self):
        original = TaskShard(
            shard_id=5,
            tasks=[
                {"scenario": Path("s1.yaml"), "seed": 1, "overrides": {"a": 1}},
                {"scenario": Path("s2.yaml"), "seed": 2, "overrides": {}},
            ],
        )
        restored = TaskShard.from_dict(original.to_dict())
        assert restored.shard_id == original.shard_id
        assert restored.num_tasks == original.num_tasks


# ============================================================================
# ShardManifest
# ============================================================================


class TestShardManifest:
    def _make_tasks(self, n: int) -> list[dict[str, Any]]:
        return [
            {"scenario": Path(f"scenario_{i}.yaml"), "seed": i, "overrides": {}}
            for i in range(n)
        ]

    def test_from_tasks_basic(self):
        tasks = self._make_tasks(10)
        manifest = ShardManifest.from_tasks(tasks, num_shards=3, template_name="test")
        assert manifest.num_shards == 3
        assert manifest.total_tasks == 10
        assert manifest.template_name == "test"
        assert manifest.manifest_id != ""

    def test_from_tasks_single_shard(self):
        tasks = self._make_tasks(5)
        manifest = ShardManifest.from_tasks(tasks, num_shards=1)
        assert manifest.num_shards == 1
        assert manifest.shards[0].num_tasks == 5

    def test_from_tasks_more_shards_than_tasks(self):
        tasks = self._make_tasks(3)
        manifest = ShardManifest.from_tasks(tasks, num_shards=10)
        assert manifest.num_shards == 3  # Clamped
        assert manifest.total_tasks == 3

    def test_from_tasks_empty(self):
        manifest = ShardManifest.from_tasks([], num_shards=4)
        assert manifest.num_shards == 1
        assert manifest.total_tasks == 0

    def test_from_tasks_invalid_shards(self):
        with pytest.raises(ValueError, match="num_shards must be >= 1"):
            ShardManifest.from_tasks([], num_shards=0)

    def test_round_robin_distribution(self):
        tasks = self._make_tasks(7)
        manifest = ShardManifest.from_tasks(tasks, num_shards=3)
        sizes = [s.num_tasks for s in manifest.shards]
        assert sizes == [3, 2, 2]

    def test_deterministic_manifest_id(self):
        tasks = self._make_tasks(5)
        m1 = ShardManifest.from_tasks(tasks, num_shards=2, template_name="t")
        m2 = ShardManifest.from_tasks(tasks, num_shards=2, template_name="t")
        assert m1.manifest_id == m2.manifest_id

    def test_different_inputs_different_id(self):
        tasks = self._make_tasks(5)
        m1 = ShardManifest.from_tasks(tasks, num_shards=2, template_name="a")
        m2 = ShardManifest.from_tasks(tasks, num_shards=2, template_name="b")
        assert m1.manifest_id != m2.manifest_id

    def test_to_dict(self):
        tasks = self._make_tasks(4)
        manifest = ShardManifest.from_tasks(tasks, num_shards=2, template_name="test")
        d = manifest.to_dict()
        assert d["num_shards"] == 2
        assert d["total_tasks"] == 4
        assert d["template_name"] == "test"
        assert len(d["shards"]) == 2

    def test_from_dict_roundtrip(self):
        tasks = self._make_tasks(6)
        original = ShardManifest.from_tasks(tasks, num_shards=3, template_name="rt")
        restored = ShardManifest.from_dict(original.to_dict())
        assert restored.manifest_id == original.manifest_id
        assert restored.num_shards == original.num_shards
        assert restored.total_tasks == original.total_tasks

    def test_save_and_load(self, tmp_path):
        tasks = self._make_tasks(4)
        manifest = ShardManifest.from_tasks(tasks, num_shards=2, template_name="io")
        path = tmp_path / "manifest.yaml"
        manifest.save(path)
        loaded = ShardManifest.load(path)
        assert loaded.manifest_id == manifest.manifest_id
        assert loaded.num_shards == manifest.num_shards
        assert loaded.total_tasks == manifest.total_tasks

    def test_balanced_shards(self):
        """All shards should differ by at most one task."""
        tasks = self._make_tasks(100)
        manifest = ShardManifest.from_tasks(tasks, num_shards=7)
        sizes = [s.num_tasks for s in manifest.shards]
        assert max(sizes) - min(sizes) <= 1


# ============================================================================
# ShardResult
# ============================================================================


class TestShardResult:
    def test_defaults(self):
        r = ShardResult(shard_id=0)
        assert r.status == "pending"
        assert r.attempts == 0
        assert r.records == []
        assert r.error is None

    def test_to_dict(self):
        r = ShardResult(
            shard_id=1,
            manifest_id="abc123",
            status="completed",
            attempts=1,
            records=[
                RunRecord(scenario="hall", seed=42, metrics={"success_rate": 1.0}),
            ],
        )
        d = r.to_dict()
        assert d["shard_id"] == 1
        assert d["status"] == "completed"
        assert len(d["records"]) == 1
        assert d["records"][0]["metrics"]["success_rate"] == 1.0

    def test_from_dict_roundtrip(self):
        original = ShardResult(
            shard_id=2,
            manifest_id="xyz",
            status="partial",
            attempts=2,
            started_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T00:01:00",
            records=[
                RunRecord(scenario="s1", seed=1, status="completed"),
                RunRecord(scenario="s2", seed=2, status="failed", error="boom"),
            ],
        )
        restored = ShardResult.from_dict(original.to_dict())
        assert restored.shard_id == original.shard_id
        assert restored.status == original.status
        assert restored.attempts == original.attempts
        assert len(restored.records) == 2
        assert restored.records[1].error == "boom"

    def test_from_dict_minimal(self):
        r = ShardResult.from_dict({"shard_id": 5})
        assert r.shard_id == 5
        assert r.status == "pending"
        assert r.records == []


# ============================================================================
# ResultStore
# ============================================================================


class TestResultStore:
    def test_save_and_load(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        result = ShardResult(shard_id=0, status="completed", attempts=1)
        result.records.append(RunRecord(scenario="s1", seed=1))
        store.save(result)

        loaded = store.load(0)
        assert loaded is not None
        assert loaded.shard_id == 0
        assert loaded.status == "completed"
        assert len(loaded.records) == 1

    def test_load_missing(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        assert store.load(99) is None

    def test_completed_shards(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        store.save(ShardResult(shard_id=1, status="failed"))
        store.save(ShardResult(shard_id=2, status="completed"))

        assert store.completed_shards(4) == [0, 2]

    def test_pending_shards(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        store.save(ShardResult(shard_id=2, status="completed"))

        pending = store.pending_shards(4)
        assert pending == [1, 3]

    def test_load_all(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        store.save(ShardResult(shard_id=2, status="failed"))

        results = store.load_all(3)
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_merge_results(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        r0 = ShardResult(shard_id=0, status="completed")
        r0.records.append(
            RunRecord(scenario="s1", seed=1, metrics={"success_rate": 1.0}, status="completed")
        )
        r1 = ShardResult(shard_id=1, status="completed")
        r1.records.append(
            RunRecord(scenario="s1", seed=2, metrics={"success_rate": 0.5}, status="completed")
        )
        store.save(r0)
        store.save(r1)

        summary = store.merge_results(2, template_name="test")
        assert summary.total_runs == 2
        assert summary.completed_runs == 2

    def test_merge_with_missing_shard(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        # Shard 1 is missing
        summary = store.merge_results(2, template_name="test")
        assert summary.total_runs == 0  # No records added

    def test_creates_directory(self, tmp_path):
        root = tmp_path / "deep" / "nested" / "results"
        store = ResultStore(root)
        assert root.exists()

    def test_shard_file_naming(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=42, status="completed"))
        assert (tmp_path / "results" / "shard_0042.json").exists()


# ============================================================================
# Orchestrator
# ============================================================================


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.num_shards == 4
        assert cfg.max_retries == 2
        assert cfg.max_workers is None
        assert cfg.render is False
        assert cfg.video is False

    def test_custom_values(self):
        cfg = OrchestratorConfig(num_shards=8, max_retries=5, max_workers=4)
        assert cfg.num_shards == 8
        assert cfg.max_retries == 5
        assert cfg.max_workers == 4


class TestOrchestrator:
    def _make_template(self, n_scenarios: int = 4) -> BatchTemplate:
        return BatchTemplate(
            name="test_template",
            description="Unit test template",
            scenarios=[f"scenario_{i}.yaml" for i in range(n_scenarios)],
            seeds=[1, 2],
        )

    def test_manifest_created_lazily(self, tmp_path):
        template = self._make_template()
        orch = Orchestrator(template, tmp_path, OrchestratorConfig(num_shards=2))
        assert orch._manifest is None
        _ = orch.manifest
        assert orch._manifest is not None

    def test_manifest_num_shards(self, tmp_path):
        template = self._make_template()
        orch = Orchestrator(template, tmp_path, OrchestratorConfig(num_shards=3))
        # Manifest will try to resolve scenarios which won't exist,
        # so total tasks will be 0, meaning 1 shard
        manifest = orch.manifest
        assert manifest.template_name == "test_template"

    def test_save_manifest(self, tmp_path):
        template = self._make_template()
        orch = Orchestrator(template, tmp_path, OrchestratorConfig(num_shards=2))
        path = orch.save_manifest()
        assert path.exists()
        loaded = ShardManifest.load(path)
        assert loaded.template_name == "test_template"

    def test_status(self, tmp_path):
        template = self._make_template()
        orch = Orchestrator(template, tmp_path, OrchestratorConfig(num_shards=2))
        status = orch.status()
        assert status["template_name"] == "test_template"
        assert "total_shards" in status
        assert "completed_shards" in status
        assert "pending_shards" in status

    def test_resume_all_completed(self, tmp_path):
        template = self._make_template(0)
        orch = Orchestrator(template, tmp_path, OrchestratorConfig(num_shards=1))
        # Pre-populate completed result
        manifest = orch.manifest
        r = ShardResult(shard_id=0, status="completed")
        orch.result_store.save(r)
        summary = orch.resume()
        assert summary is not None


# ============================================================================
# ShardWorker (with mocked simulation)
# ============================================================================


class TestShardWorker:
    def _make_shard(self) -> TaskShard:
        return TaskShard(
            shard_id=0,
            tasks=[
                {"scenario": Path("s1.yaml"), "seed": 1, "overrides": {}},
                {"scenario": Path("s2.yaml"), "seed": 2, "overrides": {"a.b": 1.0}},
            ],
        )

    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    def test_run_all_succeed(self, mock_load, mock_run, mock_metrics, tmp_path):
        mock_load.return_value = {"seed": 1, "scene": {}}

        state_dir = tmp_path / "bundle"
        state_dir.mkdir()
        state_path = state_dir / "state.json"
        state_path.write_text("[]")
        scenario_yaml = state_dir / "scenario.yaml"
        scenario_yaml.write_text("seed: 1\nscene: {}\n")

        mock_episode = MagicMock()
        mock_episode.state_path = str(state_path)
        mock_run.return_value = mock_episode
        mock_metrics.return_value.compute.return_value = {"success_rate": 1.0}

        shard = self._make_shard()
        worker = ShardWorker(shard, tmp_path / "out", manifest_id="test123")
        result = worker.run()

        assert result.status == "completed"
        assert result.shard_id == 0
        assert len(result.records) == 2
        assert all(r.status == "completed" for r in result.records)

    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    def test_run_all_fail(self, mock_load, mock_run, tmp_path):
        mock_load.side_effect = FileNotFoundError("not found")

        shard = self._make_shard()
        worker = ShardWorker(shard, tmp_path / "out")
        result = worker.run()

        assert result.status == "failed"
        assert len(result.records) == 2
        assert all(r.status == "failed" for r in result.records)

    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    def test_run_partial_failure(self, mock_load, mock_run, tmp_path):
        state_dir = tmp_path / "bundle"
        state_dir.mkdir()
        state_path = state_dir / "state.json"
        state_path.write_text("[]")
        scenario_yaml = state_dir / "scenario.yaml"
        scenario_yaml.write_text("seed: 1\nscene: {}\n")

        call_count = 0

        def load_side_effect(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"seed": 1, "scene": {}}
            raise RuntimeError("simulated failure")

        mock_load.side_effect = load_side_effect

        mock_episode = MagicMock()
        mock_episode.state_path = str(state_path)
        mock_run.return_value = mock_episode

        shard = self._make_shard()
        worker = ShardWorker(shard, tmp_path / "out")

        with patch("navirl.orchestration.worker.StandardMetrics") as mock_metrics:
            mock_metrics.return_value.compute.return_value = {"success_rate": 1.0}
            result = worker.run()

        assert result.status == "partial"
        assert result.records[0].status == "completed"
        assert result.records[1].status == "failed"

    def test_worker_saves_to_result_store(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        shard = TaskShard(shard_id=0, tasks=[])
        worker = ShardWorker(shard, tmp_path / "out", result_store=store)
        result = worker.run()
        assert result.status == "completed"  # Empty shard -> all 0 succeeded
        loaded = store.load(0)
        assert loaded is not None
        assert loaded.status == "completed"

    def test_worker_empty_shard(self, tmp_path):
        shard = TaskShard(shard_id=0, tasks=[])
        worker = ShardWorker(shard, tmp_path / "out")
        result = worker.run()
        assert result.status == "completed"
        assert result.records == []
        assert result.started_at != ""
        assert result.finished_at != ""


# ============================================================================
# Integration-style tests (still unit, but test component interaction)
# ============================================================================


class TestOrchestratorWithMockedWorker:
    @patch("navirl.orchestration.orchestrator.ShardWorker")
    def test_run_dispatches_all_shards(self, mock_worker_cls, tmp_path):
        template = BatchTemplate(
            name="integration_test",
            scenarios=[],
            seeds=[1],
        )
        config = OrchestratorConfig(num_shards=3, max_retries=1)
        orch = Orchestrator(template, tmp_path, config)

        mock_instance = MagicMock()
        mock_instance.run.return_value = ShardResult(
            shard_id=0, status="completed", attempts=1
        )
        mock_worker_cls.return_value = mock_instance

        summary = orch.run()
        assert summary is not None
        assert (tmp_path / "manifest.yaml").exists()

    @patch("navirl.orchestration.orchestrator.ShardWorker")
    def test_retry_on_failure(self, mock_worker_cls, tmp_path):
        template = BatchTemplate(name="retry_test", scenarios=[], seeds=[1])
        config = OrchestratorConfig(num_shards=1, max_retries=3)
        orch = Orchestrator(template, tmp_path, config)

        call_count = 0

        def run_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return ShardResult(shard_id=0, status="partial", attempts=call_count)
            return ShardResult(shard_id=0, status="completed", attempts=call_count)

        mock_instance = MagicMock()
        mock_instance.run.side_effect = run_side_effect
        mock_worker_cls.return_value = mock_instance

        summary = orch.run()
        assert call_count == 3
        # Final result should be saved
        loaded = orch.result_store.load(0)
        assert loaded is not None


class TestManifestEdgeCases:
    def test_single_task_single_shard(self):
        tasks = [{"scenario": Path("a.yaml"), "seed": 1, "overrides": {}}]
        manifest = ShardManifest.from_tasks(tasks, num_shards=1)
        assert manifest.num_shards == 1
        assert manifest.total_tasks == 1

    def test_many_shards_few_tasks(self):
        tasks = [{"scenario": Path("a.yaml"), "seed": i, "overrides": {}} for i in range(2)]
        manifest = ShardManifest.from_tasks(tasks, num_shards=100)
        assert manifest.num_shards == 2

    def test_tasks_preserved_in_shards(self):
        tasks = [
            {"scenario": Path(f"s{i}.yaml"), "seed": i, "overrides": {"k": i}}
            for i in range(10)
        ]
        manifest = ShardManifest.from_tasks(tasks, num_shards=3)
        all_tasks = []
        for shard in manifest.shards:
            all_tasks.extend(shard.tasks)
        assert len(all_tasks) == 10
        seeds = sorted(t["seed"] for t in all_tasks)
        assert seeds == list(range(10))


class TestResultStoreEdgeCases:
    def test_overwrite_result(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="failed", attempts=1))
        store.save(ShardResult(shard_id=0, status="completed", attempts=2))
        loaded = store.load(0)
        assert loaded is not None
        assert loaded.status == "completed"
        assert loaded.attempts == 2

    def test_merge_deterministic_order(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        # Save in reverse order
        r1 = ShardResult(shard_id=1, status="completed")
        r1.records.append(RunRecord(scenario="b", seed=2, status="completed"))
        r0 = ShardResult(shard_id=0, status="completed")
        r0.records.append(RunRecord(scenario="a", seed=1, status="completed"))

        store.save(r1)
        store.save(r0)

        summary = store.merge_results(2, template_name="order_test")
        assert summary.total_runs == 2

    def test_pending_with_no_results(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        pending = store.pending_shards(5)
        assert pending == [0, 1, 2, 3, 4]
