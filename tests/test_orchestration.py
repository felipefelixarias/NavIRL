"""Tests for navirl/orchestration/ module: models, manifest, result_store, worker, orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from navirl.experiments.aggregator import BatchSummary, RunRecord
from navirl.experiments.templates import BatchTemplate
from navirl.orchestration.executor import LocalExecutor, TaskExecutor, _execute_single_task
from navirl.orchestration.manifest import ShardManifest, TaskShard
from navirl.orchestration.models import SimulationTask, TaskResult, TaskStatus
from navirl.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from navirl.orchestration.result_store import ResultStore, ShardResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "navirl" / "scenarios" / "library"


@pytest.fixture
def sample_task():
    return SimulationTask(
        task_id="test_0001",
        scenario_path="hallway_pass.yaml",
        seed=42,
        overrides={"scene.orca.neighbor_dist": 4.0},
    )


@pytest.fixture
def sample_result():
    return TaskResult(
        task_id="test_0001",
        status=TaskStatus.COMPLETED,
        metrics={"success_rate": 1.0, "collisions_agent_agent": 0},
        bundle_dir="/tmp/out/hallway_pass_001",
        wall_time_s=1.5,
    )


@pytest.fixture
def failed_result():
    return TaskResult(
        task_id="test_0002",
        status=TaskStatus.FAILED,
        error="Scenario file not found",
        wall_time_s=0.1,
    )


@pytest.fixture
def sample_tasks():
    """Six expanded task dicts as BatchTemplate.expand_tasks() returns."""
    return [
        {"scenario": Path("a.yaml"), "seed": 1, "overrides": {}},
        {"scenario": Path("a.yaml"), "seed": 2, "overrides": {}},
        {"scenario": Path("b.yaml"), "seed": 1, "overrides": {}},
        {"scenario": Path("b.yaml"), "seed": 2, "overrides": {}},
        {"scenario": Path("c.yaml"), "seed": 1, "overrides": {}},
        {"scenario": Path("c.yaml"), "seed": 2, "overrides": {}},
    ]


@pytest.fixture
def sample_shard_result():
    return ShardResult(
        shard_id=0,
        manifest_id="abc123",
        records=[
            RunRecord(scenario="hallway", seed=42, metrics={"success_rate": 1.0}, status="completed"),
            RunRecord(scenario="kitchen", seed=42, metrics={"success_rate": 0.5}, status="completed"),
        ],
        status="completed",
        attempts=1,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:01:00+00:00",
    )


# ===========================================================================
# SimulationTask tests
# ===========================================================================


class TestSimulationTask:
    def test_creation(self, sample_task):
        assert sample_task.task_id == "test_0001"
        assert sample_task.scenario_path == "hallway_pass.yaml"
        assert sample_task.seed == 42
        assert sample_task.overrides == {"scene.orca.neighbor_dist": 4.0}

    def test_default_overrides(self):
        task = SimulationTask(task_id="t1", scenario_path="s.yaml", seed=1)
        assert task.overrides == {}

    def test_content_hash_deterministic(self, sample_task):
        h1 = sample_task.content_hash()
        h2 = sample_task.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_differs_for_different_seeds(self, sample_task):
        other = SimulationTask(
            task_id="test_0001",
            scenario_path="hallway_pass.yaml",
            seed=99,
            overrides={"scene.orca.neighbor_dist": 4.0},
        )
        assert sample_task.content_hash() != other.content_hash()

    def test_content_hash_ignores_task_id(self, sample_task):
        other = SimulationTask(
            task_id="different_id",
            scenario_path="hallway_pass.yaml",
            seed=42,
            overrides={"scene.orca.neighbor_dist": 4.0},
        )
        assert sample_task.content_hash() == other.content_hash()

    def test_content_hash_differs_for_different_overrides(self):
        t1 = SimulationTask(task_id="t", scenario_path="s.yaml", seed=1, overrides={"a": 1})
        t2 = SimulationTask(task_id="t", scenario_path="s.yaml", seed=1, overrides={"a": 2})
        assert t1.content_hash() != t2.content_hash()


# ===========================================================================
# TaskResult tests
# ===========================================================================


class TestTaskResult:
    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["task_id"] == "test_0001"
        assert d["status"] == "completed"
        assert d["metrics"]["success_rate"] == 1.0
        assert d["bundle_dir"] == "/tmp/out/hallway_pass_001"
        assert d["wall_time_s"] == 1.5
        assert d["error"] == ""

    def test_from_dict_roundtrip(self, sample_result):
        d = sample_result.to_dict()
        restored = TaskResult.from_dict(d)
        assert restored.task_id == sample_result.task_id
        assert restored.status == sample_result.status
        assert restored.metrics == sample_result.metrics
        assert restored.bundle_dir == sample_result.bundle_dir
        assert restored.wall_time_s == sample_result.wall_time_s

    def test_failed_result_to_dict(self, failed_result):
        d = failed_result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Scenario file not found"
        assert d["metrics"] == {}
        assert d["bundle_dir"] == ""

    def test_from_dict_failed(self, failed_result):
        d = failed_result.to_dict()
        restored = TaskResult.from_dict(d)
        assert restored.status == TaskStatus.FAILED
        assert restored.error == "Scenario file not found"

    def test_from_dict_minimal(self):
        d = {"task_id": "x", "status": "pending"}
        r = TaskResult.from_dict(d)
        assert r.task_id == "x"
        assert r.status == TaskStatus.PENDING
        assert r.metrics == {}
        assert r.wall_time_s == 0.0


# ===========================================================================
# TaskStatus tests
# ===========================================================================


class TestTaskStatus:
    def test_all_statuses(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"

    def test_from_string(self):
        assert TaskStatus("completed") == TaskStatus.COMPLETED
        assert TaskStatus("failed") == TaskStatus.FAILED


# ===========================================================================
# TaskShard tests
# ===========================================================================


class TestTaskShard:
    def test_creation(self):
        shard = TaskShard(shard_id=0, tasks=[{"scenario": Path("a.yaml"), "seed": 1, "overrides": {}}])
        assert shard.shard_id == 0
        assert shard.num_tasks == 1

    def test_empty_shard(self):
        shard = TaskShard(shard_id=5)
        assert shard.num_tasks == 0

    def test_to_dict(self):
        shard = TaskShard(shard_id=0, tasks=[{"scenario": Path("a.yaml"), "seed": 1, "overrides": {}}])
        d = shard.to_dict()
        assert d["shard_id"] == 0
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["scenario"] == "a.yaml"

    def test_from_dict_roundtrip(self):
        shard = TaskShard(shard_id=3, tasks=[{"scenario": Path("b.yaml"), "seed": 7, "overrides": {"x": 1}}])
        d = shard.to_dict()
        restored = TaskShard.from_dict(d)
        assert restored.shard_id == 3
        assert restored.num_tasks == 1
        assert restored.tasks[0]["seed"] == 7


# ===========================================================================
# ShardManifest tests
# ===========================================================================


class TestShardManifest:
    def test_from_tasks_basic(self, sample_tasks):
        manifest = ShardManifest.from_tasks(sample_tasks, num_shards=2, template_name="test")
        assert manifest.num_shards == 2
        assert manifest.total_tasks == 6
        assert manifest.template_name == "test"
        assert manifest.manifest_id != ""

    def test_from_tasks_round_robin(self, sample_tasks):
        manifest = ShardManifest.from_tasks(sample_tasks, num_shards=3)
        assert manifest.shards[0].num_tasks == 2
        assert manifest.shards[1].num_tasks == 2
        assert manifest.shards[2].num_tasks == 2

    def test_from_tasks_more_shards_than_tasks(self):
        tasks = [{"scenario": Path("a.yaml"), "seed": 1, "overrides": {}}]
        manifest = ShardManifest.from_tasks(tasks, num_shards=5)
        assert manifest.num_shards == 1  # Clamped to task count
        assert manifest.total_tasks == 1

    def test_from_tasks_single_shard(self, sample_tasks):
        manifest = ShardManifest.from_tasks(sample_tasks, num_shards=1)
        assert manifest.num_shards == 1
        assert manifest.shards[0].num_tasks == 6

    def test_from_tasks_invalid_num_shards(self):
        with pytest.raises(ValueError, match="num_shards must be >= 1"):
            ShardManifest.from_tasks([], num_shards=0)

    def test_manifest_id_deterministic(self, sample_tasks):
        m1 = ShardManifest.from_tasks(sample_tasks, num_shards=2, template_name="t")
        m2 = ShardManifest.from_tasks(sample_tasks, num_shards=2, template_name="t")
        assert m1.manifest_id == m2.manifest_id

    def test_manifest_id_differs_for_different_shards(self, sample_tasks):
        m1 = ShardManifest.from_tasks(sample_tasks, num_shards=2)
        m2 = ShardManifest.from_tasks(sample_tasks, num_shards=3)
        assert m1.manifest_id != m2.manifest_id

    def test_to_dict(self, sample_tasks):
        manifest = ShardManifest.from_tasks(sample_tasks, num_shards=2, template_name="t")
        d = manifest.to_dict()
        assert d["template_name"] == "t"
        assert d["num_shards"] == 2
        assert d["total_tasks"] == 6
        assert len(d["shards"]) == 2

    def test_from_dict_roundtrip(self, sample_tasks):
        manifest = ShardManifest.from_tasks(sample_tasks, num_shards=2, template_name="t")
        d = manifest.to_dict()
        restored = ShardManifest.from_dict(d)
        assert restored.template_name == manifest.template_name
        assert restored.num_shards == manifest.num_shards
        assert restored.total_tasks == manifest.total_tasks

    def test_save_and_load(self, tmp_path, sample_tasks):
        manifest = ShardManifest.from_tasks(sample_tasks, num_shards=2, template_name="t")
        path = tmp_path / "manifest.yaml"
        manifest.save(path)
        loaded = ShardManifest.load(path)
        assert loaded.template_name == "t"
        assert loaded.num_shards == 2
        assert loaded.total_tasks == 6
        assert loaded.manifest_id == manifest.manifest_id


# ===========================================================================
# ShardResult tests
# ===========================================================================


class TestShardResult:
    def test_creation(self, sample_shard_result):
        assert sample_shard_result.shard_id == 0
        assert sample_shard_result.status == "completed"
        assert len(sample_shard_result.records) == 2
        assert sample_shard_result.attempts == 1

    def test_to_dict(self, sample_shard_result):
        d = sample_shard_result.to_dict()
        assert d["shard_id"] == 0
        assert d["status"] == "completed"
        assert len(d["records"]) == 2
        assert d["records"][0]["scenario"] == "hallway"

    def test_from_dict_roundtrip(self, sample_shard_result):
        d = sample_shard_result.to_dict()
        restored = ShardResult.from_dict(d)
        assert restored.shard_id == sample_shard_result.shard_id
        assert restored.status == sample_shard_result.status
        assert len(restored.records) == len(sample_shard_result.records)
        assert restored.records[0].scenario == "hallway"
        assert restored.records[0].metrics["success_rate"] == 1.0

    def test_from_dict_defaults(self):
        d = {"shard_id": 5}
        r = ShardResult.from_dict(d)
        assert r.shard_id == 5
        assert r.status == "pending"
        assert r.records == []
        assert r.attempts == 0

    def test_empty_result(self):
        r = ShardResult(shard_id=0)
        assert r.status == "pending"
        assert r.records == []
        assert r.error is None


# ===========================================================================
# ResultStore tests
# ===========================================================================


class TestResultStore:
    def test_save_and_load(self, tmp_path, sample_shard_result):
        store = ResultStore(tmp_path / "results")
        store.save(sample_shard_result)
        loaded = store.load(0)
        assert loaded is not None
        assert loaded.shard_id == 0
        assert loaded.status == "completed"
        assert len(loaded.records) == 2

    def test_load_nonexistent(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        assert store.load(99) is None

    def test_load_all(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        store.save(ShardResult(shard_id=1, status="failed"))
        results = store.load_all(3)
        assert len(results) == 3
        assert results[0] is not None
        assert results[0].status == "completed"
        assert results[1] is not None
        assert results[1].status == "failed"
        assert results[2] is None

    def test_completed_shards(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        store.save(ShardResult(shard_id=1, status="failed"))
        store.save(ShardResult(shard_id=2, status="completed"))
        assert store.completed_shards(3) == [0, 2]

    def test_pending_shards(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(shard_id=0, status="completed"))
        store.save(ShardResult(shard_id=2, status="completed"))
        pending = store.pending_shards(4)
        assert pending == [1, 3]

    def test_merge_results(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(
            shard_id=0,
            records=[RunRecord(scenario="a", seed=1, metrics={"success_rate": 1.0}, status="completed")],
            status="completed",
        ))
        store.save(ShardResult(
            shard_id=1,
            records=[RunRecord(scenario="b", seed=2, metrics={"success_rate": 0.5}, status="completed")],
            status="completed",
        ))
        summary = store.merge_results(num_shards=2, template_name="test")
        assert isinstance(summary, BatchSummary)
        assert summary.total_runs == 2
        assert summary.completed_runs == 2

    def test_merge_with_missing_shard(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        store.save(ShardResult(
            shard_id=0,
            records=[RunRecord(scenario="a", seed=1, metrics={}, status="completed")],
            status="completed",
        ))
        # Shard 1 is missing
        summary = store.merge_results(num_shards=2, template_name="test")
        assert summary.total_runs == 1

    def test_creates_directory(self, tmp_path):
        store = ResultStore(tmp_path / "deep" / "nested" / "results")
        assert store.root.exists()

    def test_shard_file_is_valid_json(self, tmp_path, sample_shard_result):
        store = ResultStore(tmp_path / "results")
        path = store.save(sample_shard_result)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert data["shard_id"] == 0


# ===========================================================================
# OrchestratorConfig tests
# ===========================================================================


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.num_shards == 4
        assert cfg.max_retries == 2
        assert cfg.max_workers is None
        assert cfg.render is False
        assert cfg.video is False

    def test_custom_values(self):
        cfg = OrchestratorConfig(num_shards=8, max_retries=5, max_workers=2, render=True, video=True)
        assert cfg.num_shards == 8
        assert cfg.max_retries == 5
        assert cfg.max_workers == 2
        assert cfg.render is True
        assert cfg.video is True


# ===========================================================================
# Orchestrator tests
# ===========================================================================


class TestOrchestrator:
    def _make_template(self):
        return BatchTemplate(
            name="orch_test",
            description="Orchestrator test template",
            scenarios=["hallway_pass.yaml"],
            seeds=[42],
        )

    def test_manifest_property(self):
        template = self._make_template()
        orch = Orchestrator(template=template, out_root="/tmp/orch_test")
        manifest = orch.manifest
        assert isinstance(manifest, ShardManifest)
        assert manifest.template_name == "orch_test"

    def test_manifest_cached(self):
        template = self._make_template()
        orch = Orchestrator(template=template, out_root="/tmp/orch_test")
        m1 = orch.manifest
        m2 = orch.manifest
        assert m1 is m2

    def test_config_defaults(self):
        template = self._make_template()
        orch = Orchestrator(template=template, out_root="/tmp/orch_test")
        assert orch.config.num_shards == 4
        assert orch.config.max_retries == 2

    def test_custom_config(self):
        template = self._make_template()
        cfg = OrchestratorConfig(num_shards=2, max_retries=1)
        orch = Orchestrator(template=template, out_root="/tmp/orch_test", config=cfg)
        assert orch.config.num_shards == 2

    def test_save_manifest(self, tmp_path):
        template = self._make_template()
        orch = Orchestrator(template=template, out_root=str(tmp_path))
        path = orch.save_manifest()
        assert path.exists()
        loaded = ShardManifest.load(path)
        assert loaded.template_name == "orch_test"


# ===========================================================================
# TaskExecutor interface tests
# ===========================================================================


class TestTaskExecutorInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            TaskExecutor()

    def test_subclass_must_implement_execute_batch(self):
        class Incomplete(TaskExecutor):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass(self):
        class Dummy(TaskExecutor):
            def execute_batch(self, tasks, out_root, *, progress_callback=None):
                return []

        executor = Dummy()
        assert executor.execute_batch([], "/tmp") == []


# ===========================================================================
# LocalExecutor tests
# ===========================================================================


class TestLocalExecutor:
    def test_default_max_workers(self):
        executor = LocalExecutor()
        assert executor.max_workers >= 1

    def test_explicit_max_workers(self):
        executor = LocalExecutor(max_workers=4)
        assert executor.max_workers == 4

    def test_min_workers_is_one(self):
        executor = LocalExecutor(max_workers=0)
        assert executor.max_workers == 1

    def test_negative_workers_clamped(self):
        executor = LocalExecutor(max_workers=-5)
        assert executor.max_workers == 1

    def test_empty_batch_returns_empty(self):
        executor = LocalExecutor(max_workers=1)
        results = executor.execute_batch([], "/tmp/out")
        assert results == []

    @patch("navirl.orchestration.executor._worker_entry")
    def test_sequential_execution_calls_worker(self, mock_worker, tmp_path):
        mock_worker.return_value = {
            "task_id": "t1",
            "status": "completed",
            "metrics": {"success_rate": 1.0},
            "bundle_dir": str(tmp_path),
            "error": "",
            "wall_time_s": 0.5,
        }
        executor = LocalExecutor(max_workers=1)
        tasks = [SimulationTask(task_id="t1", scenario_path="s.yaml", seed=1)]
        results = executor.execute_batch(tasks, str(tmp_path))
        assert len(results) == 1
        assert results[0].task_id == "t1"
        assert results[0].status == TaskStatus.COMPLETED

    @patch("navirl.orchestration.executor._worker_entry")
    def test_progress_callback_called(self, mock_worker, tmp_path):
        mock_worker.return_value = {
            "task_id": "t1",
            "status": "completed",
            "metrics": {},
            "bundle_dir": "",
            "error": "",
            "wall_time_s": 0.1,
        }
        callback = MagicMock()
        executor = LocalExecutor(max_workers=1)
        tasks = [SimulationTask(task_id="t1", scenario_path="s.yaml", seed=1)]
        executor.execute_batch(tasks, str(tmp_path), progress_callback=callback)
        callback.assert_called_once_with(1, 1)

    @patch("navirl.orchestration.executor._worker_entry")
    def test_multiple_tasks_sequential(self, mock_worker, tmp_path):
        mock_worker.side_effect = [
            {"task_id": f"t{i}", "status": "completed", "metrics": {}, "bundle_dir": "", "error": "", "wall_time_s": 0.1}
            for i in range(3)
        ]
        executor = LocalExecutor(max_workers=1)
        tasks = [
            SimulationTask(task_id=f"t{i}", scenario_path="s.yaml", seed=i)
            for i in range(3)
        ]
        results = executor.execute_batch(tasks, str(tmp_path))
        assert len(results) == 3
        assert all(r.status == TaskStatus.COMPLETED for r in results)


# ===========================================================================
# ShardWorker tests
# ===========================================================================


class TestShardWorker:
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    def test_all_tasks_succeed(self, mock_seed, mock_load, mock_run, tmp_path):
        mock_load.return_value = {"name": "test", "seed": 1}

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        state = bundle / "state.jsonl"
        state.write_text('{"step": 0}\n')
        (bundle / "scenario.yaml").write_text("name: test\n")

        mock_episode = MagicMock()
        mock_episode.state_path = str(state)
        mock_run.return_value = mock_episode

        with patch("navirl.orchestration.worker.StandardMetrics") as mock_m:
            mock_m.return_value.compute.return_value = {"success_rate": 1.0}

            from navirl.orchestration.worker import ShardWorker

            shard = TaskShard(shard_id=0, tasks=[{"scenario": Path("a.yaml"), "seed": 1, "overrides": {}}])
            worker = ShardWorker(shard=shard, out_root=str(tmp_path))
            result = worker.run()

        assert result.status == "completed"
        assert len(result.records) == 1
        assert result.records[0].status == "completed"

    def test_all_tasks_fail(self, tmp_path):
        with patch("navirl.orchestration.worker.load_scenario", side_effect=FileNotFoundError("nope")):
            from navirl.orchestration.worker import ShardWorker

            shard = TaskShard(shard_id=0, tasks=[{"scenario": Path("bad.yaml"), "seed": 1, "overrides": {}}])
            worker = ShardWorker(shard=shard, out_root=str(tmp_path))
            result = worker.run()

        assert result.status == "failed"
        assert len(result.records) == 1
        assert result.records[0].status == "failed"

    def test_saves_to_result_store(self, tmp_path):
        store = ResultStore(tmp_path / "results")

        with patch("navirl.orchestration.worker.load_scenario", side_effect=FileNotFoundError("nope")):
            from navirl.orchestration.worker import ShardWorker

            shard = TaskShard(shard_id=0, tasks=[{"scenario": Path("x.yaml"), "seed": 1, "overrides": {}}])
            worker = ShardWorker(shard=shard, out_root=str(tmp_path), result_store=store)
            worker.run()

        loaded = store.load(0)
        assert loaded is not None
        assert loaded.shard_id == 0


# ===========================================================================
# _execute_single_task tests
# ===========================================================================


class TestExecuteSingleTask:
    @patch("navirl.orchestration.executor.run_scenario_dict")
    @patch("navirl.orchestration.executor.load_scenario")
    @patch("navirl.orchestration.executor.set_global_seed")
    def test_successful_execution(self, mock_seed, mock_load, mock_run, tmp_path):
        mock_load.return_value = {"name": "test", "seed": 1}

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        state = bundle / "state.jsonl"
        state.write_text('{"step": 0}\n')
        (bundle / "scenario.yaml").write_text("name: test\n")

        mock_episode = MagicMock()
        mock_episode.state_path = str(state)
        mock_run.return_value = mock_episode

        with patch("navirl.orchestration.executor.StandardMetrics") as mock_m:
            mock_m.return_value.compute.return_value = {"success_rate": 1.0}

            task = SimulationTask(task_id="t1", scenario_path="s.yaml", seed=42)
            result = _execute_single_task(task, str(tmp_path))

        assert result.status == TaskStatus.COMPLETED
        assert result.metrics == {"success_rate": 1.0}
        assert result.wall_time_s > 0

    def test_failed_execution_returns_error(self, tmp_path):
        with patch("navirl.orchestration.executor.load_scenario", side_effect=FileNotFoundError("nope")):
            task = SimulationTask(task_id="t1", scenario_path="nonexistent.yaml", seed=1)
            result = _execute_single_task(task, str(tmp_path))

        assert result.status == TaskStatus.FAILED
        assert "nope" in result.error
        assert result.wall_time_s > 0


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestCLIOrchestrate:
    def test_orchestrate_parser_exists(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["orchestrate", "template.yaml"])
        assert args.command == "orchestrate"
        assert args.template == "template.yaml"
        assert args.shards == 4
        assert args.workers == 0
        assert args.retries == 2
        assert args.resume is False

    def test_orchestrate_parser_with_options(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "orchestrate", "t.yaml",
            "--shards", "8",
            "--workers", "4",
            "--retries", "3",
            "--resume",
            "--out", "/tmp/custom",
        ])
        assert args.shards == 8
        assert args.workers == 4
        assert args.retries == 3
        assert args.resume is True
        assert args.out == "/tmp/custom"

    def test_orchestrate_render_video_flags(self):
        from navirl.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["orchestrate", "t.yaml", "--render", "--video"])
        assert args.render is True
        assert args.video is True


# ===========================================================================
# Module import tests
# ===========================================================================


class TestModuleImports:
    def test_top_level_imports(self):
        from navirl.orchestration import (
            Orchestrator,
            OrchestratorConfig,
            ResultStore,
            ShardManifest,
            ShardResult,
            ShardWorker,
            TaskShard,
        )

        assert Orchestrator is not None
        assert OrchestratorConfig is not None
        assert ResultStore is not None
        assert ShardManifest is not None
        assert ShardResult is not None
        assert ShardWorker is not None
        assert TaskShard is not None
