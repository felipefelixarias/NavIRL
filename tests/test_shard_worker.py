"""Tests for navirl/orchestration/worker.py — ShardWorker.run().

Covers the ShardWorker execution loop including:
  - All-tasks-succeed → status "completed"
  - All-tasks-fail → status "failed"
  - Mixed success/failure → status "partial"
  - Result store persistence
  - Timestamp population
  - Single-task shards
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from navirl.experiments.aggregator import RunRecord
from navirl.orchestration.manifest import TaskShard
from navirl.orchestration.result_store import ResultStore, ShardResult
from navirl.orchestration.worker import ShardWorker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shard(shard_id: int = 0, tasks: list | None = None) -> TaskShard:
    """Create a TaskShard with simple task dicts."""
    if tasks is None:
        tasks = [
            {"scenario": Path("hallway.yaml"), "seed": 42, "overrides": {}},
            {"scenario": Path("lobby.yaml"), "seed": 7, "overrides": {"dt": 0.05}},
        ]
    return TaskShard(shard_id=shard_id, tasks=tasks)


def _mock_successful_pipeline(scenario_dict, out_root, render_override, video_override):
    """Fake run_scenario_dict that writes a minimal state file."""
    out = Path(out_root) / "run_0001"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.npz"
    state_path.write_bytes(b"fake")
    scenario_yaml = out / "scenario.yaml"
    scenario_yaml.write_text("dt: 0.1\n", encoding="utf-8")
    log = MagicMock()
    log.state_path = str(state_path)
    return log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShardWorkerInit:
    def test_basic_construction(self, tmp_path):
        shard = _make_shard()
        worker = ShardWorker(shard, tmp_path)
        assert worker.shard is shard
        assert worker.out_root == tmp_path
        assert worker.render is False
        assert worker.video is False
        assert worker.result_store is None

    def test_with_result_store(self, tmp_path):
        store = ResultStore(tmp_path / "results")
        shard = _make_shard()
        worker = ShardWorker(shard, tmp_path, result_store=store, manifest_id="abc123")
        assert worker.result_store is store
        assert worker.manifest_id == "abc123"

    def test_render_and_video_flags(self, tmp_path):
        worker = ShardWorker(_make_shard(), tmp_path, render=True, video=True)
        assert worker.render is True
        assert worker.video is True


class TestShardWorkerAllSucceed:
    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_all_tasks_complete(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        mock_load.return_value = {"dt": 0.1}
        mock_overrides.return_value = {"dt": 0.1}
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {"success": True}
        mock_metrics.return_value = mock_metrics_instance

        shard = _make_shard(shard_id=0)
        worker = ShardWorker(shard, tmp_path, manifest_id="test_mid")
        result = worker.run()

        assert result.status == "completed"
        assert result.shard_id == 0
        assert result.manifest_id == "test_mid"
        assert result.attempts == 1
        assert len(result.records) == 2
        assert all(r.status == "completed" for r in result.records)
        assert result.started_at != ""
        assert result.finished_at != ""

    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_records_contain_correct_scenario_names(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        mock_load.return_value = {"dt": 0.1}
        mock_overrides.return_value = {"dt": 0.1}
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {}
        mock_metrics.return_value = mock_metrics_instance

        shard = _make_shard()
        result = ShardWorker(shard, tmp_path).run()

        assert result.records[0].scenario == "hallway"
        assert result.records[1].scenario == "lobby"

    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_records_contain_seeds(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        mock_load.return_value = {"dt": 0.1}
        mock_overrides.return_value = {"dt": 0.1}
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {}
        mock_metrics.return_value = mock_metrics_instance

        shard = _make_shard()
        result = ShardWorker(shard, tmp_path).run()

        assert result.records[0].seed == 42
        assert result.records[1].seed == 7


class TestShardWorkerAllFail:
    @patch("navirl.orchestration.worker.load_scenario", side_effect=FileNotFoundError("nope"))
    def test_all_tasks_fail(self, mock_load, tmp_path):
        shard = _make_shard(shard_id=1)
        result = ShardWorker(shard, tmp_path).run()

        assert result.status == "failed"
        assert result.shard_id == 1
        assert len(result.records) == 2
        assert all(r.status == "failed" for r in result.records)
        assert all("nope" in r.error for r in result.records)

    @patch("navirl.orchestration.worker.load_scenario", side_effect=RuntimeError("kaboom"))
    def test_error_messages_captured(self, mock_load, tmp_path):
        shard = _make_shard()
        result = ShardWorker(shard, tmp_path).run()

        for record in result.records:
            assert record.error == "kaboom"


class TestShardWorkerPartialFailure:
    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_mixed_results_give_partial(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        # First task succeeds, second fails
        def load_side_effect(path):
            if "lobby" in path:
                raise ValueError("bad scenario")
            return {"dt": 0.1}

        mock_load.side_effect = load_side_effect
        mock_overrides.return_value = {"dt": 0.1}
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {"ok": True}
        mock_metrics.return_value = mock_metrics_instance

        shard = _make_shard()
        result = ShardWorker(shard, tmp_path).run()

        assert result.status == "partial"
        assert result.records[0].status == "completed"
        assert result.records[1].status == "failed"


class TestShardWorkerResultStore:
    @patch("navirl.orchestration.worker.load_scenario", side_effect=Exception("err"))
    def test_result_persisted_to_store(self, mock_load, tmp_path):
        store = ResultStore(tmp_path / "results")
        shard = _make_shard(shard_id=3)
        worker = ShardWorker(shard, tmp_path, result_store=store)
        worker.run()

        loaded = store.load(3)
        assert loaded is not None
        assert loaded.shard_id == 3
        assert loaded.status == "failed"

    @patch("navirl.orchestration.worker.load_scenario", side_effect=Exception("err"))
    def test_no_store_means_no_persistence(self, mock_load, tmp_path):
        shard = _make_shard(shard_id=0)
        worker = ShardWorker(shard, tmp_path)  # no result_store
        result = worker.run()
        # Should still return result without error
        assert result.status == "failed"


class TestShardWorkerTimestamps:
    @patch("navirl.orchestration.worker.load_scenario", side_effect=Exception("err"))
    def test_timestamps_are_iso_format(self, mock_load, tmp_path):
        shard = _make_shard()
        result = ShardWorker(shard, tmp_path).run()

        # Verify both timestamps are valid ISO format
        started = datetime.fromisoformat(result.started_at)
        finished = datetime.fromisoformat(result.finished_at)
        assert finished >= started


class TestShardWorkerSingleTask:
    @patch("navirl.orchestration.worker.load_scenario", side_effect=Exception("err"))
    def test_single_task_shard(self, mock_load, tmp_path):
        tasks = [{"scenario": Path("single.yaml"), "seed": 1, "overrides": {}}]
        shard = _make_shard(shard_id=0, tasks=tasks)
        result = ShardWorker(shard, tmp_path).run()

        assert len(result.records) == 1
        assert result.records[0].scenario == "single"

    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_empty_overrides_not_applied(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        mock_load.return_value = {"dt": 0.1}
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {}
        mock_metrics.return_value = mock_metrics_instance

        # Task with no overrides key at all
        tasks = [{"scenario": Path("test.yaml"), "seed": 1}]
        shard = _make_shard(shard_id=0, tasks=tasks)
        ShardWorker(shard, tmp_path).run()

        mock_overrides.assert_not_called()


class TestShardWorkerOverrides:
    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_overrides_applied_when_present(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        mock_load.return_value = {"dt": 0.1}
        mock_overrides.return_value = {"dt": 0.05}
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {}
        mock_metrics.return_value = mock_metrics_instance

        tasks = [{"scenario": Path("test.yaml"), "seed": 1, "overrides": {"dt": 0.05}}]
        shard = _make_shard(shard_id=0, tasks=tasks)
        ShardWorker(shard, tmp_path).run()

        mock_overrides.assert_called_once()

    @patch("navirl.orchestration.worker.StandardMetrics")
    @patch("navirl.orchestration.worker.run_scenario_dict")
    @patch("navirl.orchestration.worker.load_scenario")
    @patch("navirl.orchestration.worker.set_global_seed")
    @patch("navirl.orchestration.worker._apply_overrides")
    def test_seed_set_on_scenario_and_globally(
        self, mock_overrides, mock_seed, mock_load, mock_run, mock_metrics, tmp_path
    ):
        scenario = {"dt": 0.1}
        mock_load.return_value = scenario
        mock_overrides.return_value = scenario
        mock_run.side_effect = lambda *a, **kw: _mock_successful_pipeline(*a, **kw)
        mock_metrics_instance = MagicMock()
        mock_metrics_instance.compute.return_value = {}
        mock_metrics.return_value = mock_metrics_instance

        tasks = [{"scenario": Path("test.yaml"), "seed": 99, "overrides": {"x": 1}}]
        shard = _make_shard(shard_id=0, tasks=tasks)
        ShardWorker(shard, tmp_path).run()

        # Verify seed was set on the scenario dict
        assert scenario["seed"] == 99
        # Verify global seed was set
        mock_seed.assert_called_once_with(99)
