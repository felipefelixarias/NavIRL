"""Tests for navirl.orchestration.executor — _worker_entry and the parallel branch.

The existing ``tests/test_orchestration.py`` covers ``_execute_single_task`` and
the sequential branch of ``LocalExecutor.execute_batch``.  This file targets
the multiprocessing-pool branch (``max_workers > 1``) and the
``_worker_entry`` adapter that is invoked across process boundaries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from navirl.orchestration.executor import (
    LocalExecutor,
    _execute_single_task,
    _worker_entry,
)
from navirl.orchestration.models import SimulationTask, TaskResult, TaskStatus

# ---------------------------------------------------------------------------
# _worker_entry
# ---------------------------------------------------------------------------


class TestWorkerEntry:
    @patch("navirl.orchestration.executor._execute_single_task")
    def test_reconstructs_task_and_returns_dict(self, mock_exec, tmp_path):
        mock_exec.return_value = TaskResult(
            task_id="abc",
            status=TaskStatus.COMPLETED,
            metrics={"success_rate": 1.0},
            wall_time_s=0.5,
        )
        task_data = {
            "task_id": "abc",
            "scenario_path": "s.yaml",
            "seed": 7,
            "overrides": {"a": 1},
        }
        result = _worker_entry((task_data, str(tmp_path)))

        # _execute_single_task should have been called with a reconstructed task
        mock_exec.assert_called_once()
        call_task = mock_exec.call_args[0][0]
        assert isinstance(call_task, SimulationTask)
        assert call_task.task_id == "abc"
        assert call_task.scenario_path == "s.yaml"
        assert call_task.seed == 7
        assert call_task.overrides == {"a": 1}

        # Result must be a serialisable dict (so it survives pickling)
        assert isinstance(result, dict)
        assert result["task_id"] == "abc"
        assert result["status"] == "completed"
        assert result["metrics"] == {"success_rate": 1.0}

    @patch("navirl.orchestration.executor._execute_single_task")
    def test_handles_missing_overrides_key(self, mock_exec, tmp_path):
        """``_worker_entry`` should default missing overrides to an empty dict."""
        mock_exec.return_value = TaskResult(task_id="x", status=TaskStatus.COMPLETED)
        task_data = {"task_id": "x", "scenario_path": "s.yaml", "seed": 0}
        result = _worker_entry((task_data, str(tmp_path)))
        assert mock_exec.call_args[0][0].overrides == {}
        assert result["task_id"] == "x"


# ---------------------------------------------------------------------------
# _execute_single_task — overrides path (line 78)
# ---------------------------------------------------------------------------


class TestExecuteSingleTaskOverrides:
    @patch("navirl.orchestration.executor.run_scenario_dict")
    @patch("navirl.orchestration.executor.load_scenario")
    @patch("navirl.orchestration.executor.set_global_seed")
    def test_overrides_are_applied(self, _seed, mock_load, mock_run, tmp_path):
        mock_load.return_value = {"id": "test", "scene": {"orca": {}}}

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        state = bundle / "state.jsonl"
        state.write_text('{"step": 0}\n')
        (bundle / "scenario.yaml").write_text("id: test\n")

        episode = MagicMock()
        episode.state_path = str(state)
        mock_run.return_value = episode

        with patch("navirl.orchestration.executor.StandardMetrics") as mock_m:
            mock_m.return_value.compute.return_value = {"success_rate": 1.0}

            task = SimulationTask(
                task_id="ovr",
                scenario_path="s.yaml",
                seed=1,
                overrides={"scene.orca.neighbor_dist": 9.5},
            )
            result = _execute_single_task(task, str(tmp_path))

        # The scenario passed to run_scenario_dict must have the override applied
        scenario_arg = mock_run.call_args.args[0]
        assert scenario_arg["scene"]["orca"]["neighbor_dist"] == 9.5
        # The overrides path must not corrupt the success result
        assert result.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# LocalExecutor — parallel branch (real multiprocessing.Pool)
# ---------------------------------------------------------------------------


def _success_worker(args):
    """Picklable replacement for ``_worker_entry`` used by the parallel test."""
    task_data, _out_root = args
    return {
        "task_id": task_data["task_id"],
        "status": "completed",
        "metrics": {"seed": task_data["seed"]},
        "bundle_dir": "",
        "error": "",
        "wall_time_s": 0.01,
    }


class TestLocalExecutorParallel:
    def test_pool_branch_runs_all_tasks(self, tmp_path):
        """``max_workers > 1`` should dispatch through ``mp.Pool.imap``."""
        with patch("navirl.orchestration.executor._worker_entry", new=_success_worker):
            executor = LocalExecutor(max_workers=2)
            tasks = [
                SimulationTask(task_id=f"t{i}", scenario_path="s.yaml", seed=i) for i in range(3)
            ]
            results = executor.execute_batch(tasks, str(tmp_path))
        assert len(results) == 3
        # imap preserves order
        assert [r.task_id for r in results] == ["t0", "t1", "t2"]
        for r in results:
            assert r.status == TaskStatus.COMPLETED

    def test_pool_branch_invokes_progress_callback(self, tmp_path):
        callback = MagicMock()
        with patch("navirl.orchestration.executor._worker_entry", new=_success_worker):
            executor = LocalExecutor(max_workers=2)
            tasks = [
                SimulationTask(task_id=f"t{i}", scenario_path="s.yaml", seed=i) for i in range(2)
            ]
            executor.execute_batch(tasks, str(tmp_path), progress_callback=callback)
        # Called once per task, with (completed, total)
        assert callback.call_count == 2
        callback.assert_any_call(2, 2)

    def test_pool_branch_creates_out_root(self, tmp_path):
        out = tmp_path / "new_out_root"
        assert not out.exists()
        with patch("navirl.orchestration.executor._worker_entry", new=_success_worker):
            executor = LocalExecutor(max_workers=2)
            tasks = [SimulationTask(task_id="t0", scenario_path="s.yaml", seed=0)]
            executor.execute_batch(tasks, str(out))
        assert out.exists()

    def test_pool_branch_clamps_to_task_count(self, tmp_path):
        """When max_workers > num_tasks, only num_tasks workers are spun up."""
        with (
            patch("navirl.orchestration.executor._worker_entry", new=_success_worker),
            patch("navirl.orchestration.executor.mp") as mp_mod,
        ):
            # Mimic the real mp.Pool context manager but observe args
            pool = MagicMock()
            pool.imap.return_value = iter(
                [
                    {
                        "task_id": "only",
                        "status": "completed",
                        "metrics": {},
                        "bundle_dir": "",
                        "error": "",
                        "wall_time_s": 0.01,
                    }
                ]
            )
            mp_mod.Pool.return_value.__enter__.return_value = pool
            mp_mod.Pool.return_value.__exit__.return_value = False
            mp_mod.cpu_count.return_value = 16

            executor = LocalExecutor(max_workers=8)
            tasks = [SimulationTask(task_id="only", scenario_path="s.yaml", seed=0)]
            executor.execute_batch(tasks, str(tmp_path))

            # 8 workers requested, 1 task → Pool gets created with 1
            mp_mod.Pool.assert_called_once_with(processes=1)
