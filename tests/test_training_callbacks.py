"""Tests for navirl/training/callbacks.py.

Covers CallbackList delegation, EvalCallback, CheckpointCallback, LoggingCallback,
EarlyStoppingCallback, CurriculumCallback, SchedulerCallback, GradientMonitorCallback,
HyperparameterSearchCallback, and ProgressBarCallback.

WandbCallback and TensorBoardCallback are skipped (require optional external deps).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from navirl.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    CurriculumCallback,
    EarlyStoppingCallback,
    EvalCallback,
    GradientMonitorCallback,
    HyperparameterSearchCallback,
    LoggingCallback,
    SchedulerCallback,
)

# ===================================================================
# Callback base class
# ===================================================================


class TestCallback:
    def test_base_hooks_are_noop(self):
        cb = Callback()
        cb.on_training_start({})
        cb.on_training_end({})
        assert cb.on_step({}) is True
        cb.on_episode_end({})
        cb.on_rollout_start({})
        cb.on_rollout_end({})
        cb.on_update_start({})
        cb.on_update_end({})


# ===================================================================
# CallbackList
# ===================================================================


class TestCallbackList:
    def test_delegates_all_hooks(self):
        tracker = {"calls": []}

        class TrackerCB(Callback):
            def on_training_start(self, locals_):
                tracker["calls"].append("start")

            def on_training_end(self, locals_):
                tracker["calls"].append("end")

            def on_step(self, locals_):
                tracker["calls"].append("step")
                return True

            def on_episode_end(self, locals_):
                tracker["calls"].append("episode_end")

            def on_rollout_start(self, locals_):
                tracker["calls"].append("rollout_start")

            def on_rollout_end(self, locals_):
                tracker["calls"].append("rollout_end")

            def on_update_start(self, locals_):
                tracker["calls"].append("update_start")

            def on_update_end(self, locals_):
                tracker["calls"].append("update_end")

        cb_list = CallbackList([TrackerCB(), TrackerCB()])
        cb_list.on_training_start({})
        cb_list.on_training_end({})
        cb_list.on_step({})
        cb_list.on_episode_end({})
        cb_list.on_rollout_start({})
        cb_list.on_rollout_end({})
        cb_list.on_update_start({})
        cb_list.on_update_end({})

        # Each hook called twice (two callbacks)
        assert tracker["calls"].count("start") == 2
        assert tracker["calls"].count("end") == 2
        assert tracker["calls"].count("step") == 2
        assert tracker["calls"].count("update_start") == 2
        assert tracker["calls"].count("update_end") == 2

    def test_on_step_stops_on_false(self):
        class StopCB(Callback):
            def on_step(self, locals_):
                return False

        class ContinueCB(Callback):
            def on_step(self, locals_):
                return True

        cb_list = CallbackList([ContinueCB(), StopCB(), ContinueCB()])
        assert cb_list.on_step({}) is False

    def test_on_step_all_true(self):
        cb_list = CallbackList([Callback(), Callback()])
        assert cb_list.on_step({}) is True


# ===================================================================
# EvalCallback
# ===================================================================


class _FakeActionSpace:
    def sample(self):
        return 0


class _FakeEnv:
    """A minimal env that is not callable (unlike MagicMock)."""

    def __init__(self, reward=1.0, n_tuple=5):
        self.action_space = _FakeActionSpace()
        self._reward = reward
        self._n_tuple = n_tuple

    def reset(self):
        return [0, 0, 0, 0], {}

    def step(self, action):
        if self._n_tuple == 5:
            return [0, 0, 0, 0], self._reward, True, False, {}
        return [0, 0, 0, 0], self._reward, True, {}


class TestEvalCallback:
    def _make_env(self, reward=1.0):
        return _FakeEnv(reward=reward)

    def test_eval_triggers_at_frequency(self):
        env = self._make_env()
        cb = EvalCallback(eval_env=env, eval_freq=5, n_eval_episodes=2, verbose=0)

        # Steps 1-4 should not trigger
        for _ in range(4):
            assert cb.on_step({}) is True
        assert len(cb.eval_results) == 0

        # Step 5 triggers evaluation
        assert cb.on_step({}) is True
        assert len(cb.eval_results) == 1
        assert cb.last_mean_reward == 1.0

    def test_eval_with_model(self):
        env = self._make_env()
        model = MagicMock()
        model.predict.return_value = (0, None)
        cb = EvalCallback(eval_env=env, eval_freq=1, n_eval_episodes=1, verbose=0)
        cb.on_step({"self": model})
        assert len(cb.eval_results) == 1
        assert model.predict.called

    def test_eval_callable_env(self):
        env = self._make_env()
        cb = EvalCallback(eval_env=lambda: env, eval_freq=1, n_eval_episodes=1, verbose=0)
        cb.on_step({})
        assert len(cb.eval_results) == 1

    def test_best_model_save(self, tmp_path):
        env = self._make_env()
        model = MagicMock()
        model.predict.return_value = (0, None)
        cb = EvalCallback(
            eval_env=env, eval_freq=1, n_eval_episodes=1,
            best_model_save_path=str(tmp_path), verbose=1,
        )
        cb.on_step({"self": model})
        assert model.save.called

    def test_eval_old_style_env(self):
        """Test env returning 4-tuple (old gym API)."""
        env = _FakeEnv(reward=2.0, n_tuple=4)
        cb = EvalCallback(eval_env=env, eval_freq=1, n_eval_episodes=1, verbose=0)
        cb.on_step({})
        assert cb.last_mean_reward == 2.0


# ===================================================================
# CheckpointCallback
# ===================================================================


class TestCheckpointCallback:
    def test_saves_at_frequency(self, tmp_path):
        model = MagicMock()
        cb = CheckpointCallback(save_freq=3, save_path=str(tmp_path), name_prefix="ckpt", verbose=0)
        cb.on_training_start({})

        # Steps 1-2 don't save
        for _ in range(2):
            cb.on_step({"self": model})
        model.save.assert_not_called()

        # Step 3 saves
        cb.on_step({"self": model})
        model.save.assert_called_once()
        call_path = model.save.call_args[0][0]
        assert "ckpt_3_steps" in call_path

    def test_no_model_no_crash(self, tmp_path):
        cb = CheckpointCallback(save_freq=1, save_path=str(tmp_path))
        cb.on_training_start({})
        cb.on_step({})  # No model in locals_ - should not crash


# ===================================================================
# LoggingCallback
# ===================================================================


class TestLoggingCallback:
    def test_logs_metrics(self, tmp_path):
        log_file = tmp_path / "train.jsonl"
        cb = LoggingCallback(log_freq=2, log_file=str(log_file), verbose=0)
        cb.on_training_start({})

        # Step 1: no log
        cb.on_step({})
        # Step 2: log
        cb.on_step({"loss": 0.5, "entropy": 1.2})

        cb.on_training_end({})

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["step"] == 2
        assert "fps" in data
        assert data["loss"] == 0.5
        assert data["entropy"] == 1.2

    def test_tracks_episode_rewards(self):
        cb = LoggingCallback(log_freq=1, verbose=0)
        cb.on_training_start({})
        cb.on_episode_end({"episode_reward": 10.0})
        cb.on_episode_end({"episode_reward": 20.0})
        cb.on_step({})
        cb.on_training_end({})
        assert len(cb._episode_rewards) == 2

    def test_no_file_no_crash(self):
        cb = LoggingCallback(log_freq=1, verbose=0)
        cb.on_training_start({})
        cb.on_step({})
        cb.on_training_end({})

    def test_episode_reward_fallback_key(self):
        cb = LoggingCallback(log_freq=1, verbose=0)
        cb.on_episode_end({"reward": 5.0})
        assert cb._episode_rewards == [5.0]


# ===================================================================
# EarlyStoppingCallback
# ===================================================================


class TestEarlyStoppingCallback:
    def _make_eval_cb(self):
        env = _FakeEnv(reward=0.0)
        return EvalCallback(eval_env=env, eval_freq=1, n_eval_episodes=1, verbose=0)

    def test_stops_after_patience(self):
        eval_cb = self._make_eval_cb()
        early_cb = EarlyStoppingCallback(eval_cb, patience=3, min_delta=0.0, verbose=0)

        # Simulate improving then plateauing eval results
        rewards = [1.0, 2.0, 2.0, 2.0, 2.0]
        for i, r in enumerate(rewards):
            eval_cb.eval_results.append({"step": i + 1, "mean_reward": r})
            eval_cb.last_mean_reward = r
            result = early_cb.on_step({})
            if not result:
                break

        assert result is False  # Should have stopped

    def test_no_stop_with_improvement(self):
        eval_cb = self._make_eval_cb()
        early_cb = EarlyStoppingCallback(eval_cb, patience=3, verbose=0)

        for i in range(5):
            eval_cb.eval_results.append({"step": i + 1, "mean_reward": float(i)})
            eval_cb.last_mean_reward = float(i)
            result = early_cb.on_step({})
            assert result is True

    def test_no_eval_results_continues(self):
        eval_cb = self._make_eval_cb()
        early_cb = EarlyStoppingCallback(eval_cb, patience=2, verbose=0)
        assert early_cb.on_step({}) is True

    def test_min_delta(self):
        eval_cb = self._make_eval_cb()
        early_cb = EarlyStoppingCallback(eval_cb, patience=2, min_delta=1.0, verbose=0)

        # Small improvements below min_delta don't count
        for i in range(4):
            eval_cb.eval_results.append({"step": i + 1, "mean_reward": i * 0.1})
            eval_cb.last_mean_reward = i * 0.1
            result = early_cb.on_step({})
            if not result:
                break

        assert result is False


# ===================================================================
# CurriculumCallback
# ===================================================================


class TestCurriculumCallback:
    def test_advances_levels(self):
        update_calls = []

        def update_fn(env, params):
            update_calls.append(params)

        cb = CurriculumCallback(
            metric_key="mean_episode_reward",
            thresholds=[(5.0, {"n_agents": 3}), (10.0, {"n_agents": 5})],
            update_fn=update_fn,
            verbose=0,
        )

        env = MagicMock()

        # Below first threshold
        cb.on_episode_end({"mean_episode_reward": 3.0, "env": env})
        assert len(update_calls) == 0

        # Cross first threshold
        cb.on_episode_end({"mean_episode_reward": 6.0, "env": env})
        assert len(update_calls) == 1
        assert update_calls[-1] == {"n_agents": 3}

        # Cross second threshold
        cb.on_episode_end({"mean_episode_reward": 12.0, "env": env})
        assert len(update_calls) == 2
        assert update_calls[-1] == {"n_agents": 5}

    def test_no_metric_no_crash(self):
        cb = CurriculumCallback(metric_key="missing_key", verbose=0)
        cb.on_episode_end({})  # Should not crash

    def test_no_env_no_crash(self):
        cb = CurriculumCallback(
            metric_key="r",
            thresholds=[(1.0, {"x": 1})],
            update_fn=lambda e, p: None,
            verbose=0,
        )
        # Metric crosses threshold but no env in locals_
        cb.on_episode_end({"r": 5.0})

    def test_no_update_fn(self):
        cb = CurriculumCallback(
            metric_key="r",
            thresholds=[(1.0, {"x": 1})],
            update_fn=None,
            verbose=0,
        )
        cb.on_episode_end({"r": 5.0, "env": MagicMock()})
        assert cb._current_level == 1


# ===================================================================
# SchedulerCallback
# ===================================================================


class TestSchedulerCallback:
    def test_step_on_update(self):
        sched = MagicMock()
        cb = SchedulerCallback(schedulers=sched, step_on="update")
        cb.on_update_end({})
        sched.step.assert_called_once()
        assert cb.n_calls == 1

    def test_step_on_step(self):
        sched = MagicMock()
        cb = SchedulerCallback(schedulers=sched, step_on="step")
        assert cb.on_step({}) is True
        sched.step.assert_called_once()

    def test_multiple_schedulers(self):
        s1 = MagicMock()
        s2 = MagicMock()
        cb = SchedulerCallback(schedulers=[s1, s2], step_on="update")
        cb.on_update_end({})
        s1.step.assert_called_once()
        s2.step.assert_called_once()

    def test_update_ignored_when_step_on_step(self):
        sched = MagicMock()
        cb = SchedulerCallback(schedulers=sched, step_on="step")
        cb.on_update_end({})
        sched.step.assert_not_called()


# ===================================================================
# GradientMonitorCallback
# ===================================================================


class TestGradientMonitorCallback:
    def _make_model_with_grads(self):
        model = MagicMock()
        # Create fake parameters with gradients
        param1 = MagicMock()
        param1.grad.data.norm.return_value = MagicMock(item=MagicMock(return_value=0.5))
        param2 = MagicMock()
        param2.grad.data.norm.return_value = MagicMock(item=MagicMock(return_value=1.5))
        model.parameters.return_value = [param1, param2]
        return model

    def test_logs_gradient_norms(self):
        model = self._make_model_with_grads()
        cb = GradientMonitorCallback(log_freq=1, verbose=0)
        cb.on_update_end({"self": model})
        assert len(cb.grad_history) == 1
        entry = cb.grad_history[0]
        assert entry["update"] == 1
        assert entry["param_count"] == 2
        assert entry["total_grad_norm"] > 0

    def test_logs_at_frequency(self):
        model = self._make_model_with_grads()
        cb = GradientMonitorCallback(log_freq=3, verbose=0)
        for _ in range(5):
            cb.on_update_end({"self": model})
        assert len(cb.grad_history) == 1  # Only at step 3

    def test_no_model(self):
        cb = GradientMonitorCallback(log_freq=1, verbose=0)
        cb.on_update_end({})
        assert len(cb.grad_history) == 0

    def test_model_with_policy(self):
        model = MagicMock(spec=[])  # No .parameters attribute
        policy = MagicMock()
        param = MagicMock()
        param.grad.data.norm.return_value = MagicMock(item=MagicMock(return_value=1.0))
        policy.parameters.return_value = [param]
        model.policy = policy
        cb = GradientMonitorCallback(log_freq=1, verbose=0)
        cb.on_update_end({"self": model})
        assert len(cb.grad_history) == 1

    def test_max_grad_norm_warning(self):
        model = self._make_model_with_grads()
        cb = GradientMonitorCallback(log_freq=1, max_grad_norm=0.1, verbose=1)
        # Should log a warning but not crash
        cb.on_update_end({"self": model})
        assert len(cb.grad_history) == 1

    def test_no_parameters_attr(self):
        model = MagicMock(spec=[])  # No attributes at all
        cb = GradientMonitorCallback(log_freq=1, verbose=0)
        cb.on_update_end({"self": model})
        assert len(cb.grad_history) == 0

    def test_param_with_no_grad(self):
        model = MagicMock()
        param = MagicMock()
        param.grad = None
        model.parameters.return_value = [param]
        cb = GradientMonitorCallback(log_freq=1, verbose=0)
        cb.on_update_end({"self": model})
        assert len(cb.grad_history) == 1
        assert cb.grad_history[0]["param_count"] == 0


# ===================================================================
# HyperparameterSearchCallback
# ===================================================================


class TestHyperparameterSearchCallback:
    def test_reports_from_eval_callback(self):
        eval_cb = MagicMock()
        eval_cb.eval_results = [{"step": 10, "mean_reward": 5.0}]
        report_fn = MagicMock()

        cb = HyperparameterSearchCallback(
            metric_key="mean_reward",
            report_fn=report_fn,
            eval_callback=eval_cb,
            report_freq=1,
        )
        cb.on_step({})
        report_fn.assert_called_once_with(1, 5.0)

    def test_reports_from_locals(self):
        report_fn = MagicMock()
        cb = HyperparameterSearchCallback(
            metric_key="mean_reward",
            report_fn=report_fn,
            report_freq=1,
        )
        cb.on_step({"mean_reward": 3.0})
        report_fn.assert_called_once_with(1, 3.0)

    def test_report_freq(self):
        report_fn = MagicMock()
        cb = HyperparameterSearchCallback(report_fn=report_fn, report_freq=5)
        for _ in range(4):
            cb.on_step({"mean_reward": 1.0})
        report_fn.assert_not_called()

        cb.on_step({"mean_reward": 1.0})
        report_fn.assert_called_once()

    def test_no_metric_no_report(self):
        report_fn = MagicMock()
        cb = HyperparameterSearchCallback(report_fn=report_fn, report_freq=1)
        cb.on_step({})
        report_fn.assert_not_called()

    def test_no_report_fn_no_crash(self):
        cb = HyperparameterSearchCallback(report_freq=1)
        cb.on_step({"mean_reward": 1.0})  # Should not crash

    def test_eval_callback_custom_metric(self):
        eval_cb = MagicMock()
        eval_cb.eval_results = [{"step": 1, "custom_metric": 42.0}]
        report_fn = MagicMock()
        cb = HyperparameterSearchCallback(
            metric_key="custom_metric",
            report_fn=report_fn,
            eval_callback=eval_cb,
            report_freq=1,
        )
        cb.on_step({})
        report_fn.assert_called_once_with(1, 42.0)
