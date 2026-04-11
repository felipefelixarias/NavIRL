"""Tests for uncovered paths in navirl.training.callbacks.

Focuses on WandbCallback, TensorBoardCallback, ProgressBarCallback,
VideoRecordCallback, and edge cases in existing callbacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    ProgressBarCallback,
    SchedulerCallback,
    TensorBoardCallback,
    VideoRecordCallback,
    WandbCallback,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSpace:
    def __init__(self, shape=(2,)):
        self.shape = shape

    def sample(self):
        import numpy as np

        return np.zeros(self.shape)


class _FakeEnv:
    """Env that terminates after n_steps steps."""

    def __init__(self, n_steps=3):
        self.action_space = _FakeSpace()
        self._n_steps = n_steps
        self._step = 0

    def reset(self):
        self._step = 0
        return [0.0, 0.0, 0.0, 0.0]

    def step(self, action):
        self._step += 1
        obs = [float(self._step)] * 4
        done = self._step >= self._n_steps
        return obs, 1.0, done, {}

    def render(self):
        import numpy as np

        return np.zeros((64, 64, 3), dtype=np.uint8)


class _FakeEnvGymV26(_FakeEnv):
    """Gym v26 style env returning 5-tuple."""

    def step(self, action):
        self._step += 1
        obs = [float(self._step)] * 4
        terminated = self._step >= self._n_steps
        truncated = False
        return obs, 1.0, terminated, truncated, {}


# ---------------------------------------------------------------------------
# WandbCallback
# ---------------------------------------------------------------------------


class TestWandbCallback:
    """Cover WandbCallback (lines 567-610)."""

    def test_training_lifecycle(self):
        """WandbCallback should init and finish a run."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        cb = WandbCallback(project="test", entity="user", config={"lr": 0.01})
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb.on_training_start({})
            assert cb._wandb is mock_wandb
            assert cb._run is mock_run

        cb.on_training_end({})
        mock_run.finish.assert_called_once()

    def test_on_step_logs_metrics(self):
        """on_step should log metrics at the right frequency."""
        mock_wandb = MagicMock()
        cb = WandbCallback(log_freq=2)
        cb._wandb = mock_wandb
        cb._run = MagicMock()

        locals_ = {"loss": 0.5, "episode_reward": 10.0}

        # Step 1: no log
        assert cb.on_step(locals_)
        mock_wandb.log.assert_not_called()

        # Step 2: should log
        assert cb.on_step(locals_)
        mock_wandb.log.assert_called_once()
        logged = mock_wandb.log.call_args[0][0]
        assert logged["loss"] == 0.5
        assert logged["episode_reward"] == 10.0

    def test_on_step_no_wandb(self):
        """on_step without wandb initialized should not crash."""
        cb = WandbCallback(log_freq=1)
        cb._wandb = None
        assert cb.on_step({"loss": 0.5})

    def test_training_end_no_run(self):
        """on_training_end with no run should not crash."""
        cb = WandbCallback()
        cb._run = None
        cb.on_training_end({})  # should not raise


# ---------------------------------------------------------------------------
# TensorBoardCallback
# ---------------------------------------------------------------------------


class TestTensorBoardCallback:
    """Cover TensorBoardCallback (lines 631-664)."""

    def test_lifecycle(self):
        """TensorBoardCallback should create and close a writer."""
        mock_writer = MagicMock()
        mock_sw = MagicMock(return_value=mock_writer)

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "torch.utils": MagicMock(),
                "torch.utils.tensorboard": MagicMock(SummaryWriter=mock_sw),
            },
        ):
            cb = TensorBoardCallback(log_dir="/tmp/tb_test", log_freq=1)
            cb.on_training_start({})
            assert cb._writer is mock_writer

        cb.on_training_end({})
        mock_writer.close.assert_called_once()

    def test_on_step_logs_scalars(self):
        """on_step should add scalar for known metric keys."""
        mock_writer = MagicMock()
        cb = TensorBoardCallback(log_freq=1)
        cb._writer = mock_writer

        locals_ = {"loss": 0.1, "entropy": 0.5}
        assert cb.on_step(locals_)
        assert mock_writer.add_scalar.call_count == 2

    def test_on_step_no_writer(self):
        """on_step without writer should not crash."""
        cb = TensorBoardCallback(log_freq=1)
        cb._writer = None
        assert cb.on_step({"loss": 0.1})

    def test_training_end_no_writer(self):
        """on_training_end with no writer should not crash."""
        cb = TensorBoardCallback()
        cb._writer = None
        cb.on_training_end({})


# ---------------------------------------------------------------------------
# ProgressBarCallback
# ---------------------------------------------------------------------------


class TestProgressBarCallback:
    """Cover ProgressBarCallback (lines 679-704)."""

    def test_lifecycle(self):
        """ProgressBarCallback should create and close a pbar."""
        mock_tqdm = MagicMock()
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        with patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=mock_tqdm)}):
            cb = ProgressBarCallback(total_steps=100)
            cb.on_training_start({})
            assert cb._pbar is mock_pbar

        cb.on_training_end({})
        mock_pbar.close.assert_called_once()

    def test_on_step_updates_pbar(self):
        """on_step should update progress bar."""
        mock_pbar = MagicMock()
        cb = ProgressBarCallback(total_steps=100)
        cb._pbar = mock_pbar

        assert cb.on_step({})
        mock_pbar.update.assert_called_once_with(1)

    def test_on_step_shows_reward(self):
        """on_step should show mean reward in postfix."""
        mock_pbar = MagicMock()
        cb = ProgressBarCallback(total_steps=100)
        cb._pbar = mock_pbar
        cb._episode_rewards = [1.0, 2.0, 3.0]

        cb.on_step({})
        mock_pbar.set_postfix.assert_called_once()

    def test_on_episode_end_tracks_rewards(self):
        cb = ProgressBarCallback(total_steps=100)
        cb.on_episode_end({"episode_reward": 5.0})
        assert cb._episode_rewards == [5.0]

    def test_on_step_no_pbar(self):
        """No pbar should not crash."""
        cb = ProgressBarCallback(total_steps=100)
        cb._pbar = None
        assert cb.on_step({})

    def test_training_end_no_pbar(self):
        cb = ProgressBarCallback(total_steps=100)
        cb._pbar = None
        cb.on_training_end({})


# ---------------------------------------------------------------------------
# VideoRecordCallback
# ---------------------------------------------------------------------------


class TestVideoRecordCallback:
    """Cover VideoRecordCallback (lines 735-791)."""

    def test_records_video(self, tmp_path):
        """VideoRecordCallback should capture frames and save video."""
        mock_imageio = MagicMock()

        cb = VideoRecordCallback(
            eval_env=_FakeEnv(n_steps=2),
            record_freq=1,
            video_dir=str(tmp_path / "videos"),
            n_episodes=1,
        )
        cb.on_training_start({})

        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            assert cb.on_step({})

        mock_imageio.mimsave.assert_called_once()

    def test_records_with_model(self, tmp_path):
        """Should use model.predict when available."""
        model = MagicMock()
        model.predict.return_value = ([0.0, 0.0], None)

        cb = VideoRecordCallback(
            eval_env=_FakeEnv(n_steps=2),
            record_freq=1,
            video_dir=str(tmp_path / "videos"),
        )
        cb.on_training_start({})

        mock_imageio = MagicMock()
        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            cb.on_step({"self": model})

        model.predict.assert_called()

    def test_gym_v26_env(self, tmp_path):
        """Should handle 5-tuple step returns."""
        cb = VideoRecordCallback(
            eval_env=_FakeEnvGymV26(n_steps=2),
            record_freq=1,
            video_dir=str(tmp_path / "videos"),
        )
        cb.on_training_start({})

        mock_imageio = MagicMock()
        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            assert cb.on_step({})

    def test_imageio_import_error(self, tmp_path):
        """Missing imageio should log warning, not crash."""
        import numpy as np

        cb = VideoRecordCallback(
            eval_env=_FakeEnv(n_steps=2),
            record_freq=1,
            video_dir=str(tmp_path / "videos"),
        )
        cb.on_training_start({})

        with patch.dict("sys.modules", {"imageio": None}):
            # Should not raise
            assert cb.on_step({})

    def test_callable_env(self, tmp_path):
        """Should handle callable env factory."""
        cb = VideoRecordCallback(
            eval_env=lambda: _FakeEnv(n_steps=2),
            record_freq=1,
            video_dir=str(tmp_path / "videos"),
        )
        cb.on_training_start({})

        mock_imageio = MagicMock()
        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            assert cb.on_step({})

    def test_no_frames_no_save(self, tmp_path):
        """Env that returns None from render should not try to save."""
        env = _FakeEnv(n_steps=1)
        # Patch render to return None
        env.render = lambda: None

        cb = VideoRecordCallback(
            eval_env=env,
            record_freq=1,
            video_dir=str(tmp_path / "videos"),
        )
        cb.on_training_start({})

        mock_imageio = MagicMock()
        with patch.dict("sys.modules", {"imageio": mock_imageio}):
            cb.on_step({})
        mock_imageio.mimsave.assert_not_called()


# ---------------------------------------------------------------------------
# CheckpointCallback edge case
# ---------------------------------------------------------------------------


class TestCheckpointCallbackEdge:
    """Cover CheckpointCallback verbose logging (line 347)."""

    def test_saves_and_logs(self, tmp_path):
        model = MagicMock()
        cb = CheckpointCallback(save_freq=1, save_path=str(tmp_path), verbose=1)
        cb.on_training_start({})
        cb.on_step({"self": model})
        model.save.assert_called_once()


# ---------------------------------------------------------------------------
# LoggingCallback edge case
# ---------------------------------------------------------------------------


class TestLoggingCallbackEdge:
    """Cover LoggingCallback file write path (line 419)."""

    def test_writes_to_file(self, tmp_path):
        log_file = str(tmp_path / "train.jsonl")
        cb = LoggingCallback(log_freq=1, log_file=log_file, verbose=0)
        cb.on_training_start({})

        cb.on_episode_end({"episode_reward": 5.0})
        cb.on_step({"loss": 0.1})
        cb.on_training_end({})

        lines = Path(log_file).read_text().strip().split("\n")
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert "step" in data


# ---------------------------------------------------------------------------
# EarlyStoppingCallback edge case
# ---------------------------------------------------------------------------


class TestEarlyStoppingCallbackEdge:
    """Cover EarlyStoppingCallback verbose logging (line 479)."""

    def test_verbose_logging_on_stop(self):
        eval_cb = EvalCallback(eval_env=_FakeEnv(), eval_freq=1)
        eval_cb.eval_results = [{"step": 1, "mean_reward": 1.0}]
        eval_cb.last_mean_reward = 1.0

        cb = EarlyStoppingCallback(eval_cb, patience=1, verbose=1)
        # First eval — sets baseline
        cb.on_step({})

        # Second eval — no improvement
        eval_cb.eval_results.append({"step": 2, "mean_reward": 0.5})
        eval_cb.last_mean_reward = 0.5
        result = cb.on_step({})
        assert result is False  # should stop


# ---------------------------------------------------------------------------
# CurriculumCallback edge case
# ---------------------------------------------------------------------------


class TestCurriculumCallbackEdge:
    """Cover CurriculumCallback level advancement with env (line 535)."""

    def test_calls_update_fn(self):
        update_fn = MagicMock()
        env = MagicMock()
        cb = CurriculumCallback(
            metric_key="score",
            thresholds=[(5.0, {"level": 1}), (10.0, {"level": 2})],
            update_fn=update_fn,
            verbose=1,
        )
        cb.on_episode_end({"score": 6.0, "env": env})
        update_fn.assert_called_once_with(env, {"level": 1})
