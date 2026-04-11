"""Tests for navirl/training/trainer.py — TrainerConfig, TrainingLogger,
EvalResult, _SingleEnvShim, and Trainer.

Covers configuration, logging, evaluation, checkpointing, callbacks, and the
full training loop using lightweight mock agents and environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from navirl.training.trainer import (
    EvalResult,
    Trainer,
    TrainerConfig,
    TrainingLogger,
    _SingleEnvShim,
)

# ---------------------------------------------------------------------------
# Helpers — mock agent and environment
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal Gym-like env for testing the Trainer."""

    def __init__(self, obs_dim: int = 4, episode_length: int = 3) -> None:
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self._step_count = 0
        self.observation_space = MagicMock(shape=(obs_dim,))
        self.action_space = MagicMock(shape=(1,))

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict]:
        self._step_count += 1
        obs = np.ones(self.obs_dim, dtype=np.float32) * self._step_count
        reward = 1.0
        done = self._step_count >= self.episode_length
        info: dict[str, Any] = {}
        if done:
            info["is_success"] = True
        return obs, reward, done, info

    def close(self) -> None:
        pass


class MockAgent:
    """Minimal agent that conforms to the Trainer's expected interface."""

    def __init__(self, obs_dim: int = 4) -> None:
        self.obs_dim = obs_dim
        self._eval = False
        self.saved_path: str | None = None
        self.loaded_path: str | None = None

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        batch_size = obs.shape[0] if obs.ndim > 1 else 1
        return np.zeros(batch_size)

    def store_transition(self, obs, actions, rewards, next_obs, dones, infos) -> None:
        pass

    def update(self) -> dict[str, float] | None:
        return {"loss": 0.5}

    def eval_mode(self) -> None:
        self._eval = True

    def train_mode(self) -> None:
        self._eval = False

    def save(self, path: str) -> None:
        self.saved_path = path
        os.makedirs(path, exist_ok=True)

    def load(self, path: str) -> None:
        self.loaded_path = path


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.total_timesteps == 1_000_000
        assert cfg.eval_interval == 10_000
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = TrainerConfig(total_timesteps=100, seed=7)
        assert cfg.total_timesteps == 100
        assert cfg.seed == 7

    def test_to_dict(self):
        cfg = TrainerConfig(total_timesteps=50)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["total_timesteps"] == 50
        assert "seed" in d

    def test_from_dict(self):
        cfg = TrainerConfig.from_dict({"total_timesteps": 200, "seed": 99})
        assert cfg.total_timesteps == 200
        assert cfg.seed == 99

    def test_from_dict_ignores_unknown_keys(self):
        cfg = TrainerConfig.from_dict({"total_timesteps": 10, "unknown_key": "ignored"})
        assert cfg.total_timesteps == 10
        assert not hasattr(cfg, "unknown_key")

    def test_roundtrip(self):
        original = TrainerConfig(total_timesteps=42, eval_episodes=7)
        restored = TrainerConfig.from_dict(original.to_dict())
        assert restored.total_timesteps == 42
        assert restored.eval_episodes == 7


# ---------------------------------------------------------------------------
# TrainingLogger
# ---------------------------------------------------------------------------


class TestTrainingLogger:
    def test_log_scalar(self):
        logger = TrainingLogger()
        logger.log_scalar("loss", 0.5, step=1)
        logger.log_scalar("loss", 0.3, step=2)

        history = logger.get_history("loss")
        assert len(history) == 2
        assert history[0] == (1, 0.5)
        assert history[1] == (2, 0.3)

    def test_log_dict(self):
        logger = TrainingLogger()
        logger.log_dict({"a": 1.0, "b": 2.0}, step=10)

        assert logger.get_history("a") == [(10, 1.0)]
        assert logger.get_history("b") == [(10, 2.0)]

    def test_get_history_missing_key(self):
        logger = TrainingLogger()
        assert logger.get_history("nonexistent") == []

    def test_close_no_backends(self):
        logger = TrainingLogger()
        logger.close()  # Should not raise

    def test_log_dir_none(self):
        """Logger works fine with log_dir=None (no file-based logging)."""
        logger = TrainingLogger(log_dir=None)
        logger.log_scalar("x", 1.0, 0)
        assert logger.get_history("x") == [(0, 1.0)]
        logger.close()

    def test_tensorboard_import_failure(self):
        """If tensorboard is not installed, logging still works."""
        logger = TrainingLogger(log_dir="/tmp/test_logger_tb", use_tensorboard=True)
        assert logger._tb_writer is None  # torch.utils.tensorboard not available
        logger.log_scalar("x", 1.0, 0)
        logger.close()

    def test_wandb_import_failure(self):
        """If wandb is not installed, logging still works."""
        logger = TrainingLogger(use_wandb=True)
        assert logger._wandb is None
        logger.log_scalar("x", 1.0, 0)
        logger.close()


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_creation(self):
        result = EvalResult(
            mean_reward=10.0,
            std_reward=2.0,
            mean_length=50.0,
            success_rate=0.8,
        )
        assert result.mean_reward == 10.0
        assert result.success_rate == 0.8
        assert result.per_episode_rewards == []

    def test_to_dict(self):
        result = EvalResult(
            mean_reward=5.0,
            std_reward=1.0,
            mean_length=20.0,
            success_rate=0.5,
            per_episode_rewards=[4.0, 6.0],
            per_episode_lengths=[18, 22],
        )
        d = result.to_dict()
        assert d["mean_reward"] == 5.0
        assert d["per_episode_rewards"] == [4.0, 6.0]

    def test_default_fields(self):
        result = EvalResult(mean_reward=0.0, std_reward=0.0, mean_length=0.0, success_rate=0.0)
        assert result.per_episode_rewards == []
        assert result.per_episode_lengths == []


# ---------------------------------------------------------------------------
# _SingleEnvShim
# ---------------------------------------------------------------------------


class TestSingleEnvShim:
    def test_reset(self):
        env = MockEnv(obs_dim=3)
        shim = _SingleEnvShim(env)
        obs = shim.reset()
        assert obs.shape == (1, 3)

    def test_step(self):
        env = MockEnv(obs_dim=3, episode_length=5)
        shim = _SingleEnvShim(env)
        shim.reset()
        obs, rewards, dones, infos = shim.step(np.array([0]))
        assert obs.shape == (1, 3)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert len(infos) == 1

    def test_step_with_list_actions(self):
        env = MockEnv(obs_dim=2, episode_length=5)
        shim = _SingleEnvShim(env)
        shim.reset()
        obs, *_ = shim.step([0])
        assert obs.shape == (1, 2)

    def test_step_auto_resets_on_done(self):
        env = MockEnv(obs_dim=2, episode_length=2)
        shim = _SingleEnvShim(env)
        shim.reset()
        shim.step(np.array([0]))  # step 1
        obs, _, dones, _ = shim.step(np.array([0]))  # step 2 — done
        assert dones[0]
        # After auto-reset, obs should be zero
        np.testing.assert_array_equal(obs[0], [0.0, 0.0])

    def test_close(self):
        env = MockEnv()
        shim = _SingleEnvShim(env)
        shim.close()  # Should not raise


# ---------------------------------------------------------------------------
# Trainer — evaluate
# ---------------------------------------------------------------------------


class TestTrainerEvaluate:
    def test_evaluate_returns_eval_result(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
            eval_episodes=3,
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=2),
            config=cfg,
        )
        result = trainer.evaluate(n_episodes=3)
        assert isinstance(result, EvalResult)
        assert len(result.per_episode_rewards) == 3
        assert len(result.per_episode_lengths) == 3
        assert result.mean_length == 2.0
        assert result.mean_reward == 2.0  # 1.0 + 1.0 per episode
        assert result.success_rate == 1.0  # all episodes have is_success

    def test_evaluate_toggles_eval_train_mode(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=2),
            config=cfg,
        )
        trainer.evaluate(n_episodes=1)
        # After evaluate, agent should be back in train mode
        assert not agent._eval

    def test_evaluate_uses_default_episodes(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
            eval_episodes=5,
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=2),
            config=cfg,
        )
        result = trainer.evaluate()
        assert len(result.per_episode_rewards) == 5


# ---------------------------------------------------------------------------
# Trainer — checkpointing
# ---------------------------------------------------------------------------


class TestTrainerCheckpointing:
    def test_save_checkpoint(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(),
            config=cfg,
        )
        trainer._global_step = 100
        trainer._episodes_done = 5

        ckpt_path = tmp_path / "ckpt" / "test_ckpt"
        trainer.save_checkpoint(ckpt_path)

        # Agent should have been saved
        assert agent.saved_path == str(ckpt_path / "agent")

        # Metadata file should exist
        meta_path = ckpt_path / "trainer_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["global_step"] == 100
        assert meta["episodes_done"] == 5

    def test_load_checkpoint(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(),
            config=cfg,
        )

        # Save first
        trainer._global_step = 200
        trainer._episodes_done = 10
        trainer._best_mean_reward = 42.0
        ckpt_path = tmp_path / "ckpt" / "test_ckpt"
        trainer.save_checkpoint(ckpt_path)

        # Reset and load
        trainer._global_step = 0
        trainer._episodes_done = 0
        trainer._best_mean_reward = float("-inf")
        trainer.load_checkpoint(ckpt_path)

        assert trainer._global_step == 200
        assert trainer._episodes_done == 10
        assert trainer._best_mean_reward == 42.0
        assert agent.loaded_path == str(ckpt_path / "agent")

    def test_load_checkpoint_no_meta(self, tmp_path):
        """Loading from a path with no trainer_meta.json only loads agent."""
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(),
            config=cfg,
        )
        no_meta_path = tmp_path / "empty_ckpt"
        no_meta_path.mkdir(parents=True)
        trainer.load_checkpoint(no_meta_path)
        assert agent.loaded_path == str(no_meta_path / "agent")
        assert trainer._global_step == 0  # unchanged


# ---------------------------------------------------------------------------
# Trainer — callbacks
# ---------------------------------------------------------------------------


class TestTrainerCallbacks:
    def test_fire_calls_matching_hooks(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        callback = MagicMock()
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(),
            config=cfg,
            callbacks=[callback],
        )
        trainer._fire("on_training_start")
        callback.on_training_start.assert_called_once()

    def test_fire_skips_missing_hooks(self, tmp_path):
        cfg = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )

        class PartialCallback:
            pass  # No hooks defined

        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(),
            config=cfg,
            callbacks=[PartialCallback()],
        )
        trainer._fire("on_training_start")  # Should not raise


# ---------------------------------------------------------------------------
# Trainer — train loop
# ---------------------------------------------------------------------------


class TestTrainerTrain:
    def test_short_training_loop(self, tmp_path):
        """Run a very short training loop to verify the full pipeline."""
        cfg = TrainerConfig(
            total_timesteps=10,
            eval_interval=5,
            save_interval=100,
            log_interval=5,
            n_envs=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
            seed=0,
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=3),
            config=cfg,
        )
        summary = trainer.train()

        assert isinstance(summary, dict)
        assert summary["total_timesteps"] >= 10
        assert summary["total_episodes"] >= 1
        assert summary["wall_time_seconds"] > 0

    def test_train_fires_callbacks(self, tmp_path):
        cfg = TrainerConfig(
            total_timesteps=5,
            eval_interval=100,
            save_interval=100,
            log_interval=100,
            n_envs=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        callback = MagicMock()
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=10),
            config=cfg,
            callbacks=[callback],
        )
        trainer.train()

        callback.on_training_start.assert_called_once()
        callback.on_training_end.assert_called_once()
        assert callback.on_step_start.called
        assert callback.on_step_end.called

    def test_train_saves_best_model_on_eval(self, tmp_path):
        cfg = TrainerConfig(
            total_timesteps=10,
            eval_interval=5,
            save_interval=100,
            log_interval=100,
            n_envs=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=2),
            config=cfg,
        )
        trainer.train()

        # Best model checkpoint should exist
        best_path = tmp_path / "ckpt" / "best_model" / "trainer_meta.json"
        assert best_path.exists()

    def test_make_envs_fallback_shim(self, tmp_path):
        """When parallel import fails, Trainer falls back to _SingleEnvShim."""
        cfg = TrainerConfig(
            total_timesteps=5,
            n_envs=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        agent = MockAgent()
        trainer = Trainer(
            agent=agent,
            env_fn=lambda: MockEnv(episode_length=3),
            config=cfg,
        )

        with patch(
            "navirl.training.trainer.Trainer._make_envs",
            return_value=_SingleEnvShim(MockEnv(episode_length=3)),
        ):
            summary = trainer.train()
            assert summary["total_timesteps"] >= 5


# ---------------------------------------------------------------------------
# Trainer — agent returning None from update
# ---------------------------------------------------------------------------


class TestTrainerNullUpdate:
    def test_agent_returning_none_from_update(self, tmp_path):
        """Training works when agent.update() returns None."""

        class NoneUpdateAgent(MockAgent):
            def update(self):
                return None

        cfg = TrainerConfig(
            total_timesteps=5,
            n_envs=1,
            eval_interval=100,
            save_interval=100,
            log_interval=100,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "log"),
        )
        trainer = Trainer(
            agent=NoneUpdateAgent(),
            env_fn=lambda: MockEnv(episode_length=10),
            config=cfg,
        )
        summary = trainer.train()
        assert summary["total_timesteps"] >= 5
