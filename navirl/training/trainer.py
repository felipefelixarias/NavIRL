"""
NavIRL Trainer
==============

Main training loop orchestration for reinforcement learning agents in the
NavIRL pedestrian simulation framework.  Provides a structured train/eval
cycle with checkpointing, metric logging, and an extensible callback system.

Exports
-------
* :class:`TrainerConfig` -- training hyper-parameters and paths.
* :class:`TrainingLogger` -- structured scalar/dict logging with optional
  TensorBoard and Weights & Biases integration.
* :class:`EvalResult` -- evaluation statistics container.
* :class:`Trainer` -- main training loop driver.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Configuration for the :class:`Trainer` training loop.

    Parameters
    ----------
    total_timesteps:
        Total number of environment timesteps to train for.
    eval_interval:
        Evaluate every *eval_interval* timesteps.
    eval_episodes:
        Number of episodes to run per evaluation round.
    save_interval:
        Save a checkpoint every *save_interval* timesteps.
    log_interval:
        Log training metrics every *log_interval* timesteps.
    checkpoint_dir:
        Directory to write checkpoint files into.
    log_dir:
        Directory for TensorBoard / W&B log artefacts.
    n_envs:
        Number of parallel environments for rollout collection.
    seed:
        Global random seed for reproducibility.
    """

    total_timesteps: int = 1_000_000
    eval_interval: int = 10_000
    eval_episodes: int = 10
    save_interval: int = 50_000
    log_interval: int = 1_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    n_envs: int = 4
    seed: int = 42

    # -- helpers -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainerConfig:
        """Construct from a dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Lightweight logging facade for training metrics.

    Records scalar values keyed by ``(key, step)`` pairs.  Optionally
    forwards writes to TensorBoard and/or Weights & Biases if the relevant
    libraries are installed *and* a *log_dir* is provided.

    Parameters
    ----------
    log_dir:
        Directory for log artefacts.  ``None`` disables file-based logging.
    use_tensorboard:
        Attempt to create a TensorBoard ``SummaryWriter``.
    use_wandb:
        Attempt to use an active Weights & Biases run for logging.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
    ) -> None:
        self.log_dir = log_dir
        self._history: dict[str, list[tuple]] = {}

        # -- TensorBoard -----------------------------------------------------
        self._tb_writer = None
        if use_tensorboard and log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_path = os.path.join(log_dir, "tensorboard")
                os.makedirs(tb_path, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tb_path)
                logger.info("TensorBoard logging enabled at %s", tb_path)
            except ImportError:
                logger.warning("tensorboard not installed -- TensorBoard logging disabled")

        # -- Weights & Biases ------------------------------------------------
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore[import-untyped]

                if wandb.run is not None:
                    self._wandb = wandb
                    logger.info("W&B logging enabled (run=%s)", wandb.run.name)
                else:
                    logger.warning("wandb imported but no active run -- W&B logging disabled")
            except ImportError:
                logger.warning("wandb not installed -- W&B logging disabled")

    # -- public API ----------------------------------------------------------

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a single scalar *value* under *key* at *step*."""
        self._history.setdefault(key, []).append((step, value))

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(key, value, global_step=step)

        if self._wandb is not None:
            self._wandb.log({key: value}, step=step)

        logger.debug("[step=%d] %s = %.6g", step, key, value)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        """Log every entry in *metrics* at *step*."""
        for key, value in metrics.items():
            self.log_scalar(key, value, step)

    def get_history(self, key: str) -> list[tuple]:
        """Return the list of ``(step, value)`` pairs for *key*."""
        return list(self._history.get(key, []))

    def close(self) -> None:
        """Flush and close backend writers."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Container for evaluation statistics.

    Attributes
    ----------
    mean_reward:
        Mean total reward across episodes.
    std_reward:
        Standard deviation of total rewards.
    mean_length:
        Mean episode length.
    success_rate:
        Fraction of episodes deemed successful (``info["is_success"]``).
    per_episode_rewards:
        Per-episode total rewards.
    per_episode_lengths:
        Per-episode lengths.
    """

    mean_reward: float
    std_reward: float
    mean_length: float
    success_rate: float
    per_episode_rewards: list[float] = field(default_factory=list)
    per_episode_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Main training loop driver.

    Orchestrates environment rollouts, agent updates, periodic evaluation,
    checkpointing, metric logging, and user-supplied callbacks.

    Parameters
    ----------
    agent:
        RL agent exposing ``select_action``, ``update``, ``save`` / ``load``,
        and ``eval_mode`` / ``train_mode`` methods (i.e. the
        :class:`~navirl.agents.base.BaseAgent` protocol).
    env_fn:
        Zero-argument callable that returns a *single* Gym-style environment.
        For vectorised training the trainer creates *n_envs* copies.
    config:
        :class:`TrainerConfig` with loop hyper-parameters and paths.
    callbacks:
        Optional sequence of callback objects called at well-defined points
        during training (see :mod:`navirl.training.callbacks`).
    """

    def __init__(
        self,
        agent: Any,
        env_fn: Callable[[], Any],
        config: TrainerConfig | None = None,
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        self.agent = agent
        self.env_fn = env_fn
        self.config = config or TrainerConfig()
        self.callbacks = list(callbacks) if callbacks else []

        self._logger = TrainingLogger(log_dir=self.config.log_dir)
        self._global_step: int = 0
        self._episodes_done: int = 0
        self._best_mean_reward: float = float("-inf")

        # Ensure output directories exist.
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

    # -- callbacks -----------------------------------------------------------

    def _fire(self, hook: str, **kwargs: Any) -> None:
        """Invoke *hook* on every registered callback, if it exists."""
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                fn(self, **kwargs)

    # -- train ---------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the main training loop.

        Returns
        -------
        dict
            Summary metrics collected during training.
        """
        cfg = self.config
        logger.info(
            "Starting training for %d timesteps (seed=%d, n_envs=%d)",
            cfg.total_timesteps,
            cfg.seed,
            cfg.n_envs,
        )

        # Create the (possibly vectorised) environment(s).
        envs = self._make_envs()

        self._fire("on_training_start")

        obs = envs.reset()
        episode_rewards: list[float] = [0.0] * cfg.n_envs
        episode_lengths: list[int] = [0] * cfg.n_envs
        all_train_metrics: dict[str, list[float]] = {}
        start_time = time.monotonic()

        while self._global_step < cfg.total_timesteps:
            # 1. Collect experience.
            self._fire("on_step_start", step=self._global_step)

            actions = self.agent.select_action(obs)
            next_obs, rewards, dones, infos = envs.step(actions)

            # Store transition(s) in the agent's buffer.
            self.agent.store_transition(obs, actions, rewards, next_obs, dones, infos)

            obs = next_obs
            self._global_step += cfg.n_envs

            # Track per-environment episode stats.
            for i in range(cfg.n_envs):
                episode_rewards[i] += float(rewards[i])
                episode_lengths[i] += 1
                if dones[i]:
                    self._episodes_done += 1
                    self._logger.log_scalar(
                        "train/episode_reward", episode_rewards[i], self._global_step
                    )
                    self._logger.log_scalar(
                        "train/episode_length", episode_lengths[i], self._global_step
                    )
                    self._fire(
                        "on_episode_end",
                        step=self._global_step,
                        reward=episode_rewards[i],
                        length=episode_lengths[i],
                        info=infos[i] if isinstance(infos, list) else infos,
                    )
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

            # 2. Update agent.
            update_info = self.agent.update()
            if update_info is not None:
                for k, v in update_info.items():
                    all_train_metrics.setdefault(k, []).append(v)

            # 3. Log metrics.
            if self._global_step % cfg.log_interval < cfg.n_envs:
                elapsed = time.monotonic() - start_time
                fps = self._global_step / max(elapsed, 1e-9)
                self._logger.log_dict(
                    {
                        "train/timesteps": self._global_step,
                        "train/episodes": self._episodes_done,
                        "train/fps": fps,
                    },
                    step=self._global_step,
                )
                if update_info:
                    self._logger.log_dict(
                        {f"train/{k}": v for k, v in update_info.items()},
                        step=self._global_step,
                    )
                self._fire("on_log", step=self._global_step)

            # 4. Evaluate periodically.
            if self._global_step % cfg.eval_interval < cfg.n_envs:
                eval_result = self.evaluate(cfg.eval_episodes)
                self._logger.log_dict(
                    {
                        "eval/mean_reward": eval_result.mean_reward,
                        "eval/std_reward": eval_result.std_reward,
                        "eval/mean_length": eval_result.mean_length,
                        "eval/success_rate": eval_result.success_rate,
                    },
                    step=self._global_step,
                )
                if eval_result.mean_reward > self._best_mean_reward:
                    self._best_mean_reward = eval_result.mean_reward
                    self.save_checkpoint(os.path.join(cfg.checkpoint_dir, "best_model"))
                self._fire("on_eval", step=self._global_step, eval_result=eval_result)

            # 5. Save checkpoints.
            if self._global_step % cfg.save_interval < cfg.n_envs:
                self.save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, f"checkpoint_{self._global_step}")
                )

            self._fire("on_step_end", step=self._global_step)

        # Cleanup.
        self._fire("on_training_end")
        envs.close()
        self._logger.close()

        elapsed = time.monotonic() - start_time
        summary = {
            "total_timesteps": self._global_step,
            "total_episodes": self._episodes_done,
            "wall_time_seconds": elapsed,
            "best_mean_reward": self._best_mean_reward,
        }
        logger.info("Training complete: %s", summary)
        return summary

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, n_episodes: int | None = None) -> EvalResult:
        """Run *n_episodes* evaluation episodes and return an :class:`EvalResult`.

        The agent is switched to eval mode for the duration of evaluation and
        restored to train mode afterwards.
        """
        n_episodes = n_episodes or self.config.eval_episodes
        env = self.env_fn()

        self.agent.eval_mode()

        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        successes: list[bool] = []

        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            length = 0

            while not done:
                action = self.agent.select_action(
                    np.asarray(obs) if not isinstance(obs, np.ndarray) else obs
                )
                # Handle both single and array action outputs.
                if isinstance(action, np.ndarray) and action.ndim > 0:
                    action = action[0]
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                length += 1

            ep_rewards.append(total_reward)
            ep_lengths.append(length)
            successes.append(bool(info.get("is_success", False)))

        env.close()
        self.agent.train_mode()

        result = EvalResult(
            mean_reward=float(np.mean(ep_rewards)),
            std_reward=float(np.std(ep_rewards)),
            mean_length=float(np.mean(ep_lengths)),
            success_rate=float(np.mean(successes)),
            per_episode_rewards=ep_rewards,
            per_episode_lengths=ep_lengths,
        )
        logger.info(
            "Eval (%d eps): mean_reward=%.2f (+/- %.2f), success_rate=%.2f",
            n_episodes,
            result.mean_reward,
            result.std_reward,
            result.success_rate,
        )
        return result

    # -- checkpointing -------------------------------------------------------

    def save_checkpoint(self, path: str | pathlib.Path) -> None:
        """Save agent state and trainer metadata to *path*."""
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Agent weights / optimiser state.
        self.agent.save(str(path / "agent"))

        # Trainer metadata.
        meta = {
            "global_step": self._global_step,
            "episodes_done": self._episodes_done,
            "best_mean_reward": self._best_mean_reward,
            "config": self.config.to_dict(),
        }
        with open(path / "trainer_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Checkpoint saved to %s (step=%d)", path, self._global_step)

    def load_checkpoint(self, path: str | pathlib.Path) -> None:
        """Restore agent state and trainer metadata from *path*."""
        path = pathlib.Path(path)

        self.agent.load(str(path / "agent"))

        meta_path = path / "trainer_meta.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            self._global_step = meta.get("global_step", 0)
            self._episodes_done = meta.get("episodes_done", 0)
            self._best_mean_reward = meta.get("best_mean_reward", float("-inf"))
            logger.info("Checkpoint loaded from %s (step=%d)", path, self._global_step)
        else:
            logger.warning("No trainer_meta.json at %s -- only agent state loaded", path)

    # -- internal helpers ----------------------------------------------------

    def _make_envs(self) -> Any:
        """Create a vectorised environment wrapper.

        Falls back to a lightweight shim if the full parallel module is not
        available.
        """
        n = self.config.n_envs
        try:
            from navirl.training.parallel import DummyVecEnv, SubprocVecEnv

            if n > 1:
                return SubprocVecEnv([self.env_fn for _ in range(n)])
            return DummyVecEnv([self.env_fn])
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Vectorised env wrappers unavailable: {e} -- using a single-env shim")
            return _SingleEnvShim(self.env_fn())


# ---------------------------------------------------------------------------
# Minimal single-env shim
# ---------------------------------------------------------------------------


class _SingleEnvShim:
    """Wraps a single Gym env to expose the vectorised-env interface."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        return np.expand_dims(np.asarray(obs), axis=0)

    def step(self, actions: Any) -> tuple:
        action = actions[0] if isinstance(actions, (np.ndarray, list)) else actions
        obs, reward, done, info = self._env.step(action)
        obs = np.expand_dims(np.asarray(obs), axis=0)
        if done:
            obs[0] = np.asarray(self._env.reset())
        return obs, np.array([reward]), np.array([done]), [info]

    def close(self) -> None:
        self._env.close()


__all__ = [
    "EvalResult",
    "Trainer",
    "TrainerConfig",
    "TrainingLogger",
]
