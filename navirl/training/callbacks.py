"""Composable training callbacks for NavIRL training loops."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Exports: Callback, CallbackList, EvalCallback, CheckpointCallback, LoggingCallback,
#          EarlyStoppingCallback, CurriculumCallback, WandbCallback, TensorBoardCallback,
#          ProgressBarCallback, VideoRecordCallback, GradientMonitorCallback,
#          SchedulerCallback, HyperparameterSearchCallback

__all__ = [
    "Callback",
    "CallbackList",
    "EvalCallback",
    "CheckpointCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    "CurriculumCallback",
    "WandbCallback",
    "TensorBoardCallback",
    "ProgressBarCallback",
    "VideoRecordCallback",
    "GradientMonitorCallback",
    "SchedulerCallback",
    "HyperparameterSearchCallback",
]

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks.

    Subclasses can override any of the hook methods to inject custom behaviour
    at the corresponding point of the training loop.  All hooks receive a
    *locals_* dictionary that the training loop populates with its current
    state (e.g. step count, episode reward, model reference, etc.).
    """

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Called once at the very beginning of training."""

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        """Called once at the very end of training."""

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        """Called after every environment step.

        Returns
        -------
        bool
            ``True`` to continue training, ``False`` to request an early stop.
        """
        return True

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        """Called at the end of each episode."""

    def on_rollout_start(self, locals_: Dict[str, Any]) -> None:
        """Called before a rollout collection begins."""

    def on_rollout_end(self, locals_: Dict[str, Any]) -> None:
        """Called after a rollout collection ends."""

    def on_update_start(self, locals_: Dict[str, Any]) -> None:
        """Called before a parameter update."""

    def on_update_end(self, locals_: Dict[str, Any]) -> None:
        """Called after a parameter update."""


class CallbackList(Callback):
    """Chains multiple callbacks so they are all invoked in order.

    Parameters
    ----------
    callbacks:
        Sequence of :class:`Callback` instances to invoke.
    """

    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self.callbacks: List[Callback] = list(callbacks)

    # ------------------------------------------------------------------
    # Hook delegation
    # ------------------------------------------------------------------

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_training_start(locals_)

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_training_end(locals_)

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        continue_training = True
        for cb in self.callbacks:
            if not cb.on_step(locals_):
                continue_training = False
        return continue_training

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(locals_)

    def on_rollout_start(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_rollout_start(locals_)

    def on_rollout_end(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_rollout_end(locals_)

    def on_update_start(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_update_start(locals_)

    def on_update_end(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_update_end(locals_)


class EvalCallback(Callback):
    """Periodically evaluate the policy and save the best model.

    Parameters
    ----------
    eval_env:
        Environment (or callable returning one) used for evaluation.
    eval_freq:
        Evaluate every *eval_freq* steps.
    n_eval_episodes:
        Number of episodes to run during each evaluation.
    best_model_save_path:
        Directory where the best model checkpoint is saved.
    deterministic:
        Whether to use deterministic actions during evaluation.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.verbose = verbose

        self.best_mean_reward: float = float("-inf")
        self.last_mean_reward: float = float("-inf")
        self.n_calls: int = 0
        self.eval_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.eval_freq != 0:
            return True

        model = locals_.get("self")
        env = self.eval_env() if callable(self.eval_env) else self.eval_env

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []

        for _ in range(self.n_eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            ep_reward = 0.0
            ep_length = 0
            while not done:
                if model is not None and hasattr(model, "predict"):
                    action, _ = model.predict(obs, deterministic=self.deterministic)
                else:
                    action = env.action_space.sample()
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, _info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, _info = result
                ep_reward += float(reward)
                ep_length += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        mean_length = sum(episode_lengths) / len(episode_lengths)
        self.last_mean_reward = mean_reward

        self.eval_results.append(
            {
                "step": self.n_calls,
                "mean_reward": mean_reward,
                "mean_length": mean_length,
                "rewards": episode_rewards,
            }
        )

        if self.verbose >= 1:
            logger.info(
                "Eval step=%d  mean_reward=%.4f  mean_length=%.1f",
                self.n_calls,
                mean_reward,
                mean_length,
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path is not None and model is not None:
                save_dir = Path(self.best_model_save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, "save"):
                    model.save(str(save_dir / "best_model"))
                if self.verbose >= 1:
                    logger.info("New best model saved (reward=%.4f)", mean_reward)

        return True


class CheckpointCallback(Callback):
    """Save the model at regular intervals.

    Parameters
    ----------
    save_freq:
        Save every *save_freq* steps.
    save_path:
        Directory for checkpoint files.
    name_prefix:
        Filename prefix for the checkpoint.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        save_freq: int = 50_000,
        save_path: str = "./checkpoints",
        name_prefix: str = "model",
        verbose: int = 1,
    ) -> None:
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.verbose = verbose
        self.n_calls: int = 0

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.save_freq != 0:
            return True

        model = locals_.get("self")
        if model is not None and hasattr(model, "save"):
            path = self.save_path / f"{self.name_prefix}_{self.n_calls}_steps"
            model.save(str(path))
            if self.verbose >= 1:
                logger.info("Checkpoint saved to %s", path)
        return True


class LoggingCallback(Callback):
    """Log training metrics to the console and/or a JSON-lines file.

    Parameters
    ----------
    log_freq:
        Log every *log_freq* steps.
    log_file:
        Optional path to a JSON-lines file for persistent logging.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        log_freq: int = 1_000,
        log_file: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        self.log_freq = log_freq
        self.log_file = log_file
        self.verbose = verbose
        self.n_calls: int = 0
        self._file_handle = None
        self._episode_rewards: List[float] = []
        self._start_time: float = 0.0

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        self._start_time = time.time()
        if self.log_file is not None:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_file, "a")

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        reward = locals_.get("episode_reward", locals_.get("reward"))
        if reward is not None:
            self._episode_rewards.append(float(reward))

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.log_freq != 0:
            return True

        elapsed = time.time() - self._start_time
        fps = self.n_calls / elapsed if elapsed > 0 else 0.0

        metrics: Dict[str, Any] = {
            "step": self.n_calls,
            "elapsed_sec": round(elapsed, 2),
            "fps": round(fps, 1),
        }

        if self._episode_rewards:
            recent = self._episode_rewards[-100:]
            metrics["mean_episode_reward"] = round(
                sum(recent) / len(recent), 4
            )
            metrics["episodes"] = len(self._episode_rewards)

        # Gather any extra metrics the training loop may expose.
        for key in ("loss", "policy_loss", "value_loss", "entropy"):
            if key in locals_:
                metrics[key] = round(float(locals_[key]), 6)

        if self.verbose >= 1:
            logger.info("Training  %s", metrics)

        if self._file_handle is not None:
            self._file_handle.write(json.dumps(metrics) + "\n")
            self._file_handle.flush()

        return True


class EarlyStoppingCallback(Callback):
    """Stop training when no improvement is observed for *patience* evaluations.

    This callback is designed to work alongside :class:`EvalCallback`.

    Parameters
    ----------
    eval_callback:
        An :class:`EvalCallback` whose ``last_mean_reward`` is monitored.
    patience:
        Number of evaluations with no improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        eval_callback: EvalCallback,
        patience: int = 5,
        min_delta: float = 0.0,
        verbose: int = 1,
    ) -> None:
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self._best_reward: float = float("-inf")
        self._no_improvement_count: int = 0
        self._last_eval_step: int = 0

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        # Only act when the eval callback has produced a new result.
        if not self.eval_callback.eval_results:
            return True
        latest_step = self.eval_callback.eval_results[-1]["step"]
        if latest_step == self._last_eval_step:
            return True
        self._last_eval_step = latest_step

        current_reward = self.eval_callback.last_mean_reward
        if current_reward > self._best_reward + self.min_delta:
            self._best_reward = current_reward
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            if self.verbose >= 1:
                logger.info(
                    "Early stopping triggered after %d evaluations without improvement.",
                    self.patience,
                )
            return False

        return True


class CurriculumCallback(Callback):
    """Update environment curriculum parameters based on training performance.

    Parameters
    ----------
    metric_key:
        Key in *locals_* used to read the current performance metric.
    thresholds:
        Ordered list of *(metric_value, params_dict)* pairs.  When the metric
        exceeds a threshold the corresponding parameters are applied.
    update_fn:
        Callable ``(env, params_dict) -> None`` that applies the curriculum
        parameters to the environment.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        metric_key: str = "mean_episode_reward",
        thresholds: Optional[List[tuple]] = None,
        update_fn: Optional[Any] = None,
        verbose: int = 1,
    ) -> None:
        self.metric_key = metric_key
        self.thresholds: List[tuple] = sorted(thresholds or [], key=lambda t: t[0])
        self.update_fn = update_fn
        self.verbose = verbose
        self._current_level: int = 0

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        metric_value = locals_.get(self.metric_key)
        if metric_value is None:
            return

        new_level = self._current_level
        for idx, (threshold, _params) in enumerate(self.thresholds):
            if metric_value >= threshold:
                new_level = idx + 1

        if new_level > self._current_level:
            self._current_level = new_level
            _, params = self.thresholds[new_level - 1]
            env = locals_.get("env")
            if env is not None and self.update_fn is not None:
                self.update_fn(env, params)
            if self.verbose >= 1:
                logger.info(
                    "Curriculum level %d activated (threshold met): %s",
                    self._current_level,
                    params,
                )


class WandbCallback(Callback):
    """Log metrics to Weights & Biases.

    The ``wandb`` package is imported lazily so the rest of the library does
    not depend on it.

    Parameters
    ----------
    project:
        W&B project name.
    entity:
        W&B entity (user or team).
    config:
        Dictionary of hyperparameters to log.
    log_freq:
        Log every *log_freq* steps.
    """

    def __init__(
        self,
        project: str = "navirl",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_freq: int = 1_000,
    ) -> None:
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.log_freq = log_freq
        self.n_calls: int = 0
        self._wandb: Any = None
        self._run: Any = None

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        import wandb  # lazy import

        self._wandb = wandb
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            reinit=True,
        )

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        if self._run is not None:
            self._run.finish()

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.log_freq != 0:
            return True

        metrics: Dict[str, Any] = {"step": self.n_calls}
        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "episode_reward",
            "episode_length",
        ):
            if key in locals_:
                metrics[key] = float(locals_[key])

        if self._wandb is not None:
            self._wandb.log(metrics, step=self.n_calls)

        return True


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard.

    ``torch.utils.tensorboard`` is imported lazily.

    Parameters
    ----------
    log_dir:
        Directory for TensorBoard event files.
    log_freq:
        Log every *log_freq* steps.
    """

    def __init__(
        self,
        log_dir: str = "./runs",
        log_freq: int = 1_000,
    ) -> None:
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.n_calls: int = 0
        self._writer: Any = None

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        from torch.utils.tensorboard import SummaryWriter  # lazy import

        self._writer = SummaryWriter(log_dir=self.log_dir)

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.close()

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.log_freq != 0:
            return True

        if self._writer is None:
            return True

        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "episode_reward",
            "episode_length",
        ):
            if key in locals_:
                self._writer.add_scalar(key, float(locals_[key]), self.n_calls)

        return True


class ProgressBarCallback(Callback):
    """Display a ``tqdm`` progress bar during training.

    ``tqdm`` is imported lazily.

    Parameters
    ----------
    total_steps:
        Total number of training steps (for progress percentage).
    """

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self._pbar: Any = None
        self._episode_rewards: List[float] = []

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        from tqdm import tqdm  # lazy import

        self._pbar = tqdm(total=self.total_steps, desc="Training", unit="step")

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        if self._pbar is not None:
            self._pbar.close()

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        reward = locals_.get("episode_reward", locals_.get("reward"))
        if reward is not None:
            self._episode_rewards.append(float(reward))

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        if self._pbar is not None:
            self._pbar.update(1)
            if self._episode_rewards:
                recent = self._episode_rewards[-100:]
                mean_r = sum(recent) / len(recent)
                self._pbar.set_postfix(mean_reward=f"{mean_r:.2f}")
        return True


class VideoRecordCallback(Callback):
    """Record videos of evaluation episodes.

    Parameters
    ----------
    eval_env:
        Environment (or callable returning one) used for recording.
    record_freq:
        Record every *record_freq* steps.
    video_dir:
        Directory where video files are stored.
    n_episodes:
        Number of episodes to record each time.
    fps:
        Frames per second for the output video.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        eval_env: Any,
        record_freq: int = 50_000,
        video_dir: str = "./videos",
        n_episodes: int = 1,
        fps: int = 30,
        verbose: int = 1,
    ) -> None:
        self.eval_env = eval_env
        self.record_freq = record_freq
        self.video_dir = Path(video_dir)
        self.n_episodes = n_episodes
        self.fps = fps
        self.verbose = verbose
        self.n_calls: int = 0

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.record_freq != 0:
            return True

        model = locals_.get("self")
        env = self.eval_env() if callable(self.eval_env) else self.eval_env

        for ep_idx in range(self.n_episodes):
            frames: List[Any] = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            while not done:
                if hasattr(env, "render"):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                if model is not None and hasattr(model, "predict"):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                result = env.step(action)
                if len(result) == 5:
                    obs, _reward, terminated, truncated, _info = result
                    done = terminated or truncated
                else:
                    obs, _reward, done, _info = result

            if frames:
                self._save_video(frames, ep_idx)

        return True

    def _save_video(self, frames: List[Any], episode_idx: int) -> None:
        """Write *frames* to a video file using imageio if available."""
        try:
            import imageio  # type: ignore

            filename = self.video_dir / f"step_{self.n_calls}_ep{episode_idx}.mp4"
            imageio.mimsave(str(filename), frames, fps=self.fps)
            if self.verbose >= 1:
                logger.info("Video saved to %s", filename)
        except ImportError:
            logger.warning("imageio not installed; skipping video recording.")


class GradientMonitorCallback(Callback):
    """Track and log gradient norms after each update.

    Parameters
    ----------
    log_freq:
        Log gradient statistics every *log_freq* updates.
    max_grad_norm:
        If set, warn when any parameter gradient exceeds this norm.
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        log_freq: int = 100,
        max_grad_norm: Optional[float] = None,
        verbose: int = 1,
    ) -> None:
        self.log_freq = log_freq
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self._update_count: int = 0
        self.grad_history: List[Dict[str, float]] = []

    def on_update_end(self, locals_: Dict[str, Any]) -> None:
        self._update_count += 1
        if self._update_count % self.log_freq != 0:
            return

        model = locals_.get("self")
        if model is None:
            return

        # Try to extract parameters from common model structures.
        parameters = None
        if hasattr(model, "parameters"):
            parameters = model.parameters()
        elif hasattr(model, "policy") and hasattr(model.policy, "parameters"):
            parameters = model.policy.parameters()

        if parameters is None:
            return

        import torch  # needed for gradient inspection

        total_norm = 0.0
        max_norm = 0.0
        param_count = 0
        for p in parameters:
            if p.grad is not None:
                pnorm = p.grad.data.norm(2).item()
                total_norm += pnorm ** 2
                max_norm = max(max_norm, pnorm)
                param_count += 1

        total_norm = total_norm ** 0.5

        entry = {
            "update": self._update_count,
            "total_grad_norm": round(total_norm, 6),
            "max_param_grad_norm": round(max_norm, 6),
            "param_count": param_count,
        }
        self.grad_history.append(entry)

        if self.verbose >= 1:
            logger.info("Gradient norms: %s", entry)

        if self.max_grad_norm is not None and total_norm > self.max_grad_norm:
            logger.warning(
                "Gradient norm %.4f exceeds max_grad_norm %.4f",
                total_norm,
                self.max_grad_norm,
            )


class SchedulerCallback(Callback):
    """Step one or more learning-rate schedulers at each update.

    Parameters
    ----------
    schedulers:
        A single scheduler or list of schedulers that expose a ``step()``
        method (e.g. ``torch.optim.lr_scheduler`` objects or NavIRL
        :class:`~navirl.training.schedulers.Schedule` wrappers).
    step_on:
        When to step the scheduler: ``"update"`` (after each gradient update)
        or ``"step"`` (after each environment step).
    """

    def __init__(
        self,
        schedulers: Union[Any, List[Any]],
        step_on: str = "update",
    ) -> None:
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        self.schedulers = list(schedulers)
        self.step_on = step_on
        self.n_calls: int = 0

    def _step_all(self, locals_: Dict[str, Any]) -> None:
        self.n_calls += 1
        for sched in self.schedulers:
            if hasattr(sched, "step"):
                sched.step()

    def on_update_end(self, locals_: Dict[str, Any]) -> None:
        if self.step_on == "update":
            self._step_all(locals_)

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        if self.step_on == "step":
            self._step_all(locals_)
        return True


class HyperparameterSearchCallback(Callback):
    """Report metrics for an external hyperparameter search framework.

    Works with Optuna, Ray Tune, or any framework that accepts a reporting
    callable.

    Parameters
    ----------
    metric_key:
        Key in *locals_* (or from an attached :class:`EvalCallback`) to
        report.
    report_fn:
        Callable ``(step, metric_value) -> None`` that sends the metric to the
        HP search framework.
    eval_callback:
        Optional :class:`EvalCallback` to read metrics from.
    report_freq:
        Report every *report_freq* steps.
    """

    def __init__(
        self,
        metric_key: str = "mean_reward",
        report_fn: Optional[Any] = None,
        eval_callback: Optional[EvalCallback] = None,
        report_freq: int = 10_000,
    ) -> None:
        self.metric_key = metric_key
        self.report_fn = report_fn
        self.eval_callback = eval_callback
        self.report_freq = report_freq
        self.n_calls: int = 0

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        self.n_calls += 1
        if self.n_calls % self.report_freq != 0:
            return True

        # Try to get the metric from the eval callback first.
        metric_value: Optional[float] = None
        if self.eval_callback is not None and self.eval_callback.eval_results:
            latest = self.eval_callback.eval_results[-1]
            metric_value = latest.get(self.metric_key, latest.get("mean_reward"))

        # Fall back to locals_.
        if metric_value is None:
            metric_value = locals_.get(self.metric_key)

        if metric_value is not None and self.report_fn is not None:
            self.report_fn(self.n_calls, float(metric_value))

        return True
