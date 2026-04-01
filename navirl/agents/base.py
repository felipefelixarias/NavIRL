"""
NavIRL Base Agent
=================

Abstract base class for all reinforcement learning agents in the NavIRL
pedestrian simulation framework.  Every concrete agent (PPO, SAC, TD3, DQN,
A2C, …) inherits from :class:`BaseAgent` and implements its abstract protocol.

Key responsibilities handled at this level:
* Device management (CPU / CUDA, with automatic fallback).
* Hyperparameter storage and access (dict‑like *and* attribute‑like).
* Checkpoint save / load with versioned metadata.
* Evaluation‑mode toggle that propagates to all owned ``nn.Module`` objects.
* Logging integration via Python's standard :mod:`logging` module and an
  optional structured‑metrics callback.
* Training‑loop hooks that concrete agents can override to inject custom
  behaviour at well‑defined points (epoch start/end, step start/end, etc.).
* Reproducibility helpers (seed management).
"""

from __future__ import annotations

import abc
import json
import logging
import pathlib
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
)

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover – allow import for type‑checking
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter base dataclass
# ---------------------------------------------------------------------------


@dataclass
class HyperParameters:
    """Base dataclass that all agent‑specific config dataclasses extend.

    Provides serialisation helpers and dict‑like access so that configs can be
    passed around generically.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (recursively converting nested dataclasses)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HyperParameters:
        """Construct from a dict, ignoring unknown keys."""
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(f"{key!r} is not a valid hyperparameter for {type(self).__name__}")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, d: dict[str, Any]) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Metric‑logging helpers
# ---------------------------------------------------------------------------

MetricsCallback = Callable[[dict[str, float], int], None]


class MetricsLogger:
    """Lightweight wrapper that buffers scalar metrics and forwards them to
    an optional callback (e.g. TensorBoard writer, W&B, …).
    """

    def __init__(self, callback: MetricsCallback | None = None) -> None:
        self._callback = callback
        self._buffer: dict[str, list[float]] = {}
        self._step: int = 0

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    def record(self, key: str, value: float) -> None:
        self._buffer.setdefault(key, []).append(value)

    def record_dict(self, d: dict[str, float]) -> None:
        for k, v in d.items():
            self.record(k, v)

    def dump(self, step: int | None = None) -> dict[str, float]:
        """Compute means of buffered values, forward to callback, and clear."""
        step = step if step is not None else self._step
        summary: dict[str, float] = {}
        for key, vals in self._buffer.items():
            summary[key] = float(np.mean(vals))
        if self._callback is not None and summary:
            self._callback(summary, step)
        self._buffer.clear()
        return summary

    def set_callback(self, callback: MetricsCallback) -> None:
        self._callback = callback


# ---------------------------------------------------------------------------
# Checkpoint metadata
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 2


@dataclass
class CheckpointMeta:
    """Metadata stored alongside every checkpoint."""

    agent_class: str = ""
    checkpoint_version: int = _CHECKPOINT_VERSION
    total_steps: int = 0
    total_episodes: int = 0
    wall_time: float = 0.0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Running statistics (for observation / value normalisation)
# ---------------------------------------------------------------------------


class RunningMeanStd:
    """Welford's online algorithm for computing running mean / variance.

    This is a pure‑NumPy implementation so that it works even without Torch.
    Agents that need a Torch‑backed version can wrap this.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-8) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = epsilon
        self._epsilon = epsilon

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[np.newaxis, ...]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = float(batch.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + self._epsilon),
            -clip,
            clip,
        )

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * np.sqrt(self.var + self._epsilon) + self.mean

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# ---------------------------------------------------------------------------
# BaseAgent abstract class
# ---------------------------------------------------------------------------


class BaseAgent(abc.ABC):
    """Abstract base class for all NavIRL RL agents.

    Parameters
    ----------
    config : HyperParameters
        Agent‑specific configuration dataclass.
    observation_space : gymnasium.spaces.Space
        Environment observation space.
    action_space : gymnasium.spaces.Space
        Environment action space.
    device : str | torch.device
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:N"``).  Falls back to CPU when
        CUDA is requested but unavailable.
    seed : int | None
        Random seed for reproducibility.
    metrics_callback : MetricsCallback | None
        Optional function ``(metrics_dict, step) -> None`` called after each
        logging dump.
    """

    # Registry of concrete subclasses (populated by __init_subclass__).
    _registry: dict[str, type[BaseAgent]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", set()):
            BaseAgent._registry[cls.__name__] = cls

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: HyperParameters,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        metrics_callback: MetricsCallback | None = None,
    ) -> None:
        self._config = config
        self._observation_space = observation_space
        self._action_space = action_space

        # Device selection -------------------------------------------------
        if _TORCH_AVAILABLE:
            if isinstance(device, str) and "cuda" in device and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available – falling back to CPU.")
                device = "cpu"
            self._device = torch.device(device)
        else:
            self._device = device  # type: ignore[assignment]

        # Reproducibility --------------------------------------------------
        self._seed = seed
        if seed is not None:
            self._set_seed(seed)

        # Bookkeeping ------------------------------------------------------
        self._total_steps: int = 0
        self._total_episodes: int = 0
        self._training: bool = True
        self._start_time: float = time.time()

        # Logging ----------------------------------------------------------
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self._metrics = MetricsLogger(callback=metrics_callback)

        # Modules list – subclasses should append nn.Modules here so that
        # eval()/train() toggles propagate automatically.
        self._modules: list[nn.Module] = []

        # Optimizers list – for checkpoint serialisation.
        self._optimizers: dict[str, Optimizer] = {}

        # Schedulers list.
        self._schedulers: dict[str, _LRScheduler] = {}

        self._logger.info(
            "Initialised %s  |  device=%s  seed=%s",
            type(self).__name__,
            self._device,
            seed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> HyperParameters:
        return self._config

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def is_training(self) -> bool:
        return self._training

    @property
    def metrics(self) -> MetricsLogger:
        return self._metrics

    @property
    def wall_time(self) -> float:
        return time.time() - self._start_time

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select an action given the current observation.

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation.
        deterministic : bool
            If ``True`` the agent should act greedily (no exploration).

        Returns
        -------
        action : np.ndarray
            The chosen action (compatible with ``env.step``).
        info : dict
            Auxiliary information (e.g. log‑prob, value estimate, …).
        """

    @abc.abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Run a single optimisation step on the given data.

        Parameters
        ----------
        batch
            Agent‑specific data structure (rollout buffer slice, replay
            buffer sample, etc.).

        Returns
        -------
        metrics : dict
            Scalar metrics describing the update (losses, KL, entropy, …).
        """

    @abc.abstractmethod
    def save(self, path: str | pathlib.Path) -> None:
        """Persist the agent to disk."""

    @abc.abstractmethod
    def load(self, path: str | pathlib.Path) -> None:
        """Restore the agent from a checkpoint on disk."""

    # ------------------------------------------------------------------
    # Training‑loop hooks (no‑op by default; override in subclasses)
    # ------------------------------------------------------------------

    def on_training_start(self) -> None:  # noqa: B027
        """Called once before the very first environment step."""
        pass

    def on_training_end(self) -> None:  # noqa: B027
        """Called after training is complete."""
        pass

    def on_epoch_start(self, epoch: int) -> None:  # noqa: B027
        """Called at the beginning of each training epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:  # noqa: B027
        """Called at the end of each training epoch."""
        pass

    def on_step_start(self, step: int) -> None:  # noqa: B027
        """Called before each environment step."""
        pass

    def on_step_end(self, step: int, reward: float, done: bool, info: dict[str, Any]) -> None:  # noqa: B027
        """Called after each environment step."""
        pass

    def on_episode_start(self, episode: int) -> None:  # noqa: B027
        """Called at the start of each episode."""
        pass

    def on_episode_end(self, episode: int, total_reward: float, length: int) -> None:  # noqa: B027
        """Called at the end of each episode."""
        pass

    def on_rollout_start(self) -> None:  # noqa: B027
        """Called before a rollout collection phase (on‑policy agents)."""
        pass

    def on_rollout_end(self) -> None:  # noqa: B027
        """Called after a rollout collection phase."""
        pass

    def on_update_start(self) -> None:  # noqa: B027
        """Called before each gradient update."""
        pass

    def on_update_end(self, metrics: dict[str, float]) -> None:  # noqa: B027
        """Called after each gradient update."""
        pass

    # ------------------------------------------------------------------
    # Train / eval toggle
    # ------------------------------------------------------------------

    def train_mode(self) -> BaseAgent:
        """Set the agent (and all owned modules) to training mode."""
        self._training = True
        for mod in self._modules:
            mod.train()
        return self

    def eval_mode(self) -> BaseAgent:
        """Set the agent (and all owned modules) to evaluation mode."""
        self._training = False
        for mod in self._modules:
            mod.eval()
        return self

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _build_checkpoint_meta(self, **extra: Any) -> CheckpointMeta:
        return CheckpointMeta(
            agent_class=type(self).__name__,
            checkpoint_version=_CHECKPOINT_VERSION,
            total_steps=self._total_steps,
            total_episodes=self._total_episodes,
            wall_time=self.wall_time,
            hyperparameters=self._config.to_dict(),
            extra=extra,
        )

    def _save_checkpoint(
        self,
        path: str | pathlib.Path,
        state_dicts: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        """Write a full checkpoint (model weights + optimizers + meta).

        Parameters
        ----------
        path : str or Path
            Directory *or* file path.  If a directory is given the checkpoint
            is written as ``checkpoint_<step>.pt`` inside it.
        state_dicts : dict
            Mapping of component names → ``state_dict()`` results.
        extra : dict, optional
            Additional metadata to embed in the checkpoint.

        Returns
        -------
        pathlib.Path
            The actual file that was written.
        """
        path = pathlib.Path(path)
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            filepath = path / f"checkpoint_{self._total_steps}.pt"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            filepath = path

        meta = self._build_checkpoint_meta(**(extra or {}))

        # Gather optimizer states
        optim_states = {name: opt.state_dict() for name, opt in self._optimizers.items()}
        sched_states = {name: sch.state_dict() for name, sch in self._schedulers.items()}

        payload: dict[str, Any] = {
            "meta": asdict(meta),
            "model": state_dicts,
            "optimizers": optim_states,
            "schedulers": sched_states,
        }

        if _TORCH_AVAILABLE:
            torch.save(payload, filepath)
        else:
            raise RuntimeError("Cannot save checkpoint without PyTorch installed.")

        # Also write a human‑readable JSON sidecar with metadata.
        meta_path = filepath.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2, default=str)

        self._logger.info("Checkpoint saved → %s  (%d steps)", filepath, self._total_steps)
        return filepath

    def _load_checkpoint(
        self,
        path: str | pathlib.Path,
    ) -> dict[str, Any]:
        """Load a checkpoint from disk and return the raw payload dict.

        Subclasses should call this and then apply the returned state dicts
        to their networks / optimizers.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if _TORCH_AVAILABLE:
            payload = torch.load(path, map_location=self._device, weights_only=False)
        else:
            raise RuntimeError("Cannot load checkpoint without PyTorch installed.")

        meta = payload.get("meta", {})
        ckpt_version = meta.get("checkpoint_version", 1)
        if ckpt_version > _CHECKPOINT_VERSION:
            logger.warning(
                "Checkpoint version %d is newer than agent version %d – loading may fail.",
                ckpt_version,
                _CHECKPOINT_VERSION,
            )

        # Restore bookkeeping
        self._total_steps = meta.get("total_steps", 0)
        self._total_episodes = meta.get("total_episodes", 0)

        # Restore optimizer states if present
        optim_states = payload.get("optimizers", {})
        for name, state in optim_states.items():
            if name in self._optimizers:
                self._optimizers[name].load_state_dict(state)

        sched_states = payload.get("schedulers", {})
        for name, state in sched_states.items():
            if name in self._schedulers:
                self._schedulers[name].load_state_dict(state)

        self._logger.info(
            "Checkpoint loaded ← %s  (step=%d, episodes=%d)",
            path,
            self._total_steps,
            self._total_episodes,
        )
        return payload

    # ------------------------------------------------------------------
    # Hyperparameter access shortcuts
    # ------------------------------------------------------------------

    def get_hyperparameter(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set_hyperparameter(self, key: str, value: Any) -> None:
        self._config[key] = value
        self._logger.debug("Hyperparameter %s set to %s", key, value)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        if _TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_tensor(
        self, x: np.ndarray | torch.Tensor, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Convert a numpy array (or keep a tensor) and move to agent device."""
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            t = x
        if dtype is not None:
            t = t.to(dtype)
        return t.to(self._device)

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """Polyak averaging: θ_target ← τ·θ_source + (1−τ)·θ_target."""
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters(), strict=False):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    def _hard_update(self, target: nn.Module, source: nn.Module) -> None:
        target.load_state_dict(source.state_dict())

    def _clip_grad_norm(self, parameters: Any, max_norm: float) -> float:
        """Clip gradients and return the total norm *before* clipping."""
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))

    def _count_parameters(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def _log_module_summary(self, name: str, module: nn.Module) -> None:
        n_params = self._count_parameters(module)
        self._logger.info("  %s: %s trainable parameters", name, f"{n_params:,}")

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        agent_name: str,
        config: HyperParameters,
        observation_space: Any,
        action_space: Any,
        **kwargs: Any,
    ) -> BaseAgent:
        """Instantiate a registered agent by name.

        >>> agent = BaseAgent.make("PPOAgent", config, obs_space, act_space)
        """
        if agent_name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown agent {agent_name!r}. Registered agents: {available}")
        return cls._registry[agent_name](config, observation_space, action_space, **kwargs)

    @classmethod
    def registered_agents(cls) -> list[str]:
        return sorted(cls._registry.keys())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"device={self._device}, "
            f"steps={self._total_steps}, "
            f"episodes={self._total_episodes}, "
            f"training={self._training})"
        )
