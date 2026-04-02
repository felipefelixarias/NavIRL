"""
Twin Delayed DDPG (TD3) Agent
==============================

Implementation of Twin Delayed Deep Deterministic Policy Gradient
(Fujimoto et al., 2018) for continuous action spaces in the NavIRL
pedestrian navigation framework.

TD3 addresses overestimation bias in actor-critic methods through three
key mechanisms:

* **Clipped double Q-learning** -- twin critics with minimum operator.
* **Delayed policy updates** -- actor updates less frequently than critics.
* **Target policy smoothing** -- noise added to target actions.

Additional features in this implementation:

* **Configurable exploration noise schedules** -- linear decay, cosine
  decay, or constant noise for action-space exploration.
* **Observation normalization** via running mean/variance statistics.
* **Per-component noise scaling** -- different noise levels per action
  dimension for heterogeneous action spaces.
* **Gradient clipping** with configurable max norm.
* **Warmup period** with random actions before training begins.

References
----------
Fujimoto, S., Hoof, H., & Meger, D. (2018).
Addressing Function Approximation Error in Actor-Critic Methods.
*ICML 2018*.
"""

from __future__ import annotations

import copy
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from navirl.agents.base import BaseAgent, HyperParameters, RunningMeanStd
from navirl.agents.networks import MLP, DeterministicPolicyHead, QValueHead

# ---------------------------------------------------------------------------
# Exploration noise schedules
# ---------------------------------------------------------------------------


class NoiseSchedule:
    """Base class for exploration noise schedules.

    Subclasses implement the ``value`` method which returns the current
    noise standard deviation given the training step.
    """

    def value(self, step: int) -> float:
        """Return the noise standard deviation at the given step.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        float
            Noise standard deviation.
        """
        raise NotImplementedError


class ConstantNoise(NoiseSchedule):
    """Constant exploration noise.

    Parameters
    ----------
    sigma : float
        Fixed noise standard deviation.
    """

    def __init__(self, sigma: float = 0.1) -> None:
        self.sigma = sigma

    def value(self, step: int) -> float:
        """Return constant noise level regardless of step.

        Parameters
        ----------
        step : int
            Current training step (unused).

        Returns
        -------
        float
            The constant sigma value.
        """
        return self.sigma

    def __repr__(self) -> str:
        return f"ConstantNoise(sigma={self.sigma})"


class LinearDecayNoise(NoiseSchedule):
    """Linearly decaying exploration noise.

    Parameters
    ----------
    start : float
        Initial noise standard deviation.
    end : float
        Final noise standard deviation.
    decay_steps : int
        Number of steps over which the decay occurs.
    """

    def __init__(self, start: float = 0.3, end: float = 0.05, decay_steps: int = 500_000) -> None:
        self.start = start
        self.end = end
        self.decay_steps = max(decay_steps, 1)

    def value(self, step: int) -> float:
        """Return linearly interpolated noise level.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        float
            Interpolated noise standard deviation.
        """
        fraction = min(1.0, step / self.decay_steps)
        return self.start + fraction * (self.end - self.start)

    def __repr__(self) -> str:
        return (
            f"LinearDecayNoise(start={self.start}, end={self.end}, decay_steps={self.decay_steps})"
        )


class CosineDecayNoise(NoiseSchedule):
    """Cosine-annealing exploration noise schedule.

    Parameters
    ----------
    start : float
        Initial noise standard deviation.
    end : float
        Minimum noise standard deviation.
    decay_steps : int
        Number of steps for one cosine half-period.
    """

    def __init__(self, start: float = 0.3, end: float = 0.05, decay_steps: int = 500_000) -> None:
        self.start = start
        self.end = end
        self.decay_steps = max(decay_steps, 1)

    def value(self, step: int) -> float:
        """Return cosine-annealed noise level.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        float
            Cosine-annealed noise standard deviation.
        """
        progress = min(step / self.decay_steps, 1.0)
        return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * progress))

    def __repr__(self) -> str:
        return (
            f"CosineDecayNoise(start={self.start}, end={self.end}, decay_steps={self.decay_steps})"
        )


def make_noise_schedule(
    schedule_type: str,
    start: float = 0.3,
    end: float = 0.05,
    decay_steps: int = 500_000,
) -> NoiseSchedule:
    """Factory function for exploration noise schedules.

    Parameters
    ----------
    schedule_type : str
        One of ``"constant"``, ``"linear"``, or ``"cosine"``.
    start : float
        Initial noise level.
    end : float
        Final noise level (ignored for constant).
    decay_steps : int
        Decay horizon (ignored for constant).

    Returns
    -------
    NoiseSchedule
        The constructed schedule.

    Raises
    ------
    ValueError
        If *schedule_type* is not recognised.
    """
    if schedule_type == "constant":
        return ConstantNoise(sigma=start)
    elif schedule_type == "linear":
        return LinearDecayNoise(start=start, end=end, decay_steps=decay_steps)
    elif schedule_type == "cosine":
        return CosineDecayNoise(start=start, end=end, decay_steps=decay_steps)
    else:
        raise ValueError(
            f"Unknown noise schedule {schedule_type!r}; "
            f"expected one of 'constant', 'linear', 'cosine'."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TD3Config(HyperParameters):
    """Hyperparameters for Twin Delayed DDPG.

    Attributes
    ----------
    lr_actor : float
        Learning rate for the actor (policy) network.
    lr_critic : float
        Learning rate for the twin Q-networks.
    gamma : float
        Discount factor.
    tau : float
        Soft target update coefficient (Polyak averaging).
    policy_noise : float
        Standard deviation of Gaussian noise added to target actions
        during critic updates (target policy smoothing).
    noise_clip : float
        Clipping range for the target policy smoothing noise.
    policy_delay : int
        Number of critic updates per actor update.
    hidden_dims : tuple of int
        Hidden layer widths for actor and critic networks.
    activation : str
        Activation function name for hidden layers.
    batch_size : int
        Mini-batch size for each gradient update.
    exploration_noise : float
        Initial standard deviation of exploration noise.
    exploration_noise_end : float
        Final exploration noise level (for decaying schedules).
    noise_schedule : str
        Exploration noise schedule: ``"constant"``, ``"linear"``, or
        ``"cosine"``.
    noise_decay_steps : int
        Number of steps over which noise decays.
    max_grad_norm : float or None
        If set, gradients are clipped to this maximum norm.
    normalize_observations : bool
        Whether to normalize observations via running statistics.
    observation_clip : float
        Clipping range for normalized observations.
    warmup_steps : int
        Number of initial steps with random actions (no learning).
    action_noise_per_dim : list of float or None
        Per-dimension noise scaling factors.  If ``None``, uniform noise
        is used across all action dimensions.
    """

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    batch_size: int = 256
    exploration_noise: float = 0.1
    exploration_noise_end: float = 0.05
    noise_schedule: str = "constant"
    noise_decay_steps: int = 500_000
    max_grad_norm: float | None = None
    normalize_observations: bool = False
    observation_clip: float = 10.0
    warmup_steps: int = 10_000
    action_noise_per_dim: list[float] | None = None


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG agent for continuous control.

    Implements TD3 with configurable exploration noise schedules,
    delayed policy updates, target policy smoothing, and optional
    observation normalization.

    Parameters
    ----------
    config : TD3Config
        Agent hyperparameters.
    observation_space
        Environment observation space (must expose ``.shape``).
    action_space
        Environment action space (must expose ``.shape``, ``.low``,
        ``.high``).
    device : str or torch.device
        Compute device.
    seed : int or None
        Random seed for reproducibility.
    metrics_callback : callable or None
        Optional ``(metrics_dict, step) -> None`` callback.
    """

    def __init__(
        self,
        config: TD3Config,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)

        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        # Action bounds for clipping
        self._action_low = torch.tensor(
            action_space.low,
            dtype=torch.float32,
            device=self._device,
        )
        self._action_high = torch.tensor(
            action_space.high,
            dtype=torch.float32,
            device=self._device,
        )

        # ---- Observation normalization ----
        self._obs_rms: RunningMeanStd | None = None
        if config.normalize_observations:
            self._obs_rms = RunningMeanStd(shape=observation_space.shape)

        # ---- Exploration noise schedule ----
        self._noise_schedule = make_noise_schedule(
            schedule_type=config.noise_schedule,
            start=config.exploration_noise,
            end=config.exploration_noise_end,
            decay_steps=config.noise_decay_steps,
        )

        # Per-dimension noise scaling
        if config.action_noise_per_dim is not None:
            self._noise_scale = torch.tensor(
                config.action_noise_per_dim,
                dtype=torch.float32,
                device=self._device,
            )
        else:
            self._noise_scale = torch.ones(
                action_dim,
                dtype=torch.float32,
                device=self._device,
            )

        # ---- Actor ----
        self.actor_trunk = MLP(
            input_dim=obs_dim,
            output_dim=0,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(self._device)
        trunk_dim = self.actor_trunk.feature_dim

        self.actor_head = DeterministicPolicyHead(
            input_dim=trunk_dim,
            action_dim=action_dim,
            use_tanh=True,
        ).to(self._device)

        # ---- Target actor ----
        self.actor_trunk_target = copy.deepcopy(self.actor_trunk).to(self._device)
        self.actor_head_target = copy.deepcopy(self.actor_head).to(self._device)
        for p in self.actor_trunk_target.parameters():
            p.requires_grad = False
        for p in self.actor_head_target.parameters():
            p.requires_grad = False

        # ---- Critics (two independent Q-networks) ----
        self.q1 = QValueHead(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        self.q2 = QValueHead(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        # ---- Target critics ----
        self.q1_target = copy.deepcopy(self.q1).to(self._device)
        self.q2_target = copy.deepcopy(self.q2).to(self._device)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # ---- Optimizers ----
        actor_params = list(self.actor_trunk.parameters()) + list(self.actor_head.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.lr_critic,
        )

        self._optimizers["actor"] = self.actor_optimizer
        self._optimizers["critic"] = self.critic_optimizer

        # ---- Register modules for train/eval toggling ----
        self._modules.extend(
            [
                self.actor_trunk,
                self.actor_head,
                self.actor_trunk_target,
                self.actor_head_target,
                self.q1,
                self.q2,
                self.q1_target,
                self.q2_target,
            ]
        )

        # ---- Update counter for delayed policy updates ----
        self._update_count: int = 0

        # ---- Random state for warmup ----
        self._rng = np.random.RandomState(seed)

        self._log_module_summary("actor_trunk", self.actor_trunk)
        self._log_module_summary("actor_head", self.actor_head)
        self._log_module_summary("q1", self.q1)
        self._log_module_summary("q2", self.q2)
        self._logger.info("Noise schedule: %s", self._noise_schedule)

    # ------------------------------------------------------------------
    # Observation normalization
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics if enabled.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation.

        Returns
        -------
        np.ndarray
            Possibly normalized observation.
        """
        cfg: TD3Config = self._config  # type: ignore[assignment]
        if self._obs_rms is not None:
            if self._training:
                self._obs_rms.update(obs)
            return self._obs_rms.normalize(obs, clip=cfg.observation_clip).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select an action given an observation.

        During the warmup period, returns uniformly random actions.
        After warmup, uses the deterministic policy with additive
        Gaussian exploration noise (unless *deterministic* is True).

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation.
        deterministic : bool
            If ``True``, return the deterministic policy output without
            exploration noise.

        Returns
        -------
        action : np.ndarray
            The chosen action, clipped to the action space bounds.
        info : dict
            Contains ``"noise_sigma"`` indicating the current noise level.
        """
        cfg: TD3Config = self._config  # type: ignore[assignment]

        # Warmup: random actions
        if self._training and self._total_steps < cfg.warmup_steps:
            action = self._rng.uniform(
                low=self._action_low.cpu().numpy(),
                high=self._action_high.cpu().numpy(),
            ).astype(np.float32)
            return action, {"noise_sigma": float("nan")}

        observation = self._normalize_obs(observation)
        current_sigma = self._noise_schedule.value(self._total_steps)

        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            features = self.actor_trunk(obs_t)
            action = self.actor_head(features)

            if not deterministic:
                noise = torch.randn_like(action) * current_sigma * self._noise_scale
                action = action + noise

            action = action.clamp(self._action_low, self._action_high)

        return self._to_numpy(action.squeeze(0)), {"noise_sigma": current_sigma}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> dict[str, float]:
        """Perform a single TD3 gradient step on a batch of transitions.

        The critic is updated every call.  The actor and target networks
        are updated only every ``policy_delay`` calls.

        Parameters
        ----------
        batch
            A named-tuple or object with attributes ``obs``, ``action``,
            ``reward``, ``next_obs``, ``done`` (each a tensor or ndarray).

        Returns
        -------
        dict
            Scalar training metrics: ``q_loss``, ``q1_mean``, ``q2_mean``
            (always present), and ``actor_loss``, ``noise_sigma``
            (present on actor update steps).
        """
        cfg: TD3Config = self._config  # type: ignore[assignment]

        obs = self._to_tensor(batch.obs, dtype=torch.float32)
        action = self._to_tensor(batch.action, dtype=torch.float32)
        reward = self._to_tensor(batch.reward, dtype=torch.float32).unsqueeze(-1)
        next_obs = self._to_tensor(batch.next_obs, dtype=torch.float32)
        done = self._to_tensor(batch.done, dtype=torch.float32).unsqueeze(-1)

        self._update_count += 1

        # ---- 1. Compute target Q-value ----
        with torch.no_grad():
            # Target action with smoothing noise
            next_features = self.actor_trunk_target(next_obs)
            next_action = self.actor_head_target(next_features)

            # Add clipped noise for target policy smoothing
            noise = (torch.randn_like(next_action) * cfg.policy_noise).clamp(
                -cfg.noise_clip,
                cfg.noise_clip,
            )
            next_action = (next_action + noise).clamp(
                self._action_low,
                self._action_high,
            )

            # Clipped double Q
            q1_target = self.q1_target(next_obs, next_action)
            q2_target = self.q2_target(next_obs, next_action)
            q_target_min = torch.min(q1_target, q2_target)
            target_q = reward + cfg.gamma * (1.0 - done) * q_target_min

        # ---- 2. Critic update ----
        q1_pred = self.q1(obs, action)
        q2_pred = self.q2(obs, action)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        if cfg.max_grad_norm is not None:
            critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
            self._clip_grad_norm(critic_params, cfg.max_grad_norm)
        self.critic_optimizer.step()

        metrics: dict[str, float] = {
            "q_loss": q_loss.item(),
            "q1_mean": float(q1_pred.mean().item()),
            "q2_mean": float(q2_pred.mean().item()),
        }

        # ---- 3. Delayed actor update ----
        if self._update_count % cfg.policy_delay == 0:
            # Freeze Q-networks to avoid computing gradients for them
            for p in self.q1.parameters():
                p.requires_grad = False

            features = self.actor_trunk(obs)
            actor_action = self.actor_head(features)
            actor_loss = -self.q1(obs, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if cfg.max_grad_norm is not None:
                actor_params = list(self.actor_trunk.parameters()) + list(
                    self.actor_head.parameters()
                )
                self._clip_grad_norm(actor_params, cfg.max_grad_norm)
            self.actor_optimizer.step()

            # Unfreeze Q-networks
            for p in self.q1.parameters():
                p.requires_grad = True

            metrics["actor_loss"] = actor_loss.item()
            metrics["noise_sigma"] = self._noise_schedule.value(self._total_steps)

            # ---- 4. Soft update target networks (also delayed) ----
            self._soft_update(self.q1_target, self.q1, cfg.tau)
            self._soft_update(self.q2_target, self.q2, cfg.tau)
            self._soft_update(self.actor_trunk_target, self.actor_trunk, cfg.tau)
            self._soft_update(self.actor_head_target, self.actor_head, cfg.tau)

        self._metrics.record_dict(metrics)
        self._total_steps += 1
        return metrics

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def get_q_values(self, observation: np.ndarray, action: np.ndarray) -> tuple[float, float]:
        """Compute Q-values for a given state-action pair.

        Useful for debugging and analysis.

        Parameters
        ----------
        observation : np.ndarray
            Single observation.
        action : np.ndarray
            Single action.

        Returns
        -------
        q1, q2 : float
            Q-values from both critic networks.
        """
        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32).unsqueeze(0)
            act_t = self._to_tensor(action, dtype=torch.float32).unsqueeze(0)
            q1 = self.q1(obs_t, act_t).item()
            q2 = self.q2(obs_t, act_t).item()
        return q1, q2

    def get_target_action(self, observation: np.ndarray) -> np.ndarray:
        """Compute the target policy action for a given observation.

        Parameters
        ----------
        observation : np.ndarray
            Single observation.

        Returns
        -------
        np.ndarray
            Target policy action.
        """
        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32).unsqueeze(0)
            features = self.actor_trunk_target(obs_t)
            action = self.actor_head_target(features)
        return self._to_numpy(action.squeeze(0))

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save agent checkpoint to disk.

        Persists all networks, target networks, optimizers, and
        normalization statistics.

        Parameters
        ----------
        path : str or Path
            Directory or file path for the checkpoint.
        """
        state_dicts: dict[str, Any] = {
            "actor_trunk": self.actor_trunk.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "actor_trunk_target": self.actor_trunk_target.state_dict(),
            "actor_head_target": self.actor_head_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "update_count": self._update_count,
        }
        if self._obs_rms is not None:
            state_dicts["obs_rms"] = self._obs_rms.state_dict()
        self._save_checkpoint(path, state_dicts)

    def load(self, path: str | pathlib.Path) -> None:
        """Load agent checkpoint from disk.

        Restores all networks, target networks, and normalization
        statistics.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        """
        payload = self._load_checkpoint(path)
        model = payload["model"]

        self.actor_trunk.load_state_dict(model["actor_trunk"])
        self.actor_head.load_state_dict(model["actor_head"])
        self.actor_trunk_target.load_state_dict(model["actor_trunk_target"])
        self.actor_head_target.load_state_dict(model["actor_head_target"])
        self.q1.load_state_dict(model["q1"])
        self.q2.load_state_dict(model["q2"])
        self.q1_target.load_state_dict(model["q1_target"])
        self.q2_target.load_state_dict(model["q2_target"])

        if "update_count" in model:
            self._update_count = model["update_count"]
        if "obs_rms" in model and self._obs_rms is not None:
            self._obs_rms.load_state_dict(model["obs_rms"])
