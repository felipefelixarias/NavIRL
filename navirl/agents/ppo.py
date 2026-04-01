"""
NavIRL PPO Agent
================

Proximal Policy Optimization (Schulman et al., 2017) implementation for
pedestrian navigation.  Uses clipped surrogate objective, Generalized
Advantage Estimation (GAE), mini-batch updates, learning rate annealing,
observation normalization, and supports both continuous and discrete action
spaces.

Key features
------------
* **Clipped surrogate objective** with configurable epsilon.
* **GAE** (lambda-return) for advantage estimation.
* **Value function clipping** to stabilise critic updates.
* **Entropy bonus** encouraging exploration.
* **Mini-batch SGD** over multiple epochs per rollout.
* **Learning rate annealing** (linear or cosine schedule).
* **Observation normalization** via running mean/variance.
* **Adaptive KL early stopping** as an optional safeguard.
* **Gradient clipping** via max-norm.

References
----------
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
High-Dimensional Continuous Control Using Generalized Advantage Estimation.
*ICLR 2016*.
"""

from __future__ import annotations

import logging
import math
import pathlib
from collections.abc import Sequence
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
from navirl.agents.networks import (
    MLP,
    CategoricalPolicyHead,
    GaussianPolicyHead,
    ValueHead,
)
from navirl.training.buffer import RolloutBuffer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Learning-rate schedule helpers
# ---------------------------------------------------------------------------


def _linear_schedule(initial_lr: float, current_step: int, total_steps: int) -> float:
    """Linearly anneal learning rate from *initial_lr* to zero.

    Parameters
    ----------
    initial_lr : float
        Starting learning rate.
    current_step : int
        Current optimisation step (0-indexed).
    total_steps : int
        Total number of optimisation steps planned.

    Returns
    -------
    float
        Annealed learning rate, clamped to be non-negative.
    """
    if total_steps <= 0:
        return initial_lr
    fraction = 1.0 - current_step / total_steps
    return max(0.0, initial_lr * fraction)


def _cosine_schedule(
    initial_lr: float,
    current_step: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Cosine-annealing learning rate schedule.

    Parameters
    ----------
    initial_lr : float
        Starting learning rate.
    current_step : int
        Current optimisation step.
    total_steps : int
        Total number of optimisation steps.
    min_lr : float
        Minimum learning rate at the end of the schedule.

    Returns
    -------
    float
        Annealed learning rate.
    """
    if total_steps <= 0:
        return initial_lr
    progress = min(current_step / total_steps, 1.0)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards : np.ndarray
        Rewards of shape ``(T, N)`` where *T* is the number of time steps
        and *N* is the number of parallel environments.
    values : np.ndarray
        Value estimates of shape ``(T, N)``.
    dones : np.ndarray
        Done flags of shape ``(T, N)``.
    last_values : np.ndarray
        Bootstrap value estimates of shape ``(N,)`` for the last step.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter controlling bias-variance trade-off.

    Returns
    -------
    advantages : np.ndarray
        GAE advantages of shape ``(T, N)``.
    returns : np.ndarray
        Discounted returns ``advantages + values``, shape ``(T, N)``.
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_values
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the explained variance between predictions and targets.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values (flattened).
    y_true : np.ndarray
        True values (flattened).

    Returns
    -------
    float
        Explained variance ratio in ``(-inf, 1.0]``.  A value of 1.0
        indicates a perfect prediction; 0.0 indicates predicting the mean.
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    var_y = np.var(y_true)
    if var_y == 0.0:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig(HyperParameters):
    """Hyperparameters for the PPO agent.

    Attributes
    ----------
    lr : float
        Initial learning rate for Adam.
    gamma : float
        Discount factor for future rewards.
    gae_lambda : float
        Lambda for Generalized Advantage Estimation.
    clip_epsilon : float
        Clipping range for the surrogate objective.
    ppo_epochs : int
        Number of optimisation epochs per rollout.
    mini_batch_size : int
        Mini-batch size for each gradient step within an epoch.
    value_loss_coeff : float
        Coefficient for the value-function loss term.
    entropy_coeff : float
        Coefficient for the entropy bonus.
    max_grad_norm : float
        Maximum gradient norm for gradient clipping.
    hidden_dims : tuple of int
        Hidden layer sizes for actor and critic backbones.
    activation : str
        Activation function for hidden layers.
    normalize_advantages : bool
        Whether to normalize advantages per mini-batch.
    clip_value_loss : bool
        Whether to clip the value-function loss.
    target_kl : float or None
        If set, early-stop PPO epochs when the mean approximate KL
        divergence exceeds this threshold.
    lr_schedule : str
        Learning rate schedule type: ``"constant"``, ``"linear"``, or
        ``"cosine"``.
    total_timesteps : int
        Total training timesteps (used for learning-rate annealing).
    normalize_observations : bool
        Whether to normalize observations using running statistics.
    observation_clip : float
        Clipping range for normalized observations.
    normalize_returns : bool
        Whether to normalize returns using running statistics.
    shared_backbone : bool
        If ``True``, actor and critic share the feature-extraction
        backbone (saving parameters but potentially hurting stability).
    use_sde : bool
        If ``True``, use State-Dependent Exploration (generalised
        exploration with structured noise in parameter space).
    sde_sample_freq : int
        How often (in steps) to re-sample the SDE noise matrix.
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    target_kl: float | None = None
    lr_schedule: str = "constant"
    total_timesteps: int = 1_000_000
    normalize_observations: bool = False
    observation_clip: float = 10.0
    normalize_returns: bool = False
    shared_backbone: bool = False
    use_sde: bool = False
    sde_sample_freq: int = -1


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent.

    Builds a separate (or optionally shared) actor and critic backbone.
    The actor uses a :class:`GaussianPolicyHead` for continuous action
    spaces or a :class:`CategoricalPolicyHead` for discrete ones.  The
    critic uses a :class:`ValueHead`.

    Parameters
    ----------
    config : PPOConfig
        Agent-specific hyperparameters.
    observation_space : gymnasium.spaces.Space
        Environment observation space.
    action_space : gymnasium.spaces.Space
        Environment action space.
    device : str or torch.device
        Compute device.
    seed : int or None
        Random seed for reproducibility.
    metrics_callback : callable or None
        Optional structured-metrics callback.
    """

    def __init__(
        self,
        config: PPOConfig,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)
        cfg: PPOConfig = self._config  # type: ignore[assignment]

        obs_dim = int(np.prod(observation_space.shape))
        hidden_dims: Sequence[int] = cfg.hidden_dims

        # --- Determine action space type ---
        self._continuous = hasattr(action_space, "shape") and len(action_space.shape) > 0
        if self._continuous:
            action_dim = int(np.prod(action_space.shape))
        else:
            action_dim = int(action_space.n)  # Discrete

        # --- Observation normalization ---
        self._obs_rms: RunningMeanStd | None = None
        if cfg.normalize_observations:
            self._obs_rms = RunningMeanStd(shape=observation_space.shape)

        # --- Return normalization ---
        self._ret_rms: RunningMeanStd | None = None
        if cfg.normalize_returns:
            self._ret_rms = RunningMeanStd(shape=())

        # --- Actor backbone ---
        self._actor_backbone = MLP(
            input_dim=obs_dim,
            output_dim=0,
            hidden_dims=hidden_dims,
            activation=cfg.activation,
            init="orthogonal",
        )
        feature_dim = self._actor_backbone._feature_dim

        if self._continuous:
            self._policy_head = GaussianPolicyHead(
                input_dim=feature_dim,
                action_dim=action_dim,
            )
        else:
            self._policy_head = CategoricalPolicyHead(
                input_dim=feature_dim,
                num_actions=action_dim,
            )

        # --- Critic backbone (separate or shared) ---
        if cfg.shared_backbone:
            self._critic_backbone = self._actor_backbone
        else:
            self._critic_backbone = MLP(
                input_dim=obs_dim,
                output_dim=0,
                hidden_dims=hidden_dims,
                activation=cfg.activation,
                init="orthogonal",
            )
        self._value_head = ValueHead(
            input_dim=self._critic_backbone._feature_dim,
            hidden_dims=(),
        )

        # Move to device
        self._actor_backbone.to(self._device)
        self._policy_head.to(self._device)
        if not cfg.shared_backbone:
            self._critic_backbone.to(self._device)
        self._value_head.to(self._device)

        # Register modules for train/eval toggle
        self._modules.extend([
            self._actor_backbone,
            self._policy_head,
            self._value_head,
        ])
        if not cfg.shared_backbone:
            self._modules.append(self._critic_backbone)

        # --- Optimizer ---
        all_params: list[torch.nn.Parameter] = list(self._actor_backbone.parameters()) + list(self._policy_head.parameters())
        if not cfg.shared_backbone:
            all_params += list(self._critic_backbone.parameters())
        all_params += list(self._value_head.parameters())
        self._optimizer = torch.optim.Adam(all_params, lr=cfg.lr, eps=1e-5)
        self._optimizers["ppo"] = self._optimizer

        # --- LR tracking ---
        self._initial_lr = cfg.lr
        self._lr_update_count: int = 0

        self._log_module_summary("actor_backbone", self._actor_backbone)
        self._log_module_summary("policy_head", self._policy_head)
        if not cfg.shared_backbone:
            self._log_module_summary("critic_backbone", self._critic_backbone)
        self._log_module_summary("value_head", self._value_head)

    # ------------------------------------------------------------------
    # Observation normalization helpers
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics if enabled.

        Parameters
        ----------
        obs : np.ndarray
            Raw observation from the environment.

        Returns
        -------
        np.ndarray
            Normalized (or unchanged) observation.
        """
        cfg: PPOConfig = self._config  # type: ignore[assignment]
        if self._obs_rms is not None:
            if self._training:
                self._obs_rms.update(obs)
            return self._obs_rms.normalize(obs, clip=cfg.observation_clip).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Learning rate annealing
    # ------------------------------------------------------------------

    def _update_learning_rate(self) -> float:
        """Update the learning rate according to the configured schedule.

        Returns
        -------
        float
            The new learning rate.
        """
        cfg: PPOConfig = self._config  # type: ignore[assignment]
        if cfg.lr_schedule == "constant":
            return self._initial_lr

        # Estimate progress based on total timesteps
        progress_steps = self._total_steps
        total = cfg.total_timesteps

        if cfg.lr_schedule == "linear":
            new_lr = _linear_schedule(self._initial_lr, progress_steps, total)
        elif cfg.lr_schedule == "cosine":
            new_lr = _cosine_schedule(self._initial_lr, progress_steps, total)
        else:
            self._logger.warning(
                "Unknown LR schedule %r; falling back to constant.", cfg.lr_schedule
            )
            return self._initial_lr

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _get_value(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Compute V(s) from an observation tensor.

        Parameters
        ----------
        obs_tensor : torch.Tensor
            Observation tensor of shape ``(B, obs_dim)``.

        Returns
        -------
        torch.Tensor
            Value estimates of shape ``(B,)``.
        """
        features = self._critic_backbone(obs_tensor)
        return self._value_head(features).squeeze(-1)

    def _evaluate_actions(
        self,
        obs_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate actions under the current policy.

        Parameters
        ----------
        obs_tensor : torch.Tensor
            Observations ``(B, obs_dim)``.
        actions_tensor : torch.Tensor
            Actions ``(B, action_dim)`` or ``(B,)`` for discrete.

        Returns
        -------
        log_probs : torch.Tensor ``(B,)``
            Log-probabilities of the actions under the current policy.
        entropy : torch.Tensor ``(B,)``
            Per-sample entropy of the policy distribution.
        values : torch.Tensor ``(B,)``
            Value estimates from the critic.
        """
        actor_features = self._actor_backbone(obs_tensor)
        values = self._get_value(obs_tensor)

        if self._continuous:
            mean, log_std = self._policy_head(actor_features)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self._policy_head(actor_features)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions_tensor.squeeze(-1))
            entropy = dist.entropy()

        return log_probs, entropy, values

    def _get_policy_distribution(
        self, obs_tensor: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Build the policy distribution for a batch of observations.

        Parameters
        ----------
        obs_tensor : torch.Tensor
            Observations ``(B, obs_dim)``.

        Returns
        -------
        torch.distributions.Distribution
            The action distribution.
        """
        actor_features = self._actor_backbone(obs_tensor)
        if self._continuous:
            mean, log_std = self._policy_head(actor_features)
            std = log_std.exp()
            return torch.distributions.Normal(mean, std)
        else:
            logits = self._policy_head(actor_features)
            return torch.distributions.Categorical(logits=logits)

    # ------------------------------------------------------------------
    # Abstract interface: act
    # ------------------------------------------------------------------

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
            If ``True``, return the mean action (continuous) or argmax
            (discrete) without sampling.

        Returns
        -------
        action : np.ndarray
            The chosen action, compatible with ``env.step``.
        info : dict
            Auxiliary information containing ``"log_prob"`` and ``"value"``
            as numpy scalars.
        """
        observation = self._normalize_obs(observation)
        obs_tensor = self._to_tensor(observation, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            actor_features = self._actor_backbone(obs_tensor)
            value = self._get_value(obs_tensor)

            if self._continuous:
                mean, log_std = self._policy_head(actor_features)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                if deterministic:
                    action_tensor = mean
                else:
                    action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum(dim=-1)
            else:
                logits = self._policy_head(actor_features)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action_tensor = logits.argmax(dim=-1)
                else:
                    action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)

        action = self._to_numpy(action_tensor.squeeze(0))
        info = {
            "log_prob": self._to_numpy(log_prob.squeeze(0)),
            "value": self._to_numpy(value.squeeze(0)),
        }
        return action, info

    # ------------------------------------------------------------------
    # Abstract interface: update
    # ------------------------------------------------------------------

    def update(self, batch: RolloutBuffer) -> dict[str, float]:
        """Run PPO optimisation epochs on collected rollout data.

        Performs multiple epochs of mini-batch gradient descent on the
        rollout buffer, computing the clipped surrogate loss, value loss,
        and entropy bonus.  Optionally applies KL-based early stopping
        and learning rate annealing.

        Parameters
        ----------
        batch : RolloutBuffer
            Filled rollout buffer with computed returns and advantages.

        Returns
        -------
        metrics : dict
            Scalar training metrics including ``policy_loss``,
            ``value_loss``, ``entropy``, ``approx_kl``,
            ``clip_fraction``, ``explained_variance``, and ``lr``.
        """
        self.on_update_start()
        cfg: PPOConfig = self._config  # type: ignore[assignment]

        # --- Update learning rate ---
        current_lr = self._update_learning_rate()

        total_size = batch.buffer_size * batch.n_envs
        all_obs = batch.observations.reshape(-1, *batch.obs_shape)
        all_actions = batch.actions.reshape(-1, *batch.action_shape)
        all_old_log_probs = batch.log_probs.reshape(-1)
        all_old_values = batch.values.reshape(-1)
        all_advantages = batch.advantages.reshape(-1)
        all_returns = batch.returns.reshape(-1)

        # --- Normalize returns if enabled ---
        if self._ret_rms is not None:
            self._ret_rms.update(all_returns)
            all_returns = self._ret_rms.normalize(all_returns).astype(np.float32)

        # --- Compute explained variance before training ---
        ev = explained_variance(all_old_values, all_returns)

        # Convert to tensors
        obs_t = self._to_tensor(all_obs, dtype=torch.float32)
        actions_t = self._to_tensor(all_actions, dtype=torch.float32)
        old_log_probs_t = self._to_tensor(all_old_log_probs, dtype=torch.float32)
        old_values_t = self._to_tensor(all_old_values, dtype=torch.float32)
        advantages_t = self._to_tensor(all_advantages, dtype=torch.float32)
        returns_t = self._to_tensor(all_returns, dtype=torch.float32)

        # Accumulate metrics across epochs
        epoch_policy_losses: list[float] = []
        epoch_value_losses: list[float] = []
        epoch_entropies: list[float] = []
        epoch_approx_kls: list[float] = []
        epoch_clip_fractions: list[float] = []
        epoch_total_losses: list[float] = []

        early_stopped = False
        completed_epochs = 0

        for epoch_idx in range(cfg.ppo_epochs):
            # Generate random mini-batch indices
            indices = np.random.permutation(total_size)

            for start in range(0, total_size, cfg.mini_batch_size):
                end = min(start + cfg.mini_batch_size, total_size)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_old_values = old_values_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Normalize advantages per mini-batch
                if cfg.normalize_advantages and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Evaluate actions under current policy
                new_log_probs, entropy, new_values = self._evaluate_actions(
                    mb_obs, mb_actions,
                )

                # --- Policy loss (clipped surrogate) ---
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss (optionally clipped) ---
                if cfg.clip_value_loss:
                    value_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -cfg.clip_epsilon,
                        cfg.clip_epsilon,
                    )
                    value_loss_unclipped = (new_values - mb_returns).pow(2)
                    value_loss_clipped = (value_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(new_values, mb_returns)

                # --- Entropy bonus ---
                entropy_loss = -entropy.mean()

                # --- Combined loss ---
                loss = (
                    policy_loss
                    + cfg.value_loss_coeff * value_loss
                    + cfg.entropy_coeff * entropy_loss
                )

                self._optimizer.zero_grad()
                loss.backward()
                self._clip_grad_norm(
                    list(self._actor_backbone.parameters())
                    + list(self._policy_head.parameters())
                    + list(self._critic_backbone.parameters())
                    + list(self._value_head.parameters()),
                    cfg.max_grad_norm,
                )
                self._optimizer.step()
                self._lr_update_count += 1

                # --- Metrics ---
                with torch.no_grad():
                    # Use the more numerically stable approximation:
                    # KL ~= (ratio - 1) - log(ratio)
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float().mean().item()
                    )

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(-entropy_loss.item())
                epoch_approx_kls.append(approx_kl)
                epoch_clip_fractions.append(clip_fraction)
                epoch_total_losses.append(loss.item())

            completed_epochs += 1

            # Optional early stopping on KL divergence
            if cfg.target_kl is not None:
                n_mb = max(1, total_size // cfg.mini_batch_size)
                mean_kl = float(np.mean(epoch_approx_kls[-n_mb:]))
                if mean_kl > 1.5 * cfg.target_kl:
                    self._logger.info(
                        "PPO early stopping at epoch %d/%d  "
                        "(approx_kl=%.4f > 1.5 * target_kl=%.4f)",
                        epoch_idx + 1, cfg.ppo_epochs, mean_kl, cfg.target_kl,
                    )
                    early_stopped = True
                    break

        metrics = {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
            "approx_kl": float(np.mean(epoch_approx_kls)),
            "clip_fraction": float(np.mean(epoch_clip_fractions)),
            "total_loss": float(np.mean(epoch_total_losses)),
            "explained_variance": ev,
            "lr": current_lr,
            "completed_epochs": float(completed_epochs),
            "early_stopped": float(early_stopped),
        }

        self._metrics.record_dict(metrics)
        self.on_update_end(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Predict value (useful for bootstrapping)
    # ------------------------------------------------------------------

    def predict_value(self, observation: np.ndarray) -> float:
        """Predict V(s) for a single observation.

        Useful for bootstrapping the last value at the end of a rollout.

        Parameters
        ----------
        observation : np.ndarray
            A single observation.

        Returns
        -------
        float
            The scalar value estimate.
        """
        observation = self._normalize_obs(observation)
        obs_t = self._to_tensor(observation, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            value = self._get_value(obs_t)
        return float(value.item())

    # ------------------------------------------------------------------
    # Training-loop hooks
    # ------------------------------------------------------------------

    def on_rollout_start(self) -> None:
        """Reset any per-rollout state."""
        pass

    def on_rollout_end(self) -> None:
        """Called after rollout collection finishes."""
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save the agent checkpoint to *path*.

        Persists network weights, optimizer state, normalization
        statistics, and training counters.

        Parameters
        ----------
        path : str or Path
            Directory or file path for the checkpoint.
        """
        state_dicts: dict[str, Any] = {
            "actor_backbone": self._actor_backbone.state_dict(),
            "policy_head": self._policy_head.state_dict(),
            "critic_backbone": self._critic_backbone.state_dict(),
            "value_head": self._value_head.state_dict(),
            "lr_update_count": self._lr_update_count,
        }
        if self._obs_rms is not None:
            state_dicts["obs_rms"] = self._obs_rms.state_dict()
        if self._ret_rms is not None:
            state_dicts["ret_rms"] = self._ret_rms.state_dict()
        self._save_checkpoint(path, state_dicts)

    def load(self, path: str | pathlib.Path) -> None:
        """Load agent state from a checkpoint at *path*.

        Restores network weights, optimizer state, normalization
        statistics, and training counters.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        """
        payload = self._load_checkpoint(path)
        model_states = payload.get("model", {})
        self._actor_backbone.load_state_dict(model_states["actor_backbone"])
        self._policy_head.load_state_dict(model_states["policy_head"])
        self._critic_backbone.load_state_dict(model_states["critic_backbone"])
        self._value_head.load_state_dict(model_states["value_head"])
        if "lr_update_count" in model_states:
            self._lr_update_count = model_states["lr_update_count"]
        if "obs_rms" in model_states and self._obs_rms is not None:
            self._obs_rms.load_state_dict(model_states["obs_rms"])
        if "ret_rms" in model_states and self._ret_rms is not None:
            self._ret_rms.load_state_dict(model_states["ret_rms"])

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    def get_policy_parameters(self) -> list[torch.nn.Parameter]:
        """Return a flat list of all policy (actor) parameters.

        Returns
        -------
        list of torch.nn.Parameter
            Parameters from the actor backbone and policy head.
        """
        return list(self._actor_backbone.parameters()) + list(self._policy_head.parameters())

    def get_value_parameters(self) -> list[torch.nn.Parameter]:
        """Return a flat list of all value (critic) parameters.

        Returns
        -------
        list of torch.nn.Parameter
            Parameters from the critic backbone and value head.
        """
        return list(self._critic_backbone.parameters()) + list(self._value_head.parameters())

    def get_current_lr(self) -> float:
        """Return the current learning rate from the optimizer.

        Returns
        -------
        float
            The learning rate of the first parameter group.
        """
        return self._optimizer.param_groups[0]["lr"]
