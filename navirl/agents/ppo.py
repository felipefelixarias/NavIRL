"""
NavIRL PPO Agent
================

Proximal Policy Optimization (Schulman et al., 2017) implementation for
pedestrian navigation.  Uses clipped surrogate objective, GAE for advantage
estimation, and supports both continuous and discrete action spaces.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navirl.agents.base import BaseAgent, HyperParameters
from navirl.agents.networks import (
    MLP,
    CategoricalPolicyHead,
    GaussianPolicyHead,
    ValueHead,
    init_weights_orthogonal,
)
from navirl.training.buffer import RolloutBuffer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig(HyperParameters):
    """Hyperparameters for the PPO agent."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    normalize_advantages: bool = True
    clip_value_loss: bool = True
    target_kl: Optional[float] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent.

    Builds a separate actor and critic network.  The actor uses a
    :class:`GaussianPolicyHead` for continuous action spaces or a
    :class:`CategoricalPolicyHead` for discrete ones.  The critic uses a
    :class:`ValueHead`.

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
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
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

        # --- Actor ---
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

        # --- Critic ---
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
        self._critic_backbone.to(self._device)
        self._value_head.to(self._device)

        # Register modules for train/eval toggle
        self._modules.extend([
            self._actor_backbone,
            self._policy_head,
            self._critic_backbone,
            self._value_head,
        ])

        # --- Optimizer ---
        all_params = (
            list(self._actor_backbone.parameters())
            + list(self._policy_head.parameters())
            + list(self._critic_backbone.parameters())
            + list(self._value_head.parameters())
        )
        self._optimizer = torch.optim.Adam(all_params, lr=cfg.lr)
        self._optimizers["ppo"] = self._optimizer

        self._log_module_summary("actor_backbone", self._actor_backbone)
        self._log_module_summary("policy_head", self._policy_head)
        self._log_module_summary("critic_backbone", self._critic_backbone)
        self._log_module_summary("value_head", self._value_head)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _get_value(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Compute V(s) from observation tensor."""
        features = self._critic_backbone(obs_tensor)
        return self._value_head(features).squeeze(-1)

    def _evaluate_actions(
        self,
        obs_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate actions under the current policy.

        Returns
        -------
        log_probs : Tensor ``(B,)``
        entropy : Tensor ``(B,)``
        values : Tensor ``(B,)``
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

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        info : dict
            Contains ``"log_prob"`` and ``"value"`` tensors (as numpy).
        """
        obs_tensor = self._to_tensor(observation, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            actor_features = self._actor_backbone(obs_tensor)
            value = self._get_value(obs_tensor)

            if self._continuous:
                mean, log_std = self._policy_head(actor_features)
                if deterministic:
                    action_tensor = mean
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    log_prob = dist.log_prob(action_tensor).sum(dim=-1)
                else:
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
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

    def update(self, batch: RolloutBuffer) -> Dict[str, float]:
        """Run PPO optimisation epochs on collected rollout data.

        Parameters
        ----------
        batch : RolloutBuffer
            Filled rollout buffer with computed returns and advantages.

        Returns
        -------
        metrics : dict
            Scalar training metrics.
        """
        self.on_update_start()
        cfg: PPOConfig = self._config  # type: ignore[assignment]

        total_size = batch.buffer_size * batch.n_envs
        all_obs = batch.observations.reshape(-1, *batch.obs_shape)
        all_actions = batch.actions.reshape(-1, *batch.action_shape)
        all_old_log_probs = batch.log_probs.reshape(-1)
        all_old_values = batch.values.reshape(-1)
        all_advantages = batch.advantages.reshape(-1)
        all_returns = batch.returns.reshape(-1)

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

        early_stopped = False

        for _epoch in range(cfg.ppo_epochs):
            # Generate random mini-batch indices
            indices = np.random.permutation(total_size)

            for start in range(0, total_size, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_old_values = old_values_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Normalize advantages
                if cfg.normalize_advantages and len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Evaluate actions under current policy
                new_log_probs, entropy, new_values = self._evaluate_actions(
                    mb_obs, mb_actions,
                )

                # --- Policy loss (clipped surrogate) ---
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
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
                    value_loss_unclipped = F.mse_loss(new_values, mb_returns)
                    value_loss_clipped = F.mse_loss(value_clipped, mb_returns)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = F.mse_loss(new_values, mb_returns)

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

                # --- Metrics ---
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > cfg.clip_epsilon).float().mean().item()
                    )

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(-entropy_loss.item())
                epoch_approx_kls.append(approx_kl)
                epoch_clip_fractions.append(clip_fraction)

            # Optional early stopping on KL divergence
            if cfg.target_kl is not None:
                mean_kl = float(np.mean(epoch_approx_kls[-max(1, total_size // cfg.mini_batch_size):]))
                if mean_kl > cfg.target_kl:
                    self._logger.info(
                        "Early stopping at epoch %d/%d  (approx_kl=%.4f > target_kl=%.4f)",
                        _epoch + 1, cfg.ppo_epochs, mean_kl, cfg.target_kl,
                    )
                    early_stopped = True
                    break

        metrics = {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
            "approx_kl": float(np.mean(epoch_approx_kls)),
            "clip_fraction": float(np.mean(epoch_clip_fractions)),
        }

        self._metrics.record_dict(metrics)
        self.on_update_end(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save the agent checkpoint to *path*."""
        state_dicts = {
            "actor_backbone": self._actor_backbone.state_dict(),
            "policy_head": self._policy_head.state_dict(),
            "critic_backbone": self._critic_backbone.state_dict(),
            "value_head": self._value_head.state_dict(),
        }
        self._save_checkpoint(path, state_dicts)

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """Load agent state from a checkpoint at *path*."""
        payload = self._load_checkpoint(path)
        model_states = payload.get("model", {})
        self._actor_backbone.load_state_dict(model_states["actor_backbone"])
        self._policy_head.load_state_dict(model_states["policy_head"])
        self._critic_backbone.load_state_dict(model_states["critic_backbone"])
        self._value_head.load_state_dict(model_states["value_head"])
