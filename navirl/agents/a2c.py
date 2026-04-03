"""
NavIRL A2C Agent
================

Synchronous Advantage Actor-Critic implementation for pedestrian navigation.
Uses a shared feature extractor with separate policy and value heads, and
performs a single gradient step per rollout collection.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from navirl.agents.base import BaseAgent, HyperParameters
from navirl.agents.networks import (
    MLP,
    CategoricalPolicyHead,
    GaussianPolicyHead,
    ValueHead,
)
from navirl.training.buffer import RolloutBuffer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class A2CConfig(HyperParameters):
    """Hyperparameters for the A2C agent."""

    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 1.0
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    normalize_advantages: bool = False
    n_steps: int = 5


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class A2CAgent(BaseAgent):
    """Synchronous Advantage Actor-Critic agent.

    Uses a shared feature extractor backbone with separate policy and value
    heads.  The policy head is a :class:`GaussianPolicyHead` for continuous
    action spaces or a :class:`CategoricalPolicyHead` for discrete ones.

    Parameters
    ----------
    config : A2CConfig
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
        config: A2CConfig,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)
        cfg: A2CConfig = self._config  # type: ignore[assignment]

        obs_dim = int(np.prod(observation_space.shape))
        hidden_dims: Sequence[int] = cfg.hidden_dims

        # --- Determine action space type ---
        self._continuous = hasattr(action_space, "shape") and len(action_space.shape) > 0
        action_dim = (
            int(np.prod(action_space.shape)) if self._continuous else int(action_space.n)
        )

        # --- Shared feature extractor ---
        self._feature_extractor = MLP(
            input_dim=obs_dim,
            output_dim=0,
            hidden_dims=hidden_dims,
            activation=cfg.activation,
            init="orthogonal",
        )
        feature_dim = self._feature_extractor._feature_dim

        # --- Policy head ---
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

        # --- Value head ---
        self._value_head = ValueHead(
            input_dim=feature_dim,
            hidden_dims=(),
        )

        # Move to device
        self._feature_extractor.to(self._device)
        self._policy_head.to(self._device)
        self._value_head.to(self._device)

        # Register modules for train/eval toggle
        self._modules.extend(
            [
                self._feature_extractor,
                self._policy_head,
                self._value_head,
            ]
        )

        # --- Optimizer (single optimizer for all parameters) ---
        all_params = (
            list(self._feature_extractor.parameters())
            + list(self._policy_head.parameters())
            + list(self._value_head.parameters())
        )
        self._optimizer = torch.optim.Adam(all_params, lr=cfg.lr)
        self._optimizers["a2c"] = self._optimizer

        self._log_module_summary("feature_extractor", self._feature_extractor)
        self._log_module_summary("policy_head", self._policy_head)
        self._log_module_summary("value_head", self._value_head)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _get_value(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Compute V(s) from observation tensor."""
        features = self._feature_extractor(obs_tensor)
        return self._value_head(features).squeeze(-1)

    def _evaluate_actions(
        self,
        obs_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate actions under the current policy.

        Returns
        -------
        log_probs : Tensor ``(B,)``
        entropy : Tensor ``(B,)``
        values : Tensor ``(B,)``
        """
        features = self._feature_extractor(obs_tensor)
        values = self._value_head(features).squeeze(-1)

        if self._continuous:
            mean, log_std = self._policy_head(features)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self._policy_head(features)
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
        info : dict
            Contains ``"log_prob"`` and ``"value"`` tensors (as numpy).
        """
        obs_tensor = self._to_tensor(observation, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self._feature_extractor(obs_tensor)
            value = self._value_head(features).squeeze(-1)

            if self._continuous:
                mean, log_std = self._policy_head(features)
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
                logits = self._policy_head(features)
                dist = torch.distributions.Categorical(logits=logits)
                action_tensor = logits.argmax(dim=-1) if deterministic else dist.sample()
                log_prob = dist.log_prob(action_tensor)

        action = self._to_numpy(action_tensor.squeeze(0))
        info = {
            "log_prob": self._to_numpy(log_prob.squeeze(0)),
            "value": self._to_numpy(value.squeeze(0)),
        }
        return action, info

    def update(self, batch: RolloutBuffer) -> dict[str, float]:
        """Run a single gradient step on the collected rollout data.

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
        cfg: A2CConfig = self._config  # type: ignore[assignment]

        # Flatten the buffer across time and environments
        all_obs = batch.observations.reshape(-1, *batch.obs_shape)
        all_actions = batch.actions.reshape(-1, *batch.action_shape)
        all_returns = batch.returns.reshape(-1)
        all_advantages = batch.advantages.reshape(-1)

        # Convert to tensors
        obs_t = self._to_tensor(all_obs, dtype=torch.float32)
        actions_t = self._to_tensor(all_actions, dtype=torch.float32)
        returns_t = self._to_tensor(all_returns, dtype=torch.float32)
        advantages_t = self._to_tensor(all_advantages, dtype=torch.float32)

        # Optionally normalize advantages
        if cfg.normalize_advantages and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Evaluate all actions under the current policy
        log_probs, entropy, values = self._evaluate_actions(obs_t, actions_t)

        # --- Policy loss: -mean(advantage * log_prob) ---
        policy_loss = -(advantages_t.detach() * log_probs).mean()

        # --- Value loss: MSE(values, returns) ---
        value_loss = F.mse_loss(values, returns_t)

        # --- Entropy bonus ---
        entropy_mean = entropy.mean()

        # --- Combined loss ---
        loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy_mean

        self._optimizer.zero_grad()
        loss.backward()
        self._clip_grad_norm(
            list(self._feature_extractor.parameters())
            + list(self._policy_head.parameters())
            + list(self._value_head.parameters()),
            cfg.max_grad_norm,
        )
        self._optimizer.step()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
        }

        self._metrics.record_dict(metrics)
        self.on_update_end(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save the agent checkpoint to *path*."""
        state_dicts = {
            "feature_extractor": self._feature_extractor.state_dict(),
            "policy_head": self._policy_head.state_dict(),
            "value_head": self._value_head.state_dict(),
        }
        self._save_checkpoint(path, state_dicts)

    def load(self, path: str | pathlib.Path) -> None:
        """Load agent state from a checkpoint at *path*."""
        payload = self._load_checkpoint(path)
        model_states = payload.get("model", {})
        self._feature_extractor.load_state_dict(model_states["feature_extractor"])
        self._policy_head.load_state_dict(model_states["policy_head"])
        self._value_head.load_state_dict(model_states["value_head"])
