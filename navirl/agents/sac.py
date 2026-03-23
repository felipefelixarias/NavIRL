"""
Soft Actor-Critic (SAC) Agent
=============================

Implementation of Soft Actor-Critic (Haarnoja et al., 2018) for continuous
action spaces in the NavIRL pedestrian navigation framework.

SAC is an off-policy maximum-entropy reinforcement learning algorithm that
optimises a stochastic policy by simultaneously maximising expected return
and entropy, encouraging exploration and robustness to model errors.

Key features:
* Squashed Gaussian policy with automatic entropy tuning.
* Twin Q-networks to mitigate positive bias in Q-value estimation.
* Soft target updates via Polyak averaging.

References
----------
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor. *ICML 2018*.
"""

from __future__ import annotations

import copy
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navirl.agents.base import BaseAgent, HyperParameters
from navirl.agents.networks import MLP, SquashedGaussianHead, TwinQHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SACConfig(HyperParameters):
    """Hyperparameters for Soft Actor-Critic.

    Attributes
    ----------
    lr_actor : float
        Learning rate for the actor (policy) network.
    lr_critic : float
        Learning rate for the twin Q-networks.
    lr_alpha : float
        Learning rate for the entropy temperature parameter.
    gamma : float
        Discount factor.
    tau : float
        Soft target update coefficient (Polyak averaging).
    alpha : float
        Initial entropy temperature. Used as a fixed value when
        ``auto_alpha`` is ``False``.
    auto_alpha : bool
        If ``True``, the entropy temperature is learned automatically.
    target_entropy : float or None
        Target entropy for automatic tuning. Defaults to
        ``-dim(action_space)`` when ``None``.
    hidden_dims : tuple of int
        Hidden layer widths for both actor and critic networks.
    activation : str
        Activation function name for hidden layers.
    batch_size : int
        Mini-batch size for each gradient update.
    max_grad_norm : float or None
        If set, gradients are clipped to this maximum norm.
    reward_scale : float
        Multiplicative reward scaling factor.
    """

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    batch_size: int = 256
    max_grad_norm: Optional[float] = None
    reward_scale: float = 1.0


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent for continuous control.

    Parameters
    ----------
    config : SACConfig
        Agent hyperparameters.
    observation_space
        Environment observation space (must expose ``.shape``).
    action_space
        Environment action space (must expose ``.shape``).
    device : str or torch.device
        Compute device.
    seed : int or None
        Random seed for reproducibility.
    metrics_callback : callable or None
        Optional ``(metrics_dict, step) -> None`` callback.
    """

    def __init__(
        self,
        config: SACConfig,
        observation_space: Any,
        action_space: Any,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)

        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        # ---- Actor (policy) ----
        self.actor_trunk = MLP(
            input_dim=obs_dim,
            output_dim=0,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(self._device)
        trunk_dim = self.actor_trunk.feature_dim

        self.actor_head = SquashedGaussianHead(
            input_dim=trunk_dim,
            action_dim=action_dim,
        ).to(self._device)

        # ---- Critic (twin Q-networks) ----
        self.critic = TwinQHead(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        # ---- Target critic ----
        self.critic_target = copy.deepcopy(self.critic).to(self._device)
        # Freeze target parameters
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Entropy temperature (alpha) ----
        if config.auto_alpha:
            self.target_entropy = (
                config.target_entropy
                if config.target_entropy is not None
                else -float(action_dim)
            )
            self.log_alpha = torch.tensor(
                np.log(config.alpha), dtype=torch.float32, device=self._device, requires_grad=True
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
            self._optimizers["alpha"] = self.alpha_optimizer
        else:
            self.target_entropy = None
            self.log_alpha = torch.tensor(
                np.log(config.alpha), dtype=torch.float32, device=self._device
            )
            self.alpha_optimizer = None

        self._alpha = config.alpha

        # ---- Optimizers ----
        actor_params = list(self.actor_trunk.parameters()) + list(self.actor_head.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self._optimizers["actor"] = self.actor_optimizer
        self._optimizers["critic"] = self.critic_optimizer

        # ---- Register modules for train/eval toggling ----
        self._modules.extend([self.actor_trunk, self.actor_head, self.critic, self.critic_target])

        self._log_module_summary("actor_trunk", self.actor_trunk)
        self._log_module_summary("actor_head", self.actor_head)
        self._log_module_summary("critic", self.critic)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Current entropy temperature."""
        return self.log_alpha.exp().item()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select an action given an observation.

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation.
        deterministic : bool
            If ``True``, return the mean action (no sampling).

        Returns
        -------
        action : np.ndarray
            The chosen action, scaled to ``(-1, 1)``.
        info : dict
            Contains ``"log_prob"`` when sampling stochastically.
        """
        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            features = self.actor_trunk(obs_t)

            if deterministic:
                mean, _ = self.actor_head(features)
                action = torch.tanh(mean)
                return self._to_numpy(action.squeeze(0)), {}
            else:
                action, log_prob = self.actor_head.sample(features)
                return (
                    self._to_numpy(action.squeeze(0)),
                    {"log_prob": self._to_numpy(log_prob.squeeze(0))},
                )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> Dict[str, float]:
        """Perform a single SAC gradient step on a batch of transitions.

        Parameters
        ----------
        batch
            A named-tuple or object with attributes ``obs``, ``action``,
            ``reward``, ``next_obs``, ``done`` (each a tensor or ndarray).

        Returns
        -------
        dict
            Scalar training metrics: ``q_loss``, ``actor_loss``,
            ``alpha_loss``, ``alpha``, ``entropy``.
        """
        cfg: SACConfig = self._config  # type: ignore[assignment]

        obs = self._to_tensor(batch.obs, dtype=torch.float32)
        action = self._to_tensor(batch.action, dtype=torch.float32)
        reward = self._to_tensor(batch.reward, dtype=torch.float32).unsqueeze(-1)
        next_obs = self._to_tensor(batch.next_obs, dtype=torch.float32)
        done = self._to_tensor(batch.done, dtype=torch.float32).unsqueeze(-1)

        reward = reward * cfg.reward_scale
        current_alpha = self.log_alpha.exp().detach()

        # ---- 1. Critic update ----
        with torch.no_grad():
            next_features = self.actor_trunk(next_obs)
            next_action, next_log_prob = self.actor_head.sample(next_features)
            next_log_prob = next_log_prob.unsqueeze(-1)

            q1_target, q2_target = self.critic_target(next_obs, next_action)
            q_target_min = torch.min(q1_target, q2_target)
            y = reward + cfg.gamma * (1.0 - done) * (q_target_min - current_alpha * next_log_prob)

        q1, q2 = self.critic(obs, action)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        if cfg.max_grad_norm is not None:
            self._clip_grad_norm(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        # ---- 2. Actor update ----
        features = self.actor_trunk(obs)
        new_action, log_prob = self.actor_head.sample(features)
        log_prob = log_prob.unsqueeze(-1)

        q1_new, q2_new = self.critic(obs, new_action)
        q_new_min = torch.min(q1_new, q2_new)

        actor_loss = (current_alpha * log_prob - q_new_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if cfg.max_grad_norm is not None:
            actor_params = list(self.actor_trunk.parameters()) + list(self.actor_head.parameters())
            self._clip_grad_norm(actor_params, cfg.max_grad_norm)
        self.actor_optimizer.step()

        # ---- 3. Alpha (temperature) update ----
        alpha_loss_val = 0.0
        if cfg.auto_alpha and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        # ---- 4. Soft update target networks ----
        self._soft_update(self.critic_target, self.critic, cfg.tau)

        # ---- Metrics ----
        entropy = -log_prob.mean().item()
        metrics = {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha,
            "entropy": entropy,
        }
        self._metrics.record_dict(metrics)
        self._total_steps += 1
        return metrics

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save agent checkpoint to disk.

        Parameters
        ----------
        path : str or Path
            Directory or file path for the checkpoint.
        """
        state_dicts = {
            "actor_trunk": self.actor_trunk.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        self._save_checkpoint(path, state_dicts)

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """Load agent checkpoint from disk.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        """
        payload = self._load_checkpoint(path)
        model = payload["model"]

        self.actor_trunk.load_state_dict(model["actor_trunk"])
        self.actor_head.load_state_dict(model["actor_head"])
        self.critic.load_state_dict(model["critic"])
        self.critic_target.load_state_dict(model["critic_target"])

        if "log_alpha" in model:
            self.log_alpha.data.copy_(model["log_alpha"].to(self._device))
