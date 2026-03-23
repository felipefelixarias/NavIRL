"""
Twin Delayed DDPG (TD3) Agent
==============================

Implementation of Twin Delayed Deep Deterministic Policy Gradient
(Fujimoto et al., 2018) for continuous action spaces in the NavIRL
pedestrian navigation framework.

TD3 addresses overestimation bias in actor-critic methods through three
key mechanisms:
* Clipped double Q-learning (twin critics with minimum operator).
* Delayed policy updates (actor updates less frequently than critics).
* Target policy smoothing (noise added to target actions).

References
----------
Fujimoto, S., Hoof, H., & Meger, D. (2018).
Addressing Function Approximation Error in Actor-Critic Methods.
*ICML 2018*.
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
from navirl.agents.networks import MLP, DeterministicPolicyHead, QValueHead

logger = logging.getLogger(__name__)


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
        Standard deviation of Gaussian noise added to actions during
        exploration (at act time).
    max_grad_norm : float or None
        If set, gradients are clipped to this maximum norm.
    """

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    batch_size: int = 256
    exploration_noise: float = 0.1
    max_grad_norm: Optional[float] = None


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG agent for continuous control.

    Parameters
    ----------
    config : TD3Config
        Agent hyperparameters.
    observation_space
        Environment observation space (must expose ``.shape``).
    action_space
        Environment action space (must expose ``.shape``, ``.low``, ``.high``).
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
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)

        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        # Action bounds for clipping
        self._action_low = torch.tensor(action_space.low, dtype=torch.float32, device=self._device)
        self._action_high = torch.tensor(action_space.high, dtype=torch.float32, device=self._device)

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
        self._modules.extend([
            self.actor_trunk, self.actor_head,
            self.actor_trunk_target, self.actor_head_target,
            self.q1, self.q2, self.q1_target, self.q2_target,
        ])

        # ---- Update counter for delayed policy updates ----
        self._update_count: int = 0

        self._log_module_summary("actor_trunk", self.actor_trunk)
        self._log_module_summary("actor_head", self.actor_head)
        self._log_module_summary("q1", self.q1)
        self._log_module_summary("q2", self.q2)

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
            If ``True``, return the deterministic policy output without
            exploration noise.

        Returns
        -------
        action : np.ndarray
            The chosen action, clipped to the action space bounds.
        info : dict
            Empty dict (TD3 does not produce auxiliary info at act time).
        """
        cfg: TD3Config = self._config  # type: ignore[assignment]

        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)

            features = self.actor_trunk(obs_t)
            action = self.actor_head(features)

            if not deterministic:
                noise = torch.randn_like(action) * cfg.exploration_noise
                action = action + noise

            action = action.clamp(self._action_low, self._action_high)

        return self._to_numpy(action.squeeze(0)), {}

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> Dict[str, float]:
        """Perform a single TD3 gradient step on a batch of transitions.

        The critic is updated every call. The actor and target networks are
        updated only every ``policy_delay`` calls.

        Parameters
        ----------
        batch
            A named-tuple or object with attributes ``obs``, ``action``,
            ``reward``, ``next_obs``, ``done`` (each a tensor or ndarray).

        Returns
        -------
        dict
            Scalar training metrics: ``q_loss`` (always present),
            ``actor_loss`` (present on actor update steps).
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

            noise = (torch.randn_like(next_action) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_action = (next_action + noise).clamp(self._action_low, self._action_high)

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

        metrics: Dict[str, float] = {"q_loss": q_loss.item()}

        # ---- 3. Delayed actor update ----
        if self._update_count % cfg.policy_delay == 0:
            features = self.actor_trunk(obs)
            actor_action = self.actor_head(features)
            actor_loss = -self.q1(obs, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if cfg.max_grad_norm is not None:
                actor_params = list(self.actor_trunk.parameters()) + list(self.actor_head.parameters())
                self._clip_grad_norm(actor_params, cfg.max_grad_norm)
            self.actor_optimizer.step()

            metrics["actor_loss"] = actor_loss.item()

            # ---- 4. Soft update target networks (also delayed) ----
            self._soft_update(self.q1_target, self.q1, cfg.tau)
            self._soft_update(self.q2_target, self.q2, cfg.tau)
            self._soft_update(self.actor_trunk_target, self.actor_trunk, cfg.tau)
            self._soft_update(self.actor_head_target, self.actor_head, cfg.tau)

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
            "actor_trunk_target": self.actor_trunk_target.state_dict(),
            "actor_head_target": self.actor_head_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "update_count": self._update_count,
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
        self.actor_trunk_target.load_state_dict(model["actor_trunk_target"])
        self.actor_head_target.load_state_dict(model["actor_head_target"])
        self.q1.load_state_dict(model["q1"])
        self.q2.load_state_dict(model["q2"])
        self.q1_target.load_state_dict(model["q1_target"])
        self.q2_target.load_state_dict(model["q2_target"])

        if "update_count" in model:
            self._update_count = model["update_count"]
