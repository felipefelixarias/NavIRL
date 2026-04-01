"""
Soft Actor-Critic (SAC) Agent
=============================

Implementation of Soft Actor-Critic (Haarnoja et al., 2018) for continuous
action spaces in the NavIRL pedestrian navigation framework.

SAC is an off-policy maximum-entropy reinforcement learning algorithm that
optimises a stochastic policy by simultaneously maximising expected return
and entropy, encouraging exploration and robustness to model errors.

Key features
------------
* **Squashed Gaussian policy** with reparameterised sampling.
* **Automatic entropy tuning** -- the temperature parameter alpha is
  learned to maintain a target entropy.
* **Twin Q-networks** (clipped double-Q) to mitigate positive bias.
* **Soft target updates** via Polyak averaging.
* **Distributional critic variant** -- optional categorical distributional
  representation of Q-values (Barth-Maron et al., 2018).
* **Replay buffer integration** -- accepts batches from any buffer
  implementing the standard transition interface.
* **Observation normalization** via running statistics.
* **Reward scaling** for environment-specific tuning.
* **Gradient clipping** with configurable max norm.

References
----------
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor. *ICML 2018*.

Haarnoja, T., Zhou, A., Hartikainen, K., et al. (2018).
Soft Actor-Critic Algorithms and Applications. *arXiv:1812.05905*.
"""

from __future__ import annotations

import copy
import logging
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from navirl.agents.base import BaseAgent, HyperParameters, RunningMeanStd
from navirl.agents.networks import MLP, SquashedGaussianHead, TwinQHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributional Q-network helper
# ---------------------------------------------------------------------------


class DistributionalTwinQ(nn.Module):
    """Twin distributional Q-networks using categorical (C51) representation.

    Each Q-network outputs a categorical distribution over a fixed set of
    atoms rather than a single scalar Q-value.  The expected Q-value is
    computed as the dot product of atom values and their probabilities.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state (observation) vector.
    action_dim : int
        Dimensionality of the action vector.
    hidden_dims : sequence of int
        Hidden layer widths.
    n_atoms : int
        Number of atoms in the categorical distribution.
    v_min : float
        Minimum support value.
    v_max : float
        Maximum support value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms: z_i = v_min + i * delta
        self.register_buffer(
            "atoms",
            torch.linspace(v_min, v_max, n_atoms),
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        input_dim = state_dim + action_dim
        layers1: list[nn.Module] = []
        layers2: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers1.extend([nn.Linear(prev, h), nn.ReLU()])
            layers2.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers1.append(nn.Linear(prev, n_atoms))
        layers2.append(nn.Linear(prev, n_atoms))

        self.net1 = nn.Sequential(*layers1)
        self.net2 = nn.Sequential(*layers2)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-value distributions for both networks.

        Parameters
        ----------
        state : torch.Tensor ``(B, state_dim)``
        action : torch.Tensor ``(B, action_dim)``

        Returns
        -------
        logits1, logits2 : torch.Tensor ``(B, n_atoms)``
            Raw logits (before softmax) for each network.
        """
        sa = torch.cat([state, action], dim=-1)
        return self.net1(sa), self.net2(sa)

    def q_values(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scalar Q-values as expectations over the distributions.

        Parameters
        ----------
        state : torch.Tensor ``(B, state_dim)``
        action : torch.Tensor ``(B, action_dim)``

        Returns
        -------
        q1, q2 : torch.Tensor ``(B, 1)``
        """
        logits1, logits2 = self.forward(state, action)
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        q1 = (probs1 * self.atoms).sum(dim=-1, keepdim=True)
        q2 = (probs2 * self.atoms).sum(dim=-1, keepdim=True)
        return q1, q2


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
        Initial entropy temperature.  Used as a fixed value when
        ``auto_alpha`` is ``False``.
    auto_alpha : bool
        If ``True``, the entropy temperature is learned automatically.
    target_entropy : float or None
        Target entropy for automatic tuning.  Defaults to
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
    normalize_observations : bool
        Whether to normalize observations using running mean/std.
    observation_clip : float
        Clipping range for normalized observations.
    distributional : bool
        If ``True``, use distributional (C51) critics instead of scalar.
    n_atoms : int
        Number of atoms for distributional critics.
    v_min : float
        Minimum support value for distributional critics.
    v_max : float
        Maximum support value for distributional critics.
    updates_per_step : int
        Number of gradient updates per environment step (UTD ratio).
    warmup_steps : int
        Number of random-action steps before training begins.
    actor_update_freq : int
        Actor is updated every this many critic updates.
    """

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    batch_size: int = 256
    max_grad_norm: float | None = None
    reward_scale: float = 1.0
    normalize_observations: bool = False
    observation_clip: float = 10.0
    distributional: bool = False
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    updates_per_step: int = 1
    warmup_steps: int = 1000
    actor_update_freq: int = 1


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent for continuous control.

    Implements SAC with automatic entropy tuning, twin Q-networks, and
    optional distributional critics.  Designed for off-policy training
    with a replay buffer.

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
        device: str | torch.device = "cpu",
        seed: int | None = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)

        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))

        # ---- Observation normalization ----
        self._obs_rms: RunningMeanStd | None = None
        if config.normalize_observations:
            self._obs_rms = RunningMeanStd(shape=observation_space.shape)

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
        if config.distributional:
            self.critic = DistributionalTwinQ(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=config.hidden_dims,
                n_atoms=config.n_atoms,
                v_min=config.v_min,
                v_max=config.v_max,
            ).to(self._device)
            self.critic_target = copy.deepcopy(self.critic).to(self._device)
        else:
            self.critic = TwinQHead(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=config.hidden_dims,
            ).to(self._device)
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
                np.log(config.alpha), dtype=torch.float32,
                device=self._device, requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config.lr_alpha,
            )
            self._optimizers["alpha"] = self.alpha_optimizer
        else:
            self.target_entropy = None
            self.log_alpha = torch.tensor(
                np.log(config.alpha), dtype=torch.float32,
                device=self._device,
            )
            self.alpha_optimizer = None

        self._alpha = config.alpha

        # ---- Optimizers ----
        actor_params = list(self.actor_trunk.parameters()) + list(self.actor_head.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic,
        )

        self._optimizers["actor"] = self.actor_optimizer
        self._optimizers["critic"] = self.critic_optimizer

        # ---- Register modules for train/eval toggling ----
        self._modules.extend([
            self.actor_trunk, self.actor_head,
            self.critic, self.critic_target,
        ])

        # ---- Update counter for actor update frequency ----
        self._critic_update_count: int = 0

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
        cfg: SACConfig = self._config  # type: ignore[assignment]
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

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation.
        deterministic : bool
            If ``True``, return the mean action (no sampling).

        Returns
        -------
        action : np.ndarray
            The chosen action, scaled to ``(-1, 1)`` via tanh squashing.
        info : dict
            Contains ``"log_prob"`` when sampling stochastically.
        """
        observation = self._normalize_obs(observation)

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
    # Distributional critic helpers
    # ------------------------------------------------------------------

    def _compute_distributional_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        current_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute categorical cross-entropy loss for distributional critics.

        Uses the projected Bellman update to compute target distributions
        and cross-entropy loss against current distributions.

        Parameters
        ----------
        obs, action, reward, next_obs, done : torch.Tensor
            Batch of transition data.
        current_alpha : torch.Tensor
            Current entropy temperature.

        Returns
        -------
        torch.Tensor
            Scalar loss for both critic networks combined.
        """
        cfg: SACConfig = self._config  # type: ignore[assignment]
        critic: DistributionalTwinQ = self.critic  # type: ignore[assignment]
        critic_tgt: DistributionalTwinQ = self.critic_target  # type: ignore[assignment]

        with torch.no_grad():
            next_features = self.actor_trunk(next_obs)
            next_action, next_log_prob = self.actor_head.sample(next_features)
            next_log_prob = next_log_prob.unsqueeze(-1)

            # Target distributions
            tgt_logits1, tgt_logits2 = critic_tgt(next_obs, next_action)
            tgt_probs1 = F.softmax(tgt_logits1, dim=-1)
            tgt_probs2 = F.softmax(tgt_logits2, dim=-1)

            # Use minimum Q-value network's distribution
            tgt_q1 = (tgt_probs1 * critic_tgt.atoms).sum(dim=-1, keepdim=True)
            tgt_q2 = (tgt_probs2 * critic_tgt.atoms).sum(dim=-1, keepdim=True)
            use_q1 = (tgt_q1 <= tgt_q2).float()
            tgt_probs = use_q1 * tgt_probs1 + (1.0 - use_q1) * tgt_probs2

            # Bellman projection of atoms
            # T_z = r + gamma * (1 - done) * z - alpha * log_prob
            atoms = critic_tgt.atoms.unsqueeze(0)  # (1, n_atoms)
            tz = reward + cfg.gamma * (1.0 - done) * (atoms - current_alpha * next_log_prob)
            tz = tz.clamp(cfg.v_min, cfg.v_max)

            # Projection onto fixed support
            b = (tz - cfg.v_min) / critic.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            lower = lower.clamp(0, cfg.n_atoms - 1)
            upper = upper.clamp(0, cfg.n_atoms - 1)

            target_dist = torch.zeros_like(tgt_probs)
            # Distribute probability mass
            offset = torch.arange(obs.size(0), device=obs.device).unsqueeze(1) * cfg.n_atoms
            target_dist.view(-1).index_add_(
                0, (lower + offset).view(-1),
                (tgt_probs * (upper.float() - b)).view(-1),
            )
            target_dist.view(-1).index_add_(
                0, (upper + offset).view(-1),
                (tgt_probs * (b - lower.float())).view(-1),
            )

        # Current distributions
        logits1, logits2 = critic(obs, action)
        log_probs1 = F.log_softmax(logits1, dim=-1)
        log_probs2 = F.log_softmax(logits2, dim=-1)

        loss1 = -(target_dist * log_probs1).sum(dim=-1).mean()
        loss2 = -(target_dist * log_probs2).sum(dim=-1).mean()

        return loss1 + loss2

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> dict[str, float]:
        """Perform a single SAC gradient step on a batch of transitions.

        Sequentially updates: (1) critic, (2) actor (every
        ``actor_update_freq`` critic steps), (3) entropy temperature,
        (4) target networks.

        Parameters
        ----------
        batch
            A named-tuple or object with attributes ``obs``, ``action``,
            ``reward``, ``next_obs``, ``done`` (each a tensor or ndarray).

        Returns
        -------
        dict
            Scalar training metrics: ``q_loss``, ``actor_loss``,
            ``alpha_loss``, ``alpha``, ``entropy``, ``q1_mean``,
            ``q2_mean``.
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
        if cfg.distributional:
            q_loss = self._compute_distributional_critic_loss(
                obs, action, reward, next_obs, done, current_alpha,
            )
            # For logging, compute scalar Q-values
            with torch.no_grad():
                q1_val, q2_val = self.critic.q_values(obs, action)
        else:
            with torch.no_grad():
                next_features = self.actor_trunk(next_obs)
                next_action, next_log_prob = self.actor_head.sample(next_features)
                next_log_prob = next_log_prob.unsqueeze(-1)

                q1_target, q2_target = self.critic_target(next_obs, next_action)
                q_target_min = torch.min(q1_target, q2_target)
                y = reward + cfg.gamma * (1.0 - done) * (
                    q_target_min - current_alpha * next_log_prob
                )

            q1, q2 = self.critic(obs, action)
            q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            q1_val, q2_val = q1, q2

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        if cfg.max_grad_norm is not None:
            self._clip_grad_norm(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        self._critic_update_count += 1

        # ---- 2. Actor update (possibly delayed) ----
        actor_loss_val = 0.0
        entropy_val = 0.0
        alpha_loss_val = 0.0

        if self._critic_update_count % cfg.actor_update_freq == 0:
            features = self.actor_trunk(obs)
            new_action, log_prob = self.actor_head.sample(features)
            log_prob = log_prob.unsqueeze(-1)

            if cfg.distributional:
                q1_new, q2_new = self.critic.q_values(obs, new_action)
            else:
                q1_new, q2_new = self.critic(obs, new_action)
            q_new_min = torch.min(q1_new, q2_new)

            actor_loss = (current_alpha * log_prob - q_new_min).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if cfg.max_grad_norm is not None:
                actor_params = (
                    list(self.actor_trunk.parameters())
                    + list(self.actor_head.parameters())
                )
                self._clip_grad_norm(actor_params, cfg.max_grad_norm)
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()
            entropy_val = -log_prob.mean().item()

            # ---- 3. Alpha (temperature) update ----
            if cfg.auto_alpha and self.alpha_optimizer is not None:
                alpha_loss = -(
                    self.log_alpha * (log_prob.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_val = alpha_loss.item()

        # ---- 4. Soft update target networks ----
        self._soft_update(self.critic_target, self.critic, cfg.tau)

        # ---- Metrics ----
        metrics = {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss_val,
            "alpha_loss": alpha_loss_val,
            "alpha": self.alpha,
            "entropy": entropy_val,
            "q1_mean": float(q1_val.mean().item()),
            "q2_mean": float(q2_val.mean().item()),
        }
        self._metrics.record_dict(metrics)
        self._total_steps += 1
        return metrics

    # ------------------------------------------------------------------
    # Batch update for higher UTD ratios
    # ------------------------------------------------------------------

    def update_multi(self, replay_buffer: Any, batch_size: int | None = None) -> dict[str, float]:
        """Perform multiple gradient updates per environment step.

        Useful for high Update-To-Data (UTD) ratio training, where the
        agent performs several gradient steps on replay data for each
        new transition collected.

        Parameters
        ----------
        replay_buffer
            A replay buffer supporting ``.sample(batch_size)`` that
            returns transition batches.
        batch_size : int or None
            Override batch size (defaults to ``config.batch_size``).

        Returns
        -------
        dict
            Averaged metrics across all updates in this call.
        """
        cfg: SACConfig = self._config  # type: ignore[assignment]
        bs = batch_size or cfg.batch_size
        all_metrics: list[dict[str, float]] = []

        for _ in range(cfg.updates_per_step):
            batch = replay_buffer.sample(bs)
            m = self.update(batch)
            all_metrics.append(m)

        # Average metrics
        avg: dict[str, float] = {}
        if all_metrics:
            for key in all_metrics[0]:
                avg[key] = float(np.mean([m[key] for m in all_metrics]))
        return avg

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Save agent checkpoint to disk.

        Persists actor, critic, target critic, entropy temperature,
        and normalization statistics.

        Parameters
        ----------
        path : str or Path
            Directory or file path for the checkpoint.
        """
        state_dicts: dict[str, Any] = {
            "actor_trunk": self.actor_trunk.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "critic_update_count": self._critic_update_count,
        }
        if self._obs_rms is not None:
            state_dicts["obs_rms"] = self._obs_rms.state_dict()
        self._save_checkpoint(path, state_dicts)

    def load(self, path: str | pathlib.Path) -> None:
        """Load agent checkpoint from disk.

        Restores actor, critic, target critic, entropy temperature,
        and normalization statistics.

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
        if "critic_update_count" in model:
            self._critic_update_count = model["critic_update_count"]
        if "obs_rms" in model and self._obs_rms is not None:
            self._obs_rms.load_state_dict(model["obs_rms"])

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    def get_diagnostics(self, batch: Any) -> dict[str, float]:
        """Compute diagnostic statistics on a batch without updating.

        Parameters
        ----------
        batch
            Transition batch with ``obs``, ``action`` attributes.

        Returns
        -------
        dict
            Diagnostic metrics including Q-value statistics and entropy.
        """
        with torch.no_grad():
            obs = self._to_tensor(batch.obs, dtype=torch.float32)
            action = self._to_tensor(batch.action, dtype=torch.float32)

            if hasattr(self.critic, "q_values"):
                q1, q2 = self.critic.q_values(obs, action)
            else:
                q1, q2 = self.critic(obs, action)

            features = self.actor_trunk(obs)
            _, log_prob = self.actor_head.sample(features)

        return {
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "q1_std": float(q1.std().item()),
            "q2_std": float(q2.std().item()),
            "policy_entropy": float(-log_prob.mean().item()),
            "alpha": self.alpha,
        }
