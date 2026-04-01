"""
AIRL (Adversarial Inverse Reinforcement Learning)
==================================================

Learns a disentangled reward function that generalises across environment
dynamics, alongside a policy trained via PPO with the learned reward.

The key insight is the reward network structure:
    D(s, a, s') = exp(f(s, a, s')) / (exp(f(s, a, s')) + pi(a|s))
where f(s, a, s') = g(s, a) + gamma * h(s') - h(s), and g is the
recoverable reward while h is a shaping potential.

Reference:
    Fu, Luo & Levine. "Learning Robust Rewards with Adversarial Inverse
    Reinforcement Learning", ICLR 2018.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.agents.base import BaseAgent, HyperParameters

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["AIRLConfig", "RewardNetwork", "AIRLAgent"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AIRLConfig(HyperParameters):
    """Hyperparameters for AIRL.

    Attributes:
        lr_discriminator: Learning rate for the reward / discriminator network.
        lr_policy: Learning rate for the PPO policy.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        disc_hidden_dims: Hidden layer sizes for reward and shaping networks.
        policy_hidden_dims: Hidden layer sizes for the policy MLP.
        disc_epochs: Discriminator update epochs per outer iteration.
        policy_epochs: PPO update epochs per outer iteration.
        batch_size: Mini-batch size.
        clip_eps: PPO clipping epsilon.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value-loss coefficient.
        max_grad_norm: Gradient clipping threshold.
        action_type: ``"continuous"`` or ``"discrete"``.
        state_only: If ``True``, the reward network ignores actions (g(s)
            instead of g(s, a)).
    """

    lr_discriminator: float = 3e-4
    lr_policy: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    disc_hidden_dims: tuple[int, ...] = (256, 256)
    policy_hidden_dims: tuple[int, ...] = (256, 256)
    disc_epochs: int = 5
    policy_epochs: int = 10
    batch_size: int = 64
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    action_type: str = "continuous"
    state_only: bool = False


# ---------------------------------------------------------------------------
# Reward network: g(s, a) + gamma * h(s') - h(s)
# ---------------------------------------------------------------------------


class RewardNetwork(nn.Module):
    """Disentangled reward network for AIRL.

    The output is ``f(s, a, s') = g(s, a) + gamma * h(s') - h(s)`` where *g*
    is the recoverable reward and *h* is a state-dependent shaping potential.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Action dimensionality (used only when ``state_only=False``).
    hidden_dims : Sequence[int]
        Hidden layer sizes for both *g* and *h* networks.
    gamma : float
        Discount factor (used in the shaping term).
    state_only : bool
        If ``True``, the reward network uses ``g(s)`` instead of ``g(s, a)``.
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 0,
        hidden_dims: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        state_only: bool = False,
        *,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        if obs_dim is None:
            obs_dim = state_dim
        if obs_dim is None:
            raise TypeError("RewardNetwork requires obs_dim or state_dim.")
        self.gamma = gamma
        self.state_only = state_only

        # g network (reward)
        g_input_dim = obs_dim if state_only else obs_dim + action_dim
        g_layers: list[nn.Module] = []
        prev = g_input_dim
        for h in hidden_dims:
            g_layers.append(nn.Linear(prev, h))
            g_layers.append(nn.Tanh())
            prev = h
        g_layers.append(nn.Linear(prev, 1))
        self.g_net = nn.Sequential(*g_layers)

        # h network (shaping potential)
        h_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            h_layers.append(nn.Linear(prev, h))
            h_layers.append(nn.Tanh())
            prev = h
        h_layers.append(nn.Linear(prev, 1))
        self.h_net = nn.Sequential(*h_layers)

    def g(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the reward component g(s, a) (or g(s) if state_only).

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch.
        actions : torch.Tensor, optional
            Action batch (ignored when ``state_only=True``).

        Returns
        -------
        torch.Tensor
            Reward values ``(B, 1)``.
        """
        if self.state_only:
            return self.g_net(obs)
        assert actions is not None
        return self.g_net(torch.cat([obs, actions], dim=-1))

    def h(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the shaping potential h(s).

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch.

        Returns
        -------
        torch.Tensor
            Potential values ``(B, 1)``.
        """
        return self.h_net(obs)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute AIRL reward components.

        Parameters
        ----------
        obs : torch.Tensor
            Current observations ``(B, obs_dim)``.
        actions : torch.Tensor
            Actions ``(B, action_dim)``.
        next_obs : torch.Tensor
            Next observations ``(B, obs_dim)``.
        dones : torch.Tensor, optional
            Episode termination flags ``(B, 1)`` or ``(B,)``.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            When ``dones`` is provided, returns f-values ``(B, 1)``.
            Otherwise returns ``(reward, shaping)`` for compatibility.
        """
        reward = self.g(obs, actions)
        h_s = self.h(obs)
        h_sp = self.h(next_obs)
        if dones is None:
            shaping = self.gamma * h_sp - h_s
            return reward, shaping
        dones = dones.reshape(-1, 1).float()
        shaping = self.gamma * h_sp * (1.0 - dones) - h_s
        return reward + shaping

    def predict_reward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract the learned reward g(s, a) for use in downstream tasks.

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch.
        actions : torch.Tensor, optional
            Action batch.

        Returns
        -------
        torch.Tensor
            Reward ``(B, 1)``.
        """
        with torch.no_grad():
            return self.g(obs, actions)


# ---------------------------------------------------------------------------
# Simple PPO actor-critic (shared with GAIL structure)
# ---------------------------------------------------------------------------


class _AIRLPolicyValue(nn.Module):
    """Actor-critic network for AIRL's PPO inner loop."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        action_type: str = "continuous",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev, h_dim))
            layers.append(nn.Tanh())
            prev = h_dim
        self.features = nn.Sequential(*layers)

        if action_type == "continuous":
            self.action_mean = nn.Linear(prev, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_logits = nn.Linear(prev, action_dim)

        self.value_head = nn.Linear(prev, 1)
        self.action_type = action_type

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(obs)
        value = self.value_head(feat)
        if self.action_type == "continuous":
            return self.action_mean(feat), value
        return self.action_logits(feat), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log-probs, values, and entropy for the given actions."""
        feat = self.features(obs)
        value = self.value_head(feat).squeeze(-1)

        if self.action_type == "continuous":
            mean = self.action_mean(feat)
            std = self.action_log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.action_logits(feat)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions.squeeze(-1))
            entropy = dist.entropy()

        return log_probs, value, entropy

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return log pi(a|s) for AIRL discriminator."""
        feat = self.features(obs)
        if self.action_type == "continuous":
            mean = self.action_mean(feat)
            std = self.action_log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            return dist.log_prob(actions).sum(dim=-1, keepdim=True)
        else:
            logits = self.action_logits(feat)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)


# ---------------------------------------------------------------------------
# AIRLAgent
# ---------------------------------------------------------------------------


class AIRLAgent(BaseAgent):
    """Adversarial Inverse Reinforcement Learning agent.

    Learns a reward function with a disentangled structure that generalises
    across environment dynamics, alongside a PPO policy optimised with the
    learned reward.

    Parameters
    ----------
    config : AIRLConfig
        AIRL hyperparameters.
    observation_space :
        Environment observation space.
    action_space :
        Environment action space.
    device : str or torch.device
        Compute device.
    seed : int, optional
        Random seed.
    metrics_callback : callable, optional
        Metrics callback.
    """

    def __init__(
        self,
        config: AIRLConfig,
        observation_space: Any,
        action_space: Any,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        metrics_callback: Any = None,
    ) -> None:
        super().__init__(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            seed=seed,
            metrics_callback=metrics_callback,
        )
        if not _TORCH_AVAILABLE:
            raise RuntimeError("AIRLAgent requires PyTorch.")

        self._obs_dim = int(np.prod(observation_space.shape))
        if config.action_type == "discrete":
            self._action_dim = int(action_space.n)
        else:
            self._action_dim = int(np.prod(action_space.shape))

        # Reward network (discriminator)
        self._reward_net = RewardNetwork(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_dims=config.disc_hidden_dims,
            gamma=config.gamma,
            state_only=config.state_only,
        ).to(self._device)
        self._modules.append(self._reward_net)

        self._reward_optimizer = torch.optim.Adam(
            self._reward_net.parameters(), lr=config.lr_discriminator
        )
        self._optimizers["reward"] = self._reward_optimizer

        # Policy + value (PPO)
        self._policy_value = _AIRLPolicyValue(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_dims=config.policy_hidden_dims,
            action_type=config.action_type,
        ).to(self._device)
        self._modules.append(self._policy_value)

        self._policy_optimizer = torch.optim.Adam(
            self._policy_value.parameters(), lr=config.lr_policy
        )
        self._optimizers["policy"] = self._policy_optimizer

        self._log_module_summary("reward_net", self._reward_net)
        self._log_module_summary("policy_value", self._policy_value)

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------

    def update_discriminator(
        self,
        expert_batch: dict[str, np.ndarray],
        policy_batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Update the AIRL discriminator (reward network).

        The AIRL discriminator is:
            D(s, a, s') = sigmoid(f(s, a, s') - log pi(a|s))
        where f = g(s, a) + gamma * h(s') - h(s).

        Expert transitions are labelled 1 and policy transitions 0.

        Parameters
        ----------
        expert_batch : dict
            Must contain ``"obs"``, ``"actions"``, ``"next_obs"``, ``"dones"``.
        policy_batch : dict
            Must contain ``"obs"``, ``"actions"``, ``"next_obs"``, ``"dones"``.

        Returns
        -------
        dict
            Discriminator metrics.
        """
        cfg: AIRLConfig = self._config  # type: ignore[assignment]

        def _prep(batch: dict[str, np.ndarray]) -> tuple:
            o = self._to_tensor(batch["obs"], torch.float32).reshape(-1, self._obs_dim)
            a = self._to_tensor(batch["actions"], torch.float32)
            no = self._to_tensor(batch["next_obs"], torch.float32).reshape(-1, self._obs_dim)
            d = self._to_tensor(batch["dones"], torch.float32)
            return o, a, no, d

        e_obs, e_act, e_next, e_done = _prep(expert_batch)
        p_obs, p_act, p_next, p_done = _prep(policy_batch)

        total_loss = 0.0
        n_updates = 0

        self._reward_net.train()
        for _ in range(cfg.disc_epochs):
            # f-values
            f_expert = self._reward_net(e_obs, e_act, e_next, e_done)
            f_policy = self._reward_net(p_obs, p_act, p_next, p_done)

            # log pi(a|s)
            with torch.no_grad():
                log_pi_expert = self._policy_value.get_log_prob(e_obs, e_act)
                log_pi_policy = self._policy_value.get_log_prob(p_obs, p_act)

            # D = sigmoid(f - log_pi)
            expert_logits = f_expert - log_pi_expert
            policy_logits = f_policy - log_pi_policy

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            )
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits)
            )
            loss = expert_loss + policy_loss

            self._reward_optimizer.zero_grad()
            loss.backward()
            self._reward_optimizer.step()

            total_loss += loss.item()
            n_updates += 1

        # Accuracy
        with torch.no_grad():
            e_pred = torch.sigmoid(
                self._reward_net(e_obs, e_act, e_next, e_done)
                - self._policy_value.get_log_prob(e_obs, e_act)
            )
            p_pred = torch.sigmoid(
                self._reward_net(p_obs, p_act, p_next, p_done)
                - self._policy_value.get_log_prob(p_obs, p_act)
            )
            acc = 0.5 * ((e_pred > 0.5).float().mean() + (p_pred <= 0.5).float().mean())

        metrics = {
            "airl/disc_loss": total_loss / max(n_updates, 1),
            "airl/disc_accuracy": acc.item(),
        }
        self._metrics.record_dict(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Policy update (PPO)
    # ------------------------------------------------------------------

    def update_policy(self, rollout_buffer: Any) -> dict[str, float]:
        """Update the policy via PPO using the AIRL-learned reward.

        Parameters
        ----------
        rollout_buffer :
            Rollout buffer or dict with PPO rollout data.

        Returns
        -------
        dict
            PPO update metrics.
        """
        cfg: AIRLConfig = self._config  # type: ignore[assignment]

        if isinstance(rollout_buffer, dict):
            obs_t = self._to_tensor(rollout_buffer["obs"], torch.float32)
            act_t = self._to_tensor(rollout_buffer["actions"], torch.float32)
            old_lp = self._to_tensor(rollout_buffer["log_probs"], torch.float32)
            adv = self._to_tensor(rollout_buffer["advantages"], torch.float32)
            ret = self._to_tensor(rollout_buffer["returns"], torch.float32)
        else:
            n = rollout_buffer.buffer_size * rollout_buffer.n_envs
            obs_t = self._to_tensor(rollout_buffer.observations.reshape(n, -1), torch.float32)
            act_t = self._to_tensor(rollout_buffer.actions.reshape(n, -1), torch.float32)
            old_lp = self._to_tensor(rollout_buffer.log_probs.reshape(n), torch.float32)
            adv = self._to_tensor(rollout_buffer.advantages.reshape(n), torch.float32)
            ret = self._to_tensor(rollout_buffer.returns.reshape(n), torch.float32)

        obs_t = obs_t.reshape(-1, self._obs_dim)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dataset = TensorDataset(obs_t, act_t, old_lp, adv, ret)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        total_pl = 0.0
        total_vl = 0.0
        total_ent = 0.0
        n_updates = 0

        self._policy_value.train()
        for _ in range(cfg.policy_epochs):
            for b_obs, b_act, b_old_lp, b_adv, b_ret in loader:
                lp, val, ent = self._policy_value.evaluate_actions(b_obs, b_act)
                ratio = (lp - b_old_lp).exp()
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * b_adv
                pl = -torch.min(s1, s2).mean()
                vl = F.mse_loss(val, b_ret)
                el = -ent.mean()

                loss = pl + cfg.value_coef * vl + cfg.entropy_coef * el

                self._policy_optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self._policy_value.parameters(), cfg.max_grad_norm)
                self._policy_optimizer.step()

                total_pl += pl.item()
                total_vl += vl.item()
                total_ent += ent.mean().item()
                n_updates += 1

        metrics = {
            "airl/policy_loss": total_pl / max(n_updates, 1),
            "airl/value_loss": total_vl / max(n_updates, 1),
            "airl/entropy": total_ent / max(n_updates, 1),
        }
        self._metrics.record_dict(metrics)
        return metrics

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> dict[str, float]:
        """Combined discriminator + policy update.

        Parameters
        ----------
        batch : dict
            Must contain expert batch keys (``"expert_obs"``,
            ``"expert_actions"``, ``"expert_next_obs"``, ``"expert_dones"``),
            policy batch keys (``"policy_obs"``, ``"policy_actions"``,
            ``"policy_next_obs"``, ``"policy_dones"``), and PPO rollout keys.

        Returns
        -------
        dict
            Combined metrics.
        """
        disc_metrics = self.update_discriminator(
            expert_batch={
                "obs": batch["expert_obs"],
                "actions": batch["expert_actions"],
                "next_obs": batch["expert_next_obs"],
                "dones": batch["expert_dones"],
            },
            policy_batch={
                "obs": batch["policy_obs"],
                "actions": batch["policy_actions"],
                "next_obs": batch["policy_next_obs"],
                "dones": batch["policy_dones"],
            },
        )
        policy_metrics = self.update_policy(
            {
                "obs": batch["obs"],
                "actions": batch["actions"],
                "log_probs": batch["log_probs"],
                "advantages": batch["advantages"],
                "returns": batch["returns"],
            }
        )
        self._total_steps += 1
        return {**disc_metrics, **policy_metrics}

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select an action given the current observation.

        Parameters
        ----------
        observation : np.ndarray
            Current observation.
        deterministic : bool
            If ``True``, use mean/argmax action.

        Returns
        -------
        action : np.ndarray
            Chosen action.
        info : dict
            Contains ``"log_prob"`` and ``"value"``.
        """
        self._policy_value.eval()
        obs_t = self._to_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
        )
        cfg: AIRLConfig = self._config  # type: ignore[assignment]

        with torch.no_grad():
            output, value = self._policy_value(obs_t)
            if cfg.action_type == "continuous":
                if deterministic:
                    action_t = output
                    log_prob = torch.zeros(1, device=self._device)
                else:
                    std = self._policy_value.action_log_std.exp()
                    dist = torch.distributions.Normal(output, std)
                    action_t = dist.sample()
                    log_prob = dist.log_prob(action_t).sum(dim=-1)
            else:
                if deterministic:
                    action_t = output.argmax(dim=-1, keepdim=True)
                    log_prob = torch.zeros(1, device=self._device)
                else:
                    dist = torch.distributions.Categorical(logits=output)
                    action_t = dist.sample().unsqueeze(-1)
                    log_prob = dist.log_prob(action_t.squeeze(-1))

        action = action_t.cpu().numpy().flatten()
        info = {
            "log_prob": log_prob.cpu().numpy().item(),
            "value": value.cpu().numpy().item(),
        }
        if self._training:
            self._policy_value.train()
        return action, info

    def compute_reward(
        self,
        obs: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the learned reward g(s, a) for a batch.

        Parameters
        ----------
        obs : np.ndarray
            Observation batch.
        actions : np.ndarray, optional
            Action batch (not needed when ``state_only=True``).

        Returns
        -------
        np.ndarray
            Learned reward values.
        """
        obs_t = self._to_tensor(obs, torch.float32).reshape(-1, self._obs_dim)
        act_t = self._to_tensor(actions, torch.float32) if actions is not None else None
        reward_t = self._reward_net.predict_reward(obs_t, act_t)
        return self._to_numpy(reward_t).flatten()

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the AIRL agent to disk."""
        self._save_checkpoint(
            path,
            state_dicts={
                "reward_net": self._reward_net.state_dict(),
                "policy_value": self._policy_value.state_dict(),
            },
        )

    def load(self, path: str | pathlib.Path) -> None:
        """Restore the AIRL agent from a checkpoint."""
        payload = self._load_checkpoint(path)
        self._reward_net.load_state_dict(payload["model"]["reward_net"])
        self._policy_value.load_state_dict(payload["model"]["policy_value"])
