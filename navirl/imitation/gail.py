"""
GAIL (Generative Adversarial Imitation Learning)
=================================================

Learns a policy by training a discriminator to distinguish between expert and
policy-generated state-action pairs, then using the discriminator's output as
a reward signal for on-policy reinforcement learning (PPO).

Reference:
    Ho & Ermon. "Generative Adversarial Imitation Learning", NeurIPS 2016.
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

__all__ = ["GAILConfig", "Discriminator", "GAILAgent"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GAILConfig(HyperParameters):
    """Hyperparameters for GAIL.

    Attributes:
        lr_discriminator: Learning rate for the discriminator.
        lr_policy: Learning rate for the PPO policy.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        disc_hidden_dims: Hidden layer sizes for the discriminator MLP.
        policy_hidden_dims: Hidden layer sizes for the policy MLP.
        disc_epochs: Discriminator update epochs per outer iteration.
        policy_epochs: PPO update epochs per outer iteration.
        batch_size: Mini-batch size for both discriminator and policy updates.
        clip_eps: PPO clipping parameter.
        entropy_coef: Entropy bonus coefficient for PPO.
        value_coef: Value-loss coefficient for PPO.
        max_grad_norm: Maximum gradient norm for clipping.
        gradient_penalty_coef: WGAN-GP gradient penalty coefficient.
            Set to ``0.0`` to disable.
        action_type: ``"continuous"`` or ``"discrete"``.
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
    gradient_penalty_coef: float = 10.0
    action_type: str = "continuous"


# ---------------------------------------------------------------------------
# Discriminator network
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """MLP discriminator that classifies (state, action) pairs as expert or
    policy-generated.

    Outputs a scalar logit; the probability of being expert is obtained via
    sigmoid.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality.
    action_dim : int
        Action dimensionality.
    hidden_dims : Sequence[int]
        Hidden-layer sizes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = obs_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits.

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch ``(B, obs_dim)``.
        actions : torch.Tensor
            Action batch ``(B, action_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, 1)``.
        """
        sa = torch.cat([obs, actions], dim=-1)
        return self.net(sa)

    def predict_reward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute GAIL reward: -log(1 - D(s, a)).

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch.
        actions : torch.Tensor
            Action batch.

        Returns
        -------
        torch.Tensor
            Reward of shape ``(B, 1)``.
        """
        with torch.no_grad():
            logits = self.forward(obs, actions)
            # Reward = -log(1 - sigmoid(logit)) to encourage fooling the disc.
            reward = -F.logsigmoid(-logits)
        return reward


# ---------------------------------------------------------------------------
# Simple policy + value network for PPO
# ---------------------------------------------------------------------------


class _PolicyValueNet(nn.Module):
    """Shared-backbone actor-critic network used by GAIL's PPO inner loop."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        action_type: str = "continuous",
    ) -> None:
        super().__init__()
        # Shared feature extractor
        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        self.features = nn.Sequential(*layers)

        # Policy head
        if action_type == "continuous":
            self.action_mean = nn.Linear(prev_dim, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_logits = nn.Linear(prev_dim, action_dim)

        # Value head
        self.value_head = nn.Linear(prev_dim, 1)

        self.action_type = action_type

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy distribution parameters and value estimate."""
        feat = self.features(obs)
        value = self.value_head(feat)
        if self.action_type == "continuous":
            mean = self.action_mean(feat)
            return mean, value
        else:
            logits = self.action_logits(feat)
            return logits, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions: return log-probs, values, and entropy.

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch.
        actions : torch.Tensor
            Action batch.

        Returns
        -------
        log_probs : torch.Tensor
        values : torch.Tensor
        entropy : torch.Tensor
        """
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


# ---------------------------------------------------------------------------
# GAILAgent
# ---------------------------------------------------------------------------


class GAILAgent(BaseAgent):
    """Generative Adversarial Imitation Learning agent.

    Trains a discriminator to distinguish expert vs. policy trajectories and
    uses the discriminator's output as a reward signal for a PPO-based policy
    optimiser.

    Parameters
    ----------
    config : GAILConfig
        GAIL hyperparameters.
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
        config: GAILConfig,
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
            raise RuntimeError("GAILAgent requires PyTorch.")

        self._obs_dim = int(np.prod(observation_space.shape))
        if config.action_type == "discrete":
            self._action_dim = int(action_space.n)
        else:
            self._action_dim = int(np.prod(action_space.shape))

        # Discriminator
        self._discriminator = Discriminator(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_dims=config.disc_hidden_dims,
        ).to(self._device)
        self._modules.append(self._discriminator)

        self._disc_optimizer = torch.optim.Adam(
            self._discriminator.parameters(), lr=config.lr_discriminator
        )
        self._optimizers["discriminator"] = self._disc_optimizer

        # Policy + value (PPO)
        self._policy_value = _PolicyValueNet(
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

        self._log_module_summary("discriminator", self._discriminator)
        self._log_module_summary("policy_value", self._policy_value)

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------

    def update_discriminator(
        self,
        expert_batch: dict[str, np.ndarray],
        policy_batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Update the discriminator on expert and policy data.

        Parameters
        ----------
        expert_batch : dict
            Must contain ``"obs"`` and ``"actions"`` arrays.
        policy_batch : dict
            Must contain ``"obs"`` and ``"actions"`` arrays.

        Returns
        -------
        dict
            Discriminator metrics (loss, accuracy, gradient penalty).
        """
        cfg: GAILConfig = self._config  # type: ignore[assignment]

        expert_obs = self._to_tensor(expert_batch["obs"], dtype=torch.float32).reshape(
            -1, self._obs_dim
        )
        expert_act = self._to_tensor(
            expert_batch["actions"], dtype=torch.float32
        )
        policy_obs = self._to_tensor(policy_batch["obs"], dtype=torch.float32).reshape(
            -1, self._obs_dim
        )
        policy_act = self._to_tensor(
            policy_batch["actions"], dtype=torch.float32
        )

        total_loss = 0.0
        total_gp = 0.0
        n_updates = 0

        self._discriminator.train()
        for _ in range(cfg.disc_epochs):
            # Expert: label = 1, Policy: label = 0
            expert_logits = self._discriminator(expert_obs, expert_act)
            policy_logits = self._discriminator(policy_obs, policy_act)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            )
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits)
            )
            loss = expert_loss + policy_loss

            # Optional WGAN-GP gradient penalty
            gp_val = 0.0
            if cfg.gradient_penalty_coef > 0.0:
                gp_val = self._gradient_penalty(
                    expert_obs, expert_act, policy_obs, policy_act
                )
                loss = loss + cfg.gradient_penalty_coef * gp_val

            self._disc_optimizer.zero_grad()
            loss.backward()
            self._disc_optimizer.step()

            total_loss += loss.item()
            total_gp += float(gp_val) if isinstance(gp_val, float) else gp_val.item()
            n_updates += 1

        # Accuracy
        with torch.no_grad():
            expert_pred = torch.sigmoid(self._discriminator(expert_obs, expert_act))
            policy_pred = torch.sigmoid(self._discriminator(policy_obs, policy_act))
            acc = 0.5 * (
                (expert_pred > 0.5).float().mean()
                + (policy_pred <= 0.5).float().mean()
            )

        metrics = {
            "gail/disc_loss": total_loss / max(n_updates, 1),
            "gail/disc_accuracy": acc.item(),
            "gail/gradient_penalty": total_gp / max(n_updates, 1),
        }
        self._metrics.record_dict(metrics)
        return metrics

    def _gradient_penalty(
        self,
        expert_obs: torch.Tensor,
        expert_act: torch.Tensor,
        policy_obs: torch.Tensor,
        policy_act: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WGAN-GP style gradient penalty.

        Parameters
        ----------
        expert_obs, expert_act : torch.Tensor
            Expert state-action batch.
        policy_obs, policy_act : torch.Tensor
            Policy state-action batch.

        Returns
        -------
        torch.Tensor
            Scalar gradient penalty.
        """
        batch_size = min(expert_obs.shape[0], policy_obs.shape[0])
        alpha = torch.rand(batch_size, 1, device=self._device)

        interp_obs = (
            alpha * expert_obs[:batch_size] + (1 - alpha) * policy_obs[:batch_size]
        ).requires_grad_(True)
        interp_act = (
            alpha * expert_act[:batch_size] + (1 - alpha) * policy_act[:batch_size]
        ).requires_grad_(True)

        interp_logits = self._discriminator(interp_obs, interp_act)
        grad = torch.autograd.grad(
            outputs=interp_logits,
            inputs=[interp_obs, interp_act],
            grad_outputs=torch.ones_like(interp_logits),
            create_graph=True,
            retain_graph=True,
        )
        grad_cat = torch.cat([g.reshape(batch_size, -1) for g in grad], dim=-1)
        grad_norm = grad_cat.norm(2, dim=1)
        penalty = ((grad_norm - 1.0) ** 2).mean()
        return penalty

    # ------------------------------------------------------------------
    # Policy update (PPO with discriminator reward)
    # ------------------------------------------------------------------

    def update_policy(self, rollout_buffer: Any) -> dict[str, float]:
        """Update the policy using PPO with the discriminator reward.

        Parameters
        ----------
        rollout_buffer :
            A :class:`~navirl.training.buffer.RolloutBuffer` or dict with
            keys ``"obs"``, ``"actions"``, ``"log_probs"``, ``"values"``,
            ``"returns"``, ``"advantages"``.

        Returns
        -------
        dict
            PPO update metrics.
        """
        cfg: GAILConfig = self._config  # type: ignore[assignment]

        if isinstance(rollout_buffer, dict):
            obs_t = self._to_tensor(rollout_buffer["obs"], dtype=torch.float32)
            act_t = self._to_tensor(rollout_buffer["actions"], dtype=torch.float32)
            old_log_probs = self._to_tensor(
                rollout_buffer["log_probs"], dtype=torch.float32
            )
            advantages = self._to_tensor(
                rollout_buffer["advantages"], dtype=torch.float32
            )
            returns = self._to_tensor(
                rollout_buffer["returns"], dtype=torch.float32
            )
        else:
            # Assume RolloutBuffer-like object
            n = rollout_buffer.buffer_size * rollout_buffer.n_envs
            obs_t = self._to_tensor(
                rollout_buffer.observations.reshape(n, -1), dtype=torch.float32
            )
            act_t = self._to_tensor(
                rollout_buffer.actions.reshape(n, -1), dtype=torch.float32
            )
            old_log_probs = self._to_tensor(
                rollout_buffer.log_probs.reshape(n), dtype=torch.float32
            )
            advantages = self._to_tensor(
                rollout_buffer.advantages.reshape(n), dtype=torch.float32
            )
            returns = self._to_tensor(
                rollout_buffer.returns.reshape(n), dtype=torch.float32
            )

        obs_t = obs_t.reshape(-1, self._obs_dim)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(obs_t, act_t, old_log_probs, advantages, returns)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        self._policy_value.train()
        for _ in range(cfg.policy_epochs):
            for batch in loader:
                b_obs, b_act, b_old_lp, b_adv, b_ret = batch

                log_probs, values, entropy = self._policy_value.evaluate_actions(
                    b_obs, b_act
                )

                # PPO clipped objective
                ratio = (log_probs - b_old_lp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                ) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, b_ret)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self._policy_optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self._policy_value.parameters(), cfg.max_grad_norm
                    )
                self._policy_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        metrics = {
            "gail/policy_loss": total_policy_loss / max(n_updates, 1),
            "gail/value_loss": total_value_loss / max(n_updates, 1),
            "gail/entropy": total_entropy / max(n_updates, 1),
        }
        self._metrics.record_dict(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Convenience: combined update
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> dict[str, float]:
        """Run a combined discriminator + policy update.

        Parameters
        ----------
        batch : dict
            Must contain ``"expert_obs"``, ``"expert_actions"``,
            ``"policy_obs"``, ``"policy_actions"``, and PPO rollout keys
            (``"obs"``, ``"actions"``, ``"log_probs"``, ``"advantages"``,
            ``"returns"``).

        Returns
        -------
        dict
            Combined metrics dictionary.
        """
        disc_metrics = self.update_discriminator(
            expert_batch={
                "obs": batch["expert_obs"],
                "actions": batch["expert_actions"],
            },
            policy_batch={
                "obs": batch["policy_obs"],
                "actions": batch["policy_actions"],
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
        metrics = {**disc_metrics, **policy_metrics}
        return metrics

    # ------------------------------------------------------------------
    # Act
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
            Current observation.
        deterministic : bool
            If ``True``, use the mean action (continuous) or argmax (discrete).

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
        cfg: GAILConfig = self._config  # type: ignore[assignment]

        with torch.no_grad():
            output, value = self._policy_value(obs_t)
            if cfg.action_type == "continuous":
                mean = output
                if deterministic:
                    action_t = mean
                    log_prob = torch.zeros(1, device=self._device)
                else:
                    std = self._policy_value.action_log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
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

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute the GAIL reward for a batch of state-action pairs.

        Parameters
        ----------
        obs : np.ndarray
            Observation batch.
        actions : np.ndarray
            Action batch.

        Returns
        -------
        np.ndarray
            GAIL rewards.
        """
        obs_t = self._to_tensor(obs, dtype=torch.float32).reshape(-1, self._obs_dim)
        act_t = self._to_tensor(actions, dtype=torch.float32)
        reward_t = self._discriminator.predict_reward(obs_t, act_t)
        return self._to_numpy(reward_t).flatten()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the GAIL agent to disk."""
        self._save_checkpoint(
            path,
            state_dicts={
                "discriminator": self._discriminator.state_dict(),
                "policy_value": self._policy_value.state_dict(),
            },
        )

    def load(self, path: str | pathlib.Path) -> None:
        """Restore the GAIL agent from a checkpoint."""
        payload = self._load_checkpoint(path)
        self._discriminator.load_state_dict(payload["model"]["discriminator"])
        self._policy_value.load_state_dict(payload["model"]["policy_value"])
