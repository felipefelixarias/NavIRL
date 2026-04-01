"""
Reward Learning from Preferences and Demonstrations
====================================================

Models that learn reward functions from various supervision signals:

- **PreferenceRewardModel**: Learns from pairwise trajectory preferences
  using the Bradley-Terry model.
- **DemonstrationRewardModel**: Learns reward via regression on expert
  demonstrations.
- **EnsembleRewardModel**: Maintains an ensemble of reward models for
  uncertainty estimation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.agents.base import HyperParameters

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    "PreferenceRewardModel",
    "DemonstrationRewardModel",
    "EnsembleRewardModel",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_reward_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
) -> nn.Module:
    """Build a simple MLP for reward prediction."""
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# PreferenceRewardModel  (Bradley-Terry)
# ---------------------------------------------------------------------------


@dataclass
class PreferenceRewardConfig(HyperParameters):
    """Config for preference-based reward learning.

    Attributes:
        lr: Learning rate.
        hidden_dims: Hidden layer sizes.
        batch_size: Mini-batch size for preference pairs.
        epochs: Number of training epochs.
        weight_decay: L2 regularisation.
        input_dim: Dimensionality of the (state, action) input.
    """

    lr: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 256)
    batch_size: int = 32
    epochs: int = 50
    weight_decay: float = 1e-4
    input_dim: int = 0  # Must be set by user


class PreferenceRewardModel:
    """Learns a reward function from pairwise trajectory preferences.

    Uses the Bradley-Terry model: given two trajectory segments sigma_1 and
    sigma_2, the probability that sigma_1 is preferred is:

        P(sigma_1 > sigma_2) = exp(R(sigma_1)) / (exp(R(sigma_1)) + exp(R(sigma_2)))

    where R(sigma) = sum_t r(s_t, a_t).

    Parameters
    ----------
    config : PreferenceRewardConfig
        Reward model hyperparameters.
    device : str
        Compute device.
    """

    def __init__(
        self,
        config: PreferenceRewardConfig,
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PreferenceRewardModel requires PyTorch.")

        self._config = config
        self._device = torch.device(device)

        self._reward_net = _build_reward_mlp(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        self._optimizer = torch.optim.Adam(
            self._reward_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        logger.info(
            "PreferenceRewardModel: input_dim=%d, params=%d",
            config.input_dim,
            sum(p.numel() for p in self._reward_net.parameters()),
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def predict_reward(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Predict scalar rewards for (state, action) pairs.

        Parameters
        ----------
        observations : np.ndarray
            Observation batch ``(N, obs_dim)``.
        actions : np.ndarray
            Action batch ``(N, action_dim)``.

        Returns
        -------
        np.ndarray
            Predicted rewards ``(N,)``.
        """
        self._reward_net.eval()
        sa = np.concatenate([observations, actions], axis=-1).astype(np.float32)
        sa_t = torch.as_tensor(sa, device=self._device)
        with torch.no_grad():
            rewards = self._reward_net(sa_t).squeeze(-1)
        return rewards.cpu().numpy()

    def _segment_return(self, segment: torch.Tensor) -> torch.Tensor:
        """Compute the predicted return for a trajectory segment.

        Parameters
        ----------
        segment : torch.Tensor
            ``(T, input_dim)`` tensor of (s, a) concatenations.

        Returns
        -------
        torch.Tensor
            Scalar return (sum of predicted rewards).
        """
        rewards = self._reward_net(segment).squeeze(-1)  # (T,)
        return rewards.sum()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        preferences: list[tuple[np.ndarray, np.ndarray, float]],
        *,
        verbose: bool = True,
    ) -> list[float]:
        """Train from pairwise preferences.

        Parameters
        ----------
        preferences : list of (segment_1, segment_2, label)
            Each element is a tuple of two trajectory segments (each a
            ``(T, input_dim)`` array of concatenated state-action pairs) and
            a label:  ``1.0`` means segment_1 is preferred, ``0.0`` means
            segment_2 is preferred, ``0.5`` means equal preference.
        verbose : bool
            Whether to log progress.

        Returns
        -------
        list of float
            Per-epoch average losses.
        """
        cfg = self._config
        n = len(preferences)
        epoch_losses: list[float] = []

        for epoch in range(cfg.epochs):
            indices = np.random.permutation(n)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, cfg.batch_size):
                batch_idx = indices[start : start + cfg.batch_size]
                loss = torch.tensor(0.0, device=self._device)

                for i in batch_idx:
                    seg1, seg2, label = preferences[i]
                    seg1_t = torch.as_tensor(
                        seg1.astype(np.float32), device=self._device
                    )
                    seg2_t = torch.as_tensor(
                        seg2.astype(np.float32), device=self._device
                    )

                    r1 = self._segment_return(seg1_t)
                    r2 = self._segment_return(seg2_t)

                    # Bradley-Terry: P(seg1 > seg2) = sigmoid(r1 - r2)
                    logit = r1 - r2
                    label_t = torch.tensor(label, device=self._device)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logit.unsqueeze(0), label_t.unsqueeze(0)
                    )

                loss = loss / max(len(batch_idx), 1)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if verbose and (
                epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1
            ):
                logger.info(
                    "PreferenceReward epoch %3d/%d  loss=%.6f",
                    epoch + 1,
                    cfg.epochs,
                    avg_loss,
                )

        return epoch_losses

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return model state."""
        return {
            "reward_net": self._reward_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Load model state."""
        self._reward_net.load_state_dict(d["reward_net"])
        self._optimizer.load_state_dict(d["optimizer"])


# ---------------------------------------------------------------------------
# DemonstrationRewardModel
# ---------------------------------------------------------------------------


@dataclass
class DemonstrationRewardConfig(HyperParameters):
    """Config for demonstration-based reward learning.

    Attributes:
        lr: Learning rate.
        hidden_dims: Hidden layer sizes.
        batch_size: Mini-batch size.
        epochs: Number of training epochs.
        weight_decay: L2 regularisation.
        input_dim: Dimensionality of the (state, action) input.
        target_reward: Default reward to assign to expert transitions.
    """

    lr: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 256)
    batch_size: int = 64
    epochs: int = 50
    weight_decay: float = 1e-4
    input_dim: int = 0
    target_reward: float = 1.0


class DemonstrationRewardModel:
    """Learns a reward function from expert demonstrations via regression.

    Expert (state, action) pairs are assigned a positive target reward while
    negative samples (random or policy-generated) receive zero reward.

    Parameters
    ----------
    config : DemonstrationRewardConfig
        Reward model hyperparameters.
    device : str
        Compute device.
    """

    def __init__(
        self,
        config: DemonstrationRewardConfig,
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("DemonstrationRewardModel requires PyTorch.")

        self._config = config
        self._device = torch.device(device)

        self._reward_net = _build_reward_mlp(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
        ).to(self._device)

        self._optimizer = torch.optim.Adam(
            self._reward_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        logger.info(
            "DemonstrationRewardModel: input_dim=%d, params=%d",
            config.input_dim,
            sum(p.numel() for p in self._reward_net.parameters()),
        )

    def predict_reward(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Predict scalar rewards for (state, action) pairs.

        Parameters
        ----------
        observations : np.ndarray
            Observation batch ``(N, obs_dim)``.
        actions : np.ndarray
            Action batch ``(N, action_dim)``.

        Returns
        -------
        np.ndarray
            Predicted rewards ``(N,)``.
        """
        self._reward_net.eval()
        sa = np.concatenate([observations, actions], axis=-1).astype(np.float32)
        sa_t = torch.as_tensor(sa, device=self._device)
        with torch.no_grad():
            rewards = self._reward_net(sa_t).squeeze(-1)
        return rewards.cpu().numpy()

    def train(
        self,
        expert_obs: np.ndarray,
        expert_actions: np.ndarray,
        negative_obs: np.ndarray | None = None,
        negative_actions: np.ndarray | None = None,
        *,
        verbose: bool = True,
    ) -> list[float]:
        """Train the reward model.

        Parameters
        ----------
        expert_obs : np.ndarray
            Expert observations ``(N, obs_dim)``.
        expert_actions : np.ndarray
            Expert actions ``(N, action_dim)``.
        negative_obs : np.ndarray, optional
            Non-expert observations for negative examples.
        negative_actions : np.ndarray, optional
            Non-expert actions for negative examples.
        verbose : bool
            Whether to log progress.

        Returns
        -------
        list of float
            Per-epoch average losses.
        """
        cfg = self._config

        # Build training data
        expert_sa = np.concatenate(
            [expert_obs, expert_actions], axis=-1
        ).astype(np.float32)
        expert_targets = np.full(
            len(expert_sa), cfg.target_reward, dtype=np.float32
        )

        if negative_obs is not None and negative_actions is not None:
            neg_sa = np.concatenate(
                [negative_obs, negative_actions], axis=-1
            ).astype(np.float32)
            neg_targets = np.zeros(len(neg_sa), dtype=np.float32)
            all_sa = np.concatenate([expert_sa, neg_sa], axis=0)
            all_targets = np.concatenate([expert_targets, neg_targets], axis=0)
        else:
            all_sa = expert_sa
            all_targets = expert_targets

        sa_t = torch.as_tensor(all_sa, device=self._device)
        target_t = torch.as_tensor(all_targets, device=self._device)

        dataset = torch.utils.data.TensorDataset(sa_t, target_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True
        )

        epoch_losses: list[float] = []
        self._reward_net.train()

        for epoch in range(cfg.epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_sa, batch_target in loader:
                pred = self._reward_net(batch_sa).squeeze(-1)
                loss = F.mse_loss(pred, batch_target)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if verbose and (
                epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1
            ):
                logger.info(
                    "DemoReward epoch %3d/%d  loss=%.6f",
                    epoch + 1,
                    cfg.epochs,
                    avg_loss,
                )

        return epoch_losses

    def state_dict(self) -> dict[str, Any]:
        """Return model state."""
        return {
            "reward_net": self._reward_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Load model state."""
        self._reward_net.load_state_dict(d["reward_net"])
        self._optimizer.load_state_dict(d["optimizer"])


# ---------------------------------------------------------------------------
# EnsembleRewardModel
# ---------------------------------------------------------------------------


@dataclass
class EnsembleRewardConfig(HyperParameters):
    """Config for ensemble reward models.

    Attributes:
        n_members: Number of ensemble members.
        member_config: Config for each individual member (either
            PreferenceRewardConfig or DemonstrationRewardConfig).
    """

    n_members: int = 5
    member_config: Any | None = None  # Set at runtime


class EnsembleRewardModel:
    """Ensemble of reward models for uncertainty-aware reward prediction.

    Maintains multiple independent reward models and provides the mean
    prediction as the reward, along with the standard deviation as an
    uncertainty estimate.

    Parameters
    ----------
    config : EnsembleRewardConfig
        Ensemble configuration.
    member_class : type
        The class of each ensemble member (``PreferenceRewardModel`` or
        ``DemonstrationRewardModel``).
    device : str
        Compute device.
    """

    def __init__(
        self,
        config: EnsembleRewardConfig,
        member_class: type,
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("EnsembleRewardModel requires PyTorch.")
        if config.member_config is None:
            raise ValueError("EnsembleRewardConfig.member_config must be set.")

        self._config = config
        self._device = device
        self._members: list[Any] = []

        for _i in range(config.n_members):
            member = member_class(config=config.member_config, device=device)
            self._members.append(member)

        logger.info(
            "EnsembleRewardModel: %d members of %s",
            config.n_members,
            member_class.__name__,
        )

    @property
    def members(self) -> list[Any]:
        """Access individual ensemble members."""
        return self._members

    def predict_reward(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict reward with uncertainty.

        Parameters
        ----------
        observations : np.ndarray
            Observation batch ``(N, obs_dim)``.
        actions : np.ndarray
            Action batch ``(N, action_dim)``.

        Returns
        -------
        mean_reward : np.ndarray
            Ensemble mean reward ``(N,)``.
        std_reward : np.ndarray
            Ensemble standard deviation ``(N,)`` (uncertainty).
        """
        all_rewards = np.stack(
            [m.predict_reward(observations, actions) for m in self._members],
            axis=0,
        )  # (n_members, N)
        mean_reward = all_rewards.mean(axis=0)
        std_reward = all_rewards.std(axis=0)
        return mean_reward, std_reward

    def predict_reward_mean(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Predict ensemble mean reward (no uncertainty).

        Parameters
        ----------
        observations : np.ndarray
            Observation batch.
        actions : np.ndarray
            Action batch.

        Returns
        -------
        np.ndarray
            Mean reward ``(N,)``.
        """
        mean, _ = self.predict_reward(observations, actions)
        return mean

    def state_dict(self) -> dict[str, Any]:
        """Return ensemble state."""
        return {
            f"member_{i}": m.state_dict() for i, m in enumerate(self._members)
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Load ensemble state."""
        for i, m in enumerate(self._members):
            key = f"member_{i}"
            if key in d:
                m.load_state_dict(d[key])
