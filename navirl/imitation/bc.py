"""
Behavioral Cloning (BC)
=======================

Supervised imitation learning that trains a policy network to reproduce
expert demonstrations by minimising a prediction loss (MSE for continuous
actions, cross-entropy for discrete actions).

The :class:`BCAgent` loads expert data from a
:class:`~navirl.training.buffer.DemonstrationBuffer`, optionally splits it
into train/validation sets, and runs gradient descent until convergence or
early-stopping criteria are met.
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
    from torch.utils.data import DataLoader, TensorDataset, random_split

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["BCConfig", "BCAgent"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BCConfig(HyperParameters):
    """Hyperparameters for Behavioral Cloning.

    Attributes:
        lr: Learning rate for the policy optimiser.
        batch_size: Mini-batch size for supervised training.
        epochs: Maximum number of training epochs.
        hidden_dims: Sizes of hidden layers in the policy MLP.
        weight_decay: L2 regularisation coefficient.
        dropout: Dropout probability applied after each hidden layer.
        action_type: ``"continuous"`` (MSE / NLL loss) or ``"discrete"``
            (cross-entropy loss).
        validation_split: Fraction of demonstrations held out for validation.
    """

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    hidden_dims: tuple[int, ...] = (256, 256)
    weight_decay: float = 1e-4
    dropout: float = 0.1
    action_type: str = "continuous"
    validation_split: float = 0.1


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


def _build_bc_policy(
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    action_type: str,
) -> nn.Module:
    """Build a simple MLP policy for behavioral cloning.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector.
    action_dim : int
        Dimensionality of the action vector (or number of discrete actions).
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    dropout : float
        Dropout probability.
    action_type : str
        ``"continuous"`` or ``"discrete"``.

    Returns
    -------
    nn.Module
        A feedforward policy network.
    """
    layers: list[nn.Module] = []
    prev_dim = obs_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, action_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# BCAgent
# ---------------------------------------------------------------------------


class BCAgent(BaseAgent):
    """Behavioral Cloning agent.

    Trains a policy to mimic expert demonstrations via supervised learning.
    Supports both continuous actions (MSE or negative log-likelihood loss)
    and discrete actions (cross-entropy loss).

    Parameters
    ----------
    config : BCConfig
        Behavioral cloning hyperparameters.
    observation_space :
        Environment observation space (used to infer ``obs_dim``).
    action_space :
        Environment action space (used to infer ``action_dim``).
    device : str or torch.device
        Compute device.
    seed : int, optional
        Random seed for reproducibility.
    metrics_callback : callable, optional
        Optional ``(metrics_dict, step) -> None`` callback.
    """

    def __init__(
        self,
        config: BCConfig,
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
            raise RuntimeError("BCAgent requires PyTorch.")

        self._obs_dim = int(np.prod(observation_space.shape))
        if config.action_type == "discrete":
            self._action_dim = int(action_space.n)
        else:
            self._action_dim = int(np.prod(action_space.shape))

        self._policy = _build_bc_policy(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            action_type=config.action_type,
        ).to(self._device)
        self._modules.append(self._policy)

        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self._optimizers["policy"] = self._optimizer

        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

        self._log_module_summary("bc_policy", self._policy)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the imitation loss.

        Parameters
        ----------
        pred : torch.Tensor
            Policy output.
        target : torch.Tensor
            Expert actions.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        cfg: BCConfig = self._config  # type: ignore[assignment]
        if cfg.action_type == "discrete":
            return nn.functional.cross_entropy(pred, target.long().squeeze(-1))
        else:
            return nn.functional.mse_loss(pred, target)

    # ------------------------------------------------------------------
    # Training from demonstrations
    # ------------------------------------------------------------------

    def train_from_demonstrations(
        self,
        demo_buffer: Any,
        *,
        verbose: bool = True,
        patience: int = 10,
    ) -> dict[str, list[float]]:
        """Train the policy from an expert demonstration buffer.

        Parameters
        ----------
        demo_buffer :
            A :class:`~navirl.training.buffer.DemonstrationBuffer` (or any
            object exposing ``.observations``, ``.actions``, and ``__len__``).
        verbose : bool
            Whether to log progress each epoch.
        patience : int
            Number of epochs without validation improvement before early
            stopping. Set to ``0`` to disable.

        Returns
        -------
        dict
            Dictionary with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        cfg: BCConfig = self._config  # type: ignore[assignment]

        # -- Build tensors from buffer ------------------------------------
        n = len(demo_buffer)
        obs_t = torch.as_tensor(demo_buffer.observations[:n], dtype=torch.float32)
        act_t = torch.as_tensor(demo_buffer.actions[:n], dtype=torch.float32)

        # Flatten observations if needed
        obs_t = obs_t.reshape(n, -1)

        dataset = TensorDataset(obs_t, act_t)

        # -- Train / validation split -------------------------------------
        val_size = int(n * cfg.validation_split)
        train_size = n - val_size

        if val_size > 0:
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
        else:
            train_ds = dataset
            val_ds = None

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )
        val_loader = (
            DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
            if val_ds is not None
            else None
        )

        # -- Training loop ------------------------------------------------
        self._train_losses.clear()
        self._val_losses.clear()
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        self.on_training_start()
        self._policy.train()

        for epoch in range(cfg.epochs):
            self.on_epoch_start(epoch)

            # ---- Train --------------------------------------------------
            epoch_loss = 0.0
            n_batches = 0
            for obs_batch, act_batch in train_loader:
                obs_batch_device = obs_batch.to(self._device)
                act_batch_device = act_batch.to(self._device)

                pred = self._policy(obs_batch_device)
                loss = self._compute_loss(pred, act_batch_device)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self._train_losses.append(avg_train_loss)
            self._metrics.record("bc/train_loss", avg_train_loss)
            self._total_steps += n_batches

            # ---- Validation ---------------------------------------------
            avg_val_loss = float("nan")
            if val_loader is not None:
                self._policy.eval()
                val_loss_total = 0.0
                val_batches = 0
                with torch.no_grad():
                    for obs_batch, act_batch in val_loader:
                        obs_batch_device = obs_batch.to(self._device)
                        act_batch_device = act_batch.to(self._device)
                        pred = self._policy(obs_batch_device)
                        val_loss_total += self._compute_loss(pred, act_batch_device).item()
                        val_batches += 1
                avg_val_loss = val_loss_total / max(val_batches, 1)
                self._val_losses.append(avg_val_loss)
                self._metrics.record("bc/val_loss", avg_val_loss)
                self._policy.train()

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_state = {k: v.clone() for k, v in self._policy.state_dict().items()}
                else:
                    epochs_without_improvement += 1

                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        logger.info(
                            "Early stopping at epoch %d (patience=%d)",
                            epoch,
                            patience,
                        )
                    break

            epoch_metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss}
            self.on_epoch_end(epoch, epoch_metrics)
            self._metrics.dump(step=epoch)

            if verbose and (epoch % max(1, cfg.epochs // 20) == 0 or epoch == cfg.epochs - 1):
                logger.info(
                    "Epoch %4d/%d  train_loss=%.6f  val_loss=%.6f",
                    epoch + 1,
                    cfg.epochs,
                    avg_train_loss,
                    avg_val_loss,
                )

        # Restore best model if early stopping was active
        if best_state is not None:
            self._policy.load_state_dict(best_state)
            logger.info("Restored best validation model (val_loss=%.6f).", best_val_loss)

        self.on_training_end()
        return {"train_loss": self._train_losses, "val_loss": self._val_losses}

    # ------------------------------------------------------------------
    # BaseAgent interface
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
            Ignored for BC (always deterministic). Present for API
            compatibility.

        Returns
        -------
        action : np.ndarray
            The predicted action.
        info : dict
            Empty dictionary (no auxiliary information for BC).
        """
        self._policy.eval()
        obs_t = self._to_tensor(observation.reshape(1, -1).astype(np.float32), dtype=torch.float32)
        with torch.no_grad():
            pred = self._policy(obs_t)

        cfg: BCConfig = self._config  # type: ignore[assignment]
        if cfg.action_type == "discrete":
            action = pred.argmax(dim=-1).cpu().numpy().flatten()
        else:
            action = pred.cpu().numpy().flatten()

        if self._training:
            self._policy.train()

        return action, {}

    def update(self, batch: Any) -> dict[str, float]:
        """Run a single supervised-learning update step.

        Parameters
        ----------
        batch : dict
            Must contain ``"obs"`` and ``"actions"`` keys with numpy arrays
            or torch tensors.

        Returns
        -------
        dict
            Scalar metrics: ``{"bc/loss": <float>}``.
        """
        obs_t = self._to_tensor(batch["obs"], dtype=torch.float32)
        act_t = self._to_tensor(batch["actions"], dtype=torch.float32)

        obs_t = obs_t.reshape(obs_t.shape[0], -1)

        pred = self._policy(obs_t)
        loss = self._compute_loss(pred, act_t)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._total_steps += 1
        return {"bc/loss": loss.item()}

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the BC agent to disk.

        Parameters
        ----------
        path : str or Path
            File or directory path for the checkpoint.
        """
        self._save_checkpoint(
            path,
            state_dicts={"policy": self._policy.state_dict()},
        )

    def load(self, path: str | pathlib.Path) -> None:
        """Restore the BC agent from a checkpoint.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file.
        """
        payload = self._load_checkpoint(path)
        self._policy.load_state_dict(payload["model"]["policy"])

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BCAgent(obs_dim={self._obs_dim}, action_dim={self._action_dim}, "
            f"action_type={self._config['action_type']!r}, "
            f"device={self._device})"
        )
