"""
DAgger (Dataset Aggregation)
============================

Iterative imitation learning algorithm that addresses the distribution-shift
problem of Behavioral Cloning.  At each iteration the learner's current policy
is rolled out in the environment, the expert is queried for corrective labels,
and the aggregated dataset is used for re-training.

Reference:
    Ross, Gordon & Bagnell. "A Reduction of Imitation Learning and Structured
    Prediction to No-Regret Online Learning", AISTATS 2011.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.agents.base import BaseAgent, HyperParameters

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["DAggerConfig", "DAggerAgent"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DAggerConfig(HyperParameters):
    """Hyperparameters for DAgger.

    Attributes:
        lr: Learning rate for the policy optimiser.
        batch_size: Mini-batch size for supervised training.
        num_iterations: Number of DAgger outer-loop iterations.
        epochs_per_iter: Gradient-descent epochs per DAgger iteration.
        hidden_dims: Sizes of hidden layers in the policy MLP.
        weight_decay: L2 regularisation coefficient.
        dropout: Dropout probability.
        action_type: ``"continuous"`` or ``"discrete"``.
        rollout_steps: Number of environment steps per DAgger iteration.
        beta_schedule: Callable ``(iteration) -> float`` returning the
            probability of using the expert action during rollouts.
            When *None* a default linear decay ``1 - i/N`` is used.
        expert_policy_fn: Callable ``(observation) -> action`` that returns
            the expert label for a given observation.  Must be set before
            calling :meth:`train`.
    """

    lr: float = 1e-3
    batch_size: int = 64
    num_iterations: int = 20
    epochs_per_iter: int = 10
    hidden_dims: tuple[int, ...] = (256, 256)
    weight_decay: float = 1e-4
    dropout: float = 0.1
    action_type: str = "continuous"
    rollout_steps: int = 1000
    beta_schedule: Callable[[int], float] | None = None
    expert_policy_fn: Callable[[np.ndarray], np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Policy builder (shared with BC)
# ---------------------------------------------------------------------------


def _build_dagger_policy(
    obs_dim: int,
    action_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Module:
    """Build a simple MLP policy for DAgger."""
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
# DAggerAgent
# ---------------------------------------------------------------------------


class DAggerAgent(BaseAgent):
    """DAgger (Dataset Aggregation) imitation-learning agent.

    Iteratively collects on-policy data, queries the expert for corrective
    labels, and retrains the policy on the aggregated dataset.  A *beta
    schedule* controls the mixing probability between the expert and learner
    policies during rollout collection.

    Parameters
    ----------
    config : DAggerConfig
        DAgger hyperparameters.
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
        config: DAggerConfig,
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
            raise RuntimeError("DAggerAgent requires PyTorch.")

        self._obs_dim = int(np.prod(observation_space.shape))
        if config.action_type == "discrete":
            self._action_dim = int(action_space.n)
        else:
            self._action_dim = int(np.prod(action_space.shape))

        self._policy = _build_dagger_policy(
            obs_dim=self._obs_dim,
            action_dim=self._action_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(self._device)
        self._modules.append(self._policy)

        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self._optimizers["policy"] = self._optimizer

        # Aggregated dataset stored as growing numpy arrays.
        self._all_obs: np.ndarray | None = None
        self._all_actions: np.ndarray | None = None

        self._log_module_summary("dagger_policy", self._policy)

    # ------------------------------------------------------------------
    # Beta schedule
    # ------------------------------------------------------------------

    def _get_beta(self, iteration: int) -> float:
        """Return the expert-mixing probability for the given iteration.

        Parameters
        ----------
        iteration : int
            Current DAgger iteration (0-indexed).

        Returns
        -------
        float
            Probability of choosing the expert action during rollout.
        """
        cfg: DAggerConfig = self._config  # type: ignore[assignment]
        if cfg.beta_schedule is not None:
            return float(cfg.beta_schedule(iteration))
        # Default: linear decay from 1.0 to 0.0
        return max(0.0, 1.0 - iteration / max(1, cfg.num_iterations - 1))

    # ------------------------------------------------------------------
    # Dataset aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, obs: np.ndarray, actions: np.ndarray) -> None:
        """Append new data to the aggregated dataset.

        Parameters
        ----------
        obs : np.ndarray
            Observation array of shape ``(N, obs_dim)``.
        actions : np.ndarray
            Expert-labelled action array of shape ``(N, action_dim)``.
        """
        if self._all_obs is None:
            self._all_obs = obs.copy()
            self._all_actions = actions.copy()
        else:
            self._all_obs = np.concatenate([self._all_obs, obs], axis=0)
            self._all_actions = np.concatenate([self._all_actions, actions], axis=0)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env: Any,
        expert_fn: Callable[[np.ndarray], np.ndarray],
        n_steps: int,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out the current policy, querying the expert for labels.

        At each step, with probability *beta* the expert action is executed;
        otherwise the learner's action is used.  Regardless of which action is
        executed, the expert action is always recorded as the label.

        Parameters
        ----------
        env :
            A Gymnasium-compatible environment.
        expert_fn : callable
            ``(observation) -> action`` expert oracle.
        n_steps : int
            Number of environment steps to collect.
        beta : float
            Expert-action probability.

        Returns
        -------
        obs_data : np.ndarray
            Collected observations, shape ``(n_steps, obs_dim)``.
        act_data : np.ndarray
            Expert-labelled actions, shape ``(n_steps, action_dim)``.
        """
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []

        obs, _ = env.reset()
        for _ in range(n_steps):
            obs_flat = np.asarray(obs, dtype=np.float32).flatten()
            expert_action = np.asarray(expert_fn(obs), dtype=np.float32).flatten()
            learner_action, _ = self.act(obs, deterministic=True)

            # Record observation and expert label
            obs_list.append(obs_flat)
            act_list.append(expert_action)

            # Execute mixed action
            if np.random.rand() < beta:
                action = expert_action
            else:
                action = learner_action

            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Supervised training on aggregated dataset
    # ------------------------------------------------------------------

    def _train_on_dataset(self, epochs: int) -> float:
        """Train the policy on the current aggregated dataset.

        Parameters
        ----------
        epochs : int
            Number of gradient-descent epochs.

        Returns
        -------
        float
            Average loss over the last epoch.
        """
        assert self._all_obs is not None and self._all_actions is not None
        cfg: DAggerConfig = self._config  # type: ignore[assignment]

        obs_t = torch.as_tensor(self._all_obs, dtype=torch.float32)
        act_t = torch.as_tensor(self._all_actions, dtype=torch.float32)
        dataset = TensorDataset(obs_t, act_t)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        self._policy.train()
        last_epoch_loss = 0.0
        for _epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_batch, act_batch in loader:
                obs_batch = obs_batch.to(self._device)
                act_batch = act_batch.to(self._device)

                pred = self._policy(obs_batch)
                if cfg.action_type == "discrete":
                    loss = nn.functional.cross_entropy(pred, act_batch.long().squeeze(-1))
                else:
                    loss = nn.functional.mse_loss(pred, act_batch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            last_epoch_loss = epoch_loss / max(n_batches, 1)
        return last_epoch_loss

    # ------------------------------------------------------------------
    # Full DAgger training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: Any,
        demo_buffer: Any = None,
        *,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Run the full DAgger training procedure.

        Parameters
        ----------
        env :
            A Gymnasium-compatible environment.
        demo_buffer : optional
            Initial demonstration buffer.  If provided, its data is added to
            the aggregated dataset before the first iteration.
        verbose : bool
            Whether to log progress.

        Returns
        -------
        dict
            Dictionary with ``"iteration_loss"`` and ``"beta"`` lists.

        Raises
        ------
        ValueError
            If ``config.expert_policy_fn`` is not set.
        """
        cfg: DAggerConfig = self._config  # type: ignore[assignment]

        expert_fn = cfg.expert_policy_fn
        if expert_fn is None:
            raise ValueError("DAggerConfig.expert_policy_fn must be set before calling train().")

        # Seed with initial demonstrations if available
        if demo_buffer is not None and len(demo_buffer) > 0:
            n = len(demo_buffer)
            initial_obs = demo_buffer.observations[:n].reshape(n, -1)
            initial_act = demo_buffer.actions[:n].reshape(n, -1)
            self._aggregate(initial_obs, initial_act)

        losses: list[float] = []
        betas: list[float] = []

        self.on_training_start()

        for iteration in range(cfg.num_iterations):
            self.on_epoch_start(iteration)

            beta = self._get_beta(iteration)
            betas.append(beta)

            # Collect on-policy data with expert labels
            new_obs, new_actions = self.collect_rollouts(env, expert_fn, cfg.rollout_steps, beta)
            self._aggregate(new_obs, new_actions)

            # Retrain on aggregated data
            loss = self._train_on_dataset(cfg.epochs_per_iter)
            losses.append(loss)

            self._metrics.record("dagger/loss", loss)
            self._metrics.record("dagger/beta", beta)
            self._metrics.record(
                "dagger/dataset_size",
                float(len(self._all_obs)),  # type: ignore[arg-type]
            )
            self._total_steps += cfg.rollout_steps

            epoch_metrics = {"loss": loss, "beta": beta}
            self.on_epoch_end(iteration, epoch_metrics)
            self._metrics.dump(step=iteration)

            if verbose:
                logger.info(
                    "DAgger iter %3d/%d  beta=%.3f  loss=%.6f  dataset=%d",
                    iteration + 1,
                    cfg.num_iterations,
                    beta,
                    loss,
                    len(self._all_obs),  # type: ignore[arg-type]
                )

        self.on_training_end()
        return {"iteration_loss": losses, "beta": betas}

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
            Ignored (always deterministic).

        Returns
        -------
        action : np.ndarray
            Predicted action.
        info : dict
            Empty dictionary.
        """
        self._policy.eval()
        obs_t = self._to_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred = self._policy(obs_t)

        cfg: DAggerConfig = self._config  # type: ignore[assignment]
        if cfg.action_type == "discrete":
            action = pred.argmax(dim=-1).cpu().numpy().flatten()
        else:
            action = pred.cpu().numpy().flatten()

        if self._training:
            self._policy.train()
        return action, {}

    def update(self, batch: Any) -> dict[str, float]:
        """Run a single supervised-learning update on a batch.

        Parameters
        ----------
        batch : dict
            Must contain ``"obs"`` and ``"actions"`` keys.

        Returns
        -------
        dict
            ``{"dagger/loss": <float>}``.
        """
        cfg: DAggerConfig = self._config  # type: ignore[assignment]
        obs_t = self._to_tensor(batch["obs"], dtype=torch.float32).reshape(-1, self._obs_dim)
        act_t = self._to_tensor(batch["actions"], dtype=torch.float32)

        pred = self._policy(obs_t)
        if cfg.action_type == "discrete":
            loss = nn.functional.cross_entropy(pred, act_t.long().squeeze(-1))
        else:
            loss = nn.functional.mse_loss(pred, act_t)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._total_steps += 1
        return {"dagger/loss": loss.item()}

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the DAgger agent to disk."""
        extra: dict[str, Any] = {}
        if self._all_obs is not None:
            extra["dataset_size"] = len(self._all_obs)
        self._save_checkpoint(
            path,
            state_dicts={"policy": self._policy.state_dict()},
            extra=extra,
        )

    def load(self, path: str | pathlib.Path) -> None:
        """Restore the DAgger agent from a checkpoint."""
        payload = self._load_checkpoint(path)
        self._policy.load_state_dict(payload["model"]["policy"])
