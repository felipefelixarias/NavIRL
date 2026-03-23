"""
NavIRL DQN Agent
================

Deep Q-Network agent and variants for discrete-action pedestrian navigation.

Implements the following DQN family of algorithms:

* **Vanilla DQN** -- fixed target network with periodic hard updates
  (Mnih et al., 2015).
* **Double DQN** -- decouples action selection (online) from evaluation
  (target) to reduce overestimation bias (van Hasselt et al., 2016).
* **Dueling DQN** -- separate value and advantage streams via
  :class:`~navirl.agents.networks.DuelingMLP` (Wang et al., 2016).
* **NoisyNet DQN** -- parameter-space exploration via
  :class:`~navirl.agents.networks.NoisyMLP` (Fortunato et al., 2018).
* **Prioritized Experience Replay** -- compatible with
  :class:`~navirl.training.PrioritizedReplayBuffer` (Schaul et al., 2016).

All variants are controlled by a single :class:`DQNConfig` dataclass and
share a unified :class:`DQNAgent` implementation.

Classes
-------
EpsilonSchedule
    Linear epsilon-greedy decay schedule.
DQNConfig
    Hyperparameter dataclass for the DQN agent family.
DQNAgent
    Concrete agent implementing the DQN algorithm and its variants.
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

from navirl.agents.base import BaseAgent, HyperParameters, MetricsCallback
from navirl.agents.networks import DuelingMLP, MLP, NoisyMLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------


class EpsilonSchedule:
    """Linear epsilon-greedy decay schedule.

    Epsilon is linearly annealed from *start* to *end* over a fixed number of
    steps, then held constant at *end* for the remainder of training.

    Parameters
    ----------
    start : float
        Initial exploration rate (typically 1.0).
    end : float
        Final exploration rate (e.g. 0.05).
    decay_steps : int
        Number of steps over which the linear decay occurs.
    """

    def __init__(self, start: float = 1.0, end: float = 0.05, decay_steps: int = 100_000) -> None:
        if not (0.0 <= end <= start <= 1.0):
            raise ValueError(
                f"Require 0 <= end <= start <= 1, got start={start}, end={end}"
            )
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be positive, got {decay_steps}")
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def value(self, step: int) -> float:
        """Return the epsilon value at the given training step.

        Parameters
        ----------
        step : int
            Current training step (0-indexed).

        Returns
        -------
        float
            Epsilon in ``[end, start]``.
        """
        fraction = min(1.0, step / self.decay_steps)
        return self.start + fraction * (self.end - self.start)

    def __repr__(self) -> str:
        return (
            f"EpsilonSchedule(start={self.start}, end={self.end}, "
            f"decay_steps={self.decay_steps})"
        )


# ---------------------------------------------------------------------------
# DQN config
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig(HyperParameters):
    """Hyperparameter configuration for :class:`DQNAgent`.

    Attributes
    ----------
    lr : float
        Learning rate for the Adam optimiser.
    gamma : float
        Discount factor.
    batch_size : int
        Mini-batch size sampled from the replay buffer.
    target_update_freq : int
        Number of gradient steps between hard target-network updates.
    eps_start : float
        Initial epsilon for the epsilon-greedy schedule.
    eps_end : float
        Final (minimum) epsilon.
    eps_decay_steps : int
        Number of steps over which epsilon is linearly annealed.
    hidden_dims : tuple of int
        Hidden-layer widths for the Q-network.
    activation : str
        Activation function name (passed to the network constructor).
    double_dqn : bool
        If ``True``, use Double-DQN action selection / evaluation split.
    dueling : bool
        If ``True``, use a :class:`DuelingMLP` architecture.
    noisy : bool
        If ``True``, use a :class:`NoisyMLP` for parameter-space exploration
        (epsilon-greedy is disabled when noisy is active).
    prioritized : bool
        If ``True``, the agent expects batches from a
        :class:`PrioritizedReplayBuffer` and returns updated priorities.
    n_step : int
        N-step return horizon (affects the effective discount: gamma^n).
    max_grad_norm : float
        Maximum gradient norm for gradient clipping.
    """

    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    target_update_freq: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    hidden_dims: Tuple[int, ...] = (128, 128)
    activation: str = "relu"
    double_dqn: bool = True
    dueling: bool = False
    noisy: bool = False
    prioritized: bool = False
    n_step: int = 1
    max_grad_norm: float = 10.0


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with support for Double, Dueling, Noisy, and
    Prioritized variants.

    The concrete variant is selected purely through :class:`DQNConfig` flags --
    no subclassing required.

    Parameters
    ----------
    config : DQNConfig
        Agent configuration.
    observation_space : gymnasium.spaces.Space
        Environment observation space (used to infer ``obs_dim``).
    action_space : gymnasium.spaces.Space
        Environment action space (must be ``Discrete``).
    device : str or torch.device
        Compute device.
    seed : int or None
        Random seed for reproducibility.
    metrics_callback : callable or None
        Optional ``(metrics_dict, step) -> None`` hook.
    """

    def __init__(
        self,
        config: DQNConfig,
        observation_space: Any,
        action_space: Any,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
        metrics_callback: Optional[MetricsCallback] = None,
    ) -> None:
        super().__init__(config, observation_space, action_space, device, seed, metrics_callback)

        # Validate discrete action space
        self._num_actions: int = int(action_space.n)
        obs_shape = observation_space.shape
        self._obs_dim: int = int(np.prod(obs_shape))

        # Build Q-networks ------------------------------------------------
        self._q_net = self._build_network()
        self._target_net = self._build_network()
        self._hard_update(self._target_net, self._q_net)
        # Freeze target parameters (no gradient tracking)
        for p in self._target_net.parameters():
            p.requires_grad = False

        self._q_net.to(self._device)
        self._target_net.to(self._device)

        self._modules.extend([self._q_net, self._target_net])

        # Optimiser --------------------------------------------------------
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=config.lr)
        self._optimizers["q_optimizer"] = self._optimizer

        # Epsilon schedule (ignored when noisy nets are active) ------------
        self._epsilon_schedule = EpsilonSchedule(
            start=config.eps_start,
            end=config.eps_end,
            decay_steps=config.eps_decay_steps,
        )

        # Internal step counter for target updates -------------------------
        self._update_step: int = 0

        # Effective discount for n-step returns ----------------------------
        self._gamma_n: float = config.gamma ** config.n_step

        # Random state for epsilon-greedy ----------------------------------
        self._rng = np.random.RandomState(seed)

        self._log_module_summary("q_net", self._q_net)
        self._logger.info(
            "DQN variant: double=%s  dueling=%s  noisy=%s  prioritized=%s  n_step=%d",
            config.double_dqn,
            config.dueling,
            config.noisy,
            config.prioritized,
            config.n_step,
        )

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self) -> nn.Module:
        """Instantiate the Q-network according to the current config."""
        cfg: DQNConfig = self._config  # type: ignore[assignment]

        if cfg.noisy:
            return NoisyMLP(
                input_dim=self._obs_dim,
                output_dim=self._num_actions,
                hidden_dims=cfg.hidden_dims,
                activation=cfg.activation,
            )
        if cfg.dueling:
            return DuelingMLP(
                input_dim=self._obs_dim,
                num_actions=self._num_actions,
                hidden_dims=cfg.hidden_dims,
                activation=cfg.activation,
            )
        return MLP(
            input_dim=self._obs_dim,
            output_dim=self._num_actions,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select an action using epsilon-greedy (or NoisyNet) exploration.

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation.
        deterministic : bool
            If ``True``, always act greedily (no exploration noise).

        Returns
        -------
        action : np.ndarray
            Scalar action array compatible with ``env.step``.
        info : dict
            Contains ``"q_values"`` and ``"epsilon"``.
        """
        cfg: DQNConfig = self._config  # type: ignore[assignment]
        epsilon = self._epsilon_schedule.value(self._total_steps)

        # Noisy nets handle exploration internally; skip epsilon-greedy
        use_epsilon = not cfg.noisy and not deterministic

        if use_epsilon and self._rng.random() < epsilon:
            action = self._rng.randint(self._num_actions)
            return np.array(action), {"q_values": None, "epsilon": epsilon}

        # Greedy action from Q-network
        obs_t = self._to_tensor(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self._q_net(obs_t)  # (1, num_actions)

        action = int(q_values.argmax(dim=-1).item())
        q_np = self._to_numpy(q_values.squeeze(0))

        return np.array(action), {"q_values": q_np, "epsilon": epsilon}

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update(self, batch: Any) -> Dict[str, float]:
        """Perform a single gradient step on a batch of transitions.

        Parameters
        ----------
        batch
            For standard replay: ``dict`` with keys
            ``obs, actions, rewards, next_obs, dones``.
            For prioritized replay: ``(dict, importance_weights, tree_indices)``.

        Returns
        -------
        dict
            Scalar metrics: ``q_loss``, ``q_mean``, ``epsilon``, and
            optionally ``grad_norm``.
        """
        cfg: DQNConfig = self._config  # type: ignore[assignment]

        # Unpack prioritized vs. uniform replay ----------------------------
        if cfg.prioritized:
            batch_dict, importance_weights, tree_indices = batch
            weights_t = self._to_tensor(
                np.asarray(importance_weights, dtype=np.float32)
            )
        else:
            batch_dict = batch
            weights_t = None
            tree_indices = None

        obs = self._to_tensor(batch_dict["obs"].astype(np.float32))
        actions = self._to_tensor(batch_dict["actions"].astype(np.int64)).long()
        rewards = self._to_tensor(batch_dict["rewards"].astype(np.float32))
        next_obs = self._to_tensor(batch_dict["next_obs"].astype(np.float32))
        dones = self._to_tensor(batch_dict["dones"].astype(np.float32))

        # Flatten observations if needed
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)
            next_obs = next_obs.view(next_obs.size(0), -1)

        # Ensure actions have correct shape: (B,) -> (B, 1)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        # Current Q-values for chosen actions ------------------------------
        q_all = self._q_net(obs)                       # (B, num_actions)
        q_values = q_all.gather(dim=1, index=actions)  # (B, 1)

        # Target Q-values --------------------------------------------------
        with torch.no_grad():
            if cfg.double_dqn:
                # Action selection with online network
                next_q_online = self._q_net(next_obs)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # Evaluation with target network
                next_q_target = self._target_net(next_obs)
                next_q = next_q_target.gather(dim=1, index=next_actions)
            else:
                # Standard DQN: max Q from target
                next_q_target = self._target_net(next_obs)
                next_q = next_q_target.max(dim=1, keepdim=True).values

            targets = rewards + self._gamma_n * next_q * (1.0 - dones)

        # Compute element-wise Huber loss ----------------------------------
        td_errors = q_values - targets
        elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction="none")

        if weights_t is not None:
            if weights_t.dim() == 1:
                weights_t = weights_t.unsqueeze(-1)
            loss = (elementwise_loss * weights_t).mean()
        else:
            loss = elementwise_loss.mean()

        # Gradient step ----------------------------------------------------
        self._optimizer.zero_grad()
        loss.backward()
        grad_norm = self._clip_grad_norm(self._q_net.parameters(), cfg.max_grad_norm)
        self._optimizer.step()

        # Reset NoisyNet noise after each update ---------------------------
        if cfg.noisy:
            self._q_net.reset_noise()   # type: ignore[attr-defined]
            self._target_net.reset_noise()  # type: ignore[attr-defined]

        # Periodic hard target update --------------------------------------
        self._update_step += 1
        if self._update_step % cfg.target_update_freq == 0:
            self._hard_update(self._target_net, self._q_net)
            self._logger.debug(
                "Hard target update at update_step=%d", self._update_step
            )

        # Build metrics ----------------------------------------------------
        epsilon = self._epsilon_schedule.value(self._total_steps)
        metrics: Dict[str, float] = {
            "q_loss": float(loss.item()),
            "q_mean": float(q_values.mean().item()),
            "epsilon": epsilon,
            "grad_norm": float(grad_norm),
        }

        # Priority updates for PER -----------------------------------------
        if cfg.prioritized and tree_indices is not None:
            td_abs = td_errors.detach().abs().squeeze(-1).cpu().numpy()
            metrics["_td_errors"] = float(td_abs.mean())
            # Return priority info via a special key so the training loop
            # can call buffer.update_priorities(tree_indices, td_abs).
            metrics["_tree_indices"] = tree_indices  # type: ignore[assignment]
            metrics["_new_priorities"] = td_abs  # type: ignore[assignment]

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save the agent to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory or file path for the checkpoint.
        """
        state_dicts = {
            "q_net": self._q_net.state_dict(),
            "target_net": self._target_net.state_dict(),
        }
        extra = {
            "update_step": self._update_step,
        }
        self._save_checkpoint(path, state_dicts, extra=extra)

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """Restore the agent from a checkpoint.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the checkpoint file.
        """
        payload = self._load_checkpoint(path)
        model_states = payload.get("model", {})

        if "q_net" in model_states:
            self._q_net.load_state_dict(model_states["q_net"])
        if "target_net" in model_states:
            self._target_net.load_state_dict(model_states["target_net"])

        extra = payload.get("meta", {}).get("extra", {})
        self._update_step = extra.get("update_step", 0)

        self._logger.info(
            "DQN agent restored  (update_step=%d, total_steps=%d)",
            self._update_step,
            self._total_steps,
        )
