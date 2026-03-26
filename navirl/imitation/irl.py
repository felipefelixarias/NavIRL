"""
Maximum Entropy Inverse Reinforcement Learning
===============================================

Learns a linear reward function r(s) = theta^T * phi(s) by matching feature
expectations between the expert policy and the learned policy, under the
maximum-entropy framework.

The algorithm alternates between:
1. Computing the current policy via forward RL under the current reward.
2. Updating the reward parameters using the gradient:
   grad = feature_expectations_expert - feature_expectations_policy.

Reference:
    Ziebart et al. "Maximum Entropy Inverse Reinforcement Learning",
    AAAI 2008.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from navirl.agents.base import HyperParameters

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["MaxEntIRLConfig", "MaxEntIRL"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MaxEntIRLConfig(HyperParameters):
    """Hyperparameters for Maximum Entropy IRL.

    Attributes:
        lr: Learning rate for reward parameter updates.
        num_iterations: Number of IRL outer-loop iterations.
        feature_dim: Dimensionality of the state feature vector phi(s).
        discount: Discount factor for computing feature expectations.
        temperature: Soft-value temperature for the Boltzmann policy.
    """

    lr: float = 1e-2
    num_iterations: int = 100
    feature_dim: int = 64
    discount: float = 0.99
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# MaxEntIRL
# ---------------------------------------------------------------------------


class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning.

    Learns a linear reward function ``r(s) = theta^T * phi(s)`` by matching
    feature expectations between the expert demonstrations and a policy
    computed via forward RL.

    Parameters
    ----------
    config : MaxEntIRLConfig
        IRL hyperparameters.
    feature_fn : callable
        ``(observation) -> feature_vector``.  Maps a raw observation to a
        fixed-length feature vector of dimension ``config.feature_dim``.
    forward_rl_fn : callable, optional
        ``(reward_fn, num_steps) -> trajectories``.  Runs forward RL under
        the given reward function and returns trajectories (list of lists of
        observations).  If not provided, the user must call
        :meth:`update_step` manually with pre-computed policy features.
    """

    def __init__(
        self,
        config: MaxEntIRLConfig,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        forward_rl_fn: Optional[
            Callable[[Callable[[np.ndarray], float], int], List[List[np.ndarray]]]
        ] = None,
    ) -> None:
        self._config = config
        self._feature_fn = feature_fn
        self._forward_rl_fn = forward_rl_fn

        # Linear reward parameters
        self._theta = np.zeros(config.feature_dim, dtype=np.float64)

        self._iteration: int = 0
        self._history: List[Dict[str, float]] = []

        logger.info(
            "MaxEntIRL initialised  |  feature_dim=%d  lr=%.4f  iterations=%d",
            config.feature_dim,
            config.lr,
            config.num_iterations,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def theta(self) -> np.ndarray:
        """Current reward parameters."""
        return self._theta.copy()

    @property
    def history(self) -> List[Dict[str, float]]:
        """Training history (list of per-iteration metrics)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------

    def reward(self, observation: np.ndarray) -> float:
        """Compute the learned reward for a single observation.

        Parameters
        ----------
        observation : np.ndarray
            Raw observation.

        Returns
        -------
        float
            Scalar reward ``theta^T * phi(s)``.
        """
        phi = self._feature_fn(observation)
        return float(self._theta @ phi)

    def reward_batch(self, observations: np.ndarray) -> np.ndarray:
        """Compute the learned reward for a batch of observations.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations ``(N, ...)``.

        Returns
        -------
        np.ndarray
            Reward vector ``(N,)``.
        """
        features = np.array(
            [self._feature_fn(obs) for obs in observations], dtype=np.float64
        )
        return features @ self._theta

    # ------------------------------------------------------------------
    # Feature expectations
    # ------------------------------------------------------------------

    def compute_feature_expectations(
        self,
        trajectories: List[List[np.ndarray]],
    ) -> np.ndarray:
        """Compute discounted feature expectations from trajectories.

        Parameters
        ----------
        trajectories : list of list of np.ndarray
            Each trajectory is a list of observations.

        Returns
        -------
        np.ndarray
            Mean feature expectations ``(feature_dim,)``.
        """
        cfg = self._config
        all_fe = np.zeros(cfg.feature_dim, dtype=np.float64)
        n_traj = len(trajectories)

        for traj in trajectories:
            discount = 1.0
            for obs in traj:
                phi = self._feature_fn(obs)
                all_fe += discount * phi
                discount *= cfg.discount
        return all_fe / max(n_traj, 1)

    @staticmethod
    def compute_feature_expectations_from_features(
        feature_trajectories: List[np.ndarray],
        discount: float = 0.99,
    ) -> np.ndarray:
        """Compute feature expectations from pre-computed feature arrays.

        Parameters
        ----------
        feature_trajectories : list of np.ndarray
            Each element is a ``(T, feature_dim)`` array.
        discount : float
            Discount factor.

        Returns
        -------
        np.ndarray
            Mean feature expectations.
        """
        all_fe: Optional[np.ndarray] = None
        n_traj = len(feature_trajectories)

        for feat_traj in feature_trajectories:
            T = feat_traj.shape[0]
            discounts = discount ** np.arange(T, dtype=np.float64)
            fe = (feat_traj * discounts[:, None]).sum(axis=0)
            if all_fe is None:
                all_fe = fe
            else:
                all_fe = all_fe + fe

        if all_fe is None:
            raise ValueError("No trajectories provided.")
        return all_fe / max(n_traj, 1)

    # ------------------------------------------------------------------
    # Single update step
    # ------------------------------------------------------------------

    def update_step(
        self,
        expert_features: np.ndarray,
        policy_features: np.ndarray,
    ) -> Dict[str, float]:
        """Perform a single gradient step on the reward parameters.

        Parameters
        ----------
        expert_features : np.ndarray
            Expert feature expectations ``(feature_dim,)``.
        policy_features : np.ndarray
            Current-policy feature expectations ``(feature_dim,)``.

        Returns
        -------
        dict
            Metrics including ``"grad_norm"`` and ``"theta_norm"``.
        """
        cfg = self._config
        grad = expert_features - policy_features
        self._theta += cfg.lr * grad

        grad_norm = float(np.linalg.norm(grad))
        theta_norm = float(np.linalg.norm(self._theta))

        self._iteration += 1
        metrics = {
            "irl/grad_norm": grad_norm,
            "irl/theta_norm": theta_norm,
            "irl/iteration": float(self._iteration),
        }
        self._history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        expert_trajectories: List[List[np.ndarray]],
        forward_rl_steps: int = 10000,
        *,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """Run the full MaxEnt IRL training loop.

        Parameters
        ----------
        expert_trajectories : list of list of np.ndarray
            Expert demonstrations (each trajectory is a list of observations).
        forward_rl_steps : int
            Number of environment steps for the forward RL sub-problem at
            each iteration.
        verbose : bool
            Whether to log progress.

        Returns
        -------
        list of dict
            Per-iteration metrics.

        Raises
        ------
        ValueError
            If ``forward_rl_fn`` was not provided at construction time.
        """
        if self._forward_rl_fn is None:
            raise ValueError(
                "forward_rl_fn must be provided to use the automatic training "
                "loop.  Alternatively, call update_step() manually."
            )

        cfg = self._config

        # Compute expert feature expectations once
        expert_fe = self.compute_feature_expectations(expert_trajectories)

        all_metrics: List[Dict[str, float]] = []

        for it in range(cfg.num_iterations):
            # Forward RL with current reward
            policy_trajectories = self._forward_rl_fn(self.reward, forward_rl_steps)
            policy_fe = self.compute_feature_expectations(policy_trajectories)

            # Gradient step
            metrics = self.update_step(expert_fe, policy_fe)
            all_metrics.append(metrics)

            if verbose and (it % max(1, cfg.num_iterations // 20) == 0 or it == cfg.num_iterations - 1):
                logger.info(
                    "MaxEntIRL iter %3d/%d  grad_norm=%.6f  theta_norm=%.6f",
                    it + 1,
                    cfg.num_iterations,
                    metrics["irl/grad_norm"],
                    metrics["irl/theta_norm"],
                )

        return all_metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable state dictionary."""
        return {
            "theta": self._theta.copy(),
            "iteration": self._iteration,
            "config": self._config.to_dict(),
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        """Load state from a dictionary.

        Parameters
        ----------
        d : dict
            State dictionary as returned by :meth:`state_dict`.
        """
        self._theta = np.array(d["theta"], dtype=np.float64)
        self._iteration = int(d.get("iteration", 0))

    def save(self, path: str) -> None:
        """Save reward parameters to a numpy file.

        Parameters
        ----------
        path : str
            File path (will be saved as ``.npz``).
        """
        np.savez(
            path,
            theta=self._theta,
            iteration=np.array(self._iteration),
        )
        logger.info("MaxEntIRL saved to %s", path)

    def load(self, path: str) -> None:
        """Load reward parameters from a numpy file.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.
        """
        data = np.load(path)
        self._theta = data["theta"].astype(np.float64)
        self._iteration = int(data["iteration"])
        logger.info("MaxEntIRL loaded from %s (iteration=%d)", path, self._iteration)

    def __repr__(self) -> str:
        return (
            f"MaxEntIRL(feature_dim={self._config.feature_dim}, "
            f"iteration={self._iteration}, "
            f"theta_norm={np.linalg.norm(self._theta):.4f})"
        )
