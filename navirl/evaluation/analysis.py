"""Post-hoc analysis tools: failure analysis, attention visualization, clustering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from navirl.data.trajectory import Trajectory


class AnalyzableAgent(Protocol):
    """Protocol for agents that expose internals for analysis."""

    def act(self, observation: Any) -> Any: ...


# -----------------------------------------------------------------------
# Failure analysis
# -----------------------------------------------------------------------


def failure_analysis(
    episodes: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Categorise failed episodes into failure modes.

    Each episode dict is expected to contain at least:
    - ``"success"``: bool
    - ``"timeout"``: bool (optional)
    - ``"collision"``: bool (optional)
    - ``"info"``: dict with additional details (optional)

    Parameters:
        episodes: List of episode summary dicts.

    Returns:
        Mapping from failure category name to the list of episodes in that
        category.  Categories: ``"collision"``, ``"timeout"``, ``"stuck"``,
        ``"other"``, ``"success"``.
    """
    categories: dict[str, list[dict[str, Any]]] = {
        "collision": [],
        "timeout": [],
        "stuck": [],
        "other": [],
        "success": [],
    }
    for ep in episodes:
        if ep.get("success", False):
            categories["success"].append(ep)
            continue
        if ep.get("collision", False):
            categories["collision"].append(ep)
        elif ep.get("timeout", False):
            categories["timeout"].append(ep)
        elif ep.get("info", {}).get("stuck", False):
            categories["stuck"].append(ep)
        else:
            categories["other"].append(ep)
    return categories


# -----------------------------------------------------------------------
# Attention visualization
# -----------------------------------------------------------------------


def attention_visualization(
    agent: Any,
    observation: np.ndarray,
    *,
    layer_name: str = "attention",
) -> np.ndarray:
    """Extract attention weights from an agent's forward pass.

    This is a best-effort utility that inspects the agent for a module
    named *layer_name* and hooks into its output.  If the agent does not
    expose attention weights, returns a dummy uniform distribution.

    Parameters:
        agent: Agent with a neural-network policy.
        observation: Observation array to feed through the network.
        layer_name: Name of the attention module to hook.

    Returns:
        Attention weight array (shape depends on the model architecture).
    """
    # Try to access a PyTorch model's attention layer.
    captured: list[np.ndarray] = []

    try:
        import torch

        model = getattr(agent, "model", None) or getattr(agent, "policy", None)
        if model is None:
            return np.ones(1, dtype=np.float64)

        hook_handle = None

        def _hook(_module: Any, _input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                # Many attention layers return (output, weights)
                weights = output[1] if len(output) > 1 else output[0]
            else:
                weights = output
            if isinstance(weights, torch.Tensor):
                captured.append(weights.detach().cpu().numpy())
            else:
                captured.append(np.asarray(weights))

        # Walk the model tree looking for the named layer.
        for name, module in model.named_modules():
            if layer_name in name:
                hook_handle = module.register_forward_hook(_hook)
                break

        if hook_handle is None:
            return np.ones(1, dtype=np.float64)

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            model(obs_tensor)
        hook_handle.remove()

    except ImportError:
        # PyTorch not available.
        return np.ones(1, dtype=np.float64)

    if captured:
        return captured[0]
    return np.ones(1, dtype=np.float64)


# -----------------------------------------------------------------------
# Policy entropy map
# -----------------------------------------------------------------------


def policy_entropy_map(
    agent: Any,
    state_grid: np.ndarray,
) -> np.ndarray:
    """Compute the policy entropy at each point on a 2-D state grid.

    Parameters:
        agent: Agent whose policy can be queried.  Must have a
            ``get_action_probs(obs)`` or ``predict_proba(obs)`` method.
        state_grid: Array of shape ``(N, obs_dim)`` representing a grid of states.

    Returns:
        Array of shape ``(N,)`` with entropy values (nats).
    """
    entropies: list[float] = []
    for obs in state_grid:
        probs = _get_probs(agent, obs)
        if probs is None:
            entropies.append(0.0)
            continue
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-12)
        ent = -float(np.sum(probs * np.log(probs + 1e-12)))
        entropies.append(ent)
    return np.array(entropies, dtype=np.float64)


# -----------------------------------------------------------------------
# Q-value landscape
# -----------------------------------------------------------------------


def q_value_landscape(
    agent: Any,
    state_grid: np.ndarray,
) -> np.ndarray:
    """Compute Q-values at each point on a 2-D state grid.

    Parameters:
        agent: Agent with a ``get_q_values(obs)`` method.
        state_grid: Array of shape ``(N, obs_dim)``.

    Returns:
        Array of shape ``(N, n_actions)`` or ``(N,)`` with Q-values.
    """
    q_vals: list[np.ndarray] = []
    for obs in state_grid:
        q = _get_q_values(agent, obs)
        q_vals.append(np.asarray(q, dtype=np.float64))
    return np.array(q_vals, dtype=np.float64)


# -----------------------------------------------------------------------
# Trajectory clustering
# -----------------------------------------------------------------------


def trajectory_clustering(
    trajectories: Sequence[Trajectory],
    n_clusters: int = 5,
    *,
    max_length: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Cluster trajectories based on their spatial shape.

    Trajectories are resampled to a common length, flattened, and clustered
    using k-means.

    Parameters:
        trajectories: Input trajectories.
        n_clusters: Number of clusters.
        max_length: Common resampled length for comparison.
        seed: Random seed for k-means initialisation.

    Returns:
        Array of shape ``(len(trajectories),)`` with cluster assignments.
    """
    if not trajectories:
        return np.array([], dtype=np.int64)

    features: list[np.ndarray] = []
    for traj in trajectories:
        pos = traj.positions
        # Resample to max_length points via linear interpolation.
        if len(pos) < 2:
            resampled = np.zeros((max_length, 2), dtype=np.float64)
        else:
            indices = np.linspace(0, len(pos) - 1, max_length)
            resampled = np.column_stack(
                [np.interp(indices, np.arange(len(pos)), pos[:, d]) for d in range(2)]
            )
        features.append(resampled.ravel())
    feature_matrix = np.array(features, dtype=np.float64)

    # Simple k-means (no sklearn dependency).
    labels = _kmeans(feature_matrix, n_clusters, seed=seed)
    return labels


# -----------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------


def _get_probs(agent: Any, obs: np.ndarray) -> np.ndarray | None:
    """Try various method names to get action probabilities from an agent."""
    for method_name in ("get_action_probs", "predict_proba", "action_probabilities"):
        fn = getattr(agent, method_name, None)
        if callable(fn):
            return fn(obs)
    return None


def _get_q_values(agent: Any, obs: np.ndarray) -> np.ndarray:
    """Try various method names to get Q-values from an agent."""
    for method_name in ("get_q_values", "q_values", "predict_q"):
        fn = getattr(agent, method_name, None)
        if callable(fn):
            return fn(obs)
    return np.zeros(1)


def _kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Minimal k-means implementation (Lloyd's algorithm).

    Parameters:
        X: Data matrix of shape ``(n_samples, n_features)``.
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed.

    Returns:
        Cluster labels of shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    if k <= 0:
        return np.zeros(n, dtype=np.int64)

    # Initialise centroids randomly.
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

    return labels
