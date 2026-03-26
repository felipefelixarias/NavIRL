"""Distributed consensus algorithms for multi-agent coordination.

Provides average, max, and weighted consensus protocols as well as a
distributed optimizer that combines consensus with local gradient steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


class ConsensusProtocol(ABC):
    """Abstract base class for consensus protocols.

    Subclasses must implement :meth:`step`, which takes a local value and
    the values received from neighbours and returns an updated local value.
    """

    @abstractmethod
    def step(
        self,
        local_value: np.ndarray,
        neighbor_values: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Execute one consensus step.

        Parameters:
            local_value: Current local value (array of arbitrary shape).
            neighbor_values: Values received from neighbouring agents.

        Returns:
            Updated local value after the consensus step.
        """
        ...


class AverageConsensus(ConsensusProtocol):
    """Converge to the average of all agent values.

    At each step the local value moves toward the mean of its neighbours
    by a factor of *gain*.

    Parameters:
        gain: Mixing weight in ``(0, 1]``.  A value of 1 replaces the
            local value with the neighbourhood mean each step.
    """

    def __init__(self, gain: float = 0.5) -> None:
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1].")
        self.gain = gain

    def step(
        self,
        local_value: np.ndarray,
        neighbor_values: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Execute one average-consensus step.

        Parameters:
            local_value: Current local value.
            neighbor_values: Values from neighbours.

        Returns:
            Updated local value moving toward the neighbourhood average.
        """
        local_value = np.asarray(local_value, dtype=np.float64)
        if not neighbor_values:
            return local_value

        all_values = [local_value] + [
            np.asarray(v, dtype=np.float64) for v in neighbor_values
        ]
        mean = np.mean(all_values, axis=0)
        return local_value + self.gain * (mean - local_value)


class MaxConsensus(ConsensusProtocol):
    """Converge to the element-wise maximum across all agent values.

    Each agent adopts the element-wise maximum of its own value and all
    received neighbour values.
    """

    def step(
        self,
        local_value: np.ndarray,
        neighbor_values: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Execute one max-consensus step.

        Parameters:
            local_value: Current local value.
            neighbor_values: Values from neighbours.

        Returns:
            Element-wise maximum of local and all neighbour values.
        """
        local_value = np.asarray(local_value, dtype=np.float64)
        result = local_value.copy()
        for v in neighbor_values:
            result = np.maximum(result, np.asarray(v, dtype=np.float64))
        return result


class WeightedConsensus(ConsensusProtocol):
    """Weighted average consensus using Metropolis-Hastings weights.

    The Metropolis weight for edge ``(i, j)`` is::

        w_ij = 1 / (1 + max(d_i, d_j))

    where ``d_i`` and ``d_j`` are the degrees of the two endpoints.  The
    self-weight is ``1 - sum(w_ij for j in neighbours)``.

    Parameters:
        degree: Degree (number of neighbours) of the local agent.
        neighbor_degrees: Sequence of degrees for each neighbour, in the
            same order as the ``neighbor_values`` passed to :meth:`step`.
    """

    def __init__(
        self,
        degree: int,
        neighbor_degrees: Sequence[int],
    ) -> None:
        self.degree = degree
        self.neighbor_degrees = list(neighbor_degrees)

        # Pre-compute Metropolis weights
        self._weights: List[float] = []
        for nd in self.neighbor_degrees:
            self._weights.append(1.0 / (1.0 + max(degree, nd)))
        self._self_weight = 1.0 - sum(self._weights)

    @property
    def weights(self) -> List[float]:
        """Metropolis weights for each neighbour."""
        return list(self._weights)

    @property
    def self_weight(self) -> float:
        """Weight applied to the local agent's own value."""
        return self._self_weight

    def step(
        self,
        local_value: np.ndarray,
        neighbor_values: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Execute one weighted consensus step.

        Parameters:
            local_value: Current local value.
            neighbor_values: Values from neighbours (same order as
                *neighbor_degrees* given at construction).

        Returns:
            Updated local value.

        Raises:
            ValueError: If the number of neighbour values does not match
                the configured number of neighbours.
        """
        local_value = np.asarray(local_value, dtype=np.float64)
        if len(neighbor_values) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} neighbour values, "
                f"got {len(neighbor_values)}."
            )

        result = self._self_weight * local_value
        for w, v in zip(self._weights, neighbor_values):
            result = result + w * np.asarray(v, dtype=np.float64)
        return result


class ConsensusOptimizer:
    """Distributed optimization via consensus combined with gradient descent.

    Each agent maintains a local copy of the optimization variable.  At
    each step the agent:

    1. Takes a local gradient step on its own objective.
    2. Performs a consensus step with neighbours to align variables.

    Parameters:
        consensus: A :class:`ConsensusProtocol` instance for the mixing step.
        lr: Learning rate for local gradient steps.
    """

    def __init__(
        self,
        consensus: ConsensusProtocol,
        lr: float = 0.01,
    ) -> None:
        self.consensus = consensus
        self.lr = lr

    def step(
        self,
        local_value: np.ndarray,
        local_gradient: np.ndarray,
        neighbor_values: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Execute one distributed optimisation step.

        Parameters:
            local_value: Current local variable estimate.
            local_gradient: Gradient of the local objective at *local_value*.
            neighbor_values: Current variable estimates from neighbours.

        Returns:
            Updated local variable estimate.
        """
        local_value = np.asarray(local_value, dtype=np.float64)
        local_gradient = np.asarray(local_gradient, dtype=np.float64)

        # Gradient descent
        updated = local_value - self.lr * local_gradient

        # Consensus mixing
        updated = self.consensus.step(updated, neighbor_values)

        return updated

    def run(
        self,
        initial_value: np.ndarray,
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        get_neighbors_fn: Callable[[np.ndarray], Sequence[np.ndarray]],
        num_steps: int = 100,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """Run the distributed optimisation loop for a single agent.

        Parameters:
            initial_value: Starting variable estimate.
            gradient_fn: Callable that returns the local gradient given the
                current value.
            get_neighbors_fn: Callable that returns neighbour values given
                the current local value (for simulation / testing).
            num_steps: Maximum number of iterations.
            tolerance: Early-stopping tolerance on the gradient norm.

        Returns:
            Final optimised variable estimate.
        """
        value = np.asarray(initial_value, dtype=np.float64)
        for _ in range(num_steps):
            grad = gradient_fn(value)
            if float(np.linalg.norm(grad)) < tolerance:
                break
            neighbors = get_neighbors_fn(value)
            value = self.step(value, grad, neighbors)
        return value
