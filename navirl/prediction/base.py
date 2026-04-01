from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PredictionResult:
    """Result of a trajectory prediction.

    Attributes:
        trajectories: Predicted future trajectories with shape
            ``(N_samples, T, 2)`` where *N_samples* is the number of sampled
            futures and *T* is the prediction horizon.
        probabilities: Probability (or weight) assigned to each sample,
            shape ``(N_samples,)``.  Should sum to 1.
        timestamps: Timestamps corresponding to each prediction step,
            shape ``(T,)``.
    """

    trajectories: np.ndarray  # (N_samples, T, 2)
    probabilities: np.ndarray  # (N_samples,)
    timestamps: np.ndarray  # (T,)

    # Optional extra metadata produced by the predictor.
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def num_samples(self) -> int:
        return int(self.trajectories.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.trajectories.shape[1])

    def best_trajectory(self) -> np.ndarray:
        """Return the most-likely trajectory, shape ``(T, 2)``."""
        idx = int(np.argmax(self.probabilities))
        return self.trajectories[idx]

    def mean_trajectory(self) -> np.ndarray:
        """Return the probability-weighted mean trajectory, shape ``(T, 2)``."""
        weights = self.probabilities[:, None, None]  # (N, 1, 1)
        return np.sum(self.trajectories * weights, axis=0)


class TrajectoryPredictor(ABC):
    """Abstract base class for trajectory prediction models.

    Subclasses must implement :meth:`predict` which, given observed
    trajectory data and optional environmental context, returns a
    :class:`PredictionResult` containing multiple possible future
    trajectories with associated probabilities.
    """

    @abstractmethod
    def predict(
        self,
        observed_trajectory: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict future trajectories from observations.

        Args:
            observed_trajectory: Observed positions with shape ``(T_obs, 2)``.
            context: Optional dictionary of contextual information (e.g.
                neighbor trajectories, scene image, semantic map).

        Returns:
            A :class:`PredictionResult` with multiple sampled futures.
        """
        ...
