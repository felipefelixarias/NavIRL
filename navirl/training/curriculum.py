"""
NavIRL Curriculum Learning
==========================

Curriculum learning schedulers for progressive difficulty scaling.  These
components allow training environments to start with simple scenarios and
gradually introduce harder conditions as the agent's competence improves.

Exports
-------
* :class:`DifficultyDimension` -- a single axis of difficulty.
* :class:`CurriculumScheduler` -- abstract base for schedulers.
* :class:`LinearCurriculum` -- linearly increase difficulty over *N* steps.
* :class:`PerformanceCurriculum` -- increase when performance exceeds a
  threshold.
* :class:`StagedCurriculum` -- discrete stages with promotion criteria.
* :class:`CurriculumManager` -- orchestrates multiple difficulty dimensions.
"""

from __future__ import annotations

import abc
import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Difficulty dimension
# ---------------------------------------------------------------------------


@dataclass
class DifficultyDimension:
    """A single axis of environment difficulty.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"obstacle_density"``).
    min_value:
        Value corresponding to the easiest setting.
    max_value:
        Value corresponding to the hardest setting.
    current_value:
        Starting value (defaults to *min_value*).
    """

    name: str
    min_value: float
    max_value: float
    current_value: float | None = None

    def __post_init__(self) -> None:
        if self.current_value is None:
            self.current_value = self.min_value

    def set_from_difficulty(self, difficulty: float) -> None:
        """Map a normalised difficulty in ``[0, 1]`` to this dimension's range.

        Parameters
        ----------
        difficulty:
            Value between 0 (easiest) and 1 (hardest).
        """
        difficulty = max(0.0, min(1.0, difficulty))
        self.current_value = self.min_value + difficulty * (self.max_value - self.min_value)


# ---------------------------------------------------------------------------
# Abstract scheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler(abc.ABC):
    """Abstract base class for curriculum difficulty schedulers.

    Subclasses must implement :meth:`get_difficulty` and :meth:`update`.
    """

    @abc.abstractmethod
    def get_difficulty(self, step: int, metrics: dict[str, float] | None = None) -> float:
        """Return the current difficulty level in ``[0, 1]``.

        Parameters
        ----------
        step:
            Current training timestep.
        metrics:
            Optional dictionary of recent evaluation or training metrics.
        """

    @abc.abstractmethod
    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        """Update internal state after a training step or evaluation.

        Parameters
        ----------
        step:
            Current training timestep.
        metrics:
            Optional dictionary of recent evaluation or training metrics.
        """


# ---------------------------------------------------------------------------
# Linear curriculum
# ---------------------------------------------------------------------------


class LinearCurriculum(CurriculumScheduler):
    """Linearly interpolate difficulty from *start* to *end* over *total_steps*.

    Parameters
    ----------
    start_difficulty:
        Difficulty at step 0.  Must be in ``[0, 1]``.
    end_difficulty:
        Difficulty at *total_steps*.  Must be in ``[0, 1]``.
    total_steps:
        Number of timesteps over which the ramp occurs.
    """

    def __init__(
        self,
        start_difficulty: float = 0.0,
        end_difficulty: float = 1.0,
        total_steps: int = 1_000_000,
    ) -> None:
        self.start_difficulty = float(start_difficulty)
        self.end_difficulty = float(end_difficulty)
        self.total_steps = int(total_steps)

    def get_difficulty(self, step: int, metrics: dict[str, float] | None = None) -> float:
        progress = min(step / max(self.total_steps, 1), 1.0)
        return self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty)

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        # Linear curriculum is purely step-based; nothing to update.
        pass


# ---------------------------------------------------------------------------
# Performance-based curriculum
# ---------------------------------------------------------------------------


class PerformanceCurriculum(CurriculumScheduler):
    """Adjust difficulty based on agent performance.

    Difficulty increases when the tracked metric exceeds *threshold* and
    decreases (slowly) when performance drops below it.

    Parameters
    ----------
    threshold:
        Metric value above which difficulty increases.
    increase_rate:
        Rate at which difficulty rises per :meth:`update` call.
    decrease_rate:
        Rate at which difficulty falls per :meth:`update` call.
    metric_key:
        Key in *metrics* to track.  Defaults to ``"eval/success_rate"``.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        increase_rate: float = 0.02,
        decrease_rate: float = 0.01,
        metric_key: str = "eval/success_rate",
    ) -> None:
        self.threshold = float(threshold)
        self.increase_rate = float(increase_rate)
        self.decrease_rate = float(decrease_rate)
        self.metric_key = metric_key
        self._difficulty: float = 0.0

    def get_difficulty(self, step: int, metrics: dict[str, float] | None = None) -> float:
        return self._difficulty

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if metrics is None or self.metric_key not in metrics:
            return

        value = metrics[self.metric_key]
        if value >= self.threshold:
            self._difficulty = min(1.0, self._difficulty + self.increase_rate)
            logger.debug(
                "PerformanceCurriculum: metric %.3f >= threshold %.3f -- difficulty -> %.3f",
                value,
                self.threshold,
                self._difficulty,
            )
        else:
            self._difficulty = max(0.0, self._difficulty - self.decrease_rate)
            logger.debug(
                "PerformanceCurriculum: metric %.3f < threshold %.3f -- difficulty -> %.3f",
                value,
                self.threshold,
                self._difficulty,
            )


# ---------------------------------------------------------------------------
# Staged curriculum
# ---------------------------------------------------------------------------


@dataclass
class _StageConfig:
    """Internal representation of a single curriculum stage."""

    name: str
    difficulty: float
    promotion_threshold: float
    metric_key: str = "eval/success_rate"


class StagedCurriculum(CurriculumScheduler):
    """Discrete difficulty stages with explicit promotion criteria.

    The scheduler starts at stage 0 and advances to the next stage when the
    tracked metric exceeds the stage's *promotion_threshold*.  It never
    moves backwards.

    Parameters
    ----------
    stages:
        Sequence of stage configuration dicts.  Each dict must contain:

        * ``"name"`` -- human-readable stage label.
        * ``"difficulty"`` -- difficulty value in ``[0, 1]`` for this stage.
        * ``"promotion_threshold"`` -- metric value required to advance.
        * ``"metric_key"`` (optional) -- defaults to ``"eval/success_rate"``.

    Example::

        stages = [
            {"name": "easy",   "difficulty": 0.0, "promotion_threshold": 0.8},
            {"name": "medium", "difficulty": 0.5, "promotion_threshold": 0.7},
            {"name": "hard",   "difficulty": 1.0, "promotion_threshold": 1.0},
        ]
    """

    def __init__(self, stages: Sequence[dict[str, Any]]) -> None:
        if not stages:
            raise ValueError("At least one stage must be provided.")
        self._stages: list[_StageConfig] = [
            _StageConfig(
                name=s["name"],
                difficulty=float(s["difficulty"]),
                promotion_threshold=float(s["promotion_threshold"]),
                metric_key=s.get("metric_key", "eval/success_rate"),
            )
            for s in stages
        ]
        self._current_stage: int = 0

    @property
    def current_stage(self) -> int:
        """Index of the active stage."""
        return self._current_stage

    @property
    def current_stage_name(self) -> str:
        """Name of the active stage."""
        return self._stages[self._current_stage].name

    def get_difficulty(self, step: int, metrics: dict[str, float] | None = None) -> float:
        return self._stages[self._current_stage].difficulty

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if metrics is None:
            return

        stage = self._stages[self._current_stage]
        value = metrics.get(stage.metric_key)
        if value is None:
            return

        # Check for promotion.
        if value >= stage.promotion_threshold and self._current_stage < len(self._stages) - 1:
            prev_name = stage.name
            self._current_stage += 1
            new_stage = self._stages[self._current_stage]
            logger.info(
                "StagedCurriculum: promoted from '%s' to '%s' "
                "(metric %.3f >= %.3f, new difficulty %.2f)",
                prev_name,
                new_stage.name,
                value,
                stage.promotion_threshold,
                new_stage.difficulty,
            )


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------


class CurriculumManager:
    """Manages multiple :class:`DifficultyDimension` instances controlled
    by a single :class:`CurriculumScheduler`.

    Parameters
    ----------
    dimensions:
        Difficulty dimensions to manage.
    scheduler:
        Scheduler that produces a normalised difficulty value.
    """

    def __init__(
        self,
        dimensions: Sequence[DifficultyDimension],
        scheduler: CurriculumScheduler,
    ) -> None:
        self.dimensions: list[DifficultyDimension] = [copy.deepcopy(d) for d in dimensions]
        self.scheduler = scheduler

    def update(self, step: int, metrics: dict[str, float] | None = None) -> None:
        """Update the scheduler and propagate difficulty to all dimensions."""
        self.scheduler.update(step, metrics)
        difficulty = self.scheduler.get_difficulty(step, metrics)
        for dim in self.dimensions:
            dim.set_from_difficulty(difficulty)

    def get_difficulty(self, step: int, metrics: dict[str, float] | None = None) -> float:
        """Return the current normalised difficulty from the scheduler."""
        return self.scheduler.get_difficulty(step, metrics)

    def get_env_config(self) -> dict[str, float]:
        """Return a dict mapping dimension names to their current values.

        This dict can be passed directly to an environment constructor or
        ``set_difficulty`` method to configure the next episode.
        """
        return {
            dim.name: dim.current_value  # type: ignore[dict-item]
            for dim in self.dimensions
        }


__all__ = [
    "CurriculumScheduler",
    "LinearCurriculum",
    "PerformanceCurriculum",
    "StagedCurriculum",
    "CurriculumManager",
    "DifficultyDimension",
]
