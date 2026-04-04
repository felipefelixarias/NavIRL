"""Learning rate and hyperparameter schedules for NavIRL training."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

# Exports: Schedule, LinearSchedule, CosineAnnealingSchedule, StepSchedule,
#          ExponentialSchedule, CyclicSchedule, WarmupSchedule,
#          ReduceOnPlateauSchedule, PolynomialSchedule, OneCycleSchedule,
#          CompositeSchedule, ExplorationSchedule

__all__ = [
    "CompositeSchedule",
    "CosineAnnealingSchedule",
    "CyclicSchedule",
    "ExplorationSchedule",
    "ExponentialSchedule",
    "LinearSchedule",
    "OneCycleSchedule",
    "PolynomialSchedule",
    "ReduceOnPlateauSchedule",
    "Schedule",
    "StepSchedule",
    "WarmupSchedule",
]


class Schedule(ABC):
    """Abstract base class for value schedules.

    A schedule maps a *step* (integer) to a scalar float value.
    """

    @abstractmethod
    def value(self, step: int) -> float:
        """Return the scheduled value at the given *step*."""

    def __call__(self, step: int) -> float:
        return self.value(step)


class LinearSchedule(Schedule):
    """Linearly interpolate from *start* to *end* over *total_steps*.

    After *total_steps* the value is clamped to *end*.

    Parameters
    ----------
    start:
        Initial value.
    end:
        Final value.
    total_steps:
        Number of steps over which the transition happens.
    """

    def __init__(self, start: float, end: float, total_steps: int) -> None:
        self.start = start
        self.end = end
        self.total_steps = max(total_steps, 1)

    def value(self, step: int) -> float:
        fraction = min(step / self.total_steps, 1.0)
        return self.start + (self.end - self.start) * fraction


class CosineAnnealingSchedule(Schedule):
    """Cosine annealing from *max_value* to *min_value* with optional warmup.

    Parameters
    ----------
    max_value:
        Peak value (reached after warmup).
    min_value:
        Trough value at the end of the cycle.
    total_steps:
        Total number of steps including warmup.
    warmup_steps:
        Number of linear warmup steps before cosine decay begins.
    """

    def __init__(
        self,
        max_value: float,
        min_value: float = 0.0,
        total_steps: int = 1,
        warmup_steps: int = 0,
    ) -> None:
        self.max_value = max_value
        self.min_value = min_value
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps

    def value(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup from min_value to max_value.
            return self.min_value + (self.max_value - self.min_value) * (
                step / max(self.warmup_steps, 1)
            )
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 0:
            return self.min_value
        progress = min((step - self.warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_value + (self.max_value - self.min_value) * cosine


class StepSchedule(Schedule):
    """Multiply the value by *factor* every *step_size* steps.

    Parameters
    ----------
    initial_value:
        Value at step 0.
    factor:
        Multiplicative factor applied at each milestone.
    step_size:
        Number of steps between multiplications.
    min_value:
        Floor for the value.
    """

    def __init__(
        self,
        initial_value: float,
        factor: float = 0.1,
        step_size: int = 10_000,
        min_value: float = 0.0,
    ) -> None:
        self.initial_value = initial_value
        self.factor = factor
        self.step_size = max(step_size, 1)
        self.min_value = min_value

    def value(self, step: int) -> float:
        n_drops = step // self.step_size
        val = self.initial_value * (self.factor**n_drops)
        return max(val, self.min_value)


class ExponentialSchedule(Schedule):
    """Exponential decay: ``initial_value * decay_rate ^ (step / decay_steps)``.

    Parameters
    ----------
    initial_value:
        Starting value.
    decay_rate:
        Base of the exponent (e.g. 0.99).
    decay_steps:
        Number of steps for one full decay period.
    min_value:
        Floor for the value.
    """

    def __init__(
        self,
        initial_value: float,
        decay_rate: float = 0.99,
        decay_steps: int = 1,
        min_value: float = 0.0,
    ) -> None:
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.decay_steps = max(decay_steps, 1)
        self.min_value = min_value

    def value(self, step: int) -> float:
        val = self.initial_value * (self.decay_rate ** (step / self.decay_steps))
        return max(val, self.min_value)


class CyclicSchedule(Schedule):
    """Cyclic learning rate schedule (triangular or cosine).

    Parameters
    ----------
    base_value:
        Lower bound of the cycle.
    max_value:
        Upper bound of the cycle.
    cycle_steps:
        Number of steps per full cycle.
    mode:
        ``"triangular"`` for linear ramp up/down, ``"cosine"`` for smooth
        cosine oscillation.
    """

    def __init__(
        self,
        base_value: float,
        max_value: float,
        cycle_steps: int = 10_000,
        mode: str = "triangular",
    ) -> None:
        self.base_value = base_value
        self.max_value = max_value
        self.cycle_steps = max(cycle_steps, 1)
        self.mode = mode

    def value(self, step: int) -> float:
        cycle_pos = (step % self.cycle_steps) / self.cycle_steps

        if self.mode == "cosine":
            # Cosine oscillation: 1 at pos=0, 0 at pos=0.5, 1 at pos=1
            scale = 0.5 * (1.0 + math.cos(2.0 * math.pi * cycle_pos))
        else:
            # Triangular mode repeats a shorter up/down ramp twice per cycle:
            # base -> max -> base -> max -> base.
            half_cycle = self.cycle_steps / 2.0
            half_pos = (step % half_cycle) / half_cycle
            if half_pos <= 0.5:
                scale = 2.0 * half_pos
            else:
                scale = 2.0 * (1.0 - half_pos)

        return self.base_value + (self.max_value - self.base_value) * scale


class WarmupSchedule(Schedule):
    """Wrap another schedule with a linear warmup phase.

    During the warmup the value increases linearly from *warmup_start* to the
    inner schedule's value at step *warmup_steps*.  After warmup, the inner
    schedule takes over.

    Parameters
    ----------
    schedule:
        The inner :class:`Schedule` that runs after warmup.
    warmup_steps:
        Number of warmup steps.
    warmup_start:
        Value at step 0.
    """

    def __init__(
        self,
        schedule: Schedule,
        warmup_steps: int,
        warmup_start: float = 0.0,
    ) -> None:
        self.schedule = schedule
        self.warmup_steps = max(warmup_steps, 1)
        self.warmup_start = warmup_start

    def value(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.schedule.value(step)
        # Linear interpolation towards the inner schedule's value at the
        # warmup boundary.
        target = self.schedule.value(self.warmup_steps)
        fraction = step / self.warmup_steps
        return self.warmup_start + (target - self.warmup_start) * fraction


class ReduceOnPlateauSchedule(Schedule):
    """Reduce the value when a monitored metric plateaus.

    Unlike most schedules this one is *stateful*: the caller must invoke
    :meth:`report` whenever a new metric reading is available.

    Parameters
    ----------
    initial_value:
        Starting value.
    factor:
        Multiplicative factor applied on each reduction.
    patience:
        Number of :meth:`report` calls with no improvement before reducing.
    min_value:
        Floor for the value.
    threshold:
        Minimum change to qualify as an improvement.
    mode:
        ``"max"`` means higher metric is better; ``"min"`` means lower is
        better.
    """

    def __init__(
        self,
        initial_value: float,
        factor: float = 0.5,
        patience: int = 10,
        min_value: float = 1e-7,
        threshold: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.initial_value = initial_value
        self.factor = factor
        self.patience = patience
        self.min_value = min_value
        self.threshold = threshold
        self.mode = mode

        self._current_value = initial_value
        self._best_metric: float | None = None
        self._no_improvement_count: int = 0

    def report(self, metric: float) -> None:
        """Report a new metric reading."""
        if self._best_metric is None:
            self._best_metric = metric
            return

        improved = (
            metric > self._best_metric + self.threshold
            if self.mode == "max"
            else metric < self._best_metric - self.threshold
        )

        if improved:
            self._best_metric = metric
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            self._current_value = max(self._current_value * self.factor, self.min_value)
            self._no_improvement_count = 0

    def value(self, step: int) -> float:
        return self._current_value


class PolynomialSchedule(Schedule):
    """Polynomial decay from *initial_value* to *end_value*.

    ``value = (initial - end) * (1 - step/total_steps)^power + end``

    Parameters
    ----------
    initial_value:
        Starting value.
    end_value:
        Final value after *total_steps*.
    total_steps:
        Duration of the decay.
    power:
        Exponent of the polynomial.
    """

    def __init__(
        self,
        initial_value: float,
        end_value: float = 0.0,
        total_steps: int = 1,
        power: float = 1.0,
    ) -> None:
        self.initial_value = initial_value
        self.end_value = end_value
        self.total_steps = max(total_steps, 1)
        self.power = power

    def value(self, step: int) -> float:
        fraction = min(step / self.total_steps, 1.0)
        return (self.initial_value - self.end_value) * (
            (1.0 - fraction) ** self.power
        ) + self.end_value


class OneCycleSchedule(Schedule):
    """Super-convergence 1cycle policy.

    The learning rate ramps up from *initial_value* to *max_value* over the
    first *pct_start* fraction of training, then cosine-anneals back down to
    *final_value*.

    Parameters
    ----------
    max_value:
        Peak learning rate.
    total_steps:
        Total training steps.
    initial_value:
        Starting learning rate (beginning of warm-up).
    final_value:
        Final learning rate at the end of training.
    pct_start:
        Fraction of total steps spent ramping up.
    """

    def __init__(
        self,
        max_value: float,
        total_steps: int,
        initial_value: float | None = None,
        final_value: float | None = None,
        pct_start: float = 0.3,
    ) -> None:
        self.max_value = max_value
        self.total_steps = max(total_steps, 1)
        self.initial_value = initial_value if initial_value is not None else max_value / 25.0
        self.final_value = final_value if final_value is not None else self.initial_value / 1e4
        self.pct_start = pct_start

    def value(self, step: int) -> float:
        up_steps = int(self.total_steps * self.pct_start)
        if step < up_steps:
            # Cosine ramp up
            progress = step / max(up_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
            return self.initial_value + (self.max_value - self.initial_value) * cosine
        else:
            # Cosine ramp down
            down_steps = self.total_steps - up_steps
            progress = (step - up_steps) / max(down_steps, 1)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_value + (self.max_value - self.final_value) * cosine


class CompositeSchedule(Schedule):
    """Chain multiple schedules sequentially.

    Each entry in *phases* is a ``(schedule, n_steps)`` pair.  The first
    schedule runs for the first *n_steps* steps, then the second schedule
    takes over (with its step counter reset to 0), and so on.  After all
    phases are exhausted the last schedule's final value is returned.

    Parameters
    ----------
    phases:
        Sequence of ``(Schedule, n_steps)`` pairs.
    """

    def __init__(self, phases: Sequence[tuple[Schedule, int]]) -> None:
        self.phases: list[tuple[Schedule, int]] = list(phases)

    def value(self, step: int) -> float:
        remaining = step
        for schedule, n_steps in self.phases:
            if remaining < n_steps:
                return schedule.value(remaining)
            remaining -= n_steps
        # Past the last phase - return the last schedule's terminal value.
        if self.phases:
            last_sched, last_n = self.phases[-1]
            return last_sched.value(last_n)
        return 0.0


class ExplorationSchedule(Schedule):
    """Epsilon schedule for exploration (linear or exponential decay).

    Parameters
    ----------
    initial_eps:
        Starting epsilon value.
    final_eps:
        Final epsilon value.
    total_steps:
        Duration of the decay.
    mode:
        ``"linear"`` or ``"exponential"`` decay.
    """

    def __init__(
        self,
        initial_eps: float = 1.0,
        final_eps: float = 0.01,
        total_steps: int = 100_000,
        mode: str = "linear",
    ) -> None:
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.total_steps = max(total_steps, 1)
        self.mode = mode

    def value(self, step: int) -> float:
        if self.mode == "exponential":
            # Compute decay rate so that at total_steps we reach final_eps.
            if self.initial_eps <= 0:
                return self.final_eps
            decay_rate = (self.final_eps / self.initial_eps) ** (1.0 / self.total_steps)
            val = self.initial_eps * (decay_rate**step)
            return max(val, self.final_eps)
        else:
            # Linear decay
            fraction = min(step / self.total_steps, 1.0)
            return self.initial_eps + (self.final_eps - self.initial_eps) * fraction
