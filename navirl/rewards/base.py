"""
NavIRL Reward Base Classes
==========================

Foundational abstractions for building composable, normalisable and shapeable
reward signals in pedestrian-navigation reinforcement learning.

Classes
-------
RewardFunction
    Abstract base class that every reward must implement.
RewardComponent
    A named, weighted wrapper around a ``RewardFunction`` for use inside a
    ``CompositeReward``.
CompositeReward
    Combines multiple ``RewardComponent`` instances into a single scalar reward.
RewardNormalizer
    Running mean / standard-deviation normalisation with configurable warmup.
RewardClipper
    Hard-clips reward values to a symmetric or asymmetric range.
RewardShaper
    Potential-based reward shaping (Ng et al. 1999) that preserves optimal
    policy invariance.
"""

from __future__ import annotations

import abc
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

State = dict[str, Any]
"""A generic agent / environment state dictionary.

Expected keys (not all are mandatory for every reward):
    ``position``   -- shape ``(2,)`` ndarray  (x, y)
    ``velocity``   -- shape ``(2,)`` ndarray  (vx, vy)
    ``heading``    -- float, radians
    ``goal``       -- shape ``(2,)`` ndarray  (gx, gy)
    ``pedestrians`` -- list[dict] each with ``position``, ``velocity``, etc.
    ``obstacles``  -- list[dict] or ndarray of obstacle segments
    ``time``       -- float, simulation time in seconds
    ``dt``         -- float, timestep duration
"""

Action = Any
"""An action taken by the agent (continuous or discrete)."""

# ---------------------------------------------------------------------------
# RewardFunction -- abstract base
# ---------------------------------------------------------------------------


class RewardFunction(abc.ABC):
    """Abstract base class for all reward functions.

    A reward function maps a (state, action, next_state) transition to a
    scalar float value.  Implementations must override :meth:`compute`.

    Parameters
    ----------
    name : str, optional
        Human-readable name for logging / decomposition.  Defaults to the
        class name.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name: str = name or self.__class__.__name__

    # -- properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the human-readable name of this reward."""
        return self._name

    # -- abstract interface --------------------------------------------------

    @abc.abstractmethod
    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Return the scalar reward for a single transition.

        Parameters
        ----------
        state : State
            The state *before* the action.
        action : Action
            The action taken.
        next_state : State
            The state *after* the action.
        info : dict, optional
            Extra environment information (e.g. collision flags).

        Returns
        -------
        float
            The computed reward value.
        """
        raise NotImplementedError

    # -- optional hooks ------------------------------------------------------

    def reset(self) -> None:
        """Reset any internal state (e.g. between episodes).

        The default implementation is a no-op.
        """
        return None

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic information about the last :meth:`compute` call.

        Useful for TensorBoard / Weights & Biases logging of individual
        reward components.  Default returns an empty dict.
        """
        return {}

    # -- dunder helpers ------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"

    def __call__(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Convenience alias for :meth:`compute`."""
        return self.compute(state, action, next_state, info=info)


# ---------------------------------------------------------------------------
# RewardComponent -- named, weighted wrapper
# ---------------------------------------------------------------------------


@dataclass
class RewardComponent:
    """A named, weighted wrapper around a :class:`RewardFunction`.

    Used as building blocks inside :class:`CompositeReward`.

    Parameters
    ----------
    reward_fn : RewardFunction
        The underlying reward function.
    weight : float
        Multiplicative weight applied to the raw reward.
    enabled : bool
        If ``False`` the component is skipped during evaluation.
    tags : list[str]
        Arbitrary labels for filtering / grouping (e.g. ``["social"]``).
    """

    reward_fn: RewardFunction
    weight: float = 1.0
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    # -- helpers -------------------------------------------------------------

    @property
    def name(self) -> str:
        """Delegate to the wrapped function's name."""
        return self.reward_fn.name

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Compute weighted reward if enabled, else return 0.0."""
        if not self.enabled:
            return 0.0
        raw = self.reward_fn.compute(state, action, next_state, info=info)
        return self.weight * raw

    def reset(self) -> None:
        """Forward reset to the wrapped function."""
        self.reward_fn.reset()

    def get_info(self) -> dict[str, Any]:
        """Return info dict augmented with *weight* and *enabled* status."""
        base = self.reward_fn.get_info()
        base["weight"] = self.weight
        base["enabled"] = self.enabled
        return base

    def __repr__(self) -> str:
        return f"RewardComponent(name={self.name!r}, weight={self.weight}, enabled={self.enabled})"


# ---------------------------------------------------------------------------
# CompositeReward -- combines multiple components
# ---------------------------------------------------------------------------


class CompositeReward(RewardFunction):
    """Combine multiple :class:`RewardComponent` instances into one scalar.

    The composite reward is the *sum* of weighted component outputs:

    .. math::

        r_{\\text{total}} = \\sum_i w_i \\cdot r_i

    Parameters
    ----------
    components : Sequence[RewardComponent]
        Ordered collection of reward components.
    name : str, optional
        Name for the composite reward.
    track_decomposition : bool
        If ``True``, :meth:`get_info` will include each component's
        individual (weighted) contribution.
    """

    def __init__(
        self,
        components: Sequence[RewardComponent] | None = None,
        *,
        name: str = "CompositeReward",
        track_decomposition: bool = True,
    ) -> None:
        super().__init__(name=name)
        self._components: list[RewardComponent] = list(components or [])
        self._track_decomposition = track_decomposition
        self._last_decomposition: dict[str, float] = {}

    # -- component management ------------------------------------------------

    @property
    def components(self) -> list[RewardComponent]:
        """Return the list of components (mutable)."""
        return self._components

    def add_component(self, component: RewardComponent) -> None:
        """Append a component to the composite."""
        self._components.append(component)

    def remove_component(self, name: str) -> RewardComponent | None:
        """Remove and return the first component matching *name*, or ``None``."""
        for i, comp in enumerate(self._components):
            if comp.name == name:
                return self._components.pop(i)
        return None

    def get_component(self, name: str) -> RewardComponent | None:
        """Return the first component matching *name*, or ``None``."""
        for comp in self._components:
            if comp.name == name:
                return comp
        return None

    def set_weight(self, name: str, weight: float) -> None:
        """Set the weight of the component with *name*.

        Raises
        ------
        KeyError
            If no component with the given name exists.
        """
        comp = self.get_component(name)
        if comp is None:
            raise KeyError(f"No component named {name!r}")
        comp.weight = weight

    def enable(self, name: str) -> None:
        """Enable a component by name."""
        comp = self.get_component(name)
        if comp is not None:
            comp.enabled = True

    def disable(self, name: str) -> None:
        """Disable a component by name."""
        comp = self.get_component(name)
        if comp is not None:
            comp.enabled = False

    def filter_by_tag(self, tag: str) -> list[RewardComponent]:
        """Return all components that carry the given *tag*."""
        return [c for c in self._components if tag in c.tags]

    # -- core interface ------------------------------------------------------

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Sum all enabled, weighted component rewards.

        Parameters
        ----------
        state, action, next_state, info
            Forwarded verbatim to each component's :meth:`compute`.

        Returns
        -------
        float
            Total composite reward.
        """
        total = 0.0
        decomposition: dict[str, float] = {}
        for comp in self._components:
            value = comp.compute(state, action, next_state, info=info)
            total += value
            if self._track_decomposition:
                decomposition[comp.name] = value
        if self._track_decomposition:
            self._last_decomposition = decomposition
        return total

    def reset(self) -> None:
        """Reset all components."""
        self._last_decomposition = {}
        for comp in self._components:
            comp.reset()

    def get_info(self) -> dict[str, Any]:
        """Return per-component reward decomposition.

        Returns
        -------
        dict
            ``{"decomposition": {name: weighted_value, ...}, "total": float}``
        """
        return {
            "decomposition": dict(self._last_decomposition),
            "total": sum(self._last_decomposition.values()),
        }

    def summary(self) -> str:
        """Return a multi-line human-readable summary of the composite."""
        lines = [f"CompositeReward '{self._name}' ({len(self._components)} components):"]
        for comp in self._components:
            status = "ON " if comp.enabled else "OFF"
            tags = ", ".join(comp.tags) if comp.tags else ""
            lines.append(f"  [{status}] {comp.name:30s}  w={comp.weight:+.4f}  tags=[{tags}]")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self):
        return iter(self._components)

    def __repr__(self) -> str:
        return f"CompositeReward(n_components={len(self._components)})"


# ---------------------------------------------------------------------------
# RewardNormalizer -- running mean / std
# ---------------------------------------------------------------------------


class RewardNormalizer(RewardFunction):
    """Wrap a reward function and normalise its output with running statistics.

    .. math::

        r_{\\text{norm}} = \\frac{r - \\mu}{\\max(\\sigma, \\epsilon)}

    The running mean :math:`\\mu` and standard deviation :math:`\\sigma` are
    estimated using Welford's online algorithm.

    Parameters
    ----------
    reward_fn : RewardFunction
        The function whose output is normalised.
    center : bool
        Subtract running mean.
    scale : bool
        Divide by running standard deviation.
    clip : float or None
        If not ``None``, clip the normalised reward to ``[-clip, clip]``.
    warmup : int
        Number of initial calls during which normalisation is *not* applied
        (raw values are returned) but statistics are still accumulated.
    epsilon : float
        Minimum divisor to avoid division by zero.
    gamma : float or None
        If provided, use an exponential moving average with this decay
        factor instead of Welford's algorithm.
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        *,
        center: bool = True,
        scale: bool = True,
        clip: float | None = 10.0,
        warmup: int = 100,
        epsilon: float = 1e-8,
        gamma: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"Normalized({reward_fn.name})")
        self._fn = reward_fn
        self._center = center
        self._scale = scale
        self._clip_val = clip
        self._warmup = warmup
        self._eps = epsilon
        self._gamma = gamma

        # Welford accumulators
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

        # EMA accumulators (used when gamma is set)
        self._ema_mean: float = 0.0
        self._ema_var: float = 1.0

        self._last_raw: float = 0.0
        self._last_normalised: float = 0.0

    # -- statistics helpers --------------------------------------------------

    @property
    def mean(self) -> float:
        """Current running mean estimate."""
        if self._gamma is not None:
            return self._ema_mean
        return self._mean

    @property
    def std(self) -> float:
        """Current running standard deviation estimate."""
        if self._gamma is not None:
            return math.sqrt(max(self._ema_var, 0.0))
        if self._count < 2:
            return 1.0
        return math.sqrt(self._m2 / self._count)

    def _update_welford(self, value: float) -> None:
        """Update Welford online mean / variance accumulators."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _update_ema(self, value: float) -> None:
        """Update exponential moving average mean / variance."""
        assert self._gamma is not None
        g = self._gamma
        self._ema_mean = g * self._ema_mean + (1.0 - g) * value
        diff = value - self._ema_mean
        self._ema_var = g * self._ema_var + (1.0 - g) * diff * diff

    def _update(self, value: float) -> None:
        if self._gamma is not None:
            self._update_ema(value)
        else:
            self._update_welford(value)

    # -- core interface ------------------------------------------------------

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Compute the normalised reward.

        During warmup the raw value is returned while statistics accumulate.

        Parameters
        ----------
        state, action, next_state, info
            Forwarded to the wrapped reward function.

        Returns
        -------
        float
            The normalised (and optionally clipped) reward.
        """
        raw = self._fn.compute(state, action, next_state, info=info)
        self._last_raw = raw
        self._update(raw)

        if self._count <= self._warmup:
            self._last_normalised = raw
            return raw

        normalised = raw
        if self._center:
            normalised -= self.mean
        if self._scale:
            normalised /= max(self.std, self._eps)
        if self._clip_val is not None:
            normalised = float(np.clip(normalised, -self._clip_val, self._clip_val))

        self._last_normalised = normalised
        return normalised

    def reset(self) -> None:
        """Reset the wrapped function but *not* the running statistics.

        Call :meth:`reset_stats` explicitly to clear statistics.
        """
        self._fn.reset()

    def reset_stats(self) -> None:
        """Clear running mean / variance accumulators."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._ema_mean = 0.0
        self._ema_var = 1.0

    def get_info(self) -> dict[str, Any]:
        """Return normalisation diagnostics."""
        return {
            "raw": self._last_raw,
            "normalised": self._last_normalised,
            "running_mean": self.mean,
            "running_std": self.std,
            "count": self._count,
        }


# ---------------------------------------------------------------------------
# RewardClipper -- hard clip
# ---------------------------------------------------------------------------


class RewardClipper(RewardFunction):
    """Wrap a reward and hard-clip its output to ``[low, high]``.

    Parameters
    ----------
    reward_fn : RewardFunction
        The function whose output is clipped.
    low : float
        Lower bound.
    high : float
        Upper bound.
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        low: float = -10.0,
        high: float = 10.0,
        *,
        name: str | None = None,
    ) -> None:
        if low >= high:
            raise ValueError(f"low ({low}) must be < high ({high})")
        super().__init__(name=name or f"Clipped({reward_fn.name})")
        self._fn = reward_fn
        self._low = low
        self._high = high
        self._last_raw: float = 0.0
        self._last_clipped: float = 0.0
        self._clip_count: int = 0
        self._total_count: int = 0

    @property
    def clip_fraction(self) -> float:
        """Fraction of calls that were actually clipped so far."""
        if self._total_count == 0:
            return 0.0
        return self._clip_count / self._total_count

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Compute and clip the reward.

        Parameters
        ----------
        state, action, next_state, info
            Forwarded to the wrapped reward function.

        Returns
        -------
        float
            Clipped reward in ``[low, high]``.
        """
        raw = self._fn.compute(state, action, next_state, info=info)
        self._last_raw = raw
        self._total_count += 1
        clipped = float(np.clip(raw, self._low, self._high))
        if clipped != raw:
            self._clip_count += 1
        self._last_clipped = clipped
        return clipped

    def reset(self) -> None:
        """Forward reset to the wrapped function."""
        self._fn.reset()

    def get_info(self) -> dict[str, Any]:
        """Return clipping diagnostics."""
        return {
            "raw": self._last_raw,
            "clipped": self._last_clipped,
            "clip_fraction": self.clip_fraction,
            "bounds": (self._low, self._high),
        }


# ---------------------------------------------------------------------------
# RewardShaper -- potential-based shaping (Ng et al. 1999)
# ---------------------------------------------------------------------------


class RewardShaper(RewardFunction):
    """Potential-based reward shaping with optimal-policy preservation.

    Given a base reward function :math:`r(s, a, s')` and a potential function
    :math:`\\Phi(s)`, the shaped reward is:

    .. math::

        r'(s, a, s') = r(s, a, s') + \\gamma \\Phi(s') - \\Phi(s)

    where :math:`\\gamma` is the MDP discount factor.  Ng, Harada & Russell
    (1999) proved that this transformation preserves the set of optimal
    policies under the original reward.

    Parameters
    ----------
    reward_fn : RewardFunction
        The original (unshaped) reward.
    potential_fn : callable
        Maps a ``State`` dict to a scalar potential :math:`\\Phi(s)`.
    gamma : float
        Discount factor of the MDP.  Must be in ``[0, 1]``.
    name : str, optional
        Human-readable name.
    scale : float
        An additional multiplier on the shaping bonus, useful for tuning
        the magnitude of shaping without re-defining the potential.

    Notes
    -----
    *  When *gamma* = 1 the shaping term is simply
       ``Phi(s') - Phi(s)`` and the discounted return is unchanged.
    *  A common choice for the potential is the negative distance to
       the goal: ``Phi(s) = -||s.position - s.goal||``.
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        potential_fn: Any,  # Callable[[State], float]
        gamma: float = 0.99,
        *,
        name: str | None = None,
        scale: float = 1.0,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        super().__init__(name=name or f"Shaped({reward_fn.name})")
        self._fn = reward_fn
        self._potential_fn = potential_fn
        self._gamma = gamma
        self._scale = scale

        self._last_base: float = 0.0
        self._last_shaping: float = 0.0
        self._last_total: float = 0.0

        # Cache the potential of the current state between calls so that
        # phi(s) == phi(s') from the previous call when states are shared.
        self._prev_potential: float | None = None

    # -- core interface ------------------------------------------------------

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Return the shaped reward ``r + scale * (gamma * Phi(s') - Phi(s))``.

        Parameters
        ----------
        state, action, next_state, info
            Forwarded to the base reward function.

        Returns
        -------
        float
            Shaped reward.
        """
        base = self._fn.compute(state, action, next_state, info=info)
        self._last_base = base

        phi_s = self._compute_potential(state)
        phi_s_prime = self._compute_potential(next_state)

        shaping = self._scale * (self._gamma * phi_s_prime - phi_s)
        self._last_shaping = shaping
        self._prev_potential = phi_s_prime

        total = base + shaping
        self._last_total = total
        return total

    def _compute_potential(self, state: State) -> float:
        """Evaluate the potential function with error handling.

        Parameters
        ----------
        state : State
            The environment state to evaluate.

        Returns
        -------
        float
            Potential value, or 0.0 if evaluation fails.
        """
        try:
            return float(self._potential_fn(state))
        except Exception:
            logger.warning("Potential function raised; returning 0.0", exc_info=True)
            return 0.0

    def reset(self) -> None:
        """Reset the wrapped function and clear the cached potential."""
        self._fn.reset()
        self._prev_potential = None

    def get_info(self) -> dict[str, Any]:
        """Return shaping diagnostics."""
        return {
            "base_reward": self._last_base,
            "shaping_bonus": self._last_shaping,
            "total": self._last_total,
            "gamma": self._gamma,
            "scale": self._scale,
        }

    # -- utility factory -----------------------------------------------------

    @staticmethod
    def goal_distance_potential(state: State) -> float:
        """Default potential: negative Euclidean distance to goal.

        Expects ``state["position"]`` and ``state["goal"]`` as array-like
        of shape ``(2,)``.

        Parameters
        ----------
        state : State
            Must contain ``position`` and ``goal`` keys.

        Returns
        -------
        float
            ``-||position - goal||_2``.
        """
        pos = np.asarray(state["position"], dtype=np.float64)
        goal = np.asarray(state["goal"], dtype=np.float64)
        return -float(np.linalg.norm(pos - goal))

    @classmethod
    def with_goal_potential(
        cls,
        reward_fn: RewardFunction,
        gamma: float = 0.99,
        *,
        scale: float = 1.0,
        name: str | None = None,
    ) -> RewardShaper:
        """Factory: build a shaper using negative goal-distance as potential.

        Parameters
        ----------
        reward_fn : RewardFunction
            Base reward to shape.
        gamma : float
            Discount factor.
        scale : float
            Shaping magnitude multiplier.
        name : str, optional
            Human-readable name.

        Returns
        -------
        RewardShaper
            A configured reward shaper.
        """
        return cls(
            reward_fn,
            potential_fn=cls.goal_distance_potential,
            gamma=gamma,
            scale=scale,
            name=name,
        )
