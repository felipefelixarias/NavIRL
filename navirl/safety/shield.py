"""Safety shielding for RL agents.

Provides wrappers that intercept an agent's actions and replace them with
safe alternatives when a constraint violation is detected.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from navirl.safety.constraints import ConstraintSet, SafetyConstraint

# ---------------------------------------------------------------------------
# Minimal agent protocol (avoids hard dependency on a concrete agent class)
# ---------------------------------------------------------------------------

class AgentLike(Protocol):
    """Structural protocol – any object with an ``act`` method."""

    def act(self, obs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# SafetyShield
# ---------------------------------------------------------------------------

class SafetyShield:
    """Wraps an agent and overrides unsafe actions.

    Parameters
    ----------
    agent : AgentLike
        The underlying RL agent.
    constraints : ConstraintSet | Sequence[SafetyConstraint]
        Hard constraints to enforce.
    fallback_policy : callable, optional
        ``(obs) -> action`` used when projection alone cannot produce a safe
        action.  Defaults to a zero-action (full stop).
    """

    def __init__(
        self,
        agent: AgentLike,
        constraints: ConstraintSet | Sequence[SafetyConstraint],
        fallback_policy: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.agent = agent
        if isinstance(constraints, ConstraintSet):
            self.constraints = constraints
        else:
            self.constraints = ConstraintSet(list(constraints))
        self.fallback_policy = fallback_policy

        # Bookkeeping
        self.interventions: int = 0
        self.total_steps: int = 0

    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Select a safe action for the given observation.

        1. Query the wrapped agent for a proposed action.
        2. Check constraints; if safe, return it directly.
        3. Attempt projection to a safe action.
        4. If projection still violates constraints, use the fallback policy.
        """
        self.total_steps += 1
        proposed = self.agent.act(obs)
        state = obs  # Assume obs contains at least [x, y, ...].

        if self.constraints.is_safe(state, proposed):
            return proposed

        self.interventions += 1

        # Try projection.
        projected = self.constraints.project(state, proposed)
        if self.constraints.is_safe(state, projected):
            return projected

        # Fallback.
        if self.fallback_policy is not None:
            return self.fallback_policy(obs)

        # Ultimate fallback: zero action (stop).
        return np.zeros_like(proposed)

    # ------------------------------------------------------------------

    @property
    def intervention_rate(self) -> float:
        """Fraction of steps where the shield had to intervene."""
        if self.total_steps == 0:
            return 0.0
        return self.interventions / self.total_steps

    def reset_stats(self) -> None:
        """Reset intervention counters."""
        self.interventions = 0
        self.total_steps = 0


# ---------------------------------------------------------------------------
# Control Barrier Function shield
# ---------------------------------------------------------------------------

class CBFShield:
    """Control Barrier Function-based safety shield.

    The CBF shield keeps the system in a *safe super-level set*
    ``{x : h(x) >= 0}`` by solving a small quadratic program (QP) that
    minimally modifies the proposed action to satisfy the CBF condition.

    Parameters
    ----------
    barrier_fn : callable
        ``h(state) -> float`` – barrier function.  Safe when >= 0.
    barrier_grad_fn : callable
        ``dh/dx(state) -> np.ndarray`` – gradient of *h* w.r.t. state.
    dynamics_fn : callable
        ``f(state, action) -> next_state`` – (simplified) dynamics.
    alpha : float
        Class-K function gain: ``dh/dt >= -alpha * h(x)``.
    action_dim : int
        Dimensionality of the action vector.
    """

    def __init__(
        self,
        barrier_fn: Callable[[np.ndarray], float],
        barrier_grad_fn: Callable[[np.ndarray], np.ndarray],
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        alpha: float = 1.0,
        action_dim: int = 2,
    ) -> None:
        self.barrier_fn = barrier_fn
        self.barrier_grad_fn = barrier_grad_fn
        self.dynamics_fn = dynamics_fn
        self.alpha = alpha
        self.action_dim = action_dim

    # ------------------------------------------------------------------

    def cbf_value(self, state: np.ndarray) -> float:
        """Compute the CBF value for the current state."""
        return float(self.barrier_fn(state))

    def is_safe(self, state: np.ndarray) -> bool:
        """Return ``True`` if the state is inside the safe set."""
        return self.cbf_value(state) >= 0.0

    def filter_action(
        self, state: np.ndarray, proposed_action: np.ndarray
    ) -> np.ndarray:
        """Filter *proposed_action* via a simplified QP to satisfy the CBF.

        The QP minimises ``||u - u_proposed||^2`` subject to the CBF
        constraint ``dh/dx * f(x, u) >= -alpha * h(x)``.

        For lightweight usage this implementation uses an analytic projection
        rather than a full QP solver.
        """
        h = self.barrier_fn(state)
        grad_h = self.barrier_grad_fn(state)

        # Evaluate constraint: Lfh + Lgh * u >= -alpha * h
        next_state_proposed = self.dynamics_fn(state, proposed_action)
        h_dot_proposed = float(grad_h @ (next_state_proposed - state))

        if h_dot_proposed >= -self.alpha * h:
            return proposed_action.copy()

        # Analytic single-constraint QP projection.
        Lgh = grad_h[:self.action_dim]
        Lgh_norm_sq = float(Lgh @ Lgh)
        if Lgh_norm_sq < 1e-12:
            return np.zeros_like(proposed_action)

        Lfh = float(grad_h @ (self.dynamics_fn(state, np.zeros_like(proposed_action)) - state))
        violation = Lfh + float(Lgh @ proposed_action[:self.action_dim]) + self.alpha * h
        if violation >= 0.0:
            return proposed_action.copy()

        lam = -violation / Lgh_norm_sq
        safe_action = proposed_action.copy()
        safe_action[:self.action_dim] = (
            proposed_action[:self.action_dim] + lam * Lgh
        )
        return safe_action


# ---------------------------------------------------------------------------
# Reachability shield (simplified HJ-reachability)
# ---------------------------------------------------------------------------

class ReachabilityShield:
    """Hamilton-Jacobi reachability-based safety shield (simplified).

    Pre-computes a discretised safe set over the state space and checks at
    runtime whether a proposed action keeps the agent inside that set.

    Parameters
    ----------
    safe_set : np.ndarray
        Boolean grid of shape matching the discretised state space.
        ``True`` marks safe cells.
    state_bounds : np.ndarray
        Shape ``(state_dim, 2)`` with ``[min, max]`` per dimension.
    dynamics_fn : callable
        ``f(state, action) -> next_state``.
    fallback_action : np.ndarray | None
        Action used when no safe action can be found.
    """

    def __init__(
        self,
        safe_set: np.ndarray,
        state_bounds: np.ndarray,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        fallback_action: np.ndarray | None = None,
    ) -> None:
        self.safe_set = safe_set
        self.state_bounds = np.asarray(state_bounds, dtype=np.float64)
        self.dynamics_fn = dynamics_fn
        self.fallback_action = fallback_action
        self._resolution = np.array(safe_set.shape)

    # ------------------------------------------------------------------

    def _state_to_index(self, state: np.ndarray) -> tuple[int, ...]:
        """Map a continuous state to the nearest grid index."""
        dims = self.state_bounds.shape[0]
        idx: list[int] = []
        for d in range(dims):
            lo, hi = self.state_bounds[d]
            frac = (state[d] - lo) / max(hi - lo, 1e-8)
            i = int(np.clip(np.round(frac * (self._resolution[d] - 1)), 0, self._resolution[d] - 1))
            idx.append(i)
        return tuple(idx)

    def is_state_safe(self, state: np.ndarray) -> bool:
        """Check if *state* is inside the precomputed safe set."""
        idx = self._state_to_index(state)
        return bool(self.safe_set[idx])

    def is_action_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if *action* keeps the agent in the safe set."""
        next_state = self.dynamics_fn(state, action)
        return self.is_state_safe(next_state)

    def filter_action(
        self, state: np.ndarray, proposed_action: np.ndarray
    ) -> np.ndarray:
        """Return the proposed action if safe, else the fallback."""
        if self.is_action_safe(state, proposed_action):
            return proposed_action.copy()
        if self.fallback_action is not None:
            return self.fallback_action.copy()
        return np.zeros_like(proposed_action)
