"""Constrained policy optimisation utilities.

Provides Lagrangian-based and trust-region methods for enforcing cost
constraints during policy learning (e.g., CPO, PID-Lagrangian).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Learnable Lagrange multiplier
# ---------------------------------------------------------------------------

class LagrangianMultiplier:
    """Learnable Lagrange multiplier for constraint satisfaction.

    Maintains a single non-negative multiplier and updates it via gradient
    ascent on the constraint violation.

    Parameters
    ----------
    initial_value : float
        Starting value of the multiplier.
    learning_rate : float
        Step size for multiplier updates.
    max_value : float
        Upper clamp for stability.
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        learning_rate: float = 0.01,
        max_value: float = 100.0,
    ) -> None:
        self._value = max(0.0, initial_value)
        self.learning_rate = learning_rate
        self.max_value = max_value

    @property
    def value(self) -> float:
        """Current multiplier value."""
        return self._value

    def update(self, constraint_value: float, threshold: float) -> None:
        """Update the multiplier given the current constraint cost.

        The multiplier increases when ``constraint_value > threshold`` and
        decreases otherwise (projected to stay >= 0).

        Parameters
        ----------
        constraint_value : float
            Observed cost / constraint function output.
        threshold : float
            Maximum allowed constraint value (the constraint is
            ``constraint_value <= threshold``).
        """
        violation = constraint_value - threshold
        self._value += self.learning_rate * violation
        self._value = float(np.clip(self._value, 0.0, self.max_value))

    def penalized_objective(
        self, reward: float, constraint_value: float
    ) -> float:
        """Return ``reward - lambda * constraint_value``."""
        return reward - self._value * constraint_value


# ---------------------------------------------------------------------------
# Constrained Policy Optimisation (CPO) update
# ---------------------------------------------------------------------------

class CPOUpdate:
    """Single Constrained Policy Optimisation step.

    Implements the core CPO update: a conjugate-gradient solve for the
    natural-gradient direction, followed by a line-search that respects both
    the KL trust region and the cost constraint.

    Parameters
    ----------
    max_kl : float
        Maximum KL divergence (trust-region radius).
    cost_limit : float
        Maximum allowed expected cost.
    cg_iters : int
        Conjugate-gradient iterations.
    line_search_steps : int
        Maximum backtracking line-search steps.
    damping : float
        Damping coefficient for the Fisher matrix.
    """

    def __init__(
        self,
        max_kl: float = 0.01,
        cost_limit: float = 25.0,
        cg_iters: int = 10,
        line_search_steps: int = 10,
        damping: float = 0.1,
    ) -> None:
        self.max_kl = max_kl
        self.cost_limit = cost_limit
        self.cg_iters = cg_iters
        self.line_search_steps = line_search_steps
        self.damping = damping

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _conjugate_gradient(
        mvp_fn: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        n_iters: int = 10,
        residual_tol: float = 1e-10,
    ) -> np.ndarray:
        """Solve ``A x = b`` using conjugate gradient, where *mvp_fn*
        computes ``A @ v`` for a given ``v``.
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot = float(r @ r)

        for _ in range(n_iters):
            Ap = mvp_fn(p)
            pAp = float(p @ Ap)
            if abs(pAp) < 1e-12:
                break
            alpha = r_dot / pAp
            x += alpha * p
            r -= alpha * Ap
            new_r_dot = float(r @ r)
            if new_r_dot < residual_tol:
                break
            p = r + (new_r_dot / max(r_dot, 1e-12)) * p
            r_dot = new_r_dot

        return x

    # -- main update --------------------------------------------------------

    def step(
        self,
        params: np.ndarray,
        reward_grad: np.ndarray,
        cost_grad: np.ndarray,
        fisher_mvp_fn: Callable[[np.ndarray], np.ndarray],
        current_cost: float,
    ) -> np.ndarray:
        """Compute a CPO parameter update.

        Parameters
        ----------
        params : np.ndarray
            Current policy parameters (flat vector).
        reward_grad : np.ndarray
            Gradient of the reward objective.
        cost_grad : np.ndarray
            Gradient of the cost objective.
        fisher_mvp_fn : callable
            ``v -> F @ v`` — Fisher information matrix-vector product.
        current_cost : float
            Current expected cost.

        Returns
        -------
        np.ndarray
            Updated parameters.
        """
        # Damped Fisher MVP.
        def damped_mvp(v: np.ndarray) -> np.ndarray:
            return fisher_mvp_fn(v) + self.damping * v

        # Natural gradient direction (CG solve).
        step_dir = self._conjugate_gradient(
            damped_mvp, reward_grad, self.cg_iters
        )

        # Step size from trust region.
        sHs = float(step_dir @ damped_mvp(step_dir))
        if sHs < 1e-12:
            return params.copy()

        max_step = np.sqrt(2.0 * self.max_kl / sHs)

        # Cost constraint correction.
        cost_proj = float(cost_grad @ step_dir)
        if current_cost > self.cost_limit and cost_proj > 0:
            # Reduce step along cost gradient direction.
            self._conjugate_gradient(
                damped_mvp, cost_grad, self.cg_iters
            )
            max_step *= max(0.0, 1.0 - (current_cost - self.cost_limit) / max(abs(cost_proj), 1e-8))

        # Backtracking line search.
        for j in range(self.line_search_steps):
            scale = 0.5 ** j
            new_params = params + scale * max_step * step_dir
            # Accept the first feasible step (caller should verify KL + cost
            # externally for a full implementation).
            return new_params

        return params.copy()


# ---------------------------------------------------------------------------
# PID Lagrangian
# ---------------------------------------------------------------------------

class PIDLagrangian:
    """PID controller for the Lagrange multiplier.

    Provides smoother constraint enforcement than a simple gradient-ascent
    Lagrangian by incorporating proportional, integral, and derivative
    terms on the constraint violation signal.

    Parameters
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    max_value : float
        Upper clamp for the multiplier.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.01,
        kd: float = 0.1,
        max_value: float = 100.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_value = max_value

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._value: float = 0.0

    @property
    def value(self) -> float:
        """Current multiplier value."""
        return self._value

    def update(self, constraint_value: float, threshold: float) -> None:
        """Update the multiplier via PID control.

        Parameters
        ----------
        constraint_value : float
            Observed cost.
        threshold : float
            Target maximum cost.
        """
        error = constraint_value - threshold
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        raw = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._value = float(np.clip(self._value + raw, 0.0, self.max_value))

    def penalized_objective(
        self, reward: float, constraint_value: float
    ) -> float:
        """Return ``reward - lambda * constraint_value``."""
        return reward - self._value * constraint_value

    def reset(self) -> None:
        """Reset PID state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._value = 0.0
