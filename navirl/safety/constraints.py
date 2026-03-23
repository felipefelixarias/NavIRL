"""Hard safety constraints for navigation.

Provides abstract and concrete constraint classes that verify whether a
proposed action is safe and, if not, project it onto the nearest safe action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SafetyConstraint(ABC):
    """Base class for all hard safety constraints."""

    @abstractmethod
    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Return ``True`` if *action* is safe given *state*."""

    @abstractmethod
    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Project *action* to the nearest action satisfying the constraint."""


# ---------------------------------------------------------------------------
# Collision constraint
# ---------------------------------------------------------------------------

@dataclass
class CollisionConstraint(SafetyConstraint):
    """Prevents actions that would lead to a collision within a time horizon.

    Parameters
    ----------
    obstacle_positions : np.ndarray
        Array of shape ``(N, 2)`` with current obstacle positions.
    obstacle_radii : np.ndarray | float
        Scalar or per-obstacle collision radii.
    agent_radius : float
        Collision radius of the controlled agent.
    time_horizon : float
        Look-ahead time in seconds for collision checking.
    dt : float
        Simulation timestep used for forward-integration.
    """

    obstacle_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    obstacle_radii: np.ndarray | float = 0.3
    agent_radius: float = 0.25
    time_horizon: float = 2.0
    dt: float = 0.1

    # ---- helpers ----------------------------------------------------------

    def _forward_positions(
        self, state: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """Integrate agent position forward over the time horizon.

        Returns an array of shape ``(T, 2)`` with predicted positions.
        """
        pos = state[:2].copy()
        vel = action[:2].copy()
        steps = max(1, int(self.time_horizon / self.dt))
        positions = np.empty((steps, 2))
        for t in range(steps):
            pos = pos + vel * self.dt
            positions[t] = pos
        return positions

    # ---- interface --------------------------------------------------------

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:  # noqa: D401
        if self.obstacle_positions.shape[0] == 0:
            return True
        future = self._forward_positions(state, action)
        radii = np.asarray(self.obstacle_radii)
        min_dist = self.agent_radius + radii
        for pos in future:
            dists = np.linalg.norm(self.obstacle_positions - pos, axis=1)
            if np.any(dists < min_dist):
                return False
        return True

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.is_safe(state, action):
            return action.copy()
        # Iteratively reduce speed until safe (simple bisection).
        safe_action = action.copy()
        for scale in np.linspace(0.9, 0.0, 10):
            candidate = action.copy()
            candidate[:2] *= scale
            if self.is_safe(state, candidate):
                safe_action = candidate
                break
        return safe_action


# ---------------------------------------------------------------------------
# Speed constraint
# ---------------------------------------------------------------------------

@dataclass
class SpeedConstraint(SafetyConstraint):
    """Enforces a maximum speed limit.

    Parameters
    ----------
    max_speed : float
        Maximum allowed speed (m/s).
    """

    max_speed: float = 1.5

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:  # noqa: D401
        speed = float(np.linalg.norm(action[:2]))
        return speed <= self.max_speed

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(action[:2]))
        if speed <= self.max_speed or speed < 1e-8:
            return action.copy()
        scaled = action.copy()
        scaled[:2] = action[:2] * (self.max_speed / speed)
        return scaled


# ---------------------------------------------------------------------------
# Acceleration constraint
# ---------------------------------------------------------------------------

@dataclass
class AccelerationConstraint(SafetyConstraint):
    """Limits maximum acceleration and jerk.

    Parameters
    ----------
    max_acceleration : float
        Maximum allowed acceleration magnitude (m/s^2).
    max_jerk : float | None
        Maximum allowed jerk magnitude (m/s^3).  ``None`` disables jerk limit.
    dt : float
        Simulation timestep, used to compute implied acceleration/jerk.
    """

    max_acceleration: float = 3.0
    max_jerk: float | None = None
    dt: float = 0.1
    _prev_acceleration: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    def _acceleration(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        current_vel = state[2:4] if state.shape[0] >= 4 else np.zeros(2)
        return (action[:2] - current_vel) / max(self.dt, 1e-8)

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:  # noqa: D401
        acc = self._acceleration(state, action)
        acc_mag = float(np.linalg.norm(acc))
        if acc_mag > self.max_acceleration:
            return False
        if self.max_jerk is not None and self._prev_acceleration is not None:
            jerk = (acc - self._prev_acceleration) / max(self.dt, 1e-8)
            if float(np.linalg.norm(jerk)) > self.max_jerk:
                return False
        return True

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        current_vel = state[2:4] if state.shape[0] >= 4 else np.zeros(2)
        acc = self._acceleration(state, action)
        acc_mag = float(np.linalg.norm(acc))

        if acc_mag > self.max_acceleration and acc_mag > 1e-8:
            acc = acc * (self.max_acceleration / acc_mag)

        if self.max_jerk is not None and self._prev_acceleration is not None:
            jerk = (acc - self._prev_acceleration) / max(self.dt, 1e-8)
            jerk_mag = float(np.linalg.norm(jerk))
            if jerk_mag > self.max_jerk and jerk_mag > 1e-8:
                jerk = jerk * (self.max_jerk / jerk_mag)
                acc = self._prev_acceleration + jerk * self.dt

        safe_action = action.copy()
        safe_action[:2] = current_vel + acc * self.dt
        self._prev_acceleration = acc
        return safe_action


# ---------------------------------------------------------------------------
# Proxemics constraint
# ---------------------------------------------------------------------------

@dataclass
class ProxemicsConstraint(SafetyConstraint):
    """Maintains minimum distance to pedestrians based on proxemic zones.

    The zones follow Hall's proxemic model:
    - *intimate*: 0 – 0.45 m (always forbidden)
    - *personal*: 0.45 – 1.2 m (forbidden by default)
    - *social*: 1.2 – 3.6 m (allowed but penalised outside this module)

    Parameters
    ----------
    pedestrian_positions : np.ndarray
        Shape ``(N, 2)`` with current pedestrian positions.
    intimate_radius : float
        Radius of the intimate zone.
    personal_radius : float
        Radius of the personal zone.
    min_distance : float
        Hard minimum distance the agent must maintain.  Defaults to the
        personal zone boundary.
    dt : float
        Simulation timestep.
    """

    pedestrian_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    intimate_radius: float = 0.45
    personal_radius: float = 1.2
    min_distance: float | None = None
    dt: float = 0.1

    def __post_init__(self) -> None:
        if self.min_distance is None:
            self.min_distance = self.personal_radius

    def _next_position(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state[:2] + action[:2] * self.dt

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:  # noqa: D401
        if self.pedestrian_positions.shape[0] == 0:
            return True
        next_pos = self._next_position(state, action)
        dists = np.linalg.norm(self.pedestrian_positions - next_pos, axis=1)
        return bool(np.all(dists >= self.min_distance))

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.is_safe(state, action):
            return action.copy()
        # Push velocity away from the nearest violating pedestrian.
        next_pos = self._next_position(state, action)
        diffs = next_pos - self.pedestrian_positions
        dists = np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-8)
        violations = (dists.squeeze() < self.min_distance)  # type: ignore[union-attr]
        if not np.any(violations):
            return action.copy()
        repulse = (diffs[violations] / dists[violations]).sum(axis=0)
        norm = np.linalg.norm(repulse)
        if norm < 1e-8:
            return action * 0.0
        repulse /= norm
        safe_action = action.copy()
        safe_action[:2] = repulse * float(np.linalg.norm(action[:2]))
        return safe_action


# ---------------------------------------------------------------------------
# Boundary constraint
# ---------------------------------------------------------------------------

@dataclass
class BoundaryConstraint(SafetyConstraint):
    """Keeps the agent within rectangular environment bounds.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Axis-aligned bounding box of the environment.
    dt : float
        Simulation timestep.
    """

    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    dt: float = 0.1

    def _next_position(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return state[:2] + action[:2] * self.dt

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:  # noqa: D401
        nxt = self._next_position(state, action)
        return bool(
            self.x_min <= nxt[0] <= self.x_max
            and self.y_min <= nxt[1] <= self.y_max
        )

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.is_safe(state, action):
            return action.copy()
        nxt = self._next_position(state, action)
        clamped = np.array([
            np.clip(nxt[0], self.x_min, self.x_max),
            np.clip(nxt[1], self.y_min, self.y_max),
        ])
        safe_action = action.copy()
        safe_action[:2] = (clamped - state[:2]) / max(self.dt, 1e-8)
        return safe_action


# ---------------------------------------------------------------------------
# Composite set
# ---------------------------------------------------------------------------

class ConstraintSet(SafetyConstraint):
    """Combines multiple constraints and projects actions to satisfy all.

    Projection is performed by iterating over constraints in order and
    applying each one's projection sequentially.  A fixed-point iteration
    repeats until all constraints are satisfied or ``max_iters`` is reached.

    Parameters
    ----------
    constraints : Sequence[SafetyConstraint]
        Ordered list of constraints.
    max_iters : int
        Maximum projection iterations.
    """

    def __init__(
        self,
        constraints: Sequence[SafetyConstraint] | None = None,
        max_iters: int = 10,
    ) -> None:
        self.constraints: list[SafetyConstraint] = list(constraints or [])
        self.max_iters = max_iters

    # -- mutators -----------------------------------------------------------

    def add(self, constraint: SafetyConstraint) -> None:
        """Append a constraint to the set."""
        self.constraints.append(constraint)

    # -- interface ----------------------------------------------------------

    def is_safe(self, state: np.ndarray, action: np.ndarray) -> bool:  # noqa: D401
        return all(c.is_safe(state, action) for c in self.constraints)

    def project(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        projected = action.copy()
        for _ in range(self.max_iters):
            changed = False
            for c in self.constraints:
                if not c.is_safe(state, projected):
                    new = c.project(state, projected)
                    if not np.allclose(new, projected):
                        changed = True
                    projected = new
            if not changed:
                break
        return projected
