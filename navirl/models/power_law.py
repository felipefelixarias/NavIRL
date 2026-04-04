"""Power Law anticipatory collision avoidance model.

Implements the energy-based model of Karamouzas, Skinner & Guy (2014) where
the interaction force between two pedestrians is proportional to the inverse
square of the time-to-collision (tau). This captures the empirically observed
power-law relationship in pedestrian trajectories.

Reference:
    Karamouzas, I., Skinner, B., & Guy, S. J. (2014). Universal power law
    governing pedestrian interactions. *Physical Review Letters*, 113(23),
    238701.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from navirl.core.constants import BODY, EPSILON
from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController
from navirl.utils import normalize_vector

__all__ = [
    "PowerLawConfig",
    "PowerLawModel",
    "PowerLawHumanController",
]


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class PowerLawConfig:
    """Configuration for the Power Law model.

    Attributes
    ----------
    k : float
        Interaction strength parameter (N * s^2).
    tau_0 : float
        Time threshold below which the force is capped (seconds).
        Prevents divergence at very small time-to-collision.
    agent_radius : float
        Default agent radius (metres). Used when not available from state.
    sigma : float
        Gaussian noise standard deviation added to the force (N).
        Models individual variation in anticipatory behavior.
    relaxation_time : float
        Relaxation time for the desired-velocity driving force (seconds).
    max_force : float
        Upper bound on the magnitude of the anticipatory force (N).
    neighbor_distance : float
        Cutoff distance beyond which interactions are ignored (metres).
    """

    k: float = 1.5
    tau_0: float = 3.0
    agent_radius: float = BODY.body_radius
    sigma: float = 0.1
    relaxation_time: float = 0.5
    max_force: float = 40.0
    neighbor_distance: float = 10.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _time_to_collision(px: float, py: float, vx: float, vy: float, radius_sum: float) -> float:
    """Compute the time-to-collision between two disks.

    Parameters
    ----------
    px, py : relative position (other - self)
    vx, vy : relative velocity (self - other)
    radius_sum : sum of radii

    Returns
    -------
    Time to closest approach that results in overlap, or ``float('inf')``
    if no collision will occur.
    """
    # Solve |p + v*t|^2 = radius_sum^2 for t
    a = vx * vx + vy * vy
    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - radius_sum * radius_sum

    if c < 0.0:
        # Already overlapping
        return 0.0

    if a < EPSILON:
        # No relative motion
        return float("inf")

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:
        # No collision
        return float("inf")

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # We want the earliest positive collision time
    if t1 > EPSILON:
        return t1
    if t2 > EPSILON:
        return t2
    return float("inf")


# ---------------------------------------------------------------------------
#  Power Law Model
# ---------------------------------------------------------------------------


class PowerLawModel:
    """Core Power Law anticipatory collision avoidance model.

    The interaction force between agents *i* and *j* is:

        F_ij = -k / tau_ij^2 * grad_tau_ij

    where tau_ij is the time-to-collision and grad_tau_ij is the gradient
    of tau with respect to the position of agent *i*.
    """

    def __init__(self, config: PowerLawConfig | None = None) -> None:
        self.cfg = config or PowerLawConfig()
        self._rng = np.random.default_rng()

    def compute_desired_force(
        self, state: AgentState, goal: tuple[float, float]
    ) -> tuple[float, float]:
        """Driving force toward the goal at preferred speed.

        F_desired = (v_preferred - v_current) / tau_relax
        """
        dx = goal[0] - state.x
        dy = goal[1] - state.y
        ex, ey, dist = normalize_vector(dx, dy)

        if dist < EPSILON:
            return (-state.vx / self.cfg.relaxation_time, -state.vy / self.cfg.relaxation_time)

        pref_vx = state.max_speed * ex
        pref_vy = state.max_speed * ey
        fx = (pref_vx - state.vx) / self.cfg.relaxation_time
        fy = (pref_vy - state.vy) / self.cfg.relaxation_time
        return fx, fy

    def compute_anticipatory_force(
        self, state: AgentState, other_states: Sequence[AgentState]
    ) -> tuple[float, float]:
        """Sum of anticipatory power-law forces from neighboring agents.

        For each neighbor, the force magnitude is k / max(tau, tau_0)^2,
        directed along the gradient of tau (which pushes agents apart to
        delay collision).
        """
        fx_total, fy_total = 0.0, 0.0

        for other in other_states:
            if other.agent_id == state.agent_id:
                continue

            # Relative position and velocity
            rel_px = other.x - state.x
            rel_py = other.y - state.y

            dist = math.hypot(rel_px, rel_py)
            if dist > self.cfg.neighbor_distance:
                continue

            rel_vx = state.vx - other.vx
            rel_vy = state.vy - other.vy
            radius_sum = state.radius + other.radius

            tau = _time_to_collision(rel_px, rel_py, rel_vx, rel_vy, radius_sum)

            if tau == float("inf") or tau < 0.0:
                continue

            # Clamp tau to avoid singularity
            tau_clamped = max(tau, EPSILON)

            # Force magnitude: k / max(tau, tau_0)^2
            # But we also scale by tau_0^2 / tau^2 to get stronger force
            # as collision approaches
            tau_effective = max(tau_clamped, self.cfg.tau_0)
            magnitude = self.cfg.k / (tau_effective * tau_effective)

            # If tau < tau_0, use stronger force proportional to 1/tau^2
            if tau_clamped < self.cfg.tau_0:
                magnitude = self.cfg.k / (tau_clamped * tau_clamped)

            # Cap the force
            magnitude = min(magnitude, self.cfg.max_force)

            # Direction: gradient of tau points away from collision
            # Approximate: direction from predicted collision point
            # Collision point at time tau
            pred_x = state.x + state.vx * tau_clamped
            pred_y = state.y + state.vy * tau_clamped
            other_pred_x = other.x + other.vx * tau_clamped
            other_pred_y = other.y + other.vy * tau_clamped

            # Force direction: away from predicted collision
            diff_x = pred_x - other_pred_x
            diff_y = pred_y - other_pred_y
            nx, ny, n_dist = normalize_vector(diff_x, diff_y)

            if n_dist < EPSILON:
                # Use current relative position as fallback direction
                nx, ny, _ = normalize_vector(-rel_px, -rel_py)

            fx_total += magnitude * nx
            fy_total += magnitude * ny

        # Optional noise
        if self.cfg.sigma > EPSILON:
            fx_total += self._rng.normal(0.0, self.cfg.sigma)
            fy_total += self._rng.normal(0.0, self.cfg.sigma)

        return fx_total, fy_total

    def compute_total_force(
        self, state: AgentState, other_states: Sequence[AgentState]
    ) -> tuple[float, float]:
        """Sum of desired force and anticipatory collision avoidance force."""
        goal = (state.goal_x, state.goal_y)
        f_desired = self.compute_desired_force(state, goal)
        f_anticipatory = self.compute_anticipatory_force(state, other_states)
        return (
            f_desired[0] + f_anticipatory[0],
            f_desired[1] + f_anticipatory[1],
        )

    def step(
        self,
        states: Sequence[AgentState],
        goals: dict[int, tuple[float, float]],
        dt: float = 0.04,
    ) -> dict[int, tuple[float, float]]:
        """Compute new velocities for all agents.

        Parameters
        ----------
        states : sequence of AgentState
        goals : mapping agent_id -> (goal_x, goal_y)
        dt : simulation timestep

        Returns
        -------
        dict mapping agent_id to (new_vx, new_vy)
        """
        all_states = list(states)
        new_velocities: dict[int, tuple[float, float]] = {}

        for state in all_states:
            fx, fy = self.compute_total_force(state, all_states)

            new_vx = state.vx + fx * dt
            new_vy = state.vy + fy * dt

            # Clamp to max speed
            speed = math.hypot(new_vx, new_vy)
            if speed > state.max_speed and speed > EPSILON:
                scale = state.max_speed / speed
                new_vx *= scale
                new_vy *= scale

            new_velocities[state.agent_id] = (new_vx, new_vy)

        return new_velocities


# ---------------------------------------------------------------------------
#  HumanController wrapper
# ---------------------------------------------------------------------------


class PowerLawHumanController(HumanController):
    """Human behavior controller driven by the Power Law model.

    Wraps :class:`PowerLawModel` to conform to the standard
    ``HumanController`` interface used throughout NavIRL.
    """

    def __init__(self, config: PowerLawConfig | None = None) -> None:
        self.model = PowerLawModel(config)
        self.human_ids: list[int] = []
        self.goals: dict[int, tuple[float, float]] = {}
        self.starts: dict[int, tuple[float, float]] = {}
        self.goal_tolerance: float = 0.5

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        self.human_ids = list(human_ids)
        self.starts = dict(starts)
        self.goals = dict(goals)

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        all_states = list(states.values())
        actions: dict[int, Action] = {}

        for hid in self.human_ids:
            if hid not in states:
                continue

            state = states[hid]
            goal = self.goals[hid]

            # Goal swap on arrival
            dist_to_goal = math.hypot(goal[0] - state.x, goal[1] - state.y)
            if dist_to_goal < self.goal_tolerance:
                prev_goal = self.goals[hid]
                self.goals[hid] = self.starts[hid]
                self.starts[hid] = prev_goal
                emit_event(
                    "goal_swap",
                    hid,
                    {
                        "new_goal": list(self.goals[hid]),
                        "new_start": list(self.starts[hid]),
                    },
                )
                goal = self.goals[hid]

            # Build a state with updated goal for force computation
            goal_state = AgentState(
                agent_id=state.agent_id,
                kind=state.kind,
                x=state.x,
                y=state.y,
                vx=state.vx,
                vy=state.vy,
                goal_x=goal[0],
                goal_y=goal[1],
                radius=state.radius,
                max_speed=state.max_speed,
            )

            fx, fy = self.model.compute_total_force(goal_state, all_states)

            pref_vx = state.vx + fx * dt
            pref_vy = state.vy + fy * dt

            # Clamp speed
            speed = math.hypot(pref_vx, pref_vy)
            if speed > state.max_speed and speed > EPSILON:
                scale = state.max_speed / speed
                pref_vx *= scale
                pref_vy *= scale

            actions[hid] = Action(
                pref_vx=pref_vx,
                pref_vy=pref_vy,
                behavior="GO_TO",
                metadata={
                    "model": "power_law",
                    "force_x": fx,
                    "force_y": fy,
                },
            )

        return actions
