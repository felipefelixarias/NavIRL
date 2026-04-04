"""Social Force Model (Helbing & Molnar 1995).

Implements the classical Social Force Model for pedestrian dynamics, including:
- Desired force (driving toward goal)
- Social (repulsive) force from other pedestrians with anisotropic field-of-view
- Wall repulsive force
- Contact force (body compression + sliding friction) when agents overlap

Reference:
    Helbing, D. & Molnar, P. (1995). Social force model for pedestrian dynamics.
    *Physical Review E*, 51(5), 4282.

    Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features
    of escape panic. *Nature*, 407(6803), 487-490.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from navirl.core.constants import EPSILON, SFM
from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController
from navirl.utils import normalize_vector

__all__ = [
    "SocialForceConfig",
    "SocialForceModel",
    "SocialForceHumanController",
]


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class SocialForceConfig:
    """Configuration for the Social Force Model.

    Default values are taken from ``navirl.core.constants.SFM``.
    """

    # Repulsive interaction strength and range
    A: float = SFM.A
    B: float = SFM.B

    # Wall repulsion strength and range
    A_wall: float = SFM.A_wall
    B_wall: float = SFM.B_wall

    # Anisotropy: 0 = isotropic, 1 = only-forward weighting
    lambda_anisotropy: float = SFM.lambda_anisotropy

    # Contact forces
    k_body: float = SFM.k_body
    kappa_friction: float = SFM.kappa_friction

    # Relaxation time for desired velocity
    tau: float = SFM.tau

    # Cutoff distances
    interaction_range: float = SFM.interaction_range
    wall_interaction_range: float = SFM.wall_interaction_range


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _anisotropy_weight(ex: float, ey: float, nx: float, ny: float, lam: float) -> float:
    """Compute the anisotropic field-of-view weight.

    w = lam + (1 - lam) * (1 + cos(theta)) / 2

    where theta is the angle between the agent's heading (ex, ey) and the
    direction toward the neighbor (nx, ny).
    """
    cos_theta = -(ex * nx + ey * ny)  # negative because n points *from* other
    return lam + (1.0 - lam) * (1.0 + cos_theta) / 2.0


# ---------------------------------------------------------------------------
#  Wall segment type
# ---------------------------------------------------------------------------

WallSegment = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float]:
    """Return (distance, nearest_point_x, nearest_point_y) from point to segment."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < EPSILON * EPSILON:
        # Degenerate segment
        return math.hypot(px - x1, py - y1), x1, y1

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    dist = math.hypot(px - nearest_x, py - nearest_y)
    return dist, nearest_x, nearest_y


# ---------------------------------------------------------------------------
#  Social Force Model
# ---------------------------------------------------------------------------


class SocialForceModel:
    """Core Social Force Model computation engine.

    All forces operate on ``AgentState`` objects (or plain position/velocity
    tuples) and return 2-D force vectors as ``(fx, fy)`` tuples.
    """

    def __init__(self, config: SocialForceConfig | None = None) -> None:
        self.cfg = config or SocialForceConfig()

    # -- Desired (driving) force -------------------------------------------

    def compute_desired_force(
        self, state: AgentState, goal: tuple[float, float]
    ) -> tuple[float, float]:
        """Force driving the agent toward the goal at preferred speed.

        F_desired = (v_pref * e_goal - v_current) / tau
        """
        dx = goal[0] - state.x
        dy = goal[1] - state.y
        ex, ey, dist = normalize_vector(dx, dy)

        if dist < EPSILON:
            # Already at goal — decelerate
            return (-state.vx / self.cfg.tau, -state.vy / self.cfg.tau)

        pref_vx = state.max_speed * ex
        pref_vy = state.max_speed * ey
        fx = (pref_vx - state.vx) / self.cfg.tau
        fy = (pref_vy - state.vy) / self.cfg.tau
        return fx, fy

    # -- Social (repulsive) force ------------------------------------------

    def compute_social_force(
        self, state: AgentState, other_states: Sequence[AgentState]
    ) -> tuple[float, float]:
        """Sum of repulsive interaction forces from all nearby agents.

        F_social = sum_j A * exp((r_ij - d_ij) / B) * n_ij * w(theta)

        where r_ij = radius_i + radius_j, d_ij is center-center distance,
        n_ij is the unit vector from j to i, and w(theta) applies anisotropy.
        """
        fx_total, fy_total = 0.0, 0.0

        # Agent heading for anisotropy
        speed = math.hypot(state.vx, state.vy)
        if speed > EPSILON:
            heading_x = state.vx / speed
            heading_y = state.vy / speed
        else:
            # Default heading toward goal
            heading_x, heading_y, _ = normalize_vector(
                state.goal_x - state.x, state.goal_y - state.y
            )

        for other in other_states:
            if other.agent_id == state.agent_id:
                continue

            dx = state.x - other.x
            dy = state.y - other.y
            dist = math.hypot(dx, dy)

            if dist < EPSILON or dist > self.cfg.interaction_range:
                continue

            r_ij = state.radius + other.radius
            nx = dx / dist
            ny = dy / dist

            # Exponential repulsive force
            magnitude = self.cfg.A * math.exp((r_ij - dist) / self.cfg.B)

            # Anisotropy weight
            w = _anisotropy_weight(heading_x, heading_y, nx, ny, self.cfg.lambda_anisotropy)

            fx_total += magnitude * nx * w
            fy_total += magnitude * ny * w

        return fx_total, fy_total

    # -- Wall force --------------------------------------------------------

    def compute_wall_force(
        self, state: AgentState, walls: Sequence[WallSegment]
    ) -> tuple[float, float]:
        """Repulsive force from wall segments.

        F_wall = A_wall * exp((r_i - d_iw) / B_wall) * n_iw

        where d_iw is distance to the nearest point on the wall and n_iw is
        the normal pointing from the wall toward the agent.
        """
        fx_total, fy_total = 0.0, 0.0

        for x1, y1, x2, y2 in walls:
            dist, wx, wy = _point_to_segment_distance(state.x, state.y, x1, y1, x2, y2)

            if dist < EPSILON or dist > self.cfg.wall_interaction_range:
                continue

            nx = (state.x - wx) / dist
            ny = (state.y - wy) / dist

            magnitude = self.cfg.A_wall * math.exp((state.radius - dist) / self.cfg.B_wall)

            # Contact compression if overlapping the wall
            overlap = state.radius - dist
            if overlap > 0.0:
                magnitude += self.cfg.k_body * overlap

                # Tangential friction (sliding along wall)
                tx, ty = -ny, nx  # tangent
                delta_vt = state.vx * tx + state.vy * ty
                fx_total -= self.cfg.kappa_friction * overlap * delta_vt * tx
                fy_total -= self.cfg.kappa_friction * overlap * delta_vt * ty

            fx_total += magnitude * nx
            fy_total += magnitude * ny

        return fx_total, fy_total

    # -- Contact force -----------------------------------------------------

    def compute_contact_force(
        self, state: AgentState, other_states: Sequence[AgentState]
    ) -> tuple[float, float]:
        """Body compression and sliding friction when agents physically overlap.

        Active only when d_ij < r_ij (actual body contact).

        F_contact = k_body * overlap * n_ij + kappa * overlap * delta_v_t * t_ij
        """
        fx_total, fy_total = 0.0, 0.0

        for other in other_states:
            if other.agent_id == state.agent_id:
                continue

            dx = state.x - other.x
            dy = state.y - other.y
            dist = math.hypot(dx, dy)
            if dist < EPSILON:
                dist = EPSILON

            r_ij = state.radius + other.radius
            overlap = r_ij - dist

            if overlap <= 0.0:
                continue

            # Normal direction (from other toward self)
            nx = dx / dist
            ny = dy / dist

            # Body compression (normal)
            f_normal = self.cfg.k_body * overlap

            # Tangential friction
            tx, ty = -ny, nx  # tangent perpendicular to normal
            delta_vt = (other.vx - state.vx) * tx + (other.vy - state.vy) * ty
            f_tangent = self.cfg.kappa_friction * overlap * delta_vt

            fx_total += f_normal * nx + f_tangent * tx
            fy_total += f_normal * ny + f_tangent * ty

        return fx_total, fy_total

    # -- Total force -------------------------------------------------------

    def compute_total_force(
        self,
        state: AgentState,
        other_states: Sequence[AgentState],
        walls: Sequence[WallSegment] = (),
    ) -> tuple[float, float]:
        """Sum of all forces acting on *state*."""
        goal = (state.goal_x, state.goal_y)

        f_desired = self.compute_desired_force(state, goal)
        f_social = self.compute_social_force(state, other_states)
        f_wall = self.compute_wall_force(state, walls)
        f_contact = self.compute_contact_force(state, other_states)

        fx = f_desired[0] + f_social[0] + f_wall[0] + f_contact[0]
        fy = f_desired[1] + f_social[1] + f_wall[1] + f_contact[1]
        return fx, fy

    # -- Integration step --------------------------------------------------

    def step(
        self,
        states: Sequence[AgentState],
        goals: dict[int, tuple[float, float]],
        walls: Sequence[WallSegment] = (),
        dt: float = 0.04,
    ) -> dict[int, tuple[float, float]]:
        """Compute new preferred velocities for all agents.

        Parameters
        ----------
        states:
            Current states for all agents.
        goals:
            Mapping from agent_id to (goal_x, goal_y).
        walls:
            List of wall segments as (x1, y1, x2, y2) tuples.
        dt:
            Simulation timestep in seconds.

        Returns
        -------
        dict mapping agent_id to (new_vx, new_vy).
        """
        all_states = list(states)
        new_velocities: dict[int, tuple[float, float]] = {}

        for state in all_states:
            fx, fy = self.compute_total_force(state, all_states, walls)

            # Euler integration: v_new = v_old + F * dt
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


class SocialForceHumanController(HumanController):
    """Human behavior controller driven by the Social Force Model.

    Wraps :class:`SocialForceModel` to conform to the standard
    ``HumanController`` interface used throughout NavIRL.
    """

    def __init__(
        self,
        config: SocialForceConfig | None = None,
        walls: Sequence[WallSegment] = (),
    ) -> None:
        self.model = SocialForceModel(config)
        self.walls: list[WallSegment] = list(walls)
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
        # Collect all visible agent states (humans + robot)
        all_states = list(states.values())

        # Build the list of human states for force computation
        [states[hid] for hid in self.human_ids if hid in states]

        actions: dict[int, Action] = {}

        for hid in self.human_ids:
            if hid not in states:
                continue

            state = states[hid]

            # Check goal arrival and swap
            goal = self.goals[hid]
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

            # Temporarily set goal on agent state for force computation
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

            fx, fy = self.model.compute_total_force(goal_state, all_states, self.walls)

            # Compute preferred velocity from force
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
                    "model": "social_force",
                    "force_x": fx,
                    "force_y": fy,
                },
            )

        return actions
