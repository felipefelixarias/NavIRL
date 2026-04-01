"""Velocity Obstacle family of collision avoidance algorithms.

Implements the following algorithms:
- **VO**: Basic Velocity Obstacle (Fiorini & Shiller 1998)
- **RVO**: Reciprocal Velocity Obstacle (van den Berg et al. 2008)
- **HRVO**: Hybrid Reciprocal Velocity Obstacle (Snape et al. 2011)
- **ORCA**: Optimal Reciprocal Collision Avoidance, pure-Python reference
  (van den Berg et al. 2011)

References:
    Fiorini, P. & Shiller, Z. (1998). Motion planning in dynamic environments
    using velocity obstacles. *IJRR*, 17(7), 760-772.

    van den Berg, J., Lin, M., & Manocha, D. (2008). Reciprocal velocity
    obstacles for real-time multi-agent navigation. *ICRA 2008*.

    Snape, J., van den Berg, J., Guy, S. J., & Manocha, D. (2011). The hybrid
    reciprocal velocity obstacle. *IEEE Trans. Robotics*, 27(4), 696-706.

    van den Berg, J., Guy, S. J., Lin, M., & Manocha, D. (2011). Reciprocal
    n-body collision avoidance. *ISRR 2011*, Springer STAR 70, 3-19.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from navirl.core.constants import EPSILON
from navirl.core.constants import ORCA as ORCA_DEFAULTS
from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController

__all__ = [
    "VOConfig",
    "VelocityObstacle",
    "ReciprocalVelocityObstacle",
    "HybridReciprocalVO",
    "ORCAPurePython",
    "VOHumanController",
]


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class VOConfig:
    """Configuration for Velocity Obstacle algorithms."""

    time_horizon: float = ORCA_DEFAULTS.time_horizon
    max_speed: float = 1.5
    safety_margin: float = ORCA_DEFAULTS.safety_margin
    num_samples: int = 250  # sampling resolution for VO/RVO/HRVO
    neighbor_distance: float = ORCA_DEFAULTS.neighbor_distance
    max_neighbors: int = ORCA_DEFAULTS.max_neighbors


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------


class VOCone(NamedTuple):
    """A velocity obstacle cone in velocity space.

    The cone apex is at ``(apex_vx, apex_vy)`` and its two boundary
    half-lines are defined by unit direction vectors ``(left_dx, left_dy)``
    and ``(right_dx, right_dy)``.
    """

    apex_vx: float
    apex_vy: float
    left_dx: float
    left_dy: float
    right_dx: float
    right_dy: float


class HalfPlane(NamedTuple):
    """ORCA half-plane constraint: point + outward normal.

    The permitted half-plane is {v : (v - point) . normal >= 0}.
    """

    point_x: float
    point_y: float
    normal_x: float
    normal_y: float


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _normalize(vx: float, vy: float) -> tuple[float, float, float]:
    n = math.hypot(vx, vy)
    if n < EPSILON:
        return 0.0, 0.0, 0.0
    return vx / n, vy / n, n


def _cross2d(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _in_vo_cone(vx: float, vy: float, cone: VOCone) -> bool:
    """Test whether velocity (vx, vy) is inside the VO cone."""
    rel_vx = vx - cone.apex_vx
    rel_vy = vy - cone.apex_vy
    # Inside cone if to the right of the left leg and to the left of the right leg
    cross_left = _cross2d(cone.left_dx, cone.left_dy, rel_vx, rel_vy)
    cross_right = _cross2d(rel_vx, rel_vy, cone.right_dx, cone.right_dy)
    return cross_left >= 0.0 and cross_right >= 0.0


# ---------------------------------------------------------------------------
#  Velocity Obstacle (basic)
# ---------------------------------------------------------------------------


class VelocityObstacle:
    """Basic Velocity Obstacle computation.

    For each obstacle (another agent), computes the cone in velocity space
    that would lead to collision within the time horizon.
    """

    def __init__(self, config: VOConfig | None = None) -> None:
        self.cfg = config or VOConfig()

    def compute_vo(self, agent: AgentState, obstacle: AgentState) -> VOCone | None:
        """Compute the velocity obstacle cone induced by *obstacle* on *agent*.

        Returns ``None`` if the obstacle is beyond the interaction range.
        """
        dx = obstacle.x - agent.x
        dy = obstacle.y - agent.y
        dist = math.hypot(dx, dy)

        combined_radius = agent.radius + obstacle.radius + self.cfg.safety_margin

        if dist < EPSILON:
            # Agents at the same position — degenerate; treat as full block
            return VOCone(
                apex_vx=obstacle.vx,
                apex_vy=obstacle.vy,
                left_dx=1.0,
                left_dy=0.0,
                right_dx=-1.0,
                right_dy=0.0,
            )

        if dist > self.cfg.neighbor_distance:
            return None

        # Truncated VO: shrink Minkowski sum by time_horizon
        # The cone apex is at the obstacle's velocity
        # The half-angle of the cone is arcsin(combined_radius / dist)
        if combined_radius >= dist:
            # Overlapping — full velocity space is blocked
            half_angle = math.pi / 2.0
        else:
            half_angle = math.asin(min(1.0, combined_radius / dist))

        # Direction from agent to obstacle
        nx = dx / dist
        ny = dy / dist

        cos_ha = math.cos(half_angle)
        sin_ha = math.sin(half_angle)

        # Left and right boundary directions (rotate direction to obstacle)
        left_dx = nx * cos_ha - ny * sin_ha
        left_dy = nx * sin_ha + ny * cos_ha
        right_dx = nx * cos_ha + ny * sin_ha
        right_dy = -nx * sin_ha + ny * cos_ha

        # Apex: obstacle velocity (for basic VO)
        return VOCone(
            apex_vx=obstacle.vx,
            apex_vy=obstacle.vy,
            left_dx=left_dx,
            left_dy=left_dy,
            right_dx=right_dx,
            right_dy=right_dy,
        )

    def select_velocity(
        self,
        agent: AgentState,
        vos: Sequence[VOCone],
        preferred_vel: tuple[float, float],
    ) -> tuple[float, float]:
        """Select the velocity closest to *preferred_vel* outside all VO cones.

        Uses random sampling with a penalty-based selection.
        """
        best_vx, best_vy = preferred_vel
        best_cost = float("inf")

        rng = np.random.default_rng(seed=hash(agent.agent_id) & 0xFFFFFFFF)
        max_speed = min(agent.max_speed, self.cfg.max_speed)

        # Always consider the preferred velocity
        candidates = [(preferred_vel[0], preferred_vel[1])]

        # Sample candidate velocities
        angles = rng.uniform(0, 2.0 * math.pi, self.cfg.num_samples)
        speeds = rng.uniform(0.0, max_speed, self.cfg.num_samples)
        for i in range(self.cfg.num_samples):
            candidates.append((speeds[i] * math.cos(angles[i]), speeds[i] * math.sin(angles[i])))

        for cvx, cvy in candidates:
            # Cost = distance to preferred velocity
            cost = math.hypot(cvx - preferred_vel[0], cvy - preferred_vel[1])

            # Penalty for being inside any VO cone
            in_vo = False
            for cone in vos:
                if _in_vo_cone(cvx, cvy, cone):
                    in_vo = True
                    break

            if in_vo:
                cost += 1e6  # large penalty

            # Speed constraint
            if math.hypot(cvx, cvy) > max_speed:
                cost += 1e6

            if cost < best_cost:
                best_cost = cost
                best_vx, best_vy = cvx, cvy

        return best_vx, best_vy


# ---------------------------------------------------------------------------
#  Reciprocal Velocity Obstacle
# ---------------------------------------------------------------------------


class ReciprocalVelocityObstacle(VelocityObstacle):
    """Reciprocal Velocity Obstacle (RVO).

    The VO cone apex is placed at the average of the two agents' velocities,
    splitting collision avoidance responsibility equally.
    """

    def compute_vo(self, agent: AgentState, obstacle: AgentState) -> VOCone | None:
        cone = super().compute_vo(agent, obstacle)
        if cone is None:
            return None

        # RVO: move apex to midpoint of agent and obstacle velocities
        apex_vx = 0.5 * (agent.vx + obstacle.vx)
        apex_vy = 0.5 * (agent.vy + obstacle.vy)

        return VOCone(
            apex_vx=apex_vx,
            apex_vy=apex_vy,
            left_dx=cone.left_dx,
            left_dy=cone.left_dy,
            right_dx=cone.right_dx,
            right_dy=cone.right_dy,
        )


# ---------------------------------------------------------------------------
#  Hybrid Reciprocal Velocity Obstacle
# ---------------------------------------------------------------------------


class HybridReciprocalVO(VelocityObstacle):
    """Hybrid Reciprocal Velocity Obstacle (HRVO).

    Uses the RVO apex on the side of the preferred velocity and the VO apex
    on the other side, reducing oscillations.
    """

    def compute_vo(self, agent: AgentState, obstacle: AgentState) -> VOCone | None:
        cone = super().compute_vo(agent, obstacle)
        if cone is None:
            return None

        # Determine which side the agent's velocity is on
        rvo_apex_vx = 0.5 * (agent.vx + obstacle.vx)
        rvo_apex_vy = 0.5 * (agent.vy + obstacle.vy)

        # Direction from obstacle center in velocity space
        rel_vx = agent.vx - obstacle.vx
        rel_vy = agent.vy - obstacle.vy

        # Cross product to determine side
        cross = _cross2d(cone.left_dx, cone.left_dy, rel_vx, rel_vy)

        if cross >= 0:
            # Agent velocity is to the left — use RVO apex
            apex_vx = rvo_apex_vx
            apex_vy = rvo_apex_vy
        else:
            # Agent velocity is to the right — use VO apex (obstacle velocity)
            apex_vx = obstacle.vx
            apex_vy = obstacle.vy

        return VOCone(
            apex_vx=apex_vx,
            apex_vy=apex_vy,
            left_dx=cone.left_dx,
            left_dy=cone.left_dy,
            right_dx=cone.right_dx,
            right_dy=cone.right_dy,
        )


# ---------------------------------------------------------------------------
#  ORCA — Pure-Python reference implementation
# ---------------------------------------------------------------------------


class ORCAPurePython:
    """Pure-Python reference implementation of ORCA.

    This is a simplified but correct implementation intended for validation
    and small-scale experiments. For production use, prefer the C++ backed
    ``rvo2`` library via :class:`navirl.humans.orca.controller.ORCAHumanController`.
    """

    def __init__(self, config: VOConfig | None = None) -> None:
        self.cfg = config or VOConfig()

    def compute_orca_lines(
        self,
        agent: AgentState,
        neighbors: Sequence[AgentState],
        dt: float,
    ) -> list[HalfPlane]:
        """Compute ORCA half-plane constraints for *agent* against *neighbors*.

        Each constraint is a half-plane in velocity space such that any
        velocity satisfying all constraints is collision-free for at least
        ``time_horizon`` seconds.
        """
        lines: list[HalfPlane] = []
        inv_tau = 1.0 / self.cfg.time_horizon

        for other in neighbors:
            if other.agent_id == agent.agent_id:
                continue

            rel_px = other.x - agent.x
            rel_py = other.y - agent.y
            rel_vx = agent.vx - other.vx
            rel_vy = agent.vy - other.vy
            dist_sq = rel_px * rel_px + rel_py * rel_py
            combined_radius = agent.radius + other.radius + self.cfg.safety_margin
            combined_radius_sq = combined_radius * combined_radius

            if dist_sq > combined_radius_sq:
                # No collision at current positions
                # Vector from cutoff circle center to relative velocity
                w_x = rel_vx - inv_tau * rel_px
                w_y = rel_vy - inv_tau * rel_py

                w_len_sq = w_x * w_x + w_y * w_y
                dot_product = w_x * rel_px + w_y * rel_py

                if dot_product < 0.0 and (
                    dot_product * dot_product > combined_radius_sq * w_len_sq
                ):
                    # Project on cut-off circle
                    w_len = math.sqrt(w_len_sq) if w_len_sq > EPSILON else EPSILON
                    unit_w_x = w_x / w_len
                    unit_w_y = w_y / w_len

                    normal_x = unit_w_x
                    normal_y = unit_w_y
                    u_x = (combined_radius * inv_tau - w_len) * unit_w_x
                    u_y = (combined_radius * inv_tau - w_len) * unit_w_y
                else:
                    # Project on legs
                    math.sqrt(dist_sq) if dist_sq > EPSILON else EPSILON
                    leg = math.sqrt(max(0.0, dist_sq - combined_radius_sq))

                    # Determine which leg
                    if _cross2d(rel_px, rel_py, w_x, w_y) > 0.0:
                        # Left leg
                        dir_x = (rel_px * leg - rel_py * combined_radius) / dist_sq
                        dir_y = (rel_py * leg + rel_px * combined_radius) / dist_sq
                    else:
                        # Right leg
                        dir_x = (rel_px * leg + rel_py * combined_radius) / dist_sq
                        dir_y = (rel_py * leg - rel_px * combined_radius) / dist_sq

                    dot_v_dir = rel_vx * dir_x + rel_vy * dir_y
                    u_x = dot_v_dir * dir_x - rel_vx
                    u_y = dot_v_dir * dir_y - rel_vy

                    normal_x = -dir_y
                    normal_y = dir_x
                    # Ensure normal points outward
                    if normal_x * u_x + normal_y * u_y < 0:
                        normal_x = dir_y
                        normal_y = -dir_x
            else:
                # Already colliding — resolve in one timestep
                inv_dt = 1.0 / dt if dt > EPSILON else 1.0 / EPSILON
                w_x = rel_vx - inv_dt * rel_px
                w_y = rel_vy - inv_dt * rel_py
                w_len = math.hypot(w_x, w_y)
                if w_len < EPSILON:
                    w_len = EPSILON

                unit_w_x = w_x / w_len
                unit_w_y = w_y / w_len

                normal_x = unit_w_x
                normal_y = unit_w_y
                u_x = (combined_radius * inv_dt - w_len) * unit_w_x
                u_y = (combined_radius * inv_dt - w_len) * unit_w_y

            # ORCA line: point = v_A + 0.5 * u, direction = perpendicular to normal
            point_x = agent.vx + 0.5 * u_x
            point_y = agent.vy + 0.5 * u_y
            lines.append(
                HalfPlane(
                    point_x=point_x,
                    point_y=point_y,
                    normal_x=normal_x,
                    normal_y=normal_y,
                )
            )

        return lines

    def solve_linear_program(
        self,
        orca_lines: Sequence[HalfPlane],
        preferred_vel: tuple[float, float],
        max_speed: float,
    ) -> tuple[float, float]:
        """Find the velocity closest to *preferred_vel* satisfying all ORCA constraints.

        Uses incremental linear programming (add one constraint at a time).
        Falls back to closest-point projection when infeasible.
        """
        result_vx, result_vy = preferred_vel

        for i, line_i in enumerate(orca_lines):
            # Check if current result satisfies this constraint
            det = (result_vx - line_i.point_x) * line_i.normal_x + (
                result_vy - line_i.point_y
            ) * line_i.normal_y
            if det >= 0.0:
                continue  # already satisfied

            # Project onto this constraint line
            # Line direction is perpendicular to normal
            dir_x = -line_i.normal_y
            dir_y = line_i.normal_x

            # Find closest point on line to preferred_vel
            t = (preferred_vel[0] - line_i.point_x) * dir_x + (
                preferred_vel[1] - line_i.point_y
            ) * dir_y

            # Clamp t to stay within max_speed disk
            # Solve |point + t * dir|^2 <= max_speed^2
            a = dir_x * dir_x + dir_y * dir_y  # = 1.0 for unit dir
            b = 2.0 * (line_i.point_x * dir_x + line_i.point_y * dir_y)
            c = (
                line_i.point_x * line_i.point_x
                + line_i.point_y * line_i.point_y
                - max_speed * max_speed
            )
            disc = b * b - 4.0 * a * c

            if disc < 0.0:
                # No intersection with speed disk — find closest feasible
                # Project preferred velocity onto the half-plane boundary
                result_vx = line_i.point_x + t * dir_x
                result_vy = line_i.point_y + t * dir_y
            else:
                sqrt_disc = math.sqrt(disc)
                t_min = (-b - sqrt_disc) / (2.0 * a)
                t_max = (-b + sqrt_disc) / (2.0 * a)
                t = max(t_min, min(t, t_max))
                result_vx = line_i.point_x + t * dir_x
                result_vy = line_i.point_y + t * dir_y

            # Re-check all previous constraints and adjust if needed
            for j in range(i):
                line_j = orca_lines[j]
                det_j = (result_vx - line_j.point_x) * line_j.normal_x + (
                    result_vy - line_j.point_y
                ) * line_j.normal_y
                if det_j < -EPSILON:
                    # Conflict between constraints i and j — find intersection
                    # of the two constraint lines
                    dir_j_x = -line_j.normal_y
                    dir_j_y = line_j.normal_x

                    denom = _cross2d(dir_x, dir_y, dir_j_x, dir_j_y)
                    if abs(denom) < EPSILON:
                        continue  # parallel lines

                    dp_x = line_j.point_x - line_i.point_x
                    dp_y = line_j.point_y - line_i.point_y
                    t_int = _cross2d(dir_j_x, dir_j_y, dp_x, dp_y) / denom

                    result_vx = line_i.point_x + t_int * dir_x
                    result_vy = line_i.point_y + t_int * dir_y
                    break

        # Final speed clamp
        speed = math.hypot(result_vx, result_vy)
        if speed > max_speed and speed > EPSILON:
            scale = max_speed / speed
            result_vx *= scale
            result_vy *= scale

        return result_vx, result_vy


# ---------------------------------------------------------------------------
#  HumanController wrapper
# ---------------------------------------------------------------------------


class VOHumanController(HumanController):
    """Human behavior controller using Velocity Obstacle algorithms.

    The ``algorithm`` parameter selects between ``"vo"``, ``"rvo"``,
    ``"hrvo"``, and ``"orca"`` (default).
    """

    def __init__(
        self,
        config: VOConfig | None = None,
        algorithm: str = "orca",
    ) -> None:
        self.cfg = config or VOConfig()
        self.algorithm = algorithm.lower()
        self.human_ids: list[int] = []
        self.goals: dict[int, tuple[float, float]] = {}
        self.starts: dict[int, tuple[float, float]] = {}
        self.goal_tolerance: float = 0.5

        # Instantiate the appropriate VO variant
        if self.algorithm == "orca":
            self._orca = ORCAPurePython(self.cfg)
            self._vo: VelocityObstacle | None = None
        elif self.algorithm == "rvo":
            self._orca = None
            self._vo = ReciprocalVelocityObstacle(self.cfg)
        elif self.algorithm == "hrvo":
            self._orca = None
            self._vo = HybridReciprocalVO(self.cfg)
        else:
            self._orca = None
            self._vo = VelocityObstacle(self.cfg)

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

    def _preferred_velocity(
        self, state: AgentState, goal: tuple[float, float]
    ) -> tuple[float, float]:
        """Compute preferred velocity directly toward the goal."""
        dx = goal[0] - state.x
        dy = goal[1] - state.y
        ux, uy, dist = _normalize(dx, dy)
        if dist < self.goal_tolerance:
            return 0.0, 0.0
        speed = min(state.max_speed, self.cfg.max_speed)
        return ux * speed, uy * speed

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

            pref_vel = self._preferred_velocity(state, goal)

            if self._orca is not None:
                # ORCA path
                neighbors = [s for s in all_states if s.agent_id != hid]
                orca_lines = self._orca.compute_orca_lines(state, neighbors, dt)
                vx, vy = self._orca.solve_linear_program(orca_lines, pref_vel, state.max_speed)
            else:
                # VO/RVO/HRVO path
                assert self._vo is not None
                vos: list[VOCone] = []
                for other in all_states:
                    if other.agent_id == hid:
                        continue
                    cone = self._vo.compute_vo(state, other)
                    if cone is not None:
                        vos.append(cone)
                vx, vy = self._vo.select_velocity(state, vos, pref_vel)

            actions[hid] = Action(
                pref_vx=vx,
                pref_vy=vy,
                behavior="GO_TO",
                metadata={
                    "model": f"vo_{self.algorithm}",
                },
            )

        return actions
