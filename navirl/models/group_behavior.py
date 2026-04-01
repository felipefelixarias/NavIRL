"""Social group detection and behavior modeling.

Provides group detection via proximity and velocity similarity, cohesion /
repulsion forces for maintaining group formations, and a ``GroupHumanController``
that layers group dynamics on top of individual goal-seeking behavior.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from navirl.core.constants import COMFORT, EPSILON
from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController

__all__ = ["GroupDetector", "GroupBehaviorModel", "GroupHumanController"]


# ---------------------------------------------------------------------------
#  GroupDetector
# ---------------------------------------------------------------------------


class GroupDetector:
    """Detect social groups from spatial proximity and velocity similarity."""

    @staticmethod
    def detect_groups(
        positions: dict[int, tuple[float, float]],
        velocities: dict[int, tuple[float, float]],
        distance_threshold: float = COMFORT.group_max_separation,
        velocity_threshold: float = 0.5,
    ) -> list[set[int]]:
        """Cluster agents into social groups.

        Two agents are considered *connected* if they are within
        ``distance_threshold`` metres **and** their velocity difference
        magnitude is below ``velocity_threshold`` m/s.  Connected
        components of this graph form the groups.

        Parameters
        ----------
        positions:
            Mapping of ``agent_id`` → ``(x, y)``.
        velocities:
            Mapping of ``agent_id`` → ``(vx, vy)``.
        distance_threshold:
            Maximum inter-agent distance (metres) to consider a link.
        velocity_threshold:
            Maximum velocity-difference magnitude (m/s) for a link.

        Returns
        -------
        list[set[int]]
            Each set contains the agent IDs belonging to one group.
            Singleton sets (isolated agents) are excluded.
        """
        ids = list(positions.keys())
        n = len(ids)
        if n < 2:
            return []

        # Build adjacency via union-find.
        parent: dict[int, int] = {aid: aid for aid in ids}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            ai = ids[i]
            px_i, py_i = positions[ai]
            vx_i, vy_i = velocities[ai]
            for j in range(i + 1, n):
                aj = ids[j]
                px_j, py_j = positions[aj]
                dist = math.hypot(px_j - px_i, py_j - py_i)
                if dist > distance_threshold:
                    continue
                vx_j, vy_j = velocities[aj]
                vel_diff = math.hypot(vx_j - vx_i, vy_j - vy_i)
                if vel_diff <= velocity_threshold:
                    _union(ai, aj)

        # Collect components.
        groups_map: dict[int, set[int]] = {}
        for aid in ids:
            root = _find(aid)
            groups_map.setdefault(root, set()).add(aid)

        return [g for g in groups_map.values() if len(g) > 1]


# ---------------------------------------------------------------------------
#  GroupBehaviorModel
# ---------------------------------------------------------------------------


class GroupBehaviorModel:
    """Computes cohesion / repulsion forces and preferred formations."""

    def __init__(
        self,
        cohesion_strength: float = 0.8,
        repulsion_strength: float = 1.5,
        preferred_distance: float = COMFORT.group_cohesion_distance,
        min_distance: float = COMFORT.min_comfortable_distance,
    ) -> None:
        self.cohesion_strength = cohesion_strength
        self.repulsion_strength = repulsion_strength
        self.preferred_distance = preferred_distance
        self.min_distance = min_distance

    # -- cohesion -------------------------------------------------------

    def compute_cohesion_force(
        self,
        agent: AgentState,
        group_members: list[AgentState],
    ) -> tuple[float, float]:
        """Return a force pulling *agent* toward the group centroid.

        The force magnitude scales linearly with the distance from the
        centroid, beyond the ``preferred_distance``.
        """
        if not group_members:
            return 0.0, 0.0

        cx = sum(m.x for m in group_members) / len(group_members)
        cy = sum(m.y for m in group_members) / len(group_members)

        dx = cx - agent.x
        dy = cy - agent.y
        dist = math.hypot(dx, dy)

        if dist < self.preferred_distance or dist < EPSILON:
            return 0.0, 0.0

        excess = dist - self.preferred_distance
        magnitude = self.cohesion_strength * excess
        return magnitude * dx / dist, magnitude * dy / dist

    # -- repulsion ------------------------------------------------------

    def compute_repulsion_force(
        self,
        agent: AgentState,
        group_members: list[AgentState],
    ) -> tuple[float, float]:
        """Return a force keeping *agent* at a minimum distance from peers."""
        fx, fy = 0.0, 0.0
        for m in group_members:
            if m.agent_id == agent.agent_id:
                continue
            dx = agent.x - m.x
            dy = agent.y - m.y
            dist = math.hypot(dx, dy)
            if dist >= self.min_distance or dist < EPSILON:
                continue
            overlap = self.min_distance - dist
            magnitude = self.repulsion_strength * overlap
            fx += magnitude * dx / dist
            fy += magnitude * dy / dist
        return fx, fy

    # -- formation detection -------------------------------------------

    @staticmethod
    def compute_formation(
        group_positions: list[tuple[float, float]],
    ) -> str:
        """Classify the spatial arrangement of a group.

        Returns
        -------
        str
            One of ``'line'``, ``'V'``, or ``'cluster'``.
        """
        n = len(group_positions)
        if n < 2:
            return "cluster"

        pts = np.array(group_positions, dtype=np.float64)
        centroid = pts.mean(axis=0)
        centered = pts - centroid

        if n == 2:
            return "line"

        # Use PCA: if the first principal component explains > 85% of
        # variance, the formation is approximately linear.
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        total = eigvals.sum()
        if total < EPSILON:
            return "cluster"

        ratio = eigvals[0] / total
        if ratio > 0.85:
            # Check for V-shape: compute signed angles from centroid.
            angles = np.arctan2(centered[:, 1], centered[:, 0])
            angle_range = float(np.ptp(angles))
            if angle_range > math.pi / 3:
                return "V"
            return "line"

        return "cluster"


# ---------------------------------------------------------------------------
#  GroupHumanController
# ---------------------------------------------------------------------------


class GroupHumanController(HumanController):
    """Human controller that adds group dynamics on top of goal-seeking.

    Each step:
      1. Detect groups from current positions and velocities.
      2. Compute a goal-seeking velocity for each agent.
      3. Blend in cohesion / repulsion forces for agents that belong to a
         group.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.goal_tolerance: float = float(cfg.get("goal_tolerance", 0.5))
        self.relaxation_time: float = float(cfg.get("relaxation_time", COMFORT.relaxation_time))
        self.distance_threshold: float = float(
            cfg.get("distance_threshold", COMFORT.group_max_separation)
        )
        self.velocity_threshold: float = float(cfg.get("velocity_threshold", 0.5))

        self._group_model = GroupBehaviorModel(
            cohesion_strength=float(cfg.get("cohesion_strength", 0.8)),
            repulsion_strength=float(cfg.get("repulsion_strength", 1.5)),
            preferred_distance=float(
                cfg.get("preferred_distance", COMFORT.group_cohesion_distance)
            ),
            min_distance=float(cfg.get("min_distance", COMFORT.min_comfortable_distance)),
        )

        self.human_ids: list[int] = []
        self.starts: dict[int, tuple[float, float]] = {}
        self.goals: dict[int, tuple[float, float]] = {}
        self.backend: Any = None

    # -- HumanController interface --------------------------------------

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
        self.backend = backend

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        # Build position / velocity dicts for group detection.
        positions: dict[int, tuple[float, float]] = {}
        velocities: dict[int, tuple[float, float]] = {}
        for hid in self.human_ids:
            s = states[hid]
            positions[hid] = (s.x, s.y)
            velocities[hid] = (s.vx, s.vy)

        groups = GroupDetector.detect_groups(
            positions,
            velocities,
            distance_threshold=self.distance_threshold,
            velocity_threshold=self.velocity_threshold,
        )

        # Map agent → group for quick look-up.
        agent_group: dict[int, set[int]] = {}
        for g in groups:
            for aid in g:
                agent_group[aid] = g

        actions: dict[int, Action] = {}
        for hid in self.human_ids:
            s = states[hid]
            gx, gy = self.goals[hid]

            # Goal-seeking preferred velocity.
            dx = gx - s.x
            dy = gy - s.y
            dist = math.hypot(dx, dy)
            if dist < self.goal_tolerance:
                # Swap start / goal.
                prev = self.goals[hid]
                self.goals[hid] = self.starts[hid]
                self.starts[hid] = prev
                emit_event(
                    "goal_swap",
                    hid,
                    {
                        "new_goal": list(self.goals[hid]),
                        "new_start": list(self.starts[hid]),
                    },
                )
                gx, gy = self.goals[hid]
                dx = gx - s.x
                dy = gy - s.y
                dist = math.hypot(dx, dy)

            if dist > EPSILON:
                ux, uy = dx / dist, dy / dist
            else:
                ux, uy = 0.0, 0.0

            pref_speed = min(s.max_speed, dist / max(self.relaxation_time, EPSILON))
            pvx = ux * pref_speed
            pvy = uy * pref_speed

            # Add group forces if applicable.
            group = agent_group.get(hid)
            behavior = "GO_TO"
            metadata: dict[str, Any] = {}
            if group is not None:
                members = [states[mid] for mid in group if mid != hid and mid in states]
                cfx, cfy = self._group_model.compute_cohesion_force(s, members)
                rfx, rfy = self._group_model.compute_repulsion_force(s, members)
                pvx += (cfx + rfx) * dt
                pvy += (cfy + rfy) * dt
                behavior = "GROUP_WALK"
                metadata["group_ids"] = sorted(group)
                formation = GroupBehaviorModel.compute_formation(
                    [(states[mid].x, states[mid].y) for mid in group if mid in states]
                )
                metadata["formation"] = formation

            # Clamp to max speed.
            speed = math.hypot(pvx, pvy)
            if speed > s.max_speed and speed > EPSILON:
                scale = s.max_speed / speed
                pvx *= scale
                pvy *= scale

            actions[hid] = Action(
                pref_vx=pvx,
                pref_vy=pvy,
                behavior=behavior,
                metadata=metadata,
            )

        return actions
