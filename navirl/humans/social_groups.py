"""Social group formation, dynamics, and decision-making for pedestrian groups.

Implements group formations (line, cluster, V-shape, F-formation for
conversational groups), cohesion/repulsion forces, leader-follower dynamics,
group velocity consensus, splitting/merging, and collective decision-making
at intersections.  All computations use numpy only.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navirl.humans.pedestrian_state import PedestrianState

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FormationType(enum.Enum):
    """Spatial formation pattern for a group."""

    LINE = "line"
    CLUSTER = "cluster"
    V_SHAPE = "v_shape"
    ABREAST = "abreast"
    RIVER = "river"
    F_FORMATION = "f_formation"


class GroupRole(enum.Enum):
    """Role of a member within the group."""

    LEADER = "leader"
    FOLLOWER = "follower"
    EQUAL = "equal"


# ---------------------------------------------------------------------------
# F-formation (conversational groups)
# ---------------------------------------------------------------------------


@dataclass
class FFormation:
    """Models a conversational (F-formation) arrangement.

    In an F-formation, group members arrange themselves around a shared
    focus point (the *o-space*) and each member has a *transactional
    segment* (p-space).

    Parameters
    ----------
    o_space_radius : float
        Radius of the shared interaction space (m).
    p_space_width : float
        Width of the personal transactional segment (m).
    facing_tolerance : float
        Allowed angular deviation from facing the o-space centre (rad).
    """

    o_space_radius: float = 0.8
    p_space_width: float = 0.5
    facing_tolerance: float = math.pi / 6.0

    def compute_positions(
        self,
        centre: np.ndarray,
        n_members: int,
        start_angle: float = 0.0,
    ) -> list[np.ndarray]:
        """Compute ideal member positions around *centre*.

        Members are distributed uniformly on a circle of radius
        ``o_space_radius + p_space_width``.

        Parameters
        ----------
        centre : numpy.ndarray
            2-D centre of the o-space.
        n_members : int
            Number of group members.
        start_angle : float
            Starting angle offset (rad).

        Returns
        -------
        list[numpy.ndarray]
            List of 2-D positions for each member.
        """
        r = self.o_space_radius + self.p_space_width
        positions: list[np.ndarray] = []
        for i in range(n_members):
            angle = start_angle + 2.0 * math.pi * i / max(n_members, 1)
            pos = centre + r * np.array([math.cos(angle), math.sin(angle)])
            positions.append(pos)
        return positions

    def compute_facing_angles(
        self,
        centre: np.ndarray,
        member_positions: list[np.ndarray],
    ) -> list[float]:
        """Compute the ideal facing angle for each member (towards centre).

        Parameters
        ----------
        centre : numpy.ndarray
            O-space centre.
        member_positions : list[numpy.ndarray]
            Current member positions.

        Returns
        -------
        list[float]
            Facing angles in radians.
        """
        angles: list[float] = []
        for pos in member_positions:
            diff = centre - pos
            angles.append(float(math.atan2(diff[1], diff[0])))
        return angles

    def is_valid(
        self,
        centre: np.ndarray,
        member_positions: list[np.ndarray],
        member_headings: list[float],
    ) -> bool:
        """Check whether the current configuration is a valid F-formation.

        Parameters
        ----------
        centre : numpy.ndarray
            O-space centre.
        member_positions : list[numpy.ndarray]
            Member positions.
        member_headings : list[float]
            Member body headings.

        Returns
        -------
        bool
            ``True`` if all members are within tolerance.
        """
        ideal_angles = self.compute_facing_angles(centre, member_positions)
        for heading, ideal in zip(member_headings, ideal_angles, strict=False):
            delta = math.atan2(math.sin(heading - ideal), math.cos(heading - ideal))
            if abs(delta) > self.facing_tolerance:
                return False
        return True


# ---------------------------------------------------------------------------
# SocialGroup
# ---------------------------------------------------------------------------


@dataclass
class SocialGroup:
    """Represents a social group of pedestrians with shared dynamics.

    Parameters
    ----------
    group_id : int
        Unique group identifier.
    member_ids : list[int]
        Pedestrian IDs belonging to this group.
    formation : FormationType
        Desired spatial formation.
    leader_id : int or None
        ID of the group leader, if any.
    goal : numpy.ndarray or None
        Shared group goal position.
    cohesion_strength : float
        Strength of the cohesion force keeping members together.
    repulsion_strength : float
        Strength of inter-member repulsion (prevent overlap).
    max_spread : float
        Maximum allowed distance (m) of any member from the centroid.
    desired_spacing : float
        Preferred inter-member distance (m).
    velocity_consensus_weight : float
        Weight for velocity consensus computation (0..1).
    """

    group_id: int = 0
    member_ids: list[int] = field(default_factory=list)
    formation: FormationType = FormationType.CLUSTER
    leader_id: int | None = None
    goal: np.ndarray | None = None
    cohesion_strength: float = 1.5
    repulsion_strength: float = 2.0
    max_spread: float = 3.0
    desired_spacing: float = 1.0
    velocity_consensus_weight: float = 0.6
    _f_formation: FFormation = field(default_factory=FFormation)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- properties -----------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of members in the group."""
        return len(self.member_ids)

    @property
    def has_leader(self) -> bool:
        """Whether the group has a designated leader."""
        return self.leader_id is not None

    # -- geometry helpers -----------------------------------------------------

    @staticmethod
    def _centroid(states: list[PedestrianState]) -> np.ndarray:
        """Compute the centroid position of *states*.

        Parameters
        ----------
        states : list[PedestrianState]
            Group member states.

        Returns
        -------
        numpy.ndarray
            2-D centroid.
        """
        if not states:
            return np.zeros(2, dtype=np.float64)
        return np.mean([s.position for s in states], axis=0)

    @staticmethod
    def _mean_velocity(states: list[PedestrianState]) -> np.ndarray:
        """Compute mean velocity of *states*.

        Parameters
        ----------
        states : list[PedestrianState]
            Group member states.

        Returns
        -------
        numpy.ndarray
            2-D mean velocity.
        """
        if not states:
            return np.zeros(2, dtype=np.float64)
        return np.mean([s.velocity for s in states], axis=0)

    def _get_member_states(self, all_states: dict[int, PedestrianState]) -> list[PedestrianState]:
        """Extract member states from the global state dictionary.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            All pedestrian states keyed by ID.

        Returns
        -------
        list[PedestrianState]
            States of group members (in member_ids order).
        """
        return [all_states[mid] for mid in self.member_ids if mid in all_states]

    # -- formation targets ----------------------------------------------------

    def formation_targets(
        self,
        all_states: dict[int, PedestrianState],
    ) -> dict[int, np.ndarray]:
        """Compute desired positions for each member given the current formation.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dictionary.

        Returns
        -------
        dict[int, numpy.ndarray]
            Map from member ID to desired 2-D position.
        """
        members = self._get_member_states(all_states)
        if not members:
            return {}

        centroid = self._centroid(members)
        mean_vel = self._mean_velocity(members)
        heading = (
            float(math.atan2(mean_vel[1], mean_vel[0])) if np.linalg.norm(mean_vel) > 1e-8 else 0.0
        )
        n = len(members)

        if self.formation == FormationType.F_FORMATION:
            positions = self._f_formation.compute_positions(centroid, n, heading)
            return {mid: pos for mid, pos in zip(self.member_ids[:n], positions, strict=False)}

        if self.formation == FormationType.LINE:
            return self._line_positions(centroid, heading, n)

        if self.formation == FormationType.ABREAST:
            return self._abreast_positions(centroid, heading, n)

        if self.formation == FormationType.V_SHAPE:
            return self._v_shape_positions(centroid, heading, n)

        # Default: CLUSTER - members stay near centroid with spacing.
        return self._cluster_positions(centroid, n)

    def _line_positions(
        self, centroid: np.ndarray, heading: float, n: int
    ) -> dict[int, np.ndarray]:
        """Compute single-file line formation positions.

        Parameters
        ----------
        centroid : numpy.ndarray
            Group centroid.
        heading : float
            Movement direction.
        n : int
            Number of members.

        Returns
        -------
        dict[int, numpy.ndarray]
            Target positions keyed by member ID.
        """
        direction = np.array([math.cos(heading), math.sin(heading)])
        targets: dict[int, np.ndarray] = {}
        for i, mid in enumerate(self.member_ids[:n]):
            offset = -(i - (n - 1) / 2.0) * self.desired_spacing
            targets[mid] = centroid + direction * offset
        return targets

    def _abreast_positions(
        self, centroid: np.ndarray, heading: float, n: int
    ) -> dict[int, np.ndarray]:
        """Compute side-by-side (abreast) formation positions.

        Parameters
        ----------
        centroid : numpy.ndarray
            Group centroid.
        heading : float
            Movement direction.
        n : int
            Number of members.

        Returns
        -------
        dict[int, numpy.ndarray]
            Target positions keyed by member ID.
        """
        perp = np.array([-math.sin(heading), math.cos(heading)])
        targets: dict[int, np.ndarray] = {}
        for i, mid in enumerate(self.member_ids[:n]):
            offset = (i - (n - 1) / 2.0) * self.desired_spacing
            targets[mid] = centroid + perp * offset
        return targets

    def _v_shape_positions(
        self, centroid: np.ndarray, heading: float, n: int
    ) -> dict[int, np.ndarray]:
        """Compute V-shape formation positions.

        The leader (index 0) is at the tip; others spread behind.

        Parameters
        ----------
        centroid : numpy.ndarray
            Group centroid.
        heading : float
            Movement direction.
        n : int
            Number of members.

        Returns
        -------
        dict[int, numpy.ndarray]
            Target positions keyed by member ID.
        """
        direction = np.array([math.cos(heading), math.sin(heading)])
        perp = np.array([-math.sin(heading), math.cos(heading)])
        targets: dict[int, np.ndarray] = {}
        for i, mid in enumerate(self.member_ids[:n]):
            if i == 0:
                targets[mid] = centroid + direction * self.desired_spacing * 0.5
            else:
                row = (i + 1) // 2
                side = 1 if i % 2 == 1 else -1
                behind = -direction * row * self.desired_spacing * 0.8
                lateral = perp * side * row * self.desired_spacing * 0.6
                targets[mid] = centroid + behind + lateral
        return targets

    def _cluster_positions(self, centroid: np.ndarray, n: int) -> dict[int, np.ndarray]:
        """Compute cluster formation positions (evenly spaced on a circle).

        Parameters
        ----------
        centroid : numpy.ndarray
            Group centroid.
        n : int
            Number of members.

        Returns
        -------
        dict[int, numpy.ndarray]
            Target positions keyed by member ID.
        """
        if n <= 1:
            return {self.member_ids[0]: centroid.copy()} if n == 1 else {}
        targets: dict[int, np.ndarray] = {}
        for i, mid in enumerate(self.member_ids[:n]):
            angle = 2.0 * math.pi * i / n
            offset = np.array([math.cos(angle), math.sin(angle)]) * self.desired_spacing * 0.5
            targets[mid] = centroid + offset
        return targets

    # -- forces ---------------------------------------------------------------

    def cohesion_force(
        self,
        state: PedestrianState,
        all_states: dict[int, PedestrianState],
    ) -> np.ndarray:
        """Compute the cohesion force pulling *state* towards the group centroid.

        Parameters
        ----------
        state : PedestrianState
            The member for which to compute the force.
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        numpy.ndarray
            2-D force vector.
        """
        members = self._get_member_states(all_states)
        if len(members) < 2:
            return np.zeros(2, dtype=np.float64)

        centroid = self._centroid(members)
        diff = centroid - state.position
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return np.zeros(2, dtype=np.float64)

        direction = diff / dist
        magnitude = self.cohesion_strength * max(0.0, dist - self.desired_spacing * 0.5)
        return direction * magnitude

    def repulsion_force(
        self,
        state: PedestrianState,
        all_states: dict[int, PedestrianState],
    ) -> np.ndarray:
        """Compute the intra-group repulsion force for *state*.

        Prevents group members from overlapping.

        Parameters
        ----------
        state : PedestrianState
            The member for which to compute the force.
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        numpy.ndarray
            2-D force vector.
        """
        force = np.zeros(2, dtype=np.float64)
        for mid in self.member_ids:
            if mid == state.pid or mid not in all_states:
                continue
            other = all_states[mid]
            diff = state.position - other.position
            dist = float(np.linalg.norm(diff))
            if dist < 1e-9:
                # Random direction to break symmetry.
                diff = np.array([1.0, 0.0])
                dist = 1e-9
            min_dist = state.radius + other.radius
            if dist < min_dist * 2.0:
                direction = diff / dist
                overlap = max(0.0, min_dist * 2.0 - dist)
                force += direction * self.repulsion_strength * overlap
        return force

    def formation_force(
        self,
        state: PedestrianState,
        all_states: dict[int, PedestrianState],
        strength: float = 1.0,
    ) -> np.ndarray:
        """Compute a force steering *state* towards its formation target.

        Parameters
        ----------
        state : PedestrianState
            The member.
        all_states : dict[int, PedestrianState]
            Global state dict.
        strength : float
            Force scaling factor.

        Returns
        -------
        numpy.ndarray
            2-D force vector.
        """
        targets = self.formation_targets(all_states)
        target = targets.get(state.pid)
        if target is None:
            return np.zeros(2, dtype=np.float64)

        diff = target - state.position
        return diff * strength

    # -- velocity consensus ---------------------------------------------------

    def consensus_velocity(
        self,
        all_states: dict[int, PedestrianState],
    ) -> np.ndarray:
        """Compute a consensus velocity for the group.

        Uses weighted average of member velocities.  If a leader exists, the
        leader's velocity is weighted more heavily.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        numpy.ndarray
            Consensus 2-D velocity.
        """
        members = self._get_member_states(all_states)
        if not members:
            return np.zeros(2, dtype=np.float64)

        weights: list[float] = []
        vels: list[np.ndarray] = []
        for m in members:
            w = 2.0 if m.pid == self.leader_id else 1.0
            weights.append(w)
            vels.append(m.velocity)

        w_arr = np.array(weights, dtype=np.float64)
        w_arr /= w_arr.sum()
        vel_arr = np.array(vels, dtype=np.float64)
        return np.einsum("i,ij->j", w_arr, vel_arr)

    def blended_velocity(
        self,
        state: PedestrianState,
        individual_velocity: np.ndarray,
        all_states: dict[int, PedestrianState],
    ) -> np.ndarray:
        """Blend *individual_velocity* with group consensus.

        Parameters
        ----------
        state : PedestrianState
            The member pedestrian.
        individual_velocity : numpy.ndarray
            The individually computed preferred velocity.
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        numpy.ndarray
            Blended 2-D velocity.
        """
        consensus = self.consensus_velocity(all_states)
        w = self.velocity_consensus_weight
        # Leaders are less influenced by consensus.
        if state.pid == self.leader_id:
            w *= 0.3
        return individual_velocity * (1.0 - w) + consensus * w

    # -- leader-follower ------------------------------------------------------

    def leader_follower_force(
        self,
        state: PedestrianState,
        all_states: dict[int, PedestrianState],
        follow_distance: float = 1.2,
        follow_strength: float = 1.5,
    ) -> np.ndarray:
        """Compute a force for a follower to stay behind the leader.

        Parameters
        ----------
        state : PedestrianState
            Follower state.
        all_states : dict[int, PedestrianState]
            Global state dict.
        follow_distance : float
            Desired following distance (m).
        follow_strength : float
            Force magnitude scaling.

        Returns
        -------
        numpy.ndarray
            2-D force vector. Zero if *state* is the leader or no leader set.
        """
        if self.leader_id is None or state.pid == self.leader_id:
            return np.zeros(2, dtype=np.float64)
        if self.leader_id not in all_states:
            return np.zeros(2, dtype=np.float64)

        leader = all_states[self.leader_id]
        leader_heading = leader.heading
        behind = -np.array([math.cos(leader_heading), math.sin(leader_heading)])
        target = leader.position + behind * follow_distance
        diff = target - state.position
        return diff * follow_strength

    # -- splitting / merging --------------------------------------------------

    def should_split(
        self,
        all_states: dict[int, PedestrianState],
        spread_threshold: float | None = None,
    ) -> bool:
        """Determine whether the group should split.

        Splitting is triggered when any member is further than
        *spread_threshold* from the centroid.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.
        spread_threshold : float or None
            Override for *max_spread*.

        Returns
        -------
        bool
            ``True`` if splitting is warranted.
        """
        threshold = spread_threshold if spread_threshold is not None else self.max_spread
        members = self._get_member_states(all_states)
        if len(members) < 2:
            return False
        centroid = self._centroid(members)
        for m in members:
            if float(np.linalg.norm(m.position - centroid)) > threshold:
                return True
        return False

    def split(
        self,
        all_states: dict[int, PedestrianState],
        next_group_id: int,
    ) -> SocialGroup | None:
        """Split the group by separating the furthest member(s).

        The furthest member from the centroid is removed and placed in a new
        group.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.
        next_group_id : int
            ID for the newly created group.

        Returns
        -------
        SocialGroup or None
            The new group containing the split-off members, or ``None`` if
            the group cannot be split (fewer than 2 members).
        """
        members = self._get_member_states(all_states)
        if len(members) < 2:
            return None

        centroid = self._centroid(members)
        dists = [
            (mid, float(np.linalg.norm(all_states[mid].position - centroid)))
            for mid in self.member_ids
            if mid in all_states
        ]
        dists.sort(key=lambda x: x[1], reverse=True)

        # Remove the furthest member.
        split_id = dists[0][0]
        self.member_ids.remove(split_id)
        if self.leader_id == split_id:
            self.leader_id = self.member_ids[0] if self.member_ids else None

        new_group = SocialGroup(
            group_id=next_group_id,
            member_ids=[split_id],
            formation=FormationType.CLUSTER,
            goal=self.goal.copy() if self.goal is not None else None,
        )
        return new_group

    @staticmethod
    def can_merge(
        group_a: SocialGroup,
        group_b: SocialGroup,
        all_states: dict[int, PedestrianState],
        merge_distance: float = 2.0,
    ) -> bool:
        """Check whether two groups are close enough to merge.

        Parameters
        ----------
        group_a, group_b : SocialGroup
            The two candidate groups.
        all_states : dict[int, PedestrianState]
            Global state dict.
        merge_distance : float
            Maximum centroid-to-centroid distance for merging.

        Returns
        -------
        bool
            ``True`` if the groups should merge.
        """
        members_a = group_a._get_member_states(all_states)
        members_b = group_b._get_member_states(all_states)
        if not members_a or not members_b:
            return False
        ca = SocialGroup._centroid(members_a)
        cb = SocialGroup._centroid(members_b)
        return float(np.linalg.norm(ca - cb)) < merge_distance

    def merge(self, other: SocialGroup) -> None:
        """Absorb *other* group into this group.

        Parameters
        ----------
        other : SocialGroup
            Group to absorb.
        """
        for mid in other.member_ids:
            if mid not in self.member_ids:
                self.member_ids.append(mid)
        other.member_ids.clear()

    # -- group path planning --------------------------------------------------

    def group_goal_velocity(
        self,
        all_states: dict[int, PedestrianState],
        speed: float = 1.0,
    ) -> np.ndarray:
        """Compute velocity towards the group goal from the current centroid.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.
        speed : float
            Desired approach speed.

        Returns
        -------
        numpy.ndarray
            2-D velocity towards the group goal.
        """
        if self.goal is None:
            return np.zeros(2, dtype=np.float64)
        members = self._get_member_states(all_states)
        if not members:
            return np.zeros(2, dtype=np.float64)

        centroid = self._centroid(members)
        diff = self.goal - centroid
        dist = float(np.linalg.norm(diff))
        if dist < 1e-9:
            return np.zeros(2, dtype=np.float64)
        return (diff / dist) * speed

    # -- intersection decision-making -----------------------------------------

    def decide_at_intersection(
        self,
        all_states: dict[int, PedestrianState],
        options: list[np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Group decision-making at an intersection with multiple path options.

        Uses a simple voting mechanism: each member votes for the option
        closest to their individual goal.  Ties are broken randomly.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.
        options : list[numpy.ndarray]
            Available direction vectors (2-D, unit or scaled).
        rng : numpy.random.Generator
            Random number generator for tie-breaking.

        Returns
        -------
        numpy.ndarray
            Chosen direction vector.
        """
        if not options:
            return np.zeros(2, dtype=np.float64)
        if len(options) == 1:
            return options[0].copy()

        members = self._get_member_states(all_states)
        votes = np.zeros(len(options), dtype=np.int64)

        for m in members:
            goal_dir = m.goal - m.position
            norm = float(np.linalg.norm(goal_dir))
            if norm < 1e-9:
                continue
            goal_dir = goal_dir / norm
            # Vote for option most aligned with personal goal direction.
            dots = [
                float(np.dot(goal_dir, opt / max(np.linalg.norm(opt), 1e-9))) for opt in options
            ]
            best = int(np.argmax(dots))
            votes[best] += 1

        # If leader exists, double the leader's vote.
        if self.leader_id is not None and self.leader_id in all_states:
            leader = all_states[self.leader_id]
            goal_dir = leader.goal - leader.position
            norm = float(np.linalg.norm(goal_dir))
            if norm > 1e-9:
                goal_dir = goal_dir / norm
                dots = [
                    float(np.dot(goal_dir, opt / max(np.linalg.norm(opt), 1e-9))) for opt in options
                ]
                best = int(np.argmax(dots))
                votes[best] += 1  # Extra vote for leader.

        max_votes = int(votes.max())
        winners = [i for i in range(len(options)) if votes[i] == max_votes]
        chosen_idx = int(rng.choice(winners))
        return options[chosen_idx].copy()

    # -- combined step --------------------------------------------------------

    def compute_social_forces(
        self,
        state: PedestrianState,
        all_states: dict[int, PedestrianState],
    ) -> np.ndarray:
        """Compute the total social force on *state* from group dynamics.

        Combines cohesion, repulsion, and formation forces.

        Parameters
        ----------
        state : PedestrianState
            The group member.
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        numpy.ndarray
            Total 2-D social force.
        """
        f_coh = self.cohesion_force(state, all_states)
        f_rep = self.repulsion_force(state, all_states)
        f_form = self.formation_force(state, all_states)
        f_follow = self.leader_follower_force(state, all_states)
        return f_coh + f_rep + f_form + f_follow

    def spread(self, all_states: dict[int, PedestrianState]) -> float:
        """Compute the maximum member distance from centroid.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        float
            Maximum spread distance (m).  0 if fewer than 2 members.
        """
        members = self._get_member_states(all_states)
        if len(members) < 2:
            return 0.0
        centroid = self._centroid(members)
        return float(max(np.linalg.norm(m.position - centroid) for m in members))


# ---------------------------------------------------------------------------
# Group manager
# ---------------------------------------------------------------------------


class GroupManager:
    """Manages creation, splitting, merging and querying of social groups.

    Parameters
    ----------
    merge_distance : float
        Centroid distance below which groups can merge.
    split_spread : float
        Spread threshold that triggers group splitting.
    """

    def __init__(
        self,
        merge_distance: float = 2.0,
        split_spread: float = 4.0,
    ) -> None:
        self.merge_distance: float = merge_distance
        self.split_spread: float = split_spread
        self._groups: dict[int, SocialGroup] = {}
        self._next_id: int = 0

    # -- accessors ------------------------------------------------------------

    @property
    def groups(self) -> dict[int, SocialGroup]:
        """All managed groups keyed by group ID."""
        return self._groups

    def get_group(self, group_id: int) -> SocialGroup | None:
        """Return the group with *group_id*, or ``None``.

        Parameters
        ----------
        group_id : int
            Group identifier.
        """
        return self._groups.get(group_id)

    def group_of(self, pid: int) -> SocialGroup | None:
        """Return the group containing pedestrian *pid*, or ``None``.

        Parameters
        ----------
        pid : int
            Pedestrian identifier.
        """
        for g in self._groups.values():
            if pid in g.member_ids:
                return g
        return None

    # -- mutation -------------------------------------------------------------

    def create_group(
        self,
        member_ids: list[int],
        formation: FormationType = FormationType.CLUSTER,
        leader_id: int | None = None,
        goal: np.ndarray | None = None,
        **kwargs: Any,
    ) -> SocialGroup:
        """Create and register a new group.

        Parameters
        ----------
        member_ids : list[int]
            Pedestrian IDs.
        formation : FormationType
            Desired formation.
        leader_id : int or None
            Optional leader ID.
        goal : numpy.ndarray or None
            Optional shared goal.
        **kwargs
            Extra keyword arguments forwarded to :class:`SocialGroup`.

        Returns
        -------
        SocialGroup
            The newly created group.
        """
        gid = self._next_id
        self._next_id += 1
        group = SocialGroup(
            group_id=gid,
            member_ids=list(member_ids),
            formation=formation,
            leader_id=leader_id,
            goal=goal,
            **kwargs,
        )
        self._groups[gid] = group
        return group

    def remove_group(self, group_id: int) -> None:
        """Remove a group by ID.

        Parameters
        ----------
        group_id : int
            Group to remove.
        """
        self._groups.pop(group_id, None)

    def step(
        self,
        all_states: dict[int, PedestrianState],
    ) -> list[dict[str, Any]]:
        """Run one management step: check for splits and merges.

        Parameters
        ----------
        all_states : dict[int, PedestrianState]
            Global state dict.

        Returns
        -------
        list[dict]
            List of event records (``{"type": "split"|"merge", ...}``).
        """
        events: list[dict[str, Any]] = []

        # -- splitting --------------------------------------------------------
        to_split: list[int] = []
        for gid, g in self._groups.items():
            if g.should_split(all_states, spread_threshold=self.split_spread):
                to_split.append(gid)
        for gid in to_split:
            new_g = self._groups[gid].split(all_states, self._next_id)
            if new_g is not None:
                self._next_id += 1
                self._groups[new_g.group_id] = new_g
                events.append(
                    {
                        "type": "split",
                        "parent_group": gid,
                        "new_group": new_g.group_id,
                        "new_members": list(new_g.member_ids),
                    }
                )

        # -- merging ----------------------------------------------------------
        gids = list(self._groups.keys())
        merged: set[int] = set()
        for i in range(len(gids)):
            if gids[i] in merged:
                continue
            for j in range(i + 1, len(gids)):
                if gids[j] in merged:
                    continue
                ga = self._groups[gids[i]]
                gb = self._groups[gids[j]]
                if SocialGroup.can_merge(ga, gb, all_states, self.merge_distance):
                    ga.merge(gb)
                    merged.add(gids[j])
                    events.append(
                        {
                            "type": "merge",
                            "absorbing_group": gids[i],
                            "absorbed_group": gids[j],
                        }
                    )
        for gid in merged:
            del self._groups[gid]

        # -- remove empty groups ----------------------------------------------
        empty = [gid for gid, g in self._groups.items() if g.size == 0]
        for gid in empty:
            del self._groups[gid]

        return events
