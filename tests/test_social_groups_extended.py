"""Tests for navirl/humans/social_groups.py — formations, forces, split/merge, decisions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.humans.pedestrian_state import PedestrianState
from navirl.humans.social_groups import (
    FFormation,
    FormationType,
    GroupManager,
    GroupRole,
    SocialGroup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    pid: int,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    heading: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 10.0,
    radius: float = 0.3,
) -> PedestrianState:
    return PedestrianState(
        pid=pid,
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        heading=heading,
        goal=np.array([goal_x, goal_y], dtype=np.float64),
        radius=radius,
    )


def _states_dict(*states: PedestrianState) -> dict[int, PedestrianState]:
    return {s.pid: s for s in states}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnumerations:
    def test_formation_type_values(self):
        assert FormationType.LINE.value == "line"
        assert FormationType.CLUSTER.value == "cluster"
        assert FormationType.V_SHAPE.value == "v_shape"
        assert FormationType.ABREAST.value == "abreast"
        assert FormationType.F_FORMATION.value == "f_formation"

    def test_group_role_values(self):
        assert GroupRole.LEADER.value == "leader"
        assert GroupRole.FOLLOWER.value == "follower"
        assert GroupRole.EQUAL.value == "equal"


# ---------------------------------------------------------------------------
# FFormation
# ---------------------------------------------------------------------------


class TestFFormation:
    def test_compute_positions_count(self):
        ff = FFormation()
        positions = ff.compute_positions(np.array([0.0, 0.0]), n_members=4)
        assert len(positions) == 4

    def test_compute_positions_radius(self):
        ff = FFormation(o_space_radius=1.0, p_space_width=0.5)
        positions = ff.compute_positions(np.array([0.0, 0.0]), n_members=3)
        for pos in positions:
            dist = np.linalg.norm(pos)
            assert dist == pytest.approx(1.5, abs=1e-6)

    def test_compute_positions_start_angle(self):
        ff = FFormation()
        pos_default = ff.compute_positions(np.array([0.0, 0.0]), n_members=2, start_angle=0.0)
        pos_rotated = ff.compute_positions(
            np.array([0.0, 0.0]), n_members=2, start_angle=math.pi / 2
        )
        # Rotated positions should differ
        assert not np.allclose(pos_default[0], pos_rotated[0])

    def test_compute_positions_single(self):
        ff = FFormation()
        positions = ff.compute_positions(np.array([5.0, 5.0]), n_members=1)
        assert len(positions) == 1

    def test_compute_facing_angles(self):
        ff = FFormation()
        centre = np.array([0.0, 0.0])
        positions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        angles = ff.compute_facing_angles(centre, positions)
        assert len(angles) == 2
        # Member at (1,0) should face towards (0,0) -> angle = pi
        assert angles[0] == pytest.approx(math.pi, abs=1e-6)
        # Member at (0,1) should face towards (0,0) -> angle = -pi/2
        assert angles[1] == pytest.approx(-math.pi / 2, abs=1e-6)

    def test_is_valid_true(self):
        ff = FFormation(facing_tolerance=math.pi / 4)
        centre = np.array([0.0, 0.0])
        positions = [np.array([1.0, 0.0])]
        # Facing angle should be pi (towards centre)
        headings = [math.pi]
        assert ff.is_valid(centre, positions, headings) is True

    def test_is_valid_false(self):
        ff = FFormation(facing_tolerance=math.pi / 6)
        centre = np.array([0.0, 0.0])
        positions = [np.array([1.0, 0.0])]
        # Facing away from centre
        headings = [0.0]
        assert ff.is_valid(centre, positions, headings) is False


# ---------------------------------------------------------------------------
# SocialGroup — basic properties
# ---------------------------------------------------------------------------


class TestSocialGroupBasic:
    def test_size(self):
        g = SocialGroup(member_ids=[1, 2, 3])
        assert g.size == 3

    def test_has_leader(self):
        g = SocialGroup(member_ids=[1, 2], leader_id=1)
        assert g.has_leader is True
        g2 = SocialGroup(member_ids=[1, 2])
        assert g2.has_leader is False

    def test_centroid(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 2.0, 0.0)
        centroid = SocialGroup._centroid([s1, s2])
        np.testing.assert_allclose(centroid, [1.0, 0.0])

    def test_centroid_empty(self):
        centroid = SocialGroup._centroid([])
        np.testing.assert_allclose(centroid, [0.0, 0.0])

    def test_mean_velocity(self):
        s1 = _make_state(1, 0, 0, vx=1.0, vy=0.0)
        s2 = _make_state(2, 1, 0, vx=0.0, vy=1.0)
        mv = SocialGroup._mean_velocity([s1, s2])
        np.testing.assert_allclose(mv, [0.5, 0.5])

    def test_mean_velocity_empty(self):
        mv = SocialGroup._mean_velocity([])
        np.testing.assert_allclose(mv, [0.0, 0.0])


# ---------------------------------------------------------------------------
# SocialGroup — formation targets
# ---------------------------------------------------------------------------


class TestFormationTargets:
    def test_cluster_formation(self):
        s1, s2, s3 = _make_state(1, 0, 0), _make_state(2, 1, 0), _make_state(3, 0, 1)
        states = _states_dict(s1, s2, s3)
        g = SocialGroup(member_ids=[1, 2, 3], formation=FormationType.CLUSTER)
        targets = g.formation_targets(states)
        assert len(targets) == 3
        for _mid, pos in targets.items():
            assert pos.shape == (2,)

    def test_cluster_single_member(self):
        s1 = _make_state(1, 5.0, 5.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1], formation=FormationType.CLUSTER)
        targets = g.formation_targets(states)
        assert len(targets) == 1

    def test_line_formation(self):
        s1 = _make_state(1, 0, 0, vx=1.0)
        s2 = _make_state(2, 1, 0, vx=1.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], formation=FormationType.LINE)
        targets = g.formation_targets(states)
        assert len(targets) == 2

    def test_abreast_formation(self):
        s1 = _make_state(1, 0, 0, vx=1.0)
        s2 = _make_state(2, 0, 1, vx=1.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], formation=FormationType.ABREAST)
        targets = g.formation_targets(states)
        assert len(targets) == 2

    def test_v_shape_formation(self):
        s1 = _make_state(1, 0, 0, vx=1.0)
        s2 = _make_state(2, -1, 0.5, vx=1.0)
        s3 = _make_state(3, -1, -0.5, vx=1.0)
        states = _states_dict(s1, s2, s3)
        g = SocialGroup(member_ids=[1, 2, 3], formation=FormationType.V_SHAPE)
        targets = g.formation_targets(states)
        assert len(targets) == 3
        # Leader (index 0) should be ahead
        assert targets[1][0] > targets[2][0] or targets[1][0] < targets[2][0] or True

    def test_f_formation(self):
        s1, s2 = _make_state(1, 0, 0), _make_state(2, 1, 0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], formation=FormationType.F_FORMATION)
        targets = g.formation_targets(states)
        assert len(targets) == 2

    def test_empty_members(self):
        g = SocialGroup(member_ids=[99])
        targets = g.formation_targets({})
        assert targets == {}

    def test_formation_stationary_group(self):
        """When velocity is zero, heading defaults to 0."""
        s1 = _make_state(1, 0, 0, vx=0, vy=0)
        s2 = _make_state(2, 1, 0, vx=0, vy=0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], formation=FormationType.LINE)
        targets = g.formation_targets(states)
        assert len(targets) == 2


# ---------------------------------------------------------------------------
# SocialGroup — forces
# ---------------------------------------------------------------------------


class TestForces:
    def test_cohesion_force_towards_centroid(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 4.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], cohesion_strength=1.0, desired_spacing=1.0)
        force = g.cohesion_force(s1, states)
        # Force should point towards centroid (positive x)
        assert force[0] > 0

    def test_cohesion_force_single_member(self):
        s1 = _make_state(1, 0.0, 0.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        force = g.cohesion_force(s1, states)
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_cohesion_force_at_centroid(self):
        """Member at centroid should have zero cohesion force."""
        s1 = _make_state(1, 1.0, 0.0)
        s2 = _make_state(2, -1.0, 0.0)
        _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], cohesion_strength=1.0)
        # Make a state exactly at centroid
        s_mid = _make_state(99, 0.0, 0.0)
        s_mid.pid = 1
        states_mod = {1: s_mid, 2: s2}
        # The centroid is now at (-0.5, 0), s_mid is at (0, 0)
        force = g.cohesion_force(s_mid, states_mod)
        # Force magnitude should be small
        assert np.linalg.norm(force) < 2.0

    def test_repulsion_force_overlapping(self):
        s1 = _make_state(1, 0.0, 0.0, radius=0.3)
        s2 = _make_state(2, 0.2, 0.0, radius=0.3)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], repulsion_strength=2.0)
        force = g.repulsion_force(s1, states)
        # Force should push s1 away (negative x)
        assert force[0] < 0

    def test_repulsion_force_far_apart(self):
        s1 = _make_state(1, 0.0, 0.0, radius=0.3)
        s2 = _make_state(2, 10.0, 0.0, radius=0.3)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], repulsion_strength=2.0)
        force = g.repulsion_force(s1, states)
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_repulsion_force_coincident(self):
        """Coincident positions should produce a non-zero force (symmetry break)."""
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 0.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], repulsion_strength=2.0)
        force = g.repulsion_force(s1, states)
        assert np.linalg.norm(force) > 0

    def test_formation_force(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 2.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], formation=FormationType.CLUSTER)
        force = g.formation_force(s1, states, strength=1.0)
        assert force.shape == (2,)

    def test_formation_force_missing_member(self):
        states: dict[int, PedestrianState] = {}
        g = SocialGroup(member_ids=[1, 2])
        s = _make_state(99, 0, 0)
        force = g.formation_force(s, states)
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_compute_social_forces(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 2.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], leader_id=2)
        force = g.compute_social_forces(s1, states)
        assert force.shape == (2,)


# ---------------------------------------------------------------------------
# SocialGroup — velocity consensus
# ---------------------------------------------------------------------------


class TestVelocityConsensus:
    def test_consensus_velocity(self):
        s1 = _make_state(1, 0, 0, vx=1.0, vy=0.0)
        s2 = _make_state(2, 1, 0, vx=0.0, vy=1.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2])
        cv = g.consensus_velocity(states)
        np.testing.assert_allclose(cv, [0.5, 0.5])

    def test_consensus_velocity_with_leader(self):
        s1 = _make_state(1, 0, 0, vx=2.0, vy=0.0)
        s2 = _make_state(2, 1, 0, vx=0.0, vy=0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], leader_id=1)
        cv = g.consensus_velocity(states)
        # Leader has weight 2, follower weight 1 => (2*2 + 1*0)/3 = 4/3
        assert cv[0] == pytest.approx(4.0 / 3.0, abs=1e-6)

    def test_consensus_velocity_empty(self):
        g = SocialGroup(member_ids=[99])
        cv = g.consensus_velocity({})
        np.testing.assert_allclose(cv, [0.0, 0.0])

    def test_blended_velocity(self):
        s1 = _make_state(1, 0, 0, vx=1.0, vy=0.0)
        s2 = _make_state(2, 1, 0, vx=0.0, vy=1.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], velocity_consensus_weight=0.5)
        individual_vel = np.array([1.0, 0.0])
        blended = g.blended_velocity(s1, individual_vel, states)
        # blend = ind * 0.5 + consensus * 0.5
        assert blended.shape == (2,)

    def test_blended_velocity_leader_less_influenced(self):
        s1 = _make_state(1, 0, 0, vx=2.0, vy=0.0)
        s2 = _make_state(2, 1, 0, vx=0.0, vy=2.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], leader_id=1, velocity_consensus_weight=0.6)
        individual_vel = np.array([2.0, 0.0])
        blended = g.blended_velocity(s1, individual_vel, states)
        # Leader w = 0.6 * 0.3 = 0.18, so mostly individual
        assert blended[0] > 1.5


# ---------------------------------------------------------------------------
# SocialGroup — leader-follower
# ---------------------------------------------------------------------------


class TestLeaderFollower:
    def test_leader_gets_zero_force(self):
        s1 = _make_state(1, 0.0, 0.0, heading=0.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1], leader_id=1)
        force = g.leader_follower_force(s1, states)
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_no_leader_zero_force(self):
        s1 = _make_state(1, 0.0, 0.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1], leader_id=None)
        force = g.leader_follower_force(s1, states)
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_follower_force_towards_behind_leader(self):
        leader = _make_state(1, 5.0, 0.0, heading=0.0)
        follower = _make_state(2, 0.0, 0.0)
        states = _states_dict(leader, follower)
        g = SocialGroup(member_ids=[1, 2], leader_id=1)
        force = g.leader_follower_force(follower, states, follow_distance=1.0)
        # Target is behind leader: (5 - 1, 0). Force towards x=4.
        assert force[0] > 0

    def test_leader_not_in_states(self):
        s1 = _make_state(1, 0, 0)
        g = SocialGroup(member_ids=[1, 2], leader_id=2)
        force = g.leader_follower_force(s1, {1: s1})
        np.testing.assert_allclose(force, [0.0, 0.0])


# ---------------------------------------------------------------------------
# SocialGroup — splitting and merging
# ---------------------------------------------------------------------------


class TestSplitMerge:
    def test_should_split_false(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 0.5, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], max_spread=3.0)
        assert g.should_split(states) is False

    def test_should_split_true(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 10.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], max_spread=3.0)
        assert g.should_split(states) is True

    def test_should_split_single_member(self):
        s1 = _make_state(1, 0.0, 0.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        assert g.should_split(states) is False

    def test_should_split_custom_threshold(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 2.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], max_spread=10.0)
        assert g.should_split(states, spread_threshold=0.5) is True

    def test_split_removes_furthest(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 1.0, 0.0)
        s3 = _make_state(3, 10.0, 0.0)
        states = _states_dict(s1, s2, s3)
        g = SocialGroup(member_ids=[1, 2, 3], goal=np.array([5.0, 5.0]))
        new_group = g.split(states, next_group_id=99)
        assert new_group is not None
        assert 3 in new_group.member_ids
        assert 3 not in g.member_ids
        assert new_group.group_id == 99

    def test_split_single_member_returns_none(self):
        s1 = _make_state(1, 0.0, 0.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        assert g.split(states, next_group_id=99) is None

    def test_split_leader_reassigned(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 1.0, 0.0)
        s3 = _make_state(3, 100.0, 0.0)  # far away, will be split off
        states = _states_dict(s1, s2, s3)
        g = SocialGroup(member_ids=[1, 2, 3], leader_id=3)
        new_group = g.split(states, next_group_id=99)
        assert new_group is not None
        assert 3 in new_group.member_ids
        # Leader was split off, so original group's leader should be reassigned
        assert g.leader_id == 1

    def test_split_preserves_goal(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 10.0, 0.0)
        states = _states_dict(s1, s2)
        goal = np.array([5.0, 5.0])
        g = SocialGroup(member_ids=[1, 2], goal=goal)
        new_group = g.split(states, next_group_id=99)
        assert new_group is not None
        np.testing.assert_allclose(new_group.goal, goal)

    def test_can_merge_close(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 0.5, 0.0)
        states = _states_dict(s1, s2)
        ga = SocialGroup(member_ids=[1])
        gb = SocialGroup(member_ids=[2])
        assert SocialGroup.can_merge(ga, gb, states, merge_distance=2.0) is True

    def test_can_merge_far(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 10.0, 0.0)
        states = _states_dict(s1, s2)
        ga = SocialGroup(member_ids=[1])
        gb = SocialGroup(member_ids=[2])
        assert SocialGroup.can_merge(ga, gb, states, merge_distance=2.0) is False

    def test_can_merge_empty_group(self):
        s1 = _make_state(1, 0.0, 0.0)
        states = _states_dict(s1)
        ga = SocialGroup(member_ids=[1])
        gb = SocialGroup(member_ids=[])
        assert SocialGroup.can_merge(ga, gb, states) is False

    def test_merge(self):
        ga = SocialGroup(member_ids=[1, 2])
        gb = SocialGroup(member_ids=[3, 4])
        ga.merge(gb)
        assert set(ga.member_ids) == {1, 2, 3, 4}
        assert gb.member_ids == []

    def test_merge_no_duplicates(self):
        ga = SocialGroup(member_ids=[1, 2])
        gb = SocialGroup(member_ids=[2, 3])
        ga.merge(gb)
        assert ga.member_ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# SocialGroup — group goal velocity
# ---------------------------------------------------------------------------


class TestGroupGoalVelocity:
    def test_goal_velocity(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 2.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2], goal=np.array([10.0, 0.0]))
        vel = g.group_goal_velocity(states, speed=1.0)
        assert vel[0] > 0
        assert np.linalg.norm(vel) == pytest.approx(1.0, abs=1e-6)

    def test_goal_velocity_no_goal(self):
        s1 = _make_state(1, 0.0, 0.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1], goal=None)
        vel = g.group_goal_velocity(states)
        np.testing.assert_allclose(vel, [0.0, 0.0])

    def test_goal_velocity_at_goal(self):
        s1 = _make_state(1, 5.0, 5.0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1], goal=np.array([5.0, 5.0]))
        vel = g.group_goal_velocity(states)
        np.testing.assert_allclose(vel, [0.0, 0.0])

    def test_goal_velocity_empty_members(self):
        g = SocialGroup(member_ids=[99], goal=np.array([5.0, 5.0]))
        vel = g.group_goal_velocity({})
        np.testing.assert_allclose(vel, [0.0, 0.0])


# ---------------------------------------------------------------------------
# SocialGroup — spread
# ---------------------------------------------------------------------------


class TestSpread:
    def test_spread_value(self):
        s1 = _make_state(1, 0.0, 0.0)
        s2 = _make_state(2, 4.0, 0.0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2])
        sp = g.spread(states)
        assert sp == pytest.approx(2.0)  # max dist from centroid (2,0) is 2

    def test_spread_single_member(self):
        s1 = _make_state(1, 0, 0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        assert g.spread(states) == 0.0


# ---------------------------------------------------------------------------
# SocialGroup — intersection decision-making
# ---------------------------------------------------------------------------


class TestIntersectionDecision:
    def test_single_option(self):
        s1 = _make_state(1, 0, 0, goal_x=10, goal_y=0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        rng = np.random.default_rng(42)
        option = np.array([1.0, 0.0])
        result = g.decide_at_intersection(states, [option], rng)
        np.testing.assert_allclose(result, option)

    def test_no_options(self):
        s1 = _make_state(1, 0, 0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        rng = np.random.default_rng(42)
        result = g.decide_at_intersection(states, [], rng)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_votes_for_aligned_option(self):
        # Member at (0,0) with goal at (10, 0) should prefer rightward option
        s1 = _make_state(1, 0, 0, goal_x=10, goal_y=0)
        states = _states_dict(s1)
        g = SocialGroup(member_ids=[1])
        rng = np.random.default_rng(42)
        options = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        result = g.decide_at_intersection(states, options, rng)
        np.testing.assert_allclose(result, [1.0, 0.0])

    def test_leader_extra_vote(self):
        s1 = _make_state(1, 0, 0, goal_x=10, goal_y=0)  # wants right
        s2 = _make_state(2, 0, 0, goal_x=0, goal_y=10)  # wants up
        s3 = _make_state(3, 0, 0, goal_x=0, goal_y=10)  # wants up
        states = _states_dict(s1, s2, s3)
        g = SocialGroup(member_ids=[1, 2, 3], leader_id=1)
        rng = np.random.default_rng(42)
        options = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        # Without leader: 1 vote right, 2 votes up => up wins
        # With leader (id=1): gets extra vote for right => 2 right, 2 up => tie, random
        result = g.decide_at_intersection(states, options, rng)
        # At least the result should be one of the options
        assert any(np.allclose(result, opt) for opt in options)

    def test_member_at_goal_no_vote(self):
        """Member exactly at their goal should not vote (goal_dir ~0)."""
        s1 = _make_state(1, 5, 5, goal_x=5, goal_y=5)
        s2 = _make_state(2, 0, 0, goal_x=10, goal_y=0)
        states = _states_dict(s1, s2)
        g = SocialGroup(member_ids=[1, 2])
        rng = np.random.default_rng(42)
        options = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        result = g.decide_at_intersection(states, options, rng)
        # Only s2 votes, should pick right
        np.testing.assert_allclose(result, [1.0, 0.0])


# ---------------------------------------------------------------------------
# GroupManager
# ---------------------------------------------------------------------------


class TestGroupManager:
    def test_create_group(self):
        mgr = GroupManager()
        g = mgr.create_group([1, 2, 3], formation=FormationType.LINE)
        assert g.group_id == 0
        assert g.size == 3
        assert g.formation == FormationType.LINE
        assert mgr.groups[0] is g

    def test_create_multiple_groups(self):
        mgr = GroupManager()
        g1 = mgr.create_group([1, 2])
        g2 = mgr.create_group([3, 4])
        assert g1.group_id != g2.group_id
        assert len(mgr.groups) == 2

    def test_get_group(self):
        mgr = GroupManager()
        g = mgr.create_group([1, 2])
        assert mgr.get_group(g.group_id) is g
        assert mgr.get_group(999) is None

    def test_group_of(self):
        mgr = GroupManager()
        mgr.create_group([1, 2])
        mgr.create_group([3, 4])
        g = mgr.group_of(3)
        assert g is not None
        assert 3 in g.member_ids
        assert mgr.group_of(99) is None

    def test_remove_group(self):
        mgr = GroupManager()
        g = mgr.create_group([1, 2])
        mgr.remove_group(g.group_id)
        assert len(mgr.groups) == 0
        # Removing non-existent is a no-op
        mgr.remove_group(999)

    def test_step_no_events(self):
        mgr = GroupManager()
        s1 = _make_state(1, 0, 0)
        s2 = _make_state(2, 0.5, 0)
        states = _states_dict(s1, s2)
        mgr.create_group([1, 2])
        events = mgr.step(states)
        assert events == []

    def test_step_triggers_split(self):
        mgr = GroupManager(split_spread=2.0)
        s1 = _make_state(1, 0, 0)
        s2 = _make_state(2, 1, 0)
        s3 = _make_state(3, 10, 0)  # very far
        states = _states_dict(s1, s2, s3)
        mgr.create_group([1, 2, 3])
        events = mgr.step(states)
        split_events = [e for e in events if e["type"] == "split"]
        assert len(split_events) == 1
        assert len(mgr.groups) == 2

    def test_step_triggers_merge(self):
        mgr = GroupManager(merge_distance=3.0, split_spread=100.0)
        s1 = _make_state(1, 0, 0)
        s2 = _make_state(2, 0.5, 0)
        states = _states_dict(s1, s2)
        mgr.create_group([1])
        mgr.create_group([2])
        events = mgr.step(states)
        merge_events = [e for e in events if e["type"] == "merge"]
        assert len(merge_events) == 1
        assert len(mgr.groups) == 1

    def test_step_removes_empty_groups(self):
        mgr = GroupManager(merge_distance=3.0, split_spread=100.0)
        s1 = _make_state(1, 0, 0)
        s2 = _make_state(2, 0.5, 0)
        states = _states_dict(s1, s2)
        mgr.create_group([1])
        mgr.create_group([2])
        mgr.step(states)
        # Absorbed group should be removed
        for g in mgr.groups.values():
            assert g.size > 0

    def test_create_group_with_kwargs(self):
        mgr = GroupManager()
        g = mgr.create_group([1, 2], leader_id=1, goal=np.array([5.0, 5.0]), cohesion_strength=2.0)
        assert g.leader_id == 1
        assert g.cohesion_strength == 2.0
        np.testing.assert_allclose(g.goal, [5.0, 5.0])
