"""Tests for navirl/humans/ modules: pedestrian_state, behavior_model, crowd_generator, social_groups."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.humans.pedestrian_state import (
    Activity,
    GazeDirection,
    PedestrianState,
    PersonalityTag,
    StateHistory,
    StatePredictor,
    compute_centroid,
    filter_by_activity,
    filter_by_group,
    pairwise_distances,
    states_to_array,
)
from navirl.humans.behavior_model import (
    AttentionModel,
    BehaviorModel,
    PersonalityParams,
    create_behavior_model,
    get_personality_params,
    sample_personality,
)
from navirl.humans.crowd_generator import (
    CrowdGenerator,
    DemographicDistribution,
    FlowPattern,
    GoalAssigner,
    ScenarioType,
    SpawnEvent,
    SpawnRegion,
    SpawnStrategy,
    estimate_density,
    flow_rate,
)
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


def _ped(
    pid: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
    heading: float = 0.0,
    radius: float = 0.3,
    group_id: int | None = None,
    activity: Activity = Activity.WALKING,
    personality: PersonalityTag = PersonalityTag.NORMAL,
) -> PedestrianState:
    return PedestrianState(
        pid=pid,
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        heading=heading,
        goal=np.array([goal_x, goal_y], dtype=np.float64),
        radius=radius,
        group_id=group_id,
        activity=activity,
        personality=personality,
    )


# ===========================================================================
# 1. pedestrian_state
# ===========================================================================


class TestGazeDirection:
    def test_default_unit_vector(self):
        g = GazeDirection()
        ux, uy = g.unit_vector
        assert ux == pytest.approx(1.0)
        assert uy == pytest.approx(0.0)

    def test_set_from_vector(self):
        g = GazeDirection()
        g.set_from_vector(0.0, 1.0)
        assert g.angle_rad == pytest.approx(math.pi / 2.0)

    def test_set_from_zero_vector_no_change(self):
        g = GazeDirection(angle_rad=1.0)
        g.set_from_vector(0.0, 0.0)
        assert g.angle_rad == pytest.approx(1.0)

    def test_angular_distance_wrap(self):
        g = GazeDirection(angle_rad=math.pi - 0.1)
        d = g.angular_distance(-math.pi + 0.1)
        assert abs(d) == pytest.approx(0.2, abs=1e-9)


class TestPedestrianState:
    def test_speed_property(self):
        p = _ped(vx=3.0, vy=4.0)
        assert p.speed == pytest.approx(5.0)

    def test_xy_shortcuts(self):
        p = _ped(x=1.5, y=2.5)
        assert p.x == pytest.approx(1.5)
        assert p.y == pytest.approx(2.5)

    def test_distance_to(self):
        a = _ped(x=0.0, y=0.0)
        b = _ped(x=3.0, y=4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_bearing_to(self):
        a = _ped(x=0.0, y=0.0)
        b = _ped(x=0.0, y=1.0)
        assert a.bearing_to(b) == pytest.approx(math.pi / 2.0)

    def test_in_personal_space(self):
        a = _ped(x=0.0, y=0.0, radius=0.3)
        a.personal_space_radius = 0.6
        close = _ped(x=0.5, y=0.0, radius=0.3)
        far = _ped(x=5.0, y=0.0, radius=0.3)
        assert a.in_personal_space(close)
        assert not a.in_personal_space(far)

    def test_distance_to_goal(self):
        p = _ped(x=0.0, y=0.0, goal_x=3.0, goal_y=4.0)
        assert p.distance_to_goal() == pytest.approx(5.0)

    def test_update_heading_from_velocity(self):
        p = _ped(vx=0.0, vy=1.0)
        p.update_heading_from_velocity()
        assert p.heading == pytest.approx(math.pi / 2.0)

    def test_update_heading_zero_speed_noop(self):
        p = _ped(vx=0.0, vy=0.0)
        p.heading = 1.23
        p.update_heading_from_velocity()
        assert p.heading == pytest.approx(1.23)

    def test_clone_is_independent(self):
        p = _ped(x=1.0, y=2.0)
        p.metadata["key"] = "val"
        c = p.clone()
        c.position[0] = 99.0
        c.metadata["key"] = "changed"
        assert p.x == pytest.approx(1.0)
        assert p.metadata["key"] == "val"

    def test_to_dict_roundtrip(self):
        p = _ped(pid=7, x=1.0, y=2.0, vx=0.5, vy=-0.5)
        p.personality = PersonalityTag.HURRIED
        p.activity = Activity.RUNNING
        d = p.to_dict()
        restored = PedestrianState.from_dict(d)
        assert restored.pid == 7
        assert restored.x == pytest.approx(1.0)
        assert restored.personality == PersonalityTag.HURRIED
        assert restored.activity == Activity.RUNNING

    def test_from_dict_invalid_personality_falls_back(self):
        d = {"pid": 0, "personality": "nonexistent"}
        s = PedestrianState.from_dict(d)
        assert s.personality == PersonalityTag.NORMAL


class TestStateHistory:
    def test_capacity_validation(self):
        with pytest.raises(ValueError):
            StateHistory(capacity=0)

    def test_record_and_latest(self):
        h = StateHistory(capacity=5)
        assert h.latest() is None
        p = _ped(x=1.0, y=2.0)
        h.record(p, 0.0)
        assert len(h) == 1
        assert h.latest().x == pytest.approx(1.0)

    def test_ring_buffer_eviction(self):
        h = StateHistory(capacity=2)
        h.record(_ped(x=1.0), 0.0)
        h.record(_ped(x=2.0), 1.0)
        h.record(_ped(x=3.0), 2.0)
        assert len(h) == 2
        s, t = h.at(0)
        assert s.x == pytest.approx(2.0)

    def test_positions_array_empty(self):
        h = StateHistory()
        arr = h.positions_array()
        assert arr.shape == (0, 2)

    def test_path_length(self):
        h = StateHistory()
        h.record(_ped(x=0.0, y=0.0), 0.0)
        h.record(_ped(x=3.0, y=4.0), 1.0)
        assert h.path_length() == pytest.approx(5.0)

    def test_mean_speed_empty(self):
        assert StateHistory().mean_speed() == 0.0

    def test_window(self):
        h = StateHistory()
        for i in range(5):
            h.record(_ped(x=float(i)), float(i))
        w = h.window(3)
        assert len(w) == 3
        assert w[0][0].x == pytest.approx(2.0)

    def test_clear(self):
        h = StateHistory()
        h.record(_ped(), 0.0)
        h.clear()
        assert len(h) == 0


class TestStatePredictor:
    def test_constant_velocity(self):
        pred = StatePredictor(use_acceleration=False)
        s = _ped(x=0.0, y=0.0, vx=1.0, vy=0.0)
        pos = pred.predict_position(s, 2.0)
        assert pos[0] == pytest.approx(2.0)

    def test_constant_acceleration(self):
        pred = StatePredictor(use_acceleration=True)
        s = _ped(x=0.0, y=0.0, vx=0.0, vy=0.0)
        s.acceleration = np.array([2.0, 0.0])
        pos = pred.predict_position(s, 1.0)
        assert pos[0] == pytest.approx(1.0)

    def test_predict_trajectory_shape(self):
        pred = StatePredictor()
        s = _ped(vx=1.0)
        traj = pred.predict_trajectory(s, horizon=1.0, step_dt=0.5)
        # K = floor(1.0/0.5) + 1 = 3
        assert traj.shape == (3, 2)

    def test_predict_from_history_empty(self):
        pred = StatePredictor()
        assert pred.predict_from_history(StateHistory(), 1.0) is None

    def test_predict_from_history_few_samples(self):
        pred = StatePredictor()
        h = StateHistory()
        h.record(_ped(x=0.0, vx=1.0), 0.0)
        pos = pred.predict_from_history(h, 1.0)
        assert pos is not None
        assert pos[0] == pytest.approx(1.0)

    def test_collision_time_head_on(self):
        pred = StatePredictor()
        a = _ped(x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.25)
        b = _ped(x=5.0, y=0.0, vx=-1.0, vy=0.0, radius=0.25)
        t = pred.collision_time(a, b)
        assert t is not None
        assert t == pytest.approx(2.25, abs=0.01)

    def test_collision_time_no_collision(self):
        pred = StatePredictor()
        a = _ped(x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.25)
        b = _ped(x=0.0, y=10.0, vx=1.0, vy=0.0, radius=0.25)
        assert pred.collision_time(a, b) is None


class TestBatchUtilities:
    def test_states_to_array(self):
        states = [_ped(pid=0, x=1.0, y=2.0), _ped(pid=1, x=3.0, y=4.0)]
        arr = states_to_array(states)
        assert arr.shape == (2, 8)
        assert arr[0, 0] == 0  # pid
        assert arr[1, 1] == pytest.approx(3.0)

    def test_pairwise_distances_empty(self):
        assert pairwise_distances([]).shape == (0, 0)

    def test_pairwise_distances_symmetric(self):
        states = [_ped(x=0.0, y=0.0), _ped(x=3.0, y=4.0)]
        d = pairwise_distances(states)
        assert d[0, 1] == pytest.approx(5.0)
        assert d[1, 0] == pytest.approx(5.0)
        assert d[0, 0] == pytest.approx(0.0)

    def test_filter_by_activity(self):
        states = [_ped(activity=Activity.WALKING), _ped(activity=Activity.STANDING)]
        assert len(filter_by_activity(states, Activity.WALKING)) == 1

    def test_filter_by_group(self):
        states = [_ped(group_id=1), _ped(group_id=2), _ped(group_id=1)]
        assert len(filter_by_group(states, 1)) == 2

    def test_compute_centroid(self):
        states = [_ped(x=0.0, y=0.0), _ped(x=4.0, y=0.0)]
        c = compute_centroid(states)
        assert c[0] == pytest.approx(2.0)

    def test_compute_centroid_empty_raises(self):
        with pytest.raises(ValueError):
            compute_centroid([])


# ===========================================================================
# 2. behavior_model
# ===========================================================================


class TestAttentionModel:
    def test_can_perceive_in_range_and_fov(self):
        attn = AttentionModel(awareness_radius=10.0, field_of_view=math.pi)
        obs = _ped(heading=0.0)
        assert attn.can_perceive(obs, np.array([5.0, 0.0]))

    def test_cannot_perceive_behind(self):
        attn = AttentionModel(awareness_radius=10.0, field_of_view=math.pi / 3.0)
        obs = _ped(heading=0.0)
        assert not attn.can_perceive(obs, np.array([-5.0, 0.0]))

    def test_cannot_perceive_out_of_range(self):
        attn = AttentionModel(awareness_radius=5.0)
        obs = _ped(heading=0.0)
        assert not attn.can_perceive(obs, np.array([20.0, 0.0]))

    def test_distracted_blocks_perception(self):
        attn = AttentionModel()
        attn._distracted = True
        obs = _ped(heading=0.0)
        assert not attn.can_perceive(obs, np.array([1.0, 0.0]))

    def test_effective_awareness_radius_normal_vs_distracted(self):
        attn = AttentionModel(awareness_radius=10.0)
        assert attn.effective_awareness_radius() == pytest.approx(10.0)
        attn._distracted = True
        assert attn.effective_awareness_radius() == pytest.approx(3.0)

    def test_reset(self):
        attn = AttentionModel()
        attn._distracted = True
        attn._distraction_remaining = 5.0
        attn.reset()
        assert not attn.is_distracted


class TestBehaviorModel:
    def test_default_personality(self):
        bm = BehaviorModel()
        assert bm.personality == PersonalityTag.NORMAL

    def test_preferred_speed_scales_with_personality(self):
        hurried = BehaviorModel(personality=PersonalityTag.HURRIED)
        normal = BehaviorModel(personality=PersonalityTag.NORMAL)
        assert hurried.preferred_speed > normal.preferred_speed

    def test_adapt_speed_no_neighbours(self):
        bm = BehaviorModel(rng_seed=42)
        ego = _ped(vx=1.0)
        assert bm.adapt_speed(ego, []) == pytest.approx(bm.preferred_speed)

    def test_compute_comfort_alone(self):
        bm = BehaviorModel(rng_seed=42)
        ego = _ped(vx=bm.preferred_speed)
        c = bm.compute_comfort(ego, [])
        assert 0.8 <= c <= 1.0

    def test_compute_stress_with_time_pressure(self):
        bm = BehaviorModel(personality=PersonalityTag.HURRIED, rng_seed=42)
        ego = _ped()
        s = bm.compute_stress(ego, [], time_pressure=1.0)
        assert s > 0.0

    def test_step_returns_keys(self):
        bm = BehaviorModel(rng_seed=42)
        ego = _ped(vx=1.0, heading=0.0)
        result = bm.step(ego, [], dt=0.1)
        assert "adapted_speed" in result
        assert "comfort" in result
        assert "stress" in result
        assert "is_distracted" in result

    def test_create_smartphone_user(self):
        bm = BehaviorModel.create_smartphone_user(rng_seed=0)
        assert bm.personality == PersonalityTag.DISTRACTED
        assert bm.attention.awareness_radius == pytest.approx(3.0)

    def test_create_elderly(self):
        bm = BehaviorModel.create_elderly(rng_seed=0)
        assert bm.personality == PersonalityTag.ELDERLY

    def test_create_disability(self):
        bm = BehaviorModel.create_disability(rng_seed=0)
        assert bm.preferred_speed < 1.0

    def test_choose_side_returns_valid(self):
        bm = BehaviorModel(rng_seed=42)
        ego = _ped()
        other = _ped(x=5.0)
        side = bm.choose_side(ego, other)
        assert side in (1, -1)


class TestBehaviorModelFactories:
    def test_get_personality_params_all_tags(self):
        for tag in PersonalityTag:
            p = get_personality_params(tag)
            assert isinstance(p, PersonalityParams)

    def test_sample_personality_returns_valid(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            tag = sample_personality(rng)
            assert isinstance(tag, PersonalityTag)

    def test_create_behavior_model_factory(self):
        bm = create_behavior_model(PersonalityTag.AGGRESSIVE, rng_seed=0)
        assert bm.personality == PersonalityTag.AGGRESSIVE


# ===========================================================================
# 3. crowd_generator
# ===========================================================================


class TestSpawnRegion:
    def test_area(self):
        r = SpawnRegion(0.0, 10.0, 0.0, 5.0)
        assert r.area == pytest.approx(50.0)

    def test_centre(self):
        r = SpawnRegion(-10.0, 10.0, -5.0, 5.0)
        np.testing.assert_allclose(r.centre, [0.0, 0.0])

    def test_contains(self):
        r = SpawnRegion(0.0, 10.0, 0.0, 10.0)
        assert r.contains(np.array([5.0, 5.0]))
        assert not r.contains(np.array([-1.0, 5.0]))

    def test_sample_within_bounds(self):
        r = SpawnRegion(0.0, 1.0, 0.0, 1.0)
        rng = np.random.default_rng(42)
        for _ in range(50):
            p = r.sample(rng)
            assert r.contains(p)


class TestDemographicDistribution:
    def test_sample_speed_clamped(self):
        dd = DemographicDistribution(speed_min=0.5, speed_max=2.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = dd.sample_speed(rng)
            assert 0.5 <= s <= 2.0

    def test_sample_radius_clamped(self):
        dd = DemographicDistribution(radius_min=0.2, radius_max=0.5)
        rng = np.random.default_rng(42)
        for _ in range(100):
            r = dd.sample_radius(rng)
            assert 0.2 <= r <= 0.5


class TestCrowdGenerator:
    def test_generate_batch_explicit_count(self):
        gen = CrowdGenerator(rng_seed=42)
        peds = gen.generate_batch(count=5)
        assert len(peds) == 5
        pids = [p.pid for p in peds]
        assert len(set(pids)) == 5  # unique ids

    def test_generate_batch_respects_max(self):
        gen = CrowdGenerator(max_pedestrians=3, rng_seed=42)
        peds = gen.generate_batch(count=10)
        assert len(peds) == 3

    def test_generate_batch_density_based(self):
        region = SpawnRegion(0.0, 10.0, 0.0, 10.0)
        gen = CrowdGenerator(spawn_regions=[region], density=0.1, rng_seed=42)
        peds = gen.generate_batch()
        # 0.1 * 100 = 10
        assert len(peds) == 10

    def test_step_batch_fires_once(self):
        gen = CrowdGenerator(rng_seed=42, max_pedestrians=5)
        first = gen.step(0.0, 0.1)
        second = gen.step(0.1, 0.1)
        assert len(first) > 0
        assert len(second) == 0

    def test_step_poisson(self):
        gen = CrowdGenerator(
            spawn_strategy=SpawnStrategy.POISSON,
            poisson_rate=100.0,
            max_pedestrians=50,
            rng_seed=42,
        )
        peds = gen.step(0.0, 1.0)
        # With rate=100 and dt=1, expect many arrivals
        assert len(peds) > 0

    def test_step_scheduled(self):
        schedule = [SpawnEvent(time_s=0.5, count=2, region_idx=0)]
        gen = CrowdGenerator(
            spawn_strategy=SpawnStrategy.SCHEDULED,
            schedule=schedule,
            rng_seed=42,
        )
        early = gen.step(0.0, 0.1)
        assert len(early) == 0
        at_time = gen.step(0.5, 0.1)
        assert len(at_time) == 2

    def test_reset(self):
        gen = CrowdGenerator(rng_seed=42, max_pedestrians=5)
        gen.generate_batch(count=5)
        gen.reset()
        peds = gen.generate_batch(count=3)
        assert len(peds) == 3
        assert peds[0].pid == 0  # pid counter reset

    def test_commute_scenario(self):
        gen = CrowdGenerator.commute_scenario(rng_seed=42)
        peds = gen.generate_batch()
        assert len(peds) > 0

    def test_random_walk_scenario(self):
        gen = CrowdGenerator.random_walk_scenario(n_pedestrians=10, rng_seed=42)
        peds = gen.generate_batch()
        assert len(peds) == 10


class TestEstimateDensity:
    def test_empty_positions(self):
        grid = estimate_density(np.empty((0, 2)), SpawnRegion(0, 10, 0, 10), cell_size=5.0)
        assert np.all(grid == 0.0)

    def test_single_agent(self):
        pos = np.array([[5.0, 5.0]])
        grid = estimate_density(pos, SpawnRegion(0, 10, 0, 10), cell_size=10.0)
        assert grid.shape == (1, 1)
        assert grid[0, 0] == pytest.approx(0.01)


class TestFlowRate:
    def test_zero_length_line(self):
        pos = np.array([[0.0, 0.0]])
        vel = np.array([[1.0, 0.0]])
        line_s = np.array([5.0, 5.0])
        line_e = np.array([5.0, 5.0])
        assert flow_rate(pos, vel, line_s, line_e) == 0.0

    def test_perpendicular_flow(self):
        pos = np.array([[5.0, 0.0]])
        vel = np.array([[0.0, 1.0]])
        line_s = np.array([0.0, 0.0])
        line_e = np.array([10.0, 0.0])
        result = flow_rate(pos, vel, line_s, line_e, radius=1.0)
        assert result == pytest.approx(1.0)


# ===========================================================================
# 4. social_groups
# ===========================================================================


class TestFFormation:
    def test_compute_positions_count(self):
        ff = FFormation()
        centre = np.array([0.0, 0.0])
        positions = ff.compute_positions(centre, n_members=4)
        assert len(positions) == 4

    def test_compute_facing_angles(self):
        ff = FFormation()
        centre = np.array([0.0, 0.0])
        positions = [np.array([1.0, 0.0])]
        angles = ff.compute_facing_angles(centre, positions)
        assert angles[0] == pytest.approx(math.pi)  # facing toward centre

    def test_is_valid(self):
        ff = FFormation(facing_tolerance=math.pi)
        centre = np.array([0.0, 0.0])
        positions = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
        headings = [math.pi, 0.0]  # facing inward
        assert ff.is_valid(centre, positions, headings)


class TestSocialGroup:
    def test_size_property(self):
        g = SocialGroup(member_ids=[1, 2, 3])
        assert g.size == 3

    def test_has_leader(self):
        g = SocialGroup(leader_id=1)
        assert g.has_leader
        g2 = SocialGroup()
        assert not g2.has_leader

    def test_cohesion_force_single_member_zero(self):
        g = SocialGroup(member_ids=[0])
        s = _ped(pid=0)
        force = g.cohesion_force(s, {0: s})
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_repulsion_force_no_overlap(self):
        g = SocialGroup(member_ids=[0, 1])
        s0 = _ped(pid=0, x=0.0, y=0.0, radius=0.3)
        s1 = _ped(pid=1, x=10.0, y=0.0, radius=0.3)
        force = g.repulsion_force(s0, {0: s0, 1: s1})
        np.testing.assert_allclose(force, [0.0, 0.0])

    def test_consensus_velocity_with_leader(self):
        g = SocialGroup(member_ids=[0, 1], leader_id=0)
        s0 = _ped(pid=0, vx=2.0, vy=0.0)
        s1 = _ped(pid=1, vx=0.0, vy=0.0)
        cv = g.consensus_velocity({0: s0, 1: s1})
        # Leader has weight 2, follower weight 1 -> weighted mean
        assert cv[0] == pytest.approx(2.0 * (2.0 / 3.0))

    def test_spread_single_member(self):
        g = SocialGroup(member_ids=[0])
        assert g.spread({0: _ped(pid=0)}) == 0.0

    def test_should_split(self):
        g = SocialGroup(member_ids=[0, 1], max_spread=2.0)
        s0 = _ped(pid=0, x=0.0, y=0.0)
        s1 = _ped(pid=1, x=10.0, y=0.0)
        assert g.should_split({0: s0, 1: s1})

    def test_split(self):
        g = SocialGroup(member_ids=[0, 1, 2])
        s0 = _ped(pid=0, x=0.0, y=0.0)
        s1 = _ped(pid=1, x=1.0, y=0.0)
        s2 = _ped(pid=2, x=100.0, y=0.0)  # furthest
        new = g.split({0: s0, 1: s1, 2: s2}, next_group_id=99)
        assert new is not None
        assert 2 in new.member_ids
        assert 2 not in g.member_ids

    def test_merge(self):
        ga = SocialGroup(member_ids=[0, 1])
        gb = SocialGroup(member_ids=[2, 3])
        ga.merge(gb)
        assert ga.size == 4
        assert gb.size == 0

    def test_decide_at_intersection_no_options(self):
        g = SocialGroup(member_ids=[0])
        rng = np.random.default_rng(42)
        result = g.decide_at_intersection({0: _ped(pid=0)}, [], rng)
        np.testing.assert_allclose(result, [0.0, 0.0])


class TestGroupManager:
    def test_create_and_get(self):
        gm = GroupManager()
        g = gm.create_group([1, 2, 3])
        assert gm.get_group(g.group_id) is g

    def test_group_of(self):
        gm = GroupManager()
        gm.create_group([1, 2])
        assert gm.group_of(1) is not None
        assert gm.group_of(99) is None

    def test_remove_group(self):
        gm = GroupManager()
        g = gm.create_group([1])
        gm.remove_group(g.group_id)
        assert gm.get_group(g.group_id) is None

    def test_step_triggers_split(self):
        gm = GroupManager(split_spread=2.0)
        g = gm.create_group([0, 1])
        s0 = _ped(pid=0, x=0.0, y=0.0)
        s1 = _ped(pid=1, x=50.0, y=0.0)
        events = gm.step({0: s0, 1: s1})
        split_events = [e for e in events if e["type"] == "split"]
        assert len(split_events) == 1

    def test_step_triggers_merge(self):
        gm = GroupManager(merge_distance=5.0)
        gm.create_group([0])
        gm.create_group([1])
        s0 = _ped(pid=0, x=0.0, y=0.0)
        s1 = _ped(pid=1, x=1.0, y=0.0)
        events = gm.step({0: s0, 1: s1})
        merge_events = [e for e in events if e["type"] == "merge"]
        assert len(merge_events) == 1
        # After merge only one group should remain
        assert len(gm.groups) == 1
