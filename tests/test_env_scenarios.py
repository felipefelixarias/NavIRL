"""Tests for navirl/envs/scenarios.py — scenario generators for RL navigation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.envs.scenarios import (
    BaseScenario,
    CircleCrossing,
    CorridorPassing,
    DenseRoom,
    DoorwayNegotiation,
    GroupNavigation,
    IntersectionCrossing,
    OpenField,
    ProceduralScenarioGenerator,
    RandomGoal,
    ScenarioDifficultyScaler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
REQUIRED_KEYS = {
    "map_name",
    "robot_start",
    "robot_goal",
    "human_starts",
    "human_goals",
    "num_humans",
}


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _assert_valid_config(cfg: dict, expected_humans: int | None = None) -> None:
    """Check that a scenario config has the required structure."""
    missing = REQUIRED_KEYS - cfg.keys()
    assert not missing, f"Missing keys: {missing}"

    assert isinstance(cfg["robot_start"], tuple)
    assert len(cfg["robot_start"]) == 2
    assert isinstance(cfg["robot_goal"], tuple)
    assert len(cfg["robot_goal"]) == 2

    assert isinstance(cfg["human_starts"], list)
    assert isinstance(cfg["human_goals"], list)
    assert len(cfg["human_starts"]) == len(cfg["human_goals"])

    if expected_humans is not None:
        assert cfg["num_humans"] == expected_humans
        assert len(cfg["human_starts"]) == expected_humans


# ---------------------------------------------------------------------------
# BaseScenario static helpers
# ---------------------------------------------------------------------------


class TestBaseScenarioHelpers:
    def test_uniform_position_in_range(self):
        pos = BaseScenario._uniform_position(_rng(), -5.0, 5.0)
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert -5.0 <= pos[0] <= 5.0
        assert -5.0 <= pos[1] <= 5.0

    def test_positions_on_circle_count(self):
        pts = BaseScenario._positions_on_circle(6, radius=3.0)
        assert len(pts) == 6

    def test_positions_on_circle_radius(self):
        pts = BaseScenario._positions_on_circle(8, radius=5.0)
        for x, y in pts:
            assert math.isclose(math.hypot(x, y), 5.0, rel_tol=1e-9)

    def test_positions_on_circle_with_center(self):
        center = (2.0, 3.0)
        pts = BaseScenario._positions_on_circle(4, radius=1.0, center=center)
        for x, y in pts:
            dist = math.hypot(x - center[0], y - center[1])
            assert math.isclose(dist, 1.0, rel_tol=1e-9)

    def test_positions_on_circle_with_offset(self):
        pts_no_offset = BaseScenario._positions_on_circle(4, radius=2.0, offset_angle=0.0)
        pts_offset = BaseScenario._positions_on_circle(4, radius=2.0, offset_angle=math.pi / 4)
        # First point should differ
        assert not math.isclose(pts_no_offset[0][0], pts_offset[0][0])

    def test_antipodal(self):
        positions = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        anti = BaseScenario._antipodal(positions)
        assert anti == [(-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]

    def test_antipodal_with_center(self):
        positions = [(3.0, 0.0)]
        anti = BaseScenario._antipodal(positions, center=(1.0, 0.0))
        assert anti == [(-1.0, 0.0)]

    def test_antipodal_empty(self):
        assert BaseScenario._antipodal([]) == []


# ---------------------------------------------------------------------------
# CircleCrossing
# ---------------------------------------------------------------------------


class TestCircleCrossing:
    def test_generate_default(self):
        cfg = CircleCrossing().generate(_rng())
        _assert_valid_config(cfg, expected_humans=5)
        assert cfg["map_name"] == "circle_crossing"

    def test_custom_params(self):
        sc = CircleCrossing(num_humans=10, circle_radius=8.0)
        cfg = sc.generate(_rng())
        _assert_valid_config(cfg, expected_humans=10)

    def test_positions_on_circle(self):
        sc = CircleCrossing(num_humans=3, circle_radius=5.0)
        cfg = sc.generate(_rng())
        # Robot + humans should all be ~5.0 from origin
        all_starts = [cfg["robot_start"]] + cfg["human_starts"]
        for x, y in all_starts:
            dist = math.hypot(x, y)
            assert math.isclose(dist, 5.0, rel_tol=1e-9)

    def test_goals_are_antipodal(self):
        sc = CircleCrossing(num_humans=2, circle_radius=4.0)
        cfg = sc.generate(_rng())
        all_starts = [cfg["robot_start"]] + cfg["human_starts"]
        all_goals = [cfg["robot_goal"]] + cfg["human_goals"]
        for (sx, sy), (gx, gy) in zip(all_starts, all_goals, strict=True):
            assert math.isclose(sx + gx, 0.0, abs_tol=1e-9)
            assert math.isclose(sy + gy, 0.0, abs_tol=1e-9)

    def test_reproducibility(self):
        sc = CircleCrossing()
        cfg1 = sc.generate(_rng(99))
        cfg2 = sc.generate(_rng(99))
        assert cfg1 == cfg2


# ---------------------------------------------------------------------------
# RandomGoal
# ---------------------------------------------------------------------------


class TestRandomGoal:
    def test_generate_default(self):
        cfg = RandomGoal().generate(_rng())
        _assert_valid_config(cfg, expected_humans=5)
        assert cfg["map_name"] == "random_goal"

    def test_positions_within_world(self):
        sc = RandomGoal(world_size=4.0)
        cfg = sc.generate(_rng())
        for pos in (
            [cfg["robot_start"], cfg["robot_goal"]] + cfg["human_starts"] + cfg["human_goals"]
        ):
            assert -4.0 <= pos[0] <= 4.0
            assert -4.0 <= pos[1] <= 4.0

    def test_min_goal_distance_respected(self):
        sc = RandomGoal(num_humans=3, min_goal_dist=2.0)
        cfg = sc.generate(_rng())
        # Robot
        assert math.dist(cfg["robot_start"], cfg["robot_goal"]) >= 2.0
        # Humans
        for hs, hg in zip(cfg["human_starts"], cfg["human_goals"], strict=True):
            assert math.dist(hs, hg) >= 2.0

    def test_custom_params(self):
        sc = RandomGoal(num_humans=12, world_size=10.0, min_goal_dist=5.0)
        cfg = sc.generate(_rng())
        _assert_valid_config(cfg, expected_humans=12)


# ---------------------------------------------------------------------------
# CorridorPassing
# ---------------------------------------------------------------------------


class TestCorridorPassing:
    def test_generate_default(self):
        cfg = CorridorPassing().generate(_rng())
        _assert_valid_config(cfg, expected_humans=6)
        assert cfg["map_name"] == "corridor"

    def test_robot_endpoints(self):
        sc = CorridorPassing(corridor_length=12.0)
        cfg = sc.generate(_rng())
        assert cfg["robot_start"] == (-6.0, 0.0)
        assert cfg["robot_goal"] == (6.0, 0.0)

    def test_human_positions_in_corridor(self):
        sc = CorridorPassing(corridor_length=10.0, corridor_width=2.0, num_humans=8)
        cfg = sc.generate(_rng())
        for hs in cfg["human_starts"]:
            assert -5.0 <= hs[0] <= 5.0
            assert -1.0 <= hs[1] <= 1.0

    def test_bidirectional_traffic(self):
        sc = CorridorPassing(corridor_length=10.0, num_humans=4)
        cfg = sc.generate(_rng())
        # Even-index humans go left-to-right, odd go right-to-left
        for i, (hs, hg) in enumerate(zip(cfg["human_starts"], cfg["human_goals"], strict=True)):
            if i % 2 == 0:
                assert hs[0] <= 0  # starts left
                assert hg[0] == 5.0  # goal right
            else:
                assert hs[0] >= 0  # starts right
                assert hg[0] == -5.0  # goal left

    def test_extra_keys(self):
        cfg = CorridorPassing(corridor_length=8.0, corridor_width=3.0).generate(_rng())
        assert cfg["corridor_length"] == 8.0
        assert cfg["corridor_width"] == 3.0


# ---------------------------------------------------------------------------
# DoorwayNegotiation
# ---------------------------------------------------------------------------


class TestDoorwayNegotiation:
    def test_generate_default(self):
        cfg = DoorwayNegotiation().generate(_rng())
        _assert_valid_config(cfg, expected_humans=3)
        assert cfg["map_name"] == "doorway"

    def test_robot_positions(self):
        sc = DoorwayNegotiation(room_depth=7.0)
        cfg = sc.generate(_rng())
        assert cfg["robot_start"] == (-7.0, 0.0)
        assert cfg["robot_goal"] == (7.0, 0.0)

    def test_humans_on_opposite_side(self):
        sc = DoorwayNegotiation(room_depth=5.0, num_humans=4)
        cfg = sc.generate(_rng())
        for hs in cfg["human_starts"]:
            assert hs[0] >= 0.5  # humans start on positive-x side

    def test_door_width_key(self):
        cfg = DoorwayNegotiation(door_width=1.5).generate(_rng())
        assert cfg["door_width"] == 1.5

    def test_human_y_within_door(self):
        sc = DoorwayNegotiation(door_width=2.0, num_humans=5)
        cfg = sc.generate(_rng())
        for hs in cfg["human_starts"]:
            assert -1.0 <= hs[1] <= 1.0


# ---------------------------------------------------------------------------
# IntersectionCrossing
# ---------------------------------------------------------------------------


class TestIntersectionCrossing:
    def test_generate_default(self):
        cfg = IntersectionCrossing().generate(_rng())
        _assert_valid_config(cfg, expected_humans=8)  # 2 per direction * 4
        assert cfg["map_name"] == "intersection"

    def test_robot_crosses_south_to_north(self):
        sc = IntersectionCrossing(approach_distance=10.0)
        cfg = sc.generate(_rng())
        assert cfg["robot_start"] == (0.0, -10.0)
        assert cfg["robot_goal"] == (0.0, 10.0)

    def test_num_humans_matches(self):
        sc = IntersectionCrossing(num_humans_per_direction=3)
        cfg = sc.generate(_rng())
        assert cfg["num_humans"] == 12
        assert len(cfg["human_starts"]) == 12

    def test_custom_size(self):
        sc = IntersectionCrossing(intersection_size=5.0, approach_distance=12.0)
        cfg = sc.generate(_rng())
        _assert_valid_config(cfg)


# ---------------------------------------------------------------------------
# GroupNavigation
# ---------------------------------------------------------------------------


class TestGroupNavigation:
    def test_generate_default(self):
        cfg = GroupNavigation().generate(_rng())
        _assert_valid_config(cfg, expected_humans=9)  # 3 groups * 3
        assert cfg["map_name"] == "group_navigation"

    def test_num_humans(self):
        sc = GroupNavigation(num_groups=4, group_size=5)
        cfg = sc.generate(_rng())
        assert cfg["num_humans"] == 20
        assert len(cfg["human_starts"]) == 20

    def test_group_members_clustered(self):
        sc = GroupNavigation(num_groups=2, group_size=4, group_spread=0.5, world_size=10.0)
        cfg = sc.generate(_rng())
        # Members within same group should be within 2*spread of each other
        for g in range(2):
            group_starts = cfg["human_starts"][g * 4 : (g + 1) * 4]
            xs = [p[0] for p in group_starts]
            ys = [p[1] for p in group_starts]
            assert max(xs) - min(xs) <= 1.0  # 2 * 0.5
            assert max(ys) - min(ys) <= 1.0

    def test_extra_keys(self):
        cfg = GroupNavigation(num_groups=2, group_size=3).generate(_rng())
        assert cfg["num_groups"] == 2
        assert cfg["group_size"] == 3


# ---------------------------------------------------------------------------
# DenseRoom
# ---------------------------------------------------------------------------


class TestDenseRoom:
    def test_generate_default(self):
        cfg = DenseRoom().generate(_rng())
        _assert_valid_config(cfg, expected_humans=20)
        assert cfg["map_name"] == "dense_room"

    def test_robot_starts_in_corner(self):
        sc = DenseRoom(room_size=5.0)
        cfg = sc.generate(_rng())
        assert cfg["robot_start"] == (-4.5, -4.5)
        assert cfg["robot_goal"] == (4.5, 4.5)

    def test_humans_within_room(self):
        sc = DenseRoom(room_size=3.0, num_humans=10)
        cfg = sc.generate(_rng())
        for pos in cfg["human_starts"] + cfg["human_goals"]:
            assert -3.0 <= pos[0] <= 3.0
            assert -3.0 <= pos[1] <= 3.0

    def test_room_size_key(self):
        cfg = DenseRoom(room_size=7.0).generate(_rng())
        assert cfg["room_size"] == 7.0


# ---------------------------------------------------------------------------
# OpenField
# ---------------------------------------------------------------------------


class TestOpenField:
    def test_generate_default(self):
        cfg = OpenField().generate(_rng())
        _assert_valid_config(cfg, expected_humans=8)
        assert cfg["map_name"] == "open_field"

    def test_positions_within_field(self):
        sc = OpenField(field_size=10.0, num_humans=5)
        cfg = sc.generate(_rng())
        for pos in (
            [cfg["robot_start"], cfg["robot_goal"]] + cfg["human_starts"] + cfg["human_goals"]
        ):
            assert -10.0 <= pos[0] <= 10.0
            assert -10.0 <= pos[1] <= 10.0

    def test_field_size_key(self):
        cfg = OpenField(field_size=20.0).generate(_rng())
        assert cfg["field_size"] == 20.0


# ---------------------------------------------------------------------------
# ScenarioDifficultyScaler
# ---------------------------------------------------------------------------


class TestScenarioDifficultyScaler:
    def test_difficulty_zero_no_scaling(self):
        base = CircleCrossing(num_humans=5)
        scaler = ScenarioDifficultyScaler(base_scenario=base, difficulty=0.0)
        cfg = scaler.generate(_rng())
        assert cfg["num_humans"] == 5
        assert cfg["difficulty"] == 0.0

    def test_difficulty_one_max_scaling(self):
        base = CircleCrossing(num_humans=5)
        scaler = ScenarioDifficultyScaler(
            base_scenario=base, difficulty=1.0, max_humans_multiplier=3.0
        )
        cfg = scaler.generate(_rng())
        assert cfg["num_humans"] == 15  # 5 * 3.0
        assert cfg["difficulty"] == 1.0

    def test_difficulty_half(self):
        base = CircleCrossing(num_humans=10)
        scaler = ScenarioDifficultyScaler(
            base_scenario=base, difficulty=0.5, max_humans_multiplier=3.0
        )
        cfg = scaler.generate(_rng())
        # 10 * (1 + 0.5*(3-1)) = 10 * 2 = 20
        assert cfg["num_humans"] == 20

    def test_corridor_width_narrows(self):
        base = CorridorPassing(corridor_width=4.0, num_humans=2)
        scaler = ScenarioDifficultyScaler(base_scenario=base, difficulty=1.0)
        cfg = scaler.generate(_rng())
        # corridor_width * (1 - 0.5*1.0) = 4.0 * 0.5 = 2.0
        assert cfg["corridor_width"] == pytest.approx(2.0)

    def test_door_width_narrows(self):
        base = DoorwayNegotiation(door_width=2.0, num_humans=1)
        scaler = ScenarioDifficultyScaler(base_scenario=base, difficulty=0.5)
        cfg = scaler.generate(_rng())
        # door_width * (1 - 0.5*0.5) = 2.0 * 0.75 = 1.5
        assert cfg["door_width"] == pytest.approx(1.5)

    def test_difficulty_clipped_above_one(self):
        base = CircleCrossing(num_humans=4)
        scaler = ScenarioDifficultyScaler(base_scenario=base, difficulty=2.0)
        cfg = scaler.generate(_rng())
        assert cfg["difficulty"] == 1.0

    def test_difficulty_clipped_below_zero(self):
        base = CircleCrossing(num_humans=4)
        scaler = ScenarioDifficultyScaler(base_scenario=base, difficulty=-1.0)
        cfg = scaler.generate(_rng())
        assert cfg["difficulty"] == 0.0

    def test_config_has_required_keys(self):
        base = DenseRoom(num_humans=5)
        scaler = ScenarioDifficultyScaler(base_scenario=base, difficulty=0.7)
        cfg = scaler.generate(_rng())
        _assert_valid_config(cfg)


# ---------------------------------------------------------------------------
# ProceduralScenarioGenerator
# ---------------------------------------------------------------------------


class TestProceduralScenarioGenerator:
    def test_generate_default_pool(self):
        gen = ProceduralScenarioGenerator()
        cfg = gen.generate(_rng())
        _assert_valid_config(cfg)
        assert "scenario_index" in cfg
        assert "scenario_class" in cfg

    def test_single_scenario_pool(self):
        gen = ProceduralScenarioGenerator(scenario_pool=[DenseRoom(num_humans=3)])
        cfg = gen.generate(_rng())
        assert cfg["map_name"] == "dense_room"
        assert cfg["scenario_index"] == 0
        assert cfg["scenario_class"] == "DenseRoom"

    def test_weighted_selection(self):
        pool = [CircleCrossing(), DenseRoom()]
        # Weight heavily toward DenseRoom
        gen = ProceduralScenarioGenerator(scenario_pool=pool, weights=[0.0, 1.0])
        cfg = gen.generate(_rng())
        assert cfg["map_name"] == "dense_room"

    def test_with_difficulty_range(self):
        gen = ProceduralScenarioGenerator(
            scenario_pool=[CircleCrossing(num_humans=5)],
            difficulty_range=(0.2, 0.8),
        )
        cfg = gen.generate(_rng())
        _assert_valid_config(cfg)
        assert "difficulty" in cfg
        assert 0.2 <= cfg["difficulty"] <= 0.8
        assert cfg["scenario_class"] == "ScenarioDifficultyScaler"

    def test_reproducibility(self):
        gen = ProceduralScenarioGenerator()
        cfg1 = gen.generate(_rng(123))
        cfg2 = gen.generate(_rng(123))
        assert cfg1 == cfg2

    def test_different_seeds_different_results(self):
        gen = ProceduralScenarioGenerator()
        cfg1 = gen.generate(_rng(1))
        cfg2 = gen.generate(_rng(2))
        # Very unlikely to be identical
        assert (
            cfg1["robot_start"] != cfg2["robot_start"]
            or cfg1["scenario_index"] != cfg2["scenario_index"]
        )
