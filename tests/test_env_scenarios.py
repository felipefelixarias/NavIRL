"""Tests for navirl/envs/scenarios.py — scenario generators."""

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


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestBaseHelpers:
    def test_uniform_position(self, rng):
        pos = BaseScenario._uniform_position(rng, -5.0, 5.0)
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert -5.0 <= pos[0] <= 5.0
        assert -5.0 <= pos[1] <= 5.0

    def test_positions_on_circle(self):
        positions = BaseScenario._positions_on_circle(4, radius=1.0)
        assert len(positions) == 4
        for x, y in positions:
            dist = math.sqrt(x**2 + y**2)
            assert dist == pytest.approx(1.0, abs=1e-10)

    def test_positions_on_circle_with_center(self):
        center = (3.0, 4.0)
        positions = BaseScenario._positions_on_circle(3, radius=2.0, center=center)
        for x, y in positions:
            dist = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            assert dist == pytest.approx(2.0, abs=1e-10)

    def test_antipodal(self):
        positions = [(1.0, 0.0), (0.0, 1.0)]
        anti = BaseScenario._antipodal(positions)
        assert anti == [(-1.0, 0.0), (0.0, -1.0)]

    def test_antipodal_custom_center(self):
        positions = [(3.0, 0.0)]
        anti = BaseScenario._antipodal(positions, center=(1.0, 0.0))
        assert anti == [(-1.0, 0.0)]


# ---------------------------------------------------------------------------
# Scenario config structure validation helper
# ---------------------------------------------------------------------------


def _check_common_keys(cfg: dict, expected_humans: int):
    """Verify all standard scenario config keys are present and valid."""
    assert "map_name" in cfg
    assert "robot_start" in cfg
    assert "robot_goal" in cfg
    assert "human_starts" in cfg
    assert "human_goals" in cfg
    assert "num_humans" in cfg
    assert cfg["num_humans"] == expected_humans
    assert len(cfg["human_starts"]) == expected_humans
    assert len(cfg["human_goals"]) == expected_humans
    assert len(cfg["robot_start"]) == 2
    assert len(cfg["robot_goal"]) == 2


# ---------------------------------------------------------------------------
# CircleCrossing
# ---------------------------------------------------------------------------


class TestCircleCrossing:
    def test_basic(self, rng):
        s = CircleCrossing(num_humans=5, circle_radius=4.0)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 5)
        assert cfg["map_name"] == "circle_crossing"

    def test_starts_on_circle(self, rng):
        radius = 3.0
        s = CircleCrossing(num_humans=3, circle_radius=radius)
        cfg = s.generate(rng)
        # Robot + all humans should be on the circle
        all_starts = [cfg["robot_start"]] + cfg["human_starts"]
        for x, y in all_starts:
            dist = math.sqrt(x**2 + y**2)
            assert dist == pytest.approx(radius, abs=1e-10)

    def test_goals_are_antipodal(self, rng):
        s = CircleCrossing(num_humans=2, circle_radius=5.0)
        cfg = s.generate(rng)
        all_starts = [cfg["robot_start"]] + cfg["human_starts"]
        all_goals = [cfg["robot_goal"]] + cfg["human_goals"]
        for (sx, sy), (gx, gy) in zip(all_starts, all_goals, strict=True):
            assert gx == pytest.approx(-sx, abs=1e-10)
            assert gy == pytest.approx(-sy, abs=1e-10)

    def test_deterministic_with_seed(self):
        s = CircleCrossing(num_humans=4)
        cfg1 = s.generate(np.random.default_rng(99))
        cfg2 = s.generate(np.random.default_rng(99))
        assert cfg1 == cfg2


# ---------------------------------------------------------------------------
# RandomGoal
# ---------------------------------------------------------------------------


class TestRandomGoal:
    def test_basic(self, rng):
        s = RandomGoal(num_humans=3, world_size=6.0, min_goal_dist=2.0)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 3)
        assert cfg["map_name"] == "random_goal"

    def test_min_distance_constraint(self, rng):
        min_dist = 3.0
        s = RandomGoal(num_humans=2, min_goal_dist=min_dist)
        cfg = s.generate(rng)
        # Robot start-goal distance
        d = math.dist(cfg["robot_start"], cfg["robot_goal"])
        assert d >= min_dist
        # Human start-goal distances
        for hs, hg in zip(cfg["human_starts"], cfg["human_goals"], strict=True):
            d = math.dist(hs, hg)
            assert d >= min_dist

    def test_positions_within_world(self, rng):
        ws = 4.0
        s = RandomGoal(num_humans=5, world_size=ws)
        cfg = s.generate(rng)
        for x, y in [cfg["robot_start"], cfg["robot_goal"]] + cfg["human_starts"] + cfg["human_goals"]:
            assert -ws <= x <= ws
            assert -ws <= y <= ws


# ---------------------------------------------------------------------------
# CorridorPassing
# ---------------------------------------------------------------------------


class TestCorridorPassing:
    def test_basic(self, rng):
        s = CorridorPassing(num_humans=6, corridor_length=10.0, corridor_width=2.0)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 6)
        assert cfg["map_name"] == "corridor"

    def test_robot_endpoints(self, rng):
        s = CorridorPassing(corridor_length=8.0)
        cfg = s.generate(rng)
        assert cfg["robot_start"] == (-4.0, 0.0)
        assert cfg["robot_goal"] == (4.0, 0.0)

    def test_humans_within_corridor(self, rng):
        length, width = 10.0, 2.0
        s = CorridorPassing(num_humans=10, corridor_length=length, corridor_width=width)
        cfg = s.generate(rng)
        half_l = length / 2
        half_w = width / 2
        for x, y in cfg["human_starts"]:
            assert -half_l <= x <= half_l
            assert -half_w <= y <= half_w

    def test_extra_keys(self, rng):
        s = CorridorPassing(corridor_length=12.0, corridor_width=3.0)
        cfg = s.generate(rng)
        assert cfg["corridor_length"] == 12.0
        assert cfg["corridor_width"] == 3.0


# ---------------------------------------------------------------------------
# DoorwayNegotiation
# ---------------------------------------------------------------------------


class TestDoorwayNegotiation:
    def test_basic(self, rng):
        s = DoorwayNegotiation(num_humans=3)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 3)
        assert cfg["map_name"] == "doorway"

    def test_robot_crosses_door(self, rng):
        depth = 5.0
        s = DoorwayNegotiation(room_depth=depth)
        cfg = s.generate(rng)
        assert cfg["robot_start"] == (-depth, 0.0)
        assert cfg["robot_goal"] == (depth, 0.0)

    def test_humans_start_on_other_side(self, rng):
        s = DoorwayNegotiation(num_humans=4, room_depth=5.0)
        cfg = s.generate(rng)
        for x, _ in cfg["human_starts"]:
            assert x > 0  # Humans start on the far side


# ---------------------------------------------------------------------------
# IntersectionCrossing
# ---------------------------------------------------------------------------


class TestIntersectionCrossing:
    def test_basic(self, rng):
        s = IntersectionCrossing(num_humans_per_direction=2)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 8)  # 2 per direction * 4 directions
        assert cfg["map_name"] == "intersection"

    def test_robot_south_to_north(self, rng):
        dist = 8.0
        s = IntersectionCrossing(approach_distance=dist)
        cfg = s.generate(rng)
        assert cfg["robot_start"] == (0.0, -dist)
        assert cfg["robot_goal"] == (0.0, dist)


# ---------------------------------------------------------------------------
# GroupNavigation
# ---------------------------------------------------------------------------


class TestGroupNavigation:
    def test_basic(self, rng):
        s = GroupNavigation(num_groups=3, group_size=3)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 9)
        assert cfg["map_name"] == "group_navigation"
        assert cfg["num_groups"] == 3
        assert cfg["group_size"] == 3

    def test_group_clustering(self, rng):
        """Members of each group should be close to each other."""
        s = GroupNavigation(num_groups=2, group_size=4, group_spread=0.5, world_size=20.0)
        cfg = s.generate(rng)
        # Group 1 is indices 0-3, group 2 is indices 4-7
        for g in range(2):
            group_starts = cfg["human_starts"][g * 4 : (g + 1) * 4]
            xs = [p[0] for p in group_starts]
            ys = [p[1] for p in group_starts]
            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)
            assert spread_x <= 1.0  # 2 * group_spread
            assert spread_y <= 1.0


# ---------------------------------------------------------------------------
# DenseRoom
# ---------------------------------------------------------------------------


class TestDenseRoom:
    def test_basic(self, rng):
        s = DenseRoom(room_size=5.0, num_humans=20)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 20)
        assert cfg["map_name"] == "dense_room"

    def test_robot_corner_to_corner(self, rng):
        size = 5.0
        s = DenseRoom(room_size=size)
        cfg = s.generate(rng)
        assert cfg["robot_start"] == (-size + 0.5, -size + 0.5)
        assert cfg["robot_goal"] == (size - 0.5, size - 0.5)


# ---------------------------------------------------------------------------
# OpenField
# ---------------------------------------------------------------------------


class TestOpenField:
    def test_basic(self, rng):
        s = OpenField(field_size=15.0, num_humans=8)
        cfg = s.generate(rng)
        _check_common_keys(cfg, 8)
        assert cfg["map_name"] == "open_field"
        assert cfg["field_size"] == 15.0


# ---------------------------------------------------------------------------
# ScenarioDifficultyScaler
# ---------------------------------------------------------------------------


class TestDifficultyScaler:
    def test_difficulty_in_output(self, rng):
        base = CircleCrossing(num_humans=5)
        s = ScenarioDifficultyScaler(base_scenario=base, difficulty=0.7)
        cfg = s.generate(rng)
        assert cfg["difficulty"] == pytest.approx(0.7)

    def test_high_difficulty_increases_humans(self, rng):
        base = CircleCrossing(num_humans=5)
        s = ScenarioDifficultyScaler(
            base_scenario=base, difficulty=1.0, max_humans_multiplier=3.0
        )
        cfg = s.generate(rng)
        # At difficulty=1.0, num_humans should be 5 * 3 = 15
        assert cfg["num_humans"] == 15

    def test_zero_difficulty_unchanged(self, rng):
        base = CircleCrossing(num_humans=5)
        s = ScenarioDifficultyScaler(
            base_scenario=base, difficulty=0.0, max_humans_multiplier=3.0
        )
        cfg = s.generate(rng)
        assert cfg["num_humans"] == 5

    def test_clamps_difficulty(self, rng):
        base = CircleCrossing(num_humans=2)
        s = ScenarioDifficultyScaler(base_scenario=base, difficulty=2.0)
        cfg = s.generate(rng)
        assert cfg["difficulty"] == pytest.approx(1.0)

    def test_narrows_corridor(self, rng):
        base = CorridorPassing(corridor_width=4.0, num_humans=2)
        s = ScenarioDifficultyScaler(base_scenario=base, difficulty=1.0)
        cfg = s.generate(rng)
        # At difficulty=1.0, corridor_width should be 4.0 * (1 - 0.5) = 2.0
        assert cfg["corridor_width"] == pytest.approx(2.0)

    def test_narrows_door(self, rng):
        base = DoorwayNegotiation(door_width=2.0, num_humans=1)
        s = ScenarioDifficultyScaler(base_scenario=base, difficulty=1.0)
        cfg = s.generate(rng)
        assert cfg["door_width"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ProceduralScenarioGenerator
# ---------------------------------------------------------------------------


class TestProceduralGenerator:
    def test_basic(self, rng):
        gen = ProceduralScenarioGenerator()
        cfg = gen.generate(rng)
        assert "map_name" in cfg
        assert "scenario_index" in cfg
        assert "scenario_class" in cfg

    def test_with_weights(self, rng):
        pool = [CircleCrossing(), CorridorPassing()]
        # Weight heavily toward corridor
        gen = ProceduralScenarioGenerator(scenario_pool=pool, weights=[0.0, 1.0])
        cfg = gen.generate(rng)
        assert cfg["map_name"] == "corridor"

    def test_with_difficulty_range(self, rng):
        gen = ProceduralScenarioGenerator(difficulty_range=(0.3, 0.7))
        cfg = gen.generate(rng)
        assert "difficulty" in cfg
        assert 0.3 <= cfg["difficulty"] <= 0.7

    def test_deterministic_with_seed(self):
        gen = ProceduralScenarioGenerator()
        cfg1 = gen.generate(np.random.default_rng(123))
        gen2 = ProceduralScenarioGenerator()
        cfg2 = gen2.generate(np.random.default_rng(123))
        assert cfg1 == cfg2

    def test_single_scenario_pool(self, rng):
        gen = ProceduralScenarioGenerator(scenario_pool=[OpenField()])
        cfg = gen.generate(rng)
        assert cfg["map_name"] == "open_field"
        assert cfg["scenario_index"] == 0
