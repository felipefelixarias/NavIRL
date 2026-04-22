"""Coverage tests for navirl/humans/crowd_generator.py.

Targets the uncovered paths in GoalAssigner flow-pattern dispatch, fallback
branches for sparsely-configured generators, and the evacuation /
event_gathering scenario factories.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.humans.crowd_generator import (
    CrowdGenerator,
    FlowPattern,
    GoalAssigner,
    SpawnEvent,
    SpawnRegion,
    SpawnStrategy,
)

# ===========================================================================
# GoalAssigner flow patterns
# ===========================================================================


class TestGoalAssignerFlows:
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_random_flow_prefers_goal_regions(self):
        goal = SpawnRegion(20.0, 21.0, 20.0, 21.0)
        ga = GoalAssigner(goal_regions=[goal], flow_pattern=FlowPattern.RANDOM)
        g = ga.assign(np.array([0.0, 0.0]), 0, self._rng())
        assert goal.contains(g)

    def test_random_flow_uses_arena_when_no_goal_regions(self):
        arena = SpawnRegion(-5.0, 5.0, -5.0, 5.0)
        ga = GoalAssigner(goal_regions=None, flow_pattern=FlowPattern.RANDOM)
        g = ga.assign(np.array([0.0, 0.0]), 0, self._rng(), arena_bounds=arena)
        assert arena.contains(g)

    def test_random_flow_fallback_when_no_regions_and_no_arena(self):
        ga = GoalAssigner(flow_pattern=FlowPattern.RANDOM)
        spawn = np.array([1.0, 2.0])
        g = ga.assign(spawn, 0, self._rng())
        # Fallback uses spawn + random offset at distance in [5, 15]
        dist = float(np.linalg.norm(g - spawn))
        assert 4.999 <= dist <= 15.001

    def test_unidirectional_goal_left_side(self):
        arena = SpawnRegion(-10.0, 10.0, -5.0, 5.0)
        ga = GoalAssigner(flow_pattern=FlowPattern.UNIDIRECTIONAL)
        g = ga.assign(np.array([-8.0, 0.0]), 0, self._rng(), arena_bounds=arena)
        # spawn_x=-8 is below midx=0, so goal should be on right half
        assert g[0] >= 0.0
        assert arena.y_min <= g[1] <= arena.y_max

    def test_unidirectional_goal_right_side(self):
        arena = SpawnRegion(-10.0, 10.0, -5.0, 5.0)
        ga = GoalAssigner(flow_pattern=FlowPattern.UNIDIRECTIONAL)
        g = ga.assign(np.array([8.0, 0.0]), 0, self._rng(), arena_bounds=arena)
        # spawn_x=8 is above midx=0, so goal should be on left half
        assert g[0] <= 0.0

    def test_unidirectional_goal_no_arena_fallback(self):
        ga = GoalAssigner(flow_pattern=FlowPattern.UNIDIRECTIONAL)
        spawn = np.array([1.0, 1.0])
        g = ga.assign(spawn, 0, self._rng())
        # Fallback: spawn + [20, 0]
        np.testing.assert_allclose(g, spawn + np.array([20.0, 0.0]))

    def test_bidirectional_pairs_opposite_regions(self):
        left = SpawnRegion(-10.0, -9.0, -1.0, 1.0)
        right = SpawnRegion(9.0, 10.0, -1.0, 1.0)
        ga = GoalAssigner(goal_regions=[left, right], flow_pattern=FlowPattern.BIDIRECTIONAL)
        # spawning in region 0 (left) should route goal to region 1 (right)
        g = ga.assign(np.array([-9.5, 0.0]), 0, self._rng())
        assert right.contains(g)
        # spawning in region 1 (right) should route goal to region 0 (left)
        g2 = ga.assign(np.array([9.5, 0.0]), 1, self._rng())
        assert left.contains(g2)

    def test_bidirectional_fallback_when_fewer_than_two_regions(self):
        single = SpawnRegion(-10.0, -9.0, -1.0, 1.0)
        ga = GoalAssigner(goal_regions=[single], flow_pattern=FlowPattern.BIDIRECTIONAL)
        spawn = np.array([-9.5, 0.0])
        g = ga.assign(spawn, 0, self._rng())
        # Fallback: spawn + [15, 0]
        np.testing.assert_allclose(g, spawn + np.array([15.0, 0.0]))

    def test_crossing_routes_to_opposite_index(self):
        # With 4 regions, crossing routes idx -> (idx + 2) % 4
        r0 = SpawnRegion(-10.0, -9.0, -1.0, 1.0)
        r1 = SpawnRegion(-1.0, 1.0, 9.0, 10.0)
        r2 = SpawnRegion(9.0, 10.0, -1.0, 1.0)
        r3 = SpawnRegion(-1.0, 1.0, -10.0, -9.0)
        ga = GoalAssigner(goal_regions=[r0, r1, r2, r3], flow_pattern=FlowPattern.CROSSING)
        g = ga.assign(np.array([-9.5, 0.0]), 0, self._rng())
        assert r2.contains(g)

    def test_crossing_fallback_when_fewer_than_two_regions(self):
        single = SpawnRegion(-10.0, -9.0, -1.0, 1.0)
        ga = GoalAssigner(goal_regions=[single], flow_pattern=FlowPattern.CROSSING)
        spawn = np.array([0.0, 0.0])
        g = ga.assign(spawn, 0, self._rng())
        # Fallback: spawn + 10 * unit vector on circle
        dist = float(np.linalg.norm(g - spawn))
        assert dist == pytest.approx(10.0, abs=1e-6)

    def test_radial_in_converges_on_centre(self):
        arena = SpawnRegion(-10.0, 10.0, -10.0, 10.0)
        ga = GoalAssigner(flow_pattern=FlowPattern.RADIAL_IN)
        g = ga.assign(np.array([8.0, 8.0]), 0, self._rng(), arena_bounds=arena)
        # Centre is origin; goal should be near it (normal noise with std 0.5)
        assert float(np.linalg.norm(g)) < 5.0

    def test_radial_in_without_arena_uses_origin(self):
        ga = GoalAssigner(flow_pattern=FlowPattern.RADIAL_IN)
        g = ga.assign(np.array([5.0, 5.0]), 0, self._rng())
        # Centre defaults to [0, 0]
        assert float(np.linalg.norm(g)) < 5.0

    def test_radial_out_moves_outward(self):
        arena = SpawnRegion(-5.0, 5.0, -5.0, 5.0)
        ga = GoalAssigner(flow_pattern=FlowPattern.RADIAL_OUT)
        g = ga.assign(np.array([0.0, 0.0]), 0, self._rng(), arena_bounds=arena)
        # Distance is uniform(15, 20)
        dist = float(np.linalg.norm(g))
        assert 14.999 <= dist <= 20.001


# ===========================================================================
# CrowdGenerator branch coverage
# ===========================================================================


class TestCrowdGeneratorEdges:
    def test_generate_batch_returns_empty_when_max_reached(self):
        gen = CrowdGenerator(max_pedestrians=3, rng_seed=1)
        first = gen.generate_batch(count=3)
        assert len(first) == 3
        # Second call hits the "count <= 0" early return.
        second = gen.generate_batch(count=5)
        assert second == []

    def test_generate_poisson_returns_empty_when_max_reached(self):
        gen = CrowdGenerator(
            spawn_strategy=SpawnStrategy.POISSON,
            poisson_rate=5.0,
            max_pedestrians=2,
            rng_seed=7,
        )
        gen.generate_batch(count=2)  # fill to max
        out = gen.generate_poisson(dt=1.0)
        assert out == []

    def test_step_unknown_strategy_returns_empty(self):
        gen = CrowdGenerator(rng_seed=2)
        # Corrupt the strategy to force the final fallback branch.
        gen.spawn_strategy = "not-a-real-strategy"
        assert gen.step(0.0, 0.1) == []

    def test_step_poisson_empty_before_arrivals(self):
        # Poisson rate zero => no arrivals even at long dt.
        gen = CrowdGenerator(
            spawn_strategy=SpawnStrategy.POISSON,
            poisson_rate=0.0,
            max_pedestrians=10,
            rng_seed=3,
        )
        assert gen.step(0.0, 10.0) == []

    def test_step_scheduled_multiple_events_same_time(self):
        schedule = [
            SpawnEvent(time_s=0.0, count=2, region_idx=0),
            SpawnEvent(time_s=0.0, count=1, region_idx=0),
        ]
        gen = CrowdGenerator(
            spawn_strategy=SpawnStrategy.SCHEDULED,
            schedule=schedule,
            rng_seed=4,
        )
        out = gen.step(0.0, 0.1)
        assert len(out) == 3


# ===========================================================================
# Scenario factory coverage
# ===========================================================================


class TestScenarioFactories:
    def test_evacuation_scenario_respects_n_pedestrians(self):
        gen = CrowdGenerator.evacuation_scenario(
            room_size=10.0,
            exit_width=1.5,
            n_pedestrians=12,
            rng_seed=5,
        )
        peds = gen.generate_batch()
        assert len(peds) == 12
        # Demographics default was widened for evacuation (max 2.5 m/s).
        speeds = [p.preferred_speed for p in peds]
        assert max(speeds) <= 2.5
        # Assigner should use the single exit as the goal region.
        assert gen.goal_assigner.flow_pattern == FlowPattern.UNIDIRECTIONAL
        assert len(gen.goal_assigner.goal_regions) == 1

    def test_event_gathering_scenario_uses_poisson_and_four_regions(self):
        gen = CrowdGenerator.event_gathering_scenario(
            arena_radius=12.0,
            stage_pos=(0.0, 0.0),
            poisson_rate=5.0,
            max_pedestrians=20,
            rng_seed=6,
        )
        assert gen.spawn_strategy == SpawnStrategy.POISSON
        assert len(gen.spawn_regions) == 4
        assert gen.goal_assigner.flow_pattern == FlowPattern.RADIAL_IN
        # Under Poisson arrival, several peds should come in over a long dt.
        peds = gen.step(0.0, 2.0)
        assert all(p.pid >= 0 for p in peds)

    def test_event_gathering_stage_offset_positions_goal(self):
        gen = CrowdGenerator.event_gathering_scenario(
            arena_radius=15.0,
            stage_pos=(5.0, -3.0),
            poisson_rate=1.0,
            max_pedestrians=5,
            rng_seed=8,
        )
        goal_region = gen.goal_assigner.goal_regions[0]
        # Goal region is a 6m box centered on stage_pos.
        assert goal_region.x_min == pytest.approx(2.0)
        assert goal_region.x_max == pytest.approx(8.0)
        assert goal_region.y_min == pytest.approx(-6.0)
        assert goal_region.y_max == pytest.approx(0.0)
