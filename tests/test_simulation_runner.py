"""Tests for navirl/simulation/runner.py.

Covers SimulationRunner: construction, callbacks, step, run, run_episodes,
run_parallel, run_batch, snapshots, reset, stats, and async run.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from navirl.simulation.clock import SimulationClock
from navirl.simulation.events import EventBus
from navirl.simulation.physics import SimplePhysics
from navirl.simulation.runner import (
    EpisodeResult,
    ProgressInfo,
    SimulationRunner,
)
from navirl.simulation.world import World

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def basic_world():
    """A simple world with two entities."""
    w = World(width=20, height=20)
    w.add_entity(position=(2, 3), velocity=(0.1, 0), kind="robot", radius=0.3)
    w.add_entity(position=(10, 10), velocity=(-0.1, 0), kind="pedestrian", radius=0.3)
    return w


@pytest.fixture
def runner(basic_world):
    """A SimulationRunner configured for short runs."""
    clock = SimulationClock(dt=0.1, max_sim_time=1.0)
    physics = SimplePhysics()
    event_bus = EventBus()
    return SimulationRunner(
        world=basic_world,
        clock=clock,
        physics=physics,
        event_bus=event_bus,
        headless=True,
        seed=42,
    )


# ===================================================================
# EpisodeResult
# ===================================================================


class TestEpisodeResult:
    def test_defaults(self):
        r = EpisodeResult()
        assert r.episode_id == 0
        assert r.success is False
        assert r.sim_time == 0.0
        assert r.wall_time == 0.0
        assert r.steps == 0
        assert r.total_collisions == 0
        assert r.events == []
        assert r.final_positions == {}
        assert r.metadata == {}

    def test_to_dict(self):
        r = EpisodeResult(episode_id=1, success=True, sim_time=5.0, steps=50)
        d = r.to_dict()
        assert d["episode_id"] == 1
        assert d["success"] is True
        assert d["sim_time"] == 5.0
        assert d["steps"] == 50
        assert isinstance(d["events"], list)
        assert isinstance(d["final_positions"], dict)
        assert isinstance(d["metadata"], dict)


# ===================================================================
# ProgressInfo
# ===================================================================


class TestProgressInfo:
    def test_defaults(self):
        p = ProgressInfo(episode=1, total_episodes=5, step=10, sim_time=1.0, wall_time=0.5)
        assert p.done is False
        assert p.message == ""

    def test_custom_fields(self):
        p = ProgressInfo(
            episode=3,
            total_episodes=10,
            step=100,
            sim_time=5.0,
            wall_time=2.0,
            done=True,
            message="completed",
        )
        assert p.done is True
        assert p.message == "completed"


# ===================================================================
# SimulationRunner construction
# ===================================================================


class TestSimulationRunnerInit:
    def test_basic_construction(self, basic_world):
        runner = SimulationRunner(world=basic_world)
        assert runner.world is basic_world
        assert runner.headless is True
        assert not runner.is_running

    def test_custom_components(self, basic_world):
        clock = SimulationClock(dt=0.05)
        physics = SimplePhysics()
        bus = EventBus()
        runner = SimulationRunner(
            world=basic_world,
            clock=clock,
            physics=physics,
            event_bus=bus,
            headless=False,
            seed=123,
        )
        assert runner.clock is clock
        assert runner.physics is physics
        assert runner.event_bus is bus
        assert runner.headless is False

    def test_repr(self, runner):
        text = repr(runner)
        assert "SimulationRunner" in text
        assert "running=False" in text


# ===================================================================
# Callbacks
# ===================================================================


class TestCallbacks:
    def test_pre_step_callback(self, runner):
        calls = []
        runner.on_pre_step(lambda r, dt: calls.append(("pre", dt)))
        runner.step()
        assert len(calls) == 1
        assert calls[0][0] == "pre"
        assert calls[0][1] > 0

    def test_post_step_callback(self, runner):
        calls = []
        runner.on_post_step(lambda r, dt: calls.append(("post", dt)))
        runner.step()
        assert len(calls) == 1

    def test_episode_end_callback(self, runner):
        results = []
        runner.on_episode_end(lambda r, res: results.append(res))
        runner.run(max_steps=5)
        assert len(results) == 1
        assert isinstance(results[0], EpisodeResult)

    def test_progress_callback(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=100.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        progress = []
        runner.on_progress(lambda info: progress.append(info))
        runner.run(max_steps=200, report_interval=50)
        # Should get at least one progress report (200/50 = 4 reports)
        assert len(progress) >= 1
        assert isinstance(progress[0], ProgressInfo)

    def test_termination_condition(self, runner):
        step_count = [0]

        def terminate_at_3(r):
            step_count[0] += 1
            return step_count[0] >= 3

        runner.add_termination_condition(terminate_at_3)
        result = runner.run(max_steps=100)
        assert result.steps == 3
        assert result.success is True  # terminated = True -> success = True


# ===================================================================
# Step
# ===================================================================


class TestStep:
    def test_single_step(self, runner):
        dt = runner.step()
        assert dt > 0
        assert runner.clock.sim_time > 0

    def test_multiple_steps(self, runner):
        for _ in range(10):
            runner.step()
        assert runner.clock.sim_time == pytest.approx(1.0)  # 10 * 0.1


# ===================================================================
# Run
# ===================================================================


class TestRun:
    def test_run_with_max_steps(self, runner):
        result = runner.run(max_steps=5)
        assert result.steps == 5
        assert result.sim_time > 0
        assert result.wall_time > 0
        assert not runner.is_running

    def test_run_with_max_time(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=100.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        result = runner.run(max_time=0.5)
        assert result.sim_time >= 0.5
        assert result.steps >= 5

    def test_run_until_clock_done(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=0.3)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        result = runner.run()
        # Clock should stop at max_sim_time
        assert result.sim_time <= 0.4  # approximately 0.3

    def test_episode_id_increments(self, runner):
        r1 = runner.run(max_steps=1)
        assert r1.episode_id == 1
        r2 = runner.run(max_steps=1)
        assert r2.episode_id == 2

    def test_result_has_final_positions(self, runner):
        result = runner.run(max_steps=3)
        assert len(result.final_positions) > 0
        for eid, pos in result.final_positions.items():
            assert isinstance(eid, int)
            assert len(pos) == 2

    def test_stop_interrupts_run(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=100.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        runner.on_post_step(lambda r, dt: r.stop() if r.clock.step_count >= 3 else None)
        result = runner.run()
        assert result.steps == 3


# ===================================================================
# Async run
# ===================================================================


class TestRunAsync:
    def test_async_run(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        result = asyncio.run(runner.run_async(max_steps=5, yield_interval=2))
        assert result.steps == 5
        assert result.sim_time > 0
        assert not runner.is_running

    def test_async_termination_condition(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        count = [0]

        def term(r):
            count[0] += 1
            return count[0] >= 4

        runner.add_termination_condition(term)
        result = asyncio.run(runner.run_async(max_steps=100))
        assert result.steps == 4


# ===================================================================
# Snapshots
# ===================================================================


class TestSnapshots:
    def test_save_and_load_snapshot(self, runner):
        runner.step()
        runner.save_snapshot("s1")
        runner.step()
        runner.step()
        # Restore
        assert runner.load_snapshot("s1") is True

    def test_load_nonexistent_snapshot(self, runner):
        assert runner.load_snapshot("nonexistent") is False

    def test_save_initial_state_and_reset(self, runner):
        runner.save_initial_state()
        # Run a few steps
        runner.run(max_steps=5)
        old_time = runner.clock.sim_time
        assert old_time > 0
        # Reset
        runner.reset()
        assert runner.clock.sim_time == 0.0


# ===================================================================
# Run episodes
# ===================================================================


class TestRunEpisodes:
    def test_basic_episodes(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        results = runner.run_episodes(n_episodes=3, max_steps_per_episode=5)
        assert len(results) == 3
        for r in results:
            assert r.steps == 5

    def test_episodes_without_reset(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=100.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        results = runner.run_episodes(n_episodes=2, max_steps_per_episode=3, reset_between=False)
        assert len(results) == 2
        # Without reset, sim_time accumulates
        assert results[1].sim_time > results[0].sim_time


# ===================================================================
# Parallel execution
# ===================================================================


class TestRunParallel:
    def test_parallel_episodes(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        results = runner.run_parallel(n_episodes=3, max_steps_per_episode=5, max_workers=2)
        assert len(results) == 3
        # Results should be sorted by episode_id
        ids = [r.episode_id for r in results]
        assert ids == sorted(ids)

    def test_parallel_progress_callback(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        progress = []
        runner.on_progress(lambda info: progress.append(info))
        runner.run_parallel(n_episodes=2, max_steps_per_episode=5, max_workers=2)
        assert len(progress) == 2


# ===================================================================
# Batch simulation
# ===================================================================


class TestRunBatch:
    def test_batch_configs(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        configs = [
            {"metadata": {"trial": 0}},
            {"metadata": {"trial": 1}},
        ]
        results = runner.run_batch(configs, max_steps=5)
        assert len(results) == 2
        assert results[0].episode_id == 0
        assert results[1].episode_id == 1
        assert results[0].metadata["trial"] == 0
        assert results[1].metadata["trial"] == 1

    def test_batch_with_entities(self, basic_world):
        clock = SimulationClock(dt=0.1, max_sim_time=10.0)
        runner = SimulationRunner(world=basic_world, clock=clock, seed=0)
        configs = [
            {"entities": [{"position": (5, 5), "kind": "pedestrian", "radius": 0.3}]},
        ]
        results = runner.run_batch(configs, max_steps=3)
        assert len(results) == 1


# ===================================================================
# Stats and accessors
# ===================================================================


class TestAccessors:
    def test_sim_time(self, runner):
        assert runner.sim_time == 0.0
        runner.step()
        assert runner.sim_time > 0

    def test_step_count(self, runner):
        assert runner.step_count == 0
        runner.step()
        assert runner.step_count == 1

    def test_entity_positions(self, runner):
        pos = runner.entity_positions()
        assert isinstance(pos, np.ndarray)
        assert pos.shape[1] == 2

    def test_entity_velocities(self, runner):
        vel = runner.entity_velocities()
        assert isinstance(vel, np.ndarray)
        assert vel.shape[1] == 2

    def test_stats(self, runner):
        runner.step()
        s = runner.stats()
        assert "clock" in s
        assert "physics" in s
        assert "events" in s
        assert s["episode_id"] == 0
        assert "episode_collisions" in s
