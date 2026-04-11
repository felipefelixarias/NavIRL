"""Tests for the navirl.simulation sub-package.

Covers clock, entities, events, world, and physics modules which
previously had 0% test coverage.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.simulation.clock import ClockStats, ScheduledEvent, SimulationClock
from navirl.simulation.entities import (
    Door,
    DynamicObstacle,
    EntityManager,
    NavigationGraph,
    Region,
    StaticObstacle,
    Wall,
    Waypoint,
)
from navirl.simulation.events import EventBus, EventFilter, EventRecord, EventType
from navirl.simulation.world import AABB, SpatialGrid, World, WorldBuilder

# ===================================================================
# AABB
# ===================================================================


class TestAABB:
    def test_contains(self):
        bb = AABB(0, 0, 10, 10)
        assert bb.contains(5, 5)
        assert bb.contains(0, 0)
        assert bb.contains(10, 10)
        assert not bb.contains(-1, 5)
        assert not bb.contains(5, 11)

    def test_overlaps(self):
        a = AABB(0, 0, 5, 5)
        b = AABB(3, 3, 8, 8)
        c = AABB(6, 6, 10, 10)
        assert a.overlaps(b)
        assert b.overlaps(a)
        assert not a.overlaps(c)

    def test_expand(self):
        bb = AABB(2, 3, 8, 9)
        expanded = bb.expand(1.0)
        assert expanded.x_min == pytest.approx(1.0)
        assert expanded.y_min == pytest.approx(2.0)
        assert expanded.x_max == pytest.approx(9.0)
        assert expanded.y_max == pytest.approx(10.0)

    def test_properties(self):
        bb = AABB(0, 0, 6, 4)
        assert bb.width == pytest.approx(6.0)
        assert bb.height == pytest.approx(4.0)
        assert bb.area == pytest.approx(24.0)
        np.testing.assert_allclose(bb.center, [3.0, 2.0])


# ===================================================================
# SpatialGrid
# ===================================================================


class TestSpatialGrid:
    def test_insert_and_query_point(self):
        grid = SpatialGrid(AABB(0, 0, 10, 10), cell_size=2.0)
        grid.insert(1, AABB(1, 1, 3, 3))
        result = grid.query_point(2, 2)
        assert 1 in result

    def test_remove(self):
        grid = SpatialGrid(AABB(0, 0, 10, 10), cell_size=2.0)
        grid.insert(1, AABB(1, 1, 3, 3))
        grid.remove(1)
        result = grid.query_point(2, 2)
        assert 1 not in result

    def test_query_aabb(self):
        grid = SpatialGrid(AABB(0, 0, 20, 20), cell_size=5.0)
        grid.insert(1, AABB(1, 1, 2, 2))
        grid.insert(2, AABB(15, 15, 16, 16))
        result = grid.query_aabb(AABB(0, 0, 5, 5))
        assert 1 in result
        assert 2 not in result

    def test_query_radius(self):
        grid = SpatialGrid(AABB(0, 0, 20, 20), cell_size=5.0)
        grid.insert(1, AABB(4.5, 4.5, 5.5, 5.5))
        result = grid.query_radius(5.0, 5.0, 2.0)
        assert 1 in result

    def test_update(self):
        grid = SpatialGrid(AABB(0, 0, 20, 20), cell_size=5.0)
        grid.insert(1, AABB(1, 1, 2, 2))
        grid.update(1, AABB(15, 15, 16, 16))
        assert 1 not in grid.query_point(1.5, 1.5)
        assert 1 in grid.query_point(15.5, 15.5)

    def test_clear(self):
        grid = SpatialGrid(AABB(0, 0, 10, 10), cell_size=2.0)
        grid.insert(1, AABB(1, 1, 3, 3))
        grid.clear()
        assert grid.query_point(2, 2) == set()


# ===================================================================
# SimulationClock
# ===================================================================


class TestSimulationClock:
    def test_initial_state(self):
        clock = SimulationClock(dt=0.01, max_sim_time=10.0)
        assert clock.sim_time == pytest.approx(0.0)
        assert clock.step_count == 0
        assert clock.dt == pytest.approx(0.01)
        assert not clock.paused
        assert not clock.done

    def test_tick_fixed_advances_time(self):
        clock = SimulationClock(dt=0.05)
        clock.start()
        dt = clock.tick_fixed()
        assert dt == pytest.approx(0.05)
        assert clock.sim_time == pytest.approx(0.05)
        assert clock.step_count == 1

    def test_multiple_ticks(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        for _ in range(10):
            clock.tick_fixed()
        assert clock.sim_time == pytest.approx(1.0)
        assert clock.step_count == 10

    def test_done_by_max_sim_time(self):
        clock = SimulationClock(dt=0.5, max_sim_time=1.0)
        clock.start()
        clock.tick_fixed()
        assert not clock.done
        clock.tick_fixed()
        assert clock.done

    def test_done_by_max_steps(self):
        clock = SimulationClock(dt=0.01, max_steps=3)
        clock.start()
        for _ in range(3):
            clock.tick_fixed()
        assert clock.done

    def test_pause_resume(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        clock.tick_fixed()
        clock.pause()
        assert clock.paused
        dt = clock.tick_fixed()
        assert dt == pytest.approx(0.0)
        assert clock.sim_time == pytest.approx(0.1)
        clock.resume()
        assert not clock.paused
        clock.tick_fixed()
        assert clock.sim_time == pytest.approx(0.2)

    def test_toggle_pause(self):
        clock = SimulationClock(dt=0.1)
        assert not clock.paused
        result = clock.toggle_pause()
        assert result is True
        assert clock.paused
        result = clock.toggle_pause()
        assert result is False
        assert not clock.paused

    def test_reset(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        for _ in range(5):
            clock.tick_fixed()
        clock.reset()
        assert clock.sim_time == pytest.approx(0.0)
        assert clock.step_count == 0
        assert not clock.paused

    def test_dt_setter_clamps(self):
        clock = SimulationClock(dt=0.1)
        clock.dt = -1.0
        assert clock.dt > 0

    def test_time_scale(self):
        clock = SimulationClock(dt=0.1, time_scale=2.0)
        assert clock.time_scale == pytest.approx(2.0)
        clock.time_scale = 0.5
        assert clock.time_scale == pytest.approx(0.5)
        clock.time_scale = -1.0
        assert clock.time_scale == pytest.approx(0.0)

    def test_schedule_fires_event(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        fired = []
        clock.schedule(0.25, lambda evt: fired.append(evt.sim_time), name="test")
        for _ in range(3):
            clock.tick_fixed()
        assert len(fired) == 1
        assert fired[0] == pytest.approx(0.25)

    def test_schedule_after(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        clock.tick_fixed()  # sim_time = 0.1
        fired = []
        clock.schedule_after(0.15, lambda evt: fired.append(evt.sim_time))
        # Event should fire at 0.25
        clock.tick_fixed()  # 0.2
        assert len(fired) == 0
        clock.tick_fixed()  # 0.3
        assert len(fired) == 1

    def test_cancel_event(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        fired = []
        evt = clock.schedule(0.15, lambda e: fired.append(1))
        clock.cancel_event(evt)
        for _ in range(5):
            clock.tick_fixed()
        assert len(fired) == 0

    def test_pending_events(self):
        clock = SimulationClock(dt=0.1)
        clock.schedule(1.0, lambda e: None)
        clock.schedule(2.0, lambda e: None)
        assert clock.pending_events() == 2
        evt = clock.schedule(3.0, lambda e: None)
        clock.cancel_event(evt)
        assert clock.pending_events() == 2

    def test_repeat_event(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        count = []
        clock.schedule(0.1, lambda e: count.append(1), repeat_interval=0.1)
        for _ in range(5):
            clock.tick_fixed()
        assert len(count) == 5

    def test_stats(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        for _ in range(3):
            clock.tick_fixed()
        stats = clock.stats()
        assert isinstance(stats, ClockStats)
        assert stats.step_count == 3
        assert stats.sim_time == pytest.approx(0.3)
        d = stats.as_dict()
        assert "step_count" in d

    def test_wall_elapsed_before_start(self):
        clock = SimulationClock()
        assert clock.wall_elapsed == pytest.approx(0.0)

    def test_repr(self):
        clock = SimulationClock(dt=0.05)
        assert "SimulationClock" in repr(clock)

    # --- tick() and tick_variable() ---

    def test_tick_advances_in_non_realtime(self):
        clock = SimulationClock(dt=0.05, time_scale=2.0, real_time=False)
        clock.start()
        dt = clock.tick()
        assert dt == pytest.approx(0.05 * 2.0)
        assert clock.sim_time == pytest.approx(0.1)
        assert clock.step_count == 1

    def test_tick_paused_returns_zero(self):
        clock = SimulationClock(dt=0.05)
        clock.start()
        clock.pause()
        dt = clock.tick()
        assert dt == pytest.approx(0.0)
        assert clock.step_count == 0

    def test_tick_realtime_mode(self):
        clock = SimulationClock(dt=0.05, real_time=True)
        clock.start()
        import time as _time

        _time.sleep(0.02)
        dt = clock.tick()
        assert dt > 0
        assert clock.step_count == 1

    def test_tick_variable_paused(self):
        clock = SimulationClock(dt=0.05)
        clock.start()
        clock.pause()
        dt = clock.tick_variable()
        assert dt == pytest.approx(0.0)

    def test_tick_variable_advances(self):
        clock = SimulationClock(dt=0.05, time_scale=1.0)
        clock.start()
        import time as _time

        _time.sleep(0.01)
        dt = clock.tick_variable()
        assert dt > 0
        assert clock.step_count == 1

    # --- accumulate_and_step() ---

    def test_accumulate_and_step_basic(self):
        clock = SimulationClock(dt=0.01)
        clock.start()
        import time as _time

        _time.sleep(0.03)
        steps = clock.accumulate_and_step()
        assert steps >= 1
        assert clock.step_count == steps

    def test_accumulate_and_step_no_time_elapsed(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        # Immediately call — likely 0 steps since dt is large
        steps = clock.accumulate_and_step()
        assert steps >= 0

    # --- Frame-rate control ---

    def test_begin_frame_and_wait(self):
        clock = SimulationClock(dt=0.01, target_fps=1000.0)
        clock.start()
        clock.begin_frame()
        duration = clock.wait_for_frame()
        assert duration > 0

    def test_target_fps_property(self):
        clock = SimulationClock(target_fps=30.0)
        assert clock.target_fps == pytest.approx(30.0)
        clock.target_fps = 120.0
        assert clock.target_fps == pytest.approx(120.0)

    def test_target_fps_clamps(self):
        clock = SimulationClock(target_fps=30.0)
        clock.target_fps = 0.5
        assert clock.target_fps >= 1.0

    # --- real_time_ratio ---

    def test_real_time_ratio_before_start(self):
        clock = SimulationClock()
        assert clock.real_time_ratio == pytest.approx(0.0)

    def test_real_time_ratio_after_ticks(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        for _ in range(10):
            clock.tick_fixed()
        ratio = clock.real_time_ratio
        # sim_time = 1.0, wall time is very small → ratio should be large
        assert ratio > 0

    # --- reset_stats ---

    def test_reset_stats_clears_times(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        for _ in range(3):
            clock.tick_fixed()
        assert len(clock._step_wall_times) == 3
        clock.reset_stats()
        assert len(clock._step_wall_times) == 0
        # sim_time should NOT be reset
        assert clock.sim_time == pytest.approx(0.3)

    # --- Event priority ordering ---

    def test_event_priority_ordering(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        fired = []
        clock.schedule(0.1, lambda e: fired.append("low"), priority=10)
        clock.schedule(0.1, lambda e: fired.append("high"), priority=0)
        clock.tick_fixed()
        assert fired == ["high", "low"]

    def test_event_data_payload(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        received = []
        clock.schedule(0.1, lambda e: received.append(e.data), data={"key": "val"})
        clock.tick_fixed()
        assert len(received) == 1
        assert received[0] == {"key": "val"}

    # --- Pause timing ---

    def test_pause_accumulates_time(self):
        clock = SimulationClock(dt=0.1)
        clock.start()
        clock.tick_fixed()
        clock.pause()
        import time as _time

        _time.sleep(0.05)
        clock.resume()
        stats = clock.stats()
        assert stats.total_paused_time >= 0.04


# ===================================================================
# Entities
# ===================================================================


class TestStaticObstacle:
    def test_creation(self):
        obs = StaticObstacle(entity_id=1, position=(5, 3), radius=1.0)
        assert obs.entity_id == 1
        assert obs.kind() == "static_obstacle"
        np.testing.assert_allclose(obs.position, [5.0, 3.0])

    def test_contains_point(self):
        obs = StaticObstacle(position=(0, 0), radius=2.0)
        assert obs.contains_point((1, 0))
        assert obs.contains_point((0, 2))
        assert not obs.contains_point((3, 0))

    def test_bounding_box(self):
        obs = StaticObstacle(position=(5, 5), radius=1.0)
        bb = obs.bounding_box()
        assert bb.x_min == pytest.approx(4.0)
        assert bb.y_max == pytest.approx(6.0)

    def test_to_dict(self):
        obs = StaticObstacle(entity_id=0, position=(1, 2), radius=0.5)
        d = obs.to_dict()
        assert d["type"] == "static_obstacle"
        assert d["radius"] == 0.5


class TestDynamicObstacle:
    def test_creation(self):
        obs = DynamicObstacle(entity_id=2, position=(0, 0), velocity=(1, 0), radius=0.3, mass=2.0)
        assert obs.kind() == "dynamic_obstacle"
        assert obs.mass == pytest.approx(2.0)

    def test_step(self):
        obs = DynamicObstacle(position=(0, 0), velocity=(2, 3))
        obs.step(0.5)
        np.testing.assert_allclose(obs.position, [1.0, 1.5])

    def test_speed(self):
        obs = DynamicObstacle(velocity=(3, 4))
        assert obs.speed == pytest.approx(5.0)

    def test_to_dict(self):
        obs = DynamicObstacle(entity_id=0, velocity=(1, 2))
        d = obs.to_dict()
        assert "velocity" in d
        assert "mass" in d


class TestWall:
    def test_creation(self):
        wall = Wall(start=(0, 0), end=(10, 0))
        assert wall.kind() == "wall"
        assert wall.length == pytest.approx(10.0)

    def test_direction_and_normal(self):
        wall = Wall(start=(0, 0), end=(4, 0))
        np.testing.assert_allclose(wall.direction, [1.0, 0.0])
        np.testing.assert_allclose(wall.normal, [0.0, 1.0])

    def test_closest_point(self):
        wall = Wall(start=(0, 0), end=(10, 0))
        cp = wall.closest_point((5, 3))
        np.testing.assert_allclose(cp, [5.0, 0.0])

    def test_closest_point_clamped(self):
        wall = Wall(start=(0, 0), end=(10, 0))
        cp = wall.closest_point((-5, 0))
        np.testing.assert_allclose(cp, [0.0, 0.0])

    def test_distance_to_point(self):
        wall = Wall(start=(0, 0), end=(10, 0))
        assert wall.distance_to_point((5, 3)) == pytest.approx(3.0)

    def test_degenerate_wall(self):
        wall = Wall(start=(5, 5), end=(5, 5))
        np.testing.assert_allclose(wall.direction, [1.0, 0.0])  # fallback

    def test_bounding_box(self):
        wall = Wall(start=(0, 0), end=(10, 0), thickness=0.2)
        bb = wall.bounding_box()
        assert bb.y_min == pytest.approx(-0.1)
        assert bb.y_max == pytest.approx(0.1)

    def test_to_dict(self):
        wall = Wall(start=(0, 0), end=(5, 0))
        d = wall.to_dict()
        assert d["type"] == "wall"
        assert "start" in d


class TestDoor:
    def test_open_close(self):
        door = Door(start=(0, 0), end=(1, 0))
        assert not door.is_open
        door.open(timestamp=1.0)
        assert door.is_open
        door.close()
        assert not door.is_open

    def test_toggle(self):
        door = Door()
        assert door.toggle(0.0) is True
        assert door.toggle(0.0) is False

    def test_auto_close(self):
        door = Door(auto_close_time=2.0)
        door.open(timestamp=1.0)
        door.update(sim_time=2.5)
        assert door.is_open
        door.update(sim_time=3.0)
        assert not door.is_open

    def test_callbacks(self):
        door = Door()
        opened = []
        closed = []
        door.on_open(lambda d: opened.append(1))
        door.on_close(lambda d: closed.append(1))
        door.open()
        assert len(opened) == 1
        door.close()
        assert len(closed) == 1

    def test_kind(self):
        assert Door().kind() == "door"

    def test_to_dict(self):
        d = Door(auto_close_time=5.0).to_dict()
        assert "is_open" in d
        assert d["auto_close_time"] == 5.0


class TestRegion:
    def test_contains_point(self):
        region = Region(position=(5, 5), size=(4, 4))
        assert region.contains_point((5, 5))
        assert region.contains_point((3.5, 3.5))
        assert not region.contains_point((1, 1))

    def test_sample_point(self):
        region = Region(position=(5, 5), size=(2, 2))
        rng = np.random.default_rng(42)
        pt = region.sample_point(rng)
        assert region.contains_point(pt)

    def test_sample_points(self):
        region = Region(position=(5, 5), size=(2, 2))
        rng = np.random.default_rng(42)
        pts = region.sample_points(100, rng)
        assert pts.shape == (100, 2)
        for pt in pts:
            assert region.contains_point(pt)

    def test_properties(self):
        region = Region(size=(6, 4))
        assert region.width == pytest.approx(6.0)
        assert region.height == pytest.approx(4.0)

    def test_to_dict(self):
        d = Region(label="spawn").to_dict()
        assert d["label"] == "spawn"


class TestWaypoint:
    def test_is_reached(self):
        wp = Waypoint(position=(5, 5), radius=1.0)
        assert wp.is_reached((5.5, 5.0))
        assert not wp.is_reached((7, 5))

    def test_connect(self):
        wp = Waypoint(entity_id=1)
        wp.connect(2)
        wp.connect(3)
        wp.connect(2)  # duplicate, should not add
        assert wp.connections == [2, 3]

    def test_kind(self):
        assert Waypoint().kind() == "waypoint"


class TestEntityBase:
    def test_distance_to(self):
        a = StaticObstacle(position=(0, 0))
        b = StaticObstacle(position=(3, 4))
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_point(self):
        a = StaticObstacle(position=(0, 0))
        assert a.distance_to_point((3, 4)) == pytest.approx(5.0)

    def test_tags(self):
        obs = StaticObstacle(tags={"obstacle", "fixed"})
        assert obs.has_tag("obstacle")
        assert not obs.has_tag("movable")


# ===================================================================
# NavigationGraph
# ===================================================================


class TestNavigationGraph:
    def _make_graph(self):
        """Create a simple triangle graph."""
        g = NavigationGraph()
        w1 = Waypoint(entity_id=0, position=(0, 0))
        w2 = Waypoint(entity_id=1, position=(3, 0))
        w3 = Waypoint(entity_id=2, position=(0, 4))
        g.add_node(w1)
        g.add_node(w2)
        g.add_node(w3)
        g.add_edge_bidirectional(0, 1)
        g.add_edge_bidirectional(1, 2)
        g.add_edge_bidirectional(0, 2)
        return g

    def test_add_nodes_and_edges(self):
        g = self._make_graph()
        assert g.num_nodes == 3
        assert g.num_edges == 6  # bidirectional = 2 per pair

    def test_neighbours(self):
        g = self._make_graph()
        nbrs = g.neighbours(0)
        nbr_ids = [n[0] for n in nbrs]
        assert 1 in nbr_ids
        assert 2 in nbr_ids

    def test_shortest_path(self):
        g = self._make_graph()
        path, cost = g.shortest_path(0, 2)
        assert path[0] == 0
        assert path[-1] == 2
        assert cost == pytest.approx(4.0)  # direct distance (0,0) to (0,4)

    def test_no_path(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=0, position=(0, 0))
        w2 = Waypoint(entity_id=1, position=(10, 10))
        g.add_node(w1)
        g.add_node(w2)
        path, cost = g.shortest_path(0, 1)
        assert path == []
        assert cost == float("inf")

    def test_remove_node(self):
        g = self._make_graph()
        g.remove_node(1)
        assert g.num_nodes == 2
        assert 1 not in g.node_ids

    def test_build_from_waypoints(self):
        w1 = Waypoint(entity_id=0, position=(0, 0))
        w2 = Waypoint(entity_id=1, position=(5, 0))
        w1.connect(1)
        g = NavigationGraph()
        g.build_from_waypoints([w1, w2])
        assert g.num_nodes == 2
        assert g.num_edges == 1

    def test_positions_array(self):
        g = self._make_graph()
        pos = g.positions_array()
        assert pos.shape == (3, 2)

    def test_positions_array_empty(self):
        g = NavigationGraph()
        pos = g.positions_array()
        assert pos.shape == (0, 2)

    def test_to_dict(self):
        g = self._make_graph()
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d


# ===================================================================
# EntityManager
# ===================================================================


class TestEntityManager:
    def test_add_and_get(self):
        mgr = EntityManager()
        obs = StaticObstacle(position=(5, 5), radius=1.0)
        eid = mgr.add(obs)
        assert mgr.get(eid) is obs
        assert eid in mgr
        assert len(mgr) == 1

    def test_remove(self):
        mgr = EntityManager()
        obs = StaticObstacle(position=(5, 5))
        eid = mgr.add(obs)
        removed = mgr.remove(eid)
        assert removed is obs
        assert len(mgr) == 0
        assert mgr.remove(eid) is None

    def test_by_kind(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1)))
        mgr.add(DynamicObstacle(position=(2, 2)))
        mgr.add(StaticObstacle(position=(3, 3)))
        statics = mgr.by_kind("static_obstacle")
        assert len(statics) == 2

    def test_by_tag(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1), tags={"wall"}))
        mgr.add(StaticObstacle(position=(2, 2), tags={"floor"}))
        walls = mgr.by_tag("wall")
        assert len(walls) == 1

    def test_by_type(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1)))
        mgr.add(DynamicObstacle(position=(2, 2)))
        dynamics = mgr.by_type(DynamicObstacle)
        assert len(dynamics) == 1

    def test_query_radius(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(5, 5), radius=0.5))
        mgr.add(StaticObstacle(position=(20, 20), radius=0.5))
        nearby = mgr.query_radius(5, 5, 3.0)
        assert len(nearby) == 1

    def test_nearest(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 0), radius=0.5))
        mgr.add(StaticObstacle(position=(5, 0), radius=0.5))
        mgr.add(StaticObstacle(position=(10, 0), radius=0.5))
        nearest = mgr.nearest(0, 0, k=2)
        assert len(nearest) == 2
        # First should be closest
        assert nearest[0][1] < nearest[1][1]

    def test_active_and_deactivate(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1)))
        mgr.add(StaticObstacle(position=(2, 2)))
        assert len(mgr.active()) == 2
        mgr.deactivate_all()
        assert len(mgr.active()) == 0
        mgr.activate_all()
        assert len(mgr.active()) == 2

    def test_clear(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1)))
        mgr.clear()
        assert len(mgr) == 0

    def test_summary(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1)))
        mgr.add(DynamicObstacle(position=(2, 2)))
        s = mgr.summary()
        assert s["static_obstacle"] == 1
        assert s["dynamic_obstacle"] == 1

    def test_iter(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(1, 1)))
        mgr.add(StaticObstacle(position=(2, 2)))
        entities = list(mgr)
        assert len(entities) == 2

    def test_entities_in_region(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(5, 5), radius=0.5))
        mgr.add(StaticObstacle(position=(20, 20), radius=0.5))
        region = Region(position=(5, 5), size=(4, 4))
        inside = mgr.entities_in_region(region)
        assert len(inside) == 1

    def test_query_aabb(self):
        mgr = EntityManager()
        mgr.add(StaticObstacle(position=(5, 5), radius=0.5))
        mgr.add(StaticObstacle(position=(20, 20), radius=0.5))
        result = mgr.query_aabb(AABB(4, 4, 6, 6))
        assert len(result) == 1


# ===================================================================
# EventBus
# ===================================================================


class TestEventRecord:
    def test_round_trip(self):
        rec = EventRecord(
            event_type=EventType.COLLISION,
            sim_time=1.5,
            source_id=1,
            target_id=2,
            data={"penetration": 0.1},
        )
        d = rec.to_dict()
        rec2 = EventRecord.from_dict(d)
        assert rec2.event_type == EventType.COLLISION
        assert rec2.sim_time == pytest.approx(1.5)
        assert rec2.data["penetration"] == pytest.approx(0.1)


class TestEventFilter:
    def test_basic_filter(self):
        filt = EventFilter(event_types={EventType.COLLISION})
        rec_col = EventRecord(event_type=EventType.COLLISION)
        rec_goal = EventRecord(event_type=EventType.GOAL_REACHED)
        assert filt.matches(rec_col)
        assert not filt.matches(rec_goal)

    def test_source_filter(self):
        filt = EventFilter(source_ids={1})
        rec = EventRecord(event_type=EventType.STEP, source_id=1)
        assert filt.matches(rec)
        rec2 = EventRecord(event_type=EventType.STEP, source_id=2)
        assert not filt.matches(rec2)

    def test_time_range_filter(self):
        filt = EventFilter(min_sim_time=1.0, max_sim_time=5.0)
        assert filt.matches(EventRecord(event_type=EventType.STEP, sim_time=3.0))
        assert not filt.matches(EventRecord(event_type=EventType.STEP, sim_time=0.5))
        assert not filt.matches(EventRecord(event_type=EventType.STEP, sim_time=6.0))

    def test_and_filter(self):
        f1 = EventFilter(event_types={EventType.COLLISION})
        f2 = EventFilter(source_ids={1})
        combined = f1 & f2
        rec = EventRecord(event_type=EventType.COLLISION, source_id=1)
        assert combined.matches(rec)
        rec2 = EventRecord(event_type=EventType.COLLISION, source_id=2)
        assert not combined.matches(rec2)

    def test_or_filter(self):
        f1 = EventFilter(event_types={EventType.COLLISION})
        f2 = EventFilter(event_types={EventType.GOAL_REACHED})
        combined = f1 | f2
        assert combined.matches(EventRecord(event_type=EventType.COLLISION))
        assert combined.matches(EventRecord(event_type=EventType.GOAL_REACHED))
        assert not combined.matches(EventRecord(event_type=EventType.TIMEOUT))

    def test_data_key_filter(self):
        filt = EventFilter(data_key="tag", data_value="important")
        rec1 = EventRecord(event_type=EventType.CUSTOM, data={"tag": "important"})
        rec2 = EventRecord(event_type=EventType.CUSTOM, data={"tag": "other"})
        assert filt.matches(rec1)
        assert not filt.matches(rec2)


class TestEventBus:
    def test_publish_and_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda rec: received.append(rec), event_type=EventType.COLLISION)
        bus.publish(EventType.COLLISION, sim_time=1.0)
        assert len(received) == 1
        assert received[0].event_type == EventType.COLLISION

    def test_wildcard_subscription(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda rec: received.append(rec))
        bus.publish(EventType.COLLISION)
        bus.publish(EventType.GOAL_REACHED)
        assert len(received) == 2

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        sid = bus.subscribe(lambda rec: received.append(rec))
        bus.publish(EventType.STEP)
        assert len(received) == 1
        bus.unsubscribe(sid)
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_once_subscription(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda rec: received.append(1), event_type=EventType.STEP, once=True)
        bus.publish(EventType.STEP)
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_entity_filter(self):
        bus = EventBus()
        received = []
        bus.subscribe(
            lambda rec: received.append(rec),
            event_type=EventType.COLLISION,
            entity_filter=1,
        )
        bus.publish(EventType.COLLISION, source_id=1, target_id=2)
        bus.publish(EventType.COLLISION, source_id=3, target_id=4)
        assert len(received) == 1

    def test_mute_unmute(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda rec: received.append(rec), event_type=EventType.STEP)
        bus.mute(EventType.STEP)
        bus.publish(EventType.STEP)
        assert len(received) == 0
        bus.unmute(EventType.STEP)
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_pause_resume(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda rec: received.append(rec))
        bus.pause()
        bus.publish(EventType.STEP)
        assert len(received) == 0
        bus.resume()
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_history(self):
        bus = EventBus(recording=True)
        bus.publish(EventType.COLLISION, sim_time=1.0)
        bus.publish(EventType.GOAL_REACHED, sim_time=2.0)
        assert len(bus.history) == 2
        bus.clear_history()
        assert len(bus.history) == 0

    def test_recording_off(self):
        bus = EventBus(recording=False)
        bus.publish(EventType.STEP)
        assert len(bus.history) == 0

    def test_history_by_type(self):
        bus = EventBus()
        bus.publish(EventType.COLLISION)
        bus.publish(EventType.STEP)
        bus.publish(EventType.COLLISION)
        assert len(bus.history_by_type(EventType.COLLISION)) == 2

    def test_history_in_range(self):
        bus = EventBus()
        bus.publish(EventType.STEP, sim_time=1.0)
        bus.publish(EventType.STEP, sim_time=3.0)
        bus.publish(EventType.STEP, sim_time=5.0)
        result = bus.history_in_range(2.0, 4.0)
        assert len(result) == 1

    def test_history_for_entity(self):
        bus = EventBus()
        bus.publish(EventType.COLLISION, source_id=1)
        bus.publish(EventType.COLLISION, source_id=2, target_id=1)
        bus.publish(EventType.COLLISION, source_id=3)
        result = bus.history_for_entity(1)
        assert len(result) == 2

    def test_filter_history(self):
        bus = EventBus()
        bus.publish(EventType.COLLISION, sim_time=1.0)
        bus.publish(EventType.STEP, sim_time=2.0)
        filt = EventFilter(event_types={EventType.COLLISION})
        result = bus.filter_history(filt)
        assert len(result) == 1

    def test_event_counts(self):
        bus = EventBus()
        bus.publish(EventType.COLLISION)
        bus.publish(EventType.COLLISION)
        bus.publish(EventType.STEP)
        counts = bus.event_counts()
        assert counts["collision"] == 2
        assert counts["step"] == 1

    def test_convenience_emitters(self):
        bus = EventBus()
        rec = bus.emit_collision(1.0, entity_a=1, entity_b=2, penetration=0.5)
        assert rec.event_type == EventType.COLLISION
        assert rec.data["penetration"] == 0.5

        rec = bus.emit_goal_reached(2.0, entity_id=1, goal_id=5)
        assert rec.event_type == EventType.GOAL_REACHED

        rec = bus.emit_timeout(3.0)
        assert rec.event_type == EventType.TIMEOUT

        rec = bus.emit_zone_enter(4.0, entity_id=1, zone_id=2)
        assert rec.event_type == EventType.ZONE_ENTER

        rec = bus.emit_zone_exit(5.0, entity_id=1, zone_id=2)
        assert rec.event_type == EventType.ZONE_EXIT

    def test_serialize_load_history(self):
        bus = EventBus()
        bus.publish(EventType.COLLISION, sim_time=1.0, source_id=1)
        data = bus.serialize_history()
        bus2 = EventBus()
        bus2.load_history(data)
        assert len(bus2.history) == 1
        assert bus2.history[0].event_type == EventType.COLLISION

    def test_replay_instant(self):
        bus = EventBus()
        bus.publish(EventType.STEP, sim_time=0.0)
        bus.publish(EventType.STEP, sim_time=1.0)
        replayed = []
        bus.register_replay_callback(lambda rec: replayed.append(rec))
        bus.replay(speed=0)
        assert len(replayed) == 2

    def test_stats(self):
        bus = EventBus()
        bus.subscribe(lambda r: None)
        bus.publish(EventType.STEP)
        stats = bus.stats()
        assert stats["publish_count"] == 1
        assert stats["subscriber_count"] == 1

    def test_reset(self):
        bus = EventBus()
        bus.subscribe(lambda r: None)
        bus.publish(EventType.STEP)
        bus.reset()
        assert bus.publish_count == 0
        assert bus.subscriber_count == 0
        assert len(bus.history) == 0

    def test_unsubscribe_all(self):
        bus = EventBus()
        bus.subscribe(lambda r: None)
        bus.subscribe(lambda r: None)
        bus.unsubscribe_all()
        assert bus.subscriber_count == 0


# ===================================================================
# World
# ===================================================================


class TestWorld:
    def test_add_remove_entity(self):
        world = World(width=20, height=20)
        eid = world.add_entity(position=(5, 5))
        assert world.entity_count() == 1
        world.remove_entity(eid)
        assert world.entity_count() == 0

    def test_get_entity(self):
        world = World()
        eid = world.add_entity(position=(3, 4), kind="robot")
        e = world.get_entity(eid)
        assert e is not None
        np.testing.assert_allclose(e["position"], [3.0, 4.0])
        assert e["kind"] == "robot"

    def test_entity_ids_filter(self):
        world = World()
        world.add_entity(position=(1, 1), kind="pedestrian")
        world.add_entity(position=(2, 2), kind="robot")
        world.add_entity(position=(3, 3), kind="pedestrian")
        peds = world.entity_ids(kind="pedestrian")
        assert len(peds) == 2
        all_ids = world.entity_ids()
        assert len(all_ids) == 3

    def test_add_wall(self):
        world = World()
        idx = world.add_wall((0, 0), (10, 0))
        assert idx == 0
        assert len(world.walls) == 1

    def test_add_boundary_walls(self):
        world = World(width=10, height=10)
        world.add_boundary_walls()
        assert len(world.walls) == 4

    def test_positions_array(self):
        world = World()
        world.add_entity(position=(1, 2))
        world.add_entity(position=(3, 4))
        pos = world.positions_array()
        assert pos.shape == (2, 2)

    def test_velocities_array(self):
        world = World()
        world.add_entity(position=(0, 0), velocity=(1, 2))
        vels = world.velocities_array()
        np.testing.assert_allclose(vels[0], [1.0, 2.0])

    def test_query_radius(self):
        world = World(width=50, height=50)
        world.add_entity(position=(5, 5))
        world.add_entity(position=(40, 40))
        nearby = world.query_radius(5, 5, 3.0)
        assert len(nearby) >= 1

    def test_metadata(self):
        world = World()
        world.set_metadata("scenario", "test")
        assert world.get_metadata("scenario") == "test"
        assert world.get_metadata("missing", "default") == "default"

    def test_snapshot_restore(self):
        world = World()
        world.add_entity(position=(5, 5), velocity=(1, 0))
        snap = world.snapshot()
        world.get_entity(0)["position"] = np.array([10, 10])
        world.restore(snap)
        e = world.get_entity(0)
        np.testing.assert_allclose(e["position"], [5.0, 5.0])

    def test_serialization(self):
        world = World(width=20, height=20)
        world.add_entity(position=(5, 5), kind="pedestrian")
        world.add_wall((0, 0), (20, 0))
        d = world.to_dict()
        world2 = World.from_dict(d)
        assert world2.entity_count() == 1
        assert len(world2.walls) == 1

    def test_json_serialization(self):
        world = World(width=10, height=10)
        world.add_entity(position=(2, 3))
        text = world.to_json()
        world2 = World.from_json(text)
        assert world2.entity_count() == 1

    def test_len_and_in(self):
        world = World()
        eid = world.add_entity(position=(1, 1))
        assert len(world) == 1
        assert eid in world

    def test_nearest_entities(self):
        world = World()
        world.add_entity(position=(1, 0))
        world.add_entity(position=(5, 0))
        world.add_entity(position=(10, 0))
        nearest = world.nearest_entities(0, 0, k=2)
        assert len(nearest) == 2
        # First should be closest
        assert nearest[0][1] <= nearest[1][1]


# ===================================================================
# WorldBuilder
# ===================================================================


class TestWorldBuilder:
    def test_basic_build(self):
        world = (
            WorldBuilder()
            .set_size(20, 20)
            .add_pedestrian(position=(5, 5))
            .add_robot(position=(10, 10))
            .add_obstacle(position=(15, 15))
            .add_wall((0, 0), (20, 0))
            .build()
        )
        assert world.entity_count() == 3
        assert len(world.walls) == 1

    def test_boundary_walls(self):
        world = WorldBuilder().set_size(10, 10).add_boundary_walls().build()
        assert len(world.walls) == 4

    def test_metadata(self):
        world = WorldBuilder().set_metadata("test_key", "test_value").build()
        assert world.get_metadata("test_key") == "test_value"

    def test_wrap(self):
        world = WorldBuilder().set_size(10, 10).enable_wrap().build()
        assert world.wrap is True
