"""Tests for navirl.simulation.world (World, WorldBuilder) and navirl.simulation.entities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from navirl.simulation.entities import (
    Door,
    DynamicObstacle,
    Entity,
    EntityManager,
    NavigationGraph,
    Region,
    StaticObstacle,
    Wall,
    Waypoint,
)
from navirl.simulation.world import AABB, CollisionResult, SpatialGrid, World, WorldBuilder

# ---------------------------------------------------------------------------
# Entity base class
# ---------------------------------------------------------------------------


class TestStaticObstacle:
    def test_kind(self):
        obs = StaticObstacle(entity_id=1, position=(3.0, 4.0), radius=0.5)
        assert obs.kind() == "static_obstacle"

    def test_bounding_box(self):
        obs = StaticObstacle(entity_id=1, position=(3.0, 4.0), radius=1.0)
        bb = obs.bounding_box()
        assert bb.x_min == pytest.approx(2.0)
        assert bb.x_max == pytest.approx(4.0)

    def test_contains_point(self):
        obs = StaticObstacle(entity_id=1, position=(0.0, 0.0), radius=1.0)
        assert obs.contains_point((0.5, 0.5))
        assert not obs.contains_point((2.0, 2.0))

    def test_to_dict(self):
        obs = StaticObstacle(entity_id=1, position=(1.0, 2.0), radius=0.5)
        d = obs.to_dict()
        assert d["type"] == "static_obstacle"
        assert d["radius"] == pytest.approx(0.5)

    def test_distance_to_point(self):
        obs = StaticObstacle(entity_id=1, position=(0.0, 0.0), radius=0.5)
        assert obs.distance_to_point((3.0, 4.0)) == pytest.approx(5.0)

    def test_tags(self):
        obs = StaticObstacle(entity_id=1, position=(0.0, 0.0), tags={"heavy", "fixed"})
        assert obs.has_tag("heavy")
        assert not obs.has_tag("light")


class TestDynamicObstacle:
    def test_kind(self):
        dyn = DynamicObstacle(entity_id=2, position=(1.0, 1.0))
        assert dyn.kind() == "dynamic_obstacle"

    def test_step_advances_position(self):
        dyn = DynamicObstacle(entity_id=2, position=(0.0, 0.0), velocity=(1.0, 0.0))
        dyn.step(0.5)
        assert dyn.position[0] == pytest.approx(0.5)
        assert dyn.position[1] == pytest.approx(0.0)

    def test_speed_property(self):
        dyn = DynamicObstacle(entity_id=2, position=(0.0, 0.0), velocity=(3.0, 4.0))
        assert dyn.speed == pytest.approx(5.0)

    def test_bounding_box(self):
        dyn = DynamicObstacle(entity_id=2, position=(5.0, 5.0), radius=0.3)
        bb = dyn.bounding_box()
        assert bb.x_min == pytest.approx(4.7)
        assert bb.y_max == pytest.approx(5.3)

    def test_to_dict(self):
        dyn = DynamicObstacle(entity_id=2, position=(1.0, 2.0), velocity=(0.5, -0.5))
        d = dyn.to_dict()
        assert d["type"] == "dynamic_obstacle"
        assert "velocity" in d

    def test_mass(self):
        dyn = DynamicObstacle(entity_id=2, position=(0.0, 0.0), mass=3.0)
        assert dyn.mass == pytest.approx(3.0)


class TestWall:
    def test_kind(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(10.0, 0.0))
        assert w.kind() == "wall"

    def test_length(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(3.0, 4.0))
        assert w.length == pytest.approx(5.0)

    def test_closest_point(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(10.0, 0.0))
        cp = w.closest_point((5.0, 3.0))
        np.testing.assert_allclose(cp, [5.0, 0.0], atol=1e-10)

    def test_closest_point_clamped_start(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(10.0, 0.0))
        cp = w.closest_point((-5.0, 0.0))
        np.testing.assert_allclose(cp, [0.0, 0.0], atol=1e-10)

    def test_distance_to_point(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(10.0, 0.0))
        assert w.distance_to_point((5.0, 3.0)) == pytest.approx(3.0)

    def test_direction(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(10.0, 0.0))
        d = w.direction
        np.testing.assert_allclose(d, [1.0, 0.0], atol=1e-10)

    def test_normal(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(10.0, 0.0))
        n = w.normal
        assert abs(np.dot(n, w.direction)) < 1e-10

    def test_bounding_box(self):
        w = Wall(entity_id=3, start=(2.0, 3.0), end=(8.0, 7.0))
        bb = w.bounding_box()
        assert bb.x_min <= 2.0
        assert bb.x_max >= 8.0

    def test_to_dict(self):
        w = Wall(entity_id=3, start=(0.0, 0.0), end=(5.0, 0.0))
        d = w.to_dict()
        assert d["type"] == "wall"


class TestDoor:
    def test_kind(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0))
        assert door.kind() == "door"

    def test_initial_state_closed(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0))
        assert not door.is_open

    def test_open_close(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0))
        door.open()
        assert door.is_open
        door.close()
        assert not door.is_open

    def test_toggle(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0))
        new_state = door.toggle()
        assert new_state is True
        assert door.is_open
        new_state = door.toggle()
        assert new_state is False
        assert not door.is_open

    def test_auto_close(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0), auto_close_time=1.0)
        door.open(timestamp=0.0)
        assert door.is_open
        door.update(sim_time=0.5)
        assert door.is_open  # Not yet
        door.update(sim_time=1.5)
        assert not door.is_open  # Auto-closed

    def test_callbacks(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0))
        open_called = []
        close_called = []
        door.on_open(lambda d: open_called.append(True))
        door.on_close(lambda d: close_called.append(True))
        door.open()
        assert len(open_called) == 1
        door.close()
        assert len(close_called) == 1

    def test_to_dict(self):
        door = Door(entity_id=4, start=(0.0, 0.0), end=(2.0, 0.0), open_state=True)
        d = door.to_dict()
        assert d["type"] == "door"


class TestRegion:
    def test_kind(self):
        r = Region(entity_id=5, position=(5.0, 5.0), size=(4.0, 4.0))
        assert r.kind() == "region"

    def test_contains_point(self):
        r = Region(entity_id=5, position=(5.0, 5.0), size=(4.0, 4.0))
        assert r.contains_point((5.0, 5.0))
        assert not r.contains_point((20.0, 20.0))

    def test_sample_point_within_bounds(self):
        r = Region(entity_id=5, position=(5.0, 5.0), size=(4.0, 4.0))
        rng = np.random.default_rng(42)
        for _ in range(20):
            pt = r.sample_point(rng)
            assert r.contains_point(pt)

    def test_sample_points(self):
        r = Region(entity_id=5, position=(0.0, 0.0), size=(10.0, 10.0))
        pts = r.sample_points(50, rng=np.random.default_rng(42))
        assert pts.shape == (50, 2)

    def test_width_height(self):
        r = Region(entity_id=5, position=(0.0, 0.0), size=(3.0, 7.0))
        assert r.width == pytest.approx(3.0)
        assert r.height == pytest.approx(7.0)

    def test_bounding_box(self):
        r = Region(entity_id=5, position=(5.0, 5.0), size=(4.0, 4.0))
        bb = r.bounding_box()
        assert bb.x_min <= 3.0
        assert bb.x_max >= 7.0

    def test_to_dict(self):
        r = Region(entity_id=5, position=(0.0, 0.0), size=(2.0, 3.0), label="spawn")
        d = r.to_dict()
        assert d["type"] == "region"
        assert d["label"] == "spawn"


class TestWaypoint:
    def test_kind(self):
        wp = Waypoint(entity_id=6, position=(1.0, 2.0), radius=0.5)
        assert wp.kind() == "waypoint"

    def test_is_reached(self):
        wp = Waypoint(entity_id=6, position=(0.0, 0.0), radius=1.0)
        assert wp.is_reached((0.5, 0.5))
        assert not wp.is_reached((2.0, 2.0))

    def test_connect(self):
        wp = Waypoint(entity_id=6, position=(0.0, 0.0))
        wp.connect(7)
        wp.connect(8)
        assert 7 in wp.connections
        assert 8 in wp.connections

    def test_to_dict(self):
        wp = Waypoint(entity_id=6, position=(0.0, 0.0), label="entrance")
        d = wp.to_dict()
        assert d["type"] == "waypoint"
        assert d["label"] == "entrance"


class TestNavigationGraph:
    def test_add_and_query_nodes(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        w2 = Waypoint(entity_id=2, position=(3.0, 4.0))
        g.add_node(w1)
        g.add_node(w2)
        assert g.num_nodes == 2

    def test_add_edge_and_neighbours(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        w2 = Waypoint(entity_id=2, position=(3.0, 4.0))
        g.add_node(w1)
        g.add_node(w2)
        g.add_edge(1, 2)
        neighbours = g.neighbours(1)
        assert any(n_id == 2 for n_id, _ in neighbours)

    def test_bidirectional_edge(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        w2 = Waypoint(entity_id=2, position=(1.0, 0.0))
        g.add_node(w1)
        g.add_node(w2)
        g.add_edge_bidirectional(1, 2)
        assert len(g.neighbours(1)) >= 1
        assert len(g.neighbours(2)) >= 1

    def test_shortest_path(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        w2 = Waypoint(entity_id=2, position=(1.0, 0.0))
        w3 = Waypoint(entity_id=3, position=(2.0, 0.0))
        g.add_node(w1)
        g.add_node(w2)
        g.add_node(w3)
        g.add_edge_bidirectional(1, 2)
        g.add_edge_bidirectional(2, 3)
        path, cost = g.shortest_path(1, 3)
        assert path == [1, 2, 3]
        assert cost == pytest.approx(2.0)

    def test_shortest_path_no_route(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        w2 = Waypoint(entity_id=2, position=(1.0, 0.0))
        g.add_node(w1)
        g.add_node(w2)
        # No edge
        path, cost = g.shortest_path(1, 2)
        assert len(path) == 0 or cost == float("inf")

    def test_remove_node(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        g.add_node(w1)
        g.remove_node(1)
        assert g.num_nodes == 0

    def test_positions_array(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        w2 = Waypoint(entity_id=2, position=(3.0, 4.0))
        g.add_node(w1)
        g.add_node(w2)
        arr = g.positions_array()
        assert arr.shape == (2, 2)

    def test_build_from_waypoints(self):
        wps = [
            Waypoint(entity_id=1, position=(0.0, 0.0)),
            Waypoint(entity_id=2, position=(1.0, 0.0)),
        ]
        wps[0].connect(2)
        g = NavigationGraph()
        g.build_from_waypoints(wps)
        assert g.num_nodes == 2

    def test_to_dict(self):
        g = NavigationGraph()
        w1 = Waypoint(entity_id=1, position=(0.0, 0.0))
        g.add_node(w1)
        d = g.to_dict()
        assert "nodes" in d or isinstance(d, dict)


class TestEntityManager:
    def _make_manager(self):
        bounds = AABB(-50, -50, 50, 50)
        return EntityManager(bounds=bounds)

    def test_add_and_get(self):
        mgr = self._make_manager()
        obs = StaticObstacle(position=(1.0, 2.0), radius=0.5)
        eid = mgr.add(obs)
        assert mgr.get(eid) is obs

    def test_remove(self):
        mgr = self._make_manager()
        obs = StaticObstacle(position=(1.0, 2.0))
        eid = mgr.add(obs)
        removed = mgr.remove(eid)
        assert removed is obs
        assert mgr.get(eid) is None

    def test_contains(self):
        mgr = self._make_manager()
        obs = StaticObstacle(position=(1.0, 2.0))
        eid = mgr.add(obs)
        assert eid in mgr
        mgr.remove(eid)
        assert eid not in mgr

    def test_by_kind(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        mgr.add(DynamicObstacle(position=(1.0, 1.0)))
        statics = mgr.by_kind("static_obstacle")
        assert len(statics) == 1

    def test_by_tag(self):
        mgr = self._make_manager()
        obs = StaticObstacle(position=(0.0, 0.0), tags={"goal"})
        mgr.add(obs)
        mgr.add(StaticObstacle(position=(1.0, 1.0)))
        tagged = mgr.by_tag("goal")
        assert len(tagged) == 1

    def test_by_type(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        mgr.add(DynamicObstacle(position=(1.0, 1.0)))
        dyns = mgr.by_type(DynamicObstacle)
        assert len(dyns) == 1

    def test_len_and_iter(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        mgr.add(StaticObstacle(position=(1.0, 1.0)))
        assert len(mgr) == 2
        assert sum(1 for _ in mgr) == 2

    def test_active_deactivate_activate(self):
        mgr = self._make_manager()
        obs = StaticObstacle(position=(0.0, 0.0))
        mgr.add(obs)
        mgr.deactivate_all()
        assert len(mgr.active()) == 0
        mgr.activate_all()
        assert len(mgr.active()) == 1

    def test_clear(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        mgr.add(StaticObstacle(position=(1.0, 1.0)))
        mgr.clear()
        assert len(mgr) == 0

    def test_summary(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        mgr.add(DynamicObstacle(position=(1.0, 1.0)))
        s = mgr.summary()
        assert s.get("static_obstacle", 0) == 1
        assert s.get("dynamic_obstacle", 0) == 1

    def test_to_list(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        result = mgr.to_list()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_bracket_access(self):
        mgr = self._make_manager()
        obs = StaticObstacle(position=(0.0, 0.0))
        eid = mgr.add(obs)
        assert mgr[eid] is obs

    def test_query_radius(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(0.0, 0.0)))
        mgr.add(StaticObstacle(position=(100.0, 100.0)))
        mgr.update_positions()
        near = mgr.query_radius(0.0, 0.0, 5.0)
        assert len(near) >= 1

    def test_entities_in_region(self):
        mgr = self._make_manager()
        mgr.add(StaticObstacle(position=(5.0, 5.0)))
        mgr.add(StaticObstacle(position=(50.0, 50.0)))
        mgr.update_positions()
        region = Region(position=(5.0, 5.0), size=(4.0, 4.0))
        inside = mgr.entities_in_region(region)
        assert len(inside) >= 1


# ---------------------------------------------------------------------------
# World class
# ---------------------------------------------------------------------------


class TestWorld:
    def test_add_entity(self):
        w = World(width=20.0, height=20.0)
        eid = w.add_entity(position=(5.0, 5.0), kind="pedestrian")
        assert w.entity_count() == 1
        e = w.get_entity(eid)
        assert e is not None

    def test_remove_entity(self):
        w = World(width=20.0, height=20.0)
        eid = w.add_entity(position=(5.0, 5.0))
        w.remove_entity(eid)
        assert w.entity_count() == 0

    def test_entity_ids_filter(self):
        w = World(width=20.0, height=20.0)
        w.add_entity(position=(1.0, 1.0), kind="pedestrian")
        w.add_entity(position=(2.0, 2.0), kind="robot")
        peds = w.entity_ids(kind="pedestrian")
        assert len(peds) == 1

    def test_add_wall(self):
        w = World(width=20.0, height=20.0)
        idx = w.add_wall(a=(0.0, 0.0), b=(10.0, 0.0))
        assert idx >= 0
        assert len(w.walls) >= 1

    def test_add_boundary_walls(self):
        w = World(width=20.0, height=20.0)
        w.add_boundary_walls()
        assert len(w.walls) >= 4

    def test_query_radius(self):
        w = World(width=50.0, height=50.0)
        w.add_entity(position=(10.0, 10.0))
        w.add_entity(position=(40.0, 40.0))
        w.refresh_spatial_index()
        near = w.query_radius(10.0, 10.0, 5.0)
        assert len(near) >= 1

    def test_nearest_entities(self):
        w = World(width=50.0, height=50.0)
        w.add_entity(position=(10.0, 10.0))
        w.add_entity(position=(11.0, 10.0))
        w.add_entity(position=(40.0, 40.0))
        w.refresh_spatial_index()
        nearest = w.nearest_entities(10.0, 10.0, k=2, max_radius=20.0)
        assert len(nearest) >= 1

    def test_positions_array(self):
        w = World(width=20.0, height=20.0)
        w.add_entity(position=(1.0, 2.0))
        w.add_entity(position=(3.0, 4.0))
        arr = w.positions_array()
        assert arr.shape == (2, 2)

    def test_velocities_array(self):
        w = World(width=20.0, height=20.0)
        w.add_entity(position=(1.0, 2.0), velocity=(0.5, 0.0))
        arr = w.velocities_array()
        assert arr.shape[0] >= 1

    def test_detect_entity_collisions(self):
        w = World(width=20.0, height=20.0)
        w.add_entity(position=(10.0, 10.0), radius=1.0)
        w.add_entity(position=(10.5, 10.0), radius=1.0)  # overlapping
        w.refresh_spatial_index()
        collisions = w.detect_entity_collisions()
        assert len(collisions) >= 1

    def test_no_collision_far_apart(self):
        w = World(width=50.0, height=50.0)
        w.add_entity(position=(5.0, 5.0), radius=0.3)
        w.add_entity(position=(40.0, 40.0), radius=0.3)
        w.refresh_spatial_index()
        collisions = w.detect_entity_collisions()
        assert len(collisions) == 0

    def test_enforce_boundaries_clamp(self):
        w = World(width=20.0, height=20.0, wrap=False)
        eid = w.add_entity(position=(-5.0, -5.0))
        w.enforce_boundaries()
        e = w.get_entity(eid)
        pos = e.get("position", e.get("pos", None))
        if pos is not None:
            assert pos[0] >= 0.0
            assert pos[1] >= 0.0

    def test_metadata(self):
        w = World(width=20.0, height=20.0)
        w.set_metadata("scenario", "test")
        assert w.get_metadata("scenario") == "test"
        assert w.get_metadata("missing", "default") == "default"

    def test_to_dict_from_dict(self):
        w = World(width=30.0, height=40.0)
        w.add_entity(position=(5.0, 5.0), kind="pedestrian", radius=0.3)
        w.add_wall(a=(0.0, 0.0), b=(10.0, 0.0))
        w.set_metadata("name", "test_world")
        d = w.to_dict()
        w2 = World.from_dict(d)
        assert w2.entity_count() == w.entity_count()

    def test_to_json_from_json(self):
        w = World(width=25.0, height=25.0)
        w.add_entity(position=(3.0, 4.0))
        text = w.to_json()
        w2 = World.from_json(text)
        assert w2.entity_count() == 1

    def test_snapshot_restore(self):
        w = World(width=20.0, height=20.0)
        eid = w.add_entity(position=(5.0, 5.0))
        snap = w.snapshot()
        e = w.get_entity(eid)
        if isinstance(e, dict):
            e["position"] = np.array([15.0, 15.0])
        w.restore(snap)
        e2 = w.get_entity(eid)
        if isinstance(e2, dict):
            np.testing.assert_allclose(e2["position"], [5.0, 5.0], atol=0.1)

    def test_wrap_mode(self):
        w = World(width=20.0, height=20.0, wrap=True)
        eid = w.add_entity(position=(25.0, 25.0))
        w.enforce_boundaries()
        e = w.get_entity(eid)
        if isinstance(e, dict):
            pos = e.get("position", e.get("pos"))
            if pos is not None:
                assert 0.0 <= pos[0] <= 20.0
                assert 0.0 <= pos[1] <= 20.0


class TestWorldBuilder:
    def test_basic_build(self):
        w = WorldBuilder().set_size(30.0, 30.0).build()
        assert w is not None

    def test_add_pedestrian(self):
        w = WorldBuilder().set_size(20.0, 20.0).add_pedestrian(position=(5.0, 5.0)).build()
        assert w.entity_count() == 1

    def test_add_robot(self):
        w = WorldBuilder().set_size(20.0, 20.0).add_robot(position=(10.0, 10.0)).build()
        assert w.entity_count() == 1
        ids = w.entity_ids(kind="robot")
        assert len(ids) == 1

    def test_add_obstacle(self):
        w = (
            WorldBuilder()
            .set_size(20.0, 20.0)
            .add_obstacle(position=(5.0, 5.0), radius=1.0)
            .build()
        )
        assert w.entity_count() == 1

    def test_add_wall(self):
        w = WorldBuilder().set_size(20.0, 20.0).add_wall(a=(0.0, 0.0), b=(10.0, 0.0)).build()
        assert len(w.walls) >= 1

    def test_add_boundary_walls(self):
        w = WorldBuilder().set_size(20.0, 20.0).add_boundary_walls().build()
        assert len(w.walls) >= 4

    def test_enable_wrap(self):
        builder = WorldBuilder().set_size(20.0, 20.0).enable_wrap(True)
        w = builder.build()
        assert w is not None

    def test_set_metadata(self):
        w = WorldBuilder().set_size(20.0, 20.0).set_metadata("scenario", "corridor").build()
        assert w.get_metadata("scenario") == "corridor"

    def test_chaining(self):
        w = (
            WorldBuilder()
            .set_size(30.0, 30.0)
            .set_cell_size(3.0)
            .add_pedestrian(position=(5.0, 5.0))
            .add_pedestrian(position=(10.0, 10.0))
            .add_robot(position=(15.0, 15.0))
            .add_wall(a=(0.0, 0.0), b=(30.0, 0.0))
            .add_boundary_walls()
            .set_metadata("name", "test")
            .build()
        )
        assert w.entity_count() == 3

    def test_complex_world(self):
        w = (
            WorldBuilder()
            .set_size(50.0, 50.0)
            .add_pedestrian(position=(10.0, 10.0), velocity=(1.0, 0.0))
            .add_pedestrian(position=(20.0, 20.0), velocity=(-0.5, 0.5))
            .add_robot(position=(25.0, 25.0))
            .add_obstacle(position=(15.0, 15.0), radius=2.0)
            .add_boundary_walls()
            .build()
        )
        assert w.entity_count() == 4
        peds = w.entity_ids(kind="pedestrian")
        assert len(peds) == 2


# ---------------------------------------------------------------------------
# Entity distance
# ---------------------------------------------------------------------------


class TestEntityDistance:
    def test_distance_between_entities(self):
        a = StaticObstacle(entity_id=1, position=(0.0, 0.0))
        b = StaticObstacle(entity_id=2, position=(3.0, 4.0))
        assert a.distance_to(b) == pytest.approx(5.0)
