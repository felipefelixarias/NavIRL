"""Entity system for the NavIRL simulation.

Defines a hierarchy of simulation entities – static and dynamic
obstacles, walls, doors, regions, waypoints – together with a
navigation graph and an :class:`EntityManager` supporting spatial
queries.
"""

from __future__ import annotations

import abc
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import (
    Any,
)

import numpy as np

from navirl.simulation.world import AABB

# ---------------------------------------------------------------------------
# Entity base
# ---------------------------------------------------------------------------


class Entity(abc.ABC):
    """Abstract base for every simulation entity.

    Parameters
    ----------
    entity_id : int
        Unique numeric id managed by :class:`EntityManager`.
    position : sequence of float
        2-D position ``[x, y]``.
    tags : set of str, optional
        Arbitrary string tags for filtering.
    """

    def __init__(
        self,
        entity_id: int = -1,
        position: Sequence[float] = (0.0, 0.0),
        tags: set[str] | None = None,
    ) -> None:
        self.entity_id: int = entity_id
        self.position: np.ndarray = np.asarray(position, dtype=np.float64)[:2].copy()
        self.tags: set[str] = tags or set()
        self.active: bool = True
        self._uuid: str = uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def kind(self) -> str:
        """Return a short string identifying the entity type."""

    @abc.abstractmethod
    def bounding_box(self) -> AABB:
        """Return the axis-aligned bounding box of the entity."""

    # ------------------------------------------------------------------
    def distance_to(self, other: Entity) -> float:
        """Euclidean distance to another entity."""
        return float(np.linalg.norm(self.position - other.position))

    def distance_to_point(self, point: Sequence[float]) -> float:
        """Euclidean distance to an arbitrary point."""
        return float(np.linalg.norm(self.position - np.asarray(point)[:2]))

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": self.kind(),
            "entity_id": self.entity_id,
            "position": self.position.tolist(),
            "tags": sorted(self.tags),
            "active": self.active,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.entity_id}, pos={self.position})"


# ---------------------------------------------------------------------------
# Static obstacles
# ---------------------------------------------------------------------------


class StaticObstacle(Entity):
    """Immovable circular obstacle.

    Parameters
    ----------
    radius : float
        Collision radius.
    """

    def __init__(
        self,
        entity_id: int = -1,
        position: Sequence[float] = (0.0, 0.0),
        radius: float = 0.5,
        tags: set[str] | None = None,
    ) -> None:
        super().__init__(entity_id, position, tags)
        self.radius: float = radius

    def kind(self) -> str:
        return "static_obstacle"

    def bounding_box(self) -> AABB:
        return AABB(
            self.position[0] - self.radius,
            self.position[1] - self.radius,
            self.position[0] + self.radius,
            self.position[1] + self.radius,
        )

    def contains_point(self, point: Sequence[float]) -> bool:
        """Check if *point* lies inside the obstacle."""
        return self.distance_to_point(point) <= self.radius

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["radius"] = self.radius
        return d


# ---------------------------------------------------------------------------
# Dynamic obstacles
# ---------------------------------------------------------------------------


class DynamicObstacle(Entity):
    """Movable circular obstacle with velocity.

    Parameters
    ----------
    radius : float
        Collision radius.
    velocity : sequence of float
        2-D velocity.
    mass : float
        Mass (kg).
    """

    def __init__(
        self,
        entity_id: int = -1,
        position: Sequence[float] = (0.0, 0.0),
        velocity: Sequence[float] = (0.0, 0.0),
        radius: float = 0.3,
        mass: float = 1.0,
        tags: set[str] | None = None,
    ) -> None:
        super().__init__(entity_id, position, tags)
        self.velocity: np.ndarray = np.asarray(velocity, dtype=np.float64)[:2].copy()
        self.radius: float = radius
        self.mass: float = mass
        self.orientation: float = 0.0

    def kind(self) -> str:
        return "dynamic_obstacle"

    def bounding_box(self) -> AABB:
        return AABB(
            self.position[0] - self.radius,
            self.position[1] - self.radius,
            self.position[0] + self.radius,
            self.position[1] + self.radius,
        )

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def step(self, dt: float) -> None:
        """Simple Euler integration step."""
        self.position += self.velocity * dt

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "velocity": self.velocity.tolist(),
                "radius": self.radius,
                "mass": self.mass,
            }
        )
        return d


# ---------------------------------------------------------------------------
# Wall
# ---------------------------------------------------------------------------


class Wall(Entity):
    """A line-segment wall defined by two endpoints.

    Parameters
    ----------
    start : sequence of float
        First endpoint ``[x, y]``.
    end : sequence of float
        Second endpoint ``[x, y]``.
    thickness : float
        Visual / collision thickness.
    """

    def __init__(
        self,
        entity_id: int = -1,
        start: Sequence[float] = (0.0, 0.0),
        end: Sequence[float] = (1.0, 0.0),
        thickness: float = 0.1,
        tags: set[str] | None = None,
    ) -> None:
        mid = (np.asarray(start)[:2] + np.asarray(end)[:2]) / 2.0
        super().__init__(entity_id, mid.tolist(), tags)
        self.start: np.ndarray = np.asarray(start, dtype=np.float64)[:2].copy()
        self.end: np.ndarray = np.asarray(end, dtype=np.float64)[:2].copy()
        self.thickness: float = thickness

    def kind(self) -> str:
        return "wall"

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector from start to end."""
        d = self.end - self.start
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else np.array([1.0, 0.0])

    @property
    def normal(self) -> np.ndarray:
        """Left-hand normal of the wall segment."""
        d = self.direction
        return np.array([-d[1], d[0]])

    def closest_point(self, point: Sequence[float]) -> np.ndarray:
        """Return the closest point on the segment to *point*."""
        p = np.asarray(point, dtype=np.float64)[:2]
        ab = self.end - self.start
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-12:
            return self.start.copy()
        t = float(np.dot(p - self.start, ab)) / ab_sq
        t = max(0.0, min(1.0, t))
        return self.start + t * ab

    def distance_to_point(self, point: Sequence[float]) -> float:
        """Shortest distance from the wall segment to a point."""
        cp = self.closest_point(point)
        return float(np.linalg.norm(np.asarray(point)[:2] - cp))

    def bounding_box(self) -> AABB:
        ht = self.thickness / 2.0
        return AABB(
            min(self.start[0], self.end[0]) - ht,
            min(self.start[1], self.end[1]) - ht,
            max(self.start[0], self.end[0]) + ht,
            max(self.start[1], self.end[1]) + ht,
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "start": self.start.tolist(),
                "end": self.end.tolist(),
                "thickness": self.thickness,
            }
        )
        return d


# ---------------------------------------------------------------------------
# Door
# ---------------------------------------------------------------------------


class Door(Wall):
    """A wall segment that can be opened or closed.

    When open the door is non-blocking.

    Parameters
    ----------
    open_state : bool
        Initial open/close state.
    auto_close_time : float
        Seconds after opening before auto-close (0 = never).
    """

    def __init__(
        self,
        entity_id: int = -1,
        start: Sequence[float] = (0.0, 0.0),
        end: Sequence[float] = (1.0, 0.0),
        thickness: float = 0.1,
        open_state: bool = False,
        auto_close_time: float = 0.0,
        tags: set[str] | None = None,
    ) -> None:
        super().__init__(entity_id, start, end, thickness, tags)
        self.is_open: bool = open_state
        self.auto_close_time: float = auto_close_time
        self._open_timestamp: float = 0.0
        self._on_open: list[Callable[[Door], None]] = []
        self._on_close: list[Callable[[Door], None]] = []

    def kind(self) -> str:
        return "door"

    # ------------------------------------------------------------------

    def open(self, timestamp: float = 0.0) -> None:
        """Open the door."""
        if not self.is_open:
            self.is_open = True
            self._open_timestamp = timestamp
            for cb in self._on_open:
                cb(self)

    def close(self) -> None:
        """Close the door."""
        if self.is_open:
            self.is_open = False
            for cb in self._on_close:
                cb(self)

    def toggle(self, timestamp: float = 0.0) -> bool:
        """Toggle open/close.  Returns new state."""
        if self.is_open:
            self.close()
        else:
            self.open(timestamp)
        return self.is_open

    def update(self, sim_time: float) -> None:
        """Check auto-close timer."""
        if (
            self.is_open
            and self.auto_close_time > 0.0
            and (sim_time - self._open_timestamp) >= self.auto_close_time
        ):
            self.close()

    def on_open(self, callback: Callable[[Door], None]) -> None:
        """Register callback for open events."""
        self._on_open.append(callback)

    def on_close(self, callback: Callable[[Door], None]) -> None:
        """Register callback for close events."""
        self._on_close.append(callback)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"is_open": self.is_open, "auto_close_time": self.auto_close_time})
        return d


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------


class Region(Entity):
    """Rectangular region used as spawn area, goal area, etc.

    Parameters
    ----------
    size : tuple of float
        ``(width, height)`` of the region.
    label : str
        Semantic label (e.g. ``"spawn"``, ``"goal"``).
    """

    def __init__(
        self,
        entity_id: int = -1,
        position: Sequence[float] = (0.0, 0.0),
        size: tuple[float, float] = (2.0, 2.0),
        label: str = "region",
        tags: set[str] | None = None,
    ) -> None:
        super().__init__(entity_id, position, tags)
        self.size: tuple[float, float] = size
        self.label: str = label

    def kind(self) -> str:
        return "region"

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def height(self) -> float:
        return self.size[1]

    def bounding_box(self) -> AABB:
        hw, hh = self.size[0] / 2.0, self.size[1] / 2.0
        return AABB(
            self.position[0] - hw,
            self.position[1] - hh,
            self.position[0] + hw,
            self.position[1] + hh,
        )

    def contains_point(self, point: Sequence[float]) -> bool:
        """Check whether *point* lies inside the region."""
        bb = self.bounding_box()
        p = np.asarray(point)[:2]
        return bb.contains(float(p[0]), float(p[1]))

    def sample_point(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a uniformly random point inside the region."""
        rng = rng or np.random.default_rng()
        hw, hh = self.size[0] / 2.0, self.size[1] / 2.0
        x = rng.uniform(self.position[0] - hw, self.position[0] + hw)
        y = rng.uniform(self.position[1] - hh, self.position[1] + hh)
        return np.array([x, y])

    def sample_points(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample *n* uniformly random points.  Returns (n, 2) array."""
        rng = rng or np.random.default_rng()
        hw, hh = self.size[0] / 2.0, self.size[1] / 2.0
        xs = rng.uniform(self.position[0] - hw, self.position[0] + hw, size=n)
        ys = rng.uniform(self.position[1] - hh, self.position[1] + hh, size=n)
        return np.column_stack([xs, ys])

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"size": list(self.size), "label": self.label})
        return d


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------


class Waypoint(Entity):
    """A discrete navigation point.

    Parameters
    ----------
    radius : float
        Arrival tolerance radius.
    label : str
        Semantic label.
    """

    def __init__(
        self,
        entity_id: int = -1,
        position: Sequence[float] = (0.0, 0.0),
        radius: float = 0.5,
        label: str = "",
        tags: set[str] | None = None,
    ) -> None:
        super().__init__(entity_id, position, tags)
        self.radius: float = radius
        self.label: str = label
        self.connections: list[int] = []  # ids of connected waypoints

    def kind(self) -> str:
        return "waypoint"

    def bounding_box(self) -> AABB:
        return AABB(
            self.position[0] - self.radius,
            self.position[1] - self.radius,
            self.position[0] + self.radius,
            self.position[1] + self.radius,
        )

    def is_reached(self, point: Sequence[float]) -> bool:
        """Check whether *point* is within the arrival radius."""
        return self.distance_to_point(point) <= self.radius

    def connect(self, other_id: int) -> None:
        """Add a one-way connection to another waypoint."""
        if other_id not in self.connections:
            self.connections.append(other_id)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "radius": self.radius,
                "label": self.label,
                "connections": self.connections,
            }
        )
        return d


# ---------------------------------------------------------------------------
# NavigationGraph
# ---------------------------------------------------------------------------


class NavigationGraph:
    """Graph structure over :class:`Waypoint` entities.

    Supports Dijkstra shortest-path queries and construction from a set
    of waypoints.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, Waypoint] = {}
        self._edges: dict[int, list[tuple[int, float]]] = {}  # node_id -> [(nbr, cost)]

    # ------------------------------------------------------------------

    def add_node(self, waypoint: Waypoint) -> None:
        """Add a waypoint as a graph node."""
        self._nodes[waypoint.entity_id] = waypoint
        self._edges.setdefault(waypoint.entity_id, [])

    def add_edge(self, from_id: int, to_id: int, cost: float | None = None) -> None:
        """Add a directed edge.  Cost defaults to Euclidean distance."""
        if cost is None:
            a = self._nodes[from_id]
            b = self._nodes[to_id]
            cost = a.distance_to(b)
        self._edges.setdefault(from_id, []).append((to_id, cost))

    def add_edge_bidirectional(self, a_id: int, b_id: int, cost: float | None = None) -> None:
        """Add an undirected edge (two directed edges)."""
        if cost is None:
            a = self._nodes[a_id]
            b = self._nodes[b_id]
            cost = a.distance_to(b)
        self.add_edge(a_id, b_id, cost)
        self.add_edge(b_id, a_id, cost)

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its edges."""
        self._nodes.pop(node_id, None)
        self._edges.pop(node_id, None)
        for nid in list(self._edges):
            self._edges[nid] = [(t, c) for t, c in self._edges[nid] if t != node_id]

    # ------------------------------------------------------------------

    def neighbours(self, node_id: int) -> list[tuple[int, float]]:
        """Return list of ``(neighbour_id, cost)``."""
        return list(self._edges.get(node_id, []))

    @property
    def node_ids(self) -> list[int]:
        return list(self._nodes)

    def get_node(self, node_id: int) -> Waypoint:
        return self._nodes[node_id]

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._edges.values())

    # ------------------------------------------------------------------
    # Dijkstra
    # ------------------------------------------------------------------

    def shortest_path(self, start_id: int, goal_id: int) -> tuple[list[int], float]:
        """Dijkstra shortest path.

        Returns ``(path, cost)`` where *path* is a list of node ids
        from *start_id* to *goal_id* inclusive.  If no path exists
        returns ``([], inf)``.
        """
        import heapq

        dist: dict[int, float] = {start_id: 0.0}
        prev: dict[int, int] = {}
        visited: set[int] = set()
        heap: list[tuple[float, int]] = [(0.0, start_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == goal_id:
                break
            for v, w in self._edges.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if goal_id not in dist:
            return [], float("inf")

        path: list[int] = []
        cur = goal_id
        while cur != start_id:
            path.append(cur)
            cur = prev[cur]
        path.append(start_id)
        path.reverse()
        return path, dist[goal_id]

    # ------------------------------------------------------------------

    def build_from_waypoints(self, waypoints: Sequence[Waypoint]) -> None:
        """Populate graph from a list of waypoints using their connections."""
        for wp in waypoints:
            self.add_node(wp)
        for wp in waypoints:
            for conn_id in wp.connections:
                if conn_id in self._nodes:
                    self.add_edge(wp.entity_id, conn_id)

    def positions_array(self) -> np.ndarray:
        """Return (N, 2) array of node positions."""
        if not self._nodes:
            return np.empty((0, 2))
        return np.array([n.position for n in self._nodes.values()])

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: wp.to_dict() for nid, wp in self._nodes.items()},
            "edges": {str(k): v for k, v in self._edges.items()},
        }

    def __repr__(self) -> str:
        return f"NavigationGraph(nodes={self.num_nodes}, edges={self.num_edges})"


# ---------------------------------------------------------------------------
# EntityManager
# ---------------------------------------------------------------------------


class EntityManager:
    """Registry and spatial index for :class:`Entity` instances.

    Uses a simple grid-based spatial index for range and nearest-entity
    queries.

    Parameters
    ----------
    bounds : AABB
        World bounds.
    cell_size : float
        Spatial index cell size.
    """

    def __init__(
        self,
        bounds: AABB | None = None,
        cell_size: float = 2.0,
    ) -> None:
        self._entities: dict[int, Entity] = {}
        self._next_id: int = 0
        self._bounds = bounds if bounds is not None else AABB(0, 0, 50, 50)
        self._cell_size = cell_size
        # Grid: (row, col) -> set of entity ids
        self._grid: dict[tuple[int, int], set[int]] = {}
        self._cols = max(1, int(np.ceil(self._bounds.width / cell_size)))
        self._rows = max(1, int(np.ceil(self._bounds.height / cell_size)))

    # ------------------------------------------------------------------
    # ID management
    # ------------------------------------------------------------------

    def _allocate_id(self) -> int:
        eid = self._next_id
        self._next_id += 1
        return eid

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self._bounds.x_min) / self._cell_size)
        r = int((y - self._bounds.y_min) / self._cell_size)
        c = max(0, min(c, self._cols - 1))
        r = max(0, min(r, self._rows - 1))
        return r, c

    def _insert_grid(self, eid: int, bb: AABB) -> None:
        r0, c0 = self._cell_of(bb.x_min, bb.y_min)
        r1, c1 = self._cell_of(bb.x_max, bb.y_max)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                self._grid.setdefault((r, c), set()).add(eid)

    def _remove_grid(self, eid: int) -> None:
        for cell_set in self._grid.values():
            cell_set.discard(eid)

    def _rebuild_grid(self) -> None:
        self._grid.clear()
        for eid, ent in self._entities.items():
            self._insert_grid(eid, ent.bounding_box())

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entity: Entity) -> int:
        """Add an entity, assign an id, index it.  Returns the id."""
        eid = self._allocate_id()
        entity.entity_id = eid
        self._entities[eid] = entity
        self._insert_grid(eid, entity.bounding_box())
        return eid

    def remove(self, entity_id: int) -> Entity | None:
        """Remove and return the entity, or ``None`` if not found."""
        ent = self._entities.pop(entity_id, None)
        if ent is not None:
            self._remove_grid(entity_id)
        return ent

    def get(self, entity_id: int) -> Entity | None:
        return self._entities.get(entity_id)

    def __getitem__(self, entity_id: int) -> Entity:
        return self._entities[entity_id]

    def __contains__(self, entity_id: int) -> bool:
        return entity_id in self._entities

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self) -> Iterator[Entity]:
        return iter(self._entities.values())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def by_kind(self, kind: str) -> list[Entity]:
        """Return all entities of a given kind."""
        return [e for e in self._entities.values() if e.kind() == kind]

    def by_tag(self, tag: str) -> list[Entity]:
        """Return all entities that carry *tag*."""
        return [e for e in self._entities.values() if tag in e.tags]

    def by_type(self, cls: type[Entity]) -> list[Entity]:
        """Return all entities that are instances of *cls*."""
        return [e for e in self._entities.values() if isinstance(e, cls)]

    def active(self) -> list[Entity]:
        """Return all active entities."""
        return [e for e in self._entities.values() if e.active]

    def ids(self) -> list[int]:
        return list(self._entities)

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def query_radius(self, x: float, y: float, radius: float) -> list[Entity]:
        """Return entities within *radius* of point (x, y)."""
        r0, c0 = self._cell_of(x - radius, y - radius)
        r1, c1 = self._cell_of(x + radius, y + radius)
        candidates: set[int] = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                candidates.update(self._grid.get((r, c), set()))
        centre = np.array([x, y])
        return [
            self._entities[eid]
            for eid in candidates
            if eid in self._entities
            and float(np.linalg.norm(self._entities[eid].position - centre)) <= radius
        ]

    def query_aabb(self, aabb: AABB) -> list[Entity]:
        """Return entities overlapping *aabb*."""
        r0, c0 = self._cell_of(aabb.x_min, aabb.y_min)
        r1, c1 = self._cell_of(aabb.x_max, aabb.y_max)
        candidates: set[int] = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                candidates.update(self._grid.get((r, c), set()))
        return [
            self._entities[eid]
            for eid in candidates
            if eid in self._entities and self._entities[eid].bounding_box().overlaps(aabb)
        ]

    def nearest(
        self, x: float, y: float, k: int = 1, max_radius: float = 10.0
    ) -> list[tuple[Entity, float]]:
        """Return up to *k* nearest entities within *max_radius*.

        Returns list of ``(entity, distance)`` sorted by distance.
        """
        ents = self.query_radius(x, y, max_radius)
        centre = np.array([x, y])
        dists = [(e, float(np.linalg.norm(e.position - centre))) for e in ents]
        dists.sort(key=lambda t: t[1])
        return dists[:k]

    def entities_in_region(self, region: Region) -> list[Entity]:
        """Return entities whose position lies inside *region*."""
        return [e for e in self._entities.values() if region.contains_point(e.position)]

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def update_positions(self) -> None:
        """Re-index all entities (call after positions change)."""
        self._rebuild_grid()

    def deactivate_all(self) -> None:
        """Set every entity to inactive."""
        for e in self._entities.values():
            e.active = False

    def activate_all(self) -> None:
        """Set every entity to active."""
        for e in self._entities.values():
            e.active = True

    def clear(self) -> None:
        """Remove all entities."""
        self._entities.clear()
        self._grid.clear()
        self._next_id = 0

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all entities to a list of dicts."""
        return [e.to_dict() for e in self._entities.values()]

    def summary(self) -> dict[str, int]:
        """Return counts per entity kind."""
        counts: dict[str, int] = {}
        for e in self._entities.values():
            k = e.kind()
            counts[k] = counts.get(k, 0) + 1
        return counts

    def __repr__(self) -> str:
        return f"EntityManager(entities={len(self._entities)})"
