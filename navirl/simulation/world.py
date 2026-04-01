"""World management for the NavIRL simulation.

Maintains the full simulation state including all entities, spatial
indexing structures, collision detection, and boundary conditions.
Provides serialization / deserialization helpers and a fluent
``WorldBuilder`` for ergonomic world construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Axis-Aligned Bounding Box
# ---------------------------------------------------------------------------


@dataclass
class AABB:
    """Axis-aligned bounding box used for broad-phase collision queries."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # ------------------------------------------------------------------
    def contains(self, x: float, y: float) -> bool:
        """Return ``True`` if point (x, y) lies within the box (inclusive)."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def overlaps(self, other: AABB) -> bool:
        """Return ``True`` if this AABB overlaps *other*."""
        return not (
            self.x_max < other.x_min
            or other.x_max < self.x_min
            or self.y_max < other.y_min
            or other.y_max < self.y_min
        )

    def expand(self, margin: float) -> AABB:
        """Return a new AABB expanded by *margin* on every side."""
        return AABB(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin,
        )

    @property
    def center(self) -> np.ndarray:
        """Return center as a 2-element array."""
        return np.array([0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max)])

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height


# ---------------------------------------------------------------------------
# Spatial Grid
# ---------------------------------------------------------------------------


class SpatialGrid:
    """Uniform spatial grid for fast neighbour and range queries.

    Parameters
    ----------
    bounds : AABB
        World bounds to partition.
    cell_size : float
        Side length of each grid cell.
    """

    def __init__(self, bounds: AABB, cell_size: float = 2.0) -> None:
        self.bounds = bounds
        self.cell_size = max(cell_size, 0.01)
        self.cols = max(1, int(np.ceil(bounds.width / self.cell_size)))
        self.rows = max(1, int(np.ceil(bounds.height / self.cell_size)))
        # Map (row, col) -> set of entity ids
        self._cells: dict[tuple[int, int], set[int]] = {}
        # Reverse map entity_id -> set of (row, col)
        self._entity_cells: dict[int, set[tuple[int, int]]] = {}

    # ------------------------------------------------------------------
    def _cell_index(self, x: float, y: float) -> tuple[int, int]:
        """Return (row, col) cell index for a world coordinate."""
        col = int((x - self.bounds.x_min) / self.cell_size)
        row = int((y - self.bounds.y_min) / self.cell_size)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return row, col

    def _cells_for_aabb(self, aabb: AABB) -> Iterator[tuple[int, int]]:
        """Yield all cell indices that intersect *aabb*."""
        r_min, c_min = self._cell_index(aabb.x_min, aabb.y_min)
        r_max, c_max = self._cell_index(aabb.x_max, aabb.y_max)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                yield r, c

    # ------------------------------------------------------------------
    def insert(self, entity_id: int, aabb: AABB) -> None:
        """Insert *entity_id* into all cells covered by *aabb*."""
        cells: set[tuple[int, int]] = set()
        for rc in self._cells_for_aabb(aabb):
            self._cells.setdefault(rc, set()).add(entity_id)
            cells.add(rc)
        self._entity_cells[entity_id] = cells

    def remove(self, entity_id: int) -> None:
        """Remove *entity_id* from the grid."""
        for rc in self._entity_cells.pop(entity_id, set()):
            s = self._cells.get(rc)
            if s is not None:
                s.discard(entity_id)
                if not s:
                    del self._cells[rc]

    def update(self, entity_id: int, aabb: AABB) -> None:
        """Re-index *entity_id* with a new bounding box."""
        self.remove(entity_id)
        self.insert(entity_id, aabb)

    def query_point(self, x: float, y: float) -> set[int]:
        """Return entity ids whose cells contain point (x, y)."""
        rc = self._cell_index(x, y)
        return set(self._cells.get(rc, set()))

    def query_aabb(self, aabb: AABB) -> set[int]:
        """Return entity ids in any cell overlapping *aabb*."""
        result: set[int] = set()
        for rc in self._cells_for_aabb(aabb):
            result.update(self._cells.get(rc, set()))
        return result

    def query_radius(self, x: float, y: float, radius: float) -> set[int]:
        """Return entity ids in cells overlapping a circle."""
        aabb = AABB(x - radius, y - radius, x + radius, y + radius)
        return self.query_aabb(aabb)

    def clear(self) -> None:
        """Remove all entries."""
        self._cells.clear()
        self._entity_cells.clear()


# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------


def _circle_circle(
    p1: np.ndarray, r1: float, p2: np.ndarray, r2: float
) -> tuple[np.ndarray, float] | None:
    """Return (normal, penetration) if circles overlap, else ``None``."""
    diff = p2 - p1
    dist = float(np.linalg.norm(diff))
    overlap = r1 + r2 - dist
    if overlap <= 0.0:
        return None
    normal = diff / dist if dist > 1e-12 else np.array([1.0, 0.0])
    return normal, overlap


def _circle_segment(
    center: np.ndarray, radius: float, seg_a: np.ndarray, seg_b: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Return (normal, penetration) if circle overlaps a line segment."""
    ab = seg_b - seg_a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return _circle_circle(center, radius, seg_a, 0.0)
    t = float(np.dot(center - seg_a, ab)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = seg_a + t * ab
    diff = center - closest
    dist = float(np.linalg.norm(diff))
    overlap = radius - dist
    if overlap <= 0.0:
        return None
    normal = diff / dist if dist > 1e-12 else np.array([0.0, 1.0])
    return normal, overlap


def _aabb_overlap(a: AABB, b: AABB) -> tuple[np.ndarray, float] | None:
    """Compute AABB overlap, return (mtv_direction, penetration) or None."""
    dx = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    dy = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if dx <= 0 or dy <= 0:
        return None
    if dx < dy:
        sign = 1.0 if a.center[0] < b.center[0] else -1.0
        return np.array([sign, 0.0]), dx
    sign = 1.0 if a.center[1] < b.center[1] else -1.0
    return np.array([0.0, sign]), dy


# ---------------------------------------------------------------------------
# Collision Result
# ---------------------------------------------------------------------------


@dataclass
class CollisionResult:
    """Record of a single collision between two entities."""

    entity_a_id: int
    entity_b_id: int
    normal: np.ndarray
    penetration: float
    contact_point: np.ndarray


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


class World:
    """Central simulation world managing entities and spatial structure.

    Parameters
    ----------
    width : float
        World width along the X-axis.
    height : float
        World height along the Y-axis.
    cell_size : float
        Spatial-grid cell size for indexing.
    wrap : bool
        If ``True`` entities wrap around boundaries (toroidal).  Otherwise
        entities are clamped inside the boundary.
    """

    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        cell_size: float = 2.0,
        wrap: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.wrap = wrap
        self.bounds = AABB(0.0, 0.0, width, height)
        self._grid = SpatialGrid(self.bounds, cell_size=cell_size)

        # entity_id -> dict with keys: position, velocity, radius, kind, data
        self._entities: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0

        # Wall segments stored as (a, b) pairs of 2-vectors
        self._walls: list[tuple[np.ndarray, np.ndarray]] = []

        # Metadata
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def add_entity(
        self,
        position: Sequence[float],
        velocity: Sequence[float] | None = None,
        radius: float = 0.3,
        kind: str = "pedestrian",
        mass: float = 1.0,
        **extra: Any,
    ) -> int:
        """Add an entity to the world.  Returns the new entity id."""
        eid = self._next_id
        self._next_id += 1
        pos = np.asarray(position, dtype=np.float64)[:2].copy()
        vel = np.zeros(2, dtype=np.float64)
        if velocity is not None:
            vel = np.asarray(velocity, dtype=np.float64)[:2].copy()
        entry: dict[str, Any] = {
            "position": pos,
            "velocity": vel,
            "radius": float(radius),
            "kind": kind,
            "mass": float(mass),
            "active": True,
        }
        entry.update(extra)
        self._entities[eid] = entry
        aabb = self._entity_aabb(eid)
        self._grid.insert(eid, aabb)
        return eid

    def remove_entity(self, entity_id: int) -> None:
        """Remove entity by id.  No-op if not present."""
        if entity_id in self._entities:
            self._grid.remove(entity_id)
            del self._entities[entity_id]

    def get_entity(self, entity_id: int) -> dict[str, Any]:
        """Return the mutable entity dict for *entity_id*.

        Raises ``KeyError`` if not found.
        """
        return self._entities[entity_id]

    def entity_ids(self, kind: str | None = None) -> list[int]:
        """Return list of entity ids, optionally filtered by *kind*."""
        if kind is None:
            return list(self._entities)
        return [eid for eid, e in self._entities.items() if e["kind"] == kind]

    def entity_count(self) -> int:
        """Return total number of entities."""
        return len(self._entities)

    @property
    def entities(self) -> dict[int, dict[str, Any]]:
        """Direct (mutable) access to the entity store."""
        return self._entities

    # ------------------------------------------------------------------
    # Wall management
    # ------------------------------------------------------------------

    def add_wall(self, a: Sequence[float], b: Sequence[float]) -> int:
        """Add a wall segment.  Returns the wall index."""
        idx = len(self._walls)
        self._walls.append(
            (np.asarray(a, dtype=np.float64)[:2].copy(), np.asarray(b, dtype=np.float64)[:2].copy())
        )
        return idx

    def add_boundary_walls(self) -> None:
        """Add four wall segments along the world boundary."""
        corners = [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height),
        ]
        for i in range(4):
            self.add_wall(corners[i], corners[(i + 1) % 4])

    @property
    def walls(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self._walls

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def _entity_aabb(self, eid: int) -> AABB:
        e = self._entities[eid]
        p = e["position"]
        r = e["radius"]
        return AABB(p[0] - r, p[1] - r, p[0] + r, p[1] + r)

    def query_radius(self, x: float, y: float, radius: float) -> list[int]:
        """Return entity ids whose *cells* intersect the query circle.

        For exact distance filtering, post-filter by Euclidean distance.
        """
        candidates = self._grid.query_radius(x, y, radius)
        result: list[int] = []
        centre = np.array([x, y])
        for eid in candidates:
            e = self._entities[eid]
            d = float(np.linalg.norm(e["position"] - centre)) - e["radius"]
            if d <= radius:
                result.append(eid)
        return result

    def query_aabb(self, aabb: AABB) -> list[int]:
        """Return entity ids overlapping *aabb*."""
        return list(self._grid.query_aabb(aabb))

    def nearest_entities(
        self, x: float, y: float, k: int = 5, max_radius: float = 10.0
    ) -> list[tuple[int, float]]:
        """Return up to *k* nearest entities within *max_radius*.

        Returns list of (entity_id, distance) sorted by distance.
        """
        candidates = self.query_radius(x, y, max_radius)
        centre = np.array([x, y])
        dists = []
        for eid in candidates:
            d = float(np.linalg.norm(self._entities[eid]["position"] - centre))
            dists.append((eid, d))
        dists.sort(key=lambda t: t[1])
        return dists[:k]

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def detect_entity_collisions(self) -> list[CollisionResult]:
        """Broad + narrow phase entity-entity collision detection."""
        checked: set[tuple[int, int]] = set()
        results: list[CollisionResult] = []
        for eid, edata in self._entities.items():
            if not edata.get("active", True):
                continue
            neighbours = self._grid.query_radius(
                edata["position"][0], edata["position"][1], edata["radius"] * 3.0
            )
            for nid in neighbours:
                if nid == eid:
                    continue
                pair = (min(eid, nid), max(eid, nid))
                if pair in checked:
                    continue
                checked.add(pair)
                ndata = self._entities[nid]
                if not ndata.get("active", True):
                    continue
                hit = _circle_circle(
                    edata["position"],
                    edata["radius"],
                    ndata["position"],
                    ndata["radius"],
                )
                if hit is not None:
                    normal, pen = hit
                    cp = edata["position"] + normal * (edata["radius"] - pen * 0.5)
                    results.append(CollisionResult(eid, nid, normal, pen, cp))
        return results

    def detect_wall_collisions(self) -> list[CollisionResult]:
        """Check every entity against every wall segment."""
        results: list[CollisionResult] = []
        for eid, edata in self._entities.items():
            if not edata.get("active", True):
                continue
            for widx, (wa, wb) in enumerate(self._walls):
                hit = _circle_segment(edata["position"], edata["radius"], wa, wb)
                if hit is not None:
                    normal, pen = hit
                    cp = edata["position"] - normal * (edata["radius"] - pen * 0.5)
                    results.append(CollisionResult(eid, -(widx + 1), normal, pen, cp))
        return results

    # ------------------------------------------------------------------
    # Boundary enforcement
    # ------------------------------------------------------------------

    def enforce_boundaries(self) -> None:
        """Clamp or wrap entity positions to stay inside the world bounds."""
        for _eid, edata in self._entities.items():
            pos = edata["position"]
            r = edata["radius"]
            if self.wrap:
                pos[0] = pos[0] % self.width
                pos[1] = pos[1] % self.height
            else:
                pos[0] = np.clip(pos[0], r, self.width - r)
                pos[1] = np.clip(pos[1], r, self.height - r)

    # ------------------------------------------------------------------
    # Update spatial index
    # ------------------------------------------------------------------

    def refresh_spatial_index(self) -> None:
        """Rebuild the spatial grid from current entity positions."""
        self._grid.clear()
        for eid in self._entities:
            self._grid.insert(eid, self._entity_aabb(eid))

    # ------------------------------------------------------------------
    # Positions / velocities as arrays
    # ------------------------------------------------------------------

    def positions_array(self, kind: str | None = None) -> np.ndarray:
        """Return (N, 2) positions array."""
        ids = self.entity_ids(kind)
        if not ids:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([self._entities[i]["position"] for i in ids])

    def velocities_array(self, kind: str | None = None) -> np.ndarray:
        """Return (N, 2) velocities array."""
        ids = self.entity_ids(kind)
        if not ids:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([self._entities[i]["velocity"] for i in ids])

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    # ------------------------------------------------------------------
    # Serialization / deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the world to a JSON-compatible dict."""
        ents = {}
        for eid, edata in self._entities.items():
            d = dict(edata)
            d["position"] = d["position"].tolist()
            d["velocity"] = d["velocity"].tolist()
            ents[str(eid)] = d
        walls_ser = [[a.tolist(), b.tolist()] for a, b in self._walls]
        return {
            "width": self.width,
            "height": self.height,
            "wrap": self.wrap,
            "entities": ents,
            "walls": walls_ser,
            "next_id": self._next_id,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], cell_size: float = 2.0) -> World:
        """Reconstruct a world from a dict produced by ``to_dict``."""
        w = cls(
            width=data["width"],
            height=data["height"],
            cell_size=cell_size,
            wrap=data.get("wrap", False),
        )
        for eid_str, edata in data.get("entities", {}).items():
            edata = dict(edata)
            edata["position"] = np.asarray(edata["position"], dtype=np.float64)
            edata["velocity"] = np.asarray(edata["velocity"], dtype=np.float64)
            eid = int(eid_str)
            w._entities[eid] = edata
            w._grid.insert(eid, w._entity_aabb(eid))
        w._next_id = data.get("next_id", 0)
        for seg in data.get("walls", []):
            w._walls.append(
                (np.asarray(seg[0], dtype=np.float64), np.asarray(seg[1], dtype=np.float64))
            )
        w._metadata = data.get("metadata", {})
        return w

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str, cell_size: float = 2.0) -> World:
        """Reconstruct from a JSON string."""
        return cls.from_dict(json.loads(text), cell_size=cell_size)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Create a deep-copy snapshot of entity state (fast restore)."""
        snap: dict[str, Any] = {}
        for eid, edata in self._entities.items():
            snap[eid] = {
                "position": edata["position"].copy(),
                "velocity": edata["velocity"].copy(),
                "active": edata.get("active", True),
            }
        return snap

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore entity positions/velocities from a snapshot."""
        for eid, sdata in snap.items():
            if eid in self._entities:
                self._entities[eid]["position"][:] = sdata["position"]
                self._entities[eid]["velocity"][:] = sdata["velocity"]
                self._entities[eid]["active"] = sdata["active"]
        self.refresh_spatial_index()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entities)

    def __contains__(self, entity_id: int) -> bool:
        return entity_id in self._entities

    def __repr__(self) -> str:
        return (
            f"World(width={self.width}, height={self.height}, "
            f"entities={len(self._entities)}, walls={len(self._walls)})"
        )


# ---------------------------------------------------------------------------
# WorldBuilder (fluent API)
# ---------------------------------------------------------------------------


class WorldBuilder:
    """Fluent builder for constructing :class:`World` instances.

    Example::

        world = (
            WorldBuilder()
            .set_size(40, 30)
            .enable_wrap(False)
            .add_boundary_walls()
            .add_pedestrian([5, 5])
            .add_pedestrian([10, 10], velocity=[1, 0])
            .add_robot([20, 15])
            .add_obstacle([25, 25], radius=1.0)
            .build()
        )
    """

    def __init__(self) -> None:
        self._width: float = 50.0
        self._height: float = 50.0
        self._cell_size: float = 2.0
        self._wrap: bool = False
        self._boundary_walls: bool = False
        self._entities: list[dict[str, Any]] = []
        self._walls: list[tuple[Sequence[float], Sequence[float]]] = []
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------

    def set_size(self, width: float, height: float) -> WorldBuilder:
        """Set world dimensions."""
        self._width = width
        self._height = height
        return self

    def set_cell_size(self, cell_size: float) -> WorldBuilder:
        """Set spatial grid cell size."""
        self._cell_size = cell_size
        return self

    def enable_wrap(self, wrap: bool = True) -> WorldBuilder:
        """Enable or disable toroidal wrapping."""
        self._wrap = wrap
        return self

    def add_boundary_walls(self) -> WorldBuilder:
        """Add walls along the world perimeter."""
        self._boundary_walls = True
        return self

    def add_wall(self, a: Sequence[float], b: Sequence[float]) -> WorldBuilder:
        """Add a wall segment."""
        self._walls.append((a, b))
        return self

    def add_pedestrian(
        self,
        position: Sequence[float],
        velocity: Sequence[float] | None = None,
        radius: float = 0.3,
        mass: float = 1.0,
        **extra: Any,
    ) -> WorldBuilder:
        """Queue a pedestrian entity."""
        self._entities.append(
            dict(
                position=position,
                velocity=velocity,
                radius=radius,
                kind="pedestrian",
                mass=mass,
                **extra,
            )
        )
        return self

    def add_robot(
        self,
        position: Sequence[float],
        velocity: Sequence[float] | None = None,
        radius: float = 0.35,
        mass: float = 5.0,
        **extra: Any,
    ) -> WorldBuilder:
        """Queue a robot entity."""
        self._entities.append(
            dict(
                position=position,
                velocity=velocity,
                radius=radius,
                kind="robot",
                mass=mass,
                **extra,
            )
        )
        return self

    def add_obstacle(
        self,
        position: Sequence[float],
        radius: float = 0.5,
        **extra: Any,
    ) -> WorldBuilder:
        """Queue a static obstacle entity."""
        self._entities.append(
            dict(
                position=position, velocity=None, radius=radius, kind="obstacle", mass=1e6, **extra
            )
        )
        return self

    def add_entity(self, **kwargs: Any) -> WorldBuilder:
        """Queue a generic entity from keyword arguments."""
        self._entities.append(kwargs)
        return self

    def set_metadata(self, key: str, value: Any) -> WorldBuilder:
        """Attach arbitrary metadata to the world."""
        self._metadata[key] = value
        return self

    # ------------------------------------------------------------------

    def build(self) -> World:
        """Construct and return the :class:`World`."""
        world = World(
            width=self._width,
            height=self._height,
            cell_size=self._cell_size,
            wrap=self._wrap,
        )
        if self._boundary_walls:
            world.add_boundary_walls()
        for seg in self._walls:
            world.add_wall(*seg)
        for ent in self._entities:
            world.add_entity(**ent)
        for k, v in self._metadata.items():
            world.set_metadata(k, v)
        return world

    def __repr__(self) -> str:
        return (
            f"WorldBuilder(size=({self._width}, {self._height}), "
            f"queued_entities={len(self._entities)})"
        )
