"""Obstacle representations for the continuous-space backend.

Provides geometric obstacle primitives used for collision detection
and path planning in continuous 2-D environments.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Obstacle(ABC):
    """Abstract base class for 2-D obstacles.

    All obstacles must implement collision checking methods including
    point containment, circle intersection, and ray casting.
    """

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside or on the obstacle.

        Parameters
        ----------
        point : np.ndarray
            2-D point, shape (2,).

        Returns
        -------
        bool
            True if the point is inside the obstacle.
        """

    @abstractmethod
    def distance_to_point(self, point: np.ndarray) -> float:
        """Compute the minimum distance from a point to the obstacle boundary.

        Parameters
        ----------
        point : np.ndarray
            2-D point, shape (2,).

        Returns
        -------
        float
            Distance to the nearest point on the obstacle boundary.
            Negative if the point is inside.
        """

    @abstractmethod
    def intersects_circle(self, center: np.ndarray, radius: float) -> bool:
        """Check if a circle intersects the obstacle.

        Parameters
        ----------
        center : np.ndarray
            Circle center, shape (2,).
        radius : float
            Circle radius.

        Returns
        -------
        bool
            True if the circle intersects the obstacle.
        """

    @abstractmethod
    def ray_cast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float | None:
        """Cast a ray and find the first intersection distance.

        Parameters
        ----------
        origin : np.ndarray
            Ray origin, shape (2,).
        direction : np.ndarray
            Ray direction (unit vector), shape (2,).

        Returns
        -------
        float or None
            Distance to the first intersection, or None if no hit.
        """

    @abstractmethod
    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """Find the closest point on the obstacle boundary.

        Parameters
        ----------
        point : np.ndarray
            Query point, shape (2,).

        Returns
        -------
        np.ndarray
            Closest point on the boundary, shape (2,).
        """

    @abstractmethod
    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Compute the outward normal at a point on/near the boundary.

        Parameters
        ----------
        point : np.ndarray
            Point on or near the boundary, shape (2,).

        Returns
        -------
        np.ndarray
            Outward normal vector (unit), shape (2,).
        """

    @abstractmethod
    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box.

        Returns
        -------
        tuple of np.ndarray
            (min_corner, max_corner), each shape (2,).
        """

    @abstractmethod
    def get_vertices(self) -> np.ndarray | None:
        """Get vertices for polygonal representation.

        Returns
        -------
        np.ndarray or None
            Vertices shape (N, 2), or None for non-polygonal obstacles.
        """

    def inflate(self, margin: float) -> Obstacle:
        """Create an inflated copy of the obstacle for C-space expansion.

        Parameters
        ----------
        margin : float
            Inflation margin.

        Returns
        -------
        Obstacle
            Inflated obstacle.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support inflation"
        )


@dataclass
class CircleObstacle(Obstacle):
    """Circular obstacle.

    Parameters
    ----------
    center : np.ndarray
        Center position, shape (2,).
    radius : float
        Circle radius.
    name : str
        Optional name identifier.
    """

    center: np.ndarray
    radius: float
    name: str = ""

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)

    def contains_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=np.float64)
        return float(np.linalg.norm(point - self.center)) <= self.radius

    def distance_to_point(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float64)
        return float(np.linalg.norm(point - self.center)) - self.radius

    def intersects_circle(self, center: np.ndarray, radius: float) -> bool:
        center = np.asarray(center, dtype=np.float64)
        dist = float(np.linalg.norm(center - self.center))
        return dist <= self.radius + radius

    def ray_cast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float | None:
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        oc = origin - self.center
        a = float(np.dot(direction, direction))
        b = 2.0 * float(np.dot(oc, direction))
        c = float(np.dot(oc, oc)) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0 or abs(a) < 1e-12:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        diff = point - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            return self.center + np.array([self.radius, 0.0])
        return self.center + diff / dist * self.radius

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        diff = point - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            return np.array([1.0, 0.0])
        return diff / dist

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.array([self.radius, self.radius])
        return self.center - r, self.center + r

    def get_vertices(self) -> np.ndarray | None:
        # Approximate circle with polygon
        n = max(16, int(self.radius * 8))
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return self.center + self.radius * np.column_stack([
            np.cos(angles), np.sin(angles),
        ])

    def inflate(self, margin: float) -> CircleObstacle:
        return CircleObstacle(
            center=self.center.copy(),
            radius=self.radius + margin,
            name=self.name,
        )


@dataclass
class RectangleObstacle(Obstacle):
    """Axis-aligned rectangular obstacle.

    Parameters
    ----------
    min_corner : np.ndarray
        Lower-left corner, shape (2,).
    max_corner : np.ndarray
        Upper-right corner, shape (2,).
    name : str
        Optional name identifier.
    """

    min_corner: np.ndarray
    max_corner: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.min_corner = np.asarray(self.min_corner, dtype=np.float64)
        self.max_corner = np.asarray(self.max_corner, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        """Center of the rectangle."""
        return 0.5 * (self.min_corner + self.max_corner)

    @property
    def width(self) -> float:
        """Width (x extent)."""
        return float(self.max_corner[0] - self.min_corner[0])

    @property
    def height(self) -> float:
        """Height (y extent)."""
        return float(self.max_corner[1] - self.min_corner[1])

    def contains_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=np.float64)
        return bool(
            self.min_corner[0] <= point[0] <= self.max_corner[0]
            and self.min_corner[1] <= point[1] <= self.max_corner[1]
        )

    def distance_to_point(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float64)
        dx = max(self.min_corner[0] - point[0], 0.0, point[0] - self.max_corner[0])
        dy = max(self.min_corner[1] - point[1], 0.0, point[1] - self.max_corner[1])
        outside_dist = math.sqrt(dx * dx + dy * dy)

        if outside_dist > 0:
            return outside_dist

        # Inside: return negative distance to nearest edge
        distances = [
            point[0] - self.min_corner[0],
            self.max_corner[0] - point[0],
            point[1] - self.min_corner[1],
            self.max_corner[1] - point[1],
        ]
        return -min(distances)

    def intersects_circle(self, center: np.ndarray, radius: float) -> bool:
        return self.distance_to_point(center) <= radius

    def ray_cast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float | None:
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        t_min = 0.0
        t_max = float("inf")

        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < self.min_corner[axis] or origin[axis] > self.max_corner[axis]:
                    return None
                continue

            t1 = (self.min_corner[axis] - origin[axis]) / direction[axis]
            t2 = (self.max_corner[axis] - origin[axis]) / direction[axis]

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return None

        return t_min if t_min >= 0 else None

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        clamped = np.clip(point, self.min_corner, self.max_corner)

        if not self.contains_point(point):
            return clamped

        # Inside: find nearest edge
        distances = [
            (point[0] - self.min_corner[0], np.array([self.min_corner[0], point[1]])),
            (self.max_corner[0] - point[0], np.array([self.max_corner[0], point[1]])),
            (point[1] - self.min_corner[1], np.array([point[0], self.min_corner[1]])),
            (self.max_corner[1] - point[1], np.array([point[0], self.max_corner[1]])),
        ]
        _, nearest = min(distances, key=lambda x: x[0])
        return nearest

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        closest = self.closest_point(point)
        diff = point - closest
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            # On the boundary - find which edge
            to_center = point - self.center
            if abs(to_center[0]) > abs(to_center[1]):
                return np.array([1.0 if to_center[0] > 0 else -1.0, 0.0])
            return np.array([0.0, 1.0 if to_center[1] > 0 else -1.0])
        return diff / dist

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.min_corner.copy(), self.max_corner.copy()

    def get_vertices(self) -> np.ndarray:
        return np.array([
            self.min_corner,
            [self.max_corner[0], self.min_corner[1]],
            self.max_corner,
            [self.min_corner[0], self.max_corner[1]],
        ])

    def inflate(self, margin: float) -> RectangleObstacle:
        m = np.array([margin, margin])
        return RectangleObstacle(
            min_corner=self.min_corner - m,
            max_corner=self.max_corner + m,
            name=self.name,
        )


@dataclass
class LineObstacle(Obstacle):
    """Line segment obstacle (wall).

    Parameters
    ----------
    start : np.ndarray
        Start point, shape (2,).
    end : np.ndarray
        End point, shape (2,).
    thickness : float
        Wall thickness for collision detection.
    name : str
        Optional name.
    """

    start: np.ndarray
    end: np.ndarray
    thickness: float = 0.1
    name: str = ""

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=np.float64)
        self.end = np.asarray(self.end, dtype=np.float64)

    @property
    def length(self) -> float:
        """Length of the line segment."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector along the line."""
        d = self.end - self.start
        length = np.linalg.norm(d)
        if length < 1e-12:
            return np.array([1.0, 0.0])
        return d / length

    @property
    def normal_vec(self) -> np.ndarray:
        """Outward normal vector (perpendicular, CCW rotation)."""
        d = self.direction
        return np.array([-d[1], d[0]])

    def _closest_point_on_segment(self, point: np.ndarray) -> np.ndarray:
        """Find closest point on the line segment."""
        line_vec = self.end - self.start
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-24:
            return self.start.copy()
        t = np.clip(np.dot(point - self.start, line_vec) / line_len_sq, 0.0, 1.0)
        return self.start + t * line_vec

    def contains_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=np.float64)
        closest = self._closest_point_on_segment(point)
        return float(np.linalg.norm(point - closest)) <= self.thickness / 2

    def distance_to_point(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float64)
        closest = self._closest_point_on_segment(point)
        return float(np.linalg.norm(point - closest)) - self.thickness / 2

    def intersects_circle(self, center: np.ndarray, radius: float) -> bool:
        return self.distance_to_point(center) <= radius

    def ray_cast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float | None:
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        seg_dir = self.end - self.start
        cross = direction[0] * seg_dir[1] - direction[1] * seg_dir[0]
        if abs(cross) < 1e-12:
            return None

        diff = self.start - origin
        t = (diff[0] * seg_dir[1] - diff[1] * seg_dir[0]) / cross
        u = (diff[0] * direction[1] - diff[1] * direction[0]) / cross

        if t >= 0 and 0 <= u <= 1:
            return t
        return None

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        return self._closest_point_on_segment(np.asarray(point, dtype=np.float64))

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        return self.normal_vec.copy()

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        min_c = np.minimum(self.start, self.end) - self.thickness / 2
        max_c = np.maximum(self.start, self.end) + self.thickness / 2
        return min_c, max_c

    def get_vertices(self) -> np.ndarray:
        normal = self.normal_vec * self.thickness / 2
        return np.array([
            self.start + normal,
            self.end + normal,
            self.end - normal,
            self.start - normal,
        ])

    def inflate(self, margin: float) -> LineObstacle:
        return LineObstacle(
            start=self.start.copy(),
            end=self.end.copy(),
            thickness=self.thickness + 2 * margin,
            name=self.name,
        )


@dataclass
class PolygonObstacle(Obstacle):
    """Convex polygon obstacle.

    Parameters
    ----------
    vertices : np.ndarray
        Vertices in CCW order, shape (N, 2).
    name : str
        Optional name.
    """

    vertices: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self._edges = np.diff(
            np.vstack([self.vertices, self.vertices[0:1]]), axis=0
        )
        self._normals = np.column_stack([-self._edges[:, 1], self._edges[:, 0]])
        norms = np.linalg.norm(self._normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self._normals = self._normals / norms
        self._center = np.mean(self.vertices, axis=0)

    @property
    def center(self) -> np.ndarray:
        """Centroid of the polygon."""
        return self._center.copy()

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    def contains_point(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=np.float64)
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > point[1]) != (yj > point[1])) and (
                point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside

    def distance_to_point(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float64)
        closest = self.closest_point(point)
        dist = float(np.linalg.norm(point - closest))
        if self.contains_point(point):
            return -dist
        return dist

    def intersects_circle(self, center: np.ndarray, radius: float) -> bool:
        return abs(self.distance_to_point(center)) <= radius or self.contains_point(center)

    def ray_cast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
    ) -> float | None:
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        min_t = None
        n = len(self.vertices)

        for i in range(n):
            j = (i + 1) % n
            seg_start = self.vertices[i]
            seg_end = self.vertices[j]
            seg_dir = seg_end - seg_start

            cross = direction[0] * seg_dir[1] - direction[1] * seg_dir[0]
            if abs(cross) < 1e-12:
                continue

            diff = seg_start - origin
            t = (diff[0] * seg_dir[1] - diff[1] * seg_dir[0]) / cross
            u = (diff[0] * direction[1] - diff[1] * direction[0]) / cross

            if t >= 0 and 0 <= u <= 1:
                if min_t is None or t < min_t:
                    min_t = t

        return min_t

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        min_dist = float("inf")
        best = self.vertices[0].copy()

        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            seg_start = self.vertices[i]
            seg_end = self.vertices[j]

            seg_vec = seg_end - seg_start
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq < 1e-24:
                candidate = seg_start
            else:
                t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
                candidate = seg_start + t * seg_vec

            dist = float(np.linalg.norm(point - candidate))
            if dist < min_dist:
                min_dist = dist
                best = candidate.copy()

        return best

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        # Find closest edge
        min_dist = float("inf")
        best_normal = self._normals[0].copy()

        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            seg_start = self.vertices[i]
            seg_end = self.vertices[j]

            seg_vec = seg_end - seg_start
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq < 1e-24:
                continue
            t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
            closest = seg_start + t * seg_vec
            dist = float(np.linalg.norm(point - closest))

            if dist < min_dist:
                min_dist = dist
                # Ensure normal points outward
                normal = self._normals[i]
                mid = 0.5 * (seg_start + seg_end)
                if np.dot(normal, mid - self._center) < 0:
                    normal = -normal
                best_normal = normal.copy()

        return best_normal

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def get_vertices(self) -> np.ndarray:
        return self.vertices.copy()

    def inflate(self, margin: float) -> PolygonObstacle:
        # Inflate by moving each vertex outward from center
        directions = self.vertices - self._center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        unit_dirs = directions / norms
        return PolygonObstacle(
            vertices=self.vertices + margin * unit_dirs,
            name=self.name,
        )


class ObstacleCollection:
    """Collection of obstacles with spatial indexing for efficient queries.

    Uses a simple grid-based spatial index for fast collision checking.

    Parameters
    ----------
    cell_size : float
        Spatial index cell size.
    """

    def __init__(self, cell_size: float = 5.0) -> None:
        self.cell_size = cell_size
        self._obstacles: list[Obstacle] = []
        self._grid: dict[tuple[int, int], list[int]] = {}

    def add(self, obstacle: Obstacle) -> int:
        """Add an obstacle and return its index.

        Parameters
        ----------
        obstacle : Obstacle
            Obstacle to add.

        Returns
        -------
        int
            Index of the added obstacle.
        """
        idx = len(self._obstacles)
        self._obstacles.append(obstacle)

        # Index in spatial grid
        bb_min, bb_max = obstacle.get_bounding_box()
        min_cx = int(math.floor(bb_min[0] / self.cell_size))
        min_cy = int(math.floor(bb_min[1] / self.cell_size))
        max_cx = int(math.floor(bb_max[0] / self.cell_size))
        max_cy = int(math.floor(bb_max[1] / self.cell_size))

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                key = (cx, cy)
                if key not in self._grid:
                    self._grid[key] = []
                self._grid[key].append(idx)

        return idx

    def check_collision(self, center: np.ndarray, radius: float) -> bool:
        """Check if a circle collides with any obstacle.

        Parameters
        ----------
        center : np.ndarray
            Circle center, shape (2,).
        radius : float
            Circle radius.

        Returns
        -------
        bool
            True if collision detected.
        """
        center = np.asarray(center, dtype=np.float64)
        cx = int(math.floor(center[0] / self.cell_size))
        cy = int(math.floor(center[1] / self.cell_size))
        cell_range = int(math.ceil(radius / self.cell_size)) + 1

        checked: set[int] = set()
        for dx in range(-cell_range, cell_range + 1):
            for dy in range(-cell_range, cell_range + 1):
                key = (cx + dx, cy + dy)
                if key not in self._grid:
                    continue
                for idx in self._grid[key]:
                    if idx in checked:
                        continue
                    checked.add(idx)
                    if self._obstacles[idx].intersects_circle(center, radius):
                        return True
        return False

    def nearest_obstacle_distance(self, point: np.ndarray) -> float:
        """Find distance to the nearest obstacle.

        Parameters
        ----------
        point : np.ndarray
            Query point, shape (2,).

        Returns
        -------
        float
            Distance to nearest obstacle. Returns inf if no obstacles.
        """
        if not self._obstacles:
            return float("inf")

        point = np.asarray(point, dtype=np.float64)
        min_dist = float("inf")
        for obs in self._obstacles:
            dist = obs.distance_to_point(point)
            min_dist = min(min_dist, dist)
        return min_dist

    def ray_cast(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = float("inf"),
    ) -> tuple[float, int] | None:
        """Cast a ray and find the first obstacle hit.

        Parameters
        ----------
        origin : np.ndarray
            Ray origin, shape (2,).
        direction : np.ndarray
            Ray direction (unit vector), shape (2,).
        max_distance : float
            Maximum ray length.

        Returns
        -------
        tuple or None
            (distance, obstacle_index) or None if no hit.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        best_t = max_distance
        best_idx = -1

        for idx, obs in enumerate(self._obstacles):
            t = obs.ray_cast(origin, direction)
            if t is not None and t < best_t:
                best_t = t
                best_idx = idx

        if best_idx >= 0:
            return best_t, best_idx
        return None

    def multi_ray_cast(
        self,
        origin: np.ndarray,
        num_rays: int = 36,
        max_distance: float = 10.0,
    ) -> np.ndarray:
        """Cast multiple rays in a fan pattern.

        Parameters
        ----------
        origin : np.ndarray
            Ray origin, shape (2,).
        num_rays : int
            Number of rays.
        max_distance : float
            Maximum ray length.

        Returns
        -------
        np.ndarray
            Distances for each ray, shape (num_rays,).
        """
        origin = np.asarray(origin, dtype=np.float64)
        distances = np.full(num_rays, max_distance)

        for i in range(num_rays):
            angle = 2.0 * math.pi * i / num_rays
            direction = np.array([math.cos(angle), math.sin(angle)])
            result = self.ray_cast(origin, direction, max_distance)
            if result is not None:
                distances[i] = result[0]

        return distances

    def get_all_obstacles(self) -> list[Obstacle]:
        """Get all obstacles."""
        return list(self._obstacles)

    def __len__(self) -> int:
        return len(self._obstacles)

    def __getitem__(self, idx: int) -> Obstacle:
        return self._obstacles[idx]

    def clear(self) -> None:
        """Remove all obstacles."""
        self._obstacles.clear()
        self._grid.clear()

    def inflate_all(self, margin: float) -> ObstacleCollection:
        """Create a new collection with all obstacles inflated.

        Parameters
        ----------
        margin : float
            Inflation margin.

        Returns
        -------
        ObstacleCollection
            New collection with inflated obstacles.
        """
        inflated = ObstacleCollection(self.cell_size)
        for obs in self._obstacles:
            try:
                inflated.add(obs.inflate(margin))
            except NotImplementedError:
                inflated.add(obs)
        return inflated
