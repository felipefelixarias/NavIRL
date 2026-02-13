from __future__ import annotations

import heapq
from collections import deque
from typing import Tuple

import cv2
import numpy as np

from navirl.backends.grid2d.constants import FREE_SPACE, OBSTACLE_SPACE


def _l2(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class GridEnvironment:
    """Standalone map environment for Grid2D pathing and obstacle extraction."""

    def __init__(self, name: str, binary_map: np.ndarray, pixels_per_meter: float = 100.0) -> None:
        self.name = name
        self.map = self._normalize_map(np.asarray(binary_map))
        self.map_size = self.map.shape
        self.pixels_per_meter = float(pixels_per_meter)
        if self.pixels_per_meter <= 0.0:
            raise ValueError("pixels_per_meter must be > 0")
        self.obstacle_map = self.map.copy()
        self.obstacles_meters: list[list[list[float]]] = []
        self._obstacles_polys_pixels: list[np.ndarray] = []

        self.waypoint_dist = 0.2
        self.num_waypoints = 10

        free = np.argwhere(self.map == FREE_SPACE)
        if free.size == 0:
            raise ValueError("Map has no traversable free space.")
        self._free_nodes = free

    def _normalize_map(self, map_img: np.ndarray) -> np.ndarray:
        m = map_img.copy().astype(np.uint8)
        m[m > 0] = FREE_SPACE
        m[m == 0] = OBSTACLE_SPACE
        return m

    def _map_to_world(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        center = np.array([self.map.shape[0] / 2.0, self.map.shape[1] / 2.0], dtype=float)
        delta = (xy - center) / self.pixels_per_meter
        if delta.ndim == 1:
            return np.array([delta[1], delta[0]], dtype=float)
        if delta.ndim == 2 and delta.shape[1] == 2:
            return delta[:, [1, 0]]
        raise ValueError("Expected shape (2,) or (N,2)")

    def _world_to_map(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        center = np.array([self.map.shape[0] / 2.0, self.map.shape[1] / 2.0], dtype=float)
        if xy.ndim == 1:
            rc = np.array([xy[1], xy[0]], dtype=float)
            return np.rint(rc * self.pixels_per_meter + center).astype(int)
        if xy.ndim == 2 and xy.shape[1] == 2:
            rc = xy[:, [1, 0]]
            return np.rint(rc * self.pixels_per_meter + center).astype(int)
        raise ValueError("Expected shape (2,) or (N,2)")

    def map_to_world(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        center = np.array([self.map.shape[1] / 2.0, self.map.shape[0] / 2.0], dtype=float)
        return (xy - center) / self.pixels_per_meter

    def world_to_map(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        center = np.array([self.map.shape[1] / 2.0, self.map.shape[0] / 2.0], dtype=float)
        return np.rint(xy * self.pixels_per_meter + center).astype(int)

    def process_obstacles(self) -> None:
        self.obstacles_meters = []
        self._obstacles_polys_pixels = []

        obstacle_mask = (self.map == OBSTACLE_SPACE).astype(np.uint8)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if contour.shape[0] < 3:
                continue
            pts = contour[:, 0, :].astype(float)
            self._obstacles_polys_pixels.append(pts)
            world_poly = []
            for x_pix, y_pix in pts:
                x = (x_pix - self.map.shape[1] / 2.0) / self.pixels_per_meter
                y = (y_pix - self.map.shape[0] / 2.0) / self.pixels_per_meter
                world_poly.append([float(x), float(y)])
            if len(world_poly) >= 3:
                self.obstacles_meters.append(world_poly)

    def get_obstacle_meters(self) -> list[list[list[float]]]:
        return self.obstacles_meters

    def get_random_point(self) -> list[float]:
        idx = int(np.random.randint(0, len(self._free_nodes)))
        row, col = self._free_nodes[idx]
        world = self._map_to_world(np.array([row, col]))
        return [float(world[0]), float(world[1])]

    def _in_bounds(self, node: tuple[int, int]) -> bool:
        r, c = node
        return 0 <= r < self.map.shape[0] and 0 <= c < self.map.shape[1]

    def _is_free(self, node: tuple[int, int]) -> bool:
        r, c = node
        return self.map[r, c] == FREE_SPACE

    def _nearest_free(self, start: tuple[int, int]) -> tuple[int, int]:
        if self._in_bounds(start) and self._is_free(start):
            return start

        q = deque([start])
        seen = {start}
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nxt = (r + dr, c + dc)
                if nxt in seen:
                    continue
                seen.add(nxt)
                if not self._in_bounds(nxt):
                    continue
                if self._is_free(nxt):
                    return nxt
                q.append(nxt)

        # Fallback to random free cell if search fails.
        rr, cc = self._free_nodes[int(np.random.randint(0, len(self._free_nodes)))]
        return int(rr), int(cc)

    def nearest_free_world(self, position: tuple[float, float]) -> tuple[float, float]:
        rc = tuple(self._world_to_map(np.array(position, dtype=float)).tolist())
        nearest = self._nearest_free(rc)
        world = self._map_to_world(np.array(nearest, dtype=float))
        return float(world[0]), float(world[1])

    def _astar(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        open_heap: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (_l2(start, goal), 0.0, start))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}
        visited: set[tuple[int, int]] = set()

        while open_heap:
            _, cur_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dr, dc in dirs:
                nbr = (current[0] + dr, current[1] + dc)
                if not self._in_bounds(nbr) or not self._is_free(nbr):
                    continue
                tentative = cur_g + 1.0
                if tentative < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative
                    f = tentative + _l2(nbr, goal)
                    heapq.heappush(open_heap, (f, tentative, nbr))

        return []

    def shortest_path(
        self,
        source_world: np.ndarray,
        target_world: np.ndarray,
        entire_path: bool = False,
    ) -> Tuple[np.ndarray, float]:
        src_rc = tuple(self._world_to_map(np.array(source_world, dtype=float)).tolist())
        dst_rc = tuple(self._world_to_map(np.array(target_world, dtype=float)).tolist())

        src_rc = self._nearest_free(src_rc)
        dst_rc = self._nearest_free(dst_rc)
        path_rc = self._astar(src_rc, dst_rc)

        if not path_rc:
            target = np.asarray(target_world, dtype=float)
            return np.array([target], dtype=float), float("inf")

        path_world = self._map_to_world(np.asarray(path_rc, dtype=float))
        if len(path_world) == 1:
            return path_world, 0.0

        seg = np.linalg.norm(path_world[1:] - path_world[:-1], axis=1)
        geodesic = float(np.sum(seg))

        if entire_path:
            return path_world, geodesic

        waypoints = []
        acc = 0.0
        for i in range(1, len(path_world)):
            acc += float(seg[i - 1])
            if acc >= self.waypoint_dist:
                waypoints.append(path_world[i])
                acc = 0.0

        if not waypoints:
            waypoints = [path_world[-1]]

        waypoints = np.asarray(waypoints[: self.num_waypoints], dtype=float)
        return waypoints, geodesic
