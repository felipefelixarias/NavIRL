from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np

from navirl.planning.base import Path, Planner, PlannerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_neighbors(node: tuple[int, int], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    """8-connected neighbours on a grid."""
    r, c = node
    rows, cols = grid_shape
    nbrs: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nbrs.append((nr, nc))
    return nbrs


def _pos_to_grid(pos: np.ndarray, origin: np.ndarray, resolution: float) -> tuple[int, int]:
    idx = ((pos - origin) / resolution).astype(int)
    return (int(idx[0]), int(idx[1]))


def _grid_to_pos(cell: tuple[int, int], origin: np.ndarray, resolution: float) -> np.ndarray:
    return origin + np.array(cell, dtype=np.float64) * resolution


def _is_free(cell: tuple[int, int], occ_grid: np.ndarray | None) -> bool:
    if occ_grid is None:
        return True
    r, c = cell
    if 0 <= r < occ_grid.shape[0] and 0 <= c < occ_grid.shape[1]:
        return bool(occ_grid[r, c] == 0)
    return False


def _reconstruct(came_from: dict, current: Any) -> list:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _build_path(
    cells: list[tuple[int, int]],
    origin: np.ndarray,
    resolution: float,
    speed: float = 1.0,
) -> Path:
    waypoints = np.array([_grid_to_pos(c, origin, resolution) for c in cells])
    dists = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))])
    timestamps = dists / max(speed, 1e-6)
    velocities = np.zeros_like(waypoints)
    if waypoints.shape[0] >= 2:
        for i in range(waypoints.shape[0] - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 1e-12:
                velocities[i] = (waypoints[i + 1] - waypoints[i]) / dt
        velocities[-1] = velocities[-2] if waypoints.shape[0] >= 2 else np.zeros(2)
    return Path(
        waypoints=waypoints,
        timestamps=timestamps,
        velocities=velocities,
        cost=float(dists[-1]) if dists.size else 0.0,
    )


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------


class AStarPlanner(Planner):
    """A* search on a 2-D occupancy grid with configurable heuristic."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        heuristic: str = "euclidean",
        speed: float = 1.0,
    ) -> None:
        self.config = config or PlannerConfig()
        self.heuristic = heuristic
        self.speed = speed

    def _h(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        if self.heuristic == "manhattan":
            return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
        if self.heuristic == "chebyshev":
            return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
        # Default: euclidean
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        del dynamic_agents
        res = self.config.resolution
        origin = np.zeros(2)
        s = _pos_to_grid(start, origin, res)
        g = _pos_to_grid(goal, origin, res)

        open_set: list[tuple[float, tuple[int, int]]] = [(self._h(s, g), s)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {s: 0.0}

        grid_shape = (
            obstacles.shape if obstacles is not None and obstacles.ndim == 2 else (10000, 10000)
        )
        t0 = time.monotonic()
        iterations = 0

        while open_set and iterations < self.config.max_iterations:
            if time.monotonic() - t0 > self.config.time_limit:
                break
            iterations += 1
            _, current = heapq.heappop(open_set)
            if current == g:
                cells = _reconstruct(came_from, current)
                return _build_path(cells, origin, res, self.speed)

            for nbr in _grid_neighbors(current, grid_shape):
                if not _is_free(nbr, obstacles):
                    continue
                move_cost = float(np.hypot(nbr[0] - current[0], nbr[1] - current[1])) * res
                tentative = g_score[current] + move_cost
                if tentative < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr] = tentative
                    f = tentative + self._h(nbr, g) * res
                    heapq.heappush(open_set, (f, nbr))

        # Fallback: straight line.
        return _build_path([s, g], origin, res, self.speed)


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------


class DijkstraPlanner(Planner):
    """Dijkstra's shortest-path algorithm on a grid."""

    def __init__(self, config: PlannerConfig | None = None, speed: float = 1.0) -> None:
        self.config = config or PlannerConfig()
        self.speed = speed

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        del dynamic_agents
        res = self.config.resolution
        origin = np.zeros(2)
        s = _pos_to_grid(start, origin, res)
        g = _pos_to_grid(goal, origin, res)

        grid_shape = (
            obstacles.shape if obstacles is not None and obstacles.ndim == 2 else (10000, 10000)
        )

        open_set: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        dist: dict[tuple[int, int], float] = {s: 0.0}
        t0 = time.monotonic()
        iterations = 0

        while open_set and iterations < self.config.max_iterations:
            if time.monotonic() - t0 > self.config.time_limit:
                break
            iterations += 1
            d, current = heapq.heappop(open_set)
            if current == g:
                cells = _reconstruct(came_from, current)
                return _build_path(cells, origin, res, self.speed)
            if d > dist.get(current, float("inf")):
                continue
            for nbr in _grid_neighbors(current, grid_shape):
                if not _is_free(nbr, obstacles):
                    continue
                move_cost = float(np.hypot(nbr[0] - current[0], nbr[1] - current[1])) * res
                tentative = dist[current] + move_cost
                if tentative < dist.get(nbr, float("inf")):
                    dist[nbr] = tentative
                    came_from[nbr] = current
                    heapq.heappush(open_set, (tentative, nbr))

        return _build_path([s, g], origin, res, self.speed)


# ---------------------------------------------------------------------------
# Theta*
# ---------------------------------------------------------------------------


class ThetaStarPlanner(Planner):
    """Theta* (any-angle A*) on a grid."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        speed: float = 1.0,
    ) -> None:
        self.config = config or PlannerConfig()
        self.speed = speed

    @staticmethod
    def _line_of_sight(
        a: tuple[int, int],
        b: tuple[int, int],
        occ_grid: np.ndarray | None,
    ) -> bool:
        """Bresenham-based line-of-sight check."""
        if occ_grid is None:
            return True
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        while True:
            if not _is_free((r0, c0), occ_grid):
                return False
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc
        return True

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        del dynamic_agents
        res = self.config.resolution
        origin = np.zeros(2)
        s = _pos_to_grid(start, origin, res)
        g = _pos_to_grid(goal, origin, res)

        grid_shape = (
            obstacles.shape if obstacles is not None and obstacles.ndim == 2 else (10000, 10000)
        )

        open_set: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {s: 0.0}
        t0 = time.monotonic()
        iterations = 0

        def _h(a: tuple[int, int]) -> float:
            return float(np.hypot(a[0] - g[0], a[1] - g[1])) * res

        while open_set and iterations < self.config.max_iterations:
            if time.monotonic() - t0 > self.config.time_limit:
                break
            iterations += 1
            _, current = heapq.heappop(open_set)
            if current == g:
                cells = _reconstruct(came_from, current)
                return _build_path(cells, origin, res, self.speed)

            for nbr in _grid_neighbors(current, grid_shape):
                if not _is_free(nbr, obstacles):
                    continue
                # Theta* path-2 update: check line of sight to parent.
                parent = came_from.get(current, current)
                if self._line_of_sight(parent, nbr, obstacles):
                    d = float(np.hypot(nbr[0] - parent[0], nbr[1] - parent[1])) * res
                    tentative = g_score[parent] + d
                    if tentative < g_score.get(nbr, float("inf")):
                        came_from[nbr] = parent
                        g_score[nbr] = tentative
                        heapq.heappush(open_set, (tentative + _h(nbr), nbr))
                else:
                    move_cost = float(np.hypot(nbr[0] - current[0], nbr[1] - current[1])) * res
                    tentative = g_score[current] + move_cost
                    if tentative < g_score.get(nbr, float("inf")):
                        came_from[nbr] = current
                        g_score[nbr] = tentative
                        heapq.heappush(open_set, (tentative + _h(nbr), nbr))

        return _build_path([s, g], origin, res, self.speed)


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------


class RRTPlanner(Planner):
    """Rapidly-exploring Random Tree (RRT)."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        step_size: float = 0.5,
        goal_bias: float = 0.1,
        speed: float = 1.0,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.speed = speed
        self.bounds = bounds  # (low, high) each (2,)

    def _sample(self, goal: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.goal_bias:
            return goal.copy()
        return np.random.uniform(low, high)

    @staticmethod
    def _nearest(tree: list[np.ndarray], point: np.ndarray) -> int:
        dists = [float(np.linalg.norm(n - point)) for n in tree]
        return int(np.argmin(dists))

    def _steer(self, from_pt: np.ndarray, to_pt: np.ndarray) -> np.ndarray:
        diff = to_pt - from_pt
        d = float(np.linalg.norm(diff))
        if d <= self.step_size:
            return to_pt.copy()
        return from_pt + diff / d * self.step_size

    @staticmethod
    def _collision_free(
        a: np.ndarray,
        b: np.ndarray,
        obstacles: np.ndarray | None,
        resolution: float = 0.05,
    ) -> bool:
        if obstacles is None:
            return True
        diff = b - a
        d = float(np.linalg.norm(diff))
        steps = max(int(d / resolution), 1)
        for i in range(steps + 1):
            pt = a + diff * (i / steps)
            # Point obstacles: Nx2 with implicit radius.
            if obstacles.ndim == 2 and obstacles.shape[1] >= 2:
                dists = np.linalg.norm(obstacles[:, :2] - pt, axis=1)
                radius = obstacles[:, 2] if obstacles.shape[1] > 2 else 0.3
                if np.any(dists < radius):
                    return False
        return True

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        del dynamic_agents
        if self.bounds is not None:
            low, high = self.bounds
        else:
            all_pts = [start, goal]
            low = np.min(all_pts, axis=0) - 5.0
            high = np.max(all_pts, axis=0) + 5.0

        tree: list[np.ndarray] = [start.copy()]
        parent: dict[int, int] = {}
        t0 = time.monotonic()

        for _it in range(self.config.max_iterations):
            if time.monotonic() - t0 > self.config.time_limit:
                break
            sample = self._sample(goal, low, high)
            nearest_idx = self._nearest(tree, sample)
            new_pt = self._steer(tree[nearest_idx], sample)
            if self._collision_free(tree[nearest_idx], new_pt, obstacles):
                new_idx = len(tree)
                tree.append(new_pt)
                parent[new_idx] = nearest_idx
                if float(np.linalg.norm(new_pt - goal)) < self.step_size:
                    # Trace back.
                    idx = new_idx
                    waypoints = []
                    while idx in parent:
                        waypoints.append(tree[idx])
                        idx = parent[idx]
                    waypoints.append(tree[0])
                    waypoints.reverse()
                    wp = np.array(waypoints)
                    dists = np.concatenate(
                        [[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
                    )
                    ts = dists / max(self.speed, 1e-6)
                    vels = np.zeros_like(wp)
                    for i in range(wp.shape[0] - 1):
                        dt = ts[i + 1] - ts[i]
                        if dt > 1e-12:
                            vels[i] = (wp[i + 1] - wp[i]) / dt
                    return Path(waypoints=wp, timestamps=ts, velocities=vels, cost=float(dists[-1]))

        # Fallback.
        wp = np.stack([start, goal])
        d = float(np.linalg.norm(goal - start))
        return Path(
            waypoints=wp,
            timestamps=np.array([0.0, d / max(self.speed, 1e-6)]),
            velocities=np.zeros_like(wp),
            cost=d,
        )


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------


class RRTStarPlanner(Planner):
    """Asymptotically optimal RRT*."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        step_size: float = 0.5,
        goal_bias: float = 0.1,
        rewire_radius: float = 1.5,
        speed: float = 1.0,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.speed = speed
        self.bounds = bounds

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        del dynamic_agents
        if self.bounds is not None:
            low, high = self.bounds
        else:
            low = np.minimum(start, goal) - 5.0
            high = np.maximum(start, goal) + 5.0

        tree: list[np.ndarray] = [start.copy()]
        parent: dict[int, int] = {}
        cost: dict[int, float] = {0: 0.0}
        t0 = time.monotonic()
        best_goal_idx: int | None = None
        best_goal_cost = float("inf")

        for _ in range(self.config.max_iterations):
            if time.monotonic() - t0 > self.config.time_limit:
                break

            if np.random.rand() < self.goal_bias:
                sample = goal.copy()
            else:
                sample = np.random.uniform(low, high)

            # Nearest.
            dists_to_tree = np.array([np.linalg.norm(n - sample) for n in tree])
            nearest_idx = int(np.argmin(dists_to_tree))
            diff = sample - tree[nearest_idx]
            d = float(np.linalg.norm(diff))
            if d > self.step_size:
                new_pt = tree[nearest_idx] + diff / d * self.step_size
            else:
                new_pt = sample.copy()

            if not RRTPlanner._collision_free(tree[nearest_idx], new_pt, obstacles):
                continue

            # Find nearby nodes for rewiring.
            new_idx = len(tree)
            near_indices = [
                i for i, n in enumerate(tree) if np.linalg.norm(n - new_pt) < self.rewire_radius
            ]

            # Choose best parent.
            best_parent = nearest_idx
            best_cost = cost[nearest_idx] + float(np.linalg.norm(new_pt - tree[nearest_idx]))
            for ni in near_indices:
                c = cost[ni] + float(np.linalg.norm(new_pt - tree[ni]))
                if c < best_cost and RRTPlanner._collision_free(tree[ni], new_pt, obstacles):
                    best_parent = ni
                    best_cost = c

            tree.append(new_pt)
            parent[new_idx] = best_parent
            cost[new_idx] = best_cost

            # Rewire.
            for ni in near_indices:
                c = best_cost + float(np.linalg.norm(tree[ni] - new_pt))
                if c < cost[ni] and RRTPlanner._collision_free(new_pt, tree[ni], obstacles):
                    parent[ni] = new_idx
                    cost[ni] = c

            # Check goal.
            if float(np.linalg.norm(new_pt - goal)) < self.step_size and best_cost < best_goal_cost:
                best_goal_idx = new_idx
                best_goal_cost = best_cost

        if best_goal_idx is not None:
            idx = best_goal_idx
            waypoints = []
            while idx in parent:
                waypoints.append(tree[idx])
                idx = parent[idx]
            waypoints.append(tree[0])
            waypoints.reverse()
            wp = np.array(waypoints)
            dists = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
            ts = dists / max(self.speed, 1e-6)
            vels = np.zeros_like(wp)
            for i in range(wp.shape[0] - 1):
                dt = ts[i + 1] - ts[i]
                if dt > 1e-12:
                    vels[i] = (wp[i + 1] - wp[i]) / dt
            return Path(waypoints=wp, timestamps=ts, velocities=vels, cost=best_goal_cost)

        # Fallback.
        wp = np.stack([start, goal])
        d = float(np.linalg.norm(goal - start))
        return Path(
            waypoints=wp,
            timestamps=np.array([0.0, d / max(self.speed, 1e-6)]),
            velocities=np.zeros_like(wp),
            cost=d,
        )


# ---------------------------------------------------------------------------
# PRM
# ---------------------------------------------------------------------------


class PRMPlanner(Planner):
    """Probabilistic Roadmap (PRM) planner."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        num_samples: int = 500,
        k_neighbors: int = 10,
        speed: float = 1.0,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.speed = speed
        self.bounds = bounds

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray | None = None,
        dynamic_agents: list[np.ndarray] | None = None,
    ) -> Path:
        del dynamic_agents
        if self.bounds is not None:
            low, high = self.bounds
        else:
            low = np.minimum(start, goal) - 5.0
            high = np.maximum(start, goal) + 5.0

        # Sample nodes.
        nodes = [start.copy(), goal.copy()]
        for _ in range(self.num_samples):
            pt = np.random.uniform(low, high)
            nodes.append(pt)
        nodes_arr = np.array(nodes)  # (M, 2)
        M = nodes_arr.shape[0]

        # Build adjacency (k-nearest neighbours, collision-checked).
        adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(M)}
        for i in range(M):
            dists = np.linalg.norm(nodes_arr - nodes_arr[i], axis=1)
            neighbours = np.argsort(dists)[1 : self.k_neighbors + 1]
            for j in neighbours:
                j_idx = int(j)
                d = float(dists[j_idx])
                if RRTPlanner._collision_free(nodes_arr[i], nodes_arr[j_idx], obstacles):
                    adjacency[i].append((j_idx, d))
                    adjacency[j_idx].append((i, d))

        # Dijkstra from node 0 (start) to node 1 (goal).
        dist_map: dict[int, float] = {0: 0.0}
        came_from: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == 1:
                break
            if d > dist_map.get(u, float("inf")):
                continue
            for v, w in adjacency[u]:
                nd = d + w
                if nd < dist_map.get(v, float("inf")):
                    dist_map[v] = nd
                    came_from[v] = u
                    heapq.heappush(pq, (nd, v))

        if 1 not in came_from and 1 != 0:
            # No path found; straight-line fallback.
            wp = np.stack([start, goal])
            d = float(np.linalg.norm(goal - start))
            return Path(
                waypoints=wp,
                timestamps=np.array([0.0, d / max(self.speed, 1e-6)]),
                velocities=np.zeros_like(wp),
                cost=d,
            )

        idx = 1
        path_indices = [idx]
        while idx in came_from:
            idx = came_from[idx]
            path_indices.append(idx)
        path_indices.reverse()

        wp = nodes_arr[path_indices]
        dists = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
        ts = dists / max(self.speed, 1e-6)
        vels = np.zeros_like(wp)
        for i in range(wp.shape[0] - 1):
            dt = ts[i + 1] - ts[i]
            if dt > 1e-12:
                vels[i] = (wp[i + 1] - wp[i]) / dt
        return Path(waypoints=wp, timestamps=ts, velocities=vels, cost=float(dists[-1]))
