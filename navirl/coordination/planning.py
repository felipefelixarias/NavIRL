"""Coordinated path planning for multi-agent systems.

Provides priority-based planning, Conflict-Based Search (CBS), and
velocity-obstacle-based coordination.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------


@dataclass
class PlanningResult:
    """Result of a coordinated planning computation.

    Attributes:
        paths: Mapping from agent id to a list of waypoints ``(x, y)``.
        cost: Total path cost (sum of individual path lengths).
        conflicts_resolved: Number of inter-agent conflicts resolved.
    """

    paths: dict[str, list[np.ndarray]]
    cost: float
    conflicts_resolved: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_length(path: Sequence[np.ndarray]) -> float:
    """Compute total Euclidean length of a waypoint path."""
    length = 0.0
    for i in range(1, len(path)):
        length += float(np.linalg.norm(path[i] - path[i - 1]))
    return length


def _reconstruct(came_from: dict[tuple, tuple], current: tuple) -> list[tuple]:
    """Trace back from *current* through *came_from* to reconstruct the path."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# A* on a grid (used by CBS and Priority planners)
# ---------------------------------------------------------------------------


def _astar_grid(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
    constraints: set[tuple[int, int, int]] | None = None,
    max_timesteps: int = 200,
) -> list[tuple[int, int]] | None:
    """Run A* on a 2-D occupancy *grid* respecting time-indexed constraints.

    Parameters:
        start: ``(row, col)`` start cell.
        goal: ``(row, col)`` goal cell.
        grid: 2-D array where 0 = free, 1 = obstacle.
        constraints: Set of ``(row, col, timestep)`` tuples the agent must
            not occupy.
        max_timesteps: Maximum search horizon.

    Returns:
        List of ``(row, col)`` positions per timestep, or ``None`` if no
        path was found.
    """
    constraints = constraints or set()
    rows, cols = grid.shape

    def heuristic(r: int, c: int) -> float:
        return abs(r - goal[0]) + abs(c - goal[1])  # Manhattan

    open_set: list[tuple[float, int, int, int]] = []  # (f, t, r, c)
    heapq.heappush(open_set, (heuristic(*start), 0, start[0], start[1]))
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    g_score: dict[tuple[int, int, int], float] = {(start[0], start[1], 0): 0}

    neighbours = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]  # wait + 4-connected

    while open_set:
        _f, t, r, c = heapq.heappop(open_set)

        if (r, c) == goal:
            # Reconstruct
            path_nodes: list[tuple[int, int]] = []
            node = (r, c, t)
            while node in came_from:
                path_nodes.append((node[0], node[1]))
                node = came_from[node]
            path_nodes.append((node[0], node[1]))
            path_nodes.reverse()
            return path_nodes

        if t >= max_timesteps:
            continue

        nt = t + 1
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                if (nr, nc, nt) in constraints:
                    continue
                new_g = g_score.get((r, c, t), float("inf")) + 1
                if new_g < g_score.get((nr, nc, nt), float("inf")):
                    g_score[(nr, nc, nt)] = new_g
                    came_from[(nr, nc, nt)] = (r, c, t)
                    f = new_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f, nt, nr, nc))

    return None  # no path found


# ---------------------------------------------------------------------------
# Priority-based planner
# ---------------------------------------------------------------------------


class PriorityPlanner:
    """Fixed priority-based multi-agent path planner.

    Higher-priority agents plan first on the raw grid.  Lower-priority
    agents treat the paths of all higher-priority agents as dynamic
    obstacles.

    Parameters:
        grid: 2-D occupancy grid (0 = free, 1 = obstacle).
    """

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = np.asarray(grid, dtype=np.int32)

    def plan(
        self,
        starts: dict[str, tuple[int, int]],
        goals: dict[str, tuple[int, int]],
        priorities: dict[str, int] | None = None,
    ) -> PlanningResult:
        """Plan paths for all agents respecting priority ordering.

        Parameters:
            starts: ``{agent_id: (row, col)}``.
            goals: ``{agent_id: (row, col)}``.
            priorities: ``{agent_id: priority}`` (higher = plans first).
                If ``None``, insertion order is used.

        Returns:
            :class:`PlanningResult` with per-agent paths.
        """
        if priorities is None:
            ordered = list(starts.keys())
        else:
            ordered = sorted(starts.keys(), key=lambda a: -priorities.get(a, 0))

        paths: dict[str, list[np.ndarray]] = {}
        constraints: set[tuple[int, int, int]] = set()
        total_cost = 0.0

        for agent_id in ordered:
            path = _astar_grid(starts[agent_id], goals[agent_id], self.grid, constraints)
            if path is None:
                paths[agent_id] = [np.array(starts[agent_id], dtype=np.float64)]
            else:
                paths[agent_id] = [np.array(p, dtype=np.float64) for p in path]
                total_cost += len(path) - 1
                # Add this agent's path as constraints for lower-priority agents
                for t, (r, c) in enumerate(path):
                    constraints.add((r, c, t))

        return PlanningResult(paths=paths, cost=total_cost)


# ---------------------------------------------------------------------------
# Conflict-Based Search (CBS)
# ---------------------------------------------------------------------------


@dataclass
class _Conflict:
    """Represents a conflict between two agents at a specific timestep."""

    agent_a: str
    agent_b: str
    location: tuple[int, int]
    timestep: int


@dataclass
class _CBSNode:
    """A node in the CBS high-level constraint tree."""

    constraints: dict[str, set[tuple[int, int, int]]]
    paths: dict[str, list[tuple[int, int]]]
    cost: float

    def __lt__(self, other: _CBSNode) -> bool:
        return self.cost < other.cost


class CBSPlanner:
    """Conflict-Based Search for optimal multi-agent pathfinding.

    **High level**: maintain a constraint tree; detect conflicts and branch
    by adding constraints to one of the conflicting agents.

    **Low level**: run A* with per-agent constraints.

    Parameters:
        grid: 2-D occupancy grid (0 = free, 1 = obstacle).
        max_iterations: Maximum CBS iterations before returning best-effort.
    """

    def __init__(self, grid: np.ndarray, max_iterations: int = 1000) -> None:
        self.grid = np.asarray(grid, dtype=np.int32)
        self.max_iterations = max_iterations

    def plan(
        self,
        starts: dict[str, tuple[int, int]],
        goals: dict[str, tuple[int, int]],
    ) -> PlanningResult:
        """Find conflict-free paths for all agents using CBS.

        Parameters:
            starts: ``{agent_id: (row, col)}``.
            goals: ``{agent_id: (row, col)}``.

        Returns:
            :class:`PlanningResult`.
        """
        agent_ids = list(starts.keys())

        # Root node: plan each agent independently
        root_constraints: dict[str, set[tuple[int, int, int]]] = {a: set() for a in agent_ids}
        root_paths: dict[str, list[tuple[int, int]]] = {}
        for a in agent_ids:
            path = _astar_grid(starts[a], goals[a], self.grid)
            root_paths[a] = path if path is not None else [starts[a]]

        root_cost = sum(len(p) - 1 for p in root_paths.values())
        root = _CBSNode(constraints=root_constraints, paths=root_paths, cost=root_cost)

        open_list: list[_CBSNode] = [root]
        conflicts_resolved = 0

        for _ in range(self.max_iterations):
            if not open_list:
                break
            node = heapq.heappop(open_list)

            conflict = self._find_first_conflict(node.paths, agent_ids)
            if conflict is None:
                # No conflicts — optimal solution found
                result_paths = {
                    a: [np.array(p, dtype=np.float64) for p in path]
                    for a, path in node.paths.items()
                }
                return PlanningResult(
                    paths=result_paths,
                    cost=node.cost,
                    conflicts_resolved=conflicts_resolved,
                )

            # Branch on conflict: constrain each agent in turn
            for agent in (conflict.agent_a, conflict.agent_b):
                new_constraints = {a: set(c) for a, c in node.constraints.items()}
                new_constraints[agent].add(
                    (conflict.location[0], conflict.location[1], conflict.timestep)
                )
                new_paths = dict(node.paths)
                new_path = _astar_grid(
                    starts[agent],
                    goals[agent],
                    self.grid,
                    new_constraints[agent],
                )
                if new_path is None:
                    continue  # infeasible branch
                new_paths[agent] = new_path
                new_cost = sum(len(p) - 1 for p in new_paths.values())
                child = _CBSNode(constraints=new_constraints, paths=new_paths, cost=new_cost)
                heapq.heappush(open_list, child)
                conflicts_resolved += 1

        # Return best node found so far
        best = root if not open_list else open_list[0]
        result_paths = {
            a: [np.array(p, dtype=np.float64) for p in path] for a, path in best.paths.items()
        }
        return PlanningResult(
            paths=result_paths,
            cost=best.cost,
            conflicts_resolved=conflicts_resolved,
        )

    @staticmethod
    def _find_first_conflict(
        paths: dict[str, list[tuple[int, int]]],
        agent_ids: list[str],
    ) -> _Conflict | None:
        """Detect the first vertex conflict between any pair of agents."""
        max_t = max(len(p) for p in paths.values())
        for t in range(max_t):
            occupied: dict[tuple[int, int], str] = {}
            for a in agent_ids:
                pos = paths[a][min(t, len(paths[a]) - 1)]
                if pos in occupied:
                    return _Conflict(
                        agent_a=occupied[pos],
                        agent_b=a,
                        location=pos,
                        timestep=t,
                    )
                occupied[pos] = a
        return None


# ---------------------------------------------------------------------------
# Velocity Obstacle planner (ORCA-style)
# ---------------------------------------------------------------------------


class VelocityObstaclePlanner:
    """Reciprocal velocity obstacle coordination (ORCA-style).

    Each agent computes half-plane constraints induced by all other agents
    and selects the closest feasible velocity to its preferred velocity.

    Parameters:
        time_horizon: Look-ahead time for collision avoidance (seconds).
        max_speed: Maximum agent speed (m/s).
        agent_radius: Radius of each agent for collision detection.
    """

    def __init__(
        self,
        time_horizon: float = 5.0,
        max_speed: float = 1.5,
        agent_radius: float = 0.3,
    ) -> None:
        self.time_horizon = time_horizon
        self.max_speed = max_speed
        self.agent_radius = agent_radius

    def compute_velocities(
        self,
        positions: np.ndarray,
        current_velocities: np.ndarray,
        preferred_velocities: np.ndarray,
    ) -> np.ndarray:
        """Compute ORCA-constrained velocities for all agents.

        Parameters:
            positions: ``(N, 2)`` current positions.
            current_velocities: ``(N, 2)`` current velocities.
            preferred_velocities: ``(N, 2)`` desired velocities (e.g.
                toward goal).

        Returns:
            ``(N, 2)`` adjusted velocities satisfying ORCA constraints.
        """
        positions = np.asarray(positions, dtype=np.float64)
        current_velocities = np.asarray(current_velocities, dtype=np.float64)
        preferred_velocities = np.asarray(preferred_velocities, dtype=np.float64)
        n = len(positions)
        new_velocities = np.copy(preferred_velocities)

        for i in range(n):
            orca_planes: list[tuple[np.ndarray, np.ndarray]] = []  # (point, normal)

            for j in range(n):
                if i == j:
                    continue

                relative_pos = positions[j] - positions[i]
                relative_vel = current_velocities[i] - current_velocities[j]
                dist = float(np.linalg.norm(relative_pos))
                combined_radius = 2 * self.agent_radius

                if dist < 1e-6:
                    continue

                # Direction from i to j
                direction = relative_pos / dist

                if dist > combined_radius:
                    # Outside collision — compute ORCA half-plane
                    # Simplified: project relative velocity onto the VO cone
                    w = relative_vel - relative_pos / self.time_horizon
                    w_len = float(np.linalg.norm(w))
                    if w_len < 1e-6:
                        continue
                    normal = w / w_len
                    # ORCA half-plane: each agent takes half the responsibility
                    point = current_velocities[i] + 0.5 * (np.dot(w, normal) * normal)
                else:
                    # Already overlapping — push apart
                    normal = direction if dist > 1e-6 else np.array([1.0, 0.0])
                    point = current_velocities[i] + 0.5 * normal * self.max_speed

                orca_planes.append((point, normal))

            # Project preferred velocity onto feasible region (iterative projection)
            vel = np.copy(preferred_velocities[i])
            for point, normal in orca_planes:
                if np.dot(vel - point, normal) < 0:
                    # Already on the feasible side
                    continue
                # Project onto the half-plane boundary
                vel = vel - np.dot(vel - point, normal) * normal

            # Clamp speed
            speed = float(np.linalg.norm(vel))
            if speed > self.max_speed:
                vel = vel / speed * self.max_speed

            new_velocities[i] = vel

        return new_velocities

    def plan_step(
        self,
        positions: np.ndarray,
        goals: np.ndarray,
        current_velocities: np.ndarray,
        dt: float = 0.1,
    ) -> PlanningResult:
        """Plan one coordination step and return resulting paths.

        Parameters:
            positions: ``(N, 2)`` current agent positions.
            goals: ``(N, 2)`` goal positions.
            current_velocities: ``(N, 2)`` current velocities.
            dt: Simulation timestep.

        Returns:
            :class:`PlanningResult` containing single-step paths.
        """
        positions = np.asarray(positions, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)

        # Preferred velocities: move toward goal at max speed
        diff = goals - positions
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-6)
        preferred = diff / dists * self.max_speed

        new_vel = self.compute_velocities(positions, current_velocities, preferred)
        new_positions = positions + new_vel * dt

        paths: dict[str, list[np.ndarray]] = {}
        for i in range(len(positions)):
            agent_id = str(i)
            paths[agent_id] = [positions[i].copy(), new_positions[i].copy()]

        cost = float(np.sum(np.linalg.norm(new_vel * dt, axis=1)))
        return PlanningResult(paths=paths, cost=cost)
