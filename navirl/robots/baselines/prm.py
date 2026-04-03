from __future__ import annotations

import heapq
import logging
import math
import random

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class PRMRobotController(RobotController):
    """Probabilistic Roadmap (PRM) planner for robot navigation."""

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.goal_tolerance = float(self.cfg.get("goal_tolerance", 0.2))
        self.replan_interval = int(self.cfg.get("replan_interval", 30))
        self.max_speed = float(self.cfg.get("max_speed", 0.8))
        self.slowdown_dist = float(self.cfg.get("slowdown_dist", 0.7))
        self.target_lookahead = int(self.cfg.get("target_lookahead", 4))
        self.velocity_smoothing = float(self.cfg.get("velocity_smoothing", 0.5))
        self.stop_speed = float(self.cfg.get("stop_speed", 0.02))

        # PRM parameters
        self.num_samples = int(self.cfg.get("num_samples", 100))  # Number of random samples
        self.connection_radius = float(
            self.cfg.get("connection_radius", 1.5)
        )  # Connection distance
        self.max_connections = int(self.cfg.get("max_connections", 8))  # Max edges per node
        self.roadmap_update_interval = int(self.cfg.get("roadmap_update_interval", 100))

        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)

        # PRM data structures
        self.roadmap_nodes: list[tuple[float, float]] = []
        self.roadmap_edges: dict[int, list[int]] = {}
        self.roadmap_built = False
        self.map_bounds: tuple[float, float, float, float] = (
            0,
            0,
            0,
            0,
        )  # min_x, min_y, max_x, max_y

    def _get_map_bounds(self) -> tuple[float, float, float, float]:
        """Get the bounds of the map for sampling."""
        if hasattr(self.backend, "map_metadata"):
            metadata = self.backend.map_metadata()
            width = metadata.get("width", 20)
            height = metadata.get("height", 20)
            return (0.0, 0.0, float(width), float(height))
        else:
            # Default bounds - adjust based on your typical map sizes
            return (0.0, 0.0, 20.0, 20.0)

    def _is_valid_position(self, pos: tuple[float, float]) -> bool:
        """Check if a position is valid (not in collision)."""
        try:
            return not self.backend.check_obstacle_collision(pos)
        except (AttributeError, TypeError, ValueError) as e:
            # Log the error for debugging but return safe default
            logging.getLogger(__name__).warning(f"Collision check failed for position {pos}: {e}")
            return False

    def _build_roadmap(self) -> None:
        """Build the PRM roadmap by sampling points and connecting them."""
        if not self.backend:
            return

        self.map_bounds = self._get_map_bounds()
        min_x, min_y, max_x, max_y = self.map_bounds

        # Clear existing roadmap
        self.roadmap_nodes = []
        self.roadmap_edges = {}

        # Sample random valid points
        attempts = 0
        while len(self.roadmap_nodes) < self.num_samples and attempts < self.num_samples * 3:
            attempts += 1
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            pos = (x, y)

            if self._is_valid_position(pos):
                self.roadmap_nodes.append(pos)
                self.roadmap_edges[len(self.roadmap_nodes) - 1] = []

        # Connect nearby nodes
        for i, node_i in enumerate(self.roadmap_nodes):
            if len(self.roadmap_edges[i]) >= self.max_connections:
                continue

            # Find candidates within connection radius
            candidates = []
            for j, node_j in enumerate(self.roadmap_nodes):
                if i == j:
                    continue

                dist = math.sqrt((node_i[0] - node_j[0]) ** 2 + (node_i[1] - node_j[1]) ** 2)
                if dist <= self.connection_radius:
                    candidates.append((j, dist))

            # Sort by distance and connect to closest valid neighbors
            candidates.sort(key=lambda x: x[1])
            for j, _dist in candidates[: self.max_connections - len(self.roadmap_edges[i])]:
                if len(self.roadmap_edges[j]) >= self.max_connections:
                    continue

                # Check if edge is valid (no obstacles along the way)
                if self._is_edge_valid(node_i, self.roadmap_nodes[j]):
                    self.roadmap_edges[i].append(j)
                    self.roadmap_edges[j].append(i)

        self.roadmap_built = True

    def _is_edge_valid(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> bool:
        """Check if an edge between two positions is valid (no obstacles)."""
        # Sample points along the edge
        num_checks = max(3, int(math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * 5))
        for i in range(num_checks + 1):
            t = i / num_checks
            x = pos1[0] * (1 - t) + pos2[0] * t
            y = pos1[1] * (1 - t) + pos2[1] * t
            if not self._is_valid_position((x, y)):
                return False
        return True

    def _find_nearest_roadmap_node(self, pos: tuple[float, float]) -> int:
        """Find the nearest roadmap node to a given position."""
        if not self.roadmap_nodes:
            return -1

        min_dist = float("inf")
        nearest_idx = -1

        for i, node in enumerate(self.roadmap_nodes):
            dist = math.sqrt((pos[0] - node[0]) ** 2 + (pos[1] - node[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _dijkstra_search(self, start_idx: int, goal_idx: int) -> list[int]:
        """Find path through roadmap using Dijkstra's algorithm."""
        if start_idx == -1 or goal_idx == -1:
            return []

        # Priority queue: (cost, node_index, path)
        pq = [(0.0, start_idx, [start_idx])]
        visited: set[int] = set()

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal_idx:
                return path

            # Explore neighbors
            for neighbor in self.roadmap_edges.get(current, []):
                if neighbor in visited:
                    continue

                # Calculate edge cost
                current_pos = self.roadmap_nodes[current]
                neighbor_pos = self.roadmap_nodes[neighbor]
                edge_cost = math.sqrt(
                    (current_pos[0] - neighbor_pos[0]) ** 2
                    + (current_pos[1] - neighbor_pos[1]) ** 2
                )
                new_cost = cost + edge_cost
                new_path = path + [neighbor]

                heapq.heappush(pq, (new_cost, neighbor, new_path))

        return []

    def _plan_via_roadmap(
        self, start_pos: tuple[float, float], goal_pos: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Plan path using the PRM roadmap."""
        if not self.roadmap_built:
            self._build_roadmap()

        if not self.roadmap_nodes:
            # Fallback to direct path
            if self._is_edge_valid(start_pos, goal_pos):
                return [start_pos, goal_pos]
            else:
                return self.backend.shortest_path(start_pos, goal_pos) or [goal_pos]

        # Find nearest nodes to start and goal
        start_idx = self._find_nearest_roadmap_node(start_pos)
        goal_idx = self._find_nearest_roadmap_node(goal_pos)

        if start_idx == -1 or goal_idx == -1:
            return self.backend.shortest_path(start_pos, goal_pos) or [goal_pos]

        # Try direct connection first
        start_node = self.roadmap_nodes[start_idx]
        goal_node = self.roadmap_nodes[goal_idx]

        if self._is_edge_valid(start_pos, start_node) and self._is_edge_valid(goal_node, goal_pos):
            # Search through roadmap
            node_path = self._dijkstra_search(start_idx, goal_idx)

            if node_path:
                # Convert node indices to positions
                world_path = [start_pos]
                for node_idx in node_path[1:-1]:  # Skip start and goal nodes
                    world_path.append(self.roadmap_nodes[node_idx])
                world_path.append(goal_pos)
                return world_path

        # Fallback to backend shortest path
        return self.backend.shortest_path(start_pos, goal_pos) or [goal_pos]

    def _plan(self, position: tuple[float, float]) -> None:
        """Plan path using PRM algorithm."""
        self.path = self._plan_via_roadmap(position, self.goal)
        if not self.path:
            self.path = [self.goal]
        self.path_idx = 0

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend,
    ) -> None:
        """Reset the PRM controller for a new planning episode.

        Initializes the robot's state and forces a roadmap rebuild for the new
        environment. This ensures the roadmap is appropriate for the current
        start/goal configuration and world state.

        Parameters
        ----------
        robot_id : int
            Unique identifier for this robot.
        start : tuple[float, float]
            Starting position (x, y).
        goal : tuple[float, float]
            Target goal position (x, y).
        backend : Backend
            The simulation backend providing collision checking and world info.
        """
        super().reset(robot_id, start, goal, backend)
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.backend = backend
        self.last_pref = (0.0, 0.0)
        self.roadmap_built = False  # Force roadmap rebuild for new environment
        self._plan(start)

    def _current_target(self) -> tuple[float, float]:
        if self.path_idx >= len(self.path):
            return self.goal
        look_idx = min(len(self.path) - 1, self.path_idx + max(0, self.target_lookahead - 1))
        return self.path[look_idx]

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        """Compute the robot's next action using PRM path following.

        Follows the pre-computed PRM path toward the goal, advancing waypoints
        when close enough and replanning periodically. Applies velocity smoothing
        and respects maximum speed constraints.

        Parameters
        ----------
        step : int
            Current simulation step number.
        time_s : float
            Current simulation time in seconds.
        dt : float
            Simulation time step in seconds.
        states : dict[int, AgentState]
            Current states of all agents in the simulation.
        emit_event : EventSink
            Event sink for logging and debugging.

        Returns
        -------
        Action
            The computed action with preferred velocity and behavior flag.
        """
        super().step(step, time_s, dt, states, emit_event)

        st = states[self.robot_id]
        dist_goal = math.hypot(self.goal[0] - st.x, self.goal[1] - st.y)
        if dist_goal <= self.goal_tolerance:
            return self.validate_action(Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE"))

        # Periodically rebuild roadmap for dynamic environments
        if step % self.roadmap_update_interval == 0:
            self.roadmap_built = False

        # Replan periodically
        if step % max(1, self.replan_interval) == 0:
            self._plan((st.x, st.y))
            emit_event(
                "robot_prm_replan",
                self.robot_id,
                {
                    "path_len": len(self.path),
                    "roadmap_nodes": len(self.roadmap_nodes),
                    "roadmap_edges": sum(len(edges) for edges in self.roadmap_edges.values()),
                },
            )

        target = self._current_target()
        dist_target = math.hypot(target[0] - st.x, target[1] - st.y)

        # Advance waypoint if close enough
        if dist_target <= self.goal_tolerance and self.path_idx < len(self.path) - 1:
            self.path_idx += 1
            target = self._current_target()
            dist_target = math.hypot(target[0] - st.x, target[1] - st.y)

        if dist_target < 1e-8:
            return self.validate_action(Action(pref_vx=0.0, pref_vy=0.0, behavior="WAIT"))

        # Compute velocity
        speed_scale = min(1.0, dist_target / max(self.slowdown_dist, 1e-6))
        speed = min(st.max_speed, self.max_speed) * speed_scale

        if dist_target > 0:
            ux = (target[0] - st.x) / dist_target
            uy = (target[1] - st.y) / dist_target
            vx = ux * speed
            vy = uy * speed
        else:
            vx = vy = 0.0

        # Velocity smoothing
        alpha = max(0.0, min(1.0, self.velocity_smoothing))
        vx = self.last_pref[0] * (1.0 - alpha) + vx * alpha
        vy = self.last_pref[1] * (1.0 - alpha) + vy * alpha

        if math.hypot(vx, vy) < self.stop_speed and dist_target < self.goal_tolerance:
            vx, vy = 0.0, 0.0

        self.last_pref = (vx, vy)

        return self.validate_action(
            Action(
                pref_vx=vx,
                pref_vy=vy,
                behavior="PRM_NAV",
            )
        )
