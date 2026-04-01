"""Probabilistic Roadmap (PRM) robot controller for complex environment navigation."""

from __future__ import annotations

import math
import random

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class Node:
    """A node in the PRM roadmap."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.neighbors: list[int] = []
        self.g_cost = float("inf")
        self.parent = -1

    def distance_to(self, other: Node) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class PRMController(RobotController):
    """PRM-based robot controller that builds and queries a probabilistic roadmap."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg

        # Navigation parameters
        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.2))
        self.replan_interval = int(cfg.get("replan_interval", 30))
        self.max_speed = float(cfg.get("max_speed", 0.8))
        self.slowdown_dist = float(cfg.get("slowdown_dist", 0.8))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.7))

        # PRM parameters
        self.num_samples = int(cfg.get("num_samples", 500))
        self.connection_radius = float(cfg.get("connection_radius", 2.0))
        self.max_connections = int(cfg.get("max_connections", 10))
        self.roadmap_bounds = cfg.get(
            "roadmap_bounds", {"x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5}
        )

        # Dynamic parameters
        self.dynamic_resampling = bool(cfg.get("dynamic_resampling", True))
        self.obstacle_clearance = float(cfg.get("obstacle_clearance", 0.3))
        self.roadmap_rebuild_interval = int(cfg.get("roadmap_rebuild_interval", 200))

        # State variables
        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)

        # PRM roadmap
        self.roadmap: list[Node] = []
        self.roadmap_built = False
        self.last_roadmap_build_step = 0

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend,
    ) -> None:
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.backend = backend
        self.last_pref = (0.0, 0.0)
        self.roadmap_built = False
        self.last_roadmap_build_step = 0
        self._build_initial_roadmap()

    def _is_collision_free(self, pos: tuple[float, float]) -> bool:
        """Check if a position is collision-free."""
        if not self.backend:
            return True

        # Use backend collision checking if available
        try:
            # Check if position is in obstacle space
            x, y = pos
            if hasattr(self.backend, "is_valid_position"):
                return self.backend.is_valid_position(x, y)
            elif hasattr(self.backend, "environment") and hasattr(
                self.backend.environment, "is_valid"
            ):
                return self.backend.environment.is_valid(x, y)
            else:
                # Fallback: assume valid if no collision checking available
                return True
        except Exception:
            return True

    def _is_path_collision_free(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> bool:
        """Check if a straight-line path between two positions is collision-free."""
        if not self.backend:
            return True

        # Sample points along the path
        num_checks = max(5, int(math.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1]) / 0.1))
        for i in range(num_checks + 1):
            t = i / num_checks
            x = pos1[0] + t * (pos2[0] - pos1[0])
            y = pos1[1] + t * (pos2[1] - pos1[1])

            if not self._is_collision_free((x, y)):
                return False

        return True

    def _sample_free_space(self) -> tuple[float, float]:
        """Sample a random point in free space."""
        max_attempts = 100
        bounds = self.roadmap_bounds

        for _ in range(max_attempts):
            x = random.uniform(bounds["x_min"], bounds["x_max"])
            y = random.uniform(bounds["y_min"], bounds["y_max"])

            if self._is_collision_free((x, y)):
                return (x, y)

        # Fallback to bounds center if no free space found
        return ((bounds["x_min"] + bounds["x_max"]) / 2, (bounds["y_min"] + bounds["y_max"]) / 2)

    def _build_initial_roadmap(self) -> None:
        """Build the initial PRM roadmap."""
        self.roadmap.clear()

        # Sample nodes in free space
        for _ in range(self.num_samples):
            x, y = self._sample_free_space()
            self.roadmap.append(Node(x, y))

        # Connect nearby nodes
        for i, node in enumerate(self.roadmap):
            # Find nearby nodes within connection radius
            nearby_indices = []
            for j, other_node in enumerate(self.roadmap):
                if i != j:
                    distance = node.distance_to(other_node)
                    if distance <= self.connection_radius:
                        nearby_indices.append((j, distance))

            # Sort by distance and connect to closest nodes
            nearby_indices.sort(key=lambda x: x[1])
            connections_made = 0

            for neighbor_idx, _distance in nearby_indices:
                if connections_made >= self.max_connections:
                    break

                neighbor_pos = (self.roadmap[neighbor_idx].x, self.roadmap[neighbor_idx].y)
                node_pos = (node.x, node.y)

                if self._is_path_collision_free(node_pos, neighbor_pos):
                    node.neighbors.append(neighbor_idx)
                    connections_made += 1

        self.roadmap_built = True

    def _add_temporary_nodes(self, positions: list[tuple[float, float]]) -> list[int]:
        """Add temporary nodes (start/goal) to the roadmap."""
        temp_indices = []

        for pos in positions:
            if not self._is_collision_free(pos):
                continue

            # Create temporary node
            temp_node = Node(pos[0], pos[1])
            temp_idx = len(self.roadmap)
            self.roadmap.append(temp_node)
            temp_indices.append(temp_idx)

            # Connect to nearby nodes
            connections_made = 0
            for i, existing_node in enumerate(self.roadmap[:-1]):  # Exclude the temp node itself
                if connections_made >= self.max_connections:
                    break

                distance = temp_node.distance_to(existing_node)
                if distance <= self.connection_radius:
                    existing_pos = (existing_node.x, existing_node.y)
                    if self._is_path_collision_free(pos, existing_pos):
                        temp_node.neighbors.append(i)
                        existing_node.neighbors.append(temp_idx)
                        connections_made += 1

        return temp_indices

    def _dijkstra_search(self, start_idx: int, goal_idx: int) -> list[tuple[float, float]]:
        """Find shortest path using Dijkstra's algorithm."""
        if start_idx >= len(self.roadmap) or goal_idx >= len(self.roadmap):
            return []

        # Reset all nodes
        for node in self.roadmap:
            node.g_cost = float("inf")
            node.parent = -1

        # Initialize start node
        self.roadmap[start_idx].g_cost = 0.0
        open_set = {start_idx}
        closed_set: set[int] = set()

        while open_set:
            # Find node with minimum cost
            current_idx = min(open_set, key=lambda idx: self.roadmap[idx].g_cost)
            open_set.remove(current_idx)
            closed_set.add(current_idx)

            # Goal reached
            if current_idx == goal_idx:
                break

            current_node = self.roadmap[current_idx]

            # Check all neighbors
            for neighbor_idx in current_node.neighbors:
                if neighbor_idx in closed_set:
                    continue

                neighbor_node = self.roadmap[neighbor_idx]
                edge_cost = current_node.distance_to(neighbor_node)
                new_cost = current_node.g_cost + edge_cost

                if new_cost < neighbor_node.g_cost:
                    neighbor_node.g_cost = new_cost
                    neighbor_node.parent = current_idx
                    open_set.add(neighbor_idx)

        # Reconstruct path
        path = []
        current_idx = goal_idx
        while current_idx != -1:
            node = self.roadmap[current_idx]
            path.append((node.x, node.y))
            current_idx = node.parent

        path.reverse()
        return path if path else []

    def _prm_plan(self, start_pos: tuple[float, float], goal_pos: tuple[float, float]) -> None:
        """Plan path using PRM roadmap."""
        if not self.roadmap_built:
            self._build_initial_roadmap()

        # Add temporary start and goal nodes
        temp_indices = self._add_temporary_nodes([start_pos, goal_pos])

        if len(temp_indices) < 2:
            # Fallback to direct path if can't connect to roadmap
            self.path = [goal_pos]
            self.path_idx = 0
            return

        start_idx, goal_idx = temp_indices[0], temp_indices[1]

        # Find path through roadmap
        roadmap_path = self._dijkstra_search(start_idx, goal_idx)

        # Clean up temporary nodes
        for temp_idx in reversed(temp_indices):
            # Remove connections to temporary nodes
            for node in self.roadmap[:temp_idx]:
                if temp_idx in node.neighbors:
                    node.neighbors.remove(temp_idx)
            # Remove temporary node
            self.roadmap.pop(temp_idx)

        # Set path
        if roadmap_path:
            self.path = roadmap_path
        else:
            self.path = [goal_pos]

        self.path_idx = 0

    def _current_target(self) -> tuple[float, float]:
        """Get current navigation target."""
        if self.path_idx >= len(self.path):
            return self.goal
        return self.path[self.path_idx]

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        """Execute one step of PRM-based navigation."""
        robot_state = states[self.robot_id]
        robot_pos = (robot_state.x, robot_state.y)

        # Check if goal reached
        dist_goal = math.hypot(self.goal[0] - robot_state.x, self.goal[1] - robot_state.y)
        if dist_goal <= self.goal_tolerance:
            emit_event(
                "robot_goal_reached",
                self.robot_id,
                {"planner": "PRM", "roadmap_size": len(self.roadmap)},
            )
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        # Rebuild roadmap periodically if dynamic resampling is enabled
        if (
            self.dynamic_resampling
            and step - self.last_roadmap_build_step > self.roadmap_rebuild_interval
        ):
            self._build_initial_roadmap()
            self.last_roadmap_build_step = step

        # Replan periodically
        if step % max(1, self.replan_interval) == 0:
            self._prm_plan(robot_pos, self.goal)
            emit_event(
                "robot_replan",
                self.robot_id,
                {"path_len": len(self.path), "roadmap_size": len(self.roadmap), "planner": "PRM"},
            )

        # Get current target
        target = self._current_target()
        dist_target = math.hypot(target[0] - robot_state.x, target[1] - robot_state.y)

        # Advance waypoint if close enough
        if dist_target <= self.goal_tolerance and self.path_idx < len(self.path) - 1:
            self.path_idx += 1
            target = self._current_target()
            dist_target = math.hypot(target[0] - robot_state.x, target[1] - robot_state.y)

        if dist_target < 1e-8:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="WAIT")

        # Calculate velocity
        speed_scale = min(1.0, dist_target / max(self.slowdown_dist, 1e-6))
        speed = min(robot_state.max_speed, self.max_speed) * speed_scale

        ux = (target[0] - robot_state.x) / dist_target
        uy = (target[1] - robot_state.y) / dist_target
        vx = ux * speed
        vy = uy * speed

        # Apply velocity smoothing
        alpha = max(0.0, min(1.0, self.velocity_smoothing))
        vx = self.last_pref[0] * (1.0 - alpha) + vx * alpha
        vy = self.last_pref[1] * (1.0 - alpha) + vy * alpha

        self.last_pref = (vx, vy)

        return Action(
            pref_vx=vx,
            pref_vy=vy,
            behavior="PRM_NAVIGATION",
        )
