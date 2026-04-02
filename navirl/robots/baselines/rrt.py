from __future__ import annotations

import math
import random

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class RRTNode:
    """Node in the RRT tree."""

    def __init__(self, position: tuple[float, float], parent: RRTNode | None = None):
        self.position = position
        self.parent = parent
        self.cost = 0.0 if parent is None else parent.cost + self._distance_to(parent)

    def _distance_to(self, other: RRTNode) -> float:
        """Calculate distance to another node."""
        return math.sqrt(
            (self.position[0] - other.position[0]) ** 2
            + (self.position[1] - other.position[1]) ** 2
        )

    def path_to_root(self) -> list[tuple[float, float]]:
        """Return path from this node to the root."""
        path = []
        current = self
        while current is not None:
            path.append(current.position)
            current = current.parent
        return list(reversed(path))


class RRTStarRobotController(RobotController):
    """RRT* (optimal RRT) planner for robot navigation."""

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.goal_tolerance = float(self.cfg.get("goal_tolerance", 0.2))
        self.replan_interval = int(self.cfg.get("replan_interval", 20))
        self.max_speed = float(self.cfg.get("max_speed", 0.8))
        self.slowdown_dist = float(self.cfg.get("slowdown_dist", 0.7))
        self.target_lookahead = int(self.cfg.get("target_lookahead", 4))
        self.velocity_smoothing = float(self.cfg.get("velocity_smoothing", 0.5))
        self.stop_speed = float(self.cfg.get("stop_speed", 0.02))

        # RRT* parameters
        self.max_iterations = int(self.cfg.get("max_iterations", 200))
        self.step_size = float(self.cfg.get("step_size", 0.3))
        self.goal_sample_rate = float(self.cfg.get("goal_sample_rate", 0.15))  # 15% goal bias
        self.rewire_radius = float(self.cfg.get("rewire_radius", 1.0))
        self.max_planning_time = float(self.cfg.get("max_planning_time", 0.5))  # seconds

        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)

        # RRT tree
        self.tree: list[RRTNode] = []
        self.map_bounds: tuple[float, float, float, float] = (0, 0, 0, 0)

    def _get_map_bounds(self) -> tuple[float, float, float, float]:
        """Get the bounds of the map for sampling."""
        if hasattr(self.backend, "map_metadata"):
            metadata = self.backend.map_metadata()
            width = metadata.get("width", 20)
            height = metadata.get("height", 20)
            return (0.0, 0.0, float(width), float(height))
        else:
            return (0.0, 0.0, 20.0, 20.0)

    def _is_valid_position(self, pos: tuple[float, float]) -> bool:
        """Check if a position is valid (not in collision)."""
        try:
            return not self.backend.check_obstacle_collision(pos)
        except Exception:
            return False

    def _random_sample(self) -> tuple[float, float]:
        """Generate a random sample point."""
        min_x, min_y, max_x, max_y = self.map_bounds

        # Goal biasing - sometimes sample near the goal
        if random.random() < self.goal_sample_rate:
            # Sample around the goal with some noise
            noise_x = random.uniform(-0.5, 0.5)
            noise_y = random.uniform(-0.5, 0.5)
            return (self.goal[0] + noise_x, self.goal[1] + noise_y)

        # Regular random sampling
        return (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

    def _nearest_node(self, position: tuple[float, float]) -> RRTNode:
        """Find the nearest node in the tree to the given position."""
        min_dist = float("inf")
        nearest = self.tree[0]

        for node in self.tree:
            dist = math.sqrt(
                (position[0] - node.position[0]) ** 2 + (position[1] - node.position[1]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def _steer(
        self, from_pos: tuple[float, float], to_pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Steer from one position towards another with step size limit."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist <= self.step_size:
            return to_pos

        # Normalize and scale
        ux = dx / dist
        uy = dy / dist
        return (from_pos[0] + ux * self.step_size, from_pos[1] + uy * self.step_size)

    def _is_path_valid(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> bool:
        """Check if path between two positions is collision-free."""
        num_checks = max(
            3, int(math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * 10)
        )

        for i in range(num_checks + 1):
            t = i / num_checks
            x = pos1[0] * (1 - t) + pos2[0] * t
            y = pos1[1] * (1 - t) + pos2[1] * t

            if not self._is_valid_position((x, y)):
                return False

        return True

    def _nodes_within_radius(self, position: tuple[float, float], radius: float) -> list[RRTNode]:
        """Find all nodes within a given radius of the position."""
        nearby_nodes = []
        for node in self.tree:
            dist = math.sqrt(
                (position[0] - node.position[0]) ** 2 + (position[1] - node.position[1]) ** 2
            )
            if dist <= radius:
                nearby_nodes.append(node)
        return nearby_nodes

    def _plan_rrt_star(
        self, start_pos: tuple[float, float], goal_pos: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Plan path using RRT* algorithm."""
        self.map_bounds = self._get_map_bounds()

        # Initialize tree with start position
        self.tree = [RRTNode(start_pos)]

        for _iteration in range(self.max_iterations):
            # Sample random point
            random_pos = self._random_sample()

            # Find nearest node
            nearest = self._nearest_node(random_pos)

            # Steer towards sample
            new_pos = self._steer(nearest.position, random_pos)

            # Check if new position is valid
            if not self._is_valid_position(new_pos):
                continue

            # Check if path to new position is valid
            if not self._is_path_valid(nearest.position, new_pos):
                continue

            # Find nearby nodes for potential rewiring
            nearby_nodes = self._nodes_within_radius(new_pos, self.rewire_radius)

            # Choose best parent among nearby nodes
            best_parent = nearest
            best_cost = nearest.cost + math.sqrt(
                (new_pos[0] - nearest.position[0]) ** 2 + (new_pos[1] - nearest.position[1]) ** 2
            )

            for node in nearby_nodes:
                if node == nearest:
                    continue

                edge_cost = math.sqrt(
                    (new_pos[0] - node.position[0]) ** 2 + (new_pos[1] - node.position[1]) ** 2
                )
                potential_cost = node.cost + edge_cost

                if potential_cost < best_cost and self._is_path_valid(node.position, new_pos):
                    best_parent = node
                    best_cost = potential_cost

            # Add new node with best parent
            new_node = RRTNode(new_pos, best_parent)
            new_node.cost = best_cost
            self.tree.append(new_node)

            # Rewire nearby nodes if paths through new node are better
            for node in nearby_nodes:
                if node == best_parent:
                    continue

                edge_cost = math.sqrt(
                    (node.position[0] - new_pos[0]) ** 2 + (node.position[1] - new_pos[1]) ** 2
                )
                potential_cost = new_node.cost + edge_cost

                if potential_cost < node.cost and self._is_path_valid(new_pos, node.position):
                    node.parent = new_node
                    node.cost = potential_cost

            # Check if we're close to goal
            dist_to_goal = math.sqrt(
                (new_pos[0] - goal_pos[0]) ** 2 + (new_pos[1] - goal_pos[1]) ** 2
            )

            if dist_to_goal <= self.goal_tolerance:
                # Try to connect to goal
                if self._is_path_valid(new_pos, goal_pos):
                    goal_node = RRTNode(goal_pos, new_node)
                    goal_node.cost = new_node.cost + dist_to_goal
                    return goal_node.path_to_root()

        # If we couldn't reach the goal, return path to closest node
        closest_node = min(
            self.tree,
            key=lambda n: math.sqrt(
                (n.position[0] - goal_pos[0]) ** 2 + (n.position[1] - goal_pos[1]) ** 2
            ),
        )

        path = closest_node.path_to_root()
        if path[-1] != goal_pos:
            path.append(goal_pos)
        return path

    def _plan(self, position: tuple[float, float]) -> None:
        """Plan path using RRT* algorithm."""
        try:
            self.path = self._plan_rrt_star(position, self.goal)
            if not self.path:
                # Fallback to backend path
                self.path = self.backend.shortest_path(position, self.goal) or [self.goal]
        except Exception:
            # Fallback on any error
            self.path = self.backend.shortest_path(position, self.goal) or [self.goal]

        self.path_idx = 0

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend,
    ) -> None:
        super().reset(robot_id, start, goal, backend)
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.backend = backend
        self.last_pref = (0.0, 0.0)
        self.tree = []
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
        super().step(step, time_s, dt, states, emit_event)

        st = states[self.robot_id]
        dist_goal = math.hypot(self.goal[0] - st.x, self.goal[1] - st.y)
        if dist_goal <= self.goal_tolerance:
            return self.validate_action(Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE"))

        # Replan periodically
        if step % max(1, self.replan_interval) == 0:
            self._plan((st.x, st.y))
            emit_event(
                "robot_rrt_replan",
                self.robot_id,
                {"path_len": len(self.path), "tree_size": len(self.tree)},
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
                behavior="RRT_NAV",
            )
        )
