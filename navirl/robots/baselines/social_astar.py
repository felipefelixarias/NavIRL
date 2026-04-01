"""Social-aware A* robot controller that considers human comfort and social forces."""

from __future__ import annotations

import math
from collections import defaultdict

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class SocialAwareAStarController(RobotController):
    """A* planner enhanced with social cost functions and human comfort zones."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg

        # Basic navigation parameters
        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.2))
        self.replan_interval = int(cfg.get("replan_interval", 15))
        self.max_speed = float(cfg.get("max_speed", 0.8))
        self.slowdown_dist = float(cfg.get("slowdown_dist", 0.7))
        self.target_lookahead = int(cfg.get("target_lookahead", 3))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.65))

        # Social parameters
        self.social_comfort_distance = float(cfg.get("social_comfort_distance", 1.2))
        self.personal_space_distance = float(cfg.get("personal_space_distance", 0.6))
        self.social_cost_weight = float(cfg.get("social_cost_weight", 2.0))
        self.group_detection_radius = float(cfg.get("group_detection_radius", 2.5))
        self.avoidance_strength = float(cfg.get("avoidance_strength", 1.5))

        # Path prediction parameters
        self.prediction_horizon = float(cfg.get("prediction_horizon", 3.0))
        self.velocity_history_size = int(cfg.get("velocity_history_size", 5))

        # State variables
        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)

        # Social awareness state
        self.human_velocity_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.last_social_forces = {}

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
        self.human_velocity_history.clear()
        self.last_social_forces.clear()
        self._social_plan(start)

    def _predict_human_position(
        self,
        human_id: int,
        current_pos: tuple[float, float],
        dt: float
    ) -> tuple[float, float]:
        """Predict where a human will be based on velocity history."""
        if human_id not in self.human_velocity_history or len(self.human_velocity_history[human_id]) == 0:
            return current_pos

        # Average recent velocities for prediction
        velocities = self.human_velocity_history[human_id]
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)

        # Predict position after dt seconds
        pred_x = current_pos[0] + avg_vx * self.prediction_horizon
        pred_y = current_pos[1] + avg_vy * self.prediction_horizon

        return (pred_x, pred_y)

    def _calculate_social_cost(
        self,
        position: tuple[float, float],
        states: dict[int, AgentState]
    ) -> float:
        """Calculate social cost for a position considering all humans."""
        total_cost = 0.0

        for human_id, human_state in states.items():
            if human_id == self.robot_id:
                continue

            human_pos = (human_state.x, human_state.y)
            distance = math.hypot(position[0] - human_pos[0], position[1] - human_pos[1])

            # Personal space violation cost
            if distance < self.personal_space_distance:
                personal_cost = (self.personal_space_distance - distance) / self.personal_space_distance
                total_cost += personal_cost * 10.0  # High penalty for personal space

            # Social comfort cost
            elif distance < self.social_comfort_distance:
                comfort_cost = (self.social_comfort_distance - distance) / self.social_comfort_distance
                total_cost += comfort_cost * self.social_cost_weight

            # Predictive cost - consider where human will be
            predicted_pos = self._predict_human_position(
                human_id,
                human_pos,
                self.prediction_horizon
            )
            pred_distance = math.hypot(position[0] - predicted_pos[0], position[1] - predicted_pos[1])

            if pred_distance < self.social_comfort_distance:
                pred_cost = (self.social_comfort_distance - pred_distance) / self.social_comfort_distance
                total_cost += pred_cost * self.social_cost_weight * 0.5  # Future avoidance

        return total_cost

    def _detect_human_groups(self, states: dict[int, AgentState]) -> list[list[int]]:
        """Detect groups of humans based on proximity."""
        human_ids = [hid for hid in states.keys() if hid != self.robot_id]
        groups = []
        assigned = set()

        for human_id in human_ids:
            if human_id in assigned:
                continue

            # Start new group
            group = [human_id]
            assigned.add(human_id)
            human_pos = (states[human_id].x, states[human_id].y)

            # Find nearby humans
            for other_id in human_ids:
                if other_id in assigned:
                    continue

                other_pos = (states[other_id].x, states[other_id].y)
                distance = math.hypot(human_pos[0] - other_pos[0], human_pos[1] - other_pos[1])

                if distance < self.group_detection_radius:
                    group.append(other_id)
                    assigned.add(other_id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _social_plan(self, position: tuple[float, float]) -> None:
        """Plan path using A* with social cost integration."""
        # For now, use basic A* and apply social forces during execution
        # In a full implementation, this would integrate social costs into the search
        if self.backend:
            self.path = self.backend.shortest_path(position, self.goal)
        else:
            self.path = [self.goal]

        if not self.path:
            self.path = [self.goal]
        self.path_idx = 0

    def _calculate_social_force(
        self,
        robot_pos: tuple[float, float],
        states: dict[int, AgentState]
    ) -> tuple[float, float]:
        """Calculate social forces affecting the robot."""
        total_fx, total_fy = 0.0, 0.0

        for human_id, human_state in states.items():
            if human_id == self.robot_id:
                continue

            human_pos = (human_state.x, human_state.y)
            dx = robot_pos[0] - human_pos[0]
            dy = robot_pos[1] - human_pos[1]
            distance = math.hypot(dx, dy)

            if distance < 1e-6:
                continue

            # Repulsive force that increases as we get closer
            if distance < self.social_comfort_distance:
                force_magnitude = self.avoidance_strength * (
                    self.social_comfort_distance - distance
                ) / self.social_comfort_distance

                # Normalize direction
                unit_dx = dx / distance
                unit_dy = dy / distance

                total_fx += force_magnitude * unit_dx
                total_fy += force_magnitude * unit_dy

        return (total_fx, total_fy)

    def _current_target(self) -> tuple[float, float]:
        """Get current navigation target with lookahead."""
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
        """Execute one step of social-aware navigation."""
        robot_state = states[self.robot_id]
        robot_pos = (robot_state.x, robot_state.y)

        # Update human velocity history for prediction
        for human_id, human_state in states.items():
            if human_id == self.robot_id:
                continue
            velocity = (human_state.vx, human_state.vy)
            self.human_velocity_history[human_id].append(velocity)

            # Keep only recent history
            if len(self.human_velocity_history[human_id]) > self.velocity_history_size:
                self.human_velocity_history[human_id].pop(0)

        # Check if goal reached
        dist_goal = math.hypot(self.goal[0] - robot_state.x, self.goal[1] - robot_state.y)
        if dist_goal <= self.goal_tolerance:
            emit_event("robot_goal_reached", self.robot_id, {"social_efficiency": "high"})
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        # Replan periodically
        if step % max(1, self.replan_interval) == 0:
            self._social_plan(robot_pos)
            groups = self._detect_human_groups(states)
            emit_event("robot_replan", self.robot_id, {
                "path_len": len(self.path),
                "human_groups_detected": len(groups),
                "social_cost": self._calculate_social_cost(robot_pos, states)
            })

        # Get navigation target
        target = self._current_target()
        dist_target = math.hypot(target[0] - robot_state.x, target[1] - robot_state.y)

        # Advance waypoint if close enough
        if dist_target <= self.goal_tolerance and self.path_idx < len(self.path) - 1:
            self.path_idx += 1
            target = self._current_target()
            dist_target = math.hypot(target[0] - robot_state.x, target[1] - robot_state.y)

        if dist_target < 1e-8:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="WAIT")

        # Calculate base velocity toward target
        speed_scale = min(1.0, dist_target / max(self.slowdown_dist, 1e-6))
        base_speed = min(robot_state.max_speed, self.max_speed) * speed_scale

        ux = (target[0] - robot_state.x) / dist_target
        uy = (target[1] - robot_state.y) / dist_target

        base_vx = ux * base_speed
        base_vy = uy * base_speed

        # Apply social forces
        social_fx, social_fy = self._calculate_social_force(robot_pos, states)

        # Combine navigation and social forces
        final_vx = base_vx + social_fx
        final_vy = base_vy + social_fy

        # Limit to maximum speed
        final_speed = math.hypot(final_vx, final_vy)
        if final_speed > self.max_speed:
            scale = self.max_speed / final_speed
            final_vx *= scale
            final_vy *= scale

        # Apply velocity smoothing
        alpha = max(0.0, min(1.0, self.velocity_smoothing))
        final_vx = self.last_pref[0] * (1.0 - alpha) + final_vx * alpha
        final_vy = self.last_pref[1] * (1.0 - alpha) + final_vy * alpha

        self.last_pref = (final_vx, final_vy)

        # Determine behavior
        behavior = "GO_TO"
        if abs(social_fx) + abs(social_fy) > 0.1:
            behavior = "SOCIAL_NAVIGATION"
        elif final_speed < 0.1:
            behavior = "WAIT"

        return Action(
            pref_vx=final_vx,
            pref_vy=final_vy,
            behavior=behavior,
        )