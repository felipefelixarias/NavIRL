"""Potential Fields robot controller for reactive navigation with artificial potential functions."""

from __future__ import annotations

import math

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class PotentialFieldsController(RobotController):
    """Robot controller using artificial potential fields for reactive navigation."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg

        # Navigation parameters
        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.2))
        self.max_speed = float(cfg.get("max_speed", 0.8))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.6))

        # Potential field parameters
        self.attractive_gain = float(cfg.get("attractive_gain", 2.0))
        self.repulsive_gain = float(cfg.get("repulsive_gain", 1.5))
        self.repulsive_range = float(cfg.get("repulsive_range", 1.0))

        # Human-specific parameters
        self.human_repulsive_gain = float(cfg.get("human_repulsive_gain", 2.5))
        self.human_repulsive_range = float(cfg.get("human_repulsive_range", 1.5))
        self.social_comfort_gain = float(cfg.get("social_comfort_gain", 1.0))

        # Dynamic obstacle parameters
        self.velocity_obstacle_gain = float(cfg.get("velocity_obstacle_gain", 3.0))
        self.prediction_horizon = float(cfg.get("prediction_horizon", 2.0))

        # Field parameters
        self.field_saturation_distance = float(cfg.get("field_saturation_distance", 0.1))
        self.oscillation_damping = float(cfg.get("oscillation_damping", 0.8))
        self.force_limit = float(cfg.get("force_limit", 5.0))

        # State variables
        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.last_pref = (0.0, 0.0)
        self.last_force = (0.0, 0.0)

        # Oscillation detection
        self.velocity_history: list[tuple[float, float]] = []
        self.max_history_length = 10

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
        self.last_force = (0.0, 0.0)
        self.velocity_history.clear()

    def _calculate_attractive_force(self, robot_pos: tuple[float, float]) -> tuple[float, float]:
        """Calculate attractive force toward the goal."""
        dx = self.goal[0] - robot_pos[0]
        dy = self.goal[1] - robot_pos[1]
        distance = math.hypot(dx, dy)

        if distance < self.field_saturation_distance:
            return (0.0, 0.0)

        # Linear attractive force
        force_magnitude = self.attractive_gain * distance
        fx = force_magnitude * dx / distance
        fy = force_magnitude * dy / distance

        return (fx, fy)

    def _calculate_repulsive_force_static(
        self,
        robot_pos: tuple[float, float],
        obstacle_pos: tuple[float, float],
        gain: float,
        range_limit: float
    ) -> tuple[float, float]:
        """Calculate repulsive force from a static obstacle."""
        dx = robot_pos[0] - obstacle_pos[0]
        dy = robot_pos[1] - obstacle_pos[1]
        distance = math.hypot(dx, dy)

        if distance >= range_limit or distance < self.field_saturation_distance:
            return (0.0, 0.0)

        # Quadratic repulsive force
        force_magnitude = gain * (1.0 / distance - 1.0 / range_limit) * (1.0 / distance**2)
        fx = force_magnitude * dx / distance
        fy = force_magnitude * dy / distance

        return (fx, fy)

    def _calculate_human_repulsive_force(
        self,
        robot_pos: tuple[float, float],
        human_state: AgentState
    ) -> tuple[float, float]:
        """Calculate enhanced repulsive force from humans considering velocity."""
        human_pos = (human_state.x, human_state.y)
        human_vel = (human_state.vx, human_state.vy)

        # Static repulsive force
        static_force = self._calculate_repulsive_force_static(
            robot_pos,
            human_pos,
            self.human_repulsive_gain,
            self.human_repulsive_range
        )

        # Predict future human position
        predicted_x = human_state.x + human_vel[0] * self.prediction_horizon
        predicted_y = human_state.y + human_vel[1] * self.prediction_horizon
        predicted_pos = (predicted_x, predicted_y)

        # Velocity-based repulsive force from predicted position
        velocity_force = self._calculate_repulsive_force_static(
            robot_pos,
            predicted_pos,
            self.velocity_obstacle_gain,
            self.human_repulsive_range * 1.5
        )

        # Social comfort force (gentler, longer range)
        social_force = self._calculate_repulsive_force_static(
            robot_pos,
            human_pos,
            self.social_comfort_gain,
            self.human_repulsive_range * 2.0
        )

        # Combine forces
        total_fx = static_force[0] + velocity_force[0] * 0.5 + social_force[0] * 0.3
        total_fy = static_force[1] + velocity_force[1] * 0.5 + social_force[1] * 0.3

        return (total_fx, total_fy)

    def _get_nearby_obstacles(self, robot_pos: tuple[float, float]) -> list[tuple[float, float]]:
        """Get positions of nearby obstacles from the environment."""
        obstacles = []

        if not self.backend:
            return obstacles

        # Sample points around robot to detect obstacles
        search_radius = self.repulsive_range * 2.0
        num_samples = 16

        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            check_x = robot_pos[0] + search_radius * math.cos(angle)
            check_y = robot_pos[1] + search_radius * math.sin(angle)

            # Check if this position is an obstacle
            try:
                if hasattr(self.backend, 'is_valid_position'):
                    if not self.backend.is_valid_position(check_x, check_y):
                        obstacles.append((check_x, check_y))
                elif hasattr(self.backend, 'environment'):
                    if hasattr(self.backend.environment, 'is_valid'):
                        if not self.backend.environment.is_valid(check_x, check_y):
                            obstacles.append((check_x, check_y))
            except Exception:
                continue

        return obstacles

    def _detect_oscillation(self) -> bool:
        """Detect if robot is oscillating."""
        if len(self.velocity_history) < 4:
            return False

        # Check if velocity direction is alternating
        recent_velocities = self.velocity_history[-4:]
        directions = []

        for vx, vy in recent_velocities:
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            angle = math.atan2(vy, vx)
            directions.append(angle)

        if len(directions) < 3:
            return False

        # Check for alternating directions
        oscillating = True
        for i in range(1, len(directions) - 1):
            prev_diff = abs(directions[i] - directions[i-1])
            next_diff = abs(directions[i+1] - directions[i])

            # Allow for some tolerance in angle changes
            if prev_diff < math.pi / 4 and next_diff < math.pi / 4:
                oscillating = False
                break

        return oscillating

    def _apply_oscillation_damping(self, force: tuple[float, float]) -> tuple[float, float]:
        """Apply damping to reduce oscillations."""
        fx, fy = force

        # If oscillating, reduce force magnitude and add damping
        if self._detect_oscillation():
            # Reduce force magnitude
            fx *= self.oscillation_damping
            fy *= self.oscillation_damping

            # Add small random perturbation to break deadlock
            import random
            perturbation_strength = 0.1
            fx += (random.random() - 0.5) * perturbation_strength
            fy += (random.random() - 0.5) * perturbation_strength

        return (fx, fy)

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        """Execute one step of potential fields navigation."""
        robot_state = states[self.robot_id]
        robot_pos = (robot_state.x, robot_state.y)

        # Check if goal reached
        dist_goal = math.hypot(self.goal[0] - robot_state.x, self.goal[1] - robot_state.y)
        if dist_goal <= self.goal_tolerance:
            emit_event("robot_goal_reached", self.robot_id, {"planner": "PotentialFields"})
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        # Calculate attractive force toward goal
        attractive_force = self._calculate_attractive_force(robot_pos)

        # Calculate repulsive forces from static obstacles
        static_repulsive_force = (0.0, 0.0)
        nearby_obstacles = self._get_nearby_obstacles(robot_pos)

        for obstacle_pos in nearby_obstacles:
            obs_force = self._calculate_repulsive_force_static(
                robot_pos,
                obstacle_pos,
                self.repulsive_gain,
                self.repulsive_range
            )
            static_repulsive_force = (
                static_repulsive_force[0] + obs_force[0],
                static_repulsive_force[1] + obs_force[1]
            )

        # Calculate repulsive forces from humans
        human_repulsive_force = (0.0, 0.0)
        for human_id, human_state in states.items():
            if human_id == self.robot_id:
                continue

            human_force = self._calculate_human_repulsive_force(robot_pos, human_state)
            human_repulsive_force = (
                human_repulsive_force[0] + human_force[0],
                human_repulsive_force[1] + human_force[1]
            )

        # Combine all forces
        total_fx = attractive_force[0] + static_repulsive_force[0] + human_repulsive_force[0]
        total_fy = attractive_force[1] + static_repulsive_force[1] + human_repulsive_force[1]

        # Limit force magnitude
        force_magnitude = math.hypot(total_fx, total_fy)
        if force_magnitude > self.force_limit:
            scale = self.force_limit / force_magnitude
            total_fx *= scale
            total_fy *= scale

        # Apply oscillation damping
        total_fx, total_fy = self._apply_oscillation_damping((total_fx, total_fy))

        # Convert force to velocity
        force_to_velocity_scale = 0.5  # Tune this based on dynamics
        desired_vx = total_fx * force_to_velocity_scale
        desired_vy = total_fy * force_to_velocity_scale

        # Limit to maximum speed
        desired_speed = math.hypot(desired_vx, desired_vy)
        if desired_speed > self.max_speed:
            scale = self.max_speed / desired_speed
            desired_vx *= scale
            desired_vy *= scale

        # Apply velocity smoothing
        alpha = max(0.0, min(1.0, self.velocity_smoothing))
        final_vx = self.last_pref[0] * (1.0 - alpha) + desired_vx * alpha
        final_vy = self.last_pref[1] * (1.0 - alpha) + desired_vy * alpha

        # Update history for oscillation detection
        self.velocity_history.append((final_vx, final_vy))
        if len(self.velocity_history) > self.max_history_length:
            self.velocity_history.pop(0)

        self.last_pref = (final_vx, final_vy)
        self.last_force = (total_fx, total_fy)

        # Determine behavior
        behavior = "POTENTIAL_FIELDS"
        if desired_speed < 0.1:
            behavior = "WAIT"
        elif abs(human_repulsive_force[0]) + abs(human_repulsive_force[1]) > 0.5:
            behavior = "AVOID_HUMANS"

        # Emit debugging info
        emit_event("potential_fields_debug", self.robot_id, {
            "attractive_force": attractive_force,
            "static_repulsive": static_repulsive_force,
            "human_repulsive": human_repulsive_force,
            "total_force": (total_fx, total_fy),
            "nearby_obstacles": len(nearby_obstacles),
            "oscillating": self._detect_oscillation()
        })

        return Action(
            pref_vx=final_vx,
            pref_vy=final_vy,
            behavior=behavior,
        )