from __future__ import annotations

import math

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class SocialCostAStarRobotController(RobotController):
    """Social-cost A* planner that considers human proximity and interaction costs."""

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.goal_tolerance = float(self.cfg.get("goal_tolerance", 0.2))
        self.replan_interval = int(self.cfg.get("replan_interval", 15))  # More frequent replanning
        self.max_speed = float(self.cfg.get("max_speed", 0.8))
        self.slowdown_dist = float(self.cfg.get("slowdown_dist", 0.7))
        self.target_lookahead = int(self.cfg.get("target_lookahead", 3))
        self.velocity_smoothing = float(self.cfg.get("velocity_smoothing", 0.4))
        self.stop_speed = float(self.cfg.get("stop_speed", 0.02))

        # Social cost parameters
        self.social_radius = float(
            self.cfg.get("social_radius", 2.0)
        )  # Distance to consider humans
        self.personal_space = float(
            self.cfg.get("personal_space", 0.8)
        )  # Preferred distance from humans
        self.social_weight = float(self.cfg.get("social_weight", 3.0))  # Weight for social cost
        self.crossing_penalty = float(
            self.cfg.get("crossing_penalty", 2.0)
        )  # Cost for crossing human paths

        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)

        # Cache for social cost computation
        self._human_positions: dict[int, tuple[float, float]] = {}
        self._human_velocities: dict[int, tuple[float, float]] = {}

    def _compute_social_cost(
        self, pos: tuple[float, float], states: dict[int, AgentState]
    ) -> float:
        """Compute social cost for a position based on nearby humans."""
        if not states:
            return 0.0

        social_cost = 0.0
        px, py = pos

        for agent_id, state in states.items():
            if agent_id == self.robot_id:
                continue  # Skip self

            # Distance to human
            dx = state.x - px
            dy = state.y - py
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self.social_radius:
                continue  # Too far to matter

            # Proximity cost - higher when closer than personal space
            if dist < self.personal_space:
                proximity_cost = (
                    self.social_weight * (self.personal_space - dist) / self.personal_space
                )
                social_cost += proximity_cost * proximity_cost  # Quadratic penalty

            # Path crossing cost - penalize positions that cross human movement direction
            vx = getattr(state, "vx", 0.0)
            vy = getattr(state, "vy", 0.0)
            speed = math.sqrt(vx * vx + vy * vy)

            if speed > 0.1:  # Only consider moving humans
                # Project robot position onto human's future path
                future_hx = state.x + vx * 2.0  # Look 2 seconds ahead
                future_hy = state.y + vy * 2.0

                # Distance from robot position to human's future path
                path_dist = self._point_to_line_distance(
                    px, py, state.x, state.y, future_hx, future_hy
                )

                if path_dist < self.personal_space:
                    crossing_cost = self.crossing_penalty * (self.personal_space - path_dist)
                    social_cost += crossing_cost

        return social_cost

    def _point_to_line_distance(
        self, px: float, py: float, lx1: float, ly1: float, lx2: float, ly2: float
    ) -> float:
        """Compute distance from point to line segment."""
        ldx = lx2 - lx1
        ldy = ly2 - ly1
        length_sq = ldx * ldx + ldy * ldy

        if length_sq < 1e-8:
            return math.sqrt((px - lx1) ** 2 + (py - ly1) ** 2)

        # Project point onto line
        t = max(0.0, min(1.0, ((px - lx1) * ldx + (py - ly1) * ldy) / length_sq))
        proj_x = lx1 + t * ldx
        proj_y = ly1 + t * ldy

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def _social_astar(
        self,
        start_pos: tuple[float, float],
        goal_pos: tuple[float, float],
        states: dict[int, AgentState],
    ) -> list[tuple[float, float]]:
        """Run A* with social cost consideration."""
        # Get basic path from backend
        basic_path = self.backend.shortest_path(start_pos, goal_pos)
        if not basic_path or len(basic_path) < 2:
            return basic_path or [goal_pos]

        # For long paths, add social cost optimization
        if len(basic_path) < 3:
            return basic_path

        # Sample alternative waypoints around the basic path
        optimized_path = [basic_path[0]]  # Start with first point

        for i in range(1, len(basic_path) - 1):
            current_wp = basic_path[i]
            best_wp = current_wp
            best_cost = self._compute_social_cost(current_wp, states)

            # Sample alternatives around the current waypoint
            for angle in [
                0,
                math.pi / 4,
                math.pi / 2,
                3 * math.pi / 4,
                math.pi,
                5 * math.pi / 4,
                3 * math.pi / 2,
                7 * math.pi / 4,
            ]:
                for radius in [0.3, 0.6]:
                    alt_x = current_wp[0] + radius * math.cos(angle)
                    alt_y = current_wp[1] + radius * math.sin(angle)
                    alt_pos = (alt_x, alt_y)

                    # Check if alternative is valid (not in obstacle)
                    if self.backend.check_obstacle_collision(alt_pos):
                        continue

                    # Compute social cost for alternative
                    alt_cost = self._compute_social_cost(alt_pos, states)

                    if alt_cost < best_cost:
                        best_cost = alt_cost
                        best_wp = alt_pos

            optimized_path.append(best_wp)

        optimized_path.append(basic_path[-1])  # End with goal
        return optimized_path

    def _plan(self, position: tuple[float, float], states: dict[int, AgentState]) -> None:
        """Plan path using social-cost A* algorithm."""
        self.path = self._social_astar(position, self.goal, states)
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
        super().reset(robot_id, start, goal, backend)
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.backend = backend
        self.last_pref = (0.0, 0.0)
        self._human_positions = {}
        self._human_velocities = {}
        # Initial planning without social context
        self._plan(start, {})

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

        # Update human tracking for social cost
        self._human_positions = {
            aid: (state.x, state.y) for aid, state in states.items() if aid != self.robot_id
        }
        self._human_velocities = {
            aid: (getattr(state, "vx", 0.0), getattr(state, "vy", 0.0))
            for aid, state in states.items()
            if aid != self.robot_id
        }

        # Replan with social considerations
        if step % max(1, self.replan_interval) == 0:
            self._plan((st.x, st.y), states)
            social_cost = self._compute_social_cost((st.x, st.y), states)
            emit_event(
                "robot_social_replan",
                self.robot_id,
                {"path_len": len(self.path), "social_cost": social_cost},
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

        # Compute base velocity
        speed_scale = min(1.0, dist_target / max(self.slowdown_dist, 1e-6))

        # Reduce speed in high social cost areas
        current_social_cost = self._compute_social_cost((st.x, st.y), states)
        social_scale = max(0.3, 1.0 - min(1.0, current_social_cost / 5.0))  # Reduce to 30% min

        speed = min(st.max_speed, self.max_speed) * speed_scale * social_scale

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
                behavior="SOCIAL_NAV",
            )
        )
