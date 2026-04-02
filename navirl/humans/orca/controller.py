from __future__ import annotations

import logging
import math

from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController

logger = logging.getLogger(__name__)


def _normalize(vx: float, vy: float) -> tuple[float, float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-8:
        return 0.0, 0.0, 0.0
    return vx / n, vy / n, n


class ORCAHumanController(HumanController):
    """
    Baseline human controller with map-aware waypoint following for ORCA.

    This controller implements goal-seeking behavior with path planning,
    velocity smoothing, and goal swapping when agents reach their destinations.

    Configuration parameters:
        goal_tolerance (float): Distance threshold to consider goal reached (0.01-10.0)
        waypoint_tolerance (float): Distance threshold to advance to next waypoint
        lookahead (int): Number of waypoints to look ahead for smoothing (1-100)
        min_speed (float): Minimum movement speed when not stopped
        slowdown_dist (float): Distance to begin slowing down before goal
        velocity_smoothing (float): Smoothing factor for velocity transitions (0.0-1.0)
        stop_speed (float): Speed threshold to stop movement
    """

    def __init__(self, cfg: dict | None = None):
        # Let parent handle config validation
        super().__init__(cfg)

        # Extract validated parameters with safe defaults
        self.goal_tolerance = self.cfg.get("goal_tolerance", 0.22)
        self.waypoint_tolerance = self.cfg.get("waypoint_tolerance", 0.2)
        self.lookahead = self.cfg.get("lookahead", 4)
        self.min_speed = self.cfg.get("min_speed", 0.04)
        self.slowdown_dist = self.cfg.get("slowdown_dist", 0.9)
        self.velocity_smoothing = self.cfg.get("velocity_smoothing", 0.25)
        self.stop_speed = self.cfg.get("stop_speed", 0.03)

        self.human_ids: list[int] = []
        self.starts: dict[int, tuple[float, float]] = {}
        self.goals: dict[int, tuple[float, float]] = {}
        self.backend = None

        self.paths: dict[int, list[tuple[float, float]]] = {}
        self.path_idx: dict[int, int] = {}
        self.last_pref: dict[int, tuple[float, float]] = {}

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        # Use parent validation
        super().reset(human_ids, starts, goals, backend)

        self.human_ids = list(human_ids)
        self.starts = dict(starts)
        self.goals = dict(goals)
        self.backend = backend

        self.paths = {}
        self.path_idx = {}
        self.last_pref = {}

        for hid in self.human_ids:
            try:
                self.paths[hid] = self._plan_path(self.starts[hid], self.goals[hid])
                self.path_idx[hid] = 0
                self.last_pref[hid] = (0.0, 0.0)
            except Exception as e:
                # Graceful fallback for path planning failures
                logger.warning("Failed to plan path for human %s: %s", hid, str(e))
                self.paths[hid] = [self.goals[hid]]  # Direct path to goal
                self.path_idx[hid] = 0
                self.last_pref[hid] = (0.0, 0.0)

    def _plan_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if self.backend is None:
            return [goal]

        path = self.backend.shortest_path(start, goal)
        if not path:
            return [goal]
        return [(float(x), float(y)) for x, y in path]

    def _maybe_swap_goal(self, agent_id: int, state: AgentState, emit_event: EventSink) -> bool:
        goal = self.goals[agent_id]
        dist = math.hypot(goal[0] - state.x, goal[1] - state.y)
        if dist <= self.goal_tolerance:
            prev_goal = self.goals[agent_id]
            self.goals[agent_id] = self.starts[agent_id]
            self.starts[agent_id] = prev_goal
            emit_event(
                "goal_swap",
                agent_id,
                {
                    "new_goal": [self.goals[agent_id][0], self.goals[agent_id][1]],
                    "new_start": [self.starts[agent_id][0], self.starts[agent_id][1]],
                },
            )
            return True
        return False

    def _current_waypoint(self, agent_id: int, state: AgentState) -> tuple[float, float]:
        path = self.paths.get(agent_id, [])
        idx = self.path_idx.get(agent_id, 0)

        if not path:
            replanned = self._plan_path((state.x, state.y), self.goals[agent_id])
            self.paths[agent_id] = replanned
            self.path_idx[agent_id] = 0
            path = replanned
            idx = 0

        while idx < len(path):
            wx, wy = path[idx]
            if math.hypot(wx - state.x, wy - state.y) <= self.waypoint_tolerance:
                idx += 1
            else:
                break

        if idx >= len(path):
            replanned = self._plan_path((state.x, state.y), self.goals[agent_id])
            self.paths[agent_id] = replanned
            idx = 0

        self.path_idx[agent_id] = idx
        path = self.paths[agent_id]

        look_idx = min(len(path) - 1, idx + max(0, self.lookahead - 1))
        return path[look_idx]

    def _goal_velocity(self, state: AgentState, target: tuple[float, float]) -> tuple[float, float]:
        dx = target[0] - state.x
        dy = target[1] - state.y
        ux, uy, dist = _normalize(dx, dy)
        if dist <= self.goal_tolerance:
            return 0.0, 0.0

        speed_scale = min(1.0, dist / max(self.slowdown_dist, 1e-6))
        speed = state.max_speed * speed_scale
        if dist > self.goal_tolerance:
            speed = max(self.min_speed, speed)
        speed = min(speed, state.max_speed)
        return ux * speed, uy * speed

    def _smooth_preferred_velocity(
        self, human_id: int, state: AgentState, vx: float, vy: float
    ) -> tuple[float, float]:
        alpha = max(0.0, min(1.0, self.velocity_smoothing))
        prev_vx, prev_vy = self.last_pref.get(human_id, (0.0, 0.0))
        svx = prev_vx * (1.0 - alpha) + vx * alpha
        svy = prev_vy * (1.0 - alpha) + vy * alpha

        speed = math.hypot(svx, svy)
        if speed > state.max_speed and speed > 1e-8:
            scale = state.max_speed / speed
            svx *= scale
            svy *= scale

        if speed < self.stop_speed and math.hypot(vx, vy) < self.stop_speed:
            svx, svy = 0.0, 0.0

        self.last_pref[human_id] = (svx, svy)
        return svx, svy

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        # Use parent input validation
        super().step(step, time_s, dt, states, robot_id, emit_event)

        _ = (step, time_s, dt, robot_id)  # Unused parameters
        actions: dict[int, Action] = {}

        for human_id in self.human_ids:
            try:
                # Check if agent state is available
                if human_id not in states:
                    logger.warning("No state available for human %s, using stop action", human_id)
                    actions[human_id] = Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")
                    continue

                state = states[human_id]

                # Validate state values
                if not (math.isfinite(state.x) and math.isfinite(state.y)):
                    logger.warning(
                        "Invalid position for human %s: (%s, %s)", human_id, state.x, state.y
                    )
                    actions[human_id] = Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")
                    continue

                # Handle goal swapping
                if self._maybe_swap_goal(human_id, state, emit_event):
                    try:
                        self.paths[human_id] = self._plan_path(
                            (state.x, state.y), self.goals[human_id]
                        )
                        self.path_idx[human_id] = 0
                    except Exception as e:
                        logger.warning("Failed to replan path for human %s: %s", human_id, str(e))

                # Generate action
                target = self._current_waypoint(human_id, state)
                vx, vy = self._goal_velocity(state, target)
                vx, vy = self._smooth_preferred_velocity(human_id, state, vx, vy)

                action = Action(
                    pref_vx=vx,
                    pref_vy=vy,
                    behavior="GO_TO",
                    metadata={
                        "target_waypoint": [float(target[0]), float(target[1])],
                        "path_index": int(self.path_idx.get(human_id, 0)),
                    },
                )

                # Validate the generated action
                actions[human_id] = self.validate_action(human_id, action)

            except Exception as e:
                logger.error("Error processing human %s: %s", human_id, str(e))
                # Provide safe fallback action
                actions[human_id] = Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")

        return actions
