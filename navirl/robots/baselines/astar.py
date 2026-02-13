from __future__ import annotations

import math

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class BaselineAStarRobotController(RobotController):
    """A* global path follower with periodic replanning."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.2))
        self.replan_interval = int(cfg.get("replan_interval", 25))
        self.max_speed = float(cfg.get("max_speed", 0.8))
        self.slowdown_dist = float(cfg.get("slowdown_dist", 0.7))
        self.target_lookahead = int(cfg.get("target_lookahead", 4))
        self.velocity_smoothing = float(cfg.get("velocity_smoothing", 0.55))
        self.stop_speed = float(cfg.get("stop_speed", 0.02))
        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)

    def _plan(self, position: tuple[float, float]) -> None:
        self.path = self.backend.shortest_path(position, self.goal)
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
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.backend = backend
        self.last_pref = (0.0, 0.0)
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
        _ = (time_s, dt)
        st = states[self.robot_id]
        dist_goal = math.hypot(self.goal[0] - st.x, self.goal[1] - st.y)
        if dist_goal <= self.goal_tolerance:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        if step % max(1, self.replan_interval) == 0:
            self._plan((st.x, st.y))
            emit_event("robot_replan", self.robot_id, {"path_len": len(self.path)})

        target = self._current_target()
        dist_target = math.hypot(target[0] - st.x, target[1] - st.y)
        if dist_target <= self.goal_tolerance and self.path_idx < len(self.path) - 1:
            self.path_idx += 1
            target = self._current_target()
            dist_target = math.hypot(target[0] - st.x, target[1] - st.y)

        if dist_target < 1e-8:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="WAIT")

        speed_scale = min(1.0, dist_target / max(self.slowdown_dist, 1e-6))
        speed = min(st.max_speed, self.max_speed) * speed_scale
        ux = (target[0] - st.x) / dist_target
        uy = (target[1] - st.y) / dist_target
        vx = ux * speed
        vy = uy * speed

        alpha = max(0.0, min(1.0, self.velocity_smoothing))
        vx = self.last_pref[0] * (1.0 - alpha) + vx * alpha
        vy = self.last_pref[1] * (1.0 - alpha) + vy * alpha
        if math.hypot(vx, vy) < self.stop_speed and dist_target < self.goal_tolerance:
            vx, vy = 0.0, 0.0
        self.last_pref = (vx, vy)

        return Action(
            pref_vx=vx,
            pref_vy=vy,
            behavior="GO_TO",
        )
