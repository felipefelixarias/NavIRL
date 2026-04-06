from __future__ import annotations

import math

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController


class BaselineAStarRobotController(RobotController):
    """A* global path follower with periodic replanning and stuck recovery.

    When the robot detects it hasn't made meaningful progress (ORCA zeroed
    its velocity because humans are blocking), it:
    1. Waits patiently for a few steps (yields to humans)
    2. Replans the path (humans may have moved)
    3. If still stuck after multiple waits, tries a perpendicular nudge
    """

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
        # Stuck detection: check distance over a window of N steps
        self.stuck_window = int(cfg.get("stuck_window", 40))
        self.stuck_dist_threshold = float(cfg.get("stuck_dist_threshold", 0.08))
        self.wait_duration = int(cfg.get("wait_duration", 15))
        self.max_wait_cycles = int(cfg.get("max_wait_cycles", 6))

        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend = None
        self.path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_pref = (0.0, 0.0)
        # Stuck recovery state
        self._pos_window: list[tuple[float, float]] = []
        self._wait_counter = 0
        self._wait_cycles = 0
        self._is_waiting = False
        self._grace_steps = 0  # steps after yield where stuck detection is disabled

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
        self._pos_window = [start]
        self._wait_counter = 0
        self._wait_cycles = 0
        self._is_waiting = False
        self._grace_steps = 0
        self._plan(start)

    def _current_target(self) -> tuple[float, float]:
        if self.path_idx >= len(self.path):
            return self.goal
        look_idx = min(len(self.path) - 1, self.path_idx + max(0, self.target_lookahead - 1))
        return self.path[look_idx]

    def _detect_stuck(self, st: AgentState) -> bool:
        """Check if robot has made meaningful progress over a sliding window.

        Compares current position to where it was N steps ago. Only triggers
        if total displacement over the window is tiny — avoids false positives
        from ORCA temporarily slowing the robot near other agents.
        """
        self._pos_window.append((st.x, st.y))
        if len(self._pos_window) > self.stuck_window:
            self._pos_window.pop(0)

        if len(self._pos_window) < self.stuck_window:
            return False  # not enough history yet

        old = self._pos_window[0]
        dist = math.hypot(st.x - old[0], st.y - old[1])

        if dist >= self.stuck_dist_threshold:
            self._wait_cycles = 0  # real progress — reset

        return dist < self.stuck_dist_threshold

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

        # --- Stuck detection and recovery ---
        if self._is_waiting:
            self._wait_counter -= 1
            if self._wait_counter <= 0:
                self._is_waiting = False
                self._pos_window.clear()  # reset so it doesn't re-trigger immediately
                self._grace_steps = self.stuck_window  # grace period before re-checking
                # Replan after waiting — humans likely moved
                self._plan((st.x, st.y))
                emit_event("robot_replan", self.robot_id,
                           {"reason": "post_wait", "wait_cycle": self._wait_cycles})
            # Emit tiny velocity toward goal so deadlock detector doesn't
            # flag a full stop, but ORCA will override to near-zero anyway
            dx = self.goal[0] - st.x
            dy = self.goal[1] - st.y
            d = math.hypot(dx, dy)
            crawl = 0.05  # barely moving — signals intent without forcing
            if d > 1e-8:
                return Action(pref_vx=dx / d * crawl, pref_vy=dy / d * crawl, behavior="YIELD")
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="YIELD")

        # Grace period after yield — let robot try GO_TO before re-checking
        if self._grace_steps > 0:
            self._grace_steps -= 1
            is_stuck = False
        else:
            is_stuck = self._detect_stuck(st)

        if is_stuck:
            # Enter wait mode — yield to let humans pass, then replan
            self._is_waiting = True
            self._wait_counter = self.wait_duration
            self._wait_cycles += 1
            emit_event("robot_yield", self.robot_id,
                       {"wait_cycle": self._wait_cycles, "pos": (st.x, st.y)})
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="YIELD")

        # --- Normal path following ---
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
