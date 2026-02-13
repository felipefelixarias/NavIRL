from __future__ import annotations

import math

from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController


class ScriptedHumanController(HumanController):
    """Deterministic scripted waypoints for repeatable debugging."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.2))
        self.max_speed = float(cfg.get("max_speed", 0.6))
        self.human_ids: list[int] = []
        self.scripts: dict[int, list[tuple[float, float]]] = {}
        self.indices: dict[int, int] = {}
        self.starts: dict[int, tuple[float, float]] = {}
        self.goals: dict[int, tuple[float, float]] = {}

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        _ = backend
        self.human_ids = list(human_ids)
        self.starts = dict(starts)
        self.goals = dict(goals)
        raw_scripts = self.cfg.get("waypoints", {})

        self.scripts = {}
        self.indices = {hid: 0 for hid in human_ids}
        for idx, hid in enumerate(human_ids):
            key = str(idx)
            waypoints = raw_scripts.get(key)
            if waypoints:
                self.scripts[hid] = [tuple(map(float, pt)) for pt in waypoints]
            else:
                self.scripts[hid] = [goals[hid], starts[hid]]

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        _ = (step, time_s, dt, robot_id)
        actions: dict[int, Action] = {}

        for hid in self.human_ids:
            st = states[hid]
            wp_idx = self.indices[hid]
            waypoints = self.scripts[hid]
            target = waypoints[wp_idx]

            dx, dy = target[0] - st.x, target[1] - st.y
            dist = math.hypot(dx, dy)
            if dist < self.goal_tolerance:
                self.indices[hid] = (wp_idx + 1) % max(1, len(waypoints))
                emit_event("script_waypoint_reached", hid, {"waypoint_index": wp_idx})
                target = waypoints[self.indices[hid]]
                dx, dy = target[0] - st.x, target[1] - st.y
                dist = math.hypot(dx, dy)

            if dist < 1e-8:
                actions[hid] = Action(0.0, 0.0, behavior="WAIT")
            else:
                speed = min(st.max_speed, self.max_speed)
                actions[hid] = Action(speed * dx / dist, speed * dy / dist, behavior="SCRIPT")

        return actions
