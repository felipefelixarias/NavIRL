from __future__ import annotations

import math
import random

from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink
from navirl.humans.orca.controller import ORCAHumanController, _normalize


class ORCAPlusHumanController(ORCAHumanController):
    """ORCA+ heuristics with ablation flags for social realism and doorway etiquette."""

    def __init__(self, cfg: dict | None = None, seed: int = 0):
        super().__init__(cfg=cfg)
        cfg = cfg or {}
        self.enable_doorway_token = bool(cfg.get("doorway_token", True))
        self.enable_anisotropic_space = bool(cfg.get("anisotropic_space", True))
        self.enable_speed_profile = bool(cfg.get("speed_profile", True))
        self.enable_group_cohesion = bool(cfg.get("group_cohesion", False))

        self.personal_space = float(cfg.get("personal_space", 0.9))
        self.accel_limit = float(cfg.get("accel_limit", 1.6))
        self.hesitation_prob = float(cfg.get("hesitation_prob", 0.06))
        self.hesitation_scale = float(cfg.get("hesitation_scale", 0.35))
        self.group_weight = float(cfg.get("group_weight", 0.35))

        doorway = cfg.get("doorway", {})
        self.door_center = tuple(doorway.get("center", [0.0, 0.0]))
        self.door_half_extents = tuple(doorway.get("half_extents", [0.25, 0.2]))
        self.door_approach_margin = float(doorway.get("approach_margin", 0.6))

        self.groups: list[list[int]] = [list(map(int, g)) for g in cfg.get("groups", [])]
        self.group_by_agent: dict[int, list[int]] = {}
        self.current_speed: dict[int, float] = {}
        self.token_holder: int | None = None
        self.rng = random.Random(seed)

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        super().reset(human_ids, starts, goals, backend=backend)
        self.current_speed = {hid: 0.0 for hid in human_ids}
        self.token_holder = None
        self.group_by_agent = {}
        for group in self.groups:
            for aid in group:
                self.group_by_agent[aid] = group

    def _in_doorway(self, x: float, y: float, margin: float = 0.0) -> bool:
        cx, cy = self.door_center
        hx, hy = self.door_half_extents
        return abs(x - cx) <= (hx + margin) and abs(y - cy) <= (hy + margin)

    def _doorway_candidates(self, states: dict[int, AgentState]) -> list[int]:
        cands: list[int] = []
        for hid in self.human_ids:
            st = states[hid]
            if self._in_doorway(st.x, st.y, margin=self.door_approach_margin):
                cands.append(hid)
        return sorted(cands)

    def _update_token(self, states: dict[int, AgentState], emit_event: EventSink) -> None:
        if not self.enable_doorway_token:
            return

        if self.token_holder is not None:
            holder = states[self.token_holder]
            if not self._in_doorway(holder.x, holder.y, margin=self.door_approach_margin + 0.2):
                emit_event("door_token_release", self.token_holder, {"token": "doorway"})
                self.token_holder = None

        if self.token_holder is None:
            candidates = self._doorway_candidates(states)
            if candidates:
                self.token_holder = candidates[0]
                emit_event("door_token_acquire", self.token_holder, {"token": "doorway"})

    def _apply_anisotropic_scale(
        self,
        hid: int,
        state: AgentState,
        desired: tuple[float, float],
        states: dict[int, AgentState],
    ) -> float:
        if not self.enable_anisotropic_space:
            return 1.0

        vx, vy = desired
        ux, uy, speed = _normalize(vx, vy)
        if speed < 1e-8:
            return 1.0

        closest_front = float("inf")
        for oid, other in states.items():
            if oid == hid:
                continue
            dx, dy = other.x - state.x, other.y - state.y
            dist = math.hypot(dx, dy)
            if dist < 1e-8:
                continue
            forwardness = (dx * ux + dy * uy) / dist
            if forwardness > 0.25:
                closest_front = min(closest_front, dist)

        if not math.isfinite(closest_front):
            return 1.0
        return max(0.25, min(1.0, closest_front / self.personal_space))

    def _group_velocity_bias(self, hid: int, states: dict[int, AgentState]) -> tuple[float, float]:
        if not self.enable_group_cohesion:
            return 0.0, 0.0

        group = self.group_by_agent.get(hid)
        if not group or len(group) <= 1:
            return 0.0, 0.0

        members = [states[g] for g in group if g in states and g != hid]
        if not members:
            return 0.0, 0.0

        cx = sum(m.x for m in members) / len(members)
        cy = sum(m.y for m in members) / len(members)
        st = states[hid]
        dx, dy = cx - st.x, cy - st.y
        ux, uy, _ = _normalize(dx, dy)
        return ux * self.group_weight, uy * self.group_weight

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        _ = (time_s, robot_id)
        actions = super().step(step, time_s, dt, states, robot_id, emit_event)
        self._update_token(states, emit_event)

        for hid in self.human_ids:
            st = states[hid]
            act = actions[hid]

            if self.enable_doorway_token and self.token_holder is not None and hid != self.token_holder:
                if self._in_doorway(st.x, st.y, margin=self.door_approach_margin):
                    emit_event("doorway_yield", hid, {"to_agent": self.token_holder})
                    actions[hid] = Action(pref_vx=0.0, pref_vy=0.0, behavior="YIELDING")
                    continue

            scale = self._apply_anisotropic_scale(hid, st, (act.pref_vx, act.pref_vy), states)
            vx, vy = act.pref_vx * scale, act.pref_vy * scale

            if self.enable_group_cohesion:
                gx, gy = self._group_velocity_bias(hid, states)
                vx += gx
                vy += gy

            target_speed = math.hypot(vx, vy)
            if self.enable_speed_profile:
                prev_speed = self.current_speed.get(hid, 0.0)
                max_delta = self.accel_limit * dt
                bounded_speed = max(prev_speed - max_delta, min(prev_speed + max_delta, target_speed))

                if self.rng.random() < self.hesitation_prob:
                    bounded_speed *= self.hesitation_scale
                    emit_event("hesitation", hid, {"scale": self.hesitation_scale})

                ux, uy, _ = _normalize(vx, vy)
                vx, vy = ux * bounded_speed, uy * bounded_speed
                self.current_speed[hid] = bounded_speed

            actions[hid] = Action(pref_vx=vx, pref_vy=vy, behavior=act.behavior)

        return actions
