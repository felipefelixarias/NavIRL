from __future__ import annotations

import json
import math
from pathlib import Path

from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController


class ReplayHumanController(HumanController):
    """Replay humans from an existing state log."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.human_ids: list[int] = []
        self.replay_positions: dict[int, list[tuple[float, float]]] = {}

    def _load_replay(self, replay_path: str, human_ids: list[int]) -> None:
        tracks = {hid: [] for hid in human_ids}
        with Path(replay_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for agent in row.get("agents", []):
                    aid = int(agent["id"])
                    if aid in tracks and agent.get("kind") == "human":
                        tracks[aid].append((float(agent["x"]), float(agent["y"])))
        self.replay_positions = tracks

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        _ = (starts, goals, backend)
        self.human_ids = list(human_ids)
        replay_path = self.cfg.get("path")
        if replay_path:
            self._load_replay(replay_path, human_ids)
        else:
            self.replay_positions = {hid: [] for hid in human_ids}

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        _ = (time_s, robot_id, emit_event)
        actions: dict[int, Action] = {}

        for hid in self.human_ids:
            st = states[hid]
            track = self.replay_positions.get(hid, [])
            if step >= len(track):
                actions[hid] = Action(0.0, 0.0, behavior="REPLAY_DONE")
                continue

            tx, ty = track[step]
            dx, dy = tx - st.x, ty - st.y
            dist = math.hypot(dx, dy)
            if dist < 1e-8:
                actions[hid] = Action(0.0, 0.0, behavior="REPLAY")
            else:
                speed = min(st.max_speed, dist / max(dt, 1e-6))
                actions[hid] = Action(speed * dx / dist, speed * dy / dist, behavior="REPLAY")

        return actions
