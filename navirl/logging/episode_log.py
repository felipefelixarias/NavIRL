from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from navirl.core.types import AgentState, EventRecord


class EpisodeLogger:
    """Writes NavIRL trace bundle artifacts."""

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.bundle_dir / "state.jsonl"
        self.events_path = self.bundle_dir / "events.jsonl"
        self.scenario_path = self.bundle_dir / "scenario.yaml"
        self.summary_path = self.bundle_dir / "summary.json"

        self._state_f = self.state_path.open("w", encoding="utf-8")
        self._events_f = self.events_path.open("w", encoding="utf-8")

    def write_resolved_scenario(self, scenario: dict) -> None:
        with self.scenario_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(scenario, f, sort_keys=False)

    def write_state(self, step: int, time_s: float, agents: list[AgentState]) -> None:
        row = {
            "step": int(step),
            "time_s": float(time_s),
            "agents": [
                {
                    "id": int(a.agent_id),
                    "kind": a.kind,
                    "x": float(a.x),
                    "y": float(a.y),
                    "vx": float(a.vx),
                    "vy": float(a.vy),
                    "goal_x": float(a.goal_x),
                    "goal_y": float(a.goal_y),
                    "radius": float(a.radius),
                    "max_speed": float(a.max_speed),
                    "behavior": a.behavior,
                    "metadata": a.metadata,
                }
                for a in agents
            ],
        }
        self._state_f.write(json.dumps(row, sort_keys=True) + "\n")

    def write_event(self, event: EventRecord) -> None:
        self._events_f.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def write_summary(self, summary: dict) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    def close(self) -> None:
        if not self._state_f.closed:
            self._state_f.close()
        if not self._events_f.closed:
            self._events_f.close()
