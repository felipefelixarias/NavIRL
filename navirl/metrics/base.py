from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class MetricsCollector(ABC):
    """Base class for metrics collectors."""

    @abstractmethod
    def compute(self, state_path: Path, scenario: dict) -> dict:
        raise NotImplementedError


def aggregate_reports(reports: list[dict]) -> dict:
    if not reports:
        return {"num_reports": 0}

    scalar_keys = [
        "success_rate",
        "intrusion_rate",
        "collisions_agent_agent",
        "collisions_agent_obstacle",
        "min_dist_robot_human_min",
        "oscillation_score",
        "jerk_proxy",
        "path_length_robot",
        "time_to_goal_robot",
    ]
    out = {"num_reports": len(reports)}
    for key in scalar_keys:
        vals = [r[key] for r in reports if key in r and isinstance(r[key], (float, int))]
        if vals:
            out[f"avg_{key}"] = float(sum(vals) / len(vals))

    out["pass_count"] = int(sum(1 for r in reports if r.get("success_rate", 0.0) >= 1.0))
    return out
