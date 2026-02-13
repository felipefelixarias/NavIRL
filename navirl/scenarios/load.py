from __future__ import annotations

import copy
from pathlib import Path

import yaml

from navirl.scenarios.validate import validate_scenario_dict


def _as_tuple2(v) -> tuple[float, float]:
    return float(v[0]), float(v[1])


def _apply_defaults(scenario: dict) -> dict:
    out = copy.deepcopy(scenario)
    out.setdefault("description", "")

    out.setdefault("evaluation", {})
    out["evaluation"].setdefault("intrusion_delta", 0.45)
    out["evaluation"].setdefault("max_speed", 1.25)
    out["evaluation"].setdefault("max_accel", 4.0)
    out["evaluation"].setdefault("teleport_thresh", 1.0)
    out["evaluation"].setdefault("deadlock_seconds", 4.0)
    out["evaluation"].setdefault("deadlock_speed_thresh", 0.015)
    out["evaluation"].setdefault("min_robot_progress", 0.1)
    out["evaluation"].setdefault("wall_clearance_buffer", 0.0)
    out["evaluation"].setdefault("enforce_wall_clearance_buffer", False)
    out["evaluation"].setdefault("near_wall_buffer", 0.02)
    out["evaluation"].setdefault("max_wall_proximity_fraction", 0.14)
    out["evaluation"].setdefault("max_heading_flip_rate", 0.82)
    out["evaluation"].setdefault("jitter_speed_thresh", 0.06)
    out["evaluation"].setdefault("max_agent_stop_seconds", 8.0)
    out["evaluation"].setdefault("stop_speed_thresh", 0.02)
    out["evaluation"].setdefault("resample_on_deadlock", True)
    out["evaluation"].setdefault("deadlock_resample_attempts", 4)
    out["evaluation"].setdefault("fail_on_deadlock", True)
    out["evaluation"].setdefault("auto_tune_traversability_offset", True)
    out["evaluation"].setdefault("traversability_offset_step", 0.005)
    out["evaluation"].setdefault("traversability_offset_max", 0.04)

    out.setdefault("render", {})
    out["render"].setdefault("enabled", True)
    out["render"].setdefault("fps", 12)
    out["render"].setdefault("video", False)
    out["render"].setdefault("playback_speed", 1.85)
    out["render"].setdefault("trail_length", 64)

    out.setdefault("humans", {})
    out["humans"].setdefault("radius", 0.18)
    out["humans"].setdefault("max_speed", 0.8)
    out["humans"].setdefault("starts", [])
    out["humans"].setdefault("goals", [])
    out["humans"].setdefault("groups", [])
    out["humans"].setdefault("controller", {"type": "orca", "params": {}})
    out["humans"]["controller"].setdefault("params", {})

    out.setdefault("robot", {})
    out["robot"].setdefault("radius", 0.2)
    out["robot"].setdefault("max_speed", 0.95)
    out["robot"].setdefault("controller", {"type": "baseline_astar", "params": {}})
    out["robot"]["controller"].setdefault("params", {})

    out.setdefault("scene", {})
    out["scene"].setdefault("id", out.get("id", "scenario"))
    out["scene"].setdefault("orca", {})

    out.setdefault("routine", {})
    return out


def _resolve_paths(scenario: dict, scenario_path: Path) -> dict:
    out = copy.deepcopy(scenario)
    map_cfg = out.get("scene", {}).get("map", {})
    if map_cfg.get("source") == "path" and "path" in map_cfg:
        map_path = Path(map_cfg["path"])
        if not map_path.is_absolute():
            map_cfg["path"] = str((scenario_path.parent / map_path).resolve())
    return out


def load_scenario(path: str | Path, *, validate: bool = True) -> dict:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario file must decode into an object: {scenario_path}")

    if validate:
        validate_scenario_dict(scenario)

    scenario = _resolve_paths(scenario, scenario_path)
    scenario = _apply_defaults(scenario)
    scenario["_meta"] = {
        "source_path": str(scenario_path.resolve()),
    }

    # Normalize common coordinate lists.
    scenario["robot"]["start"] = _as_tuple2(scenario["robot"]["start"])
    scenario["robot"]["goal"] = _as_tuple2(scenario["robot"]["goal"])
    scenario["humans"]["starts"] = [_as_tuple2(p) for p in scenario["humans"].get("starts", [])]
    scenario["humans"]["goals"] = [_as_tuple2(p) for p in scenario["humans"].get("goals", [])]

    return scenario
