from __future__ import annotations

import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).with_name("schema.json")


def load_schema() -> dict:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _is_point(v) -> bool:
    return (
        isinstance(v, (list, tuple))
        and len(v) == 2
        and isinstance(v[0], (int, float))
        and isinstance(v[1], (int, float))
    )


def _validate_humans(h: dict, errors: list[str]) -> None:
    allowed_ctrl = {"orca", "orca_plus", "scripted", "replay", "policy"}
    _require(isinstance(h, dict), "humans must be an object", errors)
    if not isinstance(h, dict):
        return

    _require("controller" in h, "humans.controller is required", errors)
    _require("count" in h, "humans.count is required", errors)
    if "count" in h:
        _require(
            isinstance(h["count"], int) and h["count"] >= 0, "humans.count must be >= 0", errors
        )

    ctrl = h.get("controller", {})
    _require(isinstance(ctrl, dict), "humans.controller must be an object", errors)
    if isinstance(ctrl, dict):
        _require("type" in ctrl, "humans.controller.type is required", errors)
        if "type" in ctrl:
            _require(
                isinstance(ctrl["type"], str) and ctrl["type"] in allowed_ctrl,
                f"humans.controller.type must be one of {sorted(allowed_ctrl)}",
                errors,
            )

    for key in ("starts", "goals"):
        if key in h:
            _require(isinstance(h[key], list), f"humans.{key} must be an array", errors)
            if isinstance(h[key], list):
                for i, p in enumerate(h[key]):
                    _require(_is_point(p), f"humans.{key}[{i}] must be [x, y]", errors)


def _validate_robot(r: dict, errors: list[str]) -> None:
    allowed_ctrl = {"baseline_astar", "user"}
    _require(isinstance(r, dict), "robot must be an object", errors)
    if not isinstance(r, dict):
        return

    for key in ("controller", "start", "goal"):
        _require(key in r, f"robot.{key} is required", errors)

    ctrl = r.get("controller", {})
    _require(isinstance(ctrl, dict), "robot.controller must be an object", errors)
    if isinstance(ctrl, dict):
        _require("type" in ctrl, "robot.controller.type is required", errors)
        if "type" in ctrl:
            _require(
                isinstance(ctrl["type"], str) and ctrl["type"] in allowed_ctrl,
                f"robot.controller.type must be one of {sorted(allowed_ctrl)}",
                errors,
            )

    if "start" in r:
        _require(_is_point(r["start"]), "robot.start must be [x, y]", errors)
    if "goal" in r:
        _require(_is_point(r["goal"]), "robot.goal must be [x, y]", errors)


def _validate_scene(scene: dict, errors: list[str]) -> None:
    _require(isinstance(scene, dict), "scene must be an object", errors)
    if not isinstance(scene, dict):
        return

    _require(scene.get("backend") == "grid2d", "scene.backend must be 'grid2d'", errors)
    map_cfg = scene.get("map")
    _require(isinstance(map_cfg, dict), "scene.map must be an object", errors)
    if isinstance(map_cfg, dict):
        source = map_cfg.get("source")
        _require(
            source in {"builtin", "path"}, "scene.map.source must be 'builtin' or 'path'", errors
        )
        ppm = map_cfg.get("pixels_per_meter")
        mpp = map_cfg.get("meters_per_pixel")
        downsample = map_cfg.get("downsample", 1.0)
        if ppm is not None:
            _require(
                isinstance(ppm, (int, float)) and ppm > 0.0,  # No float() needed - already numeric
                "scene.map.pixels_per_meter must be > 0 when provided",
                errors,
            )
        if mpp is not None:
            _require(
                isinstance(mpp, (int, float)) and mpp > 0.0,  # No float() needed - already numeric
                "scene.map.meters_per_pixel must be > 0 when provided",
                errors,
            )
        if ppm is not None and mpp is not None:
            exp_mpp = 1.0 / ppm  # No float() needed - already numeric
            _require(
                abs(mpp - exp_mpp) <= max(1e-9, exp_mpp * 0.02),  # No float() needed
                "scene.map.pixels_per_meter and scene.map.meters_per_pixel are inconsistent",
                errors,
            )
        _require(
            isinstance(downsample, (int, float)) and downsample > 0.0,  # No float() needed
            "scene.map.downsample must be > 0 when provided",
            errors,
        )
        if source == "builtin":
            _require(
                isinstance(map_cfg.get("id"), str),
                "scene.map.id is required for source=builtin",
                errors,
            )
        if source == "path":
            _require(
                isinstance(map_cfg.get("path"), str) and len(map_cfg.get("path")) > 0,
                "scene.map.path is required for source=path",
                errors,
            )
            _require(
                ppm is not None or mpp is not None,
                "scene.map.pixels_per_meter or scene.map.meters_per_pixel is required for source=path",
                errors,
            )


def _validate_horizon(horizon: dict, errors: list[str]) -> None:
    _require(isinstance(horizon, dict), "horizon must be an object", errors)
    if not isinstance(horizon, dict):
        return

    _require(
        isinstance(horizon.get("steps"), int) and horizon["steps"] >= 1,
        "horizon.steps must be >= 1",
        errors,
    )
    _require(
        isinstance(horizon.get("dt"), (int, float)) and float(horizon["dt"]) > 0.0,
        "horizon.dt must be > 0",
        errors,
    )


def validate_scenario_dict(scenario: dict) -> None:
    errors: list[str] = []
    _require(isinstance(scenario, dict), "scenario must be an object", errors)
    if not isinstance(scenario, dict):
        raise ValueError("Scenario validation failed:\n- scenario must be an object")

    _require(
        isinstance(scenario.get("id"), str) and len(scenario["id"]) > 0, "id is required", errors
    )
    _require(isinstance(scenario.get("seed"), int), "seed is required and must be integer", errors)

    _validate_scene(scenario.get("scene"), errors)
    _validate_horizon(scenario.get("horizon"), errors)
    _validate_humans(scenario.get("humans"), errors)
    _validate_robot(scenario.get("robot"), errors)

    if errors:
        lines = ["Scenario validation failed:"] + [f"- {e}" for e in errors[:30]]
        raise ValueError("\n".join(lines))
