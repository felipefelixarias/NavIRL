from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

from navirl.backends.grid2d.constants import OBSTACLE_SPACE
from navirl.backends.grid2d.environment import GridEnvironment
from navirl.backends.grid2d.maps import load_map_info

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit for configuration files
MAX_LINE_SIZE = 1024 * 1024  # 1MB limit per JSON line


def load_state_rows(state_path: Path) -> list[dict]:
    # Check file size for security
    if not state_path.exists():
        raise ValueError(f"State file not found: {state_path}")
    if state_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(
            f"State file too large (>{MAX_FILE_SIZE / 1024 / 1024:.1f}MB): {state_path}"
        )

    rows = []
    with state_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Check line size before parsing
                if len(line.encode("utf-8")) > MAX_LINE_SIZE:
                    raise ValueError(f"JSON line too large (>{MAX_LINE_SIZE / 1024:.1f}KB)")
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {state_path}")
    return rows


def load_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []

    # Check file size for security
    if events_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(
            f"Events file too large (>{MAX_FILE_SIZE / 1024 / 1024:.1f}MB): {events_path}"
        )

    out = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Check line size before parsing
                if len(line.encode("utf-8")) > MAX_LINE_SIZE:
                    raise ValueError(f"JSON line too large (>{MAX_LINE_SIZE / 1024:.1f}KB)")
                out.append(json.loads(line))
    return out


def _load_scenario(bundle_dir: Path) -> dict:
    scenario_path = bundle_dir / "scenario.yaml"

    # Check file size for security
    if not scenario_path.exists():
        raise ValueError(f"Scenario file not found: {scenario_path}")
    if scenario_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(
            f"Scenario file too large (>{MAX_FILE_SIZE / 1024 / 1024:.1f}MB): {scenario_path}"
        )

    with scenario_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_units_metadata(scenario: dict) -> dict:
    map_cfg = scenario.get("scene", {}).get("map", {})
    resolved = map_cfg.get("resolved", {})
    ppm = resolved.get("pixels_per_meter", map_cfg.get("pixels_per_meter"))
    mpp = resolved.get("meters_per_pixel", map_cfg.get("meters_per_pixel"))
    width_m = resolved.get("width_m")
    height_m = resolved.get("height_m")

    violations = []
    if ppm is None and mpp is None:
        violations.append({"reason": "missing_map_scale"})
    else:
        if ppm is None and mpp is not None:
            ppm = 1.0 / mpp
        if mpp is None and ppm is not None:
            mpp = 1.0 / ppm

    if ppm is not None:
        if ppm <= 0.0:
            violations.append({"reason": "pixels_per_meter_nonpositive", "pixels_per_meter": ppm})
    if mpp is not None:
        if mpp <= 0.0:
            violations.append({"reason": "meters_per_pixel_nonpositive", "meters_per_pixel": mpp})

    if ppm is not None and mpp is not None:
        expected = 1.0 / ppm
        if abs(mpp - expected) > max(1e-9, expected * 0.02):
            violations.append(
                {
                    "reason": "scale_inconsistent",
                    "pixels_per_meter": ppm,
                    "meters_per_pixel": mpp,
                }
            )

    if width_m is not None and width_m <= 0.0:
        violations.append({"reason": "width_m_nonpositive", "width_m": width_m})
    if height_m is not None and height_m <= 0.0:
        violations.append({"reason": "height_m_nonpositive", "height_m": height_m})

    return {
        "name": "units_metadata",
        "pass": len(violations) == 0,
        "pixels_per_meter": ppm,  # Already numeric, no conversion needed
        "meters_per_pixel": mpp,  # Already numeric, no conversion needed
        "width_m": width_m,  # Already numeric, no conversion needed
        "height_m": height_m,  # Already numeric, no conversion needed
        "num_violations": len(violations),
        "violations": violations,
    }


def _in_bounds(node: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = node
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _nearest_passable(passable: np.ndarray, start: tuple[int, int]) -> tuple[int, int] | None:
    if _in_bounds(start, passable.shape) and bool(passable[start[0], start[1]]):
        return start

    q = deque([start])
    seen = {start}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in seen:
                continue
            seen.add(nxt)
            if not _in_bounds(nxt, passable.shape):
                continue
            if bool(passable[nxt[0], nxt[1]]):
                return nxt
            q.append(nxt)
    return None


def _path_exists(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if not bool(passable[start[0], start[1]]) or not bool(passable[goal[0], goal[1]]):
        return False

    q = deque([start])
    seen = {start}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in seen or (not _in_bounds(nxt, passable.shape)):
                continue
            if not bool(passable[nxt[0], nxt[1]]):
                continue
            seen.add(nxt)
            q.append(nxt)
    return False


def validate_scenario_feasibility(bundle_dir: Path, max_adjust_m: float = 0.6) -> dict:
    scenario = _load_scenario(bundle_dir)
    src_path = scenario.get("_meta", {}).get("source_path")
    base_dir = Path(src_path).parent if src_path else None
    map_info = load_map_info(scenario["scene"], base_dir=base_dir)
    env = GridEnvironment(
        "verify-feasibility", map_info.binary_map, pixels_per_meter=map_info.pixels_per_meter
    )
    free_mask = (env.map != OBSTACLE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)

    humans = scenario.get("humans", {})
    h_count = int(humans.get("count", 0))
    h_radius = float(humans.get("radius", 0.16))
    h_starts = list(humans.get("starts", []))
    h_goals = list(humans.get("goals", []))

    agent_specs: list[dict] = []
    robot_cfg = scenario.get("robot", {})
    agent_specs.append(
        {
            "agent_id": 0,
            "kind": "robot",
            "radius": float(robot_cfg.get("radius", 0.18)),
            "start": tuple(map(float, robot_cfg.get("start", [0.0, 0.0]))),
            "goal": tuple(map(float, robot_cfg.get("goal", [0.0, 0.0]))),
        }
    )

    explicit_h = min(h_count, len(h_starts), len(h_goals))
    for i in range(explicit_h):
        agent_specs.append(
            {
                "agent_id": i + 1,
                "kind": "human",
                "radius": h_radius,
                "start": tuple(map(float, h_starts[i])),
                "goal": tuple(map(float, h_goals[i])),
            }
        )

    warnings: list[dict] = []
    violations: list[dict] = []
    suggestions: list[str] = []
    path_meta: dict[int, dict] = {}

    for spec in agent_specs:
        aid = int(spec["agent_id"])
        radius = float(spec["radius"])
        required_px = radius * env.pixels_per_meter
        passable = (env.map != OBSTACLE_SPACE) & (clearance_px >= required_px)

        start_rc_raw = tuple(env._world_to_map(np.asarray(spec["start"], dtype=float)).tolist())
        goal_rc_raw = tuple(env._world_to_map(np.asarray(spec["goal"], dtype=float)).tolist())
        start_rc = _nearest_passable(passable, start_rc_raw)
        goal_rc = _nearest_passable(passable, goal_rc_raw)

        if start_rc is None or goal_rc is None:
            violations.append(
                {
                    "agent_id": aid,
                    "kind": spec["kind"],
                    "reason": "no_passable_start_or_goal_for_radius",
                    "radius": radius,
                }
            )
            suggestions.append(
                f"Agent {aid}: reduce radius below {radius:.3f}m or move start/goal to a wider room."
            )
            continue

        start_world = env._map_to_world(np.asarray(start_rc, dtype=float))
        goal_world = env._map_to_world(np.asarray(goal_rc, dtype=float))
        start_shift = math.hypot(
            float(start_world[0]) - spec["start"][0], float(start_world[1]) - spec["start"][1]
        )
        goal_shift = math.hypot(
            float(goal_world[0]) - spec["goal"][0], float(goal_world[1]) - spec["goal"][1]
        )
        if start_shift > max_adjust_m or goal_shift > max_adjust_m:
            warnings.append(
                {
                    "agent_id": aid,
                    "kind": spec["kind"],
                    "reason": "large_start_goal_adjustment",
                    "start_shift_m": float(start_shift),
                    "goal_shift_m": float(goal_shift),
                }
            )
            suggestions.append(
                f"Agent {aid}: adjust explicit start/goal away from walls/obstacles "
                f"(shift {max(start_shift, goal_shift):.2f}m)."
            )

        has_path = _path_exists(passable, start_rc, goal_rc)
        if not has_path:
            violations.append(
                {
                    "agent_id": aid,
                    "kind": spec["kind"],
                    "reason": "radius_inflated_path_blocked",
                    "radius": radius,
                }
            )
            suggestions.append(
                f"Agent {aid}: no clearance-feasible path at radius {radius:.3f}m. Reduce radius or change start/goal."
            )
            continue

        path_world, _ = env.shortest_path(
            np.asarray(spec["start"]), np.asarray(spec["goal"]), entire_path=True
        )
        min_clearance_m = float("inf")
        for wp in path_world:
            rr, cc = env._world_to_map(np.asarray(wp, dtype=float))
            if 0 <= rr < env.map.shape[0] and 0 <= cc < env.map.shape[1]:
                min_clearance_m = min(
                    min_clearance_m, float(clearance_px[rr, cc] / env.pixels_per_meter)
                )
        path_meta[aid] = {
            "agent_id": aid,
            "radius": radius,
            "dir": (
                float(spec["goal"][0] - spec["start"][0]),
                float(spec["goal"][1] - spec["start"][1]),
            ),
            "min_path_clearance_m": (
                float(min_clearance_m) if math.isfinite(min_clearance_m) else None
            ),
        }

    # Bidirectional bottleneck risk detection (warning-level).
    risk_pairs = []
    ids = sorted(path_meta)
    for i, aid in enumerate(ids):
        for bid in ids[i + 1 :]:
            a = path_meta[aid]
            b = path_meta[bid]
            da = np.asarray(a["dir"], dtype=float)
            db = np.asarray(b["dir"], dtype=float)
            na = float(np.linalg.norm(da))
            nb = float(np.linalg.norm(db))
            if na < 1e-6 or nb < 1e-6:
                continue
            dot = float(np.dot(da / na, db / nb))
            if dot > -0.6:
                continue
            a_min = float(a.get("min_path_clearance_m") or 0.0)
            b_min = float(b.get("min_path_clearance_m") or 0.0)
            combined = float(a["radius"] + b["radius"])
            if min(a_min, b_min) < combined:
                risk_pairs.append(
                    {
                        "agent_a": int(aid),
                        "agent_b": int(bid),
                        "min_path_clearance_m": float(min(a_min, b_min)),
                        "required_for_side_by_side_m": float(combined),
                    }
                )

    if risk_pairs:
        warnings.append(
            {
                "reason": "bidirectional_bottleneck_risk",
                "num_pairs": int(len(risk_pairs)),
                "pairs": risk_pairs[:10],
            }
        )
        suggestions.append(
            "Detected opposing flows through sub-width bottlenecks; "
            "stagger start times or route one direction at a time."
        )

    if explicit_h < h_count:
        warnings.append(
            {
                "reason": "partial_human_start_goal_spec",
                "explicit_count": int(explicit_h),
                "human_count": int(h_count),
            }
        )

    return {
        "name": "scenario_feasibility",
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "num_warnings": len(warnings),
        "violations": violations[:30],
        "warnings": warnings[:30],
        "suggestions": sorted(set(suggestions))[:20],
    }


def validate_anchor_layout(bundle_dir: Path) -> dict:
    scenario = _load_scenario(bundle_dir)
    src_path = scenario.get("_meta", {}).get("source_path")
    base_dir = Path(src_path).parent if src_path else None
    map_info = load_map_info(scenario["scene"], base_dir=base_dir)
    env = GridEnvironment(
        "verify-anchor-layout", map_info.binary_map, pixels_per_meter=map_info.pixels_per_meter
    )
    free_mask = (env.map != OBSTACLE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)

    anchors: list[dict] = []
    robot = scenario.get("robot", {})
    robot_radius = float(robot.get("radius", 0.18))
    anchors.append(
        {
            "label": "robot.start",
            "kind": "robot",
            "phase": "start",
            "radius": robot_radius,
            "position": tuple(map(float, robot.get("start", [0.0, 0.0]))),
        }
    )
    anchors.append(
        {
            "label": "robot.goal",
            "kind": "robot",
            "phase": "goal",
            "radius": robot_radius,
            "position": tuple(map(float, robot.get("goal", [0.0, 0.0]))),
        }
    )

    humans = scenario.get("humans", {})
    h_radius = float(humans.get("radius", 0.16))
    h_count = int(humans.get("count", 0))
    starts = list(humans.get("starts", []))
    goals = list(humans.get("goals", []))
    explicit_h = min(h_count, len(starts), len(goals))
    for i in range(explicit_h):
        anchors.append(
            {
                "label": f"human.{i + 1}.start",
                "kind": "human",
                "phase": "start",
                "radius": h_radius,
                "position": tuple(map(float, starts[i])),
            }
        )
    for i in range(explicit_h):
        anchors.append(
            {
                "label": f"human.{i + 1}.goal",
                "kind": "human",
                "phase": "goal",
                "radius": h_radius,
                "position": tuple(map(float, goals[i])),
            }
        )

    obstacle_violations: list[dict] = []
    for a in anchors:
        rr, cc = env._world_to_map(np.asarray(a["position"], dtype=float))
        if rr < 0 or cc < 0 or rr >= env.map.shape[0] or cc >= env.map.shape[1]:
            obstacle_violations.append({"label": a["label"], "reason": "out_of_bounds"})
            continue
        required = (2.0 * float(a["radius"])) * env.pixels_per_meter
        available = float(clearance_px[rr, cc])
        if available + 1e-6 < required:
            obstacle_violations.append(
                {
                    "label": a["label"],
                    "reason": "obstacle_within_diameter",
                    "required_clearance_px": float(required),
                    "available_clearance_px": float(available),
                }
            )

    pair_violations: list[dict] = []
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            if str(a.get("phase")) != str(b.get("phase")):
                continue
            min_dist = max(2.0 * float(a["radius"]), 2.0 * float(b["radius"]))
            dx = float(a["position"][0]) - float(b["position"][0])
            dy = float(a["position"][1]) - float(b["position"][1])
            d = math.hypot(dx, dy)
            if d + 1e-6 < min_dist:
                pair_violations.append(
                    {
                        "a": a["label"],
                        "b": b["label"],
                        "phase": str(a.get("phase")),
                        "distance": float(d),
                        "required_min_distance": float(min_dist),
                    }
                )

    return {
        "name": "anchor_layout",
        "pass": (len(obstacle_violations) == 0 and len(pair_violations) == 0),
        "num_obstacle_violations": len(obstacle_violations),
        "num_pair_violations": len(pair_violations),
        "num_anchors": len(anchors),
        "obstacle_violations": obstacle_violations[:40],
        "pair_violations": pair_violations[:40],
    }


def validate_no_wall_penetration(state_path: Path, bundle_dir: Path) -> dict:
    rows = load_state_rows(state_path)
    scenario = _load_scenario(bundle_dir)
    src_path = scenario.get("_meta", {}).get("source_path")
    base_dir = Path(src_path).parent if src_path else None
    map_info = load_map_info(scenario["scene"], base_dir=base_dir)
    env = GridEnvironment(
        "verify-map", map_info.binary_map, pixels_per_meter=map_info.pixels_per_meter
    )
    free_mask = (env.map != OBSTACLE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)

    violations = []
    out_of_bounds = 0
    for row in rows:
        step = int(row["step"])
        for agent in row["agents"]:
            x, y = float(agent["x"]), float(agent["y"])
            aid = int(agent["id"])
            kind = str(agent.get("kind", "unknown"))
            rr, cc = env._world_to_map(np.array([x, y], dtype=float))
            if rr < 0 or cc < 0 or rr >= env.map.shape[0] or cc >= env.map.shape[1]:
                out_of_bounds += 1
                violations.append(
                    {"step": step, "agent_id": aid, "kind": kind, "reason": "out_of_bounds"}
                )
                continue

            if env.map[rr, cc] == OBSTACLE_SPACE:
                violations.append(
                    {"step": step, "agent_id": aid, "kind": kind, "reason": "inside_obstacle"}
                )
                continue

            radius_m = float(agent.get("radius", 0.0))
            required_px = radius_m * env.pixels_per_meter
            available_px = float(clearance_px[rr, cc])
            if available_px + 1e-6 < required_px:
                violations.append(
                    {
                        "step": step,
                        "agent_id": aid,
                        "kind": kind,
                        "reason": "radius_intersects_obstacle",
                        "required_clearance_px": required_px,
                        "available_clearance_px": available_px,
                    }
                )

    return {
        "name": "no_wall_penetration",
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "num_out_of_bounds": out_of_bounds,
        "pixels_per_meter": float(env.pixels_per_meter),
        "violations": violations[:40],
    }


def validate_wall_clearance_buffer(
    state_path: Path,
    bundle_dir: Path,
    clearance_buffer_m: float,
    max_fraction: float,
) -> dict:
    rows = load_state_rows(state_path)
    scenario = _load_scenario(bundle_dir)
    src_path = scenario.get("_meta", {}).get("source_path")
    base_dir = Path(src_path).parent if src_path else None
    map_info = load_map_info(scenario["scene"], base_dir=base_dir)
    env = GridEnvironment(
        "verify-wall-buffer",
        map_info.binary_map,
        pixels_per_meter=map_info.pixels_per_meter,
    )
    free_mask = (env.map != OBSTACLE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)

    violations = []
    total = 0
    req_extra_px = max(0.0, float(clearance_buffer_m) * env.pixels_per_meter)
    for row in rows:
        step = int(row["step"])
        for agent in row["agents"]:
            x, y = float(agent["x"]), float(agent["y"])
            aid = int(agent["id"])
            kind = str(agent.get("kind", "unknown"))
            rr, cc = env._world_to_map(np.array([x, y], dtype=float))
            if rr < 0 or cc < 0 or rr >= env.map.shape[0] or cc >= env.map.shape[1]:
                continue

            total += 1
            radius_m = float(agent.get("radius", 0.0))
            required_px = radius_m * env.pixels_per_meter + req_extra_px
            available_px = float(clearance_px[rr, cc])
            if available_px + 1e-6 < required_px:
                violations.append(
                    {
                        "step": step,
                        "agent_id": aid,
                        "kind": kind,
                        "required_clearance_px": required_px,
                        "available_clearance_px": available_px,
                    }
                )

    frac = 0.0 if total == 0 else float(len(violations) / total)
    return {
        "name": "wall_clearance_buffer",
        "pass": bool(frac <= max_fraction + 1e-9),
        "num_violations": len(violations),
        "violation_fraction": float(frac),
        "max_fraction": float(max_fraction),
        "samples": int(total),
        "clearance_buffer_m": float(clearance_buffer_m),
        "pixels_per_meter": float(env.pixels_per_meter),
        "violations": violations[:40],
    }


def validate_wall_proximity(
    state_path: Path,
    bundle_dir: Path,
    near_wall_buffer_m: float,
    max_fraction: float,
) -> dict:
    rows = load_state_rows(state_path)
    scenario = _load_scenario(bundle_dir)
    src_path = scenario.get("_meta", {}).get("source_path")
    base_dir = Path(src_path).parent if src_path else None
    map_info = load_map_info(scenario["scene"], base_dir=base_dir)
    env = GridEnvironment(
        "verify-wall-proximity",
        map_info.binary_map,
        pixels_per_meter=map_info.pixels_per_meter,
    )
    free_mask = (env.map != OBSTACLE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)

    total = 0
    near = 0
    by_agent: dict[int, dict[str, float]] = {}

    for row in rows:
        step = int(row["step"])
        for agent in row["agents"]:
            aid = int(agent["id"])
            radius = float(agent.get("radius", 0.0))
            rr, cc = env._world_to_map(
                np.array([float(agent["x"]), float(agent["y"])], dtype=float)
            )
            if rr < 0 or cc < 0 or rr >= env.map.shape[0] or cc >= env.map.shape[1]:
                continue

            required = (radius + max(0.0, near_wall_buffer_m)) * env.pixels_per_meter
            available = float(clearance_px[rr, cc])
            total += 1
            stats = by_agent.setdefault(aid, {"total": 0.0, "near": 0.0, "last_step": float(step)})
            stats["total"] += 1.0
            stats["last_step"] = float(step)

            if available + 1e-6 < required:
                near += 1
                stats["near"] += 1.0

    frac = 0.0 if total == 0 else float(near / total)
    top_agents = sorted(
        (
            (
                aid,
                0.0 if vals["total"] <= 0 else float(vals["near"] / vals["total"]),
                int(vals["near"]),
                int(vals["total"]),
            )
            for aid, vals in by_agent.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return {
        "name": "wall_proximity_fraction",
        "pass": bool(frac <= max_fraction + 1e-9),
        "near_wall_fraction": frac,
        "max_fraction": float(max_fraction),
        "near_wall_buffer_m": float(near_wall_buffer_m),
        "samples": int(total),
        "near_samples": int(near),
        "top_agents": [
            {
                "agent_id": int(aid),
                "near_fraction": float(agent_frac),
                "near_steps": int(near_n),
                "samples": int(n),
            }
            for aid, agent_frac, near_n, n in top_agents
        ],
    }


def validate_no_teleport(state_path: Path, teleport_thresh: float) -> dict:
    rows = load_state_rows(state_path)
    last_pos: dict[int, tuple[float, float]] = {}
    violations = []

    for row in rows:
        step = int(row["step"])
        for agent in row["agents"]:
            aid = int(agent["id"])
            pos = (float(agent["x"]), float(agent["y"]))
            if aid in last_pos:
                d = math.hypot(pos[0] - last_pos[aid][0], pos[1] - last_pos[aid][1])
                if d > teleport_thresh:
                    violations.append({"step": step, "agent_id": aid, "delta": d})
            last_pos[aid] = pos

    return {
        "name": "no_teleport",
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations[:40],
        "threshold": teleport_thresh,
    }


def validate_speed_accel_bounds(
    state_path: Path, dt: float, max_speed: float, max_accel: float
) -> dict:
    rows = load_state_rows(state_path)
    speed_viol = []
    accel_viol = []
    last_vel: dict[int, tuple[float, float]] = {}

    for row in rows:
        step = int(row["step"])
        for agent in row["agents"]:
            aid = int(agent["id"])
            vx, vy = float(agent["vx"]), float(agent["vy"])
            speed = math.hypot(vx, vy)
            speed_limit = max(max_speed, float(agent.get("max_speed", max_speed))) * 1.25
            if speed > speed_limit + 1e-6:
                speed_viol.append({"step": step, "agent_id": aid, "speed": speed})

            if aid in last_vel:
                ax = (vx - last_vel[aid][0]) / max(dt, 1e-8)
                ay = (vy - last_vel[aid][1]) / max(dt, 1e-8)
                accel = math.hypot(ax, ay)
                accel_limit = max_accel * 4.0
                if step > 1 and accel > accel_limit + 1e-6:
                    accel_viol.append({"step": step, "agent_id": aid, "accel": accel})
            last_vel[aid] = (vx, vy)

    return {
        "name": "speed_accel_bounds",
        "pass": (len(speed_viol) == 0 and len(accel_viol) == 0),
        "num_speed_violations": len(speed_viol),
        "num_accel_violations": len(accel_viol),
        "speed_violations": speed_viol[:40],
        "accel_violations": accel_viol[:40],
        "max_speed": max_speed,
        "max_accel": max_accel,
    }


def _wrap_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def validate_motion_jitter(
    state_path: Path,
    *,
    min_speed: float,
    max_flip_rate: float,
) -> dict:
    rows = load_state_rows(state_path)
    headings: dict[int, list[float]] = {}

    for row in rows:
        for agent in row["agents"]:
            aid = int(agent["id"])
            vx, vy = float(agent["vx"]), float(agent["vy"])
            speed = math.hypot(vx, vy)
            if speed < min_speed:
                continue
            headings.setdefault(aid, []).append(math.atan2(vy, vx))

    by_agent = []
    for aid, seq in headings.items():
        if len(seq) < 3:
            continue
        diffs = [_wrap_angle(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
        signs = [0 if abs(d) < 1e-3 else (1 if d > 0 else -1) for d in diffs]
        signs = [s for s in signs if s != 0]
        if len(signs) < 2:
            continue
        flips = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
        flip_rate = flips / max(1, len(signs) - 1)
        by_agent.append(
            {
                "agent_id": int(aid),
                "flip_rate": float(flip_rate),
                "samples": int(len(signs)),
            }
        )

    worst = 0.0 if not by_agent else max(float(r["flip_rate"]) for r in by_agent)
    avg = (
        0.0 if not by_agent else float(sum(float(r["flip_rate"]) for r in by_agent) / len(by_agent))
    )

    return {
        "name": "motion_jitter",
        "pass": bool(worst <= max_flip_rate + 1e-9),
        "max_flip_rate": float(max_flip_rate),
        "min_speed": float(min_speed),
        "worst_flip_rate": float(worst),
        "avg_flip_rate": float(avg),
        "agents": sorted(by_agent, key=lambda x: x["flip_rate"], reverse=True)[:12],
    }


def validate_token_exclusivity(events_path: Path) -> dict:
    events = load_events(events_path)
    holder = None
    violations = []

    for ev in events:
        et = ev.get("event_type")
        aid = ev.get("agent_id")
        step = int(ev.get("step", -1))

        if et == "door_token_acquire":
            if holder is not None and holder != aid:
                violations.append(
                    {
                        "step": step,
                        "reason": "acquire_without_release",
                        "previous_holder": holder,
                        "new_holder": aid,
                    }
                )
            holder = aid

        if et == "door_token_release":
            if holder is None:
                violations.append(
                    {"step": step, "reason": "release_without_holder", "agent_id": aid}
                )
            elif holder != aid:
                violations.append(
                    {
                        "step": step,
                        "reason": "release_by_non_holder",
                        "holder": holder,
                        "agent_id": aid,
                    }
                )
            holder = None

    return {
        "name": "token_exclusivity",
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations[:40],
    }


def validate_deadlock_bounded(
    state_path: Path,
    dt: float,
    deadlock_seconds: float,
    speed_thresh: float = 0.015,
) -> dict:
    rows = load_state_rows(state_path)
    threshold_steps = max(1, int(round(deadlock_seconds / max(dt, 1e-8))))
    goal_tol = 0.2

    streak: dict[int, int] = {}
    last_kind: dict[int, str] = {}
    final_step = int(rows[-1]["step"]) if rows else -1
    violations = []

    for row in rows:
        int(row["step"])
        for agent in row["agents"]:
            aid = int(agent["id"])
            kind = str(agent.get("kind", "unknown"))
            behavior = str(agent.get("behavior", ""))
            streak.setdefault(aid, 0)
            last_kind[aid] = kind

            if behavior in {"DONE", "WAIT", "YIELDING"}:
                streak[aid] = 0
                continue

            speed = math.hypot(float(agent["vx"]), float(agent["vy"]))
            dist_goal = math.hypot(
                float(agent["goal_x"]) - float(agent["x"]),
                float(agent["goal_y"]) - float(agent["y"]),
            )
            if speed < speed_thresh and dist_goal > goal_tol:
                streak[aid] += 1
            else:
                streak[aid] = 0

    for aid, stuck_steps in streak.items():
        if stuck_steps > threshold_steps:
            violations.append(
                {
                    "step": final_step,
                    "agent_id": int(aid),
                    "kind": str(last_kind.get(aid, "unknown")),
                    "stuck_steps": int(stuck_steps),
                }
            )

    return {
        "name": "deadlock_bounded",
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations[:40],
        "threshold_steps": threshold_steps,
        "speed_thresh": speed_thresh,
    }


def validate_agent_stop_duration(
    state_path: Path,
    dt: float,
    max_stop_seconds: float,
    stop_speed_thresh: float,
    goal_tol: float = 0.2,
) -> dict:
    rows = load_state_rows(state_path)
    threshold_steps = max(1, int(round(max_stop_seconds / max(dt, 1e-8))))

    streak: dict[int, int] = {}
    longest: dict[int, int] = {}
    first_violation_logged: dict[int, bool] = {}
    violations = []

    for row in rows:
        step = int(row["step"])
        for agent in row["agents"]:
            aid = int(agent["id"])
            kind = str(agent.get("kind", "unknown"))
            behavior = str(agent.get("behavior", ""))
            speed = math.hypot(float(agent["vx"]), float(agent["vy"]))
            dist_goal = math.hypot(
                float(agent["goal_x"]) - float(agent["x"]),
                float(agent["goal_y"]) - float(agent["y"]),
            )

            streak.setdefault(aid, 0)
            longest.setdefault(aid, 0)
            first_violation_logged.setdefault(aid, False)

            if behavior in {"DONE", "WAIT", "YIELDING"}:
                streak[aid] = 0
                first_violation_logged[aid] = False
                continue

            if speed < stop_speed_thresh and dist_goal > goal_tol:
                streak[aid] += 1
                longest[aid] = max(longest[aid], streak[aid])
                if streak[aid] > threshold_steps and not first_violation_logged[aid]:
                    violations.append(
                        {
                            "step": step,
                            "agent_id": aid,
                            "kind": kind,
                            "stopped_steps": streak[aid],
                            "stopped_seconds": streak[aid] * dt,
                        }
                    )
                    first_violation_logged[aid] = True
            else:
                streak[aid] = 0
                first_violation_logged[aid] = False

    worst = sorted(((aid, n) for aid, n in longest.items()), key=lambda x: x[1], reverse=True)[:10]
    worst_fmt = [
        {"agent_id": int(aid), "max_stopped_steps": int(n), "max_stopped_seconds": float(n * dt)}
        for aid, n in worst
    ]

    return {
        "name": "agent_stop_duration",
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "max_stop_seconds": float(max_stop_seconds),
        "stop_speed_thresh": float(stop_speed_thresh),
        "goal_tolerance": float(goal_tol),
        "top_longest_stops": worst_fmt,
        "violations": violations[:40],
    }


def validate_robot_progress(
    state_path: Path,
    min_progress: float = 0.3,
    goal_tol: float = 0.2,
    reference_steps: int = 80,
) -> dict:
    rows = load_state_rows(state_path)
    robot_rows = []
    for row in rows:
        for agent in row["agents"]:
            if agent.get("kind") == "robot":
                robot_rows.append(agent)
                break

    if not robot_rows:
        return {
            "name": "robot_progress",
            "pass": False,
            "reason": "missing_robot_agent",
            "min_progress": min_progress,
        }

    start = robot_rows[0]
    end = robot_rows[-1]
    sx, sy = float(start["x"]), float(start["y"])
    ex, ey = float(end["x"]), float(end["y"])
    gx, gy = float(start["goal_x"]), float(start["goal_y"])

    start_dist = math.hypot(gx - sx, gy - sy)
    end_dist = math.hypot(gx - ex, gy - ey)
    progress = 0.0 if start_dist < 1e-8 else (start_dist - end_dist) / start_dist
    reached = end_dist <= goal_tol
    horizon_scale = min(1.0, len(rows) / max(1.0, float(reference_steps)))
    effective_min_progress = min_progress * horizon_scale
    passed = bool(reached or (progress >= effective_min_progress))

    return {
        "name": "robot_progress",
        "pass": passed,
        "start_goal_dist": start_dist,
        "end_goal_dist": end_dist,
        "progress_fraction": progress,
        "reached_goal": reached,
        "min_progress": min_progress,
        "effective_min_progress": effective_min_progress,
        "horizon_steps": len(rows),
    }


def validate_log_render_sync(state_path: Path, frames_dir: Path) -> dict:
    rows = load_state_rows(state_path)
    frame_files = sorted(frames_dir.glob("*.png"))

    # Initial state row included; rendering should include every row.
    expected = len(rows)
    actual = len(frame_files)
    ok = abs(expected - actual) <= 1

    return {
        "name": "log_render_sync",
        "pass": bool(ok),
        "expected_frames": expected,
        "actual_frames": actual,
        "epsilon": 1,
    }


def run_numeric_invariants(bundle_dir: Path) -> dict:
    state_path = bundle_dir / "state.jsonl"
    events_path = bundle_dir / "events.jsonl"
    scenario = _load_scenario(bundle_dir)
    eval_cfg = scenario.get("evaluation", {})

    dt = float(scenario["horizon"]["dt"])
    teleport_thresh = float(eval_cfg.get("teleport_thresh", 1.0))
    max_speed = float(eval_cfg.get("max_speed", 1.25))
    max_accel = float(eval_cfg.get("max_accel", 4.0))
    deadlock_seconds = float(eval_cfg.get("deadlock_seconds", 4.0))
    deadlock_speed_thresh = float(eval_cfg.get("deadlock_speed_thresh", 0.015))
    min_robot_progress = float(eval_cfg.get("min_robot_progress", 0.1))
    wall_clearance_buffer = float(eval_cfg.get("wall_clearance_buffer", 0.0))
    enforce_wall_clearance_buffer = bool(eval_cfg.get("enforce_wall_clearance_buffer", False))
    near_wall_buffer = float(eval_cfg.get("near_wall_buffer", 0.02))
    max_wall_proximity_fraction = float(eval_cfg.get("max_wall_proximity_fraction", 0.14))
    max_heading_flip_rate = float(eval_cfg.get("max_heading_flip_rate", 0.82))
    jitter_speed_thresh = float(eval_cfg.get("jitter_speed_thresh", 0.06))
    max_agent_stop_seconds = float(eval_cfg.get("max_agent_stop_seconds", 8.0))
    stop_speed_thresh = float(eval_cfg.get("stop_speed_thresh", 0.02))

    checks = [
        validate_units_metadata(scenario),
        validate_scenario_feasibility(bundle_dir),
        validate_anchor_layout(bundle_dir),
    ]
    if not bool(eval_cfg.get("expected_wall_penetration", False)):
        checks.append(validate_no_wall_penetration(state_path, bundle_dir))
        if enforce_wall_clearance_buffer:
            checks.append(
                validate_wall_clearance_buffer(
                    state_path,
                    bundle_dir,
                    clearance_buffer_m=wall_clearance_buffer,
                    max_fraction=max_wall_proximity_fraction,
                )
            )
        checks.append(
            validate_wall_proximity(
                state_path,
                bundle_dir,
                near_wall_buffer_m=near_wall_buffer,
                max_fraction=max_wall_proximity_fraction,
            )
        )
    checks.extend(
        [
            validate_no_teleport(state_path, teleport_thresh),
            validate_speed_accel_bounds(
                state_path, dt=dt, max_speed=max_speed, max_accel=max_accel
            ),
            validate_motion_jitter(
                state_path,
                min_speed=jitter_speed_thresh,
                max_flip_rate=max_heading_flip_rate,
            ),
        ]
    )

    checks.append(
        validate_deadlock_bounded(
            state_path,
            dt=dt,
            deadlock_seconds=deadlock_seconds,
            speed_thresh=deadlock_speed_thresh,
        )
    )
    checks.append(
        validate_agent_stop_duration(
            state_path,
            dt=dt,
            max_stop_seconds=max_agent_stop_seconds,
            stop_speed_thresh=stop_speed_thresh,
        )
    )
    checks.append(validate_robot_progress(state_path, min_progress=min_robot_progress))

    if "doorway" in str(scenario.get("id", "")).lower():
        checks.append(validate_token_exclusivity(events_path))

    frames_dir = bundle_dir / "frames"
    if frames_dir.exists():
        checks.append(validate_log_render_sync(state_path, frames_dir))

    passed = all(c.get("pass", False) for c in checks)
    return {
        "overall_pass": passed,
        "checks": checks,
    }


def sample_key_frames(bundle_dir: Path, num_frames: int = 8) -> list[str]:
    frames_dir = bundle_dir / "frames"
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        return []

    if len(frame_files) <= num_frames:
        return [str(p) for p in frame_files]

    idxs = np.linspace(0, len(frame_files) - 1, num_frames).astype(int)
    return [str(frame_files[i]) for i in idxs]


def build_visual_summary(bundle_dir: Path, invariants: dict) -> dict:
    summary_path = bundle_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    scenario = _load_scenario(bundle_dir)

    diagnostics_path = bundle_dir / "frames" / "render_diagnostics.json"
    if diagnostics_path.exists():
        with diagnostics_path.open("r", encoding="utf-8") as f:
            render_diag = json.load(f)
    else:
        render_diag = {}

    return {
        "bundle_dir": str(bundle_dir),
        "scenario_id": str(summary.get("scenario_id", scenario.get("id", ""))),
        "expected_high_interaction": bool(
            scenario.get("evaluation", {}).get("expected_high_interaction", False)
        ),
        "map": scenario.get("scene", {}).get("map", {}).get("resolved", {}),
        "metrics": summary.get("metrics", {}),
        "invariants": invariants,
        "frame_count": len(list((bundle_dir / "frames").glob("*.png"))),
        "has_video": (bundle_dir / "frames" / "video.mp4").exists(),
        "render_diagnostics": render_diag,
    }


def check_video_artifact(bundle_dir: Path) -> dict:
    video_path = bundle_dir / "frames" / "video.mp4"
    if not video_path.exists():
        return {"name": "video_artifact", "pass": False, "reason": "missing_video"}

    cap = cv2.VideoCapture(str(video_path))
    ok, _ = cap.read()
    cap.release()
    return {
        "name": "video_artifact",
        "pass": bool(ok),
        "reason": "ok" if ok else "unreadable_video",
        "path": str(video_path),
    }
