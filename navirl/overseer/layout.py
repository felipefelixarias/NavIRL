"""Layout generation and spatial arrangement utilities.

Provides automated layout generation for placing agents, obstacles, and
other entities in simulation environments. Includes coordinate conversion,
collision detection, and spatial optimization algorithms.

Functions
---------
_world_from_rc -- Convert grid coordinates to world coordinates
generate_layout -- Generate optimized spatial arrangements
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import cv2
import numpy as np

from navirl.core.maps import FREE_SPACE, load_map_info


def _world_from_rc(
    row: int, col: int, *, width_px: int, height_px: int, ppm: float
) -> tuple[float, float]:
    x = (float(col) - float(width_px) * 0.5) / ppm
    y = (float(row) - float(height_px) * 0.5) / ppm
    return float(x), float(y)


def _build_candidates(map_info, min_clearance_m: float) -> list[dict]:
    free_mask = (map_info.binary_map == FREE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    required_px = max(1.0, float(min_clearance_m) * float(map_info.pixels_per_meter))
    candidate_mask = clearance_px >= required_px
    rc = np.argwhere(candidate_mask)
    if rc.size == 0:
        rc = np.argwhere(free_mask > 0)

    out = []
    for row, col in rc:
        x, y = _world_from_rc(
            int(row),
            int(col),
            width_px=int(map_info.width_px),
            height_px=int(map_info.height_px),
            ppm=float(map_info.pixels_per_meter),
        )
        out.append(
            {
                "row": int(row),
                "col": int(col),
                "x": float(x),
                "y": float(y),
                "clearance_m": float(clearance_px[row, col] / map_info.pixels_per_meter),
            }
        )
    return out


def _pick_spread_points(
    pool: list[dict],
    count: int,
    *,
    rng: random.Random,
    min_sep: float,
) -> list[dict]:
    if not pool or count <= 0:
        return []

    sampled = pool
    if len(sampled) > 4000:
        sampled = rng.sample(sampled, 4000)
    sampled = sorted(
        sampled, key=lambda p: (float(p["clearance_m"]), -abs(float(p["y"]))), reverse=True
    )

    selected: list[dict] = []
    for p in sampled:
        if len(selected) >= count:
            break
        ok = True
        for q in selected:
            d = math.hypot(float(p["x"]) - float(q["x"]), float(p["y"]) - float(q["y"]))
            if d < min_sep:
                ok = False
                break
        if ok:
            selected.append(p)

    while len(selected) < count and sampled:
        selected.append(rng.choice(sampled))
    return selected[:count]


def _edge_split(
    candidates: list[dict], *, axis: str, map_w: float, map_h: float
) -> tuple[list[dict], list[dict]]:
    if axis == "x":
        edge = 0.28 * map_w
        lo = [p for p in candidates if float(p["x"]) <= -edge]
        hi = [p for p in candidates if float(p["x"]) >= edge]
    else:
        edge = 0.28 * map_h
        lo = [p for p in candidates if float(p["y"]) <= -edge]
        hi = [p for p in candidates if float(p["y"]) >= edge]
    return lo, hi


def _pick_robot_pair(
    side_a: list[dict],
    side_b: list[dict],
    *,
    axis: str,
    rng: random.Random,
) -> tuple[dict, dict]:
    if not side_a or not side_b:
        raise ValueError("Insufficient side candidates for robot pair selection.")

    cross = "y" if axis == "x" else "x"
    rank_a = sorted(side_a, key=lambda p: (abs(float(p[cross])), -float(p["clearance_m"])))
    rank_b = sorted(side_b, key=lambda p: (abs(float(p[cross])), -float(p["clearance_m"])))
    return rank_a[0] if rank_a else rng.choice(side_a), rank_b[0] if rank_b else rng.choice(side_b)


def _bottleneck_score(map_info) -> float:
    free_mask = (map_info.binary_map == FREE_SPACE).astype(np.uint8)
    clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    vals = clearance_px[clearance_px > 0]
    if vals.size == 0:
        return 0.0
    q20 = float(np.percentile(vals, 20))
    center_r0 = int(round(map_info.height_px * 0.30))
    center_r1 = int(round(map_info.height_px * 0.70))
    center_c0 = int(round(map_info.width_px * 0.30))
    center_c1 = int(round(map_info.width_px * 0.70))
    center = clearance_px[center_r0:center_r1, center_c0:center_c1]
    if center.size == 0:
        return 0.0
    frac_narrow_center = float(np.mean(center <= q20))
    return frac_narrow_center


def suggest_layout(
    scenario: dict,
    *,
    objective: str = "auto",
    humans_count: int | None = None,
    seed: int = 17,
    base_dir: Path | None = None,
) -> dict:
    rng = random.Random(int(seed))
    map_info = load_map_info(scenario["scene"], base_dir=base_dir)

    human_radius = float(scenario.get("humans", {}).get("radius", 0.18))
    robot_radius = float(scenario.get("robot", {}).get("radius", 0.2))
    count = (
        int(humans_count)
        if humans_count is not None
        else int(scenario.get("humans", {}).get("count", 0))
    )
    count = max(0, count)

    min_clearance = max(robot_radius, human_radius) + 0.05
    candidates = _build_candidates(map_info, min_clearance_m=min_clearance)
    if not candidates:
        raise ValueError("No traversable candidates found for overseer layout suggestion.")

    map_w = float(map_info.width_m)
    map_h = float(map_info.height_m)
    major_axis = "x" if map_w >= map_h else "y"
    bottleneck = _bottleneck_score(map_info)

    resolved_objective = str(objective).strip().lower()
    if resolved_objective == "auto":
        if bottleneck > 0.20:
            resolved_objective = "bottleneck_showcase"
        else:
            resolved_objective = "cross_flow"

    side_a, side_b = _edge_split(candidates, axis=major_axis, map_w=map_w, map_h=map_h)
    if not side_a or not side_b:
        side_a = sorted(candidates, key=lambda p: float(p[major_axis]))[
            : max(1, len(candidates) // 3)
        ]
        side_b = sorted(candidates, key=lambda p: float(p[major_axis]), reverse=True)[
            : max(1, len(candidates) // 3)
        ]

    robot_start, robot_goal = _pick_robot_pair(side_a, side_b, axis=major_axis, rng=rng)
    if resolved_objective in {"comfort_showcase", "comfort"}:
        robot_goal, robot_start = robot_start, robot_goal

    starts: list[dict] = []
    goals: list[dict] = []
    min_human_sep = max(0.25, 2.1 * human_radius)
    half = count // 2
    n_a_to_b = half + (count % 2)
    n_b_to_a = half

    starts.extend(_pick_spread_points(side_a, n_a_to_b, rng=rng, min_sep=min_human_sep))
    goals.extend(_pick_spread_points(side_b, n_a_to_b, rng=rng, min_sep=min_human_sep))
    starts.extend(_pick_spread_points(side_b, n_b_to_a, rng=rng, min_sep=min_human_sep))
    goals.extend(_pick_spread_points(side_a, n_b_to_a, rng=rng, min_sep=min_human_sep))

    starts = starts[:count]
    goals = goals[:count]
    while len(starts) < count:
        starts.append(rng.choice(candidates))
    while len(goals) < count:
        goals.append(rng.choice(candidates))

    human_starts = [[float(p["x"]), float(p["y"])] for p in starts]
    human_goals = [[float(p["x"]), float(p["y"])] for p in goals]
    robot_start_xy = [float(robot_start["x"]), float(robot_start["y"])]
    robot_goal_xy = [float(robot_goal["x"]), float(robot_goal["y"])]

    min_clearance_used = min(
        [float(robot_start["clearance_m"]), float(robot_goal["clearance_m"])]
        + [float(p["clearance_m"]) for p in starts]
        + [float(p["clearance_m"]) for p in goals]
    )

    return {
        "objective": resolved_objective,
        "map_id": str(map_info.map_id),
        "map_source": str(map_info.source),
        "map_size_m": [float(map_w), float(map_h)],
        "major_axis": major_axis,
        "bottleneck_score": float(bottleneck),
        "robot_start": robot_start_xy,
        "robot_goal": robot_goal_xy,
        "human_starts": human_starts,
        "human_goals": human_goals,
        "humans_count": int(count),
        "quality": {
            "min_clearance_used_m": float(min_clearance_used),
            "target_min_clearance_m": float(min_clearance),
        },
    }


def apply_layout_to_scenario(scenario: dict, suggestion: dict) -> dict:
    patched = copy.deepcopy(scenario)
    patched.setdefault("humans", {})
    patched.setdefault("robot", {})

    patched["humans"]["count"] = int(suggestion["humans_count"])
    patched["humans"]["starts"] = [list(map(float, p)) for p in suggestion["human_starts"]]
    patched["humans"]["goals"] = [list(map(float, p)) for p in suggestion["human_goals"]]
    patched["robot"]["start"] = list(map(float, suggestion["robot_start"]))
    patched["robot"]["goal"] = list(map(float, suggestion["robot_goal"]))
    patched.setdefault("_meta", {})
    patched["_meta"]["aegis_layout"] = {
        "objective": suggestion.get("objective", "unknown"),
        "quality": suggestion.get("quality", {}),
    }
    return patched
