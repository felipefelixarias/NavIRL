from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from navirl.backends.grid2d.constants import OBSTACLE_SPACE
from navirl.backends.grid2d.maps import load_map_info
from navirl.metrics.base import MetricsCollector


def _load_state_rows(state_path: Path) -> list[dict]:
    rows = []
    with state_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _world_to_rc(
    x: float,
    y: float,
    shape: tuple[int, int],
    pixels_per_meter: float,
) -> tuple[int, int]:
    h, w = shape
    row = int(round(y * pixels_per_meter + h / 2.0))
    col = int(round(x * pixels_per_meter + w / 2.0))
    return row, col


def _agent_speed(agent: dict) -> float:
    return float(math.hypot(agent["vx"], agent["vy"]))


def _pair_dist(a: dict, b: dict) -> float:
    return float(math.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def _angle_wrap(rad: float) -> float:
    while rad > math.pi:
        rad -= 2 * math.pi
    while rad < -math.pi:
        rad += 2 * math.pi
    return rad


class StandardMetrics(MetricsCollector):
    """Stable metrics collection for social navigation episodes."""

    def compute(self, state_path: Path, scenario: dict) -> dict:
        rows = _load_state_rows(state_path)
        if not rows:
            raise ValueError(f"No rows in state log: {state_path}")

        dt = float(scenario["horizon"]["dt"])
        eval_cfg = scenario.get("evaluation", {})
        intrusion_delta = float(eval_cfg.get("intrusion_delta", 0.45))
        deadlock_seconds = float(eval_cfg.get("deadlock_seconds", 4.0))
        deadlock_speed_thresh = float(eval_cfg.get("deadlock_speed_thresh", 0.015))
        deadlock_steps = max(1, int(round(deadlock_seconds / dt)))
        goal_tol = 0.2

        scene_cfg = scenario.get("scene", {})
        src_path = scenario.get("_meta", {}).get("source_path")
        base_dir = Path(src_path).parent if src_path else None
        map_info = load_map_info(scene_cfg, base_dir=base_dir)
        map_img = map_info.binary_map
        ppm = float(map_info.pixels_per_meter)

        collisions_agent_agent = 0
        collisions_agent_obstacle = 0
        robot_human_min_dists: list[float] = []
        human_human_min_dists: list[float] = []
        intrusion_steps = 0

        path_len: dict[int, float] = {}
        last_pos: dict[int, tuple[float, float]] = {}
        heading_series: dict[int, list[float]] = {}
        vel_series: dict[int, list[tuple[float, float]]] = {}
        low_speed_streak: dict[int, int] = {}

        robot_id = None
        robot_goal_reached_step = None

        for row in rows:
            agents = row["agents"]
            by_id = {int(a["id"]): a for a in agents}

            for aid, a in by_id.items():
                if aid not in path_len:
                    path_len[aid] = 0.0
                    low_speed_streak[aid] = 0
                    heading_series[aid] = []
                    vel_series[aid] = []

                if a["kind"] == "robot":
                    robot_id = aid

                pos = (float(a["x"]), float(a["y"]))
                if aid in last_pos:
                    path_len[aid] += float(
                        math.hypot(pos[0] - last_pos[aid][0], pos[1] - last_pos[aid][1])
                    )
                last_pos[aid] = pos

                vx, vy = float(a["vx"]), float(a["vy"])
                vel_series[aid].append((vx, vy))
                if abs(vx) + abs(vy) > 1e-8:
                    heading_series[aid].append(math.atan2(vy, vx))

                speed = math.hypot(vx, vy)
                dist_goal = math.hypot(float(a["goal_x"]) - pos[0], float(a["goal_y"]) - pos[1])
                behavior = str(a.get("behavior", ""))
                if behavior in {"DONE", "WAIT", "YIELDING"}:
                    low_speed_streak[aid] = 0
                elif speed < deadlock_speed_thresh and dist_goal > goal_tol:
                    low_speed_streak[aid] += 1
                else:
                    low_speed_streak[aid] = 0

                row_rc, col_rc = _world_to_rc(pos[0], pos[1], map_img.shape, ppm)
                if (
                    row_rc < 0
                    or col_rc < 0
                    or row_rc >= map_img.shape[0]
                    or col_rc >= map_img.shape[1]
                ):
                    collisions_agent_obstacle += 1
                elif map_img[row_rc, col_rc] == OBSTACLE_SPACE:
                    collisions_agent_obstacle += 1

            # pairwise metrics
            ids = sorted(by_id)
            robot_human_frame_min = float("inf")
            human_human_frame_min = float("inf")
            for i, aid in enumerate(ids):
                for bid in ids[i + 1 :]:
                    a = by_id[aid]
                    b = by_id[bid]
                    d = _pair_dist(a, b)
                    if d < float(a["radius"]) + float(b["radius"]):
                        collisions_agent_agent += 1

                    if a["kind"] == "robot" and b["kind"] == "human":
                        robot_human_frame_min = min(robot_human_frame_min, d)
                    elif a["kind"] == "human" and b["kind"] == "robot":
                        robot_human_frame_min = min(robot_human_frame_min, d)
                    elif a["kind"] == "human" and b["kind"] == "human":
                        human_human_frame_min = min(human_human_frame_min, d)

            if math.isfinite(robot_human_frame_min):
                robot_human_min_dists.append(robot_human_frame_min)
                if robot_human_frame_min < intrusion_delta:
                    intrusion_steps += 1
            if math.isfinite(human_human_frame_min):
                human_human_min_dists.append(human_human_frame_min)

            if robot_id is not None and robot_goal_reached_step is None:
                r = by_id[robot_id]
                if math.hypot(r["goal_x"] - r["x"], r["goal_y"] - r["y"]) <= goal_tol:
                    robot_goal_reached_step = int(row["step"])

        oscillation_scores = []
        for _aid, headings in heading_series.items():
            if len(headings) < 3:
                continue
            diffs = [_angle_wrap(headings[i + 1] - headings[i]) for i in range(len(headings) - 1)]
            signs = [0 if abs(d) < 1e-4 else (1 if d > 0 else -1) for d in diffs]
            signs = [s for s in signs if s != 0]
            if len(signs) < 2:
                continue
            flips = sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])
            oscillation_scores.append(flips / (len(signs) - 1))

        jerk_values = []
        for _aid, vv in vel_series.items():
            if len(vv) < 3:
                continue
            arr = np.asarray(vv, dtype=float)
            acc = np.diff(arr, axis=0) / max(dt, 1e-8)
            if len(acc) < 2:
                continue
            jerk = np.diff(acc, axis=0) / max(dt, 1e-8)
            jerk_values.extend(np.linalg.norm(jerk, axis=1).tolist())

        success = float(robot_goal_reached_step is not None)
        horizon_steps = max(1, len(rows))
        deadlocked_ids = {
            aid for aid, streak in low_speed_streak.items() if streak >= deadlock_steps
        }

        def _safe_stats(vals: list[float]) -> tuple[float, float, float]:
            if not vals:
                return float("inf"), float("inf"), float("inf")
            arr = np.asarray(vals, dtype=float)
            return float(arr.min()), float(arr.mean()), float(np.percentile(arr, 5))

        rh_min, rh_mean, rh_p05 = _safe_stats(robot_human_min_dists)
        hh_min, hh_mean, hh_p05 = _safe_stats(human_human_min_dists)

        report = {
            "collisions_agent_agent": int(collisions_agent_agent),
            "collisions_agent_obstacle": int(collisions_agent_obstacle),
            "min_dist_robot_human_min": rh_min,
            "min_dist_robot_human_mean": rh_mean,
            "min_dist_robot_human_p05": rh_p05,
            "min_dist_human_human_min": hh_min,
            "min_dist_human_human_mean": hh_mean,
            "min_dist_human_human_p05": hh_p05,
            "intrusion_rate": float(intrusion_steps / horizon_steps),
            "deadlock_count": int(len(deadlocked_ids)),
            "oscillation_score": float(np.mean(oscillation_scores)) if oscillation_scores else 0.0,
            "jerk_proxy": float(np.mean(jerk_values)) if jerk_values else 0.0,
            "path_length_robot": float(path_len.get(robot_id, 0.0)),
            "time_to_goal_robot": float(robot_goal_reached_step * dt)
            if robot_goal_reached_step is not None
            else float("inf"),
            "success_rate": success,
            "horizon_steps": int(horizon_steps),
            "dt": float(dt),
            "map_pixels_per_meter": float(ppm),
            "map_meters_per_pixel": float(map_info.meters_per_pixel),
            "map_width_m": float(map_info.width_m),
            "map_height_m": float(map_info.height_m),
        }
        return report


def compute_metrics_from_bundle(state_path: Path) -> dict:
    bundle_dir = state_path.parent
    scenario_path = bundle_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found next to state log: {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    return StandardMetrics().compute(state_path, scenario)
