from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from navirl.overseer.provider import ProviderConfig
from navirl.overseer.review import AEGIS_NAME, run_aegis_review

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["overall_pass", "confidence", "violations", "status", "judge_type"],
    "properties": {
        "overall_pass": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "evidence"],
                "properties": {
                    "type": {"type": "string"},
                    "evidence": {"type": "string"},
                    "severity": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "status": {"type": "string", "enum": ["pass", "fail", "needs_human_review"]},
        "judge_type": {"type": "string"},
        "notes": {"type": "string"},
    },
    "additionalProperties": True,
}


def _frame_quality(frame_paths: list[str]) -> dict:
    if not frame_paths:
        return {
            "avg_edge_density": 0.0,
            "avg_motion": 0.0,
            "num_frames": 0,
        }

    sample_n = min(12, len(frame_paths))
    idxs = np.linspace(0, len(frame_paths) - 1, sample_n).astype(int)

    gray_frames = []
    edges = []
    for i in idxs:
        img = cv2.imread(frame_paths[i], cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
        edge = cv2.Canny(gray, 70, 150)
        edges.append(float(np.mean(edge > 0)))

    motions = []
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        motions.append(float(np.mean(diff)))

    return {
        "avg_edge_density": float(np.mean(edges)) if edges else 0.0,
        "avg_motion": float(np.mean(motions)) if motions else 0.0,
        "num_frames": len(gray_frames),
    }


def _heuristic_judge(
    summary: dict,
    frame_paths: list[str],
    confidence_threshold: float,
    require_video: bool,
) -> dict:
    """Perform comprehensive quality assessment of simulation scenarios using heuristic rules.

    This function implements a multi-criteria evaluation system that checks simulation
    scenarios against various quality and safety thresholds. It analyzes invariant
    violations, collision metrics, progress indicators, and infrastructure requirements
    to determine scenario validity.

    The function uses different thresholds based on whether high interaction is expected,
    allowing for more lenient collision and intrusion rates in crowded scenarios while
    maintaining strict safety requirements.

    Args:
        summary: Simulation summary containing metrics, checks, and metadata.
            Expected keys include 'invariants', 'metrics', 'map', 'frame_count',
            'has_video', 'expected_high_interaction', and 'bundle_dir'.
        frame_paths: List of paths to frame files for visual validation.
            Currently used for frame count validation (minimum 20 frames required).
        confidence_threshold: Minimum confidence level for judge decisions (0.0-1.0).
            Used to determine when human review is needed vs automatic pass/fail.
        require_video: Whether video artifacts are mandatory for this judge mode.
            When True, scenarios without video will be marked as blockers.

    Returns:
        Dictionary containing judge decision with the following structure:
        {
            'violations': List of violation dictionaries with 'type', 'evidence', and 'severity',
            'status': One of 'pass', 'fail', or 'needs_human_review',
            'confidence': Float between 0.0 and 1.0,
            'details': Additional diagnostic information
        }

    Validation Rules:
        - **Blocker violations**: Failed invariant checks, <20 frames, missing video (if required),
          missing/invalid map units, obstacle collisions, deadlocks, excessive agent collisions,
          extreme intrusion rates
        - **Major violations**: Near-limit agent stop duration, near-limit wall proximity,
          moderate collision/intrusion rates, insufficient robot progress
        - **Minor violations**: Low-level collision or intrusion concerns

    Threshold Adjustments:
        - **High interaction scenarios**: More lenient collision (1.45→1.8) and intrusion (0.92→0.98) thresholds
        - **Standard scenarios**: Stricter collision (1.0→1.3) and intrusion (0.72→0.9) thresholds

    Note:
        This function has high cyclomatic complexity (41) due to the comprehensive
        rule set. Consider refactoring into smaller validation functions if adding
        more rules.
    """
    inv = summary.get("invariants", {})
    checks = inv.get("checks", [])
    checks_by_name = {str(c.get("name", "")): c for c in checks if isinstance(c, dict)}
    failed_checks = [c for c in checks if not c.get("pass", False)]
    expected_high_interaction = bool(summary.get("expected_high_interaction", False))

    violations = []

    for check in failed_checks:
        violations.append(
            {
                "type": check.get("name", "unknown"),
                "evidence": f"check_failed: {check.get('name')} in {summary.get('bundle_dir')}",
                "severity": "blocker",
            }
        )

    frame_count = int(summary.get("frame_count", 0))
    if frame_count < 20:
        violations.append(
            {
                "type": "insufficient_frames",
                "evidence": f"frame_count={frame_count} (<20)",
                "severity": "blocker",
            }
        )

    if require_video and not bool(summary.get("has_video", False)):
        violations.append(
            {
                "type": "missing_video",
                "evidence": "full-suite visual judge requires video artifact",
                "severity": "blocker",
            }
        )

    metrics = summary.get("metrics", {})
    map_meta = summary.get("map", {})
    ppm = map_meta.get("pixels_per_meter")
    mpp = map_meta.get("meters_per_pixel")
    if ppm is None or mpp is None:
        violations.append(
            {
                "type": "missing_map_units",
                "evidence": "summary.map.{pixels_per_meter,meters_per_pixel} missing",
                "severity": "blocker",
            }
        )
    else:
        if float(ppm) <= 0.0 or float(mpp) <= 0.0:
            violations.append(
                {
                    "type": "invalid_map_units",
                    "evidence": f"pixels_per_meter={ppm}, meters_per_pixel={mpp}",
                    "severity": "blocker",
                }
            )

    obstacle_collisions = int(metrics.get("collisions_agent_obstacle", 0))
    if obstacle_collisions > 0:
        violations.append(
            {
                "type": "obstacle_collisions",
                "evidence": f"collisions_agent_obstacle={obstacle_collisions}",
                "severity": "blocker",
            }
        )

    deadlock_count = int(metrics.get("deadlock_count", 0))
    if deadlock_count > 0:
        violations.append(
            {
                "type": "deadlock_detected",
                "evidence": f"deadlock_count={deadlock_count}",
                "severity": "blocker",
            }
        )

    stop_check = checks_by_name.get("agent_stop_duration", {})
    if isinstance(stop_check, dict):
        max_stop_s = float(stop_check.get("max_stop_seconds", 0.0))
        longest_stop_s = 0.0
        for item in stop_check.get("top_longest_stops", []):
            if isinstance(item, dict):
                longest_stop_s = max(longest_stop_s, float(item.get("max_stopped_seconds", 0.0)))
        if max_stop_s > 0.0 and longest_stop_s > max_stop_s * 0.9:
            violations.append(
                {
                    "type": "near_limit_agent_stop_duration",
                    "evidence": f"longest_stop={longest_stop_s:.2f}s, limit={max_stop_s:.2f}s",
                    "severity": "major",
                }
            )

    wall_prox = checks_by_name.get("wall_proximity_fraction", {})
    if isinstance(wall_prox, dict):
        frac = float(wall_prox.get("near_wall_fraction", 0.0))
        lim = float(wall_prox.get("max_fraction", 1.0))
        if lim > 0.0 and frac > lim * 0.9:
            violations.append(
                {
                    "type": "near_limit_wall_proximity",
                    "evidence": f"near_wall_fraction={frac:.3f}, max_fraction={lim:.3f}",
                    "severity": "major",
                }
            )

    horizon = max(1, int(metrics.get("horizon_steps", 1)))
    pair_collisions = int(metrics.get("collisions_agent_agent", 0))
    pair_collision_rate = pair_collisions / horizon

    if expected_high_interaction:
        pair_major_thresh = 1.45
        pair_blocker_thresh = 1.8
    else:
        pair_major_thresh = 1.0
        pair_blocker_thresh = 1.3

    if pair_collision_rate > pair_blocker_thresh:
        violations.append(
            {
                "type": "excess_agent_collisions",
                "evidence": f"collisions_agent_agent={pair_collisions}, rate={pair_collision_rate:.3f}",
                "severity": "blocker",
            }
        )
    elif pair_collision_rate > pair_major_thresh:
        violations.append(
            {
                "type": "excess_agent_collisions",
                "evidence": f"collisions_agent_agent={pair_collisions}, rate={pair_collision_rate:.3f}",
                "severity": "major",
            }
        )
    elif pair_collision_rate > 0.35:
        violations.append(
            {
                "type": "excess_agent_collisions",
                "evidence": f"collisions_agent_agent={pair_collisions}, rate={pair_collision_rate:.3f}",
                "severity": "minor",
            }
        )

    intrusion_rate = float(metrics.get("intrusion_rate", 0.0))
    if expected_high_interaction:
        intrusion_major_thresh = 0.92
        intrusion_blocker_thresh = 0.98
    else:
        intrusion_major_thresh = 0.72
        intrusion_blocker_thresh = 0.9

    if intrusion_rate > intrusion_blocker_thresh:
        violations.append(
            {
                "type": "high_intrusion_rate",
                "evidence": f"intrusion_rate={intrusion_rate:.3f}",
                "severity": "blocker",
            }
        )
    elif intrusion_rate > intrusion_major_thresh:
        violations.append(
            {
                "type": "high_intrusion_rate",
                "evidence": f"intrusion_rate={intrusion_rate:.3f}",
                "severity": "major",
            }
        )
    elif intrusion_rate > 0.55:
        violations.append(
            {
                "type": "high_intrusion_rate",
                "evidence": f"intrusion_rate={intrusion_rate:.3f}",
                "severity": "minor",
            }
        )

    robot_progress = next((c for c in checks if c.get("name") == "robot_progress"), None)
    if robot_progress is not None and not bool(robot_progress.get("pass", False)):
        violations.append(
            {
                "type": "insufficient_robot_progress",
                "evidence": (
                    f"progress_fraction={float(robot_progress.get('progress_fraction', 0.0)):.3f}, "
                    f"effective_min_progress="
                    f"{float(robot_progress.get('effective_min_progress', robot_progress.get('min_progress', 0.0))):.3f}"
                ),
                "severity": "major",
            }
        )

    diag = summary.get("render_diagnostics", {})
    if not diag:
        violations.append(
            {
                "type": "missing_render_diagnostics",
                "evidence": "render_diagnostics.json missing",
                "severity": "blocker",
            }
        )
    else:
        total_agents = float(diag.get("total_agents_drawn", 0))
        total_arrows = float(diag.get("total_arrows_drawn", 0))
        avg_agents = float(diag.get("avg_agents_per_frame", 0.0))
        avg_trails = float(diag.get("avg_trail_segments_per_frame", 0.0))
        total_text = float(diag.get("total_text_elements", 0))
        style_version = str(diag.get("style_version", ""))

        if not (style_version.startswith("v2_") or style_version.startswith("v3_")):
            violations.append(
                {
                    "type": "unexpected_render_style",
                    "evidence": f"style_version={style_version}",
                    "severity": "major",
                }
            )

        arrow_coverage = total_arrows / total_agents if total_agents > 0 else 0.0
        if avg_agents <= 3.5:
            arrow_blocker_thresh = 0.45
            arrow_major_thresh = 0.62
        else:
            arrow_blocker_thresh = 0.55
            arrow_major_thresh = 0.7

        if total_agents > 0 and arrow_coverage < arrow_blocker_thresh:
            violations.append(
                {
                    "type": "insufficient_direction_arrows",
                    "evidence": f"arrow_coverage={arrow_coverage:.3f}",
                    "severity": "blocker",
                }
            )
        elif total_agents > 0 and arrow_coverage < arrow_major_thresh:
            violations.append(
                {
                    "type": "insufficient_direction_arrows",
                    "evidence": f"arrow_coverage={arrow_coverage:.3f}",
                    "severity": "major",
                }
            )

        if avg_trails < max(1.0, avg_agents * 0.9):
            violations.append(
                {
                    "type": "insufficient_trail_overlay",
                    "evidence": f"avg_trail_segments_per_frame={avg_trails:.2f}, avg_agents={avg_agents:.2f}",
                    "severity": "blocker",
                }
            )
        elif avg_trails < max(2.0, avg_agents * 1.6):
            violations.append(
                {
                    "type": "insufficient_trail_overlay",
                    "evidence": f"avg_trail_segments_per_frame={avg_trails:.2f}, avg_agents={avg_agents:.2f}",
                    "severity": "major",
                }
            )

        if total_text > frame_count * 2.0:
            violations.append(
                {
                    "type": "overlay_text_clutter",
                    "evidence": f"text_elements={int(total_text)} for frames={frame_count}",
                    "severity": "major",
                }
            )

    quality = _frame_quality(frame_paths)
    if quality["num_frames"] < 4:
        violations.append(
            {
                "type": "frame_decode_failure",
                "evidence": f"decoded_frames={quality['num_frames']}",
                "severity": "blocker",
            }
        )
    else:
        if quality["avg_edge_density"] < 0.005:
            violations.append(
                {
                    "type": "low_visual_detail",
                    "evidence": f"avg_edge_density={quality['avg_edge_density']:.4f}",
                    "severity": "blocker",
                }
            )
        elif quality["avg_edge_density"] < 0.008:
            violations.append(
                {
                    "type": "low_visual_detail",
                    "evidence": f"avg_edge_density={quality['avg_edge_density']:.4f}",
                    "severity": "major",
                }
            )
        avg_agents = float(diag.get("avg_agents_per_frame", 0.0)) if isinstance(diag, dict) else 0.0
        if avg_agents <= 3.5:
            motion_blocker_thresh = 0.14
            motion_major_thresh = 0.22
        elif avg_agents <= 5.0:
            motion_blocker_thresh = 0.3
            motion_major_thresh = 0.42
        else:
            motion_blocker_thresh = 0.35
            motion_major_thresh = 0.58

        if quality["avg_motion"] < motion_blocker_thresh:
            violations.append(
                {
                    "type": "low_scene_motion",
                    "evidence": f"avg_motion={quality['avg_motion']:.3f}",
                    "severity": "blocker",
                }
            )
        elif quality["avg_motion"] < motion_major_thresh:
            violations.append(
                {
                    "type": "low_scene_motion",
                    "evidence": f"avg_motion={quality['avg_motion']:.3f}",
                    "severity": "major",
                }
            )

    severities = [v.get("severity", "minor") for v in violations]
    has_blocker = any(s == "blocker" for s in severities)

    confidence = 0.98
    for sev in severities:
        if sev == "blocker":
            confidence -= 0.25
        elif sev == "major":
            confidence -= 0.14
        else:
            confidence -= 0.05
    confidence = float(max(0.0, min(1.0, confidence)))

    overall_pass = (not has_blocker) and (confidence >= confidence_threshold)
    status = "pass" if overall_pass else "fail"

    return {
        "overall_pass": bool(overall_pass),
        "confidence": confidence,
        "violations": violations,
        "status": status,
        "judge_type": "heuristic_rigorous",
        "notes": (
            f"frames_considered={len(frame_paths)}; "
            f"avg_edge_density={quality['avg_edge_density']:.4f}; avg_motion={quality['avg_motion']:.3f}"
        ),
    }


def run_visual_judge(
    bundle_dir: Path,
    summary: dict,
    frame_paths: list[str],
    *,
    confidence_threshold: float = 0.6,
    mode: str = "heuristic",
    require_video: bool = False,
    provider: str = "codex",
    model: str | None = None,
    endpoint: str | None = None,
    api_key_env: str = "NAVIRL_VLM_API_KEY",
    native_cmd: str | None = None,
    allow_fallback: bool = True,
) -> dict:
    """Run visual E2E judge and return strict JSON-compatible payload."""

    heuristic_payload = _heuristic_judge(
        summary,
        frame_paths=frame_paths,
        confidence_threshold=confidence_threshold,
        require_video=require_video,
    )
    if mode != "vlm":
        return heuristic_payload

    provider_cfg = ProviderConfig(
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key_env=api_key_env,
        native_cmd=native_cmd,
    )
    payload = run_aegis_review(
        bundle_dir=bundle_dir,
        summary=summary,
        frame_paths=frame_paths,
        heuristic_payload=heuristic_payload,
        provider_config=provider_cfg,
        confidence_threshold=confidence_threshold,
        require_video=require_video,
        allow_fallback=allow_fallback,
    )
    payload.setdefault("overseer_name", AEGIS_NAME)
    return payload


def write_judge_output(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
