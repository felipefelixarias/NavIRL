from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from navirl.overseer.provider import (
    ProviderCallError,
    ProviderConfig,
    ProviderUnavailableError,
    run_structured_vlm,
)

AEGIS_NAME = "Aegis Overseer"

AEGIS_REVIEW_SCHEMA = {
    "type": "object",
    "required": ["overall_pass", "confidence", "status", "violations"],
    "properties": {
        "overall_pass": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "status": {"type": "string", "enum": ["pass", "fail", "needs_human_review"]},
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
            },
        },
        "notes": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
}


def _compact_metrics(summary: dict) -> dict:
    m = dict(summary.get("metrics", {}))
    keep = [
        "collisions_agent_agent",
        "collisions_agent_obstacle",
        "deadlock_count",
        "intrusion_rate",
        "jerk_proxy",
        "oscillation_score",
        "success_rate",
        "time_to_goal_robot",
        "min_dist_robot_human_min",
        "min_dist_human_human_min",
    ]
    return {k: m[k] for k in keep if k in m}


def _load_bundle_scenario(bundle_dir: Path) -> dict:
    scenario_path = bundle_dir / "scenario.yaml"
    if not scenario_path.exists():
        return {}
    try:
        with scenario_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return raw if isinstance(raw, dict) else {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        # Handle file access errors, YAML parsing errors, and encoding issues
        return {}


def _load_state_rows(state_path: Path) -> list[dict]:
    rows: list[dict] = []
    with state_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _state_speed_ratio_stats(bundle_dir: Path) -> dict:
    state_path = bundle_dir / "state.jsonl"
    if not state_path.exists():
        return {"max_ratio": 0.0, "mean_ratio": 0.0, "num_samples": 0}

    rows = _load_state_rows(state_path)
    ratios: list[float] = []
    for row in rows:
        for agent in row.get("agents", []):
            vx = float(agent.get("vx", 0.0))
            vy = float(agent.get("vy", 0.0))
            speed = math.hypot(vx, vy)
            max_speed = max(1e-6, float(agent.get("max_speed", 1.0)))
            ratios.append(speed / max_speed)

    if not ratios:
        return {"max_ratio": 0.0, "mean_ratio": 0.0, "num_samples": 0}

    return {
        "max_ratio": float(max(ratios)),
        "mean_ratio": float(sum(ratios) / len(ratios)),
        "num_samples": int(len(ratios)),
    }


def _checks_by_name(summary: dict) -> dict:
    checks = summary.get("invariants", {}).get("checks", [])
    out = {}
    for c in checks:
        if isinstance(c, dict) and "name" in c:
            out[str(c["name"])] = c
    return out


def build_aegis_findings(bundle_dir: Path, summary: dict) -> list[dict]:
    findings: list[dict] = []
    metrics = dict(summary.get("metrics", {}))
    checks = _checks_by_name(summary)
    scenario = _load_bundle_scenario(bundle_dir)

    deadlock_count = int(metrics.get("deadlock_count", 0))
    if deadlock_count > 0:
        findings.append(
            {
                "type": "human_visible_deadlock",
                "severity": "blocker",
                "evidence": f"deadlock_count={deadlock_count}",
                "recommendation": "Increase retry budget, improve start/goal spacing, "
                "or tune ORCA horizons.",
            }
        )

    obstacle_collisions = int(metrics.get("collisions_agent_obstacle", 0))
    if obstacle_collisions > 0:
        findings.append(
            {
                "type": "human_visible_wall_hits",
                "severity": "blocker",
                "evidence": f"collisions_agent_obstacle={obstacle_collisions}",
                "recommendation": "Increase wall clearance buffers and validate "
                "map scale/radius assumptions.",
            }
        )

    motion_jitter = checks.get("motion_jitter", {})
    if isinstance(motion_jitter, dict):
        worst = float(motion_jitter.get("worst_flip_rate", 0.0))
        max_flip = float(motion_jitter.get("max_flip_rate", 1.0))
        if worst > max_flip:
            sev = "blocker"
        elif worst > max_flip * 0.9:
            sev = "major"
        else:
            sev = ""
        if sev:
            findings.append(
                {
                    "type": "unnatural_jitter",
                    "severity": sev,
                    "evidence": f"worst_flip_rate={worst:.3f}, limit={max_flip:.3f}",
                    "recommendation": "Increase velocity smoothing and reduce aggressive "
                    "neighbor response.",
                }
            )

    stop_check = checks.get("agent_stop_duration", {})
    if isinstance(stop_check, dict):
        lim = float(stop_check.get("max_stop_seconds", 0.0))
        longest = 0.0
        for item in stop_check.get("top_longest_stops", []):
            if isinstance(item, dict):
                longest = max(longest, float(item.get("max_stopped_seconds", 0.0)))
        if lim > 0.0 and longest > lim * 0.85:
            findings.append(
                {
                    "type": "near_wall_or_goal_stall",
                    "severity": "major",
                    "evidence": f"longest_stop={longest:.2f}s, limit={lim:.2f}s",
                    "recommendation": "Inspect bottlenecks and resample starts/goals to "
                    "reduce prolonged stalls.",
                }
            )

    wall_prox = checks.get("wall_proximity_fraction", {})
    if isinstance(wall_prox, dict):
        frac = float(wall_prox.get("near_wall_fraction", 0.0))
        lim = float(wall_prox.get("max_fraction", 1.0))
        if lim > 0.0 and frac > lim * 0.9:
            findings.append(
                {
                    "type": "excess_wall_hugging",
                    "severity": "major",
                    "evidence": f"near_wall_fraction={frac:.3f}, limit={lim:.3f}",
                    "recommendation": "Increase wall clearance and traversability offset "
                    "for narrow corridors.",
                }
            )

    speed_stats = _state_speed_ratio_stats(bundle_dir)
    max_ratio = float(speed_stats.get("max_ratio", 0.0))
    if max_ratio > 1.3:
        findings.append(
            {
                "type": "extreme_speed_spike",
                "severity": "blocker",
                "evidence": f"max_speed_ratio={max_ratio:.3f}",
                "recommendation": "Reduce acceleration response and verify dt / controller gains.",
            }
        )
    elif max_ratio > 1.12:
        findings.append(
            {
                "type": "speed_near_unrealistic",
                "severity": "major",
                "evidence": f"max_speed_ratio={max_ratio:.3f}",
                "recommendation": "Tune smoothing and horizon params to reduce bursts.",
            }
        )

    min_robot_human = float(metrics.get("min_dist_robot_human_min", 1.0))
    if min_robot_human < 0.18:
        findings.append(
            {
                "type": "unsafe_robot_human_clearance",
                "severity": "major",
                "evidence": f"min_dist_robot_human_min={min_robot_human:.3f}m",
                "recommendation": "Increase comfort radius and adjust robot slowdown/lookahead.",
            }
        )

    map_meta = dict(summary.get("map", {}))
    map_w = float(map_meta.get("width_m", 0.0) or 0.0)
    map_h = float(map_meta.get("height_m", 0.0) or 0.0)
    min_dim = max(1e-6, min(map_w if map_w > 0 else 1.0, map_h if map_h > 0 else 1.0))
    robot_radius = float(scenario.get("robot", {}).get("radius", 0.0) or 0.0)
    human_radius = float(scenario.get("humans", {}).get("radius", 0.0) or 0.0)

    if robot_radius > 0.22 * min_dim:
        findings.append(
            {
                "type": "robot_scale_implausible",
                "severity": "blocker",
                "evidence": f"robot_radius={robot_radius:.3f}m on min_map_dim={min_dim:.3f}m",
                "recommendation": "Fix map units (pixels_per_meter/meters_per_pixel) or "
                "robot radius.",
            }
        )

    if robot_radius > 0 and human_radius > 0:
        ratio = robot_radius / human_radius
        if ratio > 3.5 or ratio < 0.45:
            findings.append(
                {
                    "type": "robot_human_scale_mismatch",
                    "severity": "major",
                    "evidence": f"robot_to_human_radius_ratio={ratio:.3f}",
                    "recommendation": "Normalize human/robot radii to realistic scale proportions.",
                }
            )

    return findings


def _findings_to_violations(findings: list[dict]) -> list[dict]:
    out = []
    for f in findings:
        out.append(
            {
                "type": str(f.get("type", "aegis_finding")),
                "evidence": str(f.get("evidence", "")),
                "severity": str(f.get("severity", "minor")),
            }
        )
    return out


def _dedupe_violations(violations: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for v in violations:
        key = (str(v.get("type", "")), str(v.get("evidence", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def _confidence_penalty(violations: list[dict]) -> float:
    penalty = 0.0
    for v in violations:
        sev = str(v.get("severity", "minor"))
        if sev == "blocker":
            penalty += 0.18
        elif sev == "major":
            penalty += 0.08
        else:
            penalty += 0.03
    return penalty


def _review_prompt(
    *,
    summary: dict,
    frame_paths: list[str],
    heuristic_payload: dict,
    findings: list[dict],
    confidence_threshold: float,
    require_video: bool,
) -> str:
    prompt_payload = {
        "agent_name": AEGIS_NAME,
        "task": "qualitative_simulation_review",
        "confidence_threshold": confidence_threshold,
        "require_video": require_video,
        "scenario_id": summary.get("scenario_id", ""),
        "map": summary.get("map", {}),
        "metrics": _compact_metrics(summary),
        "heuristic_payload": {
            "overall_pass": heuristic_payload.get("overall_pass", False),
            "confidence": heuristic_payload.get("confidence", 0.0),
            "violations": heuristic_payload.get("violations", []),
        },
        "aegis_findings": findings,
        "frame_paths": frame_paths,
        "instructions": [
            "Assess realism like a human simulation overseer.",
            "Focus on visible bugs: wall sticking, jitter, scale mismatch, "
            "unrealistic speed, unnatural trajectories.",
            "Return strict JSON only.",
        ],
        "response_schema": AEGIS_REVIEW_SCHEMA,
    }
    return json.dumps(prompt_payload, indent=2, sort_keys=True)


def _normalize_provider_review(payload: dict) -> dict:
    out = dict(payload)
    out["overall_pass"] = bool(out.get("overall_pass", False))
    out["confidence"] = float(out.get("confidence", 0.0))
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    out["status"] = str(out.get("status", "fail"))
    if out["status"] not in {"pass", "fail", "needs_human_review"}:
        out["status"] = "fail"
    if not isinstance(out.get("violations"), list):
        out["violations"] = []
    return out


def run_aegis_review(
    *,
    bundle_dir: Path,
    summary: dict,
    frame_paths: list[str],
    heuristic_payload: dict,
    provider_config: ProviderConfig,
    confidence_threshold: float,
    require_video: bool,
    allow_fallback: bool,
) -> dict:
    findings = build_aegis_findings(bundle_dir, summary)
    finding_violations = _findings_to_violations(findings)

    provider_payload: dict | None = None
    provider_error = ""
    provider_used = False
    fallback_used = False

    try:
        prompt = _review_prompt(
            summary=summary,
            frame_paths=frame_paths,
            heuristic_payload=heuristic_payload,
            findings=findings,
            confidence_threshold=confidence_threshold,
            require_video=require_video,
        )
        provider_payload = _normalize_provider_review(
            run_structured_vlm(
                prompt=prompt,
                image_paths=frame_paths,
                schema=AEGIS_REVIEW_SCHEMA,
                config=provider_config,
            )
        )
        provider_used = True
    except (ProviderUnavailableError, ProviderCallError, ValueError) as exc:
        provider_error = str(exc)
        if not allow_fallback:
            return {
                "overall_pass": False,
                "confidence": 0.0,
                "violations": finding_violations,
                "status": "needs_human_review",
                "judge_type": "aegis_vlm_unavailable",
                "overseer_name": AEGIS_NAME,
                "findings": findings,
                "provider_trace": {
                    "provider": provider_config.normalized_provider(),
                    "provider_used": False,
                    "fallback_used": False,
                    "error": provider_error,
                },
                "notes": "VLM provider unavailable and fallback disabled.",
            }
        fallback_used = True

    violations = []
    violations.extend(list(heuristic_payload.get("violations", [])))
    violations.extend(finding_violations)
    if provider_payload is not None:
        violations.extend(list(provider_payload.get("violations", [])))
    violations = _dedupe_violations(violations)

    has_blocker = any(str(v.get("severity", "")) == "blocker" for v in violations)
    base_conf = float(heuristic_payload.get("confidence", 0.0))
    if provider_payload is not None:
        combined_conf = 0.55 * float(provider_payload.get("confidence", 0.0)) + 0.45 * base_conf
        combined_conf -= _confidence_penalty(finding_violations) * 0.35
    else:
        combined_conf = base_conf

    combined_conf = max(0.0, min(1.0, float(combined_conf)))

    overall_pass = (not has_blocker) and combined_conf >= confidence_threshold
    status = "pass" if overall_pass else "fail"

    recommendations: list[str] = []
    for f in findings:
        rec = str(f.get("recommendation", "")).strip()
        if rec and rec not in recommendations:
            recommendations.append(rec)

    judge_type = "aegis_vlm_hybrid" if provider_used else "aegis_vlm_fallback_heuristic"
    return {
        "overall_pass": bool(overall_pass),
        "confidence": float(combined_conf),
        "violations": violations,
        "status": status,
        "judge_type": judge_type,
        "overseer_name": AEGIS_NAME,
        "findings": findings,
        "recommendations": recommendations[:10],
        "provider_trace": {
            "provider": provider_config.normalized_provider(),
            "provider_used": provider_used,
            "fallback_used": fallback_used,
            "error": provider_error,
            "model": provider_config.model,
            "endpoint": provider_config.endpoint,
        },
        "notes": f"aegis_findings={len(findings)}; frames_considered={len(frame_paths)}",
    }
