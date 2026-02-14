from __future__ import annotations

import json
from pathlib import Path

import yaml

from navirl.overseer import ProviderConfig, run_aegis_review


def _minimal_summary() -> dict:
    return {
        "scenario_id": "unit_overseer",
        "metrics": {
            "deadlock_count": 1,
            "collisions_agent_obstacle": 1,
            "min_dist_robot_human_min": 0.12,
        },
        "map": {
            "width_m": 5.0,
            "height_m": 4.0,
            "pixels_per_meter": 100.0,
            "meters_per_pixel": 0.01,
        },
        "invariants": {
            "checks": [
                {"name": "motion_jitter", "worst_flip_rate": 0.95, "max_flip_rate": 0.8, "pass": False},
                {
                    "name": "wall_proximity_fraction",
                    "near_wall_fraction": 0.17,
                    "max_fraction": 0.18,
                    "pass": True,
                },
                {
                    "name": "agent_stop_duration",
                    "max_stop_seconds": 6.0,
                    "top_longest_stops": [{"agent_id": "h0", "max_stopped_seconds": 5.6}],
                    "pass": True,
                },
            ],
            "overall_pass": False,
        },
    }


def _heuristic_payload() -> dict:
    return {
        "overall_pass": False,
        "confidence": 0.45,
        "violations": [
            {
                "type": "missing_render_diagnostics",
                "evidence": "render_diagnostics.json missing",
                "severity": "blocker",
            }
        ],
        "status": "fail",
        "judge_type": "heuristic_rigorous",
    }


def test_aegis_review_fallback_mode(tmp_path: Path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    scenario = {
        "robot": {"radius": 1.4},
        "humans": {"radius": 0.18},
    }
    (bundle_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
    (bundle_dir / "state.jsonl").write_text(json.dumps({"agents": []}) + "\n", encoding="utf-8")

    payload = run_aegis_review(
        bundle_dir=bundle_dir,
        summary=_minimal_summary(),
        frame_paths=[],
        heuristic_payload=_heuristic_payload(),
        provider_config=ProviderConfig(provider="codex"),
        confidence_threshold=0.6,
        require_video=False,
        allow_fallback=True,
    )

    assert payload["judge_type"] == "aegis_vlm_fallback_heuristic"
    assert payload["provider_trace"]["provider_used"] is False
    assert payload["provider_trace"]["fallback_used"] is True
    assert payload["findings"]
    assert any(v["type"] == "human_visible_deadlock" for v in payload["violations"])


def test_aegis_review_without_fallback_needs_human_review(tmp_path: Path):
    bundle_dir = tmp_path / "bundle_no_fallback"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "scenario.yaml").write_text(
        yaml.safe_dump({"robot": {"radius": 0.2}, "humans": {"radius": 0.18}}, sort_keys=False),
        encoding="utf-8",
    )

    payload = run_aegis_review(
        bundle_dir=bundle_dir,
        summary=_minimal_summary(),
        frame_paths=[],
        heuristic_payload=_heuristic_payload(),
        provider_config=ProviderConfig(provider="codex"),
        confidence_threshold=0.6,
        require_video=False,
        allow_fallback=False,
    )

    assert payload["status"] == "needs_human_review"
    assert payload["judge_type"] == "aegis_vlm_unavailable"
    assert payload["provider_trace"]["provider_used"] is False
