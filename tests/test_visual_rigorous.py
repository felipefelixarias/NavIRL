from __future__ import annotations

from pathlib import Path

import pytest

rvo2 = pytest.importorskip("rvo2", reason="rvo2 not installed")

from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario
from navirl.verify.judge import run_visual_judge
from navirl.verify.validators import build_visual_summary, run_numeric_invariants, sample_key_frames


def test_render_emits_diagnostics_with_arrows_and_trails(tmp_path: Path):
    scenario = load_scenario("navirl/scenarios/library/hallway_pass.yaml")
    scenario["horizon"]["steps"] = 28
    scenario.setdefault("render", {})["enabled"] = True
    scenario["render"]["video"] = False

    log = run_scenario_dict(
        scenario=scenario,
        out_root=tmp_path,
        run_id="rigorous-render",
        render_override=True,
        video_override=False,
    )

    diag_path = Path(log.bundle_dir) / "frames" / "render_diagnostics.json"
    assert diag_path.exists()

    import json

    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    assert str(diag["style_version"]).startswith("v3_")
    assert diag["total_arrows_drawn"] > 0
    assert diag["total_trail_segments"] > 0
    assert diag["avg_trail_segments_per_frame"] >= 1.0
    assert float(diag["pixels_per_meter"]) > 0.0


def test_rigorous_judge_fails_when_diagnostics_missing():
    summary = {
        "bundle_dir": "dummy",
        "invariants": {"checks": [], "overall_pass": True},
        "metrics": {
            "collisions_agent_obstacle": 0,
            "collisions_agent_agent": 0,
            "intrusion_rate": 0.1,
        },
        "frame_count": 10,
        "has_video": True,
        "render_diagnostics": {},
    }
    payload = run_visual_judge(
        Path("dummy"),
        summary,
        frame_paths=[],
        confidence_threshold=0.6,
        mode="heuristic",
        require_video=True,
    )

    assert payload["status"] == "fail"
    assert any(v["type"] == "missing_render_diagnostics" for v in payload["violations"])


def test_rigorous_judge_passes_on_real_bundle(tmp_path: Path):
    scenario = load_scenario("navirl/scenarios/library/doorway_token_yield.yaml")
    scenario["horizon"]["steps"] = 110
    scenario.setdefault("render", {})["enabled"] = True
    scenario["render"]["video"] = False

    log = run_scenario_dict(
        scenario=scenario,
        out_root=tmp_path,
        run_id="rigorous-judge",
        render_override=True,
        video_override=False,
    )

    bundle_dir = Path(log.bundle_dir)
    invariants = run_numeric_invariants(bundle_dir)
    summary = build_visual_summary(bundle_dir, invariants)
    frames = sample_key_frames(bundle_dir, num_frames=8)

    payload = run_visual_judge(
        bundle_dir,
        summary,
        frame_paths=frames,
        confidence_threshold=0.6,
        mode="heuristic",
        require_video=False,
    )

    assert payload["judge_type"] == "heuristic_rigorous"
    assert payload["overall_pass"]
