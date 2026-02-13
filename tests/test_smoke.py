from __future__ import annotations

import json
from pathlib import Path

from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario
from navirl.verify.validators import run_numeric_invariants


def test_navirl_smoke_run(tmp_path: Path):
    scenario = load_scenario("navirl/scenarios/library/hallway_pass.yaml")
    scenario["horizon"]["steps"] = 24

    log = run_scenario_dict(
        scenario=scenario,
        out_root=tmp_path,
        run_id="smoke",
        render_override=False,
        video_override=False,
    )

    bundle = Path(log.bundle_dir)
    assert (bundle / "scenario.yaml").exists()
    assert (bundle / "state.jsonl").exists()
    assert (bundle / "events.jsonl").exists()
    assert (bundle / "summary.json").exists()

    with (bundle / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["scenario_id"] == "hallway_pass"
    assert "metrics" in summary
    assert "success_rate" in summary["metrics"]
    assert "map" in summary
    assert float(summary["map"]["pixels_per_meter"]) > 0.0

    invariants = run_numeric_invariants(bundle)
    checks = {c["name"]: c for c in invariants["checks"]}
    assert checks["units_metadata"]["pass"], checks["units_metadata"]
    assert checks["anchor_layout"]["pass"], checks["anchor_layout"]
    assert checks["no_wall_penetration"]["pass"], checks["no_wall_penetration"]
    if "agent_stop_duration" in checks:
        assert checks["agent_stop_duration"]["pass"], checks["agent_stop_duration"]
    assert "robot_progress" in checks
    assert checks["robot_progress"]["progress_fraction"] >= 0.0, checks["robot_progress"]
