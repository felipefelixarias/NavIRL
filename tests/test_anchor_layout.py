from __future__ import annotations

from pathlib import Path

from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario
from navirl.verify.validators import run_numeric_invariants


def test_overlapping_anchors_are_sanitized(tmp_path: Path):
    scenario = load_scenario("navirl/scenarios/library/hallway_pass.yaml")
    scenario["horizon"]["steps"] = 20
    scenario["humans"]["count"] = 2
    scenario["humans"]["starts"] = [(-1.2, 0.0), (-1.2, 0.0)]
    scenario["humans"]["goals"] = [(1.2, 0.0), (1.2, 0.0)]
    scenario["robot"]["start"] = (-1.2, 0.0)
    scenario["robot"]["goal"] = (1.2, 0.0)

    log = run_scenario_dict(
        scenario=scenario,
        out_root=tmp_path,
        run_id="anchor-layout",
        render_override=False,
        video_override=False,
    )

    checks = {c["name"]: c for c in run_numeric_invariants(Path(log.bundle_dir))["checks"]}
    assert checks["anchor_layout"]["pass"], checks["anchor_layout"]
