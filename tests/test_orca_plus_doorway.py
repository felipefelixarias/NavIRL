from __future__ import annotations

from pathlib import Path

import pytest

rvo2 = pytest.importorskip("rvo2", reason="rvo2 not installed")

from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario
from navirl.verify.validators import validate_token_exclusivity


def test_orca_plus_doorway_token_and_yield_events(tmp_path: Path):
    scenario = load_scenario("navirl/scenarios/library/doorway_token_yield.yaml")
    scenario["horizon"]["steps"] = 90

    log = run_scenario_dict(
        scenario=scenario,
        out_root=tmp_path,
        run_id="doorway",
        render_override=False,
        video_override=False,
    )

    bundle = Path(log.bundle_dir)
    events_path = bundle / "events.jsonl"

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert events

    text = "\n".join(events)
    assert "door_token_acquire" in text
    assert "doorway_yield" in text

    token_check = validate_token_exclusivity(events_path)
    assert token_check["pass"], token_check
