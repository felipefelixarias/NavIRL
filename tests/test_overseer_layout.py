from __future__ import annotations

from navirl.overseer import apply_layout_to_scenario, suggest_layout
from navirl.scenarios.load import load_scenario


def test_suggest_layout_generates_valid_layout():
    scenario = load_scenario("navirl/scenarios/library/hallway_pass.yaml")
    suggestion = suggest_layout(scenario, objective="auto", humans_count=6, seed=13)

    assert suggestion["objective"] in {
        "cross_flow",
        "bottleneck_showcase",
        "comfort_showcase",
        "comfort",
    }
    assert suggestion["humans_count"] == 6
    assert len(suggestion["human_starts"]) == 6
    assert len(suggestion["human_goals"]) == 6
    assert suggestion["robot_start"] != suggestion["robot_goal"]
    assert float(suggestion["quality"]["min_clearance_used_m"]) > 0.0


def test_apply_layout_to_scenario_updates_core_fields():
    base = {
        "id": "test_layout",
        "scene": {"id": "test_layout"},
        "humans": {"count": 0, "starts": [], "goals": []},
        "robot": {"start": [0.0, 0.0], "goal": [0.0, 0.0]},
    }
    suggestion = {
        "objective": "cross_flow",
        "humans_count": 2,
        "human_starts": [[-1.0, 0.0], [1.0, 0.0]],
        "human_goals": [[1.0, 0.0], [-1.0, 0.0]],
        "robot_start": [-2.0, 0.0],
        "robot_goal": [2.0, 0.0],
        "quality": {"min_clearance_used_m": 0.25, "target_min_clearance_m": 0.23},
    }

    patched = apply_layout_to_scenario(base, suggestion)

    assert patched["humans"]["count"] == 2
    assert patched["humans"]["starts"] == suggestion["human_starts"]
    assert patched["humans"]["goals"] == suggestion["human_goals"]
    assert patched["robot"]["start"] == suggestion["robot_start"]
    assert patched["robot"]["goal"] == suggestion["robot_goal"]
    assert patched["_meta"]["aegis_layout"]["objective"] == "cross_flow"
