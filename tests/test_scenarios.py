from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from navirl.backends.grid2d.maps import load_map_info
from navirl.scenarios.load import load_scenario
from navirl.scenarios.validate import validate_scenario_dict


LIB = Path("navirl/scenarios/library")


def test_scenario_library_all_validate():
    files = sorted(LIB.glob("*.yaml"))
    assert files, "Scenario library is empty."

    ids = set()
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        validate_scenario_dict(raw)

        scenario = load_scenario(path)
        assert scenario["id"] not in ids
        ids.add(scenario["id"])


def test_verify_canonical_scenarios_present():
    expected = {
        "hallway_pass.yaml",
        "doorway_token_yield.yaml",
        "kitchen_congestion.yaml",
        "group_cohesion.yaml",
        "robot_comfort_avoidance.yaml",
        "routine_cook_dinner_micro.yaml",
    }
    existing = {p.name for p in LIB.glob("*.yaml")}
    missing = sorted(expected - existing)
    assert not missing, f"Missing canonical scenarios: {missing}"


def test_path_map_requires_explicit_scale():
    bad = {
        "id": "path_map_missing_scale",
        "seed": 1,
        "scene": {
            "backend": "grid2d",
            "map": {
                "source": "path",
                "path": "data/maps/Wainscott_0.png",
            },
        },
        "horizon": {"steps": 10, "dt": 0.1},
        "humans": {"controller": {"type": "orca"}, "count": 0},
        "robot": {"controller": {"type": "baseline_astar"}, "start": [0.0, 0.0], "goal": [0.1, 0.1]},
    }

    with pytest.raises(ValueError):
        validate_scenario_dict(bad)


def test_path_map_downsample_updates_effective_scale():
    scene_base = {
        "map": {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "pixels_per_meter": 100.0,
        }
    }
    info_full = load_map_info(scene_base, base_dir=Path("."))

    scene_ds = {
        "map": {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "pixels_per_meter": 100.0,
            "downsample": 2.0,
        }
    }
    info_ds = load_map_info(scene_ds, base_dir=Path("."))

    assert info_full.pixels_per_meter == pytest.approx(100.0)
    assert info_ds.pixels_per_meter == pytest.approx(50.0)
    assert info_full.width_m == pytest.approx(info_ds.width_m, rel=0.02)
    assert info_full.height_m == pytest.approx(info_ds.height_m, rel=0.02)
