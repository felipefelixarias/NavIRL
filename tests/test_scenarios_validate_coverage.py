"""Cover the missing branches in navirl/scenarios/validate.py.

``test_scenarios.py`` exercises the happy path (library scenarios must
validate) and one ValueError path.  The uncovered lines are early-return
guards for non-dict sub-objects, the mpp-only / ppm-mpp-inconsistent map
branches, ``load_schema`` itself, and the ``scenario must be an object``
top-level rejection.
"""

from __future__ import annotations

import pytest

from navirl.scenarios.validate import load_schema, validate_scenario_dict

# ---------------------------------------------------------------------------
# Helper: a minimal scenario that passes validation, which individual tests
# mutate to exercise specific error paths.
# ---------------------------------------------------------------------------


def _base_scenario() -> dict:
    return {
        "id": "test",
        "seed": 0,
        "scene": {
            "backend": "grid2d",
            "map": {"source": "builtin", "id": "empty_10x10"},
        },
        "horizon": {"steps": 10, "dt": 0.1},
        "humans": {"controller": {"type": "orca"}, "count": 0},
        "robot": {
            "controller": {"type": "baseline_astar"},
            "start": [0.0, 0.0],
            "goal": [1.0, 1.0],
        },
    }


def _assert_has_error(scn: dict, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        validate_scenario_dict(scn)


# ---------------------------------------------------------------------------
# load_schema (lines 15-16)
# ---------------------------------------------------------------------------


class TestLoadSchema:
    def test_returns_dict(self):
        schema = load_schema()
        assert isinstance(schema, dict)

    def test_schema_file_is_valid_json_schema(self):
        schema = load_schema()
        # The schema is a JSON Schema doc; confirm a couple of well-known keys.
        assert "$schema" in schema or "type" in schema or "properties" in schema


# ---------------------------------------------------------------------------
# Top-level: scenario must be an object (line 180)
# ---------------------------------------------------------------------------


class TestTopLevelNotDict:
    def test_list_raises(self):
        with pytest.raises(ValueError, match="scenario must be an object"):
            validate_scenario_dict([])  # type: ignore[arg-type]

    def test_string_raises(self):
        with pytest.raises(ValueError, match="scenario must be an object"):
            validate_scenario_dict("not-a-dict")  # type: ignore[arg-type]

    def test_none_raises(self):
        with pytest.raises(ValueError, match="scenario must be an object"):
            validate_scenario_dict(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Early-return guards for non-dict sub-objects (lines 37, 69, 94, 154)
# ---------------------------------------------------------------------------


class TestNonDictSubObjects:
    """When a sub-object is the wrong type, validation records the error
    and short-circuits further checks in that branch rather than
    attempting attribute access.
    """

    def test_scene_not_dict(self):
        scn = _base_scenario()
        scn["scene"] = "not a scene"
        _assert_has_error(scn, "scene must be an object")

    def test_horizon_not_dict(self):
        scn = _base_scenario()
        scn["horizon"] = [10, 0.1]
        _assert_has_error(scn, "horizon must be an object")

    def test_humans_not_dict(self):
        scn = _base_scenario()
        scn["humans"] = "lots"
        _assert_has_error(scn, "humans must be an object")

    def test_robot_not_dict(self):
        scn = _base_scenario()
        scn["robot"] = 42
        _assert_has_error(scn, "robot must be an object")


# ---------------------------------------------------------------------------
# Map: mpp-only branch + inconsistent ppm/mpp (lines 114, 120-121)
# ---------------------------------------------------------------------------


class TestMapScaleBranches:
    def test_mpp_only_valid(self):
        """meters_per_pixel alone, with no ppm, validates without error."""
        scn = _base_scenario()
        scn["scene"]["map"] = {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "meters_per_pixel": 0.02,
        }
        # No ValueError expected.
        validate_scenario_dict(scn)

    def test_mpp_must_be_positive(self):
        scn = _base_scenario()
        scn["scene"]["map"] = {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "meters_per_pixel": 0.0,
        }
        _assert_has_error(scn, "meters_per_pixel must be > 0")

    def test_mpp_non_numeric(self):
        scn = _base_scenario()
        scn["scene"]["map"] = {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "meters_per_pixel": "no",
        }
        _assert_has_error(scn, "meters_per_pixel must be > 0")

    def test_ppm_and_mpp_inconsistent(self):
        """Providing both ppm and mpp such that mpp != 1/ppm triggers the
        consistency check (lines 120-121).
        """
        scn = _base_scenario()
        scn["scene"]["map"] = {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "pixels_per_meter": 100.0,
            "meters_per_pixel": 0.5,  # 1/100 = 0.01, vastly different
        }
        _assert_has_error(scn, "pixels_per_meter and scene.map.meters_per_pixel are inconsistent")

    def test_ppm_and_mpp_consistent_passes(self):
        scn = _base_scenario()
        scn["scene"]["map"] = {
            "source": "path",
            "path": "data/maps/Wainscott_0.png",
            "pixels_per_meter": 100.0,
            "meters_per_pixel": 0.01,
        }
        # Within the 2% tolerance, so this should pass.
        validate_scenario_dict(scn)


# ---------------------------------------------------------------------------
# Exercise a couple of extra branches to round out confidence — these are
# already covered but a regression here would be especially painful.
# ---------------------------------------------------------------------------


class TestOtherFailurePaths:
    def test_missing_id(self):
        scn = _base_scenario()
        del scn["id"]
        _assert_has_error(scn, "id is required")

    def test_missing_seed(self):
        scn = _base_scenario()
        del scn["seed"]
        _assert_has_error(scn, "seed is required")

    def test_humans_controller_unknown_type(self):
        scn = _base_scenario()
        scn["humans"]["controller"]["type"] = "no_such_controller"
        _assert_has_error(scn, "humans.controller.type must be one of")

    def test_robot_controller_unknown_type(self):
        scn = _base_scenario()
        scn["robot"]["controller"]["type"] = "no_such_planner"
        _assert_has_error(scn, "robot.controller.type must be one of")

    def test_humans_starts_not_point(self):
        scn = _base_scenario()
        scn["humans"]["starts"] = [[0.0, 0.0], "not-a-point"]
        _assert_has_error(scn, r"humans.starts\[1\] must be \[x, y\]")

    def test_robot_start_not_point(self):
        scn = _base_scenario()
        scn["robot"]["start"] = "origin"
        _assert_has_error(scn, r"robot.start must be \[x, y\]")

    def test_horizon_steps_must_be_positive(self):
        scn = _base_scenario()
        scn["horizon"]["steps"] = 0
        _assert_has_error(scn, "horizon.steps must be >= 1")

    def test_horizon_dt_must_be_positive(self):
        scn = _base_scenario()
        scn["horizon"]["dt"] = -0.1
        _assert_has_error(scn, "horizon.dt must be > 0")
