"""Tests for navirl.experiments.runner — _apply_overrides and run_batch_template.

Focus on the _apply_overrides utility and the batch runner's integration logic.
"""

from __future__ import annotations

from navirl.experiments.runner import _apply_overrides

# ---------------------------------------------------------------------------
# _apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    """Unit tests for dotted-path override application."""

    def test_single_top_level_key(self):
        scenario = {"seed": 42, "horizon": {"dt": 0.1}}
        result = _apply_overrides(scenario, {"seed": 99})
        assert result["seed"] == 99
        assert result["horizon"]["dt"] == 0.1

    def test_nested_key(self):
        scenario = {"scene": {"orca": {"neighbor_dist": 2.0}}}
        result = _apply_overrides(scenario, {"scene.orca.neighbor_dist": 4.0})
        assert result["scene"]["orca"]["neighbor_dist"] == 4.0

    def test_deep_nested_key(self):
        scenario = {"a": {"b": {"c": {"d": 1}}}}
        result = _apply_overrides(scenario, {"a.b.c.d": 99})
        assert result["a"]["b"]["c"]["d"] == 99

    def test_creates_intermediate_dicts(self):
        scenario = {}
        result = _apply_overrides(scenario, {"scene.map.id": "kitchen"})
        assert result["scene"]["map"]["id"] == "kitchen"

    def test_multiple_overrides(self):
        scenario = {
            "horizon": {"dt": 0.1, "max_steps": 100},
            "scene": {"map": {"id": "hallway"}},
        }
        result = _apply_overrides(
            scenario,
            {
                "horizon.dt": 0.05,
                "horizon.max_steps": 200,
                "scene.map.id": "doorway",
            },
        )
        assert result["horizon"]["dt"] == 0.05
        assert result["horizon"]["max_steps"] == 200
        assert result["scene"]["map"]["id"] == "doorway"

    def test_does_not_mutate_original(self):
        scenario = {"horizon": {"dt": 0.1}}
        original_dt = scenario["horizon"]["dt"]
        result = _apply_overrides(scenario, {"horizon.dt": 0.05})
        # Original should be unchanged
        assert scenario["horizon"]["dt"] == original_dt
        assert result["horizon"]["dt"] == 0.05

    def test_empty_overrides(self):
        scenario = {"a": 1, "b": {"c": 2}}
        result = _apply_overrides(scenario, {})
        assert result == scenario

    def test_override_with_different_types(self):
        scenario = {"x": 1}
        result = _apply_overrides(scenario, {"x": "string_value"})
        assert result["x"] == "string_value"

    def test_override_with_list_value(self):
        scenario = {"seeds": [1, 2, 3]}
        result = _apply_overrides(scenario, {"seeds": [10, 20]})
        assert result["seeds"] == [10, 20]

    def test_override_with_none(self):
        scenario = {"key": "value"}
        result = _apply_overrides(scenario, {"key": None})
        assert result["key"] is None

    def test_override_adds_new_key(self):
        scenario = {"existing": 1}
        result = _apply_overrides(scenario, {"new_key": "hello"})
        assert result["new_key"] == "hello"
        assert result["existing"] == 1

    def test_nested_override_preserves_siblings(self):
        scenario = {"scene": {"orca": {"neighbor_dist": 2.0, "max_neighbors": 10}}}
        result = _apply_overrides(scenario, {"scene.orca.neighbor_dist": 5.0})
        assert result["scene"]["orca"]["neighbor_dist"] == 5.0
        assert result["scene"]["orca"]["max_neighbors"] == 10
