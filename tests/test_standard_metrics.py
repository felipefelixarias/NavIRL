"""Tests for navirl.metrics.standard — StandardMetrics compute pipeline.

Covers the full compute() method end-to-end using synthetic state logs and
scenario configs, plus unit tests for helper functions.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from navirl.backends.grid2d.constants import FREE_SPACE, OBSTACLE_SPACE
from navirl.metrics.standard import (
    StandardMetrics,
    _load_state_rows,
    _pair_dist,
    _world_to_rc,
    compute_metrics_from_bundle,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_agent(
    aid: int,
    kind: str,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    radius: float = 0.3,
    goal_x: float = 0.0,
    goal_y: float = 0.0,
    behavior: str = "",
) -> dict:
    return {
        "id": aid,
        "kind": kind,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "radius": radius,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "behavior": behavior,
    }


def _write_state_log(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _default_scenario(map_id: str = "hallway") -> dict:
    return {
        "horizon": {"dt": 0.1},
        "scene": {"map": {"id": map_id}},
    }


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestWorldToRC:
    def test_origin(self):
        # A 200x300 map at 100 ppm: origin maps to center
        r, c = _world_to_rc(0.0, 0.0, (200, 300), 100.0)
        assert r == 100
        assert c == 150

    def test_positive_offset(self):
        r, c = _world_to_rc(1.0, 0.5, (200, 300), 100.0)
        assert r == 150  # 0.5 * 100 + 100
        assert c == 250  # 1.0 * 100 + 150

    def test_negative_position(self):
        r, c = _world_to_rc(-0.5, -0.5, (200, 300), 100.0)
        assert r == 50  # -0.5 * 100 + 100
        assert c == 100  # -0.5 * 100 + 150


class TestPairDist:
    def test_same_point(self):
        a = {"x": 1.0, "y": 2.0}
        b = {"x": 1.0, "y": 2.0}
        assert _pair_dist(a, b) == 0.0

    def test_unit_distance(self):
        a = {"x": 0.0, "y": 0.0}
        b = {"x": 3.0, "y": 4.0}
        assert _pair_dist(a, b) == pytest.approx(5.0)

    def test_negative_coords(self):
        a = {"x": -1.0, "y": -1.0}
        b = {"x": 2.0, "y": 3.0}
        assert _pair_dist(a, b) == pytest.approx(5.0)


class TestLoadStateRows:
    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "state.jsonl"
        rows = [{"step": 0, "agents": []}, {"step": 1, "agents": []}]
        _write_state_log(p, rows)
        loaded = _load_state_rows(p)
        assert len(loaded) == 2
        assert loaded[0]["step"] == 0

    def test_ignores_blank_lines(self, tmp_path):
        p = tmp_path / "state.jsonl"
        with p.open("w") as f:
            f.write(json.dumps({"step": 0, "agents": []}) + "\n")
            f.write("\n")
            f.write("  \n")
            f.write(json.dumps({"step": 1, "agents": []}) + "\n")
        loaded = _load_state_rows(p)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# StandardMetrics.compute() integration tests
# ---------------------------------------------------------------------------


class TestStandardMetricsCompute:
    """End-to-end tests for the main metrics pipeline."""

    def _run_metrics(self, rows: list[dict], scenario: dict | None = None, tmp_path=None) -> dict:
        if tmp_path is None:
            tmp_path = Path(tempfile.mkdtemp())
        state_path = tmp_path / "state.jsonl"
        _write_state_log(state_path, rows)
        return StandardMetrics().compute(state_path, scenario or _default_scenario())

    def test_empty_log_raises(self, tmp_path):
        state_path = tmp_path / "state.jsonl"
        state_path.write_text("")
        with pytest.raises(ValueError, match="No rows"):
            StandardMetrics().compute(state_path, _default_scenario())

    def test_single_robot_goal_reached(self, tmp_path):
        """Robot starts at (0,0) and goal is at (0.01, 0.01) — should count as reached."""
        rows = [
            {
                "step": 0,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, goal_x=0.01, goal_y=0.01),
                ],
            },
        ]
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["success_rate"] == 1.0
        assert report["time_to_goal_robot"] == pytest.approx(0.0)  # step 0 * dt
        assert report["collisions_agent_agent"] == 0
        assert report["collisions_agent_obstacle"] == 0
        assert report["horizon_steps"] == 1

    def test_single_robot_no_goal(self, tmp_path):
        """Robot far from goal — success should be 0."""
        rows = [
            {
                "step": 0,
                "agents": [_make_agent(0, "robot", 0.0, 0.0, goal_x=5.0, goal_y=5.0)],
            },
            {
                "step": 1,
                "agents": [_make_agent(0, "robot", 0.1, 0.0, vx=1.0, goal_x=5.0, goal_y=5.0)],
            },
        ]
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["success_rate"] == 0.0
        assert report["time_to_goal_robot"] == float("inf")
        assert report["path_length_robot"] > 0

    def test_robot_human_collision(self, tmp_path):
        """Robot and human overlap — collision should be counted."""
        rows = [
            {
                "step": 0,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, radius=0.3, goal_x=1.0, goal_y=0.0),
                    _make_agent(1, "human", 0.1, 0.0, radius=0.3, goal_x=-1.0, goal_y=0.0),
                ],
            },
        ]
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["collisions_agent_agent"] >= 1
        # Min dist should be small
        assert report["min_dist_robot_human_min"] < 0.5

    def test_no_collision_far_apart(self, tmp_path):
        """Robot and human far apart — no collision."""
        rows = []
        for step in range(5):
            rows.append({
                "step": step,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, vx=0.1, goal_x=2.0, goal_y=0.0),
                    _make_agent(1, "human", 0.0, 3.0, goal_x=0.0, goal_y=-1.0),
                ],
            })
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["collisions_agent_agent"] == 0
        assert report["min_dist_robot_human_min"] > 2.0

    def test_intrusion_rate(self, tmp_path):
        """Human within intrusion distance — intrusion_rate > 0."""
        rows = [
            {
                "step": 0,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, goal_x=1.0, goal_y=0.0),
                    _make_agent(1, "human", 0.3, 0.0, goal_x=-1.0, goal_y=0.0),
                ],
            },
            {
                "step": 1,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, goal_x=1.0, goal_y=0.0),
                    _make_agent(1, "human", 0.3, 0.0, goal_x=-1.0, goal_y=0.0),
                ],
            },
        ]
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["intrusion_rate"] > 0.0

    def test_path_length_computed(self, tmp_path):
        """Robot moves along x axis — path length should match displacement."""
        rows = []
        for step in range(10):
            x = 0.01 * step
            rows.append({
                "step": step,
                "agents": [
                    _make_agent(0, "robot", x, 0.0, vx=0.1, goal_x=1.0, goal_y=0.0),
                ],
            })
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["path_length_robot"] == pytest.approx(0.09, abs=0.01)

    def test_oscillation_detection(self, tmp_path):
        """Agent changing heading back-and-forth should have higher oscillation."""
        rows = []
        for step in range(20):
            # Alternating velocity direction each step — need distinct angles
            # to trigger heading sign flips. Use perpendicular directions.
            if step % 2 == 0:
                vx, vy = 1.0, 0.1
            else:
                vx, vy = 1.0, -0.1
            rows.append({
                "step": step,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, vx=vx, vy=vy, goal_x=5.0, goal_y=5.0),
                ],
            })
        report = self._run_metrics(rows, tmp_path=tmp_path)
        # High oscillation expected
        assert report["oscillation_score"] > 0.0

    def test_jerk_proxy_computed(self, tmp_path):
        """Robot with changing velocity should have nonzero jerk."""
        rows = []
        for step in range(10):
            # Accelerating then decelerating
            vx = 0.1 * step if step < 5 else 0.1 * (10 - step)
            rows.append({
                "step": step,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, vx=vx, vy=0.0, goal_x=5.0, goal_y=0.0),
                ],
            })
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["jerk_proxy"] > 0.0

    def test_multiple_humans_min_dist(self, tmp_path):
        """Multiple humans — should track minimum distance across all pairs."""
        rows = [
            {
                "step": 0,
                "agents": [
                    _make_agent(0, "robot", 0.0, 0.0, goal_x=1.0, goal_y=0.0),
                    _make_agent(1, "human", 1.0, 0.0, goal_x=-1.0, goal_y=0.0),
                    _make_agent(2, "human", 0.5, 0.0, goal_x=-1.0, goal_y=0.0),
                ],
            },
        ]
        report = self._run_metrics(rows, tmp_path=tmp_path)
        # Closest human is at distance 0.5
        assert report["min_dist_robot_human_min"] == pytest.approx(0.5, abs=0.01)
        # Human-human min dist is 0.5
        assert report["min_dist_human_human_min"] == pytest.approx(0.5, abs=0.01)

    def test_deadlock_detection(self, tmp_path):
        """Agent nearly stationary far from goal should be detected as deadlocked."""
        rows = []
        # 100 steps of near-zero speed, far from goal
        for step in range(100):
            rows.append({
                "step": step,
                "agents": [
                    _make_agent(
                        0, "robot", 0.0, 0.0,
                        vx=0.001, vy=0.0,
                        goal_x=5.0, goal_y=5.0,
                    ),
                ],
            })
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["deadlock_count"] >= 1

    def test_report_keys_complete(self, tmp_path):
        """Verify all expected keys are present in the report."""
        rows = [
            {
                "step": 0,
                "agents": [_make_agent(0, "robot", 0.0, 0.0, goal_x=0.01, goal_y=0.01)],
            },
        ]
        report = self._run_metrics(rows, tmp_path=tmp_path)
        expected_keys = {
            "collisions_agent_agent",
            "collisions_agent_obstacle",
            "min_dist_robot_human_min",
            "min_dist_robot_human_mean",
            "min_dist_robot_human_p05",
            "min_dist_human_human_min",
            "min_dist_human_human_mean",
            "min_dist_human_human_p05",
            "intrusion_rate",
            "deadlock_count",
            "oscillation_score",
            "jerk_proxy",
            "path_length_robot",
            "time_to_goal_robot",
            "success_rate",
            "horizon_steps",
            "dt",
            "map_pixels_per_meter",
            "map_meters_per_pixel",
            "map_width_m",
            "map_height_m",
        }
        assert expected_keys.issubset(set(report.keys()))

    def test_custom_evaluation_config(self, tmp_path):
        """Scenario with custom evaluation parameters."""
        scenario = {
            "horizon": {"dt": 0.05},
            "scene": {"map": {"id": "hallway"}},
            "evaluation": {
                "intrusion_delta": 0.5,
                "deadlock_seconds": 2.0,
                "deadlock_speed_thresh": 0.05,
            },
        }
        rows = [
            {
                "step": 0,
                "agents": [_make_agent(0, "robot", 0.0, 0.0, goal_x=0.01, goal_y=0.01)],
            },
        ]
        state_path = tmp_path / "state.jsonl"
        _write_state_log(state_path, rows)
        report = StandardMetrics().compute(state_path, scenario)
        assert report["dt"] == pytest.approx(0.05)

    def test_done_behavior_resets_deadlock(self, tmp_path):
        """Agent with DONE behavior should not count as deadlocked."""
        rows = []
        for step in range(100):
            rows.append({
                "step": step,
                "agents": [
                    _make_agent(
                        0, "robot", 0.0, 0.0,
                        vx=0.0, vy=0.0,
                        goal_x=5.0, goal_y=5.0,
                        behavior="DONE",
                    ),
                ],
            })
        report = self._run_metrics(rows, tmp_path=tmp_path)
        assert report["deadlock_count"] == 0

    def test_different_builtin_maps(self, tmp_path):
        """Should work with different builtin maps."""
        for map_id in ["hallway", "doorway", "kitchen"]:
            scenario = {
                "horizon": {"dt": 0.1},
                "scene": {"map": {"id": map_id}},
            }
            rows = [
                {
                    "step": 0,
                    "agents": [_make_agent(0, "robot", 0.0, 0.0, goal_x=0.01, goal_y=0.01)],
                },
            ]
            state_path = tmp_path / f"state_{map_id}.jsonl"
            _write_state_log(state_path, rows)
            report = StandardMetrics().compute(state_path, scenario)
            assert "success_rate" in report


# ---------------------------------------------------------------------------
# compute_metrics_from_bundle
# ---------------------------------------------------------------------------


class TestComputeMetricsFromBundle:
    def test_missing_scenario_file(self, tmp_path):
        state_path = tmp_path / "state.jsonl"
        state_path.write_text('{"step":0,"agents":[]}\n')
        with pytest.raises(FileNotFoundError, match="Scenario file not found"):
            compute_metrics_from_bundle(state_path)

    def test_full_bundle(self, tmp_path):
        import yaml

        scenario = _default_scenario()
        scenario_path = tmp_path / "scenario.yaml"
        with scenario_path.open("w") as f:
            yaml.dump(scenario, f)

        state_path = tmp_path / "state.jsonl"
        rows = [
            {
                "step": 0,
                "agents": [_make_agent(0, "robot", 0.0, 0.0, goal_x=0.01, goal_y=0.01)],
            },
        ]
        _write_state_log(state_path, rows)

        report = compute_metrics_from_bundle(state_path)
        assert report["success_rate"] == 1.0
