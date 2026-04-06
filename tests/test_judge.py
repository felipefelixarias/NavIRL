"""Tests for navirl.verify.judge — heuristic judge and visual judge (heuristic mode).

Covers: _heuristic_judge thresholds, violation detection, confidence calculation,
run_visual_judge in heuristic mode, write_judge_output, JUDGE_OUTPUT_SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from navirl.verify.judge import (
    JUDGE_OUTPUT_SCHEMA,
    _heuristic_judge,
    run_visual_judge,
    write_judge_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_summary(**overrides):
    """Build a minimal summary dict that passes all heuristic checks."""
    summary = {
        "invariants": {"checks": []},
        "metrics": {
            "horizon_steps": 100,
            "collisions_agent_agent": 0,
            "collisions_agent_obstacle": 0,
            "intrusion_rate": 0.0,
            "deadlock_count": 0,
        },
        "map": {"pixels_per_meter": 40.0, "meters_per_pixel": 0.025},
        "frame_count": 50,
        "has_video": True,
        "expected_high_interaction": False,
        "bundle_dir": "/tmp/test",
        "render_diagnostics": {
            "total_agents_drawn": 100,
            "total_arrows_drawn": 80,
            "avg_agents_per_frame": 4.0,
            "avg_trail_segments_per_frame": 10.0,
            "total_text_elements": 20,
            "style_version": "v3_default",
        },
    }
    summary.update(overrides)
    return summary


# ---------------------------------------------------------------------------
# _heuristic_judge — pass cases
# ---------------------------------------------------------------------------


class TestHeuristicJudgePass:
    @patch(
        "navirl.verify.judge._frame_quality",
        return_value={"avg_edge_density": 0.05, "avg_motion": 2.0, "num_frames": 12},
    )
    def test_clean_summary_passes(self, mock_fq):
        result = _heuristic_judge(
            _base_summary(),
            frame_paths=[f"/tmp/f{i}.png" for i in range(30)],
            confidence_threshold=0.6,
            require_video=False,
        )
        assert result["status"] == "pass"
        assert result["overall_pass"] is True
        assert result["confidence"] > 0.6
        assert result["judge_type"] == "heuristic_rigorous"

    def test_output_schema_keys(self):
        result = _heuristic_judge(
            _base_summary(),
            frame_paths=[],
            confidence_threshold=0.6,
            require_video=False,
        )
        for key in JUDGE_OUTPUT_SCHEMA["required"]:
            assert key in result


# ---------------------------------------------------------------------------
# _heuristic_judge — invariant check failures
# ---------------------------------------------------------------------------


class TestHeuristicJudgeInvariants:
    def test_failed_check_is_blocker(self):
        summary = _base_summary()
        summary["invariants"]["checks"] = [{"name": "no_teleport", "pass": False}]
        result = _heuristic_judge(summary, [], 0.6, False)
        assert result["status"] == "fail"
        blockers = [v for v in result["violations"] if v["severity"] == "blocker"]
        assert any("no_teleport" in v["type"] for v in blockers)


# ---------------------------------------------------------------------------
# _heuristic_judge — frame / video checks
# ---------------------------------------------------------------------------


class TestHeuristicJudgeFrameVideo:
    def test_insufficient_frames(self):
        summary = _base_summary(frame_count=5)
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "insufficient_frames" for v in result["violations"])

    def test_missing_video_when_required(self):
        summary = _base_summary(has_video=False)
        result = _heuristic_judge(summary, [], 0.6, require_video=True)
        assert any(v["type"] == "missing_video" for v in result["violations"])

    def test_missing_video_ok_when_not_required(self):
        summary = _base_summary(has_video=False)
        result = _heuristic_judge(summary, [], 0.6, require_video=False)
        assert not any(v["type"] == "missing_video" for v in result["violations"])


# ---------------------------------------------------------------------------
# _heuristic_judge — map units
# ---------------------------------------------------------------------------


class TestHeuristicJudgeMapUnits:
    def test_missing_map_units(self):
        summary = _base_summary()
        summary["map"] = {}
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "missing_map_units" for v in result["violations"])

    def test_invalid_map_units(self):
        summary = _base_summary()
        summary["map"] = {"pixels_per_meter": 0.0, "meters_per_pixel": 0.0}
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "invalid_map_units" for v in result["violations"])


# ---------------------------------------------------------------------------
# _heuristic_judge — collision metrics
# ---------------------------------------------------------------------------


class TestHeuristicJudgeCollisions:
    def test_obstacle_collisions_blocker(self):
        summary = _base_summary()
        summary["metrics"]["collisions_agent_obstacle"] = 3
        result = _heuristic_judge(summary, [], 0.6, False)
        blockers = [v for v in result["violations"] if v["severity"] == "blocker"]
        assert any(v["type"] == "obstacle_collisions" for v in blockers)

    def test_deadlock_blocker(self):
        summary = _base_summary()
        summary["metrics"]["deadlock_count"] = 1
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "deadlock_detected" for v in result["violations"])

    def test_agent_collision_rate_blocker(self):
        summary = _base_summary()
        summary["metrics"]["collisions_agent_agent"] = 200
        summary["metrics"]["horizon_steps"] = 100
        result = _heuristic_judge(summary, [], 0.6, False)
        blockers = [v for v in result["violations"] if v["severity"] == "blocker"]
        assert any(v["type"] == "excess_agent_collisions" for v in blockers)

    def test_agent_collision_rate_major(self):
        summary = _base_summary()
        summary["metrics"]["collisions_agent_agent"] = 120
        summary["metrics"]["horizon_steps"] = 100
        result = _heuristic_judge(summary, [], 0.6, False)
        majors = [v for v in result["violations"] if v["severity"] == "major"]
        assert any(v["type"] == "excess_agent_collisions" for v in majors)

    def test_agent_collision_rate_minor(self):
        summary = _base_summary()
        summary["metrics"]["collisions_agent_agent"] = 50
        summary["metrics"]["horizon_steps"] = 100
        result = _heuristic_judge(summary, [], 0.6, False)
        minors = [v for v in result["violations"] if v["severity"] == "minor"]
        assert any(v["type"] == "excess_agent_collisions" for v in minors)

    def test_high_interaction_lenient_thresholds(self):
        summary = _base_summary(expected_high_interaction=True)
        # Rate = 1.2, above standard blocker (1.3) threshold but below high-interaction (1.8)
        summary["metrics"]["collisions_agent_agent"] = 120
        summary["metrics"]["horizon_steps"] = 100
        result = _heuristic_judge(summary, [], 0.6, False)
        blockers = [
            v
            for v in result["violations"]
            if v["severity"] == "blocker" and v["type"] == "excess_agent_collisions"
        ]
        assert len(blockers) == 0  # Not a blocker in high interaction mode


# ---------------------------------------------------------------------------
# _heuristic_judge — intrusion rate
# ---------------------------------------------------------------------------


class TestHeuristicJudgeIntrusion:
    def test_blocker_intrusion(self):
        summary = _base_summary()
        summary["metrics"]["intrusion_rate"] = 0.95
        result = _heuristic_judge(summary, [], 0.6, False)
        blockers = [v for v in result["violations"] if v["severity"] == "blocker"]
        assert any(v["type"] == "high_intrusion_rate" for v in blockers)

    def test_major_intrusion(self):
        summary = _base_summary()
        summary["metrics"]["intrusion_rate"] = 0.8
        result = _heuristic_judge(summary, [], 0.6, False)
        majors = [v for v in result["violations"] if v["severity"] == "major"]
        assert any(v["type"] == "high_intrusion_rate" for v in majors)

    def test_minor_intrusion(self):
        summary = _base_summary()
        summary["metrics"]["intrusion_rate"] = 0.6
        result = _heuristic_judge(summary, [], 0.6, False)
        minors = [v for v in result["violations"] if v["severity"] == "minor"]
        assert any(v["type"] == "high_intrusion_rate" for v in minors)

    def test_high_interaction_intrusion_thresholds(self):
        summary = _base_summary(expected_high_interaction=True)
        summary["metrics"]["intrusion_rate"] = 0.85
        result = _heuristic_judge(summary, [], 0.6, False)
        # 0.85 < 0.92 major threshold for high interaction
        intrusion_violations = [v for v in result["violations"] if v["type"] == "high_intrusion_rate"]
        assert not any(v["severity"] in ("blocker", "major") for v in intrusion_violations)


# ---------------------------------------------------------------------------
# _heuristic_judge — render diagnostics
# ---------------------------------------------------------------------------


class TestHeuristicJudgeRenderDiagnostics:
    def test_missing_render_diagnostics(self):
        summary = _base_summary()
        summary["render_diagnostics"] = {}
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "missing_render_diagnostics" for v in result["violations"])

    def test_unexpected_style_version(self):
        summary = _base_summary()
        summary["render_diagnostics"]["style_version"] = "v1_old"
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "unexpected_render_style" for v in result["violations"])

    def test_insufficient_arrows_blocker(self):
        summary = _base_summary()
        summary["render_diagnostics"]["total_arrows_drawn"] = 10
        summary["render_diagnostics"]["total_agents_drawn"] = 100
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "insufficient_direction_arrows" for v in result["violations"])

    def test_insufficient_trails(self):
        summary = _base_summary()
        summary["render_diagnostics"]["avg_trail_segments_per_frame"] = 0.5
        summary["render_diagnostics"]["avg_agents_per_frame"] = 4.0
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "insufficient_trail_overlay" for v in result["violations"])

    def test_text_clutter(self):
        summary = _base_summary(frame_count=50)
        summary["render_diagnostics"]["total_text_elements"] = 200
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "overlay_text_clutter" for v in result["violations"])


# ---------------------------------------------------------------------------
# _heuristic_judge — robot progress
# ---------------------------------------------------------------------------


class TestHeuristicJudgeRobotProgress:
    def test_insufficient_robot_progress(self):
        summary = _base_summary()
        summary["invariants"]["checks"] = [
            {
                "name": "robot_progress",
                "pass": False,
                "progress_fraction": 0.05,
                "effective_min_progress": 0.1,
            }
        ]
        result = _heuristic_judge(summary, [], 0.6, False)
        majors = [v for v in result["violations"] if v["severity"] == "major"]
        assert any(v["type"] == "insufficient_robot_progress" for v in majors)


# ---------------------------------------------------------------------------
# _heuristic_judge — confidence calculation
# ---------------------------------------------------------------------------


class TestHeuristicJudgeConfidence:
    def test_confidence_decreases_with_violations(self):
        summary_clean = _base_summary()
        result_clean = _heuristic_judge(summary_clean, [], 0.6, False)

        summary_dirty = _base_summary()
        summary_dirty["metrics"]["collisions_agent_obstacle"] = 5
        summary_dirty["metrics"]["deadlock_count"] = 1
        result_dirty = _heuristic_judge(summary_dirty, [], 0.6, False)

        assert result_clean["confidence"] > result_dirty["confidence"]

    def test_confidence_clamps_to_zero(self):
        summary = _base_summary(frame_count=5, has_video=False)
        summary["map"] = {}
        summary["metrics"]["collisions_agent_obstacle"] = 10
        summary["metrics"]["deadlock_count"] = 5
        summary["render_diagnostics"] = {}
        summary["invariants"]["checks"] = [
            {"name": "a", "pass": False},
            {"name": "b", "pass": False},
            {"name": "c", "pass": False},
        ]
        result = _heuristic_judge(summary, [], 0.6, require_video=True)
        assert result["confidence"] >= 0.0


# ---------------------------------------------------------------------------
# _heuristic_judge — near-limit checks (stop duration, wall proximity)
# ---------------------------------------------------------------------------


class TestHeuristicJudgeNearLimit:
    def test_near_limit_stop_duration(self):
        summary = _base_summary()
        summary["invariants"]["checks"] = [
            {
                "name": "agent_stop_duration",
                "pass": True,
                "max_stop_seconds": 8.0,
                "top_longest_stops": [
                    {"max_stopped_seconds": 7.5}  # 7.5 > 8.0 * 0.9 = 7.2
                ],
            }
        ]
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "near_limit_agent_stop_duration" for v in result["violations"])

    def test_near_limit_wall_proximity(self):
        summary = _base_summary()
        summary["invariants"]["checks"] = [
            {
                "name": "wall_proximity_fraction",
                "pass": True,
                "near_wall_fraction": 0.13,
                "max_fraction": 0.14,  # 0.13 > 0.14 * 0.9 = 0.126
            }
        ]
        result = _heuristic_judge(summary, [], 0.6, False)
        assert any(v["type"] == "near_limit_wall_proximity" for v in result["violations"])


# ---------------------------------------------------------------------------
# run_visual_judge — heuristic mode
# ---------------------------------------------------------------------------


class TestRunVisualJudge:
    def test_heuristic_mode(self, tmp_path):
        result = run_visual_judge(
            tmp_path,
            _base_summary(),
            frame_paths=[],
            mode="heuristic",
        )
        assert result["judge_type"] == "heuristic_rigorous"

    def test_default_mode_is_heuristic(self, tmp_path):
        result = run_visual_judge(
            tmp_path,
            _base_summary(),
            frame_paths=[],
        )
        assert result["judge_type"] == "heuristic_rigorous"


# ---------------------------------------------------------------------------
# write_judge_output
# ---------------------------------------------------------------------------


class TestWriteJudgeOutput:
    def test_writes_json(self, tmp_path):
        payload = {"overall_pass": True, "confidence": 0.95}
        out = tmp_path / "judge.json"
        write_judge_output(out, payload)
        loaded = json.loads(out.read_text())
        assert loaded["overall_pass"] is True
        assert loaded["confidence"] == 0.95
