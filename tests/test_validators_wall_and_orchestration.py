"""Tests for validators.py — wall penetration, clearance, proximity,
run_numeric_invariants, build_visual_summary, check_video_artifact, and _load_scenario.

Targets the ~48% of validators.py that was previously uncovered.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from navirl.verify.validators import (
    _load_scenario,
    build_visual_summary,
    check_video_artifact,
    run_numeric_invariants,
    validate_no_wall_penetration,
    validate_wall_clearance_buffer,
    validate_wall_proximity,
)

# ---------------------------------------------------------------------------
# Helpers — create minimal scenario bundles on disk
# ---------------------------------------------------------------------------


def _make_map_png(path: Path, shape: tuple[int, int] = (40, 40)) -> None:
    """Create a simple binary map — white interior with 2-px black border."""
    img = np.ones(shape, dtype=np.uint8) * 255
    img[0:2, :] = 0
    img[-2:, :] = 0
    img[:, 0:2] = 0
    img[:, -2:] = 0
    cv2.imwrite(str(path), img)


def _minimal_scenario(
    *,
    robot_start=(0.5, 0.5),
    robot_goal=(0.8, 0.5),
    human_starts=None,
    human_goals=None,
    human_count=0,
    horizon_steps=10,
    dt=0.1,
    evaluation_overrides=None,
    scenario_id="test_scenario",
) -> dict:
    scene = {
        "map": {
            "image": "map.png",
            "pixels_per_meter": 40.0,
        },
        "orca": {
            "neighbor_dist": 5.0,
            "max_neighbors": 10,
            "time_horizon": 3.0,
            "time_horizon_obst": 3.0,
        },
    }
    scenario = {
        "id": scenario_id,
        "seed": 1,
        "scene": scene,
        "robot": {
            "start": list(robot_start),
            "goal": list(robot_goal),
            "radius": 0.05,
            "controller": {"type": "baseline_astar"},
        },
        "humans": {
            "count": human_count,
            "radius": 0.04,
            "starts": human_starts or [],
            "goals": human_goals or [],
            "controller": {"type": "orca"},
        },
        "horizon": {"steps": horizon_steps, "dt": dt},
        "evaluation": evaluation_overrides or {},
    }
    return scenario


def _write_bundle(
    tmp_path: Path,
    scenario: dict,
    state_rows: list[dict] | None = None,
    events: list[dict] | None = None,
    write_frames: bool = False,
    frame_count: int = 0,
    write_summary: bool = False,
    summary_data: dict | None = None,
) -> Path:
    """Write a minimal scenario bundle to tmp_path and return the bundle dir."""
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    _make_map_png(bundle / "map.png")
    scenario.setdefault("_meta", {})["source_path"] = str(bundle / "scenario.yaml")
    # Make map path relative to bundle
    scenario["scene"]["map"]["image"] = "map.png"

    (bundle / "scenario.yaml").write_text(
        yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
    )

    if state_rows is not None:
        with (bundle / "state.jsonl").open("w", encoding="utf-8") as f:
            for row in state_rows:
                f.write(json.dumps(row) + "\n")

    if events is not None:
        with (bundle / "events.jsonl").open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    if write_frames:
        frames = bundle / "frames"
        frames.mkdir(exist_ok=True)
        for i in range(frame_count):
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(frames / f"frame_{i:04d}.png"), img)

    if write_summary:
        data = summary_data or {}
        (bundle / "summary.json").write_text(json.dumps(data), encoding="utf-8")

    return bundle


def _agent_row(
    agent_id: int,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    kind: str = "robot",
    radius: float = 0.05,
    goal_x: float = 0.8,
    goal_y: float = 0.5,
    behavior: str = "NAVIGATE",
    max_speed: float = 1.0,
) -> dict:
    return {
        "id": agent_id,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "kind": kind,
        "radius": radius,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "behavior": behavior,
        "max_speed": max_speed,
    }


def _state_row(step: int, agents: list[dict]) -> dict:
    return {"step": step, "agents": agents}


# ---------------------------------------------------------------------------
# _load_scenario
# ---------------------------------------------------------------------------


class TestLoadScenario:
    def test_loads_valid_scenario(self, tmp_path):
        scenario = _minimal_scenario()
        bundle = _write_bundle(tmp_path, scenario)
        loaded = _load_scenario(bundle)
        assert loaded["id"] == "test_scenario"
        assert "robot" in loaded

    def test_missing_scenario_file_raises(self, tmp_path):
        bundle = tmp_path / "empty_bundle"
        bundle.mkdir()
        with pytest.raises(ValueError, match="Scenario file not found"):
            _load_scenario(bundle)

    def test_oversized_scenario_file_raises(self, tmp_path):
        bundle = tmp_path / "big_bundle"
        bundle.mkdir()
        # Create a file larger than 50MB would be impractical, so patch MAX_FILE_SIZE
        scenario_path = bundle / "scenario.yaml"
        scenario_path.write_text("id: test\n", encoding="utf-8")
        import navirl.verify.validators as v

        orig = v.MAX_FILE_SIZE
        try:
            v.MAX_FILE_SIZE = 1  # 1 byte limit
            with pytest.raises(ValueError, match="too large"):
                _load_scenario(bundle)
        finally:
            v.MAX_FILE_SIZE = orig


# ---------------------------------------------------------------------------
# validate_no_wall_penetration
# ---------------------------------------------------------------------------


class TestValidateNoWallPenetration:
    def test_agents_in_free_space_pass(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5)]),
            _state_row(1, [_agent_row(0, 0.55, 0.5)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_no_wall_penetration(bundle / "state.jsonl", bundle)
        assert result["pass"] is True
        assert result["num_violations"] == 0

    def test_agent_inside_obstacle_fails(self, tmp_path):
        """Place agent at (-2,-2) which maps to an obstacle pixel in the expanded map."""
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, -2.0, -2.0)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_no_wall_penetration(bundle / "state.jsonl", bundle)
        assert result["pass"] is False
        assert result["num_violations"] > 0

    def test_agent_out_of_bounds_detected(self, tmp_path):
        """Place agent far outside the map bounds (0,-5 maps below the grid)."""
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.0, -5.0)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_no_wall_penetration(bundle / "state.jsonl", bundle)
        assert result["pass"] is False
        assert result["num_out_of_bounds"] > 0
        assert any(v["reason"] == "out_of_bounds" for v in result["violations"])

    def test_radius_intersects_obstacle(self, tmp_path):
        """Place agent near the border obstacle with a large radius that overlaps."""
        scenario = _minimal_scenario()
        # (-1.5, -2.5) is near the obstacle border; large radius (1.5m = 60px)
        # will exceed the available clearance.
        rows = [
            _state_row(0, [_agent_row(0, -1.5, -2.5, radius=1.5)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_no_wall_penetration(bundle / "state.jsonl", bundle)
        assert result["num_violations"] > 0

    def test_multiple_agents_tracked(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(
                0,
                [
                    _agent_row(0, 0.5, 0.5, kind="robot"),
                    _agent_row(1, 0.5, 0.6, kind="human"),
                ],
            ),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_no_wall_penetration(bundle / "state.jsonl", bundle)
        assert result["pass"] is True
        assert result["pixels_per_meter"] == 40.0


# ---------------------------------------------------------------------------
# validate_wall_clearance_buffer
# ---------------------------------------------------------------------------


class TestValidateWallClearanceBuffer:
    def test_agents_with_clearance_pass(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, radius=0.02)]),
            _state_row(1, [_agent_row(0, 0.5, 0.5, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_clearance_buffer(
            bundle / "state.jsonl",
            bundle,
            clearance_buffer_m=0.01,
            max_fraction=1.0,
        )
        assert result["pass"] is True
        assert result["name"] == "wall_clearance_buffer"

    def test_agent_near_wall_violates_buffer(self, tmp_path):
        scenario = _minimal_scenario()
        # Agent at (-1.5, -2.5) is near border obstacles; use large buffer
        rows = [
            _state_row(0, [_agent_row(0, -1.5, -2.5, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_clearance_buffer(
            bundle / "state.jsonl",
            bundle,
            clearance_buffer_m=2.0,
            max_fraction=0.0,
        )
        assert result["pass"] is False
        assert result["num_violations"] > 0
        assert result["violation_fraction"] > 0

    def test_out_of_bounds_skipped(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.0, -5.0)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_clearance_buffer(
            bundle / "state.jsonl",
            bundle,
            clearance_buffer_m=0.01,
            max_fraction=1.0,
        )
        # Out of bounds agents are skipped, so samples=0
        assert result["samples"] == 0
        assert result["pass"] is True

    def test_max_fraction_threshold(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.06, 0.06, radius=0.02)]),
            _state_row(1, [_agent_row(0, 0.5, 0.5, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        # Allow up to 100% violations
        result = validate_wall_clearance_buffer(
            bundle / "state.jsonl",
            bundle,
            clearance_buffer_m=0.2,
            max_fraction=1.0,
        )
        assert result["pass"] is True


# ---------------------------------------------------------------------------
# validate_wall_proximity
# ---------------------------------------------------------------------------


class TestValidateWallProximity:
    def test_agents_far_from_wall_pass(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, radius=0.02)]),
            _state_row(1, [_agent_row(0, 0.5, 0.5, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_proximity(
            bundle / "state.jsonl",
            bundle,
            near_wall_buffer_m=0.01,
            max_fraction=1.0,
        )
        assert result["pass"] is True
        assert result["name"] == "wall_proximity_fraction"
        assert result["samples"] == 2

    def test_agent_near_wall_tracked(self, tmp_path):
        scenario = _minimal_scenario()
        # Agent near the border obstacles, large proximity buffer to trigger detection
        rows = [
            _state_row(0, [_agent_row(0, -1.5, -2.5, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_proximity(
            bundle / "state.jsonl",
            bundle,
            near_wall_buffer_m=2.0,
            max_fraction=0.0,
        )
        assert result["pass"] is False
        assert result["near_samples"] > 0
        assert len(result["top_agents"]) > 0

    def test_per_agent_stats(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(
                0,
                [
                    _agent_row(0, 0.5, 0.5, radius=0.02, kind="robot"),
                    _agent_row(1, 0.06, 0.06, radius=0.02, kind="human"),
                ],
            ),
            _state_row(
                1,
                [
                    _agent_row(0, 0.5, 0.5, radius=0.02, kind="robot"),
                    _agent_row(1, 0.06, 0.06, radius=0.02, kind="human"),
                ],
            ),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_proximity(
            bundle / "state.jsonl",
            bundle,
            near_wall_buffer_m=0.3,
            max_fraction=1.0,
        )
        assert result["samples"] == 4
        assert len(result["top_agents"]) == 2
        # Agent 1 (near wall) should have higher near_fraction
        agent_map = {a["agent_id"]: a for a in result["top_agents"]}
        assert agent_map[1]["near_fraction"] >= agent_map[0]["near_fraction"]

    def test_out_of_bounds_skipped(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.0, -5.0)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows)
        result = validate_wall_proximity(
            bundle / "state.jsonl",
            bundle,
            near_wall_buffer_m=0.01,
            max_fraction=1.0,
        )
        assert result["samples"] == 0
        assert result["pass"] is True


# ---------------------------------------------------------------------------
# run_numeric_invariants
# ---------------------------------------------------------------------------


class TestRunNumericInvariants:
    def _make_good_bundle(self, tmp_path):
        """Bundle where a robot moves from start toward goal, all clean."""
        scenario = _minimal_scenario(
            robot_start=(0.5, 0.5),
            robot_goal=(0.8, 0.5),
            horizon_steps=5,
            dt=0.1,
        )
        rows = []
        for step in range(5):
            x = 0.5 + step * 0.06
            rows.append(
                _state_row(
                    step,
                    [
                        _agent_row(
                            0,
                            x,
                            0.5,
                            vx=0.6,
                            vy=0.0,
                            kind="robot",
                            radius=0.02,
                            goal_x=0.8,
                            goal_y=0.5,
                        ),
                    ],
                )
            )
        return _write_bundle(tmp_path, scenario, state_rows=rows, events=[])

    def test_all_pass_clean_scenario(self, tmp_path):
        bundle = self._make_good_bundle(tmp_path)
        result = run_numeric_invariants(bundle)
        assert "overall_pass" in result
        assert "checks" in result
        assert len(result["checks"]) > 0
        check_names = [c["name"] for c in result["checks"]]
        assert "units_metadata" in check_names
        assert "no_teleport" in check_names
        assert "speed_accel_bounds" in check_names
        assert "robot_progress" in check_names

    def test_doorway_includes_token_check(self, tmp_path):
        scenario = _minimal_scenario(scenario_id="doorway_test")
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
            _state_row(1, [_agent_row(0, 0.53, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows, events=[])
        result = run_numeric_invariants(bundle)
        check_names = [c["name"] for c in result["checks"]]
        assert "token_exclusivity" in check_names

    def test_frames_dir_adds_sync_check(self, tmp_path):
        scenario = _minimal_scenario()
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
            _state_row(1, [_agent_row(0, 0.53, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
        ]
        bundle = _write_bundle(
            tmp_path,
            scenario,
            state_rows=rows,
            events=[],
            write_frames=True,
            frame_count=2,
        )
        result = run_numeric_invariants(bundle)
        check_names = [c["name"] for c in result["checks"]]
        assert "log_render_sync" in check_names

    def test_expected_wall_penetration_skips_wall_checks(self, tmp_path):
        scenario = _minimal_scenario(
            evaluation_overrides={"expected_wall_penetration": True},
        )
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows, events=[])
        result = run_numeric_invariants(bundle)
        check_names = [c["name"] for c in result["checks"]]
        assert "no_wall_penetration" not in check_names
        assert "wall_proximity_fraction" not in check_names

    def test_enforce_wall_clearance_buffer(self, tmp_path):
        scenario = _minimal_scenario(
            evaluation_overrides={
                "enforce_wall_clearance_buffer": True,
                "wall_clearance_buffer": 0.01,
            },
        )
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows, events=[])
        result = run_numeric_invariants(bundle)
        check_names = [c["name"] for c in result["checks"]]
        assert "wall_clearance_buffer" in check_names

    def test_reads_custom_eval_params(self, tmp_path):
        scenario = _minimal_scenario(
            evaluation_overrides={
                "teleport_thresh": 2.0,
                "max_speed": 3.0,
                "max_accel": 10.0,
                "deadlock_seconds": 8.0,
                "min_robot_progress": 0.05,
            },
        )
        rows = [
            _state_row(0, [_agent_row(0, 0.5, 0.5, vx=0.3, vy=0.0, radius=0.02)]),
        ]
        bundle = _write_bundle(tmp_path, scenario, state_rows=rows, events=[])
        result = run_numeric_invariants(bundle)
        # Should not crash and produce valid output
        assert isinstance(result["overall_pass"], bool)


# ---------------------------------------------------------------------------
# build_visual_summary
# ---------------------------------------------------------------------------


class TestBuildVisualSummary:
    def test_basic_summary(self, tmp_path):
        scenario = _minimal_scenario()
        bundle = _write_bundle(
            tmp_path,
            scenario,
            write_frames=True,
            frame_count=3,
        )
        invariants = {"overall_pass": True, "checks": []}
        result = build_visual_summary(bundle, invariants)
        assert result["scenario_id"] == "test_scenario"
        assert result["frame_count"] == 3
        assert result["has_video"] is False
        assert result["invariants"] == invariants

    def test_with_summary_json(self, tmp_path):
        scenario = _minimal_scenario()
        summary_data = {
            "scenario_id": "test_scenario",
            "metrics": {"success_rate": 0.8, "collisions_agent_agent": 0},
        }
        bundle = _write_bundle(
            tmp_path,
            scenario,
            write_frames=True,
            frame_count=2,
            write_summary=True,
            summary_data=summary_data,
        )
        invariants = {"overall_pass": True, "checks": []}
        result = build_visual_summary(bundle, invariants)
        assert result["metrics"]["success_rate"] == 0.8

    def test_without_summary_json(self, tmp_path):
        scenario = _minimal_scenario()
        bundle = _write_bundle(
            tmp_path,
            scenario,
            write_frames=True,
            frame_count=1,
        )
        invariants = {"overall_pass": False, "checks": []}
        result = build_visual_summary(bundle, invariants)
        assert result["metrics"] == {}

    def test_render_diagnostics(self, tmp_path):
        scenario = _minimal_scenario()
        bundle = _write_bundle(
            tmp_path,
            scenario,
            write_frames=True,
            frame_count=1,
        )
        diag_path = bundle / "frames" / "render_diagnostics.json"
        diag_path.write_text(json.dumps({"fps": 30}), encoding="utf-8")
        invariants = {"overall_pass": True, "checks": []}
        result = build_visual_summary(bundle, invariants)
        assert result["render_diagnostics"]["fps"] == 30


# ---------------------------------------------------------------------------
# check_video_artifact
# ---------------------------------------------------------------------------


class TestCheckVideoArtifact:
    def test_missing_video(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "frames").mkdir()
        result = check_video_artifact(bundle)
        assert result["pass"] is False
        assert result["reason"] == "missing_video"

    def test_unreadable_video(self, tmp_path):
        bundle = tmp_path / "bundle"
        frames = bundle / "frames"
        frames.mkdir(parents=True)
        # Write garbage data as video
        (frames / "video.mp4").write_bytes(b"not a real video file")
        result = check_video_artifact(bundle)
        assert result["pass"] is False
        assert result["reason"] == "unreadable_video"

    def test_valid_video(self, tmp_path):
        bundle = tmp_path / "bundle"
        frames = bundle / "frames"
        frames.mkdir(parents=True)
        # Create a minimal valid video with OpenCV
        video_path = frames / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (20, 20))
        for _ in range(5):
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        if video_path.exists() and video_path.stat().st_size > 0:
            result = check_video_artifact(bundle)
            assert result["pass"] is True
            assert result["reason"] == "ok"
