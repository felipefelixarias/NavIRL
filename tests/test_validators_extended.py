"""Tests for navirl.verify.validators — unit tests for individual validation
functions using synthetic state/event data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navirl.verify.validators import (
    load_events,
    load_state_rows,
    validate_agent_stop_duration,
    validate_deadlock_bounded,
    validate_motion_jitter,
    validate_no_teleport,
    validate_speed_accel_bounds,
    validate_token_exclusivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_state(path: Path, rows: list[dict]) -> Path:
    """Write state rows as JSON Lines."""
    fp = path / "state.jsonl"
    with open(fp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return fp


def _write_events(path: Path, events: list[dict]) -> Path:
    """Write events as JSON Lines."""
    fp = path / "events.jsonl"
    with open(fp, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return fp


def _make_state_row(step: int, agents: list[dict]) -> dict:
    """Build a state row dict."""
    return {"step": step, "agents": agents}


def _make_agent_entry(
    aid: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 10.0,
    goal_y: float = 0.0,
    kind: str = "human",
    behavior: str = "GO_TO",
    max_speed: float = 1.5,
) -> dict:
    return {
        "id": aid,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "kind": kind,
        "behavior": behavior,
        "max_speed": max_speed,
    }


# ===================================================================
# load_state_rows / load_events
# ===================================================================


class TestLoadHelpers:
    def test_load_state_rows(self, tmp_path):
        rows = [_make_state_row(0, [_make_agent_entry()])]
        fp = _write_state(tmp_path, rows)
        result = load_state_rows(fp)
        assert len(result) == 1
        assert result[0]["step"] == 0

    def test_load_events_missing_file(self, tmp_path):
        fp = tmp_path / "nonexistent.jsonl"
        result = load_events(fp)
        assert result == []

    def test_load_events_present(self, tmp_path):
        events = [{"event_type": "test", "step": 0}]
        fp = _write_events(tmp_path, events)
        result = load_events(fp)
        assert len(result) == 1


# ===================================================================
# validate_no_teleport
# ===================================================================


class TestValidateNoTeleport:
    def test_no_violations(self, tmp_path):
        rows = [
            _make_state_row(0, [_make_agent_entry(x=0.0, y=0.0)]),
            _make_state_row(1, [_make_agent_entry(x=0.1, y=0.0)]),
            _make_state_row(2, [_make_agent_entry(x=0.2, y=0.0)]),
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_no_teleport(fp, teleport_thresh=1.0)
        assert result["pass"] is True
        assert result["num_violations"] == 0

    def test_teleport_detected(self, tmp_path):
        rows = [
            _make_state_row(0, [_make_agent_entry(x=0.0, y=0.0)]),
            _make_state_row(1, [_make_agent_entry(x=100.0, y=0.0)]),  # teleport!
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_no_teleport(fp, teleport_thresh=1.0)
        assert result["pass"] is False
        assert result["num_violations"] == 1
        assert result["violations"][0]["delta"] == pytest.approx(100.0)

    def test_multiple_agents(self, tmp_path):
        rows = [
            _make_state_row(
                0,
                [
                    _make_agent_entry(aid=0, x=0.0),
                    _make_agent_entry(aid=1, x=5.0),
                ],
            ),
            _make_state_row(
                1,
                [
                    _make_agent_entry(aid=0, x=0.1),
                    _make_agent_entry(aid=1, x=50.0),  # teleport
                ],
            ),
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_no_teleport(fp, teleport_thresh=1.0)
        assert result["num_violations"] == 1


# ===================================================================
# validate_speed_accel_bounds
# ===================================================================


class TestValidateSpeedAccelBounds:
    def test_within_bounds(self, tmp_path):
        rows = [
            _make_state_row(0, [_make_agent_entry(vx=0.5, vy=0.0)]),
            _make_state_row(1, [_make_agent_entry(vx=0.6, vy=0.0)]),
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_speed_accel_bounds(fp, dt=0.04, max_speed=1.5, max_accel=2.0)
        assert result["pass"] is True

    def test_speed_violation(self, tmp_path):
        rows = [
            _make_state_row(0, [_make_agent_entry(vx=5.0, vy=0.0, max_speed=1.5)]),
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_speed_accel_bounds(fp, dt=0.04, max_speed=1.5, max_accel=2.0)
        assert result["pass"] is False
        assert result["num_speed_violations"] > 0

    def test_accel_violation(self, tmp_path):
        rows = [
            _make_state_row(0, [_make_agent_entry(vx=0.0, vy=0.0)]),
            _make_state_row(1, [_make_agent_entry(vx=0.0, vy=0.0)]),
            _make_state_row(2, [_make_agent_entry(vx=5.0, vy=0.0)]),  # huge accel at step 2
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_speed_accel_bounds(fp, dt=0.04, max_speed=10.0, max_accel=1.0)
        assert result["num_accel_violations"] > 0


# ===================================================================
# validate_motion_jitter
# ===================================================================


class TestValidateMotionJitter:
    def test_smooth_motion_passes(self, tmp_path):
        # Constant heading, no jitter
        rows = [
            _make_state_row(i, [_make_agent_entry(x=float(i), vx=1.0, vy=0.0)]) for i in range(20)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_motion_jitter(fp, min_speed=0.1, max_flip_rate=0.5)
        assert result["pass"] is True
        assert result["worst_flip_rate"] == pytest.approx(0.0)

    def test_jittery_motion_fails(self, tmp_path):
        # Heading alternates every step
        rows = []
        for i in range(20):
            vx = 1.0 if i % 2 == 0 else -1.0
            rows.append(_make_state_row(i, [_make_agent_entry(x=float(i), vx=vx, vy=0.0)]))
        fp = _write_state(tmp_path, rows)
        result = validate_motion_jitter(fp, min_speed=0.1, max_flip_rate=0.3)
        assert result["pass"] is False
        assert result["worst_flip_rate"] > 0.3

    def test_low_speed_ignored(self, tmp_path):
        rows = [_make_state_row(i, [_make_agent_entry(vx=0.001, vy=0.0)]) for i in range(10)]
        fp = _write_state(tmp_path, rows)
        result = validate_motion_jitter(fp, min_speed=0.1, max_flip_rate=0.5)
        assert result["pass"] is True


# ===================================================================
# validate_token_exclusivity
# ===================================================================


class TestValidateTokenExclusivity:
    def test_valid_sequence(self, tmp_path):
        events = [
            {"event_type": "door_token_acquire", "agent_id": 1, "step": 0},
            {"event_type": "door_token_release", "agent_id": 1, "step": 5},
            {"event_type": "door_token_acquire", "agent_id": 2, "step": 6},
            {"event_type": "door_token_release", "agent_id": 2, "step": 10},
        ]
        fp = _write_events(tmp_path, events)
        result = validate_token_exclusivity(fp)
        assert result["pass"] is True

    def test_acquire_without_release(self, tmp_path):
        events = [
            {"event_type": "door_token_acquire", "agent_id": 1, "step": 0},
            {"event_type": "door_token_acquire", "agent_id": 2, "step": 3},  # violation!
        ]
        fp = _write_events(tmp_path, events)
        result = validate_token_exclusivity(fp)
        assert result["pass"] is False
        assert result["violations"][0]["reason"] == "acquire_without_release"

    def test_release_without_holder(self, tmp_path):
        events = [
            {"event_type": "door_token_release", "agent_id": 1, "step": 0},
        ]
        fp = _write_events(tmp_path, events)
        result = validate_token_exclusivity(fp)
        assert result["pass"] is False
        assert result["violations"][0]["reason"] == "release_without_holder"

    def test_release_by_non_holder(self, tmp_path):
        events = [
            {"event_type": "door_token_acquire", "agent_id": 1, "step": 0},
            {"event_type": "door_token_release", "agent_id": 2, "step": 3},  # wrong agent
        ]
        fp = _write_events(tmp_path, events)
        result = validate_token_exclusivity(fp)
        assert result["pass"] is False
        assert result["violations"][0]["reason"] == "release_by_non_holder"

    def test_empty_events(self, tmp_path):
        fp = _write_events(tmp_path, [])
        result = validate_token_exclusivity(fp)
        assert result["pass"] is True


# ===================================================================
# validate_deadlock_bounded
# ===================================================================


class TestValidateDeadlockBounded:
    def test_no_deadlock(self, tmp_path):
        rows = [
            _make_state_row(i, [_make_agent_entry(x=float(i) * 0.1, vx=0.5, vy=0.0)])
            for i in range(50)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_deadlock_bounded(fp, dt=0.04, deadlock_seconds=1.0)
        assert result["pass"] is True

    def test_deadlock_detected(self, tmp_path):
        # Agent stuck at origin far from goal for many steps
        rows = [
            _make_state_row(i, [_make_agent_entry(x=0.0, y=0.0, vx=0.0, vy=0.0, goal_x=10.0)])
            for i in range(100)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_deadlock_bounded(fp, dt=0.04, deadlock_seconds=1.0)
        assert result["pass"] is False
        assert result["num_violations"] > 0

    def test_done_behavior_resets_streak(self, tmp_path):
        rows = [
            _make_state_row(
                i, [_make_agent_entry(x=0.0, vx=0.0, vy=0.0, goal_x=10.0, behavior="DONE")]
            )
            for i in range(100)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_deadlock_bounded(fp, dt=0.04, deadlock_seconds=1.0)
        assert result["pass"] is True

    def test_yielding_not_flagged(self, tmp_path):
        rows = [
            _make_state_row(
                i, [_make_agent_entry(x=0.0, vx=0.0, vy=0.0, goal_x=10.0, behavior="YIELDING")]
            )
            for i in range(100)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_deadlock_bounded(fp, dt=0.04, deadlock_seconds=1.0)
        assert result["pass"] is True


# ===================================================================
# validate_agent_stop_duration
# ===================================================================


class TestValidateAgentStopDuration:
    def test_no_stops(self, tmp_path):
        rows = [
            _make_state_row(i, [_make_agent_entry(vx=0.5, vy=0.0, x=float(i) * 0.1)])
            for i in range(50)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_agent_stop_duration(
            fp,
            dt=0.04,
            max_stop_seconds=1.0,
            stop_speed_thresh=0.1,
        )
        assert result["pass"] is True

    def test_long_stop_detected(self, tmp_path):
        rows = [
            _make_state_row(i, [_make_agent_entry(x=0.0, vx=0.0, vy=0.0, goal_x=10.0)])
            for i in range(200)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_agent_stop_duration(
            fp,
            dt=0.04,
            max_stop_seconds=1.0,
            stop_speed_thresh=0.1,
        )
        assert result["pass"] is False
        assert result["num_violations"] > 0

    def test_wait_behavior_excluded(self, tmp_path):
        rows = [
            _make_state_row(
                i, [_make_agent_entry(x=0.0, vx=0.0, vy=0.0, goal_x=10.0, behavior="WAIT")]
            )
            for i in range(200)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_agent_stop_duration(
            fp,
            dt=0.04,
            max_stop_seconds=1.0,
            stop_speed_thresh=0.1,
        )
        assert result["pass"] is True

    def test_near_goal_not_flagged(self, tmp_path):
        # Agent stopped but at goal (within goal_tol=0.2)
        rows = [
            _make_state_row(i, [_make_agent_entry(x=10.0, vx=0.0, vy=0.0, goal_x=10.0, goal_y=0.0)])
            for i in range(200)
        ]
        fp = _write_state(tmp_path, rows)
        result = validate_agent_stop_duration(
            fp,
            dt=0.04,
            max_stop_seconds=1.0,
            stop_speed_thresh=0.1,
        )
        assert result["pass"] is True
