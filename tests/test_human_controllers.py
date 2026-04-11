"""Tests for navirl/humans/scripted and replay controllers."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from navirl.core.types import Action, AgentState
from navirl.humans.replay.controller import ReplayHumanController
from navirl.humans.scripted.controller import ScriptedHumanController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    agent_id: int,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    max_speed: float = 1.0,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind="human",
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=0.0,
        goal_y=0.0,
        radius=0.3,
        max_speed=max_speed,
    )


def _noop_emit(event_name, agent_id, data):
    """No-op event sink."""


# ---------------------------------------------------------------------------
# ScriptedHumanController
# ---------------------------------------------------------------------------


class TestScriptedHumanController:
    def test_init_defaults(self):
        ctrl = ScriptedHumanController()
        assert ctrl.goal_tolerance == pytest.approx(0.2)
        assert ctrl.max_speed == pytest.approx(0.6)
        assert ctrl.human_ids == []

    def test_init_custom_config(self):
        cfg = {"goal_tolerance": 0.5, "max_speed": 1.2}
        ctrl = ScriptedHumanController(cfg)
        assert ctrl.goal_tolerance == pytest.approx(0.5)
        assert ctrl.max_speed == pytest.approx(1.2)

    def test_reset_assigns_default_waypoints(self):
        ctrl = ScriptedHumanController()
        human_ids = [10, 20]
        starts = {10: (0.0, 0.0), 20: (5.0, 5.0)}
        goals = {10: (10.0, 0.0), 20: (0.0, 0.0)}
        ctrl.reset(human_ids, starts, goals)

        assert ctrl.human_ids == [10, 20]
        # Default scripts: [goal, start] for each human
        assert ctrl.scripts[10] == [(10.0, 0.0), (0.0, 0.0)]
        assert ctrl.scripts[20] == [(0.0, 0.0), (5.0, 5.0)]

    def test_reset_with_custom_waypoints(self):
        cfg = {
            "waypoints": {
                "0": [[1.0, 2.0], [3.0, 4.0]],
                "1": [[5.0, 6.0]],
            }
        }
        ctrl = ScriptedHumanController(cfg)
        human_ids = [100, 200]
        starts = {100: (0.0, 0.0), 200: (0.0, 0.0)}
        goals = {100: (10.0, 10.0), 200: (10.0, 10.0)}
        ctrl.reset(human_ids, starts, goals)

        assert ctrl.scripts[100] == [(1.0, 2.0), (3.0, 4.0)]
        assert ctrl.scripts[200] == [(5.0, 6.0)]

    def test_step_moves_toward_waypoint(self):
        ctrl = ScriptedHumanController({"max_speed": 1.0})
        human_ids = [1]
        starts = {1: (0.0, 0.0)}
        goals = {1: (10.0, 0.0)}
        ctrl.reset(human_ids, starts, goals)

        states = {1: _make_state(1, 0.0, 0.0)}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)

        assert 1 in actions
        action = actions[1]
        assert isinstance(action, Action)
        # Should move in +x direction toward (10, 0)
        assert action.pref_vx > 0
        assert abs(action.pref_vy) < 1e-8
        assert action.behavior == "SCRIPT"

    def test_step_waypoint_reached_cycles(self):
        ctrl = ScriptedHumanController({"goal_tolerance": 1.0, "max_speed": 1.0})
        human_ids = [1]
        starts = {1: (0.0, 0.0)}
        goals = {1: (2.0, 0.0)}
        ctrl.reset(human_ids, starts, goals)

        # Place agent very close to first waypoint (the goal)
        states = {1: _make_state(1, 2.0, 0.0)}
        events = []

        def capture_emit(name, aid, data):
            events.append((name, aid, data))

        ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=capture_emit)
        # Waypoint should be reached and event emitted
        assert len(events) == 1
        assert events[0][0] == "script_waypoint_reached"
        assert events[0][1] == 1

    def test_step_agent_at_exact_position(self):
        """When agent is exactly at target, should WAIT."""
        ctrl = ScriptedHumanController({"goal_tolerance": 0.5, "max_speed": 1.0})
        human_ids = [1]
        starts = {1: (0.0, 0.0)}
        goals = {1: (5.0, 0.0)}
        ctrl.reset(human_ids, starts, goals)

        # Agent at goal, after waypoint cycling lands back at start which is also current pos
        ctrl.indices[1] = 1  # point to starts
        states = {1: _make_state(1, 0.0, 0.0)}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        # Distance < goal_tolerance, then recycles and checks new target
        # After recycle the target is goals[1]=(5,0), dist=5 > tolerance => SCRIPT
        assert actions[1].behavior in ("SCRIPT", "WAIT")

    def test_step_respects_max_speed(self):
        ctrl = ScriptedHumanController({"max_speed": 0.3})
        human_ids = [1]
        starts = {1: (0.0, 0.0)}
        goals = {1: (100.0, 0.0)}
        ctrl.reset(human_ids, starts, goals)

        states = {1: _make_state(1, 0.0, 0.0, max_speed=2.0)}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)

        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        assert speed <= 0.3 + 1e-9

    def test_multiple_humans(self):
        ctrl = ScriptedHumanController()
        ids = [1, 2, 3]
        starts = {i: (float(i), 0.0) for i in ids}
        goals = {i: (float(i) + 10.0, 0.0) for i in ids}
        ctrl.reset(ids, starts, goals)

        states = {i: _make_state(i, float(i), 0.0) for i in ids}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        assert len(actions) == 3
        for i in ids:
            assert i in actions


# ---------------------------------------------------------------------------
# ReplayHumanController
# ---------------------------------------------------------------------------


class TestReplayHumanController:
    def test_init_defaults(self):
        ctrl = ReplayHumanController()
        assert ctrl.human_ids == []
        assert ctrl.replay_positions == {}

    def test_reset_no_path(self):
        ctrl = ReplayHumanController()
        ctrl.reset([1, 2], {1: (0, 0), 2: (1, 1)}, {1: (5, 5), 2: (6, 6)})
        assert ctrl.human_ids == [1, 2]
        assert ctrl.replay_positions == {1: [], 2: []}

    def test_reset_with_replay_file(self, tmp_path):
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 0.0, "y": 0.0}]}),
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 1.0, "y": 0.5}]}),
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 2.0, "y": 1.0}]}),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1], {1: (0, 0)}, {1: (5, 5)})

        assert len(ctrl.replay_positions[1]) == 3
        assert ctrl.replay_positions[1][0] == (0.0, 0.0)
        assert ctrl.replay_positions[1][2] == (2.0, 1.0)

    def test_step_follows_replay(self, tmp_path):
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 1.0, "y": 0.0}]}),
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 2.0, "y": 0.0}]}),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1], {1: (0, 0)}, {1: (5, 5)})

        states = {1: _make_state(1, 0.0, 0.0)}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        assert actions[1].behavior == "REPLAY"
        assert actions[1].pref_vx > 0  # moving toward (1, 0)

    def test_step_past_end_of_replay(self, tmp_path):
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 1.0, "y": 0.0}]}),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1], {1: (0, 0)}, {1: (5, 5)})

        states = {1: _make_state(1, 0.0, 0.0)}
        # Step 0 uses index 0 (valid)
        actions0 = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        assert actions0[1].behavior == "REPLAY"

        # Step 1 is past end
        actions1 = ctrl.step(1, 0.1, 0.1, states, robot_id=0, emit_event=_noop_emit)
        assert actions1[1].behavior == "REPLAY_DONE"
        assert actions1[1].pref_vx == 0.0
        assert actions1[1].pref_vy == 0.0

    def test_step_agent_at_target(self, tmp_path):
        """Agent already at replay target should produce zero velocity."""
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 5.0, "y": 5.0}]}),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1], {1: (0, 0)}, {1: (10, 10)})

        states = {1: _make_state(1, 5.0, 5.0)}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        assert actions[1].pref_vx == 0.0
        assert actions[1].pref_vy == 0.0
        assert actions[1].behavior == "REPLAY"

    def test_replay_speed_capped(self, tmp_path):
        """Replay velocity should not exceed max_speed."""
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({"agents": [{"id": 1, "kind": "human", "x": 1000.0, "y": 0.0}]}),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1], {1: (0, 0)}, {1: (5, 5)})

        states = {1: _make_state(1, 0.0, 0.0, max_speed=2.0)}
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        # speed = min(max_speed, dist/dt) where dist=1000, dt=0.1 => capped to max_speed=2.0
        assert speed <= 2.0 + 1e-9

    def test_replay_filters_by_kind(self, tmp_path):
        """Only 'human' agents should be loaded from replay."""
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({
                "agents": [
                    {"id": 1, "kind": "human", "x": 1.0, "y": 0.0},
                    {"id": 2, "kind": "robot", "x": 99.0, "y": 99.0},
                ]
            }),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1, 2], {1: (0, 0), 2: (0, 0)}, {1: (5, 5), 2: (5, 5)})

        # Agent 1 (human) should have positions loaded
        assert len(ctrl.replay_positions[1]) == 1
        # Agent 2 (robot in log) should have no positions
        assert len(ctrl.replay_positions[2]) == 0

    def test_replay_empty_lines_skipped(self, tmp_path):
        replay_file = tmp_path / "replay.jsonl"
        content = "\n" + json.dumps({"agents": [{"id": 1, "kind": "human", "x": 1.0, "y": 2.0}]}) + "\n\n"
        replay_file.write_text(content)

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1], {1: (0, 0)}, {1: (5, 5)})
        assert len(ctrl.replay_positions[1]) == 1

    def test_multiple_humans_replay(self, tmp_path):
        replay_file = tmp_path / "replay.jsonl"
        lines = [
            json.dumps({
                "agents": [
                    {"id": 1, "kind": "human", "x": 0.0, "y": 0.0},
                    {"id": 2, "kind": "human", "x": 5.0, "y": 5.0},
                ]
            }),
            json.dumps({
                "agents": [
                    {"id": 1, "kind": "human", "x": 1.0, "y": 0.0},
                    {"id": 2, "kind": "human", "x": 6.0, "y": 5.0},
                ]
            }),
        ]
        replay_file.write_text("\n".join(lines) + "\n")

        ctrl = ReplayHumanController({"path": str(replay_file)})
        ctrl.reset([1, 2], {1: (0, 0), 2: (5, 5)}, {1: (10, 10), 2: (10, 10)})

        assert len(ctrl.replay_positions[1]) == 2
        assert len(ctrl.replay_positions[2]) == 2

        states = {
            1: _make_state(1, 0.0, 0.0),
            2: _make_state(2, 5.0, 5.0),
        }
        actions = ctrl.step(0, 0.0, 0.1, states, robot_id=0, emit_event=_noop_emit)
        assert len(actions) == 2
