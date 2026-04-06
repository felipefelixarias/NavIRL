"""Tests for navirl.logging.episode_log — episode logging, export, and statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from navirl.core.types import AgentState, EventRecord
from navirl.logging.episode_log import (
    AgentTrajectory,
    EpisodeEvent,
    EpisodeLogger,
    EpisodeStatistics,
    TrajectoryPoint,
    episode_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_state(agent_id=0, kind="robot", x=1.0, y=2.0, vx=0.5, vy=0.3):
    return AgentState(
        agent_id=agent_id,
        kind=kind,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=10.0,
        goal_y=10.0,
        radius=0.3,
        max_speed=1.5,
    )


def _make_event(step=0, time_s=0.0, event_type="collision", agent_id=0):
    return EventRecord(
        step=step,
        time_s=time_s,
        event_type=event_type,
        agent_id=agent_id,
    )


# ---------------------------------------------------------------------------
# TrajectoryPoint
# ---------------------------------------------------------------------------


class TestTrajectoryPoint:
    def test_from_agent_state(self):
        st = _make_agent_state(vx=3.0, vy=4.0)
        pt = TrajectoryPoint.from_agent_state(step=5, time_s=1.0, state=st)
        assert pt.step == 5
        assert pt.time_s == 1.0
        assert pt.x == st.x
        assert pt.y == st.y
        assert pt.speed == pytest.approx(5.0)
        assert pt.heading == pytest.approx(math.atan2(4.0, 3.0))


# ---------------------------------------------------------------------------
# AgentTrajectory
# ---------------------------------------------------------------------------


class TestAgentTrajectory:
    def _make_traj(self, n=10):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        for i in range(n):
            pt = TrajectoryPoint(
                step=i, time_s=float(i) * 0.1,
                x=float(i), y=float(i) * 0.5,
                vx=1.0, vy=0.5,
                speed=math.sqrt(1.0 + 0.25),
                heading=math.atan2(0.5, 1.0),
            )
            traj.add_point(pt)
        return traj

    def test_total_distance(self):
        traj = self._make_traj(3)
        # Points: (0,0), (1,0.5), (2,1.0)
        seg1 = math.sqrt(1 + 0.25)
        expected = seg1 * 2
        assert traj.total_distance == pytest.approx(expected)

    def test_total_distance_empty(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        assert traj.total_distance == 0.0

    def test_total_distance_single_point(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        traj.add_point(TrajectoryPoint(0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0))
        assert traj.total_distance == 0.0

    def test_duration(self):
        traj = self._make_traj(10)
        assert traj.duration == pytest.approx(0.9)

    def test_duration_single_point(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        traj.add_point(TrajectoryPoint(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert traj.duration == 0.0

    def test_average_speed(self):
        traj = self._make_traj(5)
        expected = math.sqrt(1.25)
        assert traj.average_speed == pytest.approx(expected)

    def test_average_speed_empty(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        assert traj.average_speed == 0.0

    def test_max_speed(self):
        traj = self._make_traj(5)
        assert traj.max_speed == pytest.approx(math.sqrt(1.25))

    def test_max_speed_empty(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        assert traj.max_speed == 0.0

    def test_displacement(self):
        traj = self._make_traj(5)
        # From (0,0) to (4,2)
        expected = math.sqrt(16 + 4)
        assert traj.displacement == pytest.approx(expected)

    def test_displacement_single_point(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        traj.add_point(TrajectoryPoint(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert traj.displacement == 0.0

    def test_path_efficiency(self):
        traj = self._make_traj(5)
        eff = traj.path_efficiency
        assert 0.0 < eff <= 1.0

    def test_path_efficiency_zero_distance(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        traj.add_point(TrajectoryPoint(0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        traj.add_point(TrajectoryPoint(1, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        assert traj.path_efficiency == 0.0

    def test_positions_array(self):
        traj = self._make_traj(3)
        arr = traj.positions_array()
        assert arr.shape == (3, 2)
        np.testing.assert_almost_equal(arr[0], [0.0, 0.0])

    def test_positions_array_empty(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        arr = traj.positions_array()
        assert arr.shape == (0, 2)

    def test_velocities_array(self):
        traj = self._make_traj(3)
        arr = traj.velocities_array()
        assert arr.shape == (3, 2)

    def test_velocities_array_empty(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        arr = traj.velocities_array()
        assert arr.shape == (0, 2)

    def test_speeds_array(self):
        traj = self._make_traj(3)
        arr = traj.speeds_array()
        assert arr.shape == (3,)

    def test_speeds_array_empty(self):
        traj = AgentTrajectory(agent_id=0, kind="robot")
        arr = traj.speeds_array()
        assert arr.shape == (0,)

    def test_to_dict(self):
        traj = self._make_traj(3)
        d = traj.to_dict()
        assert d["agent_id"] == 0
        assert d["kind"] == "robot"
        assert d["num_points"] == 3
        assert "total_distance" in d
        assert "points" in d
        assert len(d["points"]) == 3


# ---------------------------------------------------------------------------
# EpisodeEvent
# ---------------------------------------------------------------------------


class TestEpisodeEvent:
    def test_from_event_record(self):
        record = _make_event(step=5, time_s=1.0, event_type="collision", agent_id=2)
        ev = EpisodeEvent.from_event_record(record)
        assert ev.step == 5
        assert ev.event_type == "collision"
        assert ev.agent_id == 2
        assert ev.wall_time > 0

    def test_to_dict(self):
        ev = EpisodeEvent(step=1, time_s=0.5, event_type="test", agent_id=None)
        d = ev.to_dict()
        assert d["step"] == 1
        assert d["event_type"] == "test"
        assert d["agent_id"] is None


# ---------------------------------------------------------------------------
# EpisodeStatistics
# ---------------------------------------------------------------------------


class TestEpisodeStatistics:
    def test_to_dict(self):
        stats = EpisodeStatistics(
            episode_id="test_ep",
            num_steps=100,
            num_agents=3,
            event_counts={"collision": 2},
        )
        d = stats.to_dict()
        assert d["episode_id"] == "test_ep"
        assert d["num_steps"] == 100
        assert d["event_counts"] == {"collision": 2}


# ---------------------------------------------------------------------------
# EpisodeLogger
# ---------------------------------------------------------------------------


class TestEpisodeLogger:
    def test_init(self, tmp_path):
        logger = EpisodeLogger(tmp_path / "ep1")
        assert not logger.is_closed
        assert logger.bundle_dir.exists()
        logger.close()

    def test_context_manager(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep2") as logger:
            assert not logger.is_closed
        assert logger.is_closed

    def test_write_state(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state(0, "robot"), _make_agent_state(1, "human")]
            logger.write_state(0, 0.0, agents)
            logger.write_state(1, 0.1, agents)
        # State file should have data
        assert logger.state_path.stat().st_size > 0

    def test_write_state_closed_raises(self, tmp_path):
        logger = EpisodeLogger(tmp_path / "ep")
        logger.close()
        with pytest.raises(RuntimeError, match="closed"):
            logger.write_state(0, 0.0, [])

    def test_write_event(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_event(_make_event())
        assert logger.events_path.stat().st_size > 0

    def test_write_event_closed_raises(self, tmp_path):
        logger = EpisodeLogger(tmp_path / "ep")
        logger.close()
        with pytest.raises(RuntimeError, match="closed"):
            logger.write_event(_make_event())

    def test_write_custom_event(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_custom_event(0, 0.0, "custom_event", agent_id=1, data="test")
            events = logger.get_events("custom_event")
            assert len(events) == 1
            assert events[0].payload["data"] == "test"

    def test_write_reward(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_reward(0, 0.0, agent_id=1, reward=1.5, components={"goal": 1.0, "time": 0.5})
            history = logger.get_reward_history(agent_id=1)
            assert len(history) == 1
            assert history[0]["reward"] == 1.5

    def test_get_cumulative_reward(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_reward(0, 0.0, agent_id=1, reward=1.0)
            logger.write_reward(1, 0.1, agent_id=1, reward=2.0)
            logger.write_reward(0, 0.0, agent_id=2, reward=5.0)
            assert logger.get_cumulative_reward(1) == pytest.approx(3.0)
            assert logger.get_cumulative_reward(2) == pytest.approx(5.0)

    def test_metadata(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.set_metadata("scenario", "hallway")
            assert logger.get_metadata("scenario") == "hallway"
            assert logger.get_metadata("missing", "default") == "default"

    def test_compute_statistics(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            a1 = _make_agent_state(0, "robot", x=0, y=0, vx=1.0, vy=0.0)
            a2 = _make_agent_state(1, "human", x=5, y=5, vx=0.0, vy=1.0)
            for i in range(10):
                a1 = _make_agent_state(0, "robot", x=float(i), y=0, vx=1.0, vy=0.0)
                a2 = _make_agent_state(1, "human", x=5, y=float(i), vx=0.0, vy=1.0)
                logger.write_state(i, float(i) * 0.1, [a1, a2])
            logger.write_event(_make_event(event_type="collision"))
            logger.write_event(_make_event(event_type="collision"))
            logger.write_event(_make_event(event_type="goal_reached"))
            stats = logger.compute_statistics()
            assert stats.num_steps == 10
            assert stats.num_agents == 2
            assert stats.num_robots == 1
            assert stats.num_humans == 1
            assert stats.num_collisions == 2
            assert stats.num_events == 3
            assert stats.event_counts["collision"] == 2
            assert stats.mean_speed > 0

    def test_save_statistics(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state()]
            logger.write_state(0, 0.0, agents)
            stats = logger.save_statistics()
        assert logger.statistics_path.exists()
        data = json.loads(logger.statistics_path.read_text())
        assert data["episode_id"] == stats.episode_id

    def test_get_trajectory(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state(0), _make_agent_state(1)]
            logger.write_state(0, 0.0, agents)
            assert logger.get_trajectory(0) is not None
            assert logger.get_trajectory(999) is None

    def test_get_all_trajectories(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state(0), _make_agent_state(1)]
            logger.write_state(0, 0.0, agents)
            trajs = logger.get_all_trajectories()
            assert len(trajs) == 2
            assert 0 in trajs
            assert 1 in trajs

    def test_save_trajectories(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state(0)]
            logger.write_state(0, 0.0, agents)
            logger.save_trajectories()
        data = json.loads(logger.trajectories_path.read_text())
        assert "0" in data

    def test_get_events_filter(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_event(_make_event(event_type="collision"))
            logger.write_event(_make_event(event_type="goal_reached"))
            all_events = logger.get_events()
            assert len(all_events) == 2
            collisions = logger.get_events("collision")
            assert len(collisions) == 1

    def test_get_event_counts(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_event(_make_event(event_type="collision"))
            logger.write_event(_make_event(event_type="collision"))
            logger.write_event(_make_event(event_type="goal"))
            counts = logger.get_event_counts()
            assert counts["collision"] == 2
            assert counts["goal"] == 1

    def test_get_reward_history_all(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_reward(0, 0.0, 1, 1.0)
            logger.write_reward(0, 0.0, 2, 2.0)
            assert len(logger.get_reward_history()) == 2

    def test_write_summary(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_summary({"result": "success"})
        data = json.loads(logger.summary_path.read_text())
        assert data["result"] == "success"

    def test_write_resolved_scenario(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_resolved_scenario({"name": "test_scenario"})
        assert logger.scenario_path.exists()

    def test_close_idempotent(self, tmp_path):
        logger = EpisodeLogger(tmp_path / "ep")
        logger.close()
        logger.close()  # Should not raise
        assert logger.is_closed

    def test_buffer_flush(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep", buffer_size=3) as logger:
            agents = [_make_agent_state()]
            for i in range(5):
                logger.write_state(i, float(i) * 0.1, agents)
        # All should be flushed on close
        lines = logger.state_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_export_csv(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state(0), _make_agent_state(1)]
            logger.write_state(0, 0.0, agents)
            logger.write_state(1, 0.1, agents)
            csv_path = logger.export_csv()
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 5  # header + 4 agent-step rows

    def test_export_events_csv(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_event(_make_event(event_type="collision"))
            logger.write_event(_make_event(event_type="goal"))
            csv_path = logger.export_events_csv()
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 events

    def test_export_json(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state()]
            logger.write_state(0, 0.0, agents)
            logger.write_event(_make_event())
            logger.set_metadata("key", "val")
            json_path = logger.export_json()
        data = json.loads(json_path.read_text())
        assert "states" in data
        assert "events" in data
        assert "trajectories" in data
        assert data["metadata"]["key"] == "val"

    def test_export_rewards_csv(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_reward(0, 0.0, 1, 1.5, components={"goal": 1.0, "time": 0.5})
            logger.write_reward(1, 0.1, 1, 2.0, components={"goal": 1.5, "time": 0.5})
            csv_path = logger.export_rewards_csv()
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rewards
        # Header should include component columns
        assert "goal" in lines[0]
        assert "time" in lines[0]

    def test_iter_states(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state()]
            for i in range(5):
                logger.write_state(i, float(i) * 0.1, agents)
            states = list(logger.iter_states())
            assert len(states) == 5
            assert states[0]["step"] == 0

    def test_pairwise_distances(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            a1 = _make_agent_state(0, x=0.0, y=0.0)
            a2 = _make_agent_state(1, x=3.0, y=4.0)
            logger.write_state(0, 0.0, [a1, a2])
            dists = logger.pairwise_distances(0)
            assert dists is not None
            assert dists.shape == (2, 2)
            assert dists[0, 1] == pytest.approx(5.0)
            assert dists[1, 0] == pytest.approx(5.0)

    def test_pairwise_distances_no_data(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            assert logger.pairwise_distances(0) is None

    def test_pairwise_distances_missing_step(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            agents = [_make_agent_state(0), _make_agent_state(1)]
            logger.write_state(0, 0.0, agents)
            assert logger.pairwise_distances(999) is None

    def test_minimum_separation(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            a1 = _make_agent_state(0, x=0.0, y=0.0)
            a2 = _make_agent_state(1, x=3.0, y=4.0)
            logger.write_state(0, 0.0, [a1, a2])
            a1_close = _make_agent_state(0, x=0.0, y=0.0)
            a2_close = _make_agent_state(1, x=1.0, y=0.0)
            logger.write_state(1, 0.1, [a1_close, a2_close])
            sep = logger.minimum_separation()
            assert sep == pytest.approx(1.0)

    def test_minimum_separation_single_agent(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep") as logger:
            logger.write_state(0, 0.0, [_make_agent_state(0)])
            assert logger.minimum_separation() == float("inf")

    def test_no_trajectory_recording(self, tmp_path):
        with EpisodeLogger(tmp_path / "ep", record_trajectories=False) as logger:
            agents = [_make_agent_state()]
            logger.write_state(0, 0.0, agents)
            assert logger.get_trajectory(0) is None


# ---------------------------------------------------------------------------
# episode_context
# ---------------------------------------------------------------------------


class TestEpisodeContext:
    def test_basic(self, tmp_path):
        with episode_context(tmp_path / "ep", episode_id="test_ep") as logger:
            assert logger.episode_id == "test_ep"
            logger.write_state(0, 0.0, [_make_agent_state()])
        assert logger.is_closed

    def test_closes_on_exception(self, tmp_path):
        with pytest.raises(ValueError):
            with episode_context(tmp_path / "ep") as logger:
                raise ValueError("test error")
        assert logger.is_closed
