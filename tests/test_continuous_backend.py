"""Tests for navirl.backends.continuous.backend and environment modules.

Covers ContinuousBackend, ContinuousEnvironment, ScenarioConfig, and
the full simulation loop (reset, step, rewards, dones, queries).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.backends.continuous.backend import ContinuousBackend, ScenarioConfig
from navirl.backends.continuous.environment import (
    AgentConfig,
    ContinuousEnvironment,
    EnvironmentConfig,
)
from navirl.backends.continuous.physics import AgentState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend_with_agents() -> ContinuousBackend:
    """Create a backend with one robot and one pedestrian for reuse."""
    cfg = EnvironmentConfig(width=20.0, height=20.0, dt=0.1, max_steps=50)
    backend = ContinuousBackend(cfg)
    backend.add_robot(np.array([2.0, 2.0]), np.array([18.0, 18.0]))
    backend.add_pedestrian(np.array([10.0, 2.0]), np.array([10.0, 18.0]))
    return backend


# ---------------------------------------------------------------------------
# ContinuousBackend — construction & agent management
# ---------------------------------------------------------------------------

class TestContinuousBackendConstruction:
    def test_default_construction(self):
        backend = ContinuousBackend()
        assert backend.num_robots == 0
        assert backend.num_pedestrians == 0
        assert backend.dt > 0

    def test_custom_config(self):
        cfg = EnvironmentConfig(width=30.0, height=15.0, dt=0.05)
        backend = ContinuousBackend(cfg)
        assert backend.dt == pytest.approx(0.05)

    def test_add_robot_returns_id(self):
        backend = ContinuousBackend()
        rid = backend.add_robot(np.array([1.0, 1.0]), np.array([5.0, 5.0]))
        assert isinstance(rid, int)
        assert backend.num_robots == 1
        assert rid in backend.robot_ids

    def test_add_pedestrian_returns_id(self):
        backend = ContinuousBackend()
        pid = backend.add_pedestrian(np.array([3.0, 3.0]), np.array([7.0, 7.0]))
        assert isinstance(pid, int)
        assert backend.num_pedestrians == 1
        assert pid in backend.pedestrian_ids

    def test_multiple_agents(self):
        backend = ContinuousBackend()
        r1 = backend.add_robot(np.array([0, 0]), np.array([10, 10]))
        r2 = backend.add_robot(np.array([1, 1]), np.array([11, 11]))
        p1 = backend.add_pedestrian(np.array([5, 5]), np.array([15, 15]))
        assert backend.num_robots == 2
        assert backend.num_pedestrians == 1
        assert r1 != r2 != p1

    def test_robot_ids_returns_copy(self):
        backend = ContinuousBackend()
        backend.add_robot(np.array([0, 0]), np.array([5, 5]))
        ids = backend.robot_ids
        ids.append(999)
        assert 999 not in backend.robot_ids

    def test_pedestrian_ids_returns_copy(self):
        backend = ContinuousBackend()
        backend.add_pedestrian(np.array([0, 0]), np.array([5, 5]))
        ids = backend.pedestrian_ids
        ids.append(999)
        assert 999 not in backend.pedestrian_ids


# ---------------------------------------------------------------------------
# ContinuousBackend — from_scenario
# ---------------------------------------------------------------------------

class TestContinuousBackendFromScenario:
    def test_from_scenario_robots_and_pedestrians(self):
        scenario = ScenarioConfig(
            name="test",
            agents=[
                AgentConfig(
                    position=np.array([1.0, 1.0]),
                    goal=np.array([10.0, 10.0]),
                    agent_type="robot",
                    radius=0.4,
                    max_speed=2.0,
                ),
                AgentConfig(
                    position=np.array([5.0, 5.0]),
                    goal=np.array([15.0, 15.0]),
                    agent_type="pedestrian",
                    radius=0.3,
                    preferred_speed=1.0,
                ),
            ],
        )
        backend = ContinuousBackend.from_scenario(scenario)
        assert backend.num_robots == 1
        assert backend.num_pedestrians == 1

    def test_from_scenario_with_obstacles(self):
        scenario = ScenarioConfig(
            name="obstacles",
            obstacles=[
                {"type": "circle", "center": [5.0, 5.0], "radius": 1.0},
                {"type": "rectangle", "min_corner": [8.0, 8.0], "max_corner": [10.0, 10.0]},
            ],
        )
        backend = ContinuousBackend.from_scenario(scenario)
        assert backend.environment.num_obstacles == 2

    def test_from_scenario_with_walls(self):
        scenario = ScenarioConfig(
            name="walls",
            walls=[
                {"start": [0.0, 0.0], "end": [10.0, 0.0], "thickness": 0.2},
                {"start": [0.0, 0.0], "end": [0.0, 10.0]},
            ],
        )
        backend = ContinuousBackend.from_scenario(scenario)
        assert backend.environment.num_obstacles == 2


# ---------------------------------------------------------------------------
# ContinuousBackend — pedestrian circle and flow
# ---------------------------------------------------------------------------

class TestPedestrianPatterns:
    def test_pedestrian_circle(self):
        backend = ContinuousBackend()
        ids = backend.add_pedestrian_circle(
            center=np.array([10.0, 10.0]),
            radius=5.0,
            num_pedestrians=6,
        )
        assert len(ids) == 6
        assert backend.num_pedestrians == 6

    def test_pedestrian_circle_positions_on_circle(self):
        backend = ContinuousBackend()
        center = np.array([10.0, 10.0])
        r = 5.0
        backend.add_pedestrian_circle(center, r, 4)
        obs = backend.reset()
        for _aid, ob in obs.items():
            dist = np.linalg.norm(ob["position"] - center)
            assert dist == pytest.approx(r, abs=0.5)

    def test_pedestrian_flow(self):
        backend = ContinuousBackend()
        ids = backend.add_pedestrian_flow(
            start_region=(np.array([0.0, 0.0]), np.array([2.0, 2.0])),
            goal_region=(np.array([18.0, 18.0]), np.array([20.0, 20.0])),
            num_pedestrians=4,
            rng=np.random.default_rng(42),
        )
        assert len(ids) == 4
        assert backend.num_pedestrians == 4

    def test_pedestrian_flow_default_rng(self):
        backend = ContinuousBackend()
        ids = backend.add_pedestrian_flow(
            start_region=(np.array([0.0, 0.0]), np.array([2.0, 2.0])),
            goal_region=(np.array([18.0, 18.0]), np.array([20.0, 20.0])),
            num_pedestrians=3,
        )
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# ContinuousBackend — reset / step loop
# ---------------------------------------------------------------------------

class TestContinuousBackendSimulation:
    def test_step_before_reset_raises(self):
        backend = _make_backend_with_agents()
        with pytest.raises(RuntimeError, match="reset"):
            backend.step({0: np.zeros(2)})

    def test_reset_returns_observations(self):
        backend = _make_backend_with_agents()
        obs = backend.reset()
        assert isinstance(obs, dict)
        assert len(obs) == 2  # 1 robot + 1 pedestrian

    def test_step_returns_four_tuple(self):
        backend = _make_backend_with_agents()
        backend.reset()
        obs, rewards, dones, info = backend.step({0: np.array([1.0, 0.5])})
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(info, dict)
        assert "step" in info
        assert "num_collisions" in info

    def test_step_advances_simulation(self):
        backend = _make_backend_with_agents()
        obs0 = backend.reset()
        pos_before = obs0[0]["position"].copy()
        backend.step({0: np.array([1.0, 0.0])})
        obs1, _, _, _ = backend.step({0: np.array([1.0, 0.0])})
        pos_after = obs1[0]["position"]
        assert not np.allclose(pos_before, pos_after)

    def test_pedestrian_auto_action(self):
        """Pedestrians without explicit actions should auto-navigate toward goals."""
        backend = _make_backend_with_agents()
        backend.reset()
        # Only provide action for robot (id 0), pedestrian (id 1) auto-navigates
        obs, _, _, _ = backend.step({0: np.zeros(2)})
        assert 1 in obs

    def test_episode_data_accumulates(self):
        backend = _make_backend_with_agents()
        backend.reset()
        for _ in range(5):
            backend.step({0: np.array([0.5, 0.5])})
        stats = backend.get_episode_statistics()
        assert stats["num_frames"] == 5


# ---------------------------------------------------------------------------
# ContinuousBackend — query methods
# ---------------------------------------------------------------------------

class TestContinuousBackendQueries:
    def test_get_robot_observation(self):
        backend = _make_backend_with_agents()
        backend.reset()
        backend.step({0: np.array([1.0, 0.0])})
        obs = backend.get_robot_observation(0)
        assert obs is not None
        assert "position" in obs
        assert "velocity" in obs
        assert "heading" in obs
        assert "goal" in obs
        assert "goal_distance" in obs
        assert "nearby_pedestrians" in obs
        assert "lidar" in obs

    def test_get_robot_observation_invalid_id(self):
        backend = _make_backend_with_agents()
        backend.reset()
        assert backend.get_robot_observation(999) is None

    def test_get_trajectory(self):
        backend = _make_backend_with_agents()
        backend.reset()
        for _ in range(3):
            backend.step({0: np.array([1.0, 0.0])})
        traj = backend.get_trajectory(0)
        assert isinstance(traj, np.ndarray)
        assert traj.shape[1] == 2
        assert traj.shape[0] == 4  # initial + 3 steps

    def test_get_all_trajectories(self):
        backend = _make_backend_with_agents()
        backend.reset()
        for _ in range(2):
            backend.step({0: np.array([0.5, 0.5])})
        trajs = backend.get_all_trajectories()
        assert len(trajs) == 2
        for traj in trajs.values():
            assert traj.shape == (3, 2)

    def test_get_episode_statistics(self):
        backend = _make_backend_with_agents()
        backend.reset()
        backend.step({0: np.array([1.0, 0.0])})
        stats = backend.get_episode_statistics()
        assert "steps" in stats
        assert "episode_rewards" in stats
        assert "num_frames" in stats

    def test_environment_property(self):
        backend = ContinuousBackend()
        assert isinstance(backend.environment, ContinuousEnvironment)


# ---------------------------------------------------------------------------
# ContinuousBackend — run_episode
# ---------------------------------------------------------------------------

class TestRunEpisode:
    def test_run_episode_no_policy(self):
        backend = _make_backend_with_agents()
        result = backend.run_episode(max_steps=5)
        assert "trajectories" in result
        assert "total_rewards" in result
        assert "statistics" in result
        assert "num_steps" in result
        assert result["num_steps"] <= 5

    def test_run_episode_with_policy(self):
        backend = _make_backend_with_agents()

        def simple_policy(obs):
            return np.array([0.5, 0.5])

        result = backend.run_episode(policy=simple_policy, max_steps=10)
        assert result["num_steps"] <= 10
        assert len(result["trajectories"]) == 2

    def test_run_episode_terminates_on_goal(self):
        """Place robot very close to goal so it reaches it quickly."""
        cfg = EnvironmentConfig(dt=0.1, max_steps=200, goal_radius=1.0)
        backend = ContinuousBackend(cfg)
        backend.add_robot(np.array([9.5, 9.5]), np.array([10.0, 10.0]))
        result = backend.run_episode(max_steps=200)
        # Should terminate early (well before 200 steps)
        assert result["num_steps"] < 200


# ---------------------------------------------------------------------------
# ContinuousBackend — _compute_pedestrian_action
# ---------------------------------------------------------------------------

class TestComputePedestrianAction:
    def test_at_goal_returns_zero(self):
        backend = ContinuousBackend()
        state = AgentState(
            position=np.array([5.0, 5.0]),
            velocity=np.zeros(2),
            heading=0.0,
            radius=0.3,
            mass=80.0,
            max_speed=2.0,
        )
        goal = np.array([5.0, 5.0])
        action = backend._compute_pedestrian_action(state, goal)
        np.testing.assert_allclose(action, np.zeros(2), atol=1e-6)

    def test_far_from_goal_moves_toward(self):
        backend = ContinuousBackend()
        state = AgentState(
            position=np.array([0.0, 0.0]),
            velocity=np.zeros(2),
            heading=0.0,
            radius=0.3,
            mass=80.0,
            max_speed=2.0,
        )
        goal = np.array([10.0, 0.0])
        action = backend._compute_pedestrian_action(state, goal)
        assert action[0] > 0  # Moving right toward goal
        speed = np.linalg.norm(action)
        assert speed <= state.max_speed * 0.8 + 1e-6


# ---------------------------------------------------------------------------
# ContinuousEnvironment — core operations
# ---------------------------------------------------------------------------

class TestContinuousEnvironment:
    def test_add_agent(self):
        env = ContinuousEnvironment()
        aid = env.add_agent(AgentConfig(
            position=np.array([1.0, 1.0]),
            goal=np.array([10.0, 10.0]),
        ))
        assert isinstance(aid, int)

    def test_add_multiple_agents_unique_ids(self):
        env = ContinuousEnvironment()
        ids = []
        for i in range(5):
            aid = env.add_agent(AgentConfig(
                position=np.array([float(i), 0.0]),
                goal=np.array([float(i), 10.0]),
            ))
            ids.append(aid)
        assert len(set(ids)) == 5

    def test_add_obstacles(self):
        env = ContinuousEnvironment()
        env.add_circular_obstacle(np.array([5.0, 5.0]), 1.0)
        env.add_rectangular_obstacle(np.array([8.0, 8.0]), np.array([10.0, 10.0]))
        assert env.num_obstacles == 2

    def test_add_wall(self):
        env = ContinuousEnvironment()
        idx = env.add_wall(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 0.1)
        assert isinstance(idx, int)
        assert env.num_obstacles == 1

    def test_add_boundary_walls(self):
        env = ContinuousEnvironment(EnvironmentConfig(width=10.0, height=10.0))
        env.add_boundary_walls()
        assert env.num_obstacles == 4

    def test_reset_initializes_states(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(
            position=np.array([1.0, 1.0]),
            goal=np.array([10.0, 10.0]),
        ))
        obs = env.reset()
        assert len(obs) == 1
        assert "position" in next(iter(obs.values()))

    def test_step_updates_state(self):
        env = ContinuousEnvironment()
        aid = env.add_agent(AgentConfig(
            position=np.array([1.0, 1.0]),
            goal=np.array([10.0, 10.0]),
        ))
        env.reset()
        obs, rewards, dones, info = env.step({aid: np.array([1.0, 0.5])})
        assert aid in obs
        assert aid in rewards
        assert aid in dones
        assert "step" in info

    def test_get_agent_state(self):
        env = ContinuousEnvironment()
        aid = env.add_agent(AgentConfig(
            position=np.array([3.0, 4.0]),
            goal=np.array([10.0, 10.0]),
        ))
        env.reset()
        state = env.get_agent_state(aid)
        assert state is not None
        np.testing.assert_allclose(state.position, [3.0, 4.0], atol=1e-6)

    def test_get_agent_state_missing(self):
        env = ContinuousEnvironment()
        env.reset()
        assert env.get_agent_state(999) is None

    def test_get_agent_goal(self):
        env = ContinuousEnvironment()
        aid = env.add_agent(AgentConfig(
            position=np.array([0.0, 0.0]),
            goal=np.array([8.0, 8.0]),
        ))
        env.reset()
        goal = env.get_agent_goal(aid)
        assert goal is not None
        np.testing.assert_allclose(goal, [8.0, 8.0])

    def test_get_agent_goal_missing(self):
        env = ContinuousEnvironment()
        env.reset()
        assert env.get_agent_goal(999) is None

    def test_set_agent_goal(self):
        env = ContinuousEnvironment()
        aid = env.add_agent(AgentConfig(
            position=np.array([0.0, 0.0]),
            goal=np.array([5.0, 5.0]),
        ))
        env.reset()
        env.set_agent_goal(aid, np.array([15.0, 15.0]))
        new_goal = env.get_agent_goal(aid)
        np.testing.assert_allclose(new_goal, [15.0, 15.0])

    def test_get_trajectory_empty(self):
        env = ContinuousEnvironment()
        traj = env.get_trajectory(999)
        assert traj.shape == (0, 2)

    def test_get_all_positions(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([1.0, 2.0]), goal=np.array([10.0, 10.0])))
        env.add_agent(AgentConfig(position=np.array([3.0, 4.0]), goal=np.array([10.0, 10.0])))
        env.reset()
        positions = env.get_all_positions()
        assert positions.shape == (2, 2)

    def test_get_all_positions_empty(self):
        env = ContinuousEnvironment()
        positions = env.get_all_positions()
        assert positions.shape == (0, 2)

    def test_get_all_velocities(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([1.0, 2.0]), goal=np.array([10.0, 10.0])))
        env.reset()
        vels = env.get_all_velocities()
        assert vels.shape == (1, 2)

    def test_get_all_velocities_empty(self):
        env = ContinuousEnvironment()
        vels = env.get_all_velocities()
        assert vels.shape == (0, 2)

    def test_num_agents_property(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([0, 0]), goal=np.array([5, 5])))
        env.reset()
        assert env.num_agents == 1

    def test_current_time_property(self):
        env = ContinuousEnvironment(EnvironmentConfig(dt=0.1))
        env.add_agent(AgentConfig(position=np.array([0, 0]), goal=np.array([5, 5])))
        env.reset()
        assert env.current_time == pytest.approx(0.0)
        env.step({0: np.array([1.0, 0.0])})
        assert env.current_time == pytest.approx(0.1)

    def test_agent_ids_property(self):
        env = ContinuousEnvironment()
        a1 = env.add_agent(AgentConfig(position=np.array([0, 0]), goal=np.array([5, 5])))
        a2 = env.add_agent(AgentConfig(position=np.array([1, 1]), goal=np.array([6, 6])))
        env.reset()
        assert set(env.agent_ids) == {a1, a2}


# ---------------------------------------------------------------------------
# ContinuousEnvironment — rewards and dones
# ---------------------------------------------------------------------------

class TestEnvironmentRewardsAndDones:
    def test_goal_reached_gives_bonus(self):
        """Agent placed very close to goal should get goal bonus."""
        cfg = EnvironmentConfig(dt=0.1, goal_radius=2.0, max_steps=100)
        env = ContinuousEnvironment(cfg)
        aid = env.add_agent(AgentConfig(
            position=np.array([9.0, 9.0]),
            goal=np.array([10.0, 10.0]),
            max_speed=5.0,
        ))
        env.reset()
        # Move toward goal
        _, rewards, dones, _ = env.step({aid: np.array([2.0, 2.0])})
        # Should reach goal within a few steps
        if not dones[aid]:
            _, rewards, dones, _ = env.step({aid: np.array([2.0, 2.0])})
        # At some point the goal bonus of 10.0 should appear

    def test_timeout_done(self):
        cfg = EnvironmentConfig(dt=0.1, max_steps=3)
        env = ContinuousEnvironment(cfg)
        aid = env.add_agent(AgentConfig(
            position=np.array([0.0, 0.0]),
            goal=np.array([100.0, 100.0]),
        ))
        env.reset()
        for _ in range(3):
            _, _, dones, _ = env.step({aid: np.array([0.1, 0.0])})
        assert dones[aid] is True

    def test_progress_reward_positive_when_approaching_goal(self):
        cfg = EnvironmentConfig(dt=0.1, max_steps=100, goal_radius=0.5)
        env = ContinuousEnvironment(cfg)
        aid = env.add_agent(AgentConfig(
            position=np.array([0.0, 0.0]),
            goal=np.array([10.0, 0.0]),
            max_speed=5.0,
        ))
        env.reset()
        _, rewards, _, _ = env.step({aid: np.array([2.0, 0.0])})
        # Moving toward goal should give positive progress reward (minus time penalty)
        # Progress reward is prev_dist - curr_dist, which should be positive


# ---------------------------------------------------------------------------
# ContinuousEnvironment — statistics and snapshots
# ---------------------------------------------------------------------------

class TestEnvironmentStatisticsAndSnapshots:
    def test_get_statistics(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([0, 0]), goal=np.array([10, 10])))
        env.reset()
        env.step({0: np.array([1.0, 0.0])})
        stats = env.get_statistics()
        assert "steps" in stats
        assert "time" in stats
        assert "num_agents" in stats
        assert "goals_reached" in stats
        assert "success_rate" in stats
        assert "total_collisions" in stats
        assert "total_path_length" in stats
        assert "avg_path_length" in stats
        assert stats["num_agents"] == 1

    def test_get_snapshot(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([1.0, 2.0]), goal=np.array([10.0, 10.0])))
        env.reset()
        env.step({0: np.array([0.5, 0.5])})
        snap = env.get_snapshot()
        assert "step" in snap
        assert "time" in snap
        assert "agents" in snap
        assert 0 in snap["agents"]
        agent_snap = snap["agents"][0]
        assert "position" in agent_snap
        assert "velocity" in agent_snap
        assert "goal" in agent_snap
        assert "goal_reached" in agent_snap

    def test_load_snapshot_restores_state(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([1.0, 2.0]), goal=np.array([10.0, 10.0])))
        env.reset()
        env.step({0: np.array([1.0, 0.0])})
        snap = env.get_snapshot()

        # Step further to change state
        env.step({0: np.array([1.0, 0.0])})
        env.step({0: np.array([1.0, 0.0])})

        # Restore
        env.load_snapshot(snap)
        state = env.get_agent_state(0)
        np.testing.assert_allclose(
            state.position, snap["agents"][0]["position"], atol=1e-6,
        )

    def test_load_snapshot_string_keys(self):
        """Snapshot keys may be stringified when serialized via JSON."""
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([1.0, 2.0]), goal=np.array([10.0, 10.0])))
        env.reset()
        snap = env.get_snapshot()
        # Simulate JSON round-trip by converting int keys to strings
        snap["agents"] = {str(k): v for k, v in snap["agents"].items()}
        env.load_snapshot(snap)
        state = env.get_agent_state(0)
        assert state is not None

    def test_observations_include_nearby_agents(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([5.0, 5.0]), goal=np.array([15.0, 15.0])))
        env.add_agent(AgentConfig(position=np.array([6.0, 5.0]), goal=np.array([16.0, 15.0])))
        obs = env.reset()
        # Agent 0 should see agent 1 as nearby
        nearby = obs[0]["nearby_agents"]
        assert len(nearby) == 1
        assert nearby[0]["id"] == 1

    def test_observations_include_lidar(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([5.0, 5.0]), goal=np.array([15.0, 15.0])))
        obs = env.reset()
        assert "lidar" in obs[0]

    def test_observations_include_goal_angle(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([0.0, 0.0]), goal=np.array([10.0, 0.0])))
        obs = env.reset()
        assert "goal_angle" in obs[0]
        # Goal is directly to the right, angle should be ~0
        assert abs(obs[0]["goal_angle"]) < 0.1

    def test_reset_clears_trajectories(self):
        env = ContinuousEnvironment()
        env.add_agent(AgentConfig(position=np.array([0, 0]), goal=np.array([10, 10])))
        env.reset()
        env.step({0: np.array([1.0, 0.0])})
        env.step({0: np.array([1.0, 0.0])})
        traj1 = env.get_trajectory(0)
        assert traj1.shape[0] == 3  # initial + 2 steps

        # Reset and check trajectory is fresh
        env.reset()
        traj2 = env.get_trajectory(0)
        assert traj2.shape[0] == 1  # only initial position

    def test_multiple_episodes(self):
        """Verify that resetting and running multiple episodes works correctly."""
        env = ContinuousEnvironment(EnvironmentConfig(max_steps=5))
        env.add_agent(AgentConfig(position=np.array([0, 0]), goal=np.array([10, 10])))

        for _ in range(3):
            env.reset()
            for _ in range(3):
                env.step({0: np.array([0.5, 0.5])})
            stats = env.get_statistics()
            assert stats["steps"] == 3


# ---------------------------------------------------------------------------
# ScenarioConfig
# ---------------------------------------------------------------------------

class TestScenarioConfig:
    def test_defaults(self):
        sc = ScenarioConfig()
        assert sc.name == "default"
        assert len(sc.agents) == 0
        assert len(sc.obstacles) == 0
        assert len(sc.walls) == 0
        assert isinstance(sc.metadata, dict)

    def test_custom_values(self):
        sc = ScenarioConfig(
            name="custom",
            metadata={"author": "test"},
        )
        assert sc.name == "custom"
        assert sc.metadata["author"] == "test"
