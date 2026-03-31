"""ContinuousBackend - main interface for the continuous-space simulation.

Provides a high-level API that wraps the environment, physics, and
obstacle systems into a unified backend interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navirl.backends.continuous.environment import (
    AgentConfig,
    ContinuousEnvironment,
    EnvironmentConfig,
)
from navirl.backends.continuous.physics import AgentState


@dataclass
class ScenarioConfig:
    """Configuration for a simulation scenario.

    Parameters
    ----------
    name : str
        Scenario name.
    env_config : EnvironmentConfig
        Environment configuration.
    agents : list of AgentConfig
        Agent configurations.
    obstacles : list of dict
        Obstacle specifications (type + params).
    walls : list of dict
        Wall specifications (start, end, thickness).
    metadata : dict
        Additional scenario metadata.
    """

    name: str = "default"
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agents: list[AgentConfig] = field(default_factory=list)
    obstacles: list[dict[str, Any]] = field(default_factory=list)
    walls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContinuousBackend:
    """High-level continuous-space simulation backend.

    Provides a clean API for setting up and running continuous 2-D
    pedestrian simulations with physics-based motion.

    Parameters
    ----------
    config : EnvironmentConfig, optional
        Environment configuration.

    Examples
    --------
    >>> backend = ContinuousBackend()
    >>> backend.add_robot(np.array([2, 2]), np.array([18, 18]))
    >>> backend.add_pedestrian(np.array([10, 2]), np.array([10, 18]))
    >>> obs = backend.reset()
    >>> actions = {0: np.array([1.0, 0.5])}
    >>> obs, rewards, dones, info = backend.step(actions)
    """

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self._config = config or EnvironmentConfig()
        self._env = ContinuousEnvironment(self._config)
        self._robot_ids: list[int] = []
        self._pedestrian_ids: list[int] = []
        self._is_reset = False
        self._episode_rewards: dict[int, float] = {}
        self._episode_data: list[dict[str, Any]] = []

    @classmethod
    def from_scenario(cls, scenario: ScenarioConfig) -> ContinuousBackend:
        """Create a backend from a scenario configuration.

        Parameters
        ----------
        scenario : ScenarioConfig
            Scenario configuration.

        Returns
        -------
        ContinuousBackend
            Configured backend.
        """
        backend = cls(scenario.env_config)

        # Add agents
        for agent_config in scenario.agents:
            if agent_config.agent_type == "robot":
                backend.add_robot(
                    agent_config.position,
                    agent_config.goal,
                    radius=agent_config.radius,
                    max_speed=agent_config.max_speed,
                )
            else:
                backend.add_pedestrian(
                    agent_config.position,
                    agent_config.goal,
                    radius=agent_config.radius,
                    preferred_speed=agent_config.preferred_speed,
                )

        # Add obstacles
        for obs_spec in scenario.obstacles:
            obs_type = obs_spec.get("type", "circle")
            if obs_type == "circle":
                backend._env.add_circular_obstacle(
                    np.array(obs_spec["center"]),
                    obs_spec["radius"],
                )
            elif obs_type == "rectangle":
                backend._env.add_rectangular_obstacle(
                    np.array(obs_spec["min_corner"]),
                    np.array(obs_spec["max_corner"]),
                )

        # Add walls
        for wall_spec in scenario.walls:
            backend._env.add_wall(
                np.array(wall_spec["start"]),
                np.array(wall_spec["end"]),
                wall_spec.get("thickness", 0.1),
            )

        return backend

    def add_robot(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        radius: float = 0.3,
        max_speed: float = 1.5,
        mass: float = 50.0,
    ) -> int:
        """Add a robot agent.

        Parameters
        ----------
        position : np.ndarray
            Start position, shape (2,).
        goal : np.ndarray
            Goal position, shape (2,).
        radius : float
            Robot radius.
        max_speed : float
            Maximum speed.
        mass : float
            Robot mass.

        Returns
        -------
        int
            Robot agent ID.
        """
        config = AgentConfig(
            position=np.asarray(position),
            goal=np.asarray(goal),
            radius=radius,
            preferred_speed=max_speed * 0.8,
            max_speed=max_speed,
            mass=mass,
            agent_type="robot",
        )
        agent_id = self._env.add_agent(config)
        self._robot_ids.append(agent_id)
        return agent_id

    def add_pedestrian(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        radius: float = 0.3,
        preferred_speed: float = 1.2,
        mass: float = 80.0,
    ) -> int:
        """Add a pedestrian agent.

        Parameters
        ----------
        position : np.ndarray
            Start position, shape (2,).
        goal : np.ndarray
            Goal position, shape (2,).
        radius : float
            Pedestrian radius.
        preferred_speed : float
            Preferred walking speed.
        mass : float
            Pedestrian mass.

        Returns
        -------
        int
            Pedestrian agent ID.
        """
        config = AgentConfig(
            position=np.asarray(position),
            goal=np.asarray(goal),
            radius=radius,
            preferred_speed=preferred_speed,
            max_speed=preferred_speed * 1.5,
            mass=mass,
            agent_type="pedestrian",
        )
        agent_id = self._env.add_agent(config)
        self._pedestrian_ids.append(agent_id)
        return agent_id

    def add_pedestrian_circle(
        self,
        center: np.ndarray,
        radius: float,
        num_pedestrians: int,
        preferred_speed: float = 1.2,
    ) -> list[int]:
        """Add pedestrians in a circle, each heading to the opposite side.

        Parameters
        ----------
        center : np.ndarray
            Circle center.
        radius : float
            Circle radius.
        num_pedestrians : int
            Number of pedestrians.
        preferred_speed : float
            Preferred speed.

        Returns
        -------
        list of int
            Pedestrian IDs.
        """
        ids = []
        for i in range(num_pedestrians):
            angle = 2 * math.pi * i / num_pedestrians
            pos = np.array(
                [
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                ]
            )
            # Goal is diametrically opposite
            goal = np.array(
                [
                    center[0] - radius * math.cos(angle),
                    center[1] - radius * math.sin(angle),
                ]
            )
            aid = self.add_pedestrian(pos, goal, preferred_speed=preferred_speed)
            ids.append(aid)
        return ids

    def add_pedestrian_flow(
        self,
        start_region: tuple[np.ndarray, np.ndarray],
        goal_region: tuple[np.ndarray, np.ndarray],
        num_pedestrians: int,
        preferred_speed: float = 1.2,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Add pedestrians with random positions in a flow pattern.

        Parameters
        ----------
        start_region : tuple
            (min_corner, max_corner) for spawn area.
        goal_region : tuple
            (min_corner, max_corner) for goal area.
        num_pedestrians : int
            Number of pedestrians.
        preferred_speed : float
            Base preferred speed (varied +/- 20%).
        rng : np.random.Generator, optional
            Random generator.

        Returns
        -------
        list of int
            Pedestrian IDs.
        """
        if rng is None:
            rng = np.random.default_rng()

        start_min, start_max = np.asarray(start_region[0]), np.asarray(start_region[1])
        goal_min, goal_max = np.asarray(goal_region[0]), np.asarray(goal_region[1])

        ids = []
        for _ in range(num_pedestrians):
            pos = rng.uniform(start_min, start_max)
            goal = rng.uniform(goal_min, goal_max)
            speed = preferred_speed * rng.uniform(0.8, 1.2)
            aid = self.add_pedestrian(pos, goal, preferred_speed=speed)
            ids.append(aid)
        return ids

    def reset(self) -> dict[int, dict[str, Any]]:
        """Reset the simulation.

        Returns
        -------
        dict
            Initial observations.
        """
        self._is_reset = True
        self._episode_rewards = {aid: 0.0 for aid in self._robot_ids + self._pedestrian_ids}
        self._episode_data.clear()
        return self._env.reset()

    def step(
        self,
        actions: dict[int, np.ndarray],
    ) -> tuple[dict, dict[int, float], dict[int, bool], dict[str, Any]]:
        """Step the simulation.

        Parameters
        ----------
        actions : dict
            Actions for controlled agents (typically robots).

        Returns
        -------
        tuple
            (observations, rewards, dones, info)
        """
        if not self._is_reset:
            raise RuntimeError("Must call reset() before step()")

        # For pedestrians without explicit actions, compute simple goal-seeking
        full_actions = dict(actions)
        for pid in self._pedestrian_ids:
            if pid not in full_actions:
                state = self._env.get_agent_state(pid)
                goal = self._env.get_agent_goal(pid)
                if state is not None and goal is not None:
                    full_actions[pid] = self._compute_pedestrian_action(state, goal)

        obs, rewards, dones, info = self._env.step(full_actions)

        # Accumulate rewards
        for aid, r in rewards.items():
            if aid in self._episode_rewards:
                self._episode_rewards[aid] += r

        # Record frame data
        self._episode_data.append(
            {
                "step": info["step"],
                "positions": self._env.get_all_positions().tolist(),
                "collisions": info["num_collisions"],
            }
        )

        return obs, rewards, dones, info

    def _compute_pedestrian_action(
        self,
        state: AgentState,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Compute a simple goal-seeking action for a pedestrian.

        Uses a proportional controller with speed clamping.
        """
        diff = goal - state.position
        dist = float(np.linalg.norm(diff))
        if dist < 0.1:
            return np.zeros(2)

        direction = diff / dist
        speed = min(state.max_speed * 0.8, dist)
        return direction * speed

    # -----------------------------------------------------------------------
    # Query methods
    # -----------------------------------------------------------------------

    @property
    def robot_ids(self) -> list[int]:
        """Robot agent IDs."""
        return list(self._robot_ids)

    @property
    def pedestrian_ids(self) -> list[int]:
        """Pedestrian agent IDs."""
        return list(self._pedestrian_ids)

    @property
    def num_robots(self) -> int:
        """Number of robots."""
        return len(self._robot_ids)

    @property
    def num_pedestrians(self) -> int:
        """Number of pedestrians."""
        return len(self._pedestrian_ids)

    @property
    def environment(self) -> ContinuousEnvironment:
        """Underlying environment."""
        return self._env

    @property
    def dt(self) -> float:
        """Simulation time step."""
        return self._config.dt

    def get_robot_observation(self, robot_id: int) -> dict[str, Any] | None:
        """Get observation for a specific robot.

        Parameters
        ----------
        robot_id : int
            Robot ID.

        Returns
        -------
        dict or None
            Robot observation or None if invalid ID.
        """
        state = self._env.get_agent_state(robot_id)
        if state is None:
            return None

        goal = self._env.get_agent_goal(robot_id)
        goal_diff = goal - state.position if goal is not None else np.zeros(2)
        goal_dist = float(np.linalg.norm(goal_diff))

        # Nearby pedestrians
        nearby = []
        for pid in self._pedestrian_ids:
            ped_state = self._env.get_agent_state(pid)
            if ped_state is None:
                continue
            diff = ped_state.position - state.position
            dist = float(np.linalg.norm(diff))
            if dist < 10.0:
                nearby.append(
                    {
                        "id": pid,
                        "position": ped_state.position.copy(),
                        "velocity": ped_state.velocity.copy(),
                        "distance": dist,
                        "radius": ped_state.radius,
                    }
                )

        nearby.sort(key=lambda x: x["distance"])

        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "heading": state.heading,
            "speed": state.speed,
            "goal": goal.copy() if goal is not None else np.zeros(2),
            "goal_distance": goal_dist,
            "nearby_pedestrians": nearby,
            "lidar": self._env.obstacles.multi_ray_cast(state.position),
        }

    def get_episode_statistics(self) -> dict[str, Any]:
        """Get statistics for the current episode.

        Returns
        -------
        dict
            Episode statistics.
        """
        stats = self._env.get_statistics()
        stats["episode_rewards"] = dict(self._episode_rewards)
        stats["num_frames"] = len(self._episode_data)
        return stats

    def get_trajectory(self, agent_id: int) -> np.ndarray:
        """Get trajectory for an agent.

        Parameters
        ----------
        agent_id : int
            Agent ID.

        Returns
        -------
        np.ndarray
            Trajectory, shape (T, 2).
        """
        return self._env.get_trajectory(agent_id)

    def get_all_trajectories(self) -> dict[int, np.ndarray]:
        """Get trajectories for all agents.

        Returns
        -------
        dict
            Mapping from agent ID to trajectory array.
        """
        return {
            aid: self._env.get_trajectory(aid) for aid in self._robot_ids + self._pedestrian_ids
        }

    def run_episode(
        self,
        policy: Any = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run a complete episode with an optional policy.

        Parameters
        ----------
        policy : callable, optional
            Policy function: obs -> action. If None, robots do nothing.
        max_steps : int, optional
            Override max steps.

        Returns
        -------
        dict
            Episode results including trajectories, rewards, statistics.
        """
        obs = self.reset()
        total_rewards = {aid: 0.0 for aid in self._robot_ids}
        steps = max_steps or self._config.max_steps

        for step in range(steps):
            actions = {}
            if policy is not None:
                for rid in self._robot_ids:
                    if rid in obs:
                        actions[rid] = policy(obs[rid])
            else:
                for rid in self._robot_ids:
                    actions[rid] = np.zeros(2)

            obs, rewards, dones, info = self.step(actions)

            for rid in self._robot_ids:
                total_rewards[rid] = total_rewards.get(rid, 0.0) + rewards.get(rid, 0.0)

            if all(dones.values()):
                break

        return {
            "trajectories": self.get_all_trajectories(),
            "total_rewards": total_rewards,
            "statistics": self.get_episode_statistics(),
            "num_steps": step + 1,
        }
