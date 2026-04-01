"""Continuous-space environment for multi-agent pedestrian simulation.

Provides the main simulation environment that manages agents,
obstacles, and physics in a continuous 2-D world.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navirl.backends.continuous.obstacles import (
    CircleObstacle,
    LineObstacle,
    Obstacle,
    ObstacleCollection,
    RectangleObstacle,
)
from navirl.backends.continuous.physics import AgentState, PhysicsConfig, PhysicsEngine


@dataclass
class AgentConfig:
    """Configuration for creating an agent.

    Parameters
    ----------
    position : np.ndarray
        Initial position, shape (2,).
    goal : np.ndarray
        Goal position, shape (2,).
    radius : float
        Agent radius.
    preferred_speed : float
        Preferred walking speed.
    max_speed : float
        Maximum speed.
    mass : float
        Agent mass.
    agent_type : str
        Agent type identifier (e.g., "robot", "pedestrian").
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 0.3
    preferred_speed: float = 1.2
    max_speed: float = 2.0
    mass: float = 80.0
    agent_type: str = "pedestrian"


@dataclass
class EnvironmentConfig:
    """Configuration for the continuous environment.

    Parameters
    ----------
    width : float
        World width in meters.
    height : float
        World height in meters.
    dt : float
        Simulation time step in seconds.
    max_steps : int
        Maximum simulation steps per episode.
    goal_radius : float
        Distance threshold for reaching goal.
    physics : PhysicsConfig
        Physics engine configuration.
    enable_boundaries : bool
        Whether to enforce world boundaries.
    """

    width: float = 20.0
    height: float = 20.0
    dt: float = 0.1
    max_steps: int = 500
    goal_radius: float = 0.5
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    enable_boundaries: bool = True


class ContinuousEnvironment:
    """Continuous 2-D simulation environment.

    Manages agents and obstacles in a continuous space with physics-based
    motion. Supports multi-agent simulation with mixed agent types.

    Parameters
    ----------
    config : EnvironmentConfig, optional
        Environment configuration.

    Examples
    --------
    >>> env = ContinuousEnvironment()
    >>> env.add_agent(AgentConfig(
    ...     position=np.array([1.0, 1.0]),
    ...     goal=np.array([10.0, 10.0]),
    ... ))
    >>> obs = env.reset()
    >>> action = np.array([1.0, 0.0])
    >>> obs, reward, done, info = env.step({0: action})
    """

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()

        # World boundaries
        bounds = None
        if self.config.enable_boundaries:
            bounds = (0.0, 0.0, self.config.width, self.config.height)

        self.obstacles = ObstacleCollection()
        self.physics = PhysicsEngine(
            config=self.config.physics,
            obstacles=self.obstacles,
            world_bounds=bounds,
        )

        self._agent_configs: dict[int, AgentConfig] = {}
        self._agent_states: dict[int, AgentState] = {}
        self._agent_goals: dict[int, np.ndarray] = {}
        self._next_agent_id = 0
        self._step_count = 0
        self._episode_count = 0

        # Tracking
        self._trajectories: dict[int, list[np.ndarray]] = {}
        self._rewards: dict[int, list[float]] = {}
        self._collisions: list[tuple[int, int, float]] = []
        self._goal_reached: dict[int, bool] = {}

    def add_agent(self, config: AgentConfig) -> int:
        """Add an agent to the environment.

        Parameters
        ----------
        config : AgentConfig
            Agent configuration.

        Returns
        -------
        int
            Agent ID.
        """
        agent_id = self._next_agent_id
        self._next_agent_id += 1

        self._agent_configs[agent_id] = config
        self._agent_goals[agent_id] = np.asarray(config.goal, dtype=np.float64)
        self._goal_reached[agent_id] = False

        return agent_id

    def add_obstacle(self, obstacle: Obstacle) -> int:
        """Add an obstacle to the environment.

        Parameters
        ----------
        obstacle : Obstacle
            Obstacle to add.

        Returns
        -------
        int
            Obstacle index.
        """
        return self.obstacles.add(obstacle)

    def add_wall(
        self,
        start: np.ndarray,
        end: np.ndarray,
        thickness: float = 0.1,
    ) -> int:
        """Add a wall (line obstacle).

        Parameters
        ----------
        start : np.ndarray
            Wall start point.
        end : np.ndarray
            Wall end point.
        thickness : float
            Wall thickness.

        Returns
        -------
        int
            Obstacle index.
        """
        return self.obstacles.add(LineObstacle(
            start=np.asarray(start),
            end=np.asarray(end),
            thickness=thickness,
        ))

    def add_circular_obstacle(
        self,
        center: np.ndarray,
        radius: float,
    ) -> int:
        """Add a circular obstacle.

        Parameters
        ----------
        center : np.ndarray
            Circle center.
        radius : float
            Circle radius.

        Returns
        -------
        int
            Obstacle index.
        """
        return self.obstacles.add(CircleObstacle(
            center=np.asarray(center),
            radius=radius,
        ))

    def add_rectangular_obstacle(
        self,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
    ) -> int:
        """Add a rectangular obstacle.

        Parameters
        ----------
        min_corner : np.ndarray
            Lower-left corner.
        max_corner : np.ndarray
            Upper-right corner.

        Returns
        -------
        int
            Obstacle index.
        """
        return self.obstacles.add(RectangleObstacle(
            min_corner=np.asarray(min_corner),
            max_corner=np.asarray(max_corner),
        ))

    def add_boundary_walls(self) -> None:
        """Add walls around the world boundaries."""
        w, h = self.config.width, self.config.height
        self.add_wall(np.array([0, 0]), np.array([w, 0]))
        self.add_wall(np.array([w, 0]), np.array([w, h]))
        self.add_wall(np.array([w, h]), np.array([0, h]))
        self.add_wall(np.array([0, h]), np.array([0, 0]))

    def reset(self) -> dict[int, dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        dict
            Initial observations for each agent.
        """
        self._step_count = 0
        self._episode_count += 1
        self._collisions.clear()

        # Initialize agent states
        self._agent_states.clear()
        self._trajectories.clear()
        self._rewards.clear()

        for agent_id, config in self._agent_configs.items():
            position = np.asarray(config.position, dtype=np.float64)
            goal = np.asarray(config.goal, dtype=np.float64)

            # Compute initial heading toward goal
            diff = goal - position
            heading = math.atan2(diff[1], diff[0]) if np.linalg.norm(diff) > 1e-6 else 0.0

            self._agent_states[agent_id] = AgentState(
                position=position.copy(),
                velocity=np.zeros(2),
                heading=heading,
                radius=config.radius,
                mass=config.mass,
                max_speed=config.max_speed,
            )
            self._agent_goals[agent_id] = goal.copy()
            self._goal_reached[agent_id] = False
            self._trajectories[agent_id] = [position.copy()]
            self._rewards[agent_id] = []

        return self._get_observations()

    def step(
        self,
        actions: dict[int, np.ndarray],
    ) -> tuple[
        dict[int, dict[str, Any]],
        dict[int, float],
        dict[int, bool],
        dict[str, Any],
    ]:
        """Take a simulation step.

        Parameters
        ----------
        actions : dict
            Desired velocity for each agent, shape (2,).

        Returns
        -------
        tuple
            (observations, rewards, dones, info)
        """
        self._step_count += 1

        # Update physics
        self._agent_states = self.physics.step(
            self._agent_states, actions, self.config.dt
        )

        # Record trajectories
        for agent_id, state in self._agent_states.items():
            self._trajectories[agent_id].append(state.position.copy())

        # Record collisions
        collision_pairs = self.physics.get_collision_pairs()
        for pair in collision_pairs:
            self._collisions.append((*pair, self._step_count * self.config.dt))

        # Compute rewards and dones
        rewards = self._compute_rewards()
        dones = self._compute_dones()

        # Store rewards
        for agent_id, reward in rewards.items():
            self._rewards[agent_id].append(reward)

        # Info
        info = {
            "step": self._step_count,
            "time": self._step_count * self.config.dt,
            "collision_pairs": collision_pairs,
            "num_collisions": len(collision_pairs),
        }

        observations = self._get_observations()
        return observations, rewards, dones, info

    def _get_observations(self) -> dict[int, dict[str, Any]]:
        """Compute observations for all agents."""
        observations = {}
        for agent_id, state in self._agent_states.items():
            goal = self._agent_goals[agent_id]
            goal_diff = goal - state.position
            goal_dist = float(np.linalg.norm(goal_diff))
            goal_angle = math.atan2(goal_diff[1], goal_diff[0]) if goal_dist > 1e-6 else 0.0

            # Find nearby agents
            nearby_agents = []
            for other_id, other_state in self._agent_states.items():
                if other_id == agent_id:
                    continue
                diff = other_state.position - state.position
                dist = float(np.linalg.norm(diff))
                if dist < 10.0:  # Observation radius
                    nearby_agents.append({
                        "id": other_id,
                        "relative_position": diff.copy(),
                        "velocity": other_state.velocity.copy(),
                        "distance": dist,
                        "radius": other_state.radius,
                    })

            # Sort by distance
            nearby_agents.sort(key=lambda x: x["distance"])

            # Lidar-like obstacle sensing
            lidar = self.obstacles.multi_ray_cast(
                state.position, num_rays=36, max_distance=10.0
            )

            observations[agent_id] = {
                "position": state.position.copy(),
                "velocity": state.velocity.copy(),
                "speed": state.speed,
                "heading": state.heading,
                "goal": goal.copy(),
                "goal_distance": goal_dist,
                "goal_angle": goal_angle,
                "nearby_agents": nearby_agents,
                "lidar": lidar.copy(),
                "radius": state.radius,
            }

        return observations

    def _compute_rewards(self) -> dict[int, float]:
        """Compute rewards for all agents."""
        rewards = {}
        for agent_id, state in self._agent_states.items():
            reward = 0.0

            # Distance to goal reward
            goal = self._agent_goals[agent_id]
            goal_dist = float(np.linalg.norm(goal - state.position))

            # Previous distance
            if len(self._trajectories[agent_id]) >= 2:
                prev_pos = self._trajectories[agent_id][-2]
                prev_dist = float(np.linalg.norm(goal - prev_pos))
                reward += (prev_dist - goal_dist)  # Progress reward

            # Goal reached bonus
            if goal_dist < self.config.goal_radius:
                reward += 10.0
                self._goal_reached[agent_id] = True

            # Collision penalty
            for pair in self.physics.get_collision_pairs():
                if agent_id in pair:
                    reward -= 0.5

            # Time penalty
            reward -= 0.01

            rewards[agent_id] = reward

        return rewards

    def _compute_dones(self) -> dict[int, bool]:
        """Compute done flags for all agents."""
        dones = {}
        timeout = self._step_count >= self.config.max_steps

        for agent_id in self._agent_states:
            dones[agent_id] = self._goal_reached[agent_id] or timeout

        return dones

    # -----------------------------------------------------------------------
    # Query methods
    # -----------------------------------------------------------------------

    def get_agent_state(self, agent_id: int) -> AgentState | None:
        """Get the current state of an agent."""
        return self._agent_states.get(agent_id)

    def get_agent_goal(self, agent_id: int) -> np.ndarray | None:
        """Get the goal position of an agent."""
        return self._agent_goals.get(agent_id)

    def set_agent_goal(self, agent_id: int, goal: np.ndarray) -> None:
        """Update an agent's goal position."""
        self._agent_goals[agent_id] = np.asarray(goal, dtype=np.float64)
        self._goal_reached[agent_id] = False

    def get_trajectory(self, agent_id: int) -> np.ndarray:
        """Get the trajectory of an agent.

        Returns
        -------
        np.ndarray
            Trajectory positions, shape (T, 2).
        """
        if agent_id not in self._trajectories:
            return np.empty((0, 2))
        return np.array(self._trajectories[agent_id])

    def get_all_positions(self) -> np.ndarray:
        """Get current positions of all agents.

        Returns
        -------
        np.ndarray
            Positions, shape (N, 2).
        """
        if not self._agent_states:
            return np.empty((0, 2))
        return np.array([s.position for s in self._agent_states.values()])

    def get_all_velocities(self) -> np.ndarray:
        """Get current velocities of all agents.

        Returns
        -------
        np.ndarray
            Velocities, shape (N, 2).
        """
        if not self._agent_states:
            return np.empty((0, 2))
        return np.array([s.velocity for s in self._agent_states.values()])

    @property
    def num_agents(self) -> int:
        """Number of agents."""
        return len(self._agent_states)

    @property
    def num_obstacles(self) -> int:
        """Number of obstacles."""
        return len(self.obstacles)

    @property
    def current_time(self) -> float:
        """Current simulation time in seconds."""
        return self._step_count * self.config.dt

    @property
    def agent_ids(self) -> list[int]:
        """List of active agent IDs."""
        return list(self._agent_states.keys())

    def get_statistics(self) -> dict[str, Any]:
        """Get episode statistics.

        Returns
        -------
        dict
            Statistics including collision count, goal success, etc.
        """
        total_agents = len(self._agent_configs)
        goals_reached = sum(1 for v in self._goal_reached.values() if v)

        total_path_length = 0.0
        for agent_id in self._trajectories:
            traj = np.array(self._trajectories[agent_id])
            if len(traj) >= 2:
                diffs = np.diff(traj, axis=0)
                total_path_length += float(np.sum(np.linalg.norm(diffs, axis=1)))

        return {
            "steps": self._step_count,
            "time": self.current_time,
            "num_agents": total_agents,
            "goals_reached": goals_reached,
            "success_rate": goals_reached / max(1, total_agents),
            "total_collisions": len(self._collisions),
            "total_path_length": total_path_length,
            "avg_path_length": total_path_length / max(1, total_agents),
        }

    def get_snapshot(self) -> dict[str, Any]:
        """Get a serializable snapshot of the current state.

        Returns
        -------
        dict
            Snapshot data.
        """
        return {
            "step": self._step_count,
            "time": self.current_time,
            "agents": {
                aid: {
                    "position": state.position.tolist(),
                    "velocity": state.velocity.tolist(),
                    "heading": state.heading,
                    "radius": state.radius,
                    "goal": self._agent_goals[aid].tolist(),
                    "goal_reached": self._goal_reached[aid],
                }
                for aid, state in self._agent_states.items()
            },
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot.

        Parameters
        ----------
        snapshot : dict
            Snapshot data from ``get_snapshot()``.
        """
        self._step_count = snapshot["step"]

        for aid_str, agent_data in snapshot["agents"].items():
            aid = int(aid_str) if isinstance(aid_str, str) else aid_str

            if aid in self._agent_states:
                state = self._agent_states[aid]
                state.position = np.array(agent_data["position"])
                state.velocity = np.array(agent_data["velocity"])
                state.heading = agent_data["heading"]
                self._agent_goals[aid] = np.array(agent_data["goal"])
                self._goal_reached[aid] = agent_data["goal_reached"]
