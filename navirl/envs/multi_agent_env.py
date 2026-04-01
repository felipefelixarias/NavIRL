"""Multi-robot Gymnasium environment with PettingZoo-compatible interface.

Supports cooperative and competitive multi-robot navigation with optional
inter-robot communication channels.  The environment follows the PettingZoo
*parallel* API pattern: :meth:`step` accepts a ``dict[str, action]`` and
returns per-agent observations, rewards, terminations, and truncations.

Exports
-------
MultiAgentNavEnv    -- multi-robot navigation environment
MultiAgentNavConfig -- extended configuration dataclass
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as _exc:
    raise ImportError(
        "NavIRL Gymnasium environments require the 'gymnasium' package.  "
        "Install it with:  pip install gymnasium"
    ) from _exc

from navirl.core.constants import (
    EPSILON,
    PROXEMICS,
    REWARD,
    SIM,
)
from navirl.core.env import SceneBackend
from navirl.envs.base_env import _DISCRETE_ACTIONS, NavEnvConfig, _build_backend

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MultiAgentNavConfig(NavEnvConfig):
    """Configuration for :class:`MultiAgentNavEnv`.

    Parameters
    ----------
    num_robots : int
        Number of simultaneously-controlled robots.
    communication_dim : int
        Dimension of the optional inter-robot communication vector.  Set to 0
        to disable communication.  When > 0 each robot's observation includes
        the messages broadcast by all other robots at the previous time-step.
    shared_reward : bool
        If *True* all robots receive the mean reward (cooperative); otherwise
        each robot receives its own individual reward.
    inter_robot_collision_penalty : float
        Extra penalty when two robots collide with each other.
    """

    num_robots: int = 2
    communication_dim: int = 0
    shared_reward: bool = False
    inter_robot_collision_penalty: float = -5.0


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------


class MultiAgentNavEnv:
    """Multi-robot social navigation environment.

    Follows the PettingZoo *parallel* API pattern so that multi-agent RL
    libraries (e.g., PettingZoo, EPyMARL, MARLlib) can use it directly.

    Each robot is identified by a string name ``"robot_0"``, ``"robot_1"``,
    etc.  The :pyattr:`observation_space` and :pyattr:`action_space`
    attributes are **per-agent** dicts keyed by agent name.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": SIM.default_fps}

    # -----------------------------------------------------------------
    #  Construction
    # -----------------------------------------------------------------

    def __init__(self, config: MultiAgentNavConfig | None = None, **kwargs: Any) -> None:
        self.config = config if config is not None else MultiAgentNavConfig(**kwargs)
        self.render_mode = self.config.render_mode

        self._backend: SceneBackend | None = None
        self._rng: np.random.Generator = np.random.default_rng()

        # Agent naming
        self.possible_agents: list[str] = [
            f"robot_{i}" for i in range(self.config.num_robots)
        ]
        self.agents: list[str] = list(self.possible_agents)

        # Internal id mapping: agent_name -> backend agent_id
        self._name_to_id: dict[str, int] = {}
        self._robot_goals: dict[str, np.ndarray] = {}
        self._human_ids: list[int] = []
        self._human_goals: dict[int, np.ndarray] = {}
        self._step_count: int = 0
        self._prev_dists: dict[str, float] = {}
        self._terminated: dict[str, bool] = {}

        # Communication buffers
        self._comm_buffer: dict[str, np.ndarray] = {}

        # Build per-agent spaces
        single_obs_space = self._make_single_observation_space()
        single_act_space = self._make_single_action_space()

        self.observation_spaces: dict[str, spaces.Space] = {
            name: single_obs_space for name in self.possible_agents
        }
        self.action_spaces: dict[str, spaces.Space] = {
            name: single_act_space for name in self.possible_agents
        }

        # Convenience: singular forms expected by some frameworks
        self.observation_space = single_obs_space
        self.action_space = single_act_space

    # -----------------------------------------------------------------
    #  Space builders (per-agent)
    # -----------------------------------------------------------------

    def _make_single_observation_space(self) -> spaces.Space:
        cfg = self.config
        # Robot self-state: [rx, ry, rvx, rvy, gx, gy, dist_to_goal]
        robot_dim = 7
        # Other robots: [dx, dy, dvx, dvy] per other robot
        other_robots_dim = 4 * (cfg.num_robots - 1)
        # Humans (capped)
        human_dim = 5 * cfg.max_observable_humans
        # Communication
        comm_dim = cfg.communication_dim * (cfg.num_robots - 1)

        total_dim = robot_dim + other_robots_dim + human_dim + comm_dim
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _make_single_action_space(self) -> spaces.Space:
        cfg = self.config
        if cfg.action_type in ("continuous", "holonomic"):
            act_space: spaces.Space = spaces.Box(
                low=-cfg.robot_max_speed,
                high=cfg.robot_max_speed,
                shape=(2,),
                dtype=np.float32,
            )
        elif cfg.action_type == "discrete":
            act_space = spaces.Discrete(len(_DISCRETE_ACTIONS))
        else:
            raise ValueError(f"Unknown action_type: {cfg.action_type!r}")

        # If communication is enabled, wrap as a Dict space
        if cfg.communication_dim > 0:
            return spaces.Dict(
                {
                    "action": act_space,
                    "message": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(cfg.communication_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        return act_space

    # -----------------------------------------------------------------
    #  PettingZoo parallel API
    # -----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        self._rng = np.random.default_rng(seed)
        self._backend = _build_backend(self.config)
        self._step_count = 0
        self._human_ids = []
        self._human_goals = {}
        self.agents = list(self.possible_agents)
        self._terminated = {name: False for name in self.agents}

        # Init communication buffers to zeros
        if self.config.communication_dim > 0:
            for name in self.possible_agents:
                self._comm_buffer[name] = np.zeros(
                    self.config.communication_dim, dtype=np.float32
                )

        # Place robots
        self._name_to_id = {}
        self._robot_goals = {}
        self._prev_dists = {}
        for i, name in enumerate(self.possible_agents):
            agent_id = i  # robots get ids 0..num_robots-1
            pos = self._sample_free_point()
            self._backend.add_agent(
                agent_id,
                tuple(pos),
                self.config.robot_radius,
                self.config.robot_max_speed,
                kind="robot",
            )
            self._name_to_id[name] = agent_id

            # Goal
            goal = self._sample_free_point()
            while np.linalg.norm(goal - pos) < SIM.goal_radius * 2:
                goal = self._sample_free_point()
            self._robot_goals[name] = goal
            self._prev_dists[name] = float(np.linalg.norm(pos - goal))

        # Place humans
        self._spawn_humans(self.config.num_humans)

        observations = {name: self._observe(name) for name in self.agents}
        infos = {name: self._make_info(name) for name in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, Any],       # observations
        dict[str, float],     # rewards
        dict[str, bool],      # terminated
        dict[str, bool],      # truncated
        dict[str, dict[str, Any]],  # infos
    ]:
        assert self._backend is not None, "Call reset() before step()."
        cfg = self.config

        # --- Apply robot actions ---
        for name in self.agents:
            if self._terminated[name]:
                continue
            raw_action = actions.get(name)
            if raw_action is None:
                continue

            # Unpack communication if present
            if cfg.communication_dim > 0 and isinstance(raw_action, dict):
                velocity = self._action_to_velocity(raw_action["action"])
                self._comm_buffer[name] = np.asarray(
                    raw_action["message"], dtype=np.float32
                ).flatten()[: cfg.communication_dim]
            else:
                velocity = self._action_to_velocity(raw_action)

            agent_id = self._name_to_id[name]
            self._backend.set_preferred_velocity(agent_id, tuple(velocity))

        # --- Drive humans ---
        self._step_humans()

        # --- Physics ---
        self._backend.step()
        self._step_count += 1

        # --- Collect per-agent results ---
        observations: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        is_truncated = self._step_count >= cfg.max_steps

        for name in self.agents:
            if self._terminated[name]:
                # Already done – provide dummy outputs
                observations[name] = np.zeros(
                    self.observation_spaces[name].shape, dtype=np.float32  # type: ignore[union-attr]
                )
                rewards[name] = 0.0
                terminated[name] = True
                truncated[name] = False
                infos[name] = {}
                continue

            r, term, info = self._compute_reward(name)
            observations[name] = self._observe(name)
            rewards[name] = r
            terminated[name] = term
            truncated[name] = is_truncated and not term
            if truncated[name]:
                rewards[name] += REWARD.penalty_timeout
            infos[name] = info
            infos[name].update(self._make_info(name))

            if term:
                self._terminated[name] = True

        # --- Shared reward mode ---
        if cfg.shared_reward:
            mean_r = sum(rewards.values()) / max(len(rewards), 1)
            for name in rewards:
                rewards[name] = mean_r

        # Remove fully-terminated agents from active list
        self.agents = [n for n in self.agents if not (terminated[n] or truncated[n])]

        return observations, rewards, terminated, truncated, infos

    def render(self) -> np.ndarray | None:
        if self._backend is None:
            return None
        img = self._backend.map_image()
        if img is None:
            return None
        img = np.asarray(img, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        try:
            import cv2

            for name, aid in self._name_to_id.items():
                pos = self._backend.get_position(aid)
                px = self._backend.world_to_map(pos)
                cv2.circle(img, (px[1], px[0]), 5, (31, 119, 180), -1)
                goal = self._robot_goals[name]
                gx_px = self._backend.world_to_map((float(goal[0]), float(goal[1])))
                cv2.circle(img, (gx_px[1], gx_px[0]), 4, (214, 39, 40), -1)
            for hid in self._human_ids:
                hp = self._backend.get_position(hid)
                hp_px = self._backend.world_to_map(hp)
                cv2.circle(img, (hp_px[1], hp_px[0]), 4, (255, 127, 14), -1)
        except ImportError:
            pass

        if self.render_mode == "human":
            try:
                import cv2 as _cv2

                _cv2.imshow("MultiAgentNavEnv", img)
                _cv2.waitKey(1)
            except ImportError:
                pass
        return img

    def close(self) -> None:
        self._backend = None
        try:
            import cv2

            cv2.destroyAllWindows()
        except ImportError:
            pass

    # -----------------------------------------------------------------
    #  Per-agent observation
    # -----------------------------------------------------------------

    def _observe(self, name: str) -> np.ndarray:
        assert self._backend is not None
        cfg = self.config
        agent_id = self._name_to_id[name]

        # Self-state
        rx, ry = self._backend.get_position(agent_id)
        rvx, rvy = self._backend.get_velocity(agent_id)
        goal = self._robot_goals[name]
        gx, gy = float(goal[0]), float(goal[1])
        dist = math.hypot(rx - gx, ry - gy)
        self_state = np.array([rx, ry, rvx, rvy, gx, gy, dist], dtype=np.float32)

        # Other robots (relative)
        other_robots_parts: list[np.ndarray] = []
        for other_name in self.possible_agents:
            if other_name == name:
                continue
            oid = self._name_to_id[other_name]
            ox, oy = self._backend.get_position(oid)
            ovx, ovy = self._backend.get_velocity(oid)
            other_robots_parts.append(
                np.array([ox - rx, oy - ry, ovx - rvx, ovy - rvy], dtype=np.float32)
            )
        other_robots_vec = (
            np.concatenate(other_robots_parts) if other_robots_parts else np.array([], dtype=np.float32)
        )

        # Humans (relative, sorted by distance, capped)
        human_entries: list[tuple[float, np.ndarray]] = []
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            hvx, hvy = self._backend.get_velocity(hid)
            dx, dy = hx - rx, hy - ry
            d = math.hypot(dx, dy)
            dvx, dvy = hvx - rvx, hvy - rvy
            h_radius = cfg.human_radius_range[0]
            human_entries.append((d, np.array([dx, dy, dvx, dvy, h_radius], dtype=np.float32)))
        human_entries.sort(key=lambda t: t[0])

        human_vec = np.zeros(5 * cfg.max_observable_humans, dtype=np.float32)
        for i, (_, vec) in enumerate(human_entries[: cfg.max_observable_humans]):
            human_vec[i * 5 : i * 5 + 5] = vec

        parts = [self_state, other_robots_vec, human_vec]

        # Communication messages from other robots
        if cfg.communication_dim > 0:
            comm_parts: list[np.ndarray] = []
            for other_name in self.possible_agents:
                if other_name == name:
                    continue
                comm_parts.append(self._comm_buffer.get(
                    other_name, np.zeros(cfg.communication_dim, dtype=np.float32)
                ))
            parts.append(np.concatenate(comm_parts) if comm_parts else np.array([], dtype=np.float32))

        return np.concatenate(parts)

    # -----------------------------------------------------------------
    #  Per-agent reward
    # -----------------------------------------------------------------

    def _compute_reward(self, name: str) -> tuple[float, bool, dict[str, Any]]:
        assert self._backend is not None
        cfg = self.config
        agent_id = self._name_to_id[name]

        rx, ry = self._backend.get_position(agent_id)
        robot_pos = np.array([rx, ry], dtype=np.float32)
        goal = self._robot_goals[name]
        dist_to_goal = float(np.linalg.norm(robot_pos - goal))

        info: dict[str, Any] = {"dist_to_goal": dist_to_goal}
        reward = 0.0
        terminated = False

        # Goal reached
        if dist_to_goal < SIM.goal_radius:
            reward += REWARD.reward_goal_reached
            terminated = True
            info["event"] = "goal_reached"
            self._prev_dists[name] = dist_to_goal
            return reward, terminated, info

        # Obstacle collision
        if self._backend.check_obstacle_collision((rx, ry), cfg.robot_radius):
            reward += REWARD.penalty_collision_wall
            info["collision_wall"] = True

        # Human collisions / proxemics
        min_human_dist = float("inf")
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            d = math.hypot(rx - hx, ry - hy)
            min_human_dist = min(min_human_dist, d)
            combined = cfg.robot_radius + cfg.human_radius_range[0]
            if d < combined:
                reward += REWARD.penalty_collision_pedestrian
                info["collision_pedestrian"] = True
            elif d < PROXEMICS.intimate.outer + cfg.robot_radius:
                reward += REWARD.penalty_intimate_zone
        info["min_human_dist"] = min_human_dist

        # Inter-robot collision
        for other_name, oid in self._name_to_id.items():
            if other_name == name:
                continue
            ox, oy = self._backend.get_position(oid)
            d = math.hypot(rx - ox, ry - oy)
            if d < 2 * cfg.robot_radius:
                reward += cfg.inter_robot_collision_penalty
                info["collision_robot"] = True

        # Dense progress
        if cfg.reward_type == "dense":
            progress = self._prev_dists[name] - dist_to_goal
            reward += REWARD.reward_goal_progress * progress
            reward += REWARD.penalty_per_step

        self._prev_dists[name] = dist_to_goal
        return reward, terminated, info

    # -----------------------------------------------------------------
    #  Action conversion
    # -----------------------------------------------------------------

    def _action_to_velocity(self, action: np.ndarray | int) -> np.ndarray:
        cfg = self.config
        if cfg.action_type == "discrete":
            idx = int(action)
            return (_DISCRETE_ACTIONS[idx] * cfg.robot_max_speed).astype(np.float32)
        vel = np.asarray(action, dtype=np.float32).flatten()[:2]
        speed = float(np.linalg.norm(vel))
        if speed > cfg.robot_max_speed:
            vel = vel / speed * cfg.robot_max_speed
        return vel

    # -----------------------------------------------------------------
    #  Human management
    # -----------------------------------------------------------------

    def _spawn_humans(self, n: int) -> None:
        assert self._backend is not None
        cfg = self.config
        next_id = cfg.num_robots  # humans start after robot ids
        for i in range(n):
            hid = next_id + i
            radius = float(self._rng.uniform(*cfg.human_radius_range))
            speed = float(self._rng.uniform(*cfg.human_speed_range))
            pos = self._sample_free_point()
            self._backend.add_agent(hid, tuple(pos), radius, speed, kind="human")
            goal = self._sample_free_point()
            self._human_ids.append(hid)
            self._human_goals[hid] = goal

    def _step_humans(self) -> None:
        assert self._backend is not None
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            gx, gy = float(self._human_goals[hid][0]), float(self._human_goals[hid][1])
            dx, dy = gx - hx, gy - hy
            dist = math.hypot(dx, dy)
            if dist < SIM.goal_radius:
                self._human_goals[hid] = self._sample_free_point()
                gx, gy = float(self._human_goals[hid][0]), float(self._human_goals[hid][1])
                dx, dy = gx - hx, gy - hy
                dist = math.hypot(dx, dy)
            if dist > EPSILON:
                speed = self.config.human_speed_range[0]
                vx = dx / dist * speed
                vy = dy / dist * speed
            else:
                vx, vy = 0.0, 0.0
            self._backend.set_preferred_velocity(hid, (vx, vy))

    # -----------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------

    def _sample_free_point(self) -> np.ndarray:
        assert self._backend is not None
        pt = self._backend.sample_free_point()
        return np.array(pt, dtype=np.float32)

    def _make_info(self, name: str) -> dict[str, Any]:
        assert self._backend is not None
        agent_id = self._name_to_id[name]
        rx, ry = self._backend.get_position(agent_id)
        goal = self._robot_goals[name]
        return {
            "step": self._step_count,
            "time_s": self._step_count * self.config.dt,
            "robot_position": (rx, ry),
            "robot_velocity": self._backend.get_velocity(agent_id),
            "goal": (float(goal[0]), float(goal[1])),
        }

    @property
    def backend(self) -> SceneBackend | None:
        return self._backend


__all__ = ["MultiAgentNavEnv", "MultiAgentNavConfig"]
