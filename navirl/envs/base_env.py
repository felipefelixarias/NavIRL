"""Base Gymnasium environment for single-robot social navigation.

Wraps the NavIRL simulation backend (Grid2D with ORCA) as a
:class:`gymnasium.Env` so that standard RL libraries (Stable-Baselines3,
CleanRL, RLlib, etc.) can train navigation policies out of the box.

Exports
-------
NavEnv          -- single-robot Gymnasium environment
NavEnvConfig    -- dataclass holding every knob the environment exposes
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    LIDAR,
    PROXEMICS,
    REWARD,
    ROBOT_MAX_SPEED,
    ROBOT_RADIUS,
    SIM,
)
from navirl.core.env import SceneBackend

# ---------------------------------------------------------------------------
#  Default discrete-action table  (stop / +x / -x / +y / -y)
# ---------------------------------------------------------------------------

_DISCRETE_ACTIONS: np.ndarray = np.array(
    [
        [0.0, 0.0],  # 0 – stop
        [1.0, 0.0],  # 1 – forward  (+x)
        [-1.0, 0.0],  # 2 – backward (-x)
        [0.0, 1.0],  # 3 – left     (+y)
        [0.0, -1.0],  # 4 – right    (-y)
    ],
    dtype=np.float32,
)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NavEnvConfig:
    """Full configuration for :class:`NavEnv`.

    Parameters
    ----------
    scenario_path : str or None
        Path to a YAML/JSON scenario file.  If *None*, the environment is
        configured entirely via the inline parameters below.
    map_name : str
        Name / path of the occupancy-grid map when *scenario_path* is None.
    num_humans : int
        Number of ORCA-controlled human agents to spawn.
    observation_type : ``"state"`` | ``"lidar"`` | ``"combined"``
        Which observation representation to expose.
    action_type : ``"continuous"`` | ``"discrete"`` | ``"holonomic"``
        Action parameterisation.  ``"continuous"`` and ``"holonomic"`` both
        expose a ``Box(2,)`` for (vx, vy); ``"discrete"`` uses ``Discrete(5)``.
    reward_type : ``"sparse"`` | ``"dense"`` | ``"custom"``
        Built-in reward shaping strategy.
    max_steps : int
        Truncation horizon.
    dt : float
        Physics time-step (seconds).
    render_mode : str or None
        ``"human"`` for interactive window, ``"rgb_array"`` for off-screen.
    world_width, world_height : float
        World extents (metres) when no scenario file is used.
    robot_radius, robot_max_speed : float
        Robot body parameters.
    human_radius_range, human_speed_range : tuple
        Ranges from which human body and speed are sampled.
    lidar_num_beams : int
        Number of LiDAR rays (only relevant when observation includes lidar).
    lidar_max_range : float
        Maximum LiDAR sensing distance (metres).
    max_observable_humans : int
        Cap on the number of humans included in the state observation vector.
    custom_reward_fn : callable or None
        ``(env, action, info) -> float`` used when *reward_type* = ``"custom"``.
    """

    # Scenario source
    scenario_path: str | None = None
    map_name: str = "empty_30x30"
    num_humans: int = 5

    # Spaces
    observation_type: Literal["state", "lidar", "combined"] = "state"
    action_type: Literal["continuous", "discrete", "holonomic"] = "continuous"
    reward_type: Literal["sparse", "dense", "custom"] = "dense"

    # Horizon
    max_steps: int = SIM.max_steps
    dt: float = SIM.dt

    # Rendering
    render_mode: str | None = None

    # World geometry (used only when scenario_path is None)
    world_width: float = SIM.default_world_width
    world_height: float = SIM.default_world_height

    # Robot body
    robot_radius: float = ROBOT_RADIUS
    robot_max_speed: float = ROBOT_MAX_SPEED

    # Human body (ranges for randomisation)
    human_radius_range: tuple[float, float] = (0.2, 0.35)
    human_speed_range: tuple[float, float] = (0.6, 1.5)

    # Sensor
    lidar_num_beams: int = LIDAR.num_beams
    lidar_max_range: float = LIDAR.max_range

    # Observation cap
    max_observable_humans: int = 10

    # Custom reward hook
    custom_reward_fn: Any = None  # Callable[[NavEnv, np.ndarray, dict], float]


# ---------------------------------------------------------------------------
#  Helper – lazy backend construction
# ---------------------------------------------------------------------------


def _build_backend(config: NavEnvConfig) -> SceneBackend:
    """Instantiate a Grid2DBackend from *config*.

    If *scenario_path* is set the YAML is loaded; otherwise a procedural
    empty-world configuration is synthesised from the inline parameters.
    """
    from navirl.backends.grid2d.backend import Grid2DBackend

    if config.scenario_path is not None:
        import json  # noqa: E401

        import yaml

        path = Path(config.scenario_path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            cfg = yaml.safe_load(text)
        else:
            cfg = json.loads(text)
        scene_cfg = cfg.get("scene", cfg)
        horizon_cfg = cfg.get("horizon", {"dt": config.dt})
        return Grid2DBackend(scene_cfg, horizon_cfg, base_dir=path.parent)

    # Procedural / inline configuration
    scene_cfg: dict[str, Any] = {
        "id": config.map_name,
        "map": {"name": config.map_name},
        "orca": {"units": "meters"},
    }
    horizon_cfg: dict[str, Any] = {"dt": config.dt}
    return Grid2DBackend(scene_cfg, horizon_cfg)


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------


class NavEnv(gym.Env):
    """Single-robot social navigation environment.

    The robot must navigate to a sampled goal position while avoiding
    collisions with ORCA-controlled pedestrians and static obstacles.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": SIM.default_fps}

    # -----------------------------------------------------------------
    #  Construction
    # -----------------------------------------------------------------

    def __init__(self, config: NavEnvConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config if config is not None else NavEnvConfig(**kwargs)
        self.render_mode = self.config.render_mode

        # Placeholders initialised on first reset
        self._backend: SceneBackend | None = None
        self._rng: np.random.Generator = np.random.default_rng()

        # Agent bookkeeping
        self._robot_id: int = 0
        self._human_ids: list[int] = []
        self._robot_goal: np.ndarray = np.zeros(2, dtype=np.float32)
        self._human_goals: dict[int, np.ndarray] = {}
        self._step_count: int = 0
        self._prev_dist_to_goal: float = 0.0

        # Build spaces
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()

    # -----------------------------------------------------------------
    #  Space builders
    # -----------------------------------------------------------------

    def _make_observation_space(self) -> spaces.Space:
        cfg = self.config
        if cfg.observation_type == "state":
            # [rx, ry, rvx, rvy, gx, gy, dist] + per-human [dx, dy, dvx, dvy, radius]
            robot_dim = 7
            human_dim = 5 * cfg.max_observable_humans
            low = -np.inf * np.ones(robot_dim + human_dim, dtype=np.float32)
            high = np.inf * np.ones(robot_dim + human_dim, dtype=np.float32)
            return spaces.Box(low, high, dtype=np.float32)

        if cfg.observation_type == "lidar":
            return spaces.Box(
                low=0.0,
                high=float(cfg.lidar_max_range),
                shape=(cfg.lidar_num_beams,),
                dtype=np.float32,
            )

        if cfg.observation_type == "combined":
            robot_dim = 7
            human_dim = 5 * cfg.max_observable_humans
            return spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(robot_dim + human_dim,),
                        dtype=np.float32,
                    ),
                    "lidar": spaces.Box(
                        low=0.0,
                        high=float(cfg.lidar_max_range),
                        shape=(cfg.lidar_num_beams,),
                        dtype=np.float32,
                    ),
                }
            )

        raise ValueError(f"Unknown observation_type: {cfg.observation_type!r}")

    def _make_action_space(self) -> spaces.Space:
        cfg = self.config
        if cfg.action_type in ("continuous", "holonomic"):
            return spaces.Box(
                low=-cfg.robot_max_speed,
                high=cfg.robot_max_speed,
                shape=(2,),
                dtype=np.float32,
            )
        if cfg.action_type == "discrete":
            return spaces.Discrete(len(_DISCRETE_ACTIONS))
        raise ValueError(f"Unknown action_type: {cfg.action_type!r}")

    # -----------------------------------------------------------------
    #  Gymnasium API
    # -----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # (Re)create the physics backend
        self._backend = _build_backend(self.config)
        self._step_count = 0
        self._human_ids = []
        self._human_goals = {}

        # Place robot
        self._robot_id = 0
        start = self._sample_free_point()
        self._backend.add_agent(
            self._robot_id,
            tuple(start),
            self.config.robot_radius,
            self.config.robot_max_speed,
            kind="robot",
        )

        # Sample goal
        self._robot_goal = self._sample_free_point()
        while np.linalg.norm(self._robot_goal - start) < SIM.goal_radius * 2:
            self._robot_goal = self._sample_free_point()

        # Place humans
        self._spawn_humans(self.config.num_humans)

        # Cache initial distance
        robot_pos = np.array(self._backend.get_position(self._robot_id), dtype=np.float32)
        self._prev_dist_to_goal = float(np.linalg.norm(robot_pos - self._robot_goal))

        obs = self._compute_observation()
        info = self._make_info()
        return obs, info

    def step(self, action: np.ndarray | int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert self._backend is not None, "Call reset() before step()."

        # --- Convert action to preferred velocity ---
        velocity = self._action_to_velocity(action)
        self._backend.set_preferred_velocity(self._robot_id, tuple(velocity))

        # --- Drive humans toward their goals (simple ORCA autopilot) ---
        self._step_humans()

        # --- Advance physics ---
        self._backend.step()
        self._step_count += 1

        # --- Evaluate outcome ---
        obs = self._compute_observation()
        reward, terminated, info = self._compute_reward()
        truncated = self._step_count >= self.config.max_steps
        if truncated and not terminated:
            reward += REWARD.penalty_timeout

        info.update(self._make_info())
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self._backend is None:
            return None
        img = self._backend.map_image()
        if img is None:
            return None
        img = np.asarray(img, dtype=np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        # Simple overlay: draw robot and goal as coloured circles
        try:
            import cv2

            robot_pos = self._backend.get_position(self._robot_id)
            r_px = self._backend.world_to_map(robot_pos)
            g_px = self._backend.world_to_map(
                (float(self._robot_goal[0]), float(self._robot_goal[1]))
            )
            cv2.circle(img, (r_px[1], r_px[0]), 5, (31, 119, 180), -1)
            cv2.circle(img, (g_px[1], g_px[0]), 5, (214, 39, 40), -1)
            for hid in self._human_ids:
                hp = self._backend.get_position(hid)
                hp_px = self._backend.world_to_map(hp)
                cv2.circle(img, (hp_px[1], hp_px[0]), 4, (255, 127, 14), -1)
        except ImportError:
            pass

        if self.render_mode == "human":
            try:
                import cv2 as _cv2

                _cv2.imshow("NavEnv", img)
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
    #  Observation
    # -----------------------------------------------------------------

    def _compute_observation(self) -> Any:
        cfg = self.config
        if cfg.observation_type == "state":
            return self._obs_state()
        if cfg.observation_type == "lidar":
            return self._obs_lidar()
        if cfg.observation_type == "combined":
            return {"state": self._obs_state(), "lidar": self._obs_lidar()}
        raise ValueError(f"Unknown observation_type: {cfg.observation_type!r}")

    def _obs_state(self) -> np.ndarray:
        """Build the flat state vector."""
        assert self._backend is not None
        cfg = self.config
        rx, ry = self._backend.get_position(self._robot_id)
        rvx, rvy = self._backend.get_velocity(self._robot_id)
        gx, gy = float(self._robot_goal[0]), float(self._robot_goal[1])
        dist = math.hypot(rx - gx, ry - gy)

        robot_part = np.array([rx, ry, rvx, rvy, gx, gy, dist], dtype=np.float32)

        # Per-human relative state, sorted by distance
        human_entries: list[tuple[float, np.ndarray]] = []
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            hvx, hvy = self._backend.get_velocity(hid)
            dx, dy = hx - rx, hy - ry
            d = math.hypot(dx, dy)
            dvx, dvy = hvx - rvx, hvy - rvy
            radius = cfg.human_radius_range[0]  # approximate
            human_entries.append((d, np.array([dx, dy, dvx, dvy, radius], dtype=np.float32)))

        human_entries.sort(key=lambda t: t[0])
        human_part = np.zeros(5 * cfg.max_observable_humans, dtype=np.float32)
        for i, (_, vec) in enumerate(human_entries[: cfg.max_observable_humans]):
            human_part[i * 5 : i * 5 + 5] = vec

        return np.concatenate([robot_part, human_part])

    def _obs_lidar(self) -> np.ndarray:
        """Simple 2-D ray-cast LiDAR from the robot position."""
        assert self._backend is not None
        cfg = self.config
        rx, ry = self._backend.get_position(self._robot_id)
        num_beams = cfg.lidar_num_beams
        max_range = cfg.lidar_max_range
        angles = np.linspace(-math.pi, math.pi, num_beams, endpoint=False)
        ranges = np.full(num_beams, max_range, dtype=np.float32)

        # Check obstacles along each ray via discrete sampling
        num_samples = 50
        for i, angle in enumerate(angles):
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for s in range(1, num_samples + 1):
                d = max_range * s / num_samples
                px = rx + d * cos_a
                py = ry + d * sin_a
                if self._backend.check_obstacle_collision((px, py), 0.01):
                    ranges[i] = d
                    break

            # Also check human agents as lidar obstacles
            for hid in self._human_ids:
                hx, hy = self._backend.get_position(hid)
                dx, dy = hx - rx, hy - ry
                # Project onto ray direction
                proj = dx * cos_a + dy * sin_a
                if proj <= 0 or proj >= ranges[i]:
                    continue
                perp = abs(-dx * sin_a + dy * cos_a)
                h_radius = cfg.human_radius_range[0]
                if perp < h_radius:
                    hit_dist = max(0.0, proj - h_radius)
                    ranges[i] = min(ranges[i], hit_dist)

        return ranges

    # -----------------------------------------------------------------
    #  Reward
    # -----------------------------------------------------------------

    def _compute_reward(self) -> tuple[float, bool, dict[str, Any]]:
        """Return (reward, terminated, info_dict)."""
        assert self._backend is not None
        cfg = self.config

        if cfg.reward_type == "custom" and cfg.custom_reward_fn is not None:
            info: dict[str, Any] = {}
            reward = float(cfg.custom_reward_fn(self, None, info))
            terminated = info.get("terminated", False)
            return reward, terminated, info

        rx, ry = self._backend.get_position(self._robot_id)
        robot_pos = np.array([rx, ry], dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(robot_pos - self._robot_goal))

        info: dict[str, Any] = {"dist_to_goal": dist_to_goal}
        reward = 0.0
        terminated = False

        # --- Goal reached ---
        if dist_to_goal < SIM.goal_radius:
            reward += REWARD.reward_goal_reached
            terminated = True
            info["event"] = "goal_reached"
            self._prev_dist_to_goal = dist_to_goal
            return reward, terminated, info

        # --- Collision with obstacle ---
        if self._backend.check_obstacle_collision((rx, ry), cfg.robot_radius):
            reward += REWARD.penalty_collision_wall
            info["collision_wall"] = True

        # --- Collision / proxemics with humans ---
        min_human_dist = float("inf")
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            d = math.hypot(rx - hx, ry - hy)
            min_human_dist = min(min_human_dist, d)
            combined_radius = cfg.robot_radius + cfg.human_radius_range[0]
            if d < combined_radius:
                reward += REWARD.penalty_collision_pedestrian
                info["collision_pedestrian"] = True
            elif d < PROXEMICS.intimate.outer + cfg.robot_radius:
                reward += REWARD.penalty_intimate_zone

        info["min_human_dist"] = min_human_dist

        # --- Dense shaping ---
        if cfg.reward_type == "dense":
            progress = self._prev_dist_to_goal - dist_to_goal
            reward += REWARD.reward_goal_progress * progress
            reward += REWARD.penalty_per_step

        # --- Sparse only gives goal / collision signals ---
        # (already handled above; no progress bonus)

        self._prev_dist_to_goal = dist_to_goal
        return reward, terminated, info

    # -----------------------------------------------------------------
    #  Action conversion
    # -----------------------------------------------------------------

    def _action_to_velocity(self, action: np.ndarray | int) -> np.ndarray:
        cfg = self.config
        if cfg.action_type == "discrete":
            idx = int(action)
            return (_DISCRETE_ACTIONS[idx] * cfg.robot_max_speed).astype(np.float32)
        # continuous / holonomic
        vel = np.asarray(action, dtype=np.float32).flatten()[:2]
        speed = float(np.linalg.norm(vel))
        if speed > cfg.robot_max_speed:
            vel = vel / speed * cfg.robot_max_speed
        return vel

    # -----------------------------------------------------------------
    #  Human management
    # -----------------------------------------------------------------

    def _spawn_humans(self, n: int) -> None:
        """Add *n* ORCA humans to the backend."""
        assert self._backend is not None
        cfg = self.config
        for i in range(n):
            hid = i + 1  # robot is id 0
            radius = float(self._rng.uniform(*cfg.human_radius_range))
            speed = float(self._rng.uniform(*cfg.human_speed_range))
            pos = self._sample_free_point()
            self._backend.add_agent(hid, tuple(pos), radius, speed, kind="human")
            goal = self._sample_free_point()
            self._human_ids.append(hid)
            self._human_goals[hid] = goal

    def _step_humans(self) -> None:
        """Set preferred velocities for all humans toward their goals."""
        assert self._backend is not None
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            gx, gy = float(self._human_goals[hid][0]), float(self._human_goals[hid][1])
            dx, dy = gx - hx, gy - hy
            dist = math.hypot(dx, dy)
            if dist < SIM.goal_radius:
                # Assign a new random goal
                self._human_goals[hid] = self._sample_free_point()
                gx, gy = float(self._human_goals[hid][0]), float(self._human_goals[hid][1])
                dx, dy = gx - hx, gy - hy
                dist = math.hypot(dx, dy)
            if dist > EPSILON:
                speed = self.config.human_speed_range[0]  # conservative
                vx = dx / dist * speed
                vy = dy / dist * speed
            else:
                vx, vy = 0.0, 0.0
            self._backend.set_preferred_velocity(hid, (vx, vy))

    # -----------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------

    def _sample_free_point(self, max_attempts: int = 1000) -> np.ndarray:
        """Sample a free point from the environment.

        Parameters
        ----------
        max_attempts : int, default 1000
            Maximum number of sampling attempts before raising an error.

        Returns
        -------
        np.ndarray
            A valid free point as [x, y] coordinates.

        Raises
        ------
        RuntimeError
            If no free point can be found within max_attempts.
        """
        assert self._backend is not None

        for attempt in range(max_attempts):
            try:
                pt = self._backend.sample_free_point()
                if pt is None:
                    continue

                # Validate that the returned point is valid
                pt_array = np.array(pt, dtype=np.float32)
                if len(pt_array) < 2:
                    continue
                if not np.all(np.isfinite(pt_array[:2])):
                    continue

                return pt_array
            except Exception as e:
                # Log the exception but continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed to sample free point after {max_attempts} attempts"
                    ) from e
                continue

        raise RuntimeError(f"Failed to sample free point after {max_attempts} attempts")

    def _make_info(self) -> dict[str, Any]:
        """Assemble the info dict returned alongside observations."""
        assert self._backend is not None
        rx, ry = self._backend.get_position(self._robot_id)
        return {
            "step": self._step_count,
            "time_s": self._step_count * self.config.dt,
            "robot_position": (rx, ry),
            "robot_velocity": self._backend.get_velocity(self._robot_id),
            "goal": (float(self._robot_goal[0]), float(self._robot_goal[1])),
        }

    # Read-only accessors for downstream wrappers / reward functions
    @property
    def backend(self) -> SceneBackend | None:
        return self._backend

    @property
    def robot_id(self) -> int:
        return self._robot_id

    @property
    def robot_goal(self) -> np.ndarray:
        return self._robot_goal

    @property
    def human_ids(self) -> list[int]:
        return list(self._human_ids)


__all__ = ["NavEnv", "NavEnvConfig"]
