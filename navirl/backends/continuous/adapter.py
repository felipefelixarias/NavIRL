"""SceneBackend adapter for the continuous-space simulation.

Wraps :class:`ContinuousEnvironment` so that it conforms to the
:class:`~navirl.core.env.SceneBackend` interface used by the NavIRL
pipeline, controllers, and Gymnasium wrapper.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from navirl.backends.continuous.environment import (
    AgentConfig,
    ContinuousEnvironment,
    EnvironmentConfig,
)
from navirl.backends.continuous.physics import AgentState, PhysicsConfig
from navirl.core.env import SceneBackend


class ContinuousSceneBackend(SceneBackend):
    """Adapter that exposes a :class:`ContinuousEnvironment` through the
    :class:`SceneBackend` abstract interface.

    Parameters
    ----------
    scene_cfg : dict
        Scene configuration.  Recognised keys:

        * ``width``, ``height`` – world dimensions in metres (default 20).
        * ``dt`` – time-step override (usually supplied via *horizon_cfg*).
        * ``physics`` – dict forwarded to :class:`PhysicsConfig`.
        * ``obstacles`` – list of obstacle dicts (``type``, params).
        * ``walls`` – list of wall dicts (``start``, ``end``, ``thickness``).

    horizon_cfg : dict
        Must contain ``dt`` (float, seconds).
    base_dir : object, optional
        Ignored – present for factory-signature compatibility with
        :class:`Grid2DBackend`.
    """

    def __init__(
        self,
        scene_cfg: dict,
        horizon_cfg: dict,
        base_dir: object | None = None,
    ) -> None:
        dt = float(horizon_cfg.get("dt", scene_cfg.get("dt", 0.1)))
        width = float(scene_cfg.get("width", 20.0))
        height = float(scene_cfg.get("height", 20.0))

        physics_dict = scene_cfg.get("physics", {})
        physics_cfg = PhysicsConfig(**physics_dict) if physics_dict else PhysicsConfig()

        env_config = EnvironmentConfig(
            width=width,
            height=height,
            dt=dt,
            physics=physics_cfg,
            enable_boundaries=scene_cfg.get("enable_boundaries", True),
        )
        self._env = ContinuousEnvironment(env_config)
        self._dt = dt
        self._width = width
        self._height = height

        # Map from external agent_id → internal ContinuousEnvironment id.
        self._ext_to_int: dict[int, int] = {}
        self._agent_kinds: dict[int, str] = {}
        self._preferred_velocities: dict[int, tuple[float, float]] = {}
        self._rng = np.random.default_rng()

        # Pre-populate obstacles from config.
        for obs_spec in scene_cfg.get("obstacles", []):
            obs_type = obs_spec.get("type", "circle")
            if obs_type == "circle":
                self._env.add_circular_obstacle(
                    np.array(obs_spec["center"], dtype=float),
                    float(obs_spec["radius"]),
                )
            elif obs_type == "rectangle":
                self._env.add_rectangular_obstacle(
                    np.array(obs_spec["min_corner"], dtype=float),
                    np.array(obs_spec["max_corner"], dtype=float),
                )
        for wall_spec in scene_cfg.get("walls", []):
            self._env.add_wall(
                np.array(wall_spec["start"], dtype=float),
                np.array(wall_spec["end"], dtype=float),
                float(wall_spec.get("thickness", 0.1)),
            )

        # The environment must be reset before stepping; we do an initial
        # reset after setup so that agents added later via add_agent can be
        # integrated correctly.
        self._needs_reset = True
        self._map_resolution = 10  # pixels per metre for the synthetic map

    # ------------------------------------------------------------------
    # SceneBackend abstract methods
    # ------------------------------------------------------------------

    def add_agent(
        self,
        agent_id: int,
        position: tuple[float, float],
        radius: float,
        max_speed: float,
        kind: str,
    ) -> None:
        config = AgentConfig(
            position=np.array(position, dtype=float),
            goal=np.array(position, dtype=float),  # goal set later by controller
            radius=radius,
            preferred_speed=max_speed * 0.8,
            max_speed=max_speed,
            agent_type="robot" if kind == "robot" else "pedestrian",
        )
        int_id = self._env.add_agent(config)
        self._ext_to_int[agent_id] = int_id
        self._agent_kinds[agent_id] = kind
        self._preferred_velocities[agent_id] = (0.0, 0.0)
        self._needs_reset = True

    def set_preferred_velocity(self, agent_id: int, velocity: tuple[float, float]) -> None:
        self._preferred_velocities[agent_id] = (float(velocity[0]), float(velocity[1]))

    def step(self) -> None:
        self._ensure_reset()

        actions: dict[int, np.ndarray] = {}
        for ext_id, int_id in self._ext_to_int.items():
            vx, vy = self._preferred_velocities.get(ext_id, (0.0, 0.0))
            actions[int_id] = np.array([vx, vy], dtype=float)

        self._env.step(actions)

    def _ensure_reset(self) -> None:
        """Lazily reset the environment so agent states are available."""
        if self._needs_reset:
            self._env.reset()
            self._needs_reset = False

    def get_position(self, agent_id: int) -> tuple[float, float]:
        self._ensure_reset()
        int_id = self._ext_to_int[agent_id]
        state = self._env.get_agent_state(int_id)
        if state is None:
            raise KeyError(f"Agent {agent_id} not found")
        return float(state.position[0]), float(state.position[1])

    def get_velocity(self, agent_id: int) -> tuple[float, float]:
        self._ensure_reset()
        int_id = self._ext_to_int[agent_id]
        state = self._env.get_agent_state(int_id)
        if state is None:
            raise KeyError(f"Agent {agent_id} not found")
        return float(state.velocity[0]), float(state.velocity[1])

    def shortest_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Visibility-graph shortest path through obstacle-free space.

        Falls back to a straight line when the direct path is clear or when
        no shorter detour exists.
        """
        s = np.array(start, dtype=float)
        g = np.array(goal, dtype=float)

        # Fast path: direct line of sight.
        if self._line_clear(s, g):
            return [tuple(s), tuple(g)]

        # Grid-based A* over a discretised occupancy grid.
        return self._grid_astar(s, g)

    def sample_free_point(self) -> tuple[float, float]:
        for _ in range(200):
            x = self._rng.uniform(0.0, self._width)
            y = self._rng.uniform(0.0, self._height)
            if not self._env.obstacles.check_collision(np.array([x, y]), 0.3):
                return (x, y)
        # Fallback – centre of the world.
        return (self._width / 2, self._height / 2)

    def check_obstacle_collision(self, position: tuple[float, float], radius: float) -> bool:
        return self._env.obstacles.check_collision(np.array(position, dtype=float), radius)

    def world_to_map(self, position: tuple[float, float]) -> tuple[int, int]:
        res = self._map_resolution
        col = int(round(position[0] * res))
        row = int(round((self._height - position[1]) * res))
        return (row, col)

    def map_image(self) -> np.ndarray:
        res = self._map_resolution
        h = int(round(self._height * res))
        w = int(round(self._width * res))
        img = np.full((h, w), 255, dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                wx = c / res
                wy = self._height - r / res
                if self._env.obstacles.check_collision(np.array([wx, wy]), 0.0):
                    img[r, c] = 0
        return img

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def nearest_clear_point(
        self, position: tuple[float, float], radius: float
    ) -> tuple[float, float]:
        pos = np.array(position, dtype=float)
        if not self._env.obstacles.check_collision(pos, radius):
            return (float(pos[0]), float(pos[1]))
        # Simple radial search.
        for dist in np.linspace(0.1, max(self._width, self._height) / 2, 50):
            for angle_i in range(16):
                angle = 2 * math.pi * angle_i / 16
                candidate = pos + dist * np.array([math.cos(angle), math.sin(angle)])
                if (
                    0 <= candidate[0] <= self._width
                    and 0 <= candidate[1] <= self._height
                    and not self._env.obstacles.check_collision(candidate, radius)
                ):
                    return (float(candidate[0]), float(candidate[1]))
        return (float(pos[0]), float(pos[1]))

    def map_metadata(self) -> dict:
        return {
            "backend": "continuous",
            "width": self._width,
            "height": self._height,
            "resolution": self._map_resolution,
            "dt": self._dt,
            "num_obstacles": len(self._env.obstacles),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _line_clear(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """Check if a straight line between two points is obstacle-free."""
        diff = goal - start
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return True
        direction = diff / dist
        result = self._env.obstacles.ray_cast(start, direction, dist)
        return result is None

    def _grid_astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[tuple[float, float]]:
        """A* over a coarse grid to find an obstacle-avoiding path."""
        cell = 0.5  # metres per cell
        cols = max(1, int(math.ceil(self._width / cell)))
        rows = max(1, int(math.ceil(self._height / cell)))
        agent_radius = 0.3

        def to_cell(p: np.ndarray) -> tuple[int, int]:
            return (
                min(max(int(p[1] / cell), 0), rows - 1),
                min(max(int(p[0] / cell), 0), cols - 1),
            )

        def to_world(rc: tuple[int, int]) -> np.ndarray:
            return np.array([(rc[1] + 0.5) * cell, (rc[0] + 0.5) * cell])

        sc = to_cell(start)
        gc = to_cell(goal)

        # Build obstacle mask lazily.
        blocked: set[tuple[int, int]] = set()
        for r in range(rows):
            for c in range(cols):
                wp = to_world((r, c))
                if self._env.obstacles.check_collision(wp, agent_radius):
                    blocked.add((r, c))

        # A*
        open_set: list[tuple[float, tuple[int, int]]] = [(0.0, sc)]
        g_score: dict[tuple[int, int], float] = {sc: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1]) * cell

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == gc:
                # Reconstruct
                path_cells = [gc]
                while path_cells[-1] in came_from:
                    path_cells.append(came_from[path_cells[-1]])
                path_cells.reverse()
                path = [tuple(start)]
                for pc in path_cells[1:-1]:
                    w = to_world(pc)
                    path.append((float(w[0]), float(w[1])))
                path.append(tuple(goal))
                return path

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = current[0] + dr, current[1] + dc
                    nb = (nr, nc)
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if nb in blocked:
                        continue
                    step_cost = math.hypot(dr, dc) * cell
                    tentative = g_score[current] + step_cost
                    if tentative < g_score.get(nb, float("inf")):
                        g_score[nb] = tentative
                        came_from[nb] = current
                        heapq.heappush(open_set, (tentative + heuristic(nb, gc), nb))

        # No path found – straight line fallback.
        return [tuple(start), tuple(goal)]
