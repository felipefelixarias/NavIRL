from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from navirl.backends.grid2d.maps import MapInfo, load_map_info
from navirl.backends.grid2d.constants import FREE_SPACE, OBSTACLE_SPACE, RADIUS_METERS
from navirl.backends.grid2d.environment import GridEnvironment
from navirl.backends.grid2d.orca import IndoorORCASim, IndoorORCASimConfig
from navirl.core.env import SceneBackend


@dataclass(slots=True)
class AgentMeta:
    ext_id: int
    kind: str
    radius: float
    max_speed: float


class Grid2DBackend(SceneBackend):
    """Grid + ORCA backend using INDOORCA map processing and RVO2 stepping."""

    def __init__(self, scene_cfg: dict, horizon_cfg: dict, base_dir: Path | None = None):
        dt = float(horizon_cfg.get("dt", 0.1))
        orca_cfg = scene_cfg.get("orca", {})
        orca_units = str(orca_cfg.get("units", "meters")).lower()
        if orca_units not in {"meters", "pixels"}:
            raise ValueError("scene.orca.units must be 'meters' or 'pixels'")

        self.map_info: MapInfo = load_map_info(scene_cfg, base_dir=base_dir)
        self.binary_map = self.map_info.binary_map
        self.env = GridEnvironment(
            scene_cfg.get("id", "grid2d_scene"),
            self.binary_map,
            pixels_per_meter=self.map_info.pixels_per_meter,
        )
        self.env.process_obstacles()
        ppm = float(self.map_info.pixels_per_meter)

        def _distance_value(name: str, default_val: float) -> float:
            raw = float(orca_cfg.get(name, default_val))
            return raw / ppm if orca_units == "pixels" else raw

        def _speed_value(name: str, default_val: float) -> float:
            raw = float(orca_cfg.get(name, default_val))
            return raw / ppm if orca_units == "pixels" else raw

        # Tuned defaults for smoother, cleaner social motion in tight indoor maps.
        neighbor_dist = _distance_value("neighbor_dist", 3.5)
        max_neighbors = int(orca_cfg.get("max_neighbors", 24))
        time_horizon = float(orca_cfg.get("time_horizon", 3.5))
        time_horizon_obst = float(orca_cfg.get("time_horizon_obst", 2.5))
        default_radius = _distance_value("radius", RADIUS_METERS)
        default_max_speed = _speed_value("max_speed", 0.8)
        self.wall_clearance_buffer_m = _distance_value("wall_clearance_buffer_m", 0.0)
        self.pref_velocity_smoothing = float(orca_cfg.get("pref_velocity_smoothing", 0.35))

        self.orca = IndoorORCASim(
            IndoorORCASimConfig(
                time_step=dt,
                neighbor_dist=neighbor_dist,
                max_neighbors=max_neighbors,
                time_horizon=time_horizon,
                time_horizon_obst=time_horizon_obst,
                radius=default_radius,
                max_speed=default_max_speed,
            )
        )
        self.orca.add_obstacles(self.env.get_obstacle_meters())
        self.orca.process_obstacles()

        self._ext_to_orca: dict[int, int] = {}
        self._meta: dict[int, AgentMeta] = {}
        self._dt = dt
        free_mask = (self.env.map == FREE_SPACE).astype(np.uint8)
        self._clearance_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
        self._clearance_cache: dict[int, np.ndarray] = {}
        self._pref_vel_cache: dict[int, tuple[float, float]] = {}
        self._max_agent_radius = default_radius

    def _required_clearance_px(self, radius: float, with_buffer: bool = True) -> float:
        extra = self.wall_clearance_buffer_m if with_buffer else 0.0
        return max(1.0, (radius + extra) * self.env.pixels_per_meter)

    def _clearance_at(self, position: tuple[float, float]) -> float:
        row, col = self.world_to_map(position)
        h, w = self.env.map.shape
        if row < 0 or col < 0 or row >= h or col >= w:
            return 0.0
        return float(self._clearance_px[row, col])

    def _nearest_clear_world(
        self,
        position: tuple[float, float],
        radius: float,
        with_buffer: bool = True,
    ) -> tuple[float, float]:
        row, col = self.world_to_map(position)
        h, w = self.env.map.shape
        required_px = int(round(self._required_clearance_px(radius, with_buffer=with_buffer)))

        if 0 <= row < h and 0 <= col < w:
            if self.env.map[row, col] != OBSTACLE_SPACE and self._clearance_px[row, col] >= required_px:
                return float(position[0]), float(position[1])

        candidates = self._clearance_cache.get(required_px)
        if candidates is None:
            candidates = np.argwhere((self.env.map == FREE_SPACE) & (self._clearance_px >= required_px))
            self._clearance_cache[required_px] = candidates

        if candidates.size == 0:
            return self.env.nearest_free_world(position)

        target = np.array([row, col], dtype=float)
        deltas = candidates.astype(float) - target
        best_idx = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        rr, cc = candidates[best_idx]
        world = self.env._map_to_world(np.array([rr, cc], dtype=float))
        return float(world[0]), float(world[1])

    @property
    def dt(self) -> float:
        return self._dt

    def add_agent(
        self,
        agent_id: int,
        position: tuple[float, float],
        radius: float,
        max_speed: float,
        kind: str,
    ) -> None:
        pos = tuple(map(float, position))
        pos = self._nearest_clear_world(pos, radius, with_buffer=True)

        orca_id = self.orca.sim.addAgent(
            tuple(pos),
            self.orca.get_neighbor_dist(),
            self.orca.get_max_neighbors(),
            self.orca.get_time_horizon(),
            self.orca.get_time_horizon_obst(),
            radius,
            max_speed,
            (0.0, 0.0),
        )
        self._ext_to_orca[agent_id] = orca_id
        self._meta[agent_id] = AgentMeta(
            ext_id=agent_id,
            kind=kind,
            radius=radius,
            max_speed=max_speed,
        )
        self._max_agent_radius = max(self._max_agent_radius, float(radius))
        self._pref_vel_cache[agent_id] = (0.0, 0.0)

    def set_preferred_velocity(self, agent_id: int, velocity: tuple[float, float]) -> None:
        alpha = max(0.0, min(1.0, self.pref_velocity_smoothing))
        kind = self._meta.get(agent_id).kind if agent_id in self._meta else "human"
        if kind == "robot":
            alpha = max(alpha, 0.75)
        prev_vx, prev_vy = self._pref_vel_cache.get(agent_id, (0.0, 0.0))
        vx = (1.0 - alpha) * prev_vx + alpha * float(velocity[0])
        vy = (1.0 - alpha) * prev_vy + alpha * float(velocity[1])
        self._pref_vel_cache[agent_id] = (vx, vy)
        self.orca.set_agent_pref_velocity(self._ext_to_orca[agent_id], [vx, vy])

    def step(self) -> None:
        self.orca.do_step()
        # Keep agents on traversable space if ORCA pushes through obstacle cells.
        for ext_id, orca_id in self._ext_to_orca.items():
            x, y = self.get_position(ext_id)
            radius = self._meta[ext_id].radius
            row, col = self.world_to_map((x, y))
            if row < 0 or col < 0 or row >= self.env.map.shape[0] or col >= self.env.map.shape[1]:
                nx, ny = self._nearest_clear_world((x, y), radius, with_buffer=True)
                self.orca.set_agent_position(orca_id, [nx, ny])
                continue
            required_clearance = self._required_clearance_px(radius, with_buffer=True)
            if self._clearance_at((x, y)) + 1e-6 < required_clearance:
                nx, ny = self._nearest_clear_world((x, y), radius, with_buffer=True)
                self.orca.set_agent_position(orca_id, [nx, ny])

            # Dampen jitter when near-stationary to avoid noisy directional flicker.
            vx, vy = self.orca.get_agent_velocity(orca_id)
            speed = float(np.hypot(vx, vy))
            kind = self._meta[ext_id].kind
            jitter_stop = 0.01 if kind == "robot" else 0.03
            if speed < jitter_stop:
                self.orca.sim.setAgentVelocity(orca_id, (0.0, 0.0))

    def get_position(self, agent_id: int) -> tuple[float, float]:
        x, y = self.orca.get_agent_position(self._ext_to_orca[agent_id])
        return float(x), float(y)

    def get_velocity(self, agent_id: int) -> tuple[float, float]:
        vx, vy = self.orca.get_agent_velocity(self._ext_to_orca[agent_id])
        return float(vx), float(vy)

    def shortest_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        path_radius = float(self._max_agent_radius)
        waypoints, _ = self.env.shortest_path(np.array(start), np.array(goal), entire_path=True)
        if waypoints.size == 0:
            sx, sy = self._nearest_clear_world(start, path_radius, with_buffer=False)
            gx, gy = self._nearest_clear_world(goal, path_radius, with_buffer=False)
            return [(sx, sy), (gx, gy)]
        adjusted: list[tuple[float, float]] = []
        for p in waypoints:
            x, y = self._nearest_clear_world((float(p[0]), float(p[1])), path_radius, with_buffer=False)
            adjusted.append((x, y))
        return adjusted

    def sample_free_point(self) -> tuple[float, float]:
        x, y = self.env.get_random_point()
        return float(x), float(y)

    def check_obstacle_collision(self, position: tuple[float, float], radius: float) -> bool:
        required = self._required_clearance_px(radius, with_buffer=False)
        return self._clearance_at(position) + 1e-6 < required

    def nearest_clear_point(self, position: tuple[float, float], radius: float) -> tuple[float, float]:
        x, y = self._nearest_clear_world(position, radius, with_buffer=False)
        return float(x), float(y)

    def world_to_map(self, position: tuple[float, float]) -> tuple[int, int]:
        rc = self.env._world_to_map(np.array(position, dtype=float))
        return int(rc[0]), int(rc[1])

    def map_image(self):
        return self.binary_map

    def map_metadata(self) -> dict:
        return self.map_info.to_dict()

    def get_agent_meta(self, agent_id: int) -> AgentMeta:
        return self._meta[agent_id]

    def agent_ids(self) -> list[int]:
        return sorted(self._meta)
