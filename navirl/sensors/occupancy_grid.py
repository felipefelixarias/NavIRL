"""Ego-centric occupancy grid sensor.

Provides :class:`OccupancyGridSensor` which generates a multi-layer local
occupancy grid from the world state.  Layers can include static obstacles,
dynamic agents, velocity fields, and social-zone information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navirl.core.constants import EPSILON, PROXEMICS
from navirl.sensors.base import NoiseModel, SensorBase


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class OccupancyGridConfig:
    """Configuration for the ego-centric occupancy grid sensor.

    Parameters
    ----------
    grid_size : int
        Number of cells along each axis (grid is square).
    resolution : float
        Side length of each cell in metres.
    layers : tuple of str
        Which layers to include in the output.  Supported layers:

        * ``"static"`` -- binary occupancy from static obstacles
        * ``"dynamic"`` -- binary occupancy from dynamic agents
        * ``"velocity_x"`` -- x-component of agent velocity in each cell
        * ``"velocity_y"`` -- y-component of agent velocity in each cell
        * ``"social"`` -- social zone encoding (0 = empty, 1 = public,
          2 = social, 3 = personal, 4 = intimate)
    decay : float
        Temporal decay factor for dynamic layers (0 = no memory, 1 = full
        persistence). Currently unused but reserved for future temporal grids.
    """

    grid_size: int = 64
    resolution: float = 0.25  # metres per cell
    layers: Tuple[str, ...] = ("static", "dynamic", "velocity_x",
                                "velocity_y", "social")
    decay: float = 0.0


# ---------------------------------------------------------------------------
#  OccupancyGridSensor
# ---------------------------------------------------------------------------

class OccupancyGridSensor(SensorBase):
    """Local ego-centric multi-layer occupancy grid.

    The grid is centred on the robot and aligned with its heading (forward is
    +y in grid coordinates, i.e. the top of the grid image).

    World state keys
    ----------------
    * ``robot_pos`` : (2,) ndarray
    * ``robot_heading`` : float (radians)
    * ``agents`` : list of dicts with ``pos`` (2,), ``vel`` (2,), ``radius``.
    * ``obstacles_segments`` : (S, 2, 2) ndarray (optional)
    * ``obstacles_circles`` : dict with ``centres`` (N, 2), ``radii`` (N,)
      (optional)

    Returns
    -------
    np.ndarray
        Shape ``(num_layers, grid_size, grid_size)``, float32.
    """

    def __init__(
        self,
        config: Optional[OccupancyGridConfig] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> None:
        config = config or OccupancyGridConfig()
        super().__init__(config=config, noise_model=noise_model)
        self._layer_names = list(config.layers)
        self._half_extent = config.grid_size * config.resolution / 2.0

        # Pre-compute cell centre offsets (relative to grid centre)
        gs = config.grid_size
        res = config.resolution
        # Cell centres in local frame (robot at origin, heading along +y)
        xs = (np.arange(gs) - gs / 2.0 + 0.5) * res
        ys = (np.arange(gs) - gs / 2.0 + 0.5) * res
        # ys is flipped so row 0 = farthest forward
        ys = ys[::-1]
        self._cell_xs, self._cell_ys = np.meshgrid(xs, ys)  # (gs, gs)

    # -- SensorBase interface ------------------------------------------------

    def get_observation_space(self) -> Dict[str, Any]:
        gs = self.config.grid_size
        n_layers = len(self._layer_names)
        return {
            "shape": (n_layers, gs, gs),
            "dtype": np.float32,
            "low": 0.0,
            "high": 5.0,
            "layer_names": self._layer_names,
        }

    def _raw_observe(self, world_state: Dict[str, Any]) -> np.ndarray:
        gs = self.config.grid_size
        res = self.config.resolution
        n_layers = len(self._layer_names)
        grid = np.zeros((n_layers, gs, gs), dtype=np.float32)

        pos = np.asarray(world_state["robot_pos"], dtype=np.float64)
        heading = float(world_state.get("robot_heading", 0.0))
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)

        # Cell centres in world frame
        local_x = self._cell_xs.ravel()
        local_y = self._cell_ys.ravel()
        world_x = pos[0] + local_x * cos_h - local_y * sin_h  # (gs*gs,)
        world_y = pos[1] + local_x * sin_h + local_y * cos_h

        for li, layer_name in enumerate(self._layer_names):
            if layer_name == "static":
                grid[li] = self._render_static(
                    world_state, world_x, world_y, gs, res)
            elif layer_name == "dynamic":
                grid[li] = self._render_dynamic(
                    world_state, world_x, world_y, gs, res)
            elif layer_name == "velocity_x":
                grid[li] = self._render_velocity(
                    world_state, world_x, world_y, gs, res,
                    component=0, heading=heading)
            elif layer_name == "velocity_y":
                grid[li] = self._render_velocity(
                    world_state, world_x, world_y, gs, res,
                    component=1, heading=heading)
            elif layer_name == "social":
                grid[li] = self._render_social(
                    world_state, world_x, world_y, gs, res)

        return grid

    # -- Layer renderers -----------------------------------------------------

    def _render_static(
        self, ws: Dict[str, Any],
        wx: np.ndarray, wy: np.ndarray,
        gs: int, res: float,
    ) -> np.ndarray:
        """Binary static obstacle layer."""
        layer = np.zeros(gs * gs, dtype=np.float32)

        # Circular obstacles
        obs_c = ws.get("obstacles_circles", None)
        if obs_c is not None:
            centres = np.asarray(obs_c["centres"], dtype=np.float64)
            radii = np.asarray(obs_c["radii"], dtype=np.float64)
            for i in range(centres.shape[0]):
                dx = wx - centres[i, 0]
                dy = wy - centres[i, 1]
                dist_sq = dx * dx + dy * dy
                layer[dist_sq <= (radii[i] + res / 2) ** 2] = 1.0

        # Segment obstacles: mark cells near each segment
        segs = ws.get("obstacles_segments", None)
        if segs is not None:
            segs = np.asarray(segs, dtype=np.float64)
            for s in range(segs.shape[0]):
                p0, p1 = segs[s, 0], segs[s, 1]
                seg_d = p1 - p0
                seg_len_sq = np.dot(seg_d, seg_d)
                if seg_len_sq < EPSILON:
                    continue
                # Project each cell centre onto segment
                t = ((wx - p0[0]) * seg_d[0] + (wy - p0[1]) * seg_d[1]) / seg_len_sq
                t = np.clip(t, 0, 1)
                closest_x = p0[0] + t * seg_d[0]
                closest_y = p0[1] + t * seg_d[1]
                dist_sq = (wx - closest_x) ** 2 + (wy - closest_y) ** 2
                layer[dist_sq <= (res * 0.75) ** 2] = 1.0

        return layer.reshape(gs, gs)

    def _render_dynamic(
        self, ws: Dict[str, Any],
        wx: np.ndarray, wy: np.ndarray,
        gs: int, res: float,
    ) -> np.ndarray:
        """Binary dynamic agent (pedestrian) layer."""
        layer = np.zeros(gs * gs, dtype=np.float32)
        agents = ws.get("agents", [])
        for agent in agents:
            if isinstance(agent, dict):
                ap = np.asarray(agent["pos"], dtype=np.float64)
                ar = agent.get("radius", 0.25)
            else:
                ap = np.asarray(agent[:2], dtype=np.float64)
                ar = agent[2] if len(agent) > 2 else 0.25
            dx = wx - ap[0]
            dy = wy - ap[1]
            dist_sq = dx * dx + dy * dy
            layer[dist_sq <= (ar + res / 2) ** 2] = 1.0
        return layer.reshape(gs, gs)

    def _render_velocity(
        self, ws: Dict[str, Any],
        wx: np.ndarray, wy: np.ndarray,
        gs: int, res: float,
        component: int, heading: float,
    ) -> np.ndarray:
        """Velocity component layer (in ego frame)."""
        layer = np.zeros(gs * gs, dtype=np.float32)
        agents = ws.get("agents", [])
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)

        for agent in agents:
            if isinstance(agent, dict):
                ap = np.asarray(agent["pos"], dtype=np.float64)
                av = np.asarray(agent.get("vel", [0, 0]), dtype=np.float64)
                ar = agent.get("radius", 0.25)
            else:
                ap = np.asarray(agent[:2], dtype=np.float64)
                av = np.asarray(agent[2:4], dtype=np.float64) if len(agent) >= 4 else np.zeros(2)
                ar = agent[4] if len(agent) > 4 else 0.25

            dx = wx - ap[0]
            dy = wy - ap[1]
            dist_sq = dx * dx + dy * dy
            mask = dist_sq <= (ar + res / 2) ** 2

            # Rotate velocity to ego frame
            ego_vx = av[0] * cos_h - av[1] * sin_h
            ego_vy = av[0] * sin_h + av[1] * cos_h
            vel_ego = np.array([ego_vx, ego_vy])

            layer[mask] = vel_ego[component]

        return layer.reshape(gs, gs)

    def _render_social(
        self, ws: Dict[str, Any],
        wx: np.ndarray, wy: np.ndarray,
        gs: int, res: float,
    ) -> np.ndarray:
        """Social zone layer.

        Encodes the proxemic zone of the nearest agent at each cell:
        0 = empty/public, 1 = social, 2 = personal, 3 = intimate.
        """
        layer = np.zeros(gs * gs, dtype=np.float32)
        agents = ws.get("agents", [])
        if len(agents) == 0:
            return layer.reshape(gs, gs)

        # Compute minimum distance to any agent at each cell
        min_dist = np.full(gs * gs, 1e6, dtype=np.float64)
        for agent in agents:
            if isinstance(agent, dict):
                ap = np.asarray(agent["pos"], dtype=np.float64)
                ar = agent.get("radius", 0.25)
            else:
                ap = np.asarray(agent[:2], dtype=np.float64)
                ar = agent[2] if len(agent) > 2 else 0.25

            dx = wx - ap[0]
            dy = wy - ap[1]
            dist = np.sqrt(dx * dx + dy * dy) - ar
            min_dist = np.minimum(min_dist, dist)

        # Classify by proxemic zone
        layer[min_dist < PROXEMICS.intimate.outer] = 4.0
        layer[(min_dist >= PROXEMICS.intimate.outer) &
              (min_dist < PROXEMICS.personal.outer)] = 3.0
        layer[(min_dist >= PROXEMICS.personal.outer) &
              (min_dist < PROXEMICS.social.outer)] = 2.0
        layer[(min_dist >= PROXEMICS.social.outer) &
              (min_dist < PROXEMICS.public.outer)] = 1.0

        return layer.reshape(gs, gs)
