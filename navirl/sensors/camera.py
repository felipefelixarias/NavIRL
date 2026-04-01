"""Simulated camera and depth sensors.

Provides :class:`CameraSensor` (top-down or perspective RGB rendering) and
:class:`DepthSensor` (1-D or 2-D depth array).  Both sensors operate on
a world state dictionary and return numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.constants import CAMERA, DEPTH_SENSOR, EPSILON
from navirl.sensors.base import GaussianNoise, NoiseModel, SensorBase

# ---------------------------------------------------------------------------
#  Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Configuration for the simulated monocular camera.

    Defaults are pulled from :data:`navirl.core.constants.CAMERA`.
    """

    fov_horizontal: float = CAMERA.fov_horizontal
    fov_vertical: float = CAMERA.fov_vertical
    resolution_x: int = CAMERA.resolution_x
    resolution_y: int = CAMERA.resolution_y
    max_depth: float = CAMERA.max_depth
    focal_length_px: float = CAMERA.focal_length_px
    render_mode: str = "top_down"  # "top_down" or "perspective"


@dataclass
class DepthSensorConfig:
    """Configuration for the simulated depth sensor.

    Defaults are pulled from :data:`navirl.core.constants.DEPTH_SENSOR`.
    """

    fov_horizontal: float = DEPTH_SENSOR.fov_horizontal
    resolution: int = DEPTH_SENSOR.resolution
    max_range: float = DEPTH_SENSOR.max_range
    min_range: float = DEPTH_SENSOR.min_range
    noise_std: float = DEPTH_SENSOR.noise_std


# ---------------------------------------------------------------------------
#  CameraSensor
# ---------------------------------------------------------------------------

class CameraSensor(SensorBase):
    """Simulated monocular RGB camera.

    Supports two rendering modes:

    * **top_down** -- birds-eye view centred on the robot.  Agents and
      obstacles are rendered as filled circles/rectangles on a blank canvas.
    * **perspective** -- simplified pin-hole perspective projection of nearby
      entities onto an image plane.

    World state keys
    ----------------
    * ``robot_pos`` : (2,) ndarray
    * ``robot_heading`` : float (radians)
    * ``agents`` : list of dicts (``pos``, ``radius``, optional ``color``)
    * ``obstacles_circles`` : dict (``centres``, ``radii``)
    * ``obstacles_segments`` : (S, 2, 2) ndarray
    * ``world_bounds`` : (4,) -- [xmin, ymin, xmax, ymax] (optional)
    """

    def __init__(
        self,
        config: CameraConfig | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        self._config = config or CameraConfig()
        super().__init__(config=self._config, noise_model=noise_model)

    # -- SensorBase interface ------------------------------------------------

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "shape": (self.config.resolution_y, self.config.resolution_x, 3),
            "dtype": np.uint8,
            "low": 0,
            "high": 255,
        }

    def _raw_observe(self, world_state: dict[str, Any]) -> np.ndarray:
        if self.config.render_mode == "top_down":
            return self._render_top_down(world_state)
        return self._render_perspective(world_state)

    # -- Rendering helpers ---------------------------------------------------

    def _render_top_down(self, ws: dict[str, Any]) -> np.ndarray:
        """Render a simple birds-eye RGB image centred on the robot."""
        H, W = self.config.resolution_y, self.config.resolution_x
        img = np.full((H, W, 3), 245, dtype=np.uint8)  # light background

        pos = np.asarray(ws["robot_pos"], dtype=np.float64)
        heading = float(ws.get("robot_heading", 0.0))

        # Determine view window (metres per pixel)
        view_range = self.config.max_depth
        m_per_px = (2.0 * view_range) / min(H, W)

        def world_to_pixel(wx: float, wy: float):
            """Convert world coords to pixel coords (ego-centric, heading-up)."""
            dx, dy = wx - pos[0], wy - pos[1]
            cos_h, sin_h = np.cos(-heading), np.sin(-heading)
            rx = dx * cos_h - dy * sin_h
            ry = dx * sin_h + dy * cos_h
            px = int(W / 2 + rx / m_per_px)
            py = int(H / 2 - ry / m_per_px)
            return px, py

        def draw_filled_circle(img, cx, cy, r_px, color):
            """Draw a filled circle using numpy indexing."""
            yy, xx = np.ogrid[:H, :W]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_px ** 2
            img[mask] = color

        # Draw agents
        agents = ws.get("agents", [])
        for agent in agents:
            ap = np.asarray(agent["pos"]) if isinstance(agent, dict) else agent[:2]
            ar = agent["radius"] if isinstance(agent, dict) else agent[2]
            px, py = world_to_pixel(ap[0], ap[1])
            r_px = max(1, int(ar / m_per_px))
            color = agent.get("color", [255, 127, 14]) if isinstance(agent, dict) else [255, 127, 14]
            draw_filled_circle(img, px, py, r_px, color)

        # Draw static circular obstacles
        obs_c = ws.get("obstacles_circles", None)
        if obs_c is not None:
            centres = np.asarray(obs_c["centres"])
            radii = np.asarray(obs_c["radii"])
            for i in range(centres.shape[0]):
                px, py = world_to_pixel(centres[i, 0], centres[i, 1])
                r_px = max(1, int(radii[i] / m_per_px))
                draw_filled_circle(img, px, py, r_px, [80, 80, 80])

        # Draw wall segments as lines
        segs = ws.get("obstacles_segments", None)
        if segs is not None:
            segs = np.asarray(segs)
            for s in range(segs.shape[0]):
                x0, y0 = world_to_pixel(segs[s, 0, 0], segs[s, 0, 1])
                x1, y1 = world_to_pixel(segs[s, 1, 0], segs[s, 1, 1])
                self._draw_line(img, x0, y0, x1, y1, color=[40, 40, 40])

        # Draw robot at centre
        draw_filled_circle(img, W // 2, H // 2,
                           max(2, int(0.2 / m_per_px)), [31, 119, 180])

        return img

    def _render_perspective(self, ws: dict[str, Any]) -> np.ndarray:
        """Simplified perspective projection onto image plane."""
        H, W = self.config.resolution_y, self.config.resolution_x
        img = np.full((H, W, 3), 200, dtype=np.uint8)

        pos = np.asarray(ws["robot_pos"], dtype=np.float64)
        heading = float(ws.get("robot_heading", 0.0))
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        f = self.config.focal_length_px
        half_fov = self.config.fov_horizontal / 2.0

        agents = ws.get("agents", [])
        for agent in agents:
            ap = np.asarray(agent["pos"] if isinstance(agent, dict) else agent[:2],
                            dtype=np.float64)
            ar = agent["radius"] if isinstance(agent, dict) else agent[2]

            # Transform to camera frame
            dx, dy = ap[0] - pos[0], ap[1] - pos[1]
            cam_x = dx * cos_h + dy * sin_h   # forward
            cam_y = -dx * sin_h + dy * cos_h  # right

            if cam_x < EPSILON:
                continue
            angle = np.arctan2(cam_y, cam_x)
            if abs(angle) > half_fov:
                continue

            # Project
            px = int(W / 2 + f * cam_y / cam_x)
            r_px = max(1, int(f * ar / cam_x))
            # Vertical: place at horizon with size proportional to distance
            py = H // 2

            color = agent.get("color", [255, 127, 14]) if isinstance(agent, dict) else [255, 127, 14]
            yy, xx = np.ogrid[:H, :W]
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= r_px ** 2
            img[mask] = color

        return img

    @staticmethod
    def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                   color: list) -> None:
        """Bresenham line drawing on an image array."""
        H, W = img.shape[:2]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < W and 0 <= y0 < H:
                img[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy


# ---------------------------------------------------------------------------
#  DepthSensor
# ---------------------------------------------------------------------------

class DepthSensor(SensorBase):
    """Simulated 1-D depth sensor.

    Casts rays within the horizontal FOV and returns range measurements, similar
    to a LiDAR but limited to a narrower field of view and lower resolution.

    World state keys are the same as :class:`LidarSensor`.
    """

    def __init__(
        self,
        config: DepthSensorConfig | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        config = config or DepthSensorConfig()
        if noise_model is None and config.noise_std > 0:
            noise_model = GaussianNoise(std=config.noise_std)
        super().__init__(config=config, noise_model=noise_model)

        half = self.config.fov_horizontal / 2.0
        self._angles = np.linspace(-half, half, self.config.resolution,
                                   endpoint=False)
        self._cos = np.cos(self._angles)
        self._sin = np.sin(self._angles)

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "shape": (self.config.resolution,),
            "dtype": np.float64,
            "low": self.config.min_range,
            "high": self.config.max_range,
        }

    def _raw_observe(self, world_state: dict[str, Any]) -> np.ndarray:
        # Reuse LiDAR ray-casting logic with depth sensor parameters
        from navirl.sensors.lidar import (
            _ray_circle_intersection,
            _ray_segment_intersection,
        )

        pos = np.asarray(world_state["robot_pos"], dtype=np.float64)
        heading = float(world_state.get("robot_heading", 0.0))

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        world_cos = self._cos * cos_h - self._sin * sin_h
        world_sin = self._cos * sin_h + self._sin * cos_h

        ranges = np.full(self.config.resolution, self.config.max_range,
                         dtype=np.float64)

        # Segments
        segments = world_state.get("obstacles_segments", None)
        if segments is not None:
            segments = np.asarray(segments, dtype=np.float64)
            if segments.ndim == 3 and segments.shape[0] > 0:
                seg_r = _ray_segment_intersection(
                    pos, world_cos, world_sin, segments, self.config.max_range)
                ranges = np.minimum(ranges, seg_r)

        # Circular obstacles
        obs_c = world_state.get("obstacles_circles", None)
        if obs_c is not None:
            centres = np.asarray(obs_c["centres"], dtype=np.float64)
            radii_arr = np.asarray(obs_c["radii"], dtype=np.float64)
            if centres.shape[0] > 0:
                cr = _ray_circle_intersection(
                    pos, world_cos, world_sin, centres, radii_arr,
                    self.config.max_range)
                ranges = np.minimum(ranges, cr)

        # Dynamic agents
        agents = world_state.get("agents", [])
        if len(agents) > 0:
            if isinstance(agents[0], dict):
                centres = np.array([a["pos"] for a in agents], dtype=np.float64)
                radii_arr = np.array([a["radius"] for a in agents], dtype=np.float64)
            else:
                agents_arr = np.asarray(agents, dtype=np.float64)
                centres = agents_arr[:, :2]
                radii_arr = agents_arr[:, 2]
            ar = _ray_circle_intersection(
                pos, world_cos, world_sin, centres, radii_arr,
                self.config.max_range)
            ranges = np.minimum(ranges, ar)

        ranges = np.clip(ranges, self.config.min_range, self.config.max_range)
        return ranges
