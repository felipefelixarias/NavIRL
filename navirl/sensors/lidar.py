"""2-D planar LiDAR sensor simulation.

Provides :class:`LidarSensor` which performs vectorised ray casting against
an obstacle map (line segments) and circular agents, returning a range array
with optional noise.

Uses pre-computed sin/cos tables from :mod:`navirl.core.constants` for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.constants import (
    EPSILON,
    LIDAR,
)
from navirl.sensors.base import GaussianNoise, NoiseModel, SensorBase

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class LidarConfig:
    """Configuration for the 2-D planar LiDAR sensor.

    Defaults are pulled from :data:`navirl.core.constants.LIDAR`.
    """

    num_beams: int = LIDAR.num_beams
    max_range: float = LIDAR.max_range
    min_range: float = LIDAR.min_range
    fov: float = LIDAR.fov
    angular_resolution: float = LIDAR.angular_resolution
    noise_std: float = LIDAR.noise_std
    num_sectors: int = LIDAR.num_sectors

    @property
    def angles(self) -> np.ndarray:
        """Beam angles centred on the forward direction."""
        half = self.fov / 2.0
        return np.linspace(-half, half, self.num_beams, endpoint=False)


# ---------------------------------------------------------------------------
#  Ray-geometry intersection helpers (vectorised)
# ---------------------------------------------------------------------------


def _ray_circle_intersection(
    origin: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Vectorised ray-circle intersection for all beams against all circles.

    Parameters
    ----------
    origin : (2,) array
        Sensor position in world frame.
    cos_table, sin_table : (B,) arrays
        Pre-computed cos/sin of beam angles (in world frame).
    centres : (N, 2) array
        Circle centre positions.
    radii : (N,) array
        Circle radii.
    max_range : float
        Maximum sensor range (used as initial distance).

    Returns
    -------
    ranges : (B,) array
        Closest intersection distance per beam.
    """
    num_beams = cos_table.shape[0]
    ranges = np.full(num_beams, max_range, dtype=np.float64)

    if centres.shape[0] == 0:
        return ranges

    # Direction vectors: (B, 2)
    dirs = np.stack([cos_table, sin_table], axis=-1)

    # Offset from origin to each circle centre: (N, 2)
    oc = centres - origin  # (N, 2)

    # Project oc onto each ray direction:
    #   oc (N, 2) @ dirs.T (2, B) -> (N, B)
    proj = oc @ dirs.T  # (N, B)

    # Perpendicular distance squared from ray to centre:
    #   |oc|^2 - proj^2
    oc_sq = np.sum(oc**2, axis=1, keepdims=True)  # (N, 1)
    perp_sq = oc_sq - proj**2  # (N, B)

    radii_sq = (radii**2)[:, np.newaxis]  # (N, 1)

    # Discriminant: positive means the ray intersects the circle
    disc = radii_sq - perp_sq  # (N, B)

    # Closest intersection: t = proj - sqrt(disc)
    # We want only positive t (ray goes forward)
    valid = disc > 0
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    t_hit = proj - sqrt_disc  # (N, B)

    # Mask out negative t (behind the sensor) and invalid intersections
    t_hit[~valid] = max_range
    t_hit[t_hit < 0] = max_range

    # Minimum across all circles for each beam
    min_t = np.min(t_hit, axis=0)  # (B,)
    ranges = np.minimum(ranges, min_t)

    return ranges


def _ray_segment_intersection(
    origin: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    segments: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Vectorised ray-segment intersection for all beams against all segments.

    Parameters
    ----------
    origin : (2,) array
        Sensor position.
    cos_table, sin_table : (B,) arrays
        Pre-computed cos/sin of beam angles (world frame).
    segments : (S, 2, 2) array
        Line segments, each defined by two endpoints ``[[x0, y0], [x1, y1]]``.
    max_range : float
        Maximum sensor range.

    Returns
    -------
    ranges : (B,) array
        Closest intersection distance per beam.
    """
    num_beams = cos_table.shape[0]
    ranges = np.full(num_beams, max_range, dtype=np.float64)

    if segments.shape[0] == 0:
        return ranges

    # Ray directions: (B, 2)
    dx = cos_table  # (B,)
    dy = sin_table  # (B,)

    p0 = segments[:, 0, :]  # (S, 2)
    p1 = segments[:, 1, :]  # (S, 2)
    seg_d = p1 - p0  # (S, 2)

    for si in range(segments.shape[0]):
        sx, sy = seg_d[si, 0], seg_d[si, 1]
        ox, oy = p0[si, 0] - origin[0], p0[si, 1] - origin[1]

        # Solve:  origin + t * dir == p0 + u * seg_d
        #   t * dx - u * sx == ox  ... wait, rearranged:
        #   ox = p0.x - origin.x
        #   oy = p0.y - origin.y
        #   t * dx = ox + u * sx  =>  t*dx - u*sx = ox  ... nope
        # Standard form: origin + t*d = p0 + u*s
        #   t*d - u*s = p0 - origin
        # denom = dx * sy - dy * sx  (per beam)
        denom = dx * sy - dy * sx  # (B,)
        parallel = np.abs(denom) < EPSILON

        # t = (ox * sy - oy * sx) / denom
        # u = (ox * dy - oy * dx) / denom
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ox * sy - oy * sx) / denom
            u = (ox * dy - oy * dx) / denom

        hit = (~parallel) & (t > 0) & (t < ranges) & (u >= 0) & (u <= 1)
        ranges[hit] = t[hit]

    return ranges


# ---------------------------------------------------------------------------
#  LidarSensor
# ---------------------------------------------------------------------------


class LidarSensor(SensorBase):
    """Simulated 2-D planar LiDAR.

    Performs ray casting against an obstacle map (line segments) and circular
    agents, returning a 1-D array of range measurements.

    World state dictionary is expected to contain:

    * ``robot_pos`` : (2,) ndarray -- sensor position [x, y].
    * ``robot_heading`` : float -- sensor heading in radians.
    * ``agents`` : list of dicts with keys ``pos`` (2,), ``radius`` (float).
      The robot itself should **not** appear in this list.
    * ``obstacles_segments`` : (S, 2, 2) ndarray of wall segments (optional).
    * ``obstacles_circles`` : dict with ``centres`` (N, 2) and ``radii`` (N,)
      arrays (optional).
    """

    def __init__(
        self,
        config: LidarConfig | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        config = config or LidarConfig()
        if noise_model is None and config.noise_std > 0:
            noise_model = GaussianNoise(std=config.noise_std)
        super().__init__(config=config, noise_model=noise_model)

        # Pre-compute angle tables for this configuration
        self._angles = self.config.angles
        self._cos = np.cos(self._angles)
        self._sin = np.sin(self._angles)

    # -- SensorBase interface ------------------------------------------------

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "shape": (self.config.num_beams,),
            "dtype": np.float64,
            "low": self.config.min_range,
            "high": self.config.max_range,
        }

    def _raw_observe(self, world_state: dict[str, Any]) -> np.ndarray:
        pos = np.asarray(world_state["robot_pos"], dtype=np.float64)
        heading = float(world_state.get("robot_heading", 0.0))

        # Rotate pre-computed tables by the current heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        world_cos = self._cos * cos_h - self._sin * sin_h
        world_sin = self._cos * sin_h + self._sin * cos_h

        ranges = np.full(self.config.num_beams, self.config.max_range, dtype=np.float64)

        # --- Obstacle segments (walls) ---
        segments = world_state.get("obstacles_segments")
        if segments is not None:
            segments = np.asarray(segments, dtype=np.float64)
            if segments.ndim == 3 and segments.shape[0] > 0:
                seg_ranges = _ray_segment_intersection(
                    pos,
                    world_cos,
                    world_sin,
                    segments,
                    self.config.max_range,
                )
                ranges = np.minimum(ranges, seg_ranges)

        # --- Static circular obstacles ---
        obs_circles = world_state.get("obstacles_circles")
        if obs_circles is not None:
            centres = np.asarray(obs_circles["centres"], dtype=np.float64)
            radii = np.asarray(obs_circles["radii"], dtype=np.float64)
            if centres.shape[0] > 0:
                circ_ranges = _ray_circle_intersection(
                    pos,
                    world_cos,
                    world_sin,
                    centres,
                    radii,
                    self.config.max_range,
                )
                ranges = np.minimum(ranges, circ_ranges)

        # --- Dynamic agents (pedestrians) ---
        agents = world_state.get("agents", [])
        if len(agents) > 0:
            if isinstance(agents[0], dict):
                centres = np.array([a["pos"] for a in agents], dtype=np.float64)
                radii = np.array([a["radius"] for a in agents], dtype=np.float64)
            else:
                # Allow (N, 3) array: [x, y, radius]
                agents_arr = np.asarray(agents, dtype=np.float64)
                centres = agents_arr[:, :2]
                radii = agents_arr[:, 2]
            agent_ranges = _ray_circle_intersection(
                pos,
                world_cos,
                world_sin,
                centres,
                radii,
                self.config.max_range,
            )
            ranges = np.minimum(ranges, agent_ranges)

        # Clamp to sensor limits
        ranges = np.clip(ranges, self.config.min_range, self.config.max_range)
        return ranges

    # -- Convenience ---------------------------------------------------------

    def get_sector_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """Downsample *ranges* into coarser sectors (min within each sector).

        Returns
        -------
        (num_sectors,) ndarray
        """
        n = self.config.num_sectors
        beams_per_sector = self.config.num_beams // n
        trimmed = ranges[: n * beams_per_sector]
        sectors = trimmed.reshape(n, beams_per_sector)
        return sectors.min(axis=1)

    def ranges_to_cartesian(self, ranges: np.ndarray, heading: float = 0.0) -> np.ndarray:
        """Convert range measurements to 2-D Cartesian points (sensor frame).

        Parameters
        ----------
        ranges : (B,) array
        heading : float
            Sensor heading in radians (0 = forward along x-axis).

        Returns
        -------
        points : (B, 2) array
        """
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        world_cos = self._cos * cos_h - self._sin * sin_h
        world_sin = self._cos * sin_h + self._sin * cos_h
        x = ranges * world_cos
        y = ranges * world_sin
        return np.stack([x, y], axis=-1)
