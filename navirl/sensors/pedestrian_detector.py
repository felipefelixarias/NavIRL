"""Simulated pedestrian detection and tracking.

Provides :class:`PedestrianDetector` (single-frame person detection with
false-positive / false-negative modelling and occlusion) and
:class:`PedestrianTracker` (multi-frame tracking via Hungarian assignment
and Kalman filtering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.constants import EPSILON
from navirl.sensors.base import NoiseModel, SensorBase

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class PedestrianDetectorConfig:
    """Configuration for the simulated pedestrian detector.

    Parameters
    ----------
    detection_range : float
        Maximum detection distance (metres).
    fov : float
        Field of view (radians, centred on forward direction).
    false_positive_rate : float
        Probability of generating a spurious detection per frame.
    false_negative_rate : float
        Probability of missing a visible agent per frame.
    position_noise_std : float
        Std of additive Gaussian noise on relative position (metres).
    velocity_noise_std : float
        Std of additive Gaussian noise on relative velocity (m/s).
    occlusion_enabled : bool
        If True, agents behind other agents may be missed.
    occlusion_half_angle : float
        Half-angle of the shadow cone cast by an occluder (radians).
    max_false_positives : int
        Upper limit on spurious detections per frame.
    """

    detection_range: float = 10.0
    fov: float = 2.0 * np.pi  # full 360
    false_positive_rate: float = 0.02
    false_negative_rate: float = 0.05
    position_noise_std: float = 0.1
    velocity_noise_std: float = 0.15
    occlusion_enabled: bool = True
    occlusion_half_angle: float = 0.15  # ~8.6 degrees
    max_false_positives: int = 3


# ---------------------------------------------------------------------------
#  PedestrianDetector
# ---------------------------------------------------------------------------

class PedestrianDetector(SensorBase):
    """Simulated person detector that returns relative state of nearby agents.

    World state keys
    ----------------
    * ``robot_pos`` : (2,) ndarray
    * ``robot_vel`` : (2,) ndarray
    * ``robot_heading`` : float (radians)
    * ``agents`` : list of dicts with ``pos`` (2,), ``vel`` (2,), ``radius`` (float).

    Returns
    -------
    list[np.ndarray]
        Each element is a (5,) array:
        ``[relative_x, relative_y, relative_vx, relative_vy, radius]``.
    """

    def __init__(
        self,
        config: PedestrianDetectorConfig | None = None,
        noise_model: NoiseModel | None = None,
    ) -> None:
        self._cfg = config or PedestrianDetectorConfig()
        super().__init__(config=self._cfg, noise_model=noise_model)

    # -- SensorBase interface ------------------------------------------------

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "shape": ("variable", 5),
            "dtype": np.float64,
            "low": np.array([-self.config.detection_range,
                             -self.config.detection_range,
                             -10.0, -10.0, 0.0]),
            "high": np.array([self.config.detection_range,
                              self.config.detection_range,
                              10.0, 10.0, 1.0]),
        }

    def _raw_observe(self, world_state: dict[str, Any]) -> list[np.ndarray]:
        cfg = self._cfg
        pos = np.asarray(world_state["robot_pos"], dtype=np.float64)
        vel = np.asarray(world_state.get("robot_vel", [0.0, 0.0]),
                         dtype=np.float64)
        heading = float(world_state.get("robot_heading", 0.0))
        agents = world_state.get("agents", [])

        if len(agents) == 0:
            return self._maybe_add_false_positives([])

        # Build arrays
        if isinstance(agents[0], dict):
            positions = np.array([a["pos"] for a in agents], dtype=np.float64)
            velocities = np.array([a.get("vel", [0, 0]) for a in agents],
                                  dtype=np.float64)
            radii = np.array([a.get("radius", 0.25) for a in agents],
                             dtype=np.float64)
        else:
            arr = np.asarray(agents, dtype=np.float64)
            positions = arr[:, :2]
            velocities = arr[:, 2:4] if arr.shape[1] >= 4 else np.zeros((len(arr), 2))
            radii = arr[:, 4] if arr.shape[1] >= 5 else np.full(len(arr), 0.25)

        # Relative positions
        rel_pos = positions - pos
        dists = np.linalg.norm(rel_pos, axis=1)

        # Filter by range
        in_range = dists < cfg.detection_range

        # Filter by FOV
        angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0]) - heading
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        half_fov = cfg.fov / 2.0
        in_fov = np.abs(angles) < half_fov

        visible = in_range & in_fov
        visible_indices = np.where(visible)[0]

        # Sort by distance for occlusion check
        sorted_idx = visible_indices[np.argsort(dists[visible_indices])]

        # Occlusion filtering
        if cfg.occlusion_enabled and len(sorted_idx) > 1:
            unoccluded = []
            shadow_cones: list[tuple[float, float]] = []  # (angle, half_angle)
            for idx in sorted_idx:
                ang = np.arctan2(rel_pos[idx, 1], rel_pos[idx, 0]) - heading
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                occluded = False
                for cone_ang, cone_half in shadow_cones:
                    if abs(ang - cone_ang) < cone_half:
                        occluded = True
                        break
                if not occluded:
                    unoccluded.append(idx)
                    # Agent casts a shadow cone
                    d = dists[idx]
                    if d > EPSILON:
                        apparent_half = np.arctan2(radii[idx], d) + cfg.occlusion_half_angle
                        shadow_cones.append((ang, apparent_half))
            sorted_idx = np.array(unoccluded, dtype=int)

        # Build detections
        detections: list[np.ndarray] = []
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)

        for idx in sorted_idx:
            # False negative: randomly miss this agent
            if self._rng.random() < cfg.false_negative_rate:
                continue

            # Relative position in robot frame
            dx, dy = rel_pos[idx, 0], rel_pos[idx, 1]
            rx = dx * cos_h - dy * sin_h
            ry = dx * sin_h + dy * cos_h

            # Relative velocity in robot frame
            rel_v = velocities[idx] - vel
            rvx = rel_v[0] * cos_h - rel_v[1] * sin_h
            rvy = rel_v[0] * sin_h + rel_v[1] * cos_h

            # Add noise
            rx += self._rng.normal(0, cfg.position_noise_std)
            ry += self._rng.normal(0, cfg.position_noise_std)
            rvx += self._rng.normal(0, cfg.velocity_noise_std)
            rvy += self._rng.normal(0, cfg.velocity_noise_std)

            detections.append(
                np.array([rx, ry, rvx, rvy, radii[idx]], dtype=np.float64)
            )

        return self._maybe_add_false_positives(detections)

    def observe(self, world_state: dict[str, Any]) -> list[np.ndarray]:
        """Override base to skip generic noise (noise applied internally)."""
        return self._raw_observe(world_state)

    def _maybe_add_false_positives(
        self, detections: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Randomly inject false-positive detections."""
        cfg = self._cfg
        n_fp = 0
        for _ in range(cfg.max_false_positives):
            if self._rng.random() < cfg.false_positive_rate:
                n_fp += 1

        for _ in range(n_fp):
            dist = self._rng.uniform(1.0, cfg.detection_range)
            angle = self._rng.uniform(-cfg.fov / 2, cfg.fov / 2)
            rx = dist * np.cos(angle)
            ry = dist * np.sin(angle)
            rvx = self._rng.normal(0, 0.5)
            rvy = self._rng.normal(0, 0.5)
            radius = self._rng.uniform(0.18, 0.35)
            detections.append(
                np.array([rx, ry, rvx, rvy, radius], dtype=np.float64)
            )
        return detections


# ---------------------------------------------------------------------------
#  PedestrianTracker
# ---------------------------------------------------------------------------

class PedestrianTracker:
    """Multi-frame pedestrian tracker using Hungarian assignment and Kalman
    filtering.

    Maintains a set of active tracks, each with an integer ID, a Kalman-filter
    state estimate, and a confidence score.  Tracks are created, updated, and
    deleted automatically as detections arrive each frame.

    Parameters
    ----------
    max_age : int
        Number of consecutive missed frames before a track is deleted.
    min_hits : int
        Minimum consecutive hits before a track is considered confirmed.
    distance_threshold : float
        Maximum Mahalanobis / Euclidean distance for assignment (metres).
    dt : float
        Time step between frames (seconds).
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 2,
        distance_threshold: float = 2.0,
        dt: float = 0.04,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.dt = dt

        self._next_id: int = 0
        self._tracks: dict[int, _Track] = {}

    def reset(self) -> None:
        """Clear all tracks."""
        self._next_id = 0
        self._tracks.clear()

    def update(
        self, detections: list[np.ndarray]
    ) -> list[dict[str, Any]]:
        """Process a new frame of detections and return active tracks.

        Parameters
        ----------
        detections : list of (5,) arrays
            Each ``[rx, ry, rvx, rvy, radius]`` from :class:`PedestrianDetector`.

        Returns
        -------
        list of dict
            Each dict has keys: ``id``, ``state`` (4-vector [x, y, vx, vy]),
            ``radius``, ``confidence``, ``age``, ``hits``.
        """
        # Predict all existing tracks forward
        for track in self._tracks.values():
            track.predict(self.dt)

        # Build cost matrix (Euclidean distance between predicted positions
        # and detection positions)
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        if n_tracks == 0 and n_dets == 0:
            return self._get_confirmed_tracks()

        if n_tracks > 0 and n_dets > 0:
            cost = np.zeros((n_tracks, n_dets), dtype=np.float64)
            for i, tid in enumerate(track_ids):
                pred = self._tracks[tid].state[:2]
                for j, det in enumerate(detections):
                    cost[i, j] = np.linalg.norm(pred - det[:2])

            # Hungarian assignment
            row_ind, col_ind = self._hungarian(cost)

            matched_tracks = set()
            matched_dets = set()

            for r, c in zip(row_ind, col_ind, strict=False):
                if cost[r, c] < self.distance_threshold:
                    tid = track_ids[r]
                    self._tracks[tid].update_with_detection(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

            # Unmatched tracks: increment age
            for i in range(n_tracks):
                if i not in matched_tracks:
                    self._tracks[track_ids[i]].mark_missed()

            # Unmatched detections: create new tracks
            for j in range(n_dets):
                if j not in matched_dets:
                    self._create_track(detections[j])
        elif n_dets > 0:
            for det in detections:
                self._create_track(det)
        else:
            for track in self._tracks.values():
                track.mark_missed()

        # Remove dead tracks
        dead = [tid for tid, t in self._tracks.items()
                if t.time_since_update > self.max_age]
        for tid in dead:
            del self._tracks[tid]

        return self._get_confirmed_tracks()

    # -- Internal helpers ----------------------------------------------------

    def _create_track(self, detection: np.ndarray) -> None:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = _Track(
            track_id=tid,
            state=detection[:4].copy(),
            radius=detection[4],
        )

    def _get_confirmed_tracks(self) -> list[dict[str, Any]]:
        results = []
        for tid, t in self._tracks.items():
            if t.hits >= self.min_hits or t.time_since_update == 0:
                results.append({
                    "id": tid,
                    "state": t.state.copy(),
                    "radius": t.radius,
                    "confidence": t.confidence,
                    "age": t.age,
                    "hits": t.hits,
                })
        return results

    @staticmethod
    def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the linear assignment problem (Hungarian algorithm).

        Uses a simple greedy fallback if scipy is unavailable.
        """
        try:
            from scipy.optimize import linear_sum_assignment
            return linear_sum_assignment(cost)
        except ImportError:
            pass

        # Greedy fallback: assign lowest-cost pairs iteratively
        n, m = cost.shape
        rows, cols = [], []
        used_r, used_c = set(), set()
        flat = np.argsort(cost, axis=None)
        for idx in flat:
            r, c = divmod(int(idx), m)
            if r not in used_r and c not in used_c:
                rows.append(r)
                cols.append(c)
                used_r.add(r)
                used_c.add(c)
            if len(rows) == min(n, m):
                break
        return np.array(rows), np.array(cols)


# ---------------------------------------------------------------------------
#  Internal track representation with simple Kalman filter
# ---------------------------------------------------------------------------

class _Track:
    """Single tracked pedestrian with constant-velocity Kalman filter.

    State vector: [x, y, vx, vy].
    """

    def __init__(self, track_id: int, state: np.ndarray,
                 radius: float) -> None:
        self.track_id = track_id
        self.state = state.copy()  # (4,)
        self.radius = radius
        self.age: int = 0
        self.hits: int = 1
        self.time_since_update: int = 0

        # Kalman filter covariance
        self.P = np.eye(4, dtype=np.float64) * 1.0
        # Process noise
        self.Q = np.diag([0.05, 0.05, 0.1, 0.1]).astype(np.float64)
        # Measurement noise
        self.R = np.diag([0.1, 0.1, 0.15, 0.15]).astype(np.float64)

    @property
    def confidence(self) -> float:
        """Heuristic confidence score in [0, 1]."""
        return min(1.0, self.hits / 5.0) * max(0.0, 1.0 - self.time_since_update / 5.0)

    def predict(self, dt: float) -> None:
        """Constant-velocity prediction step."""
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        self.age += 1

    def update_with_detection(self, detection: np.ndarray) -> None:
        """Kalman update with a matched detection."""
        H = np.eye(4, dtype=np.float64)  # observe full state
        z = detection[:4]
        y = z - H @ self.state  # innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.radius = detection[4]
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1
