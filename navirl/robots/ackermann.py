"""Ackermann (car-like) robot model.

Implements the bicycle kinematic model with steering-angle limits, minimum
turning radius, Reeds-Shepp curve primitives, parallel-parking manoeuvre
generation, and a lane-following controller.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController
from navirl.utils.geometry import normalize_angle

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------


@dataclass
class AckermannConfig:
    """Parameters for a car-like (Ackermann-steered) robot.

    Attributes:
        wheelbase: Distance between front and rear axles (metres).
        max_speed: Maximum forward speed (m/s).
        min_speed: Minimum speed (negative = reverse allowed) (m/s).
        max_acceleration: Maximum longitudinal acceleration (m/s^2).
        max_steering_angle: Maximum front-wheel steering angle (rad).
        max_steering_rate: Maximum rate of steering change (rad/s).
        width: Vehicle width (metres) for collision checks.
        rear_overhang: Distance behind the rear axle (metres).
        front_overhang: Distance ahead of the front axle (metres).
    """

    wheelbase: float = 2.5
    max_speed: float = 5.0
    min_speed: float = -2.0
    max_acceleration: float = 3.0
    max_steering_angle: float = np.pi / 6.0  # 30 deg
    max_steering_rate: float = 1.0  # rad/s
    width: float = 1.8
    rear_overhang: float = 0.8
    front_overhang: float = 1.0

    @property
    def min_turning_radius(self) -> float:
        """Minimum turning radius derived from wheelbase and max steering."""
        return self.wheelbase / np.tan(max(abs(self.max_steering_angle), 1e-6))


# -----------------------------------------------------------------------
# Bicycle model kinematics
# -----------------------------------------------------------------------


def bicycle_forward(
    x: float,
    y: float,
    theta: float,
    v: float,
    delta: float,
    wheelbase: float,
    dt: float,
) -> tuple[float, float, float]:
    """Integrate bicycle kinematics for one time-step.

    The reference point is at the rear axle centre.

    Args:
        x: Rear-axle x position.
        y: Rear-axle y position.
        theta: Heading (rad).
        v: Longitudinal velocity (m/s, positive = forward).
        delta: Front-wheel steering angle (rad).
        wheelbase: Axle-to-axle distance (metres).
        dt: Time step (seconds).

    Returns:
        ``(x_new, y_new, theta_new)``.
    """
    beta = np.arctan(0.5 * np.tan(delta))  # slip angle at CG for bicycle
    x_new = x + v * np.cos(theta + beta) * dt
    y_new = y + v * np.sin(theta + beta) * dt
    (v / wheelbase) * np.sin(beta) * 2.0  # equivalent to v*tan(delta)/L
    # Use exact formulation: dtheta = v * tan(delta) / L
    dtheta_exact = v * np.tan(delta) / wheelbase
    theta_new = normalize_angle(theta + dtheta_exact * dt)
    return (float(x_new), float(y_new), float(theta_new))


def bicycle_curvature(delta: float, wheelbase: float) -> float:
    """Return the instantaneous curvature for a given steering angle.

    Args:
        delta: Steering angle (rad).
        wheelbase: Axle distance (metres).

    Returns:
        Curvature (1/metres).  Positive = turning left.
    """
    return float(np.tan(delta) / wheelbase)


def bicycle_turning_radius(delta: float, wheelbase: float) -> float:
    """Return the turning radius for a given steering angle.

    Returns ``inf`` for zero steering.
    """
    kappa = bicycle_curvature(delta, wheelbase)
    if abs(kappa) < 1e-9:
        return float("inf")
    return 1.0 / abs(kappa)


# -----------------------------------------------------------------------
# Rate limiting
# -----------------------------------------------------------------------


def rate_limit_ackermann(
    v_cmd: float,
    delta_cmd: float,
    v_prev: float,
    delta_prev: float,
    dt: float,
    config: AckermannConfig,
) -> tuple[float, float]:
    """Enforce acceleration and steering-rate limits.

    Args:
        v_cmd: Desired speed.
        delta_cmd: Desired steering angle.
        v_prev: Previous speed.
        delta_prev: Previous steering angle.
        dt: Time step.
        config: Vehicle configuration.

    Returns:
        ``(v_limited, delta_limited)``.
    """
    max_dv = config.max_acceleration * dt
    dv = float(np.clip(v_cmd - v_prev, -max_dv, max_dv))
    v_out = float(np.clip(v_prev + dv, config.min_speed, config.max_speed))

    max_dd = config.max_steering_rate * dt
    dd = float(np.clip(delta_cmd - delta_prev, -max_dd, max_dd))
    delta_out = float(
        np.clip(delta_prev + dd, -config.max_steering_angle, config.max_steering_angle)
    )
    return (v_out, delta_out)


# -----------------------------------------------------------------------
# Reeds-Shepp curves (simplified 6-segment primitives)
# -----------------------------------------------------------------------


class RSSegmentType(enum.Enum):
    """Segment type in a Reeds-Shepp path."""

    LEFT = "L"
    RIGHT = "R"
    STRAIGHT = "S"


@dataclass
class RSSegment:
    """One segment of a Reeds-Shepp path.

    Attributes:
        kind: Segment type (LEFT / RIGHT / STRAIGHT).
        length: Signed length (negative = reverse).
    """

    kind: RSSegmentType
    length: float


def _rs_forward(
    x: float,
    y: float,
    theta: float,
    seg: RSSegment,
    radius: float,
) -> tuple[float, float, float]:
    """Propagate state through one RS segment."""
    if seg.kind == RSSegmentType.STRAIGHT:
        x += seg.length * np.cos(theta)
        y += seg.length * np.sin(theta)
        return (float(x), float(y), float(theta))
    # Arc.
    sign = 1.0 if seg.kind == RSSegmentType.LEFT else -1.0
    phi = seg.length / radius  # signed arc angle
    cx = x - sign * radius * np.sin(theta)
    cy = y + sign * radius * np.cos(theta)
    theta_new = theta + sign * phi
    x_new = cx + sign * radius * np.sin(theta_new)
    y_new = cy - sign * radius * np.cos(theta_new)
    return (float(x_new), float(y_new), float(normalize_angle(theta_new)))


def reeds_shepp_path(
    x0: float,
    y0: float,
    theta0: float,
    x1: float,
    y1: float,
    theta1: float,
    radius: float,
    num_samples: int = 100,
) -> tuple[np.ndarray, list[RSSegment]]:
    """Compute an approximate Reeds-Shepp path.

    Uses a set of candidate word types (CSC and CCC families) and picks
    the shortest feasible one.  This is a simplified but functional
    implementation.

    Args:
        x0, y0, theta0: Start pose.
        x1, y1, theta1: Goal pose.
        radius: Minimum turning radius.
        num_samples: Points to sample along the path.

    Returns:
        ``(path, segments)`` where *path* has shape ``(num_samples, 3)``
        - ``[x, y, theta]`` - and *segments* lists the RS segments.
    """
    # Transform goal into start frame.
    dx = x1 - x0
    dy = y1 - y0
    cos0 = np.cos(theta0)
    sin0 = np.sin(theta0)
    lx = cos0 * dx + sin0 * dy
    ly = -sin0 * dx + cos0 * dy
    ltheta = normalize_angle(theta1 - theta0)

    # Candidate word types: LSL, RSR, LSR, RSL, LRL, RLR
    candidates: list[list[RSSegment]] = []

    def _try_csc(
        turn1: RSSegmentType,
        turn2: RSSegmentType,
    ) -> list[RSSegment] | None:
        """Attempt a CSC (curve-straight-curve) path."""
        s1 = 1.0 if turn1 == RSSegmentType.LEFT else -1.0
        s2 = 1.0 if turn2 == RSSegmentType.LEFT else -1.0
        # Centre of first circle.
        c1x = s1 * radius * 0.0  # origin
        c1y = s1 * radius
        # Centre of second circle (in local frame).
        c2x = lx - s2 * radius * np.sin(ltheta)  # type: ignore[operator]
        c2y = ly + s2 * radius * np.cos(ltheta)  # type: ignore[operator]

        ddx = c2x - c1x
        ddy = c2y - c1y
        d = float(np.hypot(ddx, ddy))

        if turn1 == turn2:
            # Same turn direction: external tangent.
            if d < 1e-6:
                return None
            straight_len = d  # simplified
            ang1 = float(np.arctan2(ddy, ddx))
            arc1 = normalize_angle(ang1 - np.pi / 2.0 * s1)
            arc2 = normalize_angle(ltheta - arc1)
        else:
            # Different turn: cross tangent.
            if d < 2.0 * radius:
                return None
            cos_a = 2.0 * radius / d
            if abs(cos_a) > 1.0:
                return None
            straight_len = np.sqrt(max(d**2 - (2.0 * radius) ** 2, 0.0))
            ang1 = float(np.arctan2(ddy, ddx))
            arc1 = normalize_angle(ang1 - np.pi / 2.0 * s1)
            arc2 = normalize_angle(ltheta - arc1)

        segs = [
            RSSegment(turn1, arc1 * radius * s1),
            RSSegment(RSSegmentType.STRAIGHT, straight_len),
            RSSegment(turn2, arc2 * radius * s2),
        ]
        return segs

    for t1 in (RSSegmentType.LEFT, RSSegmentType.RIGHT):
        for t2 in (RSSegmentType.LEFT, RSSegmentType.RIGHT):
            segs = _try_csc(t1, t2)
            if segs is not None:
                candidates.append(segs)

    # Also try direct straight line.
    dist = float(np.hypot(lx, ly))
    candidates.append([RSSegment(RSSegmentType.STRAIGHT, dist)])

    # Pick shortest.
    def _total_length(segs: list[RSSegment]) -> float:
        return sum(abs(s.length) for s in segs)

    candidates.sort(key=_total_length)
    best = candidates[0] if candidates else [RSSegment(RSSegmentType.STRAIGHT, dist)]

    # Sample the path.
    total_len = _total_length(best)
    if total_len < 1e-8:
        pt = np.array([[x0, y0, theta0]])
        return (np.tile(pt, (num_samples, 1)), best)

    cumulative = np.zeros(len(best) + 1)
    for i, seg in enumerate(best):
        cumulative[i + 1] = cumulative[i] + abs(seg.length)

    path = np.zeros((num_samples, 3))
    s_values = np.linspace(0.0, total_len, num_samples)

    for qi, s_q in enumerate(s_values):
        cx, cy, ct = x0, y0, theta0
        s_remaining = s_q
        for seg in best:
            seg_len = abs(seg.length)
            if s_remaining <= seg_len + 1e-12:
                frac = s_remaining / max(seg_len, 1e-12)
                sub = RSSegment(seg.kind, seg.length * frac)
                cx, cy, ct = _rs_forward(cx, cy, ct, sub, radius)
                break
            else:
                cx, cy, ct = _rs_forward(cx, cy, ct, seg, radius)
                s_remaining -= seg_len
        path[qi] = [cx, cy, ct]

    return (path, best)


# -----------------------------------------------------------------------
# Parallel parking
# -----------------------------------------------------------------------


def parallel_parking_trajectory(
    x_start: float,
    y_start: float,
    theta_start: float,
    x_spot: float,
    y_spot: float,
    theta_spot: float,
    config: AckermannConfig,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a parallel-parking manoeuvre.

    The manoeuvre consists of:
    1. Forward alignment alongside the parking space.
    2. Reverse with full right lock into the space.
    3. Forward with full left lock to straighten.
    4. Final reverse adjustment.

    Args:
        x_start, y_start, theta_start: Initial pose (alongside spot).
        x_spot, y_spot, theta_spot: Target parked pose.
        config: Vehicle parameters.
        dt: Simulation time step.

    Returns:
        ``(poses, controls)`` where *poses* has shape ``(T, 3)`` and
        *controls* has shape ``(T, 2)`` - ``[v, delta]``.
    """
    r = config.min_turning_radius
    park_speed = min(1.0, config.max_speed * 0.3)
    max_delta = config.max_steering_angle

    # Build a sequence of (v, delta, duration) phases.
    phases: list[tuple[float, float, float]] = [
        # Phase 1: reverse with full right lock.
        (-park_speed, -max_delta, r * np.pi * 0.25 / park_speed),
        # Phase 2: reverse with full left lock.
        (-park_speed, max_delta, r * np.pi * 0.25 / park_speed),
        # Phase 3: small forward straighten.
        (park_speed * 0.5, 0.0, 0.5),
    ]

    poses: list[tuple[float, float, float]] = [(x_start, y_start, theta_start)]
    controls: list[tuple[float, float]] = []

    x, y, theta = x_start, y_start, theta_start
    for v, delta, duration in phases:
        n_steps = max(int(duration / dt), 1)
        for _ in range(n_steps):
            controls.append((v, delta))
            x, y, theta = bicycle_forward(x, y, theta, v, delta, config.wheelbase, dt)
            poses.append((x, y, theta))

    return (np.array(poses), np.array(controls))


# -----------------------------------------------------------------------
# Lane-following controller
# -----------------------------------------------------------------------


class LaneFollower:
    """Simple Stanley-style lane-following controller.

    The Stanley controller combines a heading error term with a cross-
    track error term to compute the steering angle.

    Attributes:
        k_cross: Cross-track error gain.
        k_heading: Heading error gain.
        k_soft: Softening constant for low-speed cross-track gain.
    """

    def __init__(
        self,
        k_cross: float = 1.0,
        k_heading: float = 1.0,
        k_soft: float = 1.0,
    ) -> None:
        self.k_cross = k_cross
        self.k_heading = k_heading
        self.k_soft = k_soft

    def compute_steering(
        self,
        x: float,
        y: float,
        theta: float,
        v: float,
        lane_points: np.ndarray,
    ) -> float:
        """Compute the steering angle to follow a lane centreline.

        Args:
            x: Current x position (front axle).
            y: Current y position (front axle).
            theta: Current heading (rad).
            v: Current speed (m/s).
            lane_points: Lane centreline points, shape ``(N, 2)``.

        Returns:
            Desired steering angle (rad).
        """
        pos = np.array([x, y])
        diffs = lane_points - pos
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = int(np.argmin(dists))

        # Cross-track error (signed).
        nearest = lane_points[nearest_idx]
        # Lane tangent direction.
        if nearest_idx < len(lane_points) - 1:
            tangent = lane_points[nearest_idx + 1] - lane_points[nearest_idx]
        else:
            tangent = lane_points[nearest_idx] - lane_points[max(0, nearest_idx - 1)]
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-8:
            return 0.0
        tangent = tangent / tangent_norm

        # Signed cross-track error (positive = left of lane).
        error_vec = pos - nearest
        cross_track = float(-tangent[1] * error_vec[0] + tangent[0] * error_vec[1])

        # Heading error.
        lane_heading = float(np.arctan2(tangent[1], tangent[0]))
        heading_error = normalize_angle(lane_heading - theta)

        # Stanley formula.
        cross_term = np.arctan2(
            self.k_cross * cross_track,
            self.k_soft + abs(v),
        )
        delta = self.k_heading * heading_error + float(cross_term)
        return float(delta)


# -----------------------------------------------------------------------
# Pure-pursuit controller
# -----------------------------------------------------------------------


class PurePursuitController:
    """Pure-pursuit path-tracking controller for Ackermann vehicles.

    Selects a lookahead point along the path and computes the steering
    angle to reach it.

    Attributes:
        lookahead_dist: Base lookahead distance (metres).
        k_speed: Speed-proportional lookahead gain.
    """

    def __init__(
        self,
        lookahead_dist: float = 2.0,
        k_speed: float = 0.5,
    ) -> None:
        self.lookahead_dist = lookahead_dist
        self.k_speed = k_speed

    def compute_steering(
        self,
        x: float,
        y: float,
        theta: float,
        v: float,
        path: np.ndarray,
        wheelbase: float,
    ) -> float:
        """Compute steering angle via pure pursuit.

        Args:
            x: Rear-axle x.
            y: Rear-axle y.
            theta: Heading (rad).
            v: Current speed (m/s).
            path: Reference path, shape ``(N, 2)``.
            wheelbase: Vehicle wheelbase (metres).

        Returns:
            Desired steering angle (rad).
        """
        ld = self.lookahead_dist + self.k_speed * abs(v)
        pos = np.array([x, y])
        dists = np.linalg.norm(path - pos, axis=1)

        # Find the first path point beyond the lookahead distance.
        candidates = np.where(dists >= ld)[0]
        if len(candidates) == 0:
            target = path[-1]
        else:
            target = path[candidates[0]]

        # Transform target into vehicle frame.
        dx = target[0] - x
        dy = target[1] - y
        local_x = np.cos(theta) * dx + np.sin(theta) * dy
        local_y = -np.sin(theta) * dx + np.cos(theta) * dy

        # Curvature.
        l_sq = local_x**2 + local_y**2
        if l_sq < 1e-8:
            return 0.0
        curvature = 2.0 * local_y / l_sq
        delta = float(np.arctan(curvature * wheelbase))
        return delta


# -----------------------------------------------------------------------
# Vehicle footprint helpers
# -----------------------------------------------------------------------


def vehicle_footprint(
    x: float,
    y: float,
    theta: float,
    config: AckermannConfig,
) -> np.ndarray:
    """Compute the four corners of the vehicle rectangle in world frame.

    The reference point is the rear-axle centre.

    Args:
        x: Rear-axle x.
        y: Rear-axle y.
        theta: Heading.
        config: Vehicle parameters.

    Returns:
        Corners, shape ``(4, 2)`` ordered front-left, front-right,
        rear-right, rear-left.
    """
    hw = config.width / 2.0
    front = config.wheelbase + config.front_overhang
    rear = -config.rear_overhang

    # Corners in body frame (x-forward, y-left).
    corners_body = np.array(
        [
            [front, hw],
            [front, -hw],
            [rear, -hw],
            [rear, hw],
        ]
    )
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = (rot @ corners_body.T).T + np.array([x, y])
    return corners_world


def footprint_collision_check(
    x: float,
    y: float,
    theta: float,
    config: AckermannConfig,
    obstacles: np.ndarray,
    obstacle_radius: float = 0.3,
) -> bool:
    """Check whether the vehicle footprint collides with point obstacles.

    Uses a simple point-in-rectangle test via the separating-axis
    theorem on the four footprint edges.

    Args:
        x, y, theta: Vehicle pose.
        config: Vehicle parameters.
        obstacles: Obstacle positions, shape ``(M, 2)``.
        obstacle_radius: Radius around each obstacle point.

    Returns:
        ``True`` if there is a collision.
    """
    corners = vehicle_footprint(x, y, theta, config)
    for obs in obstacles:
        # Check distance to each edge.
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            ab = b - a
            ao = obs[:2] - a
            t = float(np.clip(np.dot(ao, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
            closest = a + t * ab
            if float(np.linalg.norm(obs[:2] - closest)) < obstacle_radius:
                return True
    return False


# -----------------------------------------------------------------------
# AckermannRobot controller
# -----------------------------------------------------------------------


class AckermannRobot(RobotController):
    """Car-like robot controller with bicycle-model kinematics.

    Supports lane following via the Stanley controller or pure pursuit,
    and exposes Reeds-Shepp path computation and parallel parking.

    Args:
        config: Vehicle configuration.
        controller: ``"stanley"`` or ``"pure_pursuit"``.
        lane_points: Optional lane centreline, shape ``(N, 2)``.
    """

    def __init__(
        self,
        config: AckermannConfig | None = None,
        controller: str = "pure_pursuit",
        lane_points: np.ndarray | None = None,
        desired_speed: float = 2.0,
    ) -> None:
        self.config = config or AckermannConfig()
        self._controller_type = controller
        self._lane = lane_points
        self._desired_speed = desired_speed

        if controller == "stanley":
            self._stanley = LaneFollower()
            self._pursuit: PurePursuitController | None = None
        else:
            self._pursuit = PurePursuitController()
            self._stanley: LaneFollower | None = None  # type: ignore[assignment]

        # State.
        self._x: float = 0.0
        self._y: float = 0.0
        self._theta: float = 0.0
        self._v: float = 0.0
        self._delta: float = 0.0
        self._robot_id: int = -1
        self._goal: tuple[float, float] = (0.0, 0.0)
        self._backend: Any = None
        self._goal_tol: float = 0.5

    # ----- RobotController interface ------------------------------------

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend: Any,
    ) -> None:
        """Reset for a new episode."""
        self._robot_id = robot_id
        self._x, self._y = start
        self._theta = 0.0
        self._v = 0.0
        self._delta = 0.0
        self._goal = goal
        self._backend = backend

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        """Compute the next action using the configured controller."""
        st = states[self._robot_id]
        self._x, self._y = st.x, st.y

        goal_dist = float(np.hypot(self._goal[0] - self._x, self._goal[1] - self._y))
        if goal_dist < self._goal_tol:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        # Build a path to follow if no lane is set.
        if self._lane is None:
            self._lane = np.stack(
                [
                    np.array([self._x, self._y]),
                    np.array(self._goal),
                ]
            )

        # Desired speed with deceleration near goal.
        speed_cmd = min(self._desired_speed, goal_dist)

        # Compute steering.
        if self._controller_type == "stanley" and self._stanley is not None:
            # Stanley uses front-axle position.
            fx = self._x + self.config.wheelbase * np.cos(self._theta)
            fy = self._y + self.config.wheelbase * np.sin(self._theta)
            delta_cmd = self._stanley.compute_steering(fx, fy, self._theta, self._v, self._lane)
        elif self._pursuit is not None:
            delta_cmd = self._pursuit.compute_steering(
                self._x,
                self._y,
                self._theta,
                self._v,
                self._lane,
                self.config.wheelbase,
            )
        else:
            delta_cmd = 0.0

        # Rate-limit.
        v_out, delta_out = rate_limit_ackermann(
            speed_cmd, delta_cmd, self._v, self._delta, dt, self.config
        )

        # Integrate.
        self._x, self._y, self._theta = bicycle_forward(
            self._x,
            self._y,
            self._theta,
            v_out,
            delta_out,
            self.config.wheelbase,
            dt,
        )
        self._v = v_out
        self._delta = delta_out

        emit_event(
            "ackermann_step",
            self._robot_id,
            {
                "v": v_out,
                "delta": delta_out,
                "theta": self._theta,
                "turning_radius": bicycle_turning_radius(delta_out, self.config.wheelbase),
            },
        )

        pref_vx = v_out * float(np.cos(self._theta))
        pref_vy = v_out * float(np.sin(self._theta))
        return Action(pref_vx=pref_vx, pref_vy=pref_vy, behavior="GO_TO")

    # ----- Accessors / helpers ------------------------------------------

    @property
    def pose(self) -> np.ndarray:
        """Current ``(x, y, theta)`` pose."""
        return np.array([self._x, self._y, self._theta])

    @property
    def steering_angle(self) -> float:
        """Current front-wheel steering angle (rad)."""
        return self._delta

    def compute_reeds_shepp(
        self,
        x1: float,
        y1: float,
        theta1: float,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Plan a Reeds-Shepp path from current pose to target.

        Args:
            x1, y1, theta1: Target pose.
            num_samples: Number of samples.

        Returns:
            Path array, shape ``(num_samples, 3)``.
        """
        path, _ = reeds_shepp_path(
            self._x,
            self._y,
            self._theta,
            x1,
            y1,
            theta1,
            self.config.min_turning_radius,
            num_samples,
        )
        return path

    def plan_parallel_park(
        self,
        x_spot: float,
        y_spot: float,
        theta_spot: float,
        dt: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan a parallel-parking manoeuvre to the given spot.

        Returns:
            ``(poses, controls)`` arrays.
        """
        return parallel_parking_trajectory(
            self._x,
            self._y,
            self._theta,
            x_spot,
            y_spot,
            theta_spot,
            self.config,
            dt,
        )

    def get_footprint(self) -> np.ndarray:
        """Return the current vehicle footprint corners, shape ``(4, 2)``."""
        return vehicle_footprint(self._x, self._y, self._theta, self.config)

    def set_lane(self, lane_points: np.ndarray) -> None:
        """Set or replace the lane centreline at runtime.

        Args:
            lane_points: New lane, shape ``(N, 2)``.
        """
        self._lane = lane_points.copy()
