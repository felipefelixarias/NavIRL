"""Holonomic (omnidirectional) robot model.

Provides an omnidirectional motion model with independent velocity control
in X and Y, per-axis acceleration limits, a first-order inertia model,
smooth trapezoidal and cubic-spline motion profiles, and waypoint-following
with cubic spline interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class HolonomicConfig:
    """Parameters for an omnidirectional robot.

    Attributes:
        max_speed: Maximum translational speed (m/s).
        max_acceleration: Maximum translational acceleration (m/s^2).
        max_jerk: Maximum jerk for smooth profiles (m/s^3).
        inertia_tau: First-order inertia time constant (seconds).
            0 means perfectly responsive.
        mass: Robot mass (kg), used only for force-based control.
        radius: Robot footprint radius (metres).
        heading_control: If ``True`` the robot also controls heading.
        max_angular_vel: Maximum angular velocity (rad/s).
        max_angular_acc: Maximum angular acceleration (rad/s^2).
    """

    max_speed: float = 1.5
    max_acceleration: float = 3.0
    max_jerk: float = 10.0
    inertia_tau: float = 0.0
    mass: float = 10.0
    radius: float = 0.25
    heading_control: bool = False
    max_angular_vel: float = 3.0
    max_angular_acc: float = 6.0


# -----------------------------------------------------------------------
# Inertia filter
# -----------------------------------------------------------------------

class InertiaFilter:
    """Simple first-order low-pass (exponential smoothing) filter.

    Models mechanical inertia by smoothing commanded velocities through
    a discrete first-order lag with time constant *tau*.

    Attributes:
        tau: Time constant (seconds).  Larger values increase lag.
    """

    def __init__(self, tau: float = 0.1) -> None:
        self.tau = max(tau, 0.0)
        self._vx: float = 0.0
        self._vy: float = 0.0

    def reset(self, vx: float = 0.0, vy: float = 0.0) -> None:
        """Reset filter state."""
        self._vx = vx
        self._vy = vy

    def filter(self, vx_cmd: float, vy_cmd: float, dt: float) -> tuple[float, float]:
        """Apply the inertia filter and return smoothed velocities.

        Args:
            vx_cmd: Commanded x velocity.
            vy_cmd: Commanded y velocity.
            dt: Time step.

        Returns:
            ``(vx_smooth, vy_smooth)``.
        """
        if self.tau <= 0.0 or dt <= 0.0:
            self._vx = vx_cmd
            self._vy = vy_cmd
        else:
            alpha = 1.0 - np.exp(-dt / self.tau)
            self._vx += alpha * (vx_cmd - self._vx)
            self._vy += alpha * (vy_cmd - self._vy)
        return (self._vx, self._vy)

    @property
    def velocity(self) -> tuple[float, float]:
        """Current filtered velocity."""
        return (self._vx, self._vy)


# -----------------------------------------------------------------------
# Acceleration limiter
# -----------------------------------------------------------------------

def clamp_acceleration(
    vx_cmd: float,
    vy_cmd: float,
    vx_prev: float,
    vy_prev: float,
    dt: float,
    config: HolonomicConfig,
) -> tuple[float, float]:
    """Clamp the acceleration so that the change in velocity does not
    exceed the configured maximum.

    The clamping is done *per-axis* first, then the speed magnitude is
    also clamped.

    Args:
        vx_cmd: Desired x velocity.
        vy_cmd: Desired y velocity.
        vx_prev: Previous x velocity.
        vy_prev: Previous y velocity.
        dt: Time step.
        config: Robot configuration.

    Returns:
        ``(vx_clamped, vy_clamped)``.
    """
    if dt <= 0.0:
        return (vx_cmd, vy_cmd)

    max_dv = config.max_acceleration * dt
    dvx = float(np.clip(vx_cmd - vx_prev, -max_dv, max_dv))
    dvy = float(np.clip(vy_cmd - vy_prev, -max_dv, max_dv))
    vx = vx_prev + dvx
    vy = vy_prev + dvy

    speed = float(np.hypot(vx, vy))
    if speed > config.max_speed:
        scale = config.max_speed / speed
        vx *= scale
        vy *= scale
    return (vx, vy)


# -----------------------------------------------------------------------
# Smooth motion profiles
# -----------------------------------------------------------------------

def trapezoidal_profile(
    distance: float,
    max_speed: float,
    max_acceleration: float,
    dt: float,
) -> np.ndarray:
    """Generate a 1-D trapezoidal velocity profile.

    The profile accelerates at *max_acceleration*, cruises at *max_speed*,
    and decelerates symmetrically to arrive at *distance* with zero
    velocity.

    Args:
        distance: Total distance to travel (positive).
        max_speed: Peak speed.
        max_acceleration: Acceleration / deceleration magnitude.
        dt: Time step for sampling.

    Returns:
        Array of velocities sampled at *dt* intervals, shape ``(T,)``.
    """
    if distance <= 0.0 or dt <= 0.0:
        return np.zeros(1)

    # Time to accelerate to max_speed.
    t_acc = max_speed / max_acceleration
    d_acc = 0.5 * max_acceleration * t_acc ** 2.0

    if 2.0 * d_acc >= distance:
        # Triangle profile (never reaches max_speed).
        t_acc = np.sqrt(distance / max_acceleration)
        total_time = 2.0 * t_acc
    else:
        d_cruise = distance - 2.0 * d_acc
        t_cruise = d_cruise / max_speed
        total_time = 2.0 * t_acc + t_cruise

    n_steps = max(int(np.ceil(total_time / dt)), 1)
    times = np.arange(n_steps) * dt
    velocities = np.zeros(n_steps)

    for i, t in enumerate(times):
        if t < t_acc:
            velocities[i] = max_acceleration * t
        elif t < total_time - t_acc:
            velocities[i] = max_speed if 2.0 * d_acc < distance else max_acceleration * t_acc
        else:
            velocities[i] = max(0.0, max_acceleration * (total_time - t))

    return velocities


def cubic_spline_waypoints(
    waypoints: np.ndarray,
    num_samples: int = 200,
) -> np.ndarray:
    """Interpolate waypoints using natural cubic splines.

    Each coordinate dimension is independently interpolated with a
    natural cubic spline (second derivative = 0 at endpoints).

    Args:
        waypoints: Control points, shape ``(N, 2)`` with ``N >= 2``.
        num_samples: Number of output samples along the spline.

    Returns:
        Interpolated positions, shape ``(num_samples, 2)``.
    """
    n = waypoints.shape[0]
    if n < 2:
        return np.tile(waypoints[0], (num_samples, 1))

    # Parameterise by cumulative chord length.
    diffs = np.diff(waypoints, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    t_knots = np.concatenate([[0.0], np.cumsum(seg_lens)])
    t_knots /= t_knots[-1] if t_knots[-1] > 1e-12 else 1.0

    result = np.zeros((num_samples, 2))
    t_query = np.linspace(0.0, 1.0, num_samples)

    for dim in range(2):
        y = waypoints[:, dim]
        # Solve for natural cubic spline coefficients using tridiagonal
        # system (Thomas algorithm).
        h = np.diff(t_knots)
        m = n
        a_diag = np.ones(m)
        b_diag = np.zeros(m)
        c_diag = np.zeros(m)
        d_vec = np.zeros(m)

        for i in range(1, m - 1):
            a_diag[i] = 2.0 * (h[i - 1] + h[i])
            b_diag[i] = h[i - 1]
            c_diag[i] = h[i]
            d_vec[i] = 3.0 * (
                (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]
            )

        # Natural boundary: second derivative = 0.
        a_diag[0] = 1.0
        a_diag[-1] = 1.0

        # Forward sweep.
        c_star = np.zeros(m)
        d_star = np.zeros(m)
        c_star[0] = c_diag[0] / a_diag[0]
        d_star[0] = d_vec[0] / a_diag[0]
        for i in range(1, m):
            denom = a_diag[i] - b_diag[i] * c_star[i - 1]
            if abs(denom) < 1e-15:
                denom = 1e-15
            c_star[i] = c_diag[i] / denom if i < m - 1 else 0.0
            d_star[i] = (d_vec[i] - b_diag[i] * d_star[i - 1]) / denom

        # Back substitution.
        c_coeffs = np.zeros(m)
        c_coeffs[-1] = d_star[-1]
        for i in range(m - 2, -1, -1):
            c_coeffs[i] = d_star[i] - c_star[i] * c_coeffs[i + 1]

        # Evaluate spline.
        for qi, tq in enumerate(t_query):
            # Find segment.
            seg = int(np.searchsorted(t_knots[1:], tq, side="right"))
            seg = min(seg, n - 2)
            seg = max(seg, 0)
            dt_seg = tq - t_knots[seg]
            hi = h[seg] if h[seg] > 1e-15 else 1e-15
            a_c = y[seg]
            b_c = (y[seg + 1] - y[seg]) / hi - hi * (2.0 * c_coeffs[seg] + c_coeffs[seg + 1]) / 3.0
            d_c = (c_coeffs[seg + 1] - c_coeffs[seg]) / (3.0 * hi)
            result[qi, dim] = a_c + b_c * dt_seg + c_coeffs[seg] * dt_seg ** 2 + d_c * dt_seg ** 3

    return result


# -----------------------------------------------------------------------
# Waypoint follower
# -----------------------------------------------------------------------

class WaypointFollower:
    """Follow a sequence of 2-D waypoints with configurable speed.

    The follower uses cubic-spline interpolation to produce a dense path
    and then tracks a moving reference point along it.

    Attributes:
        dense_path: The densely-interpolated path, shape ``(M, 2)``.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        desired_speed: float = 1.0,
        num_interp: int = 300,
    ) -> None:
        self._waypoints = waypoints.copy()
        self._desired_speed = desired_speed
        self.dense_path = cubic_spline_waypoints(waypoints, num_interp)
        self._ref_idx: int = 0
        self._finished: bool = False

    def reset(self) -> None:
        """Reset the follower to the beginning of the path."""
        self._ref_idx = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        """Whether the end of the path has been reached."""
        return self._finished

    def compute_velocity(
        self,
        x: float,
        y: float,
        dt: float,
        lookahead: float = 0.3,
    ) -> tuple[float, float]:
        """Compute desired velocity to track the spline path.

        Uses a pure-pursuit-style lookahead on the dense path.

        Args:
            x: Current x position.
            y: Current y position.
            dt: Time step.
            lookahead: Lookahead distance along the path.

        Returns:
            ``(vx, vy)`` desired velocity.
        """
        if self._finished or self._ref_idx >= len(self.dense_path):
            self._finished = True
            return (0.0, 0.0)

        # Advance reference index past points we are close to.
        pos = np.array([x, y])
        while self._ref_idx < len(self.dense_path) - 1:
            d = float(np.linalg.norm(self.dense_path[self._ref_idx] - pos))
            if d > lookahead:
                break
            self._ref_idx += 1

        if self._ref_idx >= len(self.dense_path) - 1:
            target = self.dense_path[-1]
            dist = float(np.linalg.norm(target - pos))
            if dist < 0.1:
                self._finished = True
                return (0.0, 0.0)
        else:
            target = self.dense_path[self._ref_idx]

        diff = target - pos
        dist = float(np.linalg.norm(diff))
        if dist < 1e-8:
            return (0.0, 0.0)
        direction = diff / dist
        speed = min(self._desired_speed, dist / max(dt, 1e-6))
        return (float(direction[0] * speed), float(direction[1] * speed))


# -----------------------------------------------------------------------
# Trajectory generation
# -----------------------------------------------------------------------

def generate_smooth_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    config: HolonomicConfig,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth straight-line trajectory between two points.

    The trajectory follows a trapezoidal velocity profile along the
    line connecting *start* and *goal*.

    Args:
        start: Start position ``(2,)``.
        goal: Goal position ``(2,)``.
        config: Robot configuration.
        dt: Sampling time step.

    Returns:
        ``(positions, velocities, times)`` each of shape ``(T, 2)`` or
        ``(T,)``.
    """
    diff = goal - start
    distance = float(np.linalg.norm(diff))
    if distance < 1e-8:
        return (
            start.reshape(1, 2).copy(),
            np.zeros((1, 2)),
            np.zeros(1),
        )
    direction = diff / distance
    speed_profile = trapezoidal_profile(distance, config.max_speed, config.max_acceleration, dt)
    n = len(speed_profile)
    positions = np.zeros((n, 2))
    velocities = np.zeros((n, 2))
    times = np.arange(n) * dt

    s = 0.0  # accumulated distance
    for i in range(n):
        positions[i] = start + direction * s
        velocities[i] = direction * speed_profile[i]
        s += speed_profile[i] * dt
        s = min(s, distance)

    return (positions, velocities, times)


def generate_waypoint_trajectory(
    waypoints: np.ndarray,
    config: HolonomicConfig,
    dt: float = 0.05,
    num_interp: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth trajectory through multiple waypoints.

    First interpolates with cubic splines, then re-parameterises by arc
    length with a trapezoidal speed profile.

    Args:
        waypoints: Shape ``(N, 2)``.
        config: Robot configuration.
        dt: Sampling time step.
        num_interp: Number of spline samples.

    Returns:
        ``(positions, velocities, times)``.
    """
    dense = cubic_spline_waypoints(waypoints, num_interp)
    seg_lens = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    total_dist = float(np.sum(seg_lens))
    if total_dist < 1e-8:
        return (dense[:1], np.zeros((1, 2)), np.zeros(1))

    speed_profile = trapezoidal_profile(total_dist, config.max_speed, config.max_acceleration, dt)
    n = len(speed_profile)
    positions = np.zeros((n, 2))
    velocities = np.zeros((n, 2))
    times = np.arange(n) * dt

    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    s = 0.0
    for i in range(n):
        # Find position at arc length s on dense path.
        seg = int(np.searchsorted(cum_lens[1:], s, side="right"))
        seg = min(seg, len(dense) - 2)
        seg = max(seg, 0)
        local = s - cum_lens[seg]
        sl = seg_lens[seg] if seg_lens[seg] > 1e-15 else 1e-15
        alpha = min(local / sl, 1.0)
        positions[i] = (1.0 - alpha) * dense[seg] + alpha * dense[seg + 1]
        direction = dense[seg + 1] - dense[seg]
        d = float(np.linalg.norm(direction))
        if d > 1e-12:
            direction = direction / d
        velocities[i] = direction * speed_profile[i]
        s += speed_profile[i] * dt
        s = min(s, total_dist)

    return (positions, velocities, times)


# -----------------------------------------------------------------------
# HolonomicRobot controller
# -----------------------------------------------------------------------

class HolonomicRobot(RobotController):
    """Omnidirectional robot controller.

    Combines inertia filtering, acceleration clamping, and waypoint
    following into a :class:`RobotController` compatible with the NavIRL
    simulation loop.

    Args:
        config: Holonomic robot configuration.
        waypoints: Optional pre-set waypoint path, shape ``(N, 2)``.
        desired_speed: Desired cruise speed (m/s).
    """

    def __init__(
        self,
        config: HolonomicConfig | None = None,
        waypoints: np.ndarray | None = None,
        desired_speed: float = 1.0,
    ) -> None:
        self.config = config or HolonomicConfig()
        self._inertia = InertiaFilter(tau=self.config.inertia_tau)
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._x: float = 0.0
        self._y: float = 0.0
        self._theta: float = 0.0
        self._omega: float = 0.0
        self._robot_id: int = -1
        self._goal: tuple[float, float] = (0.0, 0.0)
        self._backend: Any = None
        self._desired_speed = desired_speed
        self._follower: WaypointFollower | None = None
        if waypoints is not None:
            self._follower = WaypointFollower(
                waypoints, desired_speed=desired_speed
            )
        self._goal_tol: float = 0.20

    # ----- RobotController interface ------------------------------------

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend: Any,
    ) -> None:
        """Reset state for a new episode."""
        self._robot_id = robot_id
        self._x, self._y = start
        self._theta = 0.0
        self._omega = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._goal = goal
        self._backend = backend
        self._inertia.reset()
        if self._follower is not None:
            self._follower.reset()

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        """Compute the next omnidirectional velocity command."""
        st = states[self._robot_id]
        self._x, self._y = st.x, st.y

        # Check goal.
        goal_dist = float(np.hypot(self._goal[0] - self._x, self._goal[1] - self._y))
        if goal_dist < self._goal_tol:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        # Determine desired velocity.
        if self._follower is not None and not self._follower.finished:
            vx_des, vy_des = self._follower.compute_velocity(
                self._x, self._y, dt
            )
        else:
            dx = self._goal[0] - self._x
            dy = self._goal[1] - self._y
            dist = float(np.hypot(dx, dy))
            if dist < 1e-8:
                return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")
            speed = min(self._desired_speed, st.max_speed, dist / max(dt, 1e-6))
            vx_des = dx / dist * speed
            vy_des = dy / dist * speed

        # Acceleration limit.
        vx_cmd, vy_cmd = clamp_acceleration(
            vx_des, vy_des, self._vx, self._vy, dt, self.config
        )
        # Inertia.
        vx_out, vy_out = self._inertia.filter(vx_cmd, vy_cmd, dt)

        self._vx = vx_out
        self._vy = vy_out

        emit_event(
            "holonomic_step",
            self._robot_id,
            {"vx": vx_out, "vy": vy_out, "speed": float(np.hypot(vx_out, vy_out))},
        )

        return Action(pref_vx=vx_out, pref_vy=vy_out, behavior="GO_TO")

    # ----- Accessors ----------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Current ``(x, y)`` position."""
        return np.array([self._x, self._y])

    @property
    def velocity(self) -> np.ndarray:
        """Current ``(vx, vy)`` velocity."""
        return np.array([self._vx, self._vy])

    @property
    def speed(self) -> float:
        """Current scalar speed."""
        return float(np.hypot(self._vx, self._vy))

    def set_waypoints(self, waypoints: np.ndarray, desired_speed: float | None = None) -> None:
        """Replace the waypoint path at runtime.

        Args:
            waypoints: New waypoint array, shape ``(N, 2)``.
            desired_speed: Optional new cruise speed.
        """
        spd = desired_speed if desired_speed is not None else self._desired_speed
        self._follower = WaypointFollower(waypoints, desired_speed=spd)
