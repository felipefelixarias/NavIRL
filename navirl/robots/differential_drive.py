"""Differential-drive (unicycle) robot model.

Provides full unicycle kinematics, velocity/acceleration constraints, wheel
slip modelling, odometry accumulation, ICC (Instantaneous Centre of
Curvature) computation, trajectory tracking and a PID controller for path
following.  Sensor mounting points are also supported so that perception
payloads can be placed at known offsets from the robot frame.
"""

from __future__ import annotations

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
class DifferentialDriveConfig:
    """Parameters for a differential-drive robot.

    Attributes:
        wheel_radius: Radius of each drive wheel (metres).
        wheel_base: Distance between the two drive wheels (metres).
        max_linear_vel: Maximum forward speed (m/s).
        max_angular_vel: Maximum turning rate (rad/s).
        max_linear_acc: Maximum forward acceleration (m/s^2).
        max_angular_acc: Maximum angular acceleration (rad/s^2).
        slip_longitudinal: Longitudinal slip coefficient in [0, 1].
        slip_lateral: Lateral slip coefficient in [0, 1].
        odometry_noise_linear: Std-dev of odometry linear noise (m/step).
        odometry_noise_angular: Std-dev of odometry angular noise (rad/step).
    """

    wheel_radius: float = 0.05
    wheel_base: float = 0.30
    max_linear_vel: float = 1.0
    max_angular_vel: float = 2.0
    max_linear_acc: float = 2.0
    max_angular_acc: float = 4.0
    slip_longitudinal: float = 0.0
    slip_lateral: float = 0.0
    odometry_noise_linear: float = 0.0
    odometry_noise_angular: float = 0.0


@dataclass
class SensorMountPoint:
    """A sensor attached to the robot body.

    Attributes:
        name: Human-readable label.
        offset_x: Forward offset from robot centre (metres).
        offset_y: Lateral offset from robot centre (metres).
        offset_theta: Mounting angle relative to robot heading (rad).
        fov: Field-of-view half-angle (rad).
        max_range: Maximum sensing distance (metres).
    """

    name: str = "default"
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_theta: float = 0.0
    fov: float = np.pi / 3.0
    max_range: float = 5.0


# -----------------------------------------------------------------------
# PID controller
# -----------------------------------------------------------------------


@dataclass
class PIDGains:
    """PID gains for a single control channel.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        integral_limit: Anti-windup clamp for the integral term.
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.1
    integral_limit: float = 5.0


class PIDController:
    """Single-channel PID controller with anti-windup.

    The controller maintains internal integral and previous-error state
    and must be reset when a new trajectory begins.
    """

    def __init__(self, gains: PIDGains | None = None) -> None:
        self.gains = gains or PIDGains()
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._first: bool = True

    def reset(self) -> None:
        """Reset internal state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def compute(self, error: float, dt: float) -> float:
        """Return the PID output for the current *error* and time-step *dt*.

        Args:
            error: Current signed error.
            dt: Time since last call (seconds).  Must be > 0.

        Returns:
            Control signal (unbounded).
        """
        if dt <= 0.0:
            return 0.0

        self._integral += error * dt
        # Anti-windup clamp.
        limit = self.gains.integral_limit
        self._integral = float(np.clip(self._integral, -limit, limit))

        if self._first:
            derivative = 0.0
            self._first = False
        else:
            derivative = (error - self._prev_error) / dt

        self._prev_error = error
        return self.gains.kp * error + self.gains.ki * self._integral + self.gains.kd * derivative


# -----------------------------------------------------------------------
# Unicycle kinematics helpers
# -----------------------------------------------------------------------



def compute_icc(
    x: float,
    y: float,
    theta: float,
    v: float,
    omega: float,
) -> tuple[float, float, float]:
    """Compute the Instantaneous Centre of Curvature (ICC).

    For a unicycle moving with linear velocity *v* and angular velocity
    *omega*, the ICC lies perpendicular to the heading at distance
    R = v / omega.

    Args:
        x: Robot x position.
        y: Robot y position.
        theta: Robot heading (rad).
        v: Linear velocity (m/s).
        omega: Angular velocity (rad/s).

    Returns:
        ``(icc_x, icc_y, radius)`` where *radius* is ``v / omega``
        (signed).  If ``|omega| < 1e-9`` the robot is driving straight and
        the radius is set to ``inf``.
    """
    if abs(omega) < 1e-9:
        return (float("inf"), float("inf"), float("inf"))
    r = v / omega
    icc_x = x - r * np.sin(theta)
    icc_y = y + r * np.cos(theta)
    return (icc_x, icc_y, r)


def forward_kinematics(
    x: float,
    y: float,
    theta: float,
    v: float,
    omega: float,
    dt: float,
) -> tuple[float, float, float]:
    """Integrate unicycle kinematics for one time-step.

    Uses exact integration (arc) rather than Euler to reduce drift.

    Args:
        x: Current x position.
        y: Current y position.
        theta: Current heading (rad).
        v: Linear velocity (m/s).
        omega: Angular velocity (rad/s).
        dt: Time step (seconds).

    Returns:
        ``(x_new, y_new, theta_new)``.
    """
    if abs(omega) < 1e-9:
        # Straight-line motion.
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta
    else:
        # Arc motion about the ICC.
        r = v / omega
        dtheta = omega * dt
        x_new = x + r * (np.sin(theta + dtheta) - np.sin(theta))
        y_new = y - r * (np.cos(theta + dtheta) - np.cos(theta))
        theta_new = normalize_angle(theta + dtheta)
    return (float(x_new), float(y_new), float(theta_new))


def inverse_kinematics(
    v: float,
    omega: float,
    wheel_base: float,
    wheel_radius: float,
) -> tuple[float, float]:
    """Convert ``(v, omega)`` to individual wheel angular velocities.

    Args:
        v: Desired linear velocity (m/s).
        omega: Desired angular velocity (rad/s).
        wheel_base: Distance between wheels (metres).
        wheel_radius: Wheel radius (metres).

    Returns:
        ``(omega_left, omega_right)`` in rad/s.
    """
    v_left = v - omega * wheel_base / 2.0
    v_right = v + omega * wheel_base / 2.0
    return (v_left / wheel_radius, v_right / wheel_radius)


def wheel_velocities_to_body(
    omega_left: float,
    omega_right: float,
    wheel_base: float,
    wheel_radius: float,
) -> tuple[float, float]:
    """Convert wheel angular velocities to body ``(v, omega)``

    Args:
        omega_left: Left wheel angular velocity (rad/s).
        omega_right: Right wheel angular velocity (rad/s).
        wheel_base: Distance between wheels (metres).
        wheel_radius: Wheel radius (metres).

    Returns:
        ``(v, omega)`` - linear and angular body velocities.
    """
    v_left = omega_left * wheel_radius
    v_right = omega_right * wheel_radius
    v = (v_right + v_left) / 2.0
    omega = (v_right - v_left) / wheel_base
    return (v, omega)


# -----------------------------------------------------------------------
# Trajectory tracking
# -----------------------------------------------------------------------


def track_trajectory(
    waypoints: np.ndarray,
    x0: float,
    y0: float,
    theta0: float,
    dt: float,
    config: DifferentialDriveConfig,
    linear_pid: PIDGains | None = None,
    angular_pid: PIDGains | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow a sequence of 2-D waypoints using PID control.

    At each step the controller picks the nearest *un-passed* waypoint,
    computes cross-track and heading errors, and drives the unicycle
    toward it.

    Args:
        waypoints: Target positions, shape ``(N, 2)``.
        x0: Starting x.
        y0: Starting y.
        theta0: Starting heading (rad).
        dt: Simulation time step.
        config: Drive configuration.
        linear_pid: PID gains for speed control.
        angular_pid: PID gains for heading control.

    Returns:
        ``(poses, controls)`` where *poses* has shape ``(T, 3)`` –
        ``[x, y, theta]`` – and *controls* has shape ``(T, 2)`` –
        ``[v, omega]``.
    """
    pid_v = PIDController(linear_pid or PIDGains(kp=1.5, ki=0.0, kd=0.2))
    pid_w = PIDController(angular_pid or PIDGains(kp=3.0, ki=0.0, kd=0.3))

    x, y, theta = x0, y0, theta0
    wp_idx = 0
    poses: list[tuple[float, float, float]] = [(x, y, theta)]
    controls: list[tuple[float, float]] = []
    goal_tol = 0.15
    max_steps = len(waypoints) * 500

    for _ in range(max_steps):
        if wp_idx >= len(waypoints):
            break
        target = waypoints[wp_idx]
        dx = target[0] - x
        dy = target[1] - y
        dist = float(np.hypot(dx, dy))
        if dist < goal_tol:
            wp_idx += 1
            if wp_idx >= len(waypoints):
                controls.append((0.0, 0.0))
                break
            target = waypoints[wp_idx]
            dx = target[0] - x
            dy = target[1] - y
            dist = float(np.hypot(dx, dy))

        desired_heading = float(np.arctan2(dy, dx))
        heading_error = normalize_angle(desired_heading - theta)

        v_cmd = pid_v.compute(dist, dt)
        omega_cmd = pid_w.compute(heading_error, dt)

        # Slow down when not aligned.
        alignment = max(0.0, np.cos(heading_error))
        v_cmd *= alignment

        # Clamp velocities.
        v_cmd = float(np.clip(v_cmd, -config.max_linear_vel, config.max_linear_vel))
        omega_cmd = float(np.clip(omega_cmd, -config.max_angular_vel, config.max_angular_vel))

        controls.append((v_cmd, omega_cmd))
        x, y, theta = forward_kinematics(x, y, theta, v_cmd, omega_cmd, dt)
        poses.append((x, y, theta))

    return np.array(poses), np.array(controls) if controls else np.zeros((0, 2))


# -----------------------------------------------------------------------
# Odometry model
# -----------------------------------------------------------------------


class OdometryAccumulator:
    """Accumulate odometry with optional Gaussian noise.

    Maintains a running estimate of the robot pose based solely on
    wheel-encoder readings (dead reckoning).

    Attributes:
        x: Estimated x position.
        y: Estimated y position.
        theta: Estimated heading.
    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        theta0: float = 0.0,
        noise_linear: float = 0.0,
        noise_angular: float = 0.0,
    ) -> None:
        self.x = x0
        self.y = y0
        self.theta = theta0
        self._noise_lin = noise_linear
        self._noise_ang = noise_angular

    def update(self, v: float, omega: float, dt: float) -> tuple[float, float, float]:
        """Integrate one step and return the new ``(x, y, theta)``."""
        v_noisy = v + np.random.normal(0.0, self._noise_lin) if self._noise_lin > 0 else v
        omega_noisy = (
            omega + np.random.normal(0.0, self._noise_ang) if self._noise_ang > 0 else omega
        )
        self.x, self.y, self.theta = forward_kinematics(
            self.x, self.y, self.theta, v_noisy, omega_noisy, dt
        )
        return (self.x, self.y, self.theta)

    def reset(self, x: float, y: float, theta: float) -> None:
        """Reset the odometry pose."""
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def pose(self) -> np.ndarray:
        """Return current pose as ``(3,)`` array."""
        return np.array([self.x, self.y, self.theta])


# -----------------------------------------------------------------------
# Wheel slip model
# -----------------------------------------------------------------------


def apply_wheel_slip(
    v_cmd: float,
    omega_cmd: float,
    config: DifferentialDriveConfig,
) -> tuple[float, float]:
    """Apply a simple multiplicative slip model.

    Longitudinal slip reduces effective forward speed; lateral slip
    introduces a parasitic yaw disturbance.

    Args:
        v_cmd: Commanded linear velocity.
        omega_cmd: Commanded angular velocity.
        config: Drive configuration with slip parameters.

    Returns:
        ``(v_actual, omega_actual)`` after slip.
    """
    v_actual = v_cmd * (1.0 - config.slip_longitudinal)
    # Lateral slip adds a random angular disturbance proportional to speed.
    lateral_disturbance = (
        config.slip_lateral * v_cmd * np.random.normal(0.0, 1.0) if config.slip_lateral > 0 else 0.0
    )
    omega_actual = omega_cmd + lateral_disturbance
    return (v_actual, omega_actual)


# -----------------------------------------------------------------------
# Sensor mounting helpers
# -----------------------------------------------------------------------


def sensor_world_pose(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    mount: SensorMountPoint,
) -> tuple[float, float, float]:
    """Transform a sensor mount into world coordinates.

    Args:
        robot_x: Robot x position.
        robot_y: Robot y position.
        robot_theta: Robot heading (rad).
        mount: Sensor mounting specification.

    Returns:
        ``(sx, sy, stheta)`` – world-frame position and heading of the
        sensor.
    """
    cos_t = np.cos(robot_theta)
    sin_t = np.sin(robot_theta)
    sx = robot_x + cos_t * mount.offset_x - sin_t * mount.offset_y
    sy = robot_y + sin_t * mount.offset_x + cos_t * mount.offset_y
    stheta = normalize_angle(robot_theta + mount.offset_theta)
    return (float(sx), float(sy), float(stheta))


def sensor_fov_polygon(
    sensor_x: float,
    sensor_y: float,
    sensor_theta: float,
    fov: float,
    max_range: float,
    num_points: int = 32,
) -> np.ndarray:
    """Generate a polygon approximating the sensor FOV wedge.

    The polygon starts at the sensor origin, sweeps an arc from
    ``sensor_theta - fov`` to ``sensor_theta + fov``, and returns to the
    origin.

    Args:
        sensor_x: Sensor x position (world frame).
        sensor_y: Sensor y position (world frame).
        sensor_theta: Sensor heading (rad, world frame).
        fov: Half-angle of the FOV (rad).
        max_range: Maximum sensing distance.
        num_points: Number of arc segments.

    Returns:
        Polygon vertices, shape ``(num_points + 2, 2)``.
    """
    angles = np.linspace(sensor_theta - fov, sensor_theta + fov, num_points)
    arc_x = sensor_x + max_range * np.cos(angles)
    arc_y = sensor_y + max_range * np.sin(angles)
    verts = np.empty((num_points + 2, 2))
    verts[0] = [sensor_x, sensor_y]
    verts[1:-1, 0] = arc_x
    verts[1:-1, 1] = arc_y
    verts[-1] = [sensor_x, sensor_y]
    return verts


# -----------------------------------------------------------------------
# Rate limiter (acceleration constraints)
# -----------------------------------------------------------------------


def rate_limit(
    v_cmd: float,
    omega_cmd: float,
    v_prev: float,
    omega_prev: float,
    dt: float,
    config: DifferentialDriveConfig,
) -> tuple[float, float]:
    """Enforce acceleration limits on velocity commands.

    Args:
        v_cmd: Desired linear velocity.
        omega_cmd: Desired angular velocity.
        v_prev: Previous linear velocity.
        omega_prev: Previous angular velocity.
        dt: Time step.
        config: Drive configuration.

    Returns:
        ``(v_limited, omega_limited)`` after clamping the rate of change.
    """
    max_dv = config.max_linear_acc * dt
    max_dw = config.max_angular_acc * dt
    dv = float(np.clip(v_cmd - v_prev, -max_dv, max_dv))
    dw = float(np.clip(omega_cmd - omega_prev, -max_dw, max_dw))
    v_out = float(np.clip(v_prev + dv, -config.max_linear_vel, config.max_linear_vel))
    omega_out = float(np.clip(omega_prev + dw, -config.max_angular_vel, config.max_angular_vel))
    return (v_out, omega_out)


# -----------------------------------------------------------------------
# DifferentialDriveRobot controller
# -----------------------------------------------------------------------


class DifferentialDriveRobot(RobotController):
    """Differential-drive robot with full unicycle kinematics.

    This controller wraps the kinematic model, PID-based path following,
    odometry accumulation, wheel slip, and acceleration limiting into a
    single :class:`RobotController` interface suitable for use inside the
    NavIRL simulation loop.

    Args:
        config: Drive parameters.
        sensor_mounts: Optional list of sensor mounting points.
        linear_pid: PID gains for linear speed.
        angular_pid: PID gains for heading control.
        path: Optional pre-planned waypoint path, shape ``(N, 2)``.
    """

    def __init__(
        self,
        config: DifferentialDriveConfig | None = None,
        sensor_mounts: list[SensorMountPoint] | None = None,
        linear_pid: PIDGains | None = None,
        angular_pid: PIDGains | None = None,
        path: np.ndarray | None = None,
    ) -> None:
        self.config = config or DifferentialDriveConfig()
        self.sensor_mounts = sensor_mounts or []
        self._pid_v = PIDController(linear_pid or PIDGains(kp=1.5, ki=0.0, kd=0.2))
        self._pid_w = PIDController(angular_pid or PIDGains(kp=3.0, ki=0.0, kd=0.3))

        # State
        self._x: float = 0.0
        self._y: float = 0.0
        self._theta: float = 0.0
        self._v: float = 0.0
        self._omega: float = 0.0

        self._robot_id: int = -1
        self._goal: tuple[float, float] = (0.0, 0.0)
        self._backend: Any = None

        self._odometry = OdometryAccumulator(
            noise_linear=self.config.odometry_noise_linear,
            noise_angular=self.config.odometry_noise_angular,
        )
        self._path = path
        self._wp_idx: int = 0
        self._goal_tol: float = 0.20

    # ----- RobotController interface ------------------------------------

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend: Any,
    ) -> None:
        """Reset the controller for a new episode."""
        self._robot_id = robot_id
        self._x, self._y = start
        self._theta = 0.0
        self._v = 0.0
        self._omega = 0.0
        self._goal = goal
        self._backend = backend
        self._wp_idx = 0

        self._pid_v.reset()
        self._pid_w.reset()
        self._odometry.reset(self._x, self._y, self._theta)

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        """Compute the next action using PID path-following.

        If no explicit path was provided at construction, the controller
        drives directly toward the goal.
        """
        st = states[self._robot_id]
        self._x, self._y = st.x, st.y

        # Determine current target waypoint.
        if self._path is not None and self._wp_idx < len(self._path):
            target = self._path[self._wp_idx]
        else:
            target = np.array(self._goal)

        dx = target[0] - self._x
        dy = target[1] - self._y
        dist = float(np.hypot(dx, dy))

        # Advance waypoint index.
        if dist < self._goal_tol and self._path is not None:
            self._wp_idx += 1
            if self._wp_idx < len(self._path):
                target = self._path[self._wp_idx]
                dx = target[0] - self._x
                dy = target[1] - self._y
                dist = float(np.hypot(dx, dy))

        # Check if at final goal.
        goal_dist = float(np.hypot(self._goal[0] - self._x, self._goal[1] - self._y))
        if goal_dist < self._goal_tol:
            self._v = 0.0
            self._omega = 0.0
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        # PID control.
        desired_heading = float(np.arctan2(dy, dx))
        heading_error = normalize_angle(desired_heading - self._theta)

        v_cmd = self._pid_v.compute(dist, dt)
        omega_cmd = self._pid_w.compute(heading_error, dt)

        # Reduce speed when misaligned.
        v_cmd *= max(0.0, np.cos(heading_error))

        # Acceleration limiting.
        v_cmd, omega_cmd = rate_limit(v_cmd, omega_cmd, self._v, self._omega, dt, self.config)

        # Wheel slip.
        v_actual, omega_actual = apply_wheel_slip(v_cmd, omega_cmd, self.config)

        # Integrate kinematics.
        self._x, self._y, self._theta = forward_kinematics(
            self._x, self._y, self._theta, v_actual, omega_actual, dt
        )
        self._v = v_actual
        self._omega = omega_actual

        # Update odometry.
        self._odometry.update(v_actual, omega_actual, dt)

        # Emit diagnostic event.
        icc = compute_icc(self._x, self._y, self._theta, v_actual, omega_actual)
        emit_event(
            "diffdrive_step",
            self._robot_id,
            {
                "v": v_actual,
                "omega": omega_actual,
                "theta": self._theta,
                "icc_x": icc[0],
                "icc_y": icc[1],
                "icc_r": icc[2],
            },
        )

        # Convert to Cartesian preferred velocity for the sim.
        pref_vx = v_actual * float(np.cos(self._theta))
        pref_vy = v_actual * float(np.sin(self._theta))
        return Action(pref_vx=pref_vx, pref_vy=pref_vy, behavior="GO_TO")

    # ----- Accessors ----------------------------------------------------

    @property
    def pose(self) -> np.ndarray:
        """Current ``(x, y, theta)`` pose."""
        return np.array([self._x, self._y, self._theta])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current ``(v, omega)`` velocity."""
        return (self._v, self._omega)

    @property
    def odometry_pose(self) -> np.ndarray:
        """Odometry-estimated ``(x, y, theta)``."""
        return self._odometry.pose

    def get_sensor_poses(self) -> list[tuple[float, float, float]]:
        """Return world-frame poses of all mounted sensors."""
        return [sensor_world_pose(self._x, self._y, self._theta, m) for m in self.sensor_mounts]

    def get_wheel_speeds(self) -> tuple[float, float]:
        """Return current ``(omega_left, omega_right)`` wheel speeds."""
        return inverse_kinematics(
            self._v,
            self._omega,
            self.config.wheel_base,
            self.config.wheel_radius,
        )
