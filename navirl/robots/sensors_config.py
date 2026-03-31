"""Robot sensor configuration and management.

Defines sensor placement, field-of-view computation, occlusion checking,
sensor fusion configuration, and noise models for various sensor modalities.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

# -----------------------------------------------------------------------
# Sensor types
# -----------------------------------------------------------------------


class SensorType(enum.Enum):
    """Supported sensor modalities."""

    LIDAR_2D = "lidar_2d"
    LIDAR_3D = "lidar_3d"
    CAMERA_RGB = "camera_rgb"
    CAMERA_DEPTH = "camera_depth"
    CAMERA_STEREO = "camera_stereo"
    IMU = "imu"
    ULTRASONIC = "ultrasonic"
    WHEEL_ENCODER = "wheel_encoder"
    GPS = "gps"


# -----------------------------------------------------------------------
# Noise models
# -----------------------------------------------------------------------


@dataclass
class GaussianNoise:
    """Additive zero-mean Gaussian noise.

    Attributes:
        std_dev: Standard deviation of the noise.
    """

    std_dev: float = 0.01

    def sample(self, shape: tuple[int, ...] = ()) -> np.ndarray:
        """Draw a noise sample.

        Args:
            shape: Shape of the output array.

        Returns:
            Noise array.
        """
        return np.random.normal(0.0, self.std_dev, size=shape)


@dataclass
class UniformNoise:
    """Additive uniform noise.

    Attributes:
        half_range: Noise is drawn from ``[-half_range, half_range]``.
    """

    half_range: float = 0.01

    def sample(self, shape: tuple[int, ...] = ()) -> np.ndarray:
        """Draw a noise sample."""
        return np.random.uniform(-self.half_range, self.half_range, size=shape)


@dataclass
class RangeProportionalNoise:
    """Noise proportional to the measured range (common for depth sensors).

    ``noise = range * coefficient * N(0,1)``.

    Attributes:
        coefficient: Proportionality constant.
    """

    coefficient: float = 0.005

    def sample_at_range(self, ranges: np.ndarray) -> np.ndarray:
        """Draw noise proportional to *ranges*.

        Args:
            ranges: Measured ranges, any shape.

        Returns:
            Noise array of the same shape.
        """
        return ranges * self.coefficient * np.random.randn(*ranges.shape)


@dataclass
class SaltPepperNoise:
    """Salt-and-pepper noise model for range sensors.

    With probability *p_salt* a reading is replaced by *max_range*
    (miss), and with probability *p_pepper* by ``0`` (false return).

    Attributes:
        p_salt: Probability of a max-range reading.
        p_pepper: Probability of a zero reading.
    """

    p_salt: float = 0.01
    p_pepper: float = 0.005

    def apply(self, ranges: np.ndarray, max_range: float) -> np.ndarray:
        """Apply salt-and-pepper corruption to *ranges*.

        Args:
            ranges: Original range readings.
            max_range: Sensor maximum range.

        Returns:
            Corrupted copy.
        """
        out = ranges.copy()
        salt_mask = np.random.rand(*ranges.shape) < self.p_salt
        pepper_mask = np.random.rand(*ranges.shape) < self.p_pepper
        out[salt_mask] = max_range
        out[pepper_mask] = 0.0
        return out


# -----------------------------------------------------------------------
# Sensor mount
# -----------------------------------------------------------------------


@dataclass
class SensorMount:
    """Physical placement of a sensor on the robot body.

    All offsets are relative to the robot body-frame origin, which is
    typically the centre of the rear axle or the geometric centre.

    Attributes:
        name: Human-readable label (e.g. ``"front_lidar"``).
        sensor_type: Sensor modality.
        offset_x: Forward offset (metres).
        offset_y: Lateral offset (metres, positive = left).
        offset_z: Vertical offset (metres, positive = up).
        roll: Mounting roll (rad).
        pitch: Mounting pitch (rad).
        yaw: Mounting yaw relative to robot heading (rad).
        fov_horizontal: Horizontal field of view (rad).
        fov_vertical: Vertical field of view (rad).
        max_range: Maximum sensing distance (metres).
        min_range: Minimum sensing distance (metres).
        resolution_horizontal: Angular resolution horizontal (rad).
        resolution_vertical: Angular resolution vertical (rad).
        update_rate: Sensor update rate (Hz).
        enabled: Whether the sensor is active.
    """

    name: str = "sensor"
    sensor_type: SensorType = SensorType.LIDAR_2D
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    fov_horizontal: float = 2.0 * np.pi
    fov_vertical: float = np.pi / 6.0
    max_range: float = 30.0
    min_range: float = 0.1
    resolution_horizontal: float = np.radians(0.5)
    resolution_vertical: float = np.radians(2.0)
    update_rate: float = 10.0
    enabled: bool = True


# -----------------------------------------------------------------------
# World-frame transforms
# -----------------------------------------------------------------------


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def sensor_world_pose_2d(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    mount: SensorMount,
) -> tuple[float, float, float]:
    """Transform a 2-D sensor mount into world coordinates.

    Args:
        robot_x: Robot x.
        robot_y: Robot y.
        robot_theta: Robot heading (rad).
        mount: Sensor mount specification.

    Returns:
        ``(sx, sy, s_yaw)`` in world frame.
    """
    cos_t = np.cos(robot_theta)
    sin_t = np.sin(robot_theta)
    sx = robot_x + cos_t * mount.offset_x - sin_t * mount.offset_y
    sy = robot_y + sin_t * mount.offset_x + cos_t * mount.offset_y
    s_yaw = _wrap_angle(robot_theta + mount.yaw)
    return (float(sx), float(sy), float(s_yaw))


def sensor_world_pose_3d(
    robot_x: float,
    robot_y: float,
    robot_z: float,
    robot_roll: float,
    robot_pitch: float,
    robot_yaw: float,
    mount: SensorMount,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a 3-D sensor mount into world coordinates.

    Args:
        robot_x, robot_y, robot_z: Robot position.
        robot_roll, robot_pitch, robot_yaw: Robot orientation (rad).
        mount: Sensor mount specification.

    Returns:
        ``(position, rpy)`` where *position* is ``(3,)`` and *rpy* is
        ``(3,)`` roll-pitch-yaw.
    """
    # Body-to-world rotation (yaw-pitch-roll, ZYX convention).
    cy, sy = np.cos(robot_yaw), np.sin(robot_yaw)
    cp, sp = np.cos(robot_pitch), np.sin(robot_pitch)
    cr, sr = np.cos(robot_roll), np.sin(robot_roll)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    offset = np.array([mount.offset_x, mount.offset_y, mount.offset_z])
    world_pos = np.array([robot_x, robot_y, robot_z]) + R @ offset
    world_rpy = np.array(
        [
            _wrap_angle(robot_roll + mount.roll),
            _wrap_angle(robot_pitch + mount.pitch),
            _wrap_angle(robot_yaw + mount.yaw),
        ]
    )
    return (world_pos, world_rpy)


# -----------------------------------------------------------------------
# FOV computation
# -----------------------------------------------------------------------


def compute_fov_polygon(
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    fov: float,
    max_range: float,
    num_points: int = 64,
) -> np.ndarray:
    """Generate a 2-D polygon approximating the sensor FOV wedge.

    Args:
        sensor_x, sensor_y: Sensor position (world frame).
        sensor_yaw: Sensor heading (rad).
        fov: Total horizontal field of view (rad).
        max_range: Maximum range.
        num_points: Arc resolution.

    Returns:
        Polygon vertices, shape ``(num_points + 2, 2)``.
    """
    half = fov / 2.0
    angles = np.linspace(sensor_yaw - half, sensor_yaw + half, num_points)
    arc_x = sensor_x + max_range * np.cos(angles)
    arc_y = sensor_y + max_range * np.sin(angles)
    verts = np.empty((num_points + 2, 2))
    verts[0] = [sensor_x, sensor_y]
    verts[1:-1, 0] = arc_x
    verts[1:-1, 1] = arc_y
    verts[-1] = [sensor_x, sensor_y]
    return verts


def compute_fov_rays(
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    mount: SensorMount,
) -> np.ndarray:
    """Generate ray directions for a 2-D range sensor.

    Args:
        sensor_x, sensor_y: Sensor position.
        sensor_yaw: Sensor heading.
        mount: Sensor mount (uses ``fov_horizontal`` and
            ``resolution_horizontal``).

    Returns:
        Ray unit-direction vectors, shape ``(N, 2)``.
    """
    half = mount.fov_horizontal / 2.0
    n_rays = max(int(mount.fov_horizontal / max(mount.resolution_horizontal, 1e-6)), 1)
    angles = np.linspace(sensor_yaw - half, sensor_yaw + half, n_rays)
    rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return rays


# -----------------------------------------------------------------------
# Occlusion checking
# -----------------------------------------------------------------------


def check_point_visibility(
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    mount: SensorMount,
    point: np.ndarray,
) -> bool:
    """Check whether a single point is within the sensor's un-occluded FOV.

    This is a simple geometric check (range + angle) *without*
    ray-tracing against obstacles.

    Args:
        sensor_x, sensor_y: Sensor position in world frame.
        sensor_yaw: Sensor heading.
        mount: Sensor mount specification.
        point: Target point, shape ``(2,)``.

    Returns:
        ``True`` if the point is within range and angular FOV.
    """
    dx = point[0] - sensor_x
    dy = point[1] - sensor_y
    dist = float(np.hypot(dx, dy))
    if dist < mount.min_range or dist > mount.max_range:
        return False
    angle = float(np.arctan2(dy, dx))
    diff = _wrap_angle(angle - sensor_yaw)
    half = mount.fov_horizontal / 2.0
    return abs(diff) <= half


def raytrace_occlusion(
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    mount: SensorMount,
    target: np.ndarray,
    obstacles: np.ndarray,
    obstacle_radius: float = 0.3,
) -> bool:
    """Check if a target point is occluded by obstacles via ray-tracing.

    A ray is cast from the sensor toward *target*; if any obstacle
    circle intersects the ray segment before the target, the target is
    considered occluded.

    Args:
        sensor_x, sensor_y: Sensor position.
        sensor_yaw: Sensor heading (used for FOV pre-check).
        mount: Sensor mount.
        target: Target point, ``(2,)``.
        obstacles: Obstacle centres, ``(M, 2)``.
        obstacle_radius: Uniform radius of each obstacle.

    Returns:
        ``True`` if the target is **occluded** (not visible).
    """
    if not check_point_visibility(sensor_x, sensor_y, sensor_yaw, mount, target):
        return True  # Outside FOV.

    origin = np.array([sensor_x, sensor_y])
    ray = target - origin
    ray_len = float(np.linalg.norm(ray))
    if ray_len < 1e-8:
        return False
    ray_dir = ray / ray_len

    for obs in obstacles:
        oc = obs[:2] - origin
        proj = float(np.dot(oc, ray_dir))
        if proj < 0.0 or proj > ray_len:
            continue
        closest = origin + ray_dir * proj
        dist = float(np.linalg.norm(obs[:2] - closest))
        if dist < obstacle_radius:
            return True
    return False


def compute_visible_points(
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    mount: SensorMount,
    points: np.ndarray,
    obstacles: np.ndarray | None = None,
    obstacle_radius: float = 0.3,
) -> np.ndarray:
    """Filter a set of points to those visible from the sensor.

    Args:
        sensor_x, sensor_y: Sensor position.
        sensor_yaw: Sensor heading.
        mount: Sensor specification.
        points: Candidate points, ``(N, 2)``.
        obstacles: Optional obstacle centres, ``(M, 2)``.
        obstacle_radius: Obstacle radius.

    Returns:
        Boolean mask, shape ``(N,)``, ``True`` for visible points.
    """
    n = points.shape[0]
    visible = np.zeros(n, dtype=bool)
    for i in range(n):
        if not check_point_visibility(sensor_x, sensor_y, sensor_yaw, mount, points[i]):
            continue
        if obstacles is not None and obstacles.shape[0] > 0:
            if raytrace_occlusion(
                sensor_x,
                sensor_y,
                sensor_yaw,
                mount,
                points[i],
                obstacles,
                obstacle_radius,
            ):
                continue
        visible[i] = True
    return visible


# -----------------------------------------------------------------------
# Simulated range scan
# -----------------------------------------------------------------------


def simulate_range_scan(
    sensor_x: float,
    sensor_y: float,
    sensor_yaw: float,
    mount: SensorMount,
    obstacles: np.ndarray,
    obstacle_radius: float = 0.3,
    noise: GaussianNoise | RangeProportionalNoise | None = None,
    salt_pepper: SaltPepperNoise | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a 2-D LiDAR range scan.

    Casts rays from the sensor and returns the nearest intersection
    distance for each ray.

    Args:
        sensor_x, sensor_y: Sensor position.
        sensor_yaw: Sensor heading.
        mount: Sensor mount (defines FOV, resolution, range).
        obstacles: Obstacle centres, ``(M, 2)``.
        obstacle_radius: Obstacle radius.
        noise: Optional additive noise model.
        salt_pepper: Optional salt-and-pepper noise.

    Returns:
        ``(ranges, angles)`` where *ranges* and *angles* are both
        ``(N,)`` arrays.
    """
    rays = compute_fov_rays(sensor_x, sensor_y, sensor_yaw, mount)
    n_rays = rays.shape[0]
    ranges = np.full(n_rays, mount.max_range)
    half_fov = mount.fov_horizontal / 2.0
    angles = np.linspace(sensor_yaw - half_fov, sensor_yaw + half_fov, n_rays)

    origin = np.array([sensor_x, sensor_y])
    for ri in range(n_rays):
        ray_dir = rays[ri]
        for obs in obstacles:
            oc = obs[:2] - origin
            proj = float(np.dot(oc, ray_dir))
            if proj < mount.min_range or proj > mount.max_range:
                continue
            closest = origin + ray_dir * proj
            perp_dist = float(np.linalg.norm(obs[:2] - closest))
            if perp_dist < obstacle_radius:
                hit_dist = proj - np.sqrt(max(obstacle_radius**2 - perp_dist**2, 0.0))
                if mount.min_range <= hit_dist < ranges[ri]:
                    ranges[ri] = hit_dist

    # Apply noise.
    if noise is not None:
        if isinstance(noise, RangeProportionalNoise):
            ranges = ranges + noise.sample_at_range(ranges)
        else:
            ranges = ranges + noise.sample(shape=ranges.shape)
    ranges = np.clip(ranges, mount.min_range, mount.max_range)

    if salt_pepper is not None:
        ranges = salt_pepper.apply(ranges, mount.max_range)

    return (ranges, angles)


# -----------------------------------------------------------------------
# Sensor suite
# -----------------------------------------------------------------------


class SensorSuite:
    """Collection of sensors mounted on a robot.

    Manages multiple :class:`SensorMount` instances, computes their
    world-frame poses, and provides aggregate FOV and visibility
    queries.

    Args:
        mounts: List of sensor mounts.
    """

    def __init__(self, mounts: list[SensorMount] | None = None) -> None:
        self._mounts: list[SensorMount] = list(mounts) if mounts else []

    # ----- Management ---------------------------------------------------

    def add(self, mount: SensorMount) -> None:
        """Add a sensor to the suite."""
        self._mounts.append(mount)

    def remove(self, name: str) -> None:
        """Remove a sensor by name."""
        self._mounts = [m for m in self._mounts if m.name != name]

    def get(self, name: str) -> SensorMount | None:
        """Retrieve a sensor by name."""
        for m in self._mounts:
            if m.name == name:
                return m
        return None

    @property
    def mounts(self) -> list[SensorMount]:
        """All sensor mounts."""
        return list(self._mounts)

    @property
    def num_sensors(self) -> int:
        """Number of sensors."""
        return len(self._mounts)

    def enabled_mounts(self) -> list[SensorMount]:
        """Return only enabled sensors."""
        return [m for m in self._mounts if m.enabled]

    # ----- Pose computation ---------------------------------------------

    def world_poses_2d(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> list[tuple[str, float, float, float]]:
        """Compute world-frame 2-D poses of all enabled sensors.

        Args:
            robot_x, robot_y: Robot position.
            robot_theta: Robot heading.

        Returns:
            List of ``(name, sx, sy, s_yaw)`` tuples.
        """
        result = []
        for m in self._mounts:
            if not m.enabled:
                continue
            sx, sy, syaw = sensor_world_pose_2d(robot_x, robot_y, robot_theta, m)
            result.append((m.name, sx, sy, syaw))
        return result

    # ----- FOV ----------------------------------------------------------

    def combined_fov_polygon(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        num_points: int = 64,
    ) -> list[np.ndarray]:
        """Compute FOV polygons for all enabled sensors.

        Args:
            robot_x, robot_y, robot_theta: Robot pose.
            num_points: Arc resolution per sensor.

        Returns:
            List of polygon arrays, one per enabled sensor.
        """
        polygons = []
        for m in self._mounts:
            if not m.enabled:
                continue
            sx, sy, syaw = sensor_world_pose_2d(robot_x, robot_y, robot_theta, m)
            poly = compute_fov_polygon(sx, sy, syaw, m.fov_horizontal, m.max_range, num_points)
            polygons.append(poly)
        return polygons

    # ----- Visibility ---------------------------------------------------

    def any_sensor_sees(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        point: np.ndarray,
    ) -> bool:
        """Check if any enabled sensor can see *point*.

        Args:
            robot_x, robot_y, robot_theta: Robot pose.
            point: Target point, ``(2,)``.

        Returns:
            ``True`` if at least one sensor has the point in its FOV.
        """
        for m in self._mounts:
            if not m.enabled:
                continue
            sx, sy, syaw = sensor_world_pose_2d(robot_x, robot_y, robot_theta, m)
            if check_point_visibility(sx, sy, syaw, m, point):
                return True
        return False

    def visible_from_all(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        points: np.ndarray,
        obstacles: np.ndarray | None = None,
        obstacle_radius: float = 0.3,
    ) -> np.ndarray:
        """Compute a combined visibility mask across all enabled sensors.

        A point is marked visible if *any* sensor can see it.

        Args:
            robot_x, robot_y, robot_theta: Robot pose.
            points: Shape ``(N, 2)``.
            obstacles: Optional obstacle centres.
            obstacle_radius: Obstacle radius.

        Returns:
            Boolean mask, shape ``(N,)``.
        """
        n = points.shape[0]
        combined = np.zeros(n, dtype=bool)
        for m in self._mounts:
            if not m.enabled:
                continue
            sx, sy, syaw = sensor_world_pose_2d(robot_x, robot_y, robot_theta, m)
            vis = compute_visible_points(
                sx,
                sy,
                syaw,
                m,
                points,
                obstacles,
                obstacle_radius,
            )
            combined |= vis
        return combined

    # ----- Simulated scans ----------------------------------------------

    def scan_all(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        obstacles: np.ndarray,
        obstacle_radius: float = 0.3,
        noise: GaussianNoise | RangeProportionalNoise | None = None,
        salt_pepper: SaltPepperNoise | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run a range scan for every enabled range sensor.

        Args:
            robot_x, robot_y, robot_theta: Robot pose.
            obstacles: Obstacle centres, ``(M, 2)``.
            obstacle_radius: Obstacle radius.
            noise: Additive noise model.
            salt_pepper: Salt-and-pepper noise.

        Returns:
            Dict mapping sensor name to ``(ranges, angles)``.
        """
        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        range_types = {SensorType.LIDAR_2D, SensorType.LIDAR_3D, SensorType.ULTRASONIC}
        for m in self._mounts:
            if not m.enabled or m.sensor_type not in range_types:
                continue
            sx, sy, syaw = sensor_world_pose_2d(robot_x, robot_y, robot_theta, m)
            r, a = simulate_range_scan(
                sx,
                sy,
                syaw,
                m,
                obstacles,
                obstacle_radius,
                noise,
                salt_pepper,
            )
            results[m.name] = (r, a)
        return results


# -----------------------------------------------------------------------
# Sensor fusion configuration
# -----------------------------------------------------------------------


@dataclass
class FusionWeight:
    """Weight assigned to a sensor in a fusion pipeline.

    Attributes:
        sensor_name: Name of the sensor.
        weight: Relative weight (higher = more trusted).
        delay_s: Sensor delay / latency (seconds).
    """

    sensor_name: str = ""
    weight: float = 1.0
    delay_s: float = 0.0


@dataclass
class SensorFusionConfig:
    """Configuration for multi-sensor fusion.

    Attributes:
        weights: Per-sensor fusion weights.
        fusion_method: Algorithm name (e.g. ``"weighted_average"``,
            ``"kalman"``).
        outlier_rejection: If ``True`` apply outlier filtering.
        outlier_threshold: Mahalanobis-distance threshold for outliers.
    """

    weights: list[FusionWeight] = field(default_factory=list)
    fusion_method: str = "weighted_average"
    outlier_rejection: bool = False
    outlier_threshold: float = 3.0


def fuse_position_estimates(
    estimates: dict[str, np.ndarray],
    config: SensorFusionConfig,
) -> np.ndarray:
    """Fuse multiple position estimates using weighted average.

    Args:
        estimates: Mapping from sensor name to position ``(2,)`` or
            ``(3,)``.
        config: Fusion configuration.

    Returns:
        Fused position estimate.
    """
    if not estimates:
        return np.zeros(2)

    weight_map: dict[str, float] = {w.sensor_name: w.weight for w in config.weights}
    total_weight = 0.0
    dim = None
    fused = None

    for name, est in estimates.items():
        if dim is None:
            dim = est.shape[0]
            fused = np.zeros(dim)
        w = weight_map.get(name, 1.0)
        total_weight += w
        fused += w * est  # type: ignore[operator]

    if fused is None or total_weight < 1e-12:
        return np.zeros(2)
    return fused / total_weight


def fuse_with_covariance(
    estimates: dict[str, np.ndarray],
    covariances: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse position estimates using inverse-covariance weighting.

    Each estimate is weighted by the inverse of its covariance matrix
    (information filter fusion).

    Args:
        estimates: Sensor name to position, ``(D,)``.
        covariances: Sensor name to covariance, ``(D, D)``.

    Returns:
        ``(fused_mean, fused_covariance)``.
    """
    if not estimates:
        return (np.zeros(2), np.eye(2) * 1e6)

    dim = None
    info_matrix = None
    info_vector = None

    for name in estimates:
        est = estimates[name]
        cov = covariances.get(name)
        if dim is None:
            dim = est.shape[0]
            info_matrix = np.zeros((dim, dim))
            info_vector = np.zeros(dim)
        if cov is None:
            cov = np.eye(dim)
        inv_cov = np.linalg.inv(cov + np.eye(dim) * 1e-9)
        info_matrix += inv_cov  # type: ignore[operator]
        info_vector += inv_cov @ est  # type: ignore[operator]

    if info_matrix is None or info_vector is None:
        return (np.zeros(2), np.eye(2) * 1e6)

    fused_cov = np.linalg.inv(info_matrix + np.eye(info_matrix.shape[0]) * 1e-9)
    fused_mean = fused_cov @ info_vector
    return (fused_mean, fused_cov)


# -----------------------------------------------------------------------
# Preset sensor configurations
# -----------------------------------------------------------------------


def default_mobile_robot_suite() -> SensorSuite:
    """Create a typical mobile-robot sensor suite.

    Includes a front-facing 2-D LiDAR, a rear ultrasonic sensor, and
    wheel encoders.

    Returns:
        Pre-configured :class:`SensorSuite`.
    """
    front_lidar = SensorMount(
        name="front_lidar",
        sensor_type=SensorType.LIDAR_2D,
        offset_x=0.2,
        offset_y=0.0,
        fov_horizontal=np.radians(270.0),
        max_range=30.0,
        min_range=0.05,
        resolution_horizontal=np.radians(0.5),
        update_rate=10.0,
    )
    rear_ultrasonic = SensorMount(
        name="rear_ultrasonic",
        sensor_type=SensorType.ULTRASONIC,
        offset_x=-0.2,
        offset_y=0.0,
        yaw=np.pi,
        fov_horizontal=np.radians(30.0),
        max_range=3.0,
        min_range=0.02,
        resolution_horizontal=np.radians(5.0),
        update_rate=20.0,
    )
    encoders = SensorMount(
        name="wheel_encoders",
        sensor_type=SensorType.WHEEL_ENCODER,
        offset_x=0.0,
        offset_y=0.0,
        fov_horizontal=0.0,
        max_range=0.0,
        update_rate=100.0,
    )
    return SensorSuite([front_lidar, rear_ultrasonic, encoders])


def autonomous_vehicle_suite() -> SensorSuite:
    """Create a sensor suite typical for an autonomous vehicle.

    Includes roof LiDAR, front camera, and GPS.

    Returns:
        Pre-configured :class:`SensorSuite`.
    """
    roof_lidar = SensorMount(
        name="roof_lidar",
        sensor_type=SensorType.LIDAR_3D,
        offset_x=0.0,
        offset_y=0.0,
        offset_z=1.8,
        fov_horizontal=2.0 * np.pi,
        fov_vertical=np.radians(40.0),
        max_range=100.0,
        min_range=0.5,
        resolution_horizontal=np.radians(0.2),
        resolution_vertical=np.radians(0.4),
        update_rate=10.0,
    )
    front_camera = SensorMount(
        name="front_camera",
        sensor_type=SensorType.CAMERA_RGB,
        offset_x=1.0,
        offset_y=0.0,
        offset_z=1.2,
        fov_horizontal=np.radians(90.0),
        fov_vertical=np.radians(60.0),
        max_range=80.0,
        min_range=0.5,
        resolution_horizontal=np.radians(0.05),
        resolution_vertical=np.radians(0.05),
        update_rate=30.0,
    )
    gps = SensorMount(
        name="gps",
        sensor_type=SensorType.GPS,
        offset_x=0.0,
        offset_y=0.0,
        offset_z=1.8,
        fov_horizontal=0.0,
        max_range=0.0,
        update_rate=5.0,
    )
    return SensorSuite([roof_lidar, front_camera, gps])
