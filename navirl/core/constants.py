"""Physical and simulation constants for pedestrian navigation.

This module centralises every magic number used across the NavIRL framework.
Values are drawn from the pedestrian-dynamics literature (Helbing & Molnár 1995,
Hall 1966, Moussaïd et al. 2009) and standard robotics references.

All units are SI unless explicitly noted:
    length  → metres (m)
    time    → seconds (s)
    angle   → radians (rad)
    mass    → kilograms (kg)
    speed   → metres per second (m/s)
    force   → newtons (N)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
#  Fundamental physical constants
# ---------------------------------------------------------------------------

GRAVITY: Final[float] = 9.81  # m/s^2
EPSILON: Final[float] = 1e-8  # numerical guard against division by zero
TWO_PI: Final[float] = 2.0 * math.pi
HALF_PI: Final[float] = math.pi / 2.0
DEG2RAD: Final[float] = math.pi / 180.0
RAD2DEG: Final[float] = 180.0 / math.pi

# ---------------------------------------------------------------------------
#  Pedestrian body dimensions (adult, 50th‑percentile)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BodyDimensions:
    """Average adult pedestrian body measurements."""

    shoulder_width: float = 0.45  # m – bideltoid breadth
    chest_depth: float = 0.25  # m – anteroposterior diameter
    body_radius: float = 0.25  # m – bounding-circle radius (typical)
    body_radius_min: float = 0.18  # m – slim adult
    body_radius_max: float = 0.35  # m – large adult / with backpack
    height_mean: float = 1.70  # m
    height_std: float = 0.10  # m
    mass_mean: float = 75.0  # kg
    mass_std: float = 12.0  # kg


BODY: Final[BodyDimensions] = BodyDimensions()

# ---------------------------------------------------------------------------
#  Hall's proxemics – interpersonal distance zones
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProxemicZone:
    """Distance boundaries for a single proxemic zone (metres)."""

    inner: float
    outer: float

    @property
    def mid(self) -> float:
        return (self.inner + self.outer) / 2.0

    def contains(self, distance: float) -> bool:
        return self.inner <= distance < self.outer


@dataclass(frozen=True, slots=True)
class ProxemicZones:
    """Hall's four interpersonal distance zones.

    Reference: Hall, E. T. (1966) *The Hidden Dimension*.
    """

    intimate: ProxemicZone = ProxemicZone(inner=0.0, outer=0.45)
    personal: ProxemicZone = ProxemicZone(inner=0.45, outer=1.2)
    social: ProxemicZone = ProxemicZone(inner=1.2, outer=3.6)
    public: ProxemicZone = ProxemicZone(inner=3.6, outer=7.6)

    def classify(self, distance: float) -> str:
        """Return the zone name for *distance* (metres)."""
        if distance < self.intimate.outer:
            return "intimate"
        if distance < self.personal.outer:
            return "personal"
        if distance < self.social.outer:
            return "social"
        return "public"

    def all_zones(self) -> dict[str, ProxemicZone]:
        return {
            "intimate": self.intimate,
            "personal": self.personal,
            "social": self.social,
            "public": self.public,
        }


PROXEMICS: Final[ProxemicZones] = ProxemicZones()

# ---------------------------------------------------------------------------
#  Walking speed distributions
# ---------------------------------------------------------------------------


class GaitType(Enum):
    """Coarse gait categories."""

    SLOW = auto()
    NORMAL = auto()
    FAST = auto()
    RUNNING = auto()


@dataclass(frozen=True, slots=True)
class SpeedDistribution:
    """Gaussian distribution over walking speed (m/s)."""

    mean: float
    std: float
    min_speed: float
    max_speed: float

    def sample(self, rng: np.random.Generator | None = None, size: int = 1) -> np.ndarray:
        """Draw *size* samples, clipped to [min_speed, max_speed]."""
        _rng = rng if rng is not None else np.random.default_rng()
        raw = _rng.normal(self.mean, self.std, size=size)
        return np.clip(raw, self.min_speed, self.max_speed)


SPEED_DISTRIBUTIONS: Final[dict[GaitType, SpeedDistribution]] = {
    GaitType.SLOW: SpeedDistribution(mean=0.7, std=0.15, min_speed=0.2, max_speed=1.0),
    GaitType.NORMAL: SpeedDistribution(mean=1.34, std=0.26, min_speed=0.6, max_speed=2.0),
    GaitType.FAST: SpeedDistribution(mean=1.80, std=0.20, min_speed=1.2, max_speed=2.5),
    GaitType.RUNNING: SpeedDistribution(mean=3.0, std=0.50, min_speed=2.0, max_speed=5.0),
}

# Convenience shortcuts
PREFERRED_SPEED_MEAN: Final[float] = SPEED_DISTRIBUTIONS[GaitType.NORMAL].mean
PREFERRED_SPEED_STD: Final[float] = SPEED_DISTRIBUTIONS[GaitType.NORMAL].std
MAX_PEDESTRIAN_SPEED: Final[float] = 2.5  # m/s (excluding runners)
MAX_RUNNING_SPEED: Final[float] = 5.0  # m/s

# ---------------------------------------------------------------------------
#  Comfort / preference parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComfortParameters:
    """Preferred interpersonal and group-related distances."""

    # Preferred distance to maintain from strangers while walking
    preferred_stranger_distance: float = 1.0  # m
    preferred_stranger_distance_std: float = 0.3  # m
    # Minimum tolerable distance before discomfort
    min_comfortable_distance: float = 0.5  # m
    # Group cohesion: preferred inter-member distance
    group_cohesion_distance: float = 1.2  # m
    group_cohesion_distance_std: float = 0.3  # m
    # Maximum separation before a group member tries to rejoin
    group_max_separation: float = 3.0  # m
    # Preferred lateral offset when walking side-by-side
    side_by_side_offset: float = 0.7  # m
    # Personal "bubble" for collision-avoidance triggers
    collision_avoidance_horizon: float = 5.0  # m (look-ahead distance)
    collision_avoidance_time_horizon: float = 3.0  # s
    # Speed adaptation: fraction of preferred speed tolerated in dense crowds
    min_speed_fraction: float = 0.3
    # Desired time gap behind the person in front (car-following analogy)
    desired_time_gap: float = 0.8  # s
    # Relaxation time for speed adaptation
    relaxation_time: float = 0.5  # s


COMFORT: Final[ComfortParameters] = ComfortParameters()

# ---------------------------------------------------------------------------
#  Social Force Model (SFM) constants
#  Reference: Helbing, D. & Molnár, P. (1995)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SocialForceConstants:
    """Parameters for the standard Social Force Model."""

    # --- Repulsive interaction force ---
    # V_alpha_beta = A * exp((r_ab - d_ab) / B) * w(theta)
    A: float = 2.1  # N  – strength of repulsive interaction
    B: float = 0.3  # m  – range of repulsive interaction
    A_wall: float = 10.0  # N  – strength of wall repulsion
    B_wall: float = 0.2  # m  – range of wall repulsion
    # --- Anisotropy (field-of-view weighting) ---
    # w(theta) = lambda + (1 - lambda) * (1 + cos(theta)) / 2
    lambda_anisotropy: float = 0.35  # 0 = isotropic, 1 = only forward
    # --- Body compression / sliding friction (contact forces) ---
    k_body: float = 1.2e5  # N/m  – body-compression spring constant
    kappa_friction: float = 2.4e5  # kg/(m·s) – sliding friction coefficient
    # --- Desired-force relaxation ---
    tau: float = 0.5  # s  – relaxation time to preferred velocity
    # --- Attractive force (groups) ---
    C_attraction: float = 0.5  # N  – group-attraction strength
    # --- Pedestrian-to-pedestrian interaction cutoff ---
    interaction_range: float = 5.0  # m – ignore ped-ped forces beyond this
    # --- Wall interaction cutoff ---
    wall_interaction_range: float = 3.0  # m


SFM: Final[SocialForceConstants] = SocialForceConstants()

# ---------------------------------------------------------------------------
#  Optimal Reciprocal Collision Avoidance (ORCA) defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ORCADefaults:
    """Default parameters for the ORCA / RVO2 algorithm."""

    time_horizon: float = 3.0  # s
    time_horizon_obstacle: float = 1.5  # s
    neighbor_distance: float = 5.0  # m
    max_neighbors: int = 10
    # Added safety margin on top of agent radii
    safety_margin: float = 0.05  # m


ORCA: Final[ORCADefaults] = ORCADefaults()

# ---------------------------------------------------------------------------
#  Environment / architectural constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorridorDimensions:
    """Typical built-environment dimensions (metres)."""

    # Corridors
    corridor_width_narrow: float = 1.2  # single-file corridor
    corridor_width_standard: float = 2.0  # two-way traffic
    corridor_width_wide: float = 3.5  # generous hallway
    # Doors
    door_width_single: float = 0.9  # standard single door
    door_width_double: float = 1.8  # standard double door
    door_width_emergency: float = 1.2  # emergency exit minimum
    # Rooms
    room_size_small: tuple[float, float] = (4.0, 4.0)
    room_size_medium: tuple[float, float] = (8.0, 8.0)
    room_size_large: tuple[float, float] = (15.0, 15.0)
    # Stairways
    stair_width_min: float = 1.1
    stair_tread_depth: float = 0.28
    stair_riser_height: float = 0.18
    # Elevators
    elevator_width: float = 1.5
    elevator_depth: float = 1.5
    elevator_capacity: int = 8


ENVIRONMENT: Final[CorridorDimensions] = CorridorDimensions()

# ---------------------------------------------------------------------------
#  Level-of-service (Fruin) – pedestrian density thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LevelOfService:
    """Fruin level-of-service for walkways (ped/m²)."""

    A_max_density: float = 0.3  # free flow
    B_max_density: float = 0.5  # minor conflicts
    C_max_density: float = 0.7  # restricted speed
    D_max_density: float = 1.1  # severely restricted
    E_max_density: float = 1.7  # shuffling
    F_min_density: float = 1.7  # breakdown / stampede risk


LOS: Final[LevelOfService] = LevelOfService()

# ---------------------------------------------------------------------------
#  Simulation defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulationDefaults:
    """Default simulation parameters."""

    dt: float = 0.04  # s  – physics timestep (25 Hz)
    max_steps: int = 10_000  # safety cap
    max_time: float = 300.0  # s  – 5-minute timeout
    default_fps: int = 25  # frames per second for rendering
    substeps: int = 1  # physics substeps per control step
    warmup_steps: int = 0  # steps before RL control starts
    # Collision detection
    collision_eps: float = 0.01  # m – penetration tolerance
    # Goal reaching
    goal_radius: float = 0.5  # m – distance considered "arrived"
    # Random seed
    default_seed: int = 42
    # Recording
    record_every_n_steps: int = 1  # state-recording cadence
    # World bounds (if not specified by scenario)
    default_world_width: float = 30.0  # m
    default_world_height: float = 30.0  # m


SIM: Final[SimulationDefaults] = SimulationDefaults()

# ---------------------------------------------------------------------------
#  Sensor parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LidarParams:
    """2-D planar LiDAR sensor configuration."""

    num_beams: int = 360
    max_range: float = 10.0  # m
    min_range: float = 0.1  # m
    fov: float = TWO_PI  # rad (full 360°)
    angular_resolution: float = TWO_PI / 360  # rad
    noise_std: float = 0.01  # m – range noise
    # Beam grouping for lower-dimensional observations
    num_sectors: int = 36  # coarsened sectors

    @property
    def angles(self) -> np.ndarray:
        """Beam angles centred on the forward direction."""
        half = self.fov / 2.0
        return np.linspace(-half, half, self.num_beams, endpoint=False)


@dataclass(frozen=True, slots=True)
class CameraParams:
    """Simulated monocular camera parameters."""

    fov_horizontal: float = 1.22  # rad ≈ 70°
    fov_vertical: float = 0.87  # rad ≈ 50°
    resolution_x: int = 640
    resolution_y: int = 480
    max_depth: float = 20.0  # m
    focal_length_px: float = 525.0  # px (typical for 640×480 at 70° HFOV)


@dataclass(frozen=True, slots=True)
class DepthSensorParams:
    """Simulated depth sensor."""

    fov_horizontal: float = 1.22  # rad
    resolution: int = 64  # depth bins across FOV
    max_range: float = 8.0  # m
    min_range: float = 0.3  # m
    noise_std: float = 0.02  # m


LIDAR: Final[LidarParams] = LidarParams()
CAMERA: Final[CameraParams] = CameraParams()
DEPTH_SENSOR: Final[DepthSensorParams] = DepthSensorParams()

# ---------------------------------------------------------------------------
#  RL training defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RLTrainingDefaults:
    """Hyper-parameter defaults for reinforcement learning."""

    # Optimiser
    learning_rate: float = 3e-4
    learning_rate_min: float = 1e-6
    lr_schedule: str = "linear"  # "linear" | "cosine" | "constant"
    # Batching
    batch_size: int = 256
    mini_batch_size: int = 64
    buffer_size: int = 1_000_000
    # Discount
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # PPO specifics
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    # SAC specifics
    sac_alpha: float = 0.2  # entropy coefficient
    sac_tau: float = 0.005  # soft update factor
    sac_auto_alpha: bool = True  # auto-tune entropy
    # DQN specifics
    dqn_eps_start: float = 1.0
    dqn_eps_end: float = 0.05
    dqn_eps_decay_steps: int = 100_000
    target_update_freq: int = 1_000
    # General training
    total_timesteps: int = 5_000_000
    eval_interval: int = 50_000
    eval_episodes: int = 20
    save_interval: int = 100_000
    log_interval: int = 5_000
    # Normalisation
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    # Network architecture defaults
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    # Multi-agent
    n_envs: int = 8  # parallel environments


RL: Final[RLTrainingDefaults] = RLTrainingDefaults()

# ---------------------------------------------------------------------------
#  Reward shaping constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardConstants:
    """Default reward-function coefficients."""

    # Goal-reaching
    reward_goal_reached: float = 10.0
    reward_goal_progress: float = 1.0  # per metre closer
    # Collision penalties
    penalty_collision_pedestrian: float = -5.0
    penalty_collision_wall: float = -3.0
    penalty_intimate_zone: float = -1.0  # entering intimate zone
    # Time penalty
    penalty_per_step: float = -0.01
    penalty_timeout: float = -5.0
    # Comfort bonuses / penalties
    penalty_jerk: float = -0.1  # per unit of jerk
    penalty_speed_deviation: float = -0.2  # deviation from preferred speed
    bonus_smooth_path: float = 0.05
    # Social norms
    penalty_wrong_side: float = -0.5  # walking on wrong side of corridor
    bonus_right_side: float = 0.1


REWARD: Final[RewardConstants] = RewardConstants()

# ---------------------------------------------------------------------------
#  Crowd-density estimation bins
# ---------------------------------------------------------------------------

DENSITY_BINS: Final[np.ndarray] = np.array(
    [0.0, 0.3, 0.5, 0.7, 1.1, 1.7, 2.5, 4.0, 6.0], dtype=np.float64
)
DENSITY_LABELS: Final[tuple[str, ...]] = (
    "free_flow",
    "minor_conflict",
    "restricted",
    "severely_restricted",
    "shuffling",
    "congested",
    "crush_warning",
    "crush",
)

# ---------------------------------------------------------------------------
#  Colour palette (RGB 0-255) for visualisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Palette:
    """Standard colour palette for rendering pedestrians and obstacles."""

    background: tuple[int, int, int] = (245, 245, 240)
    robot: tuple[int, int, int] = (31, 119, 180)
    human: tuple[int, int, int] = (255, 127, 14)
    human_alt: tuple[int, int, int] = (44, 160, 44)
    obstacle: tuple[int, int, int] = (80, 80, 80)
    wall: tuple[int, int, int] = (40, 40, 40)
    goal: tuple[int, int, int] = (214, 39, 40)
    path: tuple[int, int, int] = (148, 103, 189)
    lidar_beam: tuple[int, int, int] = (200, 200, 200)
    collision_highlight: tuple[int, int, int] = (227, 66, 52)
    intimate_zone: tuple[int, int, int] = (255, 200, 200)
    personal_zone: tuple[int, int, int] = (255, 230, 200)
    social_zone: tuple[int, int, int] = (200, 230, 255)
    text: tuple[int, int, int] = (30, 30, 30)
    grid_line: tuple[int, int, int] = (220, 220, 220)


PALETTE: Final[Palette] = Palette()

# ---------------------------------------------------------------------------
#  Precomputed look-up tables
# ---------------------------------------------------------------------------

# Unit vectors for 8 cardinal / inter-cardinal directions (row = direction)
DIRECTION_VECTORS_8: Final[np.ndarray] = np.array(
    [
        [1.0, 0.0],  # E
        [0.707, 0.707],  # NE  (approx √2/2)
        [0.0, 1.0],  # N
        [-0.707, 0.707],  # NW
        [-1.0, 0.0],  # W
        [-0.707, -0.707],  # SW
        [0.0, -1.0],  # S
        [0.707, -0.707],  # SE
    ],
    dtype=np.float64,
)

# Pre-computed sin/cos tables for fast LiDAR ray casting
_LIDAR_ANGLES: Final[np.ndarray] = LIDAR.angles
LIDAR_COS_TABLE: Final[np.ndarray] = np.cos(_LIDAR_ANGLES)
LIDAR_SIN_TABLE: Final[np.ndarray] = np.sin(_LIDAR_ANGLES)

# ---------------------------------------------------------------------------
#  Named configuration presets (shorthand references)
# ---------------------------------------------------------------------------

PRESET_NAMES: Final[tuple[str, ...]] = (
    "debug",
    "fast_train",
    "full_train",
    "full_eval",
    "benchmark",
    "visualise",
)

# ---------------------------------------------------------------------------
#  Miscellaneous
# ---------------------------------------------------------------------------

# Maximum agents the engine is designed to support per scenario
MAX_AGENTS: Final[int] = 500

# Minimum scenario area per agent to avoid unreasonable densities (m²/agent)
MIN_AREA_PER_AGENT: Final[float] = 1.0

# Standard robot radius (differential-drive style)
ROBOT_RADIUS: Final[float] = 0.20  # m
ROBOT_MAX_SPEED: Final[float] = 1.0  # m/s
ROBOT_MAX_ACCEL: Final[float] = 2.0  # m/s²
ROBOT_MAX_ANGULAR_SPEED: Final[float] = 1.5  # rad/s

# Typical reaction time for pedestrians
REACTION_TIME_MEAN: Final[float] = 0.5  # s
REACTION_TIME_STD: Final[float] = 0.15  # s

# Fundamental diagram constants (speed-density relationship)
# v(rho) = v_free * (1 - rho / rho_max) — Weidmann
FUNDAMENTAL_DIAGRAM_V_FREE: Final[float] = 1.34  # m/s
FUNDAMENTAL_DIAGRAM_RHO_MAX: Final[float] = 5.4  # ped/m²

# Step length / cadence
STEP_LENGTH_MEAN: Final[float] = 0.70  # m
STEP_CADENCE_MEAN: Final[float] = 1.9  # steps/s

# ---------------------------------------------------------------------------
#  Navigation metrics constants
# ---------------------------------------------------------------------------

# Distance and proximity thresholds
INTRUSION_DISTANCE_THRESHOLD: Final[float] = 0.45  # m - personal space intrusion threshold
GOAL_TOLERANCE: Final[float] = 0.2  # m - distance to consider goal reached

# Deadlock detection parameters
DEADLOCK_TIMEOUT_SECONDS: Final[float] = 4.0  # s - time window for deadlock detection
DEADLOCK_SPEED_THRESHOLD: Final[float] = (
    0.015  # m/s - speed below which agent is considered stopped
)

# Angle computation thresholds
ANGLE_EPSILON: Final[float] = 1e-4  # rad - threshold for near-zero angle differences

# ---------------------------------------------------------------------------
#  Evaluation and metrics constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComfortEvaluationLimits:
    """Comfort evaluation thresholds for trajectory analysis."""

    max_speed: float = 2.0  # m/s - maximum comfortable walking speed
    max_accel: float = 1.5  # m/s² - maximum comfortable acceleration
    max_jerk: float = 3.0  # m/s³ - maximum comfortable jerk


COMFORT_LIMITS: Final[ComfortEvaluationLimits] = ComfortEvaluationLimits()

# Statistical analysis defaults
DEFAULT_BOOTSTRAP_SAMPLES: Final[int] = 1000  # number of bootstrap samples for CI estimation
DEFAULT_CONFIDENCE_LEVEL: Final[float] = 0.95  # confidence level for statistical tests

# ---------------------------------------------------------------------------
#  Aggregate export
# ---------------------------------------------------------------------------

__all__ = [
    # Fundamental
    "GRAVITY",
    "EPSILON",
    "TWO_PI",
    "HALF_PI",
    "DEG2RAD",
    "RAD2DEG",
    # Body
    "BODY",
    "BodyDimensions",
    # Proxemics
    "PROXEMICS",
    "ProxemicZone",
    "ProxemicZones",
    # Speed
    "GaitType",
    "SpeedDistribution",
    "SPEED_DISTRIBUTIONS",
    "PREFERRED_SPEED_MEAN",
    "PREFERRED_SPEED_STD",
    "MAX_PEDESTRIAN_SPEED",
    "MAX_RUNNING_SPEED",
    # Comfort
    "COMFORT",
    "ComfortParameters",
    # SFM
    "SFM",
    "SocialForceConstants",
    # ORCA
    "ORCA",
    "ORCADefaults",
    # Environment
    "ENVIRONMENT",
    "CorridorDimensions",
    # Level of service
    "LOS",
    "LevelOfService",
    # Simulation
    "SIM",
    "SimulationDefaults",
    # Sensors
    "LIDAR",
    "LidarParams",
    "CAMERA",
    "CameraParams",
    "DEPTH_SENSOR",
    "DepthSensorParams",
    "LIDAR_COS_TABLE",
    "LIDAR_SIN_TABLE",
    # RL
    "RL",
    "RLTrainingDefaults",
    # Reward
    "REWARD",
    "RewardConstants",
    # Density
    "DENSITY_BINS",
    "DENSITY_LABELS",
    # Palette
    "PALETTE",
    "Palette",
    # Directions
    "DIRECTION_VECTORS_8",
    # Presets
    "PRESET_NAMES",
    # Misc
    "MAX_AGENTS",
    "MIN_AREA_PER_AGENT",
    "ROBOT_RADIUS",
    "ROBOT_MAX_SPEED",
    "ROBOT_MAX_ACCEL",
    "ROBOT_MAX_ANGULAR_SPEED",
    "REACTION_TIME_MEAN",
    "REACTION_TIME_STD",
    "FUNDAMENTAL_DIAGRAM_V_FREE",
    "FUNDAMENTAL_DIAGRAM_RHO_MAX",
    "STEP_LENGTH_MEAN",
    "STEP_CADENCE_MEAN",
    # Navigation metrics
    "INTRUSION_DISTANCE_THRESHOLD",
    "GOAL_TOLERANCE",
    "DEADLOCK_TIMEOUT_SECONDS",
    "DEADLOCK_SPEED_THRESHOLD",
    "ANGLE_EPSILON",
    # Evaluation metrics
    "COMFORT_LIMITS",
    "ComfortEvaluationLimits",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
]
