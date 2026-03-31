"""Crowd generation with configurable density, flow patterns, and demographics.

Provides :class:`CrowdGenerator` for spawning pedestrian crowds with
controllable density, speed/radius distributions, spawn strategies (Poisson
process, batch, scheduled), goal assignment, and pre-built scenario
templates (commute, evacuation, event gathering, random walk).

All random sampling uses :mod:`numpy` only.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from navirl.humans.pedestrian_state import (
    Activity,
    PedestrianState,
    PersonalityTag,
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SpawnStrategy(enum.Enum):
    """How pedestrians are introduced over time."""

    BATCH = "batch"
    POISSON = "poisson"
    SCHEDULED = "scheduled"


class FlowPattern(enum.Enum):
    """Macroscopic flow pattern for the crowd."""

    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional"
    CROSSING = "crossing"
    RADIAL_IN = "radial_in"
    RADIAL_OUT = "radial_out"
    RANDOM = "random"


class ScenarioType(enum.Enum):
    """Pre-built crowd scenario templates."""

    COMMUTE = "commute"
    EVACUATION = "evacuation"
    EVENT_GATHERING = "event_gathering"
    RANDOM_WALK = "random_walk"


# ---------------------------------------------------------------------------
# Demographic distribution
# ---------------------------------------------------------------------------


@dataclass
class DemographicDistribution:
    """Statistical distribution parameters for pedestrian demographics.

    Parameters
    ----------
    speed_mean : float
        Mean preferred speed (m/s).
    speed_std : float
        Standard deviation of preferred speed.
    speed_min : float
        Minimum clamp for preferred speed.
    speed_max : float
        Maximum clamp for preferred speed.
    radius_mean : float
        Mean body radius (m).
    radius_std : float
        Standard deviation of body radius.
    radius_min : float
        Minimum clamp for radius.
    radius_max : float
        Maximum clamp for radius.
    personality_weights : dict[PersonalityTag, float] or None
        Relative weights for personality sampling.  ``None`` uses defaults.
    """

    speed_mean: float = 1.34
    speed_std: float = 0.26
    speed_min: float = 0.5
    speed_max: float = 2.0
    radius_mean: float = 0.3
    radius_std: float = 0.05
    radius_min: float = 0.2
    radius_max: float = 0.5
    personality_weights: dict[PersonalityTag, float] | None = None

    def sample_speed(self, rng: np.random.Generator) -> float:
        """Sample a preferred speed.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled speed clamped to ``[speed_min, speed_max]``.
        """
        s = float(rng.normal(self.speed_mean, self.speed_std))
        return max(self.speed_min, min(self.speed_max, s))

    def sample_radius(self, rng: np.random.Generator) -> float:
        """Sample a body radius.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled radius clamped to ``[radius_min, radius_max]``.
        """
        r = float(rng.normal(self.radius_mean, self.radius_std))
        return max(self.radius_min, min(self.radius_max, r))

    def sample_personality(self, rng: np.random.Generator) -> PersonalityTag:
        """Sample a personality tag.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        PersonalityTag
            Sampled personality.
        """
        weights = self.personality_weights
        if weights is None:
            weights = {
                PersonalityTag.NORMAL: 0.45,
                PersonalityTag.AGGRESSIVE: 0.10,
                PersonalityTag.PASSIVE: 0.15,
                PersonalityTag.DISTRACTED: 0.10,
                PersonalityTag.HURRIED: 0.10,
                PersonalityTag.ELDERLY: 0.05,
                PersonalityTag.CHILD: 0.05,
            }
        tags = list(weights.keys())
        probs = np.array([weights[t] for t in tags], dtype=np.float64)
        probs /= probs.sum()
        return tags[int(rng.choice(len(tags), p=probs))]


# ---------------------------------------------------------------------------
# Spawn region
# ---------------------------------------------------------------------------


@dataclass
class SpawnRegion:
    """Rectangular region from which pedestrians can be spawned.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Bounds of the rectangle in world coordinates (m).
    """

    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random position inside the region.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        numpy.ndarray
            2-D position ``[x, y]``.
        """
        x = float(rng.uniform(self.x_min, self.x_max))
        y = float(rng.uniform(self.y_min, self.y_max))
        return np.array([x, y], dtype=np.float64)

    @property
    def centre(self) -> np.ndarray:
        """Centre of the region."""
        return np.array(
            [(self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0],
            dtype=np.float64,
        )

    @property
    def area(self) -> float:
        """Area of the region (m^2)."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def contains(self, pos: np.ndarray) -> bool:
        """Check whether *pos* lies inside the region.

        Parameters
        ----------
        pos : numpy.ndarray
            2-D position.

        Returns
        -------
        bool
        """
        return self.x_min <= pos[0] <= self.x_max and self.y_min <= pos[1] <= self.y_max


# ---------------------------------------------------------------------------
# Goal assigner
# ---------------------------------------------------------------------------


class GoalAssigner:
    """Assigns goals to spawned pedestrians.

    Parameters
    ----------
    goal_regions : list[SpawnRegion]
        Candidate goal regions.
    flow_pattern : FlowPattern
        How goals are assigned relative to spawn positions.
    """

    def __init__(
        self,
        goal_regions: list[SpawnRegion] | None = None,
        flow_pattern: FlowPattern = FlowPattern.RANDOM,
    ) -> None:
        self.goal_regions: list[SpawnRegion] = goal_regions or []
        self.flow_pattern: FlowPattern = flow_pattern

    def assign(
        self,
        spawn_pos: np.ndarray,
        spawn_region_idx: int,
        rng: np.random.Generator,
        arena_bounds: SpawnRegion | None = None,
    ) -> np.ndarray:
        """Assign a goal position for a newly spawned pedestrian.

        Parameters
        ----------
        spawn_pos : numpy.ndarray
            2-D spawn position.
        spawn_region_idx : int
            Index of the spawn region from which this pedestrian was spawned.
        rng : numpy.random.Generator
            Random number generator.
        arena_bounds : SpawnRegion or None
            Full arena bounding box (used for some flow patterns).

        Returns
        -------
        numpy.ndarray
            2-D goal position.
        """
        if self.flow_pattern == FlowPattern.RANDOM:
            return self._random_goal(spawn_pos, rng, arena_bounds)

        if self.flow_pattern == FlowPattern.UNIDIRECTIONAL:
            return self._unidirectional_goal(spawn_pos, rng, arena_bounds)

        if self.flow_pattern == FlowPattern.BIDIRECTIONAL:
            return self._bidirectional_goal(spawn_pos, spawn_region_idx, rng)

        if self.flow_pattern == FlowPattern.CROSSING:
            return self._crossing_goal(spawn_pos, spawn_region_idx, rng)

        if self.flow_pattern in (FlowPattern.RADIAL_IN, FlowPattern.RADIAL_OUT):
            return self._radial_goal(spawn_pos, rng, arena_bounds)

        return self._random_goal(spawn_pos, rng, arena_bounds)

    # -- private helpers ------------------------------------------------------

    def _random_goal(
        self,
        spawn_pos: np.ndarray,
        rng: np.random.Generator,
        arena: SpawnRegion | None,
    ) -> np.ndarray:
        if self.goal_regions:
            region = self.goal_regions[int(rng.integers(len(self.goal_regions)))]
            return region.sample(rng)
        if arena is not None:
            return arena.sample(rng)
        # Fallback: random offset.
        angle = float(rng.uniform(0, 2.0 * math.pi))
        dist = float(rng.uniform(5.0, 15.0))
        return spawn_pos + dist * np.array([math.cos(angle), math.sin(angle)])

    def _unidirectional_goal(
        self,
        spawn_pos: np.ndarray,
        rng: np.random.Generator,
        arena: SpawnRegion | None,
    ) -> np.ndarray:
        # Goal is always on the opposite x-side of the arena.
        if arena is not None:
            mid_x = (arena.x_min + arena.x_max) / 2.0
            if spawn_pos[0] < mid_x:
                gx = float(rng.uniform(mid_x, arena.x_max))
            else:
                gx = float(rng.uniform(arena.x_min, mid_x))
            gy = float(rng.uniform(arena.y_min, arena.y_max))
            return np.array([gx, gy], dtype=np.float64)
        return spawn_pos + np.array([20.0, 0.0])

    def _bidirectional_goal(
        self,
        spawn_pos: np.ndarray,
        region_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Pair spawn regions: region 0 -> region 1, etc.
        if len(self.goal_regions) >= 2:
            target_idx = (region_idx + 1) % len(self.goal_regions)
            return self.goal_regions[target_idx].sample(rng)
        return spawn_pos + np.array([15.0, 0.0])

    def _crossing_goal(
        self,
        spawn_pos: np.ndarray,
        region_idx: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(self.goal_regions) >= 2:
            target_idx = (region_idx + len(self.goal_regions) // 2) % len(self.goal_regions)
            return self.goal_regions[target_idx].sample(rng)
        angle = float(rng.uniform(0, 2.0 * math.pi))
        return spawn_pos + 10.0 * np.array([math.cos(angle), math.sin(angle)])

    def _radial_goal(
        self,
        spawn_pos: np.ndarray,
        rng: np.random.Generator,
        arena: SpawnRegion | None,
    ) -> np.ndarray:
        centre = arena.centre if arena is not None else np.zeros(2)
        if self.flow_pattern == FlowPattern.RADIAL_IN:
            return centre + rng.normal(0, 0.5, size=2)
        # RADIAL_OUT
        angle = float(rng.uniform(0, 2.0 * math.pi))
        dist = 15.0 + float(rng.uniform(0, 5.0))
        return centre + dist * np.array([math.cos(angle), math.sin(angle)])


# ---------------------------------------------------------------------------
# Spawn schedule
# ---------------------------------------------------------------------------


@dataclass
class SpawnEvent:
    """A single scheduled spawn event.

    Parameters
    ----------
    time_s : float
        Simulation time at which to spawn.
    count : int
        Number of pedestrians to spawn.
    region_idx : int
        Index into the spawn regions list.
    """

    time_s: float = 0.0
    count: int = 1
    region_idx: int = 0


# ---------------------------------------------------------------------------
# CrowdGenerator
# ---------------------------------------------------------------------------


class CrowdGenerator:
    """Generates pedestrian crowds with configurable parameters.

    Parameters
    ----------
    spawn_regions : list[SpawnRegion]
        Regions from which pedestrians are spawned.
    goal_assigner : GoalAssigner or None
        Goal assignment strategy.
    demographics : DemographicDistribution or None
        Speed/radius/personality distributions.
    spawn_strategy : SpawnStrategy
        How pedestrians appear over time.
    density : float
        Target density (pedestrians / m^2) for batch spawning.
    poisson_rate : float
        Mean arrivals per second for Poisson spawning.
    schedule : list[SpawnEvent] or None
        Explicit spawn schedule (for SCHEDULED strategy).
    max_pedestrians : int
        Hard cap on the number of pedestrians.
    arena_bounds : SpawnRegion or None
        Full arena bounding box.
    rng_seed : int or None
        Random number generator seed.
    """

    def __init__(
        self,
        spawn_regions: list[SpawnRegion] | None = None,
        goal_assigner: GoalAssigner | None = None,
        demographics: DemographicDistribution | None = None,
        spawn_strategy: SpawnStrategy = SpawnStrategy.BATCH,
        density: float = 0.1,
        poisson_rate: float = 1.0,
        schedule: list[SpawnEvent] | None = None,
        max_pedestrians: int = 200,
        arena_bounds: SpawnRegion | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.spawn_regions = spawn_regions or [SpawnRegion()]
        self.goal_assigner = goal_assigner or GoalAssigner(goal_regions=self.spawn_regions)
        self.demographics = demographics or DemographicDistribution()
        self.spawn_strategy = spawn_strategy
        self.density = density
        self.poisson_rate = poisson_rate
        self.schedule = schedule or []
        self.max_pedestrians = max_pedestrians
        self.arena_bounds = arena_bounds or self.spawn_regions[0]
        self.rng = np.random.default_rng(rng_seed)

        self._next_pid: int = 0
        self._spawned_count: int = 0
        self._schedule_idx: int = 0
        self._poisson_accumulator: float = 0.0

    # -- single pedestrian creation ------------------------------------------

    def _create_pedestrian(
        self,
        region_idx: int,
        position: np.ndarray | None = None,
    ) -> PedestrianState:
        """Create one pedestrian with sampled demographics.

        Parameters
        ----------
        region_idx : int
            Index of the spawn region.
        position : numpy.ndarray or None
            Override position; if ``None``, sampled from the region.

        Returns
        -------
        PedestrianState
            Fully initialised pedestrian state.
        """
        region = self.spawn_regions[region_idx % len(self.spawn_regions)]
        pos = position if position is not None else region.sample(self.rng)
        goal = self.goal_assigner.assign(pos, region_idx, self.rng, self.arena_bounds)
        speed = self.demographics.sample_speed(self.rng)
        radius = self.demographics.sample_radius(self.rng)
        personality = self.demographics.sample_personality(self.rng)

        pid = self._next_pid
        self._next_pid += 1

        # Heading towards goal.
        diff = goal - pos
        heading = float(math.atan2(diff[1], diff[0])) if np.linalg.norm(diff) > 1e-9 else 0.0

        state = PedestrianState(
            pid=pid,
            position=pos.copy(),
            velocity=np.zeros(2, dtype=np.float64),
            acceleration=np.zeros(2, dtype=np.float64),
            heading=heading,
            goal=goal.copy(),
            intended_velocity=np.zeros(2, dtype=np.float64),
            max_speed=speed * 1.2,
            preferred_speed=speed,
            radius=radius,
            personal_space_radius=radius * 2.0,
            personality=personality,
            activity=Activity.WALKING,
        )
        return state

    # -- batch generation -----------------------------------------------------

    def generate_batch(
        self,
        count: int | None = None,
    ) -> list[PedestrianState]:
        """Generate a batch of pedestrians all at once.

        Parameters
        ----------
        count : int or None
            Number to generate.  If ``None``, derived from *density* and the
            total spawn-region area.

        Returns
        -------
        list[PedestrianState]
            Generated pedestrian states.
        """
        if count is None:
            total_area = sum(r.area for r in self.spawn_regions)
            count = max(1, int(round(self.density * total_area)))

        count = min(count, self.max_pedestrians - self._spawned_count)
        if count <= 0:
            return []

        peds: list[PedestrianState] = []
        for i in range(count):
            region_idx = i % len(self.spawn_regions)
            peds.append(self._create_pedestrian(region_idx))
        self._spawned_count += len(peds)
        return peds

    # -- Poisson process ------------------------------------------------------

    def generate_poisson(self, dt: float) -> list[PedestrianState]:
        """Generate pedestrians according to a Poisson arrival process.

        Call this once per simulation step.

        Parameters
        ----------
        dt : float
            Simulation time step (s).

        Returns
        -------
        list[PedestrianState]
            Newly spawned pedestrians (may be empty).
        """
        if self._spawned_count >= self.max_pedestrians:
            return []

        expected = self.poisson_rate * dt
        n_arrivals = int(self.rng.poisson(expected))
        n_arrivals = min(n_arrivals, self.max_pedestrians - self._spawned_count)

        peds: list[PedestrianState] = []
        for i in range(n_arrivals):
            region_idx = int(self.rng.integers(len(self.spawn_regions)))
            peds.append(self._create_pedestrian(region_idx))
        self._spawned_count += len(peds)
        return peds

    # -- scheduled spawning ---------------------------------------------------

    def generate_scheduled(self, time_s: float) -> list[PedestrianState]:
        """Generate pedestrians according to the pre-set schedule.

        Parameters
        ----------
        time_s : float
            Current simulation time (s).

        Returns
        -------
        list[PedestrianState]
            Newly spawned pedestrians.
        """
        peds: list[PedestrianState] = []
        while self._schedule_idx < len(self.schedule):
            event = self.schedule[self._schedule_idx]
            if event.time_s > time_s:
                break
            n = min(event.count, self.max_pedestrians - self._spawned_count)
            for _ in range(n):
                peds.append(self._create_pedestrian(event.region_idx))
            self._spawned_count += n
            self._schedule_idx += 1
        return peds

    # -- unified step ---------------------------------------------------------

    def step(self, time_s: float, dt: float) -> list[PedestrianState]:
        """Generate pedestrians for the current time step.

        Dispatches to the configured :attr:`spawn_strategy`.

        Parameters
        ----------
        time_s : float
            Current simulation time (s).
        dt : float
            Time step (s).

        Returns
        -------
        list[PedestrianState]
            Newly spawned pedestrians.
        """
        if self.spawn_strategy == SpawnStrategy.BATCH:
            # Batch only fires once (first call).
            if self._spawned_count == 0:
                return self.generate_batch()
            return []
        if self.spawn_strategy == SpawnStrategy.POISSON:
            return self.generate_poisson(dt)
        if self.spawn_strategy == SpawnStrategy.SCHEDULED:
            return self.generate_scheduled(time_s)
        return []

    # -- reset ----------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal counters for a new episode."""
        self._next_pid = 0
        self._spawned_count = 0
        self._schedule_idx = 0
        self._poisson_accumulator = 0.0

    # -- scenario factories ---------------------------------------------------

    @classmethod
    def commute_scenario(
        cls,
        corridor_length: float = 20.0,
        corridor_width: float = 4.0,
        density: float = 0.15,
        rng_seed: int | None = None,
    ) -> CrowdGenerator:
        """Create a bidirectional commute corridor scenario.

        Pedestrians spawn on either end of a corridor and walk to the
        opposite end.

        Parameters
        ----------
        corridor_length : float
            Length of the corridor (m).
        corridor_width : float
            Width of the corridor (m).
        density : float
            Target crowd density (ped/m^2).
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        CrowdGenerator
            Configured generator.
        """
        half_l = corridor_length / 2.0
        half_w = corridor_width / 2.0
        left = SpawnRegion(-half_l, -half_l + 2.0, -half_w, half_w)
        right = SpawnRegion(half_l - 2.0, half_l, -half_w, half_w)
        arena = SpawnRegion(-half_l, half_l, -half_w, half_w)
        assigner = GoalAssigner(goal_regions=[left, right], flow_pattern=FlowPattern.BIDIRECTIONAL)
        return cls(
            spawn_regions=[left, right],
            goal_assigner=assigner,
            spawn_strategy=SpawnStrategy.BATCH,
            density=density,
            arena_bounds=arena,
            rng_seed=rng_seed,
        )

    @classmethod
    def evacuation_scenario(
        cls,
        room_size: float = 15.0,
        exit_width: float = 2.0,
        n_pedestrians: int = 50,
        rng_seed: int | None = None,
    ) -> CrowdGenerator:
        """Create an evacuation scenario.

        All pedestrians start inside a room and converge on a narrow exit.

        Parameters
        ----------
        room_size : float
            Side length of the square room (m).
        exit_width : float
            Width of the exit (m).
        n_pedestrians : int
            Number of pedestrians.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        CrowdGenerator
            Configured generator.
        """
        half = room_size / 2.0
        room = SpawnRegion(-half, half, -half, half)
        exit_region = SpawnRegion(half, half + 2.0, -exit_width / 2.0, exit_width / 2.0)
        assigner = GoalAssigner(goal_regions=[exit_region], flow_pattern=FlowPattern.UNIDIRECTIONAL)
        demographics = DemographicDistribution(
            speed_mean=1.6, speed_std=0.3, speed_min=0.8, speed_max=2.5
        )
        gen = cls(
            spawn_regions=[room],
            goal_assigner=assigner,
            demographics=demographics,
            spawn_strategy=SpawnStrategy.BATCH,
            density=n_pedestrians / room.area,
            max_pedestrians=n_pedestrians,
            arena_bounds=room,
            rng_seed=rng_seed,
        )
        return gen

    @classmethod
    def event_gathering_scenario(
        cls,
        arena_radius: float = 20.0,
        stage_pos: tuple[float, float] = (0.0, 0.0),
        poisson_rate: float = 2.0,
        max_pedestrians: int = 100,
        rng_seed: int | None = None,
    ) -> CrowdGenerator:
        """Create an event-gathering scenario.

        Pedestrians arrive via Poisson process from the perimeter and
        converge on a stage/attraction point.

        Parameters
        ----------
        arena_radius : float
            Radius of the circular arena approximation (m).
        stage_pos : tuple[float, float]
            Position of the attraction point.
        poisson_rate : float
            Mean arrival rate (ped/s).
        max_pedestrians : int
            Maximum crowd size.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        CrowdGenerator
            Configured generator.
        """
        r = arena_radius
        # Four spawn regions on the perimeter.
        regions = [
            SpawnRegion(-r, -r + 2.0, -r, r),
            SpawnRegion(r - 2.0, r, -r, r),
            SpawnRegion(-r, r, -r, -r + 2.0),
            SpawnRegion(-r, r, r - 2.0, r),
        ]
        goal_region = SpawnRegion(
            stage_pos[0] - 3.0,
            stage_pos[0] + 3.0,
            stage_pos[1] - 3.0,
            stage_pos[1] + 3.0,
        )
        assigner = GoalAssigner(goal_regions=[goal_region], flow_pattern=FlowPattern.RADIAL_IN)
        arena = SpawnRegion(-r, r, -r, r)
        return cls(
            spawn_regions=regions,
            goal_assigner=assigner,
            spawn_strategy=SpawnStrategy.POISSON,
            poisson_rate=poisson_rate,
            max_pedestrians=max_pedestrians,
            arena_bounds=arena,
            rng_seed=rng_seed,
        )

    @classmethod
    def random_walk_scenario(
        cls,
        bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
        n_pedestrians: int = 30,
        rng_seed: int | None = None,
    ) -> CrowdGenerator:
        """Create a random-walk scenario.

        Pedestrians spawn uniformly and receive random goals within bounds.

        Parameters
        ----------
        bounds : tuple
            ``(x_min, x_max, y_min, y_max)`` of the arena.
        n_pedestrians : int
            Number of pedestrians.
        rng_seed : int or None
            RNG seed.

        Returns
        -------
        CrowdGenerator
            Configured generator.
        """
        arena = SpawnRegion(*bounds)
        assigner = GoalAssigner(goal_regions=[arena], flow_pattern=FlowPattern.RANDOM)
        return cls(
            spawn_regions=[arena],
            goal_assigner=assigner,
            spawn_strategy=SpawnStrategy.BATCH,
            density=n_pedestrians / max(arena.area, 1.0),
            max_pedestrians=n_pedestrians,
            arena_bounds=arena,
            rng_seed=rng_seed,
        )


# ---------------------------------------------------------------------------
# Utility: density estimation
# ---------------------------------------------------------------------------


def estimate_density(
    positions: np.ndarray,
    bounds: SpawnRegion,
    cell_size: float = 1.0,
) -> np.ndarray:
    """Estimate local pedestrian density on a grid.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(N, 2)`` array of pedestrian positions.
    bounds : SpawnRegion
        Arena bounding box.
    cell_size : float
        Grid cell side length (m).

    Returns
    -------
    numpy.ndarray
        2-D density map (pedestrians per m^2 per cell).
    """
    nx = max(1, int(math.ceil((bounds.x_max - bounds.x_min) / cell_size)))
    ny = max(1, int(math.ceil((bounds.y_max - bounds.y_min) / cell_size)))
    grid = np.zeros((ny, nx), dtype=np.float64)

    for p in positions:
        ci = int((p[0] - bounds.x_min) / cell_size)
        cj = int((p[1] - bounds.y_min) / cell_size)
        ci = max(0, min(nx - 1, ci))
        cj = max(0, min(ny - 1, cj))
        grid[cj, ci] += 1.0

    cell_area = cell_size * cell_size
    grid /= cell_area
    return grid


def flow_rate(
    positions: np.ndarray,
    velocities: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    radius: float = 0.5,
) -> float:
    """Compute pedestrian flow rate across a measurement line.

    Counts pedestrians within *radius* of the line segment and computes
    their velocity component perpendicular to the line.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(N, 2)`` pedestrian positions.
    velocities : numpy.ndarray
        ``(N, 2)`` pedestrian velocities.
    line_start, line_end : numpy.ndarray
        Endpoints of the measurement line.
    radius : float
        Detection radius around the line (m).

    Returns
    -------
    float
        Flow rate (pedestrians * m/s crossing the line).
    """
    line_vec = line_end - line_start
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-9:
        return 0.0
    line_dir = line_vec / line_len
    normal = np.array([-line_dir[1], line_dir[0]])

    total = 0.0
    for i in range(positions.shape[0]):
        # Project onto line to find closest point.
        ap = positions[i] - line_start
        t = float(np.dot(ap, line_dir))
        t = max(0.0, min(line_len, t))
        closest = line_start + line_dir * t
        dist = float(np.linalg.norm(positions[i] - closest))
        if dist <= radius:
            # Perpendicular velocity component.
            v_perp = float(np.dot(velocities[i], normal))
            total += abs(v_perp)
    return total
