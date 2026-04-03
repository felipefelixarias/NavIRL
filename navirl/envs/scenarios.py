"""Scenario generators for RL navigation environments.

Each scenario class creates a configuration dict that can be passed to
:class:`~navirl.envs.NavEnv` (or its subclasses) to set up a specific
pedestrian-navigation situation.

Exports
-------
CircleCrossing, RandomGoal, CorridorPassing, DoorwayNegotiation,
IntersectionCrossing, GroupNavigation, DenseRoom, OpenField,
ScenarioDifficultyScaler, ProceduralScenarioGenerator
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ScenarioConfig = dict[str, Any]
Position = tuple[float, float]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseScenario(ABC):
    """Abstract base for all scenario generators."""

    @abstractmethod
    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        """Return a scenario configuration dictionary.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        dict
            Keys: ``map_name``, ``robot_start``, ``robot_goal``,
            ``human_starts``, ``human_goals``, ``num_humans``, plus any
            scenario-specific extras.
        """

    # Convenience ------------------------------------------------------------
    @staticmethod
    def _uniform_position(
        rng: np.random.Generator,
        low: float,
        high: float,
    ) -> Position:
        x = rng.uniform(low, high)
        y = rng.uniform(low, high)
        return (float(x), float(y))

    @staticmethod
    def _positions_on_circle(
        n: int,
        radius: float,
        center: tuple[float, float] = (0.0, 0.0),
        offset_angle: float = 0.0,
    ) -> list[Position]:
        angles = [offset_angle + 2 * math.pi * i / n for i in range(n)]
        return [
            (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles
        ]

    @staticmethod
    def _antipodal(
        positions: list[Position],
        center: tuple[float, float] = (0.0, 0.0),
    ) -> list[Position]:
        return [(2 * center[0] - px, 2 * center[1] - py) for (px, py) in positions]


# ---------------------------------------------------------------------------
# Concrete scenarios
# ---------------------------------------------------------------------------
@dataclass
class CircleCrossing(BaseScenario):
    """Agents start on a circle and cross to antipodal positions.

    Parameters
    ----------
    num_humans : int
        Number of pedestrians placed on the circle.
    circle_radius : float
        Radius of the circle (metres).
    """

    num_humans: int = 5
    circle_radius: float = 4.0

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        total = self.num_humans + 1  # +1 for the robot
        offset = rng.uniform(0, 2 * math.pi)
        starts = self._positions_on_circle(total, self.circle_radius, offset_angle=offset)
        goals = self._antipodal(starts)

        robot_start = starts[0]
        robot_goal = goals[0]
        human_starts = starts[1:]
        human_goals = goals[1:]

        return {
            "map_name": "circle_crossing",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans,
        }


@dataclass
class RandomGoal(BaseScenario):
    """Random start and goal positions within a square world.

    Parameters
    ----------
    num_humans : int
        Number of pedestrians.
    world_size : float
        Half-width of the square world (positions sampled in [-world_size, world_size]).
    min_goal_dist : float
        Minimum Euclidean distance between any agent's start and goal.
    """

    num_humans: int = 5
    world_size: float = 6.0
    min_goal_dist: float = 3.0

    def _sample_pair(self, rng: np.random.Generator) -> tuple[Position, Position]:
        while True:
            start = self._uniform_position(rng, -self.world_size, self.world_size)
            goal = self._uniform_position(rng, -self.world_size, self.world_size)
            if math.dist(start, goal) >= self.min_goal_dist:
                return start, goal

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        robot_start, robot_goal = self._sample_pair(rng)
        human_starts: list[Position] = []
        human_goals: list[Position] = []
        for _ in range(self.num_humans):
            hs, hg = self._sample_pair(rng)
            human_starts.append(hs)
            human_goals.append(hg)

        return {
            "map_name": "random_goal",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans,
        }


@dataclass
class CorridorPassing(BaseScenario):
    """Bidirectional corridor traffic.

    Pedestrians walk in both directions through a corridor; the robot must
    navigate from one end to the other.

    Parameters
    ----------
    corridor_length : float
        Length of the corridor (metres).
    corridor_width : float
        Width of the corridor (metres).
    num_humans : int
        Total number of pedestrians (split roughly equally between directions).
    """

    corridor_length: float = 10.0
    corridor_width: float = 2.0
    num_humans: int = 6

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        half_l = self.corridor_length / 2.0
        half_w = self.corridor_width / 2.0

        robot_start = (-half_l, 0.0)
        robot_goal = (half_l, 0.0)

        human_starts: list[Position] = []
        human_goals: list[Position] = []

        for i in range(self.num_humans):
            y = float(rng.uniform(-half_w, half_w))
            if i % 2 == 0:
                # Walking left-to-right
                x = float(rng.uniform(-half_l, 0))
                human_starts.append((x, y))
                human_goals.append((half_l, y))
            else:
                # Walking right-to-left (oncoming)
                x = float(rng.uniform(0, half_l))
                human_starts.append((x, y))
                human_goals.append((-half_l, y))

        return {
            "map_name": "corridor",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans,
            "corridor_length": self.corridor_length,
            "corridor_width": self.corridor_width,
        }


@dataclass
class DoorwayNegotiation(BaseScenario):
    """Navigate through a doorway with oncoming traffic.

    Parameters
    ----------
    door_width : float
        Width of the doorway (metres).
    num_humans : int
        Number of pedestrians approaching from the opposite side.
    room_depth : float
        Depth of the rooms on either side of the door.
    """

    door_width: float = 1.0
    num_humans: int = 3
    room_depth: float = 5.0

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        robot_start = (-self.room_depth, 0.0)
        robot_goal = (self.room_depth, 0.0)

        human_starts: list[Position] = []
        human_goals: list[Position] = []

        for _ in range(self.num_humans):
            y_offset = float(rng.uniform(-self.door_width / 2, self.door_width / 2))
            hx = float(rng.uniform(0.5, self.room_depth))
            human_starts.append((hx, y_offset))
            human_goals.append((-self.room_depth, y_offset))

        return {
            "map_name": "doorway",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans,
            "door_width": self.door_width,
        }


@dataclass
class IntersectionCrossing(BaseScenario):
    """Four-way intersection crossing.

    Pedestrians approach from all four cardinal directions; the robot crosses
    from south to north.

    Parameters
    ----------
    num_humans_per_direction : int
        Number of pedestrians coming from each of the four directions.
    intersection_size : float
        Half-width of the intersection area.
    approach_distance : float
        How far from the centre agents start.
    """

    num_humans_per_direction: int = 2
    intersection_size: float = 3.0
    approach_distance: float = 8.0

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        robot_start = (0.0, -self.approach_distance)
        robot_goal = (0.0, self.approach_distance)

        directions = [
            ((0.0, -1.0), (0.0, 1.0)),  # south -> north
            ((0.0, 1.0), (0.0, -1.0)),  # north -> south
            ((-1.0, 0.0), (1.0, 0.0)),  # west  -> east
            ((1.0, 0.0), (-1.0, 0.0)),  # east  -> west
        ]

        human_starts: list[Position] = []
        human_goals: list[Position] = []

        for (sx, sy), (gx, gy) in directions:
            for _ in range(self.num_humans_per_direction):
                lateral = float(
                    rng.uniform(-self.intersection_size / 2, self.intersection_size / 2)
                )
                start = (
                    sx * self.approach_distance + gy * lateral,
                    sy * self.approach_distance + gx * lateral,
                )
                goal = (
                    gx * self.approach_distance + gy * lateral,
                    gy * self.approach_distance + gx * lateral,
                )
                human_starts.append(start)
                human_goals.append(goal)

        return {
            "map_name": "intersection",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans_per_direction * 4,
        }


@dataclass
class GroupNavigation(BaseScenario):
    """Navigate around walking groups of pedestrians.

    Groups of pedestrians walk together in a cluster; the robot must weave
    around them.

    Parameters
    ----------
    num_groups : int
        Number of pedestrian groups.
    group_size : int
        Number of pedestrians per group.
    world_size : float
        Half-width of the world.
    group_spread : float
        Maximum distance of group members from the group centre.
    """

    num_groups: int = 3
    group_size: int = 3
    world_size: float = 8.0
    group_spread: float = 0.8

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        robot_start = self._uniform_position(rng, -self.world_size, self.world_size)
        robot_goal = self._uniform_position(rng, -self.world_size, self.world_size)

        human_starts: list[Position] = []
        human_goals: list[Position] = []

        for _ in range(self.num_groups):
            cx, cy = self._uniform_position(rng, -self.world_size + 1, self.world_size - 1)
            gx, gy = self._uniform_position(rng, -self.world_size + 1, self.world_size - 1)

            for _ in range(self.group_size):
                dx = float(rng.uniform(-self.group_spread, self.group_spread))
                dy = float(rng.uniform(-self.group_spread, self.group_spread))
                human_starts.append((cx + dx, cy + dy))
                human_goals.append((gx + dx, gy + dy))

        return {
            "map_name": "group_navigation",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_groups * self.group_size,
            "num_groups": self.num_groups,
            "group_size": self.group_size,
        }


@dataclass
class DenseRoom(BaseScenario):
    """High-density room navigation.

    Many pedestrians mill about in a room; the robot must cross from one
    corner to the opposite.

    Parameters
    ----------
    room_size : float
        Half-width of the square room.
    num_humans : int
        Number of pedestrians in the room.
    """

    room_size: float = 5.0
    num_humans: int = 20

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        robot_start = (-self.room_size + 0.5, -self.room_size + 0.5)
        robot_goal = (self.room_size - 0.5, self.room_size - 0.5)

        human_starts: list[Position] = []
        human_goals: list[Position] = []

        for _ in range(self.num_humans):
            hs = self._uniform_position(rng, -self.room_size, self.room_size)
            hg = self._uniform_position(rng, -self.room_size, self.room_size)
            human_starts.append(hs)
            human_goals.append(hg)

        return {
            "map_name": "dense_room",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans,
            "room_size": self.room_size,
        }


@dataclass
class OpenField(BaseScenario):
    """Open space with scattered pedestrians.

    A large open field with relatively few pedestrians walking in random
    directions -- useful as an easy baseline scenario.

    Parameters
    ----------
    field_size : float
        Half-width of the square field.
    num_humans : int
        Number of scattered pedestrians.
    """

    field_size: float = 15.0
    num_humans: int = 8

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        robot_start = self._uniform_position(rng, -self.field_size, self.field_size)
        robot_goal = self._uniform_position(rng, -self.field_size, self.field_size)

        human_starts: list[Position] = []
        human_goals: list[Position] = []

        for _ in range(self.num_humans):
            hs = self._uniform_position(rng, -self.field_size, self.field_size)
            hg = self._uniform_position(rng, -self.field_size, self.field_size)
            human_starts.append(hs)
            human_goals.append(hg)

        return {
            "map_name": "open_field",
            "robot_start": robot_start,
            "robot_goal": robot_goal,
            "human_starts": human_starts,
            "human_goals": human_goals,
            "num_humans": self.num_humans,
            "field_size": self.field_size,
        }


# ---------------------------------------------------------------------------
# Meta-scenario utilities
# ---------------------------------------------------------------------------
@dataclass
class ScenarioDifficultyScaler(BaseScenario):
    """Wrap a scenario and scale its difficulty.

    Difficulty is expressed as a float in ``[0, 1]``.  At ``difficulty=0``
    the scenario uses its easiest settings; at ``difficulty=1`` it uses the
    hardest.  Intermediate values interpolate linearly.

    The scaler adjusts ``num_humans`` (and, where applicable,
    ``corridor_width`` / ``door_width`` / ``room_size``) of the underlying
    scenario before generation.

    Parameters
    ----------
    base_scenario : BaseScenario
        The underlying scenario to scale.
    difficulty : float
        Difficulty level in ``[0, 1]``.
    max_humans_multiplier : float
        At ``difficulty=1`` the human count is multiplied by this factor.
    """

    base_scenario: BaseScenario = field(default_factory=CircleCrossing)
    difficulty: float = 0.5
    max_humans_multiplier: float = 3.0

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        d = float(np.clip(self.difficulty, 0.0, 1.0))

        # Scale num_humans
        if hasattr(self.base_scenario, "num_humans"):
            base_n = self.base_scenario.num_humans  # type: ignore[union-attr]
            scaled = int(max(1, base_n * (1.0 + d * (self.max_humans_multiplier - 1.0))))
            object.__setattr__(self.base_scenario, "num_humans", scaled)

        # Make corridors / doorways narrower at higher difficulty
        if hasattr(self.base_scenario, "corridor_width"):
            base_w = self.base_scenario.corridor_width  # type: ignore[union-attr]
            object.__setattr__(self.base_scenario, "corridor_width", base_w * (1.0 - 0.5 * d))
        if hasattr(self.base_scenario, "door_width"):
            base_w = self.base_scenario.door_width  # type: ignore[union-attr]
            object.__setattr__(self.base_scenario, "door_width", base_w * (1.0 - 0.5 * d))

        config = self.base_scenario.generate(rng)
        config["difficulty"] = d
        return config


@dataclass
class ProceduralScenarioGenerator(BaseScenario):
    """Randomly select and generate scenarios from a pool.

    Useful for curriculum learning or domain randomisation over scenario
    types.

    Parameters
    ----------
    scenario_pool : list[BaseScenario]
        Collection of scenario instances to sample from.
    difficulty_range : tuple[float, float]
        If provided, each sampled scenario is wrapped in a
        :class:`ScenarioDifficultyScaler` with a difficulty drawn uniformly
        from this range.
    weights : list[float] | None
        Optional sampling weights (need not sum to 1).
    """

    scenario_pool: list[BaseScenario] = field(
        default_factory=lambda: [
            CircleCrossing(),
            RandomGoal(),
            CorridorPassing(),
            DoorwayNegotiation(),
            IntersectionCrossing(),
        ]
    )
    difficulty_range: tuple[float, float] | None = None
    weights: list[float] | None = None

    def generate(self, rng: np.random.Generator) -> ScenarioConfig:
        # Normalise weights
        if self.weights is not None:
            probs = np.array(self.weights, dtype=np.float64)
            probs /= probs.sum()
        else:
            probs = None

        idx = int(rng.choice(len(self.scenario_pool), p=probs))
        scenario = self.scenario_pool[idx]

        if self.difficulty_range is not None:
            lo, hi = self.difficulty_range
            diff = float(rng.uniform(lo, hi))
            scenario = ScenarioDifficultyScaler(base_scenario=scenario, difficulty=diff)

        config = scenario.generate(rng)
        config["scenario_index"] = idx
        config["scenario_class"] = type(scenario).__name__
        return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "BaseScenario",
    "CircleCrossing",
    "RandomGoal",
    "CorridorPassing",
    "DoorwayNegotiation",
    "IntersectionCrossing",
    "GroupNavigation",
    "DenseRoom",
    "OpenField",
    "ScenarioDifficultyScaler",
    "ProceduralScenarioGenerator",
]
