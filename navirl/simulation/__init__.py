"""NavIRL simulation sub-package.

Provides world management, physics, timing, entity modelling, event
handling, and high-level simulation orchestration for pedestrian / robot
navigation experiments.
"""

from __future__ import annotations

from navirl.simulation.clock import SimulationClock
from navirl.simulation.entities import (
    Door,
    DynamicObstacle,
    Entity,
    EntityManager,
    NavigationGraph,
    Region,
    StaticObstacle,
    Wall,
    Waypoint,
)
from navirl.simulation.events import EventBus, EventRecord, EventType
from navirl.simulation.physics import DynamicModel, KinematicModel, SimplePhysics
from navirl.simulation.runner import SimulationRunner
from navirl.simulation.world import World, WorldBuilder

__all__ = [
    "Door",
    "DynamicModel",
    "DynamicObstacle",
    "Entity",
    "EntityManager",
    "EventBus",
    "EventRecord",
    "EventType",
    "KinematicModel",
    "NavigationGraph",
    "Region",
    "SimplePhysics",
    "SimulationClock",
    "SimulationRunner",
    "StaticObstacle",
    "Wall",
    "Waypoint",
    "World",
    "WorldBuilder",
]
