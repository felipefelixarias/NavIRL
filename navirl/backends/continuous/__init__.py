"""Continuous-space simulation backend for NavIRL.

This backend provides a continuous 2-D environment where agents
move in floating-point coordinates with realistic kinematics.
"""

from __future__ import annotations

from navirl.backends.continuous.backend import ContinuousBackend
from navirl.backends.continuous.environment import ContinuousEnvironment
from navirl.backends.continuous.obstacles import (
    CircleObstacle,
    LineObstacle,
    Obstacle,
    PolygonObstacle,
    RectangleObstacle,
)
from navirl.backends.continuous.physics import PhysicsEngine

__all__ = [
    "CircleObstacle",
    "ContinuousBackend",
    "ContinuousEnvironment",
    "LineObstacle",
    "Obstacle",
    "PhysicsEngine",
    "PolygonObstacle",
    "RectangleObstacle",
]
