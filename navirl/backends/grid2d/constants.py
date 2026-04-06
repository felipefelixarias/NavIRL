"""Grid2D constants — re-exports shared constants from navirl.core.maps.

Grid-specific derived constants (agent radii in pixels, etc.) remain here.
The universal occupancy constants (FREE_SPACE, OBSTACLE_SPACE) now live in
``navirl.core.maps``.
"""

from __future__ import annotations

from math import ceil

from navirl.core.maps import FREE_SPACE, OBSTACLE_SPACE

PIX_PER_METER = 100
AGENT_RADIUS = 1.5
MAX_AGENT_RADIUS = ceil(AGENT_RADIUS) * PIX_PER_METER // 10
MIN_AGENT_RADIUS = round(AGENT_RADIUS) * PIX_PER_METER // 10
AGENT_DIAMETER = MAX_AGENT_RADIUS + MIN_AGENT_RADIUS

RADIUS_METERS = 0.125
RADIUS_PIXELS = round(RADIUS_METERS * PIX_PER_METER)

__all__ = [
    "AGENT_DIAMETER",
    "AGENT_RADIUS",
    "FREE_SPACE",
    "MAX_AGENT_RADIUS",
    "MIN_AGENT_RADIUS",
    "OBSTACLE_SPACE",
    "PIX_PER_METER",
    "RADIUS_METERS",
    "RADIUS_PIXELS",
]
