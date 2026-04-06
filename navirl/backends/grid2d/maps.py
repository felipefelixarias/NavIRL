"""Grid2D map utilities — re-exports from navirl.core.maps.

All map types, constants, and loading functions now live in
``navirl.core.maps`` so that non-grid2d backends can share them.
This module re-exports the public API for backward compatibility.
"""

from __future__ import annotations

from navirl.core.maps import (
    BUILTIN_MAPS,
    DEFAULT_PIXELS_PER_METER,
    FREE_SPACE,
    OBSTACLE_SPACE,
    MapInfo,
    apartment_micro_map,
    comfort_map,
    doorway_map,
    group_map,
    hallway_map,
    hospital_corridor_map,
    kitchen_map,
    load_map,
    load_map_info,
)

__all__ = [
    "BUILTIN_MAPS",
    "DEFAULT_PIXELS_PER_METER",
    "FREE_SPACE",
    "OBSTACLE_SPACE",
    "MapInfo",
    "apartment_micro_map",
    "comfort_map",
    "doorway_map",
    "group_map",
    "hallway_map",
    "hospital_corridor_map",
    "kitchen_map",
    "load_map",
    "load_map_info",
]
