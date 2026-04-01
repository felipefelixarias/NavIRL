from __future__ import annotations

from navirl.planning.base import Path, Planner, PlannerConfig
from navirl.planning.global_planners import (
    AStarPlanner,
    DijkstraPlanner,
    PRMPlanner,
    RRTPlanner,
    RRTStarPlanner,
    ThetaStarPlanner,
)

# TODO: These modules don't exist yet - commenting out to fix imports
# from navirl.planning.local_planners import APFPlanner, DWAPlanner, MPCPlanner, VFHPlanner
# from navirl.planning.social_planners import (
#     CooperativePlanner,
#     CrowdAwarePlanner,
#     SocialAwarePlanner,
# )
# from navirl.planning.trajectory_optimization import (
#     CHOMPOptimizer,
#     StompOptimizer,
#     TrajOptConfig,
#     TrajectoryOptimizer,
# )

__all__ = [
    "Path",
    "Planner",
    "PlannerConfig",
    # Global planners
    "AStarPlanner",
    "DijkstraPlanner",
    "PRMPlanner",
    "RRTPlanner",
    "RRTStarPlanner",
    "ThetaStarPlanner",
    # TODO: Add these back when modules are implemented
    # # Local planners
    # "APFPlanner",
    # "DWAPlanner",
    # "MPCPlanner",
    # "VFHPlanner",
    # # Social planners
    # "CooperativePlanner",
    # "CrowdAwarePlanner",
    # "SocialAwarePlanner",
    # # Trajectory optimization
    # "CHOMPOptimizer",
    # "StompOptimizer",
    # "TrajOptConfig",
    # "TrajectoryOptimizer",
]
