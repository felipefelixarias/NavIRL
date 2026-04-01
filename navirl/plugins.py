from __future__ import annotations

from navirl.backends.grid2d import Grid2DBackend
from navirl.core.plugin_validation import PluginMetadata
from navirl.core.registry import (
    register_backend,
    register_human_controller,
    register_robot_controller,
)
from navirl.humans.base import HumanController
from navirl.humans.orca import ORCAHumanController
from navirl.humans.orca_plus import ORCAPlusHumanController
from navirl.humans.replay import ReplayHumanController
from navirl.humans.scripted import ScriptedHumanController
from navirl.robots.baselines import BaselineAStarRobotController

_REGISTERED = False


def register_default_plugins() -> None:
    """Register NavIRL's default plugins with enhanced validation."""
    global _REGISTERED
    if _REGISTERED:
        return

    # Register Grid2D Backend with metadata
    grid2d_metadata = PluginMetadata(
        name="grid2d",
        version="1.0.0",
        author="NavIRL Core Team",
        description="Grid-based 2D navigation backend with obstacle and agent collision detection",
        api_version="1.0.0",
        required_params=[],
        optional_params=["scene_cfg", "horizon_cfg", "base_dir"],
        param_schema={
            "scene_cfg": dict,
            "horizon_cfg": dict,
            "base_dir": str,
        }
    )
    register_backend(
        "grid2d",
        lambda scene_cfg, horizon_cfg, base_dir=None: Grid2DBackend(
            scene_cfg=scene_cfg,
            horizon_cfg=horizon_cfg,
            base_dir=base_dir,
        ),
        metadata=grid2d_metadata,
    )

    # Register ORCA Human Controller
    orca_metadata = PluginMetadata(
        name="orca",
        version="1.0.0",
        author="NavIRL Core Team",
        description="ORCA (Optimal Reciprocal Collision Avoidance) human behavior controller",
        api_version="1.0.0",
        required_params=[],
        optional_params=["goal_tolerance", "waypoint_tolerance", "lookahead", "min_speed",
                        "slowdown_dist", "velocity_smoothing", "stop_speed"],
        param_schema={
            "goal_tolerance": float,
            "waypoint_tolerance": float,
            "lookahead": int,
            "min_speed": float,
            "slowdown_dist": float,
            "velocity_smoothing": float,
            "stop_speed": float,
        }
    )
    register_human_controller(
        "orca",
        lambda cfg, seed=0: ORCAHumanController(cfg=cfg),
        metadata=orca_metadata,
        interface_class=HumanController,
    )

    # Register ORCA+ Human Controller
    orca_plus_metadata = PluginMetadata(
        name="orca_plus",
        version="1.0.0",
        author="NavIRL Core Team",
        description="Enhanced ORCA with additional social behaviors and randomization",
        api_version="1.0.0",
        required_params=[],
        optional_params=["goal_tolerance", "social_force_strength", "noise_level", "seed"],
        param_schema={
            "goal_tolerance": float,
            "social_force_strength": float,
            "noise_level": float,
            "seed": int,
        }
    )
    register_human_controller(
        "orca_plus",
        lambda cfg, seed=0: ORCAPlusHumanController(cfg=cfg, seed=seed),
        metadata=orca_plus_metadata,
        interface_class=HumanController,
    )

    # Register Scripted Human Controller
    scripted_metadata = PluginMetadata(
        name="scripted",
        version="1.0.0",
        author="NavIRL Core Team",
        description="Scripted waypoint-following human behavior controller",
        api_version="1.0.0",
        required_params=[],
        optional_params=["max_speed", "waypoints", "loop_behavior"],
        param_schema={
            "max_speed": float,
            "waypoints": dict,
            "loop_behavior": bool,
        }
    )
    register_human_controller(
        "scripted",
        lambda cfg, seed=0: ScriptedHumanController(cfg=cfg),
        metadata=scripted_metadata,
        interface_class=HumanController,
    )

    # Register Replay Human Controller
    replay_metadata = PluginMetadata(
        name="replay",
        version="1.0.0",
        author="NavIRL Core Team",
        description="Replay recorded human trajectories from demonstration data",
        api_version="1.0.0",
        required_params=[],
        optional_params=["trajectory_file", "time_offset", "interpolation_method"],
        param_schema={
            "trajectory_file": str,
            "time_offset": float,
            "interpolation_method": str,
        }
    )
    register_human_controller(
        "replay",
        lambda cfg, seed=0: ReplayHumanController(cfg=cfg),
        metadata=replay_metadata,
        interface_class=HumanController,
    )

    # Register policy controller (alias for ORCA)
    register_human_controller(
        "policy",
        lambda cfg, seed=0: ORCAHumanController(cfg=cfg),
        metadata=orca_metadata,  # Reuse ORCA metadata
        interface_class=HumanController,
    )

    # Register Baseline A* Robot Controller
    astar_metadata = PluginMetadata(
        name="baseline_astar",
        version="1.0.0",
        author="NavIRL Core Team",
        description="A* pathfinding robot controller with dynamic replanning",
        api_version="1.0.0",
        required_params=[],
        optional_params=["replan_interval", "goal_tolerance", "velocity_smoothing",
                        "max_speed", "social_comfort_distance"],
        param_schema={
            "replan_interval": int,
            "goal_tolerance": float,
            "velocity_smoothing": float,
            "max_speed": float,
            "social_comfort_distance": float,
        }
    )
    register_robot_controller(
        "baseline_astar",
        lambda cfg: BaselineAStarRobotController(cfg=cfg),
        metadata=astar_metadata,
    )

    # Register user controller (alias for baseline A*)
    register_robot_controller(
        "user",
        lambda cfg: BaselineAStarRobotController(cfg=cfg),
        metadata=astar_metadata,  # Reuse A* metadata
    )

    _REGISTERED = True
