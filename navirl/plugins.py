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
from navirl.robots.baselines import (
    BaselineAStarRobotController,
    PotentialFieldsController,
    PRMController,
    SocialAwareAStarController,
)

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

    # Register Social-Aware A* Robot Controller
    social_astar_metadata = PluginMetadata(
        name="social_astar",
        version="1.0.0",
        author="NavIRL Core Team",
        description="A* pathfinding enhanced with social forces and human comfort zone awareness",
        api_version="1.0.0",
        required_params=[],
        optional_params=[
            "goal_tolerance", "replan_interval", "max_speed", "slowdown_dist",
            "target_lookahead", "velocity_smoothing", "social_comfort_distance",
            "personal_space_distance", "social_cost_weight", "group_detection_radius",
            "avoidance_strength", "prediction_horizon", "velocity_history_size"
        ],
        param_schema={
            "goal_tolerance": float,
            "replan_interval": int,
            "max_speed": float,
            "slowdown_dist": float,
            "target_lookahead": int,
            "velocity_smoothing": float,
            "social_comfort_distance": float,
            "personal_space_distance": float,
            "social_cost_weight": float,
            "group_detection_radius": float,
            "avoidance_strength": float,
            "prediction_horizon": float,
            "velocity_history_size": int,
        }
    )
    register_robot_controller(
        "social_astar",
        lambda cfg: SocialAwareAStarController(cfg=cfg),
        metadata=social_astar_metadata,
    )

    # Register PRM Robot Controller
    prm_metadata = PluginMetadata(
        name="prm",
        version="1.0.0",
        author="NavIRL Core Team",
        description="Probabilistic Roadmap planner for complex environment navigation",
        api_version="1.0.0",
        required_params=[],
        optional_params=[
            "goal_tolerance", "replan_interval", "max_speed", "slowdown_dist",
            "velocity_smoothing", "num_samples", "connection_radius",
            "max_connections", "roadmap_bounds", "dynamic_resampling",
            "obstacle_clearance", "roadmap_rebuild_interval"
        ],
        param_schema={
            "goal_tolerance": float,
            "replan_interval": int,
            "max_speed": float,
            "slowdown_dist": float,
            "velocity_smoothing": float,
            "num_samples": int,
            "connection_radius": float,
            "max_connections": int,
            "roadmap_bounds": dict,
            "dynamic_resampling": bool,
            "obstacle_clearance": float,
            "roadmap_rebuild_interval": int,
        }
    )
    register_robot_controller(
        "prm",
        lambda cfg: PRMController(cfg=cfg),
        metadata=prm_metadata,
    )

    # Register Potential Fields Robot Controller
    potential_fields_metadata = PluginMetadata(
        name="potential_fields",
        version="1.0.0",
        author="NavIRL Core Team",
        description="Reactive navigation using artificial potential fields with social awareness",
        api_version="1.0.0",
        required_params=[],
        optional_params=[
            "goal_tolerance", "max_speed", "velocity_smoothing", "attractive_gain",
            "repulsive_gain", "repulsive_range", "human_repulsive_gain",
            "human_repulsive_range", "social_comfort_gain", "velocity_obstacle_gain",
            "prediction_horizon", "field_saturation_distance", "oscillation_damping",
            "force_limit"
        ],
        param_schema={
            "goal_tolerance": float,
            "max_speed": float,
            "velocity_smoothing": float,
            "attractive_gain": float,
            "repulsive_gain": float,
            "repulsive_range": float,
            "human_repulsive_gain": float,
            "human_repulsive_range": float,
            "social_comfort_gain": float,
            "velocity_obstacle_gain": float,
            "prediction_horizon": float,
            "field_saturation_distance": float,
            "oscillation_damping": float,
            "force_limit": float,
        }
    )
    register_robot_controller(
        "potential_fields",
        lambda cfg: PotentialFieldsController(cfg=cfg),
        metadata=potential_fields_metadata,
    )

    # Register user controller (alias for baseline A*)
    register_robot_controller(
        "user",
        lambda cfg: BaselineAStarRobotController(cfg=cfg),
        metadata=astar_metadata,  # Reuse A* metadata
    )

    _REGISTERED = True
