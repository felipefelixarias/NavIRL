"""Default plugin registration for NavIRL components.

Provides centralized registration of built-in backends, human controllers,
and robot controllers. The registration ensures components are available
for use in simulations without requiring explicit imports.

Functions
---------
register_default_plugins -- Register all built-in components once
"""

from __future__ import annotations

from navirl.backends.grid2d import Grid2DBackend
from navirl.core.registry import (
    register_backend,
    register_human_controller,
    register_robot_controller,
)
from navirl.humans.orca import ORCAHumanController
from navirl.humans.orca_plus import ORCAPlusHumanController
from navirl.humans.replay import ReplayHumanController
from navirl.humans.scripted import ScriptedHumanController
from navirl.robots.baselines import (
    BaselineAStarRobotController,
    PRMRobotController,
    RRTStarRobotController,
    SocialCostAStarRobotController,
)

def _make_policy_human(cfg: dict):
    """Lazy factory so that torch is only imported when 'policy' type is used."""
    from navirl.models.learned_policy import PolicyHumanController

    return PolicyHumanController(cfg=cfg)


def _make_policy_robot(cfg: dict):
    """Lazy factory so that torch is only imported when 'policy' type is used."""
    from navirl.models.learned_robot_policy import PolicyRobotController

    return PolicyRobotController(cfg=cfg)


_REGISTERED = False


def register_default_plugins() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    register_backend(
        "grid2d",
        lambda scene_cfg, horizon_cfg, base_dir=None: Grid2DBackend(
            scene_cfg=scene_cfg,
            horizon_cfg=horizon_cfg,
            base_dir=base_dir,
        ),
    )

    register_human_controller("orca", lambda cfg, seed=0: ORCAHumanController(cfg=cfg))
    register_human_controller(
        "orca_plus",
        lambda cfg, seed=0: ORCAPlusHumanController(cfg=cfg, seed=seed),
    )
    register_human_controller("scripted", lambda cfg, seed=0: ScriptedHumanController(cfg=cfg))
    register_human_controller("replay", lambda cfg, seed=0: ReplayHumanController(cfg=cfg))
    register_human_controller(
        "policy",
        lambda cfg, seed=0: _make_policy_human(cfg),
    )

    register_robot_controller(
        "baseline_astar",
        lambda cfg: BaselineAStarRobotController(cfg=cfg),
    )
    register_robot_controller(
        "social_astar",
        lambda cfg: SocialCostAStarRobotController(cfg=cfg),
    )
    register_robot_controller(
        "prm",
        lambda cfg: PRMRobotController(cfg=cfg),
    )
    register_robot_controller(
        "rrt_star",
        lambda cfg: RRTStarRobotController(cfg=cfg),
    )
    register_robot_controller(
        "user",
        lambda cfg: BaselineAStarRobotController(cfg=cfg),
    )
    register_robot_controller(
        "policy",
        lambda cfg: _make_policy_robot(cfg),
    )

    _REGISTERED = True
