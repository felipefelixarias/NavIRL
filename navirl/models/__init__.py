"""Pedestrian behavior models for NavIRL.

This package provides a comprehensive suite of pedestrian dynamics models
ranging from classical physics-based approaches (Social Force, Velocity Obstacle,
Power Law) to data-driven (learned policies) and cognitive (behavior trees)
formulations. All controller classes implement the ``HumanController`` interface
defined in ``navirl.humans.base``.

Modules
-------
social_force
    Social Force Model (Helbing & Molnar 1995, Helbing et al. 2000) with
    extensions for groups, physical contact, and anisotropic interactions.
velocity_obstacle
    Velocity Obstacle family: VO, RVO, HRVO, and a pure-Python ORCA reference.
power_law
    Power-law anticipatory collision avoidance (Karamouzas et al. 2014).
crowd_dynamics
    Macroscopic crowd analysis: density estimation, flow fields, fundamental
    diagram, level of service, crowd pressure, and evacuation dynamics.
group_behavior
    Social group detection, formation patterns, cohesion / repulsion forces,
    leader-follower dynamics, and group splitting / merging.
learned_policy
    Wrappers for neural-network pedestrian policies with ensemble support and
    online adaptation.
behavior_tree
    Composable behavior-tree framework with a library of pre-built pedestrian
    behavior subtrees.
"""

from __future__ import annotations

from navirl.models.behavior_tree import (
    BehaviorTree,
    BehaviorTreeHumanController,
)
from navirl.models.crowd_dynamics import (
    CrowdAnalyzer,
    FundamentalDiagram,
    LevelOfService,
)
from navirl.models.group_behavior import (
    GroupBehaviorModel,
    GroupDetector,
    GroupHumanController,
)
from navirl.models.learned_policy import (
    PolicyHumanController,
    PolicyRobotController,
)
from navirl.models.power_law import (
    PowerLawConfig,
    PowerLawHumanController,
    PowerLawModel,
)
from navirl.models.social_force import (
    SocialForceConfig,
    SocialForceHumanController,
    SocialForceModel,
)
from navirl.models.velocity_obstacle import (
    HybridReciprocalVO,
    ORCAPurePython,
    ReciprocalVelocityObstacle,
    VelocityObstacle,
    VOConfig,
    VOHumanController,
)

__all__ = [
    # Social Force
    "SocialForceConfig",
    "SocialForceModel",
    "SocialForceHumanController",
    # Velocity Obstacle
    "VOConfig",
    "VelocityObstacle",
    "ReciprocalVelocityObstacle",
    "HybridReciprocalVO",
    "ORCAPurePython",
    "VOHumanController",
    # Power Law
    "PowerLawConfig",
    "PowerLawModel",
    "PowerLawHumanController",
    # Crowd Dynamics
    "CrowdAnalyzer",
    "FundamentalDiagram",
    "LevelOfService",
    # Group Behavior
    "GroupDetector",
    "GroupBehaviorModel",
    "GroupHumanController",
    # Learned Policy
    "PolicyHumanController",
    "PolicyRobotController",
    # Behavior Tree
    "BehaviorTree",
    "BehaviorTreeHumanController",
]
