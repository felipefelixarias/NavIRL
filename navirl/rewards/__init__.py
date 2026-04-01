"""
NavIRL Rewards Module
=====================

Composable, extensible reward functions for pedestrian navigation agents.

Submodules
----------
- **base** -- Abstract base classes, composition utilities, normalisation and shaping.
- **navigation** -- Goal-seeking, path-following, collision and motion-quality rewards.
- **social** -- Social-awareness rewards (proxemics, yielding, group coherence, ...).
- **learned** -- Neural / IRL / GAIL / curiosity-driven learned reward functions.
- **multi_objective** -- Multi-objective reward handling with Pareto front support.
"""

from __future__ import annotations

from navirl.rewards.base import (
    CompositeReward,
    RewardClipper,
    RewardComponent,
    RewardFunction,
    RewardNormalizer,
    RewardShaper,
)
from navirl.rewards.learned import (
    CuriosityReward,
    EnsembleReward,
    GAILReward,
    IRLReward,
    NeuralRewardFunction,
    RNDReward,
)
from navirl.rewards.multi_objective import (
    MultiObjectiveReward,
    ParetoFront,
)
from navirl.rewards.navigation import (
    BoundaryPenalty,
    CollisionPenalty,
    GoalReward,
    PathFollowingReward,
    ProgressReward,
    SmoothnessReward,
    TimePenaltyReward,
    VelocityReward,
)
from navirl.rewards.social import (
    ComfortReward,
    GazeAwarenessReward,
    GroupCoherenceReward,
    OvertakingReward,
    PersonalSpaceReward,
    ProxemicsReward,
    RightOfWayReward,
    SocialForceReward,
    YieldingReward,
)

__all__ = [
    # base
    "RewardFunction",
    "RewardComponent",
    "CompositeReward",
    "RewardNormalizer",
    "RewardClipper",
    "RewardShaper",
    # navigation
    "GoalReward",
    "PathFollowingReward",
    "TimePenaltyReward",
    "CollisionPenalty",
    "ProgressReward",
    "VelocityReward",
    "SmoothnessReward",
    "BoundaryPenalty",
    # social
    "PersonalSpaceReward",
    "SocialForceReward",
    "ProxemicsReward",
    "GazeAwarenessReward",
    "GroupCoherenceReward",
    "OvertakingReward",
    "YieldingReward",
    "RightOfWayReward",
    "ComfortReward",
    # learned
    "NeuralRewardFunction",
    "IRLReward",
    "GAILReward",
    "EnsembleReward",
    "CuriosityReward",
    "RNDReward",
    # multi-objective
    "MultiObjectiveReward",
    "ParetoFront",
]
