"""
NavIRL Rewards Module
=====================

Composable, extensible reward functions for pedestrian navigation agents.

Submodules
----------
- **base** -- Abstract base classes, composition utilities, normalisation and shaping.
- **navigation** -- Goal-seeking, path-following, collision and motion-quality rewards.
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
]
