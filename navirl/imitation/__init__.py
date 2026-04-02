"""Imitation learning algorithms for NavIRL.

This package contains implementations of various imitation learning algorithms
including behavioral cloning, inverse reinforcement learning, and adversarial
imitation learning methods.
"""

from .airl import AIRL
from .bc import BehavioralCloning
from .dagger import DAgger
from .dataset import ImitationDataset
from .gail import GAIL
from .irl import InverseReinforcementLearning
from .reward_learning import RewardLearning

__all__ = [
    "BehavioralCloning",
    "DAgger",
    "InverseReinforcementLearning",
    "AIRL",
    "GAIL",
    "ImitationDataset",
    "RewardLearning",
]