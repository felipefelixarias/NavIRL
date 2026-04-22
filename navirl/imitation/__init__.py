"""Imitation learning algorithms for NavIRL.

This package contains implementations of various imitation learning algorithms
including behavioral cloning, inverse reinforcement learning, and adversarial
imitation learning methods.

Classes that wrap torch networks (``RewardNetwork``, ``Discriminator``) import
without torch installed, but instantiating them — or any of the agent classes
(``AIRLAgent``, ``BCAgent``, ``DAggerAgent``, ``GAILAgent``) — requires the
``[agents]`` extra.
"""

from __future__ import annotations

from .airl import AIRLAgent, AIRLConfig, RewardNetwork
from .bc import BCAgent, BCConfig
from .dagger import DAggerAgent, DAggerConfig
from .dataset import DemonstrationDataset, FeatureStatistics
from .gail import Discriminator, GAILAgent, GAILConfig
from .irl import MaxEntIRL, MaxEntIRLConfig
from .reward_learning import (
    DemonstrationRewardConfig,
    DemonstrationRewardModel,
    EnsembleRewardConfig,
    EnsembleRewardModel,
    PreferenceRewardConfig,
    PreferenceRewardModel,
)

__all__ = [
    "AIRLAgent",
    "AIRLConfig",
    "BCAgent",
    "BCConfig",
    "DAggerAgent",
    "DAggerConfig",
    "DemonstrationDataset",
    "DemonstrationRewardConfig",
    "DemonstrationRewardModel",
    "Discriminator",
    "EnsembleRewardConfig",
    "EnsembleRewardModel",
    "FeatureStatistics",
    "GAILAgent",
    "GAILConfig",
    "MaxEntIRL",
    "MaxEntIRLConfig",
    "PreferenceRewardConfig",
    "PreferenceRewardModel",
    "RewardNetwork",
]
