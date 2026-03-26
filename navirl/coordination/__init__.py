"""Multi-agent coordination package for NavIRL.

Provides inter-agent communication, formation control, task allocation,
coordinated path planning, multi-agent reinforcement learning, and
distributed consensus algorithms.
"""

from __future__ import annotations

from .communication import (
    AttentionComm,
    BroadcastChannel,
    CommNetwork,
    DirectChannel,
    MessageProtocol,
    SharedMemory,
)
from .consensus import (
    AverageConsensus,
    ConsensusOptimizer,
    ConsensusProtocol,
    MaxConsensus,
    WeightedConsensus,
)
from .formation import ConsensusFormation, FormationController, LeaderFollower
from .marl import CentralizedCritic, MAPPOAgent, MARLConfig, QMIXMixer
from .planning import (
    CBSPlanner,
    PlanningResult,
    PriorityPlanner,
    VelocityObstaclePlanner,
)
from .task_allocation import (
    AllocationResult,
    AuctionAllocator,
    GreedyAllocator,
    HungarianAllocator,
    Task,
)

__all__ = [
    # communication
    "MessageProtocol",
    "BroadcastChannel",
    "DirectChannel",
    "SharedMemory",
    "CommNetwork",
    "AttentionComm",
    # formation
    "FormationController",
    "ConsensusFormation",
    "LeaderFollower",
    # task_allocation
    "Task",
    "AuctionAllocator",
    "HungarianAllocator",
    "GreedyAllocator",
    "AllocationResult",
    # planning
    "PriorityPlanner",
    "CBSPlanner",
    "VelocityObstaclePlanner",
    "PlanningResult",
    # marl
    "MARLConfig",
    "CentralizedCritic",
    "QMIXMixer",
    "MAPPOAgent",
    # consensus
    "ConsensusProtocol",
    "AverageConsensus",
    "MaxConsensus",
    "WeightedConsensus",
    "ConsensusOptimizer",
]
