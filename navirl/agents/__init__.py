"""
NavIRL Agents Package
=====================

Neural network architectures and reinforcement learning agents for
pedestrian navigation in social environments.

Subpackages
-----------
networks
    Neural network building blocks: MLPs, CNNs, attention mechanisms,
    feature extractors, policy/value heads, and recurrent cores.
"""

from navirl.agents.networks import (
    MLP,
    CategoricalPolicyHead,
    CombinedExtractor,
    DeterministicPolicyHead,
    DuelingMLP,
    EgoCentricCNN,
    GatedMLP,
    GaussianPolicyHead,
    GraphAttentionNetwork,
    GRUCore,
    ImpalaCNN,
    LidarEncoder,
    LSTMCore,
    MultiHeadSocialAttention,
    NatureDQN,
    NoisyMLP,
    OccupancyGridEncoder,
    QValueHead,
    RecurrentPolicy,
    ResidualMLP,
    SocialAttention,
    SquashedGaussianHead,
    TransformerEncoder,
    ValueHead,
)

__all__ = [
    "networks",
    # MLP variants
    "MLP",
    "DuelingMLP",
    "NoisyMLP",
    "ResidualMLP",
    "GatedMLP",
    # CNN variants
    "NatureDQN",
    "ImpalaCNN",
    "OccupancyGridEncoder",
    "LidarEncoder",
    "EgoCentricCNN",
    # Attention
    "SocialAttention",
    "MultiHeadSocialAttention",
    "TransformerEncoder",
    "GraphAttentionNetwork",
    # Extractors
    "CombinedExtractor",
    # Policy heads
    "GaussianPolicyHead",
    "SquashedGaussianHead",
    "CategoricalPolicyHead",
    "DeterministicPolicyHead",
    "ValueHead",
    "QValueHead",
    # Recurrent
    "LSTMCore",
    "GRUCore",
    "RecurrentPolicy",
]
