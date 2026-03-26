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
    DuelingMLP,
    NoisyMLP,
    ResidualMLP,
    GatedMLP,
    NatureDQN,
    ImpalaCNN,
    OccupancyGridEncoder,
    LidarEncoder,
    EgoCentricCNN,
    SocialAttention,
    MultiHeadSocialAttention,
    TransformerEncoder,
    GraphAttentionNetwork,
    CombinedExtractor,
    GaussianPolicyHead,
    SquashedGaussianHead,
    CategoricalPolicyHead,
    DeterministicPolicyHead,
    ValueHead,
    QValueHead,
    LSTMCore,
    GRUCore,
    RecurrentPolicy,
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
