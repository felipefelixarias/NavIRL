"""
NavIRL Neural Network Architectures
====================================

Comprehensive collection of neural network building blocks for
reinforcement learning agents in pedestrian navigation tasks.

Modules
-------
mlp
    Multi-layer perceptron variants: standard, dueling, noisy, residual, gated.
cnn
    Convolutional architectures for grid/image observations.
attention
    Attention mechanisms for social navigation (SARL, GAT, Transformer).
extractors
    Feature extractors for multi-modal observation fusion.
policy_heads
    Policy and value network output heads.
rnn
    Recurrent architectures for temporal reasoning.
"""

from navirl.agents.networks.mlp import (
    MLP,
    DuelingMLP,
    NoisyMLP,
    NoisyLinear,
    ResidualMLP,
    GatedMLP,
    init_weights_xavier,
    init_weights_orthogonal,
    init_weights_kaiming,
    init_weights_uniform,
)

from navirl.agents.networks.cnn import (
    CNNExtractor,
    NatureDQN,
    ImpalaCNN,
    OccupancyGridEncoder,
    LidarEncoder,
    EgoCentricCNN,
    SpatialAttentionModule,
    MultiScaleFeatureExtractor,
)

from navirl.agents.networks.attention import (
    SocialAttention,
    MultiHeadSocialAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
    GraphAttentionNetwork,
    GraphAttentionLayer,
    CrossAttention,
    RelationalReasoning,
    TemporalAttention,
    SpatialTransformer,
)

from navirl.agents.networks.extractors import (
    StateExtractor,
    LidarExtractor,
    OccupancyExtractor,
    SocialExtractor,
    CombinedExtractor,
    HierarchicalExtractor,
    RecurrentExtractor,
    RunningNormalizer,
)

from navirl.agents.networks.policy_heads import (
    GaussianPolicyHead,
    SquashedGaussianHead,
    CategoricalPolicyHead,
    MultiDiscreteHead,
    DeterministicPolicyHead,
    ValueHead,
    QValueHead,
    DuelingQHead,
    QuantileHead,
    TwinQHead,
)

from navirl.agents.networks.rnn import (
    LSTMCore,
    GRUCore,
    RecurrentPolicy,
    SequenceEncoder,
    HiddenStateManager,
    AttentionOverMemory,
)

__all__ = [
    # MLP
    "MLP",
    "DuelingMLP",
    "NoisyMLP",
    "NoisyLinear",
    "ResidualMLP",
    "GatedMLP",
    "init_weights_xavier",
    "init_weights_orthogonal",
    "init_weights_kaiming",
    "init_weights_uniform",
    # CNN
    "CNNExtractor",
    "NatureDQN",
    "ImpalaCNN",
    "OccupancyGridEncoder",
    "LidarEncoder",
    "EgoCentricCNN",
    "SpatialAttentionModule",
    "MultiScaleFeatureExtractor",
    # Attention
    "SocialAttention",
    "MultiHeadSocialAttention",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "GraphAttentionNetwork",
    "GraphAttentionLayer",
    "CrossAttention",
    "RelationalReasoning",
    "TemporalAttention",
    "SpatialTransformer",
    # Extractors
    "StateExtractor",
    "LidarExtractor",
    "OccupancyExtractor",
    "SocialExtractor",
    "CombinedExtractor",
    "HierarchicalExtractor",
    "RecurrentExtractor",
    "RunningNormalizer",
    # Policy heads
    "GaussianPolicyHead",
    "SquashedGaussianHead",
    "CategoricalPolicyHead",
    "MultiDiscreteHead",
    "DeterministicPolicyHead",
    "ValueHead",
    "QValueHead",
    "DuelingQHead",
    "QuantileHead",
    "TwinQHead",
    # RNN
    "LSTMCore",
    "GRUCore",
    "RecurrentPolicy",
    "SequenceEncoder",
    "HiddenStateManager",
    "AttentionOverMemory",
]
