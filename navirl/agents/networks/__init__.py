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

from navirl.agents.networks.attention import (
    CrossAttention,
    GraphAttentionLayer,
    GraphAttentionNetwork,
    MultiHeadSocialAttention,
    RelationalReasoning,
    SocialAttention,
    SpatialTransformer,
    TemporalAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from navirl.agents.networks.cnn import (
    CNNExtractor,
    EgoCentricCNN,
    ImpalaCNN,
    LidarEncoder,
    MultiScaleFeatureExtractor,
    NatureDQN,
    OccupancyGridEncoder,
    SpatialAttentionModule,
)
from navirl.agents.networks.extractors import (
    CombinedExtractor,
    HierarchicalExtractor,
    LidarExtractor,
    OccupancyExtractor,
    RecurrentExtractor,
    RunningNormalizer,
    SocialExtractor,
    StateExtractor,
)
from navirl.agents.networks.mlp import (
    MLP,
    DuelingMLP,
    GatedMLP,
    NoisyLinear,
    NoisyMLP,
    ResidualMLP,
    init_weights_kaiming,
    init_weights_orthogonal,
    init_weights_uniform,
    init_weights_xavier,
)
from navirl.agents.networks.policy_heads import (
    CategoricalPolicyHead,
    DeterministicPolicyHead,
    DuelingQHead,
    GaussianPolicyHead,
    MultiDiscreteHead,
    QuantileHead,
    QValueHead,
    SquashedGaussianHead,
    TwinQHead,
    ValueHead,
)
from navirl.agents.networks.rnn import (
    AttentionOverMemory,
    GRUCore,
    HiddenStateManager,
    LSTMCore,
    RecurrentPolicy,
    RNNEncoder,
    SequenceEncoder,
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
    "RNNEncoder",
    "HiddenStateManager",
    "AttentionOverMemory",
]
