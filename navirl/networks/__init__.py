"""
NavIRL Neural Network Architectures for Trajectory Prediction
==============================================================

Advanced generative and discriminative network architectures for
pedestrian trajectory prediction and motion forecasting, implemented
with NumPy for framework-agnostic prototyping and research.

Modules
-------
transformer
    Transformer-based architectures: multi-head self-attention,
    positional encoding, trajectory transformer, social transformer.
graph_nets
    Graph neural networks: GCN, GAT, GraphSAGE, message passing,
    heterogeneous graphs, edge convolution.
variational
    Variational models: CVAE, VQ-VAE, beta-VAE for trajectory
    prediction with diverse mode coverage.
flow_models
    Normalizing flows: RealNVP, coupling layers, autoregressive
    flows, continuous normalizing flows.
diffusion
    Diffusion models: DDPM, DDIM, conditional diffusion,
    classifier-free guidance for trajectory generation.
"""

from navirl.networks.diffusion import (
    DDPM,
    ClassifierFreeGuidance,
    ConditionalDiffusion,
    DDIMSampler,
    NoiseSchedule,
)
from navirl.networks.flow_models import (
    AffineCouplingLayer,
    AutoregressiveFlow,
    ContinuousNormalizingFlow,
    RealNVP,
)
from navirl.networks.graph_nets import (
    GAT,
    GCN,
    EdgeConvolution,
    GraphAttentionLayer,
    GraphConvolution,
    GraphSAGE,
    GraphSAGELayer,
    HeterogeneousGraphNetwork,
    MessagePassingNetwork,
)
from navirl.networks.transformer import (
    MapCrossAttention,
    MultiHeadSelfAttention,
    PositionalEncoding,
    SocialTransformer,
    TrajectoryTransformer,
    TransformerBlock,
)
from navirl.networks.variational import (
    CVAE,
    VQVAE,
    BetaVAE,
    ConditionalPrior,
    PosteriorCollapseRegularizer,
)

__all__ = [
    # Transformer
    "PositionalEncoding",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TrajectoryTransformer",
    "SocialTransformer",
    "MapCrossAttention",
    # Graph Networks
    "GraphConvolution",
    "GCN",
    "GraphAttentionLayer",
    "GAT",
    "GraphSAGELayer",
    "GraphSAGE",
    "MessagePassingNetwork",
    "HeterogeneousGraphNetwork",
    "EdgeConvolution",
    # Variational
    "CVAE",
    "VQVAE",
    "BetaVAE",
    "ConditionalPrior",
    "PosteriorCollapseRegularizer",
    # Normalizing Flows
    "AffineCouplingLayer",
    "RealNVP",
    "AutoregressiveFlow",
    "ContinuousNormalizingFlow",
    # Diffusion
    "DDPM",
    "DDIMSampler",
    "ConditionalDiffusion",
    "ClassifierFreeGuidance",
    "NoiseSchedule",
]
