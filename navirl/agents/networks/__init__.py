"""Lazy public exports for optional PyTorch network components."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "CrossAttention": "navirl.agents.networks.attention",
    "GraphAttentionLayer": "navirl.agents.networks.attention",
    "GraphAttentionNetwork": "navirl.agents.networks.attention",
    "MultiHeadSocialAttention": "navirl.agents.networks.attention",
    "RelationalReasoning": "navirl.agents.networks.attention",
    "SocialAttention": "navirl.agents.networks.attention",
    "SpatialTransformer": "navirl.agents.networks.attention",
    "TemporalAttention": "navirl.agents.networks.attention",
    "TransformerEncoder": "navirl.agents.networks.attention",
    "TransformerEncoderLayer": "navirl.agents.networks.attention",
    "EgoCentricCNN": "navirl.agents.networks.cnn",
    "ImpalaCNN": "navirl.agents.networks.cnn",
    "LidarEncoder": "navirl.agents.networks.cnn",
    "MultiScaleFeatureExtractor": "navirl.agents.networks.cnn",
    "NatureDQN": "navirl.agents.networks.cnn",
    "OccupancyGridEncoder": "navirl.agents.networks.cnn",
    "SpatialAttentionModule": "navirl.agents.networks.cnn",
    "CombinedExtractor": "navirl.agents.networks.extractors",
    "HierarchicalExtractor": "navirl.agents.networks.extractors",
    "LidarExtractor": "navirl.agents.networks.extractors",
    "OccupancyExtractor": "navirl.agents.networks.extractors",
    "RecurrentExtractor": "navirl.agents.networks.extractors",
    "RunningNormalizer": "navirl.agents.networks.extractors",
    "SocialExtractor": "navirl.agents.networks.extractors",
    "StateExtractor": "navirl.agents.networks.extractors",
    "DuelingMLP": "navirl.agents.networks.mlp",
    "GatedMLP": "navirl.agents.networks.mlp",
    "MLP": "navirl.agents.networks.mlp",
    "NoisyLinear": "navirl.agents.networks.mlp",
    "NoisyMLP": "navirl.agents.networks.mlp",
    "ResidualMLP": "navirl.agents.networks.mlp",
    "init_weights_kaiming": "navirl.agents.networks.mlp",
    "init_weights_orthogonal": "navirl.agents.networks.mlp",
    "init_weights_uniform": "navirl.agents.networks.mlp",
    "init_weights_xavier": "navirl.agents.networks.mlp",
    "CategoricalPolicyHead": "navirl.agents.networks.policy_heads",
    "DeterministicPolicyHead": "navirl.agents.networks.policy_heads",
    "DuelingQHead": "navirl.agents.networks.policy_heads",
    "GaussianPolicyHead": "navirl.agents.networks.policy_heads",
    "MultiDiscreteHead": "navirl.agents.networks.policy_heads",
    "QuantileHead": "navirl.agents.networks.policy_heads",
    "QValueHead": "navirl.agents.networks.policy_heads",
    "SquashedGaussianHead": "navirl.agents.networks.policy_heads",
    "TwinQHead": "navirl.agents.networks.policy_heads",
    "ValueHead": "navirl.agents.networks.policy_heads",
    "AttentionOverMemory": "navirl.agents.networks.rnn",
    "GRUCore": "navirl.agents.networks.rnn",
    "HiddenStateManager": "navirl.agents.networks.rnn",
    "LSTMCore": "navirl.agents.networks.rnn",
    "RecurrentPolicy": "navirl.agents.networks.rnn",
    "SequenceEncoder": "navirl.agents.networks.rnn",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
