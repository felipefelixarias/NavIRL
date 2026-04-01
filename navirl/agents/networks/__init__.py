"""Lazy exports for optional PyTorch-based network modules."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "MLP": ("navirl.agents.networks.mlp", "MLP"),
    "DuelingMLP": ("navirl.agents.networks.mlp", "DuelingMLP"),
    "NoisyLinear": ("navirl.agents.networks.mlp", "NoisyLinear"),
    "NoisyMLP": ("navirl.agents.networks.mlp", "NoisyMLP"),
    "ResidualMLP": ("navirl.agents.networks.mlp", "ResidualMLP"),
    "GatedMLP": ("navirl.agents.networks.mlp", "GatedMLP"),
    "init_weights_xavier": ("navirl.agents.networks.mlp", "init_weights_xavier"),
    "init_weights_orthogonal": ("navirl.agents.networks.mlp", "init_weights_orthogonal"),
    "init_weights_kaiming": ("navirl.agents.networks.mlp", "init_weights_kaiming"),
    "init_weights_uniform": ("navirl.agents.networks.mlp", "init_weights_uniform"),
    "NatureDQN": ("navirl.agents.networks.cnn", "NatureDQN"),
    "ImpalaCNN": ("navirl.agents.networks.cnn", "ImpalaCNN"),
    "OccupancyGridEncoder": ("navirl.agents.networks.cnn", "OccupancyGridEncoder"),
    "LidarEncoder": ("navirl.agents.networks.cnn", "LidarEncoder"),
    "EgoCentricCNN": ("navirl.agents.networks.cnn", "EgoCentricCNN"),
    "SpatialAttentionModule": ("navirl.agents.networks.cnn", "SpatialAttentionModule"),
    "MultiScaleFeatureExtractor": ("navirl.agents.networks.cnn", "MultiScaleFeatureExtractor"),
    "SocialAttention": ("navirl.agents.networks.attention", "SocialAttention"),
    "MultiHeadSocialAttention": ("navirl.agents.networks.attention", "MultiHeadSocialAttention"),
    "TransformerEncoder": ("navirl.agents.networks.attention", "TransformerEncoder"),
    "TransformerEncoderLayer": ("navirl.agents.networks.attention", "TransformerEncoderLayer"),
    "GraphAttentionNetwork": ("navirl.agents.networks.attention", "GraphAttentionNetwork"),
    "GraphAttentionLayer": ("navirl.agents.networks.attention", "GraphAttentionLayer"),
    "CrossAttention": ("navirl.agents.networks.attention", "CrossAttention"),
    "RelationalReasoning": ("navirl.agents.networks.attention", "RelationalReasoning"),
    "TemporalAttention": ("navirl.agents.networks.attention", "TemporalAttention"),
    "SpatialTransformer": ("navirl.agents.networks.attention", "SpatialTransformer"),
    "StateExtractor": ("navirl.agents.networks.extractors", "StateExtractor"),
    "LidarExtractor": ("navirl.agents.networks.extractors", "LidarExtractor"),
    "OccupancyExtractor": ("navirl.agents.networks.extractors", "OccupancyExtractor"),
    "SocialExtractor": ("navirl.agents.networks.extractors", "SocialExtractor"),
    "CombinedExtractor": ("navirl.agents.networks.extractors", "CombinedExtractor"),
    "HierarchicalExtractor": ("navirl.agents.networks.extractors", "HierarchicalExtractor"),
    "RecurrentExtractor": ("navirl.agents.networks.extractors", "RecurrentExtractor"),
    "RunningNormalizer": ("navirl.agents.networks.extractors", "RunningNormalizer"),
    "GaussianPolicyHead": ("navirl.agents.networks.policy_heads", "GaussianPolicyHead"),
    "SquashedGaussianHead": ("navirl.agents.networks.policy_heads", "SquashedGaussianHead"),
    "CategoricalPolicyHead": ("navirl.agents.networks.policy_heads", "CategoricalPolicyHead"),
    "MultiDiscreteHead": ("navirl.agents.networks.policy_heads", "MultiDiscreteHead"),
    "DeterministicPolicyHead": ("navirl.agents.networks.policy_heads", "DeterministicPolicyHead"),
    "ValueHead": ("navirl.agents.networks.policy_heads", "ValueHead"),
    "QValueHead": ("navirl.agents.networks.policy_heads", "QValueHead"),
    "DuelingQHead": ("navirl.agents.networks.policy_heads", "DuelingQHead"),
    "QuantileHead": ("navirl.agents.networks.policy_heads", "QuantileHead"),
    "TwinQHead": ("navirl.agents.networks.policy_heads", "TwinQHead"),
    "LSTMCore": ("navirl.agents.networks.rnn", "LSTMCore"),
    "GRUCore": ("navirl.agents.networks.rnn", "GRUCore"),
    "RecurrentPolicy": ("navirl.agents.networks.rnn", "RecurrentPolicy"),
    "SequenceEncoder": ("navirl.agents.networks.rnn", "SequenceEncoder"),
    "HiddenStateManager": ("navirl.agents.networks.rnn", "HiddenStateManager"),
    "AttentionOverMemory": ("navirl.agents.networks.rnn", "AttentionOverMemory"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} requires optional PyTorch dependencies. "
            "Install torch before importing network modules."
        ) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
