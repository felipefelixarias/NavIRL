"""Lazy exports for agent network building blocks.

The heavy agent stack depends on optional ML libraries such as PyTorch.
Keep package import cheap and import-safe by resolving symbols only when they
are explicitly accessed.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "networks": ("navirl.agents.networks", None),
    "MLP": ("navirl.agents.networks", "MLP"),
    "DuelingMLP": ("navirl.agents.networks", "DuelingMLP"),
    "NoisyMLP": ("navirl.agents.networks", "NoisyMLP"),
    "ResidualMLP": ("navirl.agents.networks", "ResidualMLP"),
    "GatedMLP": ("navirl.agents.networks", "GatedMLP"),
    "NatureDQN": ("navirl.agents.networks", "NatureDQN"),
    "ImpalaCNN": ("navirl.agents.networks", "ImpalaCNN"),
    "OccupancyGridEncoder": ("navirl.agents.networks", "OccupancyGridEncoder"),
    "LidarEncoder": ("navirl.agents.networks", "LidarEncoder"),
    "EgoCentricCNN": ("navirl.agents.networks", "EgoCentricCNN"),
    "SocialAttention": ("navirl.agents.networks", "SocialAttention"),
    "MultiHeadSocialAttention": ("navirl.agents.networks", "MultiHeadSocialAttention"),
    "TransformerEncoder": ("navirl.agents.networks", "TransformerEncoder"),
    "GraphAttentionNetwork": ("navirl.agents.networks", "GraphAttentionNetwork"),
    "CombinedExtractor": ("navirl.agents.networks", "CombinedExtractor"),
    "GaussianPolicyHead": ("navirl.agents.networks", "GaussianPolicyHead"),
    "SquashedGaussianHead": ("navirl.agents.networks", "SquashedGaussianHead"),
    "CategoricalPolicyHead": ("navirl.agents.networks", "CategoricalPolicyHead"),
    "DeterministicPolicyHead": ("navirl.agents.networks", "DeterministicPolicyHead"),
    "ValueHead": ("navirl.agents.networks", "ValueHead"),
    "QValueHead": ("navirl.agents.networks", "QValueHead"),
    "LSTMCore": ("navirl.agents.networks", "LSTMCore"),
    "GRUCore": ("navirl.agents.networks", "GRUCore"),
    "RecurrentPolicy": ("navirl.agents.networks", "RecurrentPolicy"),
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
            f"{name} requires optional agent dependencies. "
            "Install the ML stack before importing agent network symbols."
        ) from exc

    value: object | ModuleType
    if attr_name is None:
        value = module
    else:
        value = getattr(module, attr_name)
    globals()[name] = value
    return value
