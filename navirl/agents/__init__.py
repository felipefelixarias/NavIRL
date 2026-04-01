"""Public exports for NavIRL agent utilities and network modules.

The reinforcement-learning networks depend on optional third-party packages
such as PyTorch. Keep package import lightweight so NumPy-only utilities remain
available when those extras are not installed.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "MLP": "navirl.agents.networks",
    "CategoricalPolicyHead": "navirl.agents.networks",
    "CombinedExtractor": "navirl.agents.networks",
    "DeterministicPolicyHead": "navirl.agents.networks",
    "DuelingMLP": "navirl.agents.networks",
    "EgoCentricCNN": "navirl.agents.networks",
    "GatedMLP": "navirl.agents.networks",
    "GaussianPolicyHead": "navirl.agents.networks",
    "GraphAttentionNetwork": "navirl.agents.networks",
    "GRUCore": "navirl.agents.networks",
    "ImpalaCNN": "navirl.agents.networks",
    "LidarEncoder": "navirl.agents.networks",
    "LSTMCore": "navirl.agents.networks",
    "MultiHeadSocialAttention": "navirl.agents.networks",
    "NatureDQN": "navirl.agents.networks",
    "NoisyMLP": "navirl.agents.networks",
    "OccupancyGridEncoder": "navirl.agents.networks",
    "QValueHead": "navirl.agents.networks",
    "RecurrentPolicy": "navirl.agents.networks",
    "ResidualMLP": "navirl.agents.networks",
    "SocialAttention": "navirl.agents.networks",
    "SquashedGaussianHead": "navirl.agents.networks",
    "TransformerEncoder": "navirl.agents.networks",
    "ValueHead": "navirl.agents.networks",
}

__all__ = ["networks", *_EXPORTS]


def __getattr__(name: str) -> object:
    if name == "networks":
        return import_module("navirl.agents.networks")
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
