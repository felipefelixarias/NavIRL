"""Pre-built configuration presets for common NavIRL workflows.

Each preset bundles environment, agent, and training configurations so that
users can get started quickly without manual tuning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Preset dataclass
# ---------------------------------------------------------------------------


@dataclass
class Preset:
    """A named, self-documenting configuration bundle.

    Attributes
    ----------
    name : str
        Short identifier (used as the key in :data:`PRESETS`).
    description : str
        Human-readable summary of what the preset is tuned for.
    env_config : dict[str, Any]
        Environment-level configuration.
    agent_config : dict[str, Any]
        Agent / policy configuration.
    training_config : dict[str, Any]
        Training loop configuration.
    """

    name: str
    description: str
    env_config: dict[str, Any] = field(default_factory=dict)
    agent_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Standard presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, Preset] = {
    "debug": Preset(
        name="debug",
        description="Small network, few steps - fast iteration for debugging.",
        env_config={
            "num_humans": 3,
            "env_size": 6.0,
            "time_limit": 30,
        },
        agent_config={
            "hidden_sizes": [32, 32],
            "learning_rate": 3e-4,
            "batch_size": 32,
        },
        training_config={
            "total_steps": 5_000,
            "eval_interval": 1_000,
            "log_interval": 100,
            "seed": 0,
        },
    ),
    "fast_train": Preset(
        name="fast_train",
        description="Medium network, moderate steps - quick experiments.",
        env_config={
            "num_humans": 5,
            "env_size": 8.0,
            "time_limit": 50,
        },
        agent_config={
            "hidden_sizes": [128, 64],
            "learning_rate": 1e-4,
            "batch_size": 128,
        },
        training_config={
            "total_steps": 100_000,
            "eval_interval": 10_000,
            "log_interval": 1_000,
            "seed": 42,
        },
    ),
    "full_train": Preset(
        name="full_train",
        description="Large network, full training schedule for publication-quality results.",
        env_config={
            "num_humans": 10,
            "env_size": 10.0,
            "time_limit": 100,
        },
        agent_config={
            "hidden_sizes": [256, 128, 64],
            "learning_rate": 3e-5,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
        },
        training_config={
            "total_steps": 1_000_000,
            "eval_interval": 50_000,
            "log_interval": 5_000,
            "seed": 42,
            "num_eval_episodes": 100,
        },
    ),
    "social_nav": Preset(
        name="social_nav",
        description="Tuned for social navigation benchmarks (e.g., SocNavBench).",
        env_config={
            "num_humans": 8,
            "env_size": 10.0,
            "time_limit": 80,
            "human_policy": "orca",
            "randomise_goals": True,
        },
        agent_config={
            "hidden_sizes": [256, 128],
            "learning_rate": 5e-5,
            "batch_size": 256,
            "use_social_features": True,
            "attention_heads": 4,
        },
        training_config={
            "total_steps": 500_000,
            "eval_interval": 25_000,
            "log_interval": 2_500,
            "seed": 42,
            "reward_shaping": "social",
        },
    ),
    "crowd_dense": Preset(
        name="crowd_dense",
        description="Optimised for dense crowd scenarios (20+ agents).",
        env_config={
            "num_humans": 25,
            "env_size": 12.0,
            "time_limit": 120,
            "human_policy": "sfm",
        },
        agent_config={
            "hidden_sizes": [256, 256, 128],
            "learning_rate": 3e-5,
            "batch_size": 512,
            "use_social_features": True,
            "attention_heads": 8,
            "max_neighbours": 20,
        },
        training_config={
            "total_steps": 2_000_000,
            "eval_interval": 100_000,
            "log_interval": 10_000,
            "seed": 42,
            "safety_constraint": True,
            "cost_limit": 25.0,
        },
    ),
    "multi_robot": Preset(
        name="multi_robot",
        description="Multi-agent cooperative navigation settings.",
        env_config={
            "num_robots": 4,
            "num_humans": 6,
            "env_size": 14.0,
            "time_limit": 100,
            "cooperative": True,
        },
        agent_config={
            "hidden_sizes": [256, 128],
            "learning_rate": 1e-4,
            "batch_size": 256,
            "communication_dim": 32,
            "share_parameters": True,
        },
        training_config={
            "total_steps": 1_000_000,
            "eval_interval": 50_000,
            "log_interval": 5_000,
            "seed": 42,
            "multi_agent": True,
        },
    ),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_preset(name: str) -> Preset:
    """Retrieve a preset by name.

    Parameters
    ----------
    name : str
        Key in :data:`PRESETS`.

    Returns
    -------
    Preset
        A **deep copy** so callers can mutate freely.

    Raises
    ------
    KeyError
        If *name* is not a registered preset.
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available presets: {available}")
    return copy.deepcopy(PRESETS[name])


def list_presets() -> list[tuple[str, str]]:
    """Return a list of ``(name, description)`` pairs for all presets."""
    return [(p.name, p.description) for p in PRESETS.values()]


def merge_presets(base: str | Preset, overrides: dict[str, Any]) -> Preset:
    """Create a preset by merging *overrides* into a base preset.

    Parameters
    ----------
    base : str | Preset
        Base preset name or instance.
    overrides : dict
        Keys ``"env_config"``, ``"agent_config"``, ``"training_config"``
        whose values are dicts merged (shallowly) into the base.

    Returns
    -------
    Preset
        New preset with merged configuration.
    """
    if isinstance(base, str):
        preset = get_preset(base)
    else:
        preset = copy.deepcopy(base)

    for section in ("env_config", "agent_config", "training_config"):
        if section in overrides:
            getattr(preset, section).update(overrides[section])

    # Allow top-level name / description overrides.
    if "name" in overrides:
        preset.name = overrides["name"]
    if "description" in overrides:
        preset.description = overrides["description"]

    return preset
