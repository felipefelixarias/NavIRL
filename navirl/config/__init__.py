"""Configuration package for NavIRL – presets, registry, validation, and serialization."""

from __future__ import annotations

from navirl.config.presets import (
    PRESETS,
    Preset,
    get_preset,
    list_presets,
    merge_presets,
)
from navirl.config.registry import ComponentRegistry
from navirl.config.validation import (
    ConfigValidator,
    SchemaBuilder,
    validate_agent_config,
    validate_env_config,
    validate_training_config,
)
from navirl.config.serialization import (
    cli_args_to_config,
    config_to_cli_args,
    diff_configs,
    load_config,
    merge_configs,
    save_config,
)

__all__ = [
    "Preset",
    "PRESETS",
    "get_preset",
    "list_presets",
    "merge_presets",
    "ComponentRegistry",
    "ConfigValidator",
    "SchemaBuilder",
    "validate_agent_config",
    "validate_env_config",
    "validate_training_config",
    "save_config",
    "load_config",
    "config_to_cli_args",
    "cli_args_to_config",
    "merge_configs",
    "diff_configs",
]
