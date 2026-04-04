"""Configuration package for NavIRL - presets, registry, validation, and serialization."""

from __future__ import annotations

from navirl.config.presets import (
    PRESETS,
    Preset,
    get_preset,
    list_presets,
    merge_presets,
)
from navirl.config.registry import ComponentRegistry
from navirl.config.serialization import (
    cli_args_to_config,
    config_to_cli_args,
    diff_configs,
    load_config,
    merge_configs,
    save_config,
)
from navirl.config.validation import (
    ConfigValidator,
    SchemaBuilder,
    validate_agent_config,
    validate_env_config,
    validate_training_config,
)

__all__ = [
    "PRESETS",
    "ComponentRegistry",
    "ConfigValidator",
    "Preset",
    "SchemaBuilder",
    "cli_args_to_config",
    "config_to_cli_args",
    "diff_configs",
    "get_preset",
    "list_presets",
    "load_config",
    "merge_configs",
    "merge_presets",
    "save_config",
    "validate_agent_config",
    "validate_env_config",
    "validate_training_config",
]
