"""Plugin validation utilities for hardening external API usage."""
from __future__ import annotations

import inspect
import logging
from abc import ABC
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""


class ConfigValidationError(PluginValidationError):
    """Raised when plugin configuration is invalid."""


def validate_plugin_interface(
    plugin_class: type,
    expected_base: type[ABC],
    plugin_name: str
) -> None:
    """
    Validate that a plugin class conforms to the expected interface.

    Args:
        plugin_class: The plugin class to validate
        expected_base: The expected base class/interface
        plugin_name: Name of the plugin for error messages

    Raises:
        PluginValidationError: If validation fails
    """
    if not inspect.isclass(plugin_class):
        raise PluginValidationError(
            f"Plugin '{plugin_name}' must be a class, got {type(plugin_class)}"
        )

    if not issubclass(plugin_class, expected_base):
        raise PluginValidationError(
            f"Plugin '{plugin_name}' must inherit from {expected_base.__name__}, "
            f"got {plugin_class.__mro__}"
        )

    # Check for required methods
    abstract_methods = getattr(expected_base, '__abstractmethods__', set())
    for method_name in abstract_methods:
        if not hasattr(plugin_class, method_name):
            raise PluginValidationError(
                f"Plugin '{plugin_name}' missing required method '{method_name}'"
            )

        method = getattr(plugin_class, method_name)
        if not callable(method):
            raise PluginValidationError(
                f"Plugin '{plugin_name}' method '{method_name}' is not callable"
            )


def validate_plugin_factory(
    factory: Callable[..., Any],
    plugin_name: str
) -> None:
    """
    Validate that a plugin factory function is properly structured.

    Args:
        factory: The factory function to validate
        plugin_name: Name of the plugin for error messages

    Raises:
        PluginValidationError: If validation fails
    """
    if not callable(factory):
        raise PluginValidationError(
            f"Plugin factory for '{plugin_name}' must be callable, got {type(factory)}"
        )

    # Check function signature
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError) as e:
        raise PluginValidationError(
            f"Plugin factory for '{plugin_name}' has invalid signature: {e}"
        ) from e

    # Warn if factory has no parameters (might be too restrictive)
    if not sig.parameters:
        logger.warning(
            "Plugin factory for '%s' takes no parameters - "
            "consider accepting a config parameter for flexibility",
            plugin_name
        )


def validate_controller_config(config: dict | None, plugin_name: str) -> dict:
    """
    Validate and sanitize controller configuration.

    Args:
        config: Configuration dictionary or None
        plugin_name: Name of the plugin for error messages

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ConfigValidationError(
            f"Plugin '{plugin_name}' config must be a dictionary or None, "
            f"got {type(config)}"
        )

    # Check for potentially dangerous keys
    dangerous_keys = ['__class__', '__module__', '__globals__']
    for key in dangerous_keys:
        if key in config:
            raise ConfigValidationError(
                f"Plugin '{plugin_name}' config contains dangerous key '{key}'"
            )

    # Validate common numeric parameters
    numeric_params = {
        'goal_tolerance': (0.01, 10.0, "Goal tolerance"),
        'max_speed': (0.01, 20.0, "Maximum speed"),
        'lookahead': (1, 100, "Lookahead distance"),
        'velocity_smoothing': (0.0, 1.0, "Velocity smoothing factor"),
    }

    validated_config = dict(config)

    for param, (min_val, max_val, description) in numeric_params.items():
        if param in validated_config:
            value = validated_config[param]
            try:
                # Convert to appropriate numeric type
                if isinstance(min_val, int):
                    validated_config[param] = int(value)
                else:
                    validated_config[param] = float(value)

                # Check bounds
                if not min_val <= validated_config[param] <= max_val:
                    raise ConfigValidationError(
                        f"Plugin '{plugin_name}' config parameter '{param}' "
                        f"({description}) must be between {min_val} and {max_val}, "
                        f"got {validated_config[param]}"
                    )

            except (ValueError, TypeError) as e:
                raise ConfigValidationError(
                    f"Plugin '{plugin_name}' config parameter '{param}' "
                    f"({description}) must be numeric, got {value}"
                ) from e

    return validated_config


def safe_plugin_call(
    plugin_method: Callable[..., Any],
    *args,
    plugin_name: str = "unknown",
    method_name: str = "unknown",
    **kwargs
) -> Any:
    """
    Safely call a plugin method with error handling.

    Args:
        plugin_method: The plugin method to call
        *args: Positional arguments
        plugin_name: Name of the plugin for error messages
        method_name: Name of the method for error messages
        **kwargs: Keyword arguments

    Returns:
        Method result

    Raises:
        PluginValidationError: If the call fails in an unexpected way
    """
    try:
        return plugin_method(*args, **kwargs)
    except Exception as e:
        logger.error(
            "Plugin '%s' method '%s' failed: %s",
            plugin_name, method_name, str(e)
        )
        raise PluginValidationError(
            f"Plugin '{plugin_name}' method '{method_name}' failed: {e}"
        ) from e