"""Plugin validation utilities for hardening external API usage."""

from __future__ import annotations

import inspect
import logging
import time
from abc import ABC
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""


class ConfigValidationError(PluginValidationError):
    """Raised when plugin configuration is invalid."""


class PluginSecurityError(PluginValidationError):
    """Raised when plugin poses security risks."""


class PluginPerformanceError(PluginValidationError):
    """Raised when plugin violates performance constraints."""


def validate_plugin_interface(
    plugin_class: type, expected_base: type[ABC], plugin_name: str
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
    abstract_methods = getattr(expected_base, "__abstractmethods__", set())
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


def validate_plugin_factory(factory: Callable[..., Any], plugin_name: str) -> None:
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
            plugin_name,
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
            f"Plugin '{plugin_name}' config must be a dictionary or None, got {type(config)}"
        )

    # Check for potentially dangerous keys
    dangerous_keys = ["__class__", "__module__", "__globals__"]
    for key in dangerous_keys:
        if key in config:
            raise ConfigValidationError(
                f"Plugin '{plugin_name}' config contains dangerous key '{key}'"
            )

    # Validate common numeric parameters
    numeric_params = {
        "goal_tolerance": (0.01, 10.0, "Goal tolerance"),
        "max_speed": (0.01, 20.0, "Maximum speed"),
        "lookahead": (1, 100, "Lookahead distance"),
        "velocity_smoothing": (0.0, 1.0, "Velocity smoothing factor"),
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


def validate_plugin_security(plugin_class: type, plugin_name: str) -> None:
    """
    Validate plugin for potential security risks.

    Args:
        plugin_class: The plugin class to validate
        plugin_name: Name of the plugin for error messages

    Raises:
        PluginSecurityError: If security risks are detected
    """
    # Check for dangerous method names
    dangerous_methods = {"exec", "eval", "compile", "__import__", "open"}
    class_methods = {name for name in dir(plugin_class) if not name.startswith("_")}

    if dangerous_methods & class_methods:
        risky_methods = dangerous_methods & class_methods
        raise PluginSecurityError(
            f"Plugin '{plugin_name}' contains potentially dangerous methods: {risky_methods}"
        )

    # Check for risky imports in the module
    if hasattr(plugin_class, "__module__"):
        module = inspect.getmodule(plugin_class)
        if module and hasattr(module, "__dict__"):
            module_attrs = set(module.__dict__.keys())
            dangerous_imports = {"subprocess", "os", "sys", "importlib"}
            if dangerous_imports & module_attrs:
                logger.warning(
                    "Plugin '%s' imports potentially risky modules: %s",
                    plugin_name,
                    dangerous_imports & module_attrs,
                )


def performance_monitor(max_time_s: float = 1.0):
    """
    Decorator to monitor plugin method performance.

    Args:
        max_time_s: Maximum allowed execution time in seconds

    Returns:
        Decorated function with performance monitoring
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                if elapsed > max_time_s:
                    logger.warning(
                        "Plugin method %s.%s took %.3f seconds (limit: %.3f)",
                        func.__self__.__class__.__name__
                        if hasattr(func, "__self__")
                        else "Unknown",
                        func.__name__,
                        elapsed,
                        max_time_s,
                    )

        return wrapper

    return decorator


def safe_plugin_call(
    plugin_method: Callable[..., Any],
    *args,
    plugin_name: str = "unknown",
    method_name: str = "unknown",
    timeout_s: float = 5.0,
    **kwargs,
) -> Any:
    """
    Safely call a plugin method with error handling and timeouts.

    Args:
        plugin_method: The plugin method to call
        *args: Positional arguments
        plugin_name: Name of the plugin for error messages
        method_name: Name of the method for error messages
        timeout_s: Maximum execution time in seconds
        **kwargs: Keyword arguments

    Returns:
        Method result

    Raises:
        PluginValidationError: If the call fails in an unexpected way
        PluginPerformanceError: If the call exceeds timeout
    """
    start_time = time.perf_counter()
    try:
        result = plugin_method(*args, **kwargs)
        elapsed = time.perf_counter() - start_time

        if elapsed > timeout_s:
            raise PluginPerformanceError(
                f"Plugin '{plugin_name}' method '{method_name}' exceeded timeout "
                f"({elapsed:.3f}s > {timeout_s}s)"
            )

        return result

    except PluginPerformanceError:
        raise  # Re-raise performance errors as-is

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            "Plugin '%s' method '%s' failed after %.3f seconds: %s",
            plugin_name,
            method_name,
            elapsed,
            str(e),
        )
        raise PluginValidationError(
            f"Plugin '{plugin_name}' method '{method_name}' failed: {e}"
        ) from e


def validate_plugin_api_version(
    plugin_class: type, plugin_name: str, required_api_version: str = "1.0"
) -> None:
    """
    Validate plugin API version compatibility.

    Args:
        plugin_class: The plugin class to validate
        plugin_name: Name of the plugin for error messages
        required_api_version: Required API version

    Raises:
        PluginValidationError: If API version is incompatible
    """
    plugin_version = getattr(plugin_class, "__navirl_api_version__", "0.9")

    # Simple version check (can be enhanced for semantic versioning)
    try:
        plugin_major = float(plugin_version.split(".")[0])
        required_major = float(required_api_version.split(".")[0])

        if plugin_major < required_major:
            raise PluginValidationError(
                f"Plugin '{plugin_name}' API version {plugin_version} is too old. "
                f"Required: {required_api_version}+"
            )

        if plugin_major > required_major + 1:
            logger.warning(
                "Plugin '%s' API version %s may be too new (current: %s). "
                "Consider updating NavIRL.",
                plugin_name,
                plugin_version,
                required_api_version,
            )

    except (ValueError, IndexError) as e:
        raise PluginValidationError(
            f"Plugin '{plugin_name}' has invalid API version format: {plugin_version}"
        ) from e
