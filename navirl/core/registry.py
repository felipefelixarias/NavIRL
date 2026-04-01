from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from navirl.core.plugin_validation import (
    PluginMetadata,
    check_plugin_security,
    sanitize_plugin_factory,
)

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, Callable[..., Any]] = {}
_HUMAN_CONTROLLERS: dict[str, Callable[..., Any]] = {}
_ROBOT_CONTROLLERS: dict[str, Callable[..., Any]] = {}

# Enhanced storage for plugin metadata
_PLUGIN_METADATA: dict[str, PluginMetadata] = {}
_SECURITY_WARNINGS: dict[str, list[str]] = {}


def register_backend(
    name: str,
    factory: Callable[..., Any],
    metadata: PluginMetadata | None = None,
    validate_security: bool = True,
) -> None:
    """
    Register a backend plugin with enhanced validation.

    Args:
        name: Plugin name
        factory: Factory function for creating plugin instances
        metadata: Optional plugin metadata for validation
        validate_security: Whether to perform security checks
    """
    if metadata:
        # Create validated factory
        factory = sanitize_plugin_factory(factory, metadata)
        _PLUGIN_METADATA[f"backend:{name}"] = metadata

        # Security check if enabled
        if validate_security and hasattr(factory, "__code__"):
            warnings = check_plugin_security(factory)
            if warnings:
                _SECURITY_WARNINGS[f"backend:{name}"] = warnings
                logger.warning(f"Backend '{name}' has security warnings: {warnings}")

    _BACKENDS[name] = factory
    logger.info(f"Registered backend: {name}")


def get_backend(name: str, show_warnings: bool = True) -> Callable[..., Any]:
    """
    Get a registered backend factory with validation.

    Args:
        name: Backend name
        show_warnings: Whether to display security warnings

    Returns:
        Backend factory function

    Raises:
        KeyError: If backend is not registered
    """
    if name not in _BACKENDS:
        available = sorted(_BACKENDS.keys())
        suggestion = _suggest_similar_name(name, available)
        suggestion_text = f" (Did you mean '{suggestion}'?)" if suggestion else ""
        raise KeyError(
            f"Unknown backend '{name}'. " f"Available backends: {available}" f"{suggestion_text}"
        )

    # Show security warnings if any
    warning_key = f"backend:{name}"
    if show_warnings and warning_key in _SECURITY_WARNINGS:
        logger.warning(f"Backend '{name}' security warnings: {_SECURITY_WARNINGS[warning_key]}")

    return _BACKENDS[name]


def register_human_controller(
    name: str,
    factory: Callable[..., Any],
    metadata: PluginMetadata | None = None,
    interface_class: type | None = None,
    validate_security: bool = True,
) -> None:
    """
    Register a human controller plugin with enhanced validation.

    Args:
        name: Plugin name
        factory: Factory function for creating plugin instances
        metadata: Optional plugin metadata for validation
        interface_class: Expected interface/base class
        validate_security: Whether to perform security checks
    """
    if metadata:
        # Create validated factory
        factory = sanitize_plugin_factory(factory, metadata, interface_class)
        _PLUGIN_METADATA[f"human_controller:{name}"] = metadata

        # Security check if enabled
        if validate_security:
            # Try to get the actual class being created
            try:
                test_instance = factory({})
                warnings = check_plugin_security(test_instance.__class__)
                if warnings:
                    _SECURITY_WARNINGS[f"human_controller:{name}"] = warnings
                    logger.warning(f"Human controller '{name}' has security warnings: {warnings}")
            except Exception as e:
                logger.debug(f"Could not perform security check on {name}: {e}")

    _HUMAN_CONTROLLERS[name] = factory
    logger.info(f"Registered human controller: {name}")


def get_human_controller(name: str, show_warnings: bool = True) -> Callable[..., Any]:
    """
    Get a registered human controller factory with validation.

    Args:
        name: Controller name
        show_warnings: Whether to display security warnings

    Returns:
        Controller factory function

    Raises:
        KeyError: If controller is not registered
    """
    if name not in _HUMAN_CONTROLLERS:
        available = sorted(_HUMAN_CONTROLLERS.keys())
        suggestion = _suggest_similar_name(name, available)
        suggestion_text = f" (Did you mean '{suggestion}'?)" if suggestion else ""
        raise KeyError(
            f"Unknown human controller '{name}'. "
            f"Available controllers: {available}"
            f"{suggestion_text}"
        )

    # Show security warnings if any
    warning_key = f"human_controller:{name}"
    if show_warnings and warning_key in _SECURITY_WARNINGS:
        logger.warning(
            f"Human controller '{name}' security warnings: {_SECURITY_WARNINGS[warning_key]}"
        )

    return _HUMAN_CONTROLLERS[name]


def register_robot_controller(
    name: str,
    factory: Callable[..., Any],
    metadata: PluginMetadata | None = None,
    interface_class: type | None = None,
    validate_security: bool = True,
) -> None:
    """
    Register a robot controller plugin with enhanced validation.

    Args:
        name: Plugin name
        factory: Factory function for creating plugin instances
        metadata: Optional plugin metadata for validation
        interface_class: Expected interface/base class
        validate_security: Whether to perform security checks
    """
    if metadata:
        # Create validated factory
        factory = sanitize_plugin_factory(factory, metadata, interface_class)
        _PLUGIN_METADATA[f"robot_controller:{name}"] = metadata

        # Security check if enabled
        if validate_security:
            try:
                test_instance = factory({})
                warnings = check_plugin_security(test_instance.__class__)
                if warnings:
                    _SECURITY_WARNINGS[f"robot_controller:{name}"] = warnings
                    logger.warning(f"Robot controller '{name}' has security warnings: {warnings}")
            except Exception as e:
                logger.debug(f"Could not perform security check on {name}: {e}")

    _ROBOT_CONTROLLERS[name] = factory
    logger.info(f"Registered robot controller: {name}")


def get_robot_controller(name: str, show_warnings: bool = True) -> Callable[..., Any]:
    """
    Get a registered robot controller factory with validation.

    Args:
        name: Controller name
        show_warnings: Whether to display security warnings

    Returns:
        Controller factory function

    Raises:
        KeyError: If controller is not registered
    """
    if name not in _ROBOT_CONTROLLERS:
        available = sorted(_ROBOT_CONTROLLERS.keys())
        suggestion = _suggest_similar_name(name, available)
        suggestion_text = f" (Did you mean '{suggestion}'?)" if suggestion else ""
        raise KeyError(
            f"Unknown robot controller '{name}'. "
            f"Available controllers: {available}"
            f"{suggestion_text}"
        )

    # Show security warnings if any
    warning_key = f"robot_controller:{name}"
    if show_warnings and warning_key in _SECURITY_WARNINGS:
        logger.warning(
            f"Robot controller '{name}' security warnings: {_SECURITY_WARNINGS[warning_key]}"
        )

    return _ROBOT_CONTROLLERS[name]


def _suggest_similar_name(name: str, available: list[str]) -> str | None:
    """Suggest a similar name from available options using basic string similarity."""
    if not available:
        return None

    # Simple similarity based on common prefixes and edit distance
    best_match = None
    best_score = 0

    for candidate in available:
        # Check common prefix
        prefix_score = 0
        for i in range(min(len(name), len(candidate))):
            if name[i].lower() == candidate[i].lower():
                prefix_score += 1
            else:
                break

        # Check if name is a substring
        substring_score = 1 if name.lower() in candidate.lower() else 0

        # Simple score combining prefix and substring matching
        total_score = prefix_score * 2 + substring_score * 3

        if total_score > best_score and total_score > 1:  # Require minimum similarity
            best_score = total_score
            best_match = candidate

    return best_match


def registry_snapshot() -> dict[str, Any]:
    """
    Get a comprehensive snapshot of the current registry state.

    Returns:
        Dictionary with registered plugins and their metadata
    """
    return {
        "backends": sorted(_BACKENDS.keys()),
        "human_controllers": sorted(_HUMAN_CONTROLLERS.keys()),
        "robot_controllers": sorted(_ROBOT_CONTROLLERS.keys()),
        "metadata": dict(_PLUGIN_METADATA),
        "security_warnings": dict(_SECURITY_WARNINGS),
        "total_plugins": len(_BACKENDS) + len(_HUMAN_CONTROLLERS) + len(_ROBOT_CONTROLLERS),
    }


def get_plugin_metadata(plugin_type: str, name: str) -> PluginMetadata | None:
    """
    Get metadata for a specific plugin.

    Args:
        plugin_type: Type of plugin ('backend', 'human_controller', 'robot_controller')
        name: Plugin name

    Returns:
        Plugin metadata if available, None otherwise
    """
    key = f"{plugin_type}:{name}"
    return _PLUGIN_METADATA.get(key)


def get_security_warnings(plugin_type: str, name: str) -> list[str]:
    """
    Get security warnings for a specific plugin.

    Args:
        plugin_type: Type of plugin ('backend', 'human_controller', 'robot_controller')
        name: Plugin name

    Returns:
        List of security warnings
    """
    key = f"{plugin_type}:{name}"
    return _SECURITY_WARNINGS.get(key, [])


def clear_registry() -> None:
    """Clear all registered plugins. Useful for testing."""
    global _BACKENDS, _HUMAN_CONTROLLERS, _ROBOT_CONTROLLERS, _PLUGIN_METADATA, _SECURITY_WARNINGS
    _BACKENDS.clear()
    _HUMAN_CONTROLLERS.clear()
    _ROBOT_CONTROLLERS.clear()
    _PLUGIN_METADATA.clear()
    _SECURITY_WARNINGS.clear()
    logger.info("Registry cleared")
