from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from navirl.core.plugin_validation import (
    PluginValidationError,
    safe_plugin_call,
    validate_plugin_api_version,
    validate_plugin_factory,
    validate_plugin_security,
)

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, Callable[..., Any]] = {}
_HUMAN_CONTROLLERS: dict[str, Callable[..., Any]] = {}
_ROBOT_CONTROLLERS: dict[str, Callable[..., Any]] = {}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    """
    Register a backend factory with validation.

    Args:
        name: Unique name for the backend
        factory: Factory function that creates backend instances

    Raises:
        PluginValidationError: If validation fails
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Backend name must be a non-empty string, got: {name}")

    if name in _BACKENDS:
        logger.warning("Overriding existing backend '%s'", name)

    try:
        validate_plugin_factory(factory, name)
    except PluginValidationError:
        logger.error("Failed to register backend '%s'", name)
        raise

    _BACKENDS[name] = factory
    logger.debug("Registered backend '%s'", name)


def get_backend(name: str) -> Callable[..., Any]:
    """
    Get a registered backend factory with safe instantiation.

    Args:
        name: Name of the backend to retrieve

    Returns:
        Backend factory function

    Raises:
        KeyError: If backend is not registered
    """
    if name not in _BACKENDS:
        available = sorted(_BACKENDS.keys())
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")

    factory = _BACKENDS[name]

    def safe_factory(*args, **kwargs):
        return safe_plugin_call(factory, *args, plugin_name=name, method_name="__init__", **kwargs)

    return safe_factory


def register_human_controller(
    name: str, factory: Callable[..., Any], *, enable_security_validation: bool = True
) -> None:
    """
    Register a human controller factory with comprehensive validation.

    Args:
        name: Unique name for the human controller
        factory: Factory function that creates controller instances
        enable_security_validation: Whether to perform security checks

    Raises:
        PluginValidationError: If validation fails
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Human controller name must be a non-empty string, got: {name}")

    if name in _HUMAN_CONTROLLERS:
        logger.warning("Overriding existing human controller '%s'", name)

    try:
        # Basic factory validation
        validate_plugin_factory(factory, name)

        # Enhanced validation for class-based factories
        if inspect.isclass(factory):
            # API version validation
            validate_plugin_api_version(factory, name, "1.0")

            # Security validation (can be disabled for trusted plugins)
            if enable_security_validation:
                validate_plugin_security(factory, name)

    except PluginValidationError:
        logger.error("Failed to register human controller '%s'", name)
        raise

    _HUMAN_CONTROLLERS[name] = factory
    logger.info("Registered human controller '%s' with enhanced validation", name)


def get_human_controller(name: str) -> Callable[..., Any]:
    """
    Get a registered human controller factory with safe instantiation.

    Args:
        name: Name of the human controller to retrieve

    Returns:
        Human controller factory function

    Raises:
        KeyError: If controller is not registered
    """
    if name not in _HUMAN_CONTROLLERS:
        available = sorted(_HUMAN_CONTROLLERS.keys())
        raise KeyError(f"Unknown human controller '{name}'. Available: {available}")

    factory = _HUMAN_CONTROLLERS[name]

    def safe_factory(*args, **kwargs):
        return safe_plugin_call(
            factory,
            *args,
            plugin_name=name,
            method_name="__init__",
            timeout_s=10.0,  # Allow more time for initialization
            **kwargs,
        )

    return safe_factory


def register_robot_controller(
    name: str, factory: Callable[..., Any], *, enable_security_validation: bool = True
) -> None:
    """
    Register a robot controller factory with comprehensive validation.

    Args:
        name: Unique name for the robot controller
        factory: Factory function that creates controller instances
        enable_security_validation: Whether to perform security checks

    Raises:
        PluginValidationError: If validation fails
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Robot controller name must be a non-empty string, got: {name}")

    if name in _ROBOT_CONTROLLERS:
        logger.warning("Overriding existing robot controller '%s'", name)

    try:
        # Basic factory validation
        validate_plugin_factory(factory, name)

        # Enhanced validation for class-based factories
        if inspect.isclass(factory):
            # API version validation
            validate_plugin_api_version(factory, name, "1.0")

            # Security validation (can be disabled for trusted plugins)
            if enable_security_validation:
                validate_plugin_security(factory, name)

    except PluginValidationError:
        logger.error("Failed to register robot controller '%s'", name)
        raise

    _ROBOT_CONTROLLERS[name] = factory
    logger.info("Registered robot controller '%s' with enhanced validation", name)


def get_robot_controller(name: str) -> Callable[..., Any]:
    """
    Get a registered robot controller factory with safe instantiation.

    Args:
        name: Name of the robot controller to retrieve

    Returns:
        Robot controller factory function

    Raises:
        KeyError: If controller is not registered
    """
    if name not in _ROBOT_CONTROLLERS:
        available = sorted(_ROBOT_CONTROLLERS.keys())
        raise KeyError(f"Unknown robot controller '{name}'. Available: {available}")

    factory = _ROBOT_CONTROLLERS[name]

    def safe_factory(*args, **kwargs):
        return safe_plugin_call(
            factory,
            *args,
            plugin_name=name,
            method_name="__init__",
            timeout_s=10.0,  # Allow more time for initialization
            **kwargs,
        )

    return safe_factory


def registry_snapshot() -> dict[str, list[str]]:
    """Get a snapshot of all registered plugins."""
    return {
        "backends": sorted(_BACKENDS),
        "human_controllers": sorted(_HUMAN_CONTROLLERS),
        "robot_controllers": sorted(_ROBOT_CONTROLLERS),
    }


def get_plugin_info(plugin_type: str, plugin_name: str) -> dict:
    """
    Get detailed information about a registered plugin.

    Args:
        plugin_type: Type of plugin ('human_controller', 'robot_controller', 'backend')
        plugin_name: Name of the plugin

    Returns:
        Dictionary containing plugin information

    Raises:
        KeyError: If plugin is not found
        ValueError: If plugin_type is invalid
    """
    registries = {
        "human_controller": _HUMAN_CONTROLLERS,
        "robot_controller": _ROBOT_CONTROLLERS,
        "backend": _BACKENDS,
    }

    if plugin_type not in registries:
        valid_types = list(registries.keys())
        raise ValueError(f"Invalid plugin type '{plugin_type}'. Valid: {valid_types}")

    registry = registries[plugin_type]
    if plugin_name not in registry:
        available = sorted(registry.keys())
        raise KeyError(f"Plugin '{plugin_name}' not found in {plugin_type}. Available: {available}")

    factory = registry[plugin_name]
    info = {
        "name": plugin_name,
        "type": plugin_type,
        "factory_type": "class" if inspect.isclass(factory) else "function",
    }

    # Add additional metadata for class-based plugins
    if inspect.isclass(factory):
        info.update(
            {
                "module": factory.__module__,
                "doc": factory.__doc__ or "No documentation available",
                "api_version": getattr(factory, "__navirl_api_version__", "unknown"),
                "bases": [base.__name__ for base in factory.__bases__],
            }
        )

        # Check for configuration parameters
        try:
            sig = inspect.signature(factory.__init__)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'
            info["init_parameters"] = params
        except (ValueError, TypeError):
            info["init_parameters"] = "unknown"

    return info


def validate_all_plugins() -> dict[str, list[str]]:
    """
    Validate all registered plugins and return any issues found.

    Returns:
        Dictionary mapping plugin names to lists of validation issues
    """
    issues = {}

    all_plugins = (
        [("human_controller", name, factory) for name, factory in _HUMAN_CONTROLLERS.items()]
        + [("robot_controller", name, factory) for name, factory in _ROBOT_CONTROLLERS.items()]
        + [("backend", name, factory) for name, factory in _BACKENDS.items()]
    )

    for plugin_type, name, factory in all_plugins:
        plugin_issues = []

        try:
            # Basic factory validation
            validate_plugin_factory(factory, name)
        except PluginValidationError as e:
            plugin_issues.append(f"Factory validation: {e}")

        # Class-specific validations
        if inspect.isclass(factory):
            try:
                validate_plugin_api_version(factory, name)
            except PluginValidationError as e:
                plugin_issues.append(f"API version: {e}")

            try:
                validate_plugin_security(factory, name)
            except PluginValidationError as e:
                plugin_issues.append(f"Security: {e}")

        if plugin_issues:
            issues[f"{plugin_type}:{name}"] = plugin_issues

    return issues
