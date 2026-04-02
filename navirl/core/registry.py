from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from navirl.core.plugin_validation import (
    PluginValidationError,
    safe_plugin_call,
    validate_plugin_factory,
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
        return safe_plugin_call(
            factory, *args, plugin_name=name, method_name="__init__", **kwargs
        )

    return safe_factory


def register_human_controller(name: str, factory: Callable[..., Any]) -> None:
    """
    Register a human controller factory with validation.

    Args:
        name: Unique name for the human controller
        factory: Factory function that creates controller instances

    Raises:
        PluginValidationError: If validation fails
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Human controller name must be a non-empty string, got: {name}")

    if name in _HUMAN_CONTROLLERS:
        logger.warning("Overriding existing human controller '%s'", name)

    try:
        validate_plugin_factory(factory, name)
    except PluginValidationError:
        logger.error("Failed to register human controller '%s'", name)
        raise

    _HUMAN_CONTROLLERS[name] = factory
    logger.debug("Registered human controller '%s'", name)


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
            factory, *args, plugin_name=name, method_name="__init__", **kwargs
        )

    return safe_factory


def register_robot_controller(name: str, factory: Callable[..., Any]) -> None:
    """
    Register a robot controller factory with validation.

    Args:
        name: Unique name for the robot controller
        factory: Factory function that creates controller instances

    Raises:
        PluginValidationError: If validation fails
        ValueError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Robot controller name must be a non-empty string, got: {name}")

    if name in _ROBOT_CONTROLLERS:
        logger.warning("Overriding existing robot controller '%s'", name)

    try:
        validate_plugin_factory(factory, name)
    except PluginValidationError:
        logger.error("Failed to register robot controller '%s'", name)
        raise

    _ROBOT_CONTROLLERS[name] = factory
    logger.debug("Registered robot controller '%s'", name)


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
            factory, *args, plugin_name=name, method_name="__init__", **kwargs
        )

    return safe_factory


def registry_snapshot() -> dict[str, list[str]]:
    return {
        "backends": sorted(_BACKENDS),
        "human_controllers": sorted(_HUMAN_CONTROLLERS),
        "robot_controllers": sorted(_ROBOT_CONTROLLERS),
    }
