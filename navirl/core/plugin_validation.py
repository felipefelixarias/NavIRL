"""Enhanced plugin validation and security for NavIRL controller APIs."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Union

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for plugin registration and validation."""

    name: str
    version: str
    author: str
    description: str
    api_version: str = "1.0.0"
    required_params: list[str] = None
    optional_params: list[str] = None
    param_schema: dict[str, Any] = None

    def __post_init__(self):
        if self.required_params is None:
            self.required_params = []
        if self.optional_params is None:
            self.optional_params = []
        if self.param_schema is None:
            self.param_schema = {}


class PluginValidationError(Exception):
    """Exception raised when plugin validation fails."""

    pass


class PluginSecurityError(Exception):
    """Exception raised when plugin poses security risks."""

    pass


class ValidatedPlugin(Protocol):
    """Protocol for validated plugins with metadata."""

    __plugin_metadata__: PluginMetadata

    def __call__(self, cfg: dict[str, Any], **kwargs) -> Any:
        """Create plugin instance with validated configuration."""
        ...


def validate_plugin_config(
    cfg: dict[str, Any],
    metadata: PluginMetadata,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate plugin configuration against metadata schema.

    Args:
        cfg: Configuration dictionary
        metadata: Plugin metadata with schema
        strict: If True, reject unknown parameters

    Returns:
        Validated and sanitized configuration

    Raises:
        PluginValidationError: If validation fails
    """
    if cfg is None:
        cfg = {}

    validated_cfg = dict(cfg)

    # Check required parameters
    for param in metadata.required_params:
        if param not in cfg:
            raise PluginValidationError(
                f"Plugin '{metadata.name}' missing required parameter: '{param}'"
            )
    # Validate parameter types if schema provided
    for param in metadata.required_params + metadata.optional_params:
        if param not in cfg or param not in metadata.param_schema:
            continue
        value = cfg[param]
        expected_type = metadata.param_schema[param]
        if not _validate_type(value, expected_type):
            raise PluginValidationError(
                f"Plugin '{metadata.name}' parameter '{param}' has invalid type. "
                f"Expected: {expected_type}, got: {type(value)}"
            )

    # Check for unknown parameters in strict mode
    if strict:
        all_known = set(metadata.required_params + metadata.optional_params)
        unknown = set(cfg.keys()) - all_known
        if unknown:
            logger.warning(
                f"Plugin '{metadata.name}' received unknown parameters: {unknown}. "
                f"Known parameters: {all_known}"
            )

    return validated_cfg


def _validate_type(value: Any, expected_type: type) -> bool:
    """Validate that value matches expected type."""
    if expected_type == Any:
        return True

    if hasattr(expected_type, "__origin__"):
        # Handle generic types like Union, Optional, etc.
        if expected_type.__origin__ is Union:
            return any(_validate_type(value, arg) for arg in expected_type.__args__)
        if expected_type.__origin__ is list:
            return isinstance(value, list) and (
                not expected_type.__args__
                or all(_validate_type(item, expected_type.__args__[0]) for item in value)
            )
        if expected_type.__origin__ is dict:
            return isinstance(value, dict) and (
                not expected_type.__args__
                or all(
                    _validate_type(k, expected_type.__args__[0])
                    and _validate_type(v, expected_type.__args__[1])
                    for k, v in value.items()
                )
            )

    return isinstance(value, expected_type)


def validate_plugin_interface(plugin_class: type, expected_interface: type) -> None:
    """
    Validate that plugin class implements expected interface.

    Args:
        plugin_class: The plugin class to validate
        expected_interface: The expected interface/protocol

    Raises:
        PluginValidationError: If interface validation fails
    """
    if not hasattr(plugin_class, "__plugin_metadata__"):
        raise PluginValidationError(f"Plugin {plugin_class.__name__} missing __plugin_metadata__")

    # Check for required methods
    required_methods = [
        name
        for name, method in inspect.getmembers(expected_interface)
        if inspect.isabstract(method)
        or (hasattr(method, "__isabstractmethod__") and method.__isabstractmethod__)
    ]

    missing_methods = []
    for method_name in required_methods:
        if not hasattr(plugin_class, method_name):
            missing_methods.append(method_name)
        elif not callable(getattr(plugin_class, method_name)):
            missing_methods.append(f"{method_name} (not callable)")

    if missing_methods:
        raise PluginValidationError(
            f"Plugin {plugin_class.__name__} missing required methods: {missing_methods}"
        )


def sanitize_plugin_factory(
    factory: Callable,
    metadata: PluginMetadata,
    interface_class: type = None,
) -> Callable:
    """
    Create a sanitized wrapper around plugin factory with validation.

    Args:
        factory: Original plugin factory function
        metadata: Plugin metadata
        interface_class: Expected interface for validation

    Returns:
        Wrapped factory with validation
    """
    signature = inspect.signature(factory)
    first_param_name = next(iter(signature.parameters), None)

    def _validate_factory_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Validate non-controller factory arguments against metadata."""
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            return

        call_cfg = {
            name: value
            for name, value in bound.arguments.items()
            if name in metadata.required_params or name in metadata.optional_params
        }
        if call_cfg:
            validate_plugin_config(call_cfg, metadata)

    def validated_factory(*args, **kwargs) -> Any:
        """Validated factory wrapper."""
        try:
            # Handle different factory call patterns
            if args and first_param_name == "cfg" and isinstance(args[0], dict):
                # Controller-style: factory(cfg, **kwargs)
                cfg = args[0]
                validated_cfg = validate_plugin_config(cfg, metadata)
                instance = factory(validated_cfg, *args[1:], **kwargs)
            elif "cfg" in kwargs:
                # Named cfg parameter: factory(cfg=cfg, **other_kwargs)
                cfg = kwargs.pop("cfg")
                validated_cfg = validate_plugin_config(cfg, metadata)
                instance = factory(cfg=validated_cfg, **kwargs)
            else:
                # Backend-style or other factories validate call arguments
                # without mutating the original argument payload.
                _validate_factory_arguments(args, kwargs)
                instance = factory(*args, **kwargs)

            # Validate interface if provided and instance is a class instance
            if interface_class and hasattr(instance, "__class__") and not inspect.isclass(instance):
                try:
                    validate_plugin_interface(instance.__class__, interface_class)
                except Exception as e:
                    logger.debug(f"Interface validation skipped for {metadata.name}: {e}")

            # Attach metadata to instance if possible
            if hasattr(instance, "__dict__"):
                instance.__plugin_metadata__ = metadata
            elif hasattr(instance, "__setattr__"):
                try:
                    instance.__plugin_metadata__ = metadata
                except Exception:
                    pass

            logger.info(f"Successfully created plugin '{metadata.name}' v{metadata.version}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create plugin '{metadata.name}': {e}")
            raise PluginValidationError(f"Plugin creation failed: {e}") from e

    # Preserve original function metadata
    validated_factory.__name__ = f"validated_{factory.__name__}"
    validated_factory.__doc__ = f"Validated wrapper for {metadata.name} plugin"
    validated_factory.__plugin_metadata__ = metadata

    return validated_factory


def check_plugin_security(plugin_class: type) -> list[str]:
    """
    Basic security check for plugin classes.

    Args:
        plugin_class: Plugin class to check

    Returns:
        List of security warnings/issues
    """
    warnings = []

    # Check for dangerous imports or methods
    dangerous_patterns = [
        ("subprocess", "subprocess usage"),
        ("os.system", "system command execution"),
        ("eval", "code evaluation"),
        ("exec", "code execution"),
        ("__import__", "dynamic imports"),
        ("open", "file system access"),
    ]

    source_code = inspect.getsource(plugin_class) if hasattr(plugin_class, "__module__") else ""

    for pattern, description in dangerous_patterns:
        if pattern in source_code:
            warnings.append(f"Potentially unsafe: {description}")

    # Check for network-related imports
    network_patterns = ["socket", "urllib", "requests", "http"]
    for pattern in network_patterns:
        if pattern in source_code:
            warnings.append(f"Network access detected: {pattern}")

    return warnings
