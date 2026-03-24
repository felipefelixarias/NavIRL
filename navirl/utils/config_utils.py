"""Configuration utilities for NavIRL.

Provides helpers for managing configuration objects, including
YAML/JSON loading, merging, validation, and command-line argument
parsing integration.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Configuration merging
# ---------------------------------------------------------------------------

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Parameters
    ----------
    base : dict
        Base dictionary.
    override : dict
        Override dictionary.

    Returns
    -------
    dict
        Merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dictionary.

    Parameters
    ----------
    d : dict
        Nested dictionary.
    parent_key : str
        Prefix for keys.
    separator : str
        Key separator.

    Returns
    -------
    dict
        Flattened dictionary with dotted keys.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(
    d: dict[str, Any],
    separator: str = ".",
) -> dict[str, Any]:
    """Unflatten a dotted-key dictionary into nested dicts.

    Parameters
    ----------
    d : dict
        Flat dictionary with dotted keys.
    separator : str
        Key separator.

    Returns
    -------
    dict
        Nested dictionary.
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(separator)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


# ---------------------------------------------------------------------------
# JSON config loading / saving
# ---------------------------------------------------------------------------

def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_json_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration to save.
    path : str or Path
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# YAML-like config (simple format, no pyyaml dependency)
# ---------------------------------------------------------------------------

def load_simple_config(path: str | Path) -> dict[str, Any]:
    """Load a simple key=value config file.

    Supports basic types: strings, numbers, booleans.
    Lines starting with # are comments.
    Nested keys use dots: parent.child.key = value

    Parameters
    ----------
    path : str or Path
        Config file path.

    Returns
    -------
    dict
        Parsed configuration.
    """
    flat: dict[str, Any] = {}

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Strip inline comments
            if " #" in value:
                value = value[: value.index(" #")].strip()

            # Parse value type
            flat[key] = _parse_value(value)

    return unflatten_dict(flat)


def _parse_value(value: str) -> Any:
    """Parse a string value into the appropriate Python type."""
    # Boolean
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    if value.lower() in ("none", "null"):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]

    # String (strip quotes if present)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    return value


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------

def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a dictionary, handling nested dataclasses.

    Parameters
    ----------
    obj : dataclass
        Dataclass instance.

    Returns
    -------
    dict
        Dictionary representation.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def dict_to_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    """Create a dataclass instance from a dictionary.

    Handles nested dataclass fields by recursively constructing them.

    Parameters
    ----------
    cls : type
        Dataclass type.
    data : dict
        Dictionary data.

    Returns
    -------
    T
        Dataclass instance.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs: dict[str, Any] = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]

        # Check if the field type is itself a dataclass
        field_type = f.type
        if isinstance(field_type, str):
            # Handle forward references
            kwargs[f.name] = value
        elif is_dataclass(field_type) and isinstance(value, dict):
            kwargs[f.name] = dict_to_dataclass(field_type, value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------

def interpolate_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Replace ${VAR} patterns with environment variable values.

    Parameters
    ----------
    config : dict
        Configuration with potential env var references.

    Returns
    -------
    dict
        Configuration with env vars resolved.
    """
    result: dict[str, Any] = {}

    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = interpolate_env_vars(value)
        elif isinstance(value, str):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, list):
            result[key] = [
                _resolve_env_vars(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value

    return result


def _resolve_env_vars(s: str) -> str:
    """Resolve ${VAR} and ${VAR:-default} patterns in a string."""
    import re

    def _replace(match: re.Match) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(var_expr, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", _replace, s)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """A configuration validation error."""
    path: str
    message: str
    severity: str = "error"  # "error" or "warning"


class ConfigValidator:
    """Validates configuration dictionaries against a schema.

    The schema is a dictionary specifying expected keys, types,
    ranges, and required fields.

    Parameters
    ----------
    schema : dict
        Validation schema.

    Examples
    --------
    >>> schema = {
    ...     "learning_rate": {"type": float, "min": 0, "max": 1, "required": True},
    ...     "batch_size": {"type": int, "min": 1, "default": 32},
    ...     "optimizer": {"type": str, "choices": ["adam", "sgd"]},
    ... }
    >>> validator = ConfigValidator(schema)
    >>> errors = validator.validate({"learning_rate": 0.001})
    """

    def __init__(self, schema: dict[str, dict[str, Any]]) -> None:
        self.schema = schema

    def validate(self, config: dict[str, Any]) -> list[ValidationError]:
        """Validate a configuration against the schema.

        Parameters
        ----------
        config : dict
            Configuration to validate.

        Returns
        -------
        list of ValidationError
            List of validation errors/warnings.
        """
        errors: list[ValidationError] = []

        # Check required fields
        for key, spec in self.schema.items():
            if spec.get("required", False) and key not in config:
                errors.append(ValidationError(
                    path=key,
                    message=f"Required field '{key}' is missing",
                ))

        # Check each config value
        for key, value in config.items():
            if key not in self.schema:
                errors.append(ValidationError(
                    path=key,
                    message=f"Unknown configuration key '{key}'",
                    severity="warning",
                ))
                continue

            spec = self.schema[key]
            errors.extend(self._validate_field(key, value, spec))

        return errors

    def _validate_field(
        self,
        path: str,
        value: Any,
        spec: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate a single field."""
        errors: list[ValidationError] = []

        # Type check
        expected_type = spec.get("type")
        if expected_type is not None and not isinstance(value, expected_type):
            errors.append(ValidationError(
                path=path,
                message=(
                    f"Expected type {expected_type.__name__} for '{path}', "
                    f"got {type(value).__name__}"
                ),
            ))
            return errors  # Skip further checks if type is wrong

        # Range checks
        min_val = spec.get("min")
        if min_val is not None and value < min_val:
            errors.append(ValidationError(
                path=path,
                message=f"Value {value} for '{path}' is below minimum {min_val}",
            ))

        max_val = spec.get("max")
        if max_val is not None and value > max_val:
            errors.append(ValidationError(
                path=path,
                message=f"Value {value} for '{path}' exceeds maximum {max_val}",
            ))

        # Choice check
        choices = spec.get("choices")
        if choices is not None and value not in choices:
            errors.append(ValidationError(
                path=path,
                message=f"Value '{value}' for '{path}' not in {choices}",
            ))

        # Length check (for strings and lists)
        min_len = spec.get("min_length")
        if min_len is not None and hasattr(value, "__len__") and len(value) < min_len:
            errors.append(ValidationError(
                path=path,
                message=f"Length {len(value)} for '{path}' is below minimum {min_len}",
            ))

        max_len = spec.get("max_length")
        if max_len is not None and hasattr(value, "__len__") and len(value) > max_len:
            errors.append(ValidationError(
                path=path,
                message=f"Length {len(value)} for '{path}' exceeds maximum {max_len}",
            ))

        return errors

    def apply_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply default values for missing fields.

        Parameters
        ----------
        config : dict
            Input configuration.

        Returns
        -------
        dict
            Configuration with defaults applied.
        """
        result = dict(config)
        for key, spec in self.schema.items():
            if key not in result and "default" in spec:
                result[key] = copy.deepcopy(spec["default"])
        return result

    def get_help(self) -> str:
        """Generate a help string describing the schema.

        Returns
        -------
        str
            Formatted help text.
        """
        lines = []
        for key, spec in sorted(self.schema.items()):
            type_str = spec.get("type", Any).__name__ if "type" in spec else "any"
            required = "required" if spec.get("required") else "optional"
            default = f", default={spec['default']}" if "default" in spec else ""
            desc = spec.get("description", "")
            lines.append(f"  {key} ({type_str}, {required}{default}): {desc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parser integration
# ---------------------------------------------------------------------------

def config_to_argparse_args(config: dict[str, Any]) -> list[str]:
    """Convert a config dict to command-line argument format.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    list of str
        Command-line arguments.
    """
    flat = flatten_dict(config)
    args = []
    for key, value in flat.items():
        arg_name = f"--{key.replace('.', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        elif isinstance(value, (list, tuple)):
            args.append(arg_name)
            args.extend(str(v) for v in value)
        else:
            args.extend([arg_name, str(value)])
    return args


def argparse_args_to_config(args: list[str]) -> dict[str, Any]:
    """Convert command-line arguments to a config dict.

    Handles --key value and --key=value formats.

    Parameters
    ----------
    args : list of str
        Command-line arguments (without program name).

    Returns
    -------
    dict
        Configuration dictionary (possibly nested).
    """
    flat: dict[str, Any] = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if not arg.startswith("--"):
            i += 1
            continue

        key = arg[2:].replace("-", ".")

        if "=" in key:
            key, _, value = key.partition("=")
            flat[key] = _parse_value(value)
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            flat[key] = _parse_value(args[i + 1])
            i += 1
        else:
            flat[key] = True

        i += 1

    return unflatten_dict(flat)


# ---------------------------------------------------------------------------
# Config diffing
# ---------------------------------------------------------------------------

def config_diff(
    config_a: dict[str, Any],
    config_b: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Compute differences between two configurations.

    Parameters
    ----------
    config_a : dict
        First configuration.
    config_b : dict
        Second configuration.

    Returns
    -------
    dict
        Mapping of changed keys to (old_value, new_value) tuples.
        Uses dotted keys for nested values.
    """
    flat_a = flatten_dict(config_a)
    flat_b = flatten_dict(config_b)

    all_keys = set(flat_a.keys()) | set(flat_b.keys())
    diff = {}

    for key in sorted(all_keys):
        val_a = flat_a.get(key, "<missing>")
        val_b = flat_b.get(key, "<missing>")
        if val_a != val_b:
            diff[key] = (val_a, val_b)

    return diff


def format_config_diff(diff: dict[str, tuple[Any, Any]]) -> str:
    """Format a config diff for display.

    Parameters
    ----------
    diff : dict
        Output from ``config_diff()``.

    Returns
    -------
    str
        Formatted diff string.
    """
    if not diff:
        return "No differences."

    lines = []
    for key, (old, new) in sorted(diff.items()):
        if old == "<missing>":
            lines.append(f"  + {key}: {new}")
        elif new == "<missing>":
            lines.append(f"  - {key}: {old}")
        else:
            lines.append(f"  ~ {key}: {old} -> {new}")
    return "\n".join(lines)
