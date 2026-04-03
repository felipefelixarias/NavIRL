"""Configuration utilities for NavIRL.

Provides helpers for managing configuration objects, including
YAML/JSON loading, merging, validation, and command-line argument
parsing integration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


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
    from pathlib import Path

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if not path_obj.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    try:
        with open(path_obj) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {path}: {e}") from e


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

    with open(path) as f:
        for _line_num, line in enumerate(f, 1):
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


def interpolate_env_vars(config: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """Replace ${VAR} patterns with environment variable values.

    Parameters
    ----------
    config : dict
        Configuration with potential env var references.
    strict : bool
        If True, raise KeyError when environment variables are not found.
        If False, log warnings for missing variables.

    Returns
    -------
    dict
        Configuration with env vars resolved.

    Raises
    ------
    KeyError
        In strict mode when an environment variable is not found.
    """
    result: dict[str, Any] = {}

    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = interpolate_env_vars(value, strict=strict)
        elif isinstance(value, str):
            result[key] = _resolve_env_vars(value, strict=strict)
        elif isinstance(value, list):
            result[key] = [
                _resolve_env_vars(v, strict=strict) if isinstance(v, str) else v for v in value
            ]
        else:
            result[key] = value

    return result


def _resolve_env_vars(s: str, strict: bool = False) -> str:
    """Resolve ${VAR} and ${VAR:-default} patterns in a string.

    Parameters
    ----------
    s : str
        String containing environment variable patterns.
    strict : bool
        If True, raise KeyError when environment variables are not found.
        If False, log warnings and leave unexpanded variables as-is.

    Returns
    -------
    str
        String with environment variables resolved.

    Raises
    ------
    KeyError
        In strict mode when an environment variable is not found.
    """
    import logging
    import re

    logger = logging.getLogger(__name__)

    def _replace(match: re.Match) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            # Handle ${VAR:-default} syntax
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)

        # Handle ${VAR} syntax
        value = os.environ.get(var_expr)
        if value is None:
            if strict:
                raise KeyError(f"Environment variable '{var_expr}' not found")
            else:
                logger.warning(f"Environment variable '{var_expr}' not found, leaving unexpanded")
                return match.group(0)  # Return original ${VAR} pattern
        return value

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
