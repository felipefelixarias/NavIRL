"""Configuration serialization and merging utilities.

Supports YAML, JSON, and TOML formats, plus bidirectional CLI-argument
conversion and config diffing.
"""

from __future__ import annotations

import copy
import json
import pathlib
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_config(
    config: dict[str, Any],
    path: str | pathlib.Path,
    file_format: str | None = None,
) -> None:
    """Persist *config* to *path* in the requested format.

    Parameters
    ----------
    config : dict
        Configuration to save.
    path : str | Path
        Destination file path.
    file_format : str, optional
        ``"yaml"``, ``"json"``, or ``"toml"``.  Inferred from extension when
        ``None``.

    Raises
    ------
    ValueError
        If file_format is unsupported or config cannot be serialized.
    OSError
        If path cannot be created or written to.
    ImportError
        If required serialization library (yaml/toml) is not available.
    """
    try:
        path = pathlib.Path(path)
        fmt = _resolve_format(path, file_format)
        path = _normalize_output_path(path, fmt)

        # Validate config is not None
        if config is None:
            raise ValueError("Configuration cannot be None")

        # Create directory with proper error handling
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory {path.parent}: {e}") from e

        if fmt == "json":
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(config, fh, indent=2, default=str)
            except OSError as e:
                raise OSError(f"Cannot write JSON to {path}: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Configuration cannot be serialized to JSON: {e}") from e

        elif fmt == "yaml":
            try:
                yaml = _import_yaml()
                with open(path, "w", encoding="utf-8") as fh:
                    yaml.dump(config, fh, default_flow_style=False, sort_keys=False)
            except OSError as e:
                raise OSError(f"Cannot write YAML to {path}: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Configuration cannot be serialized to YAML: {e}") from e
            except Exception as e:
                # Catch YAML-specific errors (implementation varies by library)
                raise ValueError(f"Configuration cannot be serialized to YAML: {e}") from e

        elif fmt == "toml":
            try:
                toml_mod = _import_toml_write()
                with open(path, "w", encoding="utf-8") as fh:
                    toml_mod.dump(config, fh)
            except OSError as e:
                raise OSError(f"Cannot write TOML to {path}: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Configuration cannot be serialized to TOML: {e}") from e
            except Exception as e:
                # Catch TOML-specific errors (implementation varies by library)
                raise ValueError(f"Configuration cannot be serialized to TOML: {e}") from e

        else:
            raise ValueError(f"Unsupported format: '{fmt}'")

    except Exception as e:
        # Re-raise with context if this is an unexpected error
        if isinstance(e, (ValueError, OSError, ImportError)):
            raise
        raise RuntimeError(f"Unexpected error saving config to {path}: {e}") from e


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a configuration file (YAML, JSON, or TOML).

    The format is inferred from the file extension.

    Parameters
    ----------
    path : str | Path
        Source file path.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If file format is unsupported or content cannot be parsed.
    OSError
        If file cannot be read due to permissions or I/O errors.
    ImportError
        If required parsing library (yaml/toml) is not available.
    """
    try:
        path = pathlib.Path(path)

        # Resolve path (may add extension) before existence check
        resolved_path = _resolve_existing_path(path)

        # Validate file exists and is readable after path resolution
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Path is not a file: {resolved_path}")

        fmt = _resolve_format(resolved_path, None)

        if fmt == "json":
            try:
                with open(resolved_path, encoding="utf-8") as fh:
                    config = json.load(fh)
                    if not isinstance(config, dict):
                        raise ValueError(
                            f"JSON file must contain a dictionary, got {type(config).__name__}"
                        )
                    return config
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {resolved_path}: {e}") from e
            except OSError as e:
                raise OSError(f"Cannot read JSON file {resolved_path}: {e}") from e

        elif fmt == "yaml":
            try:
                yaml = _import_yaml()
                with open(resolved_path, encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)
                    if config is None:
                        return {}
                    if not isinstance(config, dict):
                        raise ValueError(
                            f"YAML file must contain a dictionary, got {type(config).__name__}"
                        )
                    return config
            except OSError as e:
                raise OSError(f"Cannot read YAML file {resolved_path}: {e}") from e
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid YAML in {resolved_path}: {e}") from e
            except Exception as e:
                # YAML-specific errors (implementation varies)
                if "yaml" in str(e).lower() or "parse" in str(e).lower():
                    raise ValueError(f"Invalid YAML in {resolved_path}: {e}") from e
                raise OSError(f"Cannot read YAML file {resolved_path}: {e}") from e

        elif fmt == "toml":
            try:
                toml_mod = _import_toml_read()
                with open(resolved_path, "rb") as fh:
                    config = toml_mod(fh)
                    if not isinstance(config, dict):
                        raise ValueError(
                            f"TOML file must contain a dictionary, got {type(config).__name__}"
                        )
                    return config
            except OSError as e:
                raise OSError(f"Cannot read TOML file {resolved_path}: {e}") from e
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid TOML in {resolved_path}: {e}") from e
            except Exception as e:
                # TOML-specific errors (implementation varies)
                if "toml" in str(e).lower() or "parse" in str(e).lower():
                    raise ValueError(f"Invalid TOML in {resolved_path}: {e}") from e
                raise OSError(f"Cannot read TOML file {resolved_path}: {e}") from e

        else:
            raise ValueError(f"Cannot infer format from extension: '{resolved_path.suffix}'")

    except Exception as e:
        # Re-raise known exceptions as-is
        if isinstance(e, (FileNotFoundError, ValueError, OSError, ImportError)):
            raise
        # Wrap unexpected errors
        raise RuntimeError(f"Unexpected error loading config from {path}: {e}") from e


# ---------------------------------------------------------------------------
# CLI conversion
# ---------------------------------------------------------------------------


def config_to_cli_args(config: dict[str, Any], prefix: str = "--") -> list[str]:
    """Flatten *config* into a list of CLI arguments.

    Nested dicts use dot-separated keys::

        {"agent": {"lr": 1e-4}} -> ["--agent.lr", "1e-4"]
    """
    args: list[str] = []
    _flatten_to_args(config, prefix, "", args)
    return args


def cli_args_to_config(args: Sequence[str]) -> dict[str, Any]:
    """Parse a list of ``--key value`` CLI arguments into a nested dict.

    Dot-separated keys are expanded::

        ["--agent.lr", "1e-4"] -> {"agent": {"lr": 1e-4}}
    """
    config: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--"):
            key = token.lstrip("-")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = _auto_cast(args[i + 1])
                i += 2
            else:
                value = True  # flag
                i += 1
            _set_nested(config, key.split("."), value)
        else:
            i += 1
    return config


# ---------------------------------------------------------------------------
# Merge / diff
# ---------------------------------------------------------------------------


def merge_configs(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *overrides* into *base* (non-destructive).

    Returns a new dict; neither input is modified.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def diff_configs(
    config1: dict[str, Any],
    config2: dict[str, Any],
    path: str = "",
) -> list[dict[str, Any]]:
    """Compute differences between two configs.

    Returns a list of diffs, each a dict with keys ``"path"``, ``"type"``
    (``"added"``, ``"removed"``, ``"changed"``), and ``"old"`` / ``"new"``
    values where applicable.
    """
    diffs: list[dict[str, Any]] = []
    all_keys = set(config1) | set(config2)

    for key in sorted(all_keys):
        full_path = f"{path}.{key}" if path else key
        in1 = key in config1
        in2 = key in config2

        if in1 and not in2:
            diffs.append({"path": full_path, "type": "removed", "old": config1[key]})
        elif in2 and not in1:
            diffs.append({"path": full_path, "type": "added", "new": config2[key]})
        else:
            v1, v2 = config1[key], config2[key]
            if isinstance(v1, dict) and isinstance(v2, dict):
                diffs.extend(diff_configs(v1, v2, full_path))
            elif v1 != v2:
                diffs.append({"path": full_path, "type": "changed", "old": v1, "new": v2})

    return diffs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_format(path: pathlib.Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit.lower()
    ext = path.suffix.lower()
    mapping = {
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }
    return mapping.get(ext, ext.lstrip("."))


def _resolve_existing_path(path: pathlib.Path) -> pathlib.Path:
    if path.suffix or path.exists():
        return path

    for suffix in (".json", ".yaml", ".yml", ".toml"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    return path


def _normalize_output_path(path: pathlib.Path, fmt: str) -> pathlib.Path:
    if path.suffix:
        return path

    suffix_map = {
        "json": ".json",
        "yaml": ".yaml",
        "toml": ".toml",
    }
    suffix = suffix_map.get(fmt)
    if suffix is None:
        return path
    return path.with_suffix(suffix)


def _import_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for YAML support: pip install pyyaml") from exc


def _import_toml_write() -> Any:
    try:
        import tomli_w as tw  # type: ignore[import-untyped]

        return tw
    except ImportError:
        pass
    try:
        import toml  # type: ignore[import-untyped]

        return toml
    except ImportError as exc:
        raise ImportError(
            "tomli-w or toml is required for TOML writing: pip install tomli-w"
        ) from exc


def _import_toml_read() -> Any:
    """Return a callable that reads TOML from a binary file handle."""
    try:
        import tomllib  # Python 3.11+

        return tomllib.load
    except ImportError:
        pass
    try:
        import tomli  # type: ignore[import-untyped]

        return tomli.load
    except ImportError:
        pass
    try:
        import toml  # type: ignore[import-untyped]

        # toml.load accepts a text file, not binary – wrap it.
        def _load_toml(fh: Any) -> dict[str, Any]:
            return toml.loads(fh.read().decode())

        return _load_toml
    except ImportError as exc:
        raise ImportError(
            "tomllib (3.11+), tomli, or toml is required for TOML reading: pip install tomli"
        ) from exc


def _flatten_to_args(d: dict[str, Any], prefix: str, parent_key: str, out: list[str]) -> None:
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            _flatten_to_args(value, prefix, full_key, out)
        elif isinstance(value, (list, tuple)):
            for item in value:
                out.append(f"{prefix}{full_key}")
                out.append(str(item))
        elif isinstance(value, bool):
            if value:
                out.append(f"{prefix}{full_key}")
        else:
            out.append(f"{prefix}{full_key}")
            out.append(str(value))


def _auto_cast(s: str) -> Any:
    """Attempt to cast a string to int, float, bool, or None."""
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict given a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
