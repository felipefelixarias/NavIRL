"""Configuration validation for NavIRL.

Validates configuration dictionaries against schemas, with helpers to
auto-generate schemas from dataclasses.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields as dc_fields
from typing import Any, get_type_hints

# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------

# A schema is a plain dict describing expected keys:
#   {
#       "key_name": {
#           "type": <python type or tuple of types>,
#           "required": bool,
#           "default": <value>,            # optional
#           "choices": [...],               # optional
#           "min": <number>,                # optional
#           "max": <number>,                # optional
#           "nested": <sub-schema dict>,    # optional
#       },
#       ...
#   }


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------

class ConfigValidator:
    """Validates configuration dicts against schemas.

    Example::

        schema = {
            "learning_rate": {"type": float, "required": True, "min": 0},
            "hidden_sizes": {"type": list, "required": True},
        }
        ok, errors = ConfigValidator.validate(cfg, schema)
    """

    @staticmethod
    def validate(
        config: dict[str, Any],
        schema: dict[str, dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Validate *config* against *schema*.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_valid, errors)`` where *errors* is empty when valid.
        """
        errors: list[str] = []

        # Check required keys.
        for key, spec in schema.items():
            if spec.get("required", False) and key not in config:
                errors.append(f"Missing required key: '{key}'")

        for key, value in config.items():
            if key not in schema:
                # Unknown keys are allowed but noted.
                continue

            spec = schema[key]

            # Type check.
            expected_type = spec.get("type")
            if expected_type is not None and not isinstance(value, expected_type):
                errors.append(
                    f"Key '{key}': expected type {expected_type}, "
                    f"got {type(value).__name__}"
                )
                continue  # skip further checks on wrong type

            # Choices.
            choices = spec.get("choices")
            if choices is not None and value not in choices:
                errors.append(
                    f"Key '{key}': value {value!r} not in {choices}"
                )

            # Numeric bounds.
            if "min" in spec and value < spec["min"]:
                errors.append(
                    f"Key '{key}': value {value} < minimum {spec['min']}"
                )
            if "max" in spec and value > spec["max"]:
                errors.append(
                    f"Key '{key}': value {value} > maximum {spec['max']}"
                )

            # Nested schema.
            nested = spec.get("nested")
            if nested is not None and isinstance(value, dict):
                _, sub_errors = ConfigValidator.validate(value, nested)
                for e in sub_errors:
                    errors.append(f"Key '{key}' -> {e}")

        return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# SchemaBuilder
# ---------------------------------------------------------------------------

class SchemaBuilder:
    """Build validation schemas from dataclasses."""

    _TYPE_MAP: dict[type, type | tuple[type, ...]] = {
        int: (int,),
        float: (int, float),
        str: (str,),
        bool: (bool,),
        list: (list,),
        dict: (dict,),
    }

    @classmethod
    def from_dataclass(cls, dc_cls: type[Any]) -> dict[str, dict[str, Any]]:
        """Generate a validation schema from a dataclass.

        Parameters
        ----------
        dc_cls : type
            A dataclass type.

        Returns
        -------
        dict
            Schema dict suitable for :meth:`ConfigValidator.validate`.
        """
        if not dataclasses.is_dataclass(dc_cls):
            raise TypeError(f"{dc_cls} is not a dataclass")

        schema: dict[str, dict[str, Any]] = {}
        hints = get_type_hints(dc_cls)

        for f in dc_fields(dc_cls):
            entry: dict[str, Any] = {}

            # Determine expected type(s).
            origin = getattr(hints.get(f.name), "__origin__", None)
            raw_type = hints.get(f.name, Any)
            if origin is not None:
                # e.g. list[int] -> list
                mapped = cls._TYPE_MAP.get(origin)
                if mapped is not None:
                    entry["type"] = mapped if len(mapped) > 1 else mapped[0]
            elif raw_type in cls._TYPE_MAP:
                mapped = cls._TYPE_MAP[raw_type]
                entry["type"] = mapped if len(mapped) > 1 else mapped[0]

            # Required if no default.
            has_default = (
                f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            )
            entry["required"] = not has_default

            if f.default is not dataclasses.MISSING:
                entry["default"] = f.default

            schema[f.name] = entry

        return schema


# ---------------------------------------------------------------------------
# Domain-specific validators
# ---------------------------------------------------------------------------

_AGENT_SCHEMA: dict[str, dict[str, Any]] = {
    "hidden_sizes": {"type": list, "required": True},
    "learning_rate": {"type": (int, float), "required": True, "min": 0},
    "batch_size": {"type": int, "required": False, "min": 1},
    "gamma": {"type": (int, float), "required": False, "min": 0, "max": 1},
    "tau": {"type": (int, float), "required": False, "min": 0, "max": 1},
}

_ENV_SCHEMA: dict[str, dict[str, Any]] = {
    "num_humans": {"type": int, "required": False, "min": 0},
    "env_size": {"type": (int, float), "required": False, "min": 0},
    "time_limit": {"type": (int, float), "required": False, "min": 0},
}

_TRAINING_SCHEMA: dict[str, dict[str, Any]] = {
    "total_steps": {"type": int, "required": True, "min": 1},
    "eval_interval": {"type": int, "required": False, "min": 1},
    "log_interval": {"type": int, "required": False, "min": 1},
    "seed": {"type": int, "required": False},
}


def validate_agent_config(config: dict[str, Any]) -> list[str]:
    """Validate an agent configuration dict and return errors."""
    _, errors = ConfigValidator.validate(config, _AGENT_SCHEMA)
    return errors


def validate_env_config(config: dict[str, Any]) -> list[str]:
    """Validate an environment configuration dict and return errors."""
    _, errors = ConfigValidator.validate(config, _ENV_SCHEMA)
    return errors


def validate_training_config(config: dict[str, Any]) -> list[str]:
    """Validate a training configuration dict and return errors."""
    _, errors = ConfigValidator.validate(config, _TRAINING_SCHEMA)
    return errors
