from __future__ import annotations

from collections.abc import Callable
from typing import Any


_BACKENDS: dict[str, Callable[..., Any]] = {}
_HUMAN_CONTROLLERS: dict[str, Callable[..., Any]] = {}
_ROBOT_CONTROLLERS: dict[str, Callable[..., Any]] = {}


def register_backend(name: str, factory: Callable[..., Any]) -> None:
    _BACKENDS[name] = factory


def get_backend(name: str) -> Callable[..., Any]:
    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def register_human_controller(name: str, factory: Callable[..., Any]) -> None:
    _HUMAN_CONTROLLERS[name] = factory


def get_human_controller(name: str) -> Callable[..., Any]:
    if name not in _HUMAN_CONTROLLERS:
        raise KeyError(
            f"Unknown human controller '{name}'. Registered: {sorted(_HUMAN_CONTROLLERS)}"
        )
    return _HUMAN_CONTROLLERS[name]


def register_robot_controller(name: str, factory: Callable[..., Any]) -> None:
    _ROBOT_CONTROLLERS[name] = factory


def get_robot_controller(name: str) -> Callable[..., Any]:
    if name not in _ROBOT_CONTROLLERS:
        raise KeyError(
            f"Unknown robot controller '{name}'. Registered: {sorted(_ROBOT_CONTROLLERS)}"
        )
    return _ROBOT_CONTROLLERS[name]


def registry_snapshot() -> dict[str, list[str]]:
    return {
        "backends": sorted(_BACKENDS),
        "human_controllers": sorted(_HUMAN_CONTROLLERS),
        "robot_controllers": sorted(_ROBOT_CONTROLLERS),
    }
