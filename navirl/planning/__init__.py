"""Planning exports for the currently implemented planner set."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "Path": ("navirl.planning.base", "Path"),
    "Planner": ("navirl.planning.base", "Planner"),
    "PlannerConfig": ("navirl.planning.base", "PlannerConfig"),
    "AStarPlanner": ("navirl.planning.global_planners", "AStarPlanner"),
    "DijkstraPlanner": ("navirl.planning.global_planners", "DijkstraPlanner"),
    "PRMPlanner": ("navirl.planning.global_planners", "PRMPlanner"),
    "RRTPlanner": ("navirl.planning.global_planners", "RRTPlanner"),
    "RRTStarPlanner": ("navirl.planning.global_planners", "RRTStarPlanner"),
    "ThetaStarPlanner": ("navirl.planning.global_planners", "ThetaStarPlanner"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
