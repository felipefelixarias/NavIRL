"""Lazy exports for optional Gymnasium environment components."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "NavEnv": ("navirl.envs.base_env", "NavEnv"),
    "NavEnvConfig": ("navirl.envs.base_env", "NavEnvConfig"),
    "CrowdNavEnv": ("navirl.envs.crowd_env", "CrowdNavEnv"),
    "CrowdNavConfig": ("navirl.envs.crowd_env", "CrowdNavConfig"),
    "MultiAgentNavEnv": ("navirl.envs.multi_agent_env", "MultiAgentNavEnv"),
    "MultiAgentNavConfig": ("navirl.envs.multi_agent_env", "MultiAgentNavConfig"),
    "ActionRepeat": ("navirl.envs.wrappers", "ActionRepeat"),
    "ClipAction": ("navirl.envs.wrappers", "ClipAction"),
    "CurriculumWrapper": ("navirl.envs.wrappers", "CurriculumWrapper"),
    "DomainRandomization": ("navirl.envs.wrappers", "DomainRandomization"),
    "FlattenObservation": ("navirl.envs.wrappers", "FlattenObservation"),
    "FrameStack": ("navirl.envs.wrappers", "FrameStack"),
    "GoalConditioned": ("navirl.envs.wrappers", "GoalConditioned"),
    "MonitorWrapper": ("navirl.envs.wrappers", "MonitorWrapper"),
    "NormalizeObservation": ("navirl.envs.wrappers", "NormalizeObservation"),
    "NormalizeReward": ("navirl.envs.wrappers", "NormalizeReward"),
    "RecordEpisode": ("navirl.envs.wrappers", "RecordEpisode"),
    "RelativeObservation": ("navirl.envs.wrappers", "RelativeObservation"),
    "RewardShaping": ("navirl.envs.wrappers", "RewardShaping"),
    "TimeLimit": ("navirl.envs.wrappers", "TimeLimit"),
    "VecEnvWrapper": ("navirl.envs.wrappers", "VecEnvWrapper"),
    "CircleCrossing": ("navirl.envs.scenarios", "CircleCrossing"),
    "CorridorPassing": ("navirl.envs.scenarios", "CorridorPassing"),
    "DenseRoom": ("navirl.envs.scenarios", "DenseRoom"),
    "DoorwayNegotiation": ("navirl.envs.scenarios", "DoorwayNegotiation"),
    "GroupNavigation": ("navirl.envs.scenarios", "GroupNavigation"),
    "IntersectionCrossing": ("navirl.envs.scenarios", "IntersectionCrossing"),
    "OpenField": ("navirl.envs.scenarios", "OpenField"),
    "ProceduralScenarioGenerator": ("navirl.envs.scenarios", "ProceduralScenarioGenerator"),
    "RandomGoal": ("navirl.envs.scenarios", "RandomGoal"),
    "ScenarioDifficultyScaler": ("navirl.envs.scenarios", "ScenarioDifficultyScaler"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{name} requires optional environment dependencies. "
            "Install gymnasium to use NavIRL RL environments and wrappers."
        ) from exc
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
