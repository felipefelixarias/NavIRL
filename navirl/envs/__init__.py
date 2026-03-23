"""NavIRL Gymnasium RL environments for social robot navigation.

This package provides a comprehensive suite of Gymnasium-compatible environments
for training and evaluating reinforcement learning agents in pedestrian-rich
social navigation scenarios.

Environments
------------
NavEnv
    Base single-robot environment wrapping the NavIRL Grid2D backend with
    configurable observation/action spaces and reward functions.
CrowdNavEnv
    Crowd navigation with tunable density, flow patterns, and social zones.
MultiAgentNavEnv
    Multi-robot environment with PettingZoo-compatible parallel interface,
    communication channels, and cooperative/competitive tasks.

Wrappers
--------
FrameStack, ActionRepeat, TimeLimit, RewardShaping, NormalizeObservation,
NormalizeReward, ClipAction, FlattenObservation, RecordEpisode, MonitorWrapper,
CurriculumWrapper, DomainRandomization, VecEnvWrapper, RelativeObservation,
GoalConditioned

Scenarios
---------
CircleCrossing, RandomGoal, CorridorPassing, DoorwayNegotiation,
IntersectionCrossing, GroupNavigation, DenseRoom, OpenField,
ScenarioDifficultyScaler, ProceduralScenarioGenerator
"""

from navirl.envs.base_env import NavEnv, NavEnvConfig
from navirl.envs.crowd_env import CrowdNavEnv, CrowdNavConfig
from navirl.envs.multi_agent_env import MultiAgentNavEnv, MultiAgentNavConfig
from navirl.envs.scenarios import (
    CircleCrossing,
    CorridorPassing,
    DenseRoom,
    DoorwayNegotiation,
    GroupNavigation,
    IntersectionCrossing,
    OpenField,
    ProceduralScenarioGenerator,
    RandomGoal,
    ScenarioDifficultyScaler,
)
from navirl.envs.wrappers import (
    ActionRepeat,
    ClipAction,
    CurriculumWrapper,
    DomainRandomization,
    FlattenObservation,
    FrameStack,
    GoalConditioned,
    MonitorWrapper,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisode,
    RelativeObservation,
    RewardShaping,
    TimeLimit,
    VecEnvWrapper,
)

__all__ = [
    # Core environments
    "NavEnv",
    "NavEnvConfig",
    "CrowdNavEnv",
    "CrowdNavConfig",
    "MultiAgentNavEnv",
    "MultiAgentNavConfig",
    # Wrappers
    "FrameStack",
    "ActionRepeat",
    "TimeLimit",
    "RewardShaping",
    "NormalizeObservation",
    "NormalizeReward",
    "ClipAction",
    "FlattenObservation",
    "RecordEpisode",
    "MonitorWrapper",
    "CurriculumWrapper",
    "DomainRandomization",
    "VecEnvWrapper",
    "RelativeObservation",
    "GoalConditioned",
    # Scenarios
    "CircleCrossing",
    "RandomGoal",
    "CorridorPassing",
    "DoorwayNegotiation",
    "IntersectionCrossing",
    "GroupNavigation",
    "DenseRoom",
    "OpenField",
    "ScenarioDifficultyScaler",
    "ProceduralScenarioGenerator",
]
