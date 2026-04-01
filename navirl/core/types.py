from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentKind = Literal["robot", "human"]


@dataclass(slots=True)
class AgentState:
    """State for one agent at one simulation step."""

    agent_id: int
    kind: AgentKind
    x: float
    y: float
    vx: float
    vy: float
    goal_x: float
    goal_y: float
    radius: float
    max_speed: float
    behavior: str = "GO_TO"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    """Controller output for one step."""

    pref_vx: float
    pref_vy: float
    behavior: str = "GO_TO"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventRecord:
    """Discrete event emitted during simulation."""

    step: int
    time_s: float
    event_type: str
    agent_id: int | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Observation:
    """Per-step observation passed to controllers."""

    step: int
    time_s: float
    agents: dict[int, AgentState]


@dataclass(slots=True)
class EpisodeLog:
    """Pointers to run artifacts."""

    run_id: str
    bundle_dir: str
    scenario_path: str
    state_path: str
    events_path: str
    summary_path: str
