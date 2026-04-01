from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from navirl.core.types import Action, AgentState

EventSink = Callable[[str, int | None, dict], None]


class HumanController(ABC):
    """Interface for human behavior controllers."""

    @abstractmethod
    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        raise NotImplementedError
