from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from navirl.core.types import Action, AgentState


EventSink = Callable[[str, int | None, dict], None]


class RobotController(ABC):
    """Interface for robot navigation controllers."""

    @abstractmethod
    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        raise NotImplementedError
