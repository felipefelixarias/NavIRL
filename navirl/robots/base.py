from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from navirl.core.types import Action, AgentState

EventSink = Callable[[str, int | None, dict], None]


class RobotController(ABC):
    """Abstract base class for robot navigation controllers.

    Robot controllers implement navigation algorithms that compute actions
    for mobile robots given their current state and goals. Controllers must
    handle multi-agent scenarios and work with different simulation backends.

    All controllers follow a reset/step pattern where reset() initializes
    the controller for a specific robot and episode, and step() computes
    the next action based on current observations.
    """

    @abstractmethod
    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend,
    ) -> None:
        """Initialize controller for a new episode.

        Args:
            robot_id: Unique identifier for this robot
            start: Initial position (x, y) in world coordinates
            goal: Target position (x, y) to navigate towards
            backend: Simulation backend providing environment interface
        """
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
        """Compute next action for this robot.

        Args:
            step: Current simulation step number
            time_s: Current simulation time in seconds
            dt: Time step duration in seconds
            states: Current states of all agents in the environment
            emit_event: Callback for emitting simulation events

        Returns:
            Action: Computed action (typically velocity command) for this robot
        """
        raise NotImplementedError
