from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from navirl.core.plugin_validation import ConfigValidationError, validate_controller_config
from navirl.core.types import Action, AgentState

logger = logging.getLogger(__name__)
EventSink = Callable[[str, int | None, dict], None]


class HumanController(ABC):
    """
    Interface for human behavior controllers.

    This abstract base class defines the contract that all human controllers
    must implement. Controllers are responsible for generating actions for
    human agents in the simulation based on their current state and goals.

    For external plugin developers:
    - Implement both reset() and step() methods
    - Ensure step() returns valid Action objects for all provided agent IDs
    - Handle edge cases gracefully (e.g., unreachable goals, invalid states)
    - Configuration should be validated in __init__
    """

    def __init__(self, cfg: dict | None = None, **kwargs):
        """
        Initialize the controller with validated configuration.

        Args:
            cfg: Configuration dictionary (will be validated)
            **kwargs: Additional keyword arguments

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            self.cfg = validate_controller_config(cfg, self.__class__.__name__)
        except ConfigValidationError:
            logger.error("Invalid configuration for %s", self.__class__.__name__)
            raise

    @abstractmethod
    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        """
        Reset the controller for a new episode.

        Args:
            human_ids: List of human agent IDs to control
            starts: Starting positions for each human agent
            goals: Goal positions for each human agent
            backend: Optional backend for path planning

        Raises:
            ValueError: If input data is invalid
        """
        # Input validation that subclasses can rely on
        if not isinstance(human_ids, list):
            raise ValueError(f"human_ids must be a list, got {type(human_ids)}")

        if not all(isinstance(hid, int) for hid in human_ids):
            raise ValueError("All human_ids must be integers")

        for hid in human_ids:
            if hid not in starts:
                raise ValueError(f"Missing start position for human {hid}")
            if hid not in goals:
                raise ValueError(f"Missing goal position for human {hid}")

            start = starts[hid]
            goal = goals[hid]

            if not (isinstance(start, (list, tuple)) and len(start) >= 2):
                raise ValueError(f"Invalid start position for human {hid}: {start}")
            if not (isinstance(goal, (list, tuple)) and len(goal) >= 2):
                raise ValueError(f"Invalid goal position for human {hid}: {goal}")

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
        """
        Compute actions for controlled human agents.

        Args:
            step: Current simulation step
            time_s: Current simulation time in seconds
            dt: Time step size in seconds
            states: Current states of all agents
            robot_id: ID of the robot agent
            emit_event: Function to emit events for logging/debugging

        Returns:
            Dictionary mapping human agent IDs to their actions

        Raises:
            ValueError: If input parameters are invalid
        """
        # Input validation that subclasses can rely on
        if step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")
        if time_s < 0:
            raise ValueError(f"Time must be non-negative, got {time_s}")
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
        if not isinstance(states, dict):
            raise ValueError(f"States must be a dictionary, got {type(states)}")
        if not callable(emit_event):
            raise ValueError(f"emit_event must be callable, got {type(emit_event)}")

    def validate_action(self, agent_id: int, action: Action) -> Action:
        """
        Validate and potentially clamp an action to safe bounds.

        Args:
            agent_id: ID of the agent
            action: Action to validate

        Returns:
            Validated (potentially modified) action
        """
        if not isinstance(action, Action):
            logger.warning("Invalid action type for agent %s: %s", agent_id, type(action))
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")

        # Clamp velocities to reasonable bounds
        max_vel = 5.0  # Reasonable maximum velocity (m/s)
        if abs(action.pref_vx) > max_vel:
            logger.warning("Clamping excessive velocity for agent %s", agent_id)
            action.pref_vx = max_vel if action.pref_vx > 0 else -max_vel

        if abs(action.pref_vy) > max_vel:
            logger.warning("Clamping excessive velocity for agent %s", agent_id)
            action.pref_vy = max_vel if action.pref_vy > 0 else -max_vel

        return action
