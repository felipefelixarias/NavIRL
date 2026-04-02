from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

from navirl.core.plugin_validation import ConfigValidationError, validate_controller_config
from navirl.core.types import Action, AgentState

logger = logging.getLogger(__name__)
EventSink = Callable[[str, int | None, dict], None]


class RobotController(ABC):
    """Abstract base class for robot navigation controllers.

    Robot controllers implement navigation algorithms that compute actions
    for mobile robots given their current state and goals. Controllers must
    handle multi-agent scenarios and work with different simulation backends.

    All controllers follow a reset/step pattern where reset() initializes
    the controller for a specific robot and episode, and step() computes
    the next action based on current observations.

    For external plugin developers:
    - Implement both reset() and step() methods
    - Ensure step() returns valid Action objects with bounded velocities
    - Handle edge cases gracefully (e.g., unreachable goals, invalid states)
    - Configuration should be validated in __init__
    - Consider computational efficiency - step() is called frequently

    Security and robustness guidelines:
    - Validate all inputs in public methods
    - Use reasonable defaults when inputs are invalid
    - Avoid unbounded loops or recursive calls
    - Limit memory allocation and computational complexity
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

        # Track performance metrics for monitoring
        self._step_count = 0
        self._last_computation_time = 0.0
        self._max_computation_time = 0.1  # 100ms warning threshold

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
            robot_id: Unique identifier for this robot (must be positive)
            start: Initial position (x, y) in world coordinates
            goal: Target position (x, y) to navigate towards
            backend: Simulation backend providing environment interface

        Raises:
            ValueError: If input data is invalid
        """
        # Enhanced input validation that subclasses can rely on
        if not isinstance(robot_id, int) or robot_id < 0:
            raise ValueError(f"robot_id must be a non-negative integer, got {robot_id}")

        if not (isinstance(start, (list, tuple)) and len(start) >= 2):
            raise ValueError(f"Invalid start position: {start} (must be (x, y) tuple)")

        if not (isinstance(goal, (list, tuple)) and len(goal) >= 2):
            raise ValueError(f"Invalid goal position: {goal} (must be (x, y) tuple)")

        # Validate position coordinates are finite numbers
        try:
            start_x, start_y = float(start[0]), float(start[1])
            goal_x, goal_y = float(goal[0]), float(goal[1])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Position coordinates must be numeric: {e}") from e

        # Check for NaN or infinite values
        if not all(abs(val) < 1e6 for val in [start_x, start_y, goal_x, goal_y]):
            raise ValueError("Position coordinates must be finite and reasonable")

        self._step_count = 0

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
            step: Current simulation step number (must be non-negative)
            time_s: Current simulation time in seconds (must be non-negative)
            dt: Time step duration in seconds (must be positive)
            states: Current states of all agents in the environment
            emit_event: Callback for emitting simulation events

        Returns:
            Action: Computed action (typically velocity command) for this robot

        Raises:
            ValueError: If input parameters are invalid
        """
        # Enhanced input validation that subclasses can rely on
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"Step must be a non-negative integer, got {step}")

        if not isinstance(time_s, (int, float)) or time_s < 0:
            raise ValueError(f"Time must be non-negative, got {time_s}")

        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")

        # Reasonable dt bounds (1µs to 10s)
        if dt < 1e-6 or dt > 10.0:
            raise ValueError(f"Time step out of reasonable bounds: {dt}")

        if not isinstance(states, dict):
            raise ValueError(f"States must be a dictionary, got {type(states)}")

        if not callable(emit_event):
            raise ValueError(f"emit_event must be callable, got {type(emit_event)}")

        # Track performance
        self._step_count += 1

    def validate_action(self, action: Action) -> Action:
        """
        Validate and potentially clamp an action to safe bounds.

        Args:
            action: Action to validate

        Returns:
            Validated (potentially modified) action
        """
        if not isinstance(action, Action):
            logger.warning("Invalid action type: %s, using STOP action", type(action))
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")

        # Clamp velocities to reasonable bounds
        max_vel = self.cfg.get('max_speed', 5.0)  # Default 5 m/s
        min_vel = -max_vel

        # Check for NaN or infinite values
        if not (abs(action.pref_vx) < 1e6 and abs(action.pref_vy) < 1e6):
            logger.warning("Invalid velocity values detected, using STOP action")
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="STOP")

        # Clamp velocities
        clamped_vx = max(min_vel, min(max_vel, action.pref_vx))
        clamped_vy = max(min_vel, min(max_vel, action.pref_vy))

        if clamped_vx != action.pref_vx or clamped_vy != action.pref_vy:
            logger.info(
                "Clamped velocity from (%.2f, %.2f) to (%.2f, %.2f)",
                action.pref_vx, action.pref_vy, clamped_vx, clamped_vy
            )

        return Action(
            pref_vx=clamped_vx,
            pref_vy=clamped_vy,
            behavior=action.behavior if hasattr(action, 'behavior') else "NORMAL"
        )

    def check_computational_performance(self, computation_time: float) -> None:
        """
        Monitor computational performance and warn about slow controllers.

        Args:
            computation_time: Time taken for the last computation in seconds
        """
        self._last_computation_time = computation_time

        if computation_time > self._max_computation_time:
            logger.warning(
                "Controller %s step() took %.3f seconds (step %d) - "
                "consider optimizing for real-time performance",
                self.__class__.__name__, computation_time, self._step_count
            )

        # Alert if consistently slow (last 10 steps average)
        if self._step_count % 10 == 0 and self._step_count > 10:
            if self._last_computation_time > self._max_computation_time * 0.8:
                logger.info(
                    "Controller %s consistently approaching performance limit",
                    self.__class__.__name__
                )
