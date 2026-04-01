"""Multi-robot fleet coordination.

Provides fleet management, formation control (line, circle, V-shape, and
custom formations), inter-robot collision avoidance via velocity obstacles,
greedy task assignment, fleet-level planning utilities, and a simple
communication model with range limits and message loss.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController

# -----------------------------------------------------------------------
# Formation definitions
# -----------------------------------------------------------------------

class FormationType(enum.Enum):
    """Built-in formation types."""
    LINE = "line"
    CIRCLE = "circle"
    V_SHAPE = "v_shape"
    WEDGE = "wedge"
    CUSTOM = "custom"


@dataclass
class FormationConfig:
    """Configuration for a fleet formation.

    Attributes:
        formation_type: Type of formation.
        spacing: Desired inter-robot distance (metres).
        v_angle: Half-angle for V-shape formations (rad).
        custom_offsets: Explicit offsets for CUSTOM formation, each
            relative to the formation centroid, shape ``(N, 2)``.
        heading_align: If ``True`` all robots share the leader heading.
    """

    formation_type: FormationType = FormationType.LINE
    spacing: float = 1.5
    v_angle: float = np.pi / 6.0
    custom_offsets: np.ndarray | None = None
    heading_align: bool = True


def compute_formation_offsets(
    n_robots: int,
    config: FormationConfig,
) -> np.ndarray:
    """Compute formation offsets relative to the centroid.

    Args:
        n_robots: Number of robots in the fleet.
        config: Formation configuration.

    Returns:
        Offsets, shape ``(n_robots, 2)``.
    """
    if config.formation_type == FormationType.CUSTOM and config.custom_offsets is not None:
        offsets = config.custom_offsets[:n_robots].copy()
        if len(offsets) < n_robots:
            pad = np.zeros((n_robots - len(offsets), 2))
            offsets = np.vstack([offsets, pad])
        return offsets

    offsets = np.zeros((n_robots, 2))

    if config.formation_type == FormationType.LINE:
        total = (n_robots - 1) * config.spacing
        for i in range(n_robots):
            offsets[i, 0] = 0.0
            offsets[i, 1] = -total / 2.0 + i * config.spacing

    elif config.formation_type == FormationType.CIRCLE:
        if n_robots == 1:
            return offsets
        radius = config.spacing / (2.0 * np.sin(np.pi / n_robots))
        for i in range(n_robots):
            angle = 2.0 * np.pi * i / n_robots
            offsets[i, 0] = radius * np.cos(angle)
            offsets[i, 1] = radius * np.sin(angle)

    elif config.formation_type in (FormationType.V_SHAPE, FormationType.WEDGE):
        half_angle = config.v_angle
        for i in range(n_robots):
            if i == 0:
                offsets[i] = [0.0, 0.0]
            else:
                side = 1 if i % 2 == 1 else -1
                rank = (i + 1) // 2
                offsets[i, 0] = -rank * config.spacing * np.cos(half_angle)
                offsets[i, 1] = side * rank * config.spacing * np.sin(half_angle)

    return offsets


def rotate_offsets(
    offsets: np.ndarray,
    heading: float,
) -> np.ndarray:
    """Rotate formation offsets by *heading* (rad).

    Args:
        offsets: Shape ``(N, 2)``.
        heading: Rotation angle.

    Returns:
        Rotated offsets, shape ``(N, 2)``.
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (rot @ offsets.T).T


# -----------------------------------------------------------------------
# Collision avoidance (velocity obstacle inspired)
# -----------------------------------------------------------------------

def fleet_collision_avoidance(
    positions: np.ndarray,
    velocities: np.ndarray,
    desired_velocities: np.ndarray,
    radii: np.ndarray,
    dt: float,
    safety_margin: float = 0.3,
) -> np.ndarray:
    """Adjust desired velocities to avoid inter-robot collisions.

    Uses a simple reciprocal velocity scaling approach: for each pair of
    robots on a collision course, both reduce their speed component
    along the line connecting them.

    Args:
        positions: Shape ``(N, 2)``.
        velocities: Current velocities, shape ``(N, 2)``.
        desired_velocities: Desired velocities, shape ``(N, 2)``.
        radii: Robot radii, shape ``(N,)``.
        dt: Time step.
        safety_margin: Extra clearance (metres).

    Returns:
        Adjusted desired velocities, shape ``(N, 2)``.
    """
    n = positions.shape[0]
    adjusted = desired_velocities.copy()

    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[j] - positions[i]
            dist = float(np.linalg.norm(diff))
            min_dist = radii[i] + radii[j] + safety_margin
            if dist < 1e-8:
                # Push apart.
                push = np.array([1.0, 0.0]) * min_dist * 0.5 / max(dt, 1e-6)
                adjusted[i] -= push
                adjusted[j] += push
                continue

            direction = diff / dist
            if dist < min_dist:
                # Already overlapping: push apart.
                overlap = min_dist - dist
                repulse = direction * overlap / (2.0 * max(dt, 1e-6))
                adjusted[i] -= repulse
                adjusted[j] += repulse
                continue

            # Predict future distance.
            rel_vel = adjusted[i] - adjusted[j]
            closing_speed = float(np.dot(rel_vel, direction))
            if closing_speed <= 0.0:
                continue  # Separating.

            time_to_collision = (dist - min_dist) / closing_speed
            if time_to_collision < dt * 3.0:
                # Scale down closing component.
                scale = max(0.0, time_to_collision / (dt * 3.0))
                correction = direction * closing_speed * (1.0 - scale) * 0.5
                adjusted[i] -= correction
                adjusted[j] += correction

    return adjusted


# -----------------------------------------------------------------------
# Task assignment
# -----------------------------------------------------------------------

def greedy_task_assignment(
    robot_positions: np.ndarray,
    task_positions: np.ndarray,
) -> list[int]:
    """Assign tasks to robots using greedy nearest-neighbour.

    Each robot is assigned the closest un-assigned task.

    Args:
        robot_positions: Shape ``(N, 2)``.
        task_positions: Shape ``(M, 2)``.

    Returns:
        List of task indices, one per robot.  ``-1`` means no task
        assigned.
    """
    n_robots = robot_positions.shape[0]
    n_tasks = task_positions.shape[0]
    assigned: list[int] = [-1] * n_robots
    used_tasks: set[int] = set()

    if n_tasks == 0:
        return assigned

    # Cost matrix.
    cost = np.linalg.norm(
        robot_positions[:, None, :] - task_positions[None, :, :], axis=2
    )  # (N, M)

    for _ in range(min(n_robots, n_tasks)):
        # Mask used tasks.
        mask = np.full(cost.shape, np.inf)
        for r in range(n_robots):
            if assigned[r] != -1:
                continue
            for t in range(n_tasks):
                if t not in used_tasks:
                    mask[r, t] = cost[r, t]

        idx = int(np.argmin(mask))
        ri, ti = divmod(idx, n_tasks)
        if mask[ri, ti] == np.inf:
            break
        assigned[ri] = ti
        used_tasks.add(ti)

    return assigned


def hungarian_task_assignment(
    robot_positions: np.ndarray,
    task_positions: np.ndarray,
) -> list[int]:
    """Assign tasks using a simplified auction-based approach.

    Falls back to greedy when the number of robots or tasks is small.

    Args:
        robot_positions: Shape ``(N, 2)``.
        task_positions: Shape ``(M, 2)``.

    Returns:
        List of task indices per robot (``-1`` if un-assigned).
    """
    n_robots = robot_positions.shape[0]
    n_tasks = task_positions.shape[0]
    if n_robots == 0 or n_tasks == 0:
        return [-1] * n_robots

    cost = np.linalg.norm(
        robot_positions[:, None, :] - task_positions[None, :, :], axis=2
    )
    assigned: list[int] = [-1] * n_robots
    used: set[int] = set()

    # Iterative auction.
    prices = np.zeros(n_tasks)
    epsilon = 1e-3

    for _ in range(n_robots * 5):
        updated = False
        for r in range(n_robots):
            values = -(cost[r] + prices)
            sorted_idx = np.argsort(values)[::-1]
            for ti in sorted_idx:
                ti = int(ti)
                if ti in used and assigned[r] != ti:
                    continue
                bid = -values[ti] + epsilon
                # Check if outbids current holder.
                current_holder = None
                for r2 in range(n_robots):
                    if assigned[r2] == ti and r2 != r:
                        current_holder = r2
                        break
                if current_holder is not None:
                    if cost[r, ti] < cost[current_holder, ti]:
                        assigned[current_holder] = -1
                        used.discard(ti)
                    else:
                        continue
                assigned[r] = ti
                used.add(ti)
                prices[ti] = bid
                updated = True
                break

        if not updated:
            break

    return assigned


# -----------------------------------------------------------------------
# Communication model
# -----------------------------------------------------------------------

@dataclass
class Message:
    """A message between fleet members.

    Attributes:
        sender_id: Sender robot ID.
        receiver_id: Receiver robot ID (``-1`` = broadcast).
        payload: Arbitrary data dictionary.
        timestamp: Simulation time of send.
    """

    sender_id: int
    receiver_id: int
    payload: dict[str, Any]
    timestamp: float = 0.0


class CommunicationModel:
    """Simple range-limited communication with optional packet loss.

    Attributes:
        max_range: Maximum communication range (metres).
        loss_probability: Probability of a message being dropped.
    """

    def __init__(
        self,
        max_range: float = 20.0,
        loss_probability: float = 0.0,
    ) -> None:
        self.max_range = max_range
        self.loss_probability = loss_probability
        self._inbox: dict[int, list[Message]] = {}

    def send(
        self,
        msg: Message,
        positions: dict[int, np.ndarray],
    ) -> bool:
        """Attempt to send a message.

        Delivery succeeds only if the receiver is within range and the
        message is not dropped by the loss model.

        Args:
            msg: The message to send.
            positions: Map of robot ID to ``(2,)`` position.

        Returns:
            ``True`` if the message was delivered.
        """
        if np.random.rand() < self.loss_probability:
            return False

        sender_pos = positions.get(msg.sender_id)
        if sender_pos is None:
            return False

        if msg.receiver_id == -1:
            # Broadcast.
            delivered = False
            for rid, rpos in positions.items():
                if rid == msg.sender_id:
                    continue
                dist = float(np.linalg.norm(rpos - sender_pos))
                if dist <= self.max_range:
                    self._inbox.setdefault(rid, []).append(msg)
                    delivered = True
            return delivered

        receiver_pos = positions.get(msg.receiver_id)
        if receiver_pos is None:
            return False
        dist = float(np.linalg.norm(receiver_pos - sender_pos))
        if dist > self.max_range:
            return False
        self._inbox.setdefault(msg.receiver_id, []).append(msg)
        return True

    def receive(self, robot_id: int) -> list[Message]:
        """Retrieve and clear all messages for *robot_id*."""
        msgs = self._inbox.pop(robot_id, [])
        return msgs

    def clear(self) -> None:
        """Clear all pending messages."""
        self._inbox.clear()


# -----------------------------------------------------------------------
# Fleet planner
# -----------------------------------------------------------------------

class FleetPlanner:
    """Coordinate fleet-level goals and path planning.

    Wraps task assignment, formation computation, and collision
    avoidance into a single planning step.

    Attributes:
        formation_config: Formation parameters.
        safety_margin: Extra clearance for collision avoidance.
    """

    def __init__(
        self,
        formation_config: FormationConfig | None = None,
        safety_margin: float = 0.3,
    ) -> None:
        self.formation_config = formation_config or FormationConfig()
        self.safety_margin = safety_margin

    def assign_tasks(
        self,
        robot_positions: np.ndarray,
        task_positions: np.ndarray,
        method: str = "greedy",
    ) -> list[int]:
        """Assign tasks to robots.

        Args:
            robot_positions: Shape ``(N, 2)``.
            task_positions: Shape ``(M, 2)``.
            method: ``"greedy"`` or ``"auction"``.

        Returns:
            Task index per robot.
        """
        if method == "auction":
            return hungarian_task_assignment(robot_positions, task_positions)
        return greedy_task_assignment(robot_positions, task_positions)

    def compute_formation_targets(
        self,
        centroid: np.ndarray,
        heading: float,
        n_robots: int,
    ) -> np.ndarray:
        """Compute world-frame formation target positions.

        Args:
            centroid: Formation centre, ``(2,)``.
            heading: Formation heading (rad).
            n_robots: Number of robots.

        Returns:
            Target positions, shape ``(N, 2)``.
        """
        offsets = compute_formation_offsets(n_robots, self.formation_config)
        rotated = rotate_offsets(offsets, heading)
        return rotated + centroid

    def avoid_collisions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        desired_velocities: np.ndarray,
        radii: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Run fleet collision avoidance.

        Returns adjusted desired velocities.
        """
        return fleet_collision_avoidance(
            positions, velocities, desired_velocities, radii, dt,
            safety_margin=self.safety_margin,
        )


# -----------------------------------------------------------------------
# RobotFleet controller
# -----------------------------------------------------------------------

class RobotFleet:
    """Manage a fleet of robots with formation control and coordination.

    The fleet wraps individual :class:`RobotController` instances,
    overlaying formation targets, collision avoidance, and communication.

    Args:
        controllers: Dict mapping robot ID to its controller.
        formation_config: Formation specification.
        comm_range: Communication range (metres).
        comm_loss: Communication loss probability.
        safety_margin: Inter-robot safety margin.
    """

    def __init__(
        self,
        controllers: dict[int, RobotController] | None = None,
        formation_config: FormationConfig | None = None,
        comm_range: float = 20.0,
        comm_loss: float = 0.0,
        safety_margin: float = 0.3,
    ) -> None:
        self._controllers: dict[int, RobotController] = controllers or {}
        self._planner = FleetPlanner(formation_config, safety_margin)
        self._comm = CommunicationModel(comm_range, comm_loss)
        self._formation_config = formation_config or FormationConfig()
        self._robot_ids: list[int] = sorted(self._controllers.keys())
        self._goals: dict[int, tuple[float, float]] = {}
        self._radii: dict[int, float] = {}
        self._leader_id: int | None = None

    # ----- Fleet management ---------------------------------------------

    def add_robot(self, robot_id: int, controller: RobotController) -> None:
        """Add a robot to the fleet.

        Args:
            robot_id: Unique identifier.
            controller: The robot's controller.
        """
        self._controllers[robot_id] = controller
        self._robot_ids = sorted(self._controllers.keys())

    def remove_robot(self, robot_id: int) -> None:
        """Remove a robot from the fleet."""
        self._controllers.pop(robot_id, None)
        self._goals.pop(robot_id, None)
        self._radii.pop(robot_id, None)
        self._robot_ids = sorted(self._controllers.keys())

    @property
    def size(self) -> int:
        """Number of robots in the fleet."""
        return len(self._controllers)

    @property
    def robot_ids(self) -> list[int]:
        """Sorted list of robot IDs."""
        return list(self._robot_ids)

    def set_leader(self, robot_id: int) -> None:
        """Designate a fleet leader for formation heading."""
        self._leader_id = robot_id

    # ----- Coordination -------------------------------------------------

    def reset(
        self,
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend: Any,
    ) -> None:
        """Reset all fleet members for a new episode.

        Args:
            starts: Start positions keyed by robot ID.
            goals: Goal positions keyed by robot ID.
            backend: Simulation backend reference.
        """
        self._goals = dict(goals)
        for rid, ctrl in self._controllers.items():
            s = starts.get(rid, (0.0, 0.0))
            g = goals.get(rid, s)
            ctrl.reset(rid, s, g, backend)
        self._comm.clear()

    def step(
        self,
        step_num: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> dict[int, Action]:
        """Step all fleet members with coordination.

        Args:
            step_num: Simulation step number.
            time_s: Simulation time (seconds).
            dt: Time step.
            states: All agent states.
            emit_event: Event sink callback.

        Returns:
            Dict mapping robot ID to its action.
        """
        n = self.size
        if n == 0:
            return {}

        # Collect positions, velocities, radii.
        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))
        radii = np.zeros(n)
        id_to_idx: dict[int, int] = {}

        for idx, rid in enumerate(self._robot_ids):
            id_to_idx[rid] = idx
            if rid in states:
                st = states[rid]
                positions[idx] = [st.x, st.y]
                velocities[idx] = [st.vx, st.vy]
                radii[idx] = st.radius
                self._radii[rid] = st.radius

        # Individual controller actions.
        raw_actions: dict[int, Action] = {}
        for rid, ctrl in self._controllers.items():
            raw_actions[rid] = ctrl.step(step_num, time_s, dt, states, emit_event)

        # Build desired velocity matrix.
        desired = np.zeros((n, 2))
        for rid, act in raw_actions.items():
            idx = id_to_idx[rid]
            desired[idx] = [act.pref_vx, act.pref_vy]

        # Formation overlay (optional).
        if self._formation_config.formation_type != FormationType.CUSTOM or n > 1:
            centroid = np.mean(positions, axis=0)
            # Leader heading.
            if self._leader_id is not None and self._leader_id in states:
                lst = states[self._leader_id]
                heading = float(np.arctan2(lst.vy, lst.vx)) if (lst.vx ** 2 + lst.vy ** 2) > 1e-4 else 0.0
            else:
                mean_vel = np.mean(velocities, axis=0)
                heading = float(np.arctan2(mean_vel[1], mean_vel[0]))

            targets = self._planner.compute_formation_targets(centroid, heading, n)
            formation_gain = 0.3
            for idx in range(n):
                error = targets[idx] - positions[idx]
                desired[idx] += formation_gain * error

        # Collision avoidance.
        adjusted = self._planner.avoid_collisions(
            positions, velocities, desired, radii, dt
        )

        # Rebuild actions.
        final_actions: dict[int, Action] = {}
        for rid in self._robot_ids:
            idx = id_to_idx[rid]
            final_actions[rid] = Action(
                pref_vx=float(adjusted[idx, 0]),
                pref_vy=float(adjusted[idx, 1]),
                behavior=raw_actions.get(rid, Action(0.0, 0.0)).behavior,
            )

        emit_event("fleet_step", None, {"n_robots": n, "step": step_num})
        return final_actions

    # ----- Communication ------------------------------------------------

    def broadcast(
        self,
        sender_id: int,
        payload: dict[str, Any],
        time_s: float,
        states: dict[int, AgentState],
    ) -> bool:
        """Broadcast a message from one robot to the fleet.

        Args:
            sender_id: Sending robot ID.
            payload: Message data.
            time_s: Current simulation time.
            states: Current agent states (for positions).

        Returns:
            ``True`` if at least one robot received the message.
        """
        positions: dict[int, np.ndarray] = {}
        for rid in self._robot_ids:
            if rid in states:
                st = states[rid]
                positions[rid] = np.array([st.x, st.y])
        msg = Message(sender_id=sender_id, receiver_id=-1, payload=payload, timestamp=time_s)
        return self._comm.send(msg, positions)

    def get_messages(self, robot_id: int) -> list[Message]:
        """Retrieve pending messages for a robot."""
        return self._comm.receive(robot_id)

    # ----- Task assignment ----------------------------------------------

    def assign_tasks(
        self,
        task_positions: np.ndarray,
        states: dict[int, AgentState],
        method: str = "greedy",
    ) -> dict[int, int]:
        """Assign tasks to fleet members.

        Args:
            task_positions: Shape ``(M, 2)``.
            states: Current agent states.
            method: Assignment method.

        Returns:
            Dict mapping robot ID to task index.
        """
        positions = np.zeros((self.size, 2))
        for idx, rid in enumerate(self._robot_ids):
            if rid in states:
                positions[idx] = [states[rid].x, states[rid].y]
        assignments = self._planner.assign_tasks(positions, task_positions, method)
        return {self._robot_ids[i]: assignments[i] for i in range(self.size)}

    # ----- Metrics ------------------------------------------------------

    def fleet_centroid(self, states: dict[int, AgentState]) -> np.ndarray:
        """Compute the fleet centroid position.

        Args:
            states: Current agent states.

        Returns:
            Centroid, shape ``(2,)``.
        """
        positions = []
        for rid in self._robot_ids:
            if rid in states:
                positions.append([states[rid].x, states[rid].y])
        if not positions:
            return np.zeros(2)
        return np.mean(positions, axis=0)

    def fleet_spread(self, states: dict[int, AgentState]) -> float:
        """Compute the maximum pairwise distance in the fleet.

        Args:
            states: Current agent states.

        Returns:
            Maximum inter-robot distance (metres).
        """
        positions = []
        for rid in self._robot_ids:
            if rid in states:
                positions.append([states[rid].x, states[rid].y])
        if len(positions) < 2:
            return 0.0
        pos = np.array(positions)
        max_dist = 0.0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if d > max_dist:
                    max_dist = d
        return max_dist

    def formation_error(
        self,
        states: dict[int, AgentState],
        heading: float = 0.0,
    ) -> float:
        """Compute the mean deviation from ideal formation positions.

        Args:
            states: Current agent states.
            heading: Formation heading (rad).

        Returns:
            Mean positional error (metres).
        """
        n = self.size
        if n == 0:
            return 0.0
        positions = np.zeros((n, 2))
        for idx, rid in enumerate(self._robot_ids):
            if rid in states:
                positions[idx] = [states[rid].x, states[rid].y]
        centroid = np.mean(positions, axis=0)
        targets = self._planner.compute_formation_targets(centroid, heading, n)
        errors = np.linalg.norm(positions - targets, axis=1)
        return float(np.mean(errors))
