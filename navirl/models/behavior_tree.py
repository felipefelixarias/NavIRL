"""Composable behavior-tree framework for pedestrian behaviors.

Provides a lightweight behavior-tree implementation with standard node types
(Sequence, Selector, Decorator, Action, Condition) and a library of pre-built
leaf nodes for common pedestrian behaviors.  ``BehaviorTreeHumanController``
wraps a tree to implement the ``HumanController`` interface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from navirl.core.constants import COMFORT, EPSILON
from navirl.core.types import Action, AgentState
from navirl.humans.base import EventSink, HumanController

__all__ = ["BehaviorTree", "BehaviorTreeHumanController"]


# ---------------------------------------------------------------------------
#  Tick status
# ---------------------------------------------------------------------------


class Status(Enum):
    """Result of a single ``tick()`` call on a behaviour-tree node."""

    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


# ---------------------------------------------------------------------------
#  Blackboard - shared data store for a single agent's tree evaluation
# ---------------------------------------------------------------------------


@dataclass
class Blackboard:
    """Per-agent mutable context passed through the tree during a tick."""

    agent: AgentState
    neighbours: list[AgentState] = field(default_factory=list)
    robot: AgentState | None = None
    goal: tuple[float, float] = (0.0, 0.0)
    dt: float = 0.04
    # Output accumulator (preferred velocity).
    pref_vx: float = 0.0
    pref_vy: float = 0.0
    behavior: str = "GO_TO"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Base node
# ---------------------------------------------------------------------------


class Node(ABC):
    """Abstract base class for all behaviour-tree nodes."""

    @abstractmethod
    def tick(self, bb: Blackboard) -> Status:
        """Execute one tick and return the resulting status."""
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset any internal running state (called between episodes)."""
        ...


# ---------------------------------------------------------------------------
#  Composite nodes
# ---------------------------------------------------------------------------


class Sequence(Node):
    """Ticks children left-to-right; fails on the first FAILURE.

    Returns SUCCESS only if every child succeeds.  Returns RUNNING if a
    child returns RUNNING (subsequent children are not ticked).
    """

    def __init__(self, children: list[Node]) -> None:
        self.children = children

    def tick(self, bb: Blackboard) -> Status:
        for child in self.children:
            status = child.tick(bb)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

    def reset_state(self) -> None:
        for c in self.children:
            c.reset_state()


class Selector(Node):
    """Ticks children left-to-right; succeeds on the first SUCCESS.

    Returns FAILURE only if every child fails.
    """

    def __init__(self, children: list[Node]) -> None:
        self.children = children

    def tick(self, bb: Blackboard) -> Status:
        for child in self.children:
            status = child.tick(bb)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE

    def reset_state(self) -> None:
        for c in self.children:
            c.reset_state()


# ---------------------------------------------------------------------------
#  Decorator nodes
# ---------------------------------------------------------------------------


class Inverter(Node):
    """Inverts SUCCESS ↔ FAILURE; passes RUNNING through."""

    def __init__(self, child: Node) -> None:
        self.child = child

    def tick(self, bb: Blackboard) -> Status:
        status = self.child.tick(bb)
        if status == Status.SUCCESS:
            return Status.FAILURE
        if status == Status.FAILURE:
            return Status.SUCCESS
        return Status.RUNNING

    def reset_state(self) -> None:
        self.child.reset_state()


class Repeater(Node):
    """Ticks the child up to *max_repeats* times (or forever if -1).

    Returns RUNNING while repeating.  Returns FAILURE if the child
    ever returns FAILURE.
    """

    def __init__(self, child: Node, max_repeats: int = -1) -> None:
        self.child = child
        self.max_repeats = max_repeats
        self._count = 0

    def tick(self, bb: Blackboard) -> Status:
        if self.max_repeats > 0 and self._count >= self.max_repeats:
            return Status.SUCCESS

        status = self.child.tick(bb)
        if status == Status.FAILURE:
            return Status.FAILURE

        self._count += 1
        if self.max_repeats > 0 and self._count >= self.max_repeats:
            return Status.SUCCESS
        return Status.RUNNING

    def reset_state(self) -> None:
        self._count = 0
        self.child.reset_state()


# ---------------------------------------------------------------------------
#  Condition nodes (leaf)
# ---------------------------------------------------------------------------


class Condition(Node):
    """Leaf node that evaluates a predicate without side effects."""

    def __init__(self, predicate: Any = None) -> None:
        self._predicate = predicate

    def tick(self, bb: Blackboard) -> Status:
        if self._predicate is not None:
            return Status.SUCCESS if self._predicate(bb) else Status.FAILURE
        return Status.FAILURE

    def reset_state(self) -> None:
        """Conditions are stateless — nothing to reset."""


class IsNearGoal(Condition):
    """Succeeds if the agent is within *radius* of its goal."""

    def __init__(self, radius: float = 0.5) -> None:
        super().__init__()
        self.radius = radius

    def tick(self, bb: Blackboard) -> Status:
        dx = bb.goal[0] - bb.agent.x
        dy = bb.goal[1] - bb.agent.y
        return Status.SUCCESS if math.hypot(dx, dy) < self.radius else Status.FAILURE


class IsObstacleAhead(Condition):
    """Succeeds if any neighbour is within *distance* in the forward cone."""

    def __init__(self, distance: float = 2.0, cone_half_angle: float = 0.6) -> None:
        super().__init__()
        self.distance = distance
        self.cone_half_angle = cone_half_angle

    def tick(self, bb: Blackboard) -> Status:
        speed = math.hypot(bb.agent.vx, bb.agent.vy)
        if speed < EPSILON:
            return Status.FAILURE
        fx, fy = bb.agent.vx / speed, bb.agent.vy / speed

        for n in bb.neighbours:
            dx = n.x - bb.agent.x
            dy = n.y - bb.agent.y
            dist = math.hypot(dx, dy)
            if dist > self.distance or dist < EPSILON:
                continue
            cos_angle = (dx * fx + dy * fy) / dist
            if cos_angle > math.cos(self.cone_half_angle):
                return Status.SUCCESS
        return Status.FAILURE


# ---------------------------------------------------------------------------
#  Action (leaf) nodes - modify the blackboard's preferred velocity
# ---------------------------------------------------------------------------


class ActionNode(Node):
    """Base for leaf nodes that produce motor commands."""


class GoToGoal(ActionNode):
    """Steer toward the agent's goal at preferred speed."""

    def tick(self, bb: Blackboard) -> Status:
        dx = bb.goal[0] - bb.agent.x
        dy = bb.goal[1] - bb.agent.y
        dist = math.hypot(dx, dy)
        if dist < EPSILON:
            bb.pref_vx, bb.pref_vy = 0.0, 0.0
            return Status.SUCCESS

        speed = min(bb.agent.max_speed, dist / max(COMFORT.relaxation_time, EPSILON))
        bb.pref_vx = speed * dx / dist
        bb.pref_vy = speed * dy / dist
        bb.behavior = "GO_TO"
        return Status.RUNNING

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


class AvoidCollision(ActionNode):
    """Adjust velocity to avoid the closest threatening neighbour."""

    def __init__(self, time_horizon: float = COMFORT.collision_avoidance_time_horizon) -> None:
        super().__init__()
        self.time_horizon = time_horizon

    def tick(self, bb: Blackboard) -> Status:
        agent = bb.agent
        closest_dist = float("inf")
        avoid_dx, avoid_dy = 0.0, 0.0

        for n in bb.neighbours:
            dx = n.x - agent.x
            dy = n.y - agent.y
            dist = math.hypot(dx, dy)
            if dist < closest_dist and dist < self.time_horizon * agent.max_speed:
                closest_dist = dist
                if dist < EPSILON:
                    avoid_dx, avoid_dy = -1.0, 0.0
                else:
                    avoid_dx = -dx / dist
                    avoid_dy = -dy / dist

        if closest_dist == float("inf"):
            return Status.SUCCESS  # nothing to avoid

        # Blend avoidance into existing preferred velocity.
        strength = max(0.0, 1.0 - closest_dist / (self.time_horizon * agent.max_speed + EPSILON))
        bb.pref_vx += strength * avoid_dx * agent.max_speed
        bb.pref_vy += strength * avoid_dy * agent.max_speed
        bb.behavior = "AVOID"
        return Status.SUCCESS

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


class MaintainDistance(ActionNode):
    """Keep at least *min_distance* from all neighbours."""

    def __init__(self, min_distance: float = COMFORT.preferred_stranger_distance) -> None:
        super().__init__()
        self.min_distance = min_distance

    def tick(self, bb: Blackboard) -> Status:
        agent = bb.agent
        fx, fy = 0.0, 0.0
        count = 0
        for n in bb.neighbours:
            dx = agent.x - n.x
            dy = agent.y - n.y
            dist = math.hypot(dx, dy)
            if dist < self.min_distance and dist > EPSILON:
                overlap = self.min_distance - dist
                fx += overlap * dx / dist
                fy += overlap * dy / dist
                count += 1

        if count == 0:
            return Status.SUCCESS

        bb.pref_vx += fx
        bb.pref_vy += fy
        return Status.SUCCESS

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


class YieldAtDoorway(ActionNode):
    """Slow down or stop to yield passage at a narrow opening."""

    def __init__(self, yield_distance: float = 1.5) -> None:
        super().__init__()
        self.yield_distance = yield_distance

    def tick(self, bb: Blackboard) -> Status:
        agent = bb.agent
        # Check for oncoming agent in the forward direction.
        speed = math.hypot(agent.vx, agent.vy)
        if speed < EPSILON:
            return Status.FAILURE

        fx, fy = agent.vx / speed, agent.vy / speed

        for n in bb.neighbours:
            dx = n.x - agent.x
            dy = n.y - agent.y
            dist = math.hypot(dx, dy)
            if dist > self.yield_distance or dist < EPSILON:
                continue

            # Oncoming check: other agent moving roughly opposite.
            n_speed = math.hypot(n.vx, n.vy)
            if n_speed < EPSILON:
                continue
            dot = (fx * n.vx + fy * n.vy) / n_speed
            if dot < -0.3:  # roughly opposing
                # Yield: reduce speed significantly.
                bb.pref_vx *= 0.2
                bb.pref_vy *= 0.2
                bb.behavior = "YIELD"
                bb.metadata["yielding_to"] = n.agent_id
                return Status.SUCCESS

        return Status.FAILURE

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


class FollowGroup(ActionNode):
    """Steer toward the centroid of nearby agents moving in the same direction."""

    def __init__(self, group_radius: float = COMFORT.group_max_separation) -> None:
        super().__init__()
        self.group_radius = group_radius

    def tick(self, bb: Blackboard) -> Status:
        agent = bb.agent
        cx, cy = 0.0, 0.0
        count = 0

        for n in bb.neighbours:
            dx = n.x - agent.x
            dy = n.y - agent.y
            dist = math.hypot(dx, dy)
            if dist > self.group_radius:
                continue
            # Velocity alignment check.
            dot = agent.vx * n.vx + agent.vy * n.vy
            if dot > 0:
                cx += n.x
                cy += n.y
                count += 1

        if count == 0:
            return Status.FAILURE

        cx /= count
        cy /= count
        dx = cx - agent.x
        dy = cy - agent.y
        dist = math.hypot(dx, dy)
        if dist < EPSILON:
            return Status.SUCCESS

        # Gentle pull toward group centroid.
        strength = 0.3
        bb.pref_vx += strength * dx / dist
        bb.pref_vy += strength * dy / dist
        bb.behavior = "GROUP_FOLLOW"
        return Status.SUCCESS

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


class WaitInQueue(ActionNode):
    """Stop and wait behind the agent directly ahead."""

    def __init__(self, queue_distance: float = 0.8) -> None:
        super().__init__()
        self.queue_distance = queue_distance

    def tick(self, bb: Blackboard) -> Status:
        agent = bb.agent
        speed = math.hypot(agent.vx, agent.vy)
        if speed < EPSILON:
            # Already stopped - check if there is someone very close ahead
            # by looking toward the goal.
            gx = bb.goal[0] - agent.x
            gy = bb.goal[1] - agent.y
            gdist = math.hypot(gx, gy)
            if gdist < EPSILON:
                return Status.FAILURE
            fx, fy = gx / gdist, gy / gdist
        else:
            fx, fy = agent.vx / speed, agent.vy / speed

        for n in bb.neighbours:
            dx = n.x - agent.x
            dy = n.y - agent.y
            dist = math.hypot(dx, dy)
            if dist > self.queue_distance or dist < EPSILON:
                continue
            # Is the neighbour roughly ahead?
            cos_a = (dx * fx + dy * fy) / dist
            if cos_a > 0.5:
                # Neighbour ahead is close → wait.
                n_speed = math.hypot(n.vx, n.vy)
                if n_speed < 0.3:
                    bb.pref_vx = 0.0
                    bb.pref_vy = 0.0
                    bb.behavior = "QUEUE"
                    bb.metadata["waiting_behind"] = n.agent_id
                    return Status.SUCCESS

        return Status.FAILURE

    def reset_state(self) -> None:
        """Reset any internal state (stateless, so no action needed)."""


# ---------------------------------------------------------------------------
#  BehaviorTree wrapper
# ---------------------------------------------------------------------------


class BehaviorTree:
    """Wraps a root ``Node`` and provides a convenience ``tick`` method."""

    def __init__(self, root: Node) -> None:
        self.root = root

    def tick(self, bb: Blackboard) -> Status:
        """Run one evaluation pass of the tree."""
        return self.root.tick(bb)

    def reset(self) -> None:
        """Reset internal running state on all nodes."""
        self.root.reset_state()

    # -- factory for a sensible default pedestrian tree -----------------

    @classmethod
    def default_pedestrian_tree(cls) -> BehaviorTree:
        """Create a general-purpose pedestrian behavior tree.

        Priority (highest first):
            1. Wait in queue if someone stationary is directly ahead.
            2. Yield at doorway if an oncoming agent is very close.
            3. Avoid imminent collision.
            4. Follow group centroid (if applicable).
            5. Maintain comfortable distance from neighbours.
            6. Go to goal.
        """
        root = Selector(
            [
                WaitInQueue(),
                YieldAtDoorway(),
                Sequence(
                    [
                        GoToGoal(),
                        AvoidCollision(),
                        MaintainDistance(),
                    ]
                ),
                Sequence(
                    [
                        FollowGroup(),
                        GoToGoal(),
                    ]
                ),
                GoToGoal(),
            ]
        )
        return cls(root)


# ---------------------------------------------------------------------------
#  BehaviorTreeHumanController
# ---------------------------------------------------------------------------


class BehaviorTreeHumanController(HumanController):
    """Human controller that uses a behavior tree to select actions.

    By default a general-purpose pedestrian tree is created via
    ``BehaviorTree.default_pedestrian_tree()``.  A custom tree can be
    supplied instead.

    Parameters
    ----------
    tree:
        An optional pre-built ``BehaviorTree``.  If *None*, the default
        pedestrian tree is used.
    """

    def __init__(self, tree: BehaviorTree | None = None) -> None:
        self.tree = tree or BehaviorTree.default_pedestrian_tree()
        self.human_ids: list[int] = []
        self.starts: dict[int, tuple[float, float]] = {}
        self.goals: dict[int, tuple[float, float]] = {}
        self.backend: Any = None

    # -- HumanController interface --------------------------------------

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        self.human_ids = list(human_ids)
        self.starts = dict(starts)
        self.goals = dict(goals)
        self.backend = backend
        self.tree.reset()

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        actions: dict[int, Action] = {}

        for hid in self.human_ids:
            agent = states[hid]

            # Goal swap on arrival.
            gx, gy = self.goals[hid]
            if math.hypot(gx - agent.x, gy - agent.y) < 0.5:
                prev = self.goals[hid]
                self.goals[hid] = self.starts[hid]
                self.starts[hid] = prev
                emit_event(
                    "goal_swap",
                    hid,
                    {
                        "new_goal": list(self.goals[hid]),
                        "new_start": list(self.starts[hid]),
                    },
                )
                gx, gy = self.goals[hid]

            # Build neighbour list.
            neighbours = [s for aid, s in states.items() if aid != hid]

            # Populate blackboard.
            bb = Blackboard(
                agent=agent,
                neighbours=neighbours,
                robot=states.get(robot_id),
                goal=(gx, gy),
                dt=dt,
            )

            # Tick the tree.
            self.tree.tick(bb)

            # Clamp output to max speed.
            pvx, pvy = bb.pref_vx, bb.pref_vy
            speed = math.hypot(pvx, pvy)
            if speed > agent.max_speed and speed > EPSILON:
                scale = agent.max_speed / speed
                pvx *= scale
                pvy *= scale

            actions[hid] = Action(
                pref_vx=pvx,
                pref_vy=pvy,
                behavior=bb.behavior,
                metadata=bb.metadata,
            )

        return actions
