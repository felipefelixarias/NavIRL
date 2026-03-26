"""Task allocation algorithms for multi-robot systems.

Includes market-based (auction), optimal (Hungarian), and heuristic
(greedy) allocators for assigning tasks to agents.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A task to be allocated to an agent.

    Attributes:
        id: Unique task identifier.
        location: Spatial location as an array-like ``(x, y [, z])``.
        priority: Priority level (higher = more urgent).
        requirements: List of capability tags the assigned agent must have.
        deadline: Optional deadline (seconds from epoch or sim time).
    """

    id: str
    location: np.ndarray
    priority: float = 1.0
    requirements: List[str] = field(default_factory=list)
    deadline: Optional[float] = None

    def __post_init__(self) -> None:
        self.location = np.asarray(self.location, dtype=np.float64)


@dataclass
class AllocationResult:
    """Result of a task allocation computation.

    Attributes:
        assignments: Mapping from agent id to list of assigned :class:`Task` objects.
        total_cost: Sum of individual assignment costs.
        unassigned: Tasks that could not be assigned.
    """

    assignments: Dict[str, List[Task]]
    total_cost: float
    unassigned: List[Task] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _euclidean_cost(agent_pos: np.ndarray, task: Task) -> float:
    """Default cost: Euclidean distance between agent and task location."""
    return float(np.linalg.norm(np.asarray(agent_pos) - task.location))


# ---------------------------------------------------------------------------
# Allocators
# ---------------------------------------------------------------------------

class AuctionAllocator:
    """Market-based task allocation using sequential or bundle auctions.

    Parameters:
        cost_fn: Callable ``(agent_position, task) -> float`` used to
            compute bid values.  Defaults to Euclidean distance.
    """

    def __init__(
        self,
        cost_fn: Optional[Callable[[np.ndarray, Task], float]] = None,
    ) -> None:
        self.cost_fn = cost_fn or _euclidean_cost

    def sequential_auction(
        self,
        agent_positions: Dict[str, np.ndarray],
        tasks: Sequence[Task],
    ) -> AllocationResult:
        """Run a sequential single-item auction.

        Tasks are auctioned one at a time in priority order (highest first).
        Each task is awarded to the agent with the lowest bid (cost).  An
        agent can win multiple tasks.

        Parameters:
            agent_positions: ``{agent_id: position}`` mapping.
            tasks: Iterable of :class:`Task` objects.

        Returns:
            :class:`AllocationResult` with assignments and total cost.
        """
        sorted_tasks = sorted(tasks, key=lambda t: -t.priority)
        assignments: Dict[str, List[Task]] = {aid: [] for aid in agent_positions}
        total_cost = 0.0
        unassigned: List[Task] = []

        for task in sorted_tasks:
            best_agent: Optional[str] = None
            best_cost = float("inf")
            for aid, pos in agent_positions.items():
                cost = self.cost_fn(np.asarray(pos), task)
                if cost < best_cost:
                    best_cost = cost
                    best_agent = aid
            if best_agent is not None:
                assignments[best_agent].append(task)
                total_cost += best_cost
            else:
                unassigned.append(task)

        return AllocationResult(
            assignments=assignments, total_cost=total_cost, unassigned=unassigned
        )

    def bundle_auction(
        self,
        agent_positions: Dict[str, np.ndarray],
        tasks: Sequence[Task],
        max_bundle_size: int = 3,
    ) -> AllocationResult:
        """Run a bundle auction for multi-task allocation.

        Each agent bids on bundles of up to *max_bundle_size* tasks.  The
        greedy winner-determination selects bundles that minimise total cost.

        Parameters:
            agent_positions: ``{agent_id: position}`` mapping.
            tasks: Iterable of :class:`Task` objects.
            max_bundle_size: Maximum number of tasks per bundle.

        Returns:
            :class:`AllocationResult`.
        """
        remaining = list(tasks)
        assignments: Dict[str, List[Task]] = {aid: [] for aid in agent_positions}
        total_cost = 0.0

        while remaining:
            best_agent: Optional[str] = None
            best_bundle: List[Task] = []
            best_cost = float("inf")

            for aid, pos in agent_positions.items():
                # Evaluate bundles of size 1..max_bundle_size (greedy enumeration)
                for size in range(1, min(max_bundle_size, len(remaining)) + 1):
                    # Sort remaining by cost for this agent and take cheapest bundle
                    sorted_by_cost = sorted(
                        remaining, key=lambda t: self.cost_fn(np.asarray(pos), t)
                    )
                    bundle = sorted_by_cost[:size]
                    bundle_cost = sum(self.cost_fn(np.asarray(pos), t) for t in bundle)
                    if bundle_cost < best_cost:
                        best_cost = bundle_cost
                        best_agent = aid
                        best_bundle = bundle

            if best_agent is None or not best_bundle:
                break

            assignments[best_agent].extend(best_bundle)
            total_cost += best_cost
            for t in best_bundle:
                remaining.remove(t)

        return AllocationResult(
            assignments=assignments, total_cost=total_cost, unassigned=remaining
        )


class HungarianAllocator:
    """Optimal one-to-one task assignment via the Hungarian algorithm.

    Uses ``scipy.optimize.linear_sum_assignment`` when available, falling
    back to a pure-numpy implementation for small problems otherwise.

    Parameters:
        cost_fn: Cost function ``(agent_position, task) -> float``.
    """

    def __init__(
        self,
        cost_fn: Optional[Callable[[np.ndarray, Task], float]] = None,
    ) -> None:
        self.cost_fn = cost_fn or _euclidean_cost

    def allocate(
        self,
        agent_positions: Dict[str, np.ndarray],
        tasks: Sequence[Task],
    ) -> AllocationResult:
        """Compute optimal one-to-one assignment.

        If there are more tasks than agents (or vice-versa) the cost matrix
        is padded so that the Hungarian algorithm can still be applied; extra
        items are left unassigned.

        Parameters:
            agent_positions: ``{agent_id: position}`` mapping.
            tasks: Sequence of :class:`Task` objects.

        Returns:
            :class:`AllocationResult`.
        """
        agent_ids = list(agent_positions.keys())
        n_agents = len(agent_ids)
        n_tasks = len(tasks)
        size = max(n_agents, n_tasks)

        # Build cost matrix (pad with large values)
        large = 1e9
        cost_matrix = np.full((size, size), large, dtype=np.float64)
        for i, aid in enumerate(agent_ids):
            for j, task in enumerate(tasks):
                cost_matrix[i, j] = self.cost_fn(np.asarray(agent_positions[aid]), task)

        row_ind, col_ind = self._solve(cost_matrix)

        assignments: Dict[str, List[Task]] = {aid: [] for aid in agent_ids}
        total_cost = 0.0
        assigned_tasks: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if r < n_agents and c < n_tasks and cost_matrix[r, c] < large:
                assignments[agent_ids[r]].append(tasks[c])
                total_cost += cost_matrix[r, c]
                assigned_tasks.add(c)

        unassigned = [t for j, t in enumerate(tasks) if j not in assigned_tasks]
        return AllocationResult(
            assignments=assignments, total_cost=total_cost, unassigned=unassigned
        )

    @staticmethod
    def _solve(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the assignment problem, preferring scipy when available."""
        try:
            from scipy.optimize import linear_sum_assignment

            return linear_sum_assignment(cost_matrix)
        except ImportError:
            # Greedy fallback for small matrices
            n = cost_matrix.shape[0]
            rows: List[int] = []
            cols: List[int] = []
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            flat = np.argsort(cost_matrix, axis=None)
            for idx in flat:
                r, c = divmod(int(idx), n)
                if r not in used_rows and c not in used_cols:
                    rows.append(r)
                    cols.append(c)
                    used_rows.add(r)
                    used_cols.add(c)
                if len(rows) == n:
                    break
            return np.array(rows), np.array(cols)


class GreedyAllocator:
    """Greedy nearest-task allocation.

    Each agent is assigned the nearest unassigned task until all tasks (or
    agents) are exhausted.

    Parameters:
        cost_fn: Cost function ``(agent_position, task) -> float``.
    """

    def __init__(
        self,
        cost_fn: Optional[Callable[[np.ndarray, Task], float]] = None,
    ) -> None:
        self.cost_fn = cost_fn or _euclidean_cost

    def allocate(
        self,
        agent_positions: Dict[str, np.ndarray],
        tasks: Sequence[Task],
    ) -> AllocationResult:
        """Greedily assign tasks to agents by increasing cost.

        Parameters:
            agent_positions: ``{agent_id: position}`` mapping.
            tasks: Sequence of :class:`Task` objects.

        Returns:
            :class:`AllocationResult`.
        """
        remaining = list(tasks)
        assignments: Dict[str, List[Task]] = {aid: [] for aid in agent_positions}
        total_cost = 0.0

        # Build all (agent, task) pairs sorted by cost
        pairs: List[Tuple[float, str, int]] = []
        for aid, pos in agent_positions.items():
            for j, task in enumerate(remaining):
                cost = self.cost_fn(np.asarray(pos), task)
                pairs.append((cost, aid, j))
        pairs.sort()

        assigned_indices: set[int] = set()
        for cost, aid, j in pairs:
            if j in assigned_indices:
                continue
            assignments[aid].append(remaining[j])
            total_cost += cost
            assigned_indices.add(j)
            if len(assigned_indices) == len(remaining):
                break

        unassigned = [t for j, t in enumerate(remaining) if j not in assigned_indices]
        return AllocationResult(
            assignments=assignments, total_cost=total_cost, unassigned=unassigned
        )
