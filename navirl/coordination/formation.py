"""Formation control algorithms for multi-agent coordination.

Provides geometric formation maintenance, consensus-based distributed
formation control, and leader-follower architectures.
"""

from __future__ import annotations

import numpy as np


class FormationController:
    """Maintains a group of agents in a specified geometric formation.

    Supported formation types:

    * ``"line"`` -- agents arranged in a line along the heading direction.
    * ``"wedge"`` -- V-shaped formation opening away from the heading.
    * ``"circle"`` -- agents evenly spaced on a circle around the center.
    * ``"diamond"`` -- diamond / rhombus arrangement.
    * ``"custom"`` -- user-supplied relative offsets.

    Parameters:
        spacing: Default inter-agent spacing (metres).
        custom_offsets: Array of shape ``(N, 2)`` for ``"custom"`` formation.
    """

    formation_types: tuple[str, ...] = ("line", "wedge", "circle", "diamond", "custom")

    def __init__(
        self,
        spacing: float = 2.0,
        custom_offsets: np.ndarray | None = None,
    ) -> None:
        self.spacing = spacing
        self.custom_offsets = custom_offsets

    # -- public API ---------------------------------------------------------

    def compute_desired_positions(
        self,
        center: np.ndarray,
        heading: float,
        formation_type: str,
        num_agents: int,
    ) -> np.ndarray:
        """Compute desired 2-D positions for *num_agents* in the given formation.

        Parameters:
            center: Formation centroid as ``(2,)`` array ``[x, y]``.
            heading: Heading angle in radians (0 = positive-x).
            formation_type: One of :pyattr:`formation_types`.
            num_agents: Number of agents in the formation.

        Returns:
            Array of shape ``(num_agents, 2)`` with desired positions.

        Raises:
            ValueError: If *formation_type* is unknown.
        """
        center = np.asarray(center, dtype=np.float64)
        offsets = self._compute_offsets(formation_type, num_agents)
        rotated = self._rotate(offsets, heading)
        return rotated + center

    def compute_formation_error(
        self,
        current_positions: np.ndarray,
        desired_positions: np.ndarray,
    ) -> float:
        """Compute the mean Euclidean formation error.

        Parameters:
            current_positions: ``(N, 2)`` array of current agent positions.
            desired_positions: ``(N, 2)`` array of desired agent positions.

        Returns:
            Mean Euclidean distance between current and desired positions.
        """
        current_positions = np.asarray(current_positions, dtype=np.float64)
        desired_positions = np.asarray(desired_positions, dtype=np.float64)
        return float(np.mean(np.linalg.norm(current_positions - desired_positions, axis=1)))

    # -- internal helpers ---------------------------------------------------

    def _compute_offsets(self, formation_type: str, num_agents: int) -> np.ndarray:
        """Return local-frame offsets of shape ``(num_agents, 2)``."""
        if formation_type == "line":
            return self._line_offsets(num_agents)
        elif formation_type == "wedge":
            return self._wedge_offsets(num_agents)
        elif formation_type == "circle":
            return self._circle_offsets(num_agents)
        elif formation_type == "diamond":
            return self._diamond_offsets(num_agents)
        elif formation_type == "custom":
            if self.custom_offsets is None:
                raise ValueError("custom_offsets must be provided for 'custom' formation.")
            return np.asarray(self.custom_offsets[:num_agents], dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown formation type '{formation_type}'. Choose from {self.formation_types}."
            )

    def _line_offsets(self, n: int) -> np.ndarray:
        offsets = np.zeros((n, 2), dtype=np.float64)
        start = -self.spacing * (n - 1) / 2.0
        for i in range(n):
            offsets[i, 1] = start + i * self.spacing  # lateral spread
        return offsets

    def _wedge_offsets(self, n: int) -> np.ndarray:
        offsets = np.zeros((n, 2), dtype=np.float64)
        # Leader at front; others fan out behind
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            rank = (i + 1) // 2
            offsets[i, 0] = -rank * self.spacing  # behind leader
            offsets[i, 1] = side * rank * self.spacing  # lateral
        return offsets

    def _circle_offsets(self, n: int) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = self.spacing * n / (2 * np.pi) if n > 1 else 0.0
        offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
        return offsets

    def _diamond_offsets(self, n: int) -> np.ndarray:
        offsets = np.zeros((n, 2), dtype=np.float64)
        if n >= 1:
            offsets[0] = [self.spacing, 0.0]  # front
        if n >= 2:
            offsets[1] = [-self.spacing, 0.0]  # rear
        if n >= 3:
            offsets[2] = [0.0, self.spacing]  # right
        if n >= 4:
            offsets[3] = [0.0, -self.spacing]  # left
        # Extra agents placed concentrically
        for i in range(4, n):
            angle = 2 * np.pi * (i - 4) / max(n - 4, 1)
            offsets[i] = [
                np.cos(angle) * self.spacing * 2,
                np.sin(angle) * self.spacing * 2,
            ]
        return offsets

    @staticmethod
    def _rotate(offsets: np.ndarray, heading: float) -> np.ndarray:
        """Rotate 2-D *offsets* by *heading* radians."""
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, -s], [s, c]])
        return offsets @ rotation.T


class ConsensusFormation:
    """Consensus-based formation control using a graph Laplacian protocol.

    Agents iteratively update their positions so that inter-agent distances
    converge to the desired formation offsets.

    Parameters:
        desired_offsets: ``(N, 2)`` array of desired relative positions from
            the formation centroid.
        gain: Convergence gain applied each consensus step.
        adjacency: ``(N, N)`` binary adjacency matrix.  If ``None`` a fully
            connected graph is assumed.
    """

    def __init__(
        self,
        desired_offsets: np.ndarray,
        gain: float = 0.1,
        adjacency: np.ndarray | None = None,
    ) -> None:
        self.desired_offsets = np.asarray(desired_offsets, dtype=np.float64)
        self.gain = gain
        self.num_agents = len(self.desired_offsets)

        if adjacency is not None:
            self.adjacency = np.asarray(adjacency, dtype=np.float64)
        else:
            self.adjacency = np.ones((self.num_agents, self.num_agents)) - np.eye(self.num_agents)

        self._laplacian = np.diag(self.adjacency.sum(axis=1)) - self.adjacency

    @property
    def laplacian(self) -> np.ndarray:
        """Graph Laplacian matrix ``L = D - A``."""
        return self._laplacian

    def step(self, current_positions: np.ndarray) -> np.ndarray:
        """Execute one consensus step and return updated positions.

        Parameters:
            current_positions: ``(N, 2)`` array of current agent positions.

        Returns:
            ``(N, 2)`` array of updated positions after one consensus step.
        """
        current_positions = np.asarray(current_positions, dtype=np.float64)
        centroid = current_positions.mean(axis=0)
        desired_absolute = centroid + self.desired_offsets

        # Consensus correction toward desired inter-agent structure
        error = current_positions - desired_absolute
        correction = -self.gain * error
        return current_positions + correction

    def converged(self, current_positions: np.ndarray, tolerance: float = 0.05) -> bool:
        """Check whether agents have converged to the desired formation.

        Parameters:
            current_positions: ``(N, 2)`` current positions.
            tolerance: Maximum mean error to consider converged.
        """
        centroid = np.asarray(current_positions).mean(axis=0)
        desired_absolute = centroid + self.desired_offsets
        error = np.mean(np.linalg.norm(current_positions - desired_absolute, axis=1))
        return float(error) < tolerance


class LeaderFollower:
    """Leader-follower formation controller.

    One agent is designated the *leader* and navigates autonomously.  All
    other agents maintain prescribed relative offsets to the leader.

    Parameters:
        leader_index: Index of the leader in the agent array.
        follower_offsets: ``(N-1, 2)`` desired offsets of followers relative
            to the leader in the leader's local frame.
        gain: Proportional gain for follower position correction.
    """

    def __init__(
        self,
        leader_index: int = 0,
        follower_offsets: np.ndarray | None = None,
        gain: float = 0.5,
    ) -> None:
        self.leader_index = leader_index
        self.follower_offsets = (
            np.asarray(follower_offsets, dtype=np.float64) if follower_offsets is not None else None
        )
        self.gain = gain

    def compute_follower_targets(
        self,
        leader_position: np.ndarray,
        leader_heading: float,
    ) -> np.ndarray:
        """Compute global target positions for all followers.

        Parameters:
            leader_position: ``(2,)`` position of the leader.
            leader_heading: Leader heading in radians.

        Returns:
            ``(num_followers, 2)`` target positions in the global frame.

        Raises:
            ValueError: If :pyattr:`follower_offsets` has not been set.
        """
        if self.follower_offsets is None:
            raise ValueError("follower_offsets must be set before computing targets.")
        leader_position = np.asarray(leader_position, dtype=np.float64)
        c, s = np.cos(leader_heading), np.sin(leader_heading)
        rotation = np.array([[c, -s], [s, c]])
        rotated = self.follower_offsets @ rotation.T
        return rotated + leader_position

    def step(
        self,
        positions: np.ndarray,
        leader_heading: float,
    ) -> np.ndarray:
        """Return velocity commands that drive followers toward their targets.

        Parameters:
            positions: ``(N, 2)`` positions with leader at :pyattr:`leader_index`.
            leader_heading: Leader heading in radians.

        Returns:
            ``(N, 2)`` velocity commands (leader velocity is zero).
        """
        positions = np.asarray(positions, dtype=np.float64)
        num_agents = len(positions)
        velocities = np.zeros_like(positions)

        leader_pos = positions[self.leader_index]
        targets = self.compute_follower_targets(leader_pos, leader_heading)

        follower_idx = 0
        for i in range(num_agents):
            if i == self.leader_index:
                continue
            error = targets[follower_idx] - positions[i]
            velocities[i] = self.gain * error
            follower_idx += 1

        return velocities
