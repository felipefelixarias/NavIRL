from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class SceneBackend(ABC):
    """Abstract backend interface used by NavIRL runners/controllers.

    Every concrete backend must implement all ``@abstractmethod`` members.
    Optional hooks (``nearest_clear_point``, ``map_metadata``, ``dt``) have
    sensible defaults so that minimal backends can skip them.
    """

    @property
    def dt(self) -> float:
        """Physics time-step in seconds.

        Backends that support variable time-steps should return the value
        that was active during the most recent ``step()`` call.  The default
        returns ``0.1`` so callers always get a safe fallback.
        """
        return 0.1

    @abstractmethod
    def add_agent(
        self,
        agent_id: int,
        position: tuple[float, float],
        radius: float,
        max_speed: float,
        kind: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_preferred_velocity(self, agent_id: int, velocity: tuple[float, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_position(self, agent_id: int) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_velocity(self, agent_id: int) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def shortest_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def sample_free_point(self) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def check_obstacle_collision(self, position: tuple[float, float], radius: float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def world_to_map(self, position: tuple[float, float]) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def map_image(self) -> np.ndarray:
        raise NotImplementedError

    def nearest_clear_point(
        self, position: tuple[float, float], radius: float
    ) -> tuple[float, float]:
        """Find the nearest collision-free point to the given position.

        Parameters
        ----------
        position : tuple[float, float]
            Target position as (x, y) coordinates.
        radius : float
            Search radius for finding clear space.

        Returns
        -------
        tuple[float, float]
            Nearest clear position as (x, y) coordinates.
            Base implementation returns input position unchanged.

        Note
        ----
        This base implementation does not perform collision checking.
        Subclasses should override to provide actual collision avoidance.
        """
        return tuple(map(float, position))

    def map_metadata(self) -> dict:
        """Get metadata about the environment map.

        Returns
        -------
        dict
            Dictionary containing map properties such as:
            - 'bounds': Map boundaries
            - 'resolution': Grid resolution
            - 'obstacles': Obstacle information
            Base implementation returns empty dictionary.

        Note
        ----
        Subclasses should override to provide specific map metadata.
        """
        return {}
