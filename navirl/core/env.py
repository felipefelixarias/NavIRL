from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class SceneBackend(ABC):
    """Abstract backend interface used by NavIRL runners/controllers."""

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
        return tuple(map(float, position))

    def map_metadata(self) -> dict:
        return {}
