from __future__ import annotations

from typing import NamedTuple

try:
    import rvo2
except ImportError:
    rvo2 = None


class IndoorORCASimConfig(NamedTuple):
    time_step: float = 1 / 32.0
    neighbor_dist: float = 5.0
    max_neighbors: int = 4
    time_horizon: float = 1.25
    time_horizon_obst: float = 1.0
    radius: float = 0.125
    max_speed: float = 0.5


class IndoorORCASim:
    """Thin wrapper over rvo2.PyRVOSimulator."""

    def __init__(self, config: IndoorORCASimConfig):
        if rvo2 is None:
            raise ImportError(
                "rvo2 is required for IndoorORCASim. Install it with: pip install rvo2"
            )
        self.config = config
        self.sim = rvo2.PyRVOSimulator(
            config.time_step,
            config.neighbor_dist,
            config.max_neighbors,
            config.time_horizon,
            config.time_horizon_obst,
            config.radius,
            config.max_speed,
        )
        self._no_steps = True
        self.trajectories: list[list[list[float]]] = []

    @property
    def time_step(self) -> float:
        return self.config.time_step

    @property
    def neighbor_dist(self) -> float:
        return self.config.neighbor_dist

    @property
    def max_neighbors(self) -> int:
        return self.config.max_neighbors

    @property
    def time_horizon(self) -> float:
        return self.config.time_horizon

    @property
    def time_horizon_obst(self) -> float:
        return self.config.time_horizon_obst

    @property
    def radius(self) -> float:
        return self.config.radius

    @property
    def max_speed(self) -> float:
        return self.config.max_speed

    def add_agent(self, position: list[float], velocity: list[float] | None = None) -> int:
        _ = velocity
        return self.sim.addAgent(tuple(position))

    def add_obstacle(self, vertices: list[list[float]]) -> int:
        return self.sim.addObstacle(vertices)

    def add_obstacles(self, obstacles: list[list[list[float]]]) -> list[int]:
        return [self.add_obstacle(obstacle) for obstacle in obstacles]

    def process_obstacles(self) -> None:
        self.sim.processObstacles()

    def set_agent_pref_velocity(self, agent_no: int, pref_velocity: list[float]) -> None:
        self.sim.setAgentPrefVelocity(agent_no, tuple(pref_velocity))

    def set_agent_position(self, agent_no: int, position: list[float]) -> None:
        self.sim.setAgentPosition(agent_no, tuple(position))

    @property
    def num_agents(self) -> int:
        return self.sim.getNumAgents()

    @property
    def num_obstacle_vertices(self) -> int:
        return self.sim.getNumObstacleVertices()

    def get_agent_position(self, agent_no: int) -> list[float]:
        return self.sim.getAgentPosition(agent_no)

    def get_agent_velocity(self, agent_no: int) -> list[float]:
        return self.sim.getAgentVelocity(agent_no)

    def get_global_time(self) -> float:
        return self.sim.getGlobalTime()

    def do_step(self) -> None:
        if self._no_steps:
            for i in range(self.num_agents):  # Use property instead of method
                self.trajectories.append([])
                self.trajectories[i].append(list(self.get_agent_position(i)))
            self._no_steps = False
        else:
            for i in range(self.num_agents):  # Use property instead of method
                self.trajectories[i].append(list(self.get_agent_position(i)))

        self.sim.doStep()
