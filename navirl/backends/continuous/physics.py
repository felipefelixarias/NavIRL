"""Physics engine for the continuous-space backend.

Provides kinematic and dynamic models for agent motion, including
integration methods, collision response, and force computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from navirl.backends.continuous.obstacles import ObstacleCollection


class IntegrationMethod(Enum):
    """Numerical integration methods."""

    EULER = "euler"
    SEMI_IMPLICIT_EULER = "semi_implicit_euler"
    VELOCITY_VERLET = "velocity_verlet"
    RK4 = "rk4"


@dataclass
class AgentState:
    """Full kinematic state of an agent.

    Parameters
    ----------
    position : np.ndarray
        2-D position, shape (2,).
    velocity : np.ndarray
        2-D velocity, shape (2,).
    heading : float
        Heading angle in radians.
    angular_velocity : float
        Angular velocity in rad/s.
    radius : float
        Agent collision radius.
    mass : float
        Agent mass in kg.
    max_speed : float
        Maximum speed in m/s.
    max_acceleration : float
        Maximum acceleration in m/s^2.
    max_angular_velocity : float
        Maximum angular velocity in rad/s.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    heading: float = 0.0
    angular_velocity: float = 0.0
    radius: float = 0.3
    mass: float = 80.0
    max_speed: float = 2.0
    max_acceleration: float = 3.0
    max_angular_velocity: float = 2.0 * math.pi

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

    @property
    def speed(self) -> float:
        """Current speed."""
        return float(np.linalg.norm(self.velocity))

    def copy(self) -> AgentState:
        """Create a deep copy."""
        return AgentState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=self.heading,
            angular_velocity=self.angular_velocity,
            radius=self.radius,
            mass=self.mass,
            max_speed=self.max_speed,
            max_acceleration=self.max_acceleration,
            max_angular_velocity=self.max_angular_velocity,
        )


@dataclass
class PhysicsConfig:
    """Configuration for the physics engine.

    Parameters
    ----------
    integration_method : IntegrationMethod
        Numerical integration method.
    friction_coefficient : float
        Kinetic friction coefficient.
    damping : float
        Velocity damping factor (0 = no damping, 1 = full damping).
    restitution : float
        Coefficient of restitution for collisions (0 = inelastic).
    enable_collision_response : bool
        Whether to apply collision response forces.
    collision_force_magnitude : float
        Magnitude of collision repulsion force.
    boundary_force_magnitude : float
        Force applied at world boundaries.
    min_separation : float
        Minimum allowed separation between agents.
    """

    integration_method: IntegrationMethod = IntegrationMethod.SEMI_IMPLICIT_EULER
    friction_coefficient: float = 0.0
    damping: float = 0.1
    restitution: float = 0.5
    enable_collision_response: bool = True
    collision_force_magnitude: float = 500.0
    boundary_force_magnitude: float = 200.0
    min_separation: float = 0.05


class PhysicsEngine:
    """Physics engine for continuous-space agent simulation.

    Handles agent motion integration, collision detection and response,
    and force computation.

    Parameters
    ----------
    config : PhysicsConfig, optional
        Physics configuration.
    obstacles : ObstacleCollection, optional
        Static obstacles for collision detection.
    world_bounds : tuple, optional
        (x_min, y_min, x_max, y_max) world boundaries.
    """

    def __init__(
        self,
        config: PhysicsConfig | None = None,
        obstacles: ObstacleCollection | None = None,
        world_bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.config = config or PhysicsConfig()
        self.obstacles = obstacles or ObstacleCollection()
        self.world_bounds = world_bounds
        self._force_accumulators: dict[int, np.ndarray] = {}
        self._collision_pairs: list[tuple[int, int]] = []

    def step(
        self,
        agents: dict[int, AgentState],
        actions: dict[int, np.ndarray],
        dt: float,
    ) -> dict[int, AgentState]:
        """Advance the simulation by one time step.

        Parameters
        ----------
        agents : dict
            Current agent states keyed by agent ID.
        actions : dict
            Desired velocity or force for each agent, shape (2,).
        dt : float
            Time step in seconds.

        Returns
        -------
        dict
            Updated agent states.
        """
        # 1. Compute forces
        forces = self._compute_forces(agents, actions, dt)

        # 2. Integrate motion
        new_states = {}
        method = self.config.integration_method

        for agent_id, state in agents.items():
            force = forces.get(agent_id, np.zeros(2))
            action = np.asarray(actions.get(agent_id, np.zeros(2)), dtype=np.float64)

            if method == IntegrationMethod.EULER:
                new_state = self._integrate_euler(state, action, force, dt)
            elif method == IntegrationMethod.SEMI_IMPLICIT_EULER:
                new_state = self._integrate_semi_implicit(state, action, force, dt)
            elif method == IntegrationMethod.VELOCITY_VERLET:
                new_state = self._integrate_verlet(state, action, force, dt)
            elif method == IntegrationMethod.RK4:
                new_state = self._integrate_rk4(state, action, force, dt)
            else:
                new_state = self._integrate_euler(state, action, force, dt)

            new_states[agent_id] = new_state

        # 3. Resolve collisions
        if self.config.enable_collision_response:
            self._resolve_agent_collisions(new_states)
            self._resolve_obstacle_collisions(new_states)
            self._resolve_boundary_collisions(new_states)

        # 4. Enforce speed limits
        for state in new_states.values():
            speed = state.speed
            if speed > state.max_speed:
                state.velocity *= state.max_speed / speed

        return new_states

    def _compute_forces(
        self,
        agents: dict[int, AgentState],
        actions: dict[int, np.ndarray],
        dt: float,
    ) -> dict[int, np.ndarray]:
        """Compute all forces acting on agents."""
        forces: dict[int, np.ndarray] = {aid: np.zeros(2) for aid in agents}

        # Damping forces
        if self.config.damping > 0:
            for aid, state in agents.items():
                forces[aid] -= self.config.damping * state.velocity * state.mass

        # Friction forces
        if self.config.friction_coefficient > 0:
            for aid, state in agents.items():
                speed = state.speed
                if speed > 1e-6:
                    friction = -self.config.friction_coefficient * state.mass * 9.81
                    forces[aid] += friction * state.velocity / speed

        # Agent-agent repulsion
        agent_list = list(agents.items())
        self._collision_pairs.clear()
        for i in range(len(agent_list)):
            aid_i, state_i = agent_list[i]
            for j in range(i + 1, len(agent_list)):
                aid_j, state_j = agent_list[j]

                diff = state_i.position - state_j.position
                dist = float(np.linalg.norm(diff))
                min_dist = state_i.radius + state_j.radius + self.config.min_separation

                if dist < min_dist and dist > 1e-6:
                    self._collision_pairs.append((aid_i, aid_j))
                    overlap = min_dist - dist
                    direction = diff / dist
                    force_mag = self.config.collision_force_magnitude * overlap
                    forces[aid_i] += force_mag * direction
                    forces[aid_j] -= force_mag * direction

        # Obstacle repulsion
        for aid, state in agents.items():
            for obs in self.obstacles.get_all_obstacles():
                dist = obs.distance_to_point(state.position)
                threshold = state.radius + self.config.min_separation
                if dist < threshold:
                    normal = obs.normal_at(state.position)
                    overlap = threshold - dist
                    forces[aid] += self.config.collision_force_magnitude * overlap * normal

        # Boundary forces
        if self.world_bounds is not None:
            x_min, y_min, x_max, y_max = self.world_bounds
            for aid, state in agents.items():
                pos = state.position
                r = state.radius
                bf = self.config.boundary_force_magnitude

                if pos[0] - r < x_min:
                    forces[aid][0] += bf * (x_min - pos[0] + r)
                if pos[0] + r > x_max:
                    forces[aid][0] -= bf * (pos[0] + r - x_max)
                if pos[1] - r < y_min:
                    forces[aid][1] += bf * (y_min - pos[1] + r)
                if pos[1] + r > y_max:
                    forces[aid][1] -= bf * (pos[1] + r - y_max)

        return forces

    def _integrate_euler(
        self,
        state: AgentState,
        action: np.ndarray,
        force: np.ndarray,
        dt: float,
    ) -> AgentState:
        """Forward Euler integration."""
        new_state = state.copy()
        acceleration = force / state.mass + self._action_to_acceleration(state, action, dt)

        # Clamp acceleration
        acc_mag = float(np.linalg.norm(acceleration))
        if acc_mag > state.max_acceleration:
            acceleration *= state.max_acceleration / acc_mag

        new_state.position = state.position + state.velocity * dt
        new_state.velocity = state.velocity + acceleration * dt
        new_state.heading = self._update_heading(state, new_state.velocity, dt)
        return new_state

    def _integrate_semi_implicit(
        self,
        state: AgentState,
        action: np.ndarray,
        force: np.ndarray,
        dt: float,
    ) -> AgentState:
        """Semi-implicit Euler (symplectic Euler) integration."""
        new_state = state.copy()
        acceleration = force / state.mass + self._action_to_acceleration(state, action, dt)

        acc_mag = float(np.linalg.norm(acceleration))
        if acc_mag > state.max_acceleration:
            acceleration *= state.max_acceleration / acc_mag

        # Update velocity first, then use new velocity for position
        new_state.velocity = state.velocity + acceleration * dt
        new_state.position = state.position + new_state.velocity * dt
        new_state.heading = self._update_heading(state, new_state.velocity, dt)
        return new_state

    def _integrate_verlet(
        self,
        state: AgentState,
        action: np.ndarray,
        force: np.ndarray,
        dt: float,
    ) -> AgentState:
        """Velocity Verlet integration."""
        new_state = state.copy()
        acceleration = force / state.mass + self._action_to_acceleration(state, action, dt)

        acc_mag = float(np.linalg.norm(acceleration))
        if acc_mag > state.max_acceleration:
            acceleration *= state.max_acceleration / acc_mag

        # Verlet: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_state.position = state.position + state.velocity * dt + 0.5 * acceleration * dt * dt
        # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        # Approximate a(t+dt) ≈ a(t) for simplicity
        new_state.velocity = state.velocity + acceleration * dt
        new_state.heading = self._update_heading(state, new_state.velocity, dt)
        return new_state

    def _integrate_rk4(
        self,
        state: AgentState,
        action: np.ndarray,
        force: np.ndarray,
        dt: float,
    ) -> AgentState:
        """Fourth-order Runge-Kutta integration."""
        new_state = state.copy()
        acceleration = force / state.mass + self._action_to_acceleration(state, action, dt)

        acc_mag = float(np.linalg.norm(acceleration))
        if acc_mag > state.max_acceleration:
            acceleration *= state.max_acceleration / acc_mag

        # RK4 for position and velocity
        k1_v = acceleration * dt
        k1_x = state.velocity * dt

        k2_v = acceleration * dt
        k2_x = (state.velocity + 0.5 * k1_v) * dt

        k3_v = acceleration * dt
        k3_x = (state.velocity + 0.5 * k2_v) * dt

        k4_v = acceleration * dt
        k4_x = (state.velocity + k3_v) * dt

        new_state.velocity = state.velocity + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        new_state.position = state.position + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        new_state.heading = self._update_heading(state, new_state.velocity, dt)
        return new_state

    def _action_to_acceleration(
        self,
        state: AgentState,
        action: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Convert a desired velocity action to acceleration.

        Computes the acceleration needed to reach the desired velocity
        from the current velocity within the time step.

        Parameters
        ----------
        state : AgentState
            Current state.
        action : np.ndarray
            Desired velocity, shape (2,).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Acceleration, shape (2,).
        """
        if dt <= 0:
            return np.zeros(2)

        desired_velocity = action.copy()

        # Clamp desired speed
        desired_speed = float(np.linalg.norm(desired_velocity))
        if desired_speed > state.max_speed:
            desired_velocity *= state.max_speed / desired_speed

        acceleration = (desired_velocity - state.velocity) / dt
        return acceleration

    def _update_heading(
        self,
        state: AgentState,
        new_velocity: np.ndarray,
        dt: float,
    ) -> float:
        """Update heading based on velocity direction."""
        speed = float(np.linalg.norm(new_velocity))
        if speed < 0.01:
            return state.heading

        desired_heading = math.atan2(new_velocity[1], new_velocity[0])

        # Smooth heading change
        diff = desired_heading - state.heading
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi

        max_change = state.max_angular_velocity * dt
        diff = max(-max_change, min(max_change, diff))

        heading = state.heading + diff
        while heading > math.pi:
            heading -= 2 * math.pi
        while heading < -math.pi:
            heading += 2 * math.pi

        return heading

    def _resolve_agent_collisions(
        self,
        agents: dict[int, AgentState],
    ) -> None:
        """Resolve penetrations between agents."""
        agent_list = list(agents.items())
        for i in range(len(agent_list)):
            aid_i, state_i = agent_list[i]
            for j in range(i + 1, len(agent_list)):
                aid_j, state_j = agent_list[j]

                diff = state_i.position - state_j.position
                dist = float(np.linalg.norm(diff))
                min_dist = state_i.radius + state_j.radius

                if dist < min_dist and dist > 1e-6:
                    # Push agents apart
                    direction = diff / dist
                    overlap = min_dist - dist
                    total_mass = state_i.mass + state_j.mass
                    state_i.position += direction * overlap * (state_j.mass / total_mass)
                    state_j.position -= direction * overlap * (state_i.mass / total_mass)

                    # Apply restitution to velocities
                    e = self.config.restitution
                    v_rel = state_i.velocity - state_j.velocity
                    v_rel_n = np.dot(v_rel, direction) * direction

                    if np.dot(v_rel, direction) < 0:
                        impulse = -(1 + e) * v_rel_n / (1 / state_i.mass + 1 / state_j.mass)
                        state_i.velocity += impulse / state_i.mass
                        state_j.velocity -= impulse / state_j.mass

    def _resolve_obstacle_collisions(
        self,
        agents: dict[int, AgentState],
    ) -> None:
        """Resolve agent-obstacle penetrations."""
        for state in agents.values():
            for obs in self.obstacles.get_all_obstacles():
                dist = obs.distance_to_point(state.position)
                if dist < state.radius:
                    normal = obs.normal_at(state.position)
                    overlap = state.radius - dist
                    state.position += normal * overlap

                    # Remove velocity component into obstacle
                    v_n = np.dot(state.velocity, normal)
                    if v_n < 0:
                        state.velocity -= (1 + self.config.restitution) * v_n * normal

    def _resolve_boundary_collisions(
        self,
        agents: dict[int, AgentState],
    ) -> None:
        """Resolve agent-boundary penetrations."""
        if self.world_bounds is None:
            return

        x_min, y_min, x_max, y_max = self.world_bounds

        for state in agents.values():
            r = state.radius
            e = self.config.restitution

            if state.position[0] - r < x_min:
                state.position[0] = x_min + r
                if state.velocity[0] < 0:
                    state.velocity[0] *= -e

            if state.position[0] + r > x_max:
                state.position[0] = x_max - r
                if state.velocity[0] > 0:
                    state.velocity[0] *= -e

            if state.position[1] - r < y_min:
                state.position[1] = y_min + r
                if state.velocity[1] < 0:
                    state.velocity[1] *= -e

            if state.position[1] + r > y_max:
                state.position[1] = y_max - r
                if state.velocity[1] > 0:
                    state.velocity[1] *= -e

    def get_collision_pairs(self) -> list[tuple[int, int]]:
        """Get agent-agent collision pairs from the last step.

        Returns
        -------
        list of (int, int)
            Pairs of colliding agent IDs.
        """
        return list(self._collision_pairs)

    def check_line_of_sight(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
    ) -> bool:
        """Check if there is a clear line of sight between two positions.

        Parameters
        ----------
        from_pos : np.ndarray
            Start position, shape (2,).
        to_pos : np.ndarray
            End position, shape (2,).

        Returns
        -------
        bool
            True if line of sight is clear.
        """
        from_pos = np.asarray(from_pos, dtype=np.float64)
        to_pos = np.asarray(to_pos, dtype=np.float64)
        diff = to_pos - from_pos
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return True

        direction = diff / dist
        result = self.obstacles.ray_cast(from_pos, direction, dist)
        return result is None

    def compute_energy(self, agents: dict[int, AgentState]) -> dict[str, float]:
        """Compute kinetic energy statistics.

        Parameters
        ----------
        agents : dict
            Agent states.

        Returns
        -------
        dict
            Energy statistics (total, mean, max, min).
        """
        energies = []
        for state in agents.values():
            speed = state.speed
            ke = 0.5 * state.mass * speed * speed
            energies.append(ke)

        if not energies:
            return {"total": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0}

        return {
            "total": sum(energies),
            "mean": sum(energies) / len(energies),
            "max": max(energies),
            "min": min(energies),
        }
