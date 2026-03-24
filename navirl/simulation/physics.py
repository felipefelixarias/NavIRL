"""Physics engine for NavIRL simulation.

Provides integration schemes (Euler, RK4), collision response, friction
modelling, force accumulation, and constraint solving.  Two concrete
motion models are included: :class:`KinematicModel` (velocity-driven)
and :class:`DynamicModel` (force-driven with mass & inertia).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from navirl.simulation.world import CollisionResult, World


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhysicsState:
    """Minimal physics state vector for a single entity."""

    position: np.ndarray  # (2,)
    velocity: np.ndarray  # (2,)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation: float = 0.0
    angular_velocity: float = 0.0

    def copy(self) -> PhysicsState:
        return PhysicsState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
        )


@dataclass
class MaterialProperties:
    """Friction and restitution coefficients for an entity."""

    static_friction: float = 0.6
    kinetic_friction: float = 0.4
    restitution: float = 0.3
    drag: float = 0.0


@dataclass
class ForceRecord:
    """A single force applied to an entity during the current step."""

    entity_id: int
    force: np.ndarray  # (2,)
    label: str = ""


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def _euler_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit Euler integration.  Returns (new_pos, new_vel)."""
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel


def _rk4_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """4th-order Runge-Kutta integration.

    Parameters
    ----------
    pos, vel : np.ndarray
        Current state.
    acc_fn : callable
        ``(position, velocity) -> acceleration`` for the ODE right-hand side.
    dt : float
        Time step.

    Returns
    -------
    new_pos, new_vel : np.ndarray
    """
    k1v = acc_fn(pos, vel) * dt
    k1x = vel * dt

    k2v = acc_fn(pos + 0.5 * k1x, vel + 0.5 * k1v) * dt
    k2x = (vel + 0.5 * k1v) * dt

    k3v = acc_fn(pos + 0.5 * k2x, vel + 0.5 * k2v) * dt
    k3x = (vel + 0.5 * k2v) * dt

    k4v = acc_fn(pos + k3x, vel + k3v) * dt
    k4x = (vel + k3v) * dt

    new_pos = pos + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    new_vel = vel + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    return new_pos, new_vel


# ---------------------------------------------------------------------------
# Abstract motion model
# ---------------------------------------------------------------------------

class MotionModel(abc.ABC):
    """Base class for entity motion models."""

    @abc.abstractmethod
    def compute_acceleration(
        self,
        state: PhysicsState,
        forces: np.ndarray,
        mass: float,
        dt: float,
    ) -> np.ndarray:
        """Return the acceleration vector given current state and net force."""

    @abc.abstractmethod
    def integrate(
        self,
        state: PhysicsState,
        forces: np.ndarray,
        mass: float,
        dt: float,
        method: str,
    ) -> PhysicsState:
        """Integrate one timestep and return new state."""


# ---------------------------------------------------------------------------
# KinematicModel
# ---------------------------------------------------------------------------

class KinematicModel(MotionModel):
    """Velocity-driven model.

    The entity's velocity is set directly (e.g. from a planner) and
    position is integrated.  Forces are interpreted as desired-velocity
    commands scaled by a response factor.

    Parameters
    ----------
    max_speed : float
        Maximum allowed speed.
    max_accel : float
        Maximum acceleration magnitude per step.
    response_time : float
        Time constant for velocity tracking (lower = snappier).
    """

    def __init__(
        self,
        max_speed: float = 2.0,
        max_accel: float = 4.0,
        response_time: float = 0.5,
    ) -> None:
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.response_time = max(response_time, 1e-6)

    # ------------------------------------------------------------------

    def compute_acceleration(
        self,
        state: PhysicsState,
        forces: np.ndarray,
        mass: float,
        dt: float,
    ) -> np.ndarray:
        """Compute acceleration toward a desired velocity derived from forces.

        ``forces`` is treated as a desired velocity target divided by mass.
        """
        desired_vel = forces / max(mass, 1e-6)
        speed = float(np.linalg.norm(desired_vel))
        if speed > self.max_speed:
            desired_vel = desired_vel * (self.max_speed / speed)
        acc = (desired_vel - state.velocity) / self.response_time
        acc_mag = float(np.linalg.norm(acc))
        if acc_mag > self.max_accel:
            acc = acc * (self.max_accel / acc_mag)
        return acc

    def integrate(
        self,
        state: PhysicsState,
        forces: np.ndarray,
        mass: float,
        dt: float,
        method: str = "euler",
    ) -> PhysicsState:
        """Integrate one timestep."""
        acc = self.compute_acceleration(state, forces, mass, dt)
        if method == "rk4":
            def acc_fn(p: np.ndarray, v: np.ndarray) -> np.ndarray:
                desired = forces / max(mass, 1e-6)
                sp = float(np.linalg.norm(desired))
                if sp > self.max_speed:
                    desired = desired * (self.max_speed / sp)
                a = (desired - v) / self.response_time
                am = float(np.linalg.norm(a))
                if am > self.max_accel:
                    a = a * (self.max_accel / am)
                return a

            new_pos, new_vel = _rk4_step(state.position, state.velocity, acc_fn, dt)
        else:
            new_pos, new_vel = _euler_step(state.position, state.velocity, acc, dt)

        # Enforce speed limit
        spd = float(np.linalg.norm(new_vel))
        if spd > self.max_speed:
            new_vel = new_vel * (self.max_speed / spd)

        # Orientation from velocity
        orient = state.orientation
        if spd > 0.05:
            orient = float(np.arctan2(new_vel[1], new_vel[0]))

        return PhysicsState(
            position=new_pos,
            velocity=new_vel,
            acceleration=acc,
            orientation=orient,
            angular_velocity=(orient - state.orientation) / max(dt, 1e-9),
        )


# ---------------------------------------------------------------------------
# DynamicModel
# ---------------------------------------------------------------------------

class DynamicModel(MotionModel):
    """Force-driven model with mass, inertia, and friction.

    Parameters
    ----------
    max_speed : float
        Speed cap.
    max_force : float
        Maximum net force magnitude.
    moment_of_inertia : float
        Rotational inertia.
    material : MaterialProperties | None
        Friction / drag properties.
    """

    def __init__(
        self,
        max_speed: float = 3.0,
        max_force: float = 20.0,
        moment_of_inertia: float = 0.1,
        material: MaterialProperties | None = None,
    ) -> None:
        self.max_speed = max_speed
        self.max_force = max_force
        self.moment_of_inertia = max(moment_of_inertia, 1e-9)
        self.material = material or MaterialProperties()

    # ------------------------------------------------------------------

    def compute_acceleration(
        self,
        state: PhysicsState,
        forces: np.ndarray,
        mass: float,
        dt: float,
    ) -> np.ndarray:
        """Compute acceleration = F/m with drag and friction."""
        net = forces.copy()

        # Clamp force magnitude
        fmag = float(np.linalg.norm(net))
        if fmag > self.max_force:
            net = net * (self.max_force / fmag)

        # Aerodynamic drag  F_drag = -drag * v * |v|
        spd = float(np.linalg.norm(state.velocity))
        if spd > 1e-6 and self.material.drag > 0.0:
            drag_force = -self.material.drag * state.velocity * spd
            net = net + drag_force

        # Kinetic friction opposes motion
        if spd > 1e-4:
            friction = (
                -self.material.kinetic_friction
                * mass
                * 9.81
                * (state.velocity / spd)
            )
            net = net + friction
        else:
            # Static friction prevents creep
            if fmag < self.material.static_friction * mass * 9.81:
                return np.zeros(2)

        acc = net / max(mass, 1e-6)
        return acc

    def integrate(
        self,
        state: PhysicsState,
        forces: np.ndarray,
        mass: float,
        dt: float,
        method: str = "euler",
    ) -> PhysicsState:
        """Integrate one timestep (Euler or RK4)."""
        if method == "rk4":
            def acc_fn(p: np.ndarray, v: np.ndarray) -> np.ndarray:
                tmp = PhysicsState(
                    position=p, velocity=v,
                    acceleration=np.zeros(2),
                    orientation=state.orientation,
                )
                return self.compute_acceleration(tmp, forces, mass, dt)

            new_pos, new_vel = _rk4_step(state.position, state.velocity, acc_fn, dt)
        else:
            acc = self.compute_acceleration(state, forces, mass, dt)
            new_pos, new_vel = _euler_step(state.position, state.velocity, acc, dt)

        # Speed cap
        spd = float(np.linalg.norm(new_vel))
        if spd > self.max_speed:
            new_vel = new_vel * (self.max_speed / spd)

        orient = state.orientation
        if spd > 0.05:
            orient = float(np.arctan2(new_vel[1], new_vel[0]))

        acc_out = (new_vel - state.velocity) / max(dt, 1e-9)

        return PhysicsState(
            position=new_pos,
            velocity=new_vel,
            acceleration=acc_out,
            orientation=orient,
            angular_velocity=(orient - state.orientation) / max(dt, 1e-9),
        )


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

@dataclass
class VelocityConstraint:
    """Limits an entity's speed within [min_speed, max_speed]."""

    entity_id: int
    min_speed: float = 0.0
    max_speed: float = 2.0

    def apply(self, velocity: np.ndarray) -> np.ndarray:
        """Return the constrained velocity vector."""
        spd = float(np.linalg.norm(velocity))
        if spd < 1e-9:
            return velocity
        if spd > self.max_speed:
            return velocity * (self.max_speed / spd)
        if spd < self.min_speed:
            return velocity * (self.min_speed / spd)
        return velocity


@dataclass
class WallConstraint:
    """Prevents an entity from crossing a wall segment."""

    seg_a: np.ndarray
    seg_b: np.ndarray
    restitution: float = 0.0

    def apply(
        self, position: np.ndarray, velocity: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (corrected_position, corrected_velocity)."""
        ab = self.seg_b - self.seg_a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-12:
            return position, velocity
        t = float(np.dot(position - self.seg_a, ab)) / ab_len_sq
        t = max(0.0, min(1.0, t))
        closest = self.seg_a + t * ab
        diff = position - closest
        dist = float(np.linalg.norm(diff))
        if dist >= radius or dist < 1e-12:
            return position, velocity
        normal = diff / dist
        penetration = radius - dist
        new_pos = position + normal * penetration
        vn = float(np.dot(velocity, normal))
        if vn < 0.0:
            new_vel = velocity - (1.0 + self.restitution) * vn * normal
        else:
            new_vel = velocity.copy()
        return new_pos, new_vel


# ---------------------------------------------------------------------------
# SimplePhysics engine
# ---------------------------------------------------------------------------

class SimplePhysics:
    """Physics engine that manages forces, integration, and collision response.

    Parameters
    ----------
    integration_method : str
        ``"euler"`` or ``"rk4"``.
    default_model : str
        ``"kinematic"`` or ``"dynamic"`` – used when an entity has no
        explicit model assigned.
    gravity : float
        Gravitational acceleration constant (used for friction forces only
        in the 2-D plane).
    collision_restitution : float
        Default coefficient of restitution for entity-entity collisions.
    positional_correction_factor : float
        Fraction of penetration resolved per step (Baumgarte-like).
    """

    def __init__(
        self,
        integration_method: str = "euler",
        default_model: str = "kinematic",
        gravity: float = 9.81,
        collision_restitution: float = 0.3,
        positional_correction_factor: float = 0.8,
    ) -> None:
        self.integration_method = integration_method
        self.gravity = gravity
        self.collision_restitution = collision_restitution
        self.positional_correction_factor = positional_correction_factor

        # Per-entity force accumulators: entity_id -> list of (force, label)
        self._forces: Dict[int, List[Tuple[np.ndarray, str]]] = {}

        # Per-entity motion model overrides
        self._models: Dict[int, MotionModel] = {}

        # Default models
        self._kinematic = KinematicModel()
        self._dynamic = DynamicModel()
        self._default_model_name = default_model

        # Velocity constraints
        self._velocity_constraints: Dict[int, VelocityConstraint] = {}

        # Wall constraints
        self._wall_constraints: List[WallConstraint] = []

        # Statistics
        self.total_collisions: int = 0
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def set_model(self, entity_id: int, model: MotionModel) -> None:
        """Override the motion model for a specific entity."""
        self._models[entity_id] = model

    def get_model(self, entity_id: int) -> MotionModel:
        """Return the motion model for an entity (explicit or default)."""
        if entity_id in self._models:
            return self._models[entity_id]
        if self._default_model_name == "dynamic":
            return self._dynamic
        return self._kinematic

    def set_default_kinematic(self, **kwargs: Any) -> None:
        """Replace default kinematic model parameters."""
        self._kinematic = KinematicModel(**kwargs)

    def set_default_dynamic(self, **kwargs: Any) -> None:
        """Replace default dynamic model parameters."""
        self._dynamic = DynamicModel(**kwargs)

    # ------------------------------------------------------------------
    # Force accumulation
    # ------------------------------------------------------------------

    def apply_force(
        self, entity_id: int, force: Sequence[float], label: str = ""
    ) -> None:
        """Add a force to an entity for the current step."""
        f = np.asarray(force, dtype=np.float64)[:2]
        self._forces.setdefault(entity_id, []).append((f, label))

    def clear_forces(self) -> None:
        """Clear all accumulated forces."""
        self._forces.clear()

    def net_force(self, entity_id: int) -> np.ndarray:
        """Return the summed net force for an entity."""
        forces = self._forces.get(entity_id, [])
        if not forces:
            return np.zeros(2)
        return sum(f for f, _ in forces)  # type: ignore[return-value]

    def force_records(self) -> List[ForceRecord]:
        """Return a flat list of all current force records."""
        records: List[ForceRecord] = []
        for eid, flist in self._forces.items():
            for f, label in flist:
                records.append(ForceRecord(eid, f, label))
        return records

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def add_velocity_constraint(
        self, entity_id: int, min_speed: float = 0.0, max_speed: float = 2.0
    ) -> None:
        """Add or update a velocity constraint for an entity."""
        self._velocity_constraints[entity_id] = VelocityConstraint(
            entity_id, min_speed, max_speed
        )

    def remove_velocity_constraint(self, entity_id: int) -> None:
        """Remove velocity constraint for an entity."""
        self._velocity_constraints.pop(entity_id, None)

    def add_wall_constraint(
        self, seg_a: Sequence[float], seg_b: Sequence[float],
        restitution: float = 0.0,
    ) -> None:
        """Add a wall constraint segment."""
        self._wall_constraints.append(
            WallConstraint(
                np.asarray(seg_a, dtype=np.float64)[:2],
                np.asarray(seg_b, dtype=np.float64)[:2],
                restitution,
            )
        )

    def sync_wall_constraints(self, world: World) -> None:
        """Rebuild wall constraints from the world's wall list."""
        self._wall_constraints.clear()
        for a, b in world.walls:
            self._wall_constraints.append(
                WallConstraint(a, b, self.collision_restitution)
            )

    # ------------------------------------------------------------------
    # Integration step
    # ------------------------------------------------------------------

    def step(self, world: World, dt: float) -> List[CollisionResult]:
        """Advance the physics simulation by *dt*.

        1. Integrate each entity with accumulated forces.
        2. Resolve wall constraints.
        3. Detect and respond to entity-entity collisions.
        4. Enforce velocity constraints.
        5. Enforce world boundaries.
        6. Refresh spatial index.

        Returns the list of collision results for the step.
        """
        self.step_count += 1

        # --- 1. Integrate ---
        for eid, edata in world.entities.items():
            if not edata.get("active", True):
                continue
            model = self.get_model(eid)
            state = PhysicsState(
                position=edata["position"].copy(),
                velocity=edata["velocity"].copy(),
                orientation=edata.get("orientation", 0.0),
            )
            net = self.net_force(eid)
            mass = edata.get("mass", 1.0)
            new_state = model.integrate(state, net, mass, dt, self.integration_method)
            edata["position"][:] = new_state.position
            edata["velocity"][:] = new_state.velocity
            edata["orientation"] = new_state.orientation

        # --- 2. Wall constraints ---
        for eid, edata in world.entities.items():
            if not edata.get("active", True):
                continue
            for wc in self._wall_constraints:
                pos, vel = wc.apply(
                    edata["position"], edata["velocity"], edata["radius"]
                )
                edata["position"][:] = pos
                edata["velocity"][:] = vel

        # --- 3. Collision detection & response ---
        collisions = world.detect_entity_collisions()
        self.total_collisions += len(collisions)
        for col in collisions:
            self._resolve_collision(world, col)

        # Also handle wall collisions from the world
        wall_cols = world.detect_wall_collisions()
        self.total_collisions += len(wall_cols)
        for col in wall_cols:
            self._resolve_wall_collision(world, col)

        all_collisions = collisions + wall_cols

        # --- 4. Velocity constraints ---
        for eid, vc in self._velocity_constraints.items():
            if eid in world.entities:
                world.entities[eid]["velocity"] = vc.apply(
                    world.entities[eid]["velocity"]
                )

        # --- 5. Boundaries ---
        world.enforce_boundaries()

        # --- 6. Spatial index ---
        world.refresh_spatial_index()

        return all_collisions

    # ------------------------------------------------------------------
    # Collision response
    # ------------------------------------------------------------------

    def _resolve_collision(self, world: World, col: CollisionResult) -> None:
        """Elastic-like impulse-based collision response between entities."""
        ea = world.entities[col.entity_a_id]
        eb = world.entities[col.entity_b_id]

        ma = ea.get("mass", 1.0)
        mb = eb.get("mass", 1.0)
        inv_total = 1.0 / max(ma + mb, 1e-9)

        # Positional correction (separate overlapping circles)
        correction = col.normal * col.penetration * self.positional_correction_factor
        ea["position"] -= correction * (mb * inv_total)
        eb["position"] += correction * (ma * inv_total)

        # Impulse
        rel_vel = ea["velocity"] - eb["velocity"]
        vn = float(np.dot(rel_vel, col.normal))
        if vn > 0.0:
            return  # separating

        e = self.collision_restitution
        j = -(1.0 + e) * vn * inv_total
        impulse = j * col.normal

        ea["velocity"] += impulse / max(ma, 1e-9)
        eb["velocity"] -= impulse / max(mb, 1e-9)

    def _resolve_wall_collision(self, world: World, col: CollisionResult) -> None:
        """Resolve entity-wall collision with bounce."""
        edata = world.entities[col.entity_a_id]
        # Push out
        edata["position"] += col.normal * col.penetration * self.positional_correction_factor
        # Reflect velocity
        vn = float(np.dot(edata["velocity"], col.normal))
        if vn < 0.0:
            edata["velocity"] -= (1.0 + self.collision_restitution) * vn * col.normal

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset accumulators and statistics."""
        self._forces.clear()
        self.total_collisions = 0
        self.step_count = 0

    def stats(self) -> Dict[str, Any]:
        """Return a summary dict of physics statistics."""
        return {
            "step_count": self.step_count,
            "total_collisions": self.total_collisions,
            "num_force_entries": sum(len(v) for v in self._forces.values()),
            "num_velocity_constraints": len(self._velocity_constraints),
            "num_wall_constraints": len(self._wall_constraints),
        }

    def __repr__(self) -> str:
        return (
            f"SimplePhysics(method={self.integration_method!r}, "
            f"steps={self.step_count}, collisions={self.total_collisions})"
        )
