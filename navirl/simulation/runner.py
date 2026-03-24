"""Simulation runner for NavIRL.

Orchestrates :class:`World`, :class:`SimulationClock`, physics,
entities, and the event bus into coherent run-loops.  Supports
synchronous and asynchronous execution, episode management, snapshot /
restore, headless mode, batch simulation, parallel episodes, and
progress reporting.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np

from navirl.simulation.clock import SimulationClock
from navirl.simulation.events import EventBus, EventRecord, EventType
from navirl.simulation.physics import SimplePhysics
from navirl.simulation.world import CollisionResult, World


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Summary of a single simulation episode."""

    episode_id: int = 0
    success: bool = False
    sim_time: float = 0.0
    wall_time: float = 0.0
    steps: int = 0
    total_collisions: int = 0
    events: List[EventRecord] = field(default_factory=list)
    final_positions: Dict[int, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "episode_id": self.episode_id,
            "success": self.success,
            "sim_time": self.sim_time,
            "wall_time": self.wall_time,
            "steps": self.steps,
            "total_collisions": self.total_collisions,
            "events": [e.to_dict() for e in self.events],
            "final_positions": self.final_positions,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------

@dataclass
class ProgressInfo:
    """Information passed to progress callbacks."""

    episode: int
    total_episodes: int
    step: int
    sim_time: float
    wall_time: float
    done: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------

StepCallback = Callable[["SimulationRunner", float], None]
EpisodeCallback = Callable[["SimulationRunner", EpisodeResult], None]
ProgressCallback = Callable[[ProgressInfo], None]


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """High-level simulation orchestrator.

    Parameters
    ----------
    world : World
        The simulation world.
    clock : SimulationClock or None
        Timing controller.  Created with defaults if ``None``.
    physics : SimplePhysics or None
        Physics engine.  Created with defaults if ``None``.
    event_bus : EventBus or None
        Event system.  Created if ``None``.
    headless : bool
        If ``True`` skips any rendering hooks.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        world: World,
        clock: SimulationClock | None = None,
        physics: SimplePhysics | None = None,
        event_bus: EventBus | None = None,
        headless: bool = True,
        seed: int | None = None,
    ) -> None:
        self.world = world
        self.clock = clock or SimulationClock()
        self.physics = physics or SimplePhysics()
        self.event_bus = event_bus or EventBus()
        self.headless = headless
        self.rng = np.random.default_rng(seed)

        # Callbacks
        self._pre_step_callbacks: List[StepCallback] = []
        self._post_step_callbacks: List[StepCallback] = []
        self._episode_end_callbacks: List[EpisodeCallback] = []
        self._progress_callbacks: List[ProgressCallback] = []
        self._termination_conditions: List[Callable[[SimulationRunner], bool]] = []

        # State
        self._running: bool = False
        self._episode_id: int = 0
        self._initial_snapshot: Dict[str, Any] | None = None
        self._snapshots: Dict[str, Dict[str, Any]] = {}

        # Metrics per episode
        self._episode_collisions: int = 0
        self._episode_start_wall: float = 0.0

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_pre_step(self, callback: StepCallback) -> None:
        """Register a callback invoked before each physics step."""
        self._pre_step_callbacks.append(callback)

    def on_post_step(self, callback: StepCallback) -> None:
        """Register a callback invoked after each physics step."""
        self._post_step_callbacks.append(callback)

    def on_episode_end(self, callback: EpisodeCallback) -> None:
        """Register a callback invoked at the end of each episode."""
        self._episode_end_callbacks.append(callback)

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a progress reporting callback."""
        self._progress_callbacks.append(callback)

    def add_termination_condition(
        self, condition: Callable[[SimulationRunner], bool]
    ) -> None:
        """Add a custom termination condition.

        The callable receives the runner and should return ``True`` to
        stop the current episode.
        """
        self._termination_conditions.append(condition)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def save_snapshot(self, name: str = "default") -> None:
        """Save a named snapshot of the current world state."""
        self._snapshots[name] = self.world.snapshot()

    def load_snapshot(self, name: str = "default") -> bool:
        """Restore a named snapshot.  Returns ``True`` if found."""
        snap = self._snapshots.get(name)
        if snap is None:
            return False
        self.world.restore(snap)
        return True

    def save_initial_state(self) -> None:
        """Save the current state as the initial state for :meth:`reset`."""
        self._initial_snapshot = self.world.to_dict()

    def reset(self) -> None:
        """Reset world to initial state (if saved) and reset clock/physics."""
        if self._initial_snapshot is not None:
            cell_size = self.world._grid.cell_size
            self.world = World.from_dict(self._initial_snapshot, cell_size=cell_size)
        self.clock.reset()
        self.physics.reset()
        self.event_bus.clear_history()
        self._episode_collisions = 0

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Execute a single simulation step.

        Returns the timestep used.
        """
        dt = self.clock.tick_fixed()
        if dt <= 0:
            return 0.0

        # Pre-step callbacks
        for cb in self._pre_step_callbacks:
            cb(self, dt)

        # Physics
        collisions = self.physics.step(self.world, dt)
        self._episode_collisions += len(collisions)

        # Emit collision events
        for col in collisions:
            self.event_bus.emit_collision(
                sim_time=self.clock.sim_time,
                entity_a=col.entity_a_id,
                entity_b=col.entity_b_id,
                penetration=col.penetration,
                normal=col.normal.tolist(),
            )

        # Post-step callbacks
        for cb in self._post_step_callbacks:
            cb(self, dt)

        # Clear forces for next step
        self.physics.clear_forces()

        return dt

    # ------------------------------------------------------------------
    # Run loops
    # ------------------------------------------------------------------

    def run(
        self,
        max_steps: int = 0,
        max_time: float = 0.0,
        report_interval: int = 100,
    ) -> EpisodeResult:
        """Run a synchronous episode loop.

        Parameters
        ----------
        max_steps : int
            Stop after this many steps (0 = use clock limits).
        max_time : float
            Stop after this simulation time (0 = use clock limits).
        report_interval : int
            Steps between progress reports.

        Returns
        -------
        EpisodeResult
        """
        self._running = True
        self._episode_id += 1
        self._episode_collisions = 0
        self._episode_start_wall = time.monotonic()
        self.clock.start()

        self.event_bus.publish(
            EventType.EPISODE_START,
            sim_time=self.clock.sim_time,
            data={"episode_id": self._episode_id},
        )

        step_count = 0
        terminated = False

        while self._running and not self.clock.done:
            # Max limits
            if max_steps > 0 and step_count >= max_steps:
                break
            if max_time > 0.0 and self.clock.sim_time >= max_time:
                break

            self.step()
            step_count += 1

            # Custom termination
            for cond in self._termination_conditions:
                if cond(self):
                    terminated = True
                    break
            if terminated:
                break

            # Progress
            if report_interval > 0 and step_count % report_interval == 0:
                self._report_progress(step_count)

        wall_elapsed = time.monotonic() - self._episode_start_wall
        result = self._build_result(step_count, wall_elapsed, terminated)

        self.event_bus.publish(
            EventType.EPISODE_END,
            sim_time=self.clock.sim_time,
            data={"episode_id": self._episode_id, "success": result.success},
        )

        for cb in self._episode_end_callbacks:
            cb(self, result)

        self._running = False
        return result

    async def run_async(
        self,
        max_steps: int = 0,
        max_time: float = 0.0,
        report_interval: int = 100,
        yield_interval: int = 10,
    ) -> EpisodeResult:
        """Asynchronous episode loop (cooperative multitasking).

        Parameters
        ----------
        yield_interval : int
            Yield control every N steps.
        """
        import asyncio

        self._running = True
        self._episode_id += 1
        self._episode_collisions = 0
        self._episode_start_wall = time.monotonic()
        self.clock.start()

        self.event_bus.publish(
            EventType.EPISODE_START,
            sim_time=self.clock.sim_time,
            data={"episode_id": self._episode_id},
        )

        step_count = 0
        terminated = False

        while self._running and not self.clock.done:
            if max_steps > 0 and step_count >= max_steps:
                break
            if max_time > 0.0 and self.clock.sim_time >= max_time:
                break

            self.step()
            step_count += 1

            for cond in self._termination_conditions:
                if cond(self):
                    terminated = True
                    break
            if terminated:
                break

            if report_interval > 0 and step_count % report_interval == 0:
                self._report_progress(step_count)

            if step_count % yield_interval == 0:
                await asyncio.sleep(0)

        wall_elapsed = time.monotonic() - self._episode_start_wall
        result = self._build_result(step_count, wall_elapsed, terminated)

        self.event_bus.publish(
            EventType.EPISODE_END,
            sim_time=self.clock.sim_time,
            data={"episode_id": self._episode_id, "success": result.success},
        )

        for cb in self._episode_end_callbacks:
            cb(self, result)

        self._running = False
        return result

    def stop(self) -> None:
        """Request the current run loop to stop."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def run_episodes(
        self,
        n_episodes: int,
        max_steps_per_episode: int = 1000,
        max_time_per_episode: float = 0.0,
        reset_between: bool = True,
        report_interval: int = 100,
    ) -> List[EpisodeResult]:
        """Run multiple episodes sequentially.

        Parameters
        ----------
        n_episodes : int
            Number of episodes.
        max_steps_per_episode : int
            Step limit per episode.
        max_time_per_episode : float
            Time limit per episode.
        reset_between : bool
            If ``True`` reset the world between episodes.
        report_interval : int
            Steps between progress reports.

        Returns
        -------
        list of EpisodeResult
        """
        if self._initial_snapshot is None:
            self.save_initial_state()

        results: List[EpisodeResult] = []
        for ep in range(n_episodes):
            if reset_between:
                self.reset()
            result = self.run(
                max_steps=max_steps_per_episode,
                max_time=max_time_per_episode,
                report_interval=report_interval,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Batch / parallel
    # ------------------------------------------------------------------

    @staticmethod
    def _run_single_episode(
        world_dict: Dict[str, Any],
        clock_dt: float,
        max_steps: int,
        physics_method: str,
        seed: int,
        episode_id: int,
    ) -> Dict[str, Any]:
        """Worker function for parallel episode execution.

        Reconstructs the world from a dict so it can be pickled across
        process boundaries.
        """
        world = World.from_dict(world_dict)
        clock = SimulationClock(dt=clock_dt)
        physics = SimplePhysics(integration_method=physics_method)
        event_bus = EventBus(recording=False)
        runner = SimulationRunner(
            world=world,
            clock=clock,
            physics=physics,
            event_bus=event_bus,
            headless=True,
            seed=seed,
        )
        result = runner.run(max_steps=max_steps)
        result.episode_id = episode_id
        return result.to_dict()

    def run_parallel(
        self,
        n_episodes: int,
        max_steps_per_episode: int = 1000,
        max_workers: int = 4,
        base_seed: int = 42,
    ) -> List[EpisodeResult]:
        """Run episodes in parallel using a thread pool.

        Parameters
        ----------
        n_episodes : int
            Number of episodes.
        max_steps_per_episode : int
            Step limit per episode.
        max_workers : int
            Maximum parallel workers.
        base_seed : int
            Base seed (each episode gets ``base_seed + i``).

        Returns
        -------
        list of EpisodeResult
        """
        world_dict = self.world.to_dict()
        clock_dt = self.clock.dt
        physics_method = self.physics.integration_method

        futures: List[concurrent.futures.Future[Dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i in range(n_episodes):
                fut = pool.submit(
                    self._run_single_episode,
                    world_dict,
                    clock_dt,
                    max_steps_per_episode,
                    physics_method,
                    base_seed + i,
                    i,
                )
                futures.append(fut)

            results: List[EpisodeResult] = []
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                rd = fut.result()
                er = EpisodeResult(
                    episode_id=rd["episode_id"],
                    success=rd["success"],
                    sim_time=rd["sim_time"],
                    wall_time=rd["wall_time"],
                    steps=rd["steps"],
                    total_collisions=rd["total_collisions"],
                    final_positions=rd["final_positions"],
                    metadata=rd["metadata"],
                )
                results.append(er)
                # Report progress
                for cb in self._progress_callbacks:
                    cb(ProgressInfo(
                        episode=len(results),
                        total_episodes=n_episodes,
                        step=0,
                        sim_time=er.sim_time,
                        wall_time=er.wall_time,
                        done=len(results) == n_episodes,
                        message=f"Episode {er.episode_id} complete",
                    ))

        results.sort(key=lambda r: r.episode_id)
        return results

    # ------------------------------------------------------------------
    # Batch simulation
    # ------------------------------------------------------------------

    def run_batch(
        self,
        configs: List[Dict[str, Any]],
        max_steps: int = 1000,
    ) -> List[EpisodeResult]:
        """Run a batch of episodes with different configurations.

        Each config dict can contain:
        - ``"entities"``: list of entity kwargs to add.
        - ``"walls"``: list of wall segment pairs.
        - ``"metadata"``: dict of metadata.

        Returns list of :class:`EpisodeResult`.
        """
        if self._initial_snapshot is None:
            self.save_initial_state()

        results: List[EpisodeResult] = []
        for idx, cfg in enumerate(configs):
            self.reset()
            # Apply config
            for ent in cfg.get("entities", []):
                self.world.add_entity(**ent)
            for wall in cfg.get("walls", []):
                self.world.add_wall(*wall)
            for k, v in cfg.get("metadata", {}).items():
                self.world.set_metadata(k, v)
            result = self.run(max_steps=max_steps)
            result.episode_id = idx
            result.metadata.update(cfg.get("metadata", {}))
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, steps: int, wall_time: float, terminated: bool
    ) -> EpisodeResult:
        """Construct an :class:`EpisodeResult` from current state."""
        final_pos: Dict[int, List[float]] = {}
        for eid, edata in self.world.entities.items():
            final_pos[eid] = edata["position"].tolist()

        return EpisodeResult(
            episode_id=self._episode_id,
            success=terminated,
            sim_time=self.clock.sim_time,
            wall_time=wall_time,
            steps=steps,
            total_collisions=self._episode_collisions,
            events=list(self.event_bus.history),
            final_positions=final_pos,
        )

    def _report_progress(self, step_count: int) -> None:
        """Fire progress callbacks."""
        info = ProgressInfo(
            episode=self._episode_id,
            total_episodes=self._episode_id,
            step=step_count,
            sim_time=self.clock.sim_time,
            wall_time=time.monotonic() - self._episode_start_wall,
        )
        for cb in self._progress_callbacks:
            cb(info)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        return self.clock.sim_time

    @property
    def step_count(self) -> int:
        return self.clock.step_count

    def entity_positions(self, kind: str | None = None) -> np.ndarray:
        """Return (N, 2) array of entity positions."""
        return self.world.positions_array(kind)

    def entity_velocities(self, kind: str | None = None) -> np.ndarray:
        """Return (N, 2) array of entity velocities."""
        return self.world.velocities_array(kind)

    def stats(self) -> Dict[str, Any]:
        """Return combined statistics from clock, physics, and events."""
        return {
            "clock": self.clock.stats().as_dict(),
            "physics": self.physics.stats(),
            "events": self.event_bus.stats(),
            "episode_id": self._episode_id,
            "episode_collisions": self._episode_collisions,
        }

    def __repr__(self) -> str:
        return (
            f"SimulationRunner(world={self.world!r}, "
            f"t={self.clock.sim_time:.3f}, "
            f"running={self._running})"
        )
