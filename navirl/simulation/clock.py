"""Simulation clock for NavIRL.

Provides :class:`SimulationClock` which manages fixed-timestep and
variable-timestep modes, real-time synchronisation, pause / resume,
time scaling, priority-queue event scheduling, frame-rate control, and
timing statistics.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Scheduled event
# ---------------------------------------------------------------------------


@dataclass(order=True)
class ScheduledEvent:
    """An event to fire at a specific simulation time.

    Events are ordered by *sim_time* then *priority* (lower = higher
    priority) then insertion order.
    """

    sim_time: float
    priority: int = field(compare=True, default=0)
    _seq: int = field(compare=True, default=0)
    callback: Callable[..., None] = field(compare=False, default=lambda: None)
    name: str = field(compare=False, default="")
    repeat_interval: float = field(compare=False, default=0.0)
    data: Any = field(compare=False, default=None)
    cancelled: bool = field(compare=False, default=False)


# ---------------------------------------------------------------------------
# Frame statistics
# ---------------------------------------------------------------------------


@dataclass
class ClockStats:
    """Aggregate timing statistics."""

    sim_time: float = 0.0
    wall_time: float = 0.0
    step_count: int = 0
    real_time_ratio: float = 0.0
    avg_step_wall_ms: float = 0.0
    min_step_wall_ms: float = float("inf")
    max_step_wall_ms: float = 0.0
    total_paused_time: float = 0.0
    events_fired: int = 0

    def as_dict(self) -> dict[str, float]:
        """Return a plain dict representation."""
        return {
            "sim_time": self.sim_time,
            "wall_time": self.wall_time,
            "step_count": self.step_count,
            "real_time_ratio": self.real_time_ratio,
            "avg_step_wall_ms": self.avg_step_wall_ms,
            "min_step_wall_ms": self.min_step_wall_ms,
            "max_step_wall_ms": self.max_step_wall_ms,
            "total_paused_time": self.total_paused_time,
            "events_fired": self.events_fired,
        }


# ---------------------------------------------------------------------------
# SimulationClock
# ---------------------------------------------------------------------------


class SimulationClock:
    """Central clock governing simulation timing.

    Parameters
    ----------
    dt : float
        Default fixed timestep (seconds).
    max_sim_time : float
        Maximum simulation time before auto-stop.
    time_scale : float
        Multiplier for elapsed wall time when in real-time mode.
    real_time : bool
        If ``True`` the clock synchronises simulation time with wall
        time (scaled by *time_scale*).
    max_steps : int
        Maximum number of steps (0 = unlimited).
    target_fps : float
        Target frame rate when using :meth:`wait_for_frame`.
    """

    def __init__(
        self,
        dt: float = 0.01,
        max_sim_time: float = float("inf"),
        time_scale: float = 1.0,
        real_time: bool = False,
        max_steps: int = 0,
        target_fps: float = 60.0,
    ) -> None:
        # Core timing
        self._dt: float = max(dt, 1e-9)
        self._sim_time: float = 0.0
        self._step: int = 0
        self._max_sim_time: float = max_sim_time
        self._max_steps: int = max_steps
        self._time_scale: float = max(time_scale, 0.0)
        self._real_time: bool = real_time

        # Pause state
        self._paused: bool = False
        self._total_paused: float = 0.0
        self._pause_start: float = 0.0

        # Wall-clock tracking
        self._wall_start: float = 0.0
        self._wall_last_step: float = 0.0
        self._started: bool = False

        # Frame rate control
        self._target_fps: float = max(target_fps, 1.0)
        self._frame_start: float = 0.0

        # Variable timestep accumulator
        self._accumulator: float = 0.0

        # Event scheduling (priority queue)
        self._events: list[ScheduledEvent] = []
        self._event_seq: int = 0

        # Per-step wall-time measurements
        self._step_wall_times: list[float] = []
        self._fired_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Current timestep."""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = max(value, 1e-9)

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds."""
        return self._sim_time

    @property
    def step_count(self) -> int:
        """Number of completed steps."""
        return self._step

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        self._time_scale = max(value, 0.0)

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def done(self) -> bool:
        """Return ``True`` when the clock has reached a termination condition."""
        if self._sim_time >= self._max_sim_time:
            return True
        return 0 < self._max_steps <= self._step

    @property
    def wall_elapsed(self) -> float:
        """Wall-clock seconds since :meth:`start`, excluding paused time."""
        if not self._started:
            return 0.0
        raw = time.monotonic() - self._wall_start
        return raw - self._total_paused

    @property
    def real_time_ratio(self) -> float:
        """Ratio of simulation time to wall time (>1 = faster than real-time)."""
        w = self.wall_elapsed
        if w < 1e-9:
            return 0.0
        return self._sim_time / w

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise wall-clock tracking.  Call once before the run loop."""
        self._wall_start = time.monotonic()
        self._wall_last_step = self._wall_start
        self._frame_start = self._wall_start
        self._started = True

    def reset(self) -> None:
        """Reset the clock to time zero."""
        self._sim_time = 0.0
        self._step = 0
        self._paused = False
        self._total_paused = 0.0
        self._accumulator = 0.0
        self._step_wall_times.clear()
        self._fired_count = 0
        self._started = False
        self._events.clear()
        self._event_seq = 0

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the clock."""
        if not self._paused:
            self._paused = True
            self._pause_start = time.monotonic()

    def resume(self) -> None:
        """Resume the clock."""
        if self._paused:
            self._total_paused += time.monotonic() - self._pause_start
            self._paused = False

    def toggle_pause(self) -> bool:
        """Toggle pause state; return new paused flag."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    # ------------------------------------------------------------------
    # Tick / step
    # ------------------------------------------------------------------

    def tick(self) -> float:
        """Advance one fixed-timestep tick.

        Returns the timestep used.  In real-time mode the method may
        return a variable dt corresponding to actual wall-clock elapsed.
        """
        if self._paused:
            return 0.0

        now = time.monotonic()

        if self._real_time:
            raw_dt = (now - self._wall_last_step) * self._time_scale
            # Clamp to avoid spiral-of-death
            dt = min(raw_dt, self._dt * 10.0)
        else:
            dt = self._dt * self._time_scale

        self._sim_time += dt
        self._step += 1
        step_wall = (now - self._wall_last_step) * 1000.0
        self._step_wall_times.append(step_wall)
        self._wall_last_step = now

        # Fire scheduled events
        self._process_events()

        return dt

    def tick_fixed(self) -> float:
        """Advance exactly one fixed timestep (ignores real-time mode).

        Returns dt.
        """
        if self._paused:
            return 0.0
        dt = self._dt
        self._sim_time += dt
        self._step += 1
        now = time.monotonic()
        step_wall = (now - self._wall_last_step) * 1000.0
        self._step_wall_times.append(step_wall)
        self._wall_last_step = now
        self._process_events()
        return dt

    def tick_variable(self) -> float:
        """Advance using wall-clock elapsed time (variable timestep).

        Clamps dt to at most 10x the nominal fixed timestep to prevent
        instability.
        """
        if self._paused:
            return 0.0
        now = time.monotonic()
        raw_dt = (now - self._wall_last_step) * self._time_scale
        dt = min(max(raw_dt, 1e-9), self._dt * 10.0)
        self._sim_time += dt
        self._step += 1
        step_wall = (now - self._wall_last_step) * 1000.0
        self._step_wall_times.append(step_wall)
        self._wall_last_step = now
        self._process_events()
        return dt

    def accumulate_and_step(self) -> int:
        """Accumulate wall-time and return how many fixed sub-steps to run.

        Typical usage for a semi-fixed timestep game loop::

            n = clock.accumulate_and_step()
            for _ in range(n):
                physics.step(world, clock.dt)
        """
        now = time.monotonic()
        frame_dt = (now - self._wall_last_step) * self._time_scale
        frame_dt = min(frame_dt, self._dt * 20.0)
        self._accumulator += frame_dt
        self._wall_last_step = now
        steps = 0
        while self._accumulator >= self._dt:
            self._accumulator -= self._dt
            self._sim_time += self._dt
            self._step += 1
            steps += 1
            self._process_events()
        return steps

    # ------------------------------------------------------------------
    # Frame-rate control
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Mark the start of a render/logic frame."""
        self._frame_start = time.monotonic()

    def wait_for_frame(self) -> float:
        """Sleep until the target frame time has elapsed.

        Returns the actual frame duration in seconds.
        """
        target = 1.0 / self._target_fps
        elapsed = time.monotonic() - self._frame_start
        remaining = target - elapsed
        if remaining > 0:
            time.sleep(remaining)
        actual = time.monotonic() - self._frame_start
        return actual

    @property
    def target_fps(self) -> float:
        return self._target_fps

    @target_fps.setter
    def target_fps(self, value: float) -> None:
        self._target_fps = max(value, 1.0)

    # ------------------------------------------------------------------
    # Event scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        sim_time: float,
        callback: Callable[..., None],
        name: str = "",
        priority: int = 0,
        repeat_interval: float = 0.0,
        data: Any = None,
    ) -> ScheduledEvent:
        """Schedule a callback at a future simulation time.

        Parameters
        ----------
        sim_time : float
            Simulation time at which to fire.
        callback : callable
            Function to call.  Receives the :class:`ScheduledEvent` as
            its sole argument.
        name : str
            Human-readable label.
        priority : int
            Lower values fire first at equal sim_time.
        repeat_interval : float
            If > 0 the event re-schedules itself at ``sim_time +
            repeat_interval``.
        data : Any
            Arbitrary payload.

        Returns
        -------
        ScheduledEvent
        """
        seq = self._event_seq
        self._event_seq += 1
        evt = ScheduledEvent(
            sim_time=sim_time,
            priority=priority,
            _seq=seq,
            callback=callback,
            name=name,
            repeat_interval=repeat_interval,
            data=data,
        )
        heapq.heappush(self._events, evt)
        return evt

    def schedule_after(
        self,
        delay: float,
        callback: Callable[..., None],
        name: str = "",
        priority: int = 0,
        repeat_interval: float = 0.0,
        data: Any = None,
    ) -> ScheduledEvent:
        """Schedule a callback *delay* seconds from now (sim time)."""
        return self.schedule(
            self._sim_time + delay, callback, name, priority, repeat_interval, data
        )

    def cancel_event(self, event: ScheduledEvent) -> None:
        """Cancel a scheduled event (lazy removal)."""
        event.cancelled = True

    def pending_events(self) -> int:
        """Return count of non-cancelled pending events."""
        return sum(1 for e in self._events if not e.cancelled)

    def _process_events(self) -> None:
        """Fire all events whose sim_time <= current sim_time."""
        while self._events and self._events[0].sim_time <= self._sim_time:
            evt = heapq.heappop(self._events)
            if evt.cancelled:
                continue
            evt.callback(evt)
            self._fired_count += 1
            if evt.repeat_interval > 0.0:
                self.schedule(
                    evt.sim_time + evt.repeat_interval,
                    evt.callback,
                    evt.name,
                    evt.priority,
                    evt.repeat_interval,
                    evt.data,
                )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> ClockStats:
        """Return aggregate timing statistics."""
        n = len(self._step_wall_times)
        avg = sum(self._step_wall_times) / n if n else 0.0
        mn = min(self._step_wall_times) if n else float("inf")
        mx = max(self._step_wall_times) if n else 0.0
        return ClockStats(
            sim_time=self._sim_time,
            wall_time=self.wall_elapsed,
            step_count=self._step,
            real_time_ratio=self.real_time_ratio,
            avg_step_wall_ms=avg,
            min_step_wall_ms=mn,
            max_step_wall_ms=mx,
            total_paused_time=self._total_paused,
            events_fired=self._fired_count,
        )

    def reset_stats(self) -> None:
        """Clear per-step timing history (does **not** reset sim time)."""
        self._step_wall_times.clear()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "paused" if self._paused else "running"
        return (
            f"SimulationClock(t={self._sim_time:.4f}, step={self._step}, "
            f"dt={self._dt}, state={state})"
        )
