"""Timing and profiling utilities for NavIRL.

Provides context managers, decorators, and classes for measuring
execution time, throttling function calls, and tracking performance.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Context manager and reusable timer for measuring elapsed time.

    Can be used as a context manager, a manual start/stop timer,
    or a decorator.

    Examples
    --------
    As a context manager:

    >>> with Timer("my_op") as t:
    ...     time.sleep(0.01)
    >>> t.elapsed > 0
    True

    As a manual timer:

    >>> t = Timer("manual")
    >>> t.start()
    >>> time.sleep(0.01)
    >>> t.stop()
    >>> t.elapsed > 0
    True

    As a decorator:

    >>> @Timer.decorate("my_func")
    ... def foo():
    ...     pass
    """

    def __init__(self, name: str = "timer", verbose: bool = False) -> None:
        self.name = name
        self.verbose = verbose
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._elapsed: float = 0.0
        self._lap_times: list[float] = []
        self._running = False

    def start(self) -> Timer:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._running = True
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        if self._start_time is None:
            return 0.0
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        self._running = False
        if self.verbose:
            print(f"[Timer:{self.name}] {self._elapsed:.6f}s")
        return self._elapsed

    def lap(self) -> float:
        """Record a lap time without stopping the timer.

        Returns
        -------
        float
            Time since the last lap (or start).
        """
        if self._start_time is None:
            return 0.0
        now = time.perf_counter()
        last = self._lap_times[-1] if self._lap_times else 0.0
        total_elapsed = now - self._start_time
        lap_time = total_elapsed - last
        self._lap_times.append(total_elapsed)
        return lap_time

    def reset(self) -> None:
        """Reset the timer."""
        self._start_time = None
        self._end_time = None
        self._elapsed = 0.0
        self._lap_times.clear()
        self._running = False

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds.

        If the timer is still running, returns time since start.
        """
        if self._running and self._start_time is not None:
            return time.perf_counter() - self._start_time
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000.0

    @property
    def lap_times(self) -> list[float]:
        """List of cumulative lap times."""
        return list(self._lap_times)

    @property
    def is_running(self) -> bool:
        """Whether the timer is currently running."""
        return self._running

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self._running else f"{self._elapsed:.6f}s"
        return f"Timer(name={self.name!r}, {status})"

    @staticmethod
    def decorate(name: str | None = None, verbose: bool = True) -> Callable:
        """Create a decorator that times function execution.

        Parameters
        ----------
        name : str, optional
            Timer name.  Defaults to the function name.
        verbose : bool
            If True, print elapsed time.

        Returns
        -------
        Callable
            Decorator.
        """

        def decorator(func: Callable) -> Callable:
            timer_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with Timer(timer_name, verbose=verbose):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@dataclass
class _ProfileEntry:
    """Statistics for a single profiled operation."""

    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    last_time: float = 0.0

    @property
    def avg_time(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


class _Profiler:
    """Global profiler that tracks timing for named operations.

    Used by the ``profile`` decorator/context manager.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ProfileEntry] = defaultdict(_ProfileEntry)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def record(self, name: str, elapsed: float) -> None:
        """Record a timing measurement."""
        entry = self._entries[name]
        entry.total_time += elapsed
        entry.call_count += 1
        entry.last_time = elapsed
        entry.min_time = min(entry.min_time, elapsed)
        entry.max_time = max(entry.max_time, elapsed)

    def get_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a named operation.

        Returns
        -------
        dict
            Statistics including total, avg, min, max, count.
        """
        entry = self._entries.get(name)
        if entry is None:
            return {}
        return {
            "total": entry.total_time,
            "avg": entry.avg_time,
            "min": entry.min_time if entry.min_time < float("inf") else 0.0,
            "max": entry.max_time,
            "count": entry.call_count,
            "last": entry.last_time,
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all profiled operations."""
        return {name: self.get_stats(name) for name in self._entries}

    def summary(self) -> str:
        """Generate a summary table of all profiled operations.

        Returns
        -------
        str
            Formatted summary table.
        """
        if not self._entries:
            return "No profiling data."

        lines = [
            f"{'Operation':<30} {'Calls':>8} {'Total(s)':>10} "
            f"{'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}",
            "-" * 80,
        ]

        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].total_time,
            reverse=True,
        )

        for name, entry in sorted_entries:
            lines.append(
                f"{name:<30} {entry.call_count:>8} "
                f"{entry.total_time:>10.4f} "
                f"{entry.avg_time * 1000:>10.4f} "
                f"{entry.min_time * 1000:>10.4f} "
                f"{entry.max_time * 1000:>10.4f}"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all profiling data."""
        self._entries.clear()


# Global profiler instance
_profiler = _Profiler()


class profile:
    """Context manager and decorator for profiling operations.

    Records timing data to the global profiler.

    Examples
    --------
    As a context manager:

    >>> with profile("matrix_multiply"):
    ...     result = np.dot(a, b)

    As a decorator:

    >>> @profile("compute_features")
    ... def compute(data):
    ...     return process(data)

    Getting results:

    >>> profile.stats("matrix_multiply")
    {'total': 0.123, 'avg': 0.041, ...}
    >>> print(profile.summary())
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._start: float = 0.0

    def __enter__(self) -> profile:
        if _profiler.enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if _profiler.enabled:
            elapsed = time.perf_counter() - self._start
            _profiler.record(self._name, elapsed)

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator."""
        name = self._name

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _profiler.enabled:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                _profiler.record(name, elapsed)

        return wrapper

    @staticmethod
    def stats(name: str) -> dict[str, float]:
        """Get profiling stats for an operation."""
        return _profiler.get_stats(name)

    @staticmethod
    def all_stats() -> dict[str, dict[str, float]]:
        """Get all profiling stats."""
        return _profiler.get_all_stats()

    @staticmethod
    def summary() -> str:
        """Get formatted summary of all profiling data."""
        return _profiler.summary()

    @staticmethod
    def reset() -> None:
        """Reset all profiling data."""
        _profiler.reset()

    @staticmethod
    def enable() -> None:
        """Enable profiling."""
        _profiler.enabled = True

    @staticmethod
    def disable() -> None:
        """Disable profiling."""
        _profiler.enabled = False


# ---------------------------------------------------------------------------
# Throttle / rate limiter
# ---------------------------------------------------------------------------


def throttle(min_interval: float) -> Callable:
    """Decorator that throttles function calls to a minimum interval.

    If called before the interval has elapsed, returns the previous
    result without calling the function.

    Parameters
    ----------
    min_interval : float
        Minimum time between calls in seconds.

    Returns
    -------
    Callable
        Decorator.

    Examples
    --------
    >>> @throttle(0.1)
    ... def expensive_computation(x):
    ...     return x * 2
    """

    def decorator(func: Callable) -> Callable:
        last_call_time: list[float] = [0.0]
        last_result: list[Any] = [None]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.perf_counter()
            if now - last_call_time[0] >= min_interval:
                last_result[0] = func(*args, **kwargs)
                last_call_time[0] = now
            return last_result[0]

        wrapper.reset = lambda: last_call_time.__setitem__(0, 0.0)  # type: ignore[attr-defined]
        return wrapper

    return decorator


class rate_limiter:
    """Rate limiter that allows at most N calls per time window.

    Parameters
    ----------
    max_calls : int
        Maximum number of calls per window.
    window : float
        Time window in seconds.

    Examples
    --------
    >>> limiter = rate_limiter(10, 1.0)  # 10 calls per second
    >>> if limiter.allow():
    ...     do_something()
    """

    def __init__(self, max_calls: int, window: float) -> None:
        self.max_calls = max_calls
        self.window = window
        # Use deque for O(1) operations at both ends
        self._call_times: deque[float] = deque()

    def _prune_expired(self, now: float) -> None:
        """Remove expired entries from the front of the deque.

        This is more efficient than list comprehension as it only removes
        elements from the front until we hit a non-expired entry.
        """
        cutoff = now - self.window
        # Remove expired entries from front (O(k) where k = expired entries)
        while self._call_times and self._call_times[0] <= cutoff:
            self._call_times.popleft()

    def allow(self) -> bool:
        """Check if a call is allowed and record it if so.

        Returns
        -------
        bool
            True if the call is allowed.
        """
        now = time.perf_counter()
        self._prune_expired(now)

        if len(self._call_times) < self.max_calls:
            self._call_times.append(now)  # O(1) append to deque
            return True
        return False

    def wait_time(self) -> float:
        """Time to wait before the next call is allowed.

        Returns
        -------
        float
            Seconds to wait (0.0 if a call is allowed now).
        """
        now = time.perf_counter()
        self._prune_expired(now)

        if len(self._call_times) < self.max_calls:
            return 0.0

        # After pruning, the first entry is the oldest valid call
        oldest = self._call_times[0]
        return max(0.0, oldest + self.window - now)

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._call_times.clear()

    @property
    def remaining(self) -> int:
        """Number of remaining calls allowed in current window."""
        now = time.perf_counter()
        self._prune_expired(now)
        return max(0, self.max_calls - len(self._call_times))


# ---------------------------------------------------------------------------
# Frequency tracker
# ---------------------------------------------------------------------------


class FrequencyTracker:
    """Track the frequency (rate) of events.

    Useful for monitoring frame rates, update frequencies, etc.

    Parameters
    ----------
    window : float
        Time window for frequency computation in seconds.

    Examples
    --------
    >>> tracker = FrequencyTracker(window=1.0)
    >>> for _ in range(100):
    ...     tracker.tick()
    >>> freq = tracker.frequency  # ~events per second
    """

    def __init__(self, window: float = 1.0) -> None:
        self.window = window
        # Use deque for O(1) operations at both ends
        self._ticks: deque[float] = deque()

    def _prune_expired(self, now: float) -> None:
        """Remove expired entries from the front of the deque.

        This is more efficient than list comprehension as it only removes
        elements from the front until we hit a non-expired entry.
        """
        cutoff = now - self.window
        # Remove expired entries from front (O(k) where k = expired entries)
        while self._ticks and self._ticks[0] <= cutoff:
            self._ticks.popleft()

    def tick(self) -> None:
        """Record an event occurrence."""
        now = time.perf_counter()
        self._prune_expired(now)
        self._ticks.append(now)  # O(1) append to deque

    @property
    def frequency(self) -> float:
        """Current event frequency in events per second."""
        if not self._ticks:
            return 0.0
        now = time.perf_counter()
        self._prune_expired(now)

        if len(self._ticks) < 2:
            return 0.0
        duration = self._ticks[-1] - self._ticks[0]
        if duration < 1e-9:
            return 0.0
        return (len(self._ticks) - 1) / duration

    @property
    def count(self) -> int:
        """Number of events in the current window."""
        now = time.perf_counter()
        self._prune_expired(now)
        return len(self._ticks)

    def reset(self) -> None:
        """Clear all recorded ticks."""
        self._ticks.clear()


# ---------------------------------------------------------------------------
# Stopwatch with checkpoints
# ---------------------------------------------------------------------------


class Stopwatch:
    """Stopwatch with named checkpoints for multi-phase timing.

    Useful for profiling multi-step operations where you want to
    track time spent in each phase.

    Examples
    --------
    >>> sw = Stopwatch()
    >>> sw.start()
    >>> # ... phase 1 ...
    >>> sw.checkpoint("phase1")
    >>> # ... phase 2 ...
    >>> sw.checkpoint("phase2")
    >>> sw.stop()
    >>> sw.phase_times
    {'phase1': 0.xxx, 'phase2': 0.yyy}
    """

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._checkpoints: list[tuple[str, float]] = []
        self._end_time: float | None = None

    def start(self) -> Stopwatch:
        """Start the stopwatch."""
        self._start_time = time.perf_counter()
        self._checkpoints.clear()
        self._end_time = None
        return self

    def checkpoint(self, name: str) -> float:
        """Record a named checkpoint.

        Parameters
        ----------
        name : str
            Checkpoint name.

        Returns
        -------
        float
            Time since the last checkpoint (or start).
        """
        now = time.perf_counter()
        self._checkpoints.append((name, now))

        if len(self._checkpoints) == 1:
            return now - (self._start_time or now)
        return now - self._checkpoints[-2][1]

    def stop(self) -> float:
        """Stop the stopwatch.

        Returns
        -------
        float
            Total elapsed time.
        """
        self._end_time = time.perf_counter()
        if self._start_time is None:
            return 0.0
        return self._end_time - self._start_time

    @property
    def total_elapsed(self) -> float:
        """Total elapsed time from start to stop (or now)."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @property
    def phase_times(self) -> dict[str, float]:
        """Time spent in each phase between checkpoints."""
        if not self._checkpoints or self._start_time is None:
            return {}

        result = {}
        prev_time = self._start_time

        for name, cp_time in self._checkpoints:
            result[name] = cp_time - prev_time
            prev_time = cp_time

        return result

    @property
    def checkpoint_names(self) -> list[str]:
        """Names of all checkpoints."""
        return [name for name, _ in self._checkpoints]

    def summary(self) -> str:
        """Generate a summary of phase times.

        Returns
        -------
        str
            Formatted summary.
        """
        phases = self.phase_times
        if not phases:
            return "No checkpoints recorded."

        total = self.total_elapsed
        lines = [f"Total: {total:.6f}s"]
        for name, duration in phases.items():
            pct = (duration / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {duration:.6f}s ({pct:.1f}%)")

        return "\n".join(lines)
