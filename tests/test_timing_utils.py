"""Tests for navirl.utils.timing — Timer, profile, throttle, rate_limiter, etc."""

from __future__ import annotations

import time

import pytest

from navirl.utils.timing import (
    FrequencyTracker,
    Stopwatch,
    Timer,
    profile,
    rate_limiter,
    throttle,
)

# ---------------------------------------------------------------------------
#  Timer
# ---------------------------------------------------------------------------


class TestTimer:
    def test_context_manager(self):
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed > 0
        assert not t.is_running

    def test_manual_start_stop(self):
        t = Timer("manual")
        t.start()
        assert t.is_running
        time.sleep(0.01)
        elapsed = t.stop()
        assert elapsed > 0
        assert not t.is_running

    def test_stop_without_start(self):
        t = Timer()
        assert t.stop() == 0.0

    def test_elapsed_while_running(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        assert t.elapsed > 0
        assert t.is_running

    def test_elapsed_ms(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_ms > 0
        assert t.elapsed_ms == pytest.approx(t.elapsed * 1000)

    def test_lap(self):
        t = Timer()
        assert t.lap() == 0.0  # not started
        t.start()
        time.sleep(0.01)
        lap1 = t.lap()
        time.sleep(0.01)
        lap2 = t.lap()
        assert lap1 > 0
        assert lap2 > 0
        assert len(t.lap_times) == 2

    def test_reset(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        t.stop()
        t.reset()
        assert t.elapsed == 0.0
        assert not t.is_running
        assert t.lap_times == []

    def test_repr(self):
        t = Timer("my_timer")
        r = repr(t)
        assert "my_timer" in r
        t.start()
        assert "running" in repr(t)
        t.stop()
        assert "s" in repr(t)

    def test_decorate(self):
        @Timer.decorate("decorated_fn", verbose=False)
        def foo(x):
            return x * 2

        assert foo(5) == 10


# ---------------------------------------------------------------------------
#  _Profiler / profile
# ---------------------------------------------------------------------------


class TestProfile:
    def setup_method(self):
        profile.reset()

    def test_context_manager(self):
        with profile("op1"):
            time.sleep(0.01)
        stats = profile.stats("op1")
        assert stats["count"] == 1
        assert stats["total"] > 0

    def test_decorator(self):
        @profile("my_func")
        def compute():
            return 42

        result = compute()
        assert result == 42
        stats = profile.stats("my_func")
        assert stats["count"] == 1

    def test_multiple_calls(self):
        for _ in range(3):
            with profile("repeated"):
                pass
        stats = profile.stats("repeated")
        assert stats["count"] == 3

    def test_unknown_operation(self):
        assert profile.stats("nonexistent") == {}

    def test_all_stats(self):
        with profile("a"):
            pass
        with profile("b"):
            pass
        all_s = profile.all_stats()
        assert "a" in all_s
        assert "b" in all_s

    def test_summary_empty(self):
        s = profile.summary()
        assert "No profiling data" in s

    def test_summary_with_data(self):
        with profile("test_op"):
            pass
        s = profile.summary()
        assert "test_op" in s

    def test_enable_disable(self):
        profile.disable()
        with profile("disabled_op"):
            pass
        stats = profile.stats("disabled_op")
        assert stats == {}
        profile.enable()

    def test_decorator_when_disabled(self):
        profile.disable()

        @profile("disabled_dec")
        def f():
            return 99

        assert f() == 99
        assert profile.stats("disabled_dec") == {}
        profile.enable()


# ---------------------------------------------------------------------------
#  throttle
# ---------------------------------------------------------------------------


class TestThrottle:
    def test_first_call_always_executes(self):
        call_count = [0]

        @throttle(1.0)
        def fn():
            call_count[0] += 1
            return call_count[0]

        assert fn() == 1

    def test_throttled_call_returns_cached(self):
        call_count = [0]

        @throttle(10.0)
        def fn():
            call_count[0] += 1
            return call_count[0]

        fn()  # first call
        result = fn()  # should be throttled
        assert result == 1
        assert call_count[0] == 1

    def test_call_after_interval(self):
        @throttle(0.01)
        def fn():
            return time.perf_counter()

        first = fn()
        time.sleep(0.02)
        second = fn()
        assert second > first


# ---------------------------------------------------------------------------
#  rate_limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = rate_limiter(3, 1.0)
        assert rl.allow() is True
        assert rl.allow() is True
        assert rl.allow() is True

    def test_blocks_over_limit(self):
        rl = rate_limiter(2, 10.0)
        assert rl.allow() is True
        assert rl.allow() is True
        assert rl.allow() is False

    def test_wait_time_zero_when_allowed(self):
        rl = rate_limiter(10, 1.0)
        assert rl.wait_time() == 0.0

    def test_wait_time_positive_when_blocked(self):
        rl = rate_limiter(1, 10.0)
        rl.allow()
        assert rl.wait_time() > 0

    def test_remaining(self):
        rl = rate_limiter(3, 10.0)
        assert rl.remaining == 3
        rl.allow()
        assert rl.remaining == 2

    def test_reset(self):
        rl = rate_limiter(1, 10.0)
        rl.allow()
        assert rl.allow() is False
        rl.reset()
        assert rl.allow() is True


# ---------------------------------------------------------------------------
#  FrequencyTracker
# ---------------------------------------------------------------------------


class TestFrequencyTracker:
    def test_no_ticks(self):
        ft = FrequencyTracker()
        assert ft.frequency == 0.0
        assert ft.count == 0

    def test_single_tick(self):
        ft = FrequencyTracker()
        ft.tick()
        assert ft.count == 1
        assert ft.frequency == 0.0  # need at least 2 ticks

    def test_multiple_ticks(self):
        ft = FrequencyTracker(window=10.0)
        for _ in range(10):
            ft.tick()
            time.sleep(0.001)
        assert ft.frequency > 0
        assert ft.count == 10

    def test_reset(self):
        ft = FrequencyTracker()
        ft.tick()
        ft.tick()
        ft.reset()
        assert ft.count == 0


# ---------------------------------------------------------------------------
#  Stopwatch
# ---------------------------------------------------------------------------


class TestStopwatch:
    def test_basic_flow(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        sw.checkpoint("phase1")
        time.sleep(0.01)
        sw.checkpoint("phase2")
        total = sw.stop()
        assert total > 0
        phases = sw.phase_times
        assert "phase1" in phases
        assert "phase2" in phases
        assert phases["phase1"] > 0
        assert phases["phase2"] > 0

    def test_stop_without_start(self):
        sw = Stopwatch()
        assert sw.stop() == 0.0

    def test_total_elapsed_while_running(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        assert sw.total_elapsed > 0

    def test_total_elapsed_without_start(self):
        sw = Stopwatch()
        assert sw.total_elapsed == 0.0

    def test_phase_times_no_checkpoints(self):
        sw = Stopwatch()
        assert sw.phase_times == {}

    def test_checkpoint_names(self):
        sw = Stopwatch()
        sw.start()
        sw.checkpoint("a")
        sw.checkpoint("b")
        assert sw.checkpoint_names == ["a", "b"]

    def test_summary_no_checkpoints(self):
        sw = Stopwatch()
        assert "No checkpoints" in sw.summary()

    def test_summary_with_checkpoints(self):
        sw = Stopwatch()
        sw.start()
        sw.checkpoint("init")
        sw.stop()
        s = sw.summary()
        assert "init" in s
        assert "Total" in s
