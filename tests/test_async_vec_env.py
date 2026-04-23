"""End-to-end tests for navirl.training.parallel.AsyncVecEnv and SubprocVecEnv.

Spawns real subprocesses (via the ``fork`` start method on Linux) to exercise
the inter-process step / reset / poll paths that pure-mock tests miss.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest

from navirl.training.parallel import AsyncVecEnv, SubprocVecEnv

# ---------------------------------------------------------------------------
# Picklable env / env_fn definitions (must be top-level for spawn semantics)
# ---------------------------------------------------------------------------


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _SimpleEnv:
    """Minimal env where step returns deterministic values per (instance, step)."""

    def __init__(self, episode_len: int = 4):
        self.observation_space = _Space((3,))
        self.action_space = _Space((2,))
        self._t = 0
        self._episode_len = episode_len

    def reset(self):
        self._t = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action):
        self._t += 1
        obs = np.full(3, float(self._t), dtype=np.float32)
        reward = float(self._t)
        done = self._t >= self._episode_len
        info: dict = {"step": self._t}
        return obs, reward, done, info

    def close(self):
        pass

    # Used by SubprocVecEnv.env_method
    def add(self, a, b):
        return a + b


class _SlowEnv(_SimpleEnv):
    """Env that sleeps before returning the step result (used to test poll())."""

    def step(self, action):
        time.sleep(0.05)
        return super().step(action)


def _make_simple_env():
    return _SimpleEnv(episode_len=4)


def _make_slow_env():
    return _SlowEnv(episode_len=4)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _start_method() -> str:
    """Pick a fast start method available on this platform."""
    available = mp.get_all_start_methods()
    if "fork" in available:
        return "fork"
    return "spawn"


# ---------------------------------------------------------------------------
# SubprocVecEnv — real-process round trip
# ---------------------------------------------------------------------------


class TestSubprocVecEnv:
    def test_reset_returns_stacked_obs(self):
        venv = SubprocVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            obs = venv.reset()
            assert obs.shape == (2, 3)
            np.testing.assert_array_equal(obs, 0.0)
        finally:
            venv.close()

    def test_step_returns_obs_rewards_dones_infos(self):
        venv = SubprocVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((2, 2), dtype=np.float32)
            obs, rewards, dones, infos = venv.step(actions)
            assert obs.shape == (2, 3)
            assert rewards.shape == (2,)
            assert dones.shape == (2,)
            assert len(infos) == 2
            # rewards = step count = 1 after first step
            np.testing.assert_array_equal(rewards, [1.0, 1.0])
            assert not dones.any()
        finally:
            venv.close()

    def test_step_auto_resets_when_done(self):
        venv = SubprocVecEnv([_make_simple_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((1, 2), dtype=np.float32)
            for _ in range(3):
                venv.step(actions)
            obs, rewards, dones, infos = venv.step(actions)
            assert dones[0]
            assert "terminal_observation" in infos[0]
            # After auto-reset, the returned obs should be the post-reset obs (zeros)
            np.testing.assert_array_equal(obs[0], 0.0)
        finally:
            venv.close()

    def test_get_attr_returns_each_env_attr(self):
        venv = SubprocVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            shapes = venv.get_attr("observation_space")
            assert len(shapes) == 2
            for s in shapes:
                assert s.shape == (3,)
        finally:
            venv.close()

    def test_env_method_calls_method_per_env(self):
        venv = SubprocVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            results = venv.env_method("add", 2, b=5)
            assert results == [7, 7]
        finally:
            venv.close()

    def test_close_when_step_pending(self):
        """``close()`` must drain any in-flight step results before sending close."""
        venv = SubprocVecEnv([_make_simple_env], start_method=_start_method())
        venv.reset()
        actions = np.zeros((1, 2), dtype=np.float32)
        venv.step_async(actions)
        # Don't call step_wait — close must clean up the pending recv
        venv.close()
        assert venv.closed

    def test_close_is_idempotent(self):
        venv = SubprocVecEnv([_make_simple_env], start_method=_start_method())
        venv.close()
        # Second close should be a no-op
        venv.close()
        assert venv.closed

    def test_default_start_method_is_chosen(self):
        """Passing ``start_method=None`` resolves to forkserver/spawn automatically."""
        venv = SubprocVecEnv([_make_simple_env], start_method=None)
        try:
            obs = venv.reset()
            assert obs.shape == (1, 3)
        finally:
            venv.close()


# ---------------------------------------------------------------------------
# AsyncVecEnv — full surface
# ---------------------------------------------------------------------------


class TestAsyncVecEnv:
    def test_reset_returns_stacked_obs(self):
        venv = AsyncVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            obs = venv.reset()
            assert obs.shape == (2, 3)
            np.testing.assert_array_equal(obs, 0.0)
            # All pending flags cleared
            assert not any(venv._pending)
        finally:
            venv.close()

    def test_step_async_then_step_wait(self):
        venv = AsyncVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((2, 2), dtype=np.float32)
            venv.step_async(actions)
            assert all(venv._pending)
            obs, rewards, dones, infos = venv.step_wait()
            assert obs.shape == (2, 3)
            np.testing.assert_array_equal(rewards, [1.0, 1.0])
            assert not any(venv._pending)
        finally:
            venv.close()

    def test_step_returns_full_tuple(self):
        venv = AsyncVecEnv([_make_simple_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((1, 2), dtype=np.float32)
            obs, rewards, dones, infos = venv.step(actions)
            assert obs.shape == (1, 3)
            assert rewards.dtype == np.float32
            assert dones.dtype == np.bool_
        finally:
            venv.close()

    def test_step_wait_reuses_last_result_when_not_pending(self):
        """If step_wait is called for an env that hasn't had step_async, prior result is reused."""
        venv = AsyncVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((2, 2), dtype=np.float32)
            # First full step populates _last_results for both envs
            venv.step_async(actions)
            obs1, _, _, _ = venv.step_wait()

            # Now only step env 0 — env 1's pending stays False, so step_wait
            # should reuse the previous result for env 1.
            venv.step_env(0, actions[0])
            assert venv._pending[0] is True
            assert venv._pending[1] is False
            obs2, rewards2, _, _ = venv.step_wait()
            # env 1's result is the cached one (same step count = 1)
            assert rewards2[1] == 1.0
            # env 0 advanced to step 2
            assert rewards2[0] == 2.0
        finally:
            venv.close()

    def test_recv_env_for_single_env(self):
        venv = AsyncVecEnv([_make_simple_env, _make_simple_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((2, 2), dtype=np.float32)
            venv.step_env(0, actions[0])
            assert venv._pending[0] is True
            obs, reward, done, info = venv.recv_env(0)
            assert reward == 1.0
            assert venv._pending[0] is False
            # _last_results updated for env 0
            assert venv._last_results[0] is not None
        finally:
            venv.close()

    def test_poll_returns_ready_envs(self):
        """``poll()`` returns indices of envs whose step result is ready."""
        venv = AsyncVecEnv([_make_simple_env, _make_slow_env], start_method=_start_method())
        try:
            venv.reset()
            actions = np.zeros((2, 2), dtype=np.float32)
            venv.step_async(actions)
            # The fast env should finish first; poll a few times until it appears
            ready: list[int] = []
            for _ in range(40):
                ready = venv.poll()
                if 0 in ready:
                    break
                time.sleep(0.01)
            assert 0 in ready
            # Drain all pending results before close
            venv.step_wait()
        finally:
            venv.close()

    def test_close_with_pending_drains(self):
        """``close()`` must drain pending step results before sending close."""
        venv = AsyncVecEnv([_make_simple_env], start_method=_start_method())
        venv.reset()
        actions = np.zeros((1, 2), dtype=np.float32)
        venv.step_async(actions)
        # Do not call step_wait — close should drain the pending recv
        venv.close()
        assert venv.closed

    def test_close_is_idempotent(self):
        venv = AsyncVecEnv([_make_simple_env], start_method=_start_method())
        venv.close()
        venv.close()
        assert venv.closed

    def test_default_start_method_is_chosen(self):
        venv = AsyncVecEnv([_make_simple_env], start_method=None)
        try:
            obs = venv.reset()
            assert obs.shape == (1, 3)
        finally:
            venv.close()
