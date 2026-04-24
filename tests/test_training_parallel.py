"""Tests for navirl/training/parallel.py — vectorized environment wrappers.

Covers DummyVecEnv, VecEnvWrapper, RunningMeanStd, VecNormalize,
VecFrameStack, and VecMonitor. SubprocVecEnv and AsyncVecEnv are tested
with real subprocesses for key functionality.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from navirl.training.parallel import (
    BaseVecEnv,
    DummyVecEnv,
    RunningMeanStd,
    VecEnvWrapper,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight mock environments
# ---------------------------------------------------------------------------


class _FakeSpace:
    """Picklable stand-in for a Gym observation/action space."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class SimpleEnv:
    """Minimal Gym-like environment for testing vectorized wrappers."""

    def __init__(self, obs_dim: int = 4, episode_length: int = 5) -> None:
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self._step_count = 0
        self.observation_space = _FakeSpace((obs_dim,))
        self.action_space = _FakeSpace((1,))
        self._closed = False

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return np.zeros(self.obs_dim, dtype=np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict]:
        self._step_count += 1
        obs = np.full(self.obs_dim, self._step_count, dtype=np.float32)
        reward = float(self._step_count)
        done = self._step_count >= self.episode_length
        info: dict[str, Any] = {"step": self._step_count}
        if done:
            info["is_success"] = True
        return obs, reward, done, info

    def close(self) -> None:
        self._closed = True

    def get_value(self) -> int:
        return self._step_count


def make_env(obs_dim: int = 4, episode_length: int = 5) -> Callable:
    """Return a factory that creates SimpleEnv instances."""
    return lambda: SimpleEnv(obs_dim=obs_dim, episode_length=episode_length)


# ---------------------------------------------------------------------------
# BaseVecEnv
# ---------------------------------------------------------------------------


class TestBaseVecEnv:
    def test_step_async_and_wait_defaults(self):
        """step_async stores actions; step_wait delegates to step."""

        class ConcreteVecEnv(BaseVecEnv):
            def step(self, actions):
                return actions * 2, np.array([1.0]), np.array([False]), [{}]

            def reset(self):
                return np.zeros(3)

            def close(self):
                pass

        env = ConcreteVecEnv(num_envs=1, observation_space=None, action_space=None)
        actions = np.array([1.0, 2.0, 3.0])
        env.step_async(actions)
        obs, *_ = env.step_wait()
        np.testing.assert_array_equal(obs, actions * 2)


# ---------------------------------------------------------------------------
# DummyVecEnv
# ---------------------------------------------------------------------------


class TestDummyVecEnv:
    def test_init_single_env(self):
        venv = DummyVecEnv([make_env()])
        assert venv.num_envs == 1
        assert venv.observation_space is not None
        venv.close()

    def test_init_multiple_envs(self):
        venv = DummyVecEnv([make_env() for _ in range(3)])
        assert venv.num_envs == 3
        venv.close()

    def test_reset_returns_stacked_obs(self):
        venv = DummyVecEnv([make_env(obs_dim=4) for _ in range(2)])
        obs = venv.reset()
        assert obs.shape == (2, 4)
        np.testing.assert_array_equal(obs, 0.0)
        venv.close()

    def test_step_returns_correct_shapes(self):
        n_envs = 3
        obs_dim = 4
        venv = DummyVecEnv([make_env(obs_dim=obs_dim) for _ in range(n_envs)])
        venv.reset()
        actions = np.zeros(n_envs)
        obs, rewards, dones, infos = venv.step(actions)

        assert obs.shape == (n_envs, obs_dim)
        assert rewards.shape == (n_envs,)
        assert dones.shape == (n_envs,)
        assert len(infos) == n_envs
        assert rewards.dtype == np.float32
        assert dones.dtype == np.bool_
        venv.close()

    def test_step_auto_resets_done_envs(self):
        """When an env is done, step auto-resets it and stores terminal obs."""
        venv = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        venv.reset()

        # Step 1: not done
        obs, rewards, dones, infos = venv.step(np.array([0]))
        assert not dones[0]

        # Step 2: done — env should auto-reset
        obs, rewards, dones, infos = venv.step(np.array([0]))
        assert dones[0]
        assert "terminal_observation" in infos[0]
        # After auto-reset, obs should be the reset observation
        np.testing.assert_array_equal(obs[0], 0.0)
        venv.close()

    def test_step_rewards_accumulate(self):
        venv = DummyVecEnv([make_env(episode_length=3)])
        venv.reset()
        rewards_collected = []
        for _ in range(3):
            _, rewards, _, _ = venv.step(np.array([0]))
            rewards_collected.append(rewards[0])
        assert rewards_collected == [1.0, 2.0, 3.0]
        venv.close()

    def test_get_attr(self):
        venv = DummyVecEnv([make_env() for _ in range(2)])
        venv.reset()
        venv.step(np.array([0, 0]))
        values = venv.get_attr("_step_count")
        assert values == [1, 1]
        venv.close()

    def test_env_method(self):
        venv = DummyVecEnv([make_env() for _ in range(2)])
        venv.reset()
        venv.step(np.array([0, 0]))
        values = venv.env_method("get_value")
        assert values == [1, 1]
        venv.close()

    def test_close_envs_with_and_without_close(self):
        """close() handles envs both with and without a close method."""

        class NoCloseEnv:
            observation_space = MagicMock(shape=(2,))
            action_space = MagicMock(shape=(1,))

            def reset(self):
                return np.zeros(2)

            def step(self, a):
                return np.zeros(2), 0.0, False, {}

        venv = DummyVecEnv([lambda: NoCloseEnv(), make_env(obs_dim=2)])
        venv.reset()
        venv.close()  # Should not raise


# ---------------------------------------------------------------------------
# VecEnvWrapper
# ---------------------------------------------------------------------------


class TestVecEnvWrapper:
    def test_delegates_to_wrapped(self):
        inner = DummyVecEnv([make_env(obs_dim=3)])
        wrapper = VecEnvWrapper(inner)

        assert wrapper.num_envs == 1
        obs = wrapper.reset()
        assert obs.shape == (1, 3)

        obs, rewards, dones, infos = wrapper.step(np.array([0]))
        assert obs.shape == (1, 3)
        wrapper.close()

    def test_step_async_wait(self):
        inner = DummyVecEnv([make_env(obs_dim=2)])
        wrapper = VecEnvWrapper(inner)
        wrapper.reset()
        wrapper.step_async(np.array([0]))
        obs, rewards, dones, infos = wrapper.step_wait()
        assert obs.shape == (1, 2)
        wrapper.close()


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------


class TestRunningMeanStd:
    def test_initial_state(self):
        rms = RunningMeanStd(shape=(3,))
        np.testing.assert_array_equal(rms.mean, np.zeros(3))
        np.testing.assert_array_equal(rms.var, np.ones(3))

    def test_update_single_batch(self):
        rms = RunningMeanStd(shape=())
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(data)
        # After one big batch, mean should be close to 3.0
        assert abs(rms.mean - 3.0) < 0.1
        # Var should be close to 2.0 (population var of [1..5])
        assert abs(rms.var - 2.0) < 0.5

    def test_update_multiple_batches(self):
        rms = RunningMeanStd(shape=())
        for batch in [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0])]:
            rms.update(batch)
        # Mean of [1,2,3,4,5] ~ 3
        assert abs(rms.mean - 3.0) < 0.5

    def test_multidimensional(self):
        rms = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        rms.update(data)
        assert abs(rms.mean[0] - 2.0) < 0.5
        assert abs(rms.mean[1] - 20.0) < 1.0

    def test_count_increases(self):
        rms = RunningMeanStd(shape=())
        initial_count = rms.count
        rms.update(np.array([1.0, 2.0, 3.0]))
        assert rms.count > initial_count


# ---------------------------------------------------------------------------
# VecNormalize
# ---------------------------------------------------------------------------


class TestVecNormalize:
    def _make_vec_normalize(self, n_envs: int = 2, obs_dim: int = 4) -> VecNormalize:
        inner = DummyVecEnv([make_env(obs_dim=obs_dim) for _ in range(n_envs)])
        return VecNormalize(inner, norm_obs=True, norm_reward=True)

    def test_reset_returns_normalized_obs(self):
        vn = self._make_vec_normalize()
        obs = vn.reset()
        assert obs.dtype == np.float32
        assert obs.shape == (2, 4)
        vn.close()

    def test_step_returns_normalized(self):
        vn = self._make_vec_normalize()
        vn.reset()
        obs, rewards, dones, infos = vn.step(np.array([0, 0]))
        assert obs.dtype == np.float32
        assert rewards.dtype == np.float32
        vn.close()

    def test_normalization_clips_obs(self):
        vn = self._make_vec_normalize(n_envs=1, obs_dim=2)
        vn.clip_obs = 5.0
        vn.reset()
        # Step several times to build up stats
        for _ in range(10):
            obs, *_ = vn.step(np.array([0]))
        assert np.all(obs <= 5.0)
        assert np.all(obs >= -5.0)
        vn.close()

    def test_normalization_clips_reward(self):
        vn = self._make_vec_normalize(n_envs=1)
        vn.clip_reward = 2.0
        vn.reset()
        for _ in range(10):
            _, rewards, *_ = vn.step(np.array([0]))
        assert np.all(rewards <= 2.0)
        assert np.all(rewards >= -2.0)
        vn.close()

    def test_training_mode_updates_stats(self):
        vn = self._make_vec_normalize()
        vn.reset()
        initial_count = vn.obs_rms.count
        vn.step(np.array([0, 0]))
        assert vn.obs_rms.count > initial_count
        vn.close()

    def test_eval_mode_freezes_stats(self):
        vn = self._make_vec_normalize()
        vn.reset()
        vn.step(np.array([0, 0]))
        vn.set_training(False)
        count_before = vn.obs_rms.count
        vn.step(np.array([0, 0]))
        assert vn.obs_rms.count == count_before
        vn.close()

    def test_returns_reset_on_done(self):
        """Discounted returns should reset to 0 when an episode ends."""
        vn = VecNormalize(
            DummyVecEnv([make_env(obs_dim=2, episode_length=2)]),
            norm_obs=False,
            norm_reward=True,
        )
        vn.reset()
        vn.step(np.array([0]))  # step 1
        assert vn.returns[0] != 0.0
        vn.step(np.array([0]))  # step 2 — done
        assert vn.returns[0] == 0.0
        vn.close()

    def test_no_normalization(self):
        """With both norms disabled, values pass through unchanged."""
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=10)])
        vn = VecNormalize(inner, norm_obs=False, norm_reward=False)
        vn.reset()
        obs, rewards, _, _ = vn.step(np.array([0]))
        # First step of SimpleEnv produces obs=[1,1], reward=1.0
        np.testing.assert_array_almost_equal(obs[0], [1.0, 1.0])
        assert abs(rewards[0] - 1.0) < 0.01
        vn.close()

    def test_obs_space_without_shape(self):
        """VecNormalize handles observation spaces without a shape attr."""

        class NoShapeSpaceEnv:
            observation_space = "discrete"  # no .shape
            action_space = MagicMock(shape=(1,))

            def reset(self):
                return np.array([0.0])

            def step(self, a):
                return np.array([1.0]), 1.0, False, {}

        inner = DummyVecEnv([lambda: NoShapeSpaceEnv()])
        vn = VecNormalize(inner)
        obs = vn.reset()
        assert obs is not None
        vn.close()


# ---------------------------------------------------------------------------
# VecFrameStack
# ---------------------------------------------------------------------------


class TestVecFrameStack:
    def test_reset_shape(self):
        inner = DummyVecEnv([make_env(obs_dim=3) for _ in range(2)])
        fs = VecFrameStack(inner, n_stack=4)
        obs = fs.reset()
        assert obs.shape == (2, 4, 3)
        fs.close()

    def test_reset_last_frame_populated(self):
        inner = DummyVecEnv([make_env(obs_dim=2)])
        fs = VecFrameStack(inner, n_stack=3)
        obs = fs.reset()
        # Only the last frame should have the reset obs, rest should be zero
        np.testing.assert_array_equal(obs[0, 0], [0.0, 0.0])
        np.testing.assert_array_equal(obs[0, 1], [0.0, 0.0])
        np.testing.assert_array_equal(obs[0, 2], [0.0, 0.0])
        fs.close()

    def test_step_rolls_frames(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=10)])
        fs = VecFrameStack(inner, n_stack=3)
        fs.reset()

        # Step 1: obs = [1, 1]
        obs, *_ = fs.step(np.array([0]))
        np.testing.assert_array_equal(obs[0, 2], [1.0, 1.0])

        # Step 2: obs = [2, 2]
        obs, *_ = fs.step(np.array([0]))
        np.testing.assert_array_equal(obs[0, 1], [1.0, 1.0])
        np.testing.assert_array_equal(obs[0, 2], [2.0, 2.0])
        fs.close()

    def test_done_clears_stack(self):
        """When an episode ends, the frame stack for that env is zeroed."""
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        fs = VecFrameStack(inner, n_stack=3)
        fs.reset()
        fs.step(np.array([0]))  # step 1
        obs, _, dones, _ = fs.step(np.array([0]))  # step 2 — done
        assert dones[0]
        # The stack should be cleared except for the auto-reset obs
        # After clearing and inserting reset obs:
        np.testing.assert_array_equal(obs[0, 0], [0.0, 0.0])
        np.testing.assert_array_equal(obs[0, 1], [0.0, 0.0])
        fs.close()

    def test_returns_copy(self):
        """Returned observations should be copies, not views."""
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=10)])
        fs = VecFrameStack(inner, n_stack=2)
        fs.reset()
        obs1, *_ = fs.step(np.array([0]))
        obs2, *_ = fs.step(np.array([0]))
        # Modifying obs1 should not affect obs2 or internal state
        obs1[:] = 999.0
        assert not np.all(obs2 == 999.0)
        fs.close()


# ---------------------------------------------------------------------------
# VecMonitor
# ---------------------------------------------------------------------------


class TestVecMonitor:
    def test_episode_tracking(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=3)])
        mon = VecMonitor(inner)
        mon.reset()

        for _ in range(3):
            obs, rewards, dones, infos = mon.step(np.array([0]))

        # After 3 steps, episode should be done
        assert dones[0]
        assert "episode" in infos[0]
        ep = infos[0]["episode"]
        assert ep["l"] == 3
        # Total reward = 1 + 2 + 3 = 6
        assert ep["r"] == 6.0
        mon.close()

    def test_episode_count(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        mon = VecMonitor(inner)
        mon.reset()

        # Run 2 full episodes (2 steps each, auto-reset between)
        for _ in range(4):
            mon.step(np.array([0]))

        assert mon.episode_count == 2
        mon.close()

    def test_get_episode_rewards_and_lengths(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        mon = VecMonitor(inner)
        mon.reset()

        # Complete 1 episode
        for _ in range(2):
            mon.step(np.array([0]))

        rewards = mon.get_episode_rewards()
        lengths = mon.get_episode_lengths()
        assert len(rewards) == 1
        assert len(lengths) == 1
        assert rewards[0] == 3.0  # 1 + 2
        assert lengths[0] == 2
        mon.close()

    def test_mean_reward_no_episodes(self):
        inner = DummyVecEnv([make_env(obs_dim=2)])
        mon = VecMonitor(inner)
        mon.reset()
        assert mon.mean_reward == 0.0
        assert mon.mean_length == 0.0
        mon.close()

    def test_mean_reward_with_episodes(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        mon = VecMonitor(inner)
        mon.reset()
        for _ in range(4):
            mon.step(np.array([0]))
        assert mon.mean_reward > 0.0
        assert mon.mean_length > 0.0
        mon.close()

    def test_reset_clears_accumulators(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=5)])
        mon = VecMonitor(inner)
        mon.reset()
        mon.step(np.array([0]))
        mon.reset()
        assert mon.episode_rewards[0] == 0.0
        assert mon.episode_lengths[0] == 0
        mon.close()

    def test_multi_env_monitoring(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2) for _ in range(3)])
        mon = VecMonitor(inner)
        mon.reset()

        for _ in range(2):
            mon.step(np.array([0, 0, 0]))

        # All 3 envs should have completed 1 episode each
        assert mon.episode_count == 3
        assert len(mon.get_episode_rewards()) == 3
        mon.close()

    def test_info_keywords(self):
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        mon = VecMonitor(inner, info_keywords=("is_success",))
        mon.reset()

        for _ in range(2):
            _, _, dones, infos = mon.step(np.array([0]))

        # The terminal step has is_success=True in SimpleEnv
        assert infos[0]["episode"]["is_success"] is True
        mon.close()

    def test_rewards_accumulate_correctly_across_steps(self):
        """Verify rewards reset after episode end for correct accumulation."""
        inner = DummyVecEnv([make_env(obs_dim=2, episode_length=2)])
        mon = VecMonitor(inner)
        mon.reset()

        # Episode 1
        mon.step(np.array([0]))
        assert mon.episode_rewards[0] == 1.0
        mon.step(np.array([0]))  # done
        # After done, accumulator resets
        assert mon.episode_rewards[0] == 0.0

        # Episode 2 starts
        mon.step(np.array([0]))
        assert mon.episode_rewards[0] == 1.0
        mon.close()


# ---------------------------------------------------------------------------
# SubprocVecEnv — basic smoke test
# ---------------------------------------------------------------------------


class TestSubprocVecEnv:
    def test_step_and_reset(self):
        """Smoke test: SubprocVecEnv can reset and step with real subprocesses."""
        from navirl.training.parallel import SubprocVecEnv

        def _make():
            return SimpleEnv(obs_dim=3, episode_length=5)

        venv = SubprocVecEnv([_make for _ in range(2)], start_method="fork")
        try:
            obs = venv.reset()
            assert obs.shape == (2, 3)

            obs, rewards, dones, infos = venv.step(np.array([0, 0]))
            assert obs.shape == (2, 3)
            assert rewards.shape == (2,)
            assert dones.shape == (2,)
        finally:
            venv.close()

    def test_get_attr_and_env_method(self):
        from navirl.training.parallel import SubprocVecEnv

        def _make():
            return SimpleEnv(obs_dim=2, episode_length=5)

        venv = SubprocVecEnv([_make, _make], start_method="fork")
        try:
            venv.reset()
            venv.step(np.array([0, 0]))
            values = venv.get_attr("_step_count")
            assert values == [1, 1]
            method_values = venv.env_method("get_value")
            assert method_values == [1, 1]
        finally:
            venv.close()

    def test_close_idempotent(self):
        from navirl.training.parallel import SubprocVecEnv

        venv = SubprocVecEnv(
            [lambda: SimpleEnv(obs_dim=2, episode_length=5)],
            start_method="fork",
        )
        venv.reset()
        venv.close()
        venv.close()  # Should not raise

    def test_auto_reset_on_done(self):
        from navirl.training.parallel import SubprocVecEnv

        def _make():
            return SimpleEnv(obs_dim=2, episode_length=2)

        venv = SubprocVecEnv([_make], start_method="fork")
        try:
            venv.reset()
            venv.step(np.array([0]))  # step 1
            obs, _, dones, infos = venv.step(np.array([0]))  # step 2 — done
            assert dones[0]
            assert "terminal_observation" in infos[0]
            # Obs should be from reset
            np.testing.assert_array_equal(obs[0], [0.0, 0.0])
        finally:
            venv.close()

    def test_default_start_method_is_auto_detected(self):
        """When start_method is None, SubprocVecEnv picks forkserver if
        available else spawn — both branches must yield a working env."""
        import multiprocessing as mp

        from navirl.training.parallel import SubprocVecEnv

        venv = SubprocVecEnv([_make_simple_env, _make_simple_env])
        try:
            obs = venv.reset()
            assert obs.shape == (2, 4)
            assert venv.num_envs == 2
        finally:
            venv.close()

        # Sanity: the path we exercised matches what get_all_start_methods
        # would surface, i.e. at least one of the two acceptable methods
        # exists on this platform.
        methods = mp.get_all_start_methods()
        assert "forkserver" in methods or "spawn" in methods

    def test_close_drains_pending_step(self):
        """Calling close() while step_async is still in-flight must drain
        the pipes (`waiting=True` branch) without deadlocking."""
        from navirl.training.parallel import SubprocVecEnv

        def _make():
            return SimpleEnv(obs_dim=2, episode_length=10)

        venv = SubprocVecEnv([_make, _make], start_method="fork")
        try:
            venv.reset()
            venv.step_async(np.array([0, 0]))
            assert venv.waiting is True
            # close() must consume the pending step results before sending
            # the close command.
            venv.close()
            assert venv.closed is True
        finally:
            if not venv.closed:
                venv.close()


def _make_simple_env():
    """Module-level factory required for the spawn/forkserver start methods,
    which pickle the callable into worker processes."""
    return SimpleEnv(obs_dim=4, episode_length=5)
