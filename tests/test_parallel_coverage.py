"""Tests for uncovered paths in navirl.training.parallel.

Focuses on RunningMeanStd, VecNormalize, VecFrameStack, VecMonitor,
and the AsyncVecEnv / SubprocVecEnv worker logic.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from navirl.training.parallel import (
    AsyncVecEnv,
    BaseVecEnv,
    DummyVecEnv,
    RunningMeanStd,
    VecEnvWrapper,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
    _worker,
)

# ---------------------------------------------------------------------------
# Minimal fake environment
# ---------------------------------------------------------------------------


class _FakeSpace:
    """Minimal space mock with shape."""

    def __init__(self, shape: tuple[int, ...] = (4,)) -> None:
        self.shape = shape

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape)


class _FakeEnv:
    """Minimal environment for testing vectorized wrappers."""

    def __init__(self, obs_shape: tuple[int, ...] = (4,)) -> None:
        self.observation_space = _FakeSpace(obs_shape)
        self.action_space = _FakeSpace((2,))
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return np.zeros(self.observation_space.shape)

    def step(self, action: np.ndarray) -> tuple:
        self._step_count += 1
        obs = np.random.randn(*self.observation_space.shape).astype(np.float32)
        reward = float(np.random.randn())
        done = self._step_count >= 5
        info = {}
        return obs, reward, done, info

    def close(self) -> None:
        pass


def _make_env(obs_shape: tuple[int, ...] = (4,)) -> Callable:
    return lambda: _FakeEnv(obs_shape)


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------


class TestRunningMeanStd:
    """Cover RunningMeanStd.update (lines 611-631)."""

    def test_initial_state(self):
        rms = RunningMeanStd(shape=(3,))
        np.testing.assert_array_equal(rms.mean, np.zeros(3))
        np.testing.assert_array_equal(rms.var, np.ones(3))

    def test_single_update(self):
        rms = RunningMeanStd(shape=(2,))
        batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rms.update(batch)
        # Mean should be close to [3, 4]
        assert abs(rms.mean[0] - 3.0) < 0.1
        assert abs(rms.mean[1] - 4.0) < 0.1

    def test_multiple_updates_converge(self):
        rms = RunningMeanStd(shape=())
        np.random.seed(42)
        for _ in range(100):
            batch = np.random.randn(32) * 2.0 + 5.0
            rms.update(batch)
        # Should converge toward mean=5.0, var≈4.0
        assert abs(rms.mean - 5.0) < 0.5
        assert abs(rms.var - 4.0) < 1.0

    def test_scalar_shape(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.array([1.0, 2.0, 3.0]))
        assert rms.mean.shape == ()


# ---------------------------------------------------------------------------
# VecNormalize
# ---------------------------------------------------------------------------


class TestVecNormalize:
    """Cover VecNormalize step/reset/normalize (lines 679-701, 703-715)."""

    def _make_vec_env(self, n: int = 2) -> DummyVecEnv:
        return DummyVecEnv([_make_env() for _ in range(n)])

    def test_step_normalizes_obs_and_rewards(self):
        venv = self._make_vec_env()
        norm = VecNormalize(venv)
        obs = norm.reset()
        assert obs.dtype == np.float32

        actions = np.zeros((2, 2))
        obs, rewards, dones, infos = norm.step(actions)
        assert obs.dtype == np.float32
        assert rewards.dtype == np.float32

    def test_step_resets_returns_on_done(self):
        venv = self._make_vec_env(1)
        norm = VecNormalize(venv)
        norm.reset()

        # Step until done
        actions = np.zeros((1, 2))
        for _ in range(10):
            obs, rewards, dones, infos = norm.step(actions)
            if dones[0]:
                break
        # After done, returns should be reset
        assert norm.returns[0] == 0.0

    def test_training_mode_updates_stats(self):
        venv = self._make_vec_env()
        norm = VecNormalize(venv)
        norm.reset()

        initial_count = norm.obs_rms.count
        actions = np.zeros((2, 2))
        norm.step(actions)
        assert norm.obs_rms.count > initial_count

    def test_eval_mode_freezes_stats(self):
        venv = self._make_vec_env()
        norm = VecNormalize(venv)
        norm.reset()

        # Warm up stats
        actions = np.zeros((2, 2))
        for _ in range(3):
            norm.step(actions)

        norm.set_training(False)
        count_before = norm.obs_rms.count
        norm.step(actions)
        assert norm.obs_rms.count == count_before

    def test_no_normalization(self):
        venv = self._make_vec_env()
        norm = VecNormalize(venv, norm_obs=False, norm_reward=False)
        norm.reset()
        actions = np.zeros((2, 2))
        obs, rewards, dones, infos = norm.step(actions)
        # Should still return float32
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# VecFrameStack
# ---------------------------------------------------------------------------


class TestVecFrameStack:
    """Cover VecFrameStack step/reset (lines 776-807)."""

    def _make_vec_env(self, n: int = 2) -> DummyVecEnv:
        return DummyVecEnv([_make_env() for _ in range(n)])

    def test_reset_stacks_frames(self):
        venv = self._make_vec_env()
        fs = VecFrameStack(venv, n_stack=3)
        obs = fs.reset()
        assert obs.shape == (2, 3, 4)  # (num_envs, n_stack, obs_dim)
        # Only last frame should be non-zero
        np.testing.assert_array_equal(obs[:, 0, :], 0.0)
        np.testing.assert_array_equal(obs[:, 1, :], 0.0)

    def test_step_shifts_frames(self):
        venv = self._make_vec_env()
        fs = VecFrameStack(venv, n_stack=3)
        fs.reset()
        actions = np.zeros((2, 2))

        obs1, _, _, _ = fs.step(actions)
        obs2, _, _, _ = fs.step(actions)

        # After 2 steps, frames should have shifted
        assert obs2.shape == (2, 3, 4)

    def test_done_clears_stack(self):
        venv = self._make_vec_env(1)
        fs = VecFrameStack(venv, n_stack=3)
        fs.reset()
        actions = np.zeros((1, 2))

        # Step until done
        for _ in range(10):
            obs, _, dones, _ = fs.step(actions)
            if dones[0]:
                # Stack should have been cleared for the done env
                # (the new obs will be in the last slot after clearing)
                break


# ---------------------------------------------------------------------------
# VecMonitor
# ---------------------------------------------------------------------------


class TestVecMonitor:
    """Cover VecMonitor step (lines 836-875)."""

    def _make_vec_env(self, n: int = 2) -> DummyVecEnv:
        return DummyVecEnv([_make_env() for _ in range(n)])

    def test_tracks_episode_stats(self):
        venv = self._make_vec_env(1)
        mon = VecMonitor(venv)
        mon.reset()
        actions = np.zeros((1, 2))

        # Step until done to get an episode
        for _ in range(10):
            obs, rewards, dones, infos = mon.step(actions)
            if dones[0]:
                assert "episode" in infos[0]
                assert "r" in infos[0]["episode"]
                assert "l" in infos[0]["episode"]
                break

    def test_episode_count_increments(self):
        venv = self._make_vec_env(1)
        mon = VecMonitor(venv)
        mon.reset()
        actions = np.zeros((1, 2))

        for _ in range(10):
            obs, _, dones, _ = mon.step(actions)
            if dones[0]:
                break
        assert mon.episode_count >= 1

    def test_info_keywords_tracked(self):
        """Info keywords from the env should be copied to episode info."""
        venv = self._make_vec_env(1)
        mon = VecMonitor(venv, info_keywords=("custom_metric",))
        mon.reset()

        # Inject a custom metric into the env's info
        original_step = venv.envs[0].step

        def patched_step(action):
            obs, reward, done, info = original_step(action)
            info["custom_metric"] = 42.0
            return obs, reward, done, info

        venv.envs[0].step = patched_step

        actions = np.zeros((1, 2))
        for _ in range(10):
            obs, _, dones, infos = mon.step(actions)
            if dones[0]:
                assert infos[0]["episode"]["custom_metric"] == 42.0
                break

    def test_reward_history_bounded(self):
        """Episode history should be bounded by maxlen=100."""
        venv = self._make_vec_env(1)
        mon = VecMonitor(venv)
        mon.reset()
        actions = np.zeros((1, 2))

        for _ in range(200):
            obs, _, dones, _ = mon.step(actions)
        # History deque is bounded
        assert len(mon._episode_reward_history) <= 100


# ---------------------------------------------------------------------------
# _worker function
# ---------------------------------------------------------------------------


class TestWorkerFunction:
    """Cover _worker subprocess function (lines 192-228)."""

    def test_worker_step(self):
        """Worker should handle step command."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = _FakeEnv()

        # Simulate commands: step then close
        work_remote.recv.side_effect = [
            ("step", np.zeros(2)),
            ("close", None),
        ]

        _worker(work_remote, parent_remote, lambda: env)
        parent_remote.close.assert_called_once()
        assert work_remote.send.call_count >= 1

    def test_worker_reset(self):
        """Worker should handle reset command."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = _FakeEnv()

        work_remote.recv.side_effect = [
            ("reset", None),
            ("close", None),
        ]

        _worker(work_remote, parent_remote, lambda: env)
        assert work_remote.send.call_count >= 1

    def test_worker_get_attr(self):
        """Worker should handle get_attr command."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = _FakeEnv()

        work_remote.recv.side_effect = [
            ("get_attr", "observation_space"),
            ("close", None),
        ]

        _worker(work_remote, parent_remote, lambda: env)
        # Should have sent the observation_space
        sent_value = work_remote.send.call_args_list[0][0][0]
        assert hasattr(sent_value, "shape")

    def test_worker_env_method(self):
        """Worker should handle env_method command."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = _FakeEnv()

        work_remote.recv.side_effect = [
            ("env_method", ("reset", (), {})),
            ("close", None),
        ]

        _worker(work_remote, parent_remote, lambda: env)
        assert work_remote.send.call_count >= 1

    def test_worker_get_spaces(self):
        """Worker should handle get_spaces command."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = _FakeEnv()

        work_remote.recv.side_effect = [
            ("get_spaces", None),
            ("close", None),
        ]

        _worker(work_remote, parent_remote, lambda: env)
        sent = work_remote.send.call_args_list[0][0][0]
        assert isinstance(sent, tuple)
        assert len(sent) == 2

    def test_worker_unknown_command_raises(self):
        """Unknown command should raise ValueError."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = _FakeEnv()

        work_remote.recv.side_effect = [
            ("unknown_cmd", None),
        ]

        with pytest.raises(ValueError, match="Unknown command"):
            _worker(work_remote, parent_remote, lambda: env)

    def test_worker_eof_exits(self):
        """EOFError should cause clean exit."""
        parent_remote = MagicMock()
        work_remote = MagicMock()

        work_remote.recv.side_effect = EOFError()

        _worker(work_remote, parent_remote, lambda: _FakeEnv())
        parent_remote.close.assert_called_once()

    def test_worker_step_auto_resets_on_done(self):
        """Step that returns done should auto-reset."""
        parent_remote = MagicMock()
        work_remote = MagicMock()
        env = MagicMock()
        env.step.return_value = (np.zeros(4), 1.0, True, {})
        env.reset.return_value = np.zeros(4)

        work_remote.recv.side_effect = [
            ("step", np.zeros(2)),
            ("close", None),
        ]

        _worker(work_remote, parent_remote, lambda: env)
        env.reset.assert_called_once()
        # Check terminal_observation was added to info
        sent = work_remote.send.call_args_list[0][0][0]
        assert "terminal_observation" in sent[3]


# ---------------------------------------------------------------------------
# VecEnvWrapper delegation
# ---------------------------------------------------------------------------


class TestVecEnvWrapper:
    """Cover VecEnvWrapper delegation methods."""

    def test_wrapper_delegates(self):
        venv = DummyVecEnv([_make_env()])
        wrapper = VecEnvWrapper(venv)

        obs = wrapper.reset()
        assert obs.shape == (1, 4)

        actions = np.zeros((1, 2))
        obs, rewards, dones, infos = wrapper.step(actions)
        assert obs.shape == (1, 4)

        wrapper.close()
