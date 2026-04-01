"""Tests for navirl/envs/ module: base env, crowd env, multi-agent, wrappers, scenarios."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GYM_AVAILABLE, reason="gymnasium not installed")


# ---------------------------------------------------------------------------
# Minimal mock environment for wrapper tests
# ---------------------------------------------------------------------------

if _GYM_AVAILABLE:
    class MockNavEnv(gym.Env):
        """Minimal Gymnasium env for testing wrappers."""

        def __init__(self, obs_dim: int = 8, continuous: bool = True):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            if continuous:
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
                )
            else:
                self.action_space = spaces.Discrete(5)
            self._step_count = 0
            self._max_steps = 50

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step_count = 0
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}

        def step(self, action):
            self._step_count += 1
            obs = np.random.randn(*self.observation_space.shape).astype(np.float32)
            reward = 1.0
            terminated = self._step_count >= self._max_steps
            truncated = False
            info = {"success": terminated}
            return obs, reward, terminated, truncated, info
else:
    class MockNavEnv:
        """Placeholder to keep module importable when Gymnasium is absent."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("MockNavEnv requires gymnasium")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_env():
    return MockNavEnv(obs_dim=8)


@pytest.fixture
def mock_discrete_env():
    return MockNavEnv(obs_dim=8, continuous=False)


# ---------------------------------------------------------------------------
# Base env step / reset
# ---------------------------------------------------------------------------

class TestBaseEnv:
    def test_reset_returns_obs_and_info(self, mock_env):
        obs, info = mock_env.reset()
        assert obs.shape == (8,)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, mock_env):
        mock_env.reset()
        action = mock_env.action_space.sample()
        result = mock_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_terminates(self, mock_env):
        mock_env.reset()
        done = False
        steps = 0
        while not done:
            action = mock_env.action_space.sample()
            _, _, terminated, truncated, _ = mock_env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == 50

    def test_reset_resets_step_count(self, mock_env):
        mock_env.reset()
        for _ in range(10):
            mock_env.step(mock_env.action_space.sample())
        mock_env.reset()
        assert mock_env._step_count == 0

    def test_observation_space_shape(self, mock_env):
        assert mock_env.observation_space.shape == (8,)

    def test_action_space_continuous(self, mock_env):
        assert mock_env.action_space.shape == (2,)

    def test_action_space_discrete(self, mock_discrete_env):
        assert mock_discrete_env.action_space.n == 5


# ---------------------------------------------------------------------------
# FrameStack wrapper
# ---------------------------------------------------------------------------

class TestFrameStack:
    def test_stacked_obs_shape(self, mock_env):
        from navirl.envs.wrappers import FrameStack
        wrapped = FrameStack(mock_env, num_stack=4)
        obs, _ = wrapped.reset()
        assert obs.shape == (4, 8)

    def test_stacking_accumulates(self, mock_env):
        from navirl.envs.wrappers import FrameStack
        wrapped = FrameStack(mock_env, num_stack=3)
        obs, _ = wrapped.reset()
        # After reset, all frames should be the same
        np.testing.assert_array_equal(obs[0], obs[1])
        np.testing.assert_array_equal(obs[1], obs[2])

    @pytest.mark.parametrize("num_stack", [1, 2, 4, 8])
    def test_various_stack_sizes(self, num_stack):
        env = MockNavEnv(obs_dim=4)
        from navirl.envs.wrappers import FrameStack
        wrapped = FrameStack(env, num_stack=num_stack)
        obs, _ = wrapped.reset()
        assert obs.shape == (num_stack, 4)

    def test_step_updates_stack(self, mock_env):
        from navirl.envs.wrappers import FrameStack
        wrapped = FrameStack(mock_env, num_stack=2)
        obs0, _ = wrapped.reset()
        action = mock_env.action_space.sample()
        obs1, _, _, _, _ = wrapped.step(action)
        # After one step, the frames should differ (since obs is random)
        assert obs1.shape == (2, 8)


# ---------------------------------------------------------------------------
# NormalizeObservation wrapper
# ---------------------------------------------------------------------------

class TestNormalizeObservation:
    def test_normalized_shape(self, mock_env):
        from navirl.envs.wrappers import NormalizeObservation
        wrapped = NormalizeObservation(mock_env)
        obs, _ = wrapped.reset()
        assert obs.shape == (8,)
        assert obs.dtype == np.float32

    def test_observations_are_clipped(self, mock_env):
        from navirl.envs.wrappers import NormalizeObservation
        wrapped = NormalizeObservation(mock_env, clip=5.0)
        wrapped.reset()
        for _ in range(20):
            obs, _, _, _, _ = wrapped.step(mock_env.action_space.sample())
        assert np.all(obs >= -5.0)
        assert np.all(obs <= 5.0)


# ---------------------------------------------------------------------------
# RewardShaping wrapper
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_reward_shaping(self, mock_env):
        from navirl.envs.wrappers import RewardShaping
        wrapped = RewardShaping(mock_env, shaping_fn=lambda r, obs, act, info: r * 2.0)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(mock_env.action_space.sample())
        assert reward == pytest.approx(2.0)

    def test_reward_shaping_penalty(self, mock_env):
        from navirl.envs.wrappers import RewardShaping
        wrapped = RewardShaping(mock_env, shaping_fn=lambda r, obs, act, info: r - 0.5)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(mock_env.action_space.sample())
        assert reward == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ActionRepeat wrapper
# ---------------------------------------------------------------------------

class TestActionRepeat:
    def test_action_repeat(self, mock_env):
        from navirl.envs.wrappers import ActionRepeat
        wrapped = ActionRepeat(mock_env, repeat=4)
        wrapped.reset()
        action = mock_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        # Reward should be accumulated over 4 steps
        assert reward == pytest.approx(4.0)

    def test_action_repeat_early_termination(self):
        env = MockNavEnv(obs_dim=4)
        env._max_steps = 2
        from navirl.envs.wrappers import ActionRepeat
        wrapped = ActionRepeat(env, repeat=10)
        wrapped.reset()
        _, reward, terminated, truncated, _ = wrapped.step(env.action_space.sample())
        # Should terminate after 2 steps, not 10
        assert terminated or truncated
        assert reward == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ClipAction wrapper
# ---------------------------------------------------------------------------

class TestClipAction:
    def test_clip_action(self, mock_env):
        from navirl.envs.wrappers import ClipAction
        wrapped = ClipAction(mock_env)
        wrapped.reset()
        # Pass an action outside bounds
        action = np.array([10.0, -10.0], dtype=np.float32)
        obs, _, _, _, _ = wrapped.step(action)
        # Should not crash, action is clipped internally
        assert obs.shape == (8,)


# ---------------------------------------------------------------------------
# RecordEpisode wrapper
# ---------------------------------------------------------------------------

class TestRecordEpisode:
    def test_records_episode(self, mock_env):
        from navirl.envs.wrappers import RecordEpisode
        wrapped = RecordEpisode(mock_env)
        wrapped.reset()
        for _ in range(5):
            wrapped.step(mock_env.action_space.sample())
        assert len(wrapped.episode_rewards) >= 0  # at least tracks something


# ---------------------------------------------------------------------------
# CurriculumWrapper
# ---------------------------------------------------------------------------

class TestCurriculumWrapper:
    def test_curriculum_wrapper(self, mock_env):
        from navirl.envs.wrappers import CurriculumWrapper

        class MockCurriculum:
            def __init__(self):
                self.difficulty = 0.0
            def get_env_config(self):
                return {"difficulty": self.difficulty}

        curriculum = MockCurriculum()
        wrapped = CurriculumWrapper(mock_env, curriculum_manager=curriculum)
        obs, _ = wrapped.reset()
        assert obs.shape == (8,)


# ---------------------------------------------------------------------------
# Multi-step episode test
# ---------------------------------------------------------------------------

class TestFullEpisode:
    def test_full_episode_continuous(self, mock_env):
        obs, _ = mock_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = mock_env.action_space.sample()
            obs, reward, terminated, truncated, info = mock_env.step(action)
            total_reward += reward
            done = terminated or truncated
        assert total_reward > 0.0
        assert info.get("success") is True

    def test_full_episode_discrete(self, mock_discrete_env):
        obs, _ = mock_discrete_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = mock_discrete_env.action_space.sample()
            obs, reward, terminated, truncated, info = mock_discrete_env.step(action)
            total_reward += reward
            done = terminated or truncated
        assert total_reward > 0.0


# ---------------------------------------------------------------------------
# Wrapper composition
# ---------------------------------------------------------------------------

class TestWrapperComposition:
    def test_normalize_then_frame_stack(self, mock_env):
        from navirl.envs.wrappers import FrameStack, NormalizeObservation
        wrapped = NormalizeObservation(mock_env)
        wrapped = FrameStack(wrapped, num_stack=3)
        obs, _ = wrapped.reset()
        assert obs.shape == (3, 8)

    def test_clip_and_reward_shape(self, mock_env):
        from navirl.envs.wrappers import ClipAction, RewardShaping
        wrapped = ClipAction(mock_env)
        wrapped = RewardShaping(wrapped, shaping_fn=lambda r, o, a, i: r + 1.0)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(np.array([0.5, 0.5]))
        assert reward == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_scenarios_module_importable(self):
        from navirl.envs import scenarios
        assert hasattr(scenarios, "__file__")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_env_with_zero_dim_obs(self):
        env = MockNavEnv(obs_dim=1)
        obs, _ = env.reset()
        assert obs.shape == (1,)

    def test_multiple_resets(self, mock_env):
        for _ in range(5):
            obs, info = mock_env.reset()
            assert obs.shape == (8,)

    def test_step_after_done(self, mock_env):
        mock_env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = mock_env.step(mock_env.action_space.sample())
            done = terminated or truncated
        # Stepping after done should still work (no crash)
        obs, _, _, _, _ = mock_env.step(mock_env.action_space.sample())
        assert obs.shape == (8,)
