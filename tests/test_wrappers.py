from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Build a mock gymnasium module so the wrappers can import cleanly.
# ---------------------------------------------------------------------------


def _build_mock_gym() -> types.ModuleType:
    """Create a minimal gymnasium mock with Env, Wrapper, spaces, etc."""

    gym_mod = types.ModuleType("gymnasium")
    spaces_mod = types.ModuleType("gymnasium.spaces")
    utils_mod = types.ModuleType("gymnasium.spaces.utils")

    # --- spaces.Box ---
    class Box:
        def __init__(self, low=-np.inf, high=np.inf, shape=None, dtype=np.float32):
            if shape is not None:
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
                self.shape = shape
            else:
                self.low = np.asarray(low, dtype=dtype)
                self.high = np.asarray(high, dtype=dtype)
                self.shape = self.low.shape
            self.dtype = np.dtype(dtype)

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class Dict(dict):
        """Minimal gymnasium.spaces.Dict."""

        def __init__(self, spaces_dict=None, **kwargs):
            super().__init__(spaces_dict or {}, **kwargs)

        def sample(self):
            return {k: v.sample() for k, v in self.items()}

    class Discrete:
        def __init__(self, n):
            self.n = n
            self.shape = ()
            self.dtype = np.int64

        def sample(self):
            return np.random.randint(self.n)

    def flatten_space(space):
        if isinstance(space, Dict):
            total = sum(int(np.prod(s.shape)) for s in space.values())
            return Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)
        return space

    def flatten(space, obs):
        if isinstance(obs, dict):
            return np.concatenate([np.asarray(v).flatten() for v in obs.values()])
        return np.asarray(obs).flatten()

    spaces_mod.Box = Box
    spaces_mod.Dict = Dict
    spaces_mod.Discrete = Discrete
    utils_mod.flatten_space = flatten_space
    utils_mod.flatten = flatten
    spaces_mod.utils = utils_mod

    # --- Base classes ---
    class Env:
        observation_space: Box
        action_space: Box

        def step(self, action):
            raise NotImplementedError

        def reset(self, **kwargs):
            raise NotImplementedError

        def close(self):
            pass

        @property
        def unwrapped(self):
            return self

    class Wrapper(Env):
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, "observation_space", None)
            self.action_space = getattr(env, "action_space", None)

        def step(self, action):
            return self.env.step(action)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def close(self):
            return self.env.close()

        @property
        def unwrapped(self):
            return self.env.unwrapped

    class ObservationWrapper(Wrapper):
        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(action)
            return self.observation(obs), rew, term, trunc, info

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info

        def observation(self, observation):
            raise NotImplementedError

    class ActionWrapper(Wrapper):
        def step(self, action):
            return self.env.step(self.action(action))

        def action(self, action):
            raise NotImplementedError

    class RewardWrapper(Wrapper):
        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(action)
            return obs, self.reward(rew), term, trunc, info

        def reward(self, reward):
            raise NotImplementedError

    gym_mod.Env = Env
    gym_mod.Wrapper = Wrapper
    gym_mod.ObservationWrapper = ObservationWrapper
    gym_mod.ActionWrapper = ActionWrapper
    gym_mod.RewardWrapper = RewardWrapper
    gym_mod.spaces = spaces_mod

    return gym_mod, spaces_mod, utils_mod


_gym_mod, _spaces_mod, _utils_mod = _build_mock_gym()

# Inject into sys.modules BEFORE importing wrappers
sys.modules["gymnasium"] = _gym_mod
sys.modules["gymnasium.spaces"] = _spaces_mod
sys.modules["gymnasium.spaces.utils"] = _utils_mod

# Now import the wrappers module (it will find our mock gymnasium)
# Remove any cached import of the wrappers module first
sys.modules.pop("navirl.envs.wrappers", None)

from navirl.envs.wrappers import (
    ActionRepeat,
    ClipAction,
    CurriculumWrapper,
    DomainRandomization,
    FlattenObservation,
    FrameStack,
    GoalConditioned,
    MonitorWrapper,
    NormalizeObservation,
    NormalizeReward,
    RecordEpisode,
    RelativeObservation,
    RewardShaping,
    TimeLimit,
    VecEnvWrapper,
)

# ---------------------------------------------------------------------------
# Helpers: a simple mock environment
# ---------------------------------------------------------------------------

class MockEnv(_gym_mod.Env):
    """Deterministic environment for testing wrappers."""

    def __init__(
        self,
        obs_shape=(4,),
        obs_low=-1.0,
        obs_high=1.0,
        act_shape=(2,),
        act_low=-1.0,
        act_high=1.0,
    ):
        self.observation_space = _spaces_mod.Box(
            low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float32
        )
        self.action_space = _spaces_mod.Box(
            low=act_low, high=act_high, shape=act_shape, dtype=np.float32
        )
        self._step_count = 0
        self._obs = np.zeros(obs_shape, dtype=np.float32)
        self._terminated = False
        self._truncated = False
        self._reward = 1.0
        self._info: dict = {}
        self.difficulty = 0.0  # for CurriculumWrapper / DomainRandomization

    def reset(self, **kwargs):
        self._step_count = 0
        self._obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action):
        self._step_count += 1
        self._obs = np.full(self.observation_space.shape, self._step_count * 0.1, dtype=np.float32)
        return (
            self._obs.copy(),
            self._reward,
            self._terminated,
            self._truncated,
            dict(self._info),
        )

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self


# ============================================================================
# Tests: FrameStack
# ============================================================================


class TestFrameStack:
    def test_observation_shape_after_reset(self):
        env = MockEnv(obs_shape=(3,))
        wrapped = FrameStack(env, num_stack=4)
        obs, _ = wrapped.reset()
        assert obs.shape == (4, 3)

    def test_reset_fills_all_frames_with_initial_obs(self):
        env = MockEnv(obs_shape=(2,))
        wrapped = FrameStack(env, num_stack=3)
        obs, _ = wrapped.reset()
        # All frames should be identical (the reset observation is zeros)
        for i in range(3):
            np.testing.assert_array_equal(obs[i], np.zeros(2))

    def test_step_shifts_frames(self):
        env = MockEnv(obs_shape=(2,))
        wrapped = FrameStack(env, num_stack=3)
        wrapped.reset()
        obs1, *_ = wrapped.step(np.zeros(2))
        # Frame 0,1 = reset obs (zeros), frame 2 = step obs (0.1, 0.1)
        np.testing.assert_allclose(obs1[0], np.zeros(2))
        np.testing.assert_allclose(obs1[2], np.full(2, 0.1), atol=1e-6)

    def test_observation_space_bounds(self):
        env = MockEnv(obs_shape=(3,), obs_low=-2.0, obs_high=2.0)
        wrapped = FrameStack(env, num_stack=2)
        assert wrapped.observation_space.shape == (2, 3)
        np.testing.assert_allclose(wrapped.observation_space.low, -2.0)
        np.testing.assert_allclose(wrapped.observation_space.high, 2.0)


# ============================================================================
# Tests: NormalizeObservation
# ============================================================================


class TestNormalizeObservation:
    def test_first_observation_is_zero(self):
        env = MockEnv(obs_shape=(2,))
        wrapped = NormalizeObservation(env)
        wrapped.reset()
        # After reset the observation method sees a zero-vector. With count=1,
        # mean=0, var ~1, normalised should be ~0.
        obs, _ = wrapped.reset()
        np.testing.assert_allclose(obs, 0.0, atol=1e-5)

    def test_clipping(self):
        env = MockEnv(obs_shape=(1,))
        wrapped = NormalizeObservation(env, clip=2.0)
        wrapped.reset()
        # Feed a huge observation value through the observation method
        result = wrapped.observation(np.array([1e6], dtype=np.float32))
        assert np.all(result <= 2.0)
        assert np.all(result >= -2.0)

    def test_running_stats_converge(self):
        env = MockEnv(obs_shape=(1,))
        wrapped = NormalizeObservation(env)
        wrapped.reset()
        for val in np.random.randn(200):
            wrapped.observation(np.array([val], dtype=np.float32))
        # After many observations, _mean and _var should be reasonable
        assert wrapped._count == 201  # 1 from reset + 200


# ============================================================================
# Tests: FlattenObservation
# ============================================================================


class TestFlattenObservation:
    def test_flattens_dict_observation(self):
        env = MockEnv(obs_shape=(4,))
        # Replace observation_space with a Dict space
        env.observation_space = _spaces_mod.Dict({
            "pos": _spaces_mod.Box(low=-1, high=1, shape=(2,)),
            "vel": _spaces_mod.Box(low=-1, high=1, shape=(3,)),
        })
        wrapped = FlattenObservation(env)
        assert wrapped.observation_space.shape == (5,)

    def test_observation_returns_flat_array(self):
        env = MockEnv(obs_shape=(4,))
        env.observation_space = _spaces_mod.Dict({
            "a": _spaces_mod.Box(low=0, high=1, shape=(2,)),
            "b": _spaces_mod.Box(low=0, high=1, shape=(2,)),
        })
        wrapped = FlattenObservation(env)
        obs_dict = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        result = wrapped.observation(obs_dict)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])


# ============================================================================
# Tests: RelativeObservation
# ============================================================================


class TestRelativeObservation:
    def test_translation_puts_robot_at_origin(self):
        env = MockEnv(obs_shape=(6,))
        wrapped = RelativeObservation(env, robot_pos_indices=(0, 1))
        obs = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        result = wrapped.observation(obs)
        # Robot x,y should be at origin
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        # Other positions shifted
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(2.0)

    def test_rotation_with_heading(self):
        # Use 6-element obs so even/odd slicing produces equal-length arrays
        # obs: [rx, ry, heading, pad, other_x, other_y]
        env = MockEnv(obs_shape=(6,))
        wrapped = RelativeObservation(env, robot_pos_indices=(0, 1), robot_heading_index=2)
        heading = np.pi / 2
        obs = np.array([1.0, 1.0, heading, 0.0, 2.0, 1.0], dtype=np.float32)
        result = wrapped.observation(obs)
        # After translation: [0, 0, heading-heading, -1, 1, 0]
        # Robot position should be at origin
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(0.0, abs=1e-5)

    def test_no_rotation_when_heading_is_none(self):
        env = MockEnv(obs_shape=(4,))
        wrapped = RelativeObservation(env, robot_pos_indices=(0, 1), robot_heading_index=None)
        obs = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = wrapped.observation(obs)
        np.testing.assert_allclose(result, [0.0, 0.0, 2.0, 2.0])


# ============================================================================
# Tests: GoalConditioned
# ============================================================================


class TestGoalConditioned:
    def test_observation_is_dict_with_required_keys(self):
        env = MockEnv(obs_shape=(6,))
        wrapped = GoalConditioned(env)
        obs_raw = np.arange(6, dtype=np.float32)
        result = wrapped.observation(obs_raw)
        assert "observation" in result
        assert "achieved_goal" in result
        assert "desired_goal" in result

    def test_default_goal_fns_use_first_and_second_pairs(self):
        env = MockEnv(obs_shape=(6,))
        wrapped = GoalConditioned(env)
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        result = wrapped.observation(obs)
        np.testing.assert_array_equal(result["achieved_goal"], [1.0, 2.0])
        np.testing.assert_array_equal(result["desired_goal"], [3.0, 4.0])

    def test_custom_goal_fns(self):
        env = MockEnv(obs_shape=(6,))
        wrapped = GoalConditioned(
            env,
            achieved_goal_fn=lambda o: o[-2:],
            desired_goal_fn=lambda o: o[0:1],
        )
        obs = np.arange(6, dtype=np.float32)
        result = wrapped.observation(obs)
        np.testing.assert_array_equal(result["achieved_goal"], [4.0, 5.0])
        np.testing.assert_array_equal(result["desired_goal"], [0.0])

    def test_observation_space_is_dict(self):
        env = MockEnv(obs_shape=(6,))
        wrapped = GoalConditioned(env)
        assert isinstance(wrapped.observation_space, _spaces_mod.Dict)
        assert "observation" in wrapped.observation_space
        assert "achieved_goal" in wrapped.observation_space


# ============================================================================
# Tests: ClipAction
# ============================================================================


class TestClipAction:
    def test_clips_action_to_bounds(self):
        env = MockEnv(act_shape=(2,), act_low=-0.5, act_high=0.5)
        wrapped = ClipAction(env)
        clipped = wrapped.action(np.array([2.0, -3.0]))
        np.testing.assert_allclose(clipped, [0.5, -0.5])

    def test_action_within_bounds_unchanged(self):
        env = MockEnv(act_shape=(2,), act_low=-1.0, act_high=1.0)
        wrapped = ClipAction(env)
        action = np.array([0.3, -0.7])
        clipped = wrapped.action(action)
        np.testing.assert_allclose(clipped, action)


# ============================================================================
# Tests: RewardShaping
# ============================================================================


class TestRewardShaping:
    def test_progress_weight(self):
        env = MockEnv()
        # Make reset return distance_to_goal in its info
        original_reset = env.reset

        def reset_with_dist(**kwargs):
            obs, info = original_reset(**kwargs)
            info["distance_to_goal"] = 10.0
            return obs, info

        env.reset = reset_with_dist
        wrapped = RewardShaping(env, progress_weight=1.0)
        wrapped.reset()
        # After reset, _prev_dist = 10.0
        env._info = {"distance_to_goal": 8.0}
        _, reward, *_ = wrapped.step(np.zeros(2))
        # shaped = base(1.0) + 1.0*(10.0 - 8.0) = 3.0
        assert reward == pytest.approx(3.0)

    def test_collision_penalty(self):
        env = MockEnv()
        env._info = {"collision": True}
        wrapped = RewardShaping(env, collision_penalty=5.0)
        wrapped.reset()
        _, reward, *_ = wrapped.step(np.zeros(2))
        # shaped = 1.0 - 5.0 = -4.0
        assert reward == pytest.approx(-4.0)

    def test_custom_shaping_fn_6_args(self):
        def shaper(obs, action, reward, terminated, truncated, info):
            return 100.0

        env = MockEnv()
        wrapped = RewardShaping(env, shaping_fn=shaper)
        wrapped.reset()
        _, reward, *_ = wrapped.step(np.zeros(2))
        # shaped = base(1.0) + 100.0 = 101.0
        assert reward == pytest.approx(101.0)

    def test_custom_shaping_fn_4_args(self):
        def shaper(reward, obs, action, info):
            return float(reward) * 2

        env = MockEnv()
        wrapped = RewardShaping(env, shaping_fn=shaper)
        wrapped.reset()
        _, reward, *_ = wrapped.step(np.zeros(2))
        # For arity<=4, shaped = shaping_fn(reward, obs, action, info)
        assert reward == pytest.approx(2.0)

    def test_custom_shaping_fn_with_varargs(self):
        def shaper(*args):
            return 50.0

        env = MockEnv()
        wrapped = RewardShaping(env, shaping_fn=shaper)
        # VAR_POSITIONAL => arity=6 => added to base
        wrapped.reset()
        _, reward, *_ = wrapped.step(np.zeros(2))
        assert reward == pytest.approx(51.0)


class TestInferShapingFnArity:
    """Test _infer_shaping_fn_arity independently (no gym needed)."""

    def test_none_returns_zero(self):
        assert RewardShaping._infer_shaping_fn_arity(None) == 0

    def test_lambda_no_args(self):
        assert RewardShaping._infer_shaping_fn_arity(lambda: 0) == 0

    def test_lambda_one_arg(self):
        assert RewardShaping._infer_shaping_fn_arity(lambda x: x) == 1

    def test_lambda_four_args(self):
        assert RewardShaping._infer_shaping_fn_arity(lambda a, b, c, d: 0) == 4

    def test_lambda_six_args(self):
        assert RewardShaping._infer_shaping_fn_arity(lambda a, b, c, d, e, f: 0) == 6

    def test_varargs_returns_six(self):
        assert RewardShaping._infer_shaping_fn_arity(lambda *args: 0) == 6

    def test_mixed_args_and_varargs(self):
        assert RewardShaping._infer_shaping_fn_arity(lambda a, b, *args: 0) == 6

    def test_regular_function(self):
        def my_fn(obs, action, reward):
            return 0.0

        assert RewardShaping._infer_shaping_fn_arity(my_fn) == 3

    def test_uninspectable_callable_returns_six(self):
        # Create a callable whose signature raises TypeError
        class Opaque:
            def __call__(self, *args):
                return 0.0
            # Make inspect.signature fail
            __signature__ = None
        opaque = Opaque()
        # inspect.signature raises TypeError for __signature__=None in some versions;
        # if it doesn't, the VAR_POSITIONAL path gives 6 anyway
        result = RewardShaping._infer_shaping_fn_arity(opaque)
        assert result == 6


# ============================================================================
# Tests: NormalizeReward
# ============================================================================


class TestNormalizeReward:
    def test_normalizes_constant_reward(self):
        env = MockEnv()
        wrapped = NormalizeReward(env, gamma=0.99, clip=10.0)
        wrapped.reset()
        rewards = []
        for _ in range(50):
            _, r, *_ = wrapped.step(np.zeros(2))
            rewards.append(r)
        # Should produce finite rewards
        assert all(np.isfinite(r) for r in rewards)

    def test_clipping(self):
        env = MockEnv()
        env._reward = 1e6
        wrapped = NormalizeReward(env, clip=5.0)
        wrapped.reset()
        _, r, *_ = wrapped.step(np.zeros(2))
        assert -5.0 <= r <= 5.0

    def test_reset_clears_return(self):
        env = MockEnv()
        wrapped = NormalizeReward(env)
        wrapped.reset()
        wrapped.step(np.zeros(2))
        wrapped.step(np.zeros(2))
        wrapped.reset()
        assert wrapped._return == 0.0


# ============================================================================
# Tests: ActionRepeat
# ============================================================================


class TestActionRepeat:
    def test_reward_summing(self):
        env = MockEnv()
        env._reward = 2.0
        wrapped = ActionRepeat(env, num_repeat=3)
        wrapped.reset()
        _, reward, *_ = wrapped.step(np.zeros(2))
        assert reward == pytest.approx(6.0)

    def test_early_termination(self):
        env = MockEnv()
        env._reward = 1.0
        call_count = 0
        original_step = env.step

        def terminating_step(action):
            nonlocal call_count
            call_count += 1
            obs, r, term, trunc, info = original_step(action)
            if call_count >= 2:
                term = True
            return obs, r, term, trunc, info

        env.step = terminating_step
        wrapped = ActionRepeat(env, num_repeat=5)
        wrapped.reset()
        _, reward, terminated, *_ = wrapped.step(np.zeros(2))
        # Should stop after 2 steps
        assert call_count == 2
        assert reward == pytest.approx(2.0)
        assert terminated is True


# ============================================================================
# Tests: TimeLimit
# ============================================================================


class TestTimeLimit:
    def test_truncation_at_max_steps(self):
        env = MockEnv()
        wrapped = TimeLimit(env, max_steps=3)
        wrapped.reset()
        for _i in range(2):
            _, _, _, truncated, info = wrapped.step(np.zeros(2))
            assert truncated is False
        _, _, _, truncated, info = wrapped.step(np.zeros(2))
        assert truncated is True
        assert info.get("TimeLimit.truncated") is True

    def test_reset_clears_counter(self):
        env = MockEnv()
        wrapped = TimeLimit(env, max_steps=5)
        wrapped.reset()
        wrapped.step(np.zeros(2))
        wrapped.step(np.zeros(2))
        wrapped.reset()
        assert wrapped._elapsed_steps == 0


# ============================================================================
# Tests: RecordEpisode
# ============================================================================


class TestRecordEpisode:
    def test_episode_stored_on_termination(self):
        env = MockEnv()
        wrapped = RecordEpisode(env)
        wrapped.reset()
        env._terminated = True
        wrapped.step(np.zeros(2))
        assert len(wrapped.episodes) == 1
        assert len(wrapped.episode_rewards) == 1

    def test_episode_rewards_tracking(self):
        env = MockEnv()
        env._reward = 3.0
        wrapped = RecordEpisode(env)
        wrapped.reset()
        wrapped.step(np.zeros(2))
        wrapped.step(np.zeros(2))
        env._terminated = True
        wrapped.step(np.zeros(2))
        assert wrapped.episode_rewards[-1] == pytest.approx(9.0)

    def test_fifo_buffer(self):
        env = MockEnv()
        wrapped = RecordEpisode(env, max_episodes=2)
        for _ in range(3):
            wrapped.reset()
            env._terminated = True
            wrapped.step(np.zeros(2))
        assert len(wrapped.episodes) == 2

    def test_episode_data_structure(self):
        env = MockEnv()
        wrapped = RecordEpisode(env)
        wrapped.reset()
        wrapped.step(np.zeros(2))
        env._terminated = True
        wrapped.step(np.zeros(2))
        ep = wrapped.episodes[0]
        assert "observations" in ep
        assert "actions" in ep
        assert "rewards" in ep
        assert "infos" in ep
        # 1 from reset + 2 from steps
        assert len(ep["observations"]) == 3
        assert len(ep["actions"]) == 2


# ============================================================================
# Tests: MonitorWrapper
# ============================================================================


class TestMonitorWrapper:
    def test_episode_stats_recorded(self):
        env = MockEnv()
        env._reward = 2.0
        wrapped = MonitorWrapper(env)
        wrapped.reset()
        wrapped.step(np.zeros(2))
        env._terminated = True
        _, _, _, _, info = wrapped.step(np.zeros(2))
        assert len(wrapped.episode_returns) == 1
        assert wrapped.episode_returns[0] == pytest.approx(4.0)
        assert wrapped.episode_lengths[0] == 2
        assert len(wrapped.episode_times) == 1

    def test_info_contains_episode_on_done(self):
        env = MockEnv()
        env._terminated = True
        wrapped = MonitorWrapper(env)
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.zeros(2))
        assert "episode" in info
        assert "r" in info["episode"]
        assert "l" in info["episode"]
        assert "t" in info["episode"]

    def test_reset_clears_accumulators(self):
        env = MockEnv()
        wrapped = MonitorWrapper(env)
        wrapped.reset()
        wrapped.step(np.zeros(2))
        wrapped.reset()
        assert wrapped._episode_return == 0.0
        assert wrapped._episode_length == 0


# ============================================================================
# Tests: CurriculumWrapper
# ============================================================================


class TestCurriculumWrapper:
    def test_scheduler_sets_difficulty(self):
        env = MockEnv()
        def scheduler(steps):
            return min(steps / 100, 1.0)
        wrapped = CurriculumWrapper(env, scheduler=scheduler)
        wrapped.reset()
        assert env.difficulty == 0.0  # 0 steps so far
        for _ in range(50):
            wrapped.step(np.zeros(2))
        wrapped.reset()
        assert env.difficulty == pytest.approx(0.5)

    def test_curriculum_manager_with_get_env_config(self):
        env = MockEnv()
        manager = MagicMock()
        manager.get_env_config.return_value = {"difficulty": 0.75}
        wrapped = CurriculumWrapper(env, curriculum_manager=manager)
        wrapped.reset()
        assert env.difficulty == pytest.approx(0.75)

    def test_curriculum_manager_with_difficulty_attr(self):
        env = MockEnv()
        manager = MagicMock(spec=[])  # no get_env_config
        manager.difficulty = 0.3
        wrapped = CurriculumWrapper(env, curriculum_manager=manager)
        wrapped.reset()
        assert env.difficulty == pytest.approx(0.3)

    def test_error_when_neither_scheduler_nor_manager(self):
        env = MockEnv()
        with pytest.raises(TypeError, match="requires either"):
            CurriculumWrapper(env)

    def test_error_when_manager_has_no_difficulty(self):
        env = MockEnv()
        manager = MagicMock(spec=[])  # no attributes
        del manager.difficulty  # make sure it doesn't exist
        wrapped = CurriculumWrapper(env, curriculum_manager=manager)
        with pytest.raises(AttributeError, match="curriculum_manager must define"):
            wrapped.reset()


# ============================================================================
# Tests: DomainRandomization
# ============================================================================


class TestDomainRandomization:
    def test_attributes_randomized_on_reset(self):
        env = MockEnv()
        env.friction = 0.5  # add attribute
        config = {"friction": (0.1, 0.9), "difficulty": (0.0, 1.0)}
        wrapped = DomainRandomization(env, randomization_config=config, seed=42)
        wrapped.reset()
        # The attributes should have changed from defaults
        assert 0.1 <= env.friction <= 0.9
        assert 0.0 <= env.difficulty <= 1.0

    def test_seeded_rng_is_deterministic(self):
        env1 = MockEnv()
        env1.param = 0.0
        env2 = MockEnv()
        env2.param = 0.0
        config = {"param": (0.0, 10.0)}
        w1 = DomainRandomization(env1, randomization_config=config, seed=123)
        w2 = DomainRandomization(env2, randomization_config=config, seed=123)
        w1.reset()
        w2.reset()
        assert env1.param == pytest.approx(env2.param)

    def test_missing_attribute_is_skipped(self):
        env = MockEnv()
        config = {"nonexistent_attr": (0.0, 1.0)}
        wrapped = DomainRandomization(env, randomization_config=config, seed=0)
        # Should not raise; attribute doesn't exist so it won't be set
        wrapped.reset()
        assert not hasattr(env, "nonexistent_attr")


# ============================================================================
# Tests: VecEnvWrapper
# ============================================================================


class TestVecEnvWrapper:
    def test_batched_reset(self):
        vec = VecEnvWrapper([lambda: MockEnv(obs_shape=(3,)) for _ in range(4)])
        obs, infos = vec.reset()
        assert obs.shape == (4, 3)
        assert len(infos) == 4

    def test_batched_step(self):
        vec = VecEnvWrapper([lambda: MockEnv(obs_shape=(2,), act_shape=(1,)) for _ in range(3)])
        vec.reset()
        actions = np.zeros((3, 1))
        obs, rewards, terminateds, truncateds, infos = vec.step(actions)
        assert obs.shape == (3, 2)
        assert rewards.shape == (3,)
        assert terminateds.shape == (3,)
        assert truncateds.shape == (3,)
        assert len(infos) == 3

    def test_close(self):
        envs_closed = []

        def make_env():
            e = MockEnv()
            original_close = e.close
            def tracked_close():
                envs_closed.append(True)
                original_close()
            e.close = tracked_close
            return e

        vec = VecEnvWrapper([make_env for _ in range(2)])
        vec.close()
        assert len(envs_closed) == 2

    def test_num_envs(self):
        vec = VecEnvWrapper([lambda: MockEnv() for _ in range(5)])
        assert vec.num_envs == 5
