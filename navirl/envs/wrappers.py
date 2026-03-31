"""Gymnasium wrappers for NavIRL navigation environments.

Each wrapper subclasses the appropriate ``gymnasium`` base wrapper and can be
composed around any :class:`~navirl.envs.NavEnv` (or compatible Gymnasium env).

Exports
-------
FrameStack, ActionRepeat, TimeLimit, RewardShaping, NormalizeObservation,
NormalizeReward, ClipAction, FlattenObservation, RecordEpisode, MonitorWrapper,
CurriculumWrapper, DomainRandomization, VecEnvWrapper, RelativeObservation,
GoalConditioned
"""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Callable
from typing import Any, SupportsFloat

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    try:
        import gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[no-redef]

        _GYM_AVAILABLE = True
    except ImportError:
        _GYM_AVAILABLE = False


def _require_gym() -> None:
    if not _GYM_AVAILABLE:
        raise ImportError(
            "Neither gymnasium nor gym is installed. "
            "Install gymnasium with: pip install gymnasium"
        )


# ============================================================================
# Observation wrappers
# ============================================================================


class FrameStack(gym.ObservationWrapper):
    """Stack the last *num_stack* observations along a new leading axis.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    num_stack : int
        Number of frames to stack.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        _require_gym()
        super().__init__(env)
        self.num_stack = num_stack
        self._frames: deque = deque(maxlen=num_stack)

        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self._frames.append(observation)
        return np.stack(list(self._frames), axis=0)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self._frames.append(obs)
        return self.observation(obs), info


class NormalizeObservation(gym.ObservationWrapper):
    """Normalise observations using a running mean and standard deviation.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    clip : float
        Clip normalised observations to ``[-clip, clip]``.
    epsilon : float
        Small constant to avoid division by zero.
    """

    def __init__(self, env: gym.Env, clip: float = 10.0, epsilon: float = 1e-8):
        _require_gym()
        super().__init__(env)
        self.clip = clip
        self.epsilon = epsilon
        self._count = 0.0
        self._mean = np.zeros(env.observation_space.shape, dtype=np.float64)
        self._var = np.ones(env.observation_space.shape, dtype=np.float64)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self._update_stats(observation)
        normalised = (observation - self._mean) / np.sqrt(self._var + self.epsilon)
        return np.clip(normalised, -self.clip, self.clip).astype(np.float32)

    def _update_stats(self, obs: np.ndarray) -> None:
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += (delta * delta2 - self._var) / self._count


class FlattenObservation(gym.ObservationWrapper):
    """Flatten dictionary observations into a single vector.

    Works with :class:`gymnasium.spaces.Dict` or :class:`gymnasium.spaces.Tuple`
    observation spaces.
    """

    def __init__(self, env: gym.Env):
        _require_gym()
        super().__init__(env)
        self.observation_space = spaces.utils.flatten_space(env.observation_space)

    def observation(self, observation: Any) -> np.ndarray:
        return spaces.utils.flatten(self.env.observation_space, observation)


class RelativeObservation(gym.ObservationWrapper):
    """Convert absolute positions to robot-centric (ego-centric) coordinates.

    Assumes the observation is a 1-D array where the first two elements are
    the robot's (x, y) position.  All subsequent (x, y) pairs are shifted
    so that the robot is at the origin, then optionally rotated so the
    robot's heading faces the +x axis.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    robot_pos_indices : tuple[int, int]
        Indices of the robot's x, y in the observation vector.
    robot_heading_index : int | None
        Index of the robot's heading angle (radians).  If given, the frame
        is rotated so the heading aligns with +x.
    """

    def __init__(
        self,
        env: gym.Env,
        robot_pos_indices: tuple[int, int] = (0, 1),
        robot_heading_index: int | None = None,
    ):
        _require_gym()
        super().__init__(env)
        self.robot_pos_indices = robot_pos_indices
        self.robot_heading_index = robot_heading_index

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = observation.copy()
        rx = obs[self.robot_pos_indices[0]]
        ry = obs[self.robot_pos_indices[1]]

        # Translate so robot is at origin
        # Shift every pair of coordinates (we treat even/odd as x/y)
        obs[0::2] -= rx
        obs[1::2] -= ry

        # Optionally rotate into ego frame
        if self.robot_heading_index is not None:
            theta = -obs[self.robot_heading_index]
            c, s = np.cos(theta), np.sin(theta)
            xs = obs[0::2].copy()
            ys = obs[1::2].copy()
            obs[0::2] = c * xs - s * ys
            obs[1::2] = s * xs + c * ys

        return obs


class GoalConditioned(gym.ObservationWrapper):
    """Add goal-conditioned observation structure (HER-compatible).

    Wraps the observation into a dictionary with ``observation``,
    ``achieved_goal``, and ``desired_goal`` keys suitable for Hindsight
    Experience Replay.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    achieved_goal_fn : callable
        ``(obs) -> np.ndarray`` returning the achieved goal.
    desired_goal_fn : callable
        ``(obs) -> np.ndarray`` returning the desired goal.
    """

    def __init__(
        self,
        env: gym.Env,
        achieved_goal_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        desired_goal_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        _require_gym()
        super().__init__(env)

        self._achieved_goal_fn = achieved_goal_fn or (lambda obs: obs[:2])
        self._desired_goal_fn = desired_goal_fn or (lambda obs: obs[2:4])

        # Build a Dict observation space
        sample = env.observation_space.sample()
        ag_sample = self._achieved_goal_fn(sample)
        dg_sample = self._desired_goal_fn(sample)

        self.observation_space = spaces.Dict(
            {
                "observation": env.observation_space,
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=ag_sample.shape, dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=dg_sample.shape, dtype=np.float32
                ),
            }
        )

    def observation(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "observation": observation,
            "achieved_goal": self._achieved_goal_fn(observation),
            "desired_goal": self._desired_goal_fn(observation),
        }


# ============================================================================
# Action wrappers
# ============================================================================


class ClipAction(gym.ActionWrapper):
    """Clip continuous actions to the environment's valid range."""

    def __init__(self, env: gym.Env):
        _require_gym()
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)


# ============================================================================
# Reward wrappers
# ============================================================================


class RewardShaping(gym.RewardWrapper):
    """Add shaped reward terms on top of the base reward.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    shaping_fn : callable | None
        ``(obs, action, reward, terminated, truncated, info) -> float``
        returning the *additional* reward to add.
    progress_weight : float
        Weight for a built-in goal-progress shaping term (requires
        ``info["distance_to_goal"]``).
    collision_penalty : float
        Additional penalty applied when ``info.get("collision")`` is truthy.
    """

    def __init__(
        self,
        env: gym.Env,
        shaping_fn: Callable[..., float] | None = None,
        progress_weight: float = 0.0,
        collision_penalty: float = 0.0,
    ):
        _require_gym()
        super().__init__(env)
        self._shaping_fn = shaping_fn
        self._progress_weight = progress_weight
        self._collision_penalty = collision_penalty
        self._prev_dist: float | None = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = info.get("distance_to_goal")
        return obs, info

    def reward(self, reward: SupportsFloat) -> float:
        r = float(reward)
        return r  # base return; step() post-processes below

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)

        # Built-in progress shaping
        if self._progress_weight != 0.0 and "distance_to_goal" in info:
            curr_dist = info["distance_to_goal"]
            if self._prev_dist is not None:
                shaped += self._progress_weight * (self._prev_dist - curr_dist)
            self._prev_dist = curr_dist

        # Collision penalty
        if self._collision_penalty != 0.0 and info.get("collision"):
            shaped -= self._collision_penalty

        # Custom shaping function
        if self._shaping_fn is not None:
            shaped += self._shaping_fn(obs, action, reward, terminated, truncated, info)

        return obs, shaped, terminated, truncated, info


class NormalizeReward(gym.RewardWrapper):
    """Normalise rewards using a running discounted return estimate.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    gamma : float
        Discount factor used for the running return estimate.
    clip : float
        Clip normalised rewards to ``[-clip, clip]``.
    epsilon : float
        Small constant to avoid division by zero.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ):
        _require_gym()
        super().__init__(env)
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self._return = 0.0
        self._count = 0.0
        self._mean = 0.0
        self._var = 1.0

    def reward(self, reward: SupportsFloat) -> float:
        r = float(reward)
        self._return = self._return * self.gamma + r
        self._update_stats(self._return)
        normalised = r / (np.sqrt(self._var) + self.epsilon)
        return float(np.clip(normalised, -self.clip, self.clip))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._return = 0.0
        return self.env.reset(**kwargs)

    def _update_stats(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._var += (delta * delta2 - self._var) / self._count


# ============================================================================
# General wrappers (subclass gym.Wrapper)
# ============================================================================


class ActionRepeat(gym.Wrapper):
    """Repeat the selected action for *num_repeat* environment steps.

    Rewards are summed across repeats; the wrapper terminates early if the
    episode ends during a repeat.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    num_repeat : int
        Number of times to repeat each action.
    """

    def __init__(self, env: gym.Env, num_repeat: int = 4):
        _require_gym()
        super().__init__(env)
        self.num_repeat = num_repeat

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self.num_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class TimeLimit(gym.Wrapper):
    """Override the maximum number of steps per episode.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    max_steps : int
        Maximum number of environment steps before truncation.
    """

    def __init__(self, env: gym.Env, max_steps: int = 500):
        _require_gym()
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed_steps: int = 0

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class RecordEpisode(gym.Wrapper):
    """Record episode data (observations, actions, rewards) for later replay.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    max_episodes : int
        Maximum number of episodes to retain in the buffer (FIFO).
    """

    def __init__(self, env: gym.Env, max_episodes: int = 100):
        _require_gym()
        super().__init__(env)
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes)
        self._current_episode: dict[str, list[Any]] = {}

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self._current_episode.get("observations"):
            self.episodes.append(copy.deepcopy(self._current_episode))
        self._current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
        }
        obs, info = self.env.reset(**kwargs)
        self._current_episode["observations"].append(obs)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_episode["actions"].append(action)
        self._current_episode["rewards"].append(reward)
        self._current_episode["observations"].append(obs)
        self._current_episode["infos"].append(info)
        if terminated or truncated:
            self.episodes.append(copy.deepcopy(self._current_episode))
            self._current_episode = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "infos": [],
            }
        return obs, reward, terminated, truncated, info


class MonitorWrapper(gym.Wrapper):
    """Track episode statistics: return, length, and wall-clock time.

    Statistics for completed episodes are available via
    :attr:`episode_returns`, :attr:`episode_lengths`, and
    :attr:`episode_times`.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    """

    def __init__(self, env: gym.Env):
        _require_gym()
        super().__init__(env)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._episode_start: float = 0.0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_start = time.monotonic()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        self._episode_length += 1

        if terminated or truncated:
            elapsed = time.monotonic() - self._episode_start
            self.episode_returns.append(self._episode_return)
            self.episode_lengths.append(self._episode_length)
            self.episode_times.append(elapsed)
            info["episode"] = {
                "r": self._episode_return,
                "l": self._episode_length,
                "t": elapsed,
            }
        return obs, reward, terminated, truncated, info


class CurriculumWrapper(gym.Wrapper):
    """Adjust environment difficulty over the course of training.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    scheduler : callable
        ``(total_steps: int) -> float``  returning a difficulty value
        (typically in ``[0, 1]``) that is written to
        ``env.unwrapped.difficulty`` before each reset.
    """

    def __init__(self, env: gym.Env, scheduler: Callable[[int], float]):
        _require_gym()
        super().__init__(env)
        self.scheduler = scheduler
        self._total_steps: int = 0

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        difficulty = self.scheduler(self._total_steps)
        if hasattr(self.env.unwrapped, "difficulty"):
            self.env.unwrapped.difficulty = difficulty  # type: ignore[attr-defined]
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        self._total_steps += 1
        return result


class DomainRandomization(gym.Wrapper):
    """Randomise environment parameters at the start of each episode.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    randomization_config : dict[str, tuple[float, float]]
        Mapping of attribute names on ``env.unwrapped`` to ``(low, high)``
        ranges.  At each reset, each attribute is sampled uniformly from
        its range.
    seed : int | None
        Optional seed for the internal RNG.
    """

    def __init__(
        self,
        env: gym.Env,
        randomization_config: dict[str, tuple[float, float]],
        seed: int | None = None,
    ):
        _require_gym()
        super().__init__(env)
        self.randomization_config = randomization_config
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        for attr, (lo, hi) in self.randomization_config.items():
            value = float(self._rng.uniform(lo, hi))
            if hasattr(self.env.unwrapped, attr):
                setattr(self.env.unwrapped, attr, value)
        return self.env.reset(**kwargs)


class VecEnvWrapper:
    """Thin adapter for vectorised (batched) environments.

    Wraps a list of ``gym.Env`` instances and exposes batched ``reset`` /
    ``step`` interfaces.  For production use prefer ``gymnasium.vector``
    or Stable-Baselines3 ``SubprocVecEnv``, but this lightweight version
    is handy for testing and prototyping.

    Parameters
    ----------
    env_fns : list[callable]
        List of zero-argument callables, each returning a ``gym.Env``.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        _require_gym()
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        results = [env.reset() for env in self.envs]
        obs = np.stack([r[0] for r in results])
        infos = [r[1] for r in results]
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminateds = np.array([r[2] for r in results], dtype=bool)
        truncateds = np.array([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs, rewards, terminateds, truncateds, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "FrameStack",
    "ActionRepeat",
    "TimeLimit",
    "RewardShaping",
    "NormalizeObservation",
    "NormalizeReward",
    "ClipAction",
    "FlattenObservation",
    "RecordEpisode",
    "MonitorWrapper",
    "CurriculumWrapper",
    "DomainRandomization",
    "VecEnvWrapper",
    "RelativeObservation",
    "GoalConditioned",
]
