"""Vectorized environment execution for parallel data collection.

Provides vectorized environment wrappers that run multiple environment
instances in parallel (via subprocesses or sequentially) and apply
common transformations like observation normalization, frame stacking,
and episode monitoring.
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

# Exports: SubprocVecEnv, DummyVecEnv, VecEnvWrapper, VecNormalize,
#          VecFrameStack, VecMonitor, AsyncVecEnv

__all__ = [
    "SubprocVecEnv",
    "DummyVecEnv",
    "VecEnvWrapper",
    "VecNormalize",
    "VecFrameStack",
    "VecMonitor",
    "AsyncVecEnv",
]


class BaseVecEnv(ABC):
    """Abstract base class for vectorized environments.

    Defines the common interface that all vectorized environment
    implementations must follow.

    Attributes:
        num_envs: Number of environments being managed.
        observation_space: Observation space of the environments.
        action_space: Action space of the environments.
    """

    def __init__(self, num_envs: int, observation_space: Any, action_space: Any) -> None:
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Take a step in all environments simultaneously.

        Args:
            actions: Array of actions, one per environment.

        Returns:
            Tuple of (observations, rewards, dones, infos).
        """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset all environments.

        Returns:
            Stacked observations from all environments.
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""

    def step_async(self, actions: np.ndarray) -> None:
        """Begin stepping asynchronously. Default implementation is synchronous."""
        self._pending_actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for async step to complete. Default calls step()."""
        return self.step(self._pending_actions)


class DummyVecEnv(BaseVecEnv):
    """Sequential (single-process) vectorized environment for debugging.

    Runs all environments in the same process sequentially. Useful for
    debugging since errors will be raised directly in the main process.

    Args:
        env_fns: List of callables, each returning an environment instance.
    """

    def __init__(self, env_fns: list[Callable]) -> None:
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(
            num_envs=len(env_fns),
            observation_space=getattr(env, "observation_space", None),
            action_space=getattr(env, "action_space", None),
        )
        self._observations: list[np.ndarray] | None = None

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments sequentially.

        Automatically resets environments that return done=True.

        Args:
            actions: Array of actions, shape (num_envs, *action_shape).

        Returns:
            Tuple of (obs_batch, rewards, dones, infos).
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for _i, (env, action) in enumerate(zip(self.envs, actions, strict=False)):
            obs, reward, done, info = env.step(action)
            if done:
                terminal_obs = obs
                obs = env.reset()
                info["terminal_observation"] = terminal_obs
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations.

        Returns:
            Observations array of shape (num_envs, *obs_shape).
        """
        observations = [env.reset() for env in self.envs]
        return np.stack(observations)

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    def get_attr(self, attr_name: str) -> list[Any]:
        """Get an attribute from each environment.

        Args:
            attr_name: Name of the attribute.

        Returns:
            List of attribute values, one per environment.
        """
        return [getattr(env, attr_name) for env in self.envs]

    def env_method(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method on each environment.

        Args:
            method_name: Name of the method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            List of return values, one per environment.
        """
        return [getattr(env, method_name)(*args, **kwargs) for env in self.envs]


def _worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn: Callable,
) -> None:
    """Worker function that runs in a subprocess to manage a single environment.

    Receives commands over a pipe and executes them on the environment.

    Args:
        remote: Connection to receive commands and send results.
        parent_remote: Parent's end of the pipe (closed in the child process).
        env_fn: Callable that creates the environment instance.
    """
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                info["terminal_observation"] = obs
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            remote.send(obs)
        elif cmd == "close":
            if hasattr(env, "close"):
                env.close()
            remote.close()
            break
        elif cmd == "get_attr":
            remote.send(getattr(env, data))
        elif cmd == "env_method":
            method_name, args, kwargs = data
            remote.send(getattr(env, method_name)(*args, **kwargs))
        elif cmd == "get_spaces":
            remote.send(
                (
                    getattr(env, "observation_space", None),
                    getattr(env, "action_space", None),
                )
            )
        else:
            raise ValueError(f"Unknown command: {cmd}")


class SubprocVecEnv(BaseVecEnv):
    """Multiprocess vectorized environment.

    Each environment runs in its own subprocess, allowing true parallel
    execution. Communication is done via multiprocessing pipes.

    Args:
        env_fns: List of callables, each returning an environment instance.
        start_method: Multiprocessing start method ('forkserver', 'spawn', or 'fork').
    """

    def __init__(
        self,
        env_fns: list[Callable],
        start_method: str | None = None,
    ) -> None:
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"

        ctx = mp.get_context(start_method)

        self.remotes: list[Connection] = []
        self.work_remotes: list[Connection] = []
        self.processes: list[Process] = []

        for env_fn in env_fns:
            parent_remote, work_remote = ctx.Pipe()
            self.remotes.append(parent_remote)
            self.work_remotes.append(work_remote)
            process = ctx.Process(
                target=_worker,
                args=(work_remote, parent_remote, env_fn),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(
            num_envs=n_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments in parallel via subprocesses.

        Args:
            actions: Array of actions, shape (num_envs, *action_shape).

        Returns:
            Tuple of (obs_batch, rewards, dones, infos).
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        """Send step commands to all subprocesses without waiting.

        Args:
            actions: Array of actions, one per environment.
        """
        for remote, action in zip(self.remotes, actions, strict=False):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for all subprocesses to complete their step.

        Returns:
            Tuple of (obs_batch, rewards, dones, infos).
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        observations, rewards, dones, infos = zip(*results, strict=False)
        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            list(infos),
        )

    def reset(self) -> np.ndarray:
        """Reset all environments via subprocesses.

        Returns:
            Observations array of shape (num_envs, *obs_shape).
        """
        for remote in self.remotes:
            remote.send(("reset", None))
        observations = [remote.recv() for remote in self.remotes]
        return np.stack(observations)

    def close(self) -> None:
        """Shut down all subprocesses and clean up."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name: str) -> list[Any]:
        """Get an attribute from each subprocess environment.

        Args:
            attr_name: Name of the attribute.

        Returns:
            List of attribute values.
        """
        for remote in self.remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in self.remotes]

    def env_method(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method on each subprocess environment.

        Args:
            method_name: Name of the method.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            List of return values.
        """
        for remote in self.remotes:
            remote.send(("env_method", (method_name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]


class AsyncVecEnv(BaseVecEnv):
    """Asynchronous vectorized environment that does not wait for all envs.

    Unlike SubprocVecEnv which blocks until all environments complete their
    step, AsyncVecEnv allows environments to progress independently. Fast
    environments are not bottlenecked by slow ones.

    Args:
        env_fns: List of callables, each returning an environment instance.
        start_method: Multiprocessing start method.
    """

    def __init__(
        self,
        env_fns: list[Callable],
        start_method: str | None = None,
    ) -> None:
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"

        ctx = mp.get_context(start_method)

        self.remotes: list[Connection] = []
        self.work_remotes: list[Connection] = []
        self.processes: list[Process] = []

        for env_fn in env_fns:
            parent_remote, work_remote = ctx.Pipe()
            self.remotes.append(parent_remote)
            self.work_remotes.append(work_remote)
            process = ctx.Process(
                target=_worker,
                args=(work_remote, parent_remote, env_fn),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(
            num_envs=n_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self._pending: list[bool] = [False] * n_envs
        self._last_results: list[tuple | None] = [None] * n_envs

    def step_async(self, actions: np.ndarray) -> None:
        """Send step commands to all environments without blocking.

        Args:
            actions: Array of actions, one per environment.
        """
        for i, (remote, action) in enumerate(zip(self.remotes, actions, strict=False)):
            remote.send(("step", action))
            self._pending[i] = True

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for all pending environments to finish stepping.

        Returns:
            Tuple of (obs_batch, rewards, dones, infos).
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, remote in enumerate(self.remotes):
            if self._pending[i]:
                result = remote.recv()
                self._last_results[i] = result
                self._pending[i] = False
            else:
                result = self._last_results[i]

            obs, reward, done, info = result
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def step_env(self, env_idx: int, action: np.ndarray) -> None:
        """Send a step command to a single environment.

        Args:
            env_idx: Index of the environment.
            action: Action for that environment.
        """
        self.remotes[env_idx].send(("step", action))
        self._pending[env_idx] = True

    def poll(self) -> list[int]:
        """Check which environments have completed their steps.

        Returns:
            List of environment indices that have results ready.
        """
        ready = []
        for i, remote in enumerate(self.remotes):
            if self._pending[i] and remote.poll():
                ready.append(i)
        return ready

    def recv_env(self, env_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        """Receive the result from a specific environment.

        Args:
            env_idx: Index of the environment.

        Returns:
            Tuple of (obs, reward, done, info).
        """
        result = self.remotes[env_idx].recv()
        self._pending[env_idx] = False
        self._last_results[env_idx] = result
        return result

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Synchronous step for interface compatibility.

        Args:
            actions: Array of actions.

        Returns:
            Tuple of (obs_batch, rewards, dones, infos).
        """
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> np.ndarray:
        """Reset all environments.

        Returns:
            Observations array of shape (num_envs, *obs_shape).
        """
        for remote in self.remotes:
            remote.send(("reset", None))
        observations = [remote.recv() for remote in self.remotes]
        self._pending = [False] * self.num_envs
        return np.stack(observations)

    def close(self) -> None:
        """Shut down all subprocesses."""
        if self.closed:
            return
        for i, remote in enumerate(self.remotes):
            if self._pending[i]:
                remote.recv()
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


class VecEnvWrapper(BaseVecEnv):
    """Base class for vectorized environment wrappers.

    Wraps an existing vectorized environment and delegates all calls to it
    by default. Subclasses override specific methods to modify behavior.

    Args:
        venv: The vectorized environment to wrap.
    """

    def __init__(self, venv: BaseVecEnv) -> None:
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step the wrapped environment.

        Args:
            actions: Array of actions.

        Returns:
            Tuple of (obs_batch, rewards, dones, infos).
        """
        return self.venv.step(actions)

    def reset(self) -> np.ndarray:
        """Reset the wrapped environment.

        Returns:
            Observations array.
        """
        return self.venv.reset()

    def close(self) -> None:
        """Close the wrapped environment."""
        return self.venv.close()

    def step_async(self, actions: np.ndarray) -> None:
        """Async step on the wrapped environment."""
        self.venv.step_async(actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for async step on the wrapped environment."""
        return self.venv.step_wait()


class RunningMeanStd:
    """Tracks running mean and variance using Welford's online algorithm.

    Used internally by VecNormalize to compute observation and reward
    statistics incrementally.

    Args:
        shape: Shape of the values being tracked.
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a new batch of values.

        Args:
            batch: Array of new values, shape (batch_size, *shape).
        """
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class VecNormalize(VecEnvWrapper):
    """Normalize observations and rewards across vectorized environments.

    Maintains running mean and standard deviation statistics and uses them
    to normalize observations and optionally rewards. Statistics are updated
    only during training.

    Args:
        venv: The vectorized environment to wrap.
        norm_obs: Whether to normalize observations.
        norm_reward: Whether to normalize rewards.
        clip_obs: Maximum absolute value for clipped normalized observations.
        clip_reward: Maximum absolute value for clipped normalized rewards.
        gamma: Discount factor for reward normalization (running discounted return).
        epsilon: Small constant for numerical stability in division.
    """

    def __init__(
        self,
        venv: BaseVecEnv,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(venv)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        obs_shape = ()
        if hasattr(venv.observation_space, "shape"):
            obs_shape = venv.observation_space.shape

        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(venv.num_envs, dtype=np.float32)

        self.training = True

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step and normalize observations and rewards.

        Args:
            actions: Array of actions.

        Returns:
            Tuple of (normalized_obs, normalized_rewards, dones, infos).
        """
        obs, rewards, dones, infos = self.venv.step(actions)

        self.returns = self.returns * self.gamma + rewards

        if self.training:
            self.obs_rms.update(obs)
            self.ret_rms.update(self.returns.reshape(-1))

        obs = self._normalize_obs(obs)
        rewards = self._normalize_reward(rewards)

        self.returns[dones] = 0.0

        return obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """Reset and normalize initial observations.

        Returns:
            Normalized observations array.
        """
        obs = self.venv.reset()
        self.returns = np.zeros(self.num_envs, dtype=np.float32)

        if self.training:
            self.obs_rms.update(obs)

        return self._normalize_obs(obs)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics.

        Args:
            obs: Raw observations.

        Returns:
            Normalized and clipped observations.
        """
        if self.norm_obs:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs.astype(np.float32)

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running return statistics.

        Args:
            rewards: Raw rewards.

        Returns:
            Normalized and clipped rewards.
        """
        if self.norm_reward:
            rewards = rewards / np.sqrt(self.ret_rms.var + self.epsilon)
            rewards = np.clip(rewards, -self.clip_reward, self.clip_reward)
        return rewards.astype(np.float32)

    def set_training(self, training: bool) -> None:
        """Toggle training mode for statistics updates.

        Args:
            training: If True, update running statistics. If False, freeze them.
        """
        self.training = training


class VecFrameStack(VecEnvWrapper):
    """Frame stacking wrapper for vectorized environments.

    Stacks the last n_stack observations along a new first axis. Useful
    for providing temporal context to policies that process raw frames.

    Args:
        venv: The vectorized environment to wrap.
        n_stack: Number of consecutive frames to stack.
    """

    def __init__(self, venv: BaseVecEnv, n_stack: int = 4) -> None:
        super().__init__(venv)
        self.n_stack = n_stack

        obs_shape = ()
        if hasattr(venv.observation_space, "shape"):
            obs_shape = venv.observation_space.shape

        self._obs_shape = obs_shape
        self.stacked_obs = np.zeros((venv.num_envs, n_stack, *obs_shape), dtype=np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step and update the frame stack.

        Environments that terminate have their stacked frames cleared.

        Args:
            actions: Array of actions.

        Returns:
            Tuple of (stacked_obs, rewards, dones, infos).
        """
        obs, rewards, dones, infos = self.venv.step(actions)

        for i, done in enumerate(dones):
            if done:
                self.stacked_obs[i] = 0.0

        self.stacked_obs = np.roll(self.stacked_obs, shift=-1, axis=1)
        self.stacked_obs[:, -1] = obs

        return self.stacked_obs.copy(), rewards, dones, infos

    def reset(self) -> np.ndarray:
        """Reset and initialize the frame stack.

        Returns:
            Stacked observations with the first frame replicated.
        """
        obs = self.venv.reset()
        self.stacked_obs[:] = 0.0
        self.stacked_obs[:, -1] = obs
        return self.stacked_obs.copy()


class VecMonitor(VecEnvWrapper):
    """Episode statistics monitoring for vectorized environments.

    Tracks episode rewards and lengths across all environments and provides
    access to recent episode statistics.

    Args:
        venv: The vectorized environment to wrap.
        info_keywords: Additional info dict keys to track per episode.
    """

    def __init__(
        self,
        venv: BaseVecEnv,
        info_keywords: tuple[str, ...] = (),
    ) -> None:
        super().__init__(venv)
        self.info_keywords = info_keywords

        self.episode_rewards = np.zeros(venv.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(venv.num_envs, dtype=np.int64)
        self.episode_count = 0

        self._episode_reward_history: deque = deque(maxlen=100)
        self._episode_length_history: deque = deque(maxlen=100)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step and accumulate episode statistics.

        When an episode ends, the total reward and length are recorded in
        the info dict under the 'episode' key and added to the history.

        Args:
            actions: Array of actions.

        Returns:
            Tuple of (obs_batch, rewards, dones, infos) where infos for
            completed episodes contain an 'episode' key with 'r' (reward)
            and 'l' (length) entries.
        """
        obs, rewards, dones, infos = self.venv.step(actions)

        self.episode_rewards += rewards
        self.episode_lengths += 1

        for i, done in enumerate(dones):
            if done:
                episode_info = {
                    "r": float(self.episode_rewards[i]),
                    "l": int(self.episode_lengths[i]),
                }

                for key in self.info_keywords:
                    if key in infos[i]:
                        episode_info[key] = infos[i][key]

                infos[i]["episode"] = episode_info

                self._episode_reward_history.append(self.episode_rewards[i])
                self._episode_length_history.append(self.episode_lengths[i])
                self.episode_count += 1

                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        return obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        """Reset all environments and clear current episode accumulators.

        Returns:
            Observations array.
        """
        obs = self.venv.reset()
        self.episode_rewards[:] = 0.0
        self.episode_lengths[:] = 0
        return obs

    def get_episode_rewards(self) -> list[float]:
        """Get recent episode total rewards.

        Returns:
            List of the last 100 episode rewards.
        """
        return list(self._episode_reward_history)

    def get_episode_lengths(self) -> list[int]:
        """Get recent episode lengths.

        Returns:
            List of the last 100 episode lengths.
        """
        return list(self._episode_length_history)

    @property
    def mean_reward(self) -> float:
        """Mean reward over recent episodes, or 0.0 if no episodes completed."""
        if len(self._episode_reward_history) == 0:
            return 0.0
        return float(np.mean(self._episode_reward_history))

    @property
    def mean_length(self) -> float:
        """Mean length over recent episodes, or 0.0 if no episodes completed."""
        if len(self._episode_length_history) == 0:
            return 0.0
        return float(np.mean(self._episode_length_history))
