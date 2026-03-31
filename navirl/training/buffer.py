"""Experience replay buffers for RL training.

Provides various buffer implementations for storing and sampling transitions
during reinforcement learning training, including uniform, prioritized,
n-step, hindsight, sequence, rollout, multi-agent, and demonstration buffers.
"""

from collections import deque

import numpy as np

# Exports: ReplayBuffer, PrioritizedReplayBuffer, NStepBuffer, HindsightReplayBuffer,
#          SequenceBuffer, RolloutBuffer, MultiAgentBuffer, DemonstrationBuffer

__all__ = [
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "NStepBuffer",
    "HindsightReplayBuffer",
    "SequenceBuffer",
    "RolloutBuffer",
    "MultiAgentBuffer",
    "DemonstrationBuffer",
]


class ReplayBuffer:
    """Standard uniform experience replay buffer.

    Stores transitions as (obs, action, reward, next_obs, done) tuples in
    pre-allocated numpy arrays with circular overwriting when capacity is reached.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
        """
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of transitions uniformly.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self._size


class _SumTree:
    """Binary sum tree for efficient priority-based sampling.

    Supports O(log n) update and sampling operations. Each leaf stores
    a priority value, and internal nodes store the sum of their children.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def update(self, tree_index: int, priority: float) -> None:
        """Update the priority of a leaf node and propagate changes upward.

        Args:
            tree_index: Index of the leaf node in the tree array.
            priority: New priority value.
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> tuple[int, float, int]:
        """Retrieve a leaf node by traversing the tree with a cumulative value.

        Args:
            value: A value in [0, total_priority) used to select a leaf.

        Returns:
            Tuple of (tree_index, priority, data_index).
        """
        parent_index = 0
        while True:
            left_child = 2 * parent_index + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                leaf_index = parent_index
                break

            if value <= self.tree[left_child]:
                parent_index = left_child
            else:
                value -= self.tree[left_child]
                parent_index = right_child

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index

    @property
    def total_priority(self) -> float:
        """Return the sum of all priorities (root node value)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Return the maximum priority among all leaves."""
        leaves = self.tree[self.capacity - 1 :]
        return np.max(leaves) if np.any(leaves > 0) else 1.0


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay buffer (Schaul et al. 2015).

    Uses a sum tree data structure for efficient O(log n) priority-based
    sampling. Transitions with higher TD-error are sampled more frequently.
    Importance sampling weights are provided to correct for the non-uniform
    sampling bias.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
        alpha: Priority exponent controlling how much prioritization is used.
            0.0 corresponds to uniform sampling, 1.0 to full prioritization.
        beta: Initial importance sampling exponent for bias correction.
            Should be annealed from this initial value to 1.0 during training.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.alpha = alpha
        self.beta = beta

        self._tree = _SumTree(capacity)
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0
        self._max_priority = 1.0
        self._epsilon = 1e-6

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum priority.

        New transitions are added with the current maximum priority so they
        are sampled at least once before their priority is updated.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
        """
        tree_index = self._pos + self._tree.capacity - 1
        priority = self._max_priority**self.alpha

        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = float(done)

        self._tree.update(tree_index, priority)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Sample a batch of transitions with probability proportional to priority.

        The priority range is divided into equal segments, and one transition
        is sampled uniformly from each segment for stratified sampling.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (batch_dict, importance_weights, tree_indices) where
            batch_dict contains the transition data, importance_weights are
            for bias correction, and tree_indices are needed for priority updates.
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self._tree.total_priority / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            tree_idx, priority, data_idx = self._tree.get_leaf(value)
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = priority

        sampling_probs = priorities / self._tree.total_priority
        weights = (self._size * sampling_probs) ** (-self.beta)
        weights /= weights.max()

        batch = {
            "obs": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "dones": self.dones[indices],
        }
        return batch, weights.astype(np.float32), tree_indices

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled transitions based on TD-errors.

        Args:
            tree_indices: Tree indices returned from sample().
            td_errors: Absolute TD-errors for each sampled transition.
        """
        priorities = (np.abs(td_errors) + self._epsilon) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self._tree.update(int(tree_idx), priority)
        self._max_priority = max(
            self._max_priority, float(np.max(np.abs(td_errors) + self._epsilon))
        )

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self._size


class NStepBuffer:
    """N-step return replay buffer.

    Accumulates n-step returns before storing transitions. The reward for
    each stored transition is the discounted sum of the next n rewards,
    and the next observation is the observation n steps into the future.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
        n_step: Number of steps for multi-step returns.
        gamma: Discount factor applied to future rewards.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        n_step: int = 3,
        gamma: float = 0.99,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.n_step = n_step
        self.gamma = gamma

        self._buffer = ReplayBuffer(capacity, obs_shape, action_shape)
        self._n_step_buffer: deque = deque(maxlen=n_step)

    def _compute_n_step_return(self) -> tuple[float, np.ndarray, bool]:
        """Compute the n-step discounted return from the pending buffer.

        Returns:
            Tuple of (n_step_reward, n_step_next_obs, n_step_done).
        """
        reward = 0.0
        for i, (_, _, r, next_obs, done) in enumerate(self._n_step_buffer):
            reward += (self.gamma**i) * r
            if done:
                return reward, next_obs, True
        return reward, next_obs, done

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition, storing the n-step version when ready.

        Transitions are buffered until n steps are accumulated. When the
        buffer is full or an episode ends, the n-step transition is computed
        and stored in the underlying replay buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Single-step reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
        """
        self._n_step_buffer.append((obs, action, reward, next_obs, done))

        if len(self._n_step_buffer) == self.n_step:
            n_reward, n_next_obs, n_done = self._compute_n_step_return()
            first_obs, first_action, _, _, _ = self._n_step_buffer[0]
            self._buffer.add(first_obs, first_action, n_reward, n_next_obs, n_done)

        if done:
            while len(self._n_step_buffer) > 0:
                n_reward, n_next_obs, n_done = self._compute_n_step_return()
                first_obs, first_action, _, _, _ = self._n_step_buffer[0]
                self._buffer.add(first_obs, first_action, n_reward, n_next_obs, n_done)
                self._n_step_buffer.popleft()

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of n-step transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'.
        """
        return self._buffer.sample(batch_size)

    def __len__(self) -> int:
        """Return the current number of stored n-step transitions."""
        return len(self._buffer)


class HindsightReplayBuffer:
    """Hindsight Experience Replay buffer (Andrychowicz et al. 2017).

    Stores goal-conditioned transitions and generates additional transitions
    with substituted goals during sampling. This allows the agent to learn
    from failed attempts by relabeling the achieved goal as the desired goal.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
        goal_shape: Shape of goal arrays.
        strategy: Goal relabeling strategy. One of 'future', 'final', 'episode'.
            'future' selects goals from future states in the same episode,
            'final' uses the final state of the episode,
            'episode' selects randomly from the episode.
        k: Number of additional hindsight goals per real transition.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        goal_shape: tuple[int, ...],
        strategy: str = "future",
        k: int = 4,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.goal_shape = goal_shape
        self.strategy = strategy
        self.k = k

        assert strategy in ("future", "final", "episode"), f"Unknown strategy: {strategy}"

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.desired_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        self.achieved_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)

        self._pos = 0
        self._size = 0

        self._current_episode: list[dict[str, np.ndarray]] = []
        self._episode_starts: list[int] = []
        self._episode_lengths: list[int] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> None:
        """Store a goal-conditioned transition.

        Transitions are accumulated into episodes. When an episode ends,
        hindsight transitions with relabeled goals are also stored.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
            desired_goal: The goal the agent was trying to achieve.
            achieved_goal: The goal the agent actually achieved.
        """
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }
        self._current_episode.append(transition)

        self._store_transition(obs, action, reward, next_obs, done, desired_goal, achieved_goal)

        if done:
            self._generate_hindsight_transitions()
            self._current_episode = []

    def _store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        desired_goal: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> None:
        """Write a single transition into the circular buffer arrays."""
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = float(done)
        self.desired_goals[self._pos] = desired_goal
        self.achieved_goals[self._pos] = achieved_goal

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _generate_hindsight_transitions(self) -> None:
        """Generate hindsight transitions for the completed episode.

        For each transition in the episode, generates k additional transitions
        with goals relabeled according to the chosen strategy.
        """
        episode = self._current_episode
        episode_len = len(episode)

        for idx in range(episode_len):
            t = episode[idx]
            for _ in range(self.k):
                if self.strategy == "future":
                    future_idx = np.random.randint(idx, episode_len)
                elif self.strategy == "final":
                    future_idx = episode_len - 1
                elif self.strategy == "episode":
                    future_idx = np.random.randint(0, episode_len)
                else:
                    continue

                new_goal = episode[future_idx]["achieved_goal"]
                new_reward = self._compute_reward(t["achieved_goal"], new_goal)
                self._store_transition(
                    t["obs"],
                    t["action"],
                    new_reward,
                    t["next_obs"],
                    t["done"],
                    new_goal,
                    t["achieved_goal"],
                )

    @staticmethod
    def _compute_reward(
        achieved_goal: np.ndarray, desired_goal: np.ndarray, threshold: float = 0.05
    ) -> float:
        """Compute a sparse reward based on goal distance.

        Args:
            achieved_goal: The goal actually achieved.
            desired_goal: The target goal.
            threshold: Distance threshold for success.

        Returns:
            0.0 if the achieved goal is within the threshold of the desired
            goal, -1.0 otherwise.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if distance < threshold else -1.0

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch of goal-conditioned transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys 'obs', 'actions', 'rewards', 'next_obs',
            'dones', 'desired_goals', 'achieved_goals'.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "dones": self.dones[indices],
            "desired_goals": self.desired_goals[indices],
            "achieved_goals": self.achieved_goals[indices],
        }

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self._size


class SequenceBuffer:
    """Buffer for storing and sampling fixed-length sequences for recurrent policies.

    Stores complete episodes and samples contiguous subsequences of a fixed
    length. Useful for training recurrent neural network policies (e.g., LSTMs).

    Args:
        capacity: Maximum number of individual transitions to store.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
        seq_len: Length of sequences to sample.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        seq_len: int = 20,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.seq_len = seq_len

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._size = 0
        self._episode_starts: list[int] = []
        self._episode_lengths: list[int] = []
        self._current_episode_start: int = 0
        self._current_episode_len: int = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition and track episode boundaries.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
        """
        if self._current_episode_len == 0:
            self._current_episode_start = self._pos

        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._current_episode_len += 1

        if done:
            if self._current_episode_len >= self.seq_len:
                self._episode_starts.append(self._current_episode_start)
                self._episode_lengths.append(self._current_episode_len)
            self._current_episode_len = 0

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of fixed-length sequences.

        Each sequence is a contiguous run of transitions from a single episode.

        Args:
            batch_size: Number of sequences to sample.

        Returns:
            Dictionary with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones',
            each with shape (batch_size, seq_len, ...).
        """
        batch_obs = np.zeros((batch_size, self.seq_len, *self.obs_shape), dtype=np.float32)
        batch_actions = np.zeros((batch_size, self.seq_len, *self.action_shape), dtype=np.float32)
        batch_rewards = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        batch_next_obs = np.zeros((batch_size, self.seq_len, *self.obs_shape), dtype=np.float32)
        batch_dones = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        for i in range(batch_size):
            ep_idx = np.random.randint(0, len(self._episode_starts))
            ep_start = self._episode_starts[ep_idx]
            ep_len = self._episode_lengths[ep_idx]
            offset = np.random.randint(0, ep_len - self.seq_len + 1)
            start = (ep_start + offset) % self.capacity

            for j in range(self.seq_len):
                idx = (start + j) % self.capacity
                batch_obs[i, j] = self.observations[idx]
                batch_actions[i, j] = self.actions[idx]
                batch_rewards[i, j] = self.rewards[idx]
                batch_next_obs[i, j] = self.next_observations[idx]
                batch_dones[i, j] = self.dones[idx]

        return {
            "obs": batch_obs,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "next_obs": batch_next_obs,
            "dones": batch_dones,
        }

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return self._size


class RolloutBuffer:
    """On-policy rollout storage buffer for algorithms like PPO and A2C.

    Collects a fixed number of environment steps, then computes returns
    and advantages using Generalized Advantage Estimation (GAE). The buffer
    is meant to be filled once, used for training, then reset.

    Args:
        buffer_size: Number of steps to store per environment.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
        n_envs: Number of parallel environments contributing data.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        n_envs: int = 1,
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.n_envs = n_envs

        self.observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)

        self._pos = 0
        self._full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Store a single timestep of data from all environments.

        Args:
            obs: Observations from all envs, shape (n_envs, *obs_shape).
            action: Actions taken, shape (n_envs, *action_shape).
            reward: Rewards received, shape (n_envs,).
            value: Value estimates, shape (n_envs,).
            log_prob: Log probabilities of actions, shape (n_envs,).
            done: Done flags, shape (n_envs,).
        """
        if self._pos >= self.buffer_size:
            raise BufferError("Rollout buffer is full. Call reset() before adding more data.")

        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.values[self._pos] = value
        self.log_probs[self._pos] = log_prob
        self.dones[self._pos] = done

        self._pos += 1
        if self._pos == self.buffer_size:
            self._full = True

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute returns and GAE advantages for the collected rollout.

        Uses Generalized Advantage Estimation (GAE-Lambda) to compute
        advantages, which provides a bias-variance tradeoff controlled by
        the gae_lambda parameter.

        Args:
            last_value: Value estimate for the state after the last step,
                shape (n_envs,).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter. 0.0 gives one-step TD advantage,
                1.0 gives Monte Carlo advantage.
        """
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_values = last_value
            else:
                next_values = self.values[step + 1]

            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random batch from the flattened rollout data.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys 'obs', 'actions', 'values', 'log_probs',
            'advantages', 'returns'.
        """
        total_size = self.buffer_size * self.n_envs
        indices = np.random.randint(0, total_size, size=batch_size)

        flat_obs = self.observations.reshape(-1, *self.obs_shape)
        flat_actions = self.actions.reshape(-1, *self.action_shape)
        flat_values = self.values.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)

        return {
            "obs": flat_obs[indices],
            "actions": flat_actions[indices],
            "values": flat_values[indices],
            "log_probs": flat_log_probs[indices],
            "advantages": flat_advantages[indices],
            "returns": flat_returns[indices],
        }

    def reset(self) -> None:
        """Reset the buffer for a new rollout collection cycle."""
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        """Return the current number of timesteps stored across all envs."""
        return self._pos * self.n_envs


class MultiAgentBuffer:
    """Per-agent replay buffers for multi-agent reinforcement learning.

    Maintains a separate ReplayBuffer for each agent, allowing independent
    storage and sampling of transitions per agent.

    Args:
        num_agents: Number of agents.
        capacity: Maximum number of transitions per agent buffer.
        obs_shape: Shape of observation arrays (same for all agents).
        action_shape: Shape of action arrays (same for all agents).
    """

    def __init__(
        self,
        num_agents: int,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        self.num_agents = num_agents
        self.capacity = capacity
        self.buffers = [ReplayBuffer(capacity, obs_shape, action_shape) for _ in range(num_agents)]

    def add(
        self,
        agent_id: int,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition for a specific agent.

        Args:
            agent_id: Index of the agent (0-indexed).
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
        """
        self.buffers[agent_id].add(obs, action, reward, next_obs, done)

    def add_all(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Store transitions for all agents simultaneously.

        Args:
            obs: Observations, shape (num_agents, *obs_shape).
            actions: Actions, shape (num_agents, *action_shape).
            rewards: Rewards, shape (num_agents,).
            next_obs: Next observations, shape (num_agents, *obs_shape).
            dones: Done flags, shape (num_agents,).
        """
        for i in range(self.num_agents):
            self.buffers[i].add(obs[i], actions[i], float(rewards[i]), next_obs[i], bool(dones[i]))

    def sample(
        self, batch_size: int, agent_id: int | None = None
    ) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample per agent.
            agent_id: If specified, sample only from that agent's buffer.
                If None, sample from all agents and return a list.

        Returns:
            A single batch dict if agent_id is specified, otherwise a list
            of batch dicts (one per agent).
        """
        if agent_id is not None:
            return self.buffers[agent_id].sample(batch_size)
        return [buf.sample(batch_size) for buf in self.buffers]

    def __len__(self) -> int:
        """Return the total number of transitions across all agents."""
        return sum(len(buf) for buf in self.buffers)


class DemonstrationBuffer:
    """Buffer for loading and sampling from expert demonstrations.

    Supports loading pre-recorded demonstrations from disk and sampling
    them either alone or mixed with online experience data at a configurable
    ratio.

    Args:
        capacity: Maximum number of demonstration transitions to store.
        obs_shape: Shape of observation arrays.
        action_shape: Shape of action arrays.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a demonstration transition.

        Args:
            obs: Current observation.
            action: Expert action taken.
            reward: Reward received.
            next_obs: Next observation after action.
            done: Whether the episode terminated.
        """
        if self._size >= self.capacity:
            raise BufferError("Demonstration buffer is full.")
        self.observations[self._size] = obs
        self.actions[self._size] = action
        self.rewards[self._size] = reward
        self.next_observations[self._size] = next_obs
        self.dones[self._size] = float(done)
        self._size += 1

    def load_demonstrations(self, path: str) -> int:
        """Load expert demonstrations from a numpy archive file.

        The file should be a .npz archive containing arrays named 'obs',
        'actions', 'rewards', 'next_obs', and 'dones'.

        Args:
            path: Path to the .npz demonstration file.

        Returns:
            Number of transitions loaded.

        Raises:
            BufferError: If loading would exceed buffer capacity.
        """
        data = np.load(path)
        n_transitions = len(data["obs"])

        if self._size + n_transitions > self.capacity:
            n_transitions = self.capacity - self._size
            if n_transitions <= 0:
                raise BufferError("Demonstration buffer is full.")

        start = self._size
        end = start + n_transitions

        self.observations[start:end] = data["obs"][:n_transitions]
        self.actions[start:end] = data["actions"][:n_transitions]
        self.rewards[start:end] = data["rewards"][:n_transitions]
        self.next_observations[start:end] = data["next_obs"][:n_transitions]
        self.dones[start:end] = data["dones"][:n_transitions]

        self._size += n_transitions
        return n_transitions

    def sample(
        self,
        batch_size: int,
        online_buffer: ReplayBuffer | None = None,
        demo_ratio: float = 0.25,
    ) -> dict[str, np.ndarray]:
        """Sample a batch, optionally mixing with online experience data.

        When an online buffer is provided, the batch is composed of
        demo_ratio fraction demonstration data and (1 - demo_ratio) fraction
        online data.

        Args:
            batch_size: Total number of transitions to sample.
            online_buffer: Optional online replay buffer to mix with.
            demo_ratio: Fraction of the batch drawn from demonstrations
                when an online buffer is provided. Ignored if online_buffer
                is None.

        Returns:
            Dictionary with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'.
        """
        if online_buffer is not None and len(online_buffer) > 0:
            n_demo = int(batch_size * demo_ratio)
            n_online = batch_size - n_demo

            demo_indices = np.random.randint(0, self._size, size=n_demo)
            online_batch = online_buffer.sample(n_online)

            return {
                "obs": np.concatenate([self.observations[demo_indices], online_batch["obs"]]),
                "actions": np.concatenate([self.actions[demo_indices], online_batch["actions"]]),
                "rewards": np.concatenate([self.rewards[demo_indices], online_batch["rewards"]]),
                "next_obs": np.concatenate(
                    [self.next_observations[demo_indices], online_batch["next_obs"]]
                ),
                "dones": np.concatenate([self.dones[demo_indices], online_batch["dones"]]),
            }

        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """Return the current number of demonstration transitions stored."""
        return self._size
