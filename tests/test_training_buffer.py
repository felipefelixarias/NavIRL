"""Tests for navirl/training/buffer.py — all 8 buffer types.

Covers: ReplayBuffer, PrioritizedReplayBuffer, NStepBuffer,
HindsightReplayBuffer, SequenceBuffer, RolloutBuffer,
MultiAgentBuffer, DemonstrationBuffer.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from navirl.training.buffer import (
    DemonstrationBuffer,
    HindsightReplayBuffer,
    MultiAgentBuffer,
    NStepBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    SequenceBuffer,
)

OBS_SHAPE = (4,)
ACT_SHAPE = (2,)


def _make_transition(seed: int = 0):
    rng = np.random.RandomState(seed)
    return {
        "obs": rng.randn(*OBS_SHAPE).astype(np.float32),
        "action": rng.randn(*ACT_SHAPE).astype(np.float32),
        "reward": float(rng.randn()),
        "next_obs": rng.randn(*OBS_SHAPE).astype(np.float32),
        "done": bool(rng.randint(2)),
    }


# ============================================================
# ReplayBuffer
# ============================================================


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        assert len(buf) == 0
        t = _make_transition()
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = ReplayBuffer(5, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == 5

    def test_sample_shape(self):
        buf = ReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        for i in range(20):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(8)
        assert batch["obs"].shape == (8, *OBS_SHAPE)
        assert batch["actions"].shape == (8, *ACT_SHAPE)
        assert batch["rewards"].shape == (8,)
        assert batch["next_obs"].shape == (8, *OBS_SHAPE)
        assert batch["dones"].shape == (8,)

    def test_sample_returns_stored_data(self):
        buf = ReplayBuffer(10, OBS_SHAPE, ACT_SHAPE)
        t = _make_transition(42)
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(1)
        np.testing.assert_array_almost_equal(batch["obs"][0], t["obs"])

    def test_capacity_respected(self):
        cap = 3
        buf = ReplayBuffer(cap, OBS_SHAPE, ACT_SHAPE)
        for i in range(100):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == cap


# ============================================================
# PrioritizedReplayBuffer
# ============================================================


class TestPrioritizedReplayBuffer:
    def test_add_and_len(self):
        buf = PrioritizedReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        assert len(buf) == 0
        t = _make_transition()
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == 1

    def test_sample_has_weights_and_indices(self):
        buf = PrioritizedReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        for i in range(20):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(8)
        assert "weights" in batch
        assert "indices" in batch
        assert batch["weights"].shape == (8,)
        assert batch["indices"].shape == (8,)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(4)
        td_errors = np.array([0.5, 1.0, 0.1, 2.0])
        # Should not raise
        buf.update_priorities(batch["indices"], td_errors)

    def test_alpha_beta_params(self):
        buf = PrioritizedReplayBuffer(50, OBS_SHAPE, ACT_SHAPE, alpha=0.8, beta=0.6)
        assert buf.alpha == 0.8
        assert buf.beta == 0.6

    def test_circular_overwrite(self):
        buf = PrioritizedReplayBuffer(5, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == 5

    def test_weights_are_normalized(self):
        buf = PrioritizedReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        for i in range(30):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(10)
        assert float(np.max(batch["weights"])) == pytest.approx(1.0)


# ============================================================
# NStepBuffer
# ============================================================


class TestNStepBuffer:
    def test_n_step_accumulation(self):
        buf = NStepBuffer(100, OBS_SHAPE, ACT_SHAPE, n_step=3, gamma=0.99)
        # Add 3 transitions (not done), should store 1 n-step transition
        for i in range(3):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], 1.0, t["next_obs"], False)
        assert len(buf) == 1

    def test_done_flushes_buffer(self):
        buf = NStepBuffer(100, OBS_SHAPE, ACT_SHAPE, n_step=3, gamma=0.99)
        for i in range(2):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], 1.0, t["next_obs"], False)
        # End episode early
        t = _make_transition(99)
        buf.add(t["obs"], t["action"], 1.0, t["next_obs"], True)
        # All pending transitions should be flushed
        assert len(buf) >= 1

    def test_n_step_return_value(self):
        """Verify the discounted return: r0 + gamma*r1 + gamma^2*r2."""
        gamma = 0.5
        buf = NStepBuffer(100, OBS_SHAPE, ACT_SHAPE, n_step=3, gamma=gamma)
        obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        act = np.zeros(ACT_SHAPE, dtype=np.float32)
        # rewards: 1.0, 2.0, 3.0
        buf.add(obs, act, 1.0, obs, False)
        buf.add(obs, act, 2.0, obs, False)
        buf.add(obs, act, 3.0, obs, False)
        # Expected: 1 + 0.5*2 + 0.25*3 = 2.75
        assert len(buf) == 1
        batch = buf.sample(1)
        assert batch["rewards"][0] == pytest.approx(2.75, abs=1e-5)

    def test_sample_shape(self):
        buf = NStepBuffer(100, OBS_SHAPE, ACT_SHAPE, n_step=2, gamma=0.99)
        for i in range(20):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], 1.0, t["next_obs"], i % 5 == 4)
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, *OBS_SHAPE)


# ============================================================
# HindsightReplayBuffer
# ============================================================


class TestHindsightReplayBuffer:
    GOAL_SHAPE = (2,)

    def test_add_and_len(self):
        buf = HindsightReplayBuffer(200, OBS_SHAPE, ACT_SHAPE, self.GOAL_SHAPE)
        assert len(buf) == 0
        obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        act = np.zeros(ACT_SHAPE, dtype=np.float32)
        goal = np.array([1.0, 0.0], dtype=np.float32)
        achieved = np.array([0.5, 0.0], dtype=np.float32)
        buf.add(obs, act, -1.0, obs, False, goal, achieved)
        assert len(buf) == 1

    def test_hindsight_generates_extra_transitions(self):
        """After a done episode, k extra transitions per step should be generated."""
        k = 4
        buf = HindsightReplayBuffer(1000, OBS_SHAPE, ACT_SHAPE, self.GOAL_SHAPE, k=k)
        episode_len = 5
        obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        act = np.zeros(ACT_SHAPE, dtype=np.float32)
        goal = np.array([10.0, 10.0], dtype=np.float32)
        for i in range(episode_len):
            achieved = np.array([float(i), 0.0], dtype=np.float32)
            done = i == episode_len - 1
            buf.add(obs, act, -1.0, obs, done, goal, achieved)
        # Original transitions + k hindsight per step
        expected = episode_len + episode_len * k
        assert len(buf) == expected

    def test_strategies(self):
        for strategy in ("future", "final", "episode"):
            buf = HindsightReplayBuffer(
                500, OBS_SHAPE, ACT_SHAPE, self.GOAL_SHAPE, strategy=strategy, k=2
            )
            obs = np.zeros(OBS_SHAPE, dtype=np.float32)
            act = np.zeros(ACT_SHAPE, dtype=np.float32)
            goal = np.array([5.0, 5.0], dtype=np.float32)
            for i in range(3):
                achieved = np.array([float(i), 0.0], dtype=np.float32)
                buf.add(obs, act, -1.0, obs, i == 2, goal, achieved)
            assert len(buf) > 3  # hindsight added extra

    def test_invalid_strategy_raises(self):
        with pytest.raises(AssertionError):
            HindsightReplayBuffer(100, OBS_SHAPE, ACT_SHAPE, self.GOAL_SHAPE, strategy="invalid")

    def test_sample_has_goal_fields(self):
        buf = HindsightReplayBuffer(500, OBS_SHAPE, ACT_SHAPE, self.GOAL_SHAPE, k=2)
        obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        act = np.zeros(ACT_SHAPE, dtype=np.float32)
        goal = np.array([5.0, 5.0], dtype=np.float32)
        for i in range(5):
            achieved = np.array([float(i), 0.0], dtype=np.float32)
            buf.add(obs, act, -1.0, obs, i == 4, goal, achieved)
        batch = buf.sample(4)
        assert "desired_goals" in batch
        assert "achieved_goals" in batch
        assert batch["desired_goals"].shape == (4, *self.GOAL_SHAPE)

    def test_compute_reward_sparse(self):
        achieved = np.array([1.0, 0.0])
        desired_close = np.array([1.01, 0.0])
        desired_far = np.array([5.0, 5.0])
        assert HindsightReplayBuffer._compute_reward(achieved, desired_close) == 0.0
        assert HindsightReplayBuffer._compute_reward(achieved, desired_far) == -1.0


# ============================================================
# SequenceBuffer
# ============================================================


class TestSequenceBuffer:
    def test_add_and_len(self):
        buf = SequenceBuffer(100, OBS_SHAPE, ACT_SHAPE, seq_len=5)
        assert len(buf) == 0
        t = _make_transition()
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == 1

    def test_short_episodes_not_indexed(self):
        """Episodes shorter than seq_len should not be added to episode index."""
        buf = SequenceBuffer(100, OBS_SHAPE, ACT_SHAPE, seq_len=10)
        # Add a 3-step episode (too short for seq_len=10)
        for i in range(3):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], i == 2)
        assert len(buf._episode_starts) == 0

    def test_long_episode_indexed(self):
        buf = SequenceBuffer(200, OBS_SHAPE, ACT_SHAPE, seq_len=5)
        for i in range(10):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], i == 9)
        assert len(buf._episode_starts) == 1

    def test_sample_shape(self):
        buf = SequenceBuffer(200, OBS_SHAPE, ACT_SHAPE, seq_len=5)
        # Add a 20-step episode
        for i in range(20):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], i == 19)
        batch = buf.sample(3)
        assert batch["obs"].shape == (3, 5, *OBS_SHAPE)
        assert batch["actions"].shape == (3, 5, *ACT_SHAPE)
        assert batch["rewards"].shape == (3, 5)


# ============================================================
# RolloutBuffer
# ============================================================


class TestRolloutBuffer:
    def test_add_and_len(self):
        buf = RolloutBuffer(10, OBS_SHAPE, ACT_SHAPE, n_envs=2)
        assert len(buf) == 0
        obs = np.zeros((2, *OBS_SHAPE), dtype=np.float32)
        act = np.zeros((2, *ACT_SHAPE), dtype=np.float32)
        rew = np.zeros(2, dtype=np.float32)
        val = np.zeros(2, dtype=np.float32)
        lp = np.zeros(2, dtype=np.float32)
        done = np.zeros(2, dtype=np.float32)
        buf.add(obs, act, rew, val, lp, done)
        assert len(buf) == 2  # 1 step * 2 envs

    def test_overflow_raises(self):
        buf = RolloutBuffer(2, OBS_SHAPE, ACT_SHAPE, n_envs=1)
        obs = np.zeros((1, *OBS_SHAPE), dtype=np.float32)
        act = np.zeros((1, *ACT_SHAPE), dtype=np.float32)
        rew = np.zeros(1, dtype=np.float32)
        val = np.zeros(1, dtype=np.float32)
        lp = np.zeros(1, dtype=np.float32)
        done = np.zeros(1, dtype=np.float32)
        buf.add(obs, act, rew, val, lp, done)
        buf.add(obs, act, rew, val, lp, done)
        with pytest.raises(BufferError):
            buf.add(obs, act, rew, val, lp, done)

    def test_compute_gae(self):
        buf = RolloutBuffer(3, OBS_SHAPE, ACT_SHAPE, n_envs=1)
        obs = np.zeros((1, *OBS_SHAPE), dtype=np.float32)
        act = np.zeros((1, *ACT_SHAPE), dtype=np.float32)
        for _i in range(3):
            buf.add(obs, act, np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        last_val = np.array([0.0], dtype=np.float32)
        buf.compute_returns_and_advantages(last_val, gamma=1.0, gae_lambda=1.0)
        # With gamma=1, lambda=1, values=0, rewards=[1,1,1], last_value=0
        # advantages should be [3, 2, 1] (MC returns)
        np.testing.assert_array_almost_equal(
            buf.advantages[:, 0], [3.0, 2.0, 1.0]
        )

    def test_reset(self):
        buf = RolloutBuffer(5, OBS_SHAPE, ACT_SHAPE, n_envs=1)
        obs = np.zeros((1, *OBS_SHAPE), dtype=np.float32)
        act = np.zeros((1, *ACT_SHAPE), dtype=np.float32)
        buf.add(obs, act, np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        buf.reset()
        assert len(buf) == 0

    def test_sample_shape(self):
        buf = RolloutBuffer(10, OBS_SHAPE, ACT_SHAPE, n_envs=2)
        obs = np.zeros((2, *OBS_SHAPE), dtype=np.float32)
        act = np.zeros((2, *ACT_SHAPE), dtype=np.float32)
        rew = np.ones(2, dtype=np.float32)
        val = np.zeros(2, dtype=np.float32)
        lp = np.zeros(2, dtype=np.float32)
        done = np.zeros(2, dtype=np.float32)
        for _ in range(10):
            buf.add(obs, act, rew, val, lp, done)
        buf.compute_returns_and_advantages(np.zeros(2), gamma=0.99, gae_lambda=0.95)
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, *OBS_SHAPE)
        assert "advantages" in batch
        assert "returns" in batch


# ============================================================
# MultiAgentBuffer
# ============================================================


class TestMultiAgentBuffer:
    def test_add_per_agent(self):
        buf = MultiAgentBuffer(3, 100, OBS_SHAPE, ACT_SHAPE)
        t = _make_transition()
        buf.add(0, t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        buf.add(1, t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf.buffers[0]) == 1
        assert len(buf.buffers[1]) == 1
        assert len(buf.buffers[2]) == 0
        assert len(buf) == 2

    def test_add_all(self):
        buf = MultiAgentBuffer(2, 100, OBS_SHAPE, ACT_SHAPE)
        obs = np.zeros((2, *OBS_SHAPE), dtype=np.float32)
        act = np.zeros((2, *ACT_SHAPE), dtype=np.float32)
        rew = np.array([1.0, 2.0])
        next_obs = np.zeros((2, *OBS_SHAPE), dtype=np.float32)
        dones = np.array([False, True])
        buf.add_all(obs, act, rew, next_obs, dones)
        assert len(buf) == 2

    def test_sample_single_agent(self):
        buf = MultiAgentBuffer(2, 100, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            buf.add(0, t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
            buf.add(1, t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(4, agent_id=0)
        assert isinstance(batch, dict)
        assert batch["obs"].shape == (4, *OBS_SHAPE)

    def test_sample_all_agents(self):
        buf = MultiAgentBuffer(3, 100, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            for a in range(3):
                buf.add(a, t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        result = buf.sample(4)
        assert isinstance(result, list)
        assert len(result) == 3


# ============================================================
# DemonstrationBuffer
# ============================================================


class TestDemonstrationBuffer:
    def test_add_and_len(self):
        buf = DemonstrationBuffer(100, OBS_SHAPE, ACT_SHAPE)
        assert len(buf) == 0
        t = _make_transition()
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        assert len(buf) == 1

    def test_full_raises(self):
        buf = DemonstrationBuffer(2, OBS_SHAPE, ACT_SHAPE)
        t = _make_transition()
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        with pytest.raises(BufferError):
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])

    def test_sample_demo_only(self):
        buf = DemonstrationBuffer(100, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, *OBS_SHAPE)

    def test_sample_mixed_with_online(self):
        demo_buf = DemonstrationBuffer(100, OBS_SHAPE, ACT_SHAPE)
        online_buf = ReplayBuffer(100, OBS_SHAPE, ACT_SHAPE)
        for i in range(10):
            t = _make_transition(i)
            demo_buf.add(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])
            t2 = _make_transition(i + 100)
            online_buf.add(t2["obs"], t2["action"], t2["reward"], t2["next_obs"], t2["done"])
        batch = demo_buf.sample(8, online_buffer=online_buf, demo_ratio=0.5)
        assert batch["obs"].shape == (8, *OBS_SHAPE)

    def test_load_demonstrations(self):
        buf = DemonstrationBuffer(100, OBS_SHAPE, ACT_SHAPE)
        n = 10
        data = {
            "obs": np.random.randn(n, *OBS_SHAPE).astype(np.float32),
            "actions": np.random.randn(n, *ACT_SHAPE).astype(np.float32),
            "rewards": np.random.randn(n).astype(np.float32),
            "next_obs": np.random.randn(n, *OBS_SHAPE).astype(np.float32),
            "dones": np.zeros(n, dtype=np.float32),
        }
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f, **data)
            path = f.name
        loaded = buf.load_demonstrations(path)
        assert loaded == n
        assert len(buf) == n
