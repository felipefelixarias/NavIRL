"""Tests for navirl/agents/ module: base agent, networks, agent creation."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import pathlib

import numpy as np
import pytest

from navirl.agents.base import (
    BaseAgent,
    CheckpointMeta,
    HyperParameters,
    MetricsLogger,
    RunningMeanStd,
)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_space():
    """Minimal mock observation space."""
    class _Space:
        shape = (8,)
        def __init__(self):
            self.dtype = np.float32
            self.low = np.full(8, -np.inf)
            self.high = np.full(8, np.inf)
        def sample(self):
            return np.random.randn(8).astype(np.float32)
    return _Space()


@pytest.fixture
def cont_action_space():
    """Continuous action space mock."""
    class _Space:
        shape = (2,)
        def __init__(self):
            self.dtype = np.float32
            self.low = np.array([-1.0, -1.0], dtype=np.float32)
            self.high = np.array([1.0, 1.0], dtype=np.float32)
            self.n = None
        def sample(self):
            return np.random.uniform(-1, 1, 2).astype(np.float32)
    return _Space()


@pytest.fixture
def disc_action_space():
    """Discrete action space mock."""
    class _Space:
        n = 5
        shape = ()
        def __init__(self):
            self.dtype = np.int64
        def sample(self):
            return np.random.randint(0, 5)
    return _Space()


# ---------------------------------------------------------------------------
# HyperParameters
# ---------------------------------------------------------------------------

class TestHyperParameters:
    def test_to_dict(self):
        from dataclasses import dataclass

        @dataclass
        class MyConfig(HyperParameters):
            lr: float = 1e-3
            hidden: int = 64

        cfg = MyConfig(lr=0.01, hidden=128)
        d = cfg.to_dict()
        assert d["lr"] == 0.01
        assert d["hidden"] == 128

    def test_from_dict(self):
        from dataclasses import dataclass

        @dataclass
        class MyConfig(HyperParameters):
            lr: float = 1e-3

        cfg = MyConfig.from_dict({"lr": 0.005, "unknown": True})
        assert cfg.lr == 0.005

    def test_getitem_setitem(self):
        from dataclasses import dataclass

        @dataclass
        class Cfg(HyperParameters):
            x: float = 1.0

        cfg = Cfg()
        assert cfg["x"] == 1.0
        cfg["x"] = 2.0
        assert cfg["x"] == 2.0

    def test_setitem_invalid_key(self):
        from dataclasses import dataclass

        @dataclass
        class Cfg(HyperParameters):
            x: float = 1.0

        cfg = Cfg()
        with pytest.raises(KeyError):
            cfg["nonexistent"] = 42

    def test_contains(self):
        from dataclasses import dataclass

        @dataclass
        class Cfg(HyperParameters):
            x: float = 1.0

        cfg = Cfg()
        assert "x" in cfg
        assert "z" not in cfg


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

class TestMetricsLogger:
    def test_record_and_dump(self):
        logged = {}
        def callback(metrics, step):
            logged.update(metrics)

        ml = MetricsLogger(callback=callback)
        ml.record("loss", 0.5)
        ml.record("loss", 0.3)
        summary = ml.dump(step=1)
        assert summary["loss"] == pytest.approx(0.4)
        assert "loss" in logged

    def test_step_property(self):
        ml = MetricsLogger()
        ml.step = 10
        assert ml.step == 10

    def test_record_dict(self):
        ml = MetricsLogger()
        ml.record_dict({"a": 1.0, "b": 2.0})
        summary = ml.dump()
        assert "a" in summary
        assert "b" in summary


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------

class TestRunningMeanStd:
    def test_update_single(self):
        rms = RunningMeanStd(shape=(2,))
        rms.update(np.array([1.0, 2.0]))
        assert rms.mean.shape == (2,)

    def test_update_batch(self):
        rms = RunningMeanStd(shape=(3,))
        batch = np.random.randn(100, 3)
        rms.update(batch)
        np.testing.assert_allclose(rms.mean, batch.mean(axis=0), atol=0.01)

    def test_normalize(self):
        rms = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rms.update(data)
        normalized = rms.normalize(np.array([3.0, 4.0]))
        assert normalized.shape == (2,)

    def test_denormalize_round_trip(self):
        rms = RunningMeanStd(shape=(2,))
        data = np.random.randn(50, 2) * 10 + 5
        rms.update(data)
        x = np.array([3.0, 7.0])
        normed = rms.normalize(x)
        recovered = rms.denormalize(normed)
        np.testing.assert_allclose(recovered, x, atol=1e-6)

    def test_state_dict_load(self):
        rms1 = RunningMeanStd(shape=(4,))
        rms1.update(np.random.randn(20, 4))
        state = rms1.state_dict()

        rms2 = RunningMeanStd(shape=(4,))
        rms2.load_state_dict(state)
        np.testing.assert_array_equal(rms1.mean, rms2.mean)
        np.testing.assert_array_equal(rms1.var, rms2.var)


# ---------------------------------------------------------------------------
# Concrete agent creation (PPO, SAC, DQN, A2C, TD3)
# ---------------------------------------------------------------------------

class TestPPOAgent:
    def test_create(self, obs_space, cont_action_space):
        from navirl.agents.ppo import PPOAgent, PPOConfig
        config = PPOConfig(lr=1e-3, hidden_dims=(32, 32), ppo_epochs=2)
        agent = PPOAgent(config, obs_space, cont_action_space, device="cpu")
        assert agent is not None

    def test_act(self, obs_space, cont_action_space):
        from navirl.agents.ppo import PPOAgent, PPOConfig
        config = PPOConfig(hidden_dims=(16, 16))
        agent = PPOAgent(config, obs_space, cont_action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert action.shape == (2,)
        assert isinstance(info, dict)

    def test_act_deterministic(self, obs_space, cont_action_space):
        from navirl.agents.ppo import PPOAgent, PPOConfig
        config = PPOConfig(hidden_dims=(16, 16))
        agent = PPOAgent(config, obs_space, cont_action_space, device="cpu", seed=42)
        obs = np.random.randn(8).astype(np.float32)
        a1, _ = agent.act(obs, deterministic=True)
        a2, _ = agent.act(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)


class TestSACAgent:
    def test_create(self, obs_space, cont_action_space):
        from navirl.agents.sac import SACAgent, SACConfig
        config = SACConfig(hidden_dims=(32, 32))
        agent = SACAgent(config, obs_space, cont_action_space, device="cpu")
        assert agent is not None

    def test_act(self, obs_space, cont_action_space):
        from navirl.agents.sac import SACAgent, SACConfig
        config = SACConfig(hidden_dims=(16, 16))
        agent = SACAgent(config, obs_space, cont_action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert action.shape == (2,)


class TestDQNAgent:
    def test_create(self, obs_space, disc_action_space):
        from navirl.agents.dqn import DQNAgent, DQNConfig
        config = DQNConfig(hidden_dims=(32, 32))
        agent = DQNAgent(config, obs_space, disc_action_space, device="cpu")
        assert agent is not None

    def test_act(self, obs_space, disc_action_space):
        from navirl.agents.dqn import DQNAgent, DQNConfig
        config = DQNConfig(hidden_dims=(16, 16))
        agent = DQNAgent(config, obs_space, disc_action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert isinstance(info, dict)


class TestA2CAgent:
    def test_create(self, obs_space, cont_action_space):
        from navirl.agents.a2c import A2CAgent, A2CConfig
        config = A2CConfig(hidden_dims=(32, 32))
        agent = A2CAgent(config, obs_space, cont_action_space, device="cpu")
        assert agent is not None

    def test_act(self, obs_space, cont_action_space):
        from navirl.agents.a2c import A2CAgent, A2CConfig
        config = A2CConfig(hidden_dims=(16, 16))
        agent = A2CAgent(config, obs_space, cont_action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert action.shape == (2,)


class TestTD3Agent:
    def test_create(self, obs_space, cont_action_space):
        from navirl.agents.td3 import TD3Agent, TD3Config
        config = TD3Config(hidden_dims=(32, 32))
        agent = TD3Agent(config, obs_space, cont_action_space, device="cpu")
        assert agent is not None

    def test_act(self, obs_space, cont_action_space):
        from navirl.agents.td3 import TD3Agent, TD3Config
        config = TD3Config(hidden_dims=(16, 16))
        agent = TD3Agent(config, obs_space, cont_action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert action.shape == (2,)


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

class TestMLP:
    def test_forward(self):
        from navirl.agents.networks.mlp import MLP
        net = MLP(input_dim=8, hidden_dims=[32, 32], output_dim=4)
        x = torch.randn(1, 8)
        out = net(x)
        assert out.shape == (1, 4)

    def test_different_activations(self):
        from navirl.agents.networks.mlp import MLP
        for act in ["relu", "tanh"]:
            net = MLP(input_dim=4, hidden_dims=[16], output_dim=2, activation=act)
            out = net(torch.randn(1, 4))
            assert out.shape == (1, 2)


class TestCNN:
    def test_forward(self):
        # TODO: CNNExtractor doesn't exist, using NatureDQN as placeholder
        from navirl.agents.networks.cnn import NatureDQN
        net = NatureDQN(input_channels=3, input_height=84, input_width=84, output_dim=64)
        # Assume input is (batch, C, H, W)
        x = torch.randn(1, 3, 84, 84)
        out = net(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 64


class TestRNN:
    def test_forward(self):
        # TODO: RNNEncoder doesn't exist, using SequenceEncoder as replacement
        from navirl.agents.networks.rnn import SequenceEncoder
        net = SequenceEncoder(input_dim=8, hidden_size=16, num_layers=1)
        # (batch, seq_len, input_dim)
        x = torch.randn(2, 5, 8)
        out = net(x)
        assert out.shape == (2, 16)  # hidden_size=16, bidirectional=False


class TestAttention:
    def test_forward(self):
        from navirl.agents.networks.attention import SocialAttention
        net = SocialAttention(embed_dim=16, num_heads=2)
        # (batch, num_agents, embed_dim)
        x = torch.randn(2, 6, 16)
        out = net(x)
        assert out.shape[0] == 2


class TestPolicyHeads:
    def test_gaussian_head(self):
        from navirl.agents.networks.policy_heads import GaussianPolicyHead
        head = GaussianPolicyHead(input_dim=32, action_dim=2)
        x = torch.randn(1, 32)
        dist = head(x)
        sample = dist.sample()
        assert sample.shape == (1, 2)

    def test_categorical_head(self):
        from navirl.agents.networks.policy_heads import CategoricalPolicyHead
        head = CategoricalPolicyHead(input_dim=32, n_actions=5)
        x = torch.randn(1, 32)
        dist = head(x)
        sample = dist.sample()
        assert sample.shape == (1,)

    def test_value_head(self):
        from navirl.agents.networks.policy_heads import ValueHead
        head = ValueHead(input_dim=32)
        x = torch.randn(4, 32)
        v = head(x)
        assert v.shape == (4, 1)


# ---------------------------------------------------------------------------
# BaseAgent mode toggles
# ---------------------------------------------------------------------------

class TestBaseAgentModes:
    def _make_agent(self, obs_space, cont_action_space):
        from navirl.agents.ppo import PPOAgent, PPOConfig
        config = PPOConfig(hidden_dims=(16,))
        return PPOAgent(config, obs_space, cont_action_space, device="cpu")

    def test_train_eval_toggle(self, obs_space, cont_action_space):
        agent = self._make_agent(obs_space, cont_action_space)
        assert agent.is_training is True
        agent.eval_mode()
        assert agent.is_training is False
        agent.train_mode()
        assert agent.is_training is True

    def test_total_steps_property(self, obs_space, cont_action_space):
        agent = self._make_agent(obs_space, cont_action_space)
        assert agent.total_steps == 0

    def test_repr(self, obs_space, cont_action_space):
        agent = self._make_agent(obs_space, cont_action_space)
        r = repr(agent)
        assert "PPOAgent" in r

    def test_registered_agents(self):
        names = BaseAgent.registered_agents()
        assert isinstance(names, list)
        assert len(names) > 0


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_registry_contains_agents(self):
        registered = BaseAgent.registered_agents()
        # At minimum the agents we imported should be registered
        assert any("PPO" in name for name in registered)

    def test_make_unknown_agent(self, obs_space, cont_action_space):
        from navirl.agents.ppo import PPOConfig
        with pytest.raises(ValueError, match="Unknown agent"):
            BaseAgent.make("NonexistentAgent", PPOConfig(), obs_space, cont_action_space)
