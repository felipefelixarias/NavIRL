"""Tests for navirl/imitation/ module: IRL, GAIL, AIRL, dataset."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def expert_data():
    """Synthetic expert demonstrations: 100 transitions."""
    rng = np.random.default_rng(42)
    obs = rng.standard_normal((100, 8)).astype(np.float32)
    actions = rng.standard_normal((100, 2)).astype(np.float32)
    next_obs = obs + 0.1 * actions[:, :2].repeat(4, axis=1)
    rewards = rng.standard_normal(100).astype(np.float32)
    dones = np.zeros(100, dtype=np.float32)
    dones[::20] = 1.0
    return {
        "observations": obs,
        "actions": actions,
        "next_observations": next_obs,
        "rewards": rewards,
        "dones": dones,
    }


@pytest.fixture
def obs_space():
    class _Space:
        shape = (8,)

        def __init__(self):
            self.dtype = np.float32
            self.low = np.full(8, -np.inf)
            self.high = np.full(8, np.inf)

    return _Space()


@pytest.fixture
def action_space():
    class _Space:
        shape = (2,)

        def __init__(self):
            self.dtype = np.float32
            self.low = np.array([-1, -1], dtype=np.float32)
            self.high = np.array([1, 1], dtype=np.float32)
            self.n = None

    return _Space()


# ---------------------------------------------------------------------------
# MaxEntIRL
# ---------------------------------------------------------------------------


class TestMaxEntIRL:
    def test_config_creation(self):
        from navirl.imitation.irl import MaxEntIRLConfig

        cfg = MaxEntIRLConfig(lr=0.01, feature_dim=32, num_iterations=50)
        assert cfg.lr == 0.01
        assert cfg.feature_dim == 32

    def test_config_to_dict(self):
        from navirl.imitation.irl import MaxEntIRLConfig

        cfg = MaxEntIRLConfig()
        d = cfg.to_dict()
        assert "lr" in d
        assert "feature_dim" in d

    def test_reward_initialization(self):
        from navirl.imitation.irl import MaxEntIRL, MaxEntIRLConfig

        cfg = MaxEntIRLConfig(feature_dim=16)
        irl = MaxEntIRL(cfg)
        assert irl.theta.shape == (16,)

    def test_compute_reward(self):
        from navirl.imitation.irl import MaxEntIRL, MaxEntIRLConfig

        cfg = MaxEntIRLConfig(feature_dim=8)
        irl = MaxEntIRL(cfg)
        features = np.random.randn(8).astype(np.float64)
        reward = irl.compute_reward(features)
        assert isinstance(reward, float)

    def test_update_step(self):
        from navirl.imitation.irl import MaxEntIRL, MaxEntIRLConfig

        cfg = MaxEntIRLConfig(feature_dim=8, lr=0.1)
        irl = MaxEntIRL(cfg)
        expert_features = np.ones(8)
        policy_features = np.zeros(8)
        theta_before = irl.theta.copy()
        irl.update(expert_features, policy_features)
        # Theta should change
        assert not np.allclose(irl.theta, theta_before)


# ---------------------------------------------------------------------------
# GAIL
# ---------------------------------------------------------------------------


class TestGAIL:
    def test_config_creation(self):
        from navirl.imitation.gail import GAILConfig

        cfg = GAILConfig(lr_discriminator=1e-4, disc_hidden_dims=(64, 64))
        assert cfg.lr_discriminator == 1e-4

    def test_discriminator_forward(self):
        from navirl.imitation.gail import Discriminator

        disc = Discriminator(state_dim=8, action_dim=2, hidden_dims=(32, 32))
        state = torch.randn(4, 8)
        action = torch.randn(4, 2)
        output = disc(state, action)
        assert output.shape == (4, 1)
        # Output should be in (0, 1) due to sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_discriminator_gradient(self):
        from navirl.imitation.gail import Discriminator

        disc = Discriminator(state_dim=8, action_dim=2, hidden_dims=(16,))
        state = torch.randn(2, 8)
        action = torch.randn(2, 2)
        output = disc(state, action)
        loss = output.mean()
        loss.backward()
        for p in disc.parameters():
            assert p.grad is not None

    def test_gail_agent_creation(self, obs_space, action_space):
        from navirl.imitation.gail import GAILAgent, GAILConfig

        cfg = GAILConfig(disc_hidden_dims=(16,), policy_hidden_dims=(16,))
        agent = GAILAgent(cfg, obs_space, action_space, device="cpu")
        assert agent is not None

    def test_gail_agent_act(self, obs_space, action_space):
        from navirl.imitation.gail import GAILAgent, GAILConfig

        cfg = GAILConfig(disc_hidden_dims=(16,), policy_hidden_dims=(16,))
        agent = GAILAgent(cfg, obs_space, action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert action.shape == (2,)


# ---------------------------------------------------------------------------
# AIRL
# ---------------------------------------------------------------------------


class TestAIRL:
    def test_config_creation(self):
        from navirl.imitation.airl import AIRLConfig

        cfg = AIRLConfig(gamma=0.99, state_only=True)
        assert cfg.gamma == 0.99
        assert cfg.state_only is True

    def test_reward_network_forward(self):
        from navirl.imitation.airl import RewardNetwork

        net = RewardNetwork(state_dim=8, action_dim=2, hidden_dims=(32,), gamma=0.99)
        state = torch.randn(4, 8)
        action = torch.randn(4, 2)
        next_state = torch.randn(4, 8)
        reward, shaping = net(state, action, next_state)
        assert reward.shape == (4, 1)
        assert shaping.shape == (4, 1)

    def test_reward_network_state_only(self):
        from navirl.imitation.airl import RewardNetwork

        net = RewardNetwork(
            state_dim=8,
            action_dim=2,
            hidden_dims=(16,),
            gamma=0.99,
            state_only=True,
        )
        state = torch.randn(2, 8)
        action = torch.randn(2, 2)
        next_state = torch.randn(2, 8)
        reward, shaping = net(state, action, next_state)
        assert reward.shape == (2, 1)

    def test_airl_agent_creation(self, obs_space, action_space):
        from navirl.imitation.airl import AIRLAgent, AIRLConfig

        cfg = AIRLConfig(disc_hidden_dims=(16,), policy_hidden_dims=(16,))
        agent = AIRLAgent(cfg, obs_space, action_space, device="cpu")
        assert agent is not None

    def test_airl_agent_act(self, obs_space, action_space):
        from navirl.imitation.airl import AIRLAgent, AIRLConfig

        cfg = AIRLConfig(disc_hidden_dims=(16,), policy_hidden_dims=(16,))
        agent = AIRLAgent(cfg, obs_space, action_space, device="cpu")
        obs = np.random.randn(8).astype(np.float32)
        action, info = agent.act(obs)
        assert action.shape == (2,)


# ---------------------------------------------------------------------------
# DemonstrationDataset
# ---------------------------------------------------------------------------


class TestDemonstrationDataset:
    def test_from_arrays(self, expert_data):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=expert_data["observations"],
            actions=expert_data["actions"],
        )
        assert len(ds) == 100

    def test_getitem(self, expert_data):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=expert_data["observations"],
            actions=expert_data["actions"],
        )
        item = ds[0]
        assert "observation" in item or "obs" in item or isinstance(item, tuple)

    def test_save_load_npz(self, expert_data, tmp_path):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=expert_data["observations"],
            actions=expert_data["actions"],
        )
        path = tmp_path / "demo.npz"
        ds.save(str(path))
        assert path.exists()

        ds2 = DemonstrationDataset.load(str(path))
        assert len(ds2) == len(ds)

    def test_normalize(self, expert_data):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=expert_data["observations"],
            actions=expert_data["actions"],
        )
        stats = ds.compute_statistics()
        assert stats.obs_mean.shape == (8,)
        assert stats.obs_std.shape == (8,)

    def test_split(self, expert_data):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=expert_data["observations"],
            actions=expert_data["actions"],
        )
        train, val = ds.split(train_ratio=0.8, seed=42)
        assert len(train) + len(val) == len(ds)
        assert len(train) > len(val)


# ---------------------------------------------------------------------------
# Dataset edge cases
# ---------------------------------------------------------------------------


class TestDatasetEdgeCases:
    def test_empty_dataset(self):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=np.empty((0, 4), dtype=np.float32),
            actions=np.empty((0, 2), dtype=np.float32),
        )
        assert len(ds) == 0

    def test_single_transition(self):
        from navirl.imitation.dataset import DemonstrationDataset

        ds = DemonstrationDataset(
            observations=np.ones((1, 4), dtype=np.float32),
            actions=np.ones((1, 2), dtype=np.float32),
        )
        assert len(ds) == 1

    def test_feature_statistics(self, expert_data):
        from navirl.imitation.dataset import FeatureStatistics

        stats = FeatureStatistics(
            obs_mean=np.zeros(8),
            obs_std=np.ones(8),
            action_mean=np.zeros(2),
            action_std=np.ones(2),
        )
        assert stats.obs_mean.shape == (8,)


# ---------------------------------------------------------------------------
# Discriminator training step
# ---------------------------------------------------------------------------


class TestDiscriminatorTraining:
    def test_single_update(self, expert_data):
        from navirl.imitation.gail import Discriminator

        disc = Discriminator(state_dim=8, action_dim=2, hidden_dims=(32,))
        optimizer = torch.optim.Adam(disc.parameters(), lr=1e-3)

        expert_s = torch.from_numpy(expert_data["observations"][:16])
        expert_a = torch.from_numpy(expert_data["actions"][:16])
        policy_s = torch.randn(16, 8)
        policy_a = torch.randn(16, 2)

        expert_out = disc(expert_s, expert_a)
        policy_out = disc(policy_s, policy_a)

        loss = -(torch.log(expert_out + 1e-8).mean() + torch.log(1 - policy_out + 1e-8).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0  # loss is finite

    def test_discriminator_different_sizes(self):
        from navirl.imitation.gail import Discriminator

        for hdims in [(16,), (32, 32), (64, 32, 16)]:
            disc = Discriminator(state_dim=4, action_dim=2, hidden_dims=hdims)
            out = disc(torch.randn(1, 4), torch.randn(1, 2))
            assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# Reward learning integration
# ---------------------------------------------------------------------------


class TestRewardLearningIntegration:
    def test_irl_reward_matches_expert(self):
        """After some updates, IRL reward should assign higher values to expert features."""
        from navirl.imitation.irl import MaxEntIRL, MaxEntIRLConfig

        cfg = MaxEntIRLConfig(feature_dim=4, lr=0.5)
        irl = MaxEntIRL(cfg)

        expert_features = np.array([1.0, 1.0, 1.0, 1.0])
        policy_features = np.array([0.0, 0.0, 0.0, 0.0])

        for _ in range(50):
            irl.update(expert_features, policy_features)

        r_expert = irl.compute_reward(expert_features)
        r_policy = irl.compute_reward(policy_features)
        assert r_expert > r_policy
