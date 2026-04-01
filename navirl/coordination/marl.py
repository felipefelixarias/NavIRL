"""Multi-agent reinforcement learning algorithms.

Provides centralized-training-decentralized-execution (CTDE) components
including a centralized critic, QMIX mixing network, and Multi-Agent PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MARLConfig:
    """Configuration for multi-agent RL training.

    Attributes:
        algorithm: Name of the MARL algorithm (e.g. ``"mappo"``, ``"qmix"``).
        centralized_critic: Whether to use a centralized value function.
        shared_policy: Whether all agents share a single policy network.
        communication: Whether to enable learned inter-agent communication.
        gamma: Discount factor.
        lr: Learning rate.
        num_agents: Number of agents.
    """

    algorithm: str = "mappo"
    centralized_critic: bool = True
    shared_policy: bool = True
    communication: bool = False
    gamma: float = 0.99
    lr: float = 3e-4
    num_agents: int = 2


# ---------------------------------------------------------------------------
# Neural modules (require PyTorch)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class CentralizedCritic(nn.Module):
        """Centralized critic for CTDE multi-agent training.

        The critic receives the *joint* observation (concatenation of all
        agent observations) and outputs a single state-value estimate.
        During execution, only the decentralized actors are needed.

        Parameters:
            obs_dim: Observation dimensionality for a single agent.
            num_agents: Number of agents.
            hidden_dim: Hidden-layer width.
        """

        def __init__(
            self,
            obs_dim: int,
            num_agents: int,
            hidden_dim: int = 256,
        ) -> None:
            super().__init__()
            self.obs_dim = obs_dim
            self.num_agents = num_agents
            joint_dim = obs_dim * num_agents

            self.network = nn.Sequential(
                nn.Linear(joint_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
            """Compute state value from joint observations.

            Parameters:
                joint_obs: Tensor of shape ``(batch, num_agents * obs_dim)``
                    or ``(batch, num_agents, obs_dim)``.

            Returns:
                State-value estimate of shape ``(batch, 1)``.
            """
            if joint_obs.dim() == 3:
                joint_obs = joint_obs.view(joint_obs.size(0), -1)
            return self.network(joint_obs)

    class QMIXMixer(nn.Module):
        """QMIX monotonic mixing network.

        Combines per-agent Q-values into a joint Q-value using a mixing
        network whose weights are constrained to be non-negative
        (monotonicity).  A hyper-network generates the mixing weights
        conditioned on the global state.

        Parameters:
            num_agents: Number of agents.
            state_dim: Dimensionality of the global state.
            mixing_embed_dim: Width of the mixing network hidden layer.
        """

        def __init__(
            self,
            num_agents: int,
            state_dim: int,
            mixing_embed_dim: int = 32,
        ) -> None:
            super().__init__()
            self.num_agents = num_agents
            self.state_dim = state_dim
            self.mixing_embed_dim = mixing_embed_dim

            # Hyper-networks for mixing weights (layer 1)
            self.hyper_w1 = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim),
            )
            self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

            # Hyper-networks for mixing weights (layer 2)
            self.hyper_w2 = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(mixing_embed_dim, mixing_embed_dim),
            )
            self.hyper_b2 = nn.Sequential(
                nn.Linear(state_dim, mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(mixing_embed_dim, 1),
            )

        def forward(
            self,
            agent_qs: torch.Tensor,
            state: torch.Tensor,
        ) -> torch.Tensor:
            """Mix individual Q-values into a joint Q-value.

            Parameters:
                agent_qs: Per-agent Q-values of shape ``(batch, num_agents)``.
                state: Global state of shape ``(batch, state_dim)``.

            Returns:
                Joint Q-value of shape ``(batch, 1)``.
            """
            batch_size = agent_qs.size(0)
            agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

            # First layer (non-negative weights via abs)
            w1 = torch.abs(
                self.hyper_w1(state).view(batch_size, self.num_agents, self.mixing_embed_dim)
            )
            b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_embed_dim)
            hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

            # Second layer
            w2 = torch.abs(
                self.hyper_w2(state).view(batch_size, self.mixing_embed_dim, 1)
            )
            b2 = self.hyper_b2(state).view(batch_size, 1, 1)
            q_total = torch.bmm(hidden, w2) + b2

            return q_total.squeeze(-1).squeeze(-1).unsqueeze(-1)  # (batch, 1)

    class MAPPOAgent(nn.Module):
        """Multi-Agent Proximal Policy Optimization agent.

        Supports both shared and separate policies per agent.  The value
        function is centralized (takes joint state), while actors receive
        only agent-specific observations.

        Parameters:
            obs_dim: Single-agent observation dimension.
            action_dim: Action dimension (discrete or continuous).
            num_agents: Number of agents.
            hidden_dim: Hidden-layer width.
            shared_policy: If ``True``, all agents share one actor network.
            continuous: If ``True``, output Gaussian policy; else categorical.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            num_agents: int,
            hidden_dim: int = 128,
            shared_policy: bool = True,
            continuous: bool = False,
        ) -> None:
            super().__init__()
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.num_agents = num_agents
            self.shared_policy = shared_policy
            self.continuous = continuous

            # Actor(s)
            if shared_policy:
                self.actor = self._build_actor(obs_dim, action_dim, hidden_dim)
            else:
                self.actors = nn.ModuleList(
                    [
                        self._build_actor(obs_dim, action_dim, hidden_dim)
                        for _ in range(num_agents)
                    ]
                )

            # Centralized critic (takes joint observation)
            self.critic = CentralizedCritic(obs_dim, num_agents, hidden_dim)

            # Log-std for continuous actions
            if continuous:
                self.log_std = nn.Parameter(torch.zeros(action_dim))

        # -- building blocks ------------------------------------------------

        @staticmethod
        def _build_actor(obs_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        # -- forward --------------------------------------------------------

        def forward(
            self,
            obs: torch.Tensor,
            agent_index: int = 0,
        ) -> torch.Tensor:
            """Compute action logits (discrete) or mean (continuous) for one agent.

            Parameters:
                obs: Agent-specific observation of shape ``(batch, obs_dim)``.
                agent_index: Index of the agent (used when policies are not
                    shared).

            Returns:
                Action logits or mean of shape ``(batch, action_dim)``.
            """
            if self.shared_policy:
                return self.actor(obs)
            return self.actors[agent_index](obs)

        def get_value(self, joint_obs: torch.Tensor) -> torch.Tensor:
            """Compute centralized state value.

            Parameters:
                joint_obs: Joint observations of shape
                    ``(batch, num_agents * obs_dim)`` or
                    ``(batch, num_agents, obs_dim)``.

            Returns:
                Value estimate of shape ``(batch, 1)``.
            """
            return self.critic(joint_obs)

        def get_action_and_value(
            self,
            obs: torch.Tensor,
            joint_obs: torch.Tensor,
            agent_index: int = 0,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return sampled action, log-probability, and state value.

            Parameters:
                obs: Agent-specific observation ``(batch, obs_dim)``.
                joint_obs: Joint observations for the critic.
                agent_index: Agent index for separate policies.

            Returns:
                Tuple of ``(action, log_prob, value)`` tensors.
            """
            logits = self.forward(obs, agent_index)
            value = self.get_value(joint_obs)

            if self.continuous:
                std = self.log_std.exp()
                dist = torch.distributions.Normal(logits, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(-1)

            return action, log_prob, value

else:  # pragma: no cover — torch not available

    class CentralizedCritic:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("CentralizedCritic requires PyTorch.")

    class QMIXMixer:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("QMIXMixer requires PyTorch.")

    class MAPPOAgent:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("MAPPOAgent requires PyTorch.")
