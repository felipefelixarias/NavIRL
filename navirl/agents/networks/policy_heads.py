"""
Policy and Value Network Output Heads
======================================

Output heads that sit on top of shared feature extractors to produce
actions (for policy networks) or scalar estimates (for value / Q
networks) in reinforcement learning agents.

Classes
-------
GaussianPolicyHead
    Outputs mean and log_std for continuous Gaussian actions.
SquashedGaussianHead
    Gaussian with tanh squashing and log-prob correction (SAC style).
CategoricalPolicyHead
    Softmax over discrete actions.
MultiDiscreteHead
    Multiple independent categorical distributions.
DeterministicPolicyHead
    Direct deterministic action output with optional tanh.
ValueHead
    Single scalar state-value V(s).
QValueHead
    State-action value Q(s, a) via concatenation.
DuelingQHead
    Dueling architecture splitting value and advantage streams.
QuantileHead
    Quantile regression for distributional RL.
TwinQHead
    Two independent Q-networks for SAC / TD3 style algorithms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal

from navirl.agents.networks.mlp import MLP

# =====================================================================
# Gaussian Policy Head
# =====================================================================


class GaussianPolicyHead(nn.Module):
    """Output head for continuous Gaussian policies.

    Produces a mean vector and a log-standard-deviation vector that
    parameterise a diagonal Gaussian distribution over actions.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming feature vector.
    action_dim : int
        Dimensionality of the continuous action space.
    log_std_bounds : tuple of (float, float)
        Clamping range for the log-standard-deviation.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.log_std_min, self.log_std_max = log_std_bounds

        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        # Small uniform init for the output layers
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Compute mean and log_std.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        mean : Tensor ``(B, action_dim)``
        log_std : Tensor ``(B, action_dim)``
        """
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    # ------------------------------------------------------------------
    def sample(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Sample an action and compute its log-probability.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        action : Tensor ``(B, action_dim)``
        log_prob : Tensor ``(B,)``
        """
        mean, log_std = self.forward(features)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


# =====================================================================
# Squashed Gaussian Head (SAC)
# =====================================================================


class SquashedGaussianHead(nn.Module):
    """Gaussian policy with tanh squashing (SAC style).

    Like :class:`GaussianPolicyHead` but applies ``tanh`` to the sampled
    action and corrects the log-probability for the squashing bijector.

    Parameters
    ----------
    input_dim : int
    action_dim : int
    log_std_bounds : tuple of (float, float)
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.log_std_min, self.log_std_max = log_std_bounds

        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Compute mean and log_std (before squashing).

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        mean : Tensor ``(B, action_dim)``
        log_std : Tensor ``(B, action_dim)``
        """
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    # ------------------------------------------------------------------
    def sample(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Sample a squashed action and compute the corrected log-probability.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        action : Tensor ``(B, action_dim)`` in ``(-1, 1)``
        log_prob : Tensor ``(B,)``
        """
        mean, log_std = self.forward(features)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()  # pre-squash sample
        action = torch.tanh(x_t)

        # Log-prob correction for tanh squashing
        log_prob = dist.log_prob(x_t).sum(dim=-1)
        log_prob -= (2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))).sum(dim=-1)
        return action, log_prob


# =====================================================================
# Categorical Policy Head
# =====================================================================


class CategoricalPolicyHead(nn.Module):
    """Softmax policy head for discrete action spaces.

    Parameters
    ----------
    input_dim : int
    num_actions : int
        Number of discrete actions.
    """

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.logits_head = nn.Linear(input_dim, num_actions)

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Return raw logits.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        logits : Tensor ``(B, num_actions)``
        """
        return self.logits_head(features)

    # ------------------------------------------------------------------
    def sample(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action and compute log-probability and entropy.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        action : LongTensor ``(B,)``
        log_prob : Tensor ``(B,)``
        entropy : Tensor ``(B,)``
        """
        logits = self.forward(features)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


# =====================================================================
# Multi-Discrete Head
# =====================================================================


class MultiDiscreteHead(nn.Module):
    """Multiple independent categorical distributions.

    Useful for action spaces that have several discrete sub-actions
    (e.g., steering category + speed category).

    Parameters
    ----------
    input_dim : int
    nvec : sequence of int
        Number of possible actions for each sub-action dimension.
    """

    def __init__(self, input_dim: int, nvec: Sequence[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.nvec = list(nvec)
        self.heads = nn.ModuleList([nn.Linear(input_dim, n) for n in self.nvec])

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> list[Tensor]:
        """Return logits for each sub-action dimension.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        list of Tensor, each ``(B, n_i)``
        """
        return [head(features) for head in self.heads]

    # ------------------------------------------------------------------
    def sample(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample from each sub-action independently.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        actions : LongTensor ``(B, len(nvec))``
        log_prob : Tensor ``(B,)``  (sum across sub-actions)
        entropy : Tensor ``(B,)``  (sum across sub-actions)
        """
        logits_list = self.forward(features)
        actions, log_probs, entropies = [], [], []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            a = dist.sample()
            actions.append(a)
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
        return (
            torch.stack(actions, dim=-1),
            torch.stack(log_probs, dim=-1).sum(dim=-1),
            torch.stack(entropies, dim=-1).sum(dim=-1),
        )


# =====================================================================
# Deterministic Policy Head
# =====================================================================


class DeterministicPolicyHead(nn.Module):
    """Direct deterministic action output with optional tanh squashing.

    Parameters
    ----------
    input_dim : int
    action_dim : int
    use_tanh : bool
        If ``True``, apply ``tanh`` to bound actions to ``(-1, 1)``.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.use_tanh = use_tanh

        self.action_head = nn.Linear(input_dim, action_dim)
        nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Compute the deterministic action.

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        action : Tensor ``(B, action_dim)``
        """
        action = self.action_head(features)
        if self.use_tanh:
            action = torch.tanh(action)
        return action


# =====================================================================
# Value Head  V(s)
# =====================================================================


class ValueHead(nn.Module):
    """Single scalar state-value V(s).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming feature vector.
    hidden_dims : sequence of int
        Hidden-layer widths of the internal MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256,),
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        )

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Compute V(s).

        Parameters
        ----------
        features : Tensor of shape ``(B, input_dim)``

        Returns
        -------
        Tensor ``(B, 1)``
        """
        return self.net(features)


# =====================================================================
# Q-Value Head  Q(s, a)
# =====================================================================


class QValueHead(nn.Module):
    """State-action value Q(s, a) by concatenating state and action.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_dims : sequence of int
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="relu",
        )

    # ------------------------------------------------------------------
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Compute Q(s, a).

        Parameters
        ----------
        state : Tensor ``(B, state_dim)``
        action : Tensor ``(B, action_dim)``

        Returns
        -------
        Tensor ``(B, 1)``
        """
        return self.net(torch.cat([state, action], dim=-1))


# =====================================================================
# Dueling Q Head
# =====================================================================


class DuelingQHead(nn.Module):
    """Dueling Q architecture for discrete action spaces.

    Splits into value and advantage streams and combines them as
    ``Q(s,a) = V(s) + A(s,a) - mean(A)``.

    Parameters
    ----------
    input_dim : int
    num_actions : int
    """

    def __init__(self, input_dim: int, num_actions: int) -> None:
        super().__init__()
        self.num_actions = num_actions

        self.value_stream = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=(256,),
            activation="relu",
        )
        self.advantage_stream = MLP(
            input_dim=input_dim,
            output_dim=num_actions,
            hidden_dims=(256,),
            activation="relu",
        )

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Compute Q-values for all actions.

        Parameters
        ----------
        features : Tensor ``(B, input_dim)``

        Returns
        -------
        Tensor ``(B, num_actions)``
        """
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, num_actions)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# =====================================================================
# Quantile Head
# =====================================================================


class QuantileHead(nn.Module):
    """Quantile regression head for distributional RL (QR-DQN).

    Outputs *num_quantiles* Q-value estimates per action.

    Parameters
    ----------
    input_dim : int
    num_actions : int
    num_quantiles : int
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        num_quantiles: int = 51,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.net = MLP(
            input_dim=input_dim,
            output_dim=num_actions * num_quantiles,
            hidden_dims=(256, 256),
            activation="relu",
        )

    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Compute quantile values.

        Parameters
        ----------
        features : Tensor ``(B, input_dim)``

        Returns
        -------
        Tensor ``(B, num_actions, num_quantiles)``
        """
        return self.net(features).view(-1, self.num_actions, self.num_quantiles)


# =====================================================================
# Twin Q Head  (SAC / TD3)
# =====================================================================


class TwinQHead(nn.Module):
    """Two independent Q-networks for SAC / TD3 style algorithms.

    Both networks share the same architecture but have independent
    weights.  ``forward`` returns the outputs of both; a convenience
    ``min_q`` method returns the element-wise minimum.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    hidden_dims : sequence of int
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.q1 = QValueHead(state_dim, action_dim, hidden_dims)
        self.q2 = QValueHead(state_dim, action_dim, hidden_dims)

    # ------------------------------------------------------------------
    def forward(
        self, state: Tensor, action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute Q-values from both networks.

        Parameters
        ----------
        state : Tensor ``(B, state_dim)``
        action : Tensor ``(B, action_dim)``

        Returns
        -------
        q1 : Tensor ``(B, 1)``
        q2 : Tensor ``(B, 1)``
        """
        return self.q1(state, action), self.q2(state, action)

    # ------------------------------------------------------------------
    def min_q(self, state: Tensor, action: Tensor) -> Tensor:
        """Return the element-wise minimum of the two Q estimates.

        Parameters
        ----------
        state : Tensor ``(B, state_dim)``
        action : Tensor ``(B, action_dim)``

        Returns
        -------
        Tensor ``(B, 1)``
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
