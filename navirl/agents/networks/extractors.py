"""
Feature Extractors for Multi-Modal Observation Fusion
=====================================================

Observation-specific feature extractors that wrap lower-level network
building blocks (MLP, CNN, attention) to process different sensor
modalities commonly found in pedestrian navigation environments.

Classes
-------
RunningNormalizer
    Maintains running mean / std and normalizes inputs online.
StateExtractor
    Extracts features from the robot state vector (position, velocity, goal).
LidarExtractor
    Processes LiDAR range observations with a 1-D convolutional backbone.
OccupancyExtractor
    Processes 2-D occupancy grids with a convolutional backbone.
SocialExtractor
    Processes nearby-agent state sets via multi-head attention.
CombinedExtractor
    Concatenates outputs from multiple named extractors with an optional
    fusion MLP.
HierarchicalExtractor
    Two-level extraction pipeline: local extractor -> global extractor.
RecurrentExtractor
    Wraps any base extractor with an LSTM or GRU for temporal reasoning.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from navirl.agents.networks.mlp import MLP

# =====================================================================
# Running Normalizer
# =====================================================================


class RunningNormalizer(nn.Module):
    """Online running-mean / running-std normalizer.

    Tracks first and second moments using Welford's algorithm and
    normalizes inputs to zero mean and unit variance.  Statistics are
    stored as buffers so they travel with the model but are *not*
    learnable parameters.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    clip : float
        Clamp normalized values to ``[-clip, clip]``.
    epsilon : float
        Small constant for numerical stability in the denominator.
    """

    def __init__(
        self,
        input_dim: int,
        clip: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.clip = clip
        self.epsilon = epsilon

        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("var", torch.ones(input_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float64))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update running statistics with a new batch of observations.

        Parameters
        ----------
        x : Tensor of shape ``(B, input_dim)``
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count.clamp(min=1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count.clamp(min=1)
        new_var = m2 / total_count.clamp(min=1)

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    # ------------------------------------------------------------------
    def normalize(self, x: Tensor) -> Tensor:
        """Normalize ``x`` using the current running statistics.

        Parameters
        ----------
        x : Tensor of shape ``(*, input_dim)``

        Returns
        -------
        Tensor of the same shape, normalized and clipped.
        """
        std = (self.var + self.epsilon).sqrt()
        return ((x - self.mean) / std).clamp(-self.clip, self.clip)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Update (in training mode) and normalize.

        Parameters
        ----------
        x : Tensor of shape ``(*, input_dim)``

        Returns
        -------
        Tensor – normalized ``x``.
        """
        if self.training:
            self.update(x.reshape(-1, self.input_dim))
        return self.normalize(x)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, clip={self.clip}"


# =====================================================================
# State Extractor
# =====================================================================


class StateExtractor(nn.Module):
    """Extract features from a robot state vector (position, velocity, goal).

    Applies optional running normalisation followed by a small MLP.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the raw state vector.
    output_dim : int
        Dimensionality of the extracted feature vector.
    hidden_dims : sequence of int
        Hidden-layer widths of the internal MLP.
    normalize : bool
        If ``True``, inputs are normalised with a :class:`RunningNormalizer`.
    activation : str
        Activation function name forwarded to :class:`MLP`.
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int = 64,
        hidden_dims: Sequence[int] = (128,),
        normalize: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self._feature_dim = output_dim

        self.normalizer: RunningNormalizer | None = (
            RunningNormalizer(state_dim) if normalize else None
        )
        self.mlp = MLP(
            input_dim=state_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, state: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        state : Tensor of shape ``(*, state_dim)``

        Returns
        -------
        Tensor of shape ``(*, output_dim)``
        """
        if self.normalizer is not None:
            state = self.normalizer(state)
        return self.mlp(state)


# =====================================================================
# LiDAR Extractor
# =====================================================================


class LidarExtractor(nn.Module):
    """Process LiDAR range observations with a 1-D CNN backbone.

    The raw beam array is reshaped to ``(B, 1, num_beams)`` and passed
    through three 1-D convolution layers with increasing channel counts,
    followed by adaptive pooling and a linear projection.

    Parameters
    ----------
    num_beams : int
        Number of LiDAR beams (length of the input vector).
    output_dim : int
        Dimensionality of the output feature vector.
    """

    def __init__(
        self,
        num_beams: int = 360,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_beams = num_beams
        self._feature_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, output_dim)

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, lidar: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        lidar : Tensor of shape ``(B, num_beams)`` or ``(B, 1, num_beams)``

        Returns
        -------
        Tensor of shape ``(B, output_dim)``
        """
        if lidar.dim() == 2:
            lidar = lidar.unsqueeze(1)  # (B, 1, num_beams)
        h = self.conv(lidar)  # (B, 128, 1)
        h = h.squeeze(-1)  # (B, 128)
        return F.relu(self.fc(h))


# =====================================================================
# Occupancy Grid Extractor
# =====================================================================


class OccupancyExtractor(nn.Module):
    """Process 2-D occupancy grids with a convolutional backbone.

    Applies three 2-D convolution layers with increasing channel counts,
    followed by adaptive pooling and a linear projection.

    Parameters
    ----------
    grid_size : int
        Spatial side length of the square occupancy grid.
    channels : int
        Number of input channels (e.g. 1 for a single binary grid).
    output_dim : int
        Dimensionality of the output feature vector.
    """

    def __init__(
        self,
        grid_size: int = 84,
        channels: int = 1,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.channels = channels
        self._feature_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, output_dim)

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, grid: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        grid : Tensor of shape ``(B, channels, H, W)`` or ``(B, H, W)``

        Returns
        -------
        Tensor of shape ``(B, output_dim)``
        """
        if grid.dim() == 3:
            grid = grid.unsqueeze(1)  # (B, 1, H, W)
        h = self.conv(grid)  # (B, 64, 1, 1)
        h = h.flatten(1)  # (B, 64)
        return F.relu(self.fc(h))


# =====================================================================
# Social Extractor
# =====================================================================


class SocialExtractor(nn.Module):
    """Process nearby-agent states via multi-head attention.

    Each neighbouring agent is represented as a fixed-size state vector.
    The set of neighbours is passed through a linear embedding, then
    multi-head self-attention produces a permutation-invariant summary
    via mean-pooling over the attended sequence.

    Parameters
    ----------
    agent_state_dim : int
        Dimensionality of each neighbouring agent's state vector.
    output_dim : int
        Dimensionality of the output feature vector.
    num_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        agent_state_dim: int,
        output_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.agent_state_dim = agent_state_dim
        self._feature_dim = output_dim
        self.num_heads = num_heads

        # Ensure embed_dim is divisible by num_heads
        self.embed_dim = output_dim
        self.embedding = nn.Linear(agent_state_dim, self.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, output_dim)

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        agents: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        agents : Tensor of shape ``(B, N, agent_state_dim)``
            States of *N* nearby agents.
        mask : Tensor of shape ``(B, N)``, optional
            Boolean mask where ``True`` indicates a padded (invalid) agent.

        Returns
        -------
        Tensor of shape ``(B, output_dim)``
        """
        # Embed each agent
        h = F.relu(self.embedding(agents))  # (B, N, embed_dim)

        # Self-attention
        attn_out, _ = self.attention(
            h, h, h, key_padding_mask=mask,
        )  # (B, N, embed_dim)
        h = self.layer_norm(h + attn_out)

        # Aggregate: mean-pool over agents (mask out padded agents)
        if mask is not None:
            # mask: True = invalid -> zero those out before mean
            valid = (~mask).unsqueeze(-1).float()  # (B, N, 1)
            h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)  # (B, embed_dim)

        return self.output_proj(h)


# =====================================================================
# Combined Extractor
# =====================================================================


class CombinedExtractor(nn.Module):
    """Combine multiple named extractors by concatenating their outputs.

    An optional MLP fusion head is applied on top of the concatenation.

    Parameters
    ----------
    extractors : dict[str, nn.Module]
        Mapping from observation key to extractor module.  Each extractor
        must expose a ``feature_dim`` property.
    output_dim : int or None
        If given, a linear projection (with ReLU) reduces the
        concatenated features to ``output_dim``.  Otherwise the
        concatenation is returned directly.
    """

    def __init__(
        self,
        extractors: dict[str, nn.Module],
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.extractors = nn.ModuleDict(extractors)

        concat_dim = sum(ext.feature_dim for ext in self.extractors.values())

        if output_dim is not None:
            self.fusion = MLP(
                input_dim=concat_dim,
                output_dim=output_dim,
                hidden_dims=(concat_dim,),
                activation="relu",
            )
            self._feature_dim = output_dim
        else:
            self.fusion = None
            self._feature_dim = concat_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        observations : dict[str, Tensor]
            Mapping from observation key to tensor.  Keys must match those
            provided at construction time.

        Returns
        -------
        Tensor of shape ``(B, feature_dim)``
        """
        features = [self.extractors[key](observations[key]) for key in self.extractors]
        combined = torch.cat(features, dim=-1)
        if self.fusion is not None:
            combined = self.fusion(combined)
        return combined


# =====================================================================
# Hierarchical Extractor
# =====================================================================


class HierarchicalExtractor(nn.Module):
    """Two-level extraction: local extractor then global extractor.

    The local extractor processes raw observations into an intermediate
    representation, which the global extractor refines into the final
    feature vector.

    Parameters
    ----------
    local_extractor : nn.Module
        First-stage extractor.  Must expose ``feature_dim``.
    global_extractor : nn.Module
        Second-stage extractor.  Must expose ``feature_dim``.
    output_dim : int or None
        If given, an additional linear projection is appended.
    """

    def __init__(
        self,
        local_extractor: nn.Module,
        global_extractor: nn.Module,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.local_extractor = local_extractor
        self.global_extractor = global_extractor

        if output_dim is not None:
            self.output_proj = nn.Linear(global_extractor.feature_dim, output_dim)
            self._feature_dim = output_dim
        else:
            self.output_proj = None
            self._feature_dim = global_extractor.feature_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Raw observation passed to the local extractor.

        Returns
        -------
        Tensor of shape ``(B, feature_dim)``
        """
        local_features = self.local_extractor(x)
        global_features = self.global_extractor(local_features)
        if self.output_proj is not None:
            global_features = F.relu(self.output_proj(global_features))
        return global_features


# =====================================================================
# Recurrent Extractor
# =====================================================================


class RecurrentExtractor(nn.Module):
    """Wrap any base extractor with an LSTM or GRU for temporal reasoning.

    At each time step the base extractor produces a feature vector which
    is fed into the recurrent core.  The recurrent hidden state is
    returned alongside the output so callers can manage it across
    episode boundaries.

    Parameters
    ----------
    base_extractor : nn.Module
        Any feature extractor with a ``feature_dim`` property.
    rnn_type : str
        ``"lstm"`` or ``"gru"``.
    hidden_size : int
        Number of hidden units in the recurrent layer.
    num_layers : int
        Number of stacked recurrent layers.
    """

    def __init__(
        self,
        base_extractor: nn.Module,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.base_extractor = base_extractor
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._feature_dim = hidden_size

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}
        if self.rnn_type not in rnn_cls:
            raise ValueError(
                f"Unknown rnn_type '{rnn_type}'. Choose from {list(rnn_cls.keys())}"
            )

        self.rnn = rnn_cls[self.rnn_type](
            input_size=base_extractor.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def initial_state(self, batch_size: int, device: torch.device | None = None) -> tuple[Tensor, Tensor] | Tensor:
        """Return a zero-initialised hidden state.

        Parameters
        ----------
        batch_size : int
        device : torch.device, optional

        Returns
        -------
        ``(h_0, c_0)`` for LSTM or ``h_0`` for GRU, each of shape
        ``(num_layers, batch_size, hidden_size)``.
        """
        if device is None:
            device = next(self.parameters()).device
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.rnn_type == "lstm":
            c_0 = torch.zeros_like(h_0)
            return (h_0, c_0)
        return h_0

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        hidden: tuple[Tensor, Tensor] | Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Observation tensor.  If ``x`` has shape ``(B, *obs_shape)``
            (no time dimension), it is treated as a single time step.
            If ``x`` has shape ``(B, T, *obs_shape)``, each step is
            processed sequentially through the base extractor.
        hidden : optional
            Previous hidden state.  If ``None``, a zero state is used.

        Returns
        -------
        output : Tensor of shape ``(B, hidden_size)`` (single step) or
            ``(B, T, hidden_size)`` (sequence).
        hidden : updated hidden state.
        """
        single_step = x.dim() == 2 or (x.dim() >= 2 and not self._has_time_dim(x))

        if single_step:
            features = self.base_extractor(x).unsqueeze(1)  # (B, 1, feat)
        else:
            # x: (B, T, *obs_shape) – extract features per step
            B, T = x.shape[0], x.shape[1]
            flat = x.reshape(B * T, *x.shape[2:])
            flat_features = self.base_extractor(flat)  # (B*T, feat)
            features = flat_features.reshape(B, T, -1)  # (B, T, feat)

        if hidden is None:
            hidden = self.initial_state(features.shape[0], features.device)

        rnn_out, hidden = self.rnn(features, hidden)  # (B, T, hidden_size)

        if single_step:
            rnn_out = rnn_out.squeeze(1)  # (B, hidden_size)

        return rnn_out, hidden

    # ------------------------------------------------------------------
    @staticmethod
    def _has_time_dim(x: Tensor) -> bool:
        """Heuristic: tensors with dim > 2 might have a time dimension.

        We treat 3-D+ tensors as ``(B, T, ...)`` only when the caller
        explicitly passes sequences.  For safety we default to ``False``
        for 2-D tensors (single step).
        """
        return x.dim() >= 3
