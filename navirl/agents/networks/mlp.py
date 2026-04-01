"""
Multi-Layer Perceptron Architectures
=====================================

Configurable MLP variants for reinforcement learning in pedestrian
navigation environments.

Classes
-------
MLP
    Standard MLP with configurable depth, width, activations, normalization.
DuelingMLP
    Dueling architecture splitting into value and advantage streams.
NoisyLinear
    Linear layer with learnable Gaussian noise for exploration.
NoisyMLP
    MLP built from NoisyLinear layers (NoisyNet).
ResidualMLP
    MLP with skip / residual connections.
GatedMLP
    MLP with gating mechanisms for selective feature propagation.

Functions
---------
init_weights_xavier, init_weights_orthogonal, init_weights_kaiming,
init_weights_uniform
    Weight initialization utilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
    "none": nn.Identity,
}


def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name (case-insensitive)."""
    key = name.lower().strip()
    if key not in _ACTIVATIONS:
        msg = f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS.keys())}"
        raise ValueError(msg)
    return _ACTIVATIONS[key]()


# =====================================================================
# Weight initialisation utilities
# =====================================================================


def init_weights_xavier(
    module: nn.Module,
    gain: float = 1.0,
    bias_val: float = 0.0,
) -> None:
    """Apply Xavier (Glorot) uniform initialisation to all Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_val)


def init_weights_orthogonal(
    module: nn.Module,
    gain: float = 1.0,
    bias_val: float = 0.0,
) -> None:
    """Apply orthogonal initialisation — common in PPO / A2C."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_val)


def init_weights_kaiming(
    module: nn.Module,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    bias_val: float = 0.0,
) -> None:
    """Apply Kaiming (He) normal initialisation — good for ReLU networks."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_val)


def init_weights_uniform(
    module: nn.Module,
    low: float = -3e-3,
    high: float = 3e-3,
) -> None:
    """Apply small uniform initialisation — common for final policy layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, low, high)
            if m.bias is not None:
                nn.init.uniform_(m.bias, low, high)


# =====================================================================
# Standard MLP
# =====================================================================


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    output_dim : int
        Dimensionality of the output.  Set to 0 or ``None`` to omit the
        output projection (the network then ends after the last hidden layer
        + activation).
    hidden_dims : sequence of int
        Widths of hidden layers.  ``(256, 256)`` gives a 2-hidden-layer MLP.
    activation : str
        Name of the activation function (see ``_ACTIVATIONS``).
    output_activation : str
        Activation applied after the output projection.  ``"none"`` means
        linear output.
    layer_norm : bool
        If ``True``, apply ``LayerNorm`` after each hidden linear layer
        (before activation).
    batch_norm : bool
        If ``True``, apply ``BatchNorm1d`` instead of ``LayerNorm``.
        Mutually exclusive with ``layer_norm``.
    dropout : float
        Dropout probability applied after each hidden activation.
        0.0 disables dropout.
    init : str or None
        Weight initialisation strategy: ``"xavier"``, ``"orthogonal"``,
        ``"kaiming"``, ``"uniform"``, or ``None`` (PyTorch default).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        output_activation: str = "none",
        layer_norm: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        init: str | None = None,
    ) -> None:
        super().__init__()
        if layer_norm and batch_norm:
            raise ValueError("Cannot use both layer_norm and batch_norm.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self._hidden_dims = list(hidden_dims)

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(_get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Optional output projection
        if output_dim is not None and output_dim > 0:
            self.output_layer = nn.Linear(prev_dim, output_dim)
            self.output_act = _get_activation(output_activation)
            self._feature_dim = output_dim
        else:
            self.output_layer = None
            self.output_act = None
            self._feature_dim = prev_dim

        # Initialisation
        if init is not None:
            self._apply_init(init)

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def _apply_init(self, strategy: str) -> None:
        strategy = strategy.lower()
        if strategy == "xavier":
            init_weights_xavier(self)
        elif strategy == "orthogonal":
            init_weights_orthogonal(self)
        elif strategy == "kaiming":
            init_weights_kaiming(self)
        elif strategy == "uniform":
            init_weights_uniform(self)
        else:
            msg = f"Unknown init strategy: {strategy}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(*, input_dim)``

        Returns
        -------
        Tensor of shape ``(*, output_dim)`` or ``(*, hidden_dims[-1])``
        """
        h = self.hidden_layers(x)
        if self.output_layer is not None:
            h = self.output_layer(h)
            h = self.output_act(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dims={self._hidden_dims}"
        )


# =====================================================================
# Dueling MLP
# =====================================================================


class DuelingMLP(nn.Module):
    """Dueling network architecture (Wang et al., 2016).

    Splits the representation into a *value stream* and an *advantage stream*
    and combines them as ``Q(s,a) = V(s) + A(s,a) - mean(A(s,·))``.

    Parameters
    ----------
    input_dim : int
        Input feature dimensionality.
    num_actions : int
        Number of discrete actions (output width).
    hidden_dims : sequence of int
        Shared hidden-layer widths.
    value_hidden_dims : sequence of int
        Hidden widths of the value stream after the shared trunk.
    advantage_hidden_dims : sequence of int
        Hidden widths of the advantage stream after the shared trunk.
    activation : str
        Activation function name.
    layer_norm : bool
        Whether to apply layer norm in hidden layers.
    aggregation : str
        ``"mean"`` (default) or ``"max"`` for advantage centring.
    init : str or None
        Weight initialisation strategy.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (128,),
        advantage_hidden_dims: Sequence[int] = (128,),
        activation: str = "relu",
        layer_norm: bool = False,
        aggregation: str = "mean",
        init: str | None = None,
    ) -> None:
        super().__init__()
        assert aggregation in ("mean", "max"), f"Invalid aggregation: {aggregation}"
        self.aggregation = aggregation
        self.num_actions = num_actions

        # Shared trunk (output_dim=0 means no final projection)
        self.shared = MLP(
            input_dim=input_dim,
            output_dim=0,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            init=init,
        )
        trunk_dim = self.shared.feature_dim

        # Value stream  V(s) -> scalar
        self.value_stream = MLP(
            input_dim=trunk_dim,
            output_dim=1,
            hidden_dims=value_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            init=init,
        )

        # Advantage stream  A(s,a) -> R^|A|
        self.advantage_stream = MLP(
            input_dim=trunk_dim,
            output_dim=num_actions,
            hidden_dims=advantage_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            init=init,
        )

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Return Q-values for every action.

        Parameters
        ----------
        x : Tensor ``(B, input_dim)``

        Returns
        -------
        Tensor ``(B, num_actions)``
        """
        features = self.shared(x)
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, num_actions)

        if self.aggregation == "mean":
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = value + advantage - advantage.max(dim=-1, keepdim=True).values
        return q

    # ------------------------------------------------------------------
    def q_value(self, x: Tensor, action: Tensor) -> Tensor:
        """Return Q-value for a specific action.

        Parameters
        ----------
        x : Tensor ``(B, input_dim)``
        action : LongTensor ``(B,)`` or ``(B, 1)``

        Returns
        -------
        Tensor ``(B, 1)``
        """
        q_all = self.forward(x)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        return q_all.gather(dim=-1, index=action.long())


# =====================================================================
# NoisyLinear
# =====================================================================


class NoisyLinear(nn.Module):
    """Factorised NoisyNet linear layer (Fortunato et al., 2018).

    Replaces a standard ``nn.Linear`` with learnable additive Gaussian
    noise on both weights and biases, enabling parameter-space exploration
    without explicit epsilon-greedy schedules.

    Parameters
    ----------
    in_features : int
    out_features : int
    sigma_init : float
        Initial value for the noise scale parameter.
    factorised : bool
        If ``True`` use factorised noise (cheaper); else independent noise.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
        factorised: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.factorised = factorised

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        if self.factorised:
            bound = 1.0 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-bound, bound)
            self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-bound, bound)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        else:
            bound = 1.0 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-bound, bound)
            self.weight_sigma.data.fill_(0.017)
            self.bias_mu.data.uniform_(-bound, bound)
            self.bias_sigma.data.fill_(0.017)

    # ------------------------------------------------------------------
    @staticmethod
    def _f(x: Tensor) -> Tensor:
        """Factorised noise transform: sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample the noise tensors."""
        if self.factorised:
            eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
            eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
            self.weight_epsilon.copy_(eps_out.outer(eps_in))
            self.bias_epsilon.copy_(eps_out)
        else:
            self.weight_epsilon.normal_()
            self.bias_epsilon.normal_()

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sigma_init={self.sigma_init}, factorised={self.factorised}"
        )


# =====================================================================
# NoisyMLP
# =====================================================================


class NoisyMLP(nn.Module):
    """MLP using NoisyLinear layers for parameter-space exploration.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_dims : sequence of int
    activation : str
    sigma_init : float
    factorised : bool
    layer_norm : bool
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
        output_activation: str = "none",
        sigma_init: float = 0.5,
        factorised: bool = True,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        prev = input_dim
        self._noisy_layers: list[NoisyLinear] = []

        for h in hidden_dims:
            noisy = NoisyLinear(prev, h, sigma_init=sigma_init, factorised=factorised)
            layers.append(noisy)
            self._noisy_layers.append(noisy)
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_get_activation(activation))
            prev = h

        self.hidden = nn.Sequential(*layers)

        if output_dim is not None and output_dim > 0:
            self.output_layer = NoisyLinear(
                prev, output_dim, sigma_init=sigma_init, factorised=factorised
            )
            self._noisy_layers.append(self.output_layer)
            self.output_act = _get_activation(output_activation)
            self._feature_dim = output_dim
        else:
            self.output_layer = None
            self.output_act = None
            self._feature_dim = prev

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def reset_noise(self) -> None:
        """Resample noise in all NoisyLinear layers."""
        for layer in self._noisy_layers:
            layer.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        h = self.hidden(x)
        if self.output_layer is not None:
            h = self.output_act(self.output_layer(h))
        return h


# =====================================================================
# Residual MLP
# =====================================================================


class _ResidualBlock(nn.Module):
    """A single residual block: ``x + f(x)`` with an optional projection."""

    def __init__(
        self,
        dim: int,
        activation: str = "relu",
        layer_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        modules.append(nn.Linear(dim, dim))
        if layer_norm:
            modules.append(nn.LayerNorm(dim))
        modules.append(_get_activation(activation))
        if dropout > 0.0:
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(dim, dim))
        if layer_norm:
            modules.append(nn.LayerNorm(dim))
        self.block = nn.Sequential(*modules)
        self.act = _get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class ResidualMLP(nn.Module):
    """MLP with skip connections.

    An initial linear projection maps the input to ``hidden_dim``, then
    ``num_blocks`` residual blocks are applied, followed by an optional
    output projection.

    Parameters
    ----------
    input_dim : int
    output_dim : int
        Set to 0 or ``None`` to omit.
    hidden_dim : int
        Width of every hidden layer (residual blocks are square).
    num_blocks : int
        Number of residual blocks.
    activation : str
    layer_norm : bool
    dropout : float
    init : str or None
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        activation: str = "relu",
        output_activation: str = "none",
        layer_norm: bool = False,
        dropout: float = 0.0,
        init: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_act = _get_activation(activation)
        if layer_norm:
            self.input_norm = nn.LayerNorm(hidden_dim)
        else:
            self.input_norm = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                _ResidualBlock(
                    hidden_dim,
                    activation=activation,
                    layer_norm=layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        if output_dim is not None and output_dim > 0:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
            self.output_act = _get_activation(output_activation)
            self._feature_dim = output_dim
        else:
            self.output_layer = None
            self.output_act = None
            self._feature_dim = hidden_dim

        if init is not None:
            {
                "xavier": init_weights_xavier,
                "orthogonal": init_weights_orthogonal,
                "kaiming": init_weights_kaiming,
                "uniform": init_weights_uniform,
            }[init.lower()](self)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_act(self.input_norm(self.input_proj(x)))
        h = self.blocks(h)
        if self.output_layer is not None:
            h = self.output_act(self.output_layer(h))
        return h


# =====================================================================
# Gated MLP
# =====================================================================


class _GatedBlock(nn.Module):
    """Gated linear unit block: ``sigmoid(Wg x + bg) * (Wh x + bh)``."""

    def __init__(self, input_dim: int, output_dim: int, layer_norm: bool = False) -> None:
        super().__init__()
        self.linear_gate = nn.Linear(input_dim, output_dim)
        self.linear_hidden = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if layer_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.linear_gate(x))
        hidden = self.linear_hidden(x)
        return self.norm(gate * hidden)


class GatedMLP(nn.Module):
    """MLP with gating mechanisms for selective feature propagation.

    Each hidden layer is a ``_GatedBlock`` that learns a multiplicative
    gate controlling which features pass through.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_dims : sequence of int
    layer_norm : bool
    dropout : float
    output_activation : str
    init : str or None
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        layer_norm: bool = False,
        dropout: float = 0.0,
        output_activation: str = "none",
        init: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        blocks: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            blocks.append(_GatedBlock(prev, h, layer_norm=layer_norm))
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            prev = h
        self.gated_blocks = nn.Sequential(*blocks)

        if output_dim is not None and output_dim > 0:
            self.output_layer = nn.Linear(prev, output_dim)
            self.output_act = _get_activation(output_activation)
            self._feature_dim = output_dim
        else:
            self.output_layer = None
            self.output_act = None
            self._feature_dim = prev

        if init is not None:
            {
                "xavier": init_weights_xavier,
                "orthogonal": init_weights_orthogonal,
                "kaiming": init_weights_kaiming,
                "uniform": init_weights_uniform,
            }[init.lower()](self)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: Tensor) -> Tensor:
        h = self.gated_blocks(x)
        if self.output_layer is not None:
            h = self.output_act(self.output_layer(h))
        return h
