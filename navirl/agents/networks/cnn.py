"""
Convolutional Neural Network Architectures
============================================

CNN variants for processing grid, image, and lidar observations in
reinforcement learning for pedestrian navigation environments.

Classes
-------
NatureDQN
    DeepMind Nature DQN CNN (Mnih et al., 2015).
ImpalaCNN
    IMPALA-style deep residual CNN (Espeholt et al., 2018).
OccupancyGridEncoder
    CNN encoder for 2D occupancy grid maps.
LidarEncoder
    1D CNN encoder for lidar range-beam arrays.
EgoCentricCNN
    CNN for ego-centric local map patches.
SpatialAttentionModule
    CBAM-style spatial attention mechanism for conv feature maps.
MultiScaleFeatureExtractor
    Multi-scale feature pyramid with concatenated outputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Activation helper (mirrors mlp.py)
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
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS.keys())}")
    return _ACTIVATIONS[key]()


# ---------------------------------------------------------------------------
# Utility: compute conv output size
# ---------------------------------------------------------------------------


def _conv2d_output_size(
    height: int,
    width: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> tuple[int, int]:
    """Return (H_out, W_out) after a single Conv2d layer."""
    h_out = (height - kernel_size + 2 * padding) // stride + 1
    w_out = (width - kernel_size + 2 * padding) // stride + 1
    return h_out, w_out


def _conv1d_output_size(
    length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> int:
    """Return L_out after a single Conv1d layer."""
    return (length - kernel_size + 2 * padding) // stride + 1


# =====================================================================
# NatureDQN
# =====================================================================


class NatureDQN(nn.Module):
    """DeepMind Nature DQN convolutional encoder (Mnih et al., 2015).

    Architecture: three convolutional layers (32x8x4, 64x4x2, 64x3x1)
    followed by a flatten and an optional linear projection.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g. 1 for greyscale, 4 for stacked
        frames).
    input_height : int
        Height of the input image.
    input_width : int
        Width of the input image.
    output_dim : int or None
        Dimensionality of the linear output projection.  If ``None`` or
        0, the network outputs the flattened convolutional features
        directly.
    activation : str
        Activation function applied after each conv layer.
    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        output_dim: int | None = 512,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            _get_activation(activation),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            _get_activation(activation),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            _get_activation(activation),
        )

        # Compute the flattened feature size after conv layers
        h, w = input_height, input_width
        h, w = _conv2d_output_size(h, w, kernel_size=8, stride=4)
        h, w = _conv2d_output_size(h, w, kernel_size=4, stride=2)
        h, w = _conv2d_output_size(h, w, kernel_size=3, stride=1)
        self._flat_dim = 64 * h * w

        if output_dim is not None and output_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(self._flat_dim, output_dim),
                _get_activation(activation),
            )
            self._feature_dim = output_dim
        else:
            self.fc = None
            self._feature_dim = self._flat_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, C, H, W)``

        Returns
        -------
        Tensor of shape ``(B, feature_dim)``
        """
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        if self.fc is not None:
            h = self.fc(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"input_size=({self.input_height}, {self.input_width}), "
            f"output_dim={self.output_dim}"
        )


# =====================================================================
# IMPALA CNN
# =====================================================================


class _ImpalaResidualBlock(nn.Module):
    """Single residual block used within an IMPALA conv sequence."""

    def __init__(self, channels: int, activation: str = "relu") -> None:
        super().__init__()
        self.block = nn.Sequential(
            _get_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            _get_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _ImpalaConvSequence(nn.Module):
    """One IMPALA conv sequence: conv → max-pool → 2 residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ImpalaResidualBlock(out_channels, activation=activation)
        self.res2 = _ImpalaResidualBlock(out_channels, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x)
        h = self.pool(h)
        h = self.res1(h)
        h = self.res2(h)
        return h


class ImpalaCNN(nn.Module):
    """IMPALA-style deep residual CNN encoder (Espeholt et al., 2018).

    Uses a sequence of conv-maxpool-residual stacks, producing a
    high-quality spatial encoding that is flattened and projected.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    input_height : int
        Height of the input image.
    input_width : int
        Width of the input image.
    channel_sequence : sequence of int
        Number of output channels for each conv sequence stage.
        Default ``(16, 32, 32)`` follows the original IMPALA paper.
    output_dim : int or None
        Linear projection dimensionality.  ``None`` or 0 to skip.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        channel_sequence: Sequence[int] = (16, 32, 32),
        output_dim: int | None = 256,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim

        stages: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in channel_sequence:
            stages.append(_ImpalaConvSequence(in_ch, out_ch, activation=activation))
            in_ch = out_ch
        self.conv_stages = nn.Sequential(*stages)
        self.final_act = _get_activation(activation)

        # Compute spatial dimensions after all stages (each halves via
        # MaxPool2d with stride=2, padding=1)
        h, w = input_height, input_width
        for _ in channel_sequence:
            # MaxPool2d(kernel_size=3, stride=2, padding=1)
            h = (h - 3 + 2 * 1) // 2 + 1
            w = (w - 3 + 2 * 1) // 2 + 1
        self._flat_dim = int(channel_sequence[-1]) * h * w

        if output_dim is not None and output_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(self._flat_dim, output_dim),
                _get_activation(activation),
            )
            self._feature_dim = output_dim
        else:
            self.fc = None
            self._feature_dim = self._flat_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, C, H, W)``

        Returns
        -------
        Tensor of shape ``(B, feature_dim)``
        """
        h = self.conv_stages(x)
        h = self.final_act(h)
        h = h.reshape(h.size(0), -1)
        if self.fc is not None:
            h = self.fc(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"input_size=({self.input_height}, {self.input_width}), "
            f"output_dim={self.output_dim}"
        )


# =====================================================================
# OccupancyGridEncoder
# =====================================================================


class OccupancyGridEncoder(nn.Module):
    """CNN encoder tailored for 2D occupancy grid maps.

    Designed for small-to-medium grid sizes typically used in pedestrian
    navigation (e.g. 64x64 or 84x84).  Three conv layers with batch
    normalisation progressively downsample the grid before a linear
    projection.

    Parameters
    ----------
    grid_size : int
        Spatial size of the square input grid (height = width).
    input_channels : int
        Number of input channels (e.g. 1 for binary occupancy, 3 for
        multi-layer semantic maps).
    output_dim : int
        Dimensionality of the output feature vector.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        grid_size: int,
        input_channels: int = 1,
        output_dim: int = 128,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            _get_activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            _get_activation(activation),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            _get_activation(activation),
        )

        # Compute flattened dimension
        h = w = grid_size
        h, w = _conv2d_output_size(h, w, kernel_size=5, stride=2, padding=2)
        h, w = _conv2d_output_size(h, w, kernel_size=3, stride=2, padding=1)
        h, w = _conv2d_output_size(h, w, kernel_size=3, stride=2, padding=1)
        self._flat_dim = 64 * h * w

        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, output_dim),
            _get_activation(activation),
        )
        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, C, grid_size, grid_size)``

        Returns
        -------
        Tensor of shape ``(B, output_dim)``
        """
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"grid_size={self.grid_size}, "
            f"input_channels={self.input_channels}, "
            f"output_dim={self.output_dim}"
        )


# =====================================================================
# LidarEncoder
# =====================================================================


class LidarEncoder(nn.Module):
    """1D CNN encoder for lidar range-beam arrays.

    Processes a 1D array of range measurements (e.g. 360 beams) using
    1D convolutions followed by a linear projection.

    Parameters
    ----------
    num_beams : int
        Number of lidar beams (length of the 1D input).
    output_dim : int
        Dimensionality of the output feature vector.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        num_beams: int = 360,
        output_dim: int = 128,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_beams = num_beams
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(32),
            _get_activation(activation),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            _get_activation(activation),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            _get_activation(activation),
        )

        # Compute flattened dimension
        length = num_beams
        length = _conv1d_output_size(length, kernel_size=7, stride=3, padding=3)
        length = _conv1d_output_size(length, kernel_size=5, stride=2, padding=2)
        length = _conv1d_output_size(length, kernel_size=3, stride=2, padding=1)
        self._flat_dim = 64 * length

        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, output_dim),
            _get_activation(activation),
        )
        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, num_beams)`` or ``(B, 1, num_beams)``
            If 2D, a channel dimension is added automatically.

        Returns
        -------
        Tensor of shape ``(B, output_dim)``
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, num_beams)
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"num_beams={self.num_beams}, output_dim={self.output_dim}"


# =====================================================================
# EgoCentricCNN
# =====================================================================


class EgoCentricCNN(nn.Module):
    """CNN for processing ego-centric local map patches.

    Designed for small square patches centred on the agent (e.g. 32x32
    or 48x48) that encode the agent's immediate surroundings.

    Parameters
    ----------
    patch_size : int
        Spatial size of the square input patch (height = width).
    input_channels : int
        Number of input channels.
    output_dim : int
        Dimensionality of the output feature vector.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        patch_size: int,
        input_channels: int = 1,
        output_dim: int = 128,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            _get_activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            _get_activation(activation),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            _get_activation(activation),
        )

        # Compute flattened dimension
        h = w = patch_size
        h, w = _conv2d_output_size(h, w, kernel_size=3, stride=2, padding=1)
        h, w = _conv2d_output_size(h, w, kernel_size=3, stride=2, padding=1)
        h, w = _conv2d_output_size(h, w, kernel_size=3, stride=2, padding=1)
        self._flat_dim = 64 * h * w

        self.fc = nn.Sequential(
            nn.Linear(self._flat_dim, output_dim),
            _get_activation(activation),
        )
        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, C, patch_size, patch_size)``

        Returns
        -------
        Tensor of shape ``(B, output_dim)``
        """
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, "
            f"input_channels={self.input_channels}, "
            f"output_dim={self.output_dim}"
        )


# =====================================================================
# SpatialAttentionModule
# =====================================================================


class SpatialAttentionModule(nn.Module):
    """CBAM-style spatial attention module (Woo et al., 2018).

    Applies both *channel attention* and *spatial attention* sequentially
    to refine convolutional feature maps.  Channel attention squeezes
    spatial dimensions to re-weight channels; spatial attention squeezes
    the channel dimension to produce a spatial mask.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    reduction : int
        Channel-attention bottleneck reduction ratio.
    kernel_size : int
        Kernel size for the spatial attention convolution (must be odd).
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Channel attention (shared MLP on pooled features)
        mid = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

        # Spatial attention (conv on concatenated avg/max pooled channels)
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output channels (unchanged from input)."""
        return self.channels

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Apply channel and spatial attention.

        Parameters
        ----------
        x : Tensor of shape ``(B, C, H, W)``

        Returns
        -------
        Tensor of shape ``(B, C, H, W)``
        """
        # --- Channel attention ---
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=(2, 3))  # (B, C)
        max_pool = x.amax(dim=(2, 3))  # (B, C)
        channel_att = torch.sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        )  # (B, C)
        x = x * channel_att.unsqueeze(-1).unsqueeze(-1)

        # --- Spatial attention ---
        avg_spatial = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        max_spatial = x.amax(dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_att = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        )  # (B, 1, H, W)
        x = x * spatial_att

        return x

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return f"channels={self.channels}, reduction={self.reduction}"


# =====================================================================
# MultiScaleFeatureExtractor
# =====================================================================


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature pyramid extractor.

    Processes the input image at multiple resolutions using separate
    convolutional branches, then concatenates the resulting feature
    vectors to capture both fine-grained and coarse spatial information.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    input_height : int
        Height of the input image.
    input_width : int
        Width of the input image.
    branch_channels : sequence of int
        Output channels for each scale branch.  Default ``(32, 64, 64)``
        creates three branches.
    scales : sequence of float
        Downscale factors applied to the input before each branch.
        ``1.0`` means original resolution, ``0.5`` means half, etc.
        Must have the same length as ``branch_channels``.
    output_dim : int or None
        Optional linear projection applied to the concatenated features.
        ``None`` or 0 to skip.
    activation : str
        Activation function name.
    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        branch_channels: Sequence[int] = (32, 64, 64),
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        output_dim: int | None = 256,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        assert len(branch_channels) == len(
            scales
        ), "branch_channels and scales must have the same length"
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim
        self.scales = list(scales)

        self.branches = nn.ModuleList()
        total_flat = 0

        for ch, scale in zip(branch_channels, scales, strict=False):
            # Determine spatial size at this scale
            max(int(input_height * scale), 1)
            max(int(input_width * scale), 1)

            branch = nn.Sequential(
                nn.Conv2d(input_channels, ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                _get_activation(activation),
                nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                _get_activation(activation),
                nn.AdaptiveAvgPool2d(1),
            )
            self.branches.append(branch)
            total_flat += ch  # AdaptiveAvgPool2d(1) → (B, ch, 1, 1) → ch

        self._concat_dim = total_flat

        if output_dim is not None and output_dim > 0:
            self.fc = nn.Sequential(
                nn.Linear(total_flat, output_dim),
                _get_activation(activation),
            )
            self._feature_dim = output_dim
        else:
            self.fc = None
            self._feature_dim = total_flat

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape ``(B, C, H, W)``

        Returns
        -------
        Tensor of shape ``(B, feature_dim)``
        """
        branch_outputs: list[Tensor] = []
        for branch, scale in zip(self.branches, self.scales, strict=False):
            if scale != 1.0:
                scaled = F.interpolate(
                    x,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scaled = x
            feat = branch(scaled)  # (B, ch, 1, 1)
            branch_outputs.append(feat.reshape(feat.size(0), -1))

        h = torch.cat(branch_outputs, dim=-1)  # (B, concat_dim)
        if self.fc is not None:
            h = self.fc(h)
        return h

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"input_size=({self.input_height}, {self.input_width}), "
            f"scales={self.scales}, output_dim={self.output_dim}"
        )
