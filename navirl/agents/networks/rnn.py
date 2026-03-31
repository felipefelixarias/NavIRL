"""
Recurrent Neural Network Architectures
=======================================

Recurrent building blocks for temporal reasoning in reinforcement learning
agents navigating pedestrian environments.

Classes
-------
LSTMCore
    LSTM wrapper that manages hidden states for RL rollouts.
GRUCore
    GRU wrapper with the same interface as LSTMCore.
RecurrentPolicy
    Feature extractor + RNN core + policy head combined module.
SequenceEncoder
    Encodes variable-length observation sequences into fixed-size vectors.
HiddenStateManager
    Manages hidden states for multiple parallel environments.
AttentionOverMemory
    Multi-head attention mechanism over RNN memory sequences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# =====================================================================
# LSTMCore
# =====================================================================


class LSTMCore(nn.Module):
    """LSTM wrapper designed for RL that handles hidden state management.

    Wraps ``nn.LSTM`` with convenience methods for resetting hidden states
    and handling the ``(h, c)`` tuple that LSTM requires.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features at each time step.
    hidden_size : int
        Number of hidden units in the LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output vector at each time step."""
        return self.hidden_size

    # ------------------------------------------------------------------
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return a zero hidden state tuple ``(h_0, c_0)``.

        Parameters
        ----------
        batch_size : int
            Number of parallel sequences.
        device : torch.device, optional
            Device for the tensors.  Defaults to the device of the LSTM
            parameters.

        Returns
        -------
        tuple of (Tensor, Tensor)
            ``(h_0, c_0)`` each of shape ``(num_layers, batch_size, hidden_size)``.
        """
        if device is None:
            device = next(self.parameters()).device
        h_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        c_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
        return (h_0, c_0)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through the LSTM.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, seq_len, input_dim)`` or
            ``(batch, input_dim)`` (single time step, will be unsqueezed).
        hidden_state : tuple of (Tensor, Tensor), optional
            Previous ``(h, c)`` state.  If ``None`` a zero state is created.

        Returns
        -------
        output : Tensor
            LSTM output of shape ``(batch, seq_len, hidden_size)``.
        new_hidden : tuple of (Tensor, Tensor)
            Updated ``(h_n, c_n)`` state.
        """
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)  # (B, 1, input_dim)

        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = self.reset_hidden(batch_size, device=x.device)

        output, new_hidden = self.lstm(x, hidden_state)

        if single_step:
            output = output.squeeze(1)  # (B, hidden_size)

        return output, new_hidden

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}"
        )


# =====================================================================
# GRUCore
# =====================================================================


class GRUCore(nn.Module):
    """GRU wrapper designed for RL that handles hidden state management.

    Mirrors the :class:`LSTMCore` interface but uses a GRU cell, so the
    hidden state is a single tensor rather than a ``(h, c)`` tuple.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features at each time step.
    hidden_size : int
        Number of hidden units in the GRU.
    num_layers : int
        Number of stacked GRU layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output vector at each time step."""
        return self.hidden_size

    # ------------------------------------------------------------------
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Return a zero hidden state tensor ``h_0``.

        Parameters
        ----------
        batch_size : int
            Number of parallel sequences.
        device : torch.device, optional
            Device for the tensor.  Defaults to the device of the GRU
            parameters.

        Returns
        -------
        Tensor
            ``h_0`` of shape ``(num_layers, batch_size, hidden_size)``.
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the GRU.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, seq_len, input_dim)`` or
            ``(batch, input_dim)`` (single time step, will be unsqueezed).
        hidden_state : Tensor, optional
            Previous hidden state ``h``.  If ``None`` a zero state is created.

        Returns
        -------
        output : Tensor
            GRU output of shape ``(batch, seq_len, hidden_size)``.
        new_hidden : Tensor
            Updated ``h_n`` state of shape ``(num_layers, batch, hidden_size)``.
        """
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = self.reset_hidden(batch_size, device=x.device)

        output, new_hidden = self.gru(x, hidden_state)

        if single_step:
            output = output.squeeze(1)

        return output, new_hidden

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}"
        )


# =====================================================================
# RecurrentPolicy
# =====================================================================


class RecurrentPolicy(nn.Module):
    """Combines a feature extractor, RNN core, and policy head.

    Designed for recurrent RL policies (e.g. R2D2, PPO-LSTM) that need
    to maintain hidden states across episode time steps while resetting
    them at episode boundaries.

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw observations.
    rnn_type : str
        ``"lstm"`` or ``"gru"``.
    hidden_size : int
        Number of hidden units in the RNN core.
    num_layers : int
        Number of stacked RNN layers.
    extractor_hidden_dims : sequence of int
        Hidden layer widths for the feature extractor MLP that
        preprocesses observations before the RNN.
    policy_hidden_dims : sequence of int
        Hidden layer widths for the policy head MLP after the RNN.
    activation : str
        Activation function name used in both MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 1,
        extractor_hidden_dims: Sequence[int] = (128,),
        policy_hidden_dims: Sequence[int] = (128,),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- Feature extractor MLP ---
        extractor_layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in extractor_hidden_dims:
            extractor_layers.append(nn.Linear(prev_dim, h_dim))
            extractor_layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            prev_dim = h_dim
        self.feature_extractor = nn.Sequential(*extractor_layers)
        self._extractor_out_dim = prev_dim

        # --- RNN core ---
        if self.rnn_type == "lstm":
            self.rnn_core = LSTMCore(
                input_dim=prev_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        elif self.rnn_type == "gru":
            self.rnn_core = GRUCore(
                input_dim=prev_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        else:
            raise ValueError(
                f"Unknown rnn_type '{rnn_type}'. Choose 'lstm' or 'gru'."
            )

        # --- Policy head MLP ---
        head_layers: List[nn.Module] = []
        prev_dim = hidden_size
        for h_dim in policy_hidden_dims:
            head_layers.append(nn.Linear(prev_dim, h_dim))
            head_layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            prev_dim = h_dim
        self.policy_head = nn.Sequential(*head_layers)
        self._feature_dim = prev_dim

        # Hidden state cache for rollout
        self._hidden_state = None

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the policy head output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset the internal hidden state cache.

        Call this at episode boundaries to clear temporal memory.

        Parameters
        ----------
        batch_size : int
            Number of parallel environments / sequences.
        device : torch.device, optional
            Device for the hidden state tensors.
        """
        self._hidden_state = self.rnn_core.reset_hidden(batch_size, device=device)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        hidden_state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Forward pass: extract features, run through RNN, apply policy head.

        Parameters
        ----------
        x : Tensor
            Observations of shape ``(batch, input_dim)`` (single step) or
            ``(batch, seq_len, input_dim)`` (sequence).
        hidden_state : optional
            Externally provided hidden state.  If ``None``, uses the
            internally cached state (or creates a zero state).

        Returns
        -------
        output : Tensor
            Policy features of shape ``(batch, feature_dim)`` or
            ``(batch, seq_len, feature_dim)``.
        new_hidden : Tensor or tuple of Tensor
            Updated hidden state from the RNN core.
        """
        is_sequence = x.dim() == 3
        if is_sequence:
            batch, seq_len, _ = x.shape
            features = self.feature_extractor(
                x.reshape(batch * seq_len, -1)
            ).reshape(batch, seq_len, -1)
        else:
            features = self.feature_extractor(x)

        if hidden_state is None:
            hidden_state = self._hidden_state

        rnn_out, new_hidden = self.rnn_core(features, hidden_state)

        # Cache for next call
        self._hidden_state = new_hidden

        if is_sequence:
            batch, seq_len, _ = rnn_out.shape
            output = self.policy_head(
                rnn_out.reshape(batch * seq_len, -1)
            ).reshape(batch, seq_len, -1)
        else:
            output = self.policy_head(rnn_out)

        return output, new_hidden

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, rnn_type={self.rnn_type!r}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}"
        )


# =====================================================================
# SequenceEncoder
# =====================================================================


class SequenceEncoder(nn.Module):
    """Encodes variable-length sequences into fixed-size feature vectors.

    Processes a batch of sequences (potentially of different lengths)
    through an RNN and returns a single vector per sequence, taken from
    the last valid hidden state or via mean pooling over time steps.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each element in the sequence.
    hidden_size : int
        Number of hidden units in the RNN.
    rnn_type : str
        ``"lstm"`` or ``"gru"``.
    bidirectional : bool
        Whether to use a bidirectional RNN.  When ``True`` the output
        dimensionality doubles.
    pooling : str
        How to aggregate over time: ``"last"`` uses the final hidden
        state, ``"mean"`` averages all RNN outputs.
    num_layers : int
        Number of stacked RNN layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        pooling: str = "last",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.num_layers = num_layers
        self._num_directions = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self._feature_dim = hidden_size * self._num_directions

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the encoded output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        sequence: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a batch of sequences into fixed-size vectors.

        Parameters
        ----------
        sequence : Tensor
            Padded sequences of shape ``(batch, max_seq_len, input_dim)``.
        lengths : Tensor, optional
            Actual lengths of each sequence as a 1-D ``LongTensor`` of
            shape ``(batch,)``.  If ``None``, all sequences are assumed
            to have the same length (``max_seq_len``).

        Returns
        -------
        Tensor
            Encoded vectors of shape ``(batch, feature_dim)``.
        """
        batch_size = sequence.size(0)

        if lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu().long()
            # Clamp to at least 1 to avoid zero-length packing errors
            lengths_cpu = lengths_cpu.clamp(min=1)
            packed = pack_padded_sequence(
                sequence, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            rnn_out_packed, hidden = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(
                rnn_out_packed, batch_first=True
            )
        else:
            rnn_out, hidden = self.rnn(sequence)

        if self.pooling == "last":
            # Extract the final hidden state
            if self.rnn_type == "lstm":
                h_n = hidden[0]  # (num_layers * num_directions, batch, hidden_size)
            else:
                h_n = hidden  # (num_layers * num_directions, batch, hidden_size)

            # Take the last layer's hidden states
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                h_forward = h_n[-2]  # (batch, hidden_size)
                h_backward = h_n[-1]  # (batch, hidden_size)
                encoded = torch.cat([h_forward, h_backward], dim=-1)
            else:
                encoded = h_n[-1]  # (batch, hidden_size)
        elif self.pooling == "mean":
            if lengths is not None:
                # Mask padded positions before averaging
                max_len = rnn_out.size(1)
                mask = (
                    torch.arange(max_len, device=rnn_out.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                    < lengths.unsqueeze(1)
                )  # (batch, max_len)
                mask = mask.unsqueeze(-1).float()  # (batch, max_len, 1)
                encoded = (rnn_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                encoded = rnn_out.mean(dim=1)
        else:
            raise ValueError(
                f"Unknown pooling '{self.pooling}'. Choose 'last' or 'mean'."
            )

        return encoded

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_size={self.hidden_size}, "
            f"rnn_type={self.rnn_type!r}, bidirectional={self.bidirectional}, "
            f"pooling={self.pooling!r}"
        )


# =====================================================================
# RNNEncoder (Generic interface for tests)
# =====================================================================


class RNNEncoder(nn.Module):
    """Generic RNN encoder for testing and simple use cases.

    A simple wrapper that processes sequences through an RNN and projects
    to a specified output dimension.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each element in the input sequence.
    hidden_dim : int
        Number of hidden units in the RNN.
    num_layers : int
        Number of stacked RNN layers.
    output_dim : int
        Dimensionality of the output vector.
    rnn_type : str, optional
        Type of RNN to use ('lstm' or 'gru'). Default 'lstm'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        output_dim: int = 64,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Use SequenceEncoder as the core
        self.sequence_encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_size=hidden_dim,
            rnn_type=rnn_type,
            num_layers=num_layers,
            pooling="last",  # Take last hidden state
        )

        # Add projection layer to output_dim
        self.projection = nn.Linear(self.sequence_encoder.feature_dim, output_dim)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output vector."""
        return self.output_dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input sequences of shape (batch, seq_len, input_dim)

        Returns
        -------
        Tensor
            Output of shape (batch, output_dim)
        """
        encoded = self.sequence_encoder(x)
        output = self.projection(encoded)
        return output

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, output_dim={self.output_dim}"
        )


# =====================================================================
# HiddenStateManager
# =====================================================================


class HiddenStateManager(nn.Module):
    """Manages RNN hidden states for multiple parallel environments.

    In vectorised RL setups, each environment maintains its own hidden
    state.  This manager stores the full batch of hidden states and
    provides methods to selectively reset specific environments (e.g.
    at episode boundaries) without disturbing others.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    rnn_type : str
        ``"lstm"`` or ``"gru"``.
    hidden_size : int
        Number of hidden units per RNN layer.
    num_layers : int
        Number of stacked RNN layers.
    """

    def __init__(
        self,
        num_envs: int,
        rnn_type: str,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self._feature_dim = hidden_size

        # Register as buffers so they move with .to(device) / .cuda()
        self.register_buffer(
            "_h",
            torch.zeros(num_layers, num_envs, hidden_size),
        )
        if self.rnn_type == "lstm":
            self.register_buffer(
                "_c",
                torch.zeros(num_layers, num_envs, hidden_size),
            )

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Hidden size of the managed state."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def reset(
        self,
        env_indices: Optional[Union[Tensor, List[int], int]] = None,
    ) -> None:
        """Reset hidden states for the specified environments.

        Parameters
        ----------
        env_indices : int, list of int, Tensor, or None
            Indices of environments to reset.  If ``None``, resets all
            environments.
        """
        if env_indices is None:
            self._h.zero_()
            if self.rnn_type == "lstm":
                self._c.zero_()
            return

        if isinstance(env_indices, int):
            env_indices = [env_indices]
        if isinstance(env_indices, list):
            env_indices = torch.tensor(env_indices, dtype=torch.long)

        self._h[:, env_indices] = 0.0
        if self.rnn_type == "lstm":
            self._c[:, env_indices] = 0.0

    # ------------------------------------------------------------------
    def get(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return the current hidden state(s).

        Returns
        -------
        Tensor or tuple of (Tensor, Tensor)
            For GRU: ``h`` of shape ``(num_layers, num_envs, hidden_size)``.
            For LSTM: ``(h, c)`` tuple.
        """
        if self.rnn_type == "lstm":
            return (self._h, self._c)
        return self._h

    # ------------------------------------------------------------------
    def set(
        self,
        hidden_state: Union[Tensor, Tuple[Tensor, Tensor]],
    ) -> None:
        """Replace the stored hidden state(s).

        Parameters
        ----------
        hidden_state : Tensor or tuple of (Tensor, Tensor)
            New hidden state to store.  Must match the expected shape.
        """
        if self.rnn_type == "lstm":
            assert isinstance(hidden_state, tuple) and len(hidden_state) == 2, (
                "LSTM hidden state must be a (h, c) tuple."
            )
            self._h.copy_(hidden_state[0].detach())
            self._c.copy_(hidden_state[1].detach())
        else:
            assert isinstance(hidden_state, Tensor), (
                "GRU hidden state must be a single Tensor."
            )
            self._h.copy_(hidden_state.detach())

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"num_envs={self.num_envs}, rnn_type={self.rnn_type!r}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers}"
        )


# =====================================================================
# AttentionOverMemory
# =====================================================================


class AttentionOverMemory(nn.Module):
    """Multi-head attention mechanism over an RNN memory sequence.

    Given a query vector (e.g. current observation embedding) and a
    sequence of RNN hidden states (memory), computes scaled dot-product
    attention to produce a context vector summarising the relevant parts
    of memory.

    Parameters
    ----------
    query_dim : int
        Dimensionality of the query vector.
    memory_dim : int
        Dimensionality of each memory vector in the sequence.
    num_heads : int
        Number of attention heads.
    dropout : float
        Attention weight dropout probability.
    """

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads

        # Project query and memory to a common dimension for attention
        # Use memory_dim as the internal dimension; must be divisible by num_heads
        if memory_dim % num_heads != 0:
            raise ValueError(
                f"memory_dim ({memory_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self.head_dim = memory_dim // num_heads

        self.query_proj = nn.Linear(query_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        self.output_proj = nn.Linear(memory_dim, memory_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

        self._feature_dim = memory_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the attended output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        query: Tensor,
        memory_sequence: Tensor,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Attend over the memory sequence using the query.

        Parameters
        ----------
        query : Tensor
            Query vector of shape ``(batch, query_dim)``.
        memory_sequence : Tensor
            Memory to attend over, shape ``(batch, seq_len, memory_dim)``.
        memory_mask : Tensor, optional
            Boolean mask of shape ``(batch, seq_len)`` where ``True``
            indicates a valid (non-padded) position.  Padded positions
            receive ``-inf`` attention weight.

        Returns
        -------
        Tensor
            Attended context vector of shape ``(batch, memory_dim)``.
        """
        batch_size = query.size(0)
        seq_len = memory_sequence.size(1)

        # Project to multi-head space
        q = self.query_proj(query)  # (B, memory_dim)
        k = self.key_proj(memory_sequence)  # (B, S, memory_dim)
        v = self.value_proj(memory_sequence)  # (B, S, memory_dim)

        # Reshape for multi-head: (B, num_heads, *, head_dim)
        q = q.view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)
        # q: (B, num_heads, 1, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k: (B, num_heads, S, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v: (B, num_heads, S, head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_weights: (B, num_heads, 1, S)

        if memory_mask is not None:
            # memory_mask: (B, S) -> (B, 1, 1, S)
            mask = memory_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attended = torch.matmul(attn_weights, v)
        # attended: (B, num_heads, 1, head_dim)
        attended = attended.squeeze(2)  # (B, num_heads, head_dim)
        attended = attended.reshape(batch_size, self.num_heads * self.head_dim)
        # attended: (B, memory_dim)

        output = self.output_proj(attended)
        return output

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"query_dim={self.query_dim}, memory_dim={self.memory_dim}, "
            f"num_heads={self.num_heads}"
        )
