"""Inter-agent communication primitives and learned communication modules.

Provides both classical message-passing channels (broadcast, direct, shared
memory) and differentiable neural communication modules (CommNet, TarMAC).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Message protocol
# ---------------------------------------------------------------------------

@dataclass
class MessageProtocol:
    """Defines the standard message format exchanged between agents.

    Attributes:
        sender: Identifier of the sending agent.
        receiver: Identifier of the receiving agent (``None`` for broadcast).
        content: Arbitrary payload carried by the message.
        timestamp: Unix timestamp when the message was created.
        metadata: Optional additional key-value metadata.
    """

    sender: str
    receiver: str | None
    content: Any
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classical channels
# ---------------------------------------------------------------------------

class BroadcastChannel:
    """Communication channel where all agents receive all messages.

    Messages are stored in a shared buffer and can be read by any agent.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_buffer_size: int = 1000) -> None:
        self._buffer: list[MessageProtocol] = []
        self._max_buffer_size = max_buffer_size
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def send(self, message: MessageProtocol) -> None:
        """Broadcast *message* to all agents."""
        with self._lock:
            self._buffer.append(message)
            if len(self._buffer) > self._max_buffer_size:
                self._buffer = self._buffer[-self._max_buffer_size :]

    def receive(self, agent_id: str | None = None) -> list[MessageProtocol]:
        """Return all messages in the buffer.

        Parameters:
            agent_id: Ignored for broadcast; kept for API compatibility.
        """
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        """Number of messages currently buffered."""
        return len(self._buffer)


class DirectChannel:
    """Point-to-point messaging between specific agent pairs.

    Each agent has its own mailbox; messages are delivered based on the
    ``receiver`` field of :class:`MessageProtocol`.
    """

    def __init__(self) -> None:
        self._mailboxes: dict[str, list[MessageProtocol]] = {}
        self._lock = threading.Lock()

    def send(self, message: MessageProtocol) -> None:
        """Deliver *message* to the receiver's mailbox.

        Raises:
            ValueError: If ``message.receiver`` is ``None``.
        """
        if message.receiver is None:
            raise ValueError("DirectChannel requires a specific receiver.")
        with self._lock:
            self._mailboxes.setdefault(message.receiver, []).append(message)

    def receive(self, agent_id: str) -> list[MessageProtocol]:
        """Return and clear all messages addressed to *agent_id*."""
        with self._lock:
            messages = self._mailboxes.pop(agent_id, [])
        return messages

    def peek(self, agent_id: str) -> list[MessageProtocol]:
        """Return messages for *agent_id* without removing them."""
        with self._lock:
            return list(self._mailboxes.get(agent_id, []))


class SharedMemory:
    """Shared state visible to all agents.

    Behaves as a thread-safe key-value store that any agent can read or
    write.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: Any) -> None:
        """Write *value* under *key* in shared memory."""
        with self._lock:
            self._store[key] = value

    def read(self, key: str, default: Any = None) -> Any:
        """Read the value stored under *key*, returning *default* if absent."""
        with self._lock:
            return self._store.get(key, default)

    def read_all(self) -> dict[str, Any]:
        """Return a shallow copy of the entire shared state."""
        with self._lock:
            return dict(self._store)

    def clear(self) -> None:
        """Clear all shared state."""
        with self._lock:
            self._store.clear()

    @property
    def keys(self) -> list[str]:
        """All keys currently in shared memory."""
        return list(self._store.keys())


# ---------------------------------------------------------------------------
# Learned communication — CommNet-style
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class CommNetwork(nn.Module):
        """Learned communication via mean-pooling aggregation (CommNet-style).

        Each agent produces a message from its hidden state.  Messages are
        mean-pooled across all *other* agents and concatenated with the
        agent's own hidden state to produce an updated representation.

        Parameters:
            hidden_dim: Dimensionality of agent hidden states.
            message_dim: Dimensionality of the produced messages.
            num_comm_rounds: Number of communication rounds.
        """

        def __init__(
            self,
            hidden_dim: int,
            message_dim: int = 64,
            num_comm_rounds: int = 1,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.message_dim = message_dim
            self.num_comm_rounds = num_comm_rounds

            # Message encoder: hidden -> message
            self.message_encoder = nn.Linear(hidden_dim, message_dim)
            # Integration: hidden + aggregated message -> updated hidden
            self.integration = nn.Linear(hidden_dim + message_dim, hidden_dim)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Produce updated hidden states after communication.

            Parameters:
                hidden_states: Tensor of shape ``(num_agents, hidden_dim)``
                    containing the hidden state for each agent.

            Returns:
                Updated hidden states of shape ``(num_agents, hidden_dim)``.
            """
            h = hidden_states  # (N, hidden_dim)
            num_agents = h.size(0)

            for _ in range(self.num_comm_rounds):
                # Encode messages
                messages = torch.tanh(self.message_encoder(h))  # (N, message_dim)

                # Mean-pool messages from *other* agents
                total = messages.sum(dim=0, keepdim=True)  # (1, message_dim)
                # For each agent, subtract own message and average over others
                if num_agents > 1:
                    aggregated = (total - messages) / (num_agents - 1)
                else:
                    aggregated = torch.zeros_like(messages)

                # Integrate
                h = torch.tanh(self.integration(torch.cat([h, aggregated], dim=-1)))

            return h

    class AttentionComm(nn.Module):
        """Attention-based communication module (TarMAC-style).

        Each agent attends to all other agents' messages using learned soft
        attention, producing a weighted aggregation instead of a uniform
        mean-pool.

        Parameters:
            hidden_dim: Dimensionality of agent hidden states.
            message_dim: Dimensionality of the produced messages.
            key_dim: Dimensionality of the attention keys/queries.
            num_heads: Number of attention heads (multi-head attention).
        """

        def __init__(
            self,
            hidden_dim: int,
            message_dim: int = 64,
            key_dim: int = 32,
            num_heads: int = 1,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.message_dim = message_dim
            self.key_dim = key_dim
            self.num_heads = num_heads

            # Projections
            self.query_proj = nn.Linear(hidden_dim, key_dim * num_heads)
            self.key_proj = nn.Linear(hidden_dim, key_dim * num_heads)
            self.value_proj = nn.Linear(hidden_dim, message_dim * num_heads)

            # Output projection after multi-head concat
            self.output_proj = nn.Linear(message_dim * num_heads, message_dim)
            # Integration: hidden + attended message -> updated hidden
            self.integration = nn.Linear(hidden_dim + message_dim, hidden_dim)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Produce updated hidden states after attention-based communication.

            Parameters:
                hidden_states: Tensor of shape ``(num_agents, hidden_dim)``.

            Returns:
                Updated hidden states of shape ``(num_agents, hidden_dim)``.
            """
            num_agents = hidden_states.size(0)
            h = hidden_states

            # Queries, keys, values  — (N, num_heads, dim)
            queries = self.query_proj(h).view(num_agents, self.num_heads, self.key_dim)
            keys = self.key_proj(h).view(num_agents, self.num_heads, self.key_dim)
            values = self.value_proj(h).view(num_agents, self.num_heads, self.message_dim)

            # Attention scores — (num_heads, N, N)
            # queries: (N, H, Dk) -> (H, N, Dk)
            q = queries.permute(1, 0, 2)
            k = keys.permute(1, 0, 2)
            v = values.permute(1, 0, 2)

            scores = torch.bmm(q, k.transpose(1, 2)) / (self.key_dim ** 0.5)

            # Mask self-attention (agents do not attend to themselves)
            mask = torch.eye(num_agents, device=h.device, dtype=torch.bool)
            mask = mask.unsqueeze(0).expand(self.num_heads, -1, -1)
            scores = scores.masked_fill(mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)  # (H, N, N)
            # Handle single-agent edge case (all -inf -> nan after softmax)
            attn_weights = attn_weights.nan_to_num(0.0)

            # Weighted aggregation
            aggregated = torch.bmm(attn_weights, v)  # (H, N, Dm)
            # Concatenate heads -> (N, H*Dm)
            aggregated = aggregated.permute(1, 0, 2).contiguous().view(num_agents, -1)
            aggregated = self.output_proj(aggregated)  # (N, message_dim)

            # Integrate
            updated = torch.tanh(self.integration(torch.cat([h, aggregated], dim=-1)))
            return updated

else:  # pragma: no cover — torch not available

    class CommNetwork:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("CommNetwork requires PyTorch.")

    class AttentionComm:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("AttentionComm requires PyTorch.")
