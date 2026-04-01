"""
Attention Mechanisms for Social Navigation
==========================================

Attention-based neural network modules for reasoning about social
interactions in pedestrian navigation environments.

Classes
-------
SocialAttention
    Score-based attention over nearby agents (Chen et al., SARL).
MultiHeadSocialAttention
    Multi-head variant of SocialAttention.
TransformerEncoderLayer
    Standard transformer encoder layer with self-attention and FFN.
TransformerEncoder
    Stack of TransformerEncoderLayers.
GraphAttentionLayer
    Single graph attention layer (Velickovic et al., GAT).
GraphAttentionNetwork
    Stack of GraphAttentionLayers.
CrossAttention
    Cross-attention between two sets of features.
RelationalReasoning
    Pairwise relational reasoning module.
TemporalAttention
    Attention over temporal sequence of observations.
SpatialTransformer
    Transformer operating on spatial positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =====================================================================
# SocialAttention (Chen et al., SARL)
# =====================================================================


class SocialAttention(nn.Module):
    """Score-based attention over nearby agents (Chen et al., SARL).

    Computes an importance score for each nearby agent relative to the
    robot state and returns a weighted sum of agent embeddings.

    Parameters
    ----------
    input_dim : int
        Per-agent feature dimensionality (applied to both robot and
        human state inputs after embedding).
    hidden_dim : int
        Hidden dimensionality of the attention scoring network.
    output_dim : int
        Dimensionality of the attended output feature.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 128,
        output_dim: int = 128,
        *,
        embed_dim: int | None = None,
        num_heads: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim is not None and input_dim is None:
            input_dim = embed_dim
        if input_dim is None:
            raise ValueError("input_dim or embed_dim must be provided")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Embed robot and human states into a shared space
        self.robot_embed = nn.Linear(input_dim, hidden_dim)
        self.human_embed = nn.Linear(input_dim, hidden_dim)

        # Attention scoring network: takes concatenated robot-human embedding
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Output projection applied to the weighted sum
        self.output_proj = nn.Linear(input_dim, output_dim)

        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        robot_state: Tensor,
        human_states: Tensor | None = None,
    ) -> Tensor:
        """Compute attended social feature.

        Parameters
        ----------
        robot_state : Tensor ``(B, input_dim)``
            Robot state features.
        human_states : Tensor ``(B, N, input_dim)``
            Nearby-agent state features where *N* is the number of agents.

        Returns
        -------
        Tensor ``(B, output_dim)``
            Attention-weighted social feature.
        """
        if human_states is None:
            human_states = robot_state
            robot_state = human_states.mean(dim=1)

        B, N, _ = human_states.shape

        # Embed both
        robot_emb = F.relu(self.robot_embed(robot_state))  # (B, hidden_dim)
        human_emb = F.relu(self.human_embed(human_states))  # (B, N, hidden_dim)

        # Expand robot embedding to match each human
        robot_emb_exp = robot_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        # Concatenate and compute scores
        combined = torch.cat([robot_emb_exp, human_emb], dim=-1)  # (B, N, 2*hidden_dim)
        scores = self.attention_net(combined).squeeze(-1)  # (B, N)
        weights = F.softmax(scores, dim=-1)  # (B, N)

        # Weighted sum of original human states
        attended = torch.bmm(weights.unsqueeze(1), human_states).squeeze(1)  # (B, input_dim)

        return self.output_proj(attended)  # (B, output_dim)


# =====================================================================
# MultiHeadSocialAttention
# =====================================================================


class MultiHeadSocialAttention(nn.Module):
    """Multi-head variant of SocialAttention.

    Each head independently computes attention scores over nearby agents.
    The attended features from all heads are concatenated and projected.

    Parameters
    ----------
    input_dim : int
        Per-agent feature dimensionality.
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dimensionality for each head's scoring network.
    output_dim : int
        Dimensionality of the final output feature.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        head_dim = hidden_dim // num_heads
        assert head_dim * num_heads == hidden_dim, (
            "hidden_dim must be divisible by num_heads"
        )
        self.head_dim = head_dim

        # Per-head projections
        self.robot_projs = nn.ModuleList(
            [nn.Linear(input_dim, head_dim) for _ in range(num_heads)]
        )
        self.human_projs = nn.ModuleList(
            [nn.Linear(input_dim, head_dim) for _ in range(num_heads)]
        )
        self.score_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_dim * 2, head_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(head_dim, 1),
                )
                for _ in range(num_heads)
            ]
        )

        # Project concatenated multi-head output
        self.output_proj = nn.Linear(input_dim * num_heads, output_dim)

        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        robot_state: Tensor,
        human_states: Tensor,
    ) -> Tensor:
        """Compute multi-head attended social feature.

        Parameters
        ----------
        robot_state : Tensor ``(B, input_dim)``
        human_states : Tensor ``(B, N, input_dim)``

        Returns
        -------
        Tensor ``(B, output_dim)``
        """
        B, N, _ = human_states.shape
        head_outputs = []

        for i in range(self.num_heads):
            r_emb = F.relu(self.robot_projs[i](robot_state))  # (B, head_dim)
            h_emb = F.relu(self.human_projs[i](human_states))  # (B, N, head_dim)

            r_exp = r_emb.unsqueeze(1).expand(-1, N, -1)
            combined = torch.cat([r_exp, h_emb], dim=-1)  # (B, N, 2*head_dim)
            scores = self.score_nets[i](combined).squeeze(-1)  # (B, N)
            weights = F.softmax(scores, dim=-1)  # (B, N)

            # Weighted sum over original human features for this head
            attended = torch.bmm(weights.unsqueeze(1), human_states).squeeze(1)  # (B, input_dim)
            head_outputs.append(attended)

        # Concatenate heads and project
        multi_head = torch.cat(head_outputs, dim=-1)  # (B, input_dim * num_heads)
        return self.output_proj(multi_head)  # (B, output_dim)


# =====================================================================
# TransformerEncoderLayer
# =====================================================================


class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer with self-attention and FFN.

    Implements pre-norm (LayerNorm before attention/FFN) for stable
    training, following modern transformer conventions.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        Hidden dimensionality of the position-wise FFN.
    dropout : float
        Dropout probability.
    activation : str
        Activation function in the FFN (``"relu"`` or ``"gelu"``).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Position-wise FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)

        self._feature_dim = d_model

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        src : Tensor ``(B, S, d_model)``
            Source sequence.
        src_mask : Tensor, optional
            Attention mask ``(S, S)``.
        src_key_padding_mask : Tensor, optional
            Padding mask ``(B, S)``.

        Returns
        -------
        Tensor ``(B, S, d_model)``
        """
        # Pre-norm self-attention
        x = self.norm1(src)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(attn_out)

        # Pre-norm FFN
        x = self.norm2(src)
        ffn_out = self.linear2(self.dropout_ffn(self.activation(self.linear1(x))))
        src = src + self.dropout2(ffn_out)

        return src


# =====================================================================
# TransformerEncoder
# =====================================================================


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderLayers.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    nhead : int
        Number of attention heads per layer.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        FFN hidden dimensionality.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self._feature_dim = d_model

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through all encoder layers.

        Parameters
        ----------
        src : Tensor ``(B, S, d_model)``
        src_mask : Tensor, optional
        src_key_padding_mask : Tensor, optional

        Returns
        -------
        Tensor ``(B, S, d_model)``
        """
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return self.final_norm(output)


# =====================================================================
# GraphAttentionLayer (Velickovic et al., GAT)
# =====================================================================


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer (Velickovic et al., 2018).

    Computes attention coefficients between all pairs of nodes using a
    shared attention mechanism, then aggregates neighbour features via
    the learned coefficients.

    Parameters
    ----------
    in_features : int
        Input feature dimensionality per node.
    out_features : int
        Output feature dimensionality per node (per head).
    num_heads : int
        Number of independent attention heads.  Outputs are concatenated,
        yielding ``out_features * num_heads`` total dimensions.
    dropout : float
        Dropout on attention coefficients.
    alpha : float
        Negative slope for the LeakyReLU in the attention mechanism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Linear projection for each head
        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        # Attention parameters: a = [a_left || a_right] per head
        self.a_left = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_right = nn.Parameter(torch.empty(num_heads, out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout = nn.Dropout(dropout)

        self._feature_dim = out_features * num_heads

        self._reset_parameters()

    # ------------------------------------------------------------------
    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``(B, N, in_features)``
            Node features.
        adj : Tensor, optional ``(B, N, N)`` or ``(N, N)``
            Adjacency matrix.  If ``None``, a fully-connected graph is
            assumed.

        Returns
        -------
        Tensor ``(B, N, out_features * num_heads)``
        """
        B, N, _ = x.shape

        # Project: (B, N, in_features) @ (H, in_features, out_features) -> (B, H, N, out_features)
        # Use einsum for clarity
        h = torch.einsum("bni,hio->bhno", x, self.W)  # (B, H, N, out_features)

        # Attention coefficients
        # e_ij = LeakyReLU(a_left^T h_i + a_right^T h_j)
        attn_left = torch.einsum("bhno,hok->bhnk", h, self.a_left).squeeze(-1)   # (B, H, N)
        attn_right = torch.einsum("bhno,hok->bhnk", h, self.a_right).squeeze(-1)  # (B, H, N)

        # Broadcast to (B, H, N, N): e_ij = a_left_i + a_right_j
        e = attn_left.unsqueeze(-1) + attn_right.unsqueeze(-2)  # (B, H, N, N)
        e = self.leaky_relu(e)

        # Mask non-edges
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            elif adj.dim() == 3:
                adj = adj.unsqueeze(1)  # (B, 1, N, N)
            e = e.masked_fill(adj == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)  # (B, H, N, N)
        alpha = self.dropout(alpha)

        # Aggregate
        out = torch.einsum("bhnm,bhmo->bhno", alpha, h)  # (B, H, N, out_features)

        # Concatenate heads: (B, N, H * out_features)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)

        return out


# =====================================================================
# GraphAttentionNetwork
# =====================================================================


class GraphAttentionNetwork(nn.Module):
    """Stack of GraphAttentionLayers.

    Parameters
    ----------
    input_dim : int
        Input node feature dimensionality.
    hidden_dim : int
        Hidden feature dimensionality per head in intermediate layers.
    output_dim : int
        Output feature dimensionality (total, across heads in final layer).
    num_heads : int
        Number of attention heads per layer.
    num_layers : int
        Number of GAT layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers = []
        in_dim = input_dim

        for _i in range(num_layers - 1):
            layer = GraphAttentionLayer(
                in_features=in_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            layers.append(layer)
            in_dim = hidden_dim * num_heads

        # Final layer: single head to get output_dim
        layers.append(
            GraphAttentionLayer(
                in_features=in_dim,
                out_features=output_dim,
                num_heads=1,
                dropout=dropout,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through all GAT layers.

        Parameters
        ----------
        x : Tensor ``(B, N, input_dim)``
            Node features.
        adj : Tensor, optional ``(B, N, N)`` or ``(N, N)``
            Adjacency matrix.

        Returns
        -------
        Tensor ``(B, N, output_dim)``
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj=adj)
            if i < len(self.layers) - 1:
                h = self.elu(h)
                h = self.dropout(h)
        return h


# =====================================================================
# CrossAttention
# =====================================================================


class CrossAttention(nn.Module):
    """Cross-attention between two sets of features.

    Queries come from one feature set, keys and values from another.
    Uses multi-head scaled dot-product attention.

    Parameters
    ----------
    query_dim : int
        Dimensionality of query features.
    key_dim : int
        Dimensionality of key/value features.
    num_heads : int
        Number of attention heads.
    output_dim : int
        Dimensionality of the output projection.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 4,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        # Project queries, keys, values to a common dimension
        # Use output_dim as the internal attention dimension
        assert output_dim % num_heads == 0, (
            "output_dim must be divisible by num_heads"
        )
        self.head_dim = output_dim // num_heads

        self.q_proj = nn.Linear(query_dim, output_dim)
        self.k_proj = nn.Linear(key_dim, output_dim)
        self.v_proj = nn.Linear(key_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(key_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute cross-attention.

        Parameters
        ----------
        query : Tensor ``(B, Sq, query_dim)``
            Query features.
        key_value : Tensor ``(B, Skv, key_dim)``
            Key/value features.
        mask : Tensor, optional ``(B, Sq, Skv)``
            Attention mask (``True`` values are masked out).

        Returns
        -------
        Tensor ``(B, Sq, output_dim)``
        """
        B, Sq, _ = query.shape
        Skv = key_value.shape[1]

        q = self.q_proj(self.norm_q(query))    # (B, Sq, output_dim)
        k = self.k_proj(self.norm_kv(key_value))  # (B, Skv, output_dim)
        v = self.v_proj(self.norm_kv(key_value))  # (B, Skv, output_dim)

        # Reshape for multi-head attention
        q = q.view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, Sq, hd)
        k = k.view(B, Skv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Skv, hd)
        v = v.view(B, Skv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Skv, hd)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, Sq, Skv)

        if mask is not None:
            # Expand mask for heads: (B, 1, Sq, Skv)
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, Sq, hd)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, Sq, self.output_dim)  # (B, Sq, output_dim)
        out = self.out_proj(out)
        out = self.norm_out(out)

        return out


# =====================================================================
# RelationalReasoning
# =====================================================================


class RelationalReasoning(nn.Module):
    """Pairwise relational reasoning module.

    Constructs all pairwise combinations of input entities, processes
    each pair through a shared MLP, and aggregates the result.

    Parameters
    ----------
    input_dim : int
        Feature dimensionality per entity.
    hidden_dim : int
        Hidden dimensionality of the relational MLP.
    output_dim : int
        Output dimensionality after aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Relation network: processes concatenated pairs
        self.relation_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Aggregation network: processes the summed relational features
        self.aggregate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self._feature_dim = output_dim

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(self, entities: Tensor) -> Tensor:
        """Compute relational reasoning over entities.

        Parameters
        ----------
        entities : Tensor ``(B, N, input_dim)``
            Entity features.

        Returns
        -------
        Tensor ``(B, output_dim)``
            Aggregated relational feature.
        """
        B, N, D = entities.shape

        # Build all ordered pairs (i, j) for i != j
        # Expand to (B, N, N, D) and gather pairs
        e_i = entities.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        e_j = entities.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)

        pairs = torch.cat([e_i, e_j], dim=-1)  # (B, N, N, 2D)
        pairs = pairs.view(B, N * N, 2 * D)    # (B, N*N, 2D)

        # Process each pair
        relations = self.relation_net(pairs)  # (B, N*N, hidden_dim)

        # Aggregate by summing over all pairs
        aggregated = relations.sum(dim=1)  # (B, hidden_dim)

        return self.aggregate_net(aggregated)  # (B, output_dim)


# =====================================================================
# TemporalAttention
# =====================================================================


class TemporalAttention(nn.Module):
    """Attention over a temporal sequence of observations.

    Applies multi-head self-attention with sinusoidal positional
    encoding to capture temporal dependencies across time steps.

    Parameters
    ----------
    input_dim : int
        Feature dimensionality per time step.
    num_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum supported sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim),
        )

        # Sinusoidal positional encoding (pre-computed buffer)
        pe = self._build_positional_encoding(max_seq_len, input_dim)
        self.register_buffer("pe", pe)

        self._feature_dim = input_dim

    # ------------------------------------------------------------------
    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> Tensor:
        """Build sinusoidal positional encoding table."""
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Attend over the temporal dimension.

        Parameters
        ----------
        x : Tensor ``(B, T, input_dim)``
            Temporal sequence of features.
        mask : Tensor, optional ``(B, T)``
            Key padding mask (``True`` for padded positions).

        Returns
        -------
        Tensor ``(B, T, input_dim)``
            Temporally attended features.
        """
        T = x.shape[1]

        # Add positional encoding
        x = x + self.pe[:, :T, :]

        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            key_padding_mask=mask,
        )
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


# =====================================================================
# SpatialTransformer
# =====================================================================


class SpatialTransformer(nn.Module):
    """Transformer operating on spatial positions.

    Projects per-agent spatial features into a model space, adds
    learned spatial embeddings, and applies transformer encoder layers.

    Parameters
    ----------
    input_dim : int
        Input feature dimensionality per spatial entity.
    d_model : int
        Internal model dimensionality.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Project input features into model space
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Learned spatial embedding (up to a reasonable maximum)
        self.max_entities = 256
        self.spatial_embedding = nn.Embedding(self.max_entities, d_model)

        # Transformer encoder stack
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )

        self._feature_dim = d_model

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Dimensionality of the final output vector."""
        return self._feature_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Process spatial entities through the transformer.

        Parameters
        ----------
        x : Tensor ``(B, N, input_dim)``
            Per-entity spatial features.
        mask : Tensor, optional ``(B, N)``
            Key padding mask (``True`` for invalid entities).

        Returns
        -------
        Tensor ``(B, N, d_model)``
            Transformed spatial features.
        """
        B, N, _ = x.shape

        # Project and normalize
        h = self.input_norm(F.relu(self.input_proj(x)))  # (B, N, d_model)

        # Add learned spatial positional embedding
        positions = torch.arange(N, device=x.device)
        h = h + self.spatial_embedding(positions).unsqueeze(0)  # broadcast over B

        # Apply transformer encoder
        h = self.encoder(h, src_key_padding_mask=mask)

        return h
