from __future__ import annotations

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.utils import scatter, softmax


class CategoryEnhancedGNN(nn.Module):
    """Module 1: Category-Enhanced Session Graph (RGCN-style heterogeneous encoder)."""

    def __init__(
        self,
        *,
        num_items: int,
        num_leaf_cats: int,
        num_parent_cats: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: Literal["sage", "gat"] = "sage",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        if conv_type == "gat" and hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.leaf_embedding = nn.Embedding(max(num_leaf_cats, 1), hidden_dim)
        self.parent_embedding = nn.Embedding(max(num_parent_cats, 1), hidden_dim)

        self.convs = nn.ModuleList(
            [self._build_hetero_layer(hidden_dim, conv_type, gat_heads) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "item": nn.LayerNorm(hidden_dim),
                        "leaf_cat": nn.LayerNorm(hidden_dim),
                        "parent_cat": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.session_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode_session(self, data: HeteroData) -> Tensor:
        """Return session embedding z_s for one batched HeteroData."""
        x_dict = {
            "item": self.item_embedding(data["item"].node_id),
            "leaf_cat": self.leaf_embedding(data["leaf_cat"].node_id),
            "parent_cat": self.parent_embedding(data["parent_cat"].node_id),
        }
        for conv, norm_dict in zip(self.convs, self.norms):
            prev_x_dict = x_dict
            conv_x_dict = conv(prev_x_dict, data.edge_index_dict)
            x_dict = {}
            for node_type, prev_states in prev_x_dict.items():
                updated = conv_x_dict.get(node_type)
                if updated is None:
                    x_dict[node_type] = norm_dict[node_type](prev_states)
                else:
                    updated = F.dropout(F.relu(updated), p=self.dropout, training=self.training)
                    x_dict[node_type] = norm_dict[node_type](prev_states + updated)

        item_hidden = x_dict["item"]
        last_index = data["item"].last_index
        sequence_index = data["item"].sequence_index
        if hasattr(data["item"], "ptr") and data["item"].ptr is not None:
            ptr = data["item"].ptr[:-1]
            last_index = last_index + ptr
            seq_ptr = torch.repeat_interleave(ptr, data["item"].sequence_len)
            sequence_index = sequence_index + seq_ptr

        return self.readout(
            item_hidden=item_hidden,
            sequence_index=sequence_index,
            sequence_len=data["item"].sequence_len,
            last_index=last_index,
        )

    def forward(self, data: HeteroData) -> Tensor:
        session_repr = self.encode_session(data)
        return session_repr @ self.item_embedding.weight.t()

    def readout(
        self,
        *,
        item_hidden: Tensor,
        sequence_index: Tensor,
        sequence_len: Tensor,
        last_index: Tensor,
    ) -> Tensor:
        batch_size = int(sequence_len.numel())
        device = item_hidden.device
        local_interest = item_hidden[last_index]
        sequence_hidden = item_hidden[sequence_index]
        sequence_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), sequence_len.to(device=device)
        )
        q_last = self.attn_query(local_interest)[sequence_batch]
        q_seq = self.attn_key(sequence_hidden)
        attention_logits = self.attn_score(torch.sigmoid(q_last + q_seq))
        attention = softmax(attention_logits, sequence_batch, dim=0)
        global_preference = scatter(
            attention * sequence_hidden, sequence_batch, dim=0, dim_size=batch_size, reduce="sum"
        )
        return self.session_proj(torch.cat([global_preference, local_interest], dim=-1))

    def _build_hetero_layer(
        self, hidden_dim: int, conv_type: Literal["sage", "gat"], gat_heads: int
    ) -> HeteroConv:
        if conv_type == "sage":
            def build_conv() -> SAGEConv:
                return SAGEConv((-1, -1), hidden_dim)
        else:
            out_channels = hidden_dim // gat_heads

            def build_conv() -> GATConv:
                return GATConv((-1, -1), out_channels, heads=gat_heads, add_self_loops=False)

        relation_convs = {
            ("item", "sequential", "item"): build_conv(),
            ("item", "rev_sequential", "item"): build_conv(),
            ("item", "belongs_to", "leaf_cat"): build_conv(),
            ("leaf_cat", "contains", "item"): build_conv(),
            ("leaf_cat", "child_of", "parent_cat"): build_conv(),
            ("parent_cat", "parent_of", "leaf_cat"): build_conv(),
        }
        return HeteroConv(relation_convs, aggr="sum")
