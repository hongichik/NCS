"""Module 1 encoder aligned with CatSA build guide (May 2026).

- HeteroConv with per-relation GCNConv (guide pseudocode)
- Leaf and parent nodes share one category embedding table
- SR-GNN soft-attention readout (tanh + softmax weighted sum)
"""

from __future__ import annotations

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GraphConv, HeteroConv
from torch_geometric.utils import scatter, softmax


class CategoryEnhancedGNN(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        num_leaf_cats: int,
        num_parent_cats: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: Literal["gcn"] = "gcn",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_leaf_cats = max(num_leaf_cats, 1)
        self.num_parent_cats = max(num_parent_cats, 1)

        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        # Guide: parent nodes reuse the category embedding table.
        self.cat_embedding = nn.Embedding(
            self.num_leaf_cats + self.num_parent_cats,
            hidden_dim,
        )

        self.convs = nn.ModuleList(
            [self._build_hetero_layer(hidden_dim, conv_type) for _ in range(num_layers)]
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

        # SR-GNN readout: alpha_j = softmax(q^T tanh(W1 h_j + W2 h_n))
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q = nn.Linear(hidden_dim, 1, bias=False)

    def _cat_features(self, data: HeteroData) -> dict[str, Tensor]:
        leaf_ids = data["leaf_cat"].node_id
        parent_ids = data["parent_cat"].node_id + self.num_leaf_cats
        return {
            "leaf_cat": self.cat_embedding(leaf_ids),
            "parent_cat": self.cat_embedding(parent_ids),
        }

    def encode_session(self, data: HeteroData) -> Tensor:
        cat_feats = self._cat_features(data)
        x_dict = {
            "item": self.item_embedding(data["item"].node_id),
            "leaf_cat": cat_feats["leaf_cat"],
            "parent_cat": cat_feats["parent_cat"],
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
        h_last = item_hidden[last_index]
        sequence_hidden = item_hidden[sequence_index]
        sequence_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device), sequence_len.to(device=device)
        )
        h_last_expanded = h_last[sequence_batch]
        alpha_logits = self.q(torch.tanh(self.W1(sequence_hidden) + self.W2(h_last_expanded)))
        alpha = softmax(alpha_logits, sequence_batch, dim=0)
        return scatter(
            alpha * sequence_hidden,
            sequence_batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )

    def _build_hetero_layer(
        self, hidden_dim: int, conv_type: Literal["gcn"]
    ) -> HeteroConv:
        def build_item_conv() -> GCNConv:
            return GCNConv(-1, hidden_dim, add_self_loops=True)

        def build_cross_conv() -> GraphConv:
            return GraphConv((-1, -1), hidden_dim)

        relation_convs = {
            ("item", "sequential", "item"): build_item_conv(),
            ("item", "rev_sequential", "item"): build_item_conv(),
            ("item", "belongs_to", "leaf_cat"): build_cross_conv(),
            ("leaf_cat", "contains", "item"): build_cross_conv(),
            ("leaf_cat", "child_of", "parent_cat"): build_cross_conv(),
            ("parent_cat", "parent_of", "leaf_cat"): build_cross_conv(),
        }
        return HeteroConv(relation_convs, aggr="sum")
