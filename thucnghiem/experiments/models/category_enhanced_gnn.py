from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.utils import scatter, softmax


class CategoryEnhancedGNN(nn.Module):
    """
    Module 1 of CatSA: Category-Enhanced Session Graph.

    The model receives one batched HeteroData object and produces next-item logits
    of shape [batch_size, num_items].
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_leaf_cats: int,
        num_parent_cats: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: Literal["sage", "gat"] = "sage",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        if conv_type == "gat" and hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads when conv_type='gat'")

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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.item_embedding.reset_parameters()
        self.leaf_embedding.reset_parameters()
        self.parent_embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
        self.attn_query.reset_parameters()
        self.attn_key.reset_parameters()
        self.attn_score.reset_parameters()
        self.session_proj.reset_parameters()

    def forward(self, data: HeteroData) -> Tensor:
        """
        Args:
            data:
                Batched hetero graph.

        Returns:
            logits: [batch_size, num_items]
        """
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
                updated_states = conv_x_dict.get(node_type)
                if updated_states is None:
                    x_dict[node_type] = norm_dict[node_type](prev_states)
                    continue
                updated_states = F.dropout(F.relu(updated_states), p=self.dropout, training=self.training)
                x_dict[node_type] = norm_dict[node_type](prev_states + updated_states)

        item_hidden = x_dict["item"]
        session_repr = self.readout(
            item_hidden=item_hidden,
            sequence_index=data["item"].sequence_index,
            sequence_len=data["item"].sequence_len,
            last_index=data["item"].last_index,
        )
        logits = session_repr @ self.item_embedding.weight.t()
        return logits

    def readout(
        self,
        *,
        item_hidden: Tensor,
        sequence_index: Tensor,
        sequence_len: Tensor,
        last_index: Tensor,
    ) -> Tensor:
        """
        SR-GNN style soft-attention readout.

        Shapes:
        - item_hidden: [num_item_nodes_in_batch, hidden_dim]
        - sequence_index: [sum_session_lengths]
        - sequence_len: [batch_size]
        - last_index: [batch_size]
        - returns: [batch_size, hidden_dim]
        """
        batch_size = int(sequence_len.numel())
        device = item_hidden.device

        # [batch_size, hidden_dim]
        local_interest = item_hidden[last_index]

        # [sum_session_lengths, hidden_dim]
        sequence_hidden = item_hidden[sequence_index]
        sequence_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            sequence_len.to(device=device),
        )

        # Query from the last clicked item of each session is broadcast to all positions.
        # q_last: [sum_session_lengths, hidden_dim]
        # q_seq:  [sum_session_lengths, hidden_dim]
        q_last = self.attn_query(local_interest)[sequence_batch]
        q_seq = self.attn_key(sequence_hidden)

        # Unnormalized attention logits per position: [sum_session_lengths, 1]
        attention_logits = self.attn_score(torch.sigmoid(q_last + q_seq))
        attention = softmax(attention_logits, sequence_batch, dim=0)

        # Weighted session summary after attention pooling: [batch_size, hidden_dim]
        global_preference = scatter(
            attention * sequence_hidden,
            sequence_batch,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )

        session_repr = self.session_proj(torch.cat([global_preference, local_interest], dim=-1))
        return session_repr

    def _build_hetero_layer(
        self,
        hidden_dim: int,
        conv_type: Literal["sage", "gat"],
        gat_heads: int,
    ) -> HeteroConv:
        if conv_type == "sage":
            def build_conv() -> SAGEConv:
                return SAGEConv((-1, -1), hidden_dim)
        else:
            out_channels = hidden_dim // gat_heads

            def build_conv() -> GATConv:
                return GATConv(
                    (-1, -1),
                    out_channels,
                    heads=gat_heads,
                    add_self_loops=False,
                )

        relation_convs = {
            ("item", "sequential", "item"): build_conv(),
            ("item", "rev_sequential", "item"): build_conv(),
            ("item", "belongs_to", "leaf_cat"): build_conv(),
            ("leaf_cat", "contains", "item"): build_conv(),
            ("leaf_cat", "child_of", "parent_cat"): build_conv(),
            ("parent_cat", "parent_of", "leaf_cat"): build_conv(),
        }
        return HeteroConv(relation_convs, aggr="sum")