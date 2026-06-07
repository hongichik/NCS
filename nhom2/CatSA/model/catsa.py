"""Full CatSA model: Module 1 + session-level InfoNCE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from model.category_enhanced_gnn import CategoryEnhancedGNN


def session_level_infonce(z_s: Tensor, z_s_tilde: Tensor, tau: float = 0.5) -> Tensor:
    z_s = F.normalize(z_s, dim=1)
    z_s_tilde = F.normalize(z_s_tilde, dim=1)
    sim = torch.mm(z_s, z_s_tilde.t()) / tau
    targets = torch.arange(z_s.size(0), device=z_s.device)
    loss_row = F.cross_entropy(sim, targets)
    loss_col = F.cross_entropy(sim.t(), targets)
    return (loss_row + loss_col) / 2


class CatSA(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        num_leaf_cats: int,
        num_parent_cats: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1,
        cl_weight: float = 0.1,
        temperature: float = 0.5,
        mode: str = "full",
    ) -> None:
        super().__init__()
        self.encoder = CategoryEnhancedGNN(
            num_items=num_items,
            num_leaf_cats=num_leaf_cats,
            num_parent_cats=num_parent_cats,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.cl_weight = cl_weight
        self.temperature = temperature
        self.mode = mode  # "a2" = Module 1 only, "full" = + CL

    def forward(
        self,
        graph_orig: HeteroData,
        graph_aug: HeteroData | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        z_s = self.encoder.encode_session(graph_orig)
        logits = z_s @ self.encoder.item_embedding.weight.t()
        loss_rec = F.cross_entropy(logits, graph_orig.y.view(-1))

        loss_cl = None
        if self.mode == "full" and graph_aug is not None:
            z_s_tilde = self.encoder.encode_session(graph_aug)
            loss_cl = session_level_infonce(z_s, z_s_tilde, tau=self.temperature)

        return logits, loss_rec, loss_cl

    def total_loss(self, loss_rec: Tensor, loss_cl: Tensor | None) -> Tensor:
        if loss_cl is None or self.mode != "full":
            return loss_rec
        return loss_rec + self.cl_weight * loss_cl
