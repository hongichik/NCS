"""Full-ranking metrics: HR@K, NDCG@K, MRR@K (guide stage 4)."""

from __future__ import annotations

import math

import torch
from torch_geometric.loader import DataLoader

from model.catsa import CatSA2


@torch.no_grad()
def evaluate_topk(
    model: CatSA2,
    loader: DataLoader,
    device: torch.device,
    k: int,
) -> tuple[float, float, float]:
    model.eval()
    hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        logits, _, _ = model(batch, None)
        target = batch.y.view(-1)
        topk = torch.topk(logits, min(k, logits.size(-1)), dim=-1).indices
        matches = topk.eq(target.unsqueeze(1))
        hit = matches.any(dim=1)
        hits += int(hit.sum().item())

        if hit.any():
            rows = hit.nonzero(as_tuple=True)[0]
            pos = matches[rows].to(torch.int64).argmax(dim=1) + 1
            mrr_sum += float((1.0 / pos.to(torch.float32)).sum().item())
            ndcg_sum += sum(1.0 / math.log2(int(p.item()) + 1) for p in pos)

        total += int(target.size(0))

    if total == 0:
        return 0.0, 0.0, 0.0
    scale = 100.0 / total
    return scale * hits, scale * ndcg_sum, scale * mrr_sum
