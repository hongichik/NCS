from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class EncodedTaxonomy:
    item_encoder: dict[int, int]
    leaf_encoder: dict[int, int]
    parent_encoder: dict[int, int]
    item2leaf_idx: dict[int, int]
    leaf2parent_idx: dict[int, int]

    @property
    def num_items(self) -> int:
        return len(self.item_encoder)

    @property
    def num_leaf_cats(self) -> int:
        return len(self.leaf_encoder)

    @property
    def num_parent_cats(self) -> int:
        return len(self.parent_encoder)


def build_prefix_target_pairs(session_sequences: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    prefixes: list[list[int]] = []
    targets: list[int] = []
    for session in session_sequences:
        if len(session) < 2:
            continue
        for split_index in range(1, len(session)):
            prefixes.append(list(session[:split_index]))
            targets.append(int(session[split_index]))
    return prefixes, targets


def fit_taxonomy_encoders(
    session_sequences: Sequence[Sequence[int]],
    item2leaf_dict: Mapping[int, int],
    leaf2parent_dict: Mapping[int, int],
) -> EncodedTaxonomy:
    item_ids = sorted({int(item_id) for session in session_sequences for item_id in session})
    item_encoder = {item_id: idx for idx, item_id in enumerate(item_ids)}

    leaf_ids = sorted(
        {
            int(leaf_id)
            for raw_item_id in item_ids
            if raw_item_id in item2leaf_dict
            for leaf_id in [item2leaf_dict[raw_item_id]]
        }
    )
    leaf_encoder = {leaf_id: idx for idx, leaf_id in enumerate(leaf_ids)}

    parent_ids = sorted(
        {
            int(parent_id)
            for leaf_id in leaf_ids
            if leaf_id in leaf2parent_dict
            for parent_id in [leaf2parent_dict[leaf_id]]
        }
    )
    parent_encoder = {parent_id: idx for idx, parent_id in enumerate(parent_ids)}

    item2leaf_idx: dict[int, int] = {}
    for raw_item_id, raw_leaf_id in item2leaf_dict.items():
        if raw_item_id in item_encoder and raw_leaf_id in leaf_encoder:
            item2leaf_idx[item_encoder[raw_item_id]] = leaf_encoder[raw_leaf_id]

    leaf2parent_idx: dict[int, int] = {}
    for raw_leaf_id, raw_parent_id in leaf2parent_dict.items():
        if raw_leaf_id in leaf_encoder and raw_parent_id in parent_encoder:
            leaf2parent_idx[leaf_encoder[raw_leaf_id]] = parent_encoder[raw_parent_id]

    return EncodedTaxonomy(
        item_encoder=item_encoder,
        leaf_encoder=leaf_encoder,
        parent_encoder=parent_encoder,
        item2leaf_idx=item2leaf_idx,
        leaf2parent_idx=leaf2parent_idx,
    )


class CategorySessionGraphDataset(Dataset[HeteroData]):
    def __init__(
        self,
        session_sequences: Sequence[Sequence[int]],
        item2leaf_dict: Mapping[int, int],
        leaf2parent_dict: Mapping[int, int],
        targets: Sequence[int] | None = None,
        taxonomy: EncodedTaxonomy | None = None,
        raw_sessions: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.session_sequences = [list(map(int, s)) for s in session_sequences if len(s) >= 1]
        self.raw_sessions = list(raw_sessions) if raw_sessions is not None else self.session_sequences
        self.targets = list(map(int, targets)) if targets is not None else None
        if self.targets is not None and len(self.targets) != len(self.session_sequences):
            raise ValueError("targets must match session_sequences length")

        self.taxonomy = taxonomy or fit_taxonomy_encoders(
            self.session_sequences, item2leaf_dict, leaf2parent_dict
        )
        self._graphs = [
            self._build_graph(seq, None if self.targets is None else self.targets[i])
            for i, seq in enumerate(self.session_sequences)
        ]

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index: int) -> HeteroData:
        graph = self._graphs[index].clone()
        graph.raw_session = torch.tensor(self.raw_sessions[index], dtype=torch.long)
        return graph

    def _build_graph(self, raw_session: Sequence[int], target: int | None) -> HeteroData:
        encoded_session = [
            self.taxonomy.item_encoder[item_id]
            for item_id in raw_session
            if item_id in self.taxonomy.item_encoder
        ]
        if not encoded_session:
            raise ValueError("Session has no known items")

        item_global_ids: list[int] = []
        item_local_index: dict[int, int] = {}
        sequence_local_indices: list[int] = []
        for item_id in encoded_session:
            if item_id not in item_local_index:
                item_local_index[item_id] = len(item_global_ids)
                item_global_ids.append(item_id)
            sequence_local_indices.append(item_local_index[item_id])

        sequential_src = sequence_local_indices[:-1]
        sequential_dst = sequence_local_indices[1:]
        reverse_sequential_src = sequence_local_indices[1:]
        reverse_sequential_dst = sequence_local_indices[:-1]

        leaf_global_ids: list[int] = []
        leaf_local_index: dict[int, int] = {}
        parent_global_ids: list[int] = []
        parent_local_index: dict[int, int] = {}
        item_to_leaf_edges: list[tuple[int, int]] = []
        leaf_to_parent_edges: list[tuple[int, int]] = []

        for item_local_id, item_global_id in enumerate(item_global_ids):
            leaf_global_id = self.taxonomy.item2leaf_idx.get(item_global_id)
            if leaf_global_id is None:
                continue
            if leaf_global_id not in leaf_local_index:
                leaf_local_index[leaf_global_id] = len(leaf_global_ids)
                leaf_global_ids.append(leaf_global_id)
            leaf_local_id = leaf_local_index[leaf_global_id]
            item_to_leaf_edges.append((item_local_id, leaf_local_id))

        for leaf_global_id, leaf_local_id in leaf_local_index.items():
            parent_global_id = self.taxonomy.leaf2parent_idx.get(leaf_global_id)
            if parent_global_id is None:
                continue
            if parent_global_id not in parent_local_index:
                parent_local_index[parent_global_id] = len(parent_global_ids)
                parent_global_ids.append(parent_global_id)
            parent_local_id = parent_local_index[parent_global_id]
            leaf_to_parent_edges.append((leaf_local_id, parent_local_id))

        data = HeteroData()
        data["item"].node_id = torch.tensor(item_global_ids, dtype=torch.long)
        data["item"].num_nodes = len(item_global_ids)
        data["item"].sequence_index = torch.tensor(sequence_local_indices, dtype=torch.long)
        data["item"].sequence_len = torch.tensor([len(sequence_local_indices)], dtype=torch.long)
        data["item"].last_index = torch.tensor([sequence_local_indices[-1]], dtype=torch.long)

        data["leaf_cat"].node_id = torch.tensor(leaf_global_ids, dtype=torch.long)
        data["leaf_cat"].num_nodes = len(leaf_global_ids)
        data["parent_cat"].node_id = torch.tensor(parent_global_ids, dtype=torch.long)
        data["parent_cat"].num_nodes = len(parent_global_ids)

        data[("item", "sequential", "item")].edge_index = _to_edge_index(sequential_src, sequential_dst)
        data[("item", "rev_sequential", "item")].edge_index = _to_edge_index(
            reverse_sequential_src, reverse_sequential_dst
        )
        item_leaf_src = [s for s, _ in item_to_leaf_edges]
        item_leaf_dst = [d for _, d in item_to_leaf_edges]
        data[("item", "belongs_to", "leaf_cat")].edge_index = _to_edge_index(item_leaf_src, item_leaf_dst)
        data[("leaf_cat", "contains", "item")].edge_index = _to_edge_index(item_leaf_dst, item_leaf_src)
        leaf_parent_src = [s for s, _ in leaf_to_parent_edges]
        leaf_parent_dst = [d for _, d in leaf_to_parent_edges]
        data[("leaf_cat", "child_of", "parent_cat")].edge_index = _to_edge_index(leaf_parent_src, leaf_parent_dst)
        data[("parent_cat", "parent_of", "leaf_cat")].edge_index = _to_edge_index(leaf_parent_dst, leaf_parent_src)

        if target is not None:
            encoded_target = self.taxonomy.item_encoder.get(int(target))
            if encoded_target is None:
                raise ValueError(f"Unknown target {target}")
            data.y = torch.tensor([encoded_target], dtype=torch.long)
        return data


def _to_edge_index(source: Sequence[int], destination: Sequence[int]) -> Tensor:
    if len(source) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([source, destination], dtype=torch.long)
