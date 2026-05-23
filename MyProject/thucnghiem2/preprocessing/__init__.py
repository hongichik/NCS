from __future__ import annotations

__all__ = [
    "build_sessions_from_events",
    "build_session_records_from_events",
    "load_preprocessed_artifacts",
    "load_item_to_leaf_mapping",
    "load_leaf_to_parent_mapping",
    "save_preprocessed_artifacts",
    "split_session_records_by_time",
    "CategorySessionGraphDataset",
    "EncodedTaxonomy",
    "build_prefix_target_pairs",
    "fit_taxonomy_encoders",
]


def __getattr__(name: str):
    if name in {
        "build_sessions_from_events",
        "build_session_records_from_events",
        "load_preprocessed_artifacts",
        "load_item_to_leaf_mapping",
        "load_leaf_to_parent_mapping",
        "resolve_retailrocket_file_paths",
        "save_preprocessed_artifacts",
        "split_session_records_by_time",
    }:
        from .retailrocket_preprocess import (
            build_sessions_from_events,
            build_session_records_from_events,
            load_preprocessed_artifacts,
            load_item_to_leaf_mapping,
            load_leaf_to_parent_mapping,
            resolve_retailrocket_file_paths,
            save_preprocessed_artifacts,
            split_session_records_by_time,
        )

        exports = {
            "build_sessions_from_events": build_sessions_from_events,
            "build_session_records_from_events": build_session_records_from_events,
            "load_preprocessed_artifacts": load_preprocessed_artifacts,
            "load_item_to_leaf_mapping": load_item_to_leaf_mapping,
            "load_leaf_to_parent_mapping": load_leaf_to_parent_mapping,
            "resolve_retailrocket_file_paths": resolve_retailrocket_file_paths,
            "save_preprocessed_artifacts": save_preprocessed_artifacts,
            "split_session_records_by_time": split_session_records_by_time,
        }
        return exports[name]

    if name in {
        "CategorySessionGraphDataset",
        "EncodedTaxonomy",
        "build_prefix_target_pairs",
        "fit_taxonomy_encoders",
    }:
        from .session_graph_dataset import (
            CategorySessionGraphDataset,
            EncodedTaxonomy,
            build_prefix_target_pairs,
            fit_taxonomy_encoders,
        )

        exports = {
            "CategorySessionGraphDataset": CategorySessionGraphDataset,
            "EncodedTaxonomy": EncodedTaxonomy,
            "build_prefix_target_pairs": build_prefix_target_pairs,
            "fit_taxonomy_encoders": fit_taxonomy_encoders,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")