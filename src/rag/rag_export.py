"""
src/rag/rag_export.py
------------------------
Assembles the RAG-ready output bundle: one record per cluster, combining
the Stage 6 structured profile + text narrative with the Stage 7 semantic
mapping result. This is the hand-off artifact for the (not-yet-built)
Graph-RAG + LLM report stage -- nothing here calls an LLM.

Design: every record is keyed by cluster_id and carries
representative_event_uids, which are the join keys a future retrieval
step needs to pull full event context back out of Neo4j (or wherever
the raw events are persisted) when answering a question like
"why was this cluster flagged?".
"""

import json
from dataclasses import asdict
from typing import Dict, List


def build_rag_bundle(cluster_profiles: List, cluster_mappings: List) -> List[Dict]:
    """
    cluster_profiles: list[cluster_profiler.ClusterProfile]
    cluster_mappings: list[semantic_mapper.ClusterMapping]
    (same cluster_ids, same order is NOT assumed -- joined by cluster_id)
    """
    mapping_by_cluster = {m.cluster_id: m for m in cluster_mappings}

    bundle = []
    for profile in cluster_profiles:
        mapping = mapping_by_cluster.get(profile.cluster_id)
        record = {
            "cluster_id": profile.cluster_id,
            "cluster_size": profile.cluster_size,
            "narrative": profile.text,
            "structured_profile": {
                "dominant_event_ids": profile.dominant_event_ids,
                "hostname_distribution": profile.hostname_distribution,
                "common_source_images": profile.common_source_images,
                "common_target_images": profile.common_target_images,
                "common_destination_ips": profile.common_destination_ips,
                "temporal_span": profile.temporal_span,
            },
            "evidence_snippets": profile.evidence_snippets,
            "representative_event_uids": profile.representative_event_uids,
            "mitre_candidates": (
                [t.as_dict() for t in mapping.top_k_techniques] if mapping else []
            ),
            "retrieval_hint": (
                f"To answer questions about cluster {profile.cluster_id}, retrieve "
                f"the {len(profile.representative_event_uids)} representative events "
                f"and their graph neighbors (SAME_PID / SAME_HOST_TEMPORAL edges) "
                f"from the persisted event graph using representative_event_uids."
            ),
        }
        bundle.append(record)
    return bundle


def write_rag_bundle(bundle: List[Dict], path: str):
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)
