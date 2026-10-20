"""
src/rag/retriever.py
-----------------------
Retrieval layer for the RAG stage. Two retrieval modes, deliberately kept
separate:

1. STRUCTURED LOOKUP (primary, used by "generate a report for cluster 3"):
   cluster_id is already a precise key into three existing, disk-backed
   outputs (cluster_profiles_v2.json, semantic_mitre_mappings.json,
   novelty_scores.csv + embedding_clusters.csv). This is a join on an
   exact key, not a search problem -- running it through an embedding
   index would add latency and a new failure mode (approximate retrieval)
   for a question that already has an exact answer. So: plain dict
   lookups, documented as the deliberate choice, not an oversight.

2. SEMANTIC SEARCH (secondary, optional): for a free-text query like
   "find clusters related to credential access" where the caller does
   NOT already know a cluster_id, cluster narratives are embedded (reusing
   the same TextEmbedder abstraction as src/mapping/embedders.py) and
   ranked by cosine similarity against the query. This is a real search
   problem, so embedding-based retrieval is the right tool here -- the
   two modes are not interchangeable, and this module keeps them as two
   distinct entrypoints rather than forcing one mechanism for both.

INDEXED CONTENT (what retrieval operates over):
    - cluster_profiles_v2.json  -> structured profile + text narrative
    - semantic_mitre_mappings.json -> top-K technique candidates per cluster
    - novelty_scores.csv + embedding_clusters.csv -> per-event novelty,
      joined to full cluster membership (not just the 1-3 representative
      events already carried in the profile)

Nothing here reads raw_event / raw log text directly -- only the already
-structured, already-reviewed outputs of earlier pipeline stages. See
README's "What to avoid" section for why.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ClusterContext:
    """Everything a report for one cluster needs, already joined."""
    cluster_id: int
    cluster_size: int
    narrative: str
    structured_profile: Dict
    evidence_snippets: List[str]
    representative_event_uids: List[str]

    techniques: List[Dict]  # from semantic_mitre_mappings.json, unchanged

    # novelty, aggregated at retrieval time from the full cluster membership
    mean_novelty_score: Optional[float]
    max_novelty_score: Optional[float]
    novelty_label_counts: Dict[str, int]
    high_novelty_events: List[Dict]  # top-N member events by novelty_score (event_uid, score, label)


class RagIndex:
    """Loads the three on-disk artifacts once and serves structured
    lookups + (optionally) semantic search over cluster narratives."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self._profiles_by_cluster: Dict[int, Dict] = {}
        self._mappings_by_cluster: Dict[int, Dict] = {}
        self._novelty_by_cluster: Dict[int, List[Dict]] = {}
        self._narrative_index = None  # lazily built, see build_semantic_index()

        self._load()

    def _load(self):
        profiles_path = os.path.join(self.output_dir, "cluster_profiles_v2.json")
        mappings_path = os.path.join(self.output_dir, "semantic_mitre_mappings.json")
        assignments_path = os.path.join(self.output_dir, "embedding_clusters.csv")
        novelty_path = os.path.join(self.output_dir, "novelty_scores.csv")

        for p, name in [(profiles_path, "cluster_profiles_v2.json"),
                         (mappings_path, "semantic_mitre_mappings.json"),
                         (assignments_path, "embedding_clusters.csv"),
                         (novelty_path, "novelty_scores.csv")]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"RagIndex requires {name} in {self.output_dir}/ -- run scripts/run_pipeline.py first."
                )

        with open(profiles_path) as f:
            for profile in json.load(f):
                self._profiles_by_cluster[profile["cluster_id"]] = profile

        with open(mappings_path) as f:
            for mapping in json.load(f):
                self._mappings_by_cluster[mapping["cluster_id"]] = mapping

        novelty_by_uid: Dict[str, Dict] = {}
        with open(novelty_path) as f:
            for row in csv.DictReader(f):
                row["novelty_score"] = float(row["novelty_score"])
                novelty_by_uid[row["event_uid"]] = row

        with open(assignments_path) as f:
            for row in csv.DictReader(f):
                cid = int(row["cluster_id"])
                if cid == -1:
                    continue  # noise is excluded from cluster-level retrieval, consistent with profiling/mapping
                nov = novelty_by_uid.get(row["event_uid"])
                if nov is not None:
                    self._novelty_by_cluster.setdefault(cid, []).append(nov)

    def available_cluster_ids(self) -> List[int]:
        return sorted(self._profiles_by_cluster.keys())

    def get_cluster_context(self, cluster_id: int, high_novelty_top_n: int = 5) -> ClusterContext:
        if cluster_id not in self._profiles_by_cluster:
            raise KeyError(
                f"cluster_id={cluster_id} not found. Available: {self.available_cluster_ids()} "
                f"(note: noise, cluster_id=-1, is never retrievable here -- see retriever.py docstring)."
            )

        profile = self._profiles_by_cluster[cluster_id]
        mapping = self._mappings_by_cluster.get(cluster_id, {"top_k_techniques": []})
        novelty_rows = self._novelty_by_cluster.get(cluster_id, [])

        scores = [r["novelty_score"] for r in novelty_rows]
        label_counts: Dict[str, int] = {}
        for r in novelty_rows:
            label_counts[r["novelty_label"]] = label_counts.get(r["novelty_label"], 0) + 1

        high_novelty = sorted(novelty_rows, key=lambda r: -r["novelty_score"])[:high_novelty_top_n]
        high_novelty_trimmed = [
            {"event_uid": r["event_uid"], "novelty_score": r["novelty_score"], "novelty_label": r["novelty_label"]}
            for r in high_novelty
        ]

        return ClusterContext(
            cluster_id=cluster_id,
            cluster_size=profile["cluster_size"],
            narrative=profile["text"],
            structured_profile={
                "dominant_event_ids": profile["dominant_event_ids"],
                "hostname_distribution": profile["hostname_distribution"],
                "common_source_images": profile["common_source_images"],
                "common_target_images": profile["common_target_images"],
                "common_destination_ips": profile["common_destination_ips"],
                "temporal_span": profile["temporal_span"],
            },
            evidence_snippets=profile["evidence_snippets"],
            representative_event_uids=profile["representative_event_uids"],
            techniques=mapping["top_k_techniques"],
            mean_novelty_score=round(float(np.mean(scores)), 4) if scores else None,
            max_novelty_score=round(float(np.max(scores)), 4) if scores else None,
            novelty_label_counts=label_counts,
            high_novelty_events=high_novelty_trimmed,
        )

    # ---------------------------------------------------------------
    # Secondary, optional: semantic search over cluster narratives
    # ---------------------------------------------------------------

    def build_semantic_index(self, embedder):
        """embedder: anything matching src.mapping.embedders.TextEmbedder.
        Call once after construction if you need find_clusters_by_query()."""
        cluster_ids = self.available_cluster_ids()
        narratives = [self._profiles_by_cluster[cid]["text"] for cid in cluster_ids]
        vectors = embedder.embed(narratives)
        self._narrative_index = {"cluster_ids": cluster_ids, "vectors": vectors, "embedder": embedder}

    def find_clusters_by_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """Free-text semantic search, e.g. 'credential access behavior'.
        Requires build_semantic_index() to have been called first."""
        if self._narrative_index is None:
            raise RuntimeError("Call build_semantic_index(embedder) before find_clusters_by_query().")

        embedder = self._narrative_index["embedder"]
        query_vec = embedder.embed([query])[0]
        sims = self._narrative_index["vectors"] @ query_vec
        ranked_idx = np.argsort(-sims)[:top_k]

        return [
            {"cluster_id": self._narrative_index["cluster_ids"][i], "similarity_score": float(sims[i])}
            for i in ranked_idx
        ]
