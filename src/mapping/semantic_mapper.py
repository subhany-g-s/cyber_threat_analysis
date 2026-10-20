"""
src/mapping/semantic_mapper.py
----------------------------------
Embedding-based MITRE ATT&CK mapping of behaviour narratives (clusters and,
separately, noise units) against the FULL cached ATT&CK corpus.

    narrative text -> prepare_text -> embedder -> cosine vs technique vectors
                   -> Top-K ranked candidates with an explicit outcome

What the numbers mean (read before interpreting a result):
  * `similarity_score` is a COSINE SIMILARITY between two text embeddings.
    It is NOT a probability, NOT calibrated, and NOT a confidence that the
    technique was executed. Scores from different embedding models are not
    comparable. Thresholds are uncalibrated defaults unless evaluated against
    labelled examples (src/mapping/evaluation.py).
  * Status per candidate:  CANDIDATE (>= candidate threshold)
                           WEAK_MATCH (>= weak threshold, below candidate)
                           NO_CONFIDENT_MATCH (below both)
  * Per narrative `overall_outcome`: CANDIDATES | WEAK_ONLY | NO_MATCH.
    A technique is never forced onto a narrative: candidates with a score of
    zero are dropped, and NO_MATCH is a first-class result.
  * No per-event rules, EventID tables or hand-written technique logic exist
    here or anywhere in the mapping path.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.mapping.embedders import TextEmbedder
from src.mapping.text_prep import prepare_text, technique_embedding_text

STATUS_CANDIDATE = "CANDIDATE"
STATUS_WEAK = "WEAK_MATCH"
STATUS_NONE = "NO_CONFIDENT_MATCH"

OUTCOME_CANDIDATES = "CANDIDATES"
OUTCOME_WEAK_ONLY = "WEAK_ONLY"
OUTCOME_NO_MATCH = "NO_MATCH"

SIMILARITY_DISCLAIMER = ("similarity_score is cosine similarity between text embeddings - not a probability "
                         "and not calibrated confidence; candidates are leads to investigate, not confirmed findings.")


@dataclass
class TechniqueCandidate:
    technique_id: str
    technique_name: str
    tactic: str
    similarity_score: float
    mapping_status: str                       # CANDIDATE | WEAK_MATCH | NO_CONFIDENT_MATCH
    # --- additive fields (defaults keep the original 5-field constructor working) ---
    tactics: List[str] = field(default_factory=list)
    rank: int = 0
    is_subtechnique: bool = False
    description_excerpt: str = ""
    url: Optional[str] = None
    supporting_event_uids: List[str] = field(default_factory=list)
    mapping_method: str = "semantic-cosine"

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class ClusterMapping:
    cluster_id: int
    cluster_text: str
    top_k_techniques: List[TechniqueCandidate]
    representative_event_uids: List[str]
    # --- additive fields ---
    unit_id: str = ""                         # "C7" for clusters, "N2"/"E3" for noise units
    scope: str = "cluster"                    # cluster | noise_group | noise_event
    overall_outcome: str = ""
    top_margin: Optional[float] = None        # score(top1) - score(top2); informational only
    mapping_text: str = ""                    # exact text that was embedded (after prepare_text)
    corpus_version: str = ""

    def as_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "cluster_text": self.cluster_text,
            "top_k_techniques": [t.as_dict() for t in self.top_k_techniques],
            "representative_event_uids": self.representative_event_uids,
            "unit_id": self.unit_id, "scope": self.scope, "overall_outcome": self.overall_outcome,
            "top_margin": self.top_margin, "mapping_text": self.mapping_text,
            "corpus_version": self.corpus_version,
        }


class TechniqueCorpus:
    """The set of techniques that are eligible for matching, with their
    L2-normalised embedding vectors (row i <-> techniques[i])."""

    def __init__(self, techniques: List[Dict], vectors: np.ndarray, meta: Optional[Dict] = None):
        if len(techniques) != vectors.shape[0]:
            raise ValueError(
                f"Technique count ({len(techniques)}) does not match embedding row count "
                f"({vectors.shape[0]}) -- the cache is stale relative to the corpus. "
                f"Re-run scripts/rebuild_technique_embeddings.py.")
        self.techniques = techniques
        self.vectors = vectors
        self.meta = meta or {}

    # ---- original constructors (kept for the 15-entry era API and its tests) ----
    @classmethod
    def build(cls, techniques: List[Dict], embedder: TextEmbedder) -> "TechniqueCorpus":
        texts = [technique_embedding_text(t) for t in techniques]
        return cls(techniques, embedder.embed(texts))

    @classmethod
    def load_cached(cls, techniques: List[Dict], vectors_path: str, ids_path: str) -> "TechniqueCorpus":
        import json
        vectors = np.load(vectors_path)
        with open(ids_path) as f:
            cached_ids = json.load(f)
        technique_ids = [t["technique_id"] for t in techniques]
        if cached_ids != technique_ids:
            raise ValueError(
                "Cached technique_ids.json does not match the current corpus's technique order/content -- "
                "the cache is stale. Re-run scripts/rebuild_technique_embeddings.py before mapping.")
        return cls(techniques, vectors)

    # ---- full ATT&CK corpus (cached, versioned) ----
    @classmethod
    def from_mitre(cls, mitre_corpus, embedder, cache_dir: str, platforms: Optional[Sequence[str]] = None,
                   rebuild: bool = False, progress=None) -> "TechniqueCorpus":
        from src.mitre.embedding_cache import load_or_build
        techniques, vectors, status = load_or_build(mitre_corpus, embedder, cache_dir, rebuild=rebuild,
                                                    progress=progress)
        total_active = len(techniques)
        if platforms:
            wanted = {p.lower() for p in platforms}
            keep = [i for i, t in enumerate(techniques)
                    if {p.lower() for p in t.get("platforms") or []} & wanted]
            techniques = [techniques[i] for i in keep]
            vectors = vectors[keep] if keep else vectors[:0]
        if not techniques:
            raise ValueError(f"No active ATT&CK techniques apply to platform filter {list(platforms or [])}.")
        meta = {"corpus_version": mitre_corpus.version_label, "content_hash": mitre_corpus.content_hash,
                "attack_version": mitre_corpus.attack_version, "model_id": status["model_id"],
                "cache": status, "platform_filter": list(platforms) if platforms else None,
                "active_total": total_active, "applicable": len(techniques),
                "legacy_curated": bool(mitre_corpus.meta.get("legacy_curated"))}
        return cls(techniques, vectors, meta)


class SemanticMitreMapper:
    """
        corpus = TechniqueCorpus.from_mitre(mitre, embedder, "cache/mitre", platforms=["Windows"])
        mapper = SemanticMitreMapper(corpus, embedder, top_k=5, min_similarity=0.35, weak_similarity=0.25)
        mappings = mapper.map_clusters(cluster_profiles)
    """

    def __init__(self, corpus: TechniqueCorpus, embedder: TextEmbedder, top_k: int = 3,
                 min_similarity: float = 0.35, weak_similarity: Optional[float] = None):
        if weak_similarity is not None and weak_similarity > min_similarity:
            raise ValueError(f"weak_similarity ({weak_similarity}) must be <= min_similarity ({min_similarity}).")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.corpus = corpus
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.weak_similarity = weak_similarity

    # ------------------------------------------------------------------ core
    def _status(self, score: float) -> str:
        if score >= self.min_similarity:
            return STATUS_CANDIDATE
        if self.weak_similarity is not None and score >= self.weak_similarity:
            return STATUS_WEAK
        return STATUS_NONE

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        prepared = [prepare_text(t) for t in texts]
        return np.asarray(self.embedder.embed(prepared), dtype=np.float32)

    def rank_vector(self, vec: np.ndarray, supporting_event_uids: Optional[List[str]] = None
                    ) -> List[TechniqueCandidate]:
        sims = self.corpus.vectors @ vec            # cosine: both sides are L2-normalised
        order = np.argsort(-sims, kind="stable")[: self.top_k]
        out = []
        for rank, idx in enumerate(order, start=1):
            score = float(sims[idx])
            if score <= 0.0:
                break                                 # never present a zero-similarity technique as a lead
            t = self.corpus.techniques[idx]
            desc = (t.get("description") or "").strip().replace("\n", " ")
            out.append(TechniqueCandidate(
                technique_id=t["technique_id"],
                technique_name=t.get("full_name") or t["technique_name"],
                tactic=t.get("tactic") or ", ".join(t.get("tactics") or []),
                similarity_score=round(score, 4),
                mapping_status=self._status(score),
                tactics=list(t.get("tactics") or []),
                rank=rank,
                is_subtechnique=bool(t.get("is_subtechnique")),
                description_excerpt=desc[:320] + ("..." if len(desc) > 320 else ""),
                url=t.get("url"),
                supporting_event_uids=list(supporting_event_uids or []),
            ))
        return out

    @staticmethod
    def _outcome(cands: List[TechniqueCandidate]) -> str:
        if any(c.mapping_status == STATUS_CANDIDATE for c in cands):
            return OUTCOME_CANDIDATES
        if any(c.mapping_status == STATUS_WEAK for c in cands):
            return OUTCOME_WEAK_ONLY
        return OUTCOME_NO_MATCH

    def map_vector(self, unit_id: str, scope: str, cluster_id: int, display_text: str, mapping_text: str,
                   vec: np.ndarray, representative_event_uids: List[str]) -> ClusterMapping:
        cands = self.rank_vector(vec, representative_event_uids)
        margin = round(cands[0].similarity_score - cands[1].similarity_score, 4) if len(cands) > 1 else None
        return ClusterMapping(
            cluster_id=cluster_id, cluster_text=display_text, top_k_techniques=cands,
            representative_event_uids=representative_event_uids, unit_id=unit_id, scope=scope,
            overall_outcome=self._outcome(cands), top_margin=margin,
            mapping_text=prepare_text(mapping_text),
            corpus_version=self.corpus.meta.get("corpus_version", ""))

    # ------------------------------------------------------- original API
    def map_cluster(self, cluster_id: int, cluster_text: str, representative_event_uids: List[str],
                    mapping_text: Optional[str] = None) -> ClusterMapping:
        text = mapping_text or cluster_text
        vec = self.embed_texts([text])[0]
        return self.map_vector(f"C{cluster_id}", "cluster", cluster_id, cluster_text, text, vec,
                               representative_event_uids)

    def map_clusters(self, cluster_profiles: List) -> List[ClusterMapping]:
        """cluster_profiles: ClusterProfile objects (or anything with
        .cluster_id / .text / .representative_event_uids and optionally
        .mapping_text). Noise (-1) is rejected here on purpose: noise units
        are analysed through map_unit() so they can never be mistaken for a
        normal cluster."""
        for p in cluster_profiles:
            if p.cluster_id == -1:
                raise ValueError(
                    "Received cluster_id == -1 (noise) in map_clusters() -- noise must go through "
                    "map_unit(scope='noise_group'|'noise_event'), not the cluster path.")
        texts = [getattr(p, "mapping_text", "") or p.text for p in cluster_profiles]
        if not texts:
            return []
        vecs = self.embed_texts(texts)
        return [self.map_vector(f"C{p.cluster_id}", "cluster", p.cluster_id, p.text, texts[i], vecs[i],
                                p.representative_event_uids)
                for i, p in enumerate(cluster_profiles)]

    # --------------------------------------------------------- noise units
    def map_units(self, units: List[Dict]) -> List[ClusterMapping]:
        """units: dicts with unit_id, scope ('noise_group'|'noise_event'), text (display narrative),
        mapping_text, representative_event_uids. Event-level / small-group semantic analysis: the
        result is labelled with its scope and has cluster_id == -1."""
        if not units:
            return []
        for u in units:
            if u["scope"] not in ("noise_group", "noise_event"):
                raise ValueError(f"map_units only accepts noise scopes, got {u['scope']!r}")
        texts = [u.get("mapping_text") or u["text"] for u in units]
        vecs = self.embed_texts(texts)
        return [self.map_vector(u["unit_id"], u["scope"], -1, u["text"], texts[i], vecs[i],
                                u["representative_event_uids"])
                for i, u in enumerate(units)]
