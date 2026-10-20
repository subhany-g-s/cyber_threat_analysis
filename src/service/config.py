"""
src/service/config.py
-----------------------
Every tunable of a pipeline run in one serialisable object. The CLI builds it
from flags; a future Streamlit page builds it from widgets. Paths are
resolved against `project_root` by the service, so a UI can run from any
working directory.

Ollama defaults live HERE and nowhere else (the CLI and the LLM client import
them). Override with flags (`--ollama-model`, `--ollama-host`, ...) or with the
environment variables SUBHANY_OLLAMA_MODEL / OLLAMA_HOST.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


def normalize_ollama_host(host: str) -> str:
    """Ollama's own OLLAMA_HOST is often written without a scheme ('127.0.0.1:11434')."""
    host = (host or "").strip()
    if host and "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


DEFAULT_OLLAMA_MODEL = os.environ.get("SUBHANY_OLLAMA_MODEL", "llama3.1")
DEFAULT_OLLAMA_HOST = normalize_ollama_host(os.environ.get("OLLAMA_HOST", "") or "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT_S = 180        # per request; local models can be slow on CPU
DEFAULT_OLLAMA_NUM_CTX = 8192         # Ollama's own default (2-4k) would silently truncate the evidence package


@dataclass
class PipelineConfig:
    input_path: str = "data/sample_2000_events.json"
    input_format: str = "auto"                       # auto | jsonl | json-array | csv
    project_root: str = "."

    # --- pretrained GAE (never retrained here) ---
    checkpoint_path: str = "models/gae_model.pt"
    feature_schema_path: str = "models/gae_feature_schema.json"
    precomputed_embeddings: Optional[str] = None     # .npy, used INSTEAD of running the GAE
    precomputed_mapping: Optional[str] = None        # json with "row_to_event_uid"

    # --- clustering ---
    cluster_algorithm: str = "hdbscan"
    min_cluster_size: int = 20
    min_samples: int = 10

    # --- novelty ---
    novelty_weights: Dict[str, float] = field(default_factory=lambda: {
        "density": 1.0, "embedding_distance": 1.0, "reconstruction": 1.0, "graph_rarity": 1.0})
    novelty_thresholds: Tuple[float, float, float] = (0.25, 0.50, 0.75)

    # --- MITRE semantic mapping ---
    mitre_cache_dir: str = "cache/mitre"
    corpus_file: Optional[str] = None                # explicit corpus file (e.g. legacy 15-entry JSON)
    embedder: str = "sbert"                          # sbert | lexical (explicit offline stand-in)
    embedding_model: str = "all-MiniLM-L6-v2"        # HF name or local folder
    top_k: int = 5
    candidate_threshold: float = 0.35
    weak_threshold: float = 0.25
    platforms: Tuple[str, ...] = ("Windows",)        # () => all platforms
    rebuild_embeddings: bool = False

    # --- noise review ---
    noise_review_top_n: int = 10                     # noise units that get semantic analysis + retrieval + report
    noise_group_min_size: int = 2                    # connected unclustered events >= this form a group

    # --- GraphRAG ---
    retrieval_depth: int = 2
    max_traversal_events: int = 40
    hub_limit: int = 60                              # entities touching more events are not expanded
    related_units_k: int = 3
    semantic_reference_k: int = 3                    # extra ATT&CK references retrieved beyond the candidates
    neo4j_enabled: bool = False                      # env: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE

    # --- ontology module hand-off ---
    ontology_path: Optional[str] = None              # JSON in the 'subhany.ontology-entities' contract

    # --- report ---
    llm: str = "auto"                                # auto | none | ollama
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_timeout: int = DEFAULT_OLLAMA_TIMEOUT_S
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    report_top_n: int = 5                            # findings shown in the primary report (and sent to the LLM)

    # --- optional export (default: nothing is written) ---
    export_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["novelty_thresholds"] = list(self.novelty_thresholds)
        d["platforms"] = list(self.platforms)
        return d
