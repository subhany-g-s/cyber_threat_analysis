"""
src/mapping/embed_technique_corpus.py
-----------------------------------------
One-time (re-run only when config/mitre_technique_corpus.json changes)
script: embeds the technique corpus with the PRODUCTION SBERT embedder
and caches the result, so semantic_mapper.py never re-embeds the corpus
on every mapping run.

Usage:
    python -m src.mapping.embed_technique_corpus
    python -m src.mapping.embed_technique_corpus --model all-MiniLM-L6-v2
"""

import argparse
import json
import os

from src.mapping.embedders import SBertEmbedder
from src.mapping.semantic_mapper import TechniqueCorpus

DEFAULT_CORPUS_PATH = "config/mitre_technique_corpus.json"
DEFAULT_VECTORS_PATH = "cache/technique_embeddings.npy"
DEFAULT_IDS_PATH = "cache/technique_ids.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS_PATH)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out-vectors", default=DEFAULT_VECTORS_PATH)
    ap.add_argument("--out-ids", default=DEFAULT_IDS_PATH)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_vectors), exist_ok=True)

    with open(args.corpus) as f:
        techniques = json.load(f)["techniques"]
    print(f"Loaded {len(techniques)} techniques from {args.corpus}")

    print(f"Loading embedder '{args.model}' (requires network access on first use) ...")
    embedder = SBertEmbedder(args.model)

    print("Embedding technique corpus ...")
    corpus = TechniqueCorpus.build(techniques, embedder)

    import numpy as np
    np.save(args.out_vectors, corpus.vectors)
    with open(args.out_ids, "w") as f:
        json.dump([t["technique_id"] for t in techniques], f)

    print(f"Wrote {args.out_vectors}  shape={corpus.vectors.shape}")
    print(f"Wrote {args.out_ids}")


if __name__ == "__main__":
    main()
