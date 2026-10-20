"""
src/mitre/embedding_cache.py
------------------------------
Cached ATT&CK technique embeddings, keyed by

    (corpus content hash, embedding model id, preprocessing version)

so a changed corpus, a different model, or new preprocessing can never be
served a stale vector. The file name carries the key; the .npz also stores
the technique-id order and metadata, which are re-validated on every load.

    cache/mitre/embeddings/<corpus8>__<model-slug>__<prep>.npz
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.mapping.text_prep import PREP_VERSION, technique_embedding_text
from src.mitre.corpus import MitreCorpus

EMBED_BATCH = 64


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:60]


def cache_path(cache_dir: str, corpus_hash: str, model_id: str, prep_version: str = PREP_VERSION) -> str:
    return os.path.join(cache_dir, "embeddings", f"{corpus_hash[:8]}__{_slug(model_id)}__{prep_version}.npz")


def _try_load(path: str, corpus: MitreCorpus, ids: List[str], model_id: str) -> Tuple[Optional[np.ndarray], str]:
    if not os.path.exists(path):
        return None, "no cache file"
    try:
        with np.load(path, allow_pickle=False) as z:
            vectors = z["vectors"]
            cached_ids = [str(x) for x in z["ids"]]
            meta = json.loads(str(z["meta"]))
    except Exception as e:
        return None, f"cache unreadable ({type(e).__name__})"
    if meta.get("corpus_hash") != corpus.content_hash:
        return None, "corpus changed"
    if meta.get("model_id") != model_id:
        return None, "embedding model changed"
    if meta.get("prep_version") != PREP_VERSION:
        return None, "text preprocessing changed"
    if cached_ids != ids:
        return None, "technique id order differs"
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not np.isfinite(vectors).all():
        return None, "cached vectors malformed"
    return vectors.astype(np.float32, copy=False), "ok"


def load_or_build(corpus: MitreCorpus, embedder, cache_dir: str, rebuild: bool = False,
                  progress: Optional[Callable[[int, int], None]] = None
                  ) -> Tuple[List[Dict], np.ndarray, Dict[str, object]]:
    """Returns (active_techniques, vectors[n, dim], status).

    `active_techniques` is corpus.active() over ALL platforms (platform
    filtering happens at match time so one cache serves every platform).
    status["cache"] is 'hit' | 'built' | 'rebuilt' with a reason, plus the path."""
    techniques = corpus.active()
    ids = [t["technique_id"] for t in techniques]
    model_id = getattr(embedder, "model_id", embedder.__class__.__name__)
    path = cache_path(cache_dir, corpus.content_hash, model_id)

    if not rebuild:
        vectors, why = _try_load(path, corpus, ids, model_id)
        if vectors is not None:
            return techniques, vectors, {"cache": "hit", "path": path, "model_id": model_id,
                                         "dim": int(vectors.shape[1]), "count": len(ids)}
    else:
        why = "rebuild requested"

    texts = [technique_embedding_text(t) for t in techniques]
    chunks = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunks.append(np.asarray(embedder.embed(texts[i:i + EMBED_BATCH]), dtype=np.float32))
        if progress:
            progress(min(i + EMBED_BATCH, len(texts)), len(texts))
    vectors = np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Embedder produced non-finite values for the technique corpus.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {"corpus_hash": corpus.content_hash, "corpus_version": corpus.version_label, "model_id": model_id,
            "prep_version": PREP_VERSION, "count": len(ids), "dim": int(vectors.shape[1])}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, vectors=vectors, ids=np.array(ids), meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)

    # invalidate superseded caches for the SAME model + prep (older corpus versions)
    removed = []
    edir = os.path.dirname(path)
    suffix = f"__{_slug(model_id)}__{PREP_VERSION}.npz"
    for name in os.listdir(edir):
        full = os.path.join(edir, name)
        if name.endswith(suffix) and full != path:
            try:
                os.remove(full)
                removed.append(name)
            except OSError:
                pass
    return techniques, vectors, {"cache": "built" if why == "no cache file" else "rebuilt", "reason": why,
                                 "path": path, "model_id": model_id, "dim": int(vectors.shape[1]),
                                 "count": len(ids), "pruned_stale": removed}
