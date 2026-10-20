"""
src/mapping/embedders.py
---------------------------
Embedding backend abstraction for semantic_mapper.py. Kept as a small
interface (not baked directly into the mapper) so the production SBERT
backend can be swapped for an offline backend without touching mapping
logic.

Backends:
    SBertEmbedder          PRODUCTION. sentence-transformers; the model may be
                           a Hugging Face name OR a local folder path.
    LexicalCorpusEmbedder  OFFLINE DEV/TEST STAND-IN. TF-IDF fitted on the
                           technique corpus only. Matches literal word overlap,
                           not meaning -- scores are NOT comparable to SBERT's.
                           Selected only by an explicit flag, never as a
                           silent fallback.

Every embedder exposes `model_id` (part of the embedding-cache key) and
`is_standin` (True for anything that is not a real semantic model).
"""

from typing import Iterable, List, Optional, Protocol

import numpy as np


class EmbedderUnavailable(Exception):
    """The requested embedding model cannot be loaded (missing package,
    model not downloaded, no network...)."""


class TextEmbedder(Protocol):
    """Anything with this method can be used by SemanticMitreMapper."""
    def embed(self, texts: List[str]) -> np.ndarray:
        ...  # returns shape [len(texts), dim], L2-normalized rows


class SBertEmbedder:
    """
    PRODUCTION embedder. Requires `sentence-transformers`. The model is
    downloaded from Hugging Face on first use (then cached by the library);
    pass a local directory as `model_name` to run fully offline.

        embedder = SBertEmbedder("all-MiniLM-L6-v2")
        vectors = embedder.embed(["some cluster narrative text", ...])
    """
    is_standin = False

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None,
                 batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer  # lazy: keep the rest importable
        except ImportError as e:
            raise EmbedderUnavailable(
                "sentence-transformers is not installed (pip install sentence-transformers). "
                "For an explicit offline development mode use --embedder lexical.") from e
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self._model = SentenceTransformer(model_name, cache_folder=cache_folder)
        except Exception as e:                       # network, auth, missing folder, corrupt cache ...
            raise EmbedderUnavailable(
                f"Could not load embedding model '{model_name}' ({type(e).__name__}: {str(e)[:200]}). "
                f"It must be downloaded once from Hugging Face (internet needed), or pass a local model "
                f"folder via --embedding-model. For an explicit offline development mode use --embedder lexical."
            ) from e

    @property
    def model_id(self) -> str:
        return f"sbert:{self.model_name}"

    def embed(self, texts: List[str]) -> np.ndarray:
        v = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                               batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class LexicalCorpusEmbedder:
    """OFFLINE DEV/TEST STAND-IN (see module docstring). Deterministic: the
    vocabulary is fitted on the technique texts only, never on the narratives
    being mapped, so technique vectors can be cached safely."""
    is_standin = True
    model_id = "lexical-tfidf-v1"

    def __init__(self, corpus_texts: Iterable[str], max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True,
                                    ngram_range=(1, 2), min_df=1, max_features=max_features,
                                    dtype=np.float32)
        self._vec.fit(list(corpus_texts))

    def embed(self, texts: List[str]) -> np.ndarray:
        m = self._vec.transform(list(texts)).toarray().astype(np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n


def create_embedder(kind: str, model_name: str, technique_texts: Optional[Iterable[str]] = None):
    """kind: 'sbert' (production) | 'lexical' (explicit offline stand-in)."""
    if kind == "sbert":
        return SBertEmbedder(model_name)
    if kind == "lexical":
        if technique_texts is None:
            raise ValueError("The lexical stand-in needs the technique texts to fit its vocabulary.")
        return LexicalCorpusEmbedder(technique_texts)
    raise ValueError(f"Unknown embedder kind '{kind}' (expected 'sbert' or 'lexical').")
