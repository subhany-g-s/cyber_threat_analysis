"""
tests/tfidf_embedder.py
---------------------------
OFFLINE DEV/TEST STAND-IN ONLY. Implements the same TextEmbedder
interface as src/mapping/embedders.SBertEmbedder so semantic_mapper.py's
logic can be exercised end-to-end in an environment with no network
access to a model hub (e.g. this sandbox's CI-style validation).

This is NOT a replacement for SBERT in any real deployment: it matches
on lexical overlap (term frequency), not semantic meaning, so e.g.
"powershell execution" and "script-based command interpreter use" would
score low here despite being semantically related -- exactly the gap
SBERT is supposed to close. Every place this class is used in this
project is explicitly labeled as a dev/test path, never the default.
"""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus_texts: List[str]):
        self._vectorizer.fit(corpus_texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(corpus_texts) once before embed() -- see run_pipeline.py's --embedder tfidf path.")
        vectors = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
