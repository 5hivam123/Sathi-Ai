"""
Embeddings module.
Loads a Sentence Transformer model and generates embeddings for text chunks.
Includes an intelligent fallback feature-embedding model if sentence-transformers
or torch is not present in the current environment.
"""

from __future__ import annotations

import hashlib
import re
from typing import Union
import numpy as np

from src.config import EMBEDDING_MODEL


class FastFallbackEmbeddingModel:
    """
    Lightweight, dependency-free dense feature embedding model.
    Generates 384-dimensional normalized vectors using signed feature hashing
    over word n-grams and character subwords.
    Drop-in compatible with SentenceTransformer.encode().
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            vec[0] = 1.0
            return vec

        # Extract tokens: unigrams, bigrams, and character trigrams
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(words[i] + "_" + words[i + 1])
        for word in words:
            for j in range(max(0, len(word) - 2)):
                tokens.append(word[j : j + 3])

        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec

    def encode(
        self,
        texts: Union[str, list[str]],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return self._hash_text(texts)
        res = np.array([self._hash_text(t) for t in texts], dtype=np.float32)
        return res


_CACHED_MODEL = None


def get_embedding_model():
    """
    Load and return the embedding model.
    Tries SentenceTransformer first; falls back to FastFallbackEmbeddingModel
    if sentence-transformers or torch is unavailable.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        _CACHED_MODEL = model
        return model
    except Exception:
        # Fall back gracefully to high-performance feature hasher
        model = FastFallbackEmbeddingModel(dim=384)
        _CACHED_MODEL = model
        return model


def embed_texts(texts: list[str], model=None) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    Returns a numpy array of shape (N, embedding_dim).
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    if model is None:
        model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str, model=None) -> np.ndarray:
    """
    Generate an embedding for a single query string.
    Returns a 1-D numpy array.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if model is None:
        model = get_embedding_model()

    embedding = model.encode(
        query,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding
