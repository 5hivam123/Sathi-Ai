"""
FAISS Vector Store module.
Manages indexing, saving, loading, and querying of document embeddings.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from src.config import FAISS_INDEX_PATH, TOP_K_RESULTS
from src.text_splitter import TextChunk


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStore:
    """
    FAISS-backed vector store for document chunks.
    Stores both the FAISS index and associated chunk metadata.
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunks: list[TextChunk] = []
        self.dimension: Optional[int] = None

    # ── Building ──────────────────────────────────────────────────────────────

    def build(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        """
        Build a FAISS flat-L2 index from chunks and their embeddings.
        (Embeddings should already be L2-normalized if cosine similarity is desired.)
        """
        if len(chunks) == 0:
            raise VectorStoreError("Cannot build vector store: no chunks provided.")
        if len(chunks) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        dim = embeddings.shape[1]
        self.dimension = dim

        # Using Inner Product index since embeddings are L2-normalized
        # → cosine similarity via inner product
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        self.index = index
        self.chunks = chunks

    # ── Querying ──────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = TOP_K_RESULTS
    ) -> list[tuple[TextChunk, float]]:
        """
        Search the index for the top-k most relevant chunks.
        Returns a list of (chunk, score) tuples sorted by relevance (descending).
        """
        if self.index is None:
            raise VectorStoreError(
                "Vector store is empty. Please upload a document first."
            )

        query = query_embedding.astype(np.float32).reshape(1, -1)
        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(query, k)
        results: list[tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self.chunks[idx], float(score)))

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = FAISS_INDEX_PATH) -> None:
        """Save FAISS index and chunk metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise VectorStoreError("Nothing to save: index is empty.")

        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save chunks as pickle (they are plain dataclass instances)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        # Save dimension for validation on load
        with open(path / "meta.json", "w") as f:
            json.dump({"dimension": self.dimension, "n_chunks": len(self.chunks)}, f)

    def load(self, path: Path = FAISS_INDEX_PATH) -> bool:
        """
        Load FAISS index and metadata from disk.
        Returns True on success, False if no saved index exists.
        """
        path = Path(path)
        index_file = path / "index.faiss"
        chunks_file = path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            with open(path / "meta.json") as f:
                meta = json.load(f)
            self.dimension = meta.get("dimension")
            return True
        except Exception as exc:
            raise VectorStoreError(f"Failed to load vector store: {exc}") from exc

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def source_filename(self) -> Optional[str]:
        if self.chunks:
            return self.chunks[0].source_filename
        return None
