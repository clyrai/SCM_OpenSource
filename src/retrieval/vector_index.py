"""Vector index for fast nearest-neighbor seed selection.

The concept graph already stores per-concept embeddings, but spreading
activation has historically used token-overlap to pick its seeds. That works
fine at small scale; at 10K+ concepts it falls over because every query does
an O(n) scan over descriptions.

This module adds a pluggable VectorIndex over the same embeddings. The
default implementation is pure-numpy in-memory (no extra dependencies); a
production deployment can swap to FAISS / pgvector / Qdrant by implementing
the same `add` / `search` / `remove` / `rebuild` interface.

Spreading activation calls `seed_by_vector(query_embedding)` instead of
(or in addition to) the token cue path. The graph propagation that follows
is unchanged — vectors only affect *which concepts* enter the propagation,
not how the propagation behaves.

Why this lives in retrieval/ and not core/: the index is purely a retrieval
concern. It does not own concepts (LongTermMemory does), it does not
persist (SQLite does). It is a derived structure built from concepts and
discarded freely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VectorHit:
    """A search result from a VectorIndex.

    score: cosine similarity in [-1, 1]; higher is more relevant.
    """
    concept_id: str
    score: float


class VectorIndex(ABC):
    """Interface every vector backend must implement.

    Concepts are added/removed individually; search returns top-k by cosine
    similarity. Implementations decide how to maintain their internal
    structure (matrix, FAISS HNSW, pgvector ivfflat, etc.).
    """

    @abstractmethod
    def add(self, concept_id: str, embedding: Sequence[float]) -> None:
        """Add or replace a single concept's embedding."""

    @abstractmethod
    def remove(self, concept_id: str) -> None:
        """Remove a concept from the index. No-op if not present."""

    @abstractmethod
    def search(
        self,
        embedding: Sequence[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[VectorHit]:
        """Return the top_k most similar concepts above min_score.

        Cosine similarity. Returns empty list if the index is empty.
        """

    @abstractmethod
    def rebuild(
        self,
        items: Iterable[Tuple[str, Sequence[float]]],
    ) -> None:
        """Replace the entire index with the given (id, embedding) pairs.

        Used on engine startup when restoring from SQLite, or after a sleep
        cycle that bulk-modifies many concepts.
        """

    @abstractmethod
    def size(self) -> int:
        """Number of concepts currently indexed."""


class InMemoryVectorIndex(VectorIndex):
    """Pure-numpy in-memory cosine-similarity index.

    Zero extra dependencies. Fine up to ~100K concepts on a developer
    laptop (matrix-vector multiply is ~50ms at 100K × 384-dim). Beyond
    that, swap in FaissVectorIndex or PgvectorIndex.

    Stores embeddings as float32 to halve memory vs. the float64 that
    numpy would default to from a Python list. Normalizes once on insert
    so search is a single dot product per query.
    """

    def __init__(self, dim: Optional[int] = None):
        self._dim = dim
        # Parallel arrays: matrix[i] is the unit-normalized embedding for ids[i].
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # shape (n, dim), float32, unit-normalized
        self._id_to_row: dict = {}

    def add(self, concept_id: str, embedding: Sequence[float]) -> None:
        vec = self._prepare(embedding)
        if vec is None:
            return
        existing_row = self._id_to_row.get(concept_id)
        if existing_row is not None:
            # Replace in place — embedding may have been refined by the encoder.
            self._matrix[existing_row] = vec
            return
        # Append new row.
        if self._matrix is None:
            self._matrix = vec[np.newaxis, :].copy()
        else:
            self._matrix = np.vstack([self._matrix, vec[np.newaxis, :]])
        self._id_to_row[concept_id] = len(self._ids)
        self._ids.append(concept_id)

    def remove(self, concept_id: str) -> None:
        row = self._id_to_row.pop(concept_id, None)
        if row is None:
            return
        # Drop the row by replacing with the last row, then truncate.
        # Keeps the matrix contiguous without an O(n) reallocation per delete.
        last = len(self._ids) - 1
        if row != last and self._matrix is not None:
            self._matrix[row] = self._matrix[last]
            moved_id = self._ids[last]
            self._ids[row] = moved_id
            self._id_to_row[moved_id] = row
        self._ids.pop()
        if self._matrix is not None:
            self._matrix = self._matrix[: len(self._ids)] if self._ids else None

    def search(
        self,
        embedding: Sequence[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[VectorHit]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = self._prepare(embedding)
        if q is None:
            return []
        # Both q and matrix rows are unit-normalized → dot product == cosine.
        scores = self._matrix @ q
        if top_k >= len(self._ids):
            order = np.argsort(-scores)
        else:
            # argpartition is O(n) vs O(n log n) for a full sort.
            top_idx = np.argpartition(-scores, top_k)[:top_k]
            order = top_idx[np.argsort(-scores[top_idx])]
        out: List[VectorHit] = []
        for i in order:
            s = float(scores[int(i)])
            if s < min_score:
                break
            out.append(VectorHit(concept_id=self._ids[int(i)], score=s))
            if len(out) >= top_k:
                break
        return out

    def rebuild(
        self,
        items: Iterable[Tuple[str, Sequence[float]]],
    ) -> None:
        self._ids = []
        self._id_to_row = {}
        rows: List[np.ndarray] = []
        for cid, emb in items:
            vec = self._prepare(emb)
            if vec is None:
                continue
            self._id_to_row[cid] = len(self._ids)
            self._ids.append(cid)
            rows.append(vec)
        self._matrix = np.vstack(rows) if rows else None

    def size(self) -> int:
        return len(self._ids)

    # ── helpers ───────────────────────────────────────────────────────────

    def _prepare(self, embedding: Sequence[float]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        try:
            arr = np.asarray(embedding, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim != 1 or arr.size == 0:
            return None
        if self._dim is None:
            self._dim = int(arr.size)
        elif arr.size != self._dim:
            # Mismatched dim — silently skip rather than crash retrieval.
            # An encoder swap mid-session is the only realistic path here.
            return None
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return None
        return arr / norm
