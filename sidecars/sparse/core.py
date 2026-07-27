"""BM25 sparse index helpers (no FastAPI dependency)."""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import bm25s

try:
    from rag_proxy.chunk_text import PAYLOAD_TEXT_KEYS, extract_chunk_text
except ImportError:
    from chunk_text import PAYLOAD_TEXT_KEYS, extract_chunk_text  # noqa: F401 — Docker flat layout

DEFAULT_COLLECTION = "nomad_knowledge_base"

# Small metadata kept for proxy recency boost on sparse-only hits.
_RECENCY_KEYS = ("updated_at", "mtime", "timestamp")

# Sparse-only hits never touch Qdrant, so citations come from these keys alone.
PROVENANCE_KEYS = ("source", "title", "chunk_idx")

# Match the whole-word tokens the sidecar has always indexed rather than the
# bm25s default, which drops single-character tokens.
_TOKEN_PATTERN = r"(?u)\w+"

# Stopwords carry ~0 BM25 weight but appear in nearly every document, so keeping
# them turns their rows in the score matrix dense. Dropping them is what makes a
# full-corpus index fit in RAM.
_TOKENIZE_KWARGS: dict[str, Any] = {
    "token_pattern": _TOKEN_PATTERN,
    "stopwords": "en",
    "stemmer": None,
    "show_progress": False,
}


@dataclass
class IndexedDoc:
    doc_id: str
    payload: dict[str, Any]
    text: str


def slim_payload(full: dict[str, Any], text: str) -> dict[str, Any]:
    """Keep chunk text, provenance and recency fields; drop bulky ingest metadata."""
    slim: dict[str, Any] = {}
    for key in PAYLOAD_TEXT_KEYS:
        value = full.get(key)
        if value:
            slim[key] = str(value)
            break
    if not slim and text:
        slim["text"] = text
    for key in (*PROVENANCE_KEYS, *_RECENCY_KEYS):
        value = full.get(key)
        if value is not None:
            slim[key] = value
    return slim


def point_to_doc(point: dict[str, Any]) -> IndexedDoc | None:
    payload = point.get("payload") or {}
    text = extract_chunk_text({"payload": payload})
    if not text:
        return None
    doc_id = str(point.get("id", ""))
    if not doc_id:
        return None
    slim = slim_payload(payload, text)
    # Same str object the payload already holds, so the corpus costs no extra memory.
    return IndexedDoc(doc_id=doc_id, payload=slim, text=text)


class SparseIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._docs: list[IndexedDoc] = []
        self._bm25: bm25s.BM25 | None = None
        self.collection = ""
        self.last_sync = 0.0
        self.point_count = 0

    def add_points(self, points: Iterable[dict[str, Any]]) -> None:
        for point in points:
            doc = point_to_doc(point)
            if doc is not None:
                self._docs.append(doc)

    def finalize(self, collection: str) -> int:
        bm25: bm25s.BM25 | None = None
        if self._docs:
            corpus_tokens = bm25s.tokenize(
                [doc.text for doc in self._docs], **_TOKENIZE_KWARGS
            )
            bm25 = bm25s.BM25()
            bm25.index(corpus_tokens, show_progress=False)
            # Token ids are the build-time peak; the score matrix replaces them.
            del corpus_tokens
            gc.collect()
        with self._lock:
            self._bm25 = bm25
            self.collection = collection
            self.last_sync = time.time()
            self.point_count = len(self._docs)
        return len(self._docs)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if self._bm25 is None or not self._docs:
                return []
            docs = self._docs
            bm25 = self._bm25

        if limit <= 0:
            return []
        query_tokens = bm25s.tokenize(query, **_TOKENIZE_KWARGS)
        # retrieve() raises when k exceeds the corpus size.
        k = min(limit, len(docs))
        indices, scores = bm25.retrieve(query_tokens, k=k, show_progress=False)

        results: list[dict[str, Any]] = []
        for position, score in zip(indices[0], scores[0]):
            # Every IDF is non-negative here, so a positive score means the doc
            # shares at least one query term. Zeros are padding, not matches.
            if score <= 0:
                continue
            doc = docs[int(position)]
            results.append(
                {
                    "id": doc.doc_id,
                    "score": float(score),
                    "payload": doc.payload,
                }
            )
        return results

    def rebuild(self, collection: str, points: list[dict[str, Any]]) -> int:
        fresh = SparseIndex()
        fresh.add_points(points)
        return fresh.finalize(collection)


class IndexRegistry:
    """One BM25 index per Qdrant collection name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._indexes: dict[str, SparseIndex] = {}

    def install(self, collection: str, index: SparseIndex) -> int:
        with self._lock:
            old = self._indexes.pop(collection, None)
            self._indexes[collection] = index
        del old
        gc.collect()
        return index.point_count

    def rebuild(self, collection: str, points: list[dict[str, Any]]) -> int:
        index = SparseIndex()
        index.add_points(points)
        index.finalize(collection)
        return self.install(collection, index)

    def search(self, collection: str, query: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            index = self._indexes.get(collection)
        if index is None:
            return []
        return index.search(query, limit)

    def loaded_collection(self) -> str:
        with self._lock:
            if not self._indexes:
                return ""
            return max(self._indexes.values(), key=lambda idx: idx.last_sync).collection

    def doc_count(self, collection: str) -> int:
        with self._lock:
            index = self._indexes.get(collection)
        return index.point_count if index else 0

    def last_sync(self, collection: str) -> float:
        with self._lock:
            index = self._indexes.get(collection)
        return index.last_sync if index else 0.0
