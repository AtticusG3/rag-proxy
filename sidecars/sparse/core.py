"""BM25 sparse index helpers (no FastAPI dependency)."""

from __future__ import annotations

import gc
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

from rank_bm25 import BM25Okapi

try:
    from rag_proxy.chunk_text import PAYLOAD_TEXT_KEYS, extract_chunk_text
except ImportError:
    from chunk_text import PAYLOAD_TEXT_KEYS, extract_chunk_text  # noqa: F401 — Docker flat layout

DEFAULT_COLLECTION = "nomad_knowledge_base"

# Small metadata kept for proxy recency boost on sparse-only hits.
_RECENCY_KEYS = ("updated_at", "mtime", "timestamp")

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class IndexedDoc:
    doc_id: str
    payload: dict[str, Any]
    tokens: list[str]


def slim_payload(full: dict[str, Any], text: str) -> dict[str, Any]:
    """Keep chunk text plus recency fields; drop bulky ingest metadata."""
    slim: dict[str, Any] = {}
    for key in PAYLOAD_TEXT_KEYS:
        value = full.get(key)
        if value:
            slim[key] = str(value)
            break
    if not slim and text:
        slim["text"] = text
    for key in _RECENCY_KEYS:
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
    tokens = tokenize(text)
    if not tokens:
        return None
    return IndexedDoc(
        doc_id=doc_id,
        payload=slim_payload(payload, text),
        tokens=tokens,
    )


class SparseIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._docs: list[IndexedDoc] = []
        self._bm25: BM25Okapi | None = None
        self.collection = ""
        self.last_sync = 0.0
        self.point_count = 0

    def add_points(self, points: Iterable[dict[str, Any]]) -> None:
        for point in points:
            doc = point_to_doc(point)
            if doc is not None:
                self._docs.append(doc)

    def finalize(self, collection: str) -> int:
        corpus = [doc.tokens for doc in self._docs]
        bm25 = BM25Okapi(corpus) if corpus else None
        with self._lock:
            self._bm25 = bm25
            self.collection = collection
            self.last_sync = time.time()
            self.point_count = len(self._docs)
        return len(self._docs)

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if not self._bm25 or not self._docs:
                return []
            docs = self._docs
            bm25 = self._bm25

        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)
        scores = bm25.get_scores(query_tokens)
        candidate_n = min(len(docs), max(limit * 10, limit))
        ranked = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:candidate_n]

        results: list[dict[str, Any]] = []
        for index in ranked:
            if len(results) >= limit:
                break
            doc = docs[index]
            if not query_token_set.intersection(doc.tokens):
                continue
            results.append(
                {
                    "id": doc.doc_id,
                    "score": float(scores[index]),
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
