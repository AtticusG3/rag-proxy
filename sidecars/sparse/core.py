"""BM25 sparse index helpers (no FastAPI dependency)."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

try:
    from rag_proxy.chunk_text import extract_chunk_text
except ImportError:
    from chunk_text import extract_chunk_text  # noqa: F401 — Docker flat layout

DEFAULT_COLLECTION = "nomad_knowledge_base"

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class IndexedDoc:
    doc_id: str
    payload: dict[str, Any]
    tokens: list[str]


class SparseIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._docs: list[IndexedDoc] = []
        self._bm25: BM25Okapi | None = None
        self.collection = ""
        self.last_sync = 0.0
        self.point_count = 0

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if not self._bm25 or not self._docs:
                return []
            query_tokens = tokenize(query)
            if not query_tokens:
                return []
            scores = self._bm25.get_scores(query_tokens)
            ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
            query_token_set = set(query_tokens)
            results: list[dict[str, Any]] = []
            for index in ranked:
                if len(results) >= limit:
                    break
                doc = self._docs[index]
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
        docs: list[IndexedDoc] = []
        for point in points:
            payload = point.get("payload") or {}
            text = extract_chunk_text({"payload": payload})
            if not text:
                continue
            doc_id = str(point.get("id", ""))
            if not doc_id:
                continue
            docs.append(
                IndexedDoc(
                    doc_id=doc_id,
                    payload=payload,
                    tokens=tokenize(text),
                )
            )
        corpus = [doc.tokens for doc in docs]
        bm25 = BM25Okapi(corpus) if corpus else None
        with self._lock:
            self._docs = docs
            self._bm25 = bm25
            self.collection = collection
            self.last_sync = time.time()
            self.point_count = len(docs)
        return len(docs)


class IndexRegistry:
    """One BM25 index per Qdrant collection name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._indexes: dict[str, SparseIndex] = {}

    def rebuild(self, collection: str, points: list[dict[str, Any]]) -> int:
        index = SparseIndex()
        count = index.rebuild(collection, points)
        with self._lock:
            self._indexes[collection] = index
        return count

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
            # Most recently synced collection wins for health reporting.
            return max(self._indexes.values(), key=lambda idx: idx.last_sync).collection

    def doc_count(self, collection: str) -> int:
        with self._lock:
            index = self._indexes.get(collection)
        return index.point_count if index else 0

    def last_sync(self, collection: str) -> float:
        with self._lock:
            index = self._indexes.get(collection)
        return index.last_sync if index else 0.0
