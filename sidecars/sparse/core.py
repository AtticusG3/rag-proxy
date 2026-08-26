"""BM25 sparse index helpers (no FastAPI dependency)."""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import bm25s
from bm25s.tokenization import STOPWORDS_EN, Tokenized

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
# full-corpus index fit in RAM (and keeps the mmap'd CSC from being dense).
_TOKENIZE_KWARGS: dict[str, Any] = {
    "token_pattern": _TOKEN_PATTERN,
    "stopwords": "en",
    "stemmer": None,
    "show_progress": False,
}

_SPLIT_RE = re.compile(_TOKEN_PATTERN)
_STOPWORDS = frozenset(STOPWORDS_EN)
# Below this, thread-pool split costs more than it saves (tests, tiny corpora).
_PARALLEL_MIN_DOCS = 1024

_META_NAME = "meta.json"
_PARAMS_NAME = "params.index.json"


def resolve_n_threads(n: int) -> int:
    """0 or negative means all CPUs; otherwise clamp to at least 1."""
    if n <= 0:
        return os.cpu_count() or 1
    return n


def split_text_batch(texts: list[str]) -> list[list[str]]:
    """Regex-split a batch of chunk texts (stopwords dropped). GIL-friendly."""
    findall = _SPLIT_RE.findall
    stops = _STOPWORDS
    out: list[list[str]] = []
    for text in texts:
        if not text:
            out.append([])
            continue
        out.append([tok for tok in findall(text.lower()) if tok not in stops])
    return out


def tokenized_from_token_lists(token_lists: list[list[str]]) -> Tokenized:
    token_to_index: dict[str, int] = {}
    corpus_ids: list[list[int]] = []
    for tokens in token_lists:
        doc_ids: list[int] = []
        for token in tokens:
            idx = token_to_index.get(token)
            if idx is None:
                idx = len(token_to_index)
                token_to_index[token] = idx
            doc_ids.append(idx)
        corpus_ids.append(doc_ids)
    return Tokenized(ids=corpus_ids, vocab=token_to_index)


def tokenize_corpus(texts: list[str], n_threads: int = 1) -> Any:
    """Same tokens as bm25s.tokenize; thread-pool the regex split on large corpora."""
    n_threads = max(1, n_threads)
    if n_threads == 1 or len(texts) < _PARALLEL_MIN_DOCS:
        return bm25s.tokenize(texts, **_TOKENIZE_KWARGS)
    n_chunks = min(n_threads * 4, len(texts))
    chunk_size = max(1, (len(texts) + n_chunks - 1) // n_chunks)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    token_lists: list[list[str]] = []
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for part in pool.map(split_text_batch, chunks):
            token_lists.extend(part)
    return tokenized_from_token_lists(token_lists)


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


def collection_persist_dir(root: Path, collection: str) -> Path:
    """Map a Qdrant collection name to a single path component under *root*."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in collection)
    return Path(root) / (safe or "_")


def _replace_dir(src: Path, dest: Path) -> None:
    """Atomically replace *dest* with *src* (same filesystem)."""
    dest = Path(dest)
    src = Path(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = dest.with_name(dest.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dest.exists():
        dest.rename(old)
    src.rename(dest)
    if old.exists():
        shutil.rmtree(old)


def _hit_from_doc(doc: Any, score: float, position: int) -> dict[str, Any]:
    if isinstance(doc, IndexedDoc):
        return {"id": doc.doc_id, "score": score, "payload": doc.payload}
    payload = dict(doc)
    doc_id = str(payload.pop("id", position))
    return {"id": doc_id, "score": score, "payload": payload}


class SparseIndex:
    def __init__(self, n_threads: int = 1) -> None:
        self._lock = threading.Lock()
        # JsonlCorpus seeks a shared mmap; concurrent searches must not race.
        self._corpus_lock = threading.Lock()
        self._docs: list[IndexedDoc] | Any = []
        self._bm25: bm25s.BM25 | None = None
        self.collection = ""
        self.last_sync = 0.0
        self.point_count = 0
        self.mmap = False
        self.persist_dir: Path | None = None
        self.n_threads = max(1, n_threads)

    def close(self) -> None:
        with self._lock:
            docs = self._docs
            self._docs = []
            self._bm25 = None
            self.point_count = 0
        closer = getattr(docs, "close", None)
        if callable(closer):
            closer()

    def add_points(self, points: Iterable[dict[str, Any]]) -> list[str]:
        added: list[str] = []
        for point in points:
            doc = point_to_doc(point)
            if doc is not None:
                self._docs.append(doc)
                added.append(doc.text)
        return added

    def finalize(
        self,
        collection: str,
        persist_dir: Path | None = None,
        mmap: bool = True,
        token_lists: list[list[str]] | None = None,
    ) -> int:
        docs = self._docs
        bm25: bm25s.BM25 | None = None
        if docs:
            if token_lists is not None:
                if len(token_lists) != len(docs):
                    raise ValueError(
                        f"token_lists length {len(token_lists)} != docs {len(docs)}"
                    )
                corpus_tokens = tokenized_from_token_lists(token_lists)
            else:
                corpus_tokens = tokenize_corpus(
                    [doc.text for doc in docs], n_threads=self.n_threads
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
            self.point_count = len(docs) if isinstance(docs, list) else 0
            self.mmap = False
            self.persist_dir = None
        # Locals pin the heap corpus + CSC for the whole function; drop them
        # before mmap adopt so the rebuild process can actually release RAM.
        del docs
        if persist_dir is not None and bm25 is not None:
            del bm25
            self._persist(Path(persist_dir), mmap=mmap)
        return self.point_count

    def _persist(self, persist_dir: Path, *, mmap: bool) -> None:
        staging = persist_dir.with_name(persist_dir.name + ".building")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        self._write_snapshot(staging)
        _replace_dir(staging, persist_dir)
        if mmap:
            self._adopt_mmap(persist_dir)
        else:
            self.persist_dir = persist_dir

    def _write_snapshot(self, persist_dir: Path) -> None:
        if self._bm25 is None:
            raise RuntimeError("cannot persist an empty BM25 index")

        def corpus():
            for doc in self._docs:
                yield {"id": doc.doc_id, **doc.payload}

        self._bm25.save(str(persist_dir), corpus=corpus(), show_progress=False)
        (persist_dir / _META_NAME).write_text(
            json.dumps(
                {
                    "collection": self.collection,
                    "point_count": self.point_count,
                    "last_sync": self.last_sync,
                }
            ),
            encoding="utf-8",
        )

    def _adopt_mmap(self, persist_dir: Path) -> None:
        """Drop the heap copy and reopen the snapshot as numpy/jsonl mmaps."""
        collection = self.collection
        last_sync = self.last_sync
        point_count = self.point_count
        old_docs = self._docs
        with self._lock:
            self._docs = []
            self._bm25 = None
        del old_docs
        gc.collect()
        loaded = SparseIndex.load_mmap(persist_dir, collection)
        if loaded is None:
            raise RuntimeError(f"mmap load failed for {persist_dir}")
        with self._lock:
            self._bm25 = loaded._bm25
            self._docs = loaded._docs
            self.point_count = loaded.point_count or point_count
            self.last_sync = loaded.last_sync or last_sync
            self.mmap = True
            self.persist_dir = persist_dir
        del loaded

    @classmethod
    def load_mmap(cls, persist_dir: Path, collection: str) -> SparseIndex | None:
        persist_dir = Path(persist_dir)
        if not (persist_dir / _PARAMS_NAME).is_file():
            return None
        bm25 = bm25s.BM25.load(
            str(persist_dir),
            mmap=True,
            load_corpus=True,
            show_progress=False,
        )
        corpus = bm25.corpus
        if corpus is None:
            return None
        # retrieve() prefers self.corpus and would return dicts instead of row
        # ids. We look up payloads ourselves so search stays one code path.
        bm25.corpus = None
        meta: dict[str, Any] = {}
        meta_path = persist_dir / _META_NAME
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = cls()
        index._bm25 = bm25
        index._docs = corpus
        index.collection = str(meta.get("collection") or collection)
        index.point_count = int(meta.get("point_count") or len(corpus))
        index.last_sync = float(meta.get("last_sync") or persist_dir.stat().st_mtime)
        index.mmap = True
        index.persist_dir = persist_dir
        return index

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if self._bm25 is None or self.point_count <= 0:
                return []
            docs = self._docs
            bm25 = self._bm25
            count = self.point_count

        if limit <= 0:
            return []
        query_tokens = bm25s.tokenize(query, **_TOKENIZE_KWARGS)
        # retrieve() raises when k exceeds the corpus size.
        k = min(limit, count)
        indices, scores = bm25.retrieve(query_tokens, k=k, show_progress=False)

        results: list[dict[str, Any]] = []
        for position, score in zip(indices[0], scores[0]):
            # Every IDF is non-negative here, so a positive score means the doc
            # shares at least one query term. Zeros are padding, not matches.
            if score <= 0:
                continue
            with self._corpus_lock:
                doc = docs[int(position)]
            results.append(_hit_from_doc(doc, float(score), int(position)))
        return results

    def rebuild(self, collection: str, points: list[dict[str, Any]]) -> int:
        fresh = SparseIndex(n_threads=self.n_threads)
        fresh.add_points(points)
        return fresh.finalize(collection)


class IndexRegistry:
    """One BM25 index per Qdrant collection name."""

    def __init__(
        self,
        persist_root: Path | None = None,
        mmap: bool = True,
        n_threads: int = 1,
    ) -> None:
        self._lock = threading.Lock()
        self._indexes: dict[str, SparseIndex] = {}
        self.persist_root = Path(persist_root) if persist_root else None
        self.mmap = mmap
        self.n_threads = max(1, n_threads)

    def persist_dir_for(self, collection: str) -> Path | None:
        if self.persist_root is None:
            return None
        return collection_persist_dir(self.persist_root, collection)

    def install(self, collection: str, index: SparseIndex) -> int:
        with self._lock:
            old = self._indexes.pop(collection, None)
            self._indexes[collection] = index
        if old is not None:
            old.close()
        gc.collect()
        return index.point_count

    def load_persisted(self, collection: str) -> int:
        persist_dir = self.persist_dir_for(collection)
        if persist_dir is None:
            return 0
        index = SparseIndex.load_mmap(persist_dir, collection)
        if index is None:
            return 0
        return self.install(collection, index)

    def rebuild(self, collection: str, points: list[dict[str, Any]]) -> int:
        index = SparseIndex(n_threads=self.n_threads)
        index.add_points(points)
        index.finalize(
            collection,
            persist_dir=self.persist_dir_for(collection),
            mmap=self.mmap,
        )
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

    def mmap_enabled(self, collection: str) -> bool:
        with self._lock:
            index = self._indexes.get(collection)
        return bool(index.mmap) if index else False

    def persist_path(self, collection: str) -> str:
        with self._lock:
            index = self._indexes.get(collection)
        if index is None or index.persist_dir is None:
            return ""
        return str(index.persist_dir)
