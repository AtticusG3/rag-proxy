"""Process-local cache for MemGraphRAG memory indexes.

Builds fact adjacency and row-normalized embedding matrices once per SQLite
mtime, so online retrieval avoids per-request graph construction.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_proxy.memgraphrag.memory import ThreeLayerMemory, load_memory

log = logging.getLogger("rag-proxy.memgraphrag.cache")

_lock = threading.Lock()
_cache: dict[str, tuple[float, MemoryIndex]] = {}
_load_locks: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class MemoryIndex:
    """Precomputed structures for MemGraphRAG online retrieval."""

    memory: ThreeLayerMemory
    fact_adj: dict[int, list[int]]
    fact_indices: np.ndarray
    fact_embeddings: np.ndarray


def build_memory_index(memory: ThreeLayerMemory) -> MemoryIndex:
    """Build adjacency and normalized fact embedding matrix from memory."""
    fact_adj: dict[int, list[int]] = {}
    for fi in memory.facts:
        fact_adj[fi] = memory.get_related_fact_indices(fi)

    indices: list[int] = []
    rows: list[np.ndarray] = []
    for fi, fact in memory.facts.items():
        if not fact.embedding:
            continue
        emb = np.array(fact.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            continue
        indices.append(fi)
        rows.append(emb / norm)

    if rows:
        fact_indices = np.array(indices, dtype=np.int32)
        fact_embeddings = np.vstack(rows)
    else:
        fact_indices = np.array([], dtype=np.int32)
        fact_embeddings = np.zeros((0, 0), dtype=np.float32)

    log.info(
        "Built memory index: %d fact nodes, %d embedded facts",
        len(fact_adj),
        len(indices),
    )
    return MemoryIndex(
        memory=memory,
        fact_adj=fact_adj,
        fact_indices=fact_indices,
        fact_embeddings=fact_embeddings,
    )


def _load_index(path: Path) -> MemoryIndex:
    memory = load_memory(path)
    return build_memory_index(memory)


def get_memory_index(db_path: str | Path) -> MemoryIndex:
    """Return cached MemoryIndex, reloading when the SQLite file mtime changes."""
    path = Path(db_path)
    key = str(path.resolve())
    mtime = path.stat().st_mtime if path.exists() else 0.0

    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == mtime:
            log.info(
                "Reusing cached memory index: %s (%d facts)",
                path.name,
                len(cached[1].memory.facts),
            )
            return cached[1]
        load_lock = _load_locks.setdefault(key, threading.Lock())

    with load_lock:
        with _lock:
            cached = _cache.get(key)
            if cached is not None and cached[0] == mtime:
                log.info(
                    "Reusing cached memory index: %s (%d facts)",
                    path.name,
                    len(cached[1].memory.facts),
                )
                return cached[1]

        index = _load_index(path)

        with _lock:
            _cache[key] = (mtime, index)
        return index


def invalidate_memory_index(db_path: str | Path | None = None) -> None:
    """Drop cached index(es). Call after rebuilding the SQLite file in-process."""
    with _lock:
        if db_path is None:
            _cache.clear()
        else:
            key = str(Path(db_path).resolve())
            _cache.pop(key, None)
