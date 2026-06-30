"""MemGraphRAG memory index cache: hit/miss and mtime invalidation."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np

from rag_proxy.memgraphrag.cache import (
    build_memory_index,
    get_memory_index,
    invalidate_memory_index,
)
from rag_proxy.memgraphrag.memory import ThreeLayerMemory


def _memory_with_embedding() -> ThreeLayerMemory:
    mem = ThreeLayerMemory()
    schema_idx = mem.add_schema("Person", "knows", "Person")
    passage_idx = mem.add_passage("chunk-1", "Alice knows Bob.", fact_indices=[])
    fact_idx = mem.add_fact("Alice", "knows", "Bob", schema_idx, passage_idx)
    mem.passages[passage_idx].fact_indices.append(fact_idx)
    mem.set_fact_embedding(fact_idx, [3.0, 4.0])
    return mem


def test_build_memory_index_row_normalizes_embeddings() -> None:
    """Embedded facts are stored as unit rows for cosine via matmul."""
    index = build_memory_index(_memory_with_embedding())
    assert index.fact_indices.tolist() == [0]
    assert index.fact_embeddings.shape == (1, 2)
    row_norm = float(np.linalg.norm(index.fact_embeddings[0]))
    assert abs(row_norm - 1.0) < 1e-5
    assert 0 in index.fact_adj


def test_get_memory_index_cache_hit(tmp_path) -> None:
    """Second call with unchanged mtime returns the same MemoryIndex object."""
    mem = _memory_with_embedding()
    db_path = tmp_path / "mem.sqlite"
    mem.save(db_path)
    invalidate_memory_index()

    first = get_memory_index(db_path)
    with patch("rag_proxy.memgraphrag.cache.log") as cache_log:
        second = get_memory_index(db_path)
        cache_log.info.assert_called()
    assert first is second


def test_get_memory_index_reload_on_mtime_change(tmp_path) -> None:
    """When SQLite mtime changes, cache reloads a new index."""
    mem = _memory_with_embedding()
    db_path = tmp_path / "mem.sqlite"
    mem.save(db_path)
    invalidate_memory_index()

    first = get_memory_index(db_path)
    assert first.memory.facts[0].head == "Alice"

    mem.facts[0].head = "Carol"
    mem.save(db_path)
    # Ensure mtime advances on filesystems with coarse resolution
    time.sleep(0.02)

    second = get_memory_index(db_path)
    assert second is not first
    assert second.memory.facts[0].head == "Carol"


def test_invalidate_memory_index_forces_reload(tmp_path) -> None:
    """Explicit invalidation drops the entry so the next call rebuilds."""
    mem = _memory_with_embedding()
    db_path = tmp_path / "mem.sqlite"
    mem.save(db_path)
    invalidate_memory_index()

    first = get_memory_index(db_path)
    invalidate_memory_index(db_path)

    with patch("rag_proxy.memgraphrag.cache._load_index") as load_index:
        load_index.return_value = first
        second = get_memory_index(db_path)

    load_index.assert_called_once()
    assert second is first


def test_concurrent_cold_cache_load_builds_once(tmp_path) -> None:
    """Parallel get_memory_index on cold cache must build the index only once."""
    mem = _memory_with_embedding()
    db_path = tmp_path / "mem.sqlite"
    mem.save(db_path)
    invalidate_memory_index()

    build_count = 0
    build_lock = threading.Lock()
    release = threading.Event()

    def slow_load(path):
        nonlocal build_count
        with build_lock:
            build_count += 1
        release.wait(timeout=5.0)
        return build_memory_index(mem)

    with patch("rag_proxy.memgraphrag.cache._load_index", side_effect=slow_load):
        results: list[object] = []
        errors: list[Exception] = []

        def load_once() -> None:
            try:
                results.append(get_memory_index(db_path))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=load_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        release.set()
        for thread in threads:
            thread.join(timeout=10.0)

    assert not errors
    assert build_count == 1
    assert len(results) == 2
    assert results[0] is results[1]
