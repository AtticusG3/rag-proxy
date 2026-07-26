"""Tests for MemGraphRAG memory construction from extracted chunks."""

from __future__ import annotations

from pathlib import Path

from rag_proxy.memgraphrag.memory import load_memory
from scripts.build_memgraphrag_index import build_memory


def _chunk(chunk_id: str, text: str, head: str, tail: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "triples": [
            {
                "head": head,
                "relation": "works_at",
                "tail": tail,
                "schema": ["Person", "works_at", "Org"],
            }
        ],
    }


def test_facts_link_to_their_own_passage(tmp_path: Path) -> None:
    """Each fact must resolve to the chunk it came from.

    Retrieval walks fact -> passage to build context, so a fact wired to the
    wrong passage silently injects unrelated text.
    """
    db_path = tmp_path / "mem.db"
    build_memory(
        [
            _chunk("c0", "Alice works at Acme.", "Alice", "Acme"),
            _chunk("c1", "Bob works at Globex.", "Bob", "Globex"),
            _chunk("c2", "Carol works at Initech.", "Carol", "Initech"),
        ],
        str(db_path),
    )

    memory = load_memory(db_path)
    assert memory.stats == {"schemas": 1, "facts": 3, "passages": 3}

    by_head = {fact.head: fact for fact in memory.facts.values()}
    for head, expected_chunk in (("Alice", "c0"), ("Bob", "c1"), ("Carol", "c2")):
        fact = by_head[head]
        passages = [memory.passages[pi].chunk_id for pi in fact.passage_indices]
        assert passages == [expected_chunk]


def test_passages_link_back_to_their_facts(tmp_path: Path) -> None:
    """Passage -> fact links drive PPR seeding; they must not all land on chunk 0."""
    db_path = tmp_path / "mem.db"
    build_memory(
        [
            _chunk("c0", "Alice works at Acme.", "Alice", "Acme"),
            _chunk("c1", "Bob works at Globex.", "Bob", "Globex"),
        ],
        str(db_path),
    )

    memory = load_memory(db_path)
    by_chunk = {p.chunk_id: p for p in memory.passages.values()}
    for chunk_id, head in (("c0", "Alice"), ("c1", "Bob")):
        fact_heads = [memory.facts[fi].head for fi in by_chunk[chunk_id].fact_indices]
        assert fact_heads == [head]


def test_chunk_without_triples_still_indexed(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    build_memory(
        [
            {"chunk_id": "c0", "text": "No extractable relations here.", "triples": []},
            _chunk("c1", "Bob works at Globex.", "Bob", "Globex"),
        ],
        str(db_path),
    )

    memory = load_memory(db_path)
    by_chunk = {p.chunk_id: p for p in memory.passages.values()}
    assert by_chunk["c0"].fact_indices == []
    assert len(by_chunk["c1"].fact_indices) == 1
