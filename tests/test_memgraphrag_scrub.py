"""MemGraphRAG passage scrub when a source document is removed."""

from __future__ import annotations

from rag_proxy.memgraphrag.memory import ThreeLayerMemory


def test_remove_passages_by_chunk_ids_prunes_orphan_facts_and_schemas() -> None:
    """Deleting a document's passages must not leave orphan facts/schemas behind."""
    mem = ThreeLayerMemory()
    schema = mem.add_schema("Person", "knows", "Person")
    keep = mem.add_passage("keep-id", "Alice stays.", fact_indices=[])
    drop = mem.add_passage("drop-id", "Bob leaves.", fact_indices=[])
    shared = mem.add_fact("Alice", "knows", "Carol", schema, keep)
    orphan = mem.add_fact("Bob", "knows", "Dan", schema, drop)
    mem.passages[keep].fact_indices.append(shared)
    mem.passages[drop].fact_indices.append(orphan)

    removed = mem.remove_passages_by_chunk_ids({"drop-id"})

    assert removed == 1
    assert "drop-id" not in mem._chunk_id_to_idx
    assert keep in mem.passages
    assert shared in mem.facts
    assert orphan not in mem.facts
    assert schema in mem.schemas
    assert mem.facts[shared].passage_indices == [keep]


def test_remove_passages_by_chunk_ids_drops_schema_when_last_fact_goes() -> None:
    """A schema used only by the removed document must disappear with it."""
    mem = ThreeLayerMemory()
    schema = mem.add_schema("Org", "owns", "Asset")
    passage = mem.add_passage("only-id", "Org owns asset.", fact_indices=[])
    fact = mem.add_fact("Acme", "owns", "Plant", schema, passage)
    mem.passages[passage].fact_indices.append(fact)

    assert mem.remove_passages_by_chunk_ids({"only-id"}) == 1
    assert mem.passages == {}
    assert mem.facts == {}
    assert mem.schemas == {}
