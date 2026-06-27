"""Promotion gates for captured RAG improvement records."""

import asyncio

from rag_proxy import rag_corpus_promoter as promoter
from rag_proxy.config import settings


def _record(answer: str = "Use systemd to run rag-proxy and check journal logs.") -> dict:
    return {
        "record_type": "rag_turn",
        "ts": "2026-01-01T00:00:00Z",
        "trace_id": "trace-1",
        "conversation_id": "conv-1",
        "retrieval": "full",
        "chunks_injected": 1,
        "qa_pair": {"question": "How do I deploy rag-proxy?", "answer": answer},
    }


def test_should_promote_rag_record_requires_flag_and_quality_gates(monkeypatch):
    """Derived corpus entries should only come from real, useful Q&A turns."""
    monkeypatch.setattr(settings, "enable_rag_corpus_auto_ingest", False)
    assert not promoter.should_promote_rag_record(_record())

    monkeypatch.setattr(settings, "enable_rag_corpus_auto_ingest", True)
    monkeypatch.setattr(settings, "rag_corpus_min_answer_chars", 10)
    monkeypatch.setattr(settings, "rag_corpus_require_chunks", True)
    assert promoter.should_promote_rag_record(_record())

    no_chunks = _record()
    no_chunks["chunks_injected"] = 0
    assert not promoter.should_promote_rag_record(no_chunks)

    meta = _record()
    meta["qa_pair"]["question"] = "### Task:\nSuggest follow-up questions."
    assert not promoter.should_promote_rag_record(meta)

    skipped = _record()
    skipped["retrieval"] = "skip"
    assert not promoter.should_promote_rag_record(skipped)


def test_promote_rag_record_embeds_and_upserts_to_derived_collection(monkeypatch):
    """Auto-ingest should upsert a Q&A point into the configured derived collection."""
    monkeypatch.setattr(settings, "enable_rag_corpus_auto_ingest", True)
    monkeypatch.setattr(settings, "rag_corpus_min_answer_chars", 10)
    monkeypatch.setattr(settings, "rag_corpus_require_chunks", False)
    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant")
    monkeypatch.setattr(settings, "rag_corpus_collection", "derived")
    upserts = []

    async def fake_embedding(text: str):
        assert text.startswith("Q: How do I deploy rag-proxy?")
        return [0.1, 0.2, 0.3]

    def fake_upsert(qdrant_url: str, collection: str, point: dict):
        upserts.append((qdrant_url, collection, point))

    monkeypatch.setattr(promoter, "get_embedding", fake_embedding)
    monkeypatch.setattr(promoter, "_upsert_point", fake_upsert)

    promoted = asyncio.run(promoter.promote_rag_record(_record()))

    assert promoted
    assert upserts[0][0] == "http://qdrant"
    assert upserts[0][1] == "derived"
    assert upserts[0][2]["payload"]["source"] == "capture:conv-1"
    assert upserts[0][2]["payload"]["record_type"] == "conversation_qa"
