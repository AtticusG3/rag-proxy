"""Promote high-signal captured Q&A turns into a derived Qdrant collection."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from ingest.qdrant_writer import build_point, ensure_collection, upsert_points
from rag_proxy.config import settings
from rag_proxy.legacy_rag import get_embedding, is_embeddable_user_query

log = logging.getLogger("rag-proxy")


def should_promote_rag_record(record: dict[str, Any]) -> bool:
    """Return true when a captured RAG turn is suitable for derived retrieval."""
    if not settings.enable_rag_corpus_auto_ingest:
        return False
    if record.get("record_type") != "rag_turn":
        return False
    if record.get("retrieval") == "skip":
        return False
    if settings.rag_corpus_require_chunks and int(record.get("chunks_injected") or 0) <= 0:
        return False
    qa_pair = record.get("qa_pair")
    if not isinstance(qa_pair, dict):
        return False
    question = str(qa_pair.get("question") or "").strip()
    answer = str(qa_pair.get("answer") or "").strip()
    if not question or not is_embeddable_user_query(question):
        return False
    if len(answer) < settings.rag_corpus_min_answer_chars:
        return False
    return True


async def promote_rag_record(record: dict[str, Any]) -> bool:
    """Embed and upsert a captured Q&A pair into the derived RAG corpus."""
    if not should_promote_rag_record(record):
        return False
    qa_pair = record["qa_pair"]
    question = str(qa_pair["question"]).strip()
    answer = str(qa_pair["answer"]).strip()
    text = f"Q: {question}\n\nA: {answer}"
    embedding = await get_embedding(text)
    if embedding is None:
        return False

    source = f"capture:{record.get('conversation_id') or record.get('trace_id') or 'unknown'}"
    point = build_point(
        text=text,
        source=source,
        title="Captured conversation turn",
        chunk_idx=_stable_chunk_idx(record),
        embedding=embedding,
        extra={
            "capture_trace_id": record.get("trace_id"),
            "conversation_id": record.get("conversation_id"),
            "record_type": "conversation_qa",
        },
    )
    try:
        await asyncio.to_thread(
            _upsert_point,
            settings.qdrant_url,
            settings.rag_corpus_collection,
            point,
        )
        return True
    except Exception as e:
        log.warning("RAG corpus auto-ingest failed: %s", e)
        return False


def _stable_chunk_idx(record: dict[str, Any]) -> int:
    seed = f"{record.get('trace_id') or ''}:{record.get('ts') or ''}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return int(digest, 16)


def _upsert_point(qdrant_url: str, collection: str, point: dict[str, Any]) -> None:
    ensure_collection(qdrant_url, collection)
    upsert_points(qdrant_url, collection, [point])
