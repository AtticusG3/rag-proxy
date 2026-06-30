"""Build and enqueue transcript capture records."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any

from rag_proxy.capture_writer import enqueue_records
from rag_proxy.config import settings
from rag_proxy.context import RequestContext
from rag_proxy.message_sanitize import is_exportable_turn, sanitize_client_messages
from rag_proxy.response_parse import ParsedCompletion, parse_chat_completion

log = logging.getLogger("rag-proxy")


def capture_enabled(headers: dict[str, str] | None = None) -> bool:
    """True when this request should be transcript-captured."""
    if not settings.enable_transcript_capture:
        return False
    hdr = {k.lower(): v for k, v in (headers or {}).items()}
    if settings.transcript_header_opt_in:
        if hdr.get("x-capture-log", "").lower() not in ("1", "true", "yes", "on"):
            return False
    sample_rate = max(0.0, min(1.0, settings.transcript_sample_rate))
    if sample_rate <= 0:
        return False
    if sample_rate < 1.0 and random.random() >= sample_rate:
        return False
    return True


def build_finetune_record(
    *,
    original_messages: list[dict[str, Any]],
    ctx: RequestContext,
    parsed: ParsedCompletion,
    path: str,
    stream: bool,
    timestamp: str | None = None,
) -> dict[str, Any] | None:
    """Build one supervised fine-tuning turn, or None when not exportable."""
    assistant_text = parsed.assistant_text
    query_text = ctx.query_text
    if not parsed.assistant_message or not is_exportable_turn(query_text, assistant_text):
        return None
    return {
        "record_type": "finetune_turn",
        "ts": timestamp or _timestamp(),
        "trace_id": ctx.trace_id,
        "conversation_id": ctx.conversation_id,
        "path": path.rstrip("/"),
        "model": ctx.selected_model or ctx.requested_model,
        "stream": stream,
        "messages": sanitize_client_messages(
            original_messages,
            strip_proxy_artefacts=settings.transcript_strip_proxy_artefacts,
        ),
        "assistant": parsed.assistant_message,
        "finish_reason": parsed.finish_reason,
        "usage": parsed.usage,
    }


def build_rag_record(
    *,
    original_messages: list[dict[str, Any]],
    ctx: RequestContext,
    parsed: ParsedCompletion,
    path: str,
    stream: bool,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one RAG improvement record for analysis and promotion."""
    assistant_text = parsed.assistant_text or ""
    question = ctx.effective_query() or ctx.query_text
    preview_chars = max(0, settings.transcript_hit_preview_chars)
    hits = [
        {
            "id": hit.id,
            "score": round(hit.score, 3),
            "source": hit.source,
            "text_preview": hit.text[:preview_chars],
            "metadata": hit.metadata,
        }
        for hit in ctx.hits
    ]
    record: dict[str, Any] = {
        "record_type": "rag_turn",
        "ts": timestamp or _timestamp(),
        "trace_id": ctx.trace_id,
        "conversation_id": ctx.conversation_id,
        "path": path.rstrip("/"),
        "model": ctx.selected_model or ctx.requested_model,
        "stream": stream,
        "query_text": ctx.query_text,
        "retrieval_query": ctx.retrieval_query,
        "retrieval": ctx.retrieval.value,
        "intent": ctx.intent.value,
        "tier": ctx.tier.value,
        "gating_would_skip": ctx.gating_would_skip,
        "chunks_injected": len(ctx.chunk_texts),
        "scores": [round(hit.score, 3) for hit in ctx.hits],
        "hits": hits,
        "assistant_text": assistant_text,
        "qa_pair": {"question": question, "answer": assistant_text}
        if question and assistant_text
        else None,
        "rag_mode": ctx.rag_mode_header,
        "errors": ctx.errors,
        "messages": sanitize_client_messages(
            original_messages,
            strip_proxy_artefacts=settings.transcript_strip_proxy_artefacts,
        ),
        "parse_error": parsed.parse_error,
    }
    return record


def build_capture_records(
    *,
    original_messages: list[dict[str, Any]],
    ctx: RequestContext,
    response_body: bytes,
    path: str,
    stream: bool,
) -> list[dict[str, Any]]:
    """Build all transcript records for a completed chat response."""
    parsed = parse_chat_completion(path, response_body, stream=stream)
    ts = _timestamp()
    records = [
        build_rag_record(
            original_messages=original_messages,
            ctx=ctx,
            parsed=parsed,
            path=path,
            stream=stream,
            timestamp=ts,
        )
    ]
    finetune = build_finetune_record(
        original_messages=original_messages,
        ctx=ctx,
        parsed=parsed,
        path=path,
        stream=stream,
        timestamp=ts,
    )
    if finetune is not None:
        records.append(finetune)
    return records


def capture_chat_response(
    *,
    original_messages: list[dict[str, Any]],
    ctx: RequestContext,
    response_body: bytes,
    path: str,
    stream: bool,
) -> None:
    """Build and queue capture records for a completed response."""
    try:
        enqueue_records(
            build_capture_records(
                original_messages=original_messages,
                ctx=ctx,
                response_body=response_body,
                path=path,
                stream=stream,
            )
        )
    except Exception as e:
        log.warning("transcript capture failed: %s", e)


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
