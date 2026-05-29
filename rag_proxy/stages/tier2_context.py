"""Dedupe, budget, and inject context."""

from __future__ import annotations

import hashlib
import re

from rag_proxy.clients.bundle import ClientBundle
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.legacy_rag import inject_context, user_message_text

_CONSTRAINT = re.compile(r"\b(ERROR|FATAL|failed|must not|dependency)\b", re.I)


def _norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def dedupe_chunks(hits: list[ChunkHit], enable_semantic: bool) -> list[ChunkHit]:
    seen: set[str] = set()
    out: list[ChunkHit] = []
    for h in sorted(hits, key=lambda x: len(x.text), reverse=True):
        key = _norm_hash(h.text)
        if key in seen:
            continue
        if any(h.text != o.text and h.text in o.text for o in out):
            continue
        if any(o.text != h.text and o.text in h.text for o in out):
            continue
        seen.add(key)
        out.append(h)
    return out


def estimate_message_chars(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += len(user_message_text(m))
        if isinstance(m.get("content"), str):
            total += len(m["content"])
    return total


def apply_context_budget(hits: list[ChunkHit], budget_chars: int) -> list[ChunkHit]:
    if budget_chars <= 0:
        return []
    kept: list[ChunkHit] = []
    used = 0
    sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
    for h in sorted_hits:
        lines = h.text.splitlines()
        priority = [ln for ln in lines if _CONSTRAINT.search(ln)]
        body = h.text
        piece = body
        if used + len(piece) > budget_chars:
            if priority:
                piece = "\n".join(priority)
            if used + len(piece) > budget_chars:
                piece = piece[: max(0, budget_chars - used)]
        if not piece.strip():
            continue
        if used + len(piece) > budget_chars:
            break
        kept.append(ChunkHit(id=h.id, text=piece, score=h.score, source=h.source, metadata=h.metadata))
        used += len(piece) + 8
    return kept


def resolve_inject_budget_chars(ctx: RequestContext, clients: ClientBundle) -> int:
    model_id = ctx.requested_model or ""
    ctx_limit = clients.model_registry.resolve_context_limit(model_id)
    if ctx_limit and ctx_limit < 100000:
        token_budget = int(ctx_limit * settings.context_budget_ratio)
        char_budget = token_budget * 4
    else:
        char_budget = settings.context_fallback_chars

    existing = estimate_message_chars(ctx.messages)
    reserve = settings.default_completion_reserve * 4
    return max(0, char_budget - existing - reserve)


async def run_context_assembly(ctx: RequestContext, clients: ClientBundle) -> None:
    if not ctx.chunk_texts and not ctx.hits:
        return

    hits = ctx.hits if ctx.hits else [
        ChunkHit(id=str(i), text=t, score=1.0) for i, t in enumerate(ctx.chunk_texts)
    ]
    hits = dedupe_chunks(hits, settings.enable_semantic_dedupe)
    budget = resolve_inject_budget_chars(ctx, clients)
    hits = apply_context_budget(hits, budget)
    ctx.hits = hits
    ctx.chunk_texts = [h.text for h in hits if h.text]

    if ctx.chunk_texts:
        ctx.messages = inject_context(ctx.messages, ctx.chunk_texts)
        ctx.injected_tokens_est = sum(len(c) for c in ctx.chunk_texts) // 4
        ctx.stage_trace.append(f"inject:{len(ctx.chunk_texts)}")
