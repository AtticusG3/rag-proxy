"""Dedupe, budget, and inject context."""

from __future__ import annotations

import hashlib
import re

from rag_proxy.registry.models import ModelRegistry
from rag_proxy.config import settings
from rag_proxy.context import ChunkHit, RequestContext
from rag_proxy.legacy_rag import inject_context, user_message_text
from rag_proxy.token_estimate import count_tokens, truncate_to_tokens, uses_tokenizer

_CONSTRAINT = re.compile(r"\b(ERROR|FATAL|failed|must not|dependency)\b", re.I)
_CHUNK_SEPARATOR_TOKENS = 2


def _norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def dedupe_chunks(hits: list[ChunkHit], enable_semantic: bool) -> list[ChunkHit]:
    seen: set[str] = set()
    out: list[ChunkHit] = []
    for h in sorted(hits, key=lambda x: len(x.text), reverse=True):
        key = _norm_hash(h.text)
        if key in seen:
            continue
        if enable_semantic:
            if any(h.text != o.text and h.text in o.text for o in out):
                continue
            if any(o.text != h.text and o.text in h.text for o in out):
                continue
        seen.add(key)
        out.append(h)
    return out


def estimate_message_chars(messages: list[dict]) -> int:
    """Legacy char estimate; kept for tests and char-mode budget."""
    total = 0
    for m in messages:
        total += len(user_message_text(m))
    return total


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        else:
            total += count_tokens(user_message_text(m))
    return total


def apply_context_budget(hits: list[ChunkHit], budget: int) -> list[ChunkHit]:
    """Fit hits into budget (tokens when ENABLE_TOKENIZER_ESTIMATE=true, else chars)."""
    if budget <= 0:
        return []
    if uses_tokenizer():
        return _apply_context_budget(
            hits,
            budget,
            measure=count_tokens,
            truncate=truncate_to_tokens,
            separator_cost=_CHUNK_SEPARATOR_TOKENS,
        )
    return _apply_context_budget(
        hits,
        budget,
        measure=len,
        truncate=lambda text, max_units: text[: max(0, max_units)],
        separator_cost=8,
    )


def _apply_context_budget(
    hits: list[ChunkHit],
    budget: int,
    *,
    measure,
    truncate,
    separator_cost: int,
) -> list[ChunkHit]:
    kept: list[ChunkHit] = []
    used = 0
    sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
    for h in sorted_hits:
        lines = h.text.splitlines()
        priority = [ln for ln in lines if _CONSTRAINT.search(ln)]
        piece = h.text
        piece_units = measure(piece)
        if used + piece_units > budget:
            if priority:
                piece = "\n".join(priority)
                piece_units = measure(piece)
            if used + piece_units > budget:
                remaining = max(0, budget - used)
                piece = truncate(piece, remaining)
                piece_units = measure(piece)
        if not piece.strip():
            continue
        if used + piece_units > budget:
            break
        kept.append(ChunkHit(id=h.id, text=piece, score=h.score, source=h.source, metadata=h.metadata))
        used += piece_units + separator_cost
    return kept


def resolve_inject_budget_chars(ctx: RequestContext, registry: ModelRegistry) -> int:
    """Return inject budget in chars (legacy) or tokens when estimate mode is on."""
    model_id = ctx.requested_model or None
    context_tokens = registry.resolve_context_tokens(model_id)
    existing = estimate_message_tokens(ctx.messages) if uses_tokenizer() else estimate_message_chars(ctx.messages)
    reserve = settings.default_completion_reserve

    if uses_tokenizer():
        if context_tokens is not None:
            token_budget = int(context_tokens * settings.context_budget_ratio)
        else:
            token_budget = settings.context_fallback_chars // 4
        return max(0, token_budget - existing - reserve)

    if context_tokens is not None:
        char_budget = int(context_tokens * settings.context_budget_ratio) * 4
    else:
        char_budget = settings.context_fallback_chars
    reserve_chars = reserve * 4
    return max(0, char_budget - existing - reserve_chars)


async def run_context_assembly(ctx: RequestContext, registry: ModelRegistry) -> None:
    if not ctx.hits:
        return

    hits = dedupe_chunks(list(ctx.hits), settings.enable_semantic_dedupe)
    budget = resolve_inject_budget_chars(ctx, registry)
    hits = apply_context_budget(hits, budget)
    ctx.hits = hits

    texts = ctx.chunk_texts
    if texts:
        ctx.messages = inject_context(ctx.messages, texts)
        if uses_tokenizer():
            ctx.injected_tokens_est = sum(count_tokens(t) for t in texts)
        else:
            ctx.injected_tokens_est = sum(len(c) for c in texts) // 4
        ctx.stage_trace.append(f"inject:{len(texts)}")
