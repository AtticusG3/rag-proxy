"""Deterministic and optional LLM query rewrite for retrieval."""

from __future__ import annotations

import re

from rag_proxy.clients.llama_swap import resolve_intent_model, rewrite_query_via_model
from rag_proxy.config import settings
from rag_proxy.context import RequestContext, RetrievalDecision

_LITERAL_PATTERNS = [
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(r"v?\d+\.\d+\.\d+(?:[-+][\w.]+)?"),
    re.compile(r"/[\w./-]+"),
    re.compile(r"\\[\w.\\-]+"),
]

_GLOSSARY = {
    "k8s": "kubernetes",
    "omv": "openmediavault",
}


def _extract_literals(text: str) -> list[str]:
    """Find IPs, paths, versions, and UUIDs in text."""
    found: list[str] = []
    for pat in _LITERAL_PATTERNS:
        found.extend(pat.findall(text))
    return found


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of whitespace token sets."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_safe_rewrite(original: str, candidate: str) -> bool:
    """True when rewrite preserves literals and length bounds."""
    if len(candidate) > len(original) * 1.5:
        return False
    if _token_overlap(original, candidate) < 0.3:
        return False
    return all(lit in candidate for lit in _extract_literals(original))


def rewrite_query_deterministic(query: str) -> str:
    """Expand glossary abbreviations without dropping literals."""
    literals = _extract_literals(query)
    out = query.strip()
    for abbr, full in _GLOSSARY.items():
        out = re.sub(rf"\b{re.escape(abbr)}\b", full, out, flags=re.I)
    if len(out) > len(query) * 1.5:
        return query
    if _token_overlap(query, out) < 0.3:
        return query
    for lit in literals:
        if lit not in out:
            return query
    return out


def _rewrite_query_from_dict(data: dict) -> str | None:
    """Parse query string from rewrite model JSON."""
    q = data.get("query")
    if not isinstance(q, str):
        return None
    q = q.strip()
    return q or None


async def run_rewrite(ctx: RequestContext) -> None:
    """Set ctx.retrieval_query via rules and optional LLM."""
    if not ctx.query_text:
        return
    if ctx.retrieval == RetrievalDecision.SKIP:
        return

    rewritten = rewrite_query_deterministic(ctx.query_text)
    ctx.stage_trace.append("rewrite:deterministic")

    if settings.enable_query_rewrite_llm and settings.intent_model:
        model = await resolve_intent_model()
        if model:
            data = await rewrite_query_via_model(
                model,
                ctx.query_text,
                settings.intent_timeout_ms,
                base_url=settings.intent_base_url(),
            )
            if data:
                llm_q = _rewrite_query_from_dict(data)
                if llm_q and _is_safe_rewrite(ctx.query_text, llm_q):
                    rewritten = llm_q
                    ctx.stage_trace.append("rewrite:llm")

    ctx.retrieval_query = rewritten
