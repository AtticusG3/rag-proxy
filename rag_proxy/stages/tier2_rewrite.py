"""Deterministic query rewrite for retrieval."""

from __future__ import annotations

import re

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
    found: list[str] = []
    for pat in _LITERAL_PATTERNS:
        found.extend(pat.findall(text))
    return found


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def rewrite_query_deterministic(query: str) -> str:
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


async def run_rewrite(ctx: RequestContext) -> None:
    if not settings.enable_query_rewrite or not ctx.query_text:
        return
    if ctx.retrieval == RetrievalDecision.SKIP:
        return

    rewritten = rewrite_query_deterministic(ctx.query_text)
    ctx.retrieval_query = rewritten
    ctx.stage_trace.append("rewrite:deterministic")
