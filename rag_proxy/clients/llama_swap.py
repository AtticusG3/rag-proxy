"""llama-swap HTTP helpers."""

from __future__ import annotations

import logging

import httpx

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")


async def fetch_models() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{settings.llama_swap_url.rstrip('/')}/v1/models")
            r.raise_for_status()
            data = r.json()
            return data.get("data", [])
    except Exception as e:
        log.warning(f"Model list fetch failed: {e}")
        return []


async def classify_intent_via_model(model: str, prompt: str, timeout_ms: int) -> str | None:
    """Call tiny classifier model; return raw content or None."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify the user query. Reply with JSON only: "
                    '{"intent":"<label>","confidence":0.0-1.0}. '
                    "Labels: simple_chat, infra_debug, code_generation, code_review, "
                    "research, summarization, troubleshooting, log_analysis, planning, "
                    "creative, retrieval_heavy, reasoning_heavy, unknown."
                ),
            },
            {"role": "user", "content": prompt[:500]},
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000.0 + 1) as client:
            r = await client.post(
                f"{settings.llama_swap_url.rstrip('/')}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            choices = r.json().get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content")
    except Exception as e:
        log.warning(f"Intent model call failed: {e}")
    return None


async def rewrite_query_via_model(model: str, prompt: str, timeout_ms: int) -> str | None:
    """Rewrite query for retrieval; return raw content or None."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite the user query for knowledge-base search. "
                    'Reply with JSON only: {"query":"<rewritten>"}. '
                    "Preserve IPs, paths, versions, and error codes."
                ),
            },
            {"role": "user", "content": prompt[:500]},
        ],
        "temperature": 0,
        "max_tokens": 128,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000.0 + 1) as client:
            r = await client.post(
                f"{settings.llama_swap_url.rstrip('/')}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            choices = r.json().get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content")
    except Exception as e:
        log.warning(f"Query rewrite model call failed: {e}")
    return None
