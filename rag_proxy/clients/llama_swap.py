"""llama-swap HTTP helpers."""

from __future__ import annotations

import json
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


def parse_json_object(raw: str) -> dict | None:
    """Extract the first JSON object from model output."""
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        data = json.loads(raw[start:end])
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


async def chat_completion_content(
    model: str,
    messages: list[dict],
    timeout_ms: int,
    *,
    max_tokens: int | None = None,
) -> str | None:
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
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
        log.warning(f"Chat completion failed: {e}")
    return None


async def chat_json_completion(
    model: str,
    system: str,
    user: str,
    timeout_ms: int,
    *,
    max_tokens: int,
) -> dict | None:
    raw = await chat_completion_content(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user[:500]},
        ],
        timeout_ms,
        max_tokens=max_tokens,
    )
    if not raw:
        return None
    return parse_json_object(raw)


async def classify_intent_via_model(model: str, prompt: str, timeout_ms: int) -> str | None:
    """Call tiny classifier model; return raw content or None."""
    system = (
        "Classify the user query. Reply with JSON only: "
        '{"intent":"<label>","confidence":0.0-1.0}. '
        "Labels: simple_chat, infra_debug, code_generation, code_review, "
        "research, summarization, troubleshooting, log_analysis, planning, "
        "creative, retrieval_heavy, reasoning_heavy, unknown."
    )
    data = await chat_json_completion(
        model,
        system,
        prompt,
        timeout_ms,
        max_tokens=64,
    )
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False)


async def rewrite_query_via_model(model: str, prompt: str, timeout_ms: int) -> str | None:
    """Rewrite query for retrieval; return raw content or None."""
    system = (
        "Rewrite the user query for knowledge-base search. "
        'Reply with JSON only: {"query":"<rewritten>"}. '
        "Preserve IPs, paths, versions, and error codes."
    )
    data = await chat_json_completion(
        model,
        system,
        prompt,
        timeout_ms,
        max_tokens=128,
    )
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False)
