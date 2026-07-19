"""llama-swap HTTP helpers."""

from __future__ import annotations

import json
import logging
import time

import httpx

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")

_auto_model_cache: dict[str, tuple[float, str | None]] = {}


async def fetch_models(base_url: str | None = None) -> list[dict]:
    """Fetch /v1/models from an OpenAI-compatible endpoint; fail-open to []."""
    url = (base_url or settings.llama_swap_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{url}/v1/models")
            r.raise_for_status()
            data = r.json()
            return data.get("data", [])
    except Exception as e:
        log.warning(f"Model list fetch failed: {e}")
        return []


_READY_STATES = {"ready", ""}


def _running_models(data: object) -> list[str]:
    """Extract ready model ids from a llama-swap /running response.

    Handles the shapes shipped across llama-swap builds:
    - ``{}`` (nothing loaded) -> []
    - ``{"model": "X", "state": "ready"}`` (single-object) -> ["X"] if ready
    - ``{"running": [{"model": "X", "state": "ready"}, ...]}`` -> ready ids
    - ``{"running": ["X", "Y"]}`` (names only) -> ["X", "Y"]
    """
    if not isinstance(data, dict):
        return []
    if "model" in data and "running" not in data:
        entries: object = [data]
    else:
        entries = data.get("running")
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry:
                out.append(entry)
        elif isinstance(entry, dict):
            mid = entry.get("model") or entry.get("id")
            state = str(entry.get("state", "ready")).lower()
            if mid and state in _READY_STATES:
                out.append(mid)
    return out


async def fetch_running_model(base_url: str | None = None) -> str | None:
    """Return the currently loaded model id from llama-swap's /running; None if idle."""
    url = (base_url or settings.llama_swap_url).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{url}/running")
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        log.warning(f"Running model fetch failed: {e}")
        return None
    running = _running_models(data)
    return running[0] if running else None


async def resolve_intent_model() -> str | None:
    """Resolve the configured intent model, expanding 'auto' to the loaded model.

    When INTENT_MODEL is ``auto`` the model currently loaded on the intent
    endpoint (llama-swap /running) is used, so intent/rewrite piggyback on the
    warm model instead of forcing a swap. Returns None when nothing is loaded.
    Results are cached per endpoint for INTENT_MODEL_AUTO_TTL_SEC.
    """
    configured = settings.intent_model.strip()
    if configured.lower() != "auto":
        return configured or None

    base = settings.intent_base_url()
    now = time.monotonic()
    cached = _auto_model_cache.get(base)
    if cached and now - cached[0] < settings.intent_model_auto_ttl_sec:
        return cached[1]

    resolved = await fetch_running_model(base)
    _auto_model_cache[base] = (now, resolved)
    return resolved


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
    base_url: str | None = None,
) -> str | None:
    """Post chat completion and return assistant content."""
    url = (base_url or settings.llama_swap_url).rstrip("/")
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
                f"{url}/v1/chat/completions",
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
    base_url: str | None = None,
) -> dict | None:
    """Chat completion returning the first parsed JSON object."""
    raw = await chat_completion_content(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user[:500]},
        ],
        timeout_ms,
        max_tokens=max_tokens,
        base_url=base_url,
    )
    if not raw:
        return None
    return parse_json_object(raw)


async def classify_intent_via_model(
    model: str, prompt: str, timeout_ms: int, *, base_url: str | None = None
) -> dict | None:
    """Call tiny classifier model; return parsed JSON object or None."""
    system = (
        "Classify the user query. Reply with JSON only: "
        '{"intent":"<label>","confidence":0.0-1.0}. '
        "Labels: simple_chat, infra_debug, code_generation, code_review, "
        "research, summarization, troubleshooting, log_analysis, planning, "
        "creative, retrieval_heavy, reasoning_heavy, unknown."
    )
    return await chat_json_completion(
        model,
        system,
        prompt,
        timeout_ms,
        max_tokens=64,
        base_url=base_url,
    )


async def rewrite_query_via_model(
    model: str, prompt: str, timeout_ms: int, *, base_url: str | None = None
) -> dict | None:
    """Rewrite query for retrieval; return parsed JSON object or None."""
    system = (
        "Rewrite the user query for knowledge-base search. "
        'Reply with JSON only: {"query":"<rewritten>"}. '
        "Preserve IPs, paths, versions, and error codes."
    )
    return await chat_json_completion(
        model,
        system,
        prompt,
        timeout_ms,
        max_tokens=128,
        base_url=base_url,
    )
