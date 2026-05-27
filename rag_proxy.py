#!/usr/bin/env python3
from __future__ import annotations

"""
rag_proxy.py — RAG middleware for llama-swap

Intercepts /v1/chat/completions requests, embeds the user query via nomic-embed,
retrieves relevant chunks from Qdrant (NOMAD knowledge base), injects them as
context, then forwards the augmented request to llama-swap.

All other endpoints pass through transparently. API key validation is handled
entirely by llama-swap downstream — this proxy is auth-transparent.

Flow:
    Client → rag_proxy:8088 → llama-swap:8080
                    ↓
              nomic-embed:8089  →  Qdrant:6333

Usage:
    python3 rag_proxy.py

Environment variables (all optional, defaults shown):
    LLAMA_SWAP_URL       http://127.0.0.1:8080
    EMBED_URL            http://127.0.0.1:8089
    QDRANT_URL           http://192.168.1.x:6333     ← SET THIS to your omv IP
    QDRANT_COLLECTION    nomad_knowledge_base
    TOP_K                5
    SIMILARITY_THRESHOLD 0.65
    PROXY_HOST           0.0.0.0
    PROXY_PORT           8088
    LOG_LEVEL            INFO
    EMBED_MAX_CHARS      2000   (tail-truncate; keep under nomic -ub batch size)
    EMBED_RETRIES        2
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLAMA_SWAP_URL       = os.getenv("LLAMA_SWAP_URL",       "http://127.0.0.1:8080")
EMBED_URL            = os.getenv("EMBED_URL",            "http://127.0.0.1:8089")
QDRANT_URL           = os.getenv("QDRANT_URL",           "http://192.168.1.36:6333")
QDRANT_COLLECTION    = os.getenv("QDRANT_COLLECTION",    "nomad_knowledge_base")
TOP_K                = int(os.getenv("TOP_K",            "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
PROXY_HOST           = os.getenv("PROXY_HOST",           "0.0.0.0")
PROXY_PORT           = int(os.getenv("PROXY_PORT",       "8088"))
EMBED_MAX_CHARS      = int(os.getenv("EMBED_MAX_CHARS",  "2000"))
EMBED_RETRIES        = int(os.getenv("EMBED_RETRIES",    "2"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-proxy")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Proxy", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def prepare_embed_text(text: str, max_chars: int) -> str:
    """Trim to a safe size for llama-server embed batch (-ub defaults to 512 tokens)."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    log.warning(f"Embed input truncated (tail) {len(text)} -> {max_chars} chars")
    return text[-max_chars:]


def _embed_input_too_large(response_text: str) -> bool:
    return "too large to process" in response_text


async def get_embedding(text: str) -> list[float] | None:
    """Embed text via the standalone nomic-embed server."""
    # Try full limit first; on batch overflow, retry with a smaller tail slice.
    char_limits = [EMBED_MAX_CHARS]
    if EMBED_MAX_CHARS > 1200:
        char_limits.append(1200)

    async with httpx.AsyncClient(timeout=30) as client:
        for max_chars in char_limits:
            chunk = prepare_embed_text(text, max_chars)
            payload = {"model": "nomic-embed-text-v1.5", "input": chunk}
            saw_too_large = False

            for attempt in range(EMBED_RETRIES):
                if attempt:
                    await asyncio.sleep(0.5)
                try:
                    r = await client.post(f"{EMBED_URL}/v1/embeddings", json=payload)
                    body = r.text or ""
                    if r.status_code >= 500:
                        if _embed_input_too_large(body):
                            saw_too_large = True
                            break
                        if attempt + 1 < EMBED_RETRIES:
                            log.warning(
                                f"Embedding HTTP {r.status_code}, retry {attempt + 2}/{EMBED_RETRIES}: "
                                f"{body[:200]!r}"
                            )
                            continue
                    r.raise_for_status()
                    return r.json()["data"][0]["embedding"]
                except httpx.HTTPStatusError as e:
                    body = (e.response.text or "")[:200]
                    if _embed_input_too_large(e.response.text or ""):
                        saw_too_large = True
                        break
                    if e.response.status_code >= 500 and attempt + 1 < EMBED_RETRIES:
                        log.warning(
                            f"Embedding HTTP {e.response.status_code}, retry: {body!r}"
                        )
                        continue
                    log.warning(f"Embedding failed HTTP {e.response.status_code}: {body!r}")
                    return None
                except Exception as e:
                    if attempt + 1 < EMBED_RETRIES:
                        log.warning(f"Embedding failed, retry: {e}")
                        continue
                    log.warning(f"Embedding failed: {e}")
                    return None

            if not saw_too_large:
                return None

    return None


async def search_qdrant(vector: list[float]) -> list[dict]:
    """Return top-k chunks from Qdrant above the similarity threshold."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                json={
                    "vector": vector,
                    "limit": TOP_K,
                    "with_payload": True,
                    "score_threshold": SIMILARITY_THRESHOLD,
                },
            )
            r.raise_for_status()
            return r.json().get("result", [])
    except Exception as e:
        log.warning(f"Qdrant search failed: {e}")
        return []


def extract_chunk_text(hit: dict) -> str:
    """Pull text out of a Qdrant hit payload. NOMAD may use different field names."""
    payload = hit.get("payload", {})
    # Try known field names in order of likelihood
    for key in ("text", "content", "chunk", "document", "page_content"):
        if payload.get(key):
            return payload[key]
    # Last resort — dump the whole payload as a string
    return str(payload) if payload else ""


def user_message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        ).strip()
    return ""


def is_embeddable_user_query(text: str) -> bool:
    """Skip UI meta-prompts (e.g. Open WebUI follow-up suggestion tasks)."""
    head = text.lstrip()[:400].lower()
    if head.startswith("### task:"):
        return False
    if "follow-up" in head and "suggest" in head:
        return False
    if "relevant follow-up questions" in head:
        return False
    return True


def extract_query_text(messages: list[dict]) -> str | None:
    """Last real user turn — skip automated ### Task: / follow-up prompts."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = user_message_text(msg)
        if text and is_embeddable_user_query(text):
            return text
    return None


def inject_context(messages: list[dict], chunks: list[str]) -> list[dict]:
    """Prepend retrieved context as a system message (or prefix an existing one)."""
    context_block = "\n\n---\n\n".join(chunks)
    rag_prefix = (
        "The following context was retrieved from the local knowledge base. "
        "Use it to inform your response where relevant. "
        "Do not mention the knowledge base unless the user asks.\n\n"
        f"{context_block}"
    )
    messages = list(messages)  # shallow copy — don't mutate caller's list
    if messages and messages[0]["role"] == "system":
        messages[0] = {**messages[0], "content": rag_prefix + "\n\n" + messages[0]["content"]}
    else:
        messages.insert(0, {"role": "system", "content": rag_prefix})
    return messages


# ---------------------------------------------------------------------------
# Streaming helper — must keep httpx client alive until stream ends
# ---------------------------------------------------------------------------

async def relay_upstream(
    client: httpx.AsyncClient,
    upstream: httpx.Response,
) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()
        await client.aclose()


# ---------------------------------------------------------------------------
# Proxy route — catches everything
# ---------------------------------------------------------------------------

CHAT_PATHS = {"v1/chat/completions", "api/chat"}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path: str):
    body = await request.body()

    # Strip hop-by-hop headers that shouldn't be forwarded
    skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}

    # -----------------------------------------------------------------------
    # RAG augmentation — only for chat completion POSTs
    # -----------------------------------------------------------------------
    if request.method == "POST" and path.rstrip("/") in CHAT_PATHS and body:
        try:
            data = json.loads(body)
            messages = data.get("messages", [])
            query = extract_query_text(messages)

            if query:
                vector = await get_embedding(query)
                if vector is not None:
                    hits = await search_qdrant(vector)
                    chunks = [extract_chunk_text(h) for h in hits]
                    chunks = [c for c in chunks if c]  # drop empties

                    if chunks:
                        data["messages"] = inject_context(messages, chunks)
                        body = json.dumps(data, ensure_ascii=False).encode()
                        log.info(
                            f"RAG: injected {len(chunks)} chunk(s) "
                            f"(scores: {[round(h['score'], 3) for h in hits]}) "
                            f"| query: {query[:80]!r}"
                        )
                    else:
                        log.debug(f"RAG: no chunks above threshold={SIMILARITY_THRESHOLD}")
                else:
                    log.debug("RAG: skipped (embedding returned None)")
            elif any(m.get("role") == "user" for m in messages):
                log.debug("RAG: skipped (no embeddable user query)")
        except Exception as e:
            # Never break a request because of RAG — pass through unchanged
            log.warning(f"RAG augmentation error (passing through unmodified): {e}")

    # -----------------------------------------------------------------------
    # Forward to llama-swap (streaming-aware)
    # -----------------------------------------------------------------------
    client = httpx.AsyncClient(timeout=600)
    upstream: httpx.Response | None = None
    try:
        upstream_req = client.build_request(
            method=request.method,
            url=f"{LLAMA_SWAP_URL}/{path}",
            headers=headers,
            content=body,
            params=request.query_params,
        )
        upstream = await client.send(upstream_req, stream=True)

        resp_headers = dict(upstream.headers)
        # Remove headers that conflict with FastAPI's own response handling
        for h in ("content-encoding", "transfer-encoding", "content-length"):
            resp_headers.pop(h, None)

        content_type = upstream.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # Client/response stay open until relay_upstream() finishes
            return StreamingResponse(
                relay_upstream(client, upstream),
                status_code=upstream.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )

        content = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type=content_type or "application/json",
        )
    except Exception:
        if upstream is not None:
            await upstream.aclose()
        await client.aclose()
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    log.info(f"RAG Proxy starting on {PROXY_HOST}:{PROXY_PORT}")
    log.info(f"  → llama-swap : {LLAMA_SWAP_URL}")
    log.info(f"  → embed      : {EMBED_URL}")
    log.info(f"  → qdrant     : {QDRANT_URL} / {QDRANT_COLLECTION}")
    log.info(f"  → top_k={TOP_K}  threshold={SIMILARITY_THRESHOLD}")

    if "CHANGE_ME" in QDRANT_URL:
        log.warning("QDRANT_URL still has placeholder — set it to your omv IP before use")

    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level="warning")
