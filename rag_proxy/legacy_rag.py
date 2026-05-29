"""Core RAG helpers (embed, Qdrant, message extraction, injection)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from rag_proxy.config import settings

log = logging.getLogger("rag-proxy")


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
    char_limits = [settings.embed_max_chars]
    if settings.embed_max_chars > 1200:
        char_limits.append(1200)

    async with httpx.AsyncClient(timeout=30) as client:
        for max_chars in char_limits:
            chunk = prepare_embed_text(text, max_chars)
            payload = {"model": "nomic-embed-text-v1.5", "input": chunk}
            saw_too_large = False

            for attempt in range(settings.embed_retries):
                if attempt:
                    await asyncio.sleep(0.5)
                try:
                    r = await client.post(f"{settings.embed_url}/v1/embeddings", json=payload)
                    body = r.text or ""
                    if r.status_code >= 500:
                        if _embed_input_too_large(body):
                            saw_too_large = True
                            break
                        if attempt + 1 < settings.embed_retries:
                            log.warning(
                                f"Embedding HTTP {r.status_code}, retry {attempt + 2}/"
                                f"{settings.embed_retries}: {body[:200]!r}"
                            )
                            continue
                    r.raise_for_status()
                    return r.json()["data"][0]["embedding"]
                except httpx.HTTPStatusError as e:
                    body = (e.response.text or "")[:200]
                    if _embed_input_too_large(e.response.text or ""):
                        saw_too_large = True
                        break
                    if e.response.status_code >= 500 and attempt + 1 < settings.embed_retries:
                        log.warning(
                            f"Embedding HTTP {e.response.status_code}, retry: {body!r}"
                        )
                        continue
                    log.warning(f"Embedding failed HTTP {e.response.status_code}: {body!r}")
                    return None
                except Exception as e:
                    if attempt + 1 < settings.embed_retries:
                        log.warning(f"Embedding failed, retry: {e}")
                        continue
                    log.warning(f"Embedding failed: {e}")
                    return None

            if not saw_too_large:
                return None

    return None


async def search_qdrant_dense(
    vector: list[float],
    limit: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """Return top-k chunks from Qdrant above the similarity threshold."""
    limit = limit if limit is not None else settings.top_k
    score_threshold = (
        score_threshold if score_threshold is not None else settings.similarity_threshold
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/search",
                json={
                    "vector": vector,
                    "limit": limit,
                    "with_payload": True,
                    "score_threshold": score_threshold,
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
    for key in ("text", "content", "chunk", "document", "page_content"):
        if payload.get(key):
            return payload[key]
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
    messages = list(messages)
    if messages and messages[0]["role"] == "system":
        messages[0] = {**messages[0], "content": rag_prefix + "\n\n" + messages[0]["content"]}
    else:
        messages.insert(0, {"role": "system", "content": rag_prefix})
    return messages


async def legacy_augment_messages(messages: list[dict]) -> tuple[list[dict], dict]:
    """Original always-retrieve path. Returns (messages, meta) for logging."""
    meta: dict = {"chunks": 0, "scores": [], "query": None}
    query = extract_query_text(messages)
    if not query:
        return messages, meta

    meta["query"] = query
    vector = await get_embedding(query)
    if vector is None:
        return messages, meta

    hits = await search_qdrant_dense(vector)
    chunks = [extract_chunk_text(h) for h in hits]
    chunks = [c for c in chunks if c]
    if chunks:
        meta["chunks"] = len(chunks)
        meta["scores"] = [round(h["score"], 3) for h in hits]
        return inject_context(messages, chunks), meta
    return messages, meta
