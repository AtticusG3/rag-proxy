#!/usr/bin/env python3
"""FastAPI application and proxy route."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from rag_proxy.config import CHAT_PATHS, settings
from rag_proxy.observability import metrics_enabled, render_metrics_text
from rag_proxy.orchestrator import augment_chat_payload

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-proxy")

app = FastAPI(title="RAG Proxy", docs_url=None, redoc_url=None)


@app.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    if not metrics_enabled():
        return PlainTextResponse("metrics disabled\n", status_code=404)
    return PlainTextResponse(
        render_metrics_text(),
        media_type="text/plain; charset=utf-8",
    )


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


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path: str):
    body = await request.body()

    skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}

    if request.method == "POST" and path.rstrip("/") in CHAT_PATHS and body:
        original_body = body
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(f"Invalid JSON body (passing through unmodified): {e}")
        else:
            if not isinstance(data, dict):
                log.warning("Chat JSON body is not an object (passing through unmodified)")
            else:
                try:
                    data = await augment_chat_payload(data, headers)
                    body = json.dumps(data, ensure_ascii=False).encode()
                except (TypeError, ValueError) as e:
                    log.warning(f"Failed to serialize augmented body (passing through): {e}")
                    body = original_body
                except Exception as e:
                    # Single fail-open boundary for RAG augmentation errors.
                    log.warning(f"RAG augmentation error (passing through unmodified): {e}")
                    body = original_body

    client = httpx.AsyncClient(timeout=600)
    upstream: httpx.Response | None = None
    try:
        upstream_req = client.build_request(
            method=request.method,
            url=f"{settings.llama_swap_url}/{path}",
            headers=headers,
            content=body,
            params=request.query_params,
        )
        upstream = await client.send(upstream_req, stream=True)

        resp_headers = dict(upstream.headers)
        for h in ("content-encoding", "transfer-encoding", "content-length"):
            resp_headers.pop(h, None)

        content_type = upstream.headers.get("content-type", "")

        if "text/event-stream" in content_type:
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


def main() -> None:
    import uvicorn

    log.info(f"RAG Proxy starting on {settings.proxy_host}:{settings.proxy_port}")
    log.info(f"  -> llama-swap : {settings.llama_swap_url}")
    log.info(f"  -> embed      : {settings.embed_url}")
    log.info(f"  -> qdrant     : {settings.qdrant_url} / {settings.qdrant_collection}")
    log.info(f"  -> top_k={settings.top_k}  threshold={settings.similarity_threshold}")
    log.info(f"  -> cognitive_pipeline={settings.enable_cognitive_pipeline}")

    if "CHANGE_ME" in settings.qdrant_url:
        log.warning("QDRANT_URL still has placeholder -- set it to your omv IP before use")

    if metrics_enabled():
        log.info(
            f"  -> metrics     : http://{settings.proxy_host}:{settings.proxy_port}/metrics"
        )

    uvicorn.run(app, host=settings.proxy_host, port=settings.proxy_port, log_level="warning")
