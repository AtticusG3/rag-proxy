#!/usr/bin/env python3
"""FastAPI application and proxy route."""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from rag_proxy.capture import capture_chat_response, capture_enabled
from rag_proxy.capture_writer import shutdown_capture_writer, startup_capture_writer
from rag_proxy.config import CHAT_PATHS, settings
from rag_proxy.context import RequestContext
from rag_proxy.observability import metrics_enabled, render_metrics_text
from rag_proxy.orchestrator import (
    augment_chat_payload_with_context,
    build_request_context_from_http,
)
from rag_proxy.sidecar_client import shutdown_sidecar_clients, startup_sidecar_clients
from rag_proxy.stages.tier3_graph import _ensure_schema as ensure_graph_schema
from rag_proxy.upstream_client import (
    close_upstream_response,
    ensure_upstream_client,
    relay_upstream,
    relay_upstream_capture,
    shutdown_upstream_client,
    startup_upstream_client,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag-proxy")


def _internal_token_ok(request: Request) -> bool:
    expected = settings.proxy_internal_token
    if not expected:
        return True
    provided = request.headers.get("x-internal-token", "")
    return hmac.compare_digest(provided, expected)


def _warm_memgraph_cache() -> None:
    try:
        from rag_proxy.memgraphrag.cache import get_memory_index
    except ImportError:
        return
    try:
        get_memory_index(settings.memgraphrag_db_path)
        log.info("memgraphrag memory index warmed path=%s", settings.memgraphrag_db_path)
    except Exception:
        log.warning("memgraphrag cache warm failed", exc_info=True)


@asynccontextmanager
async def _app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await startup_upstream_client()
    await startup_sidecar_clients()
    if settings.enable_graph_lookup:
        ensure_graph_schema(Path(settings.graph_db_path))
    if settings.enable_memgraphrag:
        _warm_memgraph_cache()
    await startup_capture_writer()
    try:
        yield
    finally:
        await shutdown_capture_writer()
        await shutdown_sidecar_clients()
        await shutdown_upstream_client()


app = FastAPI(title="RAG Proxy", docs_url=None, redoc_url=None, lifespan=_app_lifespan)


@app.get("/metrics")
async def prometheus_metrics(request: Request) -> PlainTextResponse:
    if not _internal_token_ok(request):
        return PlainTextResponse("unauthorized\n", status_code=401)
    if not metrics_enabled():
        return PlainTextResponse("metrics disabled\n", status_code=404)
    return PlainTextResponse(
        render_metrics_text(),
        media_type="text/plain; charset=utf-8",
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path: str):
    if not _internal_token_ok(request):
        return PlainTextResponse("unauthorized\n", status_code=401)

    body = await request.body()

    skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}
    capture_original_messages: list[dict] | None = None
    capture_ctx: RequestContext | None = None
    should_capture = False

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
                should_capture = capture_enabled(headers)
                if should_capture:
                    capture_original_messages = deepcopy(data.get("messages", []))
                try:
                    data, capture_ctx = await augment_chat_payload_with_context(data, headers)
                    body = json.dumps(data, ensure_ascii=False).encode()
                except (TypeError, ValueError) as e:
                    log.warning(f"Failed to serialize augmented body (passing through): {e}")
                    if should_capture:
                        capture_ctx = build_request_context_from_http(data, headers)
                    body = original_body
                except Exception as e:
                    # Single fail-open boundary for RAG augmentation errors.
                    log.warning(f"RAG augmentation error (passing through unmodified): {e}")
                    if should_capture:
                        capture_ctx = build_request_context_from_http(data, headers)
                    body = original_body

    client = await ensure_upstream_client()
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
            stream_body = relay_upstream(request, upstream)
            if should_capture and capture_original_messages is not None and capture_ctx is not None:

                async def _capture_stream(response_body: bytes) -> None:
                    capture_chat_response(
                        original_messages=capture_original_messages or [],
                        ctx=capture_ctx,
                        response_body=response_body,
                        path=path,
                        stream=True,
                    )

                stream_body = relay_upstream_capture(request, upstream, _capture_stream)
            return StreamingResponse(
                stream_body,
                status_code=upstream.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )

        content = await upstream.aread()
        status_code = upstream.status_code
        await close_upstream_response(upstream)
        upstream = None
        if should_capture and capture_original_messages is not None and capture_ctx is not None:
            capture_chat_response(
                original_messages=capture_original_messages,
                ctx=capture_ctx,
                response_body=content,
                path=path,
                stream=bool(capture_ctx.stream),
            )
        return Response(
            content=content,
            status_code=status_code,
            headers=resp_headers,
            media_type=content_type or "application/json",
        )
    except Exception:
        await close_upstream_response(upstream)
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
