"""Tests for Qdrant sampling in MemGraphRAG build script."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import httpx

from scripts.build_memgraphrag_index import fetch_qdrant_chunks, fetch_qdrant_chunks_via_scroll


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.request = httpx.Request("POST", "http://q")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=self.request,
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, Any]:
        return self._body


def test_fetch_qdrant_chunks_falls_back_when_facet_missing() -> None:
    """Facet 404 must route to scroll sampling on older/custom Qdrant builds."""
    scroll_chunks = [
        {
            "chunk_id": "pt-1",
            "text": "hello world",
            "source": "doc-a",
            "meta": {},
        }
    ]

    async def _run() -> list[dict]:
        class _FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
                if url.endswith("/facet"):
                    return _FakeResponse(status_code=404)
                raise AssertionError(url)

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        with patch("scripts.build_memgraphrag_index.httpx.AsyncClient", _FakeClient):
            with patch(
                "scripts.build_memgraphrag_index.fetch_qdrant_chunks_via_scroll",
            ) as scroll:
                scroll.return_value = scroll_chunks
                return await fetch_qdrant_chunks(
                    qdrant_url="http://q",
                    collection="kb",
                    target_count=1,
                )

    chunks = asyncio.run(_run())
    assert chunks == scroll_chunks


def test_scroll_sampling_returns_text_chunks() -> None:
    """Scroll path must emit chunks with text from Qdrant payloads."""

    async def _run() -> list[dict]:
        class _FakeClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
                return _FakeResponse(
                    body={
                        "result": {
                            "points": [
                                {
                                    "id": "pt-1",
                                    "payload": {"text": "chunk one", "source": "a"},
                                },
                                {
                                    "id": "pt-2",
                                    "payload": {"text": "chunk two", "source": "b"},
                                },
                            ],
                            "next_page_offset": None,
                        }
                    }
                )

            async def __aenter__(self) -> _FakeClient:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        with patch("scripts.build_memgraphrag_index.httpx.AsyncClient", _FakeClient):
            return await fetch_qdrant_chunks_via_scroll(
                qdrant_url="http://q",
                collection="kb",
                target_count=2,
                max_points_to_scan=10,
            )

    chunks = asyncio.run(_run())
    assert len(chunks) == 2
    texts = {chunk["text"] for chunk in chunks}
    assert texts == {"chunk one", "chunk two"}
