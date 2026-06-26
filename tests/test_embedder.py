"""Tests for ingest embedder resilience."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from ingest.embedder import embed_texts


def _context_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://embed/v1/embeddings")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "type": "exceed_context_size_error",
                "message": "input (517 tokens) is larger than the max context size (512 tokens)",
            }
        },
    )
    return httpx.HTTPStatusError("400", request=request, response=response)


def _ok_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"embedding": vector} for vector in vectors]
    }
    return response


def test_embed_texts_splits_batch_on_context_overflow():
    client = MagicMock()
    client.post.side_effect = [
        _context_error(),
        _ok_response([[1.0], [2.0]]),
        _ok_response([[3.0], [4.0]]),
    ]

    result = embed_texts(
        ["a", "b", "c", "d"],
        embed_url="http://embed",
        client=client,
    )

    assert result == [[1.0], [2.0], [3.0], [4.0]]
    assert client.post.call_count == 3


def test_embed_texts_truncates_single_oversized_input():
    client = MagicMock()
    client.post.side_effect = [
        _context_error(),
        _ok_response([[9.0]]),
    ]

    result = embed_texts(
        ["x" * 600],
        embed_url="http://embed",
        max_chars=600,
        client=client,
    )

    assert result == [[9.0]]
    second_payload = client.post.call_args_list[1].kwargs["json"]["input"]
    assert len(second_payload[0]) == 400
