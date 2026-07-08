"""Qdrant upsert must survive transient disconnects during bulk ingest."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from ingest.qdrant_writer import upsert_points


def _point(n: int) -> dict:
    return {"id": str(n), "vector": [0.1], "payload": {"text": f"t{n}"}}


def test_upsert_retries_transient_disconnect_before_succeeding(monkeypatch) -> None:
    """A single Qdrant disconnect must not fail the whole ingest batch."""
    calls = {"n": 0}

    def fake_put(url: str, json: dict, **_kwargs) -> MagicMock:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        response = MagicMock()
        response.raise_for_status.return_value = None
        return response

    client = MagicMock()
    client.put.side_effect = fake_put
    monkeypatch.setattr("ingest.qdrant_writer.time.sleep", lambda _s: None)

    upsert_points("http://qdrant", "col", [_point(1)], client=client, retries=2)

    assert calls["n"] == 2


def test_upsert_bisects_batch_when_retries_exhausted(monkeypatch) -> None:
    """Large upserts should shrink under sustained Qdrant pressure."""
    batch_sizes: list[int] = []

    def fake_put(url: str, json: dict, **_kwargs) -> MagicMock:
        points = json["points"]
        batch_sizes.append(len(points))
        if len(points) > 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        response = MagicMock()
        response.raise_for_status.return_value = None
        return response

    client = MagicMock()
    client.put.side_effect = fake_put
    monkeypatch.setattr("ingest.qdrant_writer.time.sleep", lambda _s: None)

    upsert_points(
        "http://qdrant",
        "col",
        [_point(i) for i in range(4)],
        client=client,
        retries=0,
    )

    assert 4 in batch_sizes
    assert 2 in batch_sizes
    assert 1 in batch_sizes


def test_upsert_does_not_retry_non_transient_http_errors(monkeypatch) -> None:
    """Client errors should fail fast instead of hammering Qdrant."""
    request = httpx.Request("PUT", "http://qdrant/collections/col/points")
    response = httpx.Response(400, request=request)
    err = httpx.HTTPStatusError("bad request", request=request, response=response)

    client = MagicMock()
    client.put.side_effect = err
    monkeypatch.setattr("ingest.qdrant_writer.time.sleep", lambda _s: None)

    with pytest.raises(httpx.HTTPStatusError):
        upsert_points("http://qdrant", "col", [_point(1)], client=client, retries=3)

    assert client.put.call_count == 1
