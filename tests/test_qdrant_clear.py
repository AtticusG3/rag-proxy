"""Tests for Qdrant collection clear helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ingest.qdrant_writer import clear_collection


def test_clear_collection_drops_and_recreates():
    client = MagicMock()
    client.get.return_value.status_code = 200
    client.get.return_value.json.return_value = {"result": {"points_count": 42}}
    client.delete.return_value.status_code = 200
    client.put.return_value.status_code = 200

    with patch("ingest.qdrant_writer.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value = client
        removed = clear_collection("http://qdrant:6333", "test_col")

    assert removed == 42
    client.delete.assert_called_once()
    client.put.assert_called_once()
