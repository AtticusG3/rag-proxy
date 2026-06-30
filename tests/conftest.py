"""Ensure repo root is on sys.path for package imports."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_proxy import upstream_client as uc

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def buffered_upstream_response(body: bytes = b'{"ok":true}') -> AsyncMock:
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(return_value=body)
    mock_response.aclose = AsyncMock()
    return mock_response


def pooled_client_mock(send_response: AsyncMock) -> MagicMock:
    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value="req")
    mock_client.send = AsyncMock(return_value=send_response)
    mock_client.aclose = AsyncMock()
    return mock_client


def capture_upstream_body(mock_client: MagicMock) -> list[bytes]:
    captured: list[bytes] = []

    def _build_request(**kwargs):
        content = kwargs.get("content")
        if content is not None:
            captured.append(content)
        return "req"

    mock_client.build_request = MagicMock(side_effect=_build_request)
    return captured


class FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in for unit tests."""

    def __init__(self, post_impl: AsyncMock) -> None:
        self.post = post_impl

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def sidecar_pool_mock() -> MagicMock:
    mock = MagicMock()
    mock.aclose = AsyncMock()
    return mock


def pooled_ctor_side_effect(upstream_client: MagicMock) -> list[MagicMock]:
    """Lifespan builds upstream + embed + qdrant pools (shared httpx.AsyncClient patch)."""
    return [upstream_client, sidecar_pool_mock(), sidecar_pool_mock()]


@pytest.fixture(autouse=True)
def _reset_upstream_client():
    """Isolate upstream pool singleton between tests."""
    uc._upstream_client = None
    uc._janitor_task = None
    uc._stream_registry.clear()
    uc._stream_registry_lock = None
    yield
    if uc._janitor_task is not None:
        uc._janitor_task.cancel()
    uc._stream_registry.clear()
    uc._upstream_client = None
    uc._janitor_task = None
    uc._stream_registry_lock = None
