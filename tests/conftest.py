"""Ensure repo root is on sys.path for package imports."""

import sys
from pathlib import Path

import pytest

from rag_proxy import upstream_client as uc

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
