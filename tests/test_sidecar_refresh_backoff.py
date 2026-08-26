"""Sparse sidecar refresh loop retries with backoff when Qdrant is unreachable.

Regression coverage for the fix that keeps the sidecar retrying after a failed
initial sync (Qdrant still booting) instead of waiting a full REFRESH_SEC.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SPARSE_DIR = ROOT / "sidecars" / "sparse"
sys.path.insert(0, str(SPARSE_DIR))

import app as sparse_app  # noqa: E402


def _drive_loop(initial_delay: float, outcomes: list, n_sleeps: int) -> list[float]:
    """Run ``_refresh_loop`` with sleep/sync stubbed; capture the first ``n_sleeps`` delays.

    ``outcomes`` is consumed in order; after it is exhausted the last outcome is
    reused so a single ``RuntimeError`` simulates a persistent outage.
    """
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        await real_sleep(0)  # yield so the driver can observe progress

    idx = 0

    async def fake_sync(collection: str):
        nonlocal idx
        outcome = outcomes[min(idx, len(outcomes) - 1)]
        idx += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def run() -> None:
        orig_sync = sparse_app.sync_collection
        asyncio.sleep = fake_sleep  # type: ignore[assignment]
        sparse_app.sync_collection = fake_sync  # type: ignore[assignment]
        task = asyncio.create_task(sparse_app._refresh_loop(initial_delay))
        try:
            while len(sleeps) < n_sleeps:
                await real_sleep(0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            asyncio.sleep = real_sleep
            sparse_app.sync_collection = orig_sync

    asyncio.run(run())
    return sleeps


def test_initial_failure_backs_off_without_waiting_an_hour() -> None:
    """After the initial sync fails, retries double: 5s, 10s, 20s... not REFRESH_SEC."""
    sleeps = _drive_loop(5.0, [RuntimeError("qdrant down")], 5)
    assert sleeps == [5.0, 10.0, 20.0, 40.0, 80.0]


def test_success_resets_to_normal_refresh_interval() -> None:
    """A successful sync resets the delay to REFRESH_SEC."""
    sleeps = _drive_loop(5.0, [RuntimeError("down"), 123], 4)
    assert sleeps[0] == 5.0
    assert sleeps[1] == 10.0
    assert sleeps[2] == sparse_app.REFRESH_SEC
    assert sleeps[3] == sparse_app.REFRESH_SEC


def test_healthy_loop_recovers_quickly_after_late_outage() -> None:
    """If a later periodic sync fails, backoff restarts from the short base."""
    sleeps = _drive_loop(sparse_app.REFRESH_SEC, [RuntimeError("down")], 4)
    assert sleeps[0] == sparse_app.REFRESH_SEC
    assert sleeps[1] == sparse_app.INITIAL_RETRY_SEC
    assert sleeps[2] == sparse_app.INITIAL_RETRY_SEC * 2
    assert sleeps[3] == min(sparse_app.INITIAL_RETRY_SEC * 4, sparse_app.RETRY_MAX_SEC)


def test_health_listens_while_initial_sync_is_blocked() -> None:
    """A first-boot Qdrant rebuild must not take the HTTP port down.

    Hybrid retrieval fail-opens on an empty sparse index. A refused connection
    is treated as sidecar-down and looks like a broken deploy.
    """
    started = threading.Event()
    release = threading.Event()

    async def hung_sync(_collection: str) -> int:
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        return 0

    with (
        patch.object(sparse_app, "sync_collection", hung_sync),
        patch.object(sparse_app, "INDEX_PATH", None),
        patch.object(sparse_app, "REFRESH_SEC", 0.0),
    ):
        with TestClient(sparse_app.app) as client:
            assert started.wait(timeout=2.0), "bootstrap never started"
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] in {"ok", "starting"}
            release.set()
