"""Tests for on-demand nomic embed lifecycle (query embed unit only).

Also covers the bounded LRU/TTL embed cache."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingest.embed_lifecycle import (
    collect_embed_stop_units,
    ensure_embed_urls,
    idle_stop_threshold_sec,
    on_demand_enabled,
    unit_for_embed_url,
    units_for_embed_urls,
)
from rag_admin.embed_idle_guard import EmbedIdleGuard
from rag_proxy.clients import retrieval_async

ROOT = Path(__file__).resolve().parents[1]


def test_unit_for_embed_url_maps_only_the_query_embed() -> None:
    assert unit_for_embed_url("http://127.0.0.1:8089") == "nomic-embed.service"
    # Non-query embed URLs (e.g. llama-swap) have no dedicated unit to manage.
    assert unit_for_embed_url("http://127.0.0.1:18089") is None
    assert unit_for_embed_url("http://127.0.0.1:8081") is None


def test_units_for_embed_urls_deduplicates() -> None:
    units = units_for_embed_urls(
        [
            "http://127.0.0.1:8089",
            "http://127.0.0.1:8089",
            "http://127.0.0.1:18089",
        ]
    )
    assert units == ["nomic-embed.service"]


def test_query_embed_always_on_spares_the_query_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_URL", "http://127.0.0.1:8089")

    monkeypatch.setenv("EMBED_QUERY_ALWAYS_ON", "false")
    assert "nomic-embed.service" in collect_embed_stop_units()

    monkeypatch.setenv("EMBED_QUERY_ALWAYS_ON", "true")
    assert collect_embed_stop_units() == []


def test_collect_embed_stop_units_ignores_non_query_embed_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_URL", "http://127.0.0.1:18080")
    monkeypatch.setenv("EMBED_QUERY_ALWAYS_ON", "false")
    assert collect_embed_stop_units() == []


def test_idle_stop_threshold_shorter_when_paused() -> None:
    assert idle_stop_threshold_sec(ingest_paused=True) < idle_stop_threshold_sec(
        ingest_paused=False
    )


def test_ensure_embed_urls_skips_probe_when_url_has_no_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama-swap embed URLs must not pay a blocking health embed."""
    monkeypatch.setattr("ingest.embed_lifecycle.on_demand_enabled", lambda: True)
    with patch("ingest.embed_lifecycle.wait_for_embed_health") as wait:
        with patch("ingest.embed_lifecycle._enable_start_unit") as start:
            result = ensure_embed_urls(["http://127.0.0.1:8081"])
    start.assert_not_called()
    wait.assert_not_called()
    assert result == ["http://127.0.0.1:8081"]


def test_ensure_embed_urls_starts_nomic_query_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated query port is the only URL on-demand is allowed to start."""
    monkeypatch.setattr("ingest.embed_lifecycle.on_demand_enabled", lambda: True)
    with patch(
        "ingest.embed_lifecycle.wait_for_embed_health",
        return_value=["http://127.0.0.1:8089"],
    ) as wait:
        with patch("ingest.embed_lifecycle._enable_start_unit") as start:
            result = ensure_embed_urls(["http://127.0.0.1:8089"])
    start.assert_called_once_with("nomic-embed.service")
    wait.assert_called_once()
    assert result == ["http://127.0.0.1:8089"]


def test_on_demand_disabled_without_systemctl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_ON_DEMAND", raising=False)
    monkeypatch.setattr("ingest.embed_lifecycle.shutil.which", lambda _name: None)
    assert on_demand_enabled() is False


def test_idle_guard_stops_when_paused_and_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_admin.embed_idle_guard.on_demand_enabled", lambda: True)
    worker = MagicMock()
    worker.paused = True
    worker.running_file_count.return_value = 0
    worker.db.list_pending_files.return_value = []
    job_runner = MagicMock()
    job_runner.active_job.return_value = None

    guard = EmbedIdleGuard(worker, job_runner, poll_sec=0.01)
    with patch(
        "rag_admin.embed_idle_guard.seconds_since_embed_activity",
        return_value=999.0,
    ):
        assert guard._should_stop_embed() is True


def test_idle_guard_keeps_warm_during_active_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_admin.embed_idle_guard.on_demand_enabled", lambda: True)
    worker = MagicMock()
    worker.paused = False
    worker.running_file_count.return_value = 1
    job_runner = MagicMock()
    job_runner.active_job.return_value = None

    guard = EmbedIdleGuard(worker, job_runner)
    assert guard._should_stop_embed() is False


def _embed_cache_env(
    monkeypatch: pytest.MonkeyPatch, max_entries: int
) -> "retrieval_async.OrderedDict[str, tuple[list[float], float]]":
    """Enable the cache with a given cap and reset the module-level store."""
    from collections import OrderedDict

    monkeypatch.setattr(retrieval_async.settings, "enable_embed_cache", True)
    monkeypatch.setattr(retrieval_async.settings, "embed_cache_max_entries", max_entries)
    cache: "OrderedDict[str, tuple[list[float], float]]" = OrderedDict()
    monkeypatch.setattr("rag_proxy.clients.retrieval_async._embed_cache", cache)
    return cache


def test_embed_cache_max_entries_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _embed_cache_env(monkeypatch, 2)
    calls: list[str] = []

    async def fake_get_embedding(text: str):
        calls.append(text)
        return [float(len(text))]

    monkeypatch.setattr(retrieval_async, "get_embedding", fake_get_embedding)

    async def run() -> None:
        for text in ("query-one", "query-two", "query-three"):
            await retrieval_async.embed_text(text)

    asyncio.run(run())

    assert len(cache) <= 2
    assert len(cache) == 2
    assert calls == ["query-one", "query-two", "query-three"]
    assert retrieval_async._cache_key("query-one") not in cache
    assert retrieval_async._cache_key("query-three") in cache
    # Evicted entry is re-embedded rather than served stale.
    assert cache[next(iter(cache))][0] == [9.0]


def test_embed_cache_expired_entry_removed_on_access(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _embed_cache_env(monkeypatch, 8)
    key = retrieval_async._cache_key("stale query")
    cache[key] = ([0.5], time.time() - retrieval_async._EMBED_CACHE_TTL - 1)

    assert retrieval_async._embed_cache_get(key, time.time()) is None
    assert key not in cache


def test_embed_cache_hit_and_miss_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_proxy.observability import RAG_EMBED_CACHE_HITS, RAG_EMBED_CACHE_MISSES

    cache = _embed_cache_env(monkeypatch, 8)
    calls: list[str] = []

    async def fake_get_embedding(text: str):
        calls.append(text)
        return [0.25]

    monkeypatch.setattr(retrieval_async, "get_embedding", fake_get_embedding)
    hits_before = RAG_EMBED_CACHE_HITS._value.get()
    misses_before = RAG_EMBED_CACHE_MISSES._value.get()

    async def run() -> None:
        first = await retrieval_async.embed_text("counter query")
        second = await retrieval_async.embed_text("counter query")
        third = await retrieval_async.embed_text("counter query", no_cache=True)
        assert first == [0.25]
        assert second == [0.25]
        assert third == [0.25]

    asyncio.run(run())

    assert RAG_EMBED_CACHE_MISSES._value.get() == misses_before + 1
    assert RAG_EMBED_CACHE_HITS._value.get() == hits_before + 1
    assert calls == ["counter query", "counter query"]
    assert len(cache) == 1
    # A hit moves the entry to the most-recently-used end.
    assert next(reversed(cache.values()))[0] == [0.25]


def test_embed_cache_max_entries_documented() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    configuration = (ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")

    assert re.search(r"(?m)^EMBED_CACHE_MAX_ENTRIES=2048\s*$", env_example)
    assert "EMBED_CACHE_MAX_ENTRIES" in configuration

