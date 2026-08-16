"""Tests for on-demand nomic embed lifecycle (query embed unit only)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingest.embed_lifecycle import (
    collect_embed_stop_units,
    idle_stop_threshold_sec,
    on_demand_enabled,
    unit_for_embed_url,
    units_for_embed_urls,
)
from rag_admin.embed_idle_guard import EmbedIdleGuard


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
