"""Tests for on-demand nomic embed lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingest.embed_lifecycle import (
    idle_stop_threshold_sec,
    on_demand_enabled,
    unit_for_embed_url,
    units_for_embed_urls,
)
from rag_admin.embed_idle_guard import EmbedIdleGuard


def test_unit_for_embed_url_maps_query_and_pool() -> None:
    assert unit_for_embed_url("http://127.0.0.1:8089") == "nomic-embed.service"
    assert unit_for_embed_url("http://127.0.0.1:18089") == "nomic-embed@18089.service"


def test_units_for_embed_urls_deduplicates() -> None:
    units = units_for_embed_urls(
        [
            "http://127.0.0.1:8089",
            "http://127.0.0.1:18089",
            "http://127.0.0.1:18089",
        ]
    )
    assert units == ["nomic-embed.service", "nomic-embed@18089.service"]


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

    guard = EmbedIdleGuard(worker, job_runner, pool_env_path="/tmp/pool.env", poll_sec=0.01)
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

    guard = EmbedIdleGuard(worker, job_runner, pool_env_path="/tmp/pool.env")
    assert guard._should_stop_embed() is False
