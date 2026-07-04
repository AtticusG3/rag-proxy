"""Tests for admin ingest status helpers."""

from __future__ import annotations

import os
import tempfile

from ingest.worker import IngestConfig
from rag_admin.helpers import flash_redirect
from rag_admin.embed_throughput import (
    embed_throughput_rates,
    format_embed_rate,
    record_embed_progress,
    reset_embed_throughput,
)
from rag_admin.ingest_status import (
    enrich_file_rows,
    ingest_config_snapshot,
    ingest_queue_stats,
    ingest_velocity_text,
)


class _FakeWorker:
    def __init__(self) -> None:
        self.paused = False
        self.config = IngestConfig(
            zim_dir="/tmp/zim",
            upload_dir="/tmp/uploads",
            embed_url="http://127.0.0.1:8089",
            qdrant_url="http://127.0.0.1:6333",
            qdrant_collection="test",
            sparse_index_url="",
            batch_size=128,
            embed_concurrency=8,
            sparse_reindex_mode="off",
            stall_seconds=900,
        )


def test_enrich_file_rows_marks_stalled_running() -> None:
    rows = enrich_file_rows(
        [
            {
                "file_path": "/tmp/a.zim",
                "status": "running",
                "updated_at": "2000-01-01T00:00:00+00:00",
                "chunks_embedded": 12,
            }
        ],
        stall_seconds=60,
    )
    assert rows[0]["display_status"] == "stalled"
    assert rows[0]["is_stalled"] is True


def test_enrich_file_rows_flags_missing_files() -> None:
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        present = handle.name
    try:
        rows = enrich_file_rows(
            [
                {"file_path": present, "status": "indexed"},
                {"file_path": "/tmp/definitely-missing-ingest-file.md", "status": "failed"},
            ],
            stall_seconds=60,
        )
        assert rows[0]["file_missing"] is False
        assert rows[1]["file_missing"] is True
    finally:
        os.unlink(present)


def test_ingest_queue_stats_counts_active_files() -> None:
    reset_embed_throughput()
    stats = ingest_queue_stats(
        [
            {"status": "pending", "chunks_embedded": 0},
            {"status": "running", "chunks_embedded": 40, "display_status": "running"},
            {"status": "indexed", "chunks_embedded": 100},
        ]
    )
    assert stats["pending"] == 1
    assert stats["running"] == 1
    assert stats["active"] == 2
    assert stats["total_chunks"] == 140
    assert stats["queue_chunks"] == 40
    assert stats["velocity_text"]
    assert "in queue" in stats["velocity_text"]


def test_embed_throughput_rates_after_progress() -> None:
    reset_embed_throughput()
    record_embed_progress(1000, now=0.0)
    record_embed_progress(1060, now=60.0)
    rates = embed_throughput_rates()
    assert rates["embed_rate_now"] == 60
    assert rates["embed_rate_5m"] is None
    assert rates["embed_rate_15m"] is None


def test_embed_throughput_window_rates_need_full_span() -> None:
    reset_embed_throughput()
    record_embed_progress(0, now=0.0)
    record_embed_progress(300, now=300.0)
    record_embed_progress(900, now=600.0)
    rates = embed_throughput_rates()
    assert rates["embed_rate_5m"] == 120
    assert rates["embed_rate_15m"] is None


def test_embed_throughput_5m_and_15m_differ_with_enough_history() -> None:
    reset_embed_throughput()
    record_embed_progress(0, now=0.0)
    record_embed_progress(300, now=300.0)
    record_embed_progress(900, now=600.0)
    record_embed_progress(1200, now=900.0)
    rates = embed_throughput_rates()
    assert rates["embed_rate_5m"] == 60
    assert rates["embed_rate_15m"] == 80
    assert rates["embed_rate_5m"] != rates["embed_rate_15m"]


def test_format_embed_rate_waiting_when_pending_only() -> None:
    assert format_embed_rate(None, running=0, pending=3, window="now") == "waiting"
    assert format_embed_rate(None, running=0, pending=3, window="5m") == "—"
    assert format_embed_rate(42, running=1, pending=0, window="now") == "42 chunks/min"


def test_ingest_velocity_text_idle_queue() -> None:
    text = ingest_velocity_text(
        {
            "active": 5,
            "total_chunks": 2_974_689,
            "running": 0,
            "pending": 5,
            "embed_rate_now": None,
            "embed_rate_5m": None,
            "embed_rate_15m": None,
        }
    )
    assert "5 in queue" in text
    assert "2,974,689 corpus chunks" in text
    assert "now waiting" in text
    assert "5m —" in text


def test_ingest_config_snapshot_exposes_tuning_knobs() -> None:
    snapshot = ingest_config_snapshot(_FakeWorker())
    assert snapshot["batch_size"] == 128
    assert snapshot["embed_concurrency"] == 8
    assert snapshot["paused"] is False


def test_flash_redirect_appends_query_params() -> None:
    response = flash_redirect("/jobs", "Scan complete.")
    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs?")
    assert "Scan+complete" in response.headers["location"]
