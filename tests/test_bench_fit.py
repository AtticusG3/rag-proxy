"""Tests for bench report fitting."""

from __future__ import annotations

from ingest.bench_fit import (
    fit_from_reports,
    pick_chunk_concurrency,
    pick_embed_settings,
)


def _chunk_run(concurrency: int, rate: float, *, semantic: bool = False) -> dict:
    return {
        "mode": "chunk",
        "params": {"chunk_concurrency": concurrency, "semantic": semantic},
        "chunks_per_min": rate,
    }


def _embed_run(concurrency: int, batch_size: int, rate: float, *, errors: int = 0) -> dict:
    return {
        "mode": "embed",
        "params": {
            "embed_concurrency": concurrency,
            "batch_size": batch_size,
            "pool_size": 4,
        },
        "chunks_per_min": rate,
        "errors": errors,
    }


def test_pick_chunk_concurrency_uses_knee_not_peak():
    runs = [
        _chunk_run(1, 6505.0),
        _chunk_run(2, 15082.0),
        _chunk_run(3, 14400.0),
    ]
    picked = pick_chunk_concurrency(runs, semantic=False)
    assert picked == (2, 15082.0)


def test_pick_embed_settings_prefers_zero_errors_and_best_rate():
    runs = [
        _embed_run(16, 32, 2250.0),
        _embed_run(32, 32, 2200.0),
        _embed_run(64, 32, 1800.0, errors=2),
    ]
    picked = pick_embed_settings(runs)
    assert picked == (16, 32, 2250.0)


def test_pick_embed_settings_respects_pool_ceiling():
    runs = [
        _embed_run(16, 32, 2250.0),
        _embed_run(64, 32, 3000.0),
    ]
    picked = pick_embed_settings(runs, max_concurrency=32)
    assert picked == (16, 32, 2250.0)


def test_fit_from_reports_builds_rationale():
    chunk = {"runs": [_chunk_run(2, 15082.0, semantic=True)]}
    embed = {"runs": [_embed_run(16, 32, 2250.0)]}
    fit = fit_from_reports(chunk, embed, semantic_requested=True)
    assert fit is not None
    assert fit.chunk_concurrency == 2
    assert fit.embed_concurrency == 16
    assert fit.batch_size == 32
    assert "bench" in fit.rationale["ingest_chunk_concurrency"]
    assert "bench" in fit.rationale["ingest_embed_concurrency"]
