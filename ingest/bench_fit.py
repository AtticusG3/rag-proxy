"""Fit ingest capacity knobs from bench_ingest_capacity.py JSON reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KNEE_RATIO = 0.9


@dataclass(frozen=True)
class BenchFit:
    """Measured throughput picks to override heuristic planner defaults."""

    chunk_concurrency: int | None = None
    chunk_chunks_per_min: float | None = None
    embed_concurrency: int | None = None
    batch_size: int | None = None
    embed_chunks_per_min: float | None = None
    rationale: dict[str, str] = field(default_factory=dict)


def load_bench_report(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def _chunk_runs(report: dict | None) -> list[dict]:
    if not report:
        return []
    return [run for run in report.get("runs", []) if run.get("mode") == "chunk"]


def _embed_runs(report: dict | None) -> list[dict]:
    if not report:
        return []
    return [run for run in report.get("runs", []) if run.get("mode") == "embed"]


def pick_chunk_concurrency(
    runs: list[dict],
    *,
    semantic: bool,
    knee_ratio: float = DEFAULT_KNEE_RATIO,
) -> tuple[int, float] | None:
    """Pick the lowest concurrency within *knee_ratio* of the best measured rate."""
    matching = [run for run in runs if run.get("params", {}).get("semantic") is semantic]
    candidates = matching or list(runs)
    rated = [run for run in candidates if run.get("chunks_per_min")]
    if not rated:
        return None

    best_rate = max(float(run["chunks_per_min"]) for run in rated)
    threshold = best_rate * knee_ratio
    knee_runs = [run for run in rated if float(run["chunks_per_min"]) >= threshold]
    knee_runs.sort(key=lambda run: int(run["params"]["chunk_concurrency"]))
    chosen = knee_runs[0]
    return int(chosen["params"]["chunk_concurrency"]), float(chosen["chunks_per_min"])


def pick_embed_settings(
    runs: list[dict],
    *,
    max_concurrency: int | None = None,
) -> tuple[int, int, float] | None:
    """Pick embed concurrency and batch size with the highest measured throughput."""
    rated = [run for run in runs if run.get("chunks_per_min")]
    if not rated:
        return None

    zero_errors = [run for run in rated if int(run.get("errors") or 0) == 0]
    candidates = zero_errors or rated
    if max_concurrency is not None:
        capped = [
            run
            for run in candidates
            if int(run["params"]["embed_concurrency"]) <= max_concurrency
        ]
        if capped:
            candidates = capped

    candidates.sort(
        key=lambda run: (
            -float(run["chunks_per_min"]),
            int(run["params"]["embed_concurrency"]),
            int(run["params"]["batch_size"]),
        )
    )
    chosen = candidates[0]
    return (
        int(chosen["params"]["embed_concurrency"]),
        int(chosen["params"]["batch_size"]),
        float(chosen["chunks_per_min"]),
    )


def fit_from_reports(
    chunk_report: dict | None,
    embed_report: dict | None,
    *,
    semantic_requested: bool,
    max_embed_concurrency: int | None = None,
    knee_ratio: float = DEFAULT_KNEE_RATIO,
) -> BenchFit | None:
    """Build a :class:`BenchFit` from chunk and/or embed benchmark reports."""
    rationale: dict[str, str] = {}
    chunk_concurrency: int | None = None
    chunk_rate: float | None = None
    embed_concurrency: int | None = None
    batch_size: int | None = None
    embed_rate: float | None = None

    chunk_runs = _chunk_runs(chunk_report)
    if chunk_runs:
        picked = pick_chunk_concurrency(
            chunk_runs,
            semantic=semantic_requested,
            knee_ratio=knee_ratio,
        )
        if picked:
            chunk_concurrency, chunk_rate = picked
            rationale["ingest_chunk_concurrency"] = (
                f"{chunk_concurrency} from bench knee "
                f"({chunk_rate:.0f} chunks/min, semantic={semantic_requested})"
            )

    embed_runs = _embed_runs(embed_report)
    if embed_runs:
        picked = pick_embed_settings(embed_runs, max_concurrency=max_embed_concurrency)
        if picked:
            embed_concurrency, batch_size, embed_rate = picked
            cap_note = (
                f", capped at pool {max_embed_concurrency}"
                if max_embed_concurrency is not None
                and embed_concurrency >= max_embed_concurrency
                else ""
            )
            rationale["ingest_embed_concurrency"] = (
                f"{embed_concurrency} from bench ({embed_rate:.0f} chunks/min{cap_note})"
            )
            rationale["ingest_batch_size"] = (
                f"{batch_size} from bench ({embed_rate:.0f} chunks/min at "
                f"concurrency {embed_concurrency})"
            )

    if chunk_concurrency is None and embed_concurrency is None:
        return None

    return BenchFit(
        chunk_concurrency=chunk_concurrency,
        chunk_chunks_per_min=chunk_rate,
        embed_concurrency=embed_concurrency,
        batch_size=batch_size,
        embed_chunks_per_min=embed_rate,
        rationale=rationale,
    )


def fit_from_report_paths(
    chunk_path: str | Path | None,
    embed_path: str | Path | None,
    *,
    semantic_requested: bool,
    max_embed_concurrency: int | None = None,
) -> BenchFit | None:
    return fit_from_reports(
        load_bench_report(chunk_path),
        load_bench_report(embed_path),
        semantic_requested=semantic_requested,
        max_embed_concurrency=max_embed_concurrency,
    )
