#!/usr/bin/env python3
"""Benchmark ingest throughput across capacity planner dimensions.

Two modes:

- chunk (default, offline): generate synthetic documents and sweep chunk
  concurrency and semantic on/off, measuring chunks per minute of the CPU-bound
  chunk stage.
- embed (live): sweep embed concurrency and batch size against a running
  nomic-embed pool (no Qdrant needed), measuring embedded chunks per minute.

Output is a JSON report (stdout or --output) with the host profile attached, so
runs on different machines can be compared and planner coefficients tuned.

Examples:
  python scripts/bench_ingest_capacity.py
  python scripts/bench_ingest_capacity.py --mode embed \
      --embed-urls http://127.0.0.1:18089,http://127.0.0.1:18090
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.chunking import ChunkConfig, chunk_text, set_chunk_concurrency  # noqa: E402
from ingest.chunking_strategy import ChunkContext  # noqa: E402
from ingest.host_profile import probe_host  # noqa: E402

_SENTENCE = (
    "Ingest capacity benchmarking exercises chunking and embedding stages with "
    "repeatable synthetic prose so throughput numbers stay comparable across hosts. "
)


def synthetic_documents(count: int, *, sentences_per_doc: int = 200) -> list[str]:
    return [
        "\n\n".join(
            f"Section {section}. " + _SENTENCE * 5
            for section in range(sentences_per_doc // 5)
        )
        + f"\n\nDocument {index}."
        for index in range(count)
    ]


def bench_chunk_stage(
    documents: list[str],
    *,
    chunk_concurrency: int,
    workers: int,
    semantic: bool,
) -> dict:
    config = ChunkConfig(semantic_enabled=semantic)
    context = ChunkContext(file_type="text", source_path="/bench/doc.txt")
    set_chunk_concurrency(chunk_concurrency)
    total_chunks = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pieces in pool.map(
            lambda text: chunk_text(text, context=context, config=config),
            documents,
        ):
            total_chunks += len(pieces)
    wall_s = time.perf_counter() - start
    return {
        "mode": "chunk",
        "params": {
            "chunk_concurrency": chunk_concurrency,
            "workers": workers,
            "semantic": semantic,
            "documents": len(documents),
        },
        "wall_s": round(wall_s, 3),
        "chunks": total_chunks,
        "chunks_per_min": round(total_chunks / wall_s * 60, 1) if wall_s > 0 else None,
    }


def bench_embed_stage(
    texts: list[str],
    *,
    embed_urls: list[str],
    embed_concurrency: int,
    batch_size: int,
) -> dict:
    import httpx

    from ingest.embedder import embed_texts
    from ingest.pipeline import make_embed_semaphore

    limiter = make_embed_semaphore(embed_concurrency)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    embedded = 0
    errors = 0

    def run_batch(index_batch: tuple[int, list[str]]) -> int:
        index, batch = index_batch
        with limiter:
            try:
                vectors = embed_texts(
                    batch,
                    embed_url=embed_urls[index % len(embed_urls)],
                    embed_urls=embed_urls,
                    max_chars=2000,
                    client=client,
                )
                return len(vectors)
            except Exception:
                return -1

    start = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        with ThreadPoolExecutor(max_workers=embed_concurrency) as pool:
            for result in pool.map(run_batch, enumerate(batches)):
                if result < 0:
                    errors += 1
                else:
                    embedded += result
    wall_s = time.perf_counter() - start
    return {
        "mode": "embed",
        "params": {
            "embed_concurrency": embed_concurrency,
            "batch_size": batch_size,
            "pool_size": len(embed_urls),
            "texts": len(texts),
        },
        "wall_s": round(wall_s, 3),
        "chunks": embedded,
        "errors": errors,
        "chunks_per_min": round(embedded / wall_s * 60, 1) if wall_s > 0 else None,
    }


def run_chunk_sweep(args: argparse.Namespace) -> list[dict]:
    documents = synthetic_documents(args.documents)
    runs: list[dict] = []
    semantic_options = [False, True] if args.semantic else [False]
    for semantic in semantic_options:
        for concurrency in args.chunk_concurrency:
            runs.append(
                bench_chunk_stage(
                    documents,
                    chunk_concurrency=concurrency,
                    workers=max(concurrency, 2),
                    semantic=semantic,
                )
            )
            print(json.dumps(runs[-1]), file=sys.stderr)
    return runs


def run_embed_sweep(args: argparse.Namespace) -> list[dict]:
    if not args.embed_urls:
        raise SystemExit("--embed-urls is required for --mode embed")
    urls = [url.strip() for url in args.embed_urls.split(",") if url.strip()]
    texts = [piece for doc in synthetic_documents(args.documents) for piece in doc.split("\n\n")]
    runs: list[dict] = []
    for concurrency in args.embed_concurrency:
        for batch_size in args.batch_size:
            runs.append(
                bench_embed_stage(
                    texts,
                    embed_urls=urls,
                    embed_concurrency=concurrency,
                    batch_size=batch_size,
                )
            )
            print(json.dumps(runs[-1]), file=sys.stderr)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=("chunk", "embed"), default="chunk")
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument(
        "--chunk-concurrency", type=int, nargs="+", default=[1, 2, 4]
    )
    parser.add_argument(
        "--semantic", action="store_true", help="Also sweep semantic chunking (chunk mode)."
    )
    parser.add_argument("--embed-urls", default="")
    parser.add_argument("--embed-concurrency", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--output", default="", help="Write JSON report to this path.")
    args = parser.parse_args()

    runs = run_chunk_sweep(args) if args.mode == "chunk" else run_embed_sweep(args)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": asdict(probe_host()),
        "runs": runs,
        "best": max(
            (run for run in runs if run.get("chunks_per_min")),
            key=lambda run: run["chunks_per_min"],
            default=None,
        ),
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"report written to {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
