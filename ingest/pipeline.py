"""Pipelined bulk embed + Qdrant upsert for ingest."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import httpx

from ingest.embedder import embed_texts
from ingest.qdrant_writer import build_point, ensure_collection, upsert_points

log = logging.getLogger("ingest.pipeline")

UpdateStateFn = Callable[..., None]
ChunkBatch = list[tuple[str, str, str]]
PendingBatch = tuple[ChunkBatch, int, Future[list[list[float]]]]


def chunk_batches(
    chunks: Iterator[tuple[str, str, str]],
    batch_size: int,
) -> Iterator[ChunkBatch]:
    batch: ChunkBatch = []
    for item in chunks:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _upsert_batch(
    batch: ChunkBatch,
    embeddings: list[list[float]],
    start_chunk_idx: int,
    *,
    qdrant_url: str,
    collection: str,
    qdrant_client: httpx.Client,
) -> int:
    points = [
        build_point(
            text=text,
            source=source,
            title=title,
            chunk_idx=start_chunk_idx + i,
            embedding=embeddings[i],
        )
        for i, (title, source, text) in enumerate(batch)
    ]
    upsert_points(qdrant_url, collection, points, client=qdrant_client)
    return len(points)


def _drain_next_upsert(
    pending: dict[int, PendingBatch],
    next_upsert: int,
    *,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_client: httpx.Client,
    on_progress: UpdateStateFn | None,
    total: int,
) -> tuple[int, int]:
    batch, start_idx, future = pending.pop(next_upsert)
    embeddings = future.result()
    if len(embeddings) != len(batch):
        raise RuntimeError(
            f"embed count mismatch: expected {len(batch)}, got {len(embeddings)}"
        )
    count = _upsert_batch(
        batch,
        embeddings,
        start_idx,
        qdrant_url=qdrant_url,
        collection=qdrant_collection,
        qdrant_client=qdrant_client,
    )
    total += count
    if on_progress:
        on_progress(chunks_embedded=total)
    return total, next_upsert + 1


def _drain_ready_upserts(
    pending: dict[int, PendingBatch],
    next_upsert: int,
    *,
    qdrant_url: str,
    qdrant_collection: str,
    qdrant_client: httpx.Client,
    on_progress: UpdateStateFn | None,
    total: int,
) -> tuple[int, int]:
    """Upsert completed batches in order as soon as they are ready."""
    while next_upsert in pending and pending[next_upsert][2].done():
        total, next_upsert = _drain_next_upsert(
            pending,
            next_upsert,
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            qdrant_client=qdrant_client,
            on_progress=on_progress,
            total=total,
        )
    return total, next_upsert


def run_ingest_pipeline(
    chunks: Iterator[tuple[str, str, str]],
    *,
    embed_url: str = "",
    embed_urls: list[str] | None = None,
    qdrant_url: str,
    qdrant_collection: str,
    batch_size: int,
    embed_max_chars: int,
    embed_concurrency: int,
    on_progress: UpdateStateFn | None = None,
) -> int:
    """Embed batches concurrently and upsert to Qdrant in chunk order."""
    urls = embed_urls or ([embed_url] if embed_url else [])
    if not urls:
        raise ValueError("embed_urls or embed_url is required")
    batch_size = max(1, batch_size)
    concurrency = max(1, embed_concurrency)
    total = 0
    next_seq = 0
    next_upsert = 0
    chunk_start = 0
    pending: dict[int, PendingBatch] = {}

    qdrant_client = httpx.Client(timeout=120.0)
    try:
        ensure_collection(qdrant_url, qdrant_collection, client=qdrant_client)
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="ingest-embed",
        ) as pool:
            for batch in chunk_batches(chunks, batch_size):
                texts = [chunk[2] for chunk in batch]
                target_url = urls[next_seq % len(urls)]
                future = pool.submit(
                    embed_texts,
                    texts,
                    embed_url=target_url,
                    embed_urls=urls,
                    max_chars=embed_max_chars,
                )
                pending[next_seq] = (batch, chunk_start, future)
                chunk_start += len(batch)
                next_seq += 1

                total, next_upsert = _drain_ready_upserts(
                    pending,
                    next_upsert,
                    qdrant_url=qdrant_url,
                    qdrant_collection=qdrant_collection,
                    qdrant_client=qdrant_client,
                    on_progress=on_progress,
                    total=total,
                )

                while (next_seq - next_upsert) > concurrency:
                    total, next_upsert = _drain_next_upsert(
                        pending,
                        next_upsert,
                        qdrant_url=qdrant_url,
                        qdrant_collection=qdrant_collection,
                        qdrant_client=qdrant_client,
                        on_progress=on_progress,
                        total=total,
                    )

            while next_upsert < next_seq:
                total, next_upsert = _drain_next_upsert(
                    pending,
                    next_upsert,
                    qdrant_url=qdrant_url,
                    qdrant_collection=qdrant_collection,
                    qdrant_client=qdrant_client,
                    on_progress=on_progress,
                    total=total,
                )
    finally:
        qdrant_client.close()

    return total
