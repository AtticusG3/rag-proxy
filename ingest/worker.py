"""Background ingest worker."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from ingest.db import IngestDatabase
from ingest.chunk_config import ChunkConfig, load_chunk_config
from ingest.chunking import chunk_text
from ingest.chunking_strategy import ChunkContext
from ingest.embed_urls import parse_ingest_embed_urls
from ingest.pdf_reader import read_pdf_text
from ingest.pipeline import run_ingest_pipeline
from ingest.qdrant_writer import delete_by_source
from ingest.scanner import scan_storage
from ingest.types import determine_file_type
from ingest.zim_reader import iter_zim_articles

from ingest.stall import interrupt_error_message, is_stalled, stall_error_message

log = logging.getLogger("ingest.worker")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IngestConfig:
    zim_dir: str
    upload_dir: str
    embed_url: str
    qdrant_url: str
    qdrant_collection: str
    sparse_index_url: str
    batch_size: int = 32
    embed_concurrency: int = 4
    max_articles: int = 0
    embed_max_chars: int = 2000
    sparse_reindex_mode: str = "idle"
    stall_seconds: int = 900
    embed_urls: list[str] | None = None
    chunk_config: ChunkConfig = field(default_factory=load_chunk_config)


UpdateStateFn = Callable[..., None]


def _read_text_file(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    title = Path(path).stem.replace("_", " ").title()
    return title, text


def _iter_chunks_for_file(
    file_path: str,
    *,
    max_articles: int,
    chunk_config: ChunkConfig,
) -> Iterator[tuple[str, str, str]]:
    """Yield (title, source, chunk_text) without loading whole ZIMs into RAM."""
    file_type = determine_file_type(file_path)
    source = file_path
    chunk_ctx = ChunkContext.from_path(file_path, file_type)

    if file_type == "text":
        title, text = _read_text_file(file_path)
        for piece in chunk_text(text, context=chunk_ctx, config=chunk_config):
            yield title, source, piece
        return

    if file_type == "zim":
        for article in iter_zim_articles(file_path, max_articles=max_articles):
            for piece in chunk_text(article.text, context=chunk_ctx, config=chunk_config):
                yield article.title, source, piece
        return

    if file_type == "pdf":
        title, text = read_pdf_text(file_path)
        if not text.strip():
            return
        for piece in chunk_text(text, context=chunk_ctx, config=chunk_config):
            yield title, source, piece
        return

    raise ValueError(f"Unsupported file type for {file_path}")


def process_file(
    file_path: str,
    config: IngestConfig,
    *,
    on_progress: UpdateStateFn | None = None,
) -> int:
    """Embed one file into Qdrant. Returns total chunks embedded."""
    chunk_iter = _iter_chunks_for_file(
        file_path,
        max_articles=config.max_articles,
        chunk_config=config.chunk_config,
    )
    return run_ingest_pipeline(
        chunk_iter,
        embed_url=config.embed_url,
        embed_urls=config.embed_urls
        or parse_ingest_embed_urls(embed_url=config.embed_url),
        qdrant_url=config.qdrant_url,
        qdrant_collection=config.qdrant_collection,
        batch_size=config.batch_size,
        embed_max_chars=config.embed_max_chars,
        embed_concurrency=config.embed_concurrency,
        on_progress=on_progress,
    )


def trigger_sparse_reindex(config: IngestConfig) -> int | None:
    if not config.sparse_index_url:
        return None
    url = f"{config.sparse_index_url.rstrip('/')}/reindex"
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json={"collection": config.qdrant_collection})
            response.raise_for_status()
            return int(response.json().get("docs", 0))
    except Exception as exc:
        log.warning("sparse reindex failed: %s", exc)
        return None


class SparseReindexScheduler:
    """Avoid full-collection BM25 rebuild after every ingested file."""

    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self._dirty = False
        self._lock = threading.Lock()

    def after_file(self) -> None:
        mode = self.config.sparse_reindex_mode.lower()
        if mode == "off":
            return
        if mode == "each":
            trigger_sparse_reindex(self.config)
            return
        with self._lock:
            self._dirty = True

    def flush(self) -> None:
        mode = self.config.sparse_reindex_mode.lower()
        if mode == "off":
            return
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
        log.info("sparse reindex flush (ingest queue idle)")
        trigger_sparse_reindex(self.config)


class IngestWorker:
    """Single-threaded job processor with SQLite-backed state."""

    def __init__(
        self,
        config: IngestConfig,
        db: IngestDatabase,
    ) -> None:
        self.config = config
        self.db = db
        self._sparse = SparseReindexScheduler(config)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = False
        self._thread: threading.Thread | None = None

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def update_config(self, config: IngestConfig) -> None:
        with self._lock:
            self.config = config
            self._sparse.config = config

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._recover_interrupted_running()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ingest-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._sparse.flush()

    def enqueue_sync(self) -> str:
        """Scan storage and queue only new or previously failed files."""
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="sync")
        directories = [self.config.zim_dir, self.config.upload_dir]
        queued_new = 0
        queued_retry = 0
        for path in scan_storage(*directories):
            row = self.db.get_file_state(path)
            if row is None:
                self.db.upsert_file_state(
                    path,
                    status="pending",
                    file_type=determine_file_type(path),
                )
                queued_new += 1
            elif row["status"] == "failed":
                self.db.retry_file_state(path)
                queued_retry += 1
        pruned = self.prune_missing_files()
        message = f"scan: {queued_new} new, {queued_retry} retries, {len(pruned)} missing removed"
        self.db.update_job(job_id, status="queued", message=message)
        return job_id

    def retry_file(self, file_path: str) -> str:
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="retry", message=file_path)
        self._prepare_file_restart(file_path)
        return job_id

    def restart_stalled_files(self) -> str:
        """Re-queue running files that have not updated recently."""
        job_id = str(uuid.uuid4())
        stalled = self._list_stalled_running()
        restarted: list[str] = []
        for row in stalled:
            path = row["file_path"]
            self._prepare_file_restart(path)
            restarted.append(Path(path).name)
        message = (
            f"restarted {len(restarted)} stalled file(s)"
            if restarted
            else "no stalled files"
        )
        self.db.create_job(job_id, job_type="restart_stalled", message=message)
        self.db.update_job(job_id, status="done", message=message)
        return job_id

    def _prepare_file_restart(self, file_path: str) -> None:
        row = self.db.get_file_state(file_path)
        if row is None:
            self.db.upsert_file_state(
                file_path,
                status="pending",
                file_type=determine_file_type(file_path),
            )
            return
        delete_by_source(
            self.config.qdrant_url,
            self.config.qdrant_collection,
            file_path,
        )
        if not self.db.retry_file_state(file_path, reset_chunks=True):
            self.db.upsert_file_state(
                file_path,
                status="pending",
                file_type=determine_file_type(file_path),
                chunks_embedded=0,
            )

    def retry_all_failed(self) -> str:
        job_id = str(uuid.uuid4())
        failed = self.db.list_failed_files()
        self.db.create_job(job_id, job_type="retry_failed")
        for row in failed:
            self._prepare_file_restart(row["file_path"])
        self.db.update_job(
            job_id,
            status="queued",
            message=f"{len(failed)} failed file(s) re-queued",
        )
        return job_id

    def requeue_all_files(self) -> str:
        """Re-queue every on-disk ingest file (clears Qdrant points per source first)."""
        job_id = str(uuid.uuid4())
        rows = self.db.list_file_states()
        requeued = 0
        skipped = 0
        self.db.create_job(job_id, job_type="requeue_all")
        for row in rows:
            file_path = str(row["file_path"])
            if not os.path.isfile(file_path):
                skipped += 1
                continue
            self._prepare_file_restart(file_path)
            requeued += 1
        message = f"{requeued} file(s) re-queued"
        if skipped:
            message += f", {skipped} missing on disk skipped"
        self.db.update_job(job_id, status="queued", message=message)
        return job_id

    def enqueue_file(self, file_path: str) -> str:
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="file", message=file_path)
        self.db.upsert_file_state(file_path, status="pending")
        return job_id

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            if self._paused:
                time.sleep(1.0)
                continue
            self._fail_stalled_running()
            pending = self.db.list_pending_files(limit=1)
            if not pending:
                self._sparse.flush()
                time.sleep(1.0)
                continue
            file_path = pending[0]["file_path"]
            if not os.path.isfile(file_path):
                log.warning("ingest file missing, removing state: %s", file_path)
                self.remove_file_from_index(file_path)
                continue
            try:
                self._process_one(file_path)
            except Exception as exc:
                log.exception("ingest failed for %s", file_path)
                self.db.update_file_state(
                    file_path,
                    status="failed",
                    last_error=str(exc),
                    finished_at=_utc_now(),
                )

    def _process_one(self, file_path: str) -> None:
        with self._lock:
            self.db.update_file_state(
                file_path,
                status="running",
                started_at=_utc_now(),
                last_error=None,
            )

        def on_progress(**kwargs: object) -> None:
            chunks = kwargs.get("chunks_embedded")
            if isinstance(chunks, int):
                self.db.update_file_state(file_path, chunks_embedded=chunks)

        count = process_file(file_path, self.config, on_progress=on_progress)

        with self._lock:
            self.db.update_file_state(
                file_path,
                status="indexed",
                chunks_embedded=count,
                finished_at=_utc_now(),
            )
        self._sparse.after_file()

    def _recover_interrupted_running(self) -> None:
        for row in self.db.list_running_files():
            message = interrupt_error_message(int(row.get("chunks_embedded") or 0))
            self.db.update_file_state(
                row["file_path"],
                status="failed",
                last_error=message,
                finished_at=_utc_now(),
            )
            log.warning(
                "recovered interrupted ingest: %s (%s)",
                row["file_path"],
                message,
            )

    def _fail_stalled_row(self, row: dict[str, object]) -> None:
        file_path = str(row["file_path"])
        chunks = int(row.get("chunks_embedded") or 0)
        message = stall_error_message(
            stall_seconds=self.config.stall_seconds,
            chunks_embedded=chunks,
        )
        self.db.update_file_state(
            file_path,
            status="failed",
            last_error=message,
            finished_at=_utc_now(),
        )
        log.warning("marked stalled ingest failed: %s", file_path)

    def _fail_stalled_running(self) -> None:
        for row in self._list_stalled_running():
            self._fail_stalled_row(row)

    def _list_stalled_running(self) -> list[dict[str, object]]:
        return [
            row
            for row in self.db.list_running_files()
            if is_stalled(row.get("updated_at"), self.config.stall_seconds)
        ]

    def remove_file_from_index(self, file_path: str) -> None:
        delete_by_source(
            self.config.qdrant_url,
            self.config.qdrant_collection,
            file_path,
        )
        if os.path.isfile(file_path):
            os.remove(file_path)
        self.db.delete_file_state(file_path)
        trigger_sparse_reindex(self.config)

    def prune_missing_files(self) -> list[str]:
        """Drop ingest rows whose files no longer exist on disk."""
        removed: list[str] = []
        for row in self.db.list_file_states():
            path = str(row["file_path"])
            if os.path.isfile(path):
                continue
            log.warning("pruning missing ingest file: %s", path)
            delete_by_source(
                self.config.qdrant_url,
                self.config.qdrant_collection,
                path,
            )
            self.db.delete_file_state(path)
            removed.append(path)
        return removed

    def dismiss_all_missing_files(self) -> list[str]:
        return self.prune_missing_files()
