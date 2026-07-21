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
from ingest.chunking import ChunkConfig, load_chunk_config
from ingest.chunking import chunk_text, set_chunk_concurrency
from ingest.chunking_strategy import ChunkContext
from ingest.embed_lifecycle import ensure_embed_urls
from ingest.embed_urls import parse_ingest_embed_urls
from ingest.pdf_reader import iter_pdf_pages
from ingest.pipeline import make_embed_semaphore, run_ingest_pipeline
from ingest.qdrant_writer import delete_by_source, list_point_ids_by_source
from ingest.scanner import scan_storage
from ingest.types import determine_file_type, IngestAborted
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
    file_concurrency: int | None = None
    chunk_concurrency: int | None = None
    chunk_config: ChunkConfig = field(default_factory=load_chunk_config)
    memgraphrag_db_path: str = ""


def resolve_file_concurrency(
    embed_urls: list[str],
    *,
    explicit: int | None = None,
) -> int:
    """Worker thread count: explicit env/config, else max(1, min(4, len(embed_urls)))."""
    if explicit is not None and explicit > 0:
        return explicit
    env_raw = os.getenv("INGEST_FILE_CONCURRENCY", "").strip()
    if env_raw:
        return max(1, int(env_raw))
    pool_size = len(embed_urls) if embed_urls else 1
    return max(1, min(4, pool_size))


UpdateStateFn = Callable[..., None]


def _read_text_file(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    title = Path(path).stem.replace("_", " ").title()
    return title, text


def _pdf_page_carry(text: str) -> str:
    """Last paragraph from a page for cross-page chunk overlap."""
    paragraphs = [piece.strip() for piece in text.split("\n\n") if piece.strip()]
    if not paragraphs:
        return ""
    return paragraphs[-1]


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
        title_base = Path(file_path).stem.replace("_", " ").title()
        carry = ""
        for page_label, page_text in iter_pdf_pages(file_path):
            if not page_text.strip():
                continue
            text_for_chunk = page_text
            if carry:
                text_for_chunk = f"{carry}\n\n{page_text}"
            page_title = f"{title_base} ({page_label})"
            for piece in chunk_text(text_for_chunk, context=chunk_ctx, config=chunk_config):
                yield page_title, source, piece
            carry = _pdf_page_carry(page_text)
        return

    raise ValueError(f"Unsupported file type for {file_path}")


def process_file(
    file_path: str,
    config: IngestConfig,
    *,
    on_progress: UpdateStateFn | None = None,
    embed_limiter: threading.Semaphore | None = None,
    should_abort: Callable[[], bool] | None = None,
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
        embed_limiter=embed_limiter,
        on_progress=on_progress,
        should_abort=should_abort,
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
        from ingest.sidecar_lifecycle import ensure_sparse_sidecar

        ensure_sparse_sidecar(self.config.sparse_index_url, wait_health=True)
        log.info("sparse reindex flush (ingest queue idle)")
        trigger_sparse_reindex(self.config)


class IngestWorker:
    """Multi-threaded job processor with SQLite-backed state."""

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
        self._abort = threading.Event()
        self._paused = False
        self._preempt_gen = 0
        self._workers: list[tuple[threading.Thread, threading.Event]] = []
        self._worker_seq = 0
        self._embed_limiter = make_embed_semaphore(config.embed_concurrency)
        if config.chunk_concurrency:
            set_chunk_concurrency(config.chunk_concurrency)

    def _file_worker_count(self) -> int:
        urls = self.config.embed_urls or parse_ingest_embed_urls(
            embed_url=self.config.embed_url
        )
        return resolve_file_concurrency(urls, explicit=self.config.file_concurrency)

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._abort.set()
        else:
            self._abort.clear()

    def _should_abort(self) -> bool:
        return self._stop.is_set() or self._abort.is_set()

    def update_config(self, config: IngestConfig) -> None:
        with self._lock:
            if config.embed_concurrency != self.config.embed_concurrency:
                self._embed_limiter = make_embed_semaphore(config.embed_concurrency)
            if (
                config.chunk_concurrency
                and config.chunk_concurrency != self.config.chunk_concurrency
            ):
                set_chunk_concurrency(config.chunk_concurrency)
            self.config = config
            self._sparse.config = config
        if self._alive_workers():
            self.resize_file_workers(self._file_worker_count())

    def _alive_workers(self) -> list[tuple[threading.Thread, threading.Event]]:
        return [(thread, event) for thread, event in self._workers if thread.is_alive()]

    def resize_file_workers(self, target: int) -> None:
        """Grow or shrink file worker threads; excess threads exit between files."""
        target = max(1, target)
        with self._lock:
            self._workers = self._alive_workers()
            current = len(self._workers)
            if current > target:
                for _, event in self._workers[target:]:
                    event.set()
                self._workers = self._workers[:target]
                log.info("ingest file workers shrinking %d -> %d", current, target)
                return
            for _ in range(current, target):
                event = threading.Event()
                thread = threading.Thread(
                    target=self._run_loop,
                    args=(event,),
                    daemon=True,
                    name=f"ingest-worker-{self._worker_seq}",
                )
                self._worker_seq += 1
                thread.start()
                self._workers.append((thread, event))
            if target > current:
                log.info("ingest file workers growing %d -> %d", current, target)

    def start(self) -> None:
        if self._alive_workers():
            self.resize_file_workers(self._file_worker_count())
            return
        self._stop.clear()
        self._recover_interrupted_running()
        self._workers = []
        self.resize_file_workers(self._file_worker_count())

    def stop(self, *, flush_sparse: bool = False) -> None:
        """Signal workers to exit. Skip BM25 flush on shutdown (can block minutes)."""
        self._stop.set()
        self._abort.set()
        if flush_sparse:
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

    def drain_active_files(self, *, timeout_s: float = 3600.0, poll_s: float = 2.0) -> bool:
        """Wait until no files are mid-ingest (status=running)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.db.list_running_files():
                return True
            time.sleep(poll_s)
        return False

    def running_file_count(self) -> int:
        return len(self.db.list_running_files())

    def enqueue_file(self, file_path: str) -> str:
        job_id = str(uuid.uuid4())
        self.db.create_job(job_id, job_type="file", message=file_path)
        self.db.upsert_file_state(file_path, status="pending")
        return job_id

    def preempt_running(self) -> int:
        """Abort files mid-ingest and re-queue them at the back of their priority band.

        Workers then claim the current top of the queue (high -> mid -> low).
        Returns the number of running files that were told to yield.
        """
        running = len(self.db.list_running_files())
        if running == 0:
            return 0
        self._preempt_gen += 1
        job_id = str(uuid.uuid4())
        message = f"preempted {running} running file(s); switching to top of queue"
        self.db.create_job(job_id, job_type="preempt", message=message)
        self.db.update_job(job_id, status="done", message=message)
        log.info("ingest preempt requested: %s running file(s)", running)
        return running

    def _run_loop(self, local_stop: threading.Event) -> None:
        while not self._stop.is_set() and not local_stop.is_set():
            if self._paused:
                time.sleep(1.0)
                continue
            self._fail_stalled_running()
            claimed = self.db.claim_pending_file()
            if claimed is None:
                self._sparse.flush()
                time.sleep(1.0)
                continue
            file_path = claimed["file_path"]
            if not os.path.isfile(file_path):
                log.warning("ingest file missing, removing state: %s", file_path)
                self.remove_file_from_index(file_path)
                continue
            preempt_gen = self._preempt_gen
            try:
                self._process_one(file_path, preempt_gen=preempt_gen)
            except IngestAborted:
                if self._stop.is_set():
                    return
                reason = (
                    "paused mid-ingest"
                    if self._paused
                    else "preempted, yielded to top of queue"
                )
                self.db.update_file_state(
                    file_path,
                    status="pending",
                    last_error=reason,
                )
                log.info("ingest aborted mid-file (%s), re-queued: %s", reason, file_path)
            except Exception as exc:
                log.exception("ingest failed for %s", file_path)
                self.db.update_file_state(
                    file_path,
                    status="failed",
                    last_error=str(exc),
                    finished_at=_utc_now(),
                )

    def _process_one(self, file_path: str, *, preempt_gen: int | None = None) -> None:
        urls = self.config.embed_urls or parse_ingest_embed_urls(
            embed_url=self.config.embed_url
        )
        ensure_embed_urls(urls)

        def on_progress(**kwargs: object) -> None:
            chunks = kwargs.get("chunks_embedded")
            if isinstance(chunks, int):
                self.db.update_file_state(file_path, chunks_embedded=chunks)

        def should_abort() -> bool:
            if self._should_abort():
                return True
            return preempt_gen is not None and self._preempt_gen != preempt_gen

        count = process_file(
            file_path,
            self.config,
            on_progress=on_progress,
            embed_limiter=self._embed_limiter,
            should_abort=should_abort,
        )

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
        """Fully drop a document: MemGraphRAG, dense Qdrant, disk, ingest state, BM25."""
        self._scrub_memgraphrag_for_source(file_path)
        delete_by_source(
            self.config.qdrant_url,
            self.config.qdrant_collection,
            file_path,
        )
        if os.path.isfile(file_path):
            os.remove(file_path)
        part_path = f"{file_path}.part"
        if os.path.isfile(part_path):
            os.remove(part_path)
        self.db.delete_file_state(file_path)
        trigger_sparse_reindex(self.config)

    def _scrub_memgraphrag_for_source(self, file_path: str) -> None:
        """Remove MemGraphRAG passages whose chunk_ids match this source's Qdrant points."""
        db_path = (self.config.memgraphrag_db_path or os.getenv("MEMGRAPHRAG_DB_PATH", "")).strip()
        if not db_path or not os.path.isfile(db_path):
            return
        try:
            point_ids = list_point_ids_by_source(
                self.config.qdrant_url,
                self.config.qdrant_collection,
                file_path,
            )
            if not point_ids:
                return
            from rag_proxy.memgraphrag.memory import load_memory

            memory = load_memory(db_path)
            removed = memory.remove_passages_by_chunk_ids(set(point_ids))
            if removed:
                memory.save(db_path)
                log.info(
                    "memgraphrag scrubbed %d passages for source=%s",
                    removed,
                    file_path,
                )
        except Exception as exc:
            log.warning("memgraphrag scrub failed for %s: %s", file_path, exc)

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
