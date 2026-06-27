"""Async JSONL writer for transcript capture streams."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rag_proxy.config import settings
from rag_proxy.rag_corpus_promoter import promote_rag_record

log = logging.getLogger("rag-proxy")

_writer_task: asyncio.Task[None] | None = None
_writer_queue: asyncio.Queue[dict[str, Any] | None] | None = None


def _target_path(record: dict[str, Any]) -> Path:
    record_type = record.get("record_type")
    if record_type == "finetune_turn":
        path = Path(settings.finetune_log_path)
    elif record_type == "rag_turn":
        path = Path(settings.rag_improvement_log_path)
    else:
        raise ValueError(f"unknown capture record_type: {record_type!r}")

    return path


def _append_jsonl(record: dict[str, Any]) -> None:
    path = _target_path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


async def startup_capture_writer() -> None:
    """Start the background writer when transcript capture is enabled."""
    global _writer_queue, _writer_task
    if not settings.enable_transcript_capture or _writer_task is not None:
        return
    _writer_queue = asyncio.Queue()
    _writer_task = asyncio.create_task(_writer_loop())
    log.info("transcript capture writer started")


async def shutdown_capture_writer() -> None:
    """Flush pending records and stop the background writer."""
    global _writer_queue, _writer_task
    if _writer_task is None or _writer_queue is None:
        return
    await _writer_queue.put(None)
    try:
        await _writer_task
    finally:
        _writer_queue = None
        _writer_task = None


def enqueue_records(records: list[dict[str, Any]]) -> None:
    """Queue records for JSONL append; never raise into the request path."""
    if not records:
        return
    if not settings.enable_transcript_capture:
        return
    if _writer_queue is None:
        log.warning("transcript capture enabled but writer is not started")
        return
    try:
        for record in records:
            _writer_queue.put_nowait(record)
    except Exception as e:
        log.warning("transcript capture enqueue failed: %s", e)


async def _writer_loop() -> None:
    if _writer_queue is None:
        return
    while True:
        record = await _writer_queue.get()
        try:
            if record is None:
                return
            try:
                await asyncio.to_thread(_append_jsonl, record)
                if record.get("record_type") == "rag_turn":
                    await promote_rag_record(record)
            except Exception as e:
                log.warning("transcript capture write failed: %s", e)
        finally:
            _writer_queue.task_done()
