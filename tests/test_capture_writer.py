"""Async transcript writer side effects."""

import asyncio
import json

from rag_proxy import capture_writer
from rag_proxy.config import settings


def test_capture_writer_flushes_records_to_stream_paths(tmp_path, monkeypatch):
    """Shutdown must flush queued records so completed turns are durable."""
    ft_path = tmp_path / "finetune.jsonl"
    rag_path = tmp_path / "rag.jsonl"
    monkeypatch.setattr(settings, "enable_transcript_capture", True)
    monkeypatch.setattr(settings, "finetune_log_path", str(ft_path))
    monkeypatch.setattr(settings, "rag_improvement_log_path", str(rag_path))

    async def _run() -> None:
        await capture_writer.startup_capture_writer()
        capture_writer.enqueue_records(
            [
                {"record_type": "finetune_turn", "trace_id": "ft"},
                {"record_type": "rag_turn", "trace_id": "rag"},
            ]
        )
        await capture_writer.shutdown_capture_writer()

    asyncio.run(_run())

    assert json.loads(ft_path.read_text(encoding="utf-8").strip())["trace_id"] == "ft"
    assert json.loads(rag_path.read_text(encoding="utf-8").strip())["trace_id"] == "rag"


def test_capture_writer_write_error_does_not_raise(monkeypatch):
    """Capture persistence failures must not escape the writer task."""
    monkeypatch.setattr(settings, "enable_transcript_capture", True)

    async def fail_append(*_args):
        raise OSError("disk full")

    async def _run() -> None:
        await capture_writer.startup_capture_writer()
        monkeypatch.setattr(capture_writer.asyncio, "to_thread", fail_append)
        capture_writer.enqueue_records([{"record_type": "rag_turn", "trace_id": "rag"}])
        await capture_writer.shutdown_capture_writer()

    asyncio.run(_run())
