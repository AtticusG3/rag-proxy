"""Thread-safety smoke tests for ingest chunking."""

from __future__ import annotations

import threading

from ingest.chunking import ChunkConfig, chunk_text
from ingest.chunking_strategy import ChunkContext


def test_chunk_text_runs_concurrently_without_error() -> None:
    texts = [f"Paragraph {index} with enough words to chunk safely." for index in range(6)]
    errors: list[Exception] = []
    results: list[int] = []
    lock = threading.Lock()

    def worker(text: str) -> None:
        try:
            pieces = chunk_text(
                text,
                context=ChunkContext(file_type="text", source_path="/tmp/doc.txt"),
                config=ChunkConfig(chunk_size=32, chunk_overlap=4, semantic_enabled=False),
            )
            with lock:
                results.append(len(pieces))
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(text,)) for text in texts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)

    assert not errors
    assert len(results) == len(texts)
    assert all(count >= 1 for count in results)
