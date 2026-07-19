"""Tests for parallel chunk execution (per-thread runners + concurrency cap)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import ingest.chunking as chunking
from ingest.chunking import ChunkConfig, set_chunk_concurrency
from ingest.chunking_strategy import ChunkStrategy


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_count = 1


class _SlowRunner:
    """Records overlapping executions to prove (non-)parallelism."""

    def __init__(self, tracker: dict[str, int], lock: threading.Lock) -> None:
        self._tracker = tracker
        self._lock = lock

    def __call__(self, text: str) -> list[_FakeChunk]:
        with self._lock:
            self._tracker["active"] += 1
            self._tracker["max_active"] = max(
                self._tracker["max_active"], self._tracker["active"]
            )
        time.sleep(0.05)
        with self._lock:
            self._tracker["active"] -= 1
        return [_FakeChunk(text)]


def _run_parallel(concurrency_cap: int, workers: int) -> int:
    tracker = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    config = ChunkConfig()
    set_chunk_concurrency(concurrency_cap)
    try:
        with patch.object(
            chunking,
            "_build_runner",
            side_effect=lambda *args: _SlowRunner(tracker, lock),
        ):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        chunking._run_strategy,
                        ChunkStrategy.RECURSIVE,
                        f"text {i}",
                        tokenizer="word",
                        config=config,
                    )
                    for i in range(workers * 2)
                ]
                results = [future.result() for future in futures]
        assert all(results)
    finally:
        set_chunk_concurrency(chunking._env_chunk_concurrency())
    return tracker["max_active"]


def test_chunking_runs_in_parallel_up_to_cap() -> None:
    assert _run_parallel(concurrency_cap=4, workers=4) >= 2


def test_chunk_concurrency_cap_of_one_serializes() -> None:
    assert _run_parallel(concurrency_cap=1, workers=4) == 1


def test_runners_are_per_thread() -> None:
    # Keep references to the runner objects (not their id()); once a thread ends its
    # thread-local runner is GC'd and a later thread can reuse the same address, so
    # comparing id() is flaky under load. Holding the objects keeps them distinct.
    runners: list[object] = []
    lock = threading.Lock()

    def collect() -> None:
        runner = chunking._thread_runner("recursive", "word", 64, 0, "model")
        again = chunking._thread_runner("recursive", "word", 64, 0, "model")
        assert runner is again  # cached within a thread
        with lock:
            runners.append(runner)

    threads = [threading.Thread(target=collect) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(runners) == 2
    assert runners[0] is not runners[1]  # distinct instances across threads
