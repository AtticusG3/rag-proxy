#!/usr/bin/env python3
"""Benchmark-driven ingest capacity scale (rag-admin stays running).

Pauses and drain are handled by the admin UI before this script starts.
This script frees GPU embed units, runs chunk+embed benchmarks, applies the
capacity plan twice (pool for embed bench, then final), and restores query
embed on :8089.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.port_avoidance import apply_config_env  # noqa: E402

DEFAULT_POOL_ENV = "/opt/ai/config/nomic-embed-pool.env"
DEFAULT_SCALE_ENV = "/opt/ai/config/nomic-embed-scale.env"
QUERY_EMBED_UNIT = "nomic-embed.service"
BENCH_CHUNK_CONCURRENCY = (1, 2, 3, 4)
BENCH_EMBED_CONCURRENCY = (8, 16, 32, 48, 64)
BENCH_BATCH_SIZES = (32, 64, 128)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _python() -> str:
    return os.environ.get("PYTHON", sys.executable)


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=False)


def _read_pool_urls(pool_env: Path) -> str:
    if not pool_env.is_file():
        return ""
    for line in pool_env.read_text(encoding="utf-8").splitlines():
        if line.startswith("INGEST_EMBED_URLS="):
            return line.split("=", 1)[1].strip()
    return ""


def stop_embed_stack(scale_env: Path) -> None:
    """Stop query + pool embed systemd units and reconcile stray GPU processes."""
    scale = _load_module("scale_ingest_capacity", REPO_ROOT / "scripts" / "scale_ingest_capacity.py")
    host = _load_module("bench_ingest_host", REPO_ROOT / "scripts" / "bench_ingest_host.py")
    host._load_bench_env(scale_env)

    print("[stop] nomic-embed pool and query embed (:8089)", flush=True)
    scale._stop_disable_unit(QUERY_EMBED_UNIT)
    for port in host._bench_stop_ports(scale_env):
        scale._stop_disable_unit(f"nomic-embed@{port}.service")
    scale._kill_stray_gpu_embeds(set())


def wait_gpu_clear(*, timeout_s: float = 90.0) -> None:
    scale = _load_module("scale_ingest_capacity", REPO_ROOT / "scripts" / "scale_ingest_capacity.py")
    host = _load_module("bench_ingest_host", REPO_ROOT / "scripts" / "bench_ingest_host.py")
    deadline = time.time() + timeout_s
    print("[stop] waiting for embed llama-server processes to exit", flush=True)
    while time.time() < deadline:
        if not scale._query_gpu_llama_pids():
            return
        host.cmd_kill_strays(argparse.Namespace())
        time.sleep(2.0)
    print("warning: llama-server still on GPU (chat model may be loaded)", file=sys.stderr, flush=True)


def restart_query_embed() -> None:
    scale = _load_module("scale_ingest_capacity", REPO_ROOT / "scripts" / "scale_ingest_capacity.py")
    try:
        from ingest.embed_lifecycle import uses_dedicated_query_embed_unit
    except ImportError:
        uses_dedicated_query_embed_unit = lambda: True  # type: ignore[assignment,misc]
    if not uses_dedicated_query_embed_unit():
        print(
            "[skip] nomic-embed.service (:8089) not started; EMBED_URL uses pool port",
            flush=True,
        )
        return
    print("[restart] query embed (:8089)", flush=True)
    scale._systemctl("enable", QUERY_EMBED_UNIT, check=False)
    scale._systemctl("restart", QUERY_EMBED_UNIT, check=False)


def run_scale_apply(
    *,
    pool_env: Path,
    scale_env: Path,
    semantic_requested: str | None,
    chunk_bench: Path | None = None,
    embed_bench: Path | None = None,
) -> int:
    cmd = [
        _python(),
        str(REPO_ROOT / "scripts" / "scale_ingest_capacity.py"),
        "--apply",
        "--pool-env",
        str(pool_env),
        "--scale-env",
        str(scale_env),
    ]
    if semantic_requested is not None:
        cmd += ["--semantic-requested", semantic_requested]
    if chunk_bench is not None and chunk_bench.is_file():
        cmd += ["--chunk-bench", str(chunk_bench)]
    if embed_bench is not None and embed_bench.is_file():
        cmd += ["--embed-bench", str(embed_bench)]
    return _run(cmd, check=False).returncode


def run_chunk_bench(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[bench] chunk stage (offline CPU)", flush=True)
    return _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "bench_ingest_capacity.py"),
            "--mode",
            "chunk",
            "--semantic",
            "--chunk-concurrency",
            *[str(value) for value in BENCH_CHUNK_CONCURRENCY],
            "--documents",
            "8",
            "--output",
            str(out_dir / "chunk.json"),
        ],
        check=False,
    ).returncode


def run_embed_bench(out_dir: Path, *, embed_urls: str) -> int:
    if not embed_urls.strip():
        print("warning: skipping embed benchmark (no pool URLs)", file=sys.stderr, flush=True)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[bench] embed stage (live pool)", flush=True)
    return _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "bench_ingest_capacity.py"),
            "--mode",
            "embed",
            "--embed-urls",
            embed_urls,
            "--embed-concurrency",
            *[str(value) for value in BENCH_EMBED_CONCURRENCY],
            "--batch-size",
            *[str(value) for value in BENCH_BATCH_SIZES],
            "--documents",
            "8",
            "--output",
            str(out_dir / "embed.json"),
        ],
        check=False,
    ).returncode


def run_planner_dry_run(*, scale_env: Path) -> None:
    print("[plan] dry-run rationale", flush=True)
    _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "scale_ingest_capacity.py"),
            "--scale-env",
            str(scale_env),
        ],
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-env", default=DEFAULT_POOL_ENV)
    parser.add_argument("--scale-env", default=DEFAULT_SCALE_ENV)
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="Skip throughput benchmarks (apply planner only).",
    )
    parser.add_argument(
        "--semantic-requested",
        default=None,
        help="Pass through to scale_ingest_capacity.py (--semantic-requested).",
    )
    args = parser.parse_args()

    pool_env = Path(args.pool_env)
    scale_env = Path(args.scale_env)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"/tmp/ingest-scale-{int(time.time())}")

    apply_config_env(config_dir=scale_env.parent, scale_env=scale_env if scale_env.is_file() else None)

    status = 0
    stop_embed_stack(scale_env)
    wait_gpu_clear()

    chunk_bench = out_dir / "chunk.json"
    embed_bench = out_dir / "embed.json"

    if not args.skip_bench:
        if run_chunk_bench(out_dir) != 0:
            print("warning: chunk benchmark failed", file=sys.stderr, flush=True)
            status = 1

    if run_scale_apply(
        pool_env=pool_env,
        scale_env=scale_env,
        semantic_requested=args.semantic_requested,
        chunk_bench=chunk_bench if chunk_bench.is_file() else None,
    ) != 0:
        print("error: initial scale apply failed", file=sys.stderr, flush=True)
        restart_query_embed()
        return 1

    if not args.skip_bench:
        embed_urls = _read_pool_urls(pool_env)
        if run_embed_bench(out_dir, embed_urls=embed_urls) != 0:
            print("warning: embed benchmark failed", file=sys.stderr, flush=True)
            status = 1
        run_planner_dry_run(scale_env=scale_env)

    if run_scale_apply(
        pool_env=pool_env,
        scale_env=scale_env,
        semantic_requested=args.semantic_requested,
        chunk_bench=chunk_bench if chunk_bench.is_file() else None,
        embed_bench=embed_bench if embed_bench.is_file() else None,
    ) != 0:
        print("error: final scale apply failed", file=sys.stderr, flush=True)
        restart_query_embed()
        return 1

    restart_query_embed()
    print(f"[done] reports in {out_dir} (exit {status})", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
