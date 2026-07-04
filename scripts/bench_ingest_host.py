#!/usr/bin/env python3
"""Helpers for bench_ingest_capacity_host.sh (pause ingest, pool fallback, probes)."""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.capacity_planner import plan_ingest_capacity, render_capacity_env  # noqa: E402
from ingest.port_avoidance import (  # noqa: E402
    apply_config_env,
    embed_pool_stop_ports,
    loopback_reserved_ports,
    merge_config_dir_env,
)
from rag_admin.settings_schema import INGEST_PAUSED_KEY  # noqa: E402

DEFAULT_ADMIN_DB = "/opt/ai/rag/admin.sqlite"
DEFAULT_LLAMA = "/opt/ai/bin/llama-server"
DEFAULT_MODEL = "/opt/ai/models/embed/nomic-embed-text-v1.5.Q8_0.gguf"
ALT_MODEL = "/opt/ai/models/nomic-embed/nomic-embed-text-v1.5.Q8_0.gguf"
PID_FILE_NAME = "nomic-bench-pool.pids"


def _load_bench_env(scale_env: Path) -> None:
    apply_config_env(config_dir=scale_env.parent, scale_env=scale_env)


def _bench_reserved_ports(scale_env: Path) -> frozenset[int]:
    file_env = merge_config_dir_env(scale_env.parent, scale_env)
    return loopback_reserved_ports(file_env)


def _bench_stop_ports(scale_env: Path, *, extra: int = 4) -> tuple[int, ...]:
    file_env = merge_config_dir_env(scale_env.parent, scale_env)
    port_base = int(file_env.get("NOMIC_POOL_PORT_BASE", os.getenv("NOMIC_POOL_PORT_BASE", "18089")))
    max_instances = int(
        file_env.get("NOMIC_POOL_MAX_INSTANCES", os.getenv("NOMIC_POOL_MAX_INSTANCES", "12"))
    )
    reserved = loopback_reserved_ports(file_env)
    return embed_pool_stop_ports(
        port_base,
        max_instances,
        extra=extra,
        reserved=reserved,
    )


def _resolve_model() -> str:
    for candidate in (
        os.getenv("NOMIC_EMBED_MODEL", ""),
        DEFAULT_MODEL,
        ALT_MODEL,
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return DEFAULT_MODEL


def _read_paused(admin_db: Path) -> bool:
    if not admin_db.is_file():
        return False
    with sqlite3.connect(admin_db) as conn:
        row = conn.execute(
            "SELECT value FROM admin_settings WHERE key = ?",
            (INGEST_PAUSED_KEY,),
        ).fetchone()
    return (row[0] if row else "").lower() == "true"


def _set_paused(admin_db: Path, paused: bool) -> None:
    if not admin_db.is_file():
        return
    value = "true" if paused else "false"
    with sqlite3.connect(admin_db) as conn:
        conn.execute(
            """
            INSERT INTO admin_settings (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (INGEST_PAUSED_KEY, value),
        )
        conn.commit()


def cmd_prepare_pause(args: argparse.Namespace) -> int:
    admin_db = Path(args.admin_db)
    was_paused = _read_paused(admin_db)
    if not was_paused:
        _set_paused(admin_db, True)
    print("true" if was_paused else "false")
    return 0


def cmd_restore_pause(args: argparse.Namespace) -> int:
    admin_db = Path(args.admin_db)
    if args.was_paused.lower() == "true":
        _set_paused(admin_db, True)
    else:
        _set_paused(admin_db, False)
    return 0


def cmd_kill_strays(_args: argparse.Namespace) -> int:
    import importlib.util

    script = REPO_ROOT / "scripts" / "scale_ingest_capacity.py"
    spec = importlib.util.spec_from_file_location("scale_ingest_capacity", script)
    if spec is None or spec.loader is None:
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    killed = mod._kill_stray_gpu_embeds(set())
    for pid in killed:
        print(pid)
    return 0


def _probe_embed(url: str, *, timeout: float = 5.0) -> bool:
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{url.rstrip('/')}/v1/embeddings",
                json={"model": "nomic-embed-text-v1.5", "input": ["pool-health"]},
            )
            response.raise_for_status()
            return "embedding" in response.text
    except Exception:
        return False


def cmd_probe_urls(args: argparse.Namespace) -> int:
    healthy: list[str] = []
    for port in args.ports:
        url = f"http://127.0.0.1:{port}"
        if _probe_embed(url):
            healthy.append(url)
            print(url)
    if args.require and len(healthy) < args.require:
        return 1
    return 0


def _llama_cmd(port: int, *, parallel: int) -> list[str]:
    bin_path = os.getenv("LLAMA_SERVER_BIN", DEFAULT_LLAMA)
    return [
        bin_path,
        "-m",
        _resolve_model(),
        "--embedding",
        "--pooling",
        "mean",
        "-ngl",
        os.getenv("NOMIC_GPU_LAYERS", "99"),
        "-c",
        os.getenv("NOMIC_CONTEXT", "8096"),
        "--parallel",
        str(parallel),
        "-b",
        "2048",
        "-ub",
        "2048",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-disable",
    ]


def cmd_start_pool(args: argparse.Namespace) -> int:
    scale_env = Path(args.scale_env)
    pool_env = Path(args.pool_env)
    pid_file = Path(args.pid_file)
    log_dir = Path(args.log_dir)
    _load_bench_env(scale_env)

    plan = plan_ingest_capacity()
    parallel = plan.nomic_pool_parallel
    ports = list(plan.embed_pool.ports)
    if not ports:
        print("error: planner returned no embed ports", file=sys.stderr)
        return 1

    log_dir.mkdir(parents=True, exist_ok=True)
    pids: list[int] = []
    for port in ports:
        url = f"http://127.0.0.1:{port}"
        if _probe_embed(url, timeout=2.0):
            continue
        log_path = log_dir / f"nomic-{port}.log"
        proc = subprocess.Popen(
            _llama_cmd(port, parallel=parallel),
            stdout=log_path.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pids.append(proc.pid)

    pid_file.write_text("\n".join(str(pid) for pid in pids) + ("\n" if pids else ""), encoding="utf-8")

    deadline = time.time() + args.wait_seconds
    healthy: list[str] = []
    while time.time() < deadline:
        healthy = [f"http://127.0.0.1:{port}" for port in ports if _probe_embed(f"http://127.0.0.1:{port}")]
        if len(healthy) >= args.min_healthy:
            break
        time.sleep(2.0)

    if len(healthy) < args.min_healthy:
        print(
            f"error: only {len(healthy)}/{args.min_healthy} healthy embed instances",
            file=sys.stderr,
        )
        return 1

    healthy_ports = tuple(int(url.rsplit(":", 1)[-1]) for url in healthy)
    if len(healthy_ports) < len(ports):

        script = REPO_ROOT / "scripts" / "scale_ingest_capacity.py"
        spec = importlib.util.spec_from_file_location("scale_ingest_capacity", script)
        if spec is None or spec.loader is None:
            return 1
        scale_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scale_mod)
        plan = scale_mod.shrink_plan_to_healthy(plan, healthy)

    pool_env.parent.mkdir(parents=True, exist_ok=True)
    try:
        pool_env.write_text(render_capacity_env(plan), encoding="utf-8")
    except PermissionError:
        import importlib.util

        script = REPO_ROOT / "scripts" / "scale_ingest_capacity.py"
        spec = importlib.util.spec_from_file_location("scale_ingest_capacity", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._write_pool_env(str(pool_env), plan)

    print(plan.embed_pool.ingest_embed_urls)
    return 0


def cmd_list_stop_ports(args: argparse.Namespace) -> int:
    scale_env = Path(args.scale_env)
    _load_bench_env(scale_env)
    reserved = _bench_reserved_ports(scale_env)
    if args.show_reserved:
        print(
            "reserved loopback ports: " + ",".join(str(port) for port in sorted(reserved)),
            file=sys.stderr,
        )
    for port in _bench_stop_ports(scale_env, extra=args.extra):
        print(port)
    return 0


def cmd_stop_pool(args: argparse.Namespace) -> int:
    pid_file = Path(args.pid_file)
    if not pid_file.is_file():
        return 0
    import signal

    for line in pid_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            os.kill(int(line), signal.SIGTERM)
        except OSError:
            pass
    pid_file.unlink(missing_ok=True)
    cmd_kill_strays(args)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-pause", help="Pause ingest in admin DB; print prior paused state")
    p.add_argument("--admin-db", default=DEFAULT_ADMIN_DB)
    p.set_defaults(func=cmd_prepare_pause)

    p = sub.add_parser("restore-pause", help="Restore ingest pause flag from before bench")
    p.add_argument("--admin-db", default=DEFAULT_ADMIN_DB)
    p.add_argument("--was-paused", required=True)
    p.set_defaults(func=cmd_restore_pause)

    p = sub.add_parser("kill-strays", help="SIGTERM stray GPU embed llama-server processes")
    p.set_defaults(func=cmd_kill_strays)

    p = sub.add_parser("probe-urls", help="Print healthy embed URLs for ports")
    p.add_argument("ports", type=int, nargs="+")
    p.add_argument("--require", type=int, default=0)
    p.set_defaults(func=cmd_probe_urls)

    p = sub.add_parser("start-pool", help="Start embed pool via llama-server (systemd fallback)")
    p.add_argument("--scale-env", required=True)
    p.add_argument("--pool-env", required=True)
    p.add_argument("--pid-file", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--min-healthy", type=int, default=1)
    p.add_argument("--wait-seconds", type=float, default=180.0)
    p.set_defaults(func=cmd_start_pool)

    p = sub.add_parser("list-stop-ports", help="Print systemd pool ports to stop/disable")
    p.add_argument("--scale-env", required=True)
    p.add_argument("--extra", type=int, default=4)
    p.add_argument("--show-reserved", action="store_true")
    p.set_defaults(func=cmd_list_stop_ports)

    p = sub.add_parser("stop-pool", help="Stop fallback pool processes")
    p.add_argument("--pid-file", required=True)
    p.set_defaults(func=cmd_stop_pool)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
