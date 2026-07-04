#!/usr/bin/env python3
"""Scale the ingest stack (nomic-embed pool + ingest knobs) to fit host resources."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.capacity_planner import (  # noqa: E402
    IngestCapacityPlan,
    plan_ingest_capacity,
    render_capacity_env,
)
from ingest.embed_pool import load_embed_pool_config  # noqa: E402
from rag_proxy.env_parse import parse_bool  # noqa: E402

DEFAULT_POOL_ENV = "/opt/ai/config/nomic-embed-pool.env"
DEFAULT_SCALE_ENV = "/opt/ai/config/nomic-embed-scale.env"
LEGACY_UNIT = "nomic-embed.service"
TEMPLATE_UNIT = "nomic-embed@"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["systemctl", *args], check=check)


def _probe_embed(url: str, *, timeout: float = 5.0) -> bool:
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


def _wait_for_health(urls: list[str], *, timeout_s: float | None = None) -> list[str]:
    if timeout_s is None:
        timeout_s = min(180.0, 20.0 + 3.0 * len(urls))
    deadline = time.time() + timeout_s
    healthy: list[str] = []
    while time.time() < deadline:
        healthy = [url for url in urls if _probe_embed(url)]
        if len(healthy) == len(urls):
            return healthy
        if healthy and time.time() + 5 > deadline:
            # Accept partial pool near deadline rather than failing the whole scale.
            return healthy
        time.sleep(1.0)
    return healthy


def _stop_legacy_pool_instances(config) -> None:
    _systemctl("stop", LEGACY_UNIT, check=False)
    _systemctl("disable", LEGACY_UNIT, check=False)
    for offset in range(config.max_instances):
        port = config.port_base + offset
        unit = f"{TEMPLATE_UNIT}{port}.service"
        _systemctl("stop", unit, check=False)
        _systemctl("disable", unit, check=False)
    # Also stop ports we previously skipped in static deployments.
    for port in range(config.port_base, config.port_base + config.max_instances + 4):
        unit = f"{TEMPLATE_UNIT}{port}.service"
        _systemctl("stop", unit, check=False)


def shrink_plan_to_healthy(
    plan: IngestCapacityPlan,
    healthy_urls: list[str],
) -> IngestCapacityPlan:
    """Rebuild the capacity plan around the subset of healthy embed instances."""
    healthy_ports = tuple(int(url.rsplit(":", 1)[-1]) for url in healthy_urls)
    count = len(healthy_ports)
    embed_pool = replace(
        plan.embed_pool,
        instance_count=count,
        ports=healthy_ports,
        ingest_embed_urls=",".join(healthy_urls),
        ingest_embed_concurrency=count * plan.nomic_pool_parallel,
    )
    rationale = dict(plan.rationale)
    rationale["embed_pool"] = (
        f"{count} healthy instance(s) after scale "
        f"(planned {plan.embed_pool.instance_count})"
    )
    return replace(
        plan,
        embed_pool=embed_pool,
        ingest_embed_concurrency=embed_pool.ingest_embed_concurrency,
        ingest_file_concurrency=max(1, min(plan.ingest_file_concurrency, count)),
        rationale=rationale,
    )


def apply_plan(plan: IngestCapacityPlan, *, pool_env_path: str, wait_health: bool) -> int:
    config = load_embed_pool_config()
    if not plan.embed_pool.use_gpu_pool:
        print("no GPU detected: writing capacity plan only (no systemd changes)")
        Path(pool_env_path).write_text(render_capacity_env(plan), encoding="utf-8")
        return 0

    _stop_legacy_pool_instances(config)

    target_urls = [f"http://127.0.0.1:{port}" for port in plan.embed_pool.ports]
    restart_cmds = " ".join(
        f"systemctl restart {TEMPLATE_UNIT}{port}.service &"
        for port in plan.embed_pool.ports
    )
    for port in plan.embed_pool.ports:
        unit = f"{TEMPLATE_UNIT}{port}.service"
        _systemctl("enable", unit, check=False)
    if restart_cmds:
        _run(["bash", "-lc", f"{restart_cmds} wait"], check=False)

    healthy_urls = _wait_for_health(target_urls) if wait_health else target_urls
    if not healthy_urls:
        print("error: no healthy nomic-embed instances after scale", file=sys.stderr)
        return 1
    if len(healthy_urls) < len(target_urls):
        print(
            f"warning: using {len(healthy_urls)}/{len(target_urls)} healthy embed instances",
            file=sys.stderr,
        )
        plan = shrink_plan_to_healthy(plan, healthy_urls)

    Path(pool_env_path).write_text(render_capacity_env(plan), encoding="utf-8")
    return 0


def build_plan(args: argparse.Namespace) -> IngestCapacityPlan:
    data_paths = tuple(
        path
        for path in (os.getenv("ZIM_DIR", ""), os.getenv("UPLOAD_DIR", ""))
        if path and os.path.isdir(path)
    )
    semantic_requested = (
        parse_bool(os.getenv("INGEST_CHUNK_SEMANTIC"), True)
        if args.semantic_requested is None
        else args.semantic_requested
    )
    return plan_ingest_capacity(
        semantic_requested=semantic_requested,
        data_paths=data_paths,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Start/stop systemd units and write pool env."
    )
    parser.add_argument("--pool-env", default=DEFAULT_POOL_ENV)
    parser.add_argument("--scale-env", default=DEFAULT_SCALE_ENV)
    parser.add_argument("--no-wait-health", action="store_true")
    parser.add_argument(
        "--semantic-requested",
        type=lambda raw: parse_bool(raw, True),
        default=None,
        help="Operator semantic chunking preference (defaults to INGEST_CHUNK_SEMANTIC env).",
    )
    args = parser.parse_args()

    scale_env = Path(args.scale_env)
    if scale_env.is_file():
        for line in scale_env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    plan = build_plan(args)
    pool = plan.embed_pool
    print(
        f"instances={pool.instance_count} "
        f"ports={','.join(str(p) for p in pool.ports)}"
    )
    if pool.gpu_free_mib is not None:
        print(
            f"gpu_free_mib={pool.gpu_free_mib} "
            f"gpu_used_mib={pool.gpu_used_mib} gpu_total_mib={pool.gpu_total_mib}"
        )
    print(
        f"cpu_cores={plan.host.cpu_logical_cores} "
        f"ram_available_mib={plan.host.ram_available_mib}"
    )
    print(f"INGEST_EMBED_CONCURRENCY={plan.ingest_embed_concurrency}")
    print(f"INGEST_FILE_CONCURRENCY={plan.ingest_file_concurrency}")
    print(f"INGEST_BATCH_SIZE={plan.ingest_batch_size}")
    for key, reason in sorted(plan.rationale.items()):
        print(f"rationale {key}: {reason}")

    if not args.apply:
        print(render_capacity_env(plan), end="")
        return 0

    return apply_plan(
        plan,
        pool_env_path=args.pool_env,
        wait_health=not args.no_wait_health,
    )


if __name__ == "__main__":
    raise SystemExit(main())
