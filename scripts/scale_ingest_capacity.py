#!/usr/bin/env python3
"""Scale the ingest stack (nomic-embed pool + ingest knobs) to fit host resources."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
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
from ingest.embed_pool import EmbedPoolConfig, load_embed_pool_config  # noqa: E402
from rag_proxy.env_parse import parse_bool  # noqa: E402

DEFAULT_POOL_ENV = "/opt/ai/config/nomic-embed-pool.env"
DEFAULT_SCALE_ENV = "/opt/ai/config/nomic-embed-scale.env"
LEGACY_UNIT = "nomic-embed.service"
TEMPLATE_UNIT = "nomic-embed@"
EXTRA_PORT_BUFFER = 4
POOL_SYSTEMCTL_WRAPPER = Path("/opt/ai/bin/nomic-pool-systemctl")
_MUTATING_SYSTEMCTL = frozenset({"start", "stop", "restart", "enable", "disable"})


def _running_as_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _pool_systemctl_wrapper_installed() -> bool:
    return POOL_SYSTEMCTL_WRAPPER.is_file()


def _systemctl_argv(*args: str) -> list[str]:
    if not args:
        return ["systemctl"]
    privileged = args[0] in _MUTATING_SYSTEMCTL
    if _running_as_root() or not privileged:
        return ["systemctl", *args]
    if _pool_systemctl_wrapper_installed():
        return ["sudo", "-n", str(POOL_SYSTEMCTL_WRAPPER), *args]
    return ["sudo", "-n", "systemctl", *args]


def _systemctl_shell() -> str:
    if _running_as_root():
        return "systemctl"
    if _pool_systemctl_wrapper_installed():
        return f"sudo -n {POOL_SYSTEMCTL_WRAPPER}"
    return "sudo -n systemctl"


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(_systemctl_argv(*args), check=check)


def _pool_unit(port: int) -> str:
    return f"{TEMPLATE_UNIT}{port}.service"


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


def _stop_disable_unit(unit: str) -> None:
    _systemctl("stop", unit, check=False)
    _systemctl("disable", unit, check=False)


def _discover_pool_ports() -> set[int]:
    """Return port numbers for all nomic-embed@ units systemd knows about."""
    result = _systemctl(
        "list-units",
        "--all",
        "--no-legend",
        f"{TEMPLATE_UNIT}*",
        check=False,
    )
    ports: set[int] = set()
    for line in result.stdout.splitlines():
        token = line.split()[0] if line.split() else ""
        if "@" not in token:
            continue
        port_raw = token.split("@", 1)[1].removesuffix(".service")
        try:
            ports.add(int(port_raw))
        except ValueError:
            continue
    return ports


def _pool_port_range(config: EmbedPoolConfig) -> range:
    return range(config.port_base, config.port_base + config.max_instances + EXTRA_PORT_BUFFER)


def _retire_pool_units(*, keep_ports: set[int], config: EmbedPoolConfig) -> None:
    """Stop and disable every pool unit outside keep_ports."""
    candidates = _discover_pool_ports() | set(_pool_port_range(config))
    for port in sorted(candidates):
        if port in keep_ports:
            continue
        _stop_disable_unit(_pool_unit(port))


def _prepare_pool_shutdown(config: EmbedPoolConfig) -> None:
    """Stop legacy query embed and all known pool units before a fresh scale."""
    _stop_disable_unit(LEGACY_UNIT)
    _retire_pool_units(keep_ports=set(), config=config)


def _unit_main_pid(unit: str) -> int | None:
    result = _systemctl("show", unit, "-p", "MainPID", "--value", check=False)
    raw = result.stdout.strip()
    if not raw or raw == "0":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def _is_embed_llama(pid: int) -> bool:
    cmd = _process_cmdline(pid)
    return "llama-server" in cmd and "--embedding" in cmd


def _query_gpu_llama_pids() -> set[int]:
    try:
        result = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()

    pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        pid_raw, name = parts
        if "llama-server" not in name:
            continue
        try:
            pids.add(int(pid_raw))
        except ValueError:
            continue
    return pids


def _expected_embed_pids(keep_ports: set[int]) -> set[int]:
    pids: set[int] = set()
    for port in keep_ports:
        pid = _unit_main_pid(_pool_unit(port))
        if pid is not None:
            pids.add(pid)
    legacy_pid = _unit_main_pid(LEGACY_UNIT)
    legacy_active = _systemctl("is-active", LEGACY_UNIT, check=False)
    if legacy_pid is not None and legacy_active.stdout.strip() == "active":
        pids.add(legacy_pid)
    return pids


def _kill_stray_gpu_embeds(keep_ports: set[int]) -> list[int]:
    """SIGTERM embed llama-server processes on GPU not owned by kept units."""
    expected = _expected_embed_pids(keep_ports)
    killed: list[int] = []
    for pid in sorted(_query_gpu_llama_pids()):
        if pid in expected or not _is_embed_llama(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        killed.append(pid)
    if killed:
        print(
            f"reconciled stray gpu embed pids: {','.join(str(pid) for pid in killed)}",
            file=sys.stderr,
        )
    return killed


def _staging_suffix() -> str:
    if hasattr(os, "getuid"):
        return str(os.getuid())
    return "local"


def _write_pool_env(path: str, plan: IngestCapacityPlan) -> None:
    content = render_capacity_env(plan)
    target = Path(path)
    try:
        target.write_text(content, encoding="utf-8")
        return
    except PermissionError:
        pass

    staging = Path(tempfile.gettempdir()) / f"nomic-embed-pool.{_staging_suffix()}.env"
    staging.write_text(content, encoding="utf-8")
    sudo_copy = _run(["sudo", "-n", "cp", str(staging), str(target)], check=False)
    if sudo_copy.returncode == 0:
        _run(["sudo", "-n", "chmod", "644", str(target)], check=False)
        return

    raise PermissionError(
        f"cannot write {target} (direct write and sudo -n cp failed); "
        f"staged plan at {staging}"
    )


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


def _finalize_pool_units(
    *,
    planned_ports: set[int],
    healthy_ports: set[int],
    config: EmbedPoolConfig,
) -> None:
    """Disable crash-looping planned units and any other stray pool units."""
    for port in sorted(planned_ports - healthy_ports):
        _stop_disable_unit(_pool_unit(port))
    _retire_pool_units(keep_ports=healthy_ports, config=config)
    _kill_stray_gpu_embeds(healthy_ports)


def apply_plan(plan: IngestCapacityPlan, *, pool_env_path: str, wait_health: bool) -> int:
    config = load_embed_pool_config()
    if not plan.embed_pool.use_gpu_pool:
        print("no GPU detected: writing capacity plan only (no systemd changes)")
        _write_pool_env(pool_env_path, plan)
        return 0

    planned_ports = set(plan.embed_pool.ports)
    _prepare_pool_shutdown(config)
    _kill_stray_gpu_embeds(set())

    target_urls = [f"http://127.0.0.1:{port}" for port in plan.embed_pool.ports]
    ctl = _systemctl_shell()
    restart_cmds = " ".join(f"{ctl} restart {_pool_unit(port)} &" for port in plan.embed_pool.ports)
    for port in plan.embed_pool.ports:
        _systemctl("enable", _pool_unit(port), check=False)
    if restart_cmds:
        _run(["bash", "-lc", f"{restart_cmds} wait"], check=False)

    healthy_urls = _wait_for_health(target_urls) if wait_health else target_urls
    healthy_ports = {int(url.rsplit(":", 1)[-1]) for url in healthy_urls}
    _finalize_pool_units(
        planned_ports=planned_ports,
        healthy_ports=healthy_ports,
        config=config,
    )

    if not healthy_urls:
        print("error: no healthy nomic-embed instances after scale", file=sys.stderr)
        return 1
    if len(healthy_urls) < len(target_urls):
        print(
            f"warning: using {len(healthy_urls)}/{len(target_urls)} healthy embed instances",
            file=sys.stderr,
        )
        plan = shrink_plan_to_healthy(plan, healthy_urls)

    try:
        _write_pool_env(pool_env_path, plan)
    except PermissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
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
