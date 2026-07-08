"""On-demand sparse/rerank sidecar systemd lifecycle for rag-admin."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from rag_proxy.env_parse import parse_bool

log = logging.getLogger("ingest.sidecar_lifecycle")

SPARSE_SIDECAR_UNIT = "sparse-sidecar.service"
RERANK_SIDECAR_UNIT = "rerank-sidecar.service"

_MUTATING_SYSTEMCTL = frozenset({"start", "stop", "restart", "enable", "disable"})


def sidecar_on_demand_enabled() -> bool:
    if not parse_bool(os.getenv("SIDECAR_ON_DEMAND"), True):
        return False
    if os.name != "posix":
        return False
    return shutil.which("systemctl") is not None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def sparse_sidecar_unit() -> str:
    return os.getenv("SPARSE_SIDECAR_UNIT", SPARSE_SIDECAR_UNIT).strip()


def rerank_sidecar_unit() -> str:
    return os.getenv("RERANK_SIDECAR_UNIT", RERANK_SIDECAR_UNIT).strip()


def _running_as_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


def _systemctl_argv(*args: str) -> list[str]:
    if not args:
        return ["systemctl"]
    privileged = args[0] in _MUTATING_SYSTEMCTL
    wrapper = Path(os.getenv("SIDECAR_SYSTEMCTL", "/opt/ai/bin/nomic-pool-systemctl"))
    if not _running_as_root() and privileged and wrapper.is_file():
        return ["sudo", "-n", str(wrapper), *args]
    if not _running_as_root() and privileged:
        return ["sudo", "-n", "systemctl", *args]
    return ["systemctl", *args]


def _systemctl(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _systemctl_argv(*args),
        check=check,
        text=True,
        capture_output=True,
    )


def _unit_active(unit: str) -> bool:
    result = _systemctl("is-active", unit, check=False)
    return result.stdout.strip() in ("active", "activating")


def _start_unit(unit: str) -> None:
    _systemctl("enable", unit, check=False)
    _systemctl("start", unit, check=False)


def _stop_unit(unit: str) -> None:
    _systemctl("stop", unit, check=False)
    _systemctl("disable", unit, check=False)


def probe_sidecar_health(url: str, *, timeout: float = 5.0) -> bool:
    if not url.strip():
        return False
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{url.rstrip('/')}/health")
            response.raise_for_status()
            body = response.json()
            return str(body.get("status", "")).lower() == "ok"
    except Exception:
        return False


def wait_for_sidecar_health(
    url: str,
    *,
    timeout_s: float | None = None,
    poll_s: float = 2.0,
) -> bool:
    if not url.strip():
        return False
    if timeout_s is None:
        timeout_s = _env_float("SIDECAR_STARTUP_TIMEOUT_SEC", 600.0)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if probe_sidecar_health(url):
            return True
        time.sleep(poll_s)
    return probe_sidecar_health(url)


def _stop_sidecar_process(pattern: str) -> bool:
    """Stop a sidecar owned by the same user when systemctl is unavailable."""
    result = subprocess.run(["pkill", "-f", pattern], check=False)
    return result.returncode == 0


def stop_sparse_sidecar() -> bool:
    if not sidecar_on_demand_enabled():
        return False
    unit = sparse_sidecar_unit()
    if not _unit_active(unit) and not probe_sidecar_health(
        os.getenv("SPARSE_INDEX_URL", "http://127.0.0.1:18096")
    ):
        return False
    log.info("sidecar on-demand: stopping %s", unit)
    _stop_unit(unit)
    url = os.getenv("SPARSE_INDEX_URL", "http://127.0.0.1:18096")
    if _unit_active(unit) or probe_sidecar_health(url, timeout=1.0):
        log.warning("sidecar on-demand: systemctl stop failed; killing sparse process")
        _stop_sidecar_process("sidecars/sparse/app.py")
    return True


def ensure_sparse_sidecar(sparse_index_url: str, *, wait_health: bool = True) -> bool:
    """Start sparse sidecar unit and optionally wait until /health is ok."""
    if not sparse_index_url.strip():
        return False
    if not sidecar_on_demand_enabled():
        return probe_sidecar_health(sparse_index_url)
    unit = sparse_sidecar_unit()
    if not _unit_active(unit):
        log.info("sidecar on-demand: starting %s", unit)
        _start_unit(unit)
    if not wait_health:
        return True
    ready = wait_for_sidecar_health(sparse_index_url)
    if not ready:
        log.warning("sparse sidecar not healthy after startup wait url=%s", sparse_index_url)
    return ready


def stop_rerank_sidecar() -> bool:
    if not sidecar_on_demand_enabled():
        return False
    if parse_bool(os.getenv("SIDECAR_RERANK_DURING_INGEST"), True):
        return False
    unit = rerank_sidecar_unit()
    if not _unit_active(unit):
        return False
    log.info("sidecar on-demand: stopping %s", unit)
    _stop_unit(unit)
    return True


def ensure_rerank_sidecar(rerank_url: str, *, wait_health: bool = True) -> bool:
    if not rerank_url.strip():
        return False
    if not sidecar_on_demand_enabled():
        return probe_sidecar_health(rerank_url)
    unit = rerank_sidecar_unit()
    if not _unit_active(unit):
        log.info("sidecar on-demand: starting %s", unit)
        _start_unit(unit)
    if not wait_health:
        return True
    timeout = _env_float("SIDECAR_RERANK_STARTUP_TIMEOUT_SEC", 180.0)
    return wait_for_sidecar_health(rerank_url, timeout_s=timeout)
