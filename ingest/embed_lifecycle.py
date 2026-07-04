"""On-demand nomic-embed systemd lifecycle: start before embed, stop when idle."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ingest.embed_pool import EmbedPoolConfig, load_embed_pool_config
from ingest.embed_urls import parse_ingest_embed_urls
from ingest.port_avoidance import embed_pool_stop_ports, loopback_reserved_ports
from rag_proxy.env_parse import parse_bool

log = logging.getLogger("ingest.embed_lifecycle")

QUERY_EMBED_UNIT = "nomic-embed.service"
POOL_UNIT_PREFIX = "nomic-embed@"
DEFAULT_QUERY_EMBED_PORT = 8089
DEFAULT_QUERY_EMBED_URL = "http://127.0.0.1:8089"
DEFAULT_ACTIVITY_PATH = "/var/lib/rag_proxy/embed_last_activity"
DEFAULT_POOL_ENV = "/opt/ai/config/nomic-embed-pool.env"
EXTRA_PORT_BUFFER = 4

_MUTATING_SYSTEMCTL = frozenset({"start", "stop", "restart", "enable", "disable"})


def on_demand_enabled() -> bool:
    """True when embed units should be started/stopped on demand (Linux + systemctl)."""
    if not parse_bool(os.getenv("EMBED_ON_DEMAND"), True):
        return False
    if os.name != "posix":
        return False
    return shutil.which("systemctl") is not None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def activity_stamp_path() -> Path:
    return Path(os.getenv("EMBED_ACTIVITY_STAMP_PATH", DEFAULT_ACTIVITY_PATH))


def touch_embed_activity() -> None:
    """Record that an embed HTTP request completed (ingest or proxy)."""
    if not on_demand_enabled():
        return
    path = activity_stamp_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.3f}\n", encoding="utf-8")
    except OSError as exc:
        log.debug("embed activity stamp failed: %s", exc)


def seconds_since_embed_activity() -> float:
    path = activity_stamp_path()
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return max(0.0, time.time() - float(raw))
    except (OSError, ValueError):
        return float("inf")


def embed_port(url: str) -> int | None:
    parsed = urlparse(url.strip())
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


def dedicated_query_embed_port() -> int:
    """Port bound by nomic-embed.service (fixed in unit file, not EMBED_URL)."""
    return _env_int("NOMIC_QUERY_EMBED_PORT", DEFAULT_QUERY_EMBED_PORT)


def query_embed_url() -> str:
    return os.getenv("EMBED_URL", DEFAULT_QUERY_EMBED_URL).rstrip("/")


def uses_dedicated_query_embed_unit() -> bool:
    """True when EMBED_URL targets nomic-embed.service (:8089), not a pool port."""
    port = embed_port(query_embed_url())
    return port == dedicated_query_embed_port()


def unit_for_embed_url(url: str) -> str | None:
    port = embed_port(url)
    if port is None:
        return None
    if port == dedicated_query_embed_port():
        return QUERY_EMBED_UNIT
    return f"{POOL_UNIT_PREFIX}{port}.service"


def units_for_embed_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        unit = unit_for_embed_url(url)
        if unit is None or unit in seen:
            continue
        seen.add(unit)
        ordered.append(unit)
    return ordered


def _running_as_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return False
    return geteuid() == 0


def _systemctl_argv(*args: str) -> list[str]:
    if not args:
        return ["systemctl"]
    privileged = args[0] in _MUTATING_SYSTEMCTL
    wrapper = Path("/opt/ai/bin/nomic-pool-systemctl")
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


def _stop_disable_unit(unit: str) -> None:
    _systemctl("stop", unit, check=False)
    _systemctl("disable", unit, check=False)


def _enable_start_unit(unit: str) -> None:
    _systemctl("enable", unit, check=False)
    _systemctl("start", unit, check=False)


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


def wait_for_embed_health(
    urls: list[str],
    *,
    timeout_s: float | None = None,
) -> list[str]:
    if not urls:
        return []
    if timeout_s is None:
        timeout_s = _env_float("EMBED_STARTUP_TIMEOUT_SEC", 120.0)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        healthy = [url for url in urls if _probe_embed(url)]
        if len(healthy) == len(urls):
            return healthy
        time.sleep(1.0)
    return [url for url in urls if _probe_embed(url)]


def _read_env_urls(pool_env_path: str) -> list[str]:
    path = Path(pool_env_path)
    if not path.is_file():
        return []
    embed_url = query_embed_url()
    ingest_raw = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("INGEST_EMBED_URLS="):
            ingest_raw = line.split("=", 1)[1].strip()
        elif line.startswith("EMBED_URL="):
            embed_url = line.split("=", 1)[1].strip().rstrip("/")
    return parse_ingest_embed_urls(embed_url=embed_url, ingest_embed_urls=ingest_raw or None)


def _discover_pool_ports() -> set[int]:
    result = _systemctl(
        "list-units",
        "--all",
        "--no-legend",
        f"{POOL_UNIT_PREFIX}*",
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


def _pool_stop_ports(config: EmbedPoolConfig) -> set[int]:
    return set(
        embed_pool_stop_ports(
            config.port_base,
            config.max_instances,
            extra=EXTRA_PORT_BUFFER,
            reserved=loopback_reserved_ports(),
        )
    )


def collect_embed_stop_units(*, pool_env_path: str | None = None) -> list[str]:
    """All nomic embed systemd units that should be stopped when idle."""
    pool_env_path = pool_env_path or os.getenv(
        "NOMIC_EMBED_POOL_ENV_FILE", DEFAULT_POOL_ENV
    )
    config = load_embed_pool_config()
    units: set[str] = set()
    if uses_dedicated_query_embed_unit():
        units.add(QUERY_EMBED_UNIT)
    for url in _read_env_urls(pool_env_path):
        unit = unit_for_embed_url(url)
        if unit:
            units.add(unit)
    for port in _discover_pool_ports() | _pool_stop_ports(config):
        units.add(f"{POOL_UNIT_PREFIX}{port}.service")
    return sorted(units)


def ensure_embed_urls(
    urls: list[str],
    *,
    wait_health: bool = True,
) -> list[str]:
    """Enable/start embed units for URLs and optionally wait until healthy."""
    if not on_demand_enabled() or not urls:
        return urls
    normalized = [url.rstrip("/") for url in urls if url.strip()]
    for unit in units_for_embed_urls(normalized):
        log.info("embed on-demand: starting %s", unit)
        _enable_start_unit(unit)
    if not wait_health:
        return normalized
    healthy = wait_for_embed_health(normalized)
    if len(healthy) < len(normalized):
        log.warning(
            "embed on-demand: %s/%s endpoints healthy after startup wait",
            len(healthy),
            len(normalized),
        )
    return healthy or normalized


def stop_idle_embed_units(*, pool_env_path: str | None = None) -> int:
    """Stop and disable all nomic embed units so VRAM is released."""
    if not on_demand_enabled():
        return 0
    stopped = 0
    for unit in collect_embed_stop_units(pool_env_path=pool_env_path):
        result = _systemctl("is-active", unit, check=False)
        if result.stdout.strip() not in ("active", "activating"):
            _systemctl("disable", unit, check=False)
            continue
        log.info("embed idle: stopping %s", unit)
        _stop_disable_unit(unit)
        stopped += 1
    if stopped:
        log.info("embed idle: stopped %s nomic unit(s)", stopped)
    return stopped


def idle_stop_threshold_sec(*, ingest_paused: bool) -> float:
    if ingest_paused:
        return float(_env_int("EMBED_IDLE_PAUSED_SEC", 30))
    return float(_env_int("EMBED_IDLE_STOP_SEC", 180))
