"""Avoid embed pool ports that collide with loopback services on the same host."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

# URL env keys scanned for loopback ports (empty/unset uses homelab default only
# when the default host is loopback).
_URL_ENV_DEFAULTS: dict[str, str] = {
    "EMBED_URL": "http://127.0.0.1:8089",
    "QDRANT_URL": "http://127.0.0.1:6333",
    "RERANKER_URL": "http://127.0.0.1:8095",
    "LLAMA_SWAP_URL": "http://127.0.0.1:8080",
    "RAG_PROXY_URL": "http://127.0.0.1:8081",
    "MEMGRAPH_BUILD_LLM_URL": "http://127.0.0.1:8080/v1",
}

_URL_ENV_KEYS = (
    "EMBED_URL",
    "QDRANT_URL",
    "SPARSE_INDEX_URL",
    "RERANKER_URL",
    "LLAMA_SWAP_URL",
    "RAG_PROXY_URL",
    "MEMGRAPH_BUILD_LLM_URL",
    "MEMGRAPH_BUILD_EMBED_URL",
)

_PORT_ENV_DEFAULTS: dict[str, int] = {
    "PROXY_PORT": 8088,
    "ADMIN_PORT": 8087,
}

_PORT_ENV_KEYS = ("PROXY_PORT", "ADMIN_PORT")

CONFIG_ENV_FILES = (
    "nomic-embed-scale.env",
    "nomic-embed.env",
    "rag-admin.env",
    "rag-proxy.env",
)


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.lower().strip("[]")
    if normalized in LOOPBACK_HOSTS:
        return True
    return normalized.startswith("127.")


def port_from_url(url: str) -> int | None:
    """Return the TCP port when *url* targets loopback; otherwise None."""
    raw = url.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    host = parsed.hostname
    if not is_loopback_host(host):
        return None
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


def ports_from_embed_urls(raw: str) -> set[int]:
    ports: set[int] = set()
    for item in raw.split(","):
        port = port_from_url(item.strip())
        if port is not None:
            ports.add(port)
    return ports


def load_env_file(path: str | Path) -> dict[str, str]:
    env: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.is_file():
        return env
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def merge_config_dir_env(
    config_dir: str | Path,
    *extra_paths: str | Path,
) -> dict[str, str]:
    """Merge homelab config env files (later files override earlier keys)."""
    merged: dict[str, str] = {}
    base = Path(config_dir)
    for name in CONFIG_ENV_FILES:
        merged.update(load_env_file(base / name))
    for path in extra_paths:
        merged.update(load_env_file(path))
    return merged


def apply_config_env(
    *,
    config_dir: str | Path | None = None,
    scale_env: str | Path | None = None,
) -> None:
    """Load homelab config env files into os.environ without overriding existing vars."""
    base = Path(config_dir or os.getenv("CONFIG_DIR", "/opt/ai/config"))
    if scale_env is not None:
        for key, value in load_env_file(scale_env).items():
            os.environ.setdefault(key, value)
    for name in CONFIG_ENV_FILES:
        for key, value in load_env_file(base / name).items():
            os.environ.setdefault(key, value)


def merged_env(*sources: dict[str, str] | None) -> dict[str, str]:
    merged = dict(os.environ)
    for source in sources:
        if source:
            merged.update(source)
    return merged


def loopback_reserved_ports(
    env: dict[str, str] | None = None,
    *,
    include_defaults: bool = True,
) -> frozenset[int]:
    """Ports used by loopback services that embed pool units must not bind."""
    values = merged_env(env)
    ports: set[int] = set()

    for key in _URL_ENV_KEYS:
        url = values.get(key, "").strip()
        if not url and include_defaults and key in _URL_ENV_DEFAULTS:
            url = _URL_ENV_DEFAULTS[key]
        port = port_from_url(url)
        if port is not None:
            ports.add(port)

    # Do not block on INGEST_EMBED_URLS — those are prior pool outputs, not fixed services.

    for key in _PORT_ENV_KEYS:
        raw = values.get(key, "").strip()
        if raw:
            ports.add(int(raw))
        elif include_defaults and key in _PORT_ENV_DEFAULTS:
            ports.add(_PORT_ENV_DEFAULTS[key])

    return frozenset(ports)


def alloc_embed_pool_ports(
    *,
    port_base: int,
    count: int,
    reserved: frozenset[int] | None = None,
    max_scan: int = 128,
) -> tuple[int, ...]:
    """Pick *count* consecutive-available ports at/above *port_base*, skipping *reserved*."""
    if count <= 0:
        return ()
    blocked = reserved or loopback_reserved_ports()
    ports: list[int] = []
    candidate = port_base
    scanned = 0
    while len(ports) < count and scanned < max_scan:
        if candidate not in blocked:
            ports.append(candidate)
        candidate += 1
        scanned += 1
    return tuple(ports)


def embed_pool_stop_ports(
    port_base: int,
    max_instances: int,
    *,
    extra: int = 4,
    reserved: frozenset[int] | None = None,
) -> tuple[int, ...]:
    """Port numbers to stop when retiring pool units (covers skips and stale units)."""
    blocked = reserved or loopback_reserved_ports()
    upper = port_base + max_instances + extra + len(blocked)
    return tuple(range(port_base, upper + 1))


def describe_port_skips(
    *,
    requested_base: int,
    ports: tuple[int, ...],
    reserved: frozenset[int] | None = None,
) -> str | None:
    if not ports:
        return "no embed pool ports allocated"
    blocked = reserved or loopback_reserved_ports()
    skipped = [
        port
        for port in range(requested_base, ports[-1] + 1)
        if port not in ports and port in blocked
    ]
    if ports[0] != requested_base:
        return (
            f"pool starts at {ports[0]} (requested base {requested_base}; "
            f"skipped reserved loopback ports)"
        )
    if skipped:
        return f"skipped reserved loopback port(s): {','.join(str(p) for p in skipped)}"
    return None
