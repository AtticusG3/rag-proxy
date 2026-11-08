"""Small shared helpers for rag-admin routes and templates."""

from __future__ import annotations

import ipaddress
import os
from datetime import datetime
from urllib.parse import urlencode

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from rag_admin.config import resolve_ingest_path, settings

_BASE = os.path.dirname(__file__)


def format_datetime(value: str | None) -> str:
    """Format ISO-8601 timestamps as DD-MMM-YYYY hh:mm:ss."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return dt.strftime("%d-%b-%Y %H:%M:%S")


def format_size(size: int | None) -> str:
    """Human-readable byte size; blank for unknown/missing files."""
    if size is None:
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    if size < 1024**3:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024**3:.2f} GB"


templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))
templates.env.filters["format_dt"] = format_datetime
templates.env.filters["format_size"] = format_size


_trusted_networks_cache: tuple[str, list[ipaddress.IPv4Network | ipaddress.IPv6Network]] | None = None


def trusted_proxy_networks() -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse ADMIN_TRUSTED_PROXIES (comma-separated IPs/CIDRs); skip bad entries.

    Results are cached at module level and only re-parsed when the raw
    environment value changes, so per-request calls stay cheap.
    """
    global _trusted_networks_cache
    raw = os.getenv("ADMIN_TRUSTED_PROXIES", "")
    if _trusted_networks_cache is not None and _trusted_networks_cache[0] == raw:
        return _trusted_networks_cache[1]
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            continue
    _trusted_networks_cache = (raw, networks)
    return networks


def reset_trusted_proxy_networks_cache() -> None:
    """Drop the cached ADMIN_TRUSTED_PROXIES parsing (for tests/reconfiguration)."""
    global _trusted_networks_cache
    _trusted_networks_cache = None


def _peer_is_trusted_proxy(peer: str, networks: list) -> bool:
    if not networks:
        return False
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(peer_ip in network for network in networks)


def _forwarded_ip(raw: str | None) -> str | None:
    """First entry of a forwarded header as a valid IP; None if blank or invalid."""
    if raw is None:
        return None
    value = raw.split(",")[0].strip()
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return None


def client_ip(request: Request) -> str:
    """Client IP for rate limits.

    CF-Connecting-IP and X-Forwarded-For are trusted only when the TCP peer
    is inside ADMIN_TRUSTED_PROXIES (comma-separated IPs/CIDRs, empty by
    default = trust no forwarding headers). Otherwise the peer address is
    used, so spoofed headers cannot bypass the login rate limit.

    A forwarded value is used only if it parses as an IP address; blank or
    invalid values fall through to the next source and finally the peer, so
    the rate-limit key is never empty. The trusted proxy is expected to
    overwrite (not append to) these headers.
    """
    peer = request.client.host if request.client is not None else ""
    if _peer_is_trusted_proxy(peer, trusted_proxy_networks()):
        cf = _forwarded_ip(request.headers.get("cf-connecting-ip"))
        if cf:
            return cf
        forwarded = _forwarded_ip(request.headers.get("x-forwarded-for"))
        if forwarded:
            return forwarded
    if peer:
        return peer
    return "unknown"


def flash_redirect(url: str, message: str, *, level: str = "info") -> RedirectResponse:
    query = urlencode({"flash": message, "flash_level": level})
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}{query}", status_code=303)


def validated_ingest_file_path(file_path: str) -> str:
    """Resolve file_path under zim_dir or upload_dir; HTTP 400 if invalid."""
    try:
        return str(
            resolve_ingest_path(
                file_path,
                zim_dir=settings.zim_dir,
                upload_dir=settings.upload_dir,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
