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


def trusted_proxy_networks() -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse ADMIN_TRUSTED_PROXIES (comma-separated IPs/CIDRs); skip bad entries."""
    raw = os.getenv("ADMIN_TRUSTED_PROXIES", "")
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            continue
    return networks


def _peer_is_trusted_proxy(peer: str, networks: list) -> bool:
    if not networks:
        return False
    try:
        peer_ip = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(peer_ip in network for network in networks)


def client_ip(request: Request) -> str:
    """Client IP for rate limits.

    CF-Connecting-IP and X-Forwarded-For are trusted only when the TCP peer
    is inside ADMIN_TRUSTED_PROXIES (comma-separated IPs/CIDRs, empty by
    default = trust no forwarding headers). Otherwise the peer address is
    used, so spoofed headers cannot bypass the login rate limit.
    """
    peer = request.client.host if request.client is not None else ""
    if _peer_is_trusted_proxy(peer, trusted_proxy_networks()):
        cf = request.headers.get("cf-connecting-ip")
        if cf:
            return cf.strip()
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
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
