"""Session cookie authentication for rag-admin."""

from __future__ import annotations

import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from rag_admin.config import settings
from rag_admin.db import AdminDatabase

COOKIE_NAME = "rag_admin_session"
PUBLIC_PATHS = frozenset({"/health", "/login", "/static", "/favicon.ico"})


def _sign(payload: str) -> str:
    return hmac.new(
        settings.session_secret.encode(),
        payload.encode(),
        "sha256",
    ).hexdigest()


def _session_expires_at(*, ttl_seconds: int | None = None) -> tuple[int, str]:
    ttl = settings.session_ttl_seconds if ttl_seconds is None else ttl_seconds
    exp = int(time.time()) + ttl
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    return exp, expires_at


def _encode_cookie(session_id: str, exp: int) -> str:
    payload = f"{session_id}.{exp}"
    return f"{payload}.{_sign(payload)}"


def _decode_cookie(token: str) -> tuple[str, int] | None:
    parts = token.rsplit(".", 1)
    if len(parts) != 2:
        return None
    payload, sig = parts
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    session_parts = payload.split(".", 1)
    if len(session_parts) != 2:
        return None
    session_id, exp_str = session_parts
    try:
        exp = int(exp_str)
    except ValueError:
        return None
    if not session_id:
        return None
    return session_id, exp


def _get_db(request: Request) -> AdminDatabase | None:
    return getattr(request.app.state, "db", None)


def set_session(
    response: Response,
    db: AdminDatabase,
    *,
    client_ip: str | None = None,
) -> str:
    session_id = secrets.token_urlsafe(32)
    exp, expires_at = _session_expires_at()
    db.create_admin_session(session_id, expires_at=expires_at, client_ip=client_ip)
    response.set_cookie(
        COOKIE_NAME,
        _encode_cookie(session_id, exp),
        httponly=True,
        samesite="lax",
        max_age=settings.session_ttl_seconds,
    )
    return session_id


def clear_session(
    response: Response,
    request: Request,
    db: AdminDatabase,
) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        decoded = _decode_cookie(token)
        if decoded is not None:
            session_id, _exp = decoded
            db.revoke_admin_session(session_id)
    response.delete_cookie(COOKIE_NAME)


def verify_password(password: str) -> bool:
    return hmac.compare_digest(password, settings.password)


def is_authenticated(request: Request, db: AdminDatabase | None = None) -> bool:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    decoded = _decode_cookie(token)
    if decoded is None:
        return False
    session_id, exp = decoded
    if exp <= int(time.time()):
        return False
    if db is None:
        db = _get_db(request)
    if db is None:
        return False
    row = db.get_admin_session(session_id)
    if row is None or row.get("revoked_at") is not None:
        return False
    return True


def require_auth(request: Request) -> None:
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if path.startswith("/static") or path in PUBLIC_PATHS:
            return await call_next(request)
        db = _get_db(request)
        if path.startswith("/api/") and not is_authenticated(request, db):
            raise HTTPException(status_code=401, detail="Not authenticated")
        if (
            not path.startswith("/api/")
            and path not in PUBLIC_PATHS
            and not is_authenticated(request, db)
        ):
            return RedirectResponse(url="/login", status_code=303)
        return await call_next(request)
