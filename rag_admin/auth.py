"""Session cookie authentication for rag-admin."""

from __future__ import annotations

import hmac
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from rag_admin.config import settings

COOKIE_NAME = "rag_admin_session"
PUBLIC_PATHS = frozenset({"/health", "/login", "/static"})


def _sign(value: str) -> str:
    return hmac.new(
        settings.session_secret.encode(),
        value.encode(),
        "sha256",
    ).hexdigest()


def set_session(response: Response) -> None:
    token = _sign("authenticated")
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,
    )


def clear_session(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME)


def verify_password(password: str) -> bool:
    return hmac.compare_digest(password, settings.password)


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    return hmac.compare_digest(token, _sign("authenticated"))


def require_auth(request: Request) -> None:
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if path.startswith("/static") or path in PUBLIC_PATHS:
            return await call_next(request)
        if path.startswith("/api/") and not is_authenticated(request):
            raise HTTPException(status_code=401, detail="Not authenticated")
        if (
            not path.startswith("/api/")
            and path not in PUBLIC_PATHS
            and not is_authenticated(request)
        ):
            from fastapi.responses import RedirectResponse

            return RedirectResponse(url="/login", status_code=303)
        return await call_next(request)
