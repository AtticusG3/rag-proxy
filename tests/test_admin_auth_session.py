"""Server-side admin session cookie tests."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import Response
from starlette.requests import Request

from rag_admin.auth import (
    COOKIE_NAME,
    _decode_cookie,
    _encode_cookie,
    _sign,
    clear_session,
    is_authenticated,
    set_session,
)
from rag_admin.config import AdminSettings
from rag_admin.db import AdminDatabase


def _admin_settings(**overrides: object) -> AdminSettings:
    defaults = {
        "host": "127.0.0.1",
        "port": 8087,
        "db_path": "/tmp/admin.sqlite",
        "zim_dir": "/tmp/zim",
        "upload_dir": "/tmp/uploads",
        "embed_url": "http://127.0.0.1:18089",
        "ingest_embed_urls": "",
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_collection": "test",
        "sparse_index_url": "http://127.0.0.1:18096",
        "batch_size": 64,
        "embed_concurrency": 4,
        "max_articles": 0,
        "embed_max_chars": 2000,
        "sparse_reindex_mode": "idle",
        "stall_seconds": 900,
        "session_secret": "test-session-secret",
        "session_ttl_seconds": 3600,
        "login_max_attempts": 5,
        "login_lockout_minutes": 15,
        "password": "secure-password",
        "rag_proxy_url": "http://127.0.0.1:8081",
        "admin_env_path": "/tmp/rag-admin.env",
        "proxy_env_path": "/tmp/rag-proxy.env",
        "repo_root": "/tmp/rag_proxy",
        "job_log_dir": "/tmp/admin_jobs",
        "proxy_restart_cmd": "systemctl restart rag-proxy",
        "admin_restart_cmd": "systemctl restart rag-admin",
        "embed_pool_restart_cmd": "systemctl restart nomic-embed-scale",
        "pool_scale_env_path": "/tmp/nomic-embed-scale.env",
        "pool_env_path": "/tmp/nomic-embed-pool.env",
        "env_example_path": "",
    }
    defaults.update(overrides)
    return AdminSettings(**defaults)


def _request_with_cookie(cookie_value: str | None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie_value is not None:
        headers.append((b"cookie", f"{COOKIE_NAME}={cookie_value}".encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


class _AppState:
    def __init__(self, db: AdminDatabase) -> None:
        self.db = db


class _FakeApp:
    def __init__(self, db: AdminDatabase) -> None:
        self.state = _AppState(db)


def _attach_app(request: Request, db: AdminDatabase) -> None:
    request.scope["app"] = _FakeApp(db)


def test_valid_session_authenticates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    response = Response()
    set_session(response, db, client_ip="10.0.0.1")
    cookie = response.headers.get("set-cookie", "")
    assert COOKIE_NAME in cookie

    token = cookie.split(f"{COOKIE_NAME}=")[1].split(";")[0]
    request = _request_with_cookie(token)
    _attach_app(request, db)
    assert is_authenticated(request, db) is True


def test_expired_cookie_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    session_id = "sess-expired"
    exp = int(time.time()) - 60
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    db.create_admin_session(session_id, expires_at=expires_at)

    token = _encode_cookie(session_id, exp)
    request = _request_with_cookie(token)
    _attach_app(request, db)
    assert is_authenticated(request, db) is False


def test_revoked_session_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    response = Response()
    session_id = set_session(response, db)
    cookie = response.headers.get("set-cookie", "")
    token = cookie.split(f"{COOKIE_NAME}=")[1].split(";")[0]

    db.revoke_admin_session(session_id)
    request = _request_with_cookie(token)
    _attach_app(request, db)
    assert is_authenticated(request, db) is False


def test_logout_revokes_session_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    login_response = Response()
    session_id = set_session(login_response, db)
    cookie = login_response.headers.get("set-cookie", "")
    token = cookie.split(f"{COOKIE_NAME}=")[1].split(";")[0]

    request = _request_with_cookie(token)
    _attach_app(request, db)
    logout_response = Response()
    clear_session(logout_response, request, db)

    row = db.get_admin_session(session_id)
    assert row is not None
    assert row["revoked_at"] is not None
    assert is_authenticated(request, db) is False


def test_legacy_static_cookie_invalidated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-session HMAC('authenticated') cookies must not authenticate."""
    db_path = tmp_path / "admin.sqlite"
    secret = "test-session-secret"
    monkeypatch.setattr(
        "rag_admin.auth.settings",
        _admin_settings(session_secret=secret),
    )
    db = AdminDatabase(str(db_path))

    legacy_token = _sign("authenticated")
    request = _request_with_cookie(legacy_token)
    _attach_app(request, db)
    assert is_authenticated(request, db) is False
    assert _decode_cookie(legacy_token) is None


def test_tampered_signature_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    session_id = "sess-tamper"
    exp = int(time.time()) + 3600
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    db.create_admin_session(session_id, expires_at=expires_at)
    token = _encode_cookie(session_id, exp) + "x"

    request = _request_with_cookie(token)
    _attach_app(request, db)
    assert is_authenticated(request, db) is False


def test_missing_db_row_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    session_id = "orphan-session"
    exp = int(time.time()) + 3600
    token = _encode_cookie(session_id, exp)

    request = _request_with_cookie(token)
    _attach_app(request, db)
    assert is_authenticated(request, db) is False


def test_login_prunes_expired_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    monkeypatch.setattr("rag_admin.auth.settings", _admin_settings())
    db = AdminDatabase(str(db_path))

    past = datetime.fromtimestamp(int(time.time()) - 3600, tz=timezone.utc).isoformat()
    future = datetime.fromtimestamp(int(time.time()) + 3600, tz=timezone.utc).isoformat()
    db.create_admin_session("expired-sess", expires_at=past)
    db.create_admin_session("active-sess", expires_at=future)

    removed = db.prune_expired_admin_sessions()
    assert removed == 1
    assert db.get_admin_session("expired-sess") is None
    assert db.get_admin_session("active-sess") is not None
