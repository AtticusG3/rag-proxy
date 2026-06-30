"""Login per-IP rate limiting tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_admin.rate_limit import LoginRateLimiter
from rag_admin.helpers import client_ip
from starlette.requests import Request


def test_login_rate_limiter_locks_after_max_failures() -> None:
    limiter = LoginRateLimiter(max_attempts=3, lockout_minutes=15)
    base = 1_700_000_000.0
    for i in range(3):
        assert limiter.is_locked("10.0.0.1", now=base + i) is False
        limiter.record_failure("10.0.0.1", now=base + i)
    assert limiter.is_locked("10.0.0.1", now=base + 3) is True


def test_client_ip_prefers_cf_connecting_ip() -> None:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [
            (b"cf-connecting-ip", b"203.0.113.1"),
            (b"x-forwarded-for", b"198.51.100.2, 10.0.0.1"),
        ],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    assert client_ip(Request(scope)) == "203.0.113.1"


def test_client_ip_uses_forwarded_for_without_cf() -> None:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [(b"x-forwarded-for", b"198.51.100.2, 10.0.0.1")],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    assert client_ip(Request(scope)) == "198.51.100.2"


def test_login_rate_limiter_clears_on_success() -> None:
    limiter = LoginRateLimiter(max_attempts=2, lockout_minutes=15)
    base = 1_700_000_000.0
    limiter.record_failure("10.0.0.2", now=base)
    limiter.record_failure("10.0.0.2", now=base + 1)
    assert limiter.is_locked("10.0.0.2", now=base + 2) is True
    limiter.clear("10.0.0.2")
    assert limiter.is_locked("10.0.0.2", now=base + 3) is False


def test_login_returns_429_after_lockout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setenv("ADMIN_ALLOW_INSECURE_DEFAULTS", "true")
    monkeypatch.setenv("ADMIN_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("ADMIN_PASSWORD", "secure-password")
    monkeypatch.setenv("ADMIN_LOGIN_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("ADMIN_LOGIN_LOCKOUT_MINUTES", "15")
    monkeypatch.setenv("ZIM_DIR", str(zim_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("RAG_ADMIN_ENV_FILE", str(tmp_path / "admin.env"))
    monkeypatch.setenv("RAG_PROXY_ENV_FILE", str(tmp_path / "proxy.env"))
    monkeypatch.setenv("RAG_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("RAG_ADMIN_JOB_LOG_DIR", str(tmp_path / "jobs"))

    from importlib import reload

    import rag_admin.auth as auth_mod
    import rag_admin.config as config_mod
    import rag_admin.app as app_mod

    reload(config_mod)
    reload(auth_mod)
    reload(app_mod)

    with TestClient(app_mod.app) as client:
        for _ in range(3):
            resp = client.post(
                "/login",
                data={"password": "wrong"},
                headers={"X-Forwarded-For": "203.0.113.50"},
            )
            assert resp.status_code == 401

        locked = client.post(
            "/login",
            data={"password": "wrong"},
            headers={"X-Forwarded-For": "203.0.113.50"},
        )
        assert locked.status_code == 429

        ok = client.post(
            "/login",
            data={"password": "secure-password"},
            headers={"X-Forwarded-For": "203.0.113.50"},
        )
        assert ok.status_code == 429


def test_successful_login_clears_failure_counter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "admin.sqlite"
    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setenv("ADMIN_ALLOW_INSECURE_DEFAULTS", "true")
    monkeypatch.setenv("ADMIN_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("ADMIN_PASSWORD", "secure-password")
    monkeypatch.setenv("ADMIN_LOGIN_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("ZIM_DIR", str(zim_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("RAG_ADMIN_ENV_FILE", str(tmp_path / "admin.env"))
    monkeypatch.setenv("RAG_PROXY_ENV_FILE", str(tmp_path / "proxy.env"))
    monkeypatch.setenv("RAG_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("RAG_ADMIN_JOB_LOG_DIR", str(tmp_path / "jobs"))

    from importlib import reload

    import rag_admin.auth as auth_mod
    import rag_admin.config as config_mod
    import rag_admin.app as app_mod

    reload(config_mod)
    reload(auth_mod)
    reload(app_mod)

    with TestClient(app_mod.app) as client:
        for _ in range(2):
            resp = client.post(
                "/login",
                data={"password": "wrong"},
                headers={"X-Forwarded-For": "198.51.100.10"},
            )
            assert resp.status_code == 401

        ok = client.post(
            "/login",
            data={"password": "secure-password"},
            headers={"X-Forwarded-For": "198.51.100.10"},
            follow_redirects=False,
        )
        assert ok.status_code == 303

        again = client.post(
            "/login",
            data={"password": "wrong"},
            headers={"X-Forwarded-For": "198.51.100.10"},
        )
        assert again.status_code == 401
