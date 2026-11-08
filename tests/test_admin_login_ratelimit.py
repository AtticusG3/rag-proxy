"""Login per-IP rate limiting tests."""

from __future__ import annotations

import ipaddress
from importlib import reload
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_admin.rate_limit import LoginRateLimiter
from rag_admin import helpers
from rag_admin.helpers import client_ip
from starlette.requests import Request


def test_login_rate_limiter_locks_after_max_failures() -> None:
    limiter = LoginRateLimiter(max_attempts=3, lockout_minutes=15)
    base = 1_700_000_000.0
    for i in range(3):
        assert limiter.is_locked("10.0.0.1", now=base + i) is False
        limiter.record_failure("10.0.0.1", now=base + i)
    assert limiter.is_locked("10.0.0.1", now=base + 3) is True


def test_client_ip_prefers_cf_connecting_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.1")
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


def test_client_ip_uses_forwarded_for_without_cf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.1")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [(b"x-forwarded-for", b"198.51.100.2, 10.0.0.1")],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    assert client_ip(Request(scope)) == "198.51.100.2"


def test_client_ip_uses_forwarded_for_when_peer_inside_trusted_cidr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "10.0.0.0/8, 192.168.1.1")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [(b"x-forwarded-for", b"198.51.100.2, 10.0.0.1")],
        "query_string": b"",
        "client": ("10.20.30.40", 12345),
    }
    assert client_ip(Request(scope)) == "198.51.100.2"


def test_client_ip_spoofed_forwarded_headers_ignored_without_trusted_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ADMIN_TRUSTED_PROXIES", raising=False)
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
    assert client_ip(Request(scope)) == "127.0.0.1"


def test_client_ip_forwarded_headers_ignored_when_peer_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "10.0.0.0/8,not-an-ip,10.0.0/33")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [
            (b"cf-connecting-ip", b"203.0.113.1"),
            (b"x-forwarded-for", b"198.51.100.2"),
        ],
        "query_string": b"",
        "client": ("172.16.0.9", 12345),
    }
    assert client_ip(Request(scope)) == "172.16.0.9"


def test_client_ip_unknown_without_peer_and_trusted_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.1")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [(b"x-forwarded-for", b"198.51.100.2")],
        "query_string": b"",
        "client": None,
    }
    assert client_ip(Request(scope)) == "unknown"


def test_client_ip_blank_forwarded_headers_fall_back_to_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank/whitespace forwarded values must not produce an empty client IP."""
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.1")
    for headers in (
        [(b"cf-connecting-ip", b" ")],
        [(b"cf-connecting-ip", b"")],
        [(b"x-forwarded-for", b" , 198.51.100.2")],
        [(b"cf-connecting-ip", b" "), (b"x-forwarded-for", b"")],
    ):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/login",
            "headers": headers,
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }
        assert client_ip(Request(scope)) == "127.0.0.1"


def test_client_ip_non_ip_forwarded_value_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forwarded value that is not a valid IP is not used as the key."""
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.1")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [
            (b"cf-connecting-ip", b"evil.example.com"),
            (b"x-forwarded-for", b"not-an-ip, 198.51.100.2"),
        ],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    assert client_ip(Request(scope)) == "127.0.0.1"


def test_client_ip_invalid_cf_falls_back_to_forwarded_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.1")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/login",
        "headers": [
            (b"cf-connecting-ip", b"junk"),
            (b"x-forwarded-for", b"198.51.100.2, 10.0.0.1"),
        ],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    assert client_ip(Request(scope)) == "198.51.100.2"


def test_trusted_proxy_networks_are_parsed_once_per_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls with the same env value must not re-parse the value."""
    helpers.reset_trusted_proxy_networks_cache()
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "10.0.0.0/8, not-an-ip ")

    calls: list[str] = []
    real_ip_network = ipaddress.ip_network

    def counting_ip_network(entry, *args, **kwargs):
        calls.append(entry)
        return real_ip_network(entry, *args, **kwargs)

    monkeypatch.setattr(ipaddress, "ip_network", counting_ip_network)
    try:
        first = helpers.trusted_proxy_networks()
        assert first == [ipaddress.ip_network("10.0.0.0/8")]
        parsed = list(calls)
        assert helpers.trusted_proxy_networks() is first
        assert calls == parsed

        monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "172.16.0.0/12")
        second = helpers.trusted_proxy_networks()
        assert calls == parsed + ["172.16.0.0/12"]
        assert second is not first
    finally:
        helpers.reset_trusted_proxy_networks_cache()


def test_reset_trusted_proxy_networks_cache_forces_reparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers.reset_trusted_proxy_networks_cache()
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "10.0.0.0/8")
    assert helpers.trusted_proxy_networks() is helpers.trusted_proxy_networks()

    helpers.reset_trusted_proxy_networks_cache()
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "192.168.0.0/16")
    assert helpers.trusted_proxy_networks() == [ipaddress.ip_network("192.168.0.0/16")]
    helpers.reset_trusted_proxy_networks_cache()


def test_login_rate_limiter_clears_on_success() -> None:
    limiter = LoginRateLimiter(max_attempts=2, lockout_minutes=15)
    base = 1_700_000_000.0
    limiter.record_failure("10.0.0.2", now=base)
    limiter.record_failure("10.0.0.2", now=base + 1)
    assert limiter.is_locked("10.0.0.2", now=base + 2) is True
    limiter.clear("10.0.0.2")
    assert limiter.is_locked("10.0.0.2", now=base + 3) is False


def _admin_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def _reload_admin():
    import rag_admin.app as app_mod
    import rag_admin.auth as auth_mod
    import rag_admin.config as config_mod

    reload(config_mod)
    reload(auth_mod)
    reload(app_mod)
    return app_mod


def test_login_returns_429_after_lockout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _admin_env(monkeypatch, tmp_path)
    monkeypatch.delenv("ADMIN_TRUSTED_PROXIES", raising=False)
    app_mod = _reload_admin()

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


def test_spoofed_forwarded_headers_ignored_without_trusted_proxies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new spoofed header per attempt must not reset the rate limit."""
    _admin_env(monkeypatch, tmp_path)
    monkeypatch.delenv("ADMIN_TRUSTED_PROXIES", raising=False)
    app_mod = _reload_admin()

    with TestClient(app_mod.app) as client:
        for i in range(3):
            resp = client.post(
                "/login",
                data={"password": "wrong"},
                headers={"X-Forwarded-For": f"203.0.113.{i}"},
            )
            assert resp.status_code == 401

        locked = client.post(
            "/login",
            data={"password": "wrong"},
            headers={"CF-Connecting-IP": "198.51.100.99"},
        )
        assert locked.status_code == 429


def test_forwarded_client_ip_used_when_peer_is_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a trusted peer, each forwarded IP gets its own rate-limit key."""
    _admin_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ADMIN_TRUSTED_PROXIES", "127.0.0.0/8")
    app_mod = _reload_admin()

    inner = app_mod.app

    class ProxyPeer:
        def __init__(self, app) -> None:
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                scope = dict(scope)
                scope["client"] = ("127.0.0.1", 54321)
            await self.app(scope, receive, send)

    with TestClient(ProxyPeer(inner)) as client:
        for _ in range(3):
            resp = client.post(
                "/login",
                data={"password": "wrong"},
                headers={"X-Forwarded-For": "203.0.113.0, 10.0.0.1"},
            )
            assert resp.status_code == 401

        # The forwarded IP is now locked out...
        locked = client.post(
            "/login",
            data={"password": "wrong"},
            headers={"X-Forwarded-For": "203.0.113.0, 10.0.0.1"},
        )
        assert locked.status_code == 429

        # ...and CF-Connecting-IP wins over X-Forwarded-For: the locked
        # forwarded IP on the same request does not trigger the lockout.
        other = client.post(
            "/login",
            data={"password": "wrong"},
            headers={
                "CF-Connecting-IP": "198.51.100.7",
                "X-Forwarded-For": "203.0.113.0",
            },
        )
        assert other.status_code == 401


def test_session_client_ip_uses_resolved_peer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful login stores the resolved peer, not a spoofed header."""
    _admin_env(monkeypatch, tmp_path)
    monkeypatch.delenv("ADMIN_TRUSTED_PROXIES", raising=False)
    app_mod = _reload_admin()

    with TestClient(app_mod.app) as client:
        ok = client.post(
            "/login",
            data={"password": "secure-password"},
            headers={"X-Forwarded-For": "203.0.113.77", "CF-Connecting-IP": "198.51.100.88"},
            follow_redirects=False,
        )
        assert ok.status_code == 303

        db = app_mod.app.state.db
        with db._conn() as conn:
            rows = [
                dict(r)
                for r in conn.execute("SELECT * FROM admin_sessions").fetchall()
            ]
        assert len(rows) == 1
        assert rows[0]["client_ip"] == "testclient"


def test_successful_login_clears_failure_counter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _admin_env(monkeypatch, tmp_path)
    monkeypatch.delenv("ADMIN_TRUSTED_PROXIES", raising=False)
    monkeypatch.setenv("ADMIN_LOGIN_MAX_ATTEMPTS", "5")
    app_mod = _reload_admin()

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
