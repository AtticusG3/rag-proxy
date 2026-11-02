"""Regression tests for settings save errors and persistent side effects."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_admin.db import AdminDatabase
from rag_admin.env_file import read_env_file
from rag_admin.routes.settings import router as settings_router
from rag_admin.settings_store import SettingsStore


def test_settings_save_rejects_newline_injection_before_persisting(
    tmp_path: Path,
) -> None:
    admin_env = tmp_path / "rag-admin.env"
    proxy_env = tmp_path / "rag-proxy.env"
    admin_original = b"INGEST_BATCH_SIZE=64\nEMBED_URL=http://admin-old\n"
    proxy_original = b"EMBED_URL=http://proxy-old\n"
    admin_env.write_bytes(admin_original)
    proxy_env.write_bytes(proxy_original)

    db = AdminDatabase(str(tmp_path / "admin.sqlite"))
    db.set_setting("INGEST_BATCH_SIZE", "96")
    store = SettingsStore(
        db,
        admin_env_path=str(admin_env),
        proxy_env_path=str(proxy_env),
    )
    app = FastAPI()
    app.include_router(settings_router)
    app.state.settings_store = store
    app.state.worker = Mock()

    with TestClient(app) as client:
        response = client.post(
            "/settings/save/ingest",
            data={
                "INGEST_BATCH_SIZE": "",
                "EMBED_URL": "http://127.0.0.1:18089\nRAG_ADMIN_RESTART_CMD=calc.exe",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["flash_level"] == ["error"]
    assert "EMBED_URL must not contain CR, LF, or NUL" in query["flash"][0]
    assert admin_env.read_bytes() == admin_original
    assert proxy_env.read_bytes() == proxy_original
    assert db.get_setting("INGEST_BATCH_SIZE") == "96"

    with TestClient(app) as client:
        retry = client.post(
            "/settings/save/ingest",
            data={"EMBED_URL": "http://127.0.0.1:18090"},
            follow_redirects=False,
        )

    assert retry.status_code == 303
    retry_query = parse_qs(urlsplit(retry.headers["location"]).query)
    assert retry_query["flash_level"] == ["info"]
    assert read_env_file(str(admin_env))["EMBED_URL"] == "http://127.0.0.1:18090"
    assert read_env_file(str(proxy_env))["EMBED_URL"] == "http://127.0.0.1:18090"
