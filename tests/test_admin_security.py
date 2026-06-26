"""Security checks for rag-admin config and ingest path validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from rag_admin.config import (
    DEFAULT_PASSWORD,
    DEFAULT_SESSION_SECRET,
    AdminSettings,
    resolve_ingest_path,
    validate_settings,
)


def _admin_settings(**overrides: object) -> AdminSettings:
    defaults = {
        "host": "127.0.0.1",
        "port": 8087,
        "db_path": "/tmp/admin.sqlite",
        "zim_dir": "/tmp/zim",
        "upload_dir": "/tmp/uploads",
        "embed_url": "http://127.0.0.1:18089",
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_collection": "test",
        "sparse_index_url": "http://127.0.0.1:18096",
        "batch_size": 64,
        "max_articles": 0,
        "embed_max_chars": 2000,
        "sparse_reindex_mode": "idle",
        "stall_seconds": 900,
        "session_secret": "secure-secret",
        "password": "secure-password",
        "rag_proxy_url": "http://127.0.0.1:8081",
    }
    defaults.update(overrides)
    return AdminSettings(**defaults)


def test_insecure_defaults_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_ALLOW_INSECURE_DEFAULTS", raising=False)
    settings = _admin_settings(
        session_secret=DEFAULT_SESSION_SECRET,
        password=DEFAULT_PASSWORD,
    )
    with pytest.raises(RuntimeError, match="refused to start"):
        validate_settings(settings)


def test_default_session_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_ALLOW_INSECURE_DEFAULTS", raising=False)
    settings = _admin_settings(
        session_secret=DEFAULT_SESSION_SECRET,
        password="secure-password",
    )
    with pytest.raises(RuntimeError, match="ADMIN_SESSION_SECRET") as exc_info:
        validate_settings(settings)
    assert "ADMIN_PASSWORD" not in str(exc_info.value)


def test_default_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_ALLOW_INSECURE_DEFAULTS", raising=False)
    settings = _admin_settings(
        session_secret="secure-secret",
        password=DEFAULT_PASSWORD,
    )
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD") as exc_info:
        validate_settings(settings)
    assert "ADMIN_SESSION_SECRET" not in str(exc_info.value)


def test_insecure_defaults_allowed_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_ALLOW_INSECURE_DEFAULTS", "true")
    settings = _admin_settings(
        session_secret=DEFAULT_SESSION_SECRET,
        password=DEFAULT_PASSWORD,
    )
    validate_settings(settings)


def test_secure_settings_pass_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_ALLOW_INSECURE_DEFAULTS", raising=False)
    validate_settings(_admin_settings())


def test_resolve_ingest_path_accepts_zim_file(tmp_path: Path) -> None:
    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    zim_file = zim_dir / "wiki.zim"
    zim_file.write_text("data", encoding="ascii")
    resolved = resolve_ingest_path(
        str(zim_file),
        zim_dir=str(zim_dir),
        upload_dir=str(upload_dir),
    )
    assert resolved == zim_file.resolve()


def test_resolve_ingest_path_accepts_upload_file(tmp_path: Path) -> None:
    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    pdf = upload_dir / "doc.pdf"
    pdf.write_text("pdf", encoding="ascii")
    resolved = resolve_ingest_path(
        str(pdf),
        zim_dir=str(zim_dir),
        upload_dir=str(upload_dir),
    )
    assert resolved == pdf.resolve()


def test_resolve_ingest_path_rejects_outside_roots(tmp_path: Path) -> None:
    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    outside = tmp_path / "outside.zim"
    outside.write_text("x", encoding="ascii")
    with pytest.raises(ValueError, match="must be under"):
        resolve_ingest_path(
            str(outside),
            zim_dir=str(zim_dir),
            upload_dir=str(upload_dir),
        )


def test_resolve_ingest_path_rejects_traversal(tmp_path: Path) -> None:
    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    outside = tmp_path / "evil.zim"
    outside.write_text("x", encoding="ascii")
    traversal = zim_dir / ".." / "evil.zim"
    with pytest.raises(ValueError, match="must be under"):
        resolve_ingest_path(
            str(traversal),
            zim_dir=str(zim_dir),
            upload_dir=str(upload_dir),
        )


def _patch_paths_settings(
    monkeypatch: pytest.MonkeyPatch, zim_dir: Path, upload_dir: Path
) -> None:
    monkeypatch.setattr(
        "rag_admin.paths.settings",
        _admin_settings(zim_dir=str(zim_dir), upload_dir=str(upload_dir)),
    )


def test_validated_ingest_file_path_accepts_zim_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_admin.paths import validated_ingest_file_path

    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    zim_file = zim_dir / "wiki.zim"
    zim_file.write_text("data", encoding="ascii")
    _patch_paths_settings(monkeypatch, zim_dir, upload_dir)

    assert validated_ingest_file_path(str(zim_file)) == str(zim_file.resolve())


def test_validated_ingest_file_path_rejects_outside_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_admin.paths import validated_ingest_file_path

    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    outside = tmp_path / "outside.zim"
    outside.write_text("x", encoding="ascii")
    _patch_paths_settings(monkeypatch, zim_dir, upload_dir)

    with pytest.raises(HTTPException) as exc_info:
        validated_ingest_file_path(str(outside))
    assert exc_info.value.status_code == 400
    assert "must be under" in str(exc_info.value.detail)


def test_validated_ingest_file_path_rejects_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rag_admin.paths import validated_ingest_file_path

    zim_dir = tmp_path / "zim"
    upload_dir = tmp_path / "uploads"
    zim_dir.mkdir()
    upload_dir.mkdir()
    outside = tmp_path / "evil.zim"
    outside.write_text("x", encoding="ascii")
    traversal = zim_dir / ".." / "evil.zim"
    _patch_paths_settings(monkeypatch, zim_dir, upload_dir)

    with pytest.raises(HTTPException) as exc_info:
        validated_ingest_file_path(str(traversal))
    assert exc_info.value.status_code == 400
    assert "must be under" in str(exc_info.value.detail)
