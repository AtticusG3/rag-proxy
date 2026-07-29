"""Version bumps must retire the previous ZIM, not leave both indexed."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_admin.catalog.download_manager import CatalogDownloadManager
from rag_admin.catalog.listing_parser import CatalogItem


class _StreamResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int = 1024) -> list[bytes]:
        return [self._body]

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.body = b"new-zim-bytes"

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stream(self, method: str, url: str) -> _StreamResponse:
        assert method == "GET"
        return _StreamResponse(self.body)


def test_download_version_bump_retires_previous_archive(tmp_path: Path) -> None:
    """Risk: dated auto-update kept both April and July ZIMs INDEXED and injectable.

    After downloading the newer archive, the previous package sibling must be
    fully purged from disk/index so retrieval keeps only the latest stamp.
    """
    zim_dir = tmp_path / "zim"
    zim_dir.mkdir()
    old_path = zim_dir / "devdocs_en_cpp_2026-04.zim"
    old_path.write_bytes(b"old-zim")
    new_path = zim_dir / "devdocs_en_cpp_2026-07.zim"

    db = MagicMock()
    db.ingest = MagicMock()
    db.ingest.list_file_states.return_value = [
        {
            "file_path": str(old_path),
            "file_name": old_path.name,
            "status": "indexed",
        }
    ]
    worker = MagicMock()
    worker.config.qdrant_url = "http://qdrant"
    worker.config.qdrant_collection = "kb"
    # Real purge removes the file; mimic that so the oracle can assert disk state.
    worker.remove_file_from_index.side_effect = lambda path: Path(path).unlink(
        missing_ok=True
    )

    mgr = CatalogDownloadManager(db, str(zim_dir), str(tmp_path / "upload"), worker)
    row = {
        "id": 7,
        "source_id": "kiwix",
        "remote_url": "https://example.test/devdocs_en_cpp_2026-07.zim",
        "local_path": str(old_path),
        "package_key": "devdocs_en_cpp",
        "file_name": old_path.name,
    }

    with (
        patch(
            "rag_admin.catalog.download_manager.fetch_remote_meta",
            return_value={"size_bytes": 12, "modified": "2026-07-01"},
        ),
        patch("rag_admin.catalog.download_manager.httpx.Client", _FakeClient),
    ):
        mgr._download_one(row)

    assert new_path.is_file()
    assert new_path.read_bytes() == b"new-zim-bytes"
    assert not old_path.exists()
    worker.remove_file_from_index.assert_called_with(str(old_path))
    downloaded_calls = [
        c
        for c in db.update_subscription.call_args_list
        if c.args and c.args[0] == 7 and c.kwargs.get("status") == "downloaded"
    ]
    assert downloaded_calls
    assert downloaded_calls[-1].kwargs["local_path"] == str(new_path)
    assert downloaded_calls[-1].kwargs["file_name"] == new_path.name
    worker.enqueue_file.assert_called_once_with(str(new_path))


def test_check_updates_purges_orphan_previous_when_already_latest(
    tmp_path: Path,
) -> None:
    """Heals hosts that already run the latest URL but still have the prior ZIM indexed."""
    zim_dir = tmp_path / "zim"
    zim_dir.mkdir()
    old_path = zim_dir / "devdocs_en_numpy_2026-04.zim"
    new_path = zim_dir / "devdocs_en_numpy_2026-07.zim"
    old_path.write_bytes(b"old")
    new_path.write_bytes(b"new")

    db = MagicMock()
    db.ingest = MagicMock()
    db.ingest.list_file_states.return_value = [
        {"file_path": str(old_path), "file_name": old_path.name},
        {"file_path": str(new_path), "file_name": new_path.name},
    ]
    db.list_auto_update_subscriptions.return_value = [
        {
            "id": 3,
            "source_id": "kiwix",
            "remote_url": f"https://example.test/{new_path.name}",
            "local_path": str(new_path),
            "package_key": "devdocs_en_numpy",
            "catalog_path": "devdocs",
            "remote_modified": "2026-07-01",
            "remote_size": 3,
        }
    ]
    worker = MagicMock()
    worker.remove_file_from_index.side_effect = lambda path: Path(path).unlink(
        missing_ok=True
    )
    mgr = CatalogDownloadManager(db, str(zim_dir), str(tmp_path / "upload"), worker)

    latest = CatalogItem(
        name=new_path.name,
        href=new_path.name,
        url=f"https://example.test/{new_path.name}",
        is_directory=False,
        size_bytes=3,
        modified="2026-07-01",
        subscribable=True,
        package_key="devdocs_en_numpy",
    )

    with (
        patch(
            "rag_admin.catalog.download_manager.browse_source",
            return_value={"items": [latest], "error": None},
        ),
        patch(
            "rag_admin.catalog.download_manager.fetch_remote_meta",
            return_value={"size_bytes": 3, "modified": "2026-07-01"},
        ),
    ):
        queued = mgr.check_updates()

    assert queued == []
    assert new_path.is_file()
    assert not old_path.exists()
    worker.remove_file_from_index.assert_called_with(str(old_path))


def test_maybe_queue_newer_zim_preserves_previous_local_path(tmp_path: Path) -> None:
    """Queuing a newer stamp must not rewrite local_path before download retires it."""
    zim_dir = tmp_path / "zim"
    zim_dir.mkdir()
    old_path = zim_dir / "devdocs_en_css_2026-04.zim"
    db = MagicMock()
    mgr = CatalogDownloadManager(db, str(zim_dir), str(tmp_path / "upload"), None)

    older = CatalogItem(
        name="devdocs_en_css_2026-04.zim",
        href="devdocs_en_css_2026-04.zim",
        url="https://example.test/devdocs_en_css_2026-04.zim",
        is_directory=False,
        size_bytes=1,
        modified="2026-04-01",
        subscribable=True,
        package_key="devdocs_en_css",
    )
    newer = CatalogItem(
        name="devdocs_en_css_2026-07.zim",
        href="devdocs_en_css_2026-07.zim",
        url="https://example.test/devdocs_en_css_2026-07.zim",
        is_directory=False,
        size_bytes=1,
        modified="2026-07-01",
        subscribable=True,
        package_key="devdocs_en_css",
    )
    row = {
        "id": 9,
        "source_id": "kiwix",
        "remote_url": older.url,
        "local_path": str(old_path),
        "package_key": "devdocs_en_css",
        "catalog_path": "devdocs",
    }

    with patch(
        "rag_admin.catalog.download_manager.browse_source",
        return_value={"items": [older, newer], "error": None},
    ):
        assert mgr._maybe_queue_newer_zim(row) is True

    kwargs = db.update_subscription.call_args.kwargs
    assert kwargs["remote_url"] == newer.url
    assert kwargs["file_name"] == newer.name
    assert kwargs["status"] == "update_queued"
    assert "local_path" not in kwargs
