"""Explorer defaults to the official Kiwix mirror; dotsrc stays selectable (#55)."""

from __future__ import annotations

import re
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_admin.catalog.providers import SOURCES
from rag_admin.routes import explorer


@pytest.fixture
def client_and_calls(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_browse(source_id: str, subpath: str = "") -> dict:
        calls.append(source_id)
        return {"items": [], "error": None}

    monkeypatch.setattr(explorer, "browse_source", fake_browse)
    app = FastAPI()
    app.include_router(explorer.router)
    app.state.db = Mock()
    app.state.db.list_subscriptions.return_value = []
    with TestClient(app) as client:
        yield client, calls


def _tab_class(html: str, source_id: str) -> str:
    match = re.search(rf'href="/explorer\?source={source_id}" class="tab ([^"]*)"', html)
    assert match, f"no tab for {source_id}"
    return match.group(1)


def test_explorer_without_source_browses_kiwix(client_and_calls) -> None:
    client, calls = client_and_calls

    response = client.get("/explorer")

    assert response.status_code == 200
    assert calls == ["kiwix"]
    assert "active" in _tab_class(response.text, "kiwix")


def test_api_browse_without_source_uses_kiwix(client_and_calls) -> None:
    client, calls = client_and_calls

    response = client.get("/api/catalog/browse")

    assert response.json()["source"] == "kiwix"
    assert calls == ["kiwix"]


def test_dotsrc_stays_enabled_and_selectable(client_and_calls) -> None:
    client, calls = client_and_calls
    assert SOURCES["dotsrc"].enabled

    default_page = client.get("/explorer").text
    dotsrc_page = client.get("/explorer", params={"source": "dotsrc"}).text

    assert "disabled" not in _tab_class(default_page, "dotsrc")
    assert "active" in _tab_class(dotsrc_page, "dotsrc")
    assert calls == ["kiwix", "dotsrc"]
