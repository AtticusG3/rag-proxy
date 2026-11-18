"""Tools stage only reads files the query names."""

import asyncio

from rag_proxy.context import RequestContext
from rag_proxy.pipeline_stages import _tools_should_run
from rag_proxy.stages.tier3_tools import run_tools


def _setup(monkeypatch, tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.setattr("rag_proxy.config.settings.tool_allowed_roots", str(tmp_path))


def test_unrelated_query_adds_no_hits(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    ctx = RequestContext(query_text="how do I make sourdough?")
    asyncio.run(run_tools(ctx))
    assert ctx.hits == []
    assert ctx.errors == []


def test_query_naming_a_file_adds_only_that_file(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    ctx = RequestContext(query_text="What ports does Docker-Compose.yml publish?")
    asyncio.run(run_tools(ctx))
    assert [h.id for h in ctx.hits] == ["tool:file:docker-compose.yml"]


def test_should_run_needs_a_query():
    assert not _tools_should_run(RequestContext(query_text=None))
    assert _tools_should_run(RequestContext(query_text="edit .env.example"))
