"""MCP sidecar must answer a sessionless 2026-07-28 client.

SDK 2 serves that protocol in-process with no initialize handshake. Personal
notes live in SQLite, so a later call must see the note the earlier call stored
even though nothing is pinned to a protocol session.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.mcpserver")

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_RAG = REPO_ROOT / "sidecars" / "mcp_rag"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MCP_RAG))

_spec = importlib.util.spec_from_file_location("mcp_rag_app", MCP_RAG / "app.py")
assert _spec is not None and _spec.loader is not None
mcp_rag_app = importlib.util.module_from_spec(_spec)
sys.modules["mcp_rag_app"] = mcp_rag_app
_spec.loader.exec_module(mcp_rag_app)


def test_sessionless_client_recalls_note_stored_on_an_earlier_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2026 client must read back the personal note it just stored."""
    from mcp import Client

    monkeypatch.setenv("MCP_PERSONAL_STORE_PATH", str(tmp_path / "personal.sqlite"))

    async def _run() -> None:
        async with Client(mcp_rag_app.mcp) as client:
            assert client.session.protocol_version == "2026-07-28"
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == {
                "search_knowledge_base",
                "knowledge_base_status",
                "memory_store",
                "memory_recall",
                "memory_forget",
            }
            stored = await client.call_tool(
                "memory_store",
                {"key": "pref.editor", "text": "Use dark theme"},
            )
            stored_text = stored.content[0].text
            assert stored_text is not None
            assert "pref.editor" in stored_text
            recalled = await client.call_tool("memory_recall", {"query": "dark theme"})
            recalled_text = recalled.content[0].text
            assert recalled_text is not None
            assert "Use dark theme" in recalled_text

    asyncio.run(_run())
