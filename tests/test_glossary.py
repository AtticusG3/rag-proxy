"""The glossary is the shared vocabulary issues, code and docs must agree on."""

from __future__ import annotations

import re
from pathlib import Path

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "GLOSSARY.md"


def defined_terms() -> set[str]:
    entries = re.findall(r"^\*\*(.+?)\*\*:$", GLOSSARY_PATH.read_text(encoding="utf-8"), re.M)
    return set(entries)


def test_glossary_defines_the_shared_project_vocabulary() -> None:
    # Retrieval/injection behaviour and operator docs all refer to these by
    # name; a missing entry means the code, issues and docs are drifting apart.
    expected = {
        "Proxy",
        "Upstream",
        "Fail-open",
        "Inject",
        "Cognitive pipeline",
        "Tier0",
        "Gating",
        "Embed pool",
        "Dual-write",
        "MemGraphRAG layers",
        "Sidecar",
        "Dense retrieval",
        "Sparse retrieval",
        "Hybrid retrieval",
        "Chunk",
        "Ingest",
    }
    assert expected <= defined_terms()


def test_every_glossary_entry_has_a_definition_and_avoid_list() -> None:
    text = GLOSSARY_PATH.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^\*\*)", text, flags=re.M)
    for block in blocks:
        if not block.startswith("**"):
            continue
        head, _, rest = block.partition("\n")
        assert "_Avoid_:" in rest, f"{head!r} has no _Avoid_ line"
        definition = rest.split("_Avoid_:", 1)[0].strip()
        assert len(definition.split()) > 5, f"{head!r} has no real definition"
