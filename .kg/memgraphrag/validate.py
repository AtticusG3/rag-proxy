#!/usr/bin/env python3
"""Validate .kg/memgraphrag nodes and edges."""

from __future__ import annotations

import json
import sys
from pathlib import Path

KG_DIR = Path(__file__).resolve().parent
ALLOWED_EDGE_TYPES = {
    "ALLOWS_CLASS",
    "CONFORMS_TO",
    "CONTAINS",
    "EXTRACTED_IN",
    "HAS_HEAD",
    "HAS_HEAD_CLASS",
    "HAS_TAIL",
    "HAS_TAIL_CLASS",
    "IMPLEMENTS",
    "PRECEDES",
    "USES",
}
REQUIRED_NODE_FIELDS = {"id", "type", "source", "extracted_at"}
REQUIRED_EDGE_FIELDS = {"from", "to", "type", "source", "extracted_at"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    nodes = load_jsonl(KG_DIR / "nodes.jsonl")
    edges = load_jsonl(KG_DIR / "edges.jsonl")
    violations: list[dict] = []
    node_ids: set[str] = set()
    for node in nodes:
        missing = REQUIRED_NODE_FIELDS - node.keys()
        if missing:
            violations.append({"rule": "structural", "entity": node.get("id"), "detail": str(sorted(missing))})
            continue
        if node["id"] in node_ids:
            violations.append({"rule": "unique_node_id", "entity": node["id"], "detail": "duplicate"})
        node_ids.add(node["id"])
    for edge in edges:
        missing = REQUIRED_EDGE_FIELDS - edge.keys()
        if missing:
            violations.append({"rule": "structural", "detail": str(sorted(missing))})
            continue
        if edge["type"] not in ALLOWED_EDGE_TYPES:
            violations.append({"rule": "predicate_in_schema", "entity": edge["type"]})
        if edge["from"] not in node_ids:
            violations.append({"rule": "edge_endpoints_exist", "entity": edge["from"]})
        if edge["to"] not in node_ids:
            violations.append({"rule": "edge_endpoints_exist", "entity": edge["to"]})
    out = KG_DIR / "violations.jsonl"
    out.write_text(
        "\n".join(json.dumps(v, ensure_ascii=True) for v in violations) + ("\n" if violations else ""),
        encoding="utf-8",
    )
    print(f"nodes: {len(nodes)} edges: {len(edges)} violations: {len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
