#!/usr/bin/env python3
"""Run competency-question queries over .kg JSONL."""

from __future__ import annotations

import json
from pathlib import Path

KG = Path(__file__).resolve().parent.parent
REPO_ID = "repo:rag-proxy"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_order(edges: list[dict]) -> list[str]:
    precedes = {e["from"]: e["to"] for e in edges if e["type"] == "PRECEDES"}
    starts = [e["from"] for e in edges if e["type"] == "PRECEDES" and e["from"].startswith("stage:")]
    starts = [s for s in starts if s not in precedes.values()]
    if not starts:
        return []
    order: list[str] = []
    cur = starts[0]
    while cur:
        order.append(cur.removeprefix("stage:"))
        cur = precedes.get(cur, "")
    return order


def main() -> None:
    nodes, edges = load(KG / "nodes.jsonl"), load(KG / "edges.jsonl")
    node_by_id = {n["id"]: n for n in nodes}
    results = {
        "CQ-01": sorted({n["type"] for n in nodes}),
        "CQ-02": node_by_id.get(REPO_ID),
        "CQ-03": sorted(
            n["name"]
            for n in nodes
            if n["type"] == "Package"
        ),
        "CQ-04": sorted(n["id"] for n in nodes if n["type"] == "Config"),
        "CQ-05": sorted(
            e["to"]
            for e in edges
            if e["from"] == REPO_ID and e["type"] == "DEPENDS_ON"
        ),
        "CQ-06": sorted(n["id"] for n in nodes if n["type"] == "Script"),
        "CQ-07": sorted(n["id"] for n in nodes if n["type"] == "Document"),
        "CQ-08": "pass via validate.py",
        "CQ-09": stage_order(edges),
        "CQ-10": {
            e["from"].removeprefix("stage:"): node_by_id[e["to"]]["name"]
            for e in edges
            if e["type"] == "TOGGLED_BY" and e["to"] in node_by_id
        },
        "CQ-11": sorted(
            e["from"]
            for e in edges
            if e["type"] == "IMPLEMENTS"
        ),
        "CQ-12": sorted(
            f"{e['from']} -> {e['to']}"
            for e in edges
            if e["type"] == "DEPENDS_ON" and e["from"].startswith("service:")
        ),
        "CQ-13": sorted(n["id"] for n in nodes if n["type"] == "EnvVar" and n["name"].startswith("ENABLE_")),
        "CQ-14": sorted(n["id"] for n in nodes if n["type"] == "Module" and n.get("path", "").startswith("rag_admin/")),
        "CQ-15": sorted(n["id"] for n in nodes if n["type"] == "Module" and n.get("path", "").startswith("ingest/")),
        "CQ-16": next(
            (e["to"] for e in edges if e["from"] == "stage:memgraphrag" and e["type"] == "USES_ONTOLOGY"),
            None,
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
