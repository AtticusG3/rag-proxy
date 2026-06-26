#!/usr/bin/env python3
"""Run MemGraphRAG ontology competency-question queries."""

from __future__ import annotations

import json
from pathlib import Path

KG = Path(__file__).resolve().parent.parent
ROOT_ID = "mgr-root:memgraphrag"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def step_order(edges: list[dict]) -> list[str]:
    precedes = {e["from"]: e["to"] for e in edges if e["type"] == "PRECEDES"}
    starts = [
        e["from"]
        for e in edges
        if e["type"] == "PRECEDES" and e["from"].startswith("mgr-step:")
    ]
    starts = [s for s in starts if s not in precedes.values()]
    if not starts:
        return []
    order: list[str] = []
    cur = starts[0]
    while cur:
        order.append(cur.removeprefix("mgr-step:"))
        cur = precedes.get(cur, "")
    return order


def main() -> None:
    nodes, edges = load(KG / "nodes.jsonl"), load(KG / "edges.jsonl")
    node_by_id = {n["id"]: n for n in nodes}
    facts = [n for n in nodes if n["type"] == "MgrFact"]
    results = {
        "CQ-M01": sorted(n["name"] for n in nodes if n["type"] == "MgrLayer"),
        "CQ-M02": sorted(n["name"] for n in nodes if n["type"] == "MgrEntityClass"),
        "CQ-M03": node_by_id.get(ROOT_ID),
        "CQ-M04": "pass via validate.py",
        "CQ-M05": [
            {
                "id": n["id"],
                "pattern": (n.get("head_type"), n.get("relation"), n.get("tail_type")),
                "frequency": n.get("frequency"),
            }
            for n in nodes
            if n["type"] == "MgrOntologyPattern"
        ],
        "CQ-M06": {
            e["to"]: [
                e["from"]
                for x in edges
                if x["type"] == "CONFORMS_TO" and x["to"] == e["to"]
            ]
            for e in edges
            if e["type"] == "CONFORMS_TO"
        },
        "CQ-M07": {
            f["id"]: [
                e["to"]
                for e in edges
                if e["type"] == "EXTRACTED_IN" and e["from"] == f["id"]
            ]
            for f in facts
        },
        "CQ-M08": sorted(
            n["name"] for n in nodes if n["type"] == "MgrComponent" and n.get("role") != "retrieval_step"
        ),
        "CQ-M09": step_order(edges),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
