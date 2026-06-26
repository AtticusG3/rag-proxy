#!/usr/bin/env python3
"""Rule-based extraction for rag_proxy architecture knowledge graph."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KG = Path(__file__).resolve().parent

STAGE_ORDER = [
    "tier0",
    "intent",
    "gating",
    "routing",
    "rewrite",
    "retrieve",
    "rerank",
    "graph",
    "memgraphrag",
    "tools",
    "memory",
    "context",
]

STAGE_MODULES: dict[str, str] = {
    "tier0": "rag_proxy/stages/tier0_heuristics.py",
    "intent": "rag_proxy/stages/tier1_intent.py",
    "gating": "rag_proxy/stages/tier1_gating.py",
    "routing": "rag_proxy/stages/routing.py",
    "rewrite": "rag_proxy/stages/tier2_rewrite.py",
    "retrieve": "rag_proxy/stages/tier2_retrieval.py",
    "rerank": "rag_proxy/stages/tier2_rerank.py",
    "graph": "rag_proxy/stages/tier3_graph.py",
    "memgraphrag": "rag_proxy/stages/tier3_memgraphrag.py",
    "tools": "rag_proxy/stages/tier3_tools.py",
    "memory": "rag_proxy/stages/tier3_memory.py",
    "context": "rag_proxy/stages/tier2_context.py",
}

STAGE_FLAGS: dict[str, str] = {
    "intent": "ENABLE_INTENT_ROUTER",
    "gating": "ENABLE_RETRIEVAL_GATING",
    "routing": "ENABLE_MODEL_ROUTING",
    "rewrite": "ENABLE_QUERY_REWRITE",
    "rerank": "ENABLE_RERANKER",
    "graph": "ENABLE_GRAPH_LOOKUP",
    "memgraphrag": "ENABLE_MEMGRAPHRAG",
    "tools": "ENABLE_TOOLS",
    "memory": "ENABLE_ROLLING_MEMORY",
}

PACKAGE_ROOTS = {
    "rag_proxy": "package:rag-proxy",
    "rag_admin": "package:rag-admin",
    "ingest": "package:ingest",
    "sidecars": "package:sidecars",
}

ROUTE_RE = re.compile(
    r'@(?:app|router)\.(get|post|put|delete|patch|api_route)\(\s*["\']([^"\']+)'
)
DOCKER_SERVICE_RE = re.compile(r"^\s{2}([a-z][a-z0-9-]+):\s*$")
DEPENDS_ON_RE = re.compile(r"^\s{4}depends_on:\s*$")
SERVICE_KEY_RE = re.compile(r"^\s{6}([a-z][a-z0-9-]+):\s*$")
ENV_VAR_RE = re.compile(r"^([A-Z][A-Z0-9_]+)=")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def module_id(rel_path: str) -> str:
    return f"module:{slug(rel_path.replace('/', '-').replace('.', '-'))}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def discover_python_modules() -> list[str]:
    roots = ["rag_proxy", "rag_admin", "ingest", "sidecars"]
    paths: list[str] = []
    for root in roots:
        base = REPO / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            paths.append(path.relative_to(REPO).as_posix())
    for path in sorted((REPO / "scripts").glob("*.py")):
        paths.append(path.relative_to(REPO).as_posix())
    if (REPO / "rag_proxy.py").exists():
        paths.append("rag_proxy.py")
    return paths


def package_for_path(rel_path: str) -> str | None:
    for prefix, pkg_id in PACKAGE_ROOTS.items():
        if rel_path.startswith(prefix + "/") or rel_path == prefix:
            return pkg_id
    return None


def extract_routes(rel_path: str, text: str) -> list[tuple[str, str, int]]:
    routes: list[tuple[str, str, int]] = []
    for i, line in enumerate(text.splitlines(), 1):
        match = ROUTE_RE.search(line)
        if not match:
            continue
        method, prefix = match.group(1), match.group(2)
        route_key = f"{rel_path}:{prefix}:{method}"
        routes.append((route_key, prefix, i))
    return routes


def extract_docker_services() -> list[str]:
    compose = REPO / "docker-compose.yml"
    if not compose.exists():
        return []
    services: list[str] = []
    in_services = False
    for line in compose.read_text(encoding="utf-8").splitlines():
        if line.strip() == "services:":
            in_services = True
            continue
        if in_services and line and not line.startswith(" "):
            break
        match = DOCKER_SERVICE_RE.match(line)
        if match:
            services.append(match.group(1))
    return services


def extract_service_deps() -> list[tuple[str, str]]:
    compose = REPO / "docker-compose.yml"
    if not compose.exists():
        return []
    lines = compose.read_text(encoding="utf-8").splitlines()
    deps: list[tuple[str, str]] = []
    current: str | None = None
    in_depends = False
    for line in lines:
        svc = DOCKER_SERVICE_RE.match(line)
        if svc:
            current = svc.group(1)
            in_depends = False
            continue
        if current and line.strip() == "depends_on:":
            in_depends = True
            continue
        if in_depends:
            dep = SERVICE_KEY_RE.match(line)
            if dep:
                deps.append((current, dep.group(1)))
                continue
            if line and not line.startswith("      "):
                in_depends = False
    return deps


def extract_env_vars() -> list[str]:
    env_example = REPO / ".env.example"
    if not env_example.exists():
        return []
    names: list[str] = []
    for line in env_example.read_text(encoding="utf-8").splitlines():
        match = ENV_VAR_RE.match(line.strip())
        if match:
            names.append(match.group(1))
    return names


def extract_requirements() -> list[str]:
    req = REPO / "requirements.txt"
    if not req.exists():
        return []
    deps: list[str] = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        if name:
            deps.append(name)
    return deps


def discover_scripts() -> list[str]:
    scripts_dir = REPO / "scripts"
    if not scripts_dir.exists():
        return []
    paths: list[str] = []
    for path in sorted(scripts_dir.iterdir()):
        if path.suffix in {".sh", ".ps1", ".py", ".conf"}:
            paths.append(path.relative_to(REPO).as_posix())
    return paths


def discover_docs() -> list[str]:
    docs: list[str] = []
    readme = REPO / "README.md"
    if readme.exists():
        docs.append("README.md")
    docs_dir = REPO / "docs"
    if docs_dir.exists():
        docs.extend(sorted(p.relative_to(REPO).as_posix() for p in docs_dir.glob("*.md")))
    agents = REPO / "AGENTS.md"
    if agents.exists():
        docs.append("AGENTS.md")
    return docs


def discover_configs() -> list[str]:
    configs = ["docker-compose.yml", "Dockerfile", ".env.example"]
    for path in REPO.glob("sidecars/**/Dockerfile"):
        configs.append(path.relative_to(REPO).as_posix())
    return sorted(set(configs))


def main() -> None:
    extracted_at = now_iso()
    nodes: list[dict] = []
    edges: list[dict] = []

    def add_node(node: dict) -> None:
        node.setdefault("extracted_at", extracted_at)
        nodes.append(node)

    def add_edge(fr: str, to: str, etype: str, source: str, **extra: object) -> None:
        edge = {
            "from": fr,
            "to": to,
            "type": etype,
            "source": source,
            "extracted_at": extracted_at,
            "confidence": 1.0,
        }
        edge.update(extra)
        edges.append(edge)

    add_node(
        {
            "id": "repo:rag-proxy",
            "type": "Repository",
            "name": "rag_proxy",
            "kind": "python-app",
            "source": "repo-root",
            "purpose": "Transparent RAG middleware with optional cognitive pipeline",
        }
    )

    for pkg_name, pkg_id in PACKAGE_ROOTS.items():
        add_node(
            {
                "id": pkg_id,
                "type": "Package",
                "name": pkg_name,
                "path": pkg_name,
                "source": pkg_name,
            }
        )
        add_edge("repo:rag-proxy", pkg_id, "CONTAINS", pkg_name)

    module_paths = discover_python_modules()
    module_ids: dict[str, str] = {}
    for rel in module_paths:
        mid = module_id(rel)
        module_ids[rel] = mid
        name = Path(rel).stem
        add_node({"id": mid, "type": "Module", "name": name, "path": rel, "source": rel})
        pkg = package_for_path(rel)
        if pkg:
            add_edge(pkg, mid, "CONTAINS", rel)
        else:
            add_edge("repo:rag-proxy", mid, "CONTAINS", rel)

        full = REPO / rel
        if full.exists():
            for route_key, prefix, line_no in extract_routes(rel, full.read_text(encoding="utf-8")):
                rid = f"route:{slug(route_key)}"
                add_node(
                    {
                        "id": rid,
                        "type": "Route",
                        "name": prefix,
                        "prefix": prefix,
                        "source": f"{rel}:{line_no}",
                    }
                )
                add_edge(mid, rid, "EXPOSES", f"{rel}:{line_no}")

    for stage in STAGE_ORDER:
        sid = f"stage:{stage}"
        add_node({"id": sid, "type": "PipelineStage", "name": stage, "source": "rag_proxy/pipeline_stages.py"})
        add_edge("repo:rag-proxy", sid, "CONTAINS", "rag_proxy/pipeline_stages.py")
        mod_path = STAGE_MODULES.get(stage)
        if mod_path and mod_path in module_ids:
            add_edge(module_ids[mod_path], sid, "IMPLEMENTS", mod_path)
        flag = STAGE_FLAGS.get(stage)
        if flag:
            eid = f"env:{slug(flag)}"
            add_edge(sid, eid, "TOGGLED_BY", "rag_proxy/pipeline_stages.py")

    add_node(
        {
            "id": "subgraph:memgraphrag-ontology",
            "type": "Subgraph",
            "name": "memgraphrag-ontology",
            "path": ".kg/memgraphrag",
            "source": ".kg/memgraphrag",
        }
    )
    add_edge("stage:memgraphrag", "subgraph:memgraphrag-ontology", "USES_ONTOLOGY", ".kg/memgraphrag")
    add_edge("repo:rag-proxy", "subgraph:memgraphrag-ontology", "CONTAINS", ".kg/memgraphrag")

    for i in range(len(STAGE_ORDER) - 1):
        add_edge(
            f"stage:{STAGE_ORDER[i]}",
            f"stage:{STAGE_ORDER[i + 1]}",
            "PRECEDES",
            "rag_proxy/pipeline_stages.py",
        )

    master = "env:enable-cognitive-pipeline"
    add_edge("repo:rag-proxy", master, "CONFIGURED_BY", ".env.example")

    for name in extract_env_vars():
        add_node(
            {
                "id": f"env:{slug(name)}",
                "type": "EnvVar",
                "name": name,
                "source": ".env.example",
            }
        )

    for svc in extract_docker_services():
        sid = f"service:{slug(svc)}"
        add_node({"id": sid, "type": "Service", "name": svc, "source": "docker-compose.yml"})
        add_edge("config:docker-compose-yml", sid, "DEFINES", f"docker-compose.yml:services.{svc}")

    for fr, to in extract_service_deps():
        add_edge(f"service:{slug(fr)}", f"service:{slug(to)}", "DEPENDS_ON", "docker-compose.yml")

    for rel in discover_configs():
        cid = f"config:{slug(rel.replace('/', '-'))}"
        add_node({"id": cid, "type": "Config", "name": Path(rel).name, "path": rel, "source": rel})
        add_edge("repo:rag-proxy", cid, "HAS_CONFIG", rel)

    for dep in extract_requirements():
        did = f"dependency:{slug(dep)}"
        add_node({"id": did, "type": "Dependency", "name": dep, "source": "requirements.txt"})
        add_edge("repo:rag-proxy", did, "DEPENDS_ON", "requirements.txt")

    for rel in discover_scripts():
        if rel.endswith(".py") and rel in module_ids:
            continue
        sid = f"script:{slug(rel.replace('/', '-').replace('.', '-'))}"
        add_node({"id": sid, "type": "Script", "name": Path(rel).stem, "path": rel, "source": rel})
        add_edge("repo:rag-proxy", sid, "CONTAINS", rel)

    for rel in discover_docs():
        did = f"document:{slug(rel.replace('/', '-').replace('.', '-'))}"
        title = Path(rel).stem.replace("-", " ").replace("_", " ")
        add_node({"id": did, "type": "Document", "name": title, "path": rel, "source": rel})
        add_edge("repo:rag-proxy", did, "CONTAINS", rel)

    # Deduplicate nodes by id (last wins)
    by_id: dict[str, dict] = {}
    for node in nodes:
        by_id[node["id"]] = node
    nodes = list(by_id.values())

    (KG / "nodes.jsonl").write_text(
        "\n".join(json.dumps(n, ensure_ascii=True) for n in sorted(nodes, key=lambda x: x["id"]))
        + "\n",
        encoding="utf-8",
    )
    (KG / "edges.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=True) for e in edges) + "\n",
        encoding="utf-8",
    )
    print(f"extracted nodes={len(nodes)} edges={len(edges)}")


if __name__ == "__main__":
    main()
