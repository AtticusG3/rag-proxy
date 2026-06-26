# Competency Questions

Graph: `rag-proxy-architecture`
Scope: Full-stack rag_proxy — architecture, cognitive pipeline, ops/deploy, ingest, and admin UI.

## Must answer (P0)

| ID | Type | Question | Query pattern |
|----|------|----------|---------------|
| CQ-01 | SCQ | What entity types exist in this project graph? | List distinct node types |
| CQ-02 | FCQ | What is the root repository node? | Lookup `repo:rag-proxy` |
| CQ-03 | SCQ | What top-level packages exist? | Nodes where `type=Package` |
| CQ-04 | RCQ | What configuration artifacts exist? | List nodes where `type=Config` |
| CQ-05 | RCQ | What Python packages does the repo declare? | `DEPENDS_ON` from repo (requirements.txt) |
| CQ-09 | RCQ | What is the cognitive pipeline stage order? | Traverse `PRECEDES` from `stage:tier0` |
| CQ-08 | VCQ | Are all edge endpoints valid? | `validate.py` pass |

## Should answer (P1)

| ID | Type | Question | Query pattern |
|----|------|----------|---------------|
| CQ-06 | RCQ | What scripts automate this project? | List nodes where `type=Script` |
| CQ-07 | RCQ | What documentation is indexed? | List nodes where `type=Document` |
| CQ-10 | FCQ | Which env var toggles each optional stage? | `TOGGLED_BY` from stage nodes |
| CQ-11 | RCQ | Which modules implement pipeline stages? | `IMPLEMENTS` edges |
| CQ-12 | RCQ | What runtime service dependencies exist in Docker? | `DEPENDS_ON` between `Service` nodes |
| CQ-13 | SCQ | What feature-flag env vars exist? | `EnvVar` nodes with `ENABLE_*` prefix |
| CQ-14 | RCQ | What rag_admin modules exist? | Modules under `rag_admin/` |
| CQ-15 | RCQ | What ingest worker modules exist? | Modules under `ingest/` |
| CQ-16 | RCQ | Where is the MemGraphRAG ontology subgraph? | `USES_ONTOLOGY` from `stage:memgraphrag` |

## Verification log

| ID | Status | Notes |
|----|--------|-------|
| CQ-01 | pass | See `queries/cq-results.md` |
| CQ-02 | pass | |
| CQ-03 | pass | |
| CQ-04 | pass | |
| CQ-05 | pass | |
| CQ-06 | pass | |
| CQ-07 | pass | |
| CQ-08 | pass | `validate.py` |
| CQ-09 | pass | |
| CQ-10 | pass | |
| CQ-11 | pass | |
| CQ-12 | pass | |
| CQ-13 | pass | |
| CQ-14 | pass | |
| CQ-15 | pass | |
| CQ-16 | pass | `subgraph:memgraphrag-ontology` |
