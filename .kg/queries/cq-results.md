# CQ Results

Generated: 2026-06-26T09:38:18Z

| ID | Status | Result |
|----|--------|--------|
| CQ-01 | pass | 11 types: Config, Dependency, Document, EnvVar, Module, Package, PipelineStage, Repository, Route, Script, Service |
| CQ-02 | pass | `repo:rag-proxy` |
| CQ-03 | pass | 4 packages: ingest, rag_admin, rag_proxy, sidecars |
| CQ-04 | pass | 5 config nodes |
| CQ-05 | pass | 4 Python deps (fastapi, httpx, numpy, uvicorn) |
| CQ-06 | pass | 9 scripts |
| CQ-07 | pass | 13 documents |
| CQ-08 | pass | 0 violations (244 nodes, 204 edges) |
| CQ-09 | pass | 12-stage order: tier0 -> ... -> context |
| CQ-10 | pass | 9 stage-to-env mappings |
| CQ-11 | pass | 12 IMPLEMENTS edges |
| CQ-12 | pass | Docker service DEPENDS_ON chain extracted |
| CQ-13 | pass | 19 ENABLE_* feature flags |
| CQ-14 | pass | 22 rag_admin modules |
| CQ-15 | pass | 11 ingest modules |

Refresh: `.venv/Scripts/python.exe .kg/extract.py`
