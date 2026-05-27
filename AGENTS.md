# rag_proxy — Agent Guide

Transparent RAG middleware: embed last user message → Qdrant search → inject context → forward to llama-swap. Everything else passes through unchanged.

## Engineering principles

All work follows `.cursor/rules/engineering-principles.mdc` (Rules 1–8). Summary:

| Rule | Agent behavior |
|------|----------------|
| 1 Think first | State assumptions; ask; push back on over-engineering |
| 2 Simplicity | Smallest change; no speculative abstractions |
| 3 Surgical | Touch only required lines; match `rag_proxy.py` style |
| 4 Goal-driven | Define success criteria; verify (pytest, manual curl) before done |
| 5 Conflicts | Pick one pattern; explain; flag the loser for cleanup |
| 6 Read first | Read helpers, proxy route, tests before editing |
| 7 Tests = intent | Tests must fail when business rules change, not just outputs |
| 8 Conventions | Follow repo patterns; surface harmful conventions, don't fork |

## Repository map

| Path | Purpose |
|------|---------|
| `rag_proxy.py` | App, RAG helpers, proxy route |
| `tests/test_rag_helpers.py` | Offline unit tests (import module directly) |
| `.env.example` | Env var template |
| `rag-proxy.service` | systemd unit example |
| `README.md` | Architecture and ops |

## Skills (project)

Use when the task matches:

| Skill | Use when |
|-------|----------|
| `rag-proxy-change` | Editing RAG logic, paths, injection, env config |
| `rag-proxy-test` | Adding or fixing tests |
| `rag-proxy-debug` | RAG not injecting, embed/Qdrant/upstream issues |
| `rag-proxy-deploy` | systemd, `.env`, homelab deployment |

## Default success criteria

- **Code change**: `pytest tests/ -q` passes; no new network deps in unit tests.
- **Behavior change**: success criteria name the user-visible outcome (e.g. "multimodal user message still embeds text parts only").
- **Deploy change**: `.env.example` and module docstring updated if env vars change.

## Out of scope unless asked

- Refactoring unrelated code, new frameworks, splitting `rag_proxy.py` into packages.
- Committing `.env` or secrets.
- Changing llama-swap or Qdrant server configs outside this repo.
