# Admin Codemap

**Last Updated:** 2026-07-27
**Entry Points:** `python -m rag_admin`, `rag_admin/app.py` (`create_app`)

## Architecture

```text
Browser
   |
   v
rag_admin FastAPI (:ADMIN_PORT, default 8087)
   |
   +-- routes/dashboard, explorer, zim, ingest, settings
   +-- settings_store -> RAG_ADMIN_ENV_FILE / RAG_PROXY_ENV_FILE
   +-- IngestWorker (background)
   +-- embed / sidecar lifecycle guards (Linux systemd)
```

## Key Modules

| Module | Purpose |
| --- | --- |
| `rag_admin/app.py` | App factory, lifespan (worker + guards) |
| `rag_admin/auth.py` | Session cookie HMAC + SQLite sessions |
| `rag_admin/config.py` | Admin/ingest env defaults |
| `rag_admin/db.py` | Admin SQLite (queue, sessions, catalog) |
| `rag_admin/settings_schema.py` | Settings field groups / defaults |
| `rag_admin/settings_store.py` | Persist settings to env files + SQLite |
| `rag_admin/settings_ui.py` | Settings form rendering helpers |
| `rag_admin/job_runner.py` | Background jobs (MemGraph build, scale) |
| `rag_admin/routes/dashboard.py` | Dashboard / jobs views |
| `rag_admin/routes/explorer.py` | Content explorer + subscriptions |
| `rag_admin/routes/zim.py` | ZIM / upload pages |
| `rag_admin/routes/ingest.py` | `/api/ingest/*` queue API |
| `rag_admin/routes/settings.py` | Settings HTML + save / status / restart |

## UI templates

`dashboard`, `jobs`, `explorer`, `subscriptions`, `zim`, `upload`, `settings`, `login` under `rag_admin/templates/`.

## Data Flow

1. Operator configures via Settings UI (or env files).
2. Ingest worker consumes queue rows from admin SQLite.
3. Proxy cognitive flags written to `RAG_PROXY_ENV_FILE` require proxy restart.
4. Some ingest keys hot-reload in the worker; capacity scale jobs rewrite pool env.

## Related Areas

- [ingest.md](ingest.md)
- [Ingest and admin](../ingest-and-admin.md)
- [Configuration](../configuration.md) — admin/ingest env tables
