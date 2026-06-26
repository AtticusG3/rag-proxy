---
name: rag-proxy-deploy
description: >-
  Deploys or configures rag_proxy on a Linux host — venv, .env, systemd
  units, nomic-embed dependency. Use when setting up rag-proxy.service,
  EnvironmentFile, ports 8088/8089, or llama-swap integration.
---

# rag_proxy — Deploy

## Success criteria

```
- [ ] QDRANT_URL set to real host (no CHANGE_ME)
- [ ] nomic-embed reachable at EMBED_URL
- [ ] rag_proxy listens PROXY_HOST:PROXY_PORT (default 8088)
- [ ] Client can hit /v1/chat/completions via proxy to llama-swap
```

## Stack order

1. Qdrant (vector DB) — collection exists with expected payload fields
2. `nomic-embed` on :8089 (CPU embed server)
3. llama-swap on :8080
4. `rag_proxy.py` on :8088

## Linux setup

```bash
cd rag_proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — QDRANT_URL required
python rag_proxy.py
```

## systemd

- Unit: `rag-proxy.service` — adjust `User`, `WorkingDirectory`, `ExecStart` paths.
- `EnvironmentFile=-/path/to/.env` loads config; keep secrets out of git.
- `After=nomic-embed.service`; proxy does not load embed weights itself.

```bash
sudo cp rag-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-proxy
sudo systemctl status rag-proxy
```

## Env sync checklist

When adding/changing variables:

- [ ] `rag_proxy.py` module docstring defaults
- [ ] `.env.example`
- [ ] README environment table (if user-facing)

## Client pointer

OpenAI-compatible base: `http://<host>:8088/v1` — same paths as llama-swap; API keys unchanged (validated downstream).

## Surgical deploy changes (Rule 3)

Only edit service files or docs touched by the deploy task. Do not rewrite `config.yaml` llama-swap examples unless requested.
