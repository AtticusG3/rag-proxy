---
name: rag-proxy-deploy
description: >-
  Deploys or configures rag_proxy — venv, .env, systemd rag-proxy.service,
  optional legacy nomic-embed, or Docker split-host (docker/.env). Use when
  setting up ports 8088, llama-swap / embed-pool, or remote QDRANT_URL.
---

# rag_proxy — Deploy

## Success criteria

```
- [ ] QDRANT_URL set to real host (no CHANGE_ME)
- [ ] EMBED_URL reachable (embed-pool / Qwen typical; legacy nomic :8089 optional)
- [ ] rag_proxy listens PROXY_HOST:PROXY_PORT (default 8088)
- [ ] Client can hit /v1/chat/completions via proxy to LLAMA_SWAP_URL
```

## Stack order (bare metal)

1. Qdrant — collection exists with expected payload fields and vector width (2560 for Qwen)
2. Embeddings — llama-swap `embed-pool` on `EMBED_URL` (`EMBED_ON_DEMAND=false`) **or** optional `nomic-embed` on `:8089` (768-d)
3. Upstream chat — `LLAMA_SWAP_URL` (llama-swap `:8080` typical)
4. `rag_proxy.py` on `:8088`

## Docker (split-host)

```bash
cp docker/.env.example docker/.env
# Set QDRANT_URL, LLAMA_SWAP_URL, EMBED_URL to reachable hosts
docker compose up -d --build
```

Default compose profile is **rag-proxy only** — no bundled llama-swap or nomic container. Optional: `--profile qdrant`, `cognitive`, `llama-swap`. See `docker/README.md`.

## Linux setup (venv)

```bash
cd rag_proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — QDRANT_URL required; prefer embed-pool + EMBED_ON_DEMAND=false
python rag_proxy.py
```

## systemd

- Unit: `rag-proxy.service` — adjust `User`, `WorkingDirectory`, `ExecStart` paths.
- `EnvironmentFile=-/path/to/.env` loads config; keep secrets out of git.
- Shipped unit does not hard-require `nomic-embed.service`.

```bash
sudo cp rag-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-proxy
sudo systemctl status rag-proxy
```

## Env sync checklist

When adding/changing variables:

- [ ] `rag_proxy.py` module docstring defaults
- [ ] `.env.example` and `docker/.env.example` when operator-facing
- [ ] README environment table (if user-facing)

## Client pointer

OpenAI-compatible base: `http://<host>:8088/v1` — same paths as upstream; API keys unchanged (validated downstream).

## Surgical deploy changes (Rule 3)

Only edit service files or docs touched by the deploy task. Do not rewrite `config.yaml` llama-swap examples unless requested.
