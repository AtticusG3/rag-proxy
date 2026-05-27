# rag_proxy

Transparent RAG middleware in front of [llama-swap](https://github.com/mostlygeek/llama-swap). Clients point at this proxy instead of llama-swap directly; chat requests are augmented with Qdrant context before they reach any model.

## Architecture

```
Client
  -> rag_proxy :8088
        | embed last user message (nomic-embed :8089)
        | search Qdrant :6333
        | inject retrieved chunks as system context
        -> llama-swap :8080 -> llama-server / Ollama models
```

- **Auth**: unchanged. API keys are validated by llama-swap downstream; this proxy forwards headers as-is.
- **Passthrough**: non-chat routes and non-POST traffic are proxied without modification.
- **Failure mode**: embedding or Qdrant errors never break the request; the original body is forwarded.

## Prerequisites

| Service | Role | Default |
|---------|------|---------|
| llama-swap | Model router | `http://127.0.0.1:8080` |
| nomic-embed (llama-server `--embedding`) | Query vectors | `http://127.0.0.1:8089` |
| Qdrant | Vector store | set `QDRANT_URL` |

`config.yaml` in this repo is an example llama-swap config (paths are host-specific). The `nomic-embed` entry is a **proxy** model pointing at the always-on CPU embed server so llama-swap can warm the connection on startup without loading the weights into VRAM.

## Quick start (Linux)

```bash
cd rag_proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — at minimum set QDRANT_URL

python rag_proxy.py
```

Point OpenAI-compatible clients at `http://<host>:8088/v1` (same paths as llama-swap).

## Environment

See [.env.example](.env.example). All variables are optional; defaults are documented in `rag_proxy.py`.

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` | Qdrant HTTP API base (required for RAG) |
| `QDRANT_COLLECTION` | Collection name |
| `TOP_K` | Max chunks to retrieve |
| `SIMILARITY_THRESHOLD` | Minimum Qdrant score (0–1) |
| `EMBED_URL` | nomic-embed server |
| `LLAMA_SWAP_URL` | Upstream llama-swap |

## systemd

Example units (adjust `User`, paths, and model paths):

1. `nomic-embed.service` — always-on CPU embedding server on port 8089
2. `rag-proxy.service` — this proxy on port 8088
3. llama-swap — uses `config.yaml`

```bash
sudo cp nomic-embed.service rag-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nomic-embed rag-proxy
```

## Tests

```bash
pip install pytest
pytest tests/ -q
```

Tests cover message extraction and context injection only (no live Qdrant or embed server).

## Embedding size limits

llama-server defaults to `-ub 512` (physical batch). Inputs over ~512 tokens return:

`input (N tokens) is too large to process. increase the physical batch size`

Fixes (use both):

1. **nomic-embed.service** — set `-b 2048 -ub 2048` to match `-c 2048`, then `sudo systemctl restart nomic-embed`
2. **rag_proxy** — tail-truncates the last user message to `EMBED_MAX_CHARS` (default 2000) before embed; retries with 1200 chars if the server still rejects the payload

Long chat clients often stuff prior turns into the last `user` message; tail truncation keeps the newest text for retrieval.

## RAG behavior

On `POST /v1/chat/completions` or `POST /api/chat`:

1. Last user message text is embedded via `nomic-embed-text-v1.5`.
2. Qdrant vector search returns up to `TOP_K` hits above `SIMILARITY_THRESHOLD`.
3. Chunk text is read from payload fields: `text`, `content`, `chunk`, `document`, `page_content`.
4. Chunks are prepended to an existing system message, or a new system message is inserted.

## License

Private / homelab — add a license if you open-source this repo.
