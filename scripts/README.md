# Scripts

Helper scripts for operators and homelab dev. Run from repo root unless noted.

## Operator scripts

| Script | Purpose |
| --- | --- |
| [`build_memgraphrag_index.py`](build_memgraphrag_index.py) | Offline MemGraphRAG index from Qdrant or text files |
| [`catalog_weekly_update.py`](catalog_weekly_update.py) | Cron: check catalog subscriptions and download updates |
| [`requeue_all_ingest.py`](requeue_all_ingest.py) | Re-chunk and re-embed all tracked ingest files |
| [`scale_nomic_embed_pool.py`](scale_nomic_embed_pool.py) | VRAM-aware nomic-embed pool sizing (`INGEST_EMBED_URLS`) |
| [`export_finetune_dataset.py`](export_finetune_dataset.py) | Export transcript JSONL to fine-tuning message format |
| [`promote_rag_corpus.py`](promote_rag_corpus.py) | Promote RAG improvement JSONL pairs to Qdrant |
| [`update-buster-embed-gpu.sh`](update-buster-embed-gpu.sh) | Deploy GPU nomic-embed units on `/opt/ai` hosts |
| [`run-tests.ps1`](run-tests.ps1) | Offline pytest (Windows; uses `.venv` when present) |

See also: [Ingest and admin](../docs/ingest-and-admin.md), [MemGraphRAG](../docs/memgraphrag.md), [Configuration](../docs/configuration.md).

## Dev / homelab scripts

Not required for a standard deploy.

| Script | Purpose |
| --- | --- |
| [`remote-setup-nomad.sh`](remote-setup-nomad.sh) | Remote host bootstrap (nomad) |
| [`remote-setup-clanker.sh`](remote-setup-clanker.sh) | Remote host bootstrap (clanker) |
| [`restart-dev-nomad.sh`](restart-dev-nomad.sh) | Dev restart helper |
| [`remote-smoke-chat.sh`](remote-smoke-chat.sh) | Remote chat smoke test |
| [`install-dev-logrotate-nomad.sh`](install-dev-logrotate-nomad.sh) | Install dev logrotate config |
| [`dev-log-cap.sh`](dev-log-cap.sh) | Cap dev log file size |
| [`logrotate-rag-proxy-dev.conf`](logrotate-rag-proxy-dev.conf) | Logrotate config for dev proxy logs |
| [`clean-workspace.ps1`](clean-workspace.ps1) | Windows workspace cleanup |
| [`ingest_books.py`](ingest_books.py) | One-off book ingest (hardcoded paths) |
