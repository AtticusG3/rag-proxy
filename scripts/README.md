# Scripts

Helper scripts for operators and homelab dev. Run from repo root unless noted.

## Operator scripts

| Script | Purpose |
| --- | --- |
| [`build_memgraphrag_index.py`](build_memgraphrag_index.py) | Offline MemGraphRAG index from Qdrant or text files |
| [`catalog_weekly_update.py`](catalog_weekly_update.py) | Cron: check catalog subscriptions and download updates |
| [`requeue_all_ingest.py`](requeue_all_ingest.py) | Re-chunk and re-embed all tracked ingest files |
| [`scale_ingest_capacity.py`](scale_ingest_capacity.py) | Multi-resource ingest capacity planner (VRAM pool + CPU/RAM/disk caps) |
| [`scale_nomic_embed_pool.py`](scale_nomic_embed_pool.py) | Legacy wrapper for `scale_ingest_capacity.py` (systemd compat) |
| [`run_ingest_capacity_scale.py`](run_ingest_capacity_scale.py) | Admin background job: bench, apply plan, sync ingest env (Settings button) |
| [`bench_ingest_capacity.py`](bench_ingest_capacity.py) | Benchmark chunk/embed throughput to tune planner coefficients |
| [`bench_ingest_capacity_host.sh`](bench_ingest_capacity_host.sh) | Stop ingest, free GPU, run chunk+embed benches, apply planner, restart admin (`--skip-restart` to leave services down) |
| [`export_finetune_dataset.py`](export_finetune_dataset.py) | Export transcript JSONL to fine-tuning message format |
| [`promote_rag_corpus.py`](promote_rag_corpus.py) | Promote RAG improvement JSONL pairs to Qdrant |
| [`update-buster-embed-gpu.sh`](update-buster-embed-gpu.sh) | Deploy GPU nomic-embed units on `/opt/ai` hosts |
| [`run-tests.ps1`](run-tests.ps1) | Offline pytest (Windows; uses `.venv` when present) |
| [`check_ingest_queue.py`](check_ingest_queue.py) | Print ingest queue status counts from admin SQLite (`kb_ingest_state`) |
| [`clear_qdrant_collection.py`](clear_qdrant_collection.py) | Drop and recreate ingest Qdrant collection (destructive; loads `rag-admin.env`) |

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
| [`diag_buster.py`](diag_buster.py) | One-shot buster health dump (admin SQLite, Qdrant, embed, systemd; `--fix-embed-url`) |
| [`buster-smoke-remote.py`](buster-smoke-remote.py) | Post-deploy smoke checks on buster host (embed, proxy metrics) |
| [`bench_ingest_host.py`](bench_ingest_host.py) | Internal helper for `bench_ingest_capacity_host.sh` (pause ingest, pool fallback, probes) |
