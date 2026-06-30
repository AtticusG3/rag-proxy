#!/usr/bin/env bash
# Apply GPU nomic-embed units on buster (or any /opt/ai host).
# Run on the host: bash scripts/update-buster-embed-gpu.sh
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/rag_proxy}"
CONFIG_DIR="${CONFIG_DIR:-/opt/ai/config}"
USER_NAME="${DEPLOY_USER:-kevyn}"

if [[ ! -d "$REPO/.git" ]]; then
  echo "error: repo not found at $REPO" >&2
  exit 1
fi

cd "$REPO"
echo "[pull] $REPO"
git pull --ff-only origin main

sudo mkdir -p "$CONFIG_DIR"

if [[ ! -f "$CONFIG_DIR/nomic-embed.env" ]]; then
  echo "[config] installing $CONFIG_DIR/nomic-embed.env"
  sudo cp nomic-embed.env.example "$CONFIG_DIR/nomic-embed.env"
else
  echo "[config] keeping existing $CONFIG_DIR/nomic-embed.env"
fi

if [[ ! -f "$CONFIG_DIR/nomic-embed-scale.env" ]]; then
  echo "[config] installing $CONFIG_DIR/nomic-embed-scale.env"
  sudo cp nomic-embed-scale.env.example "$CONFIG_DIR/nomic-embed-scale.env"
else
  echo "[config] keeping existing $CONFIG_DIR/nomic-embed-scale.env"
fi

echo "[systemd] installing units"
sudo cp nomic-embed.service nomic-embed@.service nomic-embed-scale.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "[systemd] enable query embed (:8089)"
sudo systemctl enable nomic-embed.service
sudo systemctl restart nomic-embed.service

echo "[systemd] scale GPU pool and restart instances"
sudo systemctl enable nomic-embed-scale.service
sudo systemctl start nomic-embed-scale.service

if systemctl is-active --quiet rag-admin.service 2>/dev/null; then
  echo "[systemd] restart rag-admin"
  sudo systemctl restart rag-admin.service
fi

echo "[wait] embed health on :8089"
for _ in $(seq 1 30); do
  if curl -sf -m 3 -X POST http://127.0.0.1:8089/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"model":"nomic-embed-text-v1.5","input":"gpu-check"}' \
    | grep -q embedding; then
    echo "[ok] embed :8089"
    break
  fi
  sleep 2
done

if [[ -f "$CONFIG_DIR/nomic-embed-pool.env" ]]; then
  echo "[pool] $(grep -E '^INGEST_EMBED_URLS=|^INGEST_EMBED_CONCURRENCY=' "$CONFIG_DIR/nomic-embed-pool.env" || true)"
fi

echo "[units] running pool:"
systemctl list-units 'nomic-embed@*' --state=running --no-pager --no-legend || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu]"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
fi

echo "[done] GPU embed rollout complete"
