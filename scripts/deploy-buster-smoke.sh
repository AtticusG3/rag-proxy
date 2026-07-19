#!/usr/bin/env bash
# Deploy rag_proxy on buster (/opt/ai) and run smoke checks.
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
VENV="${VENV_PYTHON:-/opt/ai/venv/bin/python}"
PIP="${VENV_PIP:-/opt/ai/venv/bin/pip}"
PROXY_ENV="${PROXY_ENV_FILE:-/opt/ai/config/rag-proxy.env}"
ADMIN_ENV="${RAG_ADMIN_ENV_FILE:-/opt/ai/config/rag-admin.env}"

# Read a KEY=VALUE from a systemd EnvironmentFile without sourcing it.
read_env_value() {
  local file="$1" key="$2" default="$3" val=""
  if [[ -f "$file" ]]; then
    val=$(grep -E "^${key}=" "$file" | tail -1 | sed -E 's/^[^=]+=//; s/^"//; s/"$//; s/[[:space:]]//g' || true)
  fi
  echo "${val:-$default}"
}

# Ports match the host config so smoke checks hit the real listeners.
PROXY_PORT="${PROXY_PORT:-$(read_env_value "$PROXY_ENV" PROXY_PORT 8088)}"
ADMIN_PORT="${ADMIN_PORT:-$(read_env_value "$ADMIN_ENV" ADMIN_PORT 8087)}"
EMBED_PORT="${EMBED_PORT:-8089}"

cd "$REPO"
echo "=== [pull] $REPO ==="
git pull --ff-only origin main

echo "=== [pip] ==="
"$PIP" install -q -r requirements.txt -r requirements-admin.txt -r requirements-dev.txt

echo "=== [pytest] ==="
"$VENV" -m pytest tests/ -q --tb=line 2>&1 | tail -8

restart_unit() {
  local unit="$1"
  if ! systemctl list-unit-files "$unit" >/dev/null 2>&1; then
    echo "[skip] no unit $unit"
    return 0
  fi
  if sudo -n systemctl restart "$unit" 2>/dev/null; then
    echo "[ok] restarted $unit (sudo)"
    return 0
  fi
  if systemctl restart "$unit" 2>/dev/null; then
    echo "[ok] restarted $unit"
    return 0
  fi
  local pid
  pid=$(systemctl show "$unit" -p MainPID --value 2>/dev/null || echo 0)
  if [[ -n "$pid" && "$pid" != "0" ]]; then
    kill "$pid" 2>/dev/null || true
    sleep 3
    systemctl start "$unit" 2>/dev/null || true
    echo "[ok] restarted $unit via kill"
  fi
}

echo "=== [restart services] ==="
restart_unit rag-proxy.service
restart_unit rag-admin.service
sleep 3

echo "=== [service status] ==="
systemctl is-active rag-proxy.service rag-admin.service nomic-embed.service 2>/dev/null || true

echo "=== [embed smoke :${EMBED_PORT}] ==="
curl -sf -m 15 -X POST "http://127.0.0.1:${EMBED_PORT}/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":"deploy-check"}' \
  | grep -q embedding && echo "[ok] embed returned vector"

echo "=== [proxy metrics :${PROXY_PORT}] ==="
curl -sf -m 5 "http://127.0.0.1:${PROXY_PORT}/metrics" | head -3 || echo "[warn] proxy /metrics not reachable on :${PROXY_PORT}"

echo "=== [admin health :${ADMIN_PORT}] ==="
curl -sf -m 5 "http://127.0.0.1:${ADMIN_PORT}/health" && echo || echo "[warn] admin /health not reachable on :${ADMIN_PORT}"

echo "=== [ingest queue] ==="
sqlite3 /opt/ai/rag/admin.sqlite "SELECT status, COUNT(*) FROM kb_ingest_state GROUP BY status;" 2>/dev/null || echo "[skip] no admin sqlite"

echo "=== [done] ==="
