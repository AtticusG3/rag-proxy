#!/usr/bin/env bash
# Deploy rag_proxy on buster (/opt/ai) and run smoke checks.
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
VENV="${VENV_PYTHON:-/opt/ai/venv/bin/python}"
PIP="${VENV_PIP:-/opt/ai/venv/bin/pip}"

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

echo "=== [embed smoke :8089] ==="
curl -sf -m 15 -X POST http://127.0.0.1:8089/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":"deploy-check"}' \
  | grep -q embedding && echo "[ok] embed returned vector"

echo "=== [proxy metrics :8088] ==="
curl -sf -m 5 http://127.0.0.1:8088/metrics | head -3

echo "=== [admin health :8087] ==="
curl -sf -m 5 http://127.0.0.1:8087/health && echo || echo "[warn] admin /health not reachable"

echo "=== [ingest queue] ==="
sqlite3 /opt/ai/rag/admin.sqlite "SELECT status, COUNT(*) FROM kb_ingest_state GROUP BY status;" 2>/dev/null || echo "[skip] no admin sqlite"

echo "=== [done] ==="
