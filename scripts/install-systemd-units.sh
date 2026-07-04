#!/usr/bin/env bash
# Install rag-proxy.service and rag-admin.service with paths for this host.
# Run on the Linux host: bash scripts/install-systemd-units.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEPLOY_USER="${DEPLOY_USER:-${SUDO_USER:-$USER}}"
PROXY_ENV="${RAG_PROXY_ENV_FILE:-/opt/ai/config/rag-proxy.env}"
ADMIN_ENV="${RAG_ADMIN_ENV_FILE:-/opt/ai/config/rag-admin.env}"

if [[ -z "${VENV_PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
  elif [[ -x /opt/ai/venv/bin/python ]]; then
    VENV_PYTHON=/opt/ai/venv/bin/python
  else
    echo "error: no python venv found. Set VENV_PYTHON or create $REPO_ROOT/.venv" >&2
    exit 1
  fi
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "error: VENV_PYTHON is not executable: $VENV_PYTHON" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/rag_proxy.py" ]]; then
  echo "error: repo root does not look like rag_proxy: $REPO_ROOT" >&2
  exit 1
fi

echo "[install] repo=$REPO_ROOT"
echo "[install] user=$DEPLOY_USER"
echo "[install] python=$VENV_PYTHON"
echo "[install] proxy env=$PROXY_ENV"
echo "[install] admin env=$ADMIN_ENV"

write_unit() {
  local dest="$1"
  local body="$2"
  if [[ "${INSTALL_DRY_RUN:-}" == "1" ]]; then
    echo "----- $dest -----"
    printf '%s\n' "$body"
    return 0
  fi
  printf '%s\n' "$body" | sudo tee "$dest" >/dev/null
}

PROXY_UNIT="$(cat <<EOF
[Unit]
Description=RAG proxy for llama-swap (Qdrant context injection)
After=network.target
Before=llama-swap.service

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=-${PROXY_ENV}
ExecStart=${VENV_PYTHON} rag_proxy.py
Restart=always
RestartSec=3
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF
)"

ADMIN_UNIT="$(cat <<EOF
[Unit]
Description=RAG admin UI and dense ingest worker
After=network.target

[Service]
Type=simple
User=${DEPLOY_USER}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=-${ADMIN_ENV}
ExecStart=${VENV_PYTHON} -m rag_admin
Restart=always
RestartSec=3
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=multi-user.target
EOF
)"

write_unit /etc/systemd/system/rag-proxy.service "$PROXY_UNIT"
write_unit /etc/systemd/system/rag-admin.service "$ADMIN_UNIT"

if [[ "${INSTALL_DRY_RUN:-}" == "1" ]]; then
  echo "[dry-run] not running daemon-reload"
  exit 0
fi

sudo systemctl daemon-reload
echo "[ok] installed rag-proxy.service and rag-admin.service"
echo "     sudo systemctl enable --now rag-proxy rag-admin"
echo "     sudo systemctl status rag-proxy rag-admin"
