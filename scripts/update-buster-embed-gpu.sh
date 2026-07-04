#!/usr/bin/env bash
# Apply GPU nomic-embed units on buster (or any /opt/ai host).
# Run on the host: bash scripts/update-buster-embed-gpu.sh
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
CONFIG_DIR="${CONFIG_DIR:-/opt/ai/config}"
BIN_DIR="${BIN_DIR:-/opt/ai/bin}"
USER_NAME="${DEPLOY_USER:-kevyn}"

if [[ ! -d "$REPO/.git" ]]; then
  echo "error: repo not found at $REPO" >&2
  exit 1
fi

cd "$REPO"
echo "[pull] $REPO"
git pull --ff-only origin main

sudo mkdir -p "$CONFIG_DIR" "$BIN_DIR"

if [[ ! -f "$CONFIG_DIR/nomic-embed.env" ]]; then
  echo "[config] installing $CONFIG_DIR/nomic-embed.env"
  sudo cp nomic-embed.env.example "$CONFIG_DIR/nomic-embed.env"
else
  echo "[config] keeping existing $CONFIG_DIR/nomic-embed.env"
  if grep -q 'models/nomic-embed/' "$CONFIG_DIR/nomic-embed.env" \
    && [[ -f /opt/ai/models/embed/nomic-embed-text-v1.5.Q8_0.gguf ]]; then
    echo "[config] fixing NOMIC_EMBED_MODEL path for /opt/ai/models/embed/"
    sudo sed -i 's|models/nomic-embed/|models/embed/|g' "$CONFIG_DIR/nomic-embed.env"
  fi
fi

if [[ ! -f "$CONFIG_DIR/nomic-embed-scale.env" ]]; then
  echo "[config] installing $CONFIG_DIR/nomic-embed-scale.env"
  sudo cp nomic-embed-scale.env.example "$CONFIG_DIR/nomic-embed-scale.env"
else
  echo "[config] keeping existing $CONFIG_DIR/nomic-embed-scale.env"
fi

echo "[sudoers] install nomic pool systemctl wrapper + passwordless sudo"
sudo install -m 0755 scripts/nomic-pool-systemctl.sh "$BIN_DIR/nomic-pool-systemctl"
CP_BIN="$(command -v cp)"
CHMOD_BIN="$(command -v chmod)"
sed \
  -e "s|@DEPLOY_USER@|${USER_NAME}|g" \
  -e "s|@BIN_DIR@|${BIN_DIR}|g" \
  -e "s|@CONFIG_DIR@|${CONFIG_DIR}|g" \
  -e "s|@CP_BIN@|${CP_BIN}|g" \
  -e "s|@CHMOD_BIN@|${CHMOD_BIN}|g" \
  deploy/rag-proxy-nomic-pool.sudoers.in | sudo tee /etc/sudoers.d/rag-proxy-nomic-pool >/dev/null
sudo chmod 440 /etc/sudoers.d/rag-proxy-nomic-pool
sudo visudo -cf /etc/sudoers.d/rag-proxy-nomic-pool

echo "[systemd] installing units"
sudo cp nomic-embed.service nomic-embed@.service nomic-embed-scale.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "[systemd] enable query embed (:8089)"
sudo systemctl enable nomic-embed.service

echo "[systemd] scale GPU pool (restart oneshot so --apply re-runs)"
sudo systemctl enable nomic-embed-scale.service
sudo systemctl restart nomic-embed-scale.service

echo "[systemd] ensure query embed (:8089) after pool scale"
sudo systemctl restart nomic-embed.service

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
  if /opt/ai/venv/bin/python - "$CONFIG_DIR" "$CONFIG_DIR/nomic-embed-pool.env" <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/opt/ai/repo/rag_proxy")))
from ingest.port_avoidance import (
    apply_config_env,
    load_env_file,
    loopback_reserved_ports,
    port_from_url,
    ports_from_embed_urls,
)

config_dir = Path(sys.argv[1])
pool_env = load_env_file(sys.argv[2])
apply_config_env(config_dir=config_dir)
reserved = loopback_reserved_ports()
pool_ports = ports_from_embed_urls(pool_env.get("INGEST_EMBED_URLS", ""))
overlap = sorted(pool_ports & reserved)
if overlap:
    print(f"error: pool env uses reserved loopback port(s): {','.join(str(p) for p in overlap)}", file=sys.stderr)
    sparse = os.getenv("SPARSE_INDEX_URL", "")
    if sparse:
        print(f"  SPARSE_INDEX_URL={sparse}", file=sys.stderr)
    sys.exit(1)
PY
  then
    :
  else
    echo "[error] pool env conflicts with a reserved loopback service port; check journalctl -u nomic-embed-scale" >&2
    exit 1
  fi
fi

echo "[units] running pool:"
systemctl list-units 'nomic-embed@*' --state=running --no-pager --no-legend || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu]"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
fi

echo "[done] GPU embed rollout complete"
