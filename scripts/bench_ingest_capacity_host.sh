#!/usr/bin/env bash
# Stop active ingest, free GPU VRAM, run capacity benchmarks, apply planner, restart admin.
#
# Run on the ingest host (e.g. buster):
#   bash scripts/bench_ingest_capacity_host.sh
#
# Requires sudo for systemctl (install deploy/rag-proxy-nomic-pool.sudoers via
# scripts/update-buster-embed-gpu.sh) or run relevant steps as root.
set -euo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
PYTHON="${PYTHON:-/opt/ai/venv/bin/python}"
CONFIG_DIR="${CONFIG_DIR:-/opt/ai/config}"
SCALE_ENV="${SCALE_ENV:-$CONFIG_DIR/nomic-embed-scale.env}"
POOL_ENV="${POOL_ENV:-$CONFIG_DIR/nomic-embed-pool.env}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/tmp/ingest-bench-$STAMP}"
PORT_BASE="${NOMIC_POOL_PORT_BASE:-18089}"
MAX_PORTS="${NOMIC_POOL_MAX_INSTANCES:-12}"

mkdir -p "$OUT_DIR"

_systemctl() {
  if [[ -x /opt/ai/bin/nomic-pool-systemctl ]]; then
    sudo -n /opt/ai/bin/nomic-pool-systemctl "$@" 2>/dev/null && return 0
  fi
  if sudo -n systemctl "$@" 2>/dev/null; then
    return 0
  fi
  systemctl "$@" 2>/dev/null || sudo systemctl "$@"
}

_stop_embed_units() {
  echo "[gpu] stopping nomic-embed pool and query embed"
  _systemctl stop nomic-embed.service 2>/dev/null || true
  _systemctl disable nomic-embed.service 2>/dev/null || true
  local offset port
  for offset in $(seq 0 $((MAX_PORTS + 4))); do
    port=$((PORT_BASE + offset))
    _systemctl stop "nomic-embed@${port}.service" 2>/dev/null || true
    _systemctl disable "nomic-embed@${port}.service" 2>/dev/null || true
  done
}

_wait_gpu_clear() {
  echo "[gpu] waiting for embed llama-server processes to exit"
  local tries=0
  while (( tries < 45 )); do
    if ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
      | grep -q llama-server; then
      nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
      return 0
    fi
    # SIGTERM stray embed processes (--embedding only) via scale helper
    (cd "$REPO" && "$PYTHON" - <<'PY' || true
import importlib.util
import sys
from pathlib import Path
script = Path("/opt/ai/repo/rag_proxy/scripts/scale_ingest_capacity.py")
spec = importlib.util.spec_from_file_location("scale_ingest_capacity", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._kill_stray_gpu_embeds(set())
PY
    )
    sleep 2
    tries=$((tries + 1))
  done
  echo "warning: llama-server still on GPU; chat model may be loaded" >&2
  nvidia-smi 2>/dev/null || true
}

cd "$REPO"

echo "[ingest] stopping rag-admin (ingest worker)"
_systemctl stop rag-admin.service 2>/dev/null || true
sleep 2

_stop_embed_units
_wait_gpu_clear

echo "[bench] chunk stage (offline CPU)"
"$PYTHON" scripts/bench_ingest_capacity.py \
  --mode chunk \
  --semantic \
  --chunk-concurrency 1 2 3 4 \
  --documents 8 \
  --output "$OUT_DIR/chunk.json" \
  2>"$OUT_DIR/chunk-progress.log"

echo "[plan] apply capacity scale with GPU free"
if sudo -n systemctl start nomic-embed-scale.service 2>/dev/null; then
  echo "started nomic-embed-scale.service"
else
  "$PYTHON" scripts/scale_ingest_capacity.py \
    --apply \
    --scale-env "$SCALE_ENV" \
    --pool-env "$POOL_ENV" \
    | tee "$OUT_DIR/scale-apply.log"
fi

EMBED_URLS=""
if [[ -f "$POOL_ENV" ]]; then
  EMBED_URLS="$(grep -E '^INGEST_EMBED_URLS=' "$POOL_ENV" | cut -d= -f2- || true)"
fi
if [[ -z "$EMBED_URLS" ]]; then
  echo "error: no INGEST_EMBED_URLS in $POOL_ENV" >&2
  exit 1
fi

echo "[bench] embed stage (live pool: $EMBED_URLS)"
"$PYTHON" scripts/bench_ingest_capacity.py \
  --mode embed \
  --embed-urls "$EMBED_URLS" \
  --embed-concurrency 8 16 32 48 64 \
  --batch-size 32 64 128 \
  --documents 8 \
  --output "$OUT_DIR/embed.json" \
  2>"$OUT_DIR/embed-progress.log"

echo "[plan] dry-run rationale"
"$PYTHON" scripts/scale_ingest_capacity.py \
  --scale-env "$SCALE_ENV" \
  | tee "$OUT_DIR/planner-rationale.txt"

echo "[ingest] restarting rag-admin"
_systemctl start rag-admin.service 2>/dev/null || true

echo "[done] reports in $OUT_DIR"
ls -la "$OUT_DIR"
