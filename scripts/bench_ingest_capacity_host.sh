#!/usr/bin/env bash
# Stop active ingest, free GPU VRAM, benchmark chunk+embed throughput, apply planner, restart.
#
# Usage:
#   bash scripts/bench_ingest_capacity_host.sh
#   bash scripts/bench_ingest_capacity_host.sh --skip-restart   # stop+bench only
#
# Full systemd pool start/stop needs passwordless sudo:
#   bash scripts/update-buster-embed-gpu.sh
set -uo pipefail

REPO="${REPO_ROOT:-/opt/ai/repo/rag_proxy}"
PYTHON="${PYTHON:-/opt/ai/venv/bin/python}"
HOST_PY="${HOST_PY:-$REPO/scripts/bench_ingest_host.py}"
CONFIG_DIR="${CONFIG_DIR:-/opt/ai/config}"
ADMIN_DB="${ADMIN_DB:-/opt/ai/rag/admin.sqlite}"
SCALE_ENV="${SCALE_ENV:-$CONFIG_DIR/nomic-embed-scale.env}"
POOL_ENV="${POOL_ENV:-$CONFIG_DIR/nomic-embed-pool.env}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-/tmp/ingest-bench-$STAMP}"
PID_FILE="${PID_FILE:-$OUT_DIR/nomic-bench-pool.pids}"
POOL_LOG_DIR="${POOL_LOG_DIR:-$OUT_DIR/pool-logs}"
MIN_HEALTHY="${MIN_HEALTHY:-1}"

SKIP_RESTART=0
WAS_PAUSED="false"
RAG_ADMIN_WAS_ACTIVE=0
USED_FALLBACK_POOL=0
BENCH_STATUS=0

usage() {
  sed -n '2,10p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-restart) SKIP_RESTART=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR" "$POOL_LOG_DIR"

_systemctl() {
  if [[ -x /opt/ai/bin/nomic-pool-systemctl ]]; then
    sudo -n /opt/ai/bin/nomic-pool-systemctl "$@" 2>/dev/null && return 0
  fi
  if sudo -n systemctl "$@" 2>/dev/null; then
    return 0
  fi
  systemctl "$@" 2>/dev/null || sudo systemctl "$@" 2>/dev/null || return 1
}

_rag_admin_active() {
  systemctl is-active --quiet rag-admin.service 2>/dev/null
}

_stop_embed_units() {
  echo "[stop] nomic-embed pool and query embed (:8089)"
  _systemctl stop nomic-embed.service 2>/dev/null || true
  _systemctl disable nomic-embed.service 2>/dev/null || true
  local port
  while read -r port; do
    [[ -z "$port" ]] && continue
    _systemctl stop "nomic-embed@${port}.service" 2>/dev/null || true
    _systemctl disable "nomic-embed@${port}.service" 2>/dev/null || true
  done < <("$PYTHON" "$HOST_PY" list-stop-ports --scale-env "$SCALE_ENV" --show-reserved)
}

_wait_gpu_clear() {
  echo "[stop] waiting for embed llama-server processes to exit"
  local tries=0
  while (( tries < 45 )); do
    if ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
      | grep -q llama-server; then
      nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || true
      return 0
    fi
    "$PYTHON" "$HOST_PY" kill-strays >/dev/null 2>&1 || true
    sleep 2
    tries=$((tries + 1))
  done
  echo "warning: llama-server still on GPU (chat model may be loaded)" >&2
  nvidia-smi 2>/dev/null || true
}

_stop_phase() {
  echo "[stop] pausing ingest in admin DB"
  WAS_PAUSED="$("$PYTHON" "$HOST_PY" prepare-pause --admin-db "$ADMIN_DB" || echo false)"

  if _rag_admin_active; then
    RAG_ADMIN_WAS_ACTIVE=1
  fi
  echo "[stop] stopping rag-admin (ingest worker)"
  _systemctl stop rag-admin.service 2>/dev/null || true
  sleep 2

  _stop_embed_units
  "$PYTHON" "$HOST_PY" kill-strays >/dev/null 2>&1 || true
  _wait_gpu_clear
}

_start_pool_systemd() {
  if sudo -n systemctl start nomic-embed-scale.service 2>/dev/null; then
    echo "[pool] started via nomic-embed-scale.service"
    sleep 5
    return 0
  fi
  if "$PYTHON" "$REPO/scripts/scale_ingest_capacity.py" \
    --apply \
    --scale-env "$SCALE_ENV" \
    --pool-env "$POOL_ENV" \
    | tee "$OUT_DIR/scale-apply.log"; then
    echo "[pool] started via scale_ingest_capacity.py --apply"
    return 0
  fi
  return 1
}

_start_pool_fallback() {
  echo "[pool] systemd apply failed; starting llama-server fallback pool"
  USED_FALLBACK_POOL=1
  "$PYTHON" "$HOST_PY" start-pool \
    --scale-env "$SCALE_ENV" \
    --pool-env "$POOL_ENV" \
    --pid-file "$PID_FILE" \
    --log-dir "$POOL_LOG_DIR" \
    --min-healthy "$MIN_HEALTHY" \
    | tee "$OUT_DIR/pool-urls.txt"
}

_ensure_pool() {
  local urls=""
  if [[ -f "$POOL_ENV" ]]; then
    urls="$(grep -E '^INGEST_EMBED_URLS=' "$POOL_ENV" | cut -d= -f2- || true)"
  fi
  if [[ -n "$urls" ]]; then
    local first="${urls%%,*}"
    if curl -sf -m 3 -X POST "${first}/v1/embeddings" \
      -H 'Content-Type: application/json' \
      -d '{"model":"nomic-embed-text-v1.5","input":["ok"]}' \
      | grep -q embedding; then
      echo "[pool] already healthy: $urls"
      return 0
    fi
  fi

  if _start_pool_systemd; then
    :
  else
    _start_pool_fallback || return 1
  fi

  if [[ -f "$POOL_ENV" ]]; then
    urls="$(grep -E '^INGEST_EMBED_URLS=' "$POOL_ENV" | cut -d= -f2- || true)"
  fi
  if [[ -z "$urls" ]]; then
    echo "error: no INGEST_EMBED_URLS after pool start" >&2
    return 1
  fi
  echo "[pool] embed URLs: $urls"
}

_bench_chunk() {
  echo "[bench] chunk stage (offline CPU)"
  if ! "$PYTHON" "$REPO/scripts/bench_ingest_capacity.py" \
    --mode chunk \
    --semantic \
    --chunk-concurrency 1 2 3 4 \
    --documents 8 \
    --output "$OUT_DIR/chunk.json" \
    2>"$OUT_DIR/chunk-progress.log"; then
    echo "warning: chunk benchmark failed (see $OUT_DIR/chunk-progress.log)" >&2
    BENCH_STATUS=1
  fi
}

_bench_embed() {
  local urls
  urls="$(grep -E '^INGEST_EMBED_URLS=' "$POOL_ENV" | cut -d= -f2- || true)"
  if [[ -z "$urls" ]]; then
    echo "warning: skipping embed benchmark (no pool URLs)" >&2
    BENCH_STATUS=1
    return 1
  fi
  echo "[bench] embed stage (live pool)"
  if ! "$PYTHON" "$REPO/scripts/bench_ingest_capacity.py" \
    --mode embed \
    --embed-urls "$urls" \
    --embed-concurrency 8 16 32 48 64 \
    --batch-size 32 64 128 \
    --documents 8 \
    --output "$OUT_DIR/embed.json" \
    2>"$OUT_DIR/embed-progress.log"; then
    echo "warning: embed benchmark failed (see $OUT_DIR/embed-progress.log)" >&2
    BENCH_STATUS=1
    return 1
  fi
}

_planner_report() {
  echo "[plan] dry-run rationale"
  "$PYTHON" "$REPO/scripts/scale_ingest_capacity.py" \
    --scale-env "$SCALE_ENV" \
    | tee "$OUT_DIR/planner-rationale.txt" || true
}

_restart_phase() {
  if (( SKIP_RESTART )); then
    echo "[restart] skipped (--skip-restart)"
    return 0
  fi

  echo "[restart] applying final capacity plan"
  if _start_pool_systemd; then
    if (( USED_FALLBACK_POOL )); then
      echo "[restart] stopping fallback llama-server processes"
      "$PYTHON" "$HOST_PY" stop-pool --pid-file "$PID_FILE" || true
    fi
  else
    if (( ! USED_FALLBACK_POOL )); then
      _start_pool_fallback || echo "warning: final pool start failed" >&2
    else
      echo "[restart] fallback pool left running (PID file $PID_FILE)"
      echo "[restart] install sudoers + nomic-embed-scale.service for systemd pool"
    fi
  fi

  echo "[restart] restoring ingest pause flag"
  "$PYTHON" "$HOST_PY" restore-pause --admin-db "$ADMIN_DB" --was-paused "$WAS_PAUSED" || true

  echo "[restart] starting rag-admin"
  if _systemctl start rag-admin.service 2>/dev/null; then
    echo "[restart] rag-admin started"
  elif (( RAG_ADMIN_WAS_ACTIVE )); then
    echo "warning: could not start rag-admin (sudo may be required)" >&2
    BENCH_STATUS=1
  fi
}

_cleanup() {
  _restart_phase
}

trap _cleanup EXIT

cd "$REPO"

_stop_phase
_bench_chunk
_ensure_pool || BENCH_STATUS=1
if [[ -f "$POOL_ENV" ]]; then
  _bench_embed || true
fi
_planner_report

echo "[done] reports in $OUT_DIR (exit $BENCH_STATUS)"
ls -la "$OUT_DIR" || true
exit "$BENCH_STATUS"
