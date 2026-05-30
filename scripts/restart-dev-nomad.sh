#!/usr/bin/env bash
# Run on nomad: restart dev rag_proxy (8087) and rotate its log file.
# Set DEV_DIR to the dev checkout (default: $HOME/rag_proxy).
set -euo pipefail

DEV_DIR="${DEV_DIR:-${HOME}/rag_proxy}"
LOG="${LOG:-/tmp/rag_proxy_test.log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=dev-log-cap.sh
source "${SCRIPT_DIR}/dev-log-cap.sh"

if [ ! -d "${DEV_DIR}" ]; then
  echo "[X] dev dir missing: ${DEV_DIR} (set DEV_DIR)" >&2
  exit 1
fi

rotate_dev_log_for_restart

cd "${DEV_DIR}"
if [ ! -f .env ]; then
  echo "[X] missing ${DEV_DIR}/.env" >&2
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
set -a
# shellcheck disable=SC1091
. ./.env
set +a

dev_port="${PROXY_PORT:-8087}"
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${dev_port}/tcp" 2>/dev/null || true
  sleep 1
else
  listener_pid=""
  if command -v ss >/dev/null 2>&1; then
    listener_pid="$(
      ss -tlnp 2>/dev/null |
        grep -E ":${dev_port}([^0-9]|$)" |
        sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' |
        head -1 || true
    )"
  fi
  if [ -z "${listener_pid}" ] && command -v lsof >/dev/null 2>&1; then
    listener_pid="$(
      lsof -t -iTCP:"${dev_port}" -sTCP:LISTEN 2>/dev/null | head -1 || true
    )"
  fi
  if [ -n "${listener_pid}" ]; then
    kill -TERM "${listener_pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "${listener_pid}" 2>/dev/null || true
  else
    echo "[!] no listener found on port ${dev_port}" >&2
  fi
  sleep 1
fi

nohup python rag_proxy.py >"${LOG}" 2>&1 &
sleep 3

echo "[pid]"
ss -tlnp 2>/dev/null | grep -E ":${dev_port}([^0-9]|$)" || true

echo "[health]"
if ! curl -s -m 5 -o /dev/null -w "${dev_port} metrics: HTTP %{http_code}\n" \
  "http://127.0.0.1:${dev_port}/metrics"; then
  echo "[!] metrics probe failed on port ${dev_port}" >&2
fi

echo "[log]"
tail -15 "${LOG}"
