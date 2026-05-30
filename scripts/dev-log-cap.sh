#!/usr/bin/env bash
# Bound /tmp/rag_proxy_test.log size for the nomad dev instance (nohup stdout).
# Sourced by restart-dev-nomad.sh or run via logrotate (copytruncate while running).
set -euo pipefail

LOG="${LOG:-/tmp/rag_proxy_test.log}"
MAX_LOG_MB="${MAX_LOG_MB:-10}"
KEEP_ROTATIONS="${KEEP_ROTATIONS:-2}"

max_bytes=$((MAX_LOG_MB * 1024 * 1024))

_prune_rotations() {
  local extra
  extra=$((KEEP_ROTATIONS + 1))
  # shellcheck disable=SC2012
  ls -1t "${LOG}".*.gz 2>/dev/null | tail -n +"${extra}" | xargs -r rm -f
  # shellcheck disable=SC2012
  ls -1t "${LOG}".* 2>/dev/null | grep -v '\.gz$' | tail -n +"${extra}" | xargs -r rm -f
}

# Rotate before a fresh nohup start (process not writing yet).
rotate_dev_log_for_restart() {
  if [ ! -f "$LOG" ]; then
    return 0
  fi
  local ts rotated
  ts=$(date +%Y%m%d%H%M%S)
  rotated="${LOG}.${ts}"
  mv -f "$LOG" "${rotated}"
  if command -v gzip >/dev/null 2>&1; then
    gzip -f "${rotated}" 2>/dev/null || true
  fi
  _prune_rotations
}

# If log grew past cap while process runs, use logrotate copytruncate instead.
# This helper is for manual/cron use when logrotate is unavailable.
truncate_dev_log_if_huge() {
  [ -f "$LOG" ] || return 0
  local sz
  sz=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "${sz}" -gt "${max_bytes}" ]; then
    local ts rotated
    ts=$(date +%Y%m%d%H%M%S)
    rotated="${LOG}.${ts}"
    cp -f "$LOG" "${rotated}"
    if command -v gzip >/dev/null 2>&1; then
      gzip -f "${rotated}" 2>/dev/null || true
    fi
    : >"$LOG"
    _prune_rotations
  fi
}
