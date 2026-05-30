#!/usr/bin/env bash
# Bound /tmp/rag_proxy_test.log size for the nomad dev instance (nohup stdout).
# Sourced by restart-dev-nomad.sh or run via logrotate (copytruncate while running).
set -euo pipefail

LOG="${LOG:-/tmp/rag_proxy_test.log}"
MAX_LOG_MB="${MAX_LOG_MB:-10}"
KEEP_ROTATIONS="${KEEP_ROTATIONS:-3}"

max_bytes=$((MAX_LOG_MB * 1024 * 1024))

_file_mtime() {
  stat -c%Y "$1" 2>/dev/null || stat -f%m "$1" 2>/dev/null || echo 0
}

_log_size_bytes() {
  local sz
  sz=$(stat -c%s "$LOG" 2>/dev/null) && { echo "$sz"; return; }
  sz=$(stat -f%z "$LOG" 2>/dev/null) && { echo "$sz"; return; }
  wc -c <"$LOG" 2>/dev/null | tr -d ' \n' || echo 0
}

_prune_file_list() {
  local extra=$1
  shift
  local -a files=("$@")
  local n=${#files[@]}
  if [ "$n" -lt "$extra" ]; then
    return 0
  fi
  local i=0 f
  while IFS= read -r f; do
    i=$((i + 1))
    if [ "$i" -ge "$extra" ]; then
      rm -f -- "$f"
    fi
  done < <(
    for f in "${files[@]}"; do
      printf '%s\t%s\n' "$(_file_mtime "$f")" "$f"
    done | sort -t$'\t' -k1,1rn | cut -f2-
  )
}

_prune_rotations() {
  local extra gz plain f
  extra=$((KEEP_ROTATIONS + 1))
  gz=()
  plain=()
  shopt -s nullglob
  for f in "${LOG}".*.gz; do
    gz+=("$f")
  done
  for f in "${LOG}".*; do
    case "$f" in
      *.gz) ;;
      "${LOG}") ;;
      *) plain+=("$f") ;;
    esac
  done
  shopt -u nullglob
  if [ ${#gz[@]} -gt 0 ]; then
    _prune_file_list "$extra" "${gz[@]}"
  fi
  if [ ${#plain[@]} -gt 0 ]; then
    _prune_file_list "$extra" "${plain[@]}"
  fi
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
  sz=$(_log_size_bytes)
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
