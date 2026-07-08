#!/usr/bin/env bash
# Allowlisted systemctl wrapper for nomic embed pool lifecycle.
# Installed to /opt/ai/bin/nomic-pool-systemctl; invoked via passwordless sudo.
set -euo pipefail

SYSTEMCTL="${SYSTEMCTL:-/usr/bin/systemctl}"

_allowed_unit() {
  case "$1" in
    nomic-embed@*.service | nomic-embed.service | nomic-embed-scale.service | sparse-sidecar.service | rerank-sidecar.service) return 0 ;;
  esac
  return 1
}

cmd="${1:-}"
shift || true

case "$cmd" in
  start | stop | restart | enable | disable | show | is-active)
    unit="${1:-}"
    if [[ -z "$unit" ]] || ! _allowed_unit "$unit"; then
      echo "nomic-pool-systemctl: unit not allowed: ${unit:-<missing>}" >&2
      exit 1
    fi
    exec "$SYSTEMCTL" "$cmd" "$@"
    ;;
  list-units)
    for arg in "$@"; do
      if [[ "$arg" == nomic-embed@* ]]; then
        exec "$SYSTEMCTL" list-units "$@"
      fi
    done
    echo "nomic-pool-systemctl: list-units requires nomic-embed@* filter" >&2
    exit 1
    ;;
  *)
    echo "nomic-pool-systemctl: command not allowed: $cmd" >&2
    exit 1
    ;;
esac
