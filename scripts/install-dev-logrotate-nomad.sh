#!/usr/bin/env bash
# Run on nomad: hourly logrotate for dev nohup log (copytruncate, 10M x 3).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONF_SRC="${REPO_ROOT}/deploy/logrotate-rag-proxy-dev.conf"
CONF_DST="${HOME}/.config/rag-proxy/logrotate-dev.conf"
STATE="${HOME}/.config/rag-proxy/logrotate-dev.state"
CRON_LOG="${HOME}/.config/rag-proxy/logrotate-dev.cron.log"
CRON_MARK="# rag-proxy-dev-logrotate"
# Stagger from top-of-hour cron load (override with CRON_MINUTE).
CRON_MINUTE="${CRON_MINUTE:-17}"

if [ ! -f "${CONF_SRC}" ]; then
  echo "[X] missing logrotate config: ${CONF_SRC}" >&2
  exit 1
fi

mkdir -p "$(dirname "${CONF_DST}")"
cp -f "${CONF_SRC}" "${CONF_DST}"

LOGROTATE="$(command -v logrotate || true)"
if [ -z "${LOGROTATE}" ]; then
  echo "[X] logrotate not found; install it (e.g. apt install logrotate)" >&2
  exit 1
fi

CRON_LINE="${CRON_MINUTE} * * * * ${LOGROTATE} -s ${STATE} ${CONF_DST} >>${CRON_LOG} 2>&1"
TMP="$(mktemp)"
(crontab -l 2>/dev/null | grep -v "${CRON_MARK}" || true) >"${TMP}"
echo "${CRON_MARK}" >>"${TMP}"
echo "${CRON_LINE} ${CRON_MARK}" >>"${TMP}"
crontab "${TMP}"
rm -f "${TMP}"

echo "[OK] installed ${CONF_DST}"
echo "[OK] crontab:"
crontab -l | grep "${CRON_MARK}" || true
