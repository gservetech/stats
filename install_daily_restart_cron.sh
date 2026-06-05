#!/bin/bash
#
# Install a cron job to restart Streamlit + FastAPI every day at 08:00.
# Run on the VPS as root: sudo bash install_daily_restart_cron.sh
#
# Options (environment variables):
#   STATS_ROOT          App root on VPS (default: /opt/stats)
#   CRON_HOUR           Hour 0-23 (default: 8)
#   CRON_MINUTE         Minute 0-59 (default: 0)
#   CRON_TZ             IANA timezone, e.g. America/New_York (optional)
#   USE_SYSTEM_CRON_D   Use /etc/cron.d/ instead of root crontab (default: 1)

set -euo pipefail

STATS_ROOT="${STATS_ROOT:-/opt/stats}"
CRON_HOUR="${CRON_HOUR:-8}"
CRON_MINUTE="${CRON_MINUTE:-0}"
CRON_TZ="${CRON_TZ:-}"
USE_SYSTEM_CRON_D="${USE_SYSTEM_CRON_D:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTART_SCRIPT="${STATS_ROOT}/restart_services.sh"
CRON_D_FILE="/etc/cron.d/stats-daily-restart"
CRON_MARKER="# stats-daily-restart"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: Run as root: sudo bash $0"
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/restart_services.sh" ]]; then
  echo "ERROR: restart_services.sh not found in $SCRIPT_DIR"
  exit 1
fi

mkdir -p "$(dirname "$STATS_ROOT")"
if [[ "$SCRIPT_DIR" != "$STATS_ROOT" ]]; then
  cp -f "$SCRIPT_DIR/restart_services.sh" "$RESTART_SCRIPT"
  echo "Copied restart_services.sh -> $RESTART_SCRIPT"
fi

chmod +x "$RESTART_SCRIPT"
touch /var/log/stats_restart.log
chmod 644 /var/log/stats_restart.log

CRON_SCHEDULE="${CRON_MINUTE} ${CRON_HOUR} * * *"
TZ_LINE=""
if [[ -n "$CRON_TZ" ]]; then
  TZ_LINE="CRON_TZ=${CRON_TZ}"
fi

ENV_PREFIX="STATS_ROOT=${STATS_ROOT} BACKEND_DIR=${STATS_ROOT}/backend"

if [[ "$USE_SYSTEM_CRON_D" == "1" ]]; then
  {
    echo "SHELL=/bin/bash"
    echo "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    [[ -n "$TZ_LINE" ]] && echo "$TZ_LINE"
    echo "${CRON_SCHEDULE} root ${ENV_PREFIX} ${RESTART_SCRIPT} ${CRON_MARKER}"
  } > "$CRON_D_FILE"
  chmod 644 "$CRON_D_FILE"
  echo "Installed system cron: $CRON_D_FILE"
  echo "Schedule: every day at $(printf '%02d:%02d' "$CRON_HOUR" "$CRON_MINUTE")${CRON_TZ:+ ($CRON_TZ)}"
else
  TMP="$(mktemp)"
  crontab -l 2>/dev/null | grep -v "$CRON_MARKER" > "$TMP" || true
  {
    [[ -n "$TZ_LINE" ]] && echo "$TZ_LINE"
    echo "${CRON_SCHEDULE} ${ENV_PREFIX} ${RESTART_SCRIPT} ${CRON_MARKER}"
  } >> "$TMP"
  crontab "$TMP"
  rm -f "$TMP"
  echo "Installed root crontab entry."
  echo "Schedule: every day at $(printf '%02d:%02d' "$CRON_HOUR" "$CRON_MINUTE")${CRON_TZ:+ ($CRON_TZ)}"
fi

echo ""
echo "Verify:"
echo "  cat $CRON_D_FILE 2>/dev/null || crontab -l | grep stats"
echo "  sudo bash $RESTART_SCRIPT"
echo "  tail -f /var/log/stats_restart.log"