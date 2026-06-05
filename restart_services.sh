#!/bin/bash
#
# Restart Stats production services (FastAPI Docker + Streamlit systemd).
# Intended for daily cron at 08:00 — see install_daily_restart_cron.sh
#
# Optional overrides (export before run or set in cron):
#   STATS_ROOT=/opt/stats
#   STREAMLIT_SERVICE=stats-streamlit
#   LOG_FILE=/var/log/stats_restart.log

set -u

STATS_ROOT="${STATS_ROOT:-/opt/stats}"
BACKEND_DIR="${BACKEND_DIR:-${STATS_ROOT}/backend}"
STREAMLIT_SERVICE="${STREAMLIT_SERVICE:-stats-streamlit}"
LOG_FILE="${LOG_FILE:-/var/log/stats_restart.log}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

restart_backend() {
  if [[ ! -d "$BACKEND_DIR" ]]; then
    log "ERROR: Backend directory not found: $BACKEND_DIR"
    return 1
  fi

  log "Restarting FastAPI backend ($BACKEND_DIR)..."
  if (cd "$BACKEND_DIR" && docker compose restart >> "$LOG_FILE" 2>&1); then
    log "FastAPI backend restarted successfully (docker compose)."
    return 0
  fi

  if (cd "$BACKEND_DIR" && docker-compose restart >> "$LOG_FILE" 2>&1); then
    log "FastAPI backend restarted successfully (docker-compose)."
    return 0
  fi

  log "ERROR: Failed to restart FastAPI backend."
  return 1
}

restart_frontend() {
  log "Restarting Streamlit ($STREAMLIT_SERVICE)..."
  if systemctl restart "$STREAMLIT_SERVICE" >> "$LOG_FILE" 2>&1; then
    log "Streamlit restarted successfully."
    return 0
  fi
  log "ERROR: Failed to restart Streamlit ($STREAMLIT_SERVICE)."
  return 1
}

wait_for_health() {
  local url="$1"
  local label="$2"
  local i

  for i in {1..30}; do
    if curl -sf "$url" >/dev/null 2>&1; then
      log "$label health check OK ($url)."
      return 0
    fi
    sleep 2
  done

  log "WARN: $label health check failed after 60s ($url)."
  return 1
}

main() {
  local backend_ok=0
  local frontend_ok=0

  {
    echo "----------------------------------------"
    log "Starting daily service restart..."
  } >> "$LOG_FILE" 2>&1

  if restart_backend; then
    backend_ok=1
    wait_for_health "http://127.0.0.1:8000/health" "Backend" || true
  fi

  if restart_frontend; then
    frontend_ok=1
    sleep 3
    if curl -sf -o /dev/null "http://127.0.0.1:8501" 2>/dev/null; then
      log "Frontend health check OK (http://127.0.0.1:8501)."
    else
      log "WARN: Frontend may still be starting (http://127.0.0.1:8501)."
    fi
  fi

  if [[ "$backend_ok" -eq 1 && "$frontend_ok" -eq 1 ]]; then
    log "Daily restart completed successfully."
    echo "----------------------------------------" >> "$LOG_FILE"
    exit 0
  fi

  log "Daily restart completed with errors (backend=$backend_ok frontend=$frontend_ok)."
  echo "----------------------------------------" >> "$LOG_FILE"
  exit 1
}

main "$@"