#!/bin/bash

# ==========================================
# Service Restart Script for VPS (Hostinger)
# ==========================================
# This script restarts the Streamlit systemd service and the FastAPI Docker containers.
# It is designed to be run via cron every day to keep the services fresh.

# Log file for the restarts
LOG_FILE="/var/log/stats_restart.log"

echo "----------------------------------------" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting daily restart of services..." >> "$LOG_FILE"

# 1. Restart Streamlit Service
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting Streamlit (stats-streamlit.service)..." >> "$LOG_FILE"
if systemctl restart stats-streamlit; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Streamlit restarted successfully." >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to restart Streamlit." >> "$LOG_FILE"
fi

# 2. Restart FastAPI Backend (Docker Compose)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting FastAPI Backend (/opt/barchart-api)..." >> "$LOG_FILE"
if cd /opt/barchart-api && docker-compose restart; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FastAPI Backend restarted successfully." >> "$LOG_FILE"
else
    # Fallback in case they use 'docker compose' instead of 'docker-compose'
    if cd /opt/barchart-api && docker compose restart; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FastAPI Backend restarted successfully." >> "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to restart FastAPI Backend." >> "$LOG_FILE"
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restart process completed." >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"
