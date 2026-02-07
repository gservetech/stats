# Stats Streamlit App - VPS Deployment Guide

This guide is for deploying the Streamlit frontend only.

Backend deployment is documented separately in `backend/DEPLOY.md`.

## Target Architecture

- `api.kdsinsured.com` -> FastAPI (`127.0.0.1:8000`)
- `research.kdsinsured.com` -> Streamlit (`127.0.0.1:8501`)
- Nginx terminates SSL and reverse-proxies both services.

## 1. Prepare Streamlit App on VPS

```bash
# Example app path
mkdir -p /opt/stats
cd /opt/stats

# Copy full Streamlit project here (app.py, stats_app/, requirements.txt, etc.)
# Example:
# scp -r . user@your-vps-ip:/opt/stats/

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure Streamlit Secrets

Create `/opt/stats/.streamlit/secrets.toml`:

```toml
API_BASE_URL = "http://127.0.0.1:8000"
FINNHUB_API_KEY = "YOUR_FINNHUB_KEY"
```

Notes:

- Use local backend URL (`127.0.0.1`) for best latency and reliability.
- Do not commit secrets to git.

## 3. (Optional) Streamlit Runtime Config

Create `/opt/stats/.streamlit/config.toml`:

```toml
[server]
headless = true
address = "127.0.0.1"
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## 4. Run Streamlit with systemd

Create service file:

```bash
sudo nano /etc/systemd/system/stats-streamlit.service
```

Paste:

```ini
[Unit]
Description=Stats Streamlit App
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/stats
Environment="PYTHONUNBUFFERED=1"
ExecStart=/opt/stats/.venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stats-streamlit
sudo systemctl status stats-streamlit
```

Logs:

```bash
journalctl -u stats-streamlit -f
```

## 5. Nginx for Streamlit Subdomain

Create config:

```bash
sudo nano /etc/nginx/sites-available/research.kdsinsured.com
```

```nginx
server {
    listen 80;
    server_name research.kdsinsured.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/app.kdsinsured.com /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 6. SSL Certificate

```bash
sudo certbot --nginx -d app.kdsinsured.com
sudo certbot renew --dry-run
```

## 7. Verify End-to-End

```bash
# Local checks on VPS
curl http://127.0.0.1:8000/health
curl -I http://127.0.0.1:8501

# Public checks
curl -I https://app.kdsinsured.com
curl https://api.kdsinsured.com/health
```

## 8. Operations

```bash
# Streamlit service
sudo systemctl restart stats-streamlit
sudo systemctl stop stats-streamlit
sudo systemctl status stats-streamlit

# Nginx
sudo nginx -t
sudo systemctl reload nginx

# Logs
journalctl -u stats-streamlit -n 200 --no-pager
sudo tail -f /var/log/nginx/access.log /var/log/nginx/error.log
```

## 9. DNS Records Required

- `A` record: `api` -> VPS IP
- `A` record: `app` -> VPS IP

## 10. Common Issues

- Infinite loading with no backend logs:
  - Verify Streamlit secret `API_BASE_URL` is `http://127.0.0.1:8000` on VPS.
  - Confirm Streamlit service can reach backend: `curl http://127.0.0.1:8000/health`.
- WebSocket/disconnect issues:
  - Ensure Nginx has `Upgrade`/`Connection` headers in Streamlit vhost.
- 502 on `app.kdsinsured.com`:
  - Check Streamlit service status and logs.
