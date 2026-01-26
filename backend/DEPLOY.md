# Barchart Options API - VPS Deployment Guide

## Domain: api.kdsinsured.com

## Quick Deployment Steps

### 1. Copy files to your VPS

```bash
# SSH into your VPS
ssh user@your-vps-ip

# Create directory
mkdir -p /opt/barchart-api
cd /opt/barchart-api

# Copy files (from your local machine)
scp -r backend/* user@your-vps-ip:/opt/barchart-api/
```

### 2. Build and Run with Docker

```bash
cd /opt/barchart-api

# Build and start
docker-compose up -d --build

# Verify it's running
docker ps | grep barchart-api
curl http://localhost:8000/health
```

### 3. Configure Nginx for api.kdsinsured.com

Create Nginx config:

```bash
sudo nano /etc/nginx/sites-available/api.kdsinsured.com
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name api.kdsinsured.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Longer timeouts for scraping (60 seconds)
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/api.kdsinsured.com /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. Setup SSL with Let's Encrypt

```bash
# Install certbot if not already installed
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d api.kdsinsured.com

# Auto-renewal is automatic, but you can test:
sudo certbot renew --dry-run
```

### 5. Verify Deployment

Test your API:

```bash
# Health check
curl https://api.kdsinsured.com/health

# Get options data
curl "https://api.kdsinsured.com/options?symbol=AAPL&date=2026-01-17"
```

## DNS Configuration

Make sure your domain DNS is configured:

- **Type:** A Record
- **Name:** api
- **Value:** Your VPS IP address
- **TTL:** 3600 (or auto)

## Update Streamlit Frontend

Create/update `.streamlit/secrets.toml`:

```toml
API_BASE_URL = "https://api.kdsinsured.com"
```

## Common Commands

```bash
# View logs
docker-compose logs -f barchart-api

# Restart
docker-compose restart

# Stop
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Check status
docker stats barchart-api
```

## Performance Tuning (Optional)

Update environment variables in `docker-compose.yml` to balance speed vs reliability:

- `BROWSER_CONCURRENCY=1` (lower = fewer bans, higher = more throughput)
- `OPTIONS_PAGELOAD_TIMEOUT=60` (page load timeout in seconds)
- `OPTIONS_WAIT_SECONDS=45` (wait for network requests in seconds)
- `OPTIONS_RETRY_COUNT=1` (retries for transient failures)
- `SPOT_TTL_SECONDS=15` / `SPOT_STALE_SECONDS=60` (spot freshness vs stability)

## API Endpoints

| Endpoint                                                             | Method | Description  |
| -------------------------------------------------------------------- | ------ | ------------ |
| `https://api.kdsinsured.com/`                                        | GET    | API info     |
| `https://api.kdsinsured.com/health`                                  | GET    | Health check |
| `https://api.kdsinsured.com/options?symbol=AAPL&date=2026-01-17`     | GET    | Options JSON |
| `https://api.kdsinsured.com/options/csv?symbol=AAPL&date=2026-01-17` | GET    | Options CSV  |

## Troubleshooting

### Container not starting

```bash
docker-compose logs barchart-api
```

### Chrome memory issues

Ensure your VPS has at least 2GB RAM available. The container is configured with 4GB limit.

### 502 Bad Gateway

Check if container is running: `docker ps`
Check container logs: `docker-compose logs -f`

### Slow responses

Scraping takes 30-60 seconds per request. This is normal.
