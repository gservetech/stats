# Barchart Options API

A FastAPI-based service that fetches options data from Barchart.com and returns it as JSON or CSV.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment (VPS)

```bash
# Build and run with Docker Compose
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📡 API Endpoints

### Health Check

```
GET /
GET /health
```

### Get Options Data (JSON)

```
GET /options?symbol=$SPX&expiration=2026-01-07-w
```

**Parameters:**

- `symbol` (required): Stock symbol (e.g., `$SPX`, `AAPL`, `TSLA`)
- `expiration` (required): Expiration date in format `YYYY-MM-DD` or `YYYY-MM-DD-w` (weekly)

**Response:**

```json
{
  "success": true,
  "symbol": "$SPX",
  "expiration": "2026-01-07-w",
  "count": 150,
  "data": [
    {
      "Call Latest": "50.20",
      "Call Bid": "49.80",
      "Call Ask": "50.60",
      "Call Change": "+1.25",
      "Call Volume": "1,234",
      "Call Open Int": "5,678",
      "Call IV": "18.50%",
      "Call Last Trade": "01/07/26",
      "Strike": "5,900.00",
      "Put Latest": "45.30",
      "Put Bid": "44.90",
      "Put Ask": "45.70",
      "Put Change": "-0.85",
      "Put Volume": "2,345",
      "Put Open Int": "6,789",
      "Put IV": "19.20%",
      "Put Last Trade": "01/07/26"
    }
  ]
}
```

### Get Options Data (CSV)

```
GET /options/csv?symbol=$SPX&expiration=2026-01-07-w
```

Returns a downloadable CSV file.

## 🔌 Calling from Streamlit

```python
import requests
import pandas as pd

# API base URL (update with your VPS IP/domain)
API_URL = "http://your-vps-ip:8000"

def get_options_data(symbol: str, expiration: str):
    """Fetch options data from the Barchart API."""
    response = requests.get(
        f"{API_URL}/options",
        params={"symbol": symbol, "expiration": expiration}
    )

    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data["data"])
    else:
        raise Exception(response.json().get("detail", "Unknown error"))

# Example usage in Streamlit
symbol = st.text_input("Symbol", value="$SPX")
expiration = st.text_input("Expiration", value="2026-01-07-w")

if st.button("Fetch Options"):
    df = get_options_data(symbol, expiration)
    st.dataframe(df)
```

## 📁 Project Structure

```
barchart-data/
├── api.py              # FastAPI server
├── main.py             # CLI script (original)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image config
├── docker-compose.yml  # Docker Compose config
└── README.md           # This file
```

## ⚙️ VPS Deployment Steps

1. **SSH into your VPS:**

   ```bash
   ssh user@your-vps-ip
   ```

2. **Install Docker:**

   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo apt install docker-compose -y
   ```

3. **Clone/upload the project:**

   ```bash
   git clone <your-repo> barchart-api
   cd barchart-api
   ```

4. **Deploy:**

   ```bash
   docker-compose up -d --build
   ```

5. **Configure firewall:**

   ```bash
   sudo ufw allow 8000/tcp
   ```

6. **Test the API:**
   ```bash
   curl "http://localhost:8000/options?symbol=\$SPX&expiration=2026-01-07-w"
   ```

## 🔒 Production Notes

- For production, consider adding:
  - API key authentication
  - Rate limiting
  - HTTPS (use nginx as reverse proxy)
  - Proper logging
  - Error monitoring

## 📝 License

MIT
