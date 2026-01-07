"""
Barchart Options API Server
API to scrape side-by-side options data from Barchart

Endpoints:
- /options - Get options data (JSON) with symbol & date
- /options/csv - Get options data (CSV) with symbol & date
"""
import asyncio
import json
import base64
import re
from io import StringIO
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions

app = FastAPI(
    title="Barchart Options API",
    description="API to scrape Barchart options data with symbol and date",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def _to_float(val, default=None):
    if val is None: return default
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    if s in ("", "N/A", "na", "None"): return default
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ("", "-", "."): return default
    try: return float(s)
    except: return default

def _to_int(val, default=0):
    f = _to_float(val, None)
    return int(round(f)) if f is not None else default

def _fmt_price(x): return f"{x:,.2f}" if x is not None else ""
def _fmt_int(x): return f"{int(x):,}" if x is not None else ""
def _fmt_iv(val):
    if val is None: return ""
    if isinstance(val, str): return val.strip()
    if isinstance(val, (int, float)):
        return f"{val * 100:.2f}%" if val <= 10 else f"{val:.2f}%"
    return str(val)

def _pick(option_obj):
    """Extract and format option data."""
    if not option_obj: 
        return {k: "" for k in ["Latest", "Bid", "Ask", "Change", "Volume", "Open Int", "IV", "Last Trade", "raw_trade_time"]}
    raw = option_obj.get("raw") or {}
    latest = _to_float(option_obj.get("lastPrice"), raw.get("lastPrice"))
    bid = _to_float(option_obj.get("bidPrice"), raw.get("bidPrice"))
    ask = _to_float(option_obj.get("askPrice"), raw.get("askPrice"))
    volume = _to_int(option_obj.get("volume"), raw.get("volume"))
    oi = _to_int(option_obj.get("openInterest"), raw.get("openInterest"))
    iv_val = option_obj.get("volatility") or raw.get("volatility")
    
    return {
        "Latest": _fmt_price(latest),
        "Bid": _fmt_price(bid),
        "Ask": _fmt_price(ask),
        "Change": str(option_obj.get("priceChange") or ""),
        "Volume": _fmt_int(volume),
        "Open Int": _fmt_int(oi),
        "IV": _fmt_iv(iv_val),
        "Last Trade": str(option_obj.get("tradeTime") or ""),
        "raw_trade_time": raw.get("tradeTime", 0)
    }


def process_options_data(opt_json):
    """Process options JSON data into side-by-side format."""
    data = opt_json.get("data", {})
    rows = []

    strike_items = {}
    if isinstance(data, dict):
        if "Call" in data or "Put" in data:
            # Standard View
            for t in ["Call", "Put"]:
                for item in data.get(t, []):
                    s = item.get("strikePrice")
                    if s not in strike_items: strike_items[s] = []
                    strike_items[s].append(item)
        else:
            # SBS View (Keys are already Strikes)
            strike_items = data
    elif isinstance(data, list):
        for item in data:
            s = item.get("strikePrice")
            if s not in strike_items: strike_items[s] = []
            strike_items[s].append(item)

    for strike_str, items in strike_items.items():
        if not isinstance(items, list):
            items = [items]
        call_obj = next((i for i in items if i.get("optionType") == "Call"), None)
        put_obj = next((i for i in items if i.get("optionType") == "Put"), None)
        
        c = _pick(call_obj)
        p = _pick(put_obj)
        strike_num = _to_float(strike_str, 0)
        
        row = {
            "Call Latest": c["Latest"], 
            "Call Bid": c["Bid"], 
            "Call Ask": c["Ask"], 
            "Call Change": c["Change"], 
            "Call Volume": c["Volume"], 
            "Call OI": c["Open Int"], 
            "Call IV": c["IV"],
            "Strike": f"{strike_num:,.2f}" if strike_num else strike_str,
            "Put Latest": p["Latest"], 
            "Put Bid": p["Bid"], 
            "Put Ask": p["Ask"], 
            "Put Change": p["Change"], 
            "Put Volume": p["Volume"], 
            "Put OI": p["Open Int"], 
            "Put IV": p["IV"],
        }
        rows.append((strike_num, row))
    
    rows.sort(key=lambda x: x[0])
    return [r for _, r in rows]


async def scrape_options(symbol: str, date: str):
    """
    Scrape options data from Barchart for the given symbol and date.
    Uses network interception to capture API responses.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, $SPX, TSLA)
        date: Expiration date (e.g., 2026-01-17 or 2026-01-10-w for weekly)
    
    Returns:
        List of options data in side-by-side format
    """
    # Build URL - use SBS view for side-by-side
    url = f"https://www.barchart.com/stocks/quotes/{symbol}/options?expiration={date}&view=sbs"
    print(f"[INFO] Scraping: {url}")
    
    # Dictionary to store captured requests
    captured_requests = {}

    async def on_response(response_log):
        params = response_log.get("params", {})
        response = params.get("response", {})
        resp_url = response.get("url", "")
        
        # Capture Options Data (main table)
        if "/proxies/core-api/v1/options/get" in resp_url and "options" not in captured_requests:
            print(f"[INFO] Detected Options API call: {resp_url[:80]}...")
            captured_requests["options"] = (params.get("requestId"), resp_url)
            
        # Capture Expirations/Volume Data (summary stats)
        elif "/proxies/core-api/v1/options-expirations/get" in resp_url and "expirations" not in captured_requests:
            print(f"[INFO] Detected Expirations API call: {resp_url[:80]}...")
            captured_requests["expirations"] = (params.get("requestId"), resp_url)

    # Configure Chrome for Docker/Server environment
    options = ChromiumOptions()
    
    # Set Chrome binary path explicitly for Docker
    options.binary_location = "/usr/bin/google-chrome"
    
    # REQUIRED for Docker: Run in headless mode
    options.add_argument("--headless=new")
    
    # REQUIRED for Docker: Disable sandbox (needed when running as root)
    options.add_argument("--no-sandbox")
    
    # REQUIRED for Docker: Use /dev/shm workaround
    options.add_argument("--disable-dev-shm-usage")
    
    # Performance and stability flags for Docker
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--remote-debugging-port=0")
    
    # Anti-detection flags
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    print("[INFO] Starting browser (headless mode)...")
    print(f"[INFO] Chrome binary: {options.binary_location}")
    async with Chrome(options=options) as browser:
        tab = await browser.start()
        
        # Enable and listen to network events
        await tab.enable_network_events()
        await tab.on("Network.responseReceived", on_response)
        
        print(f"[INFO] Navigating to: {url}")
        try:
            await tab.go_to(url)
        except Exception as e:
            print(f"[ERROR] Navigation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load page: {str(e)}")

        # Wait for the page to load and API calls to fire (max 25 seconds)
        print("[INFO] Waiting for network requests (max 25s)...")
        for i in range(25):
            await asyncio.sleep(1)
            # If we've caught the options API, we can stop waiting
            if "options" in captured_requests:
                await asyncio.sleep(2)  # Wait a bit more for data to complete
                break
            if i % 5 == 0 and i > 0:
                print(f"[INFO] Waiting... {i}/25 seconds")
        
        if "options" not in captured_requests:
            print("[ERROR] Options API call not captured")
            raise HTTPException(
                status_code=404, 
                detail=f"Options data not found for {symbol} on {date}. Please verify the symbol and expiration date are valid."
            )
        
        # Extract response body
        request_id, api_url = captured_requests["options"]
        print(f"[INFO] Processing options data from request {request_id}...")
        
        try:
            # get_network_response_body returns a dict with 'body' and 'base64Encoded'
            body_data = await tab.get_network_response_body(request_id)
            
            if isinstance(body_data, dict):
                body = body_data.get("body", "")
                if body_data.get("base64Encoded"):
                    body = base64.b64decode(body).decode('utf-8', errors='ignore')
            else:
                body = body_data
            
            # Parse JSON
            opt_json = json.loads(body)
            rows = process_options_data(opt_json)
            
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=f"No options data found for {symbol} on {date}. The expiration date may be invalid."
                )
            
            print(f"[INFO] Successfully extracted {len(rows)} strikes")
            return rows
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parse error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse options data: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[ERROR] Data extraction error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")


# --- API Endpoints ---

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Barchart Options API",
        "version": "1.0.0",
        "endpoints": {
            "/options": "GET - JSON options data (params: symbol, date)",
            "/options/csv": "GET - CSV download (params: symbol, date)",
            "/health": "GET - Health check"
        },
        "example": "/options?symbol=AAPL&date=2026-01-17"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/options")
async def get_options_json(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL, $SPX, TSLA)"),
    date: str = Query(..., description="Expiration date (e.g., 2026-01-17)")
):
    """
    Get options data as JSON.
    
    - **symbol**: Stock symbol (AAPL, TSLA, $SPX, etc.)
    - **date**: Expiration date (2026-01-17 or 2026-01-10-w for weekly)
    
    Returns side-by-side Call/Put options data.
    """
    rows = await scrape_options(symbol, date)
    
    return {
        "success": True,
        "symbol": symbol,
        "date": date,
        "count": len(rows),
        "data": rows
    }


@app.get("/options/csv")
async def get_options_csv(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL, $SPX, TSLA)"),
    date: str = Query(..., description="Expiration date (e.g., 2026-01-17)")
):
    """
    Get options data as CSV file download.
    
    - **symbol**: Stock symbol (AAPL, TSLA, $SPX, etc.)
    - **date**: Expiration date (2026-01-17 or 2026-01-10-w for weekly)
    """
    rows = await scrape_options(symbol, date)
    
    df = pd.DataFrame(rows)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    filename = f"options_{symbol.replace('$', '')}_{date}.csv"
    
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
