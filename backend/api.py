"""
Barchart Options API Server
Deploy to VPS and call from Streamlit frontend
"""
import asyncio
import json
import base64
import re
from io import StringIO
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydoll.browser import Chrome

app = FastAPI(
    title="Barchart Options API",
    description="API to fetch options data from Barchart",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit domain
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
    """Process options JSON data and return rows."""
    data = opt_json.get("data", {})
    rows = []

    strike_items = {}
    if "Call" in data or "Put" in data:
        for t in ["Call", "Put"]:
            for item in data.get(t, []):
                s = item.get("strikePrice")
                if s not in strike_items: strike_items[s] = []
                strike_items[s].append(item)
    else:
        strike_items = data

    for strike_str, items in strike_items.items():
        call_obj = next((i for i in items if i.get("optionType") == "Call"), None)
        put_obj = next((i for i in items if i.get("optionType") == "Put"), None)
        
        c_data = _pick(call_obj)
        p_data = _pick(put_obj)
        strike_num = _to_float(strike_str, 0)
        
        row = {
            "Call Latest": c_data["Latest"], "Call Bid": c_data["Bid"], "Call Ask": c_data["Ask"], 
            "Call Change": c_data["Change"], "Call Volume": c_data["Volume"], 
            "Call Open Int": c_data["Open Int"], "Call IV": c_data["IV"], 
            "Call Last Trade": c_data["Last Trade"],
            "Strike": f"{strike_num:,.2f}" if strike_num else strike_str,
            "Put Latest": p_data["Latest"], "Put Bid": p_data["Bid"], "Put Ask": p_data["Ask"], 
            "Put Change": p_data["Change"], "Put Volume": p_data["Volume"], 
            "Put Open Int": p_data["Open Int"], "Put IV": p_data["IV"], 
            "Put Last Trade": p_data["Last Trade"]
        }
        rows.append(row)
    
    return rows

async def fetch_barchart_options(symbol: str, expiration: str):
    """Fetch options data from Barchart for given symbol and expiration."""
    # Construct the URL
    # Handle symbols like $SPX (need to encode the $)
    encoded_symbol = symbol if symbol.startswith("$") else symbol
    target_url = f"https://www.barchart.com/stocks/quotes/{encoded_symbol}/options?expiration={expiration}&view=sbs"
    
    captured_request = {}

    async def on_response(response_log):
        params = response_log.get("params", {})
        response = params.get("response", {})
        url = response.get("url", "")
        
        if "/proxies/core-api/v1/options/get" in url and "options" not in captured_request:
            captured_request["options"] = (params.get("requestId"), url)

    async with Chrome(options=["--headless=new", "--no-sandbox", "--disable-gpu"]) as browser:
        tab = await browser.start()
        
        await tab.enable_network_events()
        await tab.on("Network.responseReceived", on_response)
        
        try:
            await tab.go_to(target_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to navigate: {str(e)}")

        # Wait for API call
        for _ in range(30):
            await asyncio.sleep(1)
            if "options" in captured_request:
                break
        
        if "options" not in captured_request:
            raise HTTPException(status_code=404, detail="No options data found. Check symbol and expiration date.")
        
        request_id, _ = captured_request["options"]
        
        try:
            body_data = await tab.get_network_response_body(request_id)
            
            if isinstance(body_data, dict):
                body = body_data.get("body", "")
                if body_data.get("base64Encoded"):
                    body = base64.b64decode(body).decode('utf-8', errors='ignore')
            else:
                body = body_data
            
            opt_json = json.loads(body)
            return process_options_data(opt_json)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")


# --- API Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Barchart Options API",
        "version": "1.0.0",
        "endpoints": {
            "/options": "Get options data (JSON)",
            "/options/csv": "Get options data (CSV)",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check for VPS monitoring"""
    return {"status": "healthy"}

@app.get("/options")
async def get_options(
    symbol: str = Query(..., description="Stock symbol (e.g., $SPX, AAPL, TSLA)"),
    expiration: str = Query(..., description="Expiration date (e.g., 2026-01-07-w for weekly, 2026-01-17 for monthly)")
):
    """
    Fetch options data from Barchart.
    
    **Parameters:**
    - `symbol`: Stock symbol (e.g., $SPX, AAPL, TSLA)
    - `expiration`: Expiration date in format YYYY-MM-DD or YYYY-MM-DD-w (for weekly)
    
    **Returns:** JSON array of options data with Call/Put side-by-side
    
    **Example:** `/options?symbol=$SPX&expiration=2026-01-07-w`
    """
    rows = await fetch_barchart_options(symbol, expiration)
    
    if not rows:
        raise HTTPException(status_code=404, detail="No options data found")
    
    return JSONResponse(content={
        "success": True,
        "symbol": symbol,
        "expiration": expiration,
        "count": len(rows),
        "data": rows
    })

@app.get("/options/csv")
async def get_options_csv(
    symbol: str = Query(..., description="Stock symbol (e.g., $SPX, AAPL, TSLA)"),
    expiration: str = Query(..., description="Expiration date (e.g., 2026-01-07-w)")
):
    """
    Fetch options data from Barchart as CSV.
    
    **Returns:** CSV file download
    """
    rows = await fetch_barchart_options(symbol, expiration)
    
    if not rows:
        raise HTTPException(status_code=404, detail="No options data found")
    
    df = pd.DataFrame(rows)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    filename = f"options_{symbol.replace('$', '')}_{expiration}.csv"
    
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
