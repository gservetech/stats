"""
Barchart Options API Server
API to scrape side-by-side options data from Barchart

Endpoints:
- /options         - Get options data (JSON) with symbol & date
- /options/csv     - Get options data (CSV) with symbol & date
- /weekly/summary  - Weekly PCR + GEX summary (requires spot)
- /weekly/gex      - Per-strike GEX table (requires spot)
- /health          - Health check
"""

import asyncio
import json
import base64
import re
import os
import sys
import math
from io import StringIO
from datetime import datetime
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions

app = FastAPI(
    title="Barchart Options API",
    description="API to scrape Barchart options data with symbol and date",
    version="1.0.2"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- JSON Sanitizer (fix NaN/Inf) ----------------

def sanitize_json(obj):
    """
    Convert NaN/Inf floats (not JSON compliant) into None recursively.
    This prevents: ValueError: Out of range float values are not JSON compliant
    """
    if obj is None:
        return None

    # Python float
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # Numpy floats/ints
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)

    # dict/list/tuple
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]

    return obj


# ---------------- Helper Functions ----------------

def _to_float(val, default=None):
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return float(val)
        except Exception:
            return default
    s = str(val).strip()
    if s in ("", "N/A", "na", "None"):
        return default
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ("", "-", "."):
        return default
    try:
        return float(s)
    except Exception:
        return default


def _to_int(val, default=0):
    f = _to_float(val, None)
    return int(round(f)) if f is not None else default


def _fmt_price(x):
    return f"{x:,.2f}" if x is not None else ""


def _fmt_int(x):
    return f"{int(x):,}" if x is not None else ""


def _fmt_iv(val):
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
        # Some APIs return 0.23 (23%), others return 23
        return f"{v * 100:.2f}%" if v <= 10 else f"{v:.2f}%"
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
        "raw_trade_time": raw.get("tradeTime", 0),
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
                    if s not in strike_items:
                        strike_items[s] = []
                    strike_items[s].append(item)
        else:
            # SBS View (Keys are already Strikes)
            strike_items = data

    elif isinstance(data, list):
        for item in data:
            s = item.get("strikePrice")
            if s not in strike_items:
                strike_items[s] = []
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
            "Strike": f"{strike_num:,.2f}" if strike_num else str(strike_str),
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


def build_chrome_options() -> ChromiumOptions:
    """
    Build ChromiumOptions safely for both:
    - Windows local development
    - Linux/Docker VPS
    """
    opts = ChromiumOptions()

    if sys.platform.startswith("linux"):
        opts.binary_location = os.getenv("CHROME_BINARY", "/usr/bin/google-chrome")
    elif sys.platform.startswith("win"):
        # Optional: uncomment if auto-detection fails
        # chrome1 = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        # chrome2 = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        # if os.path.exists(chrome1):
        #     opts.binary_location = chrome1
        # elif os.path.exists(chrome2):
        #     opts.binary_location = chrome2
        pass

    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--window-size=1920,1080")

    # Docker/Linux stability flags (fine on Windows too)
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    return opts


# ---------------- Scraper ----------------

async def scrape_options(symbol: str, date: str):
    """
    Scrape options data from Barchart for the given symbol and date.
    Uses network interception to capture API responses.
    """
    symbol_q = quote(symbol, safe="")
    url = f"https://www.barchart.com/stocks/quotes/{symbol_q}/options?expiration={date}&view=sbs"
    print(f"[INFO] Scraping: {url}")

    captured_requests = {}

    async def on_response(response_log):
        params = response_log.get("params", {})
        response = params.get("response", {})
        resp_url = response.get("url", "")

        if "/proxies/core-api/v1/options/get" in resp_url and "options" not in captured_requests:
            print(f"[INFO] Detected Options API call: {resp_url[:120]}...")
            captured_requests["options"] = (params.get("requestId"), resp_url)

        elif "/proxies/core-api/v1/options-expirations/get" in resp_url and "expirations" not in captured_requests:
            print(f"[INFO] Detected Expirations API call: {resp_url[:120]}...")
            captured_requests["expirations"] = (params.get("requestId"), resp_url)

    options = build_chrome_options()

    print("[INFO] Starting browser (headless mode)...")
    if getattr(options, "binary_location", None):
        print(f"[INFO] Chrome binary: {options.binary_location}")
    else:
        print("[INFO] Chrome binary: (auto-detect)")

    async with Chrome(options=options) as browser:
        tab = await browser.start()

        await tab.enable_network_events()
        await tab.on("Network.responseReceived", on_response)

        print(f"[INFO] Navigating to: {url}")
        try:
            await tab.go_to(url)
        except Exception as e:
            print(f"[ERROR] Navigation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load page: {str(e)}")

        print("[INFO] Waiting for network requests (max 30s)...")
        for i in range(30):
            await asyncio.sleep(1)
            if "options" in captured_requests:
                await asyncio.sleep(2)
                break
            if i in (5, 10, 15, 20, 25):
                print(f"[INFO] Waiting... {i}/30 seconds")

        if "options" not in captured_requests:
            print("[ERROR] Options API call not captured")
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Options data not found for {symbol} on {date}. "
                    f"Verify symbol + expiration date, or Barchart may be blocking headless requests."
                )
            )

        request_id, api_url = captured_requests["options"]
        print(f"[INFO] Processing options data from request {request_id}...")
        print(f"[INFO] API URL: {api_url[:140]}...")

        try:
            body_data = await tab.get_network_response_body(request_id)

            if isinstance(body_data, dict):
                body = body_data.get("body", "")
                if body_data.get("base64Encoded"):
                    body = base64.b64decode(body).decode("utf-8", errors="ignore")
            else:
                body = body_data

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


# ---------------- Gamma / GEX helpers ----------------

def _iv_to_decimal(iv_str_or_num):
    """
    Convert:
      "23.50%" -> 0.235
      "23.5"   -> 0.235  (assume percent if > 3)
      0.235    -> 0.235
    """
    if iv_str_or_num is None:
        return None
    s = str(iv_str_or_num).strip()
    if s == "" or s.lower() in ("na", "n/a", "none"):
        return None
    if s.endswith("%"):
        v = _to_float(s[:-1], None)
        return (v / 100.0) if v is not None else None
    v = _to_float(s, None)
    if v is None:
        return None
    return (v / 100.0) if v > 3 else v


def _years_to_expiry(date_yyyy_mm_dd: str) -> float:
    """Approx time to expiry in years (expiry assumed 16:00 local)."""
    now = datetime.now()
    exp = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(hour=16, minute=0, second=0, microsecond=0)
    dt_seconds = (exp - now).total_seconds()
    return max(dt_seconds, 0.0) / (365.0 * 24 * 3600.0)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for call and put)."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def _compute_weekly_gex(rows, spot: float, date: str, r: float = 0.05, multiplier: int = 100) -> pd.DataFrame:
    """
    Builds per-strike Gamma + GEX table from your /options output.
    GEX approximation:
        call_gex = gamma * call_OI * multiplier * S^2
        put_gex  = gamma * put_OI  * multiplier * S^2
        net_gex  = call_gex - put_gex
    """
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df

    df["strike"] = df["Strike"].apply(lambda x: _to_float(x, 0.0))
    df["call_oi"] = df["Call OI"].apply(lambda x: _to_int(x, 0))
    df["put_oi"] = df["Put OI"].apply(lambda x: _to_int(x, 0))
    df["call_vol"] = df["Call Volume"].apply(lambda x: _to_int(x, 0))
    df["put_vol"] = df["Put Volume"].apply(lambda x: _to_int(x, 0))

    df["call_iv_dec"] = df["Call IV"].apply(_iv_to_decimal)
    df["put_iv_dec"] = df["Put IV"].apply(_iv_to_decimal)

    T = _years_to_expiry(date)

    def pick_iv(row):
        return row["call_iv_dec"] if row["call_iv_dec"] is not None else row["put_iv_dec"]

    df["iv_used"] = df.apply(pick_iv, axis=1)

    df["gamma"] = df.apply(
        lambda row: _bs_gamma(spot, row["strike"], T, r, row["iv_used"]) if row["iv_used"] else float("nan"),
        axis=1
    )

    S2 = spot * spot
    gamma_safe = df["gamma"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["call_gex"] = gamma_safe * df["call_oi"] * multiplier * S2
    df["put_gex"]  = gamma_safe * df["put_oi"]  * multiplier * S2
    df["net_gex"]  = df["call_gex"] - df["put_gex"]

    out = df[[
        "strike",
        "Call IV", "Put IV", "iv_used",
        "gamma",
        "call_oi", "put_oi",
        "call_vol", "put_vol",
        "call_gex", "put_gex", "net_gex"
    ]].copy()

    return out


# ---------------- API Endpoints ----------------

@app.get("/")
async def root():
    return {
        "service": "Barchart Options API",
        "version": "1.0.2",
        "endpoints": {
            "/options": "GET - JSON options data (params: symbol, date)",
            "/options/csv": "GET - CSV download (params: symbol, date)",
            "/weekly/summary": "GET - PCR + GEX summary (params: symbol, date, spot)",
            "/weekly/gex": "GET - per-strike GEX (params: symbol, date, spot)",
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
    rows = await scrape_options(symbol, date)
    payload = {
        "success": True,
        "symbol": symbol,
        "date": date,
        "count": len(rows),
        "data": rows
    }
    return sanitize_json(payload)


@app.get("/options/csv")
async def get_options_csv(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL, $SPX, TSLA)"),
    date: str = Query(..., description="Expiration date (e.g., 2026-01-17)")
):
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


@app.get("/weekly/summary")
async def weekly_summary(
    symbol: str = Query(...),
    date: str = Query(..., description="Expiration date YYYY-MM-DD (example: 2026-01-16)"),
    spot: float = Query(..., description="Underlying spot price (required for gamma)"),
    r: float = Query(0.05, description="Risk-free rate (decimal), default 0.05"),
    multiplier: int = Query(100, description="Contract multiplier, default 100"),
):
    rows = await scrape_options(symbol, date)
    gex_df = _compute_weekly_gex(rows, spot=spot, date=date, r=r, multiplier=multiplier)
    if gex_df.empty:
        raise HTTPException(status_code=404, detail="No data returned for GEX computation.")

    total_call_oi = float(gex_df["call_oi"].sum())
    total_put_oi  = float(gex_df["put_oi"].sum())
    total_call_vol = float(gex_df["call_vol"].sum())
    total_put_vol  = float(gex_df["put_vol"].sum())

    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else None

    totals = {
        "call_oi": total_call_oi,
        "put_oi": total_put_oi,
        "call_volume": total_call_vol,
        "put_volume": total_put_vol,
        "call_gex": float(gex_df["call_gex"].sum()),
        "put_gex": float(gex_df["put_gex"].sum()),
        "net_gex": float(gex_df["net_gex"].sum()),
    }

    top_call = gex_df.sort_values("call_gex", ascending=False).head(5)[["strike", "call_gex", "call_oi", "call_vol"]]
    top_put  = gex_df.sort_values("put_gex",  ascending=False).head(5)[["strike", "put_gex", "put_oi", "put_vol"]]
    top_net  = gex_df.reindex(gex_df["net_gex"].abs().sort_values(ascending=False).index).head(5)[["strike", "net_gex"]]

    payload = {
        "success": True,
        "symbol": symbol,
        "date": date,
        "spot": spot,
        "pcr": {"oi": pcr_oi, "volume": pcr_vol},
        "totals": totals,
        "top_strikes": {
            "call_gex": top_call.to_dict(orient="records"),
            "put_gex": top_put.to_dict(orient="records"),
            "net_gex_abs": top_net.to_dict(orient="records")
        }
    }
    return sanitize_json(payload)


@app.get("/weekly/gex")
async def weekly_gex(
    symbol: str = Query(...),
    date: str = Query(...),
    spot: float = Query(...),
    r: float = Query(0.05),
    multiplier: int = Query(100),
):
    rows = await scrape_options(symbol, date)
    gex_df = _compute_weekly_gex(rows, spot=spot, date=date, r=r, multiplier=multiplier)
    if gex_df.empty:
        raise HTTPException(status_code=404, detail="No data returned for GEX computation.")

    payload = {
        "success": True,
        "symbol": symbol,
        "date": date,
        "spot": spot,
        "count": int(len(gex_df)),
        "data": gex_df.to_dict(orient="records")
    }
    return sanitize_json(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
