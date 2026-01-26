"""
Barchart Options API Server
API to scrape side-by-side options data from Barchart

Endpoints:
- /options         - Get options data (JSON) with symbol & date
- /options/csv     - Get options data (CSV) with symbol & date
- /weekly/summary  - Weekly PCR + GEX summary (requires spot)
- /weekly/gex      - Per-strike GEX table (requires spot)
- /health          - Health check

✅ Cloud-ready improvements included:
- CORS enabled (localhost + your Streamlit domain + optional wildcard fallback)
- Headless Chrome flags improved for Linux cloud environments
- Simple in-memory cache to avoid double-scraping (reduces bans & speeds up)
- Health endpoint includes platform + chrome info
- Windows-safe unique Chrome profile dir to avoid Crashpad locks
- Avoids adding duplicate pydoll default flags (prevents ArgumentAlreadyExistsInOptions)
"""

import asyncio
import json
import base64
import re
import os
import sys
import math
from io import StringIO
from time import time
from datetime import datetime
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions

app = FastAPI(
    title="Barchart Options API",
    description="API to scrape Barchart options data with symbol and date",
    version="1.0.4"
)

# ---------------- CORS ----------------
# Tip: keep allow_origins=["*"] while testing, then lock to your domains.
ALLOW_ALL_CORS = os.getenv("ALLOW_ALL_CORS", "1") == "1"

allowed_origins = [
    "http://localhost:8501",
    "http://localhost:3000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
    "https://gservetech.streamlit.app",
]

if ALLOW_ALL_CORS:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Cache (avoid double-scraping) ----------------
_OPTIONS_CACHE = {}  # (symbol,date) -> {"ts": float, "rows": list}
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 min default (was 5 min)
CACHE_STALE_SECONDS = int(os.getenv("CACHE_STALE_SECONDS", "3600"))  # serve stale up to 1 hour
# Spot cache: short TTL for "real-time" with stale fallback for stability.
_SPOT_CACHE = {}  # symbol -> {"ts": float, "data": dict}
SPOT_TTL_SECONDS = int(os.getenv("SPOT_TTL_SECONDS", "5"))
SPOT_STALE_SECONDS = int(os.getenv("SPOT_STALE_SECONDS", "60"))
CNBC_QUOTE_URL = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
_BROWSER_CONCURRENCY = int(os.getenv("BROWSER_CONCURRENCY", "2")) # Increased to 2 for production
_BROWSER_SEMAPHORE = asyncio.Semaphore(_BROWSER_CONCURRENCY)
_BROWSER_ACQUIRE_TIMEOUT = 100 # Max wait for a browser slot before erroring
_EXPIRY_TZ = None
if ZoneInfo:
    try:
        _EXPIRY_TZ = ZoneInfo(os.getenv("EXPIRY_TZ", "America/New_York"))
    except ZoneInfoNotFoundError:
        _EXPIRY_TZ = None

# Request deduplication: prevent multiple identical scrapes
_PENDING_OPTIONS = {}  # (symbol,date) -> asyncio.Event (signals when scrape completes)
_PENDING_OPTIONS_LOCK = asyncio.Lock()
_PENDING_SPOT = {}  # symbol -> asyncio.Event (signals when spot fetch completes)
_PENDING_SPOT_LOCK = asyncio.Lock()


async def get_rows_cached(symbol: str, date: str):
    """
    Fetch options with:
    1. Cache hit -> return immediately
    2. Stale cache + busy browser -> return stale data (fast)
    3. Request deduplication -> wait for in-flight scrape instead of starting new one
    4. Otherwise -> scrape fresh data
    """
    key = (symbol.upper().strip(), date.strip())
    now = time()

    # 1. Fresh cache hit
    hit = _OPTIONS_CACHE.get(key)
    if hit and (now - hit["ts"]) < CACHE_TTL_SECONDS:
        print(f"[CACHE] Fresh cache hit for {key}")
        return hit["rows"]

    # 2. Stale cache available + browser busy -> return stale immediately
    if hit and (now - hit["ts"]) < CACHE_STALE_SECONDS and _BROWSER_SEMAPHORE.locked():
        print(f"[CACHE] Returning stale data for {key} (browser busy)")
        return hit["rows"]

    # 3. Check if another request is already scraping this key
    async with _PENDING_OPTIONS_LOCK:
        if key in _PENDING_OPTIONS:
            # Wait for the other request to finish
            event = _PENDING_OPTIONS[key]
            print(f"[DEDUP] Waiting for in-flight scrape of {key}")
    
    # If there's an in-flight request, wait for it
    if key in _PENDING_OPTIONS:
        try:
            await asyncio.wait_for(_PENDING_OPTIONS[key].wait(), timeout=120)
        except asyncio.TimeoutError:
            pass
        # Check cache again after waiting
        hit = _OPTIONS_CACHE.get(key)
        if hit:
            print(f"[DEDUP] Got result from parallel scrape for {key}")
            return hit["rows"]

    # 4. Start a new scrape (register as pending first)
    event = asyncio.Event()
    async with _PENDING_OPTIONS_LOCK:
        _PENDING_OPTIONS[key] = event

    try:
        # Wait for the result with a timeout slightly shorter than the frontend timeout
        async with asyncio.timeout(110): 
            rows = await scrape_options(symbol, date)
            _OPTIONS_CACHE[key] = {"ts": time(), "rows": rows}
            return rows
    except asyncio.TimeoutError:
        print(f"[TIMEOUT] Scrape for {key} took too long.")
        raise HTTPException(status_code=504, detail="Scrape timed out. Please try again.")
    finally:
        # Signal other waiters that we're done
        event.set()
        async with _PENDING_OPTIONS_LOCK:
            _PENDING_OPTIONS.pop(key, None)


async def get_spot_cached(symbol: str, date: str | None):
    """
    Fetch spot with a short TTL cache + stale fallback.
    This reduces intermittent failures and rate limiting while keeping data fresh.
    """
    key = (symbol or "").strip().upper()
    now = time()

    hit = _SPOT_CACHE.get(key)
    if hit and (now - hit["ts"]) < SPOT_TTL_SECONDS:
        data = dict(hit["data"])
        data["cached"] = True
        data["stale"] = False
        return data

    # If another request is already fetching this symbol, wait briefly.
    async with _PENDING_SPOT_LOCK:
        if key in _PENDING_SPOT:
            event = _PENDING_SPOT[key]
        else:
            event = None

    if event is not None:
        try:
            await asyncio.wait_for(event.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass
        hit = _SPOT_CACHE.get(key)
        if hit:
            data = dict(hit["data"])
            data["cached"] = True
            data["stale"] = (now - hit["ts"]) >= SPOT_TTL_SECONDS
            return data

    # Register in-flight fetch
    event = asyncio.Event()
    async with _PENDING_SPOT_LOCK:
        _PENDING_SPOT[key] = event

    try:
        data = await scrape_spot(symbol, date)
        _SPOT_CACHE[key] = {"ts": time(), "data": data}
        data = dict(data)
        data["cached"] = False
        data["stale"] = False
        return data
    except HTTPException:
        # Serve stale value if we have one
        hit = _SPOT_CACHE.get(key)
        if hit and (now - hit["ts"]) < SPOT_STALE_SECONDS:
            data = dict(hit["data"])
            data["cached"] = True
            data["stale"] = True
            data["stale_age_seconds"] = int(now - hit["ts"])
            return data
        raise
    finally:
        event.set()
        async with _PENDING_SPOT_LOCK:
            _PENDING_SPOT.pop(key, None)


# ---------------- JSON Sanitizer (fix NaN/Inf) ----------------
def sanitize_json(obj):
    """Convert NaN/Inf floats into None recursively (JSON-safe)."""
    if obj is None:
        return None

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    if isinstance(obj, (np.integer,)):
        return int(obj)

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
    s = str(val).strip().lower()
    if s in ("", "n/a", "na", "none"):
        return default
    
    # Handle multipliers (k/m) from display strings
    multiplier = 1.0
    if s.endswith("k"):
        multiplier = 1000.0
        s = s[:-1]
    elif s.endswith("m"):
        multiplier = 1000000.0
        s = s[:-1]
    
    # Remove commas and other non-numeric chars
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ("", "-", "."):
        return default
    try:
        return float(s) * multiplier
    except Exception:
        return default


def _first_present(d: dict, keys: list[str]):
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return None


def _find_cnbc_quote(obj):
    if isinstance(obj, dict):
        for key in ("QuickQuoteResult", "FormattedQuoteResult", "ExtendedQuoteResult"):
            if key in obj:
                found = _find_cnbc_quote(obj.get(key))
                if found:
                    return found
        if any(k in obj for k in ("last", "lastPrice", "last_price", "price", "lastTrade", "lastSale")):
            return obj
        for val in obj.values():
            found = _find_cnbc_quote(val)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_cnbc_quote(item)
            if found:
                return found
    return None


def _to_iso_time(val):
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        num = _to_float(s, None)
        if num is None:
            return s
        ts = num
    else:
        ts = _to_float(val, None)
        if ts is None:
            return None
    if ts > 1e10:
        ts = ts / 1000.0
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except Exception:
        return None


def _format_percent_text(val):
    if val is None:
        return None
    if isinstance(val, str) and "%" in val:
        return val.strip()
    num = _to_float(val, None)
    if num is None:
        return str(val)
    return f"{num:.2f}%"


def _to_int(val, default: int | float | str | None = 0) -> int:
    f = _to_float(val, None)
    if f is not None:
        return int(round(f))
    d = _to_float(default, None)
    return int(round(d)) if d is not None else 0


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
        return f"{v * 100:.2f}%" if v <= 10 else f"{v:.2f}%"
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
        "raw_trade_time": raw.get("tradeTime", 0),
    }


def process_options_data(opt_json):
    data = opt_json.get("data", {})
    rows = []
    strike_items = {}

    if isinstance(data, dict):
        if "Call" in data or "Put" in data:
            for t in ["Call", "Put"]:
                for item in data.get(t, []):
                    s = item.get("strikePrice")
                    strike_items.setdefault(s, []).append(item)
        else:
            strike_items = data

    elif isinstance(data, list):
        for item in data:
            s = item.get("strikePrice")
            strike_items.setdefault(s, []).append(item)

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
    IMPORTANT: pydoll adds some default flags (including --no-first-run).
    If you add them again, pydoll throws ArgumentAlreadyExistsInOptions.
    """
    opts = ChromiumOptions()

    # ---- pick Chrome binary ----
    if sys.platform.startswith("linux"):
        opts.binary_location = os.getenv("CHROME_BINARY", "/usr/bin/google-chrome")

    elif sys.platform.startswith("win"):
        env_bin = os.getenv("CHROME_BINARY")
        if env_bin and os.path.exists(env_bin):
            opts.binary_location = env_bin
        else:
            candidates = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                str(Path(os.getenv("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe"),
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    opts.binary_location = c
                    break

    # ---- unique profile dir (prevents Crashpad locks on Windows) ----
    user_data_dir = Path(tempfile.gettempdir()) / f"pydoll_profile_{os.getpid()}_{int(time())}"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # ---- args ----
    opts.add_argument("--headless=new")
    # speed & stability
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-setuid-sandbox")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--proxy-server='direct://'")
    opts.add_argument("--proxy-bypass-list=*")
    opts.add_argument("--blink-settings=imagesEnabled=false") # don't load images

    # DO NOT force remote debugging port (pydoll handles this)

    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    return opts


# ---------------- Scraper ----------------
async def scrape_options(symbol: str, date: str):
    symbol_q = quote(symbol, safe="")
    url = f"https://www.barchart.com/stocks/quotes/{symbol_q}/options?expiration={date}&view=sbs"
    print(f"[INFO] Scraping: {url}")

    captured_requests = {}

    async def on_response(response_log):
        params = response_log.get("params", {})
        response = params.get("response", {})
        resp_url = response.get("url", "")

        if "/proxies/core-api/v1/options/get" in resp_url and "options" not in captured_requests:
            captured_requests["options"] = (params.get("requestId"), resp_url)

        elif "/proxies/core-api/v1/options-expirations/get" in resp_url and "expirations" not in captured_requests:
            captured_requests["expirations"] = (params.get("requestId"), resp_url)

    async with _BROWSER_SEMAPHORE:
        options = build_chrome_options()

        print("[INFO] Starting browser (headless mode)...")
        print("[INFO] Chrome binary:", getattr(options, "binary_location", None) or "(auto-detect)")

        async with Chrome(options=options) as browser:
            tab = await browser.start()

            await tab.enable_network_events()
            await tab.on("Network.responseReceived", on_response)

            try:
                # Add a timeout to the initial page load to avoid hanging the semaphore
                await asyncio.wait_for(tab.go_to(url), timeout=45)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Barchart took too long to respond.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load page: {str(e)}")

            print("[INFO] Waiting for network requests (max 30s)...")
            # Poll much faster (every 0.2s) instead of every 1s
            for _ in range(150): 
                await asyncio.sleep(0.2)
                if "options" in captured_requests:
                    # Once request is seen, wait just a tiny bit for the body to be populateable
                    await asyncio.sleep(0.5) 
                    break

            if "options" not in captured_requests:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Options data not found for {symbol} on {date}. "
                        f"Verify symbol + expiration date, or Barchart may be blocking headless requests."
                    )
                )

            request_id, _api_url = captured_requests["options"]

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
                    raise HTTPException(status_code=404, detail=f"No options data found for {symbol} on {date}.")

                return rows

            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse options data: {str(e)}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")


async def _query_text_any(tab, selectors: list[str]) -> str | None:
    for selector in selectors:
        try:
            el = await tab.query(selector, timeout=0, raise_exc=False)
        except Exception:
            el = None
        if el:
            try:
                text = (await el.text).strip()
            except Exception:
                text = ""
            if text:
                return text
    return None


async def scrape_spot(symbol: str, date: str | None = None):
    """
    Fetches CNBC quote data via the quote JSON endpoint.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    
    params = {
        "symbols": symbol,
        "requestMethod": "itv",
        "noform": 1,
        "partnerId": 2,
        "fund": 1,
        "exthrs": 1,
        "output": "json",
        "events": 1,
    }
    url = f"{CNBC_QUOTE_URL}?{urlencode(params)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    try:
        # Run requests in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(CNBC_QUOTE_URL, params=params, headers=headers, timeout=15))
        response.raise_for_status()
        data = response.json()
        quote = _find_cnbc_quote(data)
        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote data not found for {symbol}")

        price_raw = _first_present(quote, ["last", "lastPrice", "last_price", "price", "lastTrade", "lastSale", "regularMarketLast", "regularMarketPrice"])
        price = _to_float(price_raw, None)
        if price is None:
            raise HTTPException(status_code=404, detail=f"Price data not found for {symbol}")

        change_raw = _first_present(quote, ["change", "priceChange", "netChange", "changePoints", "lastChange"])
        pct_raw = _first_present(quote, ["change_pct", "changePct", "changePercent", "percentChange", "pctChange", "percent_change"])

        after_price_raw = _first_present(quote, ["alt_last", "altLast", "extendedHoursLast", "afterHoursLast", "afterHoursPrice"])
        after_change_raw = _first_present(quote, ["alt_change", "altChange", "extendedHoursChange", "afterHoursChange"])
        after_pct_raw = _first_present(quote, ["alt_change_pct", "altChangePct", "altChangePercent", "extendedHoursChangePct", "afterHoursChangePct"])

        after_price = _to_float(after_price_raw, None)
        after_hours_data = None
        if after_price is not None:
            after_hours_data = {
                "price": after_price,
                "change": _to_float(after_change_raw, 0.0),
                "percent_change": _to_float(after_pct_raw, 0.0),
            }

        trade_time = _first_present(quote, ["last_time", "last_time_msec", "lastTime", "lastTimeMsec", "lastTradeTime", "tradeTime"])
        exchange = _first_present(quote, ["exchange", "exchangeName", "exchange_name"])
        session = _first_present(quote, ["session", "sessionStatus", "marketStatus", "market_status"])
        bid_raw = _first_present(quote, ["bid", "bidPrice", "bid_price", "bestBid"])
        ask_raw = _first_present(quote, ["ask", "askPrice", "ask_price", "bestAsk"])

        return {
            "url": url,
            "spot": price,
            "spot_text": str(price_raw) if price_raw is not None else None,
            "change": _to_float(change_raw, 0.0),
            "change_text": str(change_raw) if change_raw is not None else None,
            "percent_change": _to_float(pct_raw, 0.0),
            "percent_text": _format_percent_text(pct_raw),
            "after_hours": after_hours_data,
            "trade_time": _to_iso_time(trade_time),
            "exchange": exchange,
            "session": session,
            "bid_text": str(bid_raw) if bid_raw is not None else None,
            "ask_text": str(ask_raw) if ask_raw is not None else None,
            "source": "CNBC"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping error: {str(e)}")


# ---------------- Gamma / GEX helpers ----------------
def _iv_to_decimal(iv_str_or_num):
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
    if _EXPIRY_TZ:
        now = datetime.now(_EXPIRY_TZ)
        exp = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=_EXPIRY_TZ,
        )
        dt_seconds = (exp - now).total_seconds()
    else:
        now = datetime.now()
        exp = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
        )
        dt_seconds = (exp - now).total_seconds()
    return max(dt_seconds, 0.0) / (365.0 * 24 * 3600.0)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Standard Black-Scholes Gamma: e^(-qT) * N'(d1) / (S * sigma * sqrt(T))
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    
    # d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return (math.exp(-q * T) * _norm_pdf(d1)) / (S * sigma * math.sqrt(T))


def _compute_weekly_gex(rows, spot: float, date: str, r: float = 0.05, q: float = 0.0, multiplier: int = 100) -> pd.DataFrame:
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

    # Calculate gamma separately for calls and puts to use their specific IVs (skew)
    def calc_gamma(iv):
        if iv is None or iv <= 0:
            return 0.0
        return _bs_gamma(spot, row["strike"], T, r, q, iv)

    gammas_call = []
    gammas_put = []
    
    for _, row in df.iterrows():
        civ = row["call_iv_dec"]
        piv = row["put_iv_dec"]
        
        # If one IV is missing, use the other as fallback
        g_call = _bs_gamma(spot, row["strike"], T, r, q, civ) if civ else 0.0
        g_put = _bs_gamma(spot, row["strike"], T, r, q, piv) if piv else 0.0
        
        if not g_call and g_put: g_call = g_put
        if not g_put and g_call: g_put = g_call
        
        gammas_call.append(g_call)
        gammas_put.append(g_put)

    df["gamma_call"] = gammas_call
    df["gamma_put"] = gammas_put

    # GEX = 0.01 * S^2 * Gamma * OI * Multiplier 
    # (Representing dollar delta change for a 1% spot move)
    S2 = spot * spot
    df["call_gex"] = 0.01 * df["gamma_call"] * df["call_oi"] * multiplier * S2
    df["put_gex"] = 0.01 * df["gamma_put"] * df["put_oi"] * multiplier * S2
    
    # Dealer Net GEX (Standard convention)
    # Positive = Call Dominant (Supportive/Mean-Reverting)
    # Negative = Put Dominant (Whipsaw/Acceleration)
    df["net_gex"] = df["call_gex"] - df["put_gex"]

    return df[[
        "strike",
        "Call IV", "Put IV", "gamma_call", "gamma_put",
        "call_oi", "put_oi",
        "call_vol", "put_vol",
        "call_gex", "put_gex", "net_gex"
    ]].copy()


# ---------------- API Endpoints ----------------
@app.get("/")
async def root():
    return {
        "service": "Barchart Options API",
        "version": "1.0.4",
        "endpoints": {
            "/options": "GET - JSON options data (params: symbol, date)",
            "/options/csv": "GET - CSV download (params: symbol, date)",
            "/weekly/summary": "GET - PCR + GEX summary (params: symbol, date, spot)",
            "/weekly/gex": "GET - per-strike GEX (params: symbol, date, spot)",
            "/spot": "GET - spot quote (params: symbol, date optional)",
            "/health": "GET - Health check"
        },
        "example": "/options?symbol=AAPL&date=2026-01-16"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "platform": sys.platform,
        "chrome_binary": os.getenv("CHROME_BINARY", ""),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "browser_concurrency": _BROWSER_CONCURRENCY,
    }


@app.get("/spot")
async def get_spot(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL, $SPX, TSLA)"),
    date: str | None = Query(None, description="Optional expiration date (YYYY-MM-DD)"),
):
    data = await get_spot_cached(symbol, date)
    payload = {
        "success": True,
        "symbol": symbol,
        "date": date,
        "spot": data.get("spot"),
        "spot_text": data.get("spot_text"),
        "change": data.get("change"),
        "change_text": data.get("change_text"),
        "percent_change": data.get("percent_change"),
        "percent_text": data.get("percent_text"),
        "trade_time": data.get("trade_time"),
        "exchange": data.get("exchange"),
        "session": data.get("session"),
        "bid_text": data.get("bid_text"),
        "ask_text": data.get("ask_text"),
        "url": data.get("url"),
        "source": data.get("source"),
        "cached": data.get("cached"),
        "stale": data.get("stale"),
        "stale_age_seconds": data.get("stale_age_seconds"),
        "fetched_at": datetime.now().isoformat(),
    }
    return sanitize_json(payload)


@app.get("/options")
async def get_options_json(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL, $SPX, TSLA)"),
    date: str = Query(..., description="Expiration date (e.g., 2026-01-16)")
):
    rows = await get_rows_cached(symbol, date)
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
    date: str = Query(..., description="Expiration date (e.g., 2026-01-16)")
):
    rows = await get_rows_cached(symbol, date)
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
    q: float = Query(0.0, description="Dividend yield (decimal), default 0.0"),
    multiplier: int = Query(100, description="Contract multiplier, default 100"),
):
    rows = await get_rows_cached(symbol, date)
    gex_df = _compute_weekly_gex(rows, spot=spot, date=date, r=r, q=q, multiplier=multiplier)
    if gex_df.empty:
        raise HTTPException(status_code=404, detail="No data returned for GEX computation.")

    total_call_oi = float(gex_df["call_oi"].sum())
    total_put_oi = float(gex_df["put_oi"].sum())
    total_call_vol = float(gex_df["call_vol"].sum())
    total_put_vol = float(gex_df["put_vol"].sum())

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
    top_put = gex_df.sort_values("put_gex", ascending=False).head(5)[["strike", "put_gex", "put_oi", "put_vol"]]
    net_abs = gex_df["net_gex"].abs()
    top_net_idx = net_abs.nlargest(5).index
    top_net = gex_df.loc[top_net_idx, ["strike", "net_gex"]].copy()
    top_net["net_gex_abs"] = net_abs.loc[top_net_idx].values
    top_net = top_net[["strike", "net_gex_abs", "net_gex"]]

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
    q: float = Query(0.0),
    multiplier: int = Query(100),
):
    rows = await get_rows_cached(symbol, date)
    gex_df = _compute_weekly_gex(rows, spot=spot, date=date, r=r, q=q, multiplier=multiplier)
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
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
