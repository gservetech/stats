import os
import time
import datetime as dt
import re
import asyncio
from urllib.parse import quote
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .api_client import safe_cache_data

# Crawl4AI imports for reliable scraping
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

_SPOT_SESSION = None

def _spot_session() -> requests.Session:
    global _SPOT_SESSION
    if _SPOT_SESSION is None:
        s = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SPOT_SESSION = s
    return _SPOT_SESSION

@safe_cache_data(ttl=900, show_spinner=False)
def fetch_cnbc_chart_data(symbol: str, time_range: str = "1D") -> dict | None:
    """
    Fetches chart data from CNBC GraphQL API.
    Time ranges: 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, ALL
    """
    url = "https://webql-redesign.cnbcfm.com/graphql"
    # Ensure symbol is uppercase
    symbol = symbol.upper().strip()
    
    # Map time range to CNBC format if needed (though they seem to match the display labels)
    # 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, ALL
    
    params = {
        "operationName": "getQuoteChartData",
        "variables": f'{{"symbol":"{symbol}","timeRange":"{time_range}"}}',
        "extensions": '{"persistedQuery":{"version":1,"sha256Hash":"9e1670c29a10707c417a1efd327d4b2b1d456b77f1426e7e84fb7d399416bb6b"}}'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"CNBC API Error: Status {resp.status_code}, Body: {resp.text[:200]}")
            return {}
        data = resp.json()
        if "errors" in data:
            print(f"CNBC GraphQL Errors: {data['errors']}")
            return {}
        
        # The key in the 'data' object usually matches the operation name, 
        # but in some redesign versions it is simply 'chartData'
        data_obj = data.get("data", {})
        return data_obj.get("getQuoteChartData") or data_obj.get("chartData") or {}
    except Exception as e:
        print(f"CNBC API Exception: {e}")
        return {}

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
CNBC_QUOTE_URL = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"

def _pick_first_series(obj):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] >= 1:
            return obj.iloc[:, 0]
        return pd.Series(dtype="float64")
    return obj

def _parse_float(val, default=None):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return default
    s = str(val).strip()
    if not s:
        return default
    s = s.replace(",", "").replace("%", "").replace("+", "").replace("(", "").replace(")", "")
    try:
        return float(s)
    except Exception:
        return default

def _parse_yahoo_number(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("N/A", "-", "—"):
        return None
    s = s.replace(",", "").replace("%", "")
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)([KMBT])?$", s, flags=re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suffix, 1.0)
    return num * mult

def _normalize_share_stat_label(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    # Remove trailing footnote markers like " 3" or " 4"
    t = re.sub(r"\s*\d+$", "", t).strip()
    return t

def _extract_paren_text(label: str) -> str | None:
    if not label:
        return None
    m = re.search(r"\(([^)]*)\)", label)
    return m.group(1).strip() if m else None

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

def get_finnhub_api_key() -> str | None:
    try:
        if "FINNHUB_API_KEY" in st.secrets:
            return str(st.secrets["FINNHUB_API_KEY"])
    except Exception:
        pass
    return os.getenv("FINNHUB_API_KEY")

def get_spot_from_cnbc(symbol: str) -> dict | None:
    """
    Fetches CNBC quote data via the quote JSON endpoint.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    try:
        response = _spot_session().get(CNBC_QUOTE_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        quote = _find_cnbc_quote(data)
        if not quote:
            return None

        price_raw = _first_present(quote, ["last", "lastPrice", "last_price", "price", "lastTrade", "lastSale", "regularMarketLast", "regularMarketPrice"])
        price = _parse_float(price_raw)
        if price is None:
            return None

        change_raw = _first_present(quote, ["change", "priceChange", "netChange", "changePoints", "lastChange"])
        pct_raw = _first_present(quote, ["change_pct", "changePct", "changePercent", "percentChange", "pctChange", "percent_change"])

        after_price_raw = _first_present(quote, ["alt_last", "altLast", "extendedHoursLast", "afterHoursLast", "afterHoursPrice"])
        after_change_raw = _first_present(quote, ["alt_change", "altChange", "extendedHoursChange", "afterHoursChange"])
        after_pct_raw = _first_present(quote, ["alt_change_pct", "altChangePct", "altChangePercent", "extendedHoursChangePct", "afterHoursChangePct"])

        after_hours_data = None
        after_price = _parse_float(after_price_raw)
        if after_price is not None:
            after_hours_data = {
                "price": after_price,
                "change": _parse_float(after_change_raw, 0.0),
                "percent_change": _parse_float(after_pct_raw, 0.0),
            }
        else:
            ext = (
                quote.get("ExtendedMktQuote")
                or quote.get("extendedMktQuote")
                or quote.get("ExtendedMarketQuote")
                or quote.get("extendedMarketQuote")
            )
            if isinstance(ext, dict):
                ext_last_raw = _first_present(ext, ["last", "lastPrice", "last_price", "price", "lastTrade", "lastSale"])
                ext_change_raw = _first_present(ext, ["change", "priceChange", "netChange", "changePoints", "lastChange"])
                ext_pct_raw = _first_present(ext, ["change_pct", "changePct", "changePercent", "percentChange", "pctChange", "percent_change"])
                ext_last = _parse_float(ext_last_raw)
                if ext_last is not None:
                    after_hours_data = {
                        "price": ext_last,
                        "change": _parse_float(ext_change_raw, 0.0),
                        "percent_change": _parse_float(ext_pct_raw, 0.0),
                        "session": ext.get("type"),
                        "time": ext.get("last_time") or ext.get("last_timedate"),
                    }

        return {
            "spot": price,
            "change": _parse_float(change_raw, 0.0),
            "percent_change": _parse_float(pct_raw, 0.0),
            "after_hours": after_hours_data if after_hours_data else None,
            "source": "CNBC"
        }
    except Exception:
        pass
    return None

def get_spot_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    
    # Try CNBC first as requested
    cnbc_data = get_spot_from_cnbc(symbol)
    if cnbc_data:
        return cnbc_data

    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/quote"
        params = {"symbol": symbol, "token": api_key}
        resp = _spot_session().get(url, params=params, timeout=10)
        resp.raise_for_status()
        quote = resp.json()
        if quote and quote.get('c') is not None and float(quote.get('c', 0)) > 0:
            timestamp = quote.get('t', 0)
            return {
                "spot": float(quote['c']),
                "open": float(quote.get('o', 0)),
                "high": float(quote.get('h', 0)),
                "low": float(quote.get('l', 0)),
                "prev_close": float(quote.get('pc', 0)),
                "change": float(quote.get('d') or 0),
                "percent_change": float(quote.get('dp') or 0),
                "timestamp": dt.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else None,
                "source": "Finnhub"
            }
    except Exception:
        pass
    return None

def get_stock_candles_from_finnhub(symbol: str, resolution: str = "D", from_ts: int = None, to_ts: int = None, return_raw: bool = False) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return {"s": "error", "error": "missing_symbol"} if return_raw else None
    api_key = get_finnhub_api_key()
    if not api_key: return {"s": "error", "error": "missing_api_key"} if return_raw else None
    if to_ts is None: to_ts = int(time.time())
    if from_ts is None: from_ts = to_ts - (30 * 24 * 60 * 60)
    try:
        url = f"{FINNHUB_BASE_URL}/stock/candle"
        params = {"symbol": symbol, "resolution": resolution, "from": from_ts, "to": to_ts, "token": api_key}
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if return_raw: return data
        if data and data.get("s") == "ok":
            return {
                "open": data.get("o", []), "high": data.get("h", []), "low": data.get("l", []),
                "close": data.get("c", []), "volume": data.get("v", []), "timestamps": data.get("t", []),
                "source": "Finnhub",
            }
    except Exception:
        pass
    return None

def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        col_map = {}
        for col in out.columns:
            for target in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
                if target not in col_map and any(str(x).strip().lower() == target.lower() for x in col):
                    col_map[target] = col
        new_df = pd.DataFrame(index=out.index)
        for target, tup in col_map.items():
            new_df[target] = out[tup]
        out = new_df

    if "Close" not in out.columns:
        for alt in ["close", "Adj Close", "adj close", "Adj_Close", "adjclose"]:
            if alt in out.columns:
                out["Close"] = out[alt]
                break

    if "Close" not in out.columns: return pd.DataFrame()
    
    # Ensure all columns are numeric and 1D
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            v = out[col]
            if isinstance(v, pd.DataFrame): v = v.iloc[:, 0] if v.shape[1] else pd.Series(dtype=float)
            out[col] = pd.to_numeric(v, errors='coerce')
    
    return out

@safe_cache_data(ttl=900, show_spinner=False)
def fetch_price_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    return _normalize_yf_df(raw)

@safe_cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw is None or raw.empty: return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        col_map = {}
        for col in df.columns:
            for target in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
                if target not in col_map and any(str(x).strip().lower() == target.lower() for x in col):
                    col_map[target] = col
        for target, tup in col_map.items():
            try: df[target] = df[tup]
            except Exception: pass
    df = df.reset_index()
    try: df.columns = [str(c).strip() for c in df.columns]
    except Exception: df.columns = [str(c) for c in df.columns]
    if "Close" not in df.columns:
        for candidate in ["Adj Close", "close", "adj close", "AdjClose", "adjclose"]:
            if candidate in df.columns:
                df["Close"] = df[candidate]
                break
    if "Close" not in df.columns:
        close_like = [c for c in df.columns if "close" in str(c).lower()]
        if close_like: df["Close"] = df[close_like[0]]
    if "Close" not in df.columns: return pd.DataFrame()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            v = df[col]
            if isinstance(v, pd.DataFrame): v = v.iloc[:, 0] if v.shape[1] else pd.Series(dtype=float)
            df[col] = pd.to_numeric(v, errors="coerce")
    return df

@safe_cache_data(ttl=900, show_spinner=False)
def fetch_finnhub_candles_df(symbol: str, lookback_days: int = 730, resolution: str = "D") -> tuple:
    symbol = (symbol or "").strip().upper()
    if not symbol: return pd.DataFrame(), None, "missing_symbol", None
    resolution = str(resolution).strip().upper()
    to_ts = int(time.time())
    attempts = [int(lookback_days)]
    if lookback_days > 365: attempts.append(365)
    if lookback_days > 180: attempts.append(180)
    for days in attempts:
        from_ts = to_ts - int(days * 24 * 60 * 60)
        raw = get_stock_candles_from_finnhub(symbol, resolution=resolution, from_ts=from_ts, to_ts=to_ts, return_raw=True)
        if not raw: continue
        status = raw.get("s") or "unknown"
        error = raw.get("message") or raw.get("error")
        if status != "ok": continue
        ts = raw.get("t") or []
        if not ts: continue
        df = pd.DataFrame({
            "Date": pd.to_datetime(ts, unit="s", errors="coerce"),
            "Open": raw.get("o", []), "High": raw.get("h", []), "Low": raw.get("l", []),
            "Close": raw.get("c", []), "Volume": raw.get("v", []),
        })
        df = df.dropna(subset=["Date", "Close"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
        if not df.empty: return df, days, status, error
    return pd.DataFrame(), None, "failed", None

def get_company_profile_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/profile2"
        params = {"symbol": symbol, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and data.get('name'): return data
    except Exception: pass
    return None

def get_basic_financials_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/metric"
        params = {"symbol": symbol, "metric": "all", "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and data.get('metric'): return data
    except Exception: pass
    return None

def get_recommendation_trends_from_finnhub(symbol: str) -> list | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/recommendation"
        params = {"symbol": symbol, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0: return data
    except Exception: pass
    return None

def get_price_target_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/price-target"
        params = {"symbol": symbol, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and data.get('targetMean'): return data
    except Exception: pass
    return None

def get_company_news_from_finnhub(symbol: str, from_date: str = None, to_date: str = None) -> list | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    if not to_date: to_date = dt.date.today().isoformat()
    if not from_date: from_date = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    try:
        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {"symbol": symbol, "from": from_date, "to": to_date, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and isinstance(data, list): return data[:10]
    except Exception: pass
    return None

def get_earnings_from_finnhub(symbol: str) -> list | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/earnings"
        params = {"symbol": symbol, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0: return data
    except Exception: pass
    return None

def get_support_resistance_from_finnhub(symbol: str, resolution: str = "D") -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/scan/support-resistance"
        params = {"symbol": symbol, "resolution": resolution, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and data.get('levels'): return data
    except Exception: pass
    return None

def get_technical_indicator_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/scan/technical-indicator"
        params = {"symbol": symbol, "resolution": "D", "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data: return data
    except Exception: pass
    return None

def get_stock_peers_from_finnhub(symbol: str) -> list | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/peers"
        params = {"symbol": symbol, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and isinstance(data, list): return data
    except Exception: pass
    return None

def get_insider_sentiment_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/insider-sentiment"
        from_date = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
        to_date = dt.date.today().strftime("%Y-%m-%d")
        params = {"symbol": symbol, "from": from_date, "to": to_date, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and data.get('data'): return data
    except Exception: pass
    return None

def get_upgrade_downgrade_from_finnhub(symbol: str) -> list | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/upgrade-downgrade"
        params = {"symbol": symbol, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data and isinstance(data, list): return data[:20]
    except Exception: pass
    return None

def get_social_sentiment_from_finnhub(symbol: str) -> dict | None:
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    api_key = get_finnhub_api_key()
    if not api_key: return None
    try:
        url = f"{FINNHUB_BASE_URL}/stock/social-sentiment"
        from_date = (dt.date.today() - dt.timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = dt.date.today().strftime("%Y-%m-%d")
        params = {"symbol": symbol, "from": from_date, "to": to_date, "token": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data: return data
    except Exception: pass
    return None

def _parse_yahoo_share_statistics_html(html: str, symbol: str, url: str, source: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    # Yahoo layouts vary; parse all tables to find share-stat labels reliably.
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = _normalize_share_stat_label(cells[0].get_text(" ", strip=True))
        value = cells[1].get_text(" ", strip=True)
        if label and label not in data:
            data[label] = value

    if not data:
        return {"success": False, "error": "No statistics tables found", "url": url}

    data_lower = {k.lower(): v for k, v in data.items()}
    avg_vol_10d = _parse_yahoo_number(data_lower.get("avg vol (10 day)"))
    float_shares = _parse_yahoo_number(data_lower.get("float"))

    short_shares = None
    short_shares_prior = None
    short_ratio = None
    short_shares_label = None
    short_shares_prior_label = None
    short_ratio_label = None

    for label, value in data.items():
        l = label.lower()
        if l.startswith("shares short") and "prior month" in l:
            short_shares_prior = _parse_yahoo_number(value)
            short_shares_prior_label = label
        elif l.startswith("shares short"):
            short_shares = _parse_yahoo_number(value)
            short_shares_label = label
        elif l.startswith("short ratio"):
            short_ratio = _parse_yahoo_number(value)
            short_ratio_label = label

    if all(v is None for v in [avg_vol_10d, float_shares, short_shares, short_shares_prior, short_ratio]):
        return {"success": False, "error": "Share Statistics labels not found", "url": url}

    return {
        "success": True,
        "symbol": symbol,
        "url": url,
        "raw": data,
        "avg_vol_10d": avg_vol_10d,
        "float_shares": float_shares,
        "short_shares": short_shares,
        "short_shares_prior": short_shares_prior,
        "short_ratio": short_ratio,
        "short_shares_label": short_shares_label,
        "short_shares_prior_label": short_shares_prior_label,
        "short_ratio_label": short_ratio_label,
        "short_shares_asof": _extract_paren_text(short_shares_label),
        "short_shares_prior_asof": _extract_paren_text(short_shares_prior_label),
        "short_ratio_asof": _extract_paren_text(short_ratio_label),
        "source": source,
    }


async def _fetch_yahoo_share_statistics_crawl4ai(symbol: str) -> dict:
    """
    Async function to fetch Yahoo Finance key-statistics using crawl4ai.
    Uses a headless browser for more reliable scraping of dynamic content.
    """
    safe_symbol = quote(symbol)
    url = f"https://finance.yahoo.com/quote/{safe_symbol}/key-statistics"

    try:
        # Configure browser for stealth mode
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        )

        # Wait for the Share Statistics content to render
        wait_condition = """() => {
            const text = document.body ? document.body.innerText : '';
            return text.includes('Share Statistics') && (
                text.includes('Avg Vol (10 day)') ||
                text.includes('Shares Short') ||
                text.includes('Short Ratio')
            );
        }"""

        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_for=f"js:{wait_condition}",
            page_timeout=45000,  # 45 seconds timeout
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)

            if not result.success:
                return {"success": False, "error": "Crawl4AI failed to fetch page", "url": url}

            html = getattr(result, "cleaned_html", None) or getattr(result, "html", None) or ""
            if not html:
                return {"success": False, "error": "Crawl4AI returned empty HTML", "url": url}

            return _parse_yahoo_share_statistics_html(html, symbol, url, "crawl4ai")

    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


@safe_cache_data(ttl=900, show_spinner=False)
def fetch_yahoo_share_statistics(symbol: str) -> dict:
    """
    Fetch Yahoo Finance key-statistics using crawl4ai for reliable scraping.
    No fallbacks: if crawl4ai is unavailable or fails, return the error.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {"success": False, "error": "missing_symbol"}

    if not CRAWL4AI_AVAILABLE:
        return {
            "success": False,
            "error": "crawl4ai_not_available",
            "detail": "crawl4ai is not installed or could not be imported",
        }

    # Playwright requires a Proactor event loop on Windows for subprocess support
    if os.name == "nt":
        try:
            policy = asyncio.get_event_loop_policy()
            if not isinstance(policy, asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass

    try:
        # Run the async function in a new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _fetch_yahoo_share_statistics_crawl4ai(symbol))
                    result = future.result(timeout=60)
            else:
                result = loop.run_until_complete(_fetch_yahoo_share_statistics_crawl4ai(symbol))
        except RuntimeError:
            # No event loop exists, create one
            result = asyncio.run(_fetch_yahoo_share_statistics_crawl4ai(symbol))

        return result
    except Exception as e:
        detail = str(e)
        if isinstance(e, NotImplementedError):
            return {
                "success": False,
                "error": "playwright_subprocess_not_supported",
                "detail": "Playwright needs a Proactor event loop on Windows (subprocess support).",
            }
        return {"success": False, "error": "crawl4ai_failed", "detail": detail}

def get_quote_peers_from_finnhub(peers: list) -> dict:
    if not peers: return {}
    api_key = get_finnhub_api_key()
    if not api_key: return {}
    result = {}
    for peer in peers[:5]:
        try:
            url = f"{FINNHUB_BASE_URL}/quote"
            params = {"symbol": peer, "token": api_key}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            if data and data.get('c'):
                result[peer] = {"price": data.get('c'), "change": data.get('d'), "percent_change": data.get('dp')}
        except Exception: pass
    return result

def fetch_all_finnhub_data(symbol: str) -> dict:
    result = {
        "symbol": symbol, "quote": None, "profile": None, "financials": None, "candles": None,
        "recommendations": None, "price_target": None, "news": None, "earnings": None,
        "support_resistance": None, "technical": None, "peers": None, "peers_quotes": None,
        "insider_sentiment": None, "upgrades_downgrades": None, "social_sentiment": None,
        "success": False, "errors": []
    }
    quote = get_spot_from_finnhub(symbol)
    if quote:
        result["quote"] = quote
        result["success"] = True
    else: result["errors"].append("Quote unavailable")
    profile = get_company_profile_from_finnhub(symbol)
    if profile: result["profile"] = profile
    financials = get_basic_financials_from_finnhub(symbol)
    if financials: result["financials"] = financials
    to_ts = int(time.time())
    from_ts = to_ts - (365 * 24 * 60 * 60)
    candles = get_stock_candles_from_finnhub(symbol, "D", from_ts, to_ts)
    if candles: result["candles"] = candles
    recommendations = get_recommendation_trends_from_finnhub(symbol)
    if recommendations: result["recommendations"] = recommendations
    price_target = get_price_target_from_finnhub(symbol)
    if price_target: result["price_target"] = price_target
    news = get_company_news_from_finnhub(symbol)
    if news: result["news"] = news
    earnings = get_earnings_from_finnhub(symbol)
    if earnings: result["earnings"] = earnings
    sr = get_support_resistance_from_finnhub(symbol)
    if sr: result["support_resistance"] = sr
    tech = get_technical_indicator_from_finnhub(symbol)
    if tech: result["technical"] = tech
    peers = get_stock_peers_from_finnhub(symbol)
    if peers:
        result["peers"] = peers
        result["peers_quotes"] = get_quote_peers_from_finnhub(peers)
    insider = get_insider_sentiment_from_finnhub(symbol)
    if insider: result["insider_sentiment"] = insider
    upgrades = get_upgrade_downgrade_from_finnhub(symbol)
    if upgrades: result["upgrades_downgrades"] = upgrades
    social = get_social_sentiment_from_finnhub(symbol)
    if social: result["social_sentiment"] = social
    return result
