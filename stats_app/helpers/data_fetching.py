import os
import time
import datetime as dt
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from .api_client import safe_cache_data

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

def _pick_first_series(obj):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] >= 1:
            return obj.iloc[:, 0]
        return pd.Series(dtype="float64")
    return obj

def get_finnhub_api_key() -> str | None:
    try:
        if "FINNHUB_API_KEY" in st.secrets:
            return str(st.secrets["FINNHUB_API_KEY"])
    except Exception:
        pass
    return os.getenv("FINNHUB_API_KEY")

def get_spot_from_cnbc(symbol: str) -> dict | None:
    """
    Scrapes CNBC for live quote data.
    Ported from user-provided scraping sample.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol: return None
    
    url = f"https://www.cnbc.com/quotes/{symbol}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Target the main QuoteStrip container
        strip = soup.find(id="quote-page-strip")
        if not strip:
            strip = soup.find(class_="QuoteStrip-container")
            
        if not strip:
            return None

        def extract_price_group(container):
            if not container: return None
            data = {}
            price_tag = container.find(class_="QuoteStrip-lastPrice")
            if not price_tag: return None
            
            data['price'] = float(price_tag.get_text(strip=True).replace(",", ""))
            
            change_tag = (container.find(class_="QuoteStrip-changeUp") or 
                          container.find(class_="QuoteStrip-changeDown") or 
                          container.find(class_="QuoteStrip-unchanged"))
            
            if change_tag:
                parts = [s.get_text(strip=True) for s in change_tag.find_all("span") if s.get_text(strip=True)]
                try: 
                    data['change'] = float(parts[0].replace(",", "").replace("+", ""))
                except: data['change'] = 0.0
                try: 
                    data['percent_change'] = float(parts[1].replace(",", "").replace("+", "").replace("%", "").replace("(", "").replace(")", ""))
                except: data['percent_change'] = 0.0
            else:
                data['change'] = 0.0
                data['percent_change'] = 0.0

            return data

        # Try regular market data first
        reg_market = strip.find(class_="QuoteStrip-extendedHours")
        market_data = extract_price_group(reg_market)
        
        # If not found, try the main price tag directly in strip (sometimes it's there)
        if not market_data:
            market_data = extract_price_group(strip)

        if not market_data:
            return None

        # After Hours Data (optional)
        after_hours_div = strip.find(class_="QuoteStrip-extendedDataContainer")
        after_hours_data = extract_price_group(after_hours_div)

        return {
            "spot": market_data['price'],
            "change": market_data['change'],
            "percent_change": market_data['percent_change'],
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
        resp = requests.get(url, params=params, timeout=10)
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
