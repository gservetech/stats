
def _pick_first_series(obj):
    """If obj is a DataFrame (e.g., duplicate columns), return its first column as a Series."""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] >= 1:
            return obj.iloc[:, 0]
        return pd.Series(dtype="float64")
    return obj


def get_close_series(hist):
    """
    Return a 1-D numeric Close Series from many possible yfinance shapes.
    """
    import pandas as pd
    if hist is None:
        return pd.Series(dtype="float64")

    # Series input
    if isinstance(hist, pd.Series):
        s = pd.to_numeric(hist, errors="coerce")
        return s.dropna()

    h = hist.copy()
    if getattr(h, "empty", True):
        return pd.Series(dtype="float64")

    # Normalize column labels
    try:
        if not isinstance(h.columns, pd.MultiIndex):
            h.columns = [str(c).strip() for c in h.columns]
    except Exception:
        pass

    # Direct Close
    if "Close" in getattr(h, "columns", []):
        s = _pick_first_series(h["Close"])
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s

    # Variants
    for cand in ["Adj Close", "adj close", "close", "AdjClose", "adjclose"]:
        if cand in getattr(h, "columns", []):
            s = _pick_first_series(h[cand])
            s = pd.to_numeric(s, errors="coerce").dropna()
            return s

    # MultiIndex
    try:
        if isinstance(h.columns, pd.MultiIndex):
            close_cols = [c for c in h.columns if any("close" in str(level).lower() for level in c)]
            if close_cols:
                s = _pick_first_series(h[close_cols[0]])
                s = pd.to_numeric(s, errors="coerce").dropna()
                return s
    except Exception:
        pass

    # Last resort: any column containing 'close'
    try:
        close_like = [c for c in h.columns if "close" in str(c).lower()]
        if close_like:
            s = _pick_first_series(h[close_like[0]])
            s = pd.to_numeric(s, errors="coerce").dropna()
            return s
    except Exception:
        pass

    return pd.Series(dtype="float64")


# -*- coding: utf-8 -*-
"""
Barchart Options Dashboard - Streamlit Frontend
Connects to FastAPI backend for options data scraping + Weekly Gamma/GEX summary + Gamma Map + Noise Filters.

✅ Works BOTH:
- Local dev (defaults to http://localhost:8000)
- Streamlit Cloud (uses st.secrets["API_BASE_URL"] or env var API_BASE_URL)

How to set for Streamlit Cloud:
App → Settings → Secrets
API_BASE_URL = "https://api.kdsinsured.com"
"""

import os
import re
import math
import datetime as dt
import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Page configuration (MUST be the first Streamlit call)
# -----------------------------
try:
    st.set_page_config(
        page_title="Stats Dashboard",
        page_icon="📊",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    # Avoid crashing if another Streamlit command ran before config (e.g., reloads).
    pass


# ---------------- Streamlit cache wrapper (avoids TokenError on some Streamlit/Python combos) ----------------
def safe_cache_data(*dargs, **dkwargs):
    """A defensive wrapper around st.cache_data.

    On some environments (notably Streamlit Cloud with newer Python),
    Streamlit may fail to introspect source code for caching and raise
    tokenize.TokenError/inspect errors at import-time. If that happens,
    we gracefully disable caching instead of crashing the app.
    """

    def _decorator(func):
        try:
            return st.cache_data(*dargs, **dkwargs)(func)
        except Exception:
            return func

    return _decorator


# ---------------- Yahoo Finance Spot Helper ----------------
# Uses Yahoo Finance quote endpoint (via yfinance if installed, else direct HTTP).
try:
    import yfinance as yf  # optional
except Exception:
    yf = None


def get_spot_from_yahoo(symbol: str) -> float | None:
    """
    Fetch latest spot price for a symbol from Yahoo Finance.
    Returns None if unavailable.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None


def get_today_open_from_yahoo(symbol: str) -> float | None:
    """
    Fetch today's regular session OPEN from Yahoo.
    Returns None if unavailable.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None

    # yfinance path
    if yf is not None:
        try:
            t = yf.Ticker(symbol)
            # fast_info may contain open
            try:
                o = t.fast_info.get("open")
                if o is not None and float(o) > 0:
                    return float(o)
            except Exception:
                pass

            hist = t.history(period="1d", interval="1d")
            if hist is not None and not hist.empty and "Open" in hist.columns:
                o = float(hist["Open"].iloc[0])
                if o > 0:
                    return o
        except Exception:
            pass

    # direct quote endpoint
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("quoteResponse") or {}).get("result") or []
        if result:
            v = result[0].get("regularMarketOpen")
            if v is not None:
                o = float(v)
                if o > 0:
                    return o
    except Exception:
        pass

    return None


def atr_14_from_history(hist_df: pd.DataFrame) -> float | None:
    """
    Compute ATR(14) from a daily OHLC history dataframe returned by get_price_history_from_yahoo.
    If OHLC is missing, fall back to mean absolute close-to-close move (rough).
    """
    if hist_df is None or hist_df.empty:
        return None

    dfh = hist_df.copy()
    # Ensure sorting by Date if present
    if "Date" in dfh.columns:
        dfh = dfh.sort_values("Date")

    # If we have High/Low/Close, compute True Range
    if all(c in dfh.columns for c in ["High", "Low", "Close"]):
        hi = pd.to_numeric(dfh["High"], errors="coerce")
        lo = pd.to_numeric(dfh["Low"], errors="coerce")
        cl = pd.to_numeric(_pick_first_series(dfh['Close']), errors='coerce')
        prev_cl = cl.shift(1)

        tr = pd.concat([
            (hi - lo).abs(),
            (hi - prev_cl).abs(),
            (lo - prev_cl).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(14, min_periods=5).mean().iloc[-1]
        if pd.notna(atr) and float(atr) > 0:
            return float(atr)

    # Fallback: average absolute close change
    if "Close" in dfh.columns:
        cl = pd.to_numeric(_pick_first_series(dfh['Close']), errors='coerce').dropna()
        if len(cl) >= 6:
            atr = cl.diff().abs().rolling(14, min_periods=5).mean().iloc[-1]
            if pd.notna(atr) and float(atr) > 0:
                return float(atr)

    return None
    # Try yfinance first (more resilient)
    if yf is not None:
        try:
            t = yf.Ticker(symbol)
            # fast_info is lightweight; falls back to history if needed
            price = None
            try:
                price = t.fast_info.get("last_price")
            except Exception:
                price = None
            if price is None:
                hist = t.history(period="1d")
                if hist is not None and not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is not None and float(price) > 0:
                return float(price)
        except Exception:
            pass

    # Direct HTTP fallback (no extra dependency)
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        resp = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("quoteResponse") or {}).get("result") or []
        if result and result[0].get("regularMarketPrice") is not None:
            px = float(result[0]["regularMarketPrice"])
            if px > 0:
                return px
    except Exception:
        pass

    return None


import yfinance as yf


def get_price_history_from_yahoo(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame | None:
    """
    Fetch historical prices for charts (moving averages + Fibonacci + ATR).

    Returns a DataFrame with columns:
      ['Date', 'Open', 'High', 'Low', 'Close'] (best effort; may fall back to Close-only).

    Fallback order:
      1) yfinance (if installed)
      2) Yahoo public chart endpoint (no yfinance)
      3) Stooq daily CSV (often works when Yahoo is blocked)

    Note: Fibonacci and ATR need a daily time series (not just spot).
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return None

    # 1) yfinance
    if yf is not None:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period=period, interval=interval)
            if hist is not None and not hist.empty:
                dfh = hist.reset_index()
                if "Date" not in dfh.columns and "Datetime" in dfh.columns:
                    dfh.rename(columns={"Datetime": "Date"}, inplace=True)

                # Standardize columns
                keep = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in dfh.columns]
                if "Date" in keep and "Close" in keep:
                    dfh = dfh[keep].copy()
                    for c in ["Open", "High", "Low", "Close"]:
                        if c in dfh.columns:
                            dfh[c] = pd.to_numeric(dfh[c], errors="coerce")
                    dfh = dfh.dropna(subset=["Date", "Close"])
                    if not dfh.empty:
                        return dfh.sort_values("Date")
        except Exception:
            pass

    # 2) Yahoo chart endpoint (no yfinance)
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"range": period, "interval": interval}
        r = requests.get(url, params=params, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        j = r.json()

        result = j.get("chart", {}).get("result")
        if result:
            r0 = result[0]
            ts = r0.get("timestamp", [])
            quote = (r0.get("indicators", {}) or {}).get("quote", [{}])[0] or {}
            closes = quote.get("close", [])
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])

            if ts and closes:
                dfh = pd.DataFrame({
                    "Date": pd.to_datetime(ts, unit="s"),
                    "Open": opens if opens else [None] * len(ts),
                    "High": highs if highs else [None] * len(ts),
                    "Low": lows if lows else [None] * len(ts),
                    "Close": closes,
                })
                for c in ["Open", "High", "Low", "Close"]:
                    dfh[c] = pd.to_numeric(dfh[c], errors="coerce")
                dfh = dfh.dropna(subset=["Date", "Close"]).sort_values("Date")
                if not dfh.empty:
                    return dfh
    except Exception:
        pass

    # 3) Stooq daily CSV fallback
    try:
        sym = symbol.lower()
        # Stooq uses aapl.us for US stocks
        stooq_symbol = f"{sym}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "Date" in r.text and "Close" in r.text:
            from io import StringIO
            dfh = pd.read_csv(StringIO(r.text))
            # Stooq columns: Date, Open, High, Low, Close, Volume
            keep = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in dfh.columns]
            if "Date" in keep and "Close" in keep:
                dfh = dfh[keep].copy()
                dfh["Date"] = pd.to_datetime(dfh["Date"], errors="coerce")
                for c in ["Open", "High", "Low", "Close"]:
                    if c in dfh.columns:
                        dfh[c] = pd.to_numeric(dfh[c], errors="coerce")
                dfh = dfh.dropna(subset=["Date", "Close"]).sort_values("Date")
                if not dfh.empty:
                    return dfh
    except Exception:
        pass

    return None


def get_api_base_url() -> str:
    """
    Priority:
    1) Streamlit Secrets: st.secrets["API_BASE_URL"]
    2) Environment variable: API_BASE_URL
    3) Local default: http://localhost:8000
    """
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass

    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    return "http://localhost:8000"


API_BASE_URL = get_api_base_url()


# -----------------------------
# Streamlit "width" safe wrappers (future-proof)
# -----------------------------
def st_df(df: pd.DataFrame, height=None, hide_index: bool = True):
    """
    Streamlit 2026+ prefers width=... and rejects height=None.
    This wrapper:
      - uses width="stretch" when supported
      - only passes height if it's a real value (int/"stretch"/"content")
      - falls back to use_container_width for older Streamlit
    """
    kwargs = {"hide_index": hide_index}

    # New API (preferred)
    kwargs["width"] = "stretch"

    if height is not None:
        if isinstance(height, str):
            kwargs["height"] = height  # "stretch" or "content"
        else:
            kwargs["height"] = int(height)

    # Hard guard: NEVER pass height=None
    if kwargs.get("height", "__missing__") is None:
        kwargs.pop("height", None)

    try:
        st.dataframe(df, **kwargs)
    except TypeError:
        # Older Streamlit fallback
        if height is None:
            st.dataframe(df, use_container_width=True, hide_index=hide_index)
        else:
            st.dataframe(df, use_container_width=True, height=int(height), hide_index=hide_index)


def st_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def st_btn(label: str, disabled: bool = False, key: str | None = None):
    try:
        return st.button(label, width="stretch", disabled=disabled, key=key)
    except TypeError:
        return st.button(label, use_container_width=True, disabled=disabled, key=key)


# -----------------------------
# Barchart-inspired dark theme CSS
# -----------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: #1a1d21; }
    .stApp { background: linear-gradient(180deg, #1a1d21 0%, #0d0f11 100%); }
    .header {
        background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0;
    }
    .header h1 { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
    .header p { color: rgba(255,255,255,0.85); margin-top: 0.3rem; font-size: 0.9rem; }

    .status-ok {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 0.4rem 0.8rem;
        background: rgba(0, 215, 117, 0.15);
        border: 1px solid #00d775;
        border-radius: 15px;
        color: #00d775;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-error {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 0.4rem 0.8rem;
        background: rgba(255, 71, 87, 0.15);
        border: 1px solid #ff4757;
        border-radius: 15px;
        color: #ff4757;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2328 0%, #15181c 100%);
        border-right: 1px solid #3d4450;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

/* ---- Responsive typography (mobile + desktop) ---- */
html, body, [class*="css"]  {
  font-size: 16px;
}

/* Make headers/labels easier to read */
h1, h2, h3, h4 { letter-spacing: 0.2px; }

/* Desktop: bump overall size */
@media (min-width: 992px) {
  html, body, [class*="css"] { font-size: 18px; }
  .header h1 { font-size: 2.2rem !important; }
  .header p  { font-size: 1.05rem !important; }
}

/* Mobile: keep compact, prevent overflow */
@media (max-width: 600px) {
  .header { padding: 1rem 1rem !important; }
  .header h1 { font-size: 1.5rem !important; }
  .header p { font-size: 0.9rem !important; }
  .stButton > button { padding: 0.6rem 0.8rem !important; }
}

/* Bigger metric cards readability */
div[data-testid="stMetric"] > div {
  padding: 10px 12px;
}
div[data-testid="stMetricLabel"] p {
  font-size: 0.95rem !important;
}
div[data-testid="stMetricValue"] {
  font-size: 1.6rem !important;
}

/* Dataframes: increase font on desktop, allow horizontal scroll on mobile */
div[data-testid="stDataFrame"] { border-radius: 10px; }
@media (min-width: 992px) {
  div[data-testid="stDataFrame"] * { font-size: 0.95rem !important; }
}
@media (max-width: 600px) {
  div[data-testid="stDataFrame"] { overflow-x: auto; }
}


    /* 🚀 FORCE FULL WIDTH ON DESKTOP (override Streamlit default max-width) */
    [data-testid="stAppViewContainer"] .main .block-container,
    section.main > div.block-container,
    .block-container {
        max-width: 100% !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* Make charts & dataframes stretch to container */
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stDataFrame"] > div {
        width: 100% !important;
    }

    /* Desktop font sizing (fix "too small" look) */
    @media (min-width: 1200px) {
        html, body, [class*="css"] { font-size: 18px !important; }
    }

    /* Mobile: tighter padding + slightly smaller font */
    @media (max-width: 768px) {
        [data-testid="stAppViewContainer"] .main .block-container,
        section.main > div.block-container,
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        html, body, [class*="css"] { font-size: 15px !important; }
    }
</style>
""",
    unsafe_allow_html=True
)


# -----------------------------
# Helpers (API)
# -----------------------------
def check_api() -> bool:
    """Health check endpoint: GET /health"""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@safe_cache_data(ttl=300, show_spinner=False)
def fetch_options(symbol: str, date: str):
    """GET /options?symbol=...&date=..."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/options",
            params={"symbol": symbol, "date": date},
            timeout=120
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}

        try:
            detail = r.json().get("detail", f"HTTP {r.status_code}")
        except Exception:
            detail = f"HTTP {r.status_code}"

        return {"success": False, "error": detail, "status_code": r.status_code}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout calling backend.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}


@safe_cache_data(ttl=300, show_spinner=False)
def fetch_weekly_summary(symbol: str, date: str, spot: float, r: float = 0.05, multiplier: int = 100):
    """GET /weekly/summary?symbol=...&date=...&spot=..."""
    try:
        rqs = requests.get(
            f"{API_BASE_URL}/weekly/summary",
            params={"symbol": symbol, "date": date, "spot": spot, "r": r, "multiplier": multiplier},
            timeout=180
        )
        if rqs.status_code == 200:
            return {"success": True, "data": rqs.json()}

        try:
            detail = rqs.json().get("detail", f"HTTP {rqs.status_code}")
        except Exception:
            detail = f"HTTP {rqs.status_code}"

        return {"success": False, "error": detail, "status_code": rqs.status_code}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout calculating weekly summary (backend scraping may be slow).",
                "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}


@safe_cache_data(ttl=300, show_spinner=False)
def fetch_weekly_gex(symbol: str, date: str, spot: float, r: float = 0.05, multiplier: int = 100):
    """GET /weekly/gex?symbol=...&date=...&spot=..."""
    try:
        rqs = requests.get(
            f"{API_BASE_URL}/weekly/gex",
            params={"symbol": symbol, "date": date, "spot": spot, "r": r, "multiplier": multiplier},
            timeout=180
        )
        if rqs.status_code == 200:
            return {"success": True, "data": rqs.json()}

        try:
            detail = rqs.json().get("detail", f"HTTP {rqs.status_code}")
        except Exception:
            detail = f"HTTP {rqs.status_code}"

        return {"success": False, "error": detail, "status_code": rqs.status_code}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout fetching weekly gex.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}


# -----------------------------
# Helpers (Options charts)
# -----------------------------
def _parse_num(val):
    if pd.isna(val) or val == "":
        return 0.0
    return float(str(val).replace(",", ""))


def create_oi_charts(df: pd.DataFrame):
    df = df.copy()
    df["call_oi"] = df["Call OI"].apply(_parse_num)
    df["put_oi"] = df["Put OI"].apply(_parse_num)
    df["strike_num"] = df["Strike"].apply(_parse_num)
    df_sorted = df.sort_values("strike_num")

    bar_fig = make_subplots(rows=1, cols=2, subplot_titles=("📈 Calls OI (Bar)", "📉 Puts OI (Bar)"))
    bar_fig.add_trace(go.Bar(x=df_sorted["strike_num"], y=df_sorted["call_oi"], name="Calls"), row=1, col=1)
    bar_fig.add_trace(go.Bar(x=df_sorted["strike_num"], y=df_sorted["put_oi"], name="Puts"), row=1, col=2)
    bar_fig.update_layout(template="plotly_dark", height=350, showlegend=False)
    bar_fig.update_xaxes(title_text="Strike")
    bar_fig.update_yaxes(title_text="Open Interest")

    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(x=df_sorted["strike_num"], y=df_sorted["call_oi"], mode="lines+markers", name="Call OI"))
    line_fig.add_trace(
        go.Scatter(x=df_sorted["strike_num"], y=df_sorted["put_oi"], mode="lines+markers", name="Put OI"))
    line_fig.update_layout(
        title="📊 Call vs Put Open Interest by Strike",
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )
    line_fig.update_xaxes(title="Strike")
    line_fig.update_yaxes(title="Open Interest")

    return bar_fig, line_fig


def create_top_strikes_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x_col], y=df[y_col]))
    fig.update_layout(template="plotly_dark", height=320, title=title)
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title=y_col)
    return fig


# -----------------------------
# Volatility + Greeks helpers (no heatmap)
# -----------------------------
def _find_col(df: pd.DataFrame, side: str, key: str):
    """
    side: "call" or "put"
    key:  "iv", "delta", "gamma", "theta", "vega"
    Tries to find a column like:
      - "Call IV", "Put IV"
      - "Call Delta", "Put Delta"
      - "IV (Call)" etc
    """
    side = side.lower()
    key = key.lower()

    cols = list(df.columns)
    # quick exact-ish patterns first
    patterns = [
        rf"^{side}\s*{key}$",
        rf"^{side}\s*{key}\b",
        rf"^{side}\b.*\b{key}$",
        rf"^.*\b{side}\b.*\b{key}\b.*$",
        rf"^.*\b{key}\b.*\b{side}\b.*$",
    ]
    for pat in patterns:
        for c in cols:
            if re.search(pat, str(c), flags=re.IGNORECASE):
                return c
    return None


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")


# -----------------------------
# Black-Scholes Greeks (fallback if backend doesn't provide greeks)
# -----------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _bs_greeks(S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0):
    """
    Black-Scholes(-Merton) greeks for 1 option (call+put).
    Returns: (call_delta, put_delta, gamma, vega_per_1pct, call_theta_per_day, put_theta_per_day)
    - sigma is IV as a decimal (0.2342)
    - vega returned per 1% IV change
    - theta returned per day
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return (float('nan'),) * 6

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    Nmd1 = _norm_cdf(-d1)
    Nmd2 = _norm_cdf(-d2)
    pdf_d1 = _norm_pdf(d1)

    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    call_delta = disc_q * Nd1
    put_delta = disc_q * (Nd1 - 1.0)

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)

    # Vega: per 1.00 vol. Convert to per 1% by /100
    vega_per_1 = S * disc_q * pdf_d1 * sqrtT
    vega_per_1pct = vega_per_1 / 100.0

    # Theta: per year. Convert to per day by /365
    call_theta_y = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * disc_r * Nd2 + q * S * disc_q * Nd1
    put_theta_y = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * disc_r * Nmd2 - q * S * disc_q * Nmd1
    call_theta_d = call_theta_y / 365.0
    put_theta_d = put_theta_y / 365.0

    return call_delta, put_delta, gamma, vega_per_1pct, call_theta_d, put_theta_d


# -----------------------------
# Spot move matrix + Fibonacci helpers
# -----------------------------
def _build_spot_move_matrix(spot: float, call_delta: float, put_delta: float, gamma: float) -> pd.DataFrame:
    """Delta+Gamma approximation of option price change for a set of spot moves."""
    moves = [-20, -15, -10, -7.5, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7.5, 10, 15, 20]
    rows = []
    for dS in moves:
        call_chg = (call_delta * dS) + 0.5 * gamma * (dS ** 2)
        put_chg = (put_delta * dS) + 0.5 * gamma * (dS ** 2)
        rows.append({
            "Spot Move ($)": dS,
            "New Spot": spot + dS,
            "Call Δ+Γ Est. Change": call_chg,
            "Put Δ+Γ Est. Change": put_chg,
        })
    return pd.DataFrame(rows)


def _swing_high_low_from_history(hist_df: pd.DataFrame, lookback_days: int):
    """Return (low, high) from the last N rows of daily history."""
    try:
        sub = hist_df.tail(int(lookback_days)).copy()
        if sub.empty:
            return None
        # Prefer High/Low columns if present, else Close range
        if "Low" in sub.columns and "High" in sub.columns:
            lo = float(sub["Low"].min())
            hi = float(sub["High"].max())
        else:
            lo = float(sub["Close"].min())
            hi = float(sub["Close"].max())
        if lo == hi:
            return None
        return lo, hi
    except Exception:
        return None


def _fib_levels_from_swing(swing_low: float, swing_high: float):
    """Return (retracements, extensions) dicts for a swing."""
    lo, hi = float(swing_low), float(swing_high)
    rng = hi - lo
    if rng == 0:
        return None

    retr = {
        "0% (Low)": lo,
        "23.6%": hi - 0.236 * rng,
        "38.2%": hi - 0.382 * rng,
        "50.0%": hi - 0.500 * rng,
        "61.8%": hi - 0.618 * rng,
        "78.6%": hi - 0.786 * rng,
        "100% (High)": hi,
    }
    ext = {
        "Upper 127.2%": hi + 0.272 * rng,
        "Upper 161.8%": hi + 0.618 * rng,
        "Lower -27.2%": lo - 0.272 * rng,
        "Lower -61.8%": lo - 0.618 * rng,
    }
    return retr, ext


def plot_iv_and_greeks(df: pd.DataFrame, spot: float, T: float | None = None, r: float = 0.041, q: float = 0.004):
    """
    Builds a multi-line figure:
      - IV smile (call/put)
      - Delta, Gamma, Vega, Theta (call/put) if present
    Returns: (fig_iv, fig_greeks, atm_metrics_dict)
    """
    d = df.copy()

    # Strike column can be named differently depending on the source/export.
    # Try to find a reasonable strike column (e.g. "Strike", "Strike Price", etc.).
    strike_col = None
    for c in d.columns:
        if re.search(r"strike", str(c), flags=re.IGNORECASE):
            strike_col = c
            break
    if strike_col is None:
        # fallback: choose the first mostly-numeric column
        for c in d.columns:
            s = _to_float_series(d[c])
            if s.notna().mean() > 0.6:
                strike_col = c
                break
    if strike_col is None:
        return None, None, {}

    d["strike_num"] = _to_float_series(d[strike_col])
    d = d.dropna(subset=["strike_num"]).sort_values("strike_num")

    # --- IV
    call_iv_col = _find_col(d, "call", "iv")
    put_iv_col = _find_col(d, "put", "iv")
    fig_iv = None

    if call_iv_col or put_iv_col:
        fig_iv = go.Figure()
        if call_iv_col:
            d["call_iv"] = _to_float_series(d[call_iv_col])
            fig_iv.add_trace(
                go.Scatter(x=d["strike_num"], y=d["call_iv"], mode="lines+markers", name=f"Call IV ({call_iv_col})"))
        if put_iv_col:
            d["put_iv"] = _to_float_series(d[put_iv_col])
            fig_iv.add_trace(
                go.Scatter(x=d["strike_num"], y=d["put_iv"], mode="lines+markers", name=f"Put IV ({put_iv_col})"))

        fig_iv.add_vline(x=float(spot), line_width=2, line_dash="dash", annotation_text="Spot",
                         annotation_position="top")
        fig_iv.update_layout(
            template="plotly_dark",
            height=420,
            title="📉 IV Smile (by Strike)",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility (raw units from source)",
            hovermode="x unified",
        )

    # --- Greeks
    greek_keys = ["delta", "gamma", "vega", "theta"]
    fig_g = go.Figure()
    any_greek = False

    for gk in greek_keys:
        ccol = _find_col(d, "call", gk)
        pcol = _find_col(d, "put", gk)
        if ccol:
            any_greek = True
            fig_g.add_trace(go.Scatter(
                x=d["strike_num"], y=_to_float_series(d[ccol]),
                mode="lines", name=f"Call {gk.title()} ({ccol})"
            ))
        if pcol:
            any_greek = True
            fig_g.add_trace(go.Scatter(
                x=d["strike_num"], y=_to_float_series(d[pcol]),
                mode="lines", name=f"Put {gk.title()} ({pcol})"
            ))

    # If backend doesn't provide greeks, approximate them from IV using Black-Scholes
    if (not any_greek) and (T is not None) and (call_iv_col or put_iv_col):
        # Ensure IV numeric columns exist
        if call_iv_col and "call_iv" not in d.columns:
            d["call_iv"] = _to_float_series(d[call_iv_col])
        if put_iv_col and "put_iv" not in d.columns:
            d["put_iv"] = _to_float_series(d[put_iv_col])

        def _iv_to_sigma(iv_val: float) -> float:
            if pd.isna(iv_val):
                return float('nan')
            # Many chains store IV as percent (e.g., 23.42). If so, convert to decimal.
            return (iv_val / 100.0) if iv_val > 3.0 else float(iv_val)

        # Vectorized-ish calculation (row-wise due to CDF/PDF)
        S = float(spot)
        T_val = float(T)
        r_val = float(r)
        q_val = float(q)

        # Use call IV for call greeks and put IV for put greeks (common in chains)
        call_sig = d.get("call_iv", pd.Series([float('nan')] * len(d))).apply(_iv_to_sigma)
        put_sig = d.get("put_iv", pd.Series([float('nan')] * len(d))).apply(_iv_to_sigma)

        call_delta = []
        put_delta = []
        call_gamma = []
        put_gamma = []
        call_vega = []
        put_vega = []
        call_theta = []
        put_theta = []

        for idx_row, K in enumerate(d["strike_num"].tolist()):
            # Call side greeks from call IV
            sig_c = call_sig.iloc[idx_row] if idx_row < len(call_sig) else float('nan')
            dc, dp, gm, vg, th_c, th_p = _bs_greeks(
                S, float(K), T_val, float(sig_c) if not pd.isna(sig_c) else float('nan'), r_val, q_val
            )
            call_delta.append(dc)
            call_gamma.append(gm)
            call_vega.append(vg)
            call_theta.append(th_c)

            # Put side greeks from put IV
            sig_p = put_sig.iloc[idx_row] if idx_row < len(put_sig) else float('nan')
            dc2, dp2, gm2, vg2, th_c2, th_p2 = _bs_greeks(
                S, float(K), T_val, float(sig_p) if not pd.isna(sig_p) else float('nan'), r_val, q_val
            )
            put_delta.append(dp2)
            put_gamma.append(gm2)
            put_vega.append(vg2)
            put_theta.append(th_p2)
        d["Call Delta"] = pd.Series(call_delta, index=d.index)
        d["Put Delta"] = pd.Series(put_delta, index=d.index)
        d["Call Gamma"] = pd.Series(call_gamma, index=d.index)
        d["Put Gamma"] = pd.Series(put_gamma, index=d.index)
        d["Call Vega"] = pd.Series(call_vega, index=d.index)
        d["Put Vega"] = pd.Series(put_vega, index=d.index)
        d["Call Theta"] = pd.Series(call_theta, index=d.index)
        d["Put Theta"] = pd.Series(put_theta, index=d.index)

        # Build greeks plot from computed columns
        fig_g = go.Figure()
        for name in ["Call Delta", "Put Delta", "Call Gamma", "Put Gamma", "Call Vega", "Put Vega", "Call Theta",
                     "Put Theta"]:
            fig_g.add_trace(go.Scatter(x=d["strike_num"], y=_to_float_series(d[name]), mode="lines", name=name))
        any_greek = True

    fig_greeks = None
    if any_greek:
        fig_g.add_vline(x=float(spot), line_width=2, line_dash="dash", annotation_text="Spot",
                        annotation_position="top")
        fig_g.update_layout(
            template="plotly_dark",
            height=520,
            title="🧮 Greeks by Strike (if available from backend)",
            xaxis_title="Strike",
            yaxis_title="Greek value (raw units from source)",
            hovermode="x unified",
        )
        fig_greeks = fig_g

    # --- ATM snapshot (nearest strike to spot)
    atm_metrics = {}
    if len(d) > 0:
        atm_row = d.iloc[(d["strike_num"] - float(spot)).abs().argsort()[:1]].iloc[0]
        atm_metrics["atm_strike"] = float(atm_row["strike_num"])

        # capture key fields if they exist
        for label, side, key in [
            ("Call IV", "call", "iv"),
            ("Put IV", "put", "iv"),
            ("Call Delta", "call", "delta"),
            ("Put Delta", "put", "delta"),
            ("Call Gamma", "call", "gamma"),
            ("Put Gamma", "put", "gamma"),
            ("Call Vega", "call", "vega"),
            ("Put Vega", "put", "vega"),
            ("Call Theta", "call", "theta"),
            ("Put Theta", "put", "theta"),
        ]:
            col = _find_col(d, side, key)
            if col and col in d.columns:
                val = _to_float_series(pd.Series([atm_row[col]])).iloc[0]
                if pd.notna(val):
                    atm_metrics[label] = float(val)

    return fig_iv, fig_greeks, atm_metrics


def approx_skew_25d(df: pd.DataFrame):
    """
    Rough 25-delta skew estimate if Delta + IV exist:
      25d call IV  -  (-25d put IV)
    Uses nearest delta to +0.25 for calls and -0.25 for puts.
    Returns dict or {}.
    """
    d = df.copy()
    if "Strike" not in d.columns:
        return {}

    d["strike_num"] = _to_float_series(d["Strike"])
    d = d.dropna(subset=["strike_num"]).sort_values("strike_num")

    c_iv = _find_col(d, "call", "iv")
    p_iv = _find_col(d, "put", "iv")
    c_del = _find_col(d, "call", "delta")
    p_del = _find_col(d, "put", "delta")
    if not (c_iv and p_iv and c_del and p_del):
        return {}

    d["call_iv"] = _to_float_series(d[c_iv])
    d["put_iv"] = _to_float_series(d[p_iv])
    d["call_delta"] = _to_float_series(d[c_del])
    d["put_delta"] = _to_float_series(d[p_del])

    # nearest +0.25 call delta
    dc = d.dropna(subset=["call_iv", "call_delta"])
    dp = d.dropna(subset=["put_iv", "put_delta"])
    if dc.empty or dp.empty:
        return {}

    c_row = dc.iloc[(dc["call_delta"] - 0.25).abs().argsort()[:1]].iloc[0]
    p_row = dp.iloc[(dp["put_delta"] + 0.25).abs().argsort()[:1]].iloc[0]  # put delta near -0.25

    skew = float(c_row["call_iv"] - p_row["put_iv"])
    return {
        "call_25d_strike": float(c_row["strike_num"]),
        "call_25d_iv": float(c_row["call_iv"]),
        "call_25d_delta": float(c_row["call_delta"]),
        "put_25d_strike": float(p_row["strike_num"]),
        "put_25d_iv": float(p_row["put_iv"]),
        "put_25d_delta": float(p_row["put_delta"]),
        "skew_call_minus_put": skew,
    }


# -----------------------------
# Gamma Map helpers
# -----------------------------
def build_gamma_levels(gex_df: pd.DataFrame, spot: float, top_n: int = 5):
    df = gex_df.copy()
    if df.empty:
        return None

    for c in ["strike", "call_gex", "put_gex", "net_gex", "gamma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    magnets = (
        df.assign(net_abs=df["net_gex"].abs())
        .sort_values("net_abs", ascending=False)
        .head(top_n)[["strike", "net_gex"]]
    )

    call_walls = df.sort_values("call_gex", ascending=False).head(top_n)[["strike", "call_gex"]]
    put_walls = df.sort_values("put_gex", ascending=False).head(top_n)[["strike", "put_gex"]]

    put_below = put_walls[put_walls["strike"] <= spot].sort_values("strike", ascending=False)
    call_above = call_walls[call_walls["strike"] >= spot].sort_values("strike", ascending=True)

    lower = float(put_below.iloc[0]["strike"]) if len(put_below) else None
    upper = float(call_above.iloc[0]["strike"]) if len(call_above) else None

    return {
        "magnets": magnets,
        "call_walls": call_walls,
        "put_walls": put_walls,
        "gamma_box": {"lower": lower, "upper": upper},
    }


def plot_net_gex_map(gex_df: pd.DataFrame, spot: float, levels: dict):
    df = gex_df.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["net_gex"] = pd.to_numeric(df["net_gex"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["strike"]).sort_values("strike")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["strike"], y=df["net_gex"], mode="lines+markers", name="Net GEX"))

    fig.add_vline(
        x=spot, line_width=2, line_dash="dash",
        annotation_text=f"Spot {spot:g}", annotation_position="top"
    )

    for _, row in levels["magnets"].iterrows():
        s = float(row["strike"])
        fig.add_vline(
            x=s, line_width=1, line_dash="dot",
            annotation_text=f"Magnet {s:g}", annotation_position="bottom"
        )

    lower = levels["gamma_box"]["lower"]
    upper = levels["gamma_box"]["upper"]
    if lower is not None:
        fig.add_vline(
            x=lower, line_width=2, line_dash="dash",
            annotation_text=f"Lower wall {lower:g}", annotation_position="top left"
        )
    if upper is not None:
        fig.add_vline(
            x=upper, line_width=2, line_dash="dash",
            annotation_text=f"Upper wall {upper:g}", annotation_position="top right"
        )

    fig.update_layout(
        template="plotly_dark",
        height=450,
        title="🧲 Gamma Map (Net GEX by Strike)",
        xaxis_title="Strike",
        yaxis_title="Net GEX"
    )
    return fig


# -----------------------------
# Noise Filters (McGinley / KAMA / Kalman)
# -----------------------------
def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance may return MultiIndex columns.
    Normalize to a plain DataFrame containing ONLY 'Close'.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # If MultiIndex, find a Close-like column
    if isinstance(out.columns, pd.MultiIndex):
        close_col = None
        for col in out.columns:
            # possible forms: ('Close','AAPL') or ('AAPL','Close')
            if any(str(x).lower() == "close" for x in col):
                close_col = col
                break
        if close_col is not None:
            out = pd.DataFrame({"Close": out[close_col]})
        else:
            # flatten then try to find close
            out.columns = ["_".join(map(str, c)).strip() for c in out.columns]

    # If not MultiIndex, ensure Close exists
    if "Close" not in out.columns:
        for alt in ["close", "Adj Close", "adj close", "Adj_Close", "adjclose"]:
            if alt in out.columns:
                out["Close"] = out[alt]
                break

    if "Close" not in out.columns:
        return pd.DataFrame()

    out["Close"] = pd.to_numeric(_pick_first_series(out['Close']), errors='coerce')
    out = out
    return out[["Close"]]


@safe_cache_data(ttl=900, show_spinner=False)
def fetch_price_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    return _normalize_yf_df(raw)


def mcginley_dynamic(price: pd.Series, length: int = 14) -> pd.Series:
    p = pd.Series(price).astype(float).copy()
    out = pd.Series(index=p.index, dtype=float)

    md_prev = float(p.iloc[0])
    out.iloc[0] = md_prev

    for i in range(1, len(p)):
        pi = float(p.iloc[i])
        denom = length * (pi / md_prev) ** 4 if md_prev != 0 else length
        md_prev = md_prev + (pi - md_prev) / denom
        out.iloc[i] = md_prev

    return out


def kama(price: pd.Series, er_length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    p = pd.Series(price).astype(float).copy()
    out = pd.Series(index=p.index, dtype=float)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    out.iloc[0] = float(p.iloc[0])
    abs_diff = p.diff().abs()

    for i in range(1, len(p)):
        if i < er_length:
            out.iloc[i] = out.iloc[i - 1]
            continue

        change = abs(float(p.iloc[i]) - float(p.iloc[i - er_length]))
        volatility = float(abs_diff.iloc[i - er_length + 1: i + 1].sum())
        er = (change / volatility) if volatility != 0 else 0.0

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out.iloc[i] = out.iloc[i - 1] + sc * (float(p.iloc[i]) - float(out.iloc[i - 1]))

    return out


def kalman_filter_1d(close, process_var=1e-5, meas_var=1e-2) -> np.ndarray:
    """
    Simple 1D Kalman filter for price smoothing.
    - Accepts Series / list / numpy / DataFrame
    - Forces STRICT 1D float array to avoid 'sequence' assignment errors.
    """
    if isinstance(close, pd.DataFrame):
        z = close.iloc[:, 0].to_numpy()
    elif isinstance(close, pd.Series):
        z = close.to_numpy()
    else:
        z = np.asarray(close)

    # STRICT 1D float array
    z = np.asarray(z, dtype=float).reshape(-1)

    n = int(len(z))
    if n == 0:
        return np.array([], dtype=float)

    x = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)

    x[0] = float(z[0])
    p[0] = 1.0

    q = float(process_var)
    r = float(meas_var)

    for k in range(1, n):
        x_pred = x[k - 1]
        p_pred = p[k - 1] + q

        K = p_pred / (p_pred + r)
        x[k] = x_pred + K * (float(z[k]) - x_pred)
        p[k] = (1 - K) * p_pred

    return x


def kalman_message(close_series, kalman_series, lookback: int = 20, band_pct: float = 0.003):
    """More confident Kalman interpretation.

    Output fields:
      - trend: UPTREND / DOWNTREND / RANGE / TRANSITION / N/A
      - bias: PRICE ABOVE/BELOW/NEAR KALMAN
      - crossings: sign flips of (price - kalman) over lookback (chop proxy)
      - trend_strength: 0..100 score (higher = stronger)
      - regime: TRENDING / RANGEBOUND / TRANSITION
      - structure: HH/HL, LH/LL, or MIXED (very simple market structure)
      - msg: short readable summary
      - reasons: list of bullet reasons

    Notes:
      - This is heuristic (not a guarantee). Use as a *context* tool.
    """
    close = np.asarray(close_series, dtype=float).reshape(-1)
    kf = np.asarray(kalman_series, dtype=float).reshape(-1)

    n = int(min(len(close), len(kf)))
    if n < 10:
        return {
            "trend": "N/A",
            "bias": "N/A",
            "crossings": 0,
            "trend_strength": 0,
            "regime": "N/A",
            "structure": "N/A",
            "msg": "Not enough data for Kalman interpretation.",
            "reasons": ["Need at least ~10 bars."],
        }

    close = close[-n:]
    kf = kf[-n:]

    lb = min(int(lookback), n - 1)
    c = close[-lb:]
    k = kf[-lb:]

    # --- Bias (price vs Kalman) ---
    band = abs(float(k[-1])) * float(band_pct)
    diff = float(c[-1] - k[-1])
    if diff > band:
        bias = "PRICE ABOVE KALMAN"
    elif diff < -band:
        bias = "PRICE BELOW KALMAN"
    else:
        bias = "PRICE NEAR KALMAN"

    # --- Chop proxy: crossings ---
    sign = np.sign(c - k)
    sign[sign == 0] = 1
    crossings = int(np.sum(sign[1:] != sign[:-1]))

    # --- Trend slope of Kalman ---
    slope = float(k[-1] - k[0])

    # --- Volatility scale (ATR-ish from closes) ---
    # Use mean absolute close-to-close move over lookback as a simple volatility proxy.
    abs_moves = np.abs(np.diff(c))
    vol = float(np.mean(abs_moves)) if len(abs_moves) else 0.0

    # Strength = slope relative to recent vol (capped)
    # If vol is tiny, avoid divide-by-zero and treat slope as weak unless it's meaningful.
    ratio = abs(slope) / (vol + 1e-9)
    # Map to 0..100 with diminishing returns
    trend_strength = int(max(0, min(100, round(100 * (1 - np.exp(-0.35 * ratio))))))

    # --- Simple structure (HH/HL vs LH/LL) ---
    # Compare last 3 pivot-ish points via rolling extremes.
    # This is intentionally simple + robust.
    hi1 = float(np.max(c[-lb:]))
    lo1 = float(np.min(c[-lb:]))
    mid = lb // 2
    hi0 = float(np.max(c[:mid])) if mid > 2 else hi1
    lo0 = float(np.min(c[:mid])) if mid > 2 else lo1

    if hi1 > hi0 and lo1 > lo0:
        structure = "HH/HL"
    elif hi1 < hi0 and lo1 < lo0:
        structure = "LH/LL"
    else:
        structure = "MIXED"

    # --- Regime label ---
    # Many crossings => range/chop.
    chop_threshold = max(6, lb // 4)
    is_choppy = crossings >= chop_threshold

    if is_choppy and trend_strength < 35:
        trend = "RANGE"
        regime = "RANGEBOUND"
    else:
        if slope > 0:
            trend = "UPTREND"
        elif slope < 0:
            trend = "DOWNTREND"
        else:
            trend = "RANGE"

        # If slope says trend but price keeps crossing or strength is weak => transition
        if (trend_strength < 35) or (crossings >= max(3, chop_threshold // 2)):
            regime = "TRANSITION"
            if trend != "RANGE":
                trend = "TRANSITION"
        else:
            regime = "TRENDING"

    # --- Message + reasons ---
    reasons = []
    reasons.append(f"Kalman slope over last {lb} bars: {slope:+.4f}")
    reasons.append(f"Crossings (chop) over {lb} bars: {crossings}")
    reasons.append(f"Trend strength score: {trend_strength}/100")
    reasons.append(f"Structure: {structure}")
    reasons.append(f"Bias: {bias}")

    msg = f"Kalman regime: {regime}. Signal: {trend}. {bias}."

    # Extra trader hint
    if trend == "UPTREND" and "ABOVE" in bias:
        msg += " Strong trend; pullbacks toward Kalman can act as support."
    if trend == "UPTREND" and "BELOW" in bias:
        msg += " Trend up but price below Kalman → watch reclaim; failure can mean weakness."
    if trend == "DOWNTREND" and "BELOW" in bias:
        msg += " Downtrend; rallies toward Kalman often fade (resistance)."
    if trend == "DOWNTREND" and "ABOVE" in bias:
        msg += " Price above Kalman in downtrend → possible transition if it holds."
    if trend == "RANGE":
        msg += " Expect mean-reversion; Kalman can act as the midline."

    return {
        "trend": trend,
        "bias": bias,
        "crossings": crossings,
        "trend_strength": trend_strength,
        "regime": regime,
        "structure": structure,
        "msg": msg,
        "reasons": reasons,
    }


def plot_filters(df_prices: pd.DataFrame, length_md: int, kama_er: int, kama_fast: int, kama_slow: int,
                 kf_q: float, kf_r: float):
    close = df_prices["Close"].astype(float)

    md = mcginley_dynamic(close, length=length_md)
    k = kama(close, er_length=kama_er, fast=kama_fast, slow=kama_slow)

    kf_arr = kalman_filter_1d(close, process_var=kf_q, meas_var=kf_r)
    # IMPORTANT: give Kalman the SAME index so plotting never crashes
    kf = pd.Series(kf_arr, index=close.index, name="Kalman")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=md.index, y=md, mode="lines", name=f"McGinley ({length_md})"))
    fig.add_trace(go.Scatter(x=k.index, y=k, mode="lines", name=f"KAMA (ER={kama_er}, {kama_fast}/{kama_slow})"))
    fig.add_trace(go.Scatter(x=kf.index, y=kf, mode="lines", name=f"Kalman (Q={kf_q:g}, R={kf_r:g})"))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title="📈 Market Noise Filters (McGinley / KAMA / Kalman)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    return fig, kf


# -----------------------------
# Main App
# -----------------------------
# ---------------- PRO EDGE HELPERS (Trend / IV / Vanna / Charm / Levels) ----------------

def _slope(series: pd.Series, lookback: int = 5) -> float:
    """Simple slope proxy: last - value N bars ago."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) <= lookback:
            return float("nan")
        return float(s.iloc[-1] - s.iloc[-1 - lookback])
    except Exception:
        return float("nan")


def compute_ma_stack_and_regime(hist) -> dict:
    """
    Compute MA stacking, slopes, and a trend regime label from DAILY history.

    This version is bulletproof against:
    - MultiIndex columns (yfinance can return ('Close','AAPL'))
    - Adj Close only
    - lowercase close
    - Series input
    - weird provider schemas

    It ALWAYS rebuilds a clean DataFrame with a single 'Close' column.
    """
    out = {"ok": False, "label": "N/A", "strength": 0, "details": {}}

    if hist is None:
        return out

    # -----------------------------
    # 1) Extract a CLOSE series safely
    # -----------------------------
    close_series = None

    # If it's already a Series
    if isinstance(hist, pd.Series):
        close_series = hist.copy()

    # If it's a DataFrame
    elif isinstance(hist, pd.DataFrame) and len(hist) > 0:
        h = hist.copy()

        # Normalize non-multiindex column names (strip spaces)
        if not isinstance(h.columns, pd.MultiIndex):
            try:
                h.columns = [str(c).strip() for c in h.columns]
            except Exception:
                pass

        # Case A: normal columns
        for candidate in ["Close", "Adj Close", "close", "adj close", "AdjClose", "adjclose"]:
            if candidate in h.columns:
                close_series = h[candidate]
                break

        # Case B: MultiIndex columns (e.g. ('Close','AAPL'))
        if close_series is None and isinstance(h.columns, pd.MultiIndex):
            # try to find any column whose level-0 name is close/adj close
            pick = None
            for c in h.columns:
                try:
                    lvl0 = str(c[0]).strip().lower()
                except Exception:
                    continue
                if lvl0 in ("close", "adj close", "adjclose"):
                    pick = c
                    break
            if pick is not None:
                close_series = h[pick]

        # Case C: last resort: any column containing 'close'
        if close_series is None:
            try:
                close_like = [c for c in h.columns if "close" in str(c).lower()]
                if close_like:
                    close_series = h[close_like[0]]
            except Exception:
                pass

    # Nothing usable found
    if close_series is None:
        out["details"]["error"] = "Could not find a Close series in hist_daily."
        return out

    # If we got a DF instead of Series (rare), flatten it
    if isinstance(close_series, pd.DataFrame):
        if close_series.shape[1] == 0:
            out["details"]["error"] = "Close selection returned empty DataFrame."
            return out
        close_series = close_series.iloc[:, 0]

    # Force numeric + clean
    close_series = pd.to_numeric(close_series, errors="coerce")

    # -----------------------------
    # 2) Rebuild clean DataFrame (prevents KeyError forever)
    # -----------------------------
    h = pd.DataFrame({"Close": close_series}).dropna()

    # Need enough bars for MA60 at least (and ideally SMA200)
    if len(h) < 210:
        out["details"]["error"] = f"Not enough daily bars: {len(h)} (need ~210 for SMA200)."
        return out

    close = h["Close"]

    # -----------------------------
    # 3) Moving averages
    # -----------------------------
    for w in [20, 50, 200]:
        h[f"SMA{w}"] = close.rolling(w).mean()

    short_windows = [15, 20, 30, 45, 60]
    for w in short_windows:
        h[f"MA{w}"] = close.rolling(w).mean()

    last = h.iloc[-1]
    c = float(last["Close"])

    sma20 = float(last["SMA20"])
    sma50 = float(last["SMA50"])
    sma200 = float(last["SMA200"])

    # slopes over 10 bars
    def slope(col: str, lookback: int = 10) -> float:
        a = h[col].iloc[-1]
        b = h[col].iloc[-lookback - 1]
        return float(a - b)

    s20 = slope("SMA20")
    s50 = slope("SMA50")
    s200 = slope("SMA200")

    up_slopes = sum(1 for v in (s20, s50, s200) if np.isfinite(v) and v > 0)
    dn_slopes = sum(1 for v in (s20, s50, s200) if np.isfinite(v) and v < 0)

    # stacking (macro)
    bull_stack = (c > sma20 > sma50 > sma200)
    bear_stack = (c < sma20 < sma50 < sma200)

    # short stack (weekly-friendly)
    ma_vals = [float(last[f"MA{w}"]) for w in short_windows]

    def is_desc(vals):
        return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

    def is_asc(vals):
        return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    short_bull_stack = is_desc(ma_vals)
    short_bear_stack = is_asc(ma_vals)

    # strength score
    strength = 0
    strength += 30 if bull_stack else 0
    strength += 30 if bear_stack else 0
    strength += 20 if short_bull_stack else 0
    strength += 20 if short_bear_stack else 0
    strength += 15 if up_slopes >= 2 else 0
    strength += 15 if dn_slopes >= 2 else 0
    strength += 10 if (c > sma200 and not bear_stack) else 0
    strength += 10 if (c < sma200 and not bull_stack) else 0
    strength = int(min(100, strength))

    # label
    if bull_stack and up_slopes >= 2:
        label = "STRONG UPTREND 📈"
    elif bear_stack and dn_slopes >= 2:
        label = "STRONG DOWNTREND 📉"
    elif c > sma200:
        label = "WEAK / CHOPPY UPTREND ⚠️"
    elif c < sma200:
        label = "WEAK / CHOPPY DOWNTREND ⚠️"
    else:
        label = "NO TREND 😴"

    out["ok"] = True
    out["label"] = label
    out["strength"] = strength
    out["details"] = {
        "close": c,
        "SMA20": sma20, "SMA50": sma50, "SMA200": sma200,
        "slope20_10": s20, "slope50_10": s50, "slope200_10": s200,
        "bull_stack": bool(bull_stack), "bear_stack": bool(bear_stack),
        "short_bull_stack": bool(short_bull_stack), "short_bear_stack": bool(short_bear_stack),
    }
    return out

    # --- slope over last 10 bars (simple) ---
    def slope(col: str, lookback: int = 10) -> float:
        if col not in h.columns or len(h) <= lookback:
            return float("nan")
        a = h[col].iloc[-1]
        b = h[col].iloc[-lookback-1]
        try:
            return float(a - b)
        except Exception:
            return float("nan")

    s20 = slope("SMA20")
    s50 = slope("SMA50")
    s200 = slope("SMA200")

    up_slopes = sum([1 for v in (s20, s50, s200) if pd.notna(v) and v > 0])
    dn_slopes = sum([1 for v in (s20, s50, s200) if pd.notna(v) and v < 0])

    # --- stacking ---
    bull_stack = pd.notna(sma20) and pd.notna(sma50) and pd.notna(sma200) and (c > sma20 > sma50 > sma200)
    bear_stack = pd.notna(sma20) and pd.notna(sma50) and pd.notna(sma200) and (c < sma20 < sma50 < sma200)

    ma_cols = [f"MA{w}" for w in short_windows]
    ma_vals = [last.get(col, np.nan) for col in ma_cols]

    def is_desc(vals):
        for i in range(len(vals)-1):
            if not (pd.notna(vals[i]) and pd.notna(vals[i+1]) and float(vals[i]) > float(vals[i+1])):
                return False
        return True

    def is_asc(vals):
        for i in range(len(vals)-1):
            if not (pd.notna(vals[i]) and pd.notna(vals[i+1]) and float(vals[i]) < float(vals[i+1])):
                return False
        return True

    short_bull_stack = is_desc(ma_vals)
    short_bear_stack = is_asc(ma_vals)

    # --- strength score (0-100) ---
    strength = 0
    strength += 30 if bull_stack else 0
    strength += 30 if bear_stack else 0
    strength += 20 if short_bull_stack else 0
    strength += 20 if short_bear_stack else 0
    strength += 15 if up_slopes >= 2 else 0
    strength += 15 if dn_slopes >= 2 else 0
    # price vs 200 as macro filter
    strength += 10 if (pd.notna(sma200) and c > sma200 and not bear_stack) else 0
    strength += 10 if (pd.notna(sma200) and c < sma200 and not bull_stack) else 0
    strength = int(min(100, strength))

    if bull_stack and up_slopes >= 2:
        label = "STRONG UPTREND 📈"
    elif bear_stack and dn_slopes >= 2:
        label = "STRONG DOWNTREND 📉"
    elif pd.notna(sma200) and c > sma200:
        label = "WEAK / CHOPPY UPTREND ⚠️"
    elif pd.notna(sma200) and c < sma200:
        label = "WEAK / CHOPPY DOWNTREND ⚠️"
    else:
        label = "NO TREND 😴"

    out["ok"] = True
    out["label"] = label
    out["strength"] = strength
    out["details"] = {
        "close": c,
        "SMA20": sma20, "SMA50": sma50, "SMA200": sma200,
        "slope20_10": s20, "slope50_10": s50, "slope200_10": s200,
        "bull_stack": bool(bull_stack), "bear_stack": bool(bear_stack),
        "short_bull_stack": bool(short_bull_stack), "short_bear_stack": bool(short_bear_stack),
    }
    return out

def realized_vol_annualized(hist: pd.DataFrame, window: int = 20) -> float:
    """Annualized realized vol using log returns (daily)."""
    if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < window + 2:
        return float("nan")
    h = hist.copy()
    c = pd.to_numeric(_pick_first_series(h['Close']), errors='coerce').dropna()
    r = np.log(c / c.shift(1)).dropna()
    rv = r.rolling(window).std() * math.sqrt(252)
    try:
        return float(rv.iloc[-1])
    except Exception:
        return float("nan")


def iv_proxy_rank(current_iv: float, hist: pd.DataFrame, window: int = 20) -> dict:
    """
    IV Rank is hard to do for free without historical option IV series.
    This returns a *proxy* rank by comparing current IV to the last year's realized vol range.
    """
    out = {"ok": False, "iv_proxy_rank": None, "rv_min": None, "rv_max": None, "details": {}}

    if hist is None or hist.empty or not (current_iv and current_iv > 0):
        return out

    h = hist.copy()

    # Normalize MultiIndex / duplicate columns so we can reliably extract Close as a 1-D series
    if isinstance(getattr(h, "columns", None), pd.MultiIndex):
        col_map = {}
        for col in h.columns:
            for target in ["Close", "Adj Close", "close", "adj close"]:
                if target not in col_map and any(str(x).strip().lower() == target.lower() for x in col):
                    col_map[target] = col
        for target, tup in col_map.items():
            try:
                h[target] = h[tup]
            except Exception:
                pass

    close_obj = None
    for candidate in ["Close", "Adj Close", "close", "adj close", "AdjClose", "adjclose"]:
        if candidate in h.columns:
            close_obj = h[candidate]
            break

    if close_obj is None:
        out["details"]["error"] = "Close column not found."
        out["details"]["columns"] = str(list(getattr(h, "columns", [])))
        return out

    # If selection yields a DataFrame (duplicate column name), take first column
    if isinstance(close_obj, pd.DataFrame):
        if close_obj.shape[1] == 0:
            out["details"]["error"] = "Close selection returned empty DataFrame."
            return out
        close_obj = close_obj.iloc[:, 0]

    c = pd.to_numeric(close_obj, errors="coerce").dropna()
    if c.empty:
        out["details"]["error"] = "Close series empty after numeric coercion."
        return out

    r = np.log(c / c.shift(1)).dropna()
    rv = r.rolling(window).std() * math.sqrt(252)
    rv = rv.dropna()
    if rv.empty:
        return out

    rv_min = float(rv.min())
    rv_max = float(rv.max())
    if rv_max == rv_min:
        rank = 50.0
    else:
        rank = 100.0 * (float(current_iv) - rv_min) / (rv_max - rv_min)
        rank = max(0.0, min(100.0, rank))

    out.update({"ok": True, "iv_proxy_rank": float(rank), "rv_min": rv_min, "rv_max": rv_max})
    return out


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    # Abramowitz & Stegun approximation (good enough for dashboard)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2.0)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + 1.330274 * t))))
    return 1.0 - prob if x > 0 else prob


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if is_call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_vanna_charm(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> dict:
    """
    Approximate Vanna and Charm for a single option using Black-Scholes.
    - Vanna: dDelta/dVol (per 1.0 change in vol, e.g. +0.01 = 1 vol point)
    - Charm: dDelta/dt (per year). We also provide per day.
    Notes: This is an approximation (equity-style), but good for directional flow intuition.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return {"delta": float("nan"), "vanna": float("nan"), "charm_per_year": float("nan"), "charm_per_day": float("nan")}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = _norm_pdf(d1)

    delta = bs_delta(S, K, T, r, sigma, is_call)

    # Common vanna approximation (per 1.0 vol change). Convert to per 1 vol point by /100 later if needed.
    vanna = pdf * (1.0 - d1 / (sigma * sqrtT)) / sigma

    # Charm approximation (per year)
    # Call charm = -pdf * (2rT - d2*sigma*sqrtT) / (2T*sigma*sqrtT)
    # Put charm  = call charm (same core term) because delta differs by -1
    charm = -pdf * (2.0 * r * T - d2 * sigma * sqrtT) / (2.0 * T * sigma * sqrtT)

    return {
        "delta": float(delta),
        "vanna": float(vanna),
        "charm_per_year": float(charm),
        "charm_per_day": float(charm / 365.0),
    }


def compute_key_levels(hist_daily: pd.DataFrame) -> dict:
    """Prior day + last 5 days (weekly) levels."""
    out = {'ok': False, 'details': {}}
    if hist_daily is None or hist_daily.empty:
        return out
    h = hist_daily.copy()
    # Normalize MultiIndex columns from yfinance (e.g., ("Close","AAPL") or ("AAPL","Close"))
    if isinstance(h.columns, pd.MultiIndex):
        col_map = {}
        for col in h.columns:
            for target in ["Open", "High", "Low", "Close", "Volume"]:
                if target not in col_map and any(str(x).lower() == target.lower() for x in col):
                    col_map[target] = col
        for target, col_tup in col_map.items():
            try:
                h[target] = h[col_tup]
            except Exception:
                # If assignment fails, skip and let subsequent checks handle missing data
                pass

    if "Date" not in h.columns:
        h = h.reset_index().rename(columns={"index": "Date"})

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in h.columns:
            # Ensure we have a single Series (not a DataFrame) before numeric conversion
            col_val = h[col]
            if isinstance(col_val, pd.DataFrame):
                col_val = col_val.iloc[:, 0]
            h[col] = pd.to_numeric(col_val, errors="coerce")

    # Ensure we have a usable Close column (handle variants like 'Adj Close', lowercase, or provider-specific names)
    if "Close" not in h.columns:
        # Try common alternatives first
        for candidate in ["Close", "Adj Close", "close", "adj close", "AdjClose", "adjclose"]:
            if candidate in h.columns:
                h["Close"] = h[candidate]
                break

        # Last resort: any column containing the word 'close'
        if "Close" not in h.columns:
            try:
                close_like = [c for c in h.columns if "close" in str(c).lower()]
                if close_like:
                    h["Close"] = h[close_like[0]]
            except Exception:
                pass

    # If still missing, return a clear error instead of raising KeyError
    if "Close" not in h.columns:
        out["details"]["error"] = "Could not find a Close series in hist_daily."
        return out

    # Ensure Close is a single 1-D Series and numeric, then drop rows without it
    close_val = h["Close"]
    # If it's a DataFrame, pick a single sensible column
    if isinstance(close_val, pd.DataFrame):
        if close_val.shape[1] == 1:
            close_val = close_val.iloc[:, 0]
        else:
            # Prefer numeric dtype columns when available
            try:
                numeric_cols = close_val.select_dtypes(include=["number"]).columns.tolist()
                if numeric_cols:
                    close_val = close_val[numeric_cols[0]]
                else:
                    close_val = close_val.iloc[:, 0]
            except Exception:
                close_val = close_val.iloc[:, 0]

    # If series elements themselves are sequences (e.g., lists/arrays), try to extract first element
    if isinstance(close_val, pd.Series):
        try:
            if close_val.apply(lambda x: hasattr(x, "__len__") and not isinstance(x, (str, bytes))).any():
                close_val = close_val.apply(lambda x: x[0] if hasattr(x, "__len__") and not isinstance(x, (str, bytes)) else x)
        except Exception:
            pass

    # Coerce to numeric and try to assign the Close column robustly
    try:
        numeric_close = pd.to_numeric(close_val, errors="coerce")
    except Exception as e:
        out["details"]["error"] = f"Failed to coerce Close to numeric: {e}"
        return out

    try:
        # Preferred way: use .loc to set a top-level 'Close' column
        h.loc[:, "Close"] = numeric_close
    except Exception:
        # Fallback: concat a new single-level 'Close' column DataFrame
        try:
            h = pd.concat([h.copy(), pd.DataFrame({"Close": numeric_close}, index=h.index)], axis=1)
        except Exception as e:
            out["details"]["error"] = f"Could not set Close column in hist_daily: {e}"
            return out

    # Ensure Close actually exists before calling dropna
    if "Close" not in h.columns:
        out["details"]["error"] = "Could not set Close column in hist_daily."
        out["details"]["columns"] = str(list(h.columns))
        out["details"]["type"] = str(type(h))
        return out

    # Safely drop NaNs from Close (catch KeyError if columns are unexpected)
    try:
        h = h
    except KeyError as e:
        out["details"]["error"] = f"dropna(subset=['Close']) failed: {e}"
        out["details"]["columns"] = str(list(h.columns))
        out["details"]["type"] = str(type(h))
        try:
            out["details"]["sample"] = h.head(5).to_dict()
        except Exception:
            pass
        return out

    try:
        h = h.sort_values("Date")
    except Exception as e:
        out["details"]["error"] = f"sort_values('Date') failed: {e}"
        out["details"]["columns"] = str(list(h.columns))
        return out

    if len(h) < 2:
        return out

    prev = h.iloc[-2]
    last5 = h.tail(5)

    out.update({
        "ok": True,
        "prev_high": float(prev.get("High", float("nan"))),
        "prev_low": float(prev.get("Low", float("nan"))),
        "prev_close": float(prev.get("Close", float("nan"))),
        "wk_high": float(last5["High"].max()) if "High" in last5.columns else float("nan"),
        "wk_low": float(last5["Low"].min()) if "Low" in last5.columns else float("nan"),
    })
    return out


@safe_cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    # --- Normalize MultiIndex columns from yfinance (e.g., ("Close","AAPL")) ---
    if isinstance(df.columns, pd.MultiIndex):
        col_map = {}
        for col in df.columns:
            for target in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
                if target not in col_map and any(str(x).strip().lower() == target.lower() for x in col):
                    col_map[target] = col
        for target, tup in col_map.items():
            try:
                df[target] = df[tup]
            except Exception:
                pass

    df = df.reset_index()

    # Normalize columns to strings
    try:
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        df.columns = [str(c) for c in df.columns]

    # Ensure Close exists (handle variants)
    if "Close" not in df.columns:
        for candidate in ["Adj Close", "close", "adj close", "AdjClose", "adjclose"]:
            if candidate in df.columns:
                df["Close"] = df[candidate]
                break

    # Last resort: first column containing 'close'
    if "Close" not in df.columns:
        close_like = [c for c in df.columns if "close" in str(c).lower()]
        if close_like:
            df["Close"] = df[close_like[0]]

    if "Close" not in df.columns:
        return pd.DataFrame()

    # Coerce numerics and ensure 1-D Series (not a DataFrame)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            v = df[col]
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0] if v.shape[1] else pd.Series(dtype=float)
            df[col] = pd.to_numeric(v, errors="coerce")

    df = df
    return df


def compute_opening_range(intra: pd.DataFrame, minutes: int = 30) -> dict:
    """Opening range for the most recent session in intraday dataframe."""
    out = {'ok': False, 'details': {}}
    if intra is None or intra.empty:
        return out
    df = intra.copy()
    # Find the last session date
    dt_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col, "Close"])
    df["session_date"] = df[dt_col].dt.date
    last_day = df["session_date"].max()
    d = df[df["session_date"] == last_day].copy()
    if d.empty:
        return out
    d = d.sort_values(dt_col)

    # Market open approximation: first bar in dataset
    start = d.iloc[0][dt_col]
    end = start + pd.Timedelta(minutes=minutes)
    or_df = d[(d[dt_col] >= start) & (d[dt_col] < end)]
    if or_df.empty:
        return out

    out.update({
        "ok": True,
        "session_date": str(last_day),
        "or_high": float(or_df["High"].max()) if "High" in or_df.columns else float("nan"),
        "or_low": float(or_df["Low"].min()) if "Low" in or_df.columns else float("nan"),
        "last_close": float(d.iloc[-1]["Close"]),
    })
    return out


def _as_1d_series(x):
    """Ensure a 1-D Series (handles 1-col DataFrame / duplicate-col selection)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            x = x.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    return x if isinstance(x, pd.Series) else pd.Series(x)


def structure_label(hist_daily: pd.DataFrame, lookback: int = 40) -> dict:
    """Simple HH/HL vs LH/LL structure using swing points.

    Robust to:
      - MultiIndex columns (yfinance)
      - duplicate column names (Close/High/Low returning a DataFrame)
      - Close-only inputs (falls back gracefully)
    """
    out = {"ok": False, "label": "N/A"}
    if hist_daily is None or hist_daily.empty:
        return out

    h = hist_daily.copy()

    # If MultiIndex columns, flatten to strings like "Close AAPL"
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = [" ".join([str(v) for v in tup if v is not None]).strip() for tup in h.columns]

    # Strip column labels
    try:
        h.columns = [str(c).strip() for c in h.columns]
    except Exception:
        pass

    # Remove duplicated column names (prevents h["High"] returning a DataFrame)
    if getattr(h.columns, "duplicated", None) is not None and h.columns.duplicated().any():
        h = h.loc[:, ~h.columns.duplicated(keep="first")]

    # If Date not present, try to recover it from index
    if "Date" not in h.columns:
        h = h.reset_index().rename(columns={"index": "Date"})

    # Ensure we have required columns; if not, try common alternates
    if "High" not in h.columns or "Low" not in h.columns or "Close" not in h.columns:
        # yfinance sometimes uses "Adj Close"
        if "Close" not in h.columns:
            for alt in ["Adj Close", "adj close", "close", "Adj_Close", "adjclose"]:
                if alt in h.columns:
                    h["Close"] = h[alt]
                    break

    # Hard requirement: need High/Low/Close for pivots
    if not all(c in h.columns for c in ["High", "Low", "Close"]):
        return out

    h = h.sort_values("Date").tail(int(lookback)).reset_index(drop=True)

    # Numeric conversion (force 1-D)
    for c in ["High", "Low", "Close"]:
        s = _as_1d_series(h[c])
        h[c] = pd.to_numeric(s, errors="coerce")

    h = h.dropna(subset=["High", "Low", "Close"])
    if len(h) < 10:
        return out

    # Pivot detection
    win = 2
    piv_hi = []
    piv_lo = []
    for i in range(win, len(h) - win):
        hi = float(h.loc[i, "High"])
        lo = float(h.loc[i, "Low"])
        if hi == float(h["High"].iloc[i-win:i+win+1].max()):
            piv_hi.append((i, hi))
        if lo == float(h["Low"].iloc[i-win:i+win+1].min()):
            piv_lo.append((i, lo))

    piv_hi = piv_hi[-3:]
    piv_lo = piv_lo[-3:]

    label = "RANGE / UNCLEAR 😴"
    if len(piv_hi) >= 2 and len(piv_lo) >= 2:
        hh = piv_hi[-1][1] > piv_hi[-2][1]
        hl = piv_lo[-1][1] > piv_lo[-2][1]
        lh = piv_hi[-1][1] < piv_hi[-2][1]
        ll = piv_lo[-1][1] < piv_lo[-2][1]
        if hh and hl:
            label = "BULL STRUCTURE 📈 (HH + HL)"
        elif lh and ll:
            label = "BEAR STRUCTURE 📉 (LH + LL)"
        elif hh and not hl:
            label = "RISKY UPTREND ⚠️ (HH but no HL)"
        elif ll and not lh:
            label = "RISKY DOWNTREND ⚠️ (LL but no LH)"

    out.update({
        "ok": True,
        "label": label,
        "pivot_highs": piv_hi,
        "pivot_lows": piv_lo,
    })
    return out


def build_trade_bias(trend_label: str, gex_regime: str, iv_rank_proxy: float | None) -> str:
    """Simple trade bias text. Educational only."""
    bias = []
    if "UPTREND" in trend_label:
        bias.append("Bias: **Bullish** → favor call spreads / put sells (defined risk).")
    elif "DOWNTREND" in trend_label:
        bias.append("Bias: **Bearish** → favor put spreads / call sells (defined risk).")
    else:
        bias.append("Bias: **Neutral/Chop** → favor premium-selling structures (iron condor / butterflies) when IV is elevated.")

    if "NEGATIVE" in gex_regime:
        bias.append("GEX regime: **Negative gamma** → expect faster moves + whipsaws; size smaller, use defined risk.")
    elif "PIN" in gex_regime:
        bias.append("GEX regime: **Pin/Mean-revert** → mean-reversion near big strikes can work better than breakouts.")

    if iv_rank_proxy is not None:
        if iv_rank_proxy >= 70:
            bias.append("IV is **high** (proxy) → buying naked options is harder; spreads/premium-selling often fit better.")
        elif iv_rank_proxy <= 30:
            bias.append("IV is **low** (proxy) → directional option buying can be more reasonable (still manage risk).")

    return "\n".join(bias)


def confidence_score(trend_strength: int, structure_ok: bool, vol_ok: bool, gex_ok: bool, or_ok: bool) -> int:
    score = 0
    score += min(40, int(trend_strength * 0.4))
    score += 15 if structure_ok else 0
    score += 15 if vol_ok else 0
    score += 20 if gex_ok else 0
    score += 10 if or_ok else 0
    return int(min(100, score))

def main():
    st.markdown(
        """
    <div class="header">
        <h1>📊 Stats Dashboard</h1>
        <p>Options chain + Weekly Gamma / GEX (dealer positioning) + Filters</p>
    </div>
    """,
        unsafe_allow_html=True
    )

    api_ok = check_api()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Options Query")

        symbol = st.text_input("Symbol", value="AAPL").upper().strip()
        default_expiry = dt.date.fromisoformat("2026-01-16")
        expiry_date = st.date_input(
            "Expiration Date",
            value=default_expiry,
            help="Pick the option expiry date.",
        )
        date = expiry_date.isoformat()
        spot_input = st.number_input("Spot Price (manual fallback)", value=260.00, step=0.50)
        use_yahoo_spot = st.checkbox("Use live Yahoo spot (recommended)", value=True)
        yahoo_spot = get_spot_from_yahoo(symbol) if use_yahoo_spot else None
        if yahoo_spot is not None:
            st.caption(f"Yahoo spot: {float(yahoo_spot):,.2f}")
        else:
            if use_yahoo_spot:
                st.caption("Yahoo spot: unavailable (using manual spot).")
        spot = float(yahoo_spot) if yahoo_spot is not None else float(spot_input)

        fetch_btn = st_btn("🔄 Fetch Data", disabled=not api_ok)

        st.markdown("---")
        st.markdown("### 🔥 Quick Symbols")
        popular = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "AMZN"]
        cols = st.columns(3)
        for i, s in enumerate(popular):
            with cols[i % 3]:
                if st_btn(s, key=f"q_{s}"):
                    st.session_state["symbol_override"] = s

        if "symbol_override" in st.session_state:
            symbol = st.session_state["symbol_override"]

        st.markdown("---")
        if api_ok:
            st.markdown('<div class="status-ok">✓ API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">✗ API Offline</div>', unsafe_allow_html=True)

        st.caption(f"Backend: {API_BASE_URL}")
        if API_BASE_URL.startswith("http://localhost"):
            st.caption("Tip: On Streamlit Cloud, set API_BASE_URL in Secrets (App → Settings → Secrets).")

    if not api_ok:
        st.error(
            f"Cannot connect to API at `{API_BASE_URL}`.\n\n"
            f"Local backend example:\n"
            f"`uvicorn api:app --port 8000 --reload`"
        )
        return

    if fetch_btn or st.session_state.get("last_fetch"):
        st.session_state["last_fetch"] = {"symbol": symbol, "date": date, "spot": spot}

        with st.spinner(f"Scraping {symbol} options for {date}..."):
            options_result = fetch_options(symbol, date)

        with st.spinner(f"Computing Weekly Gamma/GEX for {symbol} {date} (spot={spot})..."):
            weekly_result = fetch_weekly_summary(symbol, date, spot)

        if not options_result.get("success"):
            st.error(f"Options error: {options_result.get('error')}")
            return
        if not weekly_result.get("success"):
            st.error(f"Weekly summary error: {weekly_result.get('error')}")
            return

        api_data = options_result["data"]
        df = pd.DataFrame(api_data.get("data", []))

        w = weekly_result["data"]
        totals = w.get("totals", {}) or {}
        pcr = w.get("pcr", {}) or {}
        top = w.get("top_strikes", {}) or {}

        top_call = pd.DataFrame(top.get("call_gex", []) or [])
        top_put = pd.DataFrame(top.get("put_gex", []) or [])
        top_net = pd.DataFrame(top.get("net_gex_abs", []) or [])

        # ---------------- PRICE + MOVING AVERAGES ----------------
        hist_df = None
        with st.expander("📈 Price + Moving Averages (15/20/30/45/60 days)", expanded=True):
            hist_df = get_price_history_from_yahoo(symbol, period="6mo", interval="1d")
            if hist_df is None or hist_df.empty:
                st.info("Price history unavailable from Yahoo (moving averages not shown).")
            else:
                hist_df = hist_df.sort_values("Date").reset_index(drop=True)
                for w_ in [15, 20, 30, 45, 60]:
                    hist_df[f"MA{w_}"] = hist_df["Close"].rolling(window=w_, min_periods=1).mean()

                fig_px = go.Figure()
                fig_px.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Close"], name="Close"))
                for w_ in [15, 20, 30, 45, 60]:
                    fig_px.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df[f"MA{w_}"], name=f"MA{w_}"))

                # Mark the spot used by the app (Yahoo/manual)
                try:
                    fig_px.add_hline(y=float(spot), line_dash="dot", annotation_text="Spot used",
                                     annotation_position="top left")
                except Exception:
                    pass

                fig_px.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend_title="Series",
                )
                st_plot(fig_px)

                st.caption(
                    "Moving averages smooth price action: shorter MAs react faster (15/20), longer MAs react slower (45/60). "
                    "Crossovers and slope help label short-term vs long-term trend."
                )

        st.success(f"✓ Loaded {len(df)} strikes for **{symbol}** expiring **{date}**")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📋 Options Chain", "📊 OI Charts", "📌 Weekly Gamma / GEX", "🧲 Gamma Map + Filters", "🧮 Volatility & Greeks", "🏆 Pro Edge"]
        )

        with tab1:
            st_df(df, height=520)

        with tab2:
            required_cols = {"Strike", "Call OI", "Put OI"}
            if not required_cols.issubset(set(df.columns)):
                st.warning(
                    f"Options data is missing expected columns: {sorted(list(required_cols - set(df.columns)))}.\n\n"
                    "Backend must return: Strike, Call OI, Put OI"
                )
            else:
                bar_fig, line_fig = create_oi_charts(df)
                st.subheader("📈 Open Interest Comparison")
                st_plot(line_fig)
                st.subheader("📊 Open Interest Distribution")
                st_plot(bar_fig)

        with tab3:
            st.subheader("📌 Weekly Gamma / GEX (Dealer Positioning)")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Put/Call Ratio (OI)", f"{(pcr.get('oi') or 0):.3f}" if pcr.get("oi") is not None else "N/A")
            c2.metric("Put/Call Ratio (Volume)",
                      f"{(pcr.get('volume') or 0):.3f}" if pcr.get("volume") is not None else "N/A")
            c3.metric("Total Net GEX", f"{(totals.get('net_gex') or 0):,.0f}")
            c4.metric("Spot Used", f"{float(w.get('spot') or spot):,.2f}")

            st.markdown("### 🧲 Top Strikes (Gamma Walls / Magnets)")

            colA, colB, colC = st.columns(3)

            with colA:
                st.markdown("**Top Call GEX**")
                if not top_call.empty:
                    st_df(top_call)
                    if {"strike", "call_gex"}.issubset(top_call.columns):
                        st_plot(create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX"))
                else:
                    st.info("No top call GEX data returned.")

            with colB:
                st.markdown("**Top Put GEX**")
                if not top_put.empty:
                    st_df(top_put)
                    if {"strike", "put_gex"}.issubset(top_put.columns):
                        st_plot(create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX"))
                else:
                    st.info("No top put GEX data returned.")

            with colC:
                st.markdown("**Top Net GEX (abs)**")
                if not top_net.empty:
                    st_df(top_net)
                    if {"strike", "net_gex"}.issubset(top_net.columns):
                        st_plot(create_top_strikes_chart(top_net, "strike", "net_gex", "Top Net GEX (abs)"))
                else:
                    st.info("No top net GEX data returned.")

            st.caption("Note: GEX is an approximation from IV + OI using Black-Scholes gamma; educational only.")

        with tab4:
            st.subheader("🧭 Gamma Map (Magnets / Walls / Box)")

            with st.spinner("Loading per-strike GEX (weekly/gex) ..."):
                gex_result = fetch_weekly_gex(symbol, date, spot)

            if not gex_result.get("success"):
                st.warning(f"Could not load /weekly/gex: {gex_result.get('error')}")
            else:
                gex_payload = gex_result["data"]
                gex_df = pd.DataFrame(gex_payload.get("data", []) or [])

                if gex_df.empty:
                    st.warning("No per-strike GEX returned from backend.")
                else:
                    levels = build_gamma_levels(gex_df, spot=spot, top_n=5)
                    if not levels:
                        st.warning("Could not compute gamma levels.")
                    else:
                        lower = levels["gamma_box"]["lower"]
                        upper = levels["gamma_box"]["upper"]

                        cA, cB, cC = st.columns(3)
                        cA.metric("Main Magnet", f"{float(levels['magnets'].iloc[0]['strike']):g}" if not levels[
                            "magnets"].empty else "N/A")
                        cB.metric("Lower Wall", f"{lower:g}" if lower is not None else "N/A")
                        cC.metric("Upper Wall", f"{upper:g}" if upper is not None else "N/A")

                        st_plot(plot_net_gex_map(gex_df, spot=spot, levels=levels))

            st.markdown("---")
            st.subheader("📈 Noise Filters (McGinley / KAMA / Kalman)")

            period = st.selectbox("History Period", ["3mo", "6mo", "1y", "2y"], index=1)
            interval = st.selectbox("Interval", ["1d", "1h", "30m"], index=0)

            c1, c2, c3 = st.columns(3)
            with c1:
                length_md = st.number_input("McGinley Length", min_value=3, max_value=200, value=14, step=1)
            with c2:
                kama_er = st.number_input("KAMA ER Length", min_value=2, max_value=200, value=10, step=1)
            with c3:
                kama_fast = st.number_input("KAMA Fast", min_value=2, max_value=50, value=2, step=1)

            kama_slow = st.number_input("KAMA Slow", min_value=int(kama_fast) + 1, max_value=300, value=30, step=1)

            st.markdown("### Kalman settings (advanced)")
            k1, k2 = st.columns(2)
            with k1:
                kf_q = st.number_input("Process variance Q", value=1e-5, format="%.8f")
            with k2:
                kf_r = st.number_input("Measurement variance R", value=1e-2, format="%.6f")

            with st.spinner(f"Loading {symbol} price history..."):
                px = fetch_price_history(symbol, period=period, interval=interval)

            if px.empty or "Close" not in px.columns:
                st.error("No price data returned. Try a different symbol/period/interval.")
            else:
                fig2, kf_series = plot_filters(px, int(length_md), int(kama_er), int(kama_fast), int(kama_slow),
                                               float(kf_q), float(kf_r))
                st_plot(fig2)

                # ✅ Kalman “what it says” message
                km = kalman_message(px["Close"].values, kf_series.values, lookback=20, band_pct=0.003)
                st.markdown(
                    f"""
**Kalman Read:** {km['msg']}

- **Regime:** **{km.get('regime', 'N/A')}**
- **Trend:** **{km.get('trend', 'N/A')}**
- **Bias:** **{km.get('bias', 'N/A')}**
- **Trend strength:** **{km.get('trend_strength', 'N/A')}**
- **Structure:** **{km.get('structure', 'N/A')}**
- **Chop (crossings/{km.get('lookback', 20)}):** **{km.get('crossings', 'N/A')}**
- **Confidence:** **{km.get('confidence', 'N/A')}**

**Why this label?**{km.get('why', '')}

**Notes:**- "UPTREND + price below Kalman" often = *pullback inside an uptrend* (watch for reclaim).
- "DOWNTREND + price below Kalman" often = *sell-the-rip* behavior (Kalman acts as resistance).
- Higher crossings = more range/chop → mean-reversion works better than breakout.
"""
                )

                st.caption(
                    "Tip: McGinley adapts to speed, KAMA adapts via Efficiency Ratio, Kalman adapts via Q/R confidence.")

        with tab5:
            st.subheader("🧮 Volatility & Greeks (from this expiry chain)")

            if df.empty:
                st.info("No options data loaded yet.")
            else:
                # --- Greeks inputs (used only if backend doesn't provide greeks)
                with st.expander('Greek Inputs (Black-Scholes fallback)', expanded=False):
                    # If your backend doesn't provide greeks, we can compute them from IV using Black-Scholes.
                    # Inputs:
                    #   r = risk-free rate (annual)
                    #   q = dividend yield (annual; 0 if you want to ignore dividends)
                    #   spot = current underlying price used for greeks
                    r_in = st.number_input('Risk-free rate r (annual, decimal)', value=0.041, step=0.001, format='%.4f')
                    q_in = st.number_input('Dividend yield q (annual, decimal)', value=0.004, step=0.001, format='%.4f')

                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        use_yahoo_spot = st.checkbox('Use live Yahoo Finance spot for Greeks', value=True)
                    with col_b:
                        spot_override = st.number_input('Spot override (0 = auto)', value=0.0, step=0.1, format='%.2f')

                    use_trading_days = st.checkbox('Use trading-day year (252) for T (otherwise calendar 365)',
                                                   value=False)

                    yahoo_spot = None
                    if use_yahoo_spot:
                        # cache per-symbol per-session so we don't spam Yahoo
                        cache_key = f"yahoo_spot_{symbol}"
                        if cache_key in st.session_state and st.session_state[cache_key]:
                            yahoo_spot = st.session_state[cache_key]
                        else:
                            yahoo_spot = get_spot_from_yahoo(symbol)
                            if yahoo_spot:
                                st.session_state[cache_key] = float(yahoo_spot)

                        if yahoo_spot:
                            st.caption(f"Yahoo spot for **{symbol}**: **{float(yahoo_spot):.2f}**")
                        else:
                            st.caption(
                                "Yahoo spot unavailable (network/blocked). Falling back to backend/override spot.")

                spot_for_greeks = None
                # priority: manual override > yahoo > backend spot
                if spot_override and float(spot_override) > 0:
                    spot_for_greeks = float(spot_override)
                elif yahoo_spot and float(yahoo_spot) > 0:
                    spot_for_greeks = float(yahoo_spot)
                else:
                    spot_for_greeks = float(spot)

                    spot_override_val = float(spot_override) if spot_override and float(spot_override) > 0 else None
                    yahoo_spot_val = float(yahoo_spot) if yahoo_spot and float(yahoo_spot) > 0 else None
                    backend_spot_val = float(spot) if spot is not None and str(spot).strip() != '' else None

                    if spot_override_val is not None:
                        spot_source = 'Override'
                    elif yahoo_spot_val is not None:
                        spot_source = 'Yahoo'
                    elif backend_spot_val is not None:
                        spot_source = 'Backend'
                    else:
                        spot_source = 'Fallback'

                    st.markdown('#### Spot used for Greeks')
                    st.write({
                        'Override': spot_override_val,
                        'Yahoo': yahoo_spot_val,
                        'Backend': backend_spot_val,
                        'Using': spot_source,
                        'Spot used (S)': float(spot_for_greeks) if spot_for_greeks is not None else None,
                    })
                    if use_yahoo_spot and spot_source != 'Yahoo':
                        st.warning(
                            'Yahoo spot was enabled but unavailable, so Greeks are using a fallback spot. Install yfinance or allow Yahoo endpoints if blocked.')

                # Assume equity options expire at market close (4:00pm local) on the selected expiry date
                _now_ts = pd.Timestamp.now()
                _exp_ts = pd.Timestamp(date) + pd.Timedelta(hours=16)
                if use_trading_days:
                    days = max(int((_exp_ts.normalize() - _now_ts.normalize()).days), 0)
                    T = max(days / 252.0, 1e-6)
                else:
                    T = max(float((_exp_ts - _now_ts).total_seconds()) / (365.0 * 24 * 3600), 1e-6)

                fig_iv, fig_greeks, atm = plot_iv_and_greeks(df, spot=spot_for_greeks, T=T, r=float(r_in),
                                                             q=float(q_in))

                if not atm:
                    st.warning("Could not compute ATM snapshot (Strike column missing or invalid).")
                else:
                    atm_strike = atm.get("atm_strike")
                    st.markdown(f"**ATM strike (nearest to spot):** `{atm_strike:g}`")

                    # show a few key ATM metrics if present
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Call IV", f"{atm.get('Call IV', float('nan')):.4f}" if "Call IV" in atm else "N/A")
                    m2.metric("Put IV", f"{atm.get('Put IV', float('nan')):.4f}" if "Put IV" in atm else "N/A")
                    m3.metric("Call Delta",
                              f"{atm.get('Call Delta', float('nan')):.3f}" if "Call Delta" in atm else "N/A")
                    m4.metric("Put Delta", f"{atm.get('Put Delta', float('nan')):.3f}" if "Put Delta" in atm else "N/A")

                    st.markdown("### 🧠 How to read the Greeks for **this** ATM strike (spot up/down, benefits & risks)")

                    # Pull values if available (from backend or Black-Scholes fallback)
                    spot_used = float(spot_for_greeks) if spot_for_greeks is not None else float("nan")
                    K = float(atm_strike) if atm_strike is not None else float("nan")

                    call_delta = float(atm.get("Call Delta", float("nan"))) if isinstance(atm, dict) else float("nan")
                    put_delta = float(atm.get("Put Delta", float("nan"))) if isinstance(atm, dict) else float("nan")
                    gamma = float(atm.get("Gamma", float("nan"))) if isinstance(atm, dict) else float("nan")
                    vega = float(atm.get("Vega", float("nan"))) if isinstance(atm, dict) else float(
                        "nan")  # per 1.00 (100%) IV
                    call_theta = float(atm.get("Call Theta", float("nan"))) if isinstance(atm, dict) else float(
                        "nan")  # per year
                    put_theta = float(atm.get("Put Theta", float("nan"))) if isinstance(atm, dict) else float(
                        "nan")  # per year

                    # Common unit conversions
                    vega_per_1pct = vega / 100.0 if pd.notna(vega) else float("nan")
                    call_theta_per_day = call_theta / 365.0 if pd.notna(call_theta) else float("nan")
                    put_theta_per_day = put_theta / 365.0 if pd.notna(put_theta) else float("nan")

                    # Scenario helpers (very rough: "all else equal")
                    def _fmt(x, fmt):
                        try:
                            return format(float(x), fmt) if pd.notna(x) else "N/A"
                        except Exception:
                            return "N/A"

                    dS_1 = 1.0
                    call_move_up_1 = call_delta * dS_1 if pd.notna(call_delta) else float("nan")
                    put_move_up_1 = put_delta * dS_1 if pd.notna(put_delta) else float("nan")
                    call_move_dn_1 = -call_delta * dS_1 if pd.notna(call_delta) else float("nan")
                    put_move_dn_1 = -put_delta * dS_1 if pd.notna(put_delta) else float("nan")

                    # Gamma effect: delta changes by ~ Gamma * ΔS
                    call_delta_up_1 = call_delta + gamma * dS_1 if (
                                pd.notna(call_delta) and pd.notna(gamma)) else float("nan")
                    call_delta_dn_1 = call_delta - gamma * dS_1 if (
                                pd.notna(call_delta) and pd.notna(gamma)) else float("nan")

                    # Vega effect: 1% IV move ≈ vega/100
                    iv_bump_1pct = vega_per_1pct if pd.notna(vega_per_1pct) else float("nan")

                    st.markdown(f"""
                    **Inputs used for Greeks**
                    - Spot used (S): `{_fmt(spot_used, '.2f')}`
                    - ATM strike (K): `{_fmt(K, '.0f')}`

                    **ATM Greeks (approx)**
                    - Call Δ: `{_fmt(call_delta, '.3f')}`  |  Put Δ: `{_fmt(put_delta, '.3f')}`
                    - Γ (Gamma): `{_fmt(gamma, '.5f')}`
                    - Vega: `{_fmt(vega, '.3f')}` per 1.00 IV  (**≈ `{_fmt(vega_per_1pct, '.3f')}` per +1% IV**)
                    - Call Θ: `{_fmt(call_theta, '.3f')}`/yr (**≈ `{_fmt(call_theta_per_day, '.3f')}` per day**)
                    - Put  Θ: `{_fmt(put_theta, '.3f')}`/yr (**≈ `{_fmt(put_theta_per_day, '.3f')}` per day**)
                    """)

                    st.markdown("#### 📈 If spot moves UP or DOWN (rough P/L impact from Δ)")
                    st.markdown(f"""
                    - **Spot +$1**: Call ≈ `{_fmt(call_move_up_1, '.3f')}` | Put ≈ `{_fmt(put_move_up_1, '.3f')}`
                    - **Spot -$1**: Call ≈ `{_fmt(call_move_dn_1, '.3f')}` | Put ≈ `{_fmt(put_move_dn_1, '.3f')}`
                    """)

                    st.markdown("#### 🚀 Gamma: why winners speed up")
                    st.markdown(f"""
                    - After a **+$1** move, Call Δ becomes ~ `{_fmt(call_delta_up_1, '.3f')}` (more sensitive to further upside).
                    - After a **-$1** move, Call Δ becomes ~ `{_fmt(call_delta_dn_1, '.3f')}` (less sensitive; you “lose speed”).
                    """)

                    st.markdown("#### 🌪 Vega: what IV does to your option")
                    st.markdown(f"""
                    - **IV +1%** → option changes about **`{_fmt(iv_bump_1pct, '.3f')}`** (all else equal).
                    - **IV -1%** → loses about the same magnitude.
                    """)
                    st.write(
                        "ATM + longer-dated expiries usually have **bigger Vega**, so IV changes can matter a lot.")

                    st.markdown("#### ⏳ Theta: the daily rent")
                    st.markdown(f"""
                    - If price/IV stay flat, **Theta is what you bleed each day** as a long option.
                    - Approx daily decay here: Call ≈ `{_fmt(call_theta_per_day, '.3f')}` per day, Put ≈ `{_fmt(put_theta_per_day, '.3f')}` per day.
                    """)

                    st.markdown("#### ✅ Benefits vs ⚠️ Risks (for this strike near this spot)")
                    st.markdown("""
                    - ✅ **Benefit**: If spot moves your way, **Gamma** can increase Δ → you can gain faster if the move continues.
                    - ✅ **Benefit**: If IV rises (fear/news), **Vega** can add profit even without a huge spot move.
                    - ⚠️ **Risk**: If spot chops sideways, **Theta** bleeds value day after day.
                    - ⚠️ **Risk**: If IV drops (IV crush), you can lose value even if spot is near your strike.
                    - ⚠️ **Reminder**: These are **“all else equal”** approximations - in real trading, Δ/Γ/Vega/Θ move together.
                    """)

                    # ---------------- Matrix: multiple spot moves (Delta + Gamma) ----------------
                    st.subheader("📊 Spot Move Matrix (Delta + Gamma)")
                    if pd.notna(call_delta) and pd.notna(put_delta) and pd.notna(gamma) and pd.notna(spot_used):
                        df_matrix = _build_spot_move_matrix(float(spot_used), float(call_delta), float(put_delta),
                                                            float(gamma))
                        # Pretty formatting
                        df_matrix["New Spot"] = df_matrix["New Spot"].map(lambda x: round(float(x), 2))
                        df_matrix["Call Δ+Γ Est. Change"] = df_matrix["Call Δ+Γ Est. Change"].map(
                            lambda x: round(float(x), 3))
                        df_matrix["Put Δ+Γ Est. Change"] = df_matrix["Put Δ+Γ Est. Change"].map(
                            lambda x: round(float(x), 3))
                        st.dataframe(df_matrix, use_container_width=True, height=420)
                        st.caption(
                            "Approximation: Δ and Γ are held constant and IV/time are assumed unchanged. Bigger moves = less accurate.")
                    else:
                        st.info("Matrix unavailable (need valid Δ/Γ and spot).")

                    # ---------------- EOD Fibonacci Projection (today open -> spot) ----------------
                    st.subheader("📌 EOD Fibonacci Projection (today open -> current spot)")

                    today_open = get_today_open_from_yahoo(symbol)
                    if today_open is None or not (today_open > 0):
                        st.info(
                            "Today's open not available from Yahoo. EOD projection uses the best available open; if it stays missing, check network or yfinance.")
                    else:
                        # Use the same spot used for greeks (spot_used) if available, otherwise fall back to sidebar spot
                        try:
                            S_now = float(spot_used) if pd.notna(spot_used) else float(spot)
                        except Exception:
                            S_now = float(spot)

                        O = float(today_open)
                        direction = "UP" if S_now >= O else "DOWN"
                        impulse = abs(S_now - O)

                        # Multipliers for extension targets
                        mults = [1.0, 1.272, 1.618]

                        if impulse <= 0:
                            st.info(
                                "Impulse is 0 (spot equals open). EOD projection needs movement to project targets.")
                        else:
                            rows = []
                            for mlt in mults:
                                if direction == "UP":
                                    upper = S_now + mlt * impulse
                                    lower = S_now - mlt * impulse
                                else:
                                    lower = S_now - mlt * impulse
                                    upper = S_now + mlt * impulse
                                rows.append({
                                    "Band": f"{mlt:.3f}x",
                                    "Lower": round(lower, 2),
                                    "Upper": round(upper, 2),
                                    "Width ($)": round(upper - lower, 2),
                                })

                            proj_df = pd.DataFrame(rows)

                            c_e1, c_e2, c_e3 = st.columns(3)
                            c_e1.metric("Today Open", f"{O:,.2f}")
                            c_e2.metric("Current Spot", f"{S_now:,.2f}")
                            c_e3.metric("Impulse |S-O|", f"{impulse:,.2f}")

                            st.dataframe(proj_df, use_container_width=True, height=200)
                            st.caption(
                                "Interpretation: Uses today's open to measure the current impulse, then projects symmetric extension bands around spot. These are NOT guarantees - they are reference levels.")

                            # ATR check (daily)
                            hist_for_atr = get_price_history_from_yahoo(symbol, period="3mo", interval="1d")
                            atr14 = atr_14_from_history(hist_for_atr)
                            if atr14 is not None:
                                atr_low = S_now - atr14
                                atr_high = S_now + atr14
                                st.markdown("**ATR(14) reality check (daily expected range):**")
                                st.write({
                                    "ATR(14)": round(atr14, 2),
                                    "ATR Low": round(atr_low, 2),
                                    "ATR High": round(atr_high, 2),
                                })
                                st.caption(
                                    "ATR band is a sanity check: if EOD extension targets are far beyond ATR, they are less likely without a catalyst.")

                    # ---------------- Fibonacci: auto swing ranges from price history ----------------
                    st.subheader("🧵 Fibonacci Range (auto swing by lookback)")

                    # Fetch daily closes for fib (independent from the MA expander)
                    hist_fib = get_price_history_from_yahoo(symbol, period="6mo", interval="1d")
                    if hist_fib is not None and not hist_fib.empty and "Close" in hist_fib.columns:
                        lookbacks = [15, 20, 30, 45, 60]
                        fib_rows = []
                        for lb in lookbacks:
                            swing = _swing_high_low_from_history(hist_fib, lb)
                            if not swing:
                                continue
                            lo, hi = swing
                            out = _fib_levels_from_swing(lo, hi)
                            if not out:
                                continue
                            retr, ext = out
                            fib_rows.append({
                                "Lookback (days)": lb,
                                "Swing Low": round(lo, 2),
                                "Swing High": round(hi, 2),
                                "Upper 161.8% (End)": round(ext["Upper 161.8%"], 2),
                                "Lower -61.8% (End)": round(ext["Lower -61.8%"], 2),
                                "Key Retrace 61.8%": round(retr["61.8%"], 2),
                                "Key Retrace 38.2%": round(retr["38.2%"], 2),
                            })
                        if fib_rows:
                            fib_df = pd.DataFrame(fib_rows).sort_values("Lookback (days)")
                            # ---- EOD Fibonacci distances (from latest daily close) ----
                            try:
                                eod_close = float(hist_fib.sort_values("Date")["Close"].iloc[-1])
                            except Exception:
                                eod_close = None

                            if eod_close is not None:
                                fib_df["EOD Close"] = eod_close
                                # dollar and percent distance to the extension "end" levels
                                fib_df["To Upper 161.8% ($)"] = (fib_df["Upper 161.8% (End)"] - eod_close).round(2)
                                fib_df["To Upper 161.8% (%)"] = (
                                            (fib_df["Upper 161.8% (End)"] / eod_close - 1.0) * 100.0).round(2)
                                fib_df["To Lower -61.8% ($)"] = (fib_df["Lower -61.8% (End)"] - eod_close).round(2)
                                fib_df["To Lower -61.8% (%)"] = (
                                            (fib_df["Lower -61.8% (End)"] / eod_close - 1.0) * 100.0).round(2)

                            st.dataframe(fib_df, use_container_width=True, height=260)
                            st.caption(
                                "‘End’ levels are common extension targets from the lookback swing (High + 0.618×Range, Low - 0.618×Range). The EOD columns show distance from the latest daily close.")
                        else:
                            st.info("Could not compute fib ranges from history.")
                    else:
                        st.info(
                            "Price history not available - Fibonacci ranges require daily close history (Yahoo/Stooq).")

                st.caption(
                    "This tab uses backend greeks if provided. If greeks are missing, it computes greeks from IV using Black-Scholes (your r/q/spot inputs) and then builds a spot-move matrix + Fibonacci ranges.")
                if fig_iv is not None:
                    st_plot(fig_iv)
                else:
                    st.info(
                        "IV columns not found in your backend payload (look for columns like 'Call IV' / 'Put IV').")

                if fig_greeks is not None:
                    st_plot(fig_greeks)
                else:
                    st.info(
                        "Greeks columns not found (Delta/Gamma/Vega/Theta). If you add them to the backend, this tab will auto-plot them.")

                skew = approx_skew_25d(df)
                if skew:
                    st.markdown("### 📐 25-Delta Skew (rough)")
                    st.write(
                        f"- 25d Call: strike {skew['call_25d_strike']:g}, Δ={skew['call_25d_delta']:.3f}, IV={skew['call_25d_iv']:.4f}\\n"
                        f"- 25d Put:  strike {skew['put_25d_strike']:g}, Δ={skew['put_25d_delta']:.3f}, IV={skew['put_25d_iv']:.4f}\\n"
                        f"- **Skew (Call IV - Put IV)**: **{skew['skew_call_minus_put']:.4f}**"
                    )
                    st.caption("Skew helps you see if downside protection (puts) is getting expensive vs upside calls.")
        with tab6:
            st.subheader("🏆 Pro Edge (Trend + Volatility + Flow + Levels)")

            st.caption(
                "This page combines multiple *free* signals (trend, IV, structure, levels, and gamma regime) into a simple checklist + confidence score. "
                "Educational use only — not financial advice."
            )

            # ---------------- Resolve spot ----------------
            try:
                spot_now = float(spot_override) if spot_override and float(spot_override) > 0 else None
            except Exception:
                spot_now = None
            if spot_now is None:
                try:
                    spot_now = float(yahoo_spot) if yahoo_spot and float(yahoo_spot) > 0 else None
                except Exception:
                    spot_now = None
            if spot_now is None:
                try:
                    spot_now = float(spot) if spot is not None and str(spot).strip() != "" else None
                except Exception:
                    spot_now = None

            if spot_now is None or spot_now <= 0:
                st.warning("Spot price is not available. Enter a manual spot to unlock all calculations.")
                st.stop()

            # ---------------- Price history (daily) ----------------
            if hist_df is None or hist_df.empty:
                # fallback daily history
                try:
                    hist_daily = fetch_price_history(symbol, period="1y", interval="1d")
                    if "Date" not in hist_daily.columns:
                        hist_daily = hist_daily.reset_index().rename(columns={"index": "Date"})
                except Exception:
                    hist_daily = pd.DataFrame()
            else:
                hist_daily = hist_df.copy()

            # Ensure daily OHLCV if available via yfinance (for levels/volume)
            if hist_daily is not None and not hist_daily.empty:
                # If only Close exists, refresh with OHLCV
                need_ohlc = not set(["Open", "High", "Low", "Volume"]).issubset(set(hist_daily.columns))
                if need_ohlc:
                    try:
                        raw = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)
                        hist_daily = raw.reset_index()
                    except Exception:
                        pass

            # ---------------- Trend regime (MAs) ----------------
            ma = compute_ma_stack_and_regime(hist_daily)
            if ma["ok"]:
                c1, c2, c3 = st.columns([1.6, 1, 1])
                with c1:
                    st.markdown("### Trend (Moving Averages)")
                    st.write(ma["label"])
                with c2:
                    st.metric("Trend strength", f"{ma['strength']}/100")
                with c3:
                    rv20 = realized_vol_annualized(hist_daily, window=20)
                    st.metric("Realized vol (20d)", f"{rv20:.2%}" if pd.notna(rv20) else "N/A")
            else:
                st.warning("Not enough daily history to compute MA regime (need more candles).")

            # ---------------- Structure (HH/HL) ----------------
            st.markdown("### Structure (HH/HL vs LH/LL)")
            struct = structure_label(hist_daily, lookback=50)
            if struct["ok"]:
                st.write(struct["label"])
                with st.expander("Show recent swing pivots", expanded=False):
                    st.write({"pivot_highs": struct.get("pivot_highs"), "pivot_lows": struct.get("pivot_lows")})
            else:
                st.info("Structure label unavailable (insufficient candles).")

            # ---------------- Levels (support/resistance) ----------------
            st.markdown("### Key Levels (Support/Resistance)")
            lvl = compute_key_levels(hist_daily)
            if lvl["ok"]:
                level_df = pd.DataFrame([{
                    "Prev High": lvl["prev_high"],
                    "Prev Low": lvl["prev_low"],
                    "Prev Close": lvl["prev_close"],
                    "Weekly High (5d)": lvl["wk_high"],
                    "Weekly Low (5d)": lvl["wk_low"],
                }])
                st_df(level_df, height=80)

                # Distance from spot
                try:
                    dist_df = pd.DataFrame([{
                        "Spot": spot_now,
                        "To Prev High": lvl["prev_high"] - spot_now,
                        "To Prev Low": spot_now - lvl["prev_low"],
                        "To Wk High": lvl["wk_high"] - spot_now,
                        "To Wk Low": spot_now - lvl["wk_low"],
                    }])
                    st_df(dist_df, height=80)
                except Exception:
                    pass
            else:
                st.info("Levels unavailable.")

            # ---------------- Intraday Opening Range (30m) ----------------
            st.markdown("### Opening Range (first 30 minutes)")
            intra = fetch_intraday(symbol, period="5d", interval="5m")
            orr = compute_opening_range(intra, minutes=30)
            if orr["ok"]:
                or_high = orr["or_high"]
                or_low = orr["or_low"]
                st.write(f"Session: **{orr['session_date']}** | OR High: **{or_high:.2f}** | OR Low: **{or_low:.2f}**")
                if spot_now > or_high:
                    st.success("OR Breakout ↑ (spot above OR high)")
                elif spot_now < or_low:
                    st.error("OR Breakdown ↓ (spot below OR low)")
                else:
                    st.info("Inside Opening Range (chop risk)")
            else:
                st.info("Intraday data not available (this can happen on Streamlit Cloud / Yahoo blocks).")

            # ---------------- Gamma regime (GEX) ----------------
            st.markdown("### Gamma Regime (from your weekly GEX tab)")
            net_gex = None
            try:
                net_gex = float(totals.get("net_gex", totals.get("Net GEX", float("nan"))))
            except Exception:
                net_gex = None

            if net_gex is None or (isinstance(net_gex, float) and pd.isna(net_gex)):
                st.info("Weekly GEX totals not available from backend for this symbol/expiry.")
                gex_regime = "UNKNOWN"
            else:
                if net_gex < 0:
                    gex_regime = "NEGATIVE GAMMA 💥 (fast moves / whipsaws)"
                    st.error(f"{gex_regime} | Net GEX: {net_gex:,.0f}")
                elif net_gex > 0:
                    gex_regime = "POSITIVE GAMMA 🧱 (pin / mean-revert risk)"
                    st.success(f"{gex_regime} | Net GEX: {net_gex:,.0f}")
                else:
                    gex_regime = "NEUTRAL GAMMA"
                    st.write(f"{gex_regime} | Net GEX: {net_gex:,.0f}")

            # ---------------- IV + Vanna + Charm (approx) ----------------
            st.markdown("### IV + Vanna + Charm (Approx)")
            atm_iv = None

            # 1) Try from your loaded chain df
            try:
                if not df.empty and "Strike" in df.columns:
                    tmp = df.copy()
                    tmp["Strike"] = pd.to_numeric(tmp["Strike"], errors="coerce")
                    tmp = tmp.dropna(subset=["Strike"])
                    tmp["dist"] = (tmp["Strike"] - float(spot_now)).abs()
                    atm_row = tmp.sort_values("dist").iloc[0]
                    c_iv = pd.to_numeric(atm_row.get("Call IV", np.nan), errors="coerce")
                    p_iv = pd.to_numeric(atm_row.get("Put IV", np.nan), errors="coerce")
                    if pd.notna(c_iv) and pd.notna(p_iv):
                        atm_iv = float((c_iv + p_iv) / 2.0)
                    elif pd.notna(c_iv):
                        atm_iv = float(c_iv)
                    elif pd.notna(p_iv):
                        atm_iv = float(p_iv)
            except Exception:
                pass

            # 2) Fallback: yfinance option chain (if available)
            if atm_iv is None:
                try:
                    tkr = yf.Ticker(symbol)
                    oc = tkr.option_chain(date)
                    calls = oc.calls.copy()
                    puts = oc.puts.copy()
                    calls["dist"] = (pd.to_numeric(calls["strike"], errors="coerce") - float(spot_now)).abs()
                    puts["dist"] = (pd.to_numeric(puts["strike"], errors="coerce") - float(spot_now)).abs()
                    c_atm = calls.sort_values("dist").iloc[0]
                    p_atm = puts.sort_values("dist").iloc[0]
                    civ = float(c_atm.get("impliedVolatility", float("nan")))
                    piv = float(p_atm.get("impliedVolatility", float("nan")))
                    if pd.notna(civ) and pd.notna(piv):
                        atm_iv = float((civ + piv) / 2.0)
                except Exception:
                    pass

            if atm_iv is None or atm_iv <= 0:
                st.warning("ATM IV not available (free feeds sometimes block this). IV/Vanna/Charm section will be limited.")
                iv_rank = {"ok": False}
                iv_rank_proxy = None
            else:
                iv_rank = iv_proxy_rank(atm_iv, hist_daily, window=20)
                iv_rank_proxy = float(iv_rank["iv_proxy_rank"]) if iv_rank.get("ok") else None

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("ATM IV", f"{atm_iv:.2%}")
                with c2:
                    st.metric("IV Rank (proxy)", f"{iv_rank_proxy:.0f}/100" if iv_rank_proxy is not None else "N/A")
                with c3:
                    rv = realized_vol_annualized(hist_daily, window=20)
                    st.metric("RV20 vs IV", f"{rv/atm_iv:.2f}x" if (pd.notna(rv) and atm_iv > 0) else "N/A")

                # Vanna/Charm for ATM option (call+put average)
                try:
                    expiry_dt = datetime.strptime(date, "%Y-%m-%d")
                    expiry_dt = expiry_dt.replace(hour=16, minute=0, second=0)
                    now_dt = datetime.now(TZ)
                    T = max(0.0, (expiry_dt - now_dt).total_seconds() / (365.0 * 24 * 3600))
                    K = float(round(spot_now))  # proxy ATM strike
                    r = 0.04  # rough default risk-free rate
                    call_vc = bs_vanna_charm(spot_now, K, T, r, atm_iv, True)
                    put_vc = bs_vanna_charm(spot_now, K, T, r, atm_iv, False)

                    vanna_avg = (call_vc["vanna"] + put_vc["vanna"]) / 2.0
                    charm_day_avg = (call_vc["charm_per_day"] + put_vc["charm_per_day"]) / 2.0

                    st.write({
                        "ATM strike (proxy K)": K,
                        "T (years)": round(T, 4),
                        "Delta (call)": round(call_vc["delta"], 4),
                        "Delta (put)": round(put_vc["delta"], 4),
                        "Vanna (avg)": round(vanna_avg, 6),
                        "Charm per day (avg)": round(charm_day_avg, 6),
                    })
                    st.caption("Interpretation: Vanna relates to hedging flows when IV changes; Charm relates to delta drift as time passes (strong near expiry).")
                except Exception:
                    st.info("Vanna/Charm not computed (date/time parsing issue).")

            # ---------------- Volume confirmation (daily) ----------------
            st.markdown("### Volume Confirmation (daily)")
            vol_ok = False
            try:
                if "Volume" in hist_daily.columns:
                    v = pd.to_numeric(hist_daily["Volume"], errors="coerce").dropna()
                    if len(v) >= 21:
                        v_last = float(v.iloc[-1])
                        v_avg = float(v.iloc[-21:-1].mean())
                        vol_ok = v_last > v_avg
                        st.write({"Last volume": v_last, "20d avg (prev)": v_avg, "Above avg?": vol_ok})
            except Exception:
                pass
            if not vol_ok:
                st.caption("If volume is below average, breakouts can fail more often (chop risk).")

            # ---------------- Trade bias + checklist ----------------
            st.markdown("### Trade Bias + Checklist")
            trend_label = ma["label"] if ma.get("ok") else "N/A"
            bias_text = build_trade_bias(trend_label, gex_regime, iv_rank_proxy)
            st.markdown(bias_text)

            checklist = []
            trend_ok = ("UPTREND" in trend_label) or ("DOWNTREND" in trend_label)
            structure_ok = struct.get("ok") and ("BULL STRUCTURE" in struct.get("label", "") or "BEAR STRUCTURE" in struct.get("label", ""))
            gex_ok = (gex_regime != "UNKNOWN")
            or_ok = bool(orr.get("ok")) and (spot_now > orr.get("or_high", float("inf")) or spot_now < orr.get("or_low", float("-inf")))

            # IV suitability (proxy)
            if iv_rank_proxy is None:
                iv_ok = False
            else:
                # For weeklies: mid IV is often easiest; very high IV = prefer spreads, very low IV = prefer buying
                iv_ok = 30 <= iv_rank_proxy <= 80

            checklist.append(("Trend regime defined", trend_ok))
            checklist.append(("Structure confirms (HH/HL or LH/LL)", structure_ok))
            checklist.append(("Key levels computed", lvl.get("ok", False)))
            checklist.append(("Opening range signal available", orr.get("ok", False)))
            checklist.append(("Volume above average", vol_ok))
            checklist.append(("GEX regime available", gex_ok))
            checklist.append(("IV data available", iv_rank_proxy is not None))

            chk_df = pd.DataFrame([{"Item": k, "OK": v} for k, v in checklist])
            st_df(chk_df, height=260)

            score = confidence_score(ma.get("strength", 0), structure_ok, iv_ok, gex_ok, or_ok)
            st.markdown("### Confidence Score (0–100)")
            st.metric("Score", score)

            if score >= 70:
                st.success("Higher alignment across signals (still manage risk).")
            elif score >= 45:
                st.warning("Mixed alignment — trade smaller or wait for cleaner setup.")
            else:
                st.error("Low alignment — chop/uncertainty risk is high. Consider waiting.")

    else:
        st.info("👆 Enter symbol/date/spot and click **Fetch Data**.")

    st.markdown("---")
    st.caption("📊 Stats Dashboard | For educational purposes only")


if __name__ == "__main__":
    main()
