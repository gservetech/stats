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
import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# -----------------------------
# Page configuration (MUST be first Streamlit call)
# -----------------------------
st.set_page_config(
    page_title="stats Dashboard",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# API Base URL (local + cloud)
# -----------------------------
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


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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
        return {"success": False, "error": "Timeout calculating weekly summary (backend scraping may be slow).", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}


@st.cache_data(ttl=300, show_spinner=False)
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
    line_fig.add_trace(go.Scatter(x=df_sorted["strike_num"], y=df_sorted["call_oi"], mode="lines+markers", name="Call OI"))
    line_fig.add_trace(go.Scatter(x=df_sorted["strike_num"], y=df_sorted["put_oi"], mode="lines+markers", name="Put OI"))
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
    put_delta  = disc_q * (Nd1 - 1.0)

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)

    # Vega: per 1.00 vol. Convert to per 1% by /100
    vega_per_1 = S * disc_q * pdf_d1 * sqrtT
    vega_per_1pct = vega_per_1 / 100.0

    # Theta: per year. Convert to per day by /365
    call_theta_y = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * disc_r * Nd2 + q * S * disc_q * Nd1
    put_theta_y  = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * disc_r * Nmd2 - q * S * disc_q * Nmd1
    call_theta_d = call_theta_y / 365.0
    put_theta_d  = put_theta_y / 365.0

    return call_delta, put_delta, gamma, vega_per_1pct, call_theta_d, put_theta_d

def plot_iv_and_greeks(df: pd.DataFrame, spot: float, T: float | None = None, r: float = 0.041, q: float = 0.004):
    """
    Builds a multi-line figure:
      - IV smile (call/put)
      - Delta, Gamma, Vega, Theta (call/put) if present
    Returns: (fig_iv, fig_greeks, atm_metrics_dict)
    """
    d = df.copy()

    if "Strike" not in d.columns:
        return None, None, {}

    d["strike_num"] = _to_float_series(d["Strike"])
    d = d.dropna(subset=["strike_num"]).sort_values("strike_num")

    # --- IV
    call_iv_col = _find_col(d, "call", "iv")
    put_iv_col = _find_col(d, "put", "iv")
    fig_iv = None

    if call_iv_col or put_iv_col:
        fig_iv = go.Figure()
        if call_iv_col:
            d["call_iv"] = _to_float_series(d[call_iv_col])
            fig_iv.add_trace(go.Scatter(x=d["strike_num"], y=d["call_iv"], mode="lines+markers", name=f"Call IV ({call_iv_col})"))
        if put_iv_col:
            d["put_iv"] = _to_float_series(d[put_iv_col])
            fig_iv.add_trace(go.Scatter(x=d["strike_num"], y=d["put_iv"], mode="lines+markers", name=f"Put IV ({put_iv_col})"))

        fig_iv.add_vline(x=float(spot), line_width=2, line_dash="dash", annotation_text="Spot", annotation_position="top")
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
        call_sig = d.get("call_iv", pd.Series([float('nan')]*len(d))).apply(_iv_to_sigma)
        put_sig  = d.get("put_iv",  pd.Series([float('nan')]*len(d))).apply(_iv_to_sigma)

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
        d["Put Delta"]  = pd.Series(put_delta, index=d.index)
        d["Call Gamma"] = pd.Series(call_gamma, index=d.index)
        d["Put Gamma"]  = pd.Series(put_gamma, index=d.index)
        d["Call Vega"]  = pd.Series(call_vega, index=d.index)
        d["Put Vega"]   = pd.Series(put_vega, index=d.index)
        d["Call Theta"] = pd.Series(call_theta, index=d.index)
        d["Put Theta"]  = pd.Series(put_theta, index=d.index)

        # Build greeks plot from computed columns
        fig_g = go.Figure()
        for name in ["Call Delta", "Put Delta", "Call Gamma", "Put Gamma", "Call Vega", "Put Vega", "Call Theta", "Put Theta"]:
            fig_g.add_trace(go.Scatter(x=d["strike_num"], y=_to_float_series(d[name]), mode="lines", name=name))
        any_greek = True

    fig_greeks = None
    if any_greek:
        fig_g.add_vline(x=float(spot), line_width=2, line_dash="dash", annotation_text="Spot", annotation_position="top")
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
def _normalize_yf_df(df: pd.DataFrame | None) -> pd.DataFrame:
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

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"])
    return out[["Close"]]


@st.cache_data(ttl=900, show_spinner=False)
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
        date = st.text_input("Expiration Date", value="2026-01-16", help="Format: YYYY-MM-DD (ex: 2026-01-16)")
        spot = st.number_input("Spot Price (required for Gamma/GEX)", value=260.00, step=0.50)

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

        st.success(f"✓ Loaded {len(df)} strikes for **{symbol}** expiring **{date}**")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📋 Options Chain", "📊 OI Charts", "📌 Weekly Gamma / GEX", "🧲 Gamma Map + Filters", "🧮 Volatility & Greeks"]
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
            c2.metric("Put/Call Ratio (Volume)", f"{(pcr.get('volume') or 0):.3f}" if pcr.get("volume") is not None else "N/A")
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
                        cA.metric("Main Magnet", f"{float(levels['magnets'].iloc[0]['strike']):g}" if not levels["magnets"].empty else "N/A")
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
                fig2, kf_series = plot_filters(px, int(length_md), int(kama_er), int(kama_fast), int(kama_slow), float(kf_q), float(kf_r))
                st_plot(fig2)

                # ✅ Kalman “what it says” message
                km = kalman_message(px["Close"].values, kf_series.values, lookback=20, band_pct=0.003)
                st.markdown(
                    f"""
**Kalman Read:** {km['msg']}

- **Regime:** **{km.get('regime','N/A')}**
- **Trend:** **{km.get('trend','N/A')}**
- **Bias:** **{km.get('bias','N/A')}**
- **Trend strength:** **{km.get('trend_strength','N/A')}**
- **Structure:** **{km.get('structure','N/A')}**
- **Chop (crossings/{km.get('lookback',20)}):** **{km.get('crossings','N/A')}**
- **Confidence:** **{km.get('confidence','N/A')}**

**Why this label?**{km.get('why','')}

**Notes:**- "UPTREND + price below Kalman" often = *pullback inside an uptrend* (watch for reclaim).
- "DOWNTREND + price below Kalman" often = *sell-the-rip* behavior (Kalman acts as resistance).
- Higher crossings = more range/chop → mean-reversion works better than breakout.
"""
                )

                st.caption("Tip: McGinley adapts to speed, KAMA adapts via Efficiency Ratio, Kalman adapts via Q/R confidence.")

        with tab5:
            st.subheader("🧮 Volatility & Greeks (from this expiry chain)")

            if df.empty:
                st.info("No options data loaded yet.")
            else:
                # --- Greeks inputs (used only if backend doesn't provide greeks)
                with st.expander('Greek Inputs (Black-Scholes fallback)', expanded=False):
                    # If your backend doesn't provide greeks, we can compute them from IV using Black–Scholes.
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

                    use_trading_days = st.checkbox('Use trading-day year (252) for T (otherwise calendar 365)', value=False)

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
                            st.caption("Yahoo spot unavailable (network/blocked). Falling back to backend/override spot.")

                # priority: manual override > yahoo > backend spot
                if spot_override and float(spot_override) > 0:
                    _spot_for_greeks = float(spot_override)
                elif yahoo_spot and float(yahoo_spot) > 0:
                    _spot_for_greeks = float(yahoo_spot)
                else:
                    _spot_for_greeks = float(spot)


                _spot_for_greeks = float(spot_override) if spot_override and float(spot_override) > 0 else float(spot)
                # Assume equity options expire at market close (4:00pm local) on the selected expiry date
                _now_ts = pd.Timestamp.now()
                _exp_ts = pd.Timestamp(date) + pd.Timedelta(hours=16)
                if use_trading_days:
                    days = max(int((_exp_ts.normalize() - _now_ts.normalize()).days), 0)
                    T = max(days / 252.0, 1e-6)
                else:
                    T = max(float((_exp_ts - _now_ts).total_seconds()) / (365.0 * 24 * 3600), 1e-6)

                fig_iv, fig_greeks, atm = plot_iv_and_greeks(df, spot=_spot_for_greeks, T=T, r=float(r_in), q=float(q_in))

                if not atm:
                    st.warning("Could not compute ATM snapshot (Strike column missing or invalid).")
                else:
                    atm_strike = atm.get("atm_strike")
                    st.markdown(f"**ATM strike (nearest to spot):** `{atm_strike:g}`")

                    # show a few key ATM metrics if present
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Call IV", f"{atm.get('Call IV', float('nan')):.4f}" if "Call IV" in atm else "N/A")
                    m2.metric("Put IV", f"{atm.get('Put IV', float('nan')):.4f}" if "Put IV" in atm else "N/A")
                    m3.metric("Call Delta", f"{atm.get('Call Delta', float('nan')):.3f}" if "Call Delta" in atm else "N/A")
                    m4.metric("Put Delta", f"{atm.get('Put Delta', float('nan')):.3f}" if "Put Delta" in atm else "N/A")

                    st.markdown("### 🧠 How to read the Greeks for **this** ATM strike (spot up/down, benefits & risks)")

                    # Pull values if available (from backend or Black–Scholes fallback)
                    spot_used = float(spot_for_greeks) if spot_for_greeks is not None else float("nan")
                    K = float(atm_strike) if atm_strike is not None else float("nan")

                    call_delta = float(atm.get("Call Delta", float("nan"))) if isinstance(atm, dict) else float("nan")
                    put_delta  = float(atm.get("Put Delta",  float("nan"))) if isinstance(atm, dict) else float("nan")
                    gamma      = float(atm.get("Gamma",      float("nan"))) if isinstance(atm, dict) else float("nan")
                    vega       = float(atm.get("Vega",       float("nan"))) if isinstance(atm, dict) else float("nan")  # per 1.00 (100%) IV
                    call_theta = float(atm.get("Call Theta", float("nan"))) if isinstance(atm, dict) else float("nan")  # per year
                    put_theta  = float(atm.get("Put Theta",  float("nan"))) if isinstance(atm, dict) else float("nan")  # per year

                    # Common unit conversions
                    vega_per_1pct = vega / 100.0 if pd.notna(vega) else float("nan")
                    call_theta_per_day = call_theta / 365.0 if pd.notna(call_theta) else float("nan")
                    put_theta_per_day  = put_theta  / 365.0 if pd.notna(put_theta)  else float("nan")

                    # Scenario helpers (very rough: "all else equal")
                    def _fmt(x, fmt):
                        try:
                            return format(float(x), fmt) if pd.notna(x) else "N/A"
                        except Exception:
                            return "N/A"

                    dS_1 = 1.0
                    call_move_up_1  = call_delta * dS_1 if pd.notna(call_delta) else float("nan")
                    put_move_up_1   = put_delta  * dS_1 if pd.notna(put_delta)  else float("nan")
                    call_move_dn_1  = -call_delta * dS_1 if pd.notna(call_delta) else float("nan")
                    put_move_dn_1   = -put_delta  * dS_1 if pd.notna(put_delta)  else float("nan")

                    # Gamma effect: delta changes by ~ Gamma * ΔS
                    call_delta_up_1 = call_delta + gamma * dS_1 if (pd.notna(call_delta) and pd.notna(gamma)) else float("nan")
                    call_delta_dn_1 = call_delta - gamma * dS_1 if (pd.notna(call_delta) and pd.notna(gamma)) else float("nan")

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
                    - Put  Θ: `{_fmt(put_theta,  '.3f')}`/yr (**≈ `{_fmt(put_theta_per_day,  '.3f')}` per day**)
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
                    st.write("ATM + longer-dated expiries usually have **bigger Vega**, so IV changes can matter a lot.")

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
                    - ⚠️ **Reminder**: These are **“all else equal”** approximations — in real trading, Δ/Γ/Vega/Θ move together.
                    """)

                    st.caption("This tab uses backend greeks if provided. If not, it computes greeks from IV using Black–Scholes (your r/q/spot inputs above).")


                if fig_iv is not None:
                    st_plot(fig_iv)
                else:
                    st.info("IV columns not found in your backend payload (look for columns like 'Call IV' / 'Put IV').")

                if fig_greeks is not None:
                    st_plot(fig_greeks)
                else:
                    st.info("Greeks columns not found (Delta/Gamma/Vega/Theta). If you add them to the backend, this tab will auto-plot them.")

                skew = approx_skew_25d(df)
                if skew:
                    st.markdown("### 📐 25-Delta Skew (rough)")
                    st.write(
                        f"- 25d Call: strike {skew['call_25d_strike']:g}, Δ={skew['call_25d_delta']:.3f}, IV={skew['call_25d_iv']:.4f}\\n"
                        f"- 25d Put:  strike {skew['put_25d_strike']:g}, Δ={skew['put_25d_delta']:.3f}, IV={skew['put_25d_iv']:.4f}\\n"
                        f"- **Skew (Call IV − Put IV)**: **{skew['skew_call_minus_put']:.4f}**"
                    )
                    st.caption("Skew helps you see if downside protection (puts) is getting expensive vs upside calls.")


    else:
        st.info("👆 Enter symbol/date/spot and click **Fetch Data**.")

    st.markdown("---")
    st.caption("📊 Stats Dashboard | For educational purposes only")


if __name__ == "__main__":
    main()