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
import numpy as np
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    fig.add_vline(x=spot, line_width=2, line_dash="dash",
                  annotation_text=f"Spot {spot:g}", annotation_position="top")

    for _, row in levels["magnets"].iterrows():
        s = float(row["strike"])
        fig.add_vline(x=s, line_width=1, line_dash="dot",
                      annotation_text=f"Magnet {s:g}", annotation_position="bottom")

    lower = levels["gamma_box"]["lower"]
    upper = levels["gamma_box"]["upper"]
    if lower is not None:
        fig.add_vline(x=lower, line_width=2, line_dash="dash",
                      annotation_text=f"Lower wall {lower:g}", annotation_position="top left")
    if upper is not None:
        fig.add_vline(x=upper, line_width=2, line_dash="dash",
                      annotation_text=f"Upper wall {upper:g}", annotation_position="top right")

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
    yfinance sometimes returns MultiIndex columns (Price x Ticker) or duplicate columns.
    This normalizes to plain columns with at least 'Close'.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # If columns are MultiIndex, prefer the 'Close' level
    if isinstance(out.columns, pd.MultiIndex):
        # Try common forms:
        # ('Close', 'AAPL') or ('AAPL', 'Close') depending on yfinance version.
        close_col = None
        for col in out.columns:
            if "Close" in col:
                close_col = col
                break
        if close_col is not None:
            out = pd.DataFrame({"Close": out[close_col]})
        else:
            # fallback: flatten then hope for Close
            out.columns = ["_".join(map(str, c)).strip() for c in out.columns]
    else:
        # ensure Close exists; if there are multiple Close columns, pick the first
        if "Close" not in out.columns:
            # sometimes lower-case?
            for alt in ["close", "Adj Close", "adjclose", "Adj_Close"]:
                if alt in out.columns:
                    out["Close"] = out[alt]
                    break

    if "Close" not in out.columns:
        return pd.DataFrame()

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Close"])
    return out


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


def kalman_filter_1d(price: pd.Series, process_var: float = 1e-5, meas_var: float = 1e-2) -> pd.Series:
    """
    Robust 1D Kalman filter.
    Fixes your crash by forcing the input to a 1D float array.
    """
    s = pd.Series(price).astype(float).dropna()
    z = np.asarray(s.values, dtype=float).reshape(-1)  # force 1D
    n = int(z.shape[0])
    if n == 0:
        return pd.Series(dtype=float)

    x = np.zeros(n, dtype=float)
    P = 1.0
    x[0] = float(z[0])

    Q = float(process_var)
    R = float(meas_var)

    for k in range(1, n):
        x_pred = x[k - 1]
        P_pred = P + Q

        K = P_pred / (P_pred + R)
        x[k] = x_pred + K * (float(z[k]) - x_pred)
        P = (1 - K) * P_pred

    return pd.Series(x, index=s.index)


def plot_filters(df_prices: pd.DataFrame, length_md: int, kama_er: int, kama_fast: int, kama_slow: int,
                 kf_q: float, kf_r: float):
    close = df_prices["Close"].astype(float)

    md = mcginley_dynamic(close, length=length_md)
    k = kama(close, er_length=kama_er, fast=kama_fast, slow=kama_slow)
    kf = kalman_filter_1d(close, process_var=kf_q, meas_var=kf_r)

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
    return fig


# -----------------------------
# Main App
# -----------------------------
def main():
    st.markdown(
        """
    <div class="header">
        <h1>📊 Stats Dashboard</h1>
        <p>Options chain + Weekly Gamma / GEX (dealer positioning)</p>
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

        fetch_btn = st.button("🔄 Fetch Data", width="stretch", disabled=not api_ok)

        st.markdown("---")
        st.markdown("### 🔥 Quick Symbols")
        popular = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "AMZN"]
        cols = st.columns(3)
        for i, s in enumerate(popular):
            with cols[i % 3]:
                if st.button(s, key=f"q_{s}", width="stretch"):
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

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Options Chain", "📊 OI Charts", "📌 Weekly Gamma / GEX", "🧲 Gamma Map + Filters"])

        with tab1:
            st.dataframe(df, width="stretch", height=520, hide_index=True)

        with tab2:
            required_cols = {"Strike", "Call OI", "Put OI"}
            if not required_cols.issubset(set(df.columns)):
                st.warning(
                    f"Options data is missing expected columns: {sorted(list(required_cols - set(df.columns)))}.\n\n"
                    "Make sure backend returns columns named exactly: Strike, Call OI, Put OI"
                )
            else:
                bar_fig, line_fig = create_oi_charts(df)
                st.subheader("📈 Open Interest Comparison")
                st.plotly_chart(line_fig, width="stretch")
                st.subheader("📊 Open Interest Distribution")
                st.plotly_chart(bar_fig, width="stretch")

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
                    st.dataframe(top_call, width="stretch", hide_index=True)
                    if {"strike", "call_gex"}.issubset(top_call.columns):
                        fig = create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX")
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No top call GEX data returned.")

            with colB:
                st.markdown("**Top Put GEX**")
                if not top_put.empty:
                    st.dataframe(top_put, width="stretch", hide_index=True)
                    if {"strike", "put_gex"}.issubset(top_put.columns):
                        fig = create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX")
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No top put GEX data returned.")

            with colC:
                st.markdown("**Top Net GEX (abs)**")
                if not top_net.empty:
                    st.dataframe(top_net, width="stretch", hide_index=True)
                    if {"strike", "net_gex"}.issubset(top_net.columns):
                        fig = create_top_strikes_chart(top_net, "strike", "net_gex", "Top Net GEX (abs)")
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No top net GEX data returned.")

            st.caption("Note: GEX is an approximation from IV + OI using Black-Scholes gamma; for educational analysis.")

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

                        t1, t2, t3 = st.columns(3)

                        with t1:
                            st.markdown("### 🧲 Magnets (Top Net GEX abs)")
                            m = levels["magnets"].copy()
                            m["net_gex"] = pd.to_numeric(m["net_gex"], errors="coerce").fillna(0.0).map(lambda x: f"{x:,.0f}")
                            st.dataframe(m, width="stretch", hide_index=True)

                        with t2:
                            st.markdown("### 🧱 Call Walls (Top Call GEX)")
                            cw = levels["call_walls"].copy()
                            cw["call_gex"] = pd.to_numeric(cw["call_gex"], errors="coerce").fillna(0.0).map(lambda x: f"{x:,.0f}")
                            st.dataframe(cw, width="stretch", hide_index=True)

                        with t3:
                            st.markdown("### 🧱 Put Walls (Top Put GEX)")
                            pw = levels["put_walls"].copy()
                            pw["put_gex"] = pd.to_numeric(pw["put_gex"], errors="coerce").fillna(0.0).map(lambda x: f"{x:,.0f}")
                            st.dataframe(pw, width="stretch", hide_index=True)

                        fig = plot_net_gex_map(gex_df, spot=spot, levels=levels)
                        st.plotly_chart(fig, width="stretch")

                        st.markdown("### 📌 Scenario Read (auto)")
                        main_mag = float(levels["magnets"].iloc[0]["strike"]) if not levels["magnets"].empty else None
                        next_mags = [float(x) for x in levels["magnets"]["strike"].iloc[1:4].tolist()] if len(levels["magnets"]) > 1 else []
                        box_str = f"[{lower:g}, {upper:g}]" if (lower is not None and upper is not None) else "N/A"

                        st.write(f"- **Spot**: {spot:g}")
                        st.write(f"- **Main magnet**: {main_mag:g}" if main_mag is not None else "- **Main magnet**: N/A")
                        if next_mags:
                            st.write(f"- **Next magnets**: {', '.join([str(int(x)) if float(x).is_integer() else str(x) for x in next_mags])}")
                        st.write(f"- **Gamma box (nearest walls)**: {box_str}")
                        st.write("- If price is inside the box, expect more **pin / range** behavior. If it breaks beyond a wall, watch for **faster moves**.")

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
                fig2 = plot_filters(px, int(length_md), int(kama_er), int(kama_fast), int(kama_slow), float(kf_q), float(kf_r))
                st.plotly_chart(fig2, width="stretch")

                st.caption("Tip: McGinley adapts to speed, KAMA adapts via Efficiency Ratio, Kalman adapts via Q/R confidence.")

    else:
        st.info("👆 Enter symbol/date/spot and click **Fetch Data**.")

    st.markdown("---")
    st.caption("📊 Stats Dashboard |  | For educational purposes only")


if __name__ == "__main__":
    main()
