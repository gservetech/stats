"""
Barchart Options Dashboard - Streamlit Frontend
Connects to FastAPI backend for options data scraping + Weekly Gamma/GEX summary.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration - MUST be first
st.set_page_config(
    page_title="Barchart Options Dashboard",
    page_icon="📊",
    layout="wide"
)

# API Configuration
DEFAULT_API_URL = "http://localhost:8000"
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", DEFAULT_API_URL)
except Exception:
    API_BASE_URL = DEFAULT_API_URL

# Barchart-inspired dark theme CSS
st.markdown("""
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
""", unsafe_allow_html=True)


def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options(symbol: str, date: str):
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
    """
    Calls backend:
      /weekly/summary?symbol=...&date=...&spot=...
    """
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

    # Bar Chart
    bar_fig = make_subplots(rows=1, cols=2, subplot_titles=("📈 Calls OI (Bar)", "📉 Puts OI (Bar)"))
    bar_fig.add_trace(go.Bar(x=df_sorted["strike_num"], y=df_sorted["call_oi"], name="Calls"), row=1, col=1)
    bar_fig.add_trace(go.Bar(x=df_sorted["strike_num"], y=df_sorted["put_oi"], name="Puts"), row=1, col=2)
    bar_fig.update_layout(template="plotly_dark", height=350, showlegend=False)
    bar_fig.update_xaxes(title_text="Strike")
    bar_fig.update_yaxes(title_text="Open Interest")

    # Line Chart
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


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>📊 Barchart Options Dashboard</h1>
        <p>Options chain + Weekly Gamma / GEX (dealer positioning)</p>
    </div>
    """, unsafe_allow_html=True)

    api_ok = check_api()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Options Query")

        symbol = st.text_input("Symbol", value="AAPL").upper().strip()
        date = st.text_input("Expiration Date", value="2026-01-16", help="Format: YYYY-MM-DD (ex: 2026-01-16)")
        spot = st.number_input("Spot Price (required for Gamma/GEX)", value=260.00, step=0.50)

        fetch_btn = st.button("🔄 Fetch Data", use_container_width=True, disabled=not api_ok)

        st.markdown("---")
        st.markdown("### 🔥 Quick Symbols")
        popular = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "AMZN"]
        cols = st.columns(3)
        for i, s in enumerate(popular):
            with cols[i % 3]:
                if st.button(s, key=f"q_{s}", use_container_width=True):
                    st.session_state["symbol_override"] = s

        if "symbol_override" in st.session_state:
            symbol = st.session_state["symbol_override"]

        st.markdown("---")
        if api_ok:
            st.markdown('<div class="status-ok">✓ API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">✗ API Offline</div>', unsafe_allow_html=True)

        st.caption(f"Backend: {API_BASE_URL}")

    if not api_ok:
        st.error(f"Cannot connect to API at `{API_BASE_URL}`. Start backend: `uvicorn api:app --port 8000 --reload`")
        return

    if fetch_btn or st.session_state.get("last_fetch"):
        st.session_state["last_fetch"] = {"symbol": symbol, "date": date, "spot": spot}

        # Fetch options + weekly summary
        with st.spinner(f"Scraping {symbol} options for {date}..."):
            options_result = fetch_options(symbol, date)

        with st.spinner(f"Computing Weekly Gamma/GEX for {symbol} {date} (spot={spot})..."):
            weekly_result = fetch_weekly_summary(symbol, date, spot)

        # Handle errors
        if not options_result.get("success"):
            st.error(f"Options error: {options_result.get('error')}")
            return
        if not weekly_result.get("success"):
            st.error(f"Weekly summary error: {weekly_result.get('error')}")
            return

        # Dataframes
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

        tab1, tab2, tab3 = st.tabs(["📋 Options Chain", "📊 OI Charts", "📌 Weekly Gamma / GEX"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=520, hide_index=True)

        with tab2:
            bar_fig, line_fig = create_oi_charts(df)
            st.subheader("📈 Open Interest Comparison")
            st.plotly_chart(line_fig, use_container_width=True)
            st.subheader("📊 Open Interest Distribution")
            st.plotly_chart(bar_fig, use_container_width=True)

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
                    st.dataframe(top_call, use_container_width=True, hide_index=True)
                    fig = create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No top call GEX data returned.")

            with colB:
                st.markdown("**Top Put GEX**")
                if not top_put.empty:
                    st.dataframe(top_put, use_container_width=True, hide_index=True)
                    fig = create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No top put GEX data returned.")

            with colC:
                st.markdown("**Top Net GEX (abs)**")
                if not top_net.empty:
                    st.dataframe(top_net, use_container_width=True, hide_index=True)
                    fig = create_top_strikes_chart(top_net, "strike", "net_gex", "Top Net GEX (abs)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No top net GEX data returned.")

            st.markdown("---")
            st.caption("Note: GEX here is an approximation from IV + OI using Black-Scholes gamma; use for educational analysis.")

    else:
        st.info("👆 Enter symbol/date/spot and click **Fetch Data**.")

    st.markdown("---")
    st.caption("📊 Barchart Options Dashboard | Data scraped from Barchart.com | For educational purposes only")


if __name__ == "__main__":
    main()
