import streamlit as st
import pandas as pd
import datetime as dt
import plotly.graph_objects as go

# Import our modular components
from stats_app.styles import apply_custom_styles
from stats_app.helpers.api_client import check_api, fetch_options, fetch_weekly_summary, API_BASE_URL
from stats_app.helpers.data_fetching import get_spot_from_finnhub, get_finnhub_api_key, fetch_price_history
from stats_app.helpers.ui_components import st_plot, st_btn, st_df
from stats_app.tabs.tab_options_chain import render_tab_options_chain
from stats_app.tabs.tab_oi_charts import render_tab_oi_charts
from stats_app.tabs.tab_weekly_gamma import render_tab_weekly_gamma
from stats_app.tabs.tab_gamma_map_filters import render_tab_gamma_map_filters
from stats_app.tabs.tab_vol_greeks import render_tab_vol_greeks
from stats_app.tabs.tab_pro_edge import render_tab_pro_edge
from stats_app.tabs.tab_market_folding import render_tab_market_folding
from stats_app.tabs.tab_vwap_obv import render_tab_vwap_obv
from stats_app.tabs.tab_vol_cone import render_tab_vol_cone

# Configure Streamlit Page
st.set_page_config(
    page_title="Stats Dashboard | Options & Gamma",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    # Apply global styles
    apply_custom_styles()
    
    # Header
    st.markdown(
        """
    <div class="header">
        <h1>📊 Stats Dashboard</h1>
        <p>Options chain + Weekly Gamma / GEX (dealer positioning) + Filters</p>
    </div>
    """,
        unsafe_allow_html=True
    )

    # Custom Tab Styling (Multi-line + Premium)
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap !important; gap: 12px 8px !important; padding: 10px 0 !important; }
        .stTabs [data-baseweb="tab"] {
            background-color: #1e2328 !important; border: 1px solid #3d4450 !important;
            border-radius: 8px !important; padding: 8px 16px !important; color: #b0b5bc !important;
            font-weight: 500 !important; flex-shrink: 0 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #00875a 0%, #00a86b 100%) !important;
            color: white !important; border-color: #00d775 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    api_ok = check_api()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Options Query")
        symbol = st.text_input("Symbol", value="TSLA").upper().strip()
        # Jan 16, 2026 has expired. Defaulting to next Friday: Jan 23, 2026
        expiry_date = st.date_input("Expiration Date", value=dt.date(2026, 1, 23))
        date = expiry_date.isoformat()
        
        spot_source = st.selectbox("Spot Price Source", options=["Finnhub (spot only)", "Manual"])
        refresh_spot_btn = st_btn("🔄 Refresh Spot")

        if "spot_data" not in st.session_state: st.session_state["spot_data"] = None
        if "spot_error" not in st.session_state: st.session_state["spot_error"] = None

        if (refresh_spot_btn or (st.session_state["spot_data"] is None)) and spot_source != "Manual" and symbol:
            finnhub_data = get_spot_from_finnhub(symbol)
            if finnhub_data:
                st.session_state["spot_data"] = finnhub_data
                st.session_state["spot_error"] = None
            else:
                st.session_state["spot_error"] = "Finnhub API key not set or no data"

        live_spot = st.session_state["spot_data"]["spot"] if st.session_state["spot_data"] else None
        if live_spot: st.success(f"📈 Finnhub: ${live_spot:.2f}")
        
        spot_input = st.number_input("Spot Price (manual fallback)", value=float(live_spot or 260.0), step=0.50)
        spot = float(live_spot) if live_spot else spot_input

        fetch_btn = st_btn("🔄 Fetch Data", disabled=not api_ok)

        st.markdown("---")
        st.markdown("### 🔥 Quick Symbols")
        popular = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]
        cols = st.columns(len(popular))
        for i, s in enumerate(popular):
            if cols[i].button(s): st.session_state["symbol_override"] = s
        if "symbol_override" in st.session_state: symbol = st.session_state["symbol_override"]

        st.markdown("---")
        if api_ok: st.markdown('<div class="status-ok">✓ API Connected</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="status-error">✗ API Offline</div>', unsafe_allow_html=True)
        st.caption(f"Backend: {API_BASE_URL}")

    # Data Fetching Logic
    if fetch_btn and api_ok:
        with st.spinner("Fetching data..."):
            st.session_state["options_result"] = fetch_options(symbol, date)
            st.session_state["weekly_result"] = fetch_weekly_summary(symbol, date, spot)
            st.session_state["spot_at_fetch"] = spot

    options_result = st.session_state.get("options_result")
    weekly_result = st.session_state.get("weekly_result")

    if options_result is not None and weekly_result is not None:
        # Check if BOTH succeeded.
        if options_result.get("success") and weekly_result.get("success"):
            df = pd.DataFrame(options_result["data"].get("data", []))
            w = weekly_result["data"]
            totals, pcr, top = w.get("totals", {}), w.get("pcr", {}), w.get("top_strikes", {})
            top_call = pd.DataFrame(top.get("call_gex", []))
            top_put = pd.DataFrame(top.get("put_gex", []))
            top_net = pd.DataFrame(top.get("net_gex_abs", []))

            # Price History Expander
            with st.expander("📈 Price + Moving Averages", expanded=True):
                hist_df = fetch_price_history(symbol)
                if not hist_df.empty:
                    for w_ in [15, 20, 50]: hist_df[f"MA{w_}"] = hist_df["Close"].rolling(w_).mean()
                    fig_px = go.Figure()
                    fig_px.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Close"], name="Close"))
                    for w_ in [15, 20, 50]: fig_px.add_trace(go.Scatter(x=hist_df.index, y=hist_df[f"MA{w_}"], name=f"MA{w_}"))
                    fig_px.update_layout(template="plotly_dark", height=400)
                    st_plot(fig_px)

            st.success(f"✓ Loaded {len(df)} strikes for **{symbol}**")

            # Tabs
            t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(
                ["📋 Chain", "📊 OI", "📌 Weekly GEX", "🧲 Map", "🧮 Greeks", "🏆 Pro Edge", "🔳 Folding", "📈 VWAP", "🎯 Vol Cone"]
            )
            
            with t1: render_tab_options_chain(df)
            with t2: render_tab_oi_charts(df)
            with t3: render_tab_weekly_gamma(pcr, totals, w, spot, top_call, top_put, top_net)
            with t4: render_tab_gamma_map_filters(symbol, date, spot)
            with t5: render_tab_vol_greeks(df, spot, symbol, date)
            with t6: render_tab_pro_edge(symbol, date, spot, hist_df, totals, df)
            with t7: render_tab_market_folding(symbol)
            with t8: render_tab_vwap_obv(symbol)
            with t9: render_tab_vol_cone(symbol)
        else:
            # Report specific error if failed.
            if not options_result.get("success"):
                st.error(f"Options API Error: {options_result.get('error')}")
            if not weekly_result.get("success"):
                st.error(f"Weekly Summary API Error: {weekly_result.get('error')}")
            
            st.warning("No data found for the selected symbol and expiration. Jan 16, 2026 has expired; please try Jan 23, 2026 or later.")
            st.info("Ensure the symbol exists on Barchart and you have selected an active expiration date.")
    else:
        st.info("Query a symbol and click 'Fetch Data' to begin.")

if __name__ == "__main__":
    main()
