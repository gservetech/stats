import sys
import os

# Ensure project root is on sys.path (Streamlit-safe)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import numpy as np
import requests

# Import modular components (ABSOLUTE imports only)
from stats_app.styles import apply_custom_styles
from stats_app.helpers.api_client import (
    check_api,
    fetch_options,
    fetch_weekly_summary,
    API_BASE_URL,
)
from stats_app.helpers.data_fetching import get_spot_from_finnhub, fetch_price_history
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
from stats_app.tabs.tab_friday_predictor import render_tab_friday_predictor
from stats_app.tabs.tab_friday_predictor_plus import render_tab_friday_predictor_plus
from stats_app.tabs.tab_vanna_charm import render_tab_vanna_charm
from stats_app.tabs.tab_interpretation_engine import render_tab_interpretation_engine


# OPTIONAL: if you add the vanna tab in your UI
# from stats_app.tabs.tab_vanna_charm import render_tab_vanna_charm


# Configure Streamlit Page
st.set_page_config(
    page_title="Stats Dashboard | Options & Gamma",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    apply_custom_styles()

    # Header
    st.markdown(
        """
        <div class="header">
            <h1>📊 Stats Dashboard</h1>
            <p>Options chain + Weekly Gamma / GEX + Friday Price Predictor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Custom Tab Styling
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
        unsafe_allow_html=True,
    )

    api_ok = check_api()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Options Query")
        symbol = st.text_input("Symbol", value="MU").upper().strip()
        expiry_date = st.date_input("Expiration Date", value=dt.date.today())
        date = expiry_date.isoformat()

        spot_source = st.selectbox("Spot Price Source", options=["CNBC (Scraping)", "Manual"])
        refresh_spot_btn = st_btn("🔄 Refresh Spot")

        # --------- per-symbol spot cache (prevents symbol cross-talk) ---------
        spot_key = f"spot_data_{symbol}"
        if spot_key not in st.session_state:
            st.session_state[spot_key] = None

        if (refresh_spot_btn or (st.session_state[spot_key] is None)) and spot_source != "Manual" and symbol:
            st.session_state[spot_key] = get_spot_from_finnhub(symbol)

        live_spot_data = st.session_state[spot_key]
        live_spot = live_spot_data["spot"] if live_spot_data else None
        
        if live_spot_data:
            source_name = live_spot_data.get("source", "Source")
            st.success(f"📈 {source_name}: ${live_spot:.2f}")
            
            if "after_hours" in live_spot_data and live_spot_data["after_hours"]:
                ah = live_spot_data["after_hours"]
                st.info(f"🌙 After Hours: ${ah['price']:.2f} ({ah['change']:+.2f})")

        spot_input = st.number_input("Spot Price (manual fallback)", value=float(live_spot or 260.0), step=0.50)
        spot = float(live_spot) if live_spot else float(spot_input)

        fetch_btn = st_btn("🔄 Fetch Data", disabled=not api_ok)

    # --------- reset session_state when symbol changes (prevents stale mixing) ---------
    if "last_symbol" not in st.session_state:
        st.session_state["last_symbol"] = symbol

    if st.session_state["last_symbol"] != symbol:
        st.session_state["options_result"] = None
        st.session_state["weekly_result"] = None
        st.session_state["hist_df"] = pd.DataFrame()
        st.session_state["spot_at_fetch"] = None
        st.session_state["last_symbol"] = symbol

    # Data Fetching Logic
    if fetch_btn and api_ok:
        with st.spinner("Analyzing market structure..."):
            st.session_state["options_result"] = fetch_options(symbol, date)
            st.session_state["weekly_result"] = fetch_weekly_summary(symbol, date, spot)
            st.session_state["hist_df"] = fetch_price_history(symbol).copy()
            st.session_state["spot_at_fetch"] = spot

    options_result = st.session_state.get("options_result")
    weekly_result = st.session_state.get("weekly_result")
    hist_df = st.session_state.get("hist_df")

    if options_result and weekly_result and options_result.get("success"):
        df = pd.DataFrame(options_result["data"].get("data", []))
        w = weekly_result["data"]
        totals, pcr, top = w.get("totals", {}), w.get("pcr", {}), w.get("top_strikes", {})

        top_call = pd.DataFrame(top.get("call_gex", []))
        top_put = pd.DataFrame(top.get("put_gex", []))
        top_net = pd.DataFrame(top.get("net_gex_abs", []))

        # Price History Expander
        with st.expander("📈 Price + Moving Averages", expanded=True):
            if hist_df is not None and not hist_df.empty:
                px_df = hist_df.copy()
                for w_ in [15, 20, 50]:
                    px_df[f"MA{w_}"] = px_df["Close"].rolling(w_).mean()

                fig_px = go.Figure()
                fig_px.add_trace(go.Scatter(x=px_df.index, y=px_df["Close"], name="Close"))
                for w_ in [15, 20, 50]:
                    fig_px.add_trace(go.Scatter(x=px_df.index, y=px_df[f"MA{w_}"], name=f"MA{w_}"))
                fig_px.update_layout(template="plotly_dark", height=400)
                st_plot(fig_px)

        st.success(f"✓ Loaded {len(df)} strikes for **{symbol}**")

        # Tabs
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13 = st.tabs(
            [
                "📋 Chain",
                "📊 OI",
                "📌 Weekly GEX",
                "🧲 Map",
                "🧮 Greeks",
                "🏆 Pro Edge",
                "🔳 Folding",
                "📈 VWAP",
                "🎯 Vol Cone",
                "🔮 Friday Predictor",
                "🧠 Friday Predictor+",
                "🌊 Vanna/Charm",
                "🧠 Interpretation",
            ]
        )

        with t1:
            render_tab_options_chain(df)
        with t2:
            render_tab_oi_charts(df)
        with t3:
            render_tab_weekly_gamma(pcr, totals, w, spot, top_call, top_put, top_net)
        with t4:
            render_tab_gamma_map_filters(symbol, date, spot)
        with t5:
            render_tab_vol_greeks(df, spot, symbol, date)
        with t6:
            render_tab_pro_edge(symbol, date, spot, hist_df, totals, df)
        with t7:
            render_tab_market_folding(symbol)
        with t8:
            render_tab_vwap_obv(symbol)
        with t9:
            render_tab_vol_cone(symbol)
        with t10:
            render_tab_friday_predictor(symbol, date, hist_df, spot)
        with t11:
            render_tab_friday_predictor_plus(symbol, w, hist_df, spot)
        with t12:
            render_tab_vanna_charm(symbol, date, spot, hist_df)
        with t13:
            render_tab_interpretation_engine(symbol, spot, df, hist_df, expiry_date=str(date))



    else:
        st.info("Query a symbol and click 'Fetch Data' to begin.")


if __name__ == "__main__":
    main()