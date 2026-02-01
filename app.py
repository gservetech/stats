import sys
import os

# Ensure project root is on sys.path (Streamlit-safe)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import datetime as dt
import time
import plotly.graph_objects as go
import numpy as np
import requests

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# Import modular components (ABSOLUTE imports only)
from stats_app.styles import apply_custom_styles
from stats_app.helpers.api_client import (
    check_api,
    fetch_spot_quote,
    fetch_options,
    fetch_weekly_summary,
    API_BASE_URL,
)
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
from stats_app.tabs.tab_friday_predictor import render_tab_friday_predictor
from stats_app.tabs.tab_friday_predictor_plus import render_tab_friday_predictor_plus
from stats_app.tabs.tab_vanna_charm import render_tab_vanna_charm
from stats_app.tabs.tab_interpretation_engine import render_tab_interpretation_engine
from stats_app.tabs.tab_orderflow_delta import render_tab_orderflow_delta

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

        def _next_friday(d: dt.date) -> dt.date:
            days_ahead = (4 - d.weekday()) % 7
            return d + dt.timedelta(days=days_ahead)

        today = dt.date.today()
        default_friday = _next_friday(today)
        expiry_date = st.date_input("Expiration Date", value=default_friday)

        # Always use the next Friday for calculations (e.g., if user picks Saturday)
        effective_expiry = _next_friday(expiry_date)
        if effective_expiry != expiry_date:
            st.caption(f"Using next Friday: {effective_expiry.isoformat()}")
        date = effective_expiry.isoformat()

        spot_source = st.selectbox("Spot Price Source", options=["CNBC", "Manual"])
        refresh_spot_btn = st_btn("🔄 Refresh Spot")
        auto_refresh = st.checkbox("Auto-refresh spot", value=False)
        refresh_interval = st.slider("Refresh interval (sec)", 5, 60, 15, step=5)
        if auto_refresh and st_autorefresh:
            st_autorefresh(interval=refresh_interval * 1000, key=f"spot_refresh_{symbol}")
        elif auto_refresh and not st_autorefresh:
            st.info("Auto-refresh unavailable (missing streamlit-autorefresh).")

        # --------- per-symbol spot cache (prevents symbol cross-talk) ---------
        spot_key = f"spot_data_{symbol}"
        spot_ts_key = f"spot_ts_{symbol}"
        spot_err_key = f"spot_err_{symbol}"
        if spot_key not in st.session_state:
            st.session_state[spot_key] = None
        if spot_ts_key not in st.session_state:
            st.session_state[spot_ts_key] = 0.0
        if spot_err_key not in st.session_state:
            st.session_state[spot_err_key] = None

        should_refresh = (
                refresh_spot_btn
                or (st.session_state[spot_key] is None)
                or (auto_refresh and (time.time() - st.session_state[spot_ts_key] >= refresh_interval))
        )

        if should_refresh and spot_source != "Manual" and symbol:
            spot_data = None
            spot_error = None

            # 1) Prefer backend /spot (cached + stable)
            backend = fetch_spot_quote(symbol, date)
            if backend and backend.get("success"):
                spot_data = backend.get("data")
                if spot_data and not spot_data.get("source"):
                    spot_data["source"] = "Backend"
            else:
                spot_error = backend.get("error") if backend else "Backend spot fetch failed"

            # 2) Fallback to direct CNBC/Finnhub
            if not spot_data:
                spot_data = get_spot_from_finnhub(symbol)
                if not spot_data and not get_finnhub_api_key():
                    if spot_error:
                        spot_error = f"{spot_error}; FINNHUB_API_KEY missing"
                    else:
                        spot_error = "FINNHUB_API_KEY missing"

            if spot_data:
                st.session_state[spot_key] = spot_data
                st.session_state[spot_ts_key] = time.time()
                st.session_state[spot_err_key] = None
            else:
                st.session_state[spot_err_key] = spot_error or "Spot fetch failed"

        live_spot_data = st.session_state[spot_key]
        spot_error = st.session_state.get(spot_err_key)
        live_spot = live_spot_data["spot"] if live_spot_data else None

        if live_spot_data:
            source_name = live_spot_data.get("source", "Source")
            stale_tag = " (stale)" if live_spot_data.get("stale") else ""
            st.success(f"📈 {source_name}{stale_tag}: ${live_spot:.2f}")
            last_ts = st.session_state.get(spot_ts_key, 0.0)
            if last_ts:
                st.caption(f"Last update: {dt.datetime.fromtimestamp(last_ts).strftime('%H:%M:%S')}")

            if "after_hours" in live_spot_data and live_spot_data["after_hours"]:
                ah = live_spot_data["after_hours"]
                st.info(f"🌙 After Hours: ${ah['price']:.2f} ({ah['change']:+.2f})")
        elif spot_error:
            st.warning(spot_error)

        spot_input = st.number_input(
            "Spot Price (manual fallback)",
            value=float(live_spot or 260.0),
            step=0.50
        )
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

    # -------------------------------------------------------------------------
    # IMPROVED FETCHING LOGIC WITH RETRIES
    # -------------------------------------------------------------------------
    if fetch_btn and api_ok:

        # Helper function to retry API calls
        def fetch_with_retry(fetch_func, func_name, max_retries=3, *args):
            placeholder = st.empty()
            for attempt in range(max_retries):
                try:
                    res = fetch_func(*args)
                    # Check for valid dict with success=True or valid DataFrame
                    if isinstance(res, dict) and res.get("success"):
                        placeholder.empty()
                        return res
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        placeholder.empty()
                        return res

                    # If we got a result but it says error, log it
                    err = res.get("error") if isinstance(res, dict) else "Unknown Error"
                    if attempt < max_retries - 1:
                        placeholder.warning(
                            f"⚠️ {func_name} (Attempt {attempt + 1}/{max_retries}) failed: {err}. Retrying...")
                        time.sleep(2)
                    else:
                        placeholder.error(f"❌ {func_name} failed: {err}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        placeholder.warning(
                            f"⚠️ {func_name} (Attempt {attempt + 1}/{max_retries}) crashed: {e}. Retrying...")
                        time.sleep(2)
                    else:
                        placeholder.error(f"❌ {func_name} crashed: {e}")
            return None

        with st.spinner(f"Analyzing market structure for {symbol}..."):
            # 1. Fetch Options (Critical)
            st.session_state["options_result"] = fetch_with_retry(
                fetch_options, "Options Chain", 3, symbol, date
            )

            # 2. Fetch Weekly Summary (Critical)
            st.session_state["weekly_result"] = fetch_with_retry(
                fetch_weekly_summary, "Weekly Summary", 3, symbol, date, spot
            )

            # 3. Fetch History (Usually reliable, but good to be safe)
            try:
                st.session_state["hist_df"] = fetch_price_history(symbol).copy()
            except Exception as e:
                st.warning(f"Could not load price history: {e}")
                st.session_state["hist_df"] = pd.DataFrame()

            st.session_state["spot_at_fetch"] = spot

    options_result = st.session_state.get("options_result")
    weekly_result = st.session_state.get("weekly_result")
    hist_df = st.session_state.get("hist_df")

    # Only proceed if we actually have successful data
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

        # Tabs (14 labels for 14 tab variables)
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = st.tabs(
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
                "📊 Orderflow/Delta",
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
            render_tab_orderflow_delta(symbol, hist_df, spot)
        with t14:
            render_tab_interpretation_engine(symbol, spot, df, hist_df, expiry_date=str(date))

    else:
        if fetch_btn and api_ok:
            # If we clicked fetch but ended up here, it means the retry failed.
            st.error("Data fetch failed after multiple retries. Please check the backend connection.")
        else:
            st.info("Query a symbol and click 'Fetch Data' to begin.")


if __name__ == "__main__":
    main()