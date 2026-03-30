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
    fetch_expirations,
    fetch_spot_quote,
    fetch_options,
    fetch_weekly_summary,
    fetch_weekly_gex,
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
from stats_app.tabs.tab_friday_calculation_6_weeks import render_tab_friday_calculation_6_weeks
from stats_app.tabs.tab_vanna_charm import render_tab_vanna_charm
from stats_app.tabs.tab_interpretation_engine import render_tab_interpretation_engine
from stats_app.tabs.tab_orderflow_delta import render_tab_orderflow_delta
from stats_app.tabs.tab_share_statistics import render_tab_share_statistics
from stats_app.tabs.tab_yahoo_data import render_tab_yahoo_data
from stats_app.tabs.tab_market_signals import render_tab_market_signals
from stats_app.tabs.tab_envelope_gator_signal import render_tab_envelope_gator_signals
from stats_app.tabs.tab_ml_rsi_pro import render_tab_ml_rsi_pro
from stats_app.tabs.tab_crossover_strategy_guide import render_tab_crossover_strategy_guide
from stats_app.tabs.tab_trend_engine import render_tab_trend_engine
from stats_app.tabs.tab_friday_playbook import render_tab_friday_playbook
from stats_app.tabs.tab_capital_flow import render_tab_capital_flow

# ✅ Existing Nobel tab (you already added)
from stats_app.tabs.tab_nobel_pattern import render_tab_nobel_pattern
from stats_app.tabs.tab_nobel_predictor_firecrawl import render_tab_nobel_predictor_firecrawl

# ✅ NEW TABS (existing in your app)
from stats_app.tabs.tab_expected_move import render_tab_expected_move
from stats_app.tabs.tab_gamma_flip_detector import render_tab_gamma_flip_detector
from stats_app.tabs.tab_iv_term_structure import render_tab_iv_term_structure
from stats_app.tabs.tab_trending_oi import render_tab_trending_oi

# ✅ NEW TAB (Microstructure Engine)
from stats_app.tabs.tab_microstructure_engine import render_tab_microstructure_engine


# Configure Streamlit Page
st.set_page_config(
    page_title="Stats Dashboard | Options & Gamma",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _to_date(date_str: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(date_str))
    except Exception:
        return None


def _week_expiry_candidates(target_expiry_str: str) -> list[str]:
    d = _to_date(target_expiry_str)
    if d is None:
        return [target_expiry_str]
    candidates = [d.isoformat()]
    if d.weekday() == 4:
        prev = (d - dt.timedelta(days=1)).isoformat()
        if prev not in candidates:
            candidates.append(prev)
    return candidates


def _is_not_found_result(res) -> bool:
    if not isinstance(res, dict) or res.get("success"):
        return False
    status_code = res.get("status_code")
    err_text = str(res.get("error") or "").lower()
    if status_code == 404:
        return True
    not_found_signals = (
        "404",
        "not found",
        "no options data found",
        "no data returned",
        "no contracts found",
    )
    return any(signal in err_text for signal in not_found_signals)


def main():
    apply_custom_styles()

    def _backend_health_state() -> tuple[bool, int]:
        now_ts = time.time()
        last_check_ts = float(st.session_state.get("backend_health_last_check_ts", 0.0))
        last_status = st.session_state.get("backend_health_status")
        fail_streak = int(st.session_state.get("backend_health_fail_streak", 0))

        success_interval = 30.0
        failure_interval = 8.0
        check_interval = failure_interval if last_status is False else success_interval

        should_check = (last_status is None) or ((now_ts - last_check_ts) >= check_interval)
        if should_check:
            is_ok = check_api()
            st.session_state["backend_health_last_check_ts"] = now_ts
            st.session_state["backend_health_status"] = is_ok
            if is_ok:
                fail_streak = 0
                st.session_state["backend_health_last_ok_ts"] = now_ts
            else:
                fail_streak += 1
            st.session_state["backend_health_fail_streak"] = fail_streak
            last_status = is_ok

        return bool(last_status), fail_streak

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

    api_ok, api_fail_streak = _backend_health_state()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Options Query")
        symbol = st.text_input("Symbol", value="MU").upper().strip()

        def _next_friday(d: dt.date) -> dt.date:
            days_ahead = (4 - d.weekday()) % 7
            return d + dt.timedelta(days=days_ahead)

        today = dt.date.today()
        default_friday = _next_friday(today)
        default_date = default_friday.isoformat()
        expirations_res = fetch_expirations(symbol, default_date) if api_ok and symbol else None
        expirations_payload = expirations_res.get("data", {}) if isinstance(expirations_res, dict) and expirations_res.get("success") else {}
        expirations = expirations_payload.get("expirations", []) if isinstance(expirations_payload, dict) else []

        if expirations:
            label_map = {
                str(item.get("date") or ""): str(item.get("label") or item.get("date") or "")
                for item in expirations
                if str(item.get("date") or "").strip()
            }
            date_options = list(label_map.keys())
            default_index = date_options.index(default_date) if default_date in label_map else 0
            date = st.selectbox(
                "Expiration Date",
                options=date_options,
                index=default_index,
                format_func=lambda d: label_map.get(str(d), str(d)),
            )
        else:
            expiry_date = st.date_input("Expiration Date", value=default_friday)
            date = expiry_date.isoformat()

        spot_source = st.selectbox("Spot Price Source", options=["CNBC", "Manual"])
        refresh_spot_btn = st_btn("🔄 Refresh Spot")
        auto_refresh = st.checkbox("Auto-refresh spot", value=False)
        refresh_interval = st.slider("Refresh interval (sec)", 5, 60, 15, step=5)
        if auto_refresh and st_autorefresh:
            st_autorefresh(interval=refresh_interval * 1000, key=f"spot_refresh_{symbol}")
        elif auto_refresh and not st_autorefresh:
            st.info("Auto-refresh unavailable (missing streamlit-autorefresh).")

        st.markdown("## ⏱️ Friday Stock Confirmation")
        friday_twelve_interval = st.selectbox(
            "TwelveData Interval",
            options=["1min", "5min", "15min"],
            index=1,
        )
        friday_twelve_outputsize = st.slider(
            "TwelveData Candles",
            min_value=30,
            max_value=200,
            value=100,
            step=10,
        )

        # --------- per-symbol spot cache ---------
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

            backend = fetch_spot_quote(symbol, date, force_refresh=refresh_spot_btn)
            if backend and backend.get("success"):
                spot_data = backend.get("data")
                if spot_data and not spot_data.get("source"):
                    spot_data["source"] = "Backend"
            else:
                spot_error = backend.get("error") if backend else "Backend spot fetch failed"

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

        if not api_ok:
            if api_fail_streak >= 3:
                st.warning("Backend health check failed repeatedly. Fetch may fail or be slow.")
            else:
                st.caption("Backend health check is unstable (transient). Retrying automatically.")
        fetch_btn = st_btn("🔄 Fetch Data")

    # --------- reset session_state when symbol changes ---------
    if "last_symbol" not in st.session_state:
        st.session_state["last_symbol"] = symbol

    if st.session_state["last_symbol"] != symbol:
        st.session_state["options_result"] = None
        st.session_state["weekly_result"] = None
        st.session_state["gex_result"] = None
        st.session_state["hist_df"] = pd.DataFrame()
        st.session_state["spot_at_fetch"] = None
        st.session_state["effective_expiry_date"] = None
        st.session_state["barchart_direct_auth"] = None
        st.session_state["last_symbol"] = symbol

    # -------------------------------------------------------------------------
    # FETCHING LOGIC WITH RETRIES
    # -------------------------------------------------------------------------
    if fetch_btn:

        def fetch_with_retry(fetch_func, func_name, max_retries=3, *args):
            placeholder = st.empty()
            for attempt in range(max_retries):
                try:
                    res = fetch_func(*args)
                    if isinstance(res, dict) and res.get("success"):
                        placeholder.empty()
                        return res
                    if isinstance(res, pd.DataFrame) and not res.empty:
                        placeholder.empty()
                        return res

                    err = res.get("error") if isinstance(res, dict) else "Unknown Error"
                    if attempt < max_retries - 1:
                        placeholder.warning(
                            f"⚠️ {func_name} (Attempt {attempt + 1}/{max_retries}) failed: {err}. Retrying..."
                        )
                        time.sleep(2)
                    else:
                        placeholder.error(f"❌ {func_name} failed: {err}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        placeholder.warning(
                            f"⚠️ {func_name} (Attempt {attempt + 1}/{max_retries}) crashed: {e}. Retrying..."
                        )
                        time.sleep(2)
                    else:
                        placeholder.error(f"❌ {func_name} crashed: {e}")
            return None

        def fetch_core_market_data(requested_date: str):
            candidate_dates = _week_expiry_candidates(requested_date)
            last_options_res = None
            last_weekly_res = None
            last_gex_res = None

            for idx, candidate_date in enumerate(candidate_dates):
                if candidate_date != requested_date:
                    st.info(
                        f"No chain found for {requested_date}. Retrying weekly expiry fallback {candidate_date}."
                    )

                options_res = fetch_with_retry(
                    fetch_options, "Options Chain", 3, symbol, candidate_date, True
                )
                last_options_res = options_res
                if not (isinstance(options_res, dict) and options_res.get("success")):
                    if idx < len(candidate_dates) - 1 and _is_not_found_result(options_res):
                        continue
                    return candidate_date, options_res, last_weekly_res, last_gex_res

                weekly_res = fetch_with_retry(
                    fetch_weekly_summary, "Weekly Summary", 3, symbol, candidate_date, spot
                )
                last_weekly_res = weekly_res
                if not (isinstance(weekly_res, dict) and weekly_res.get("success")):
                    if idx < len(candidate_dates) - 1 and _is_not_found_result(weekly_res):
                        continue
                    gex_res = fetch_with_retry(
                        fetch_weekly_gex, "Weekly GEX", 3, symbol, candidate_date, spot
                    )
                    last_gex_res = gex_res
                    return candidate_date, options_res, weekly_res, gex_res

                gex_res = fetch_with_retry(
                    fetch_weekly_gex, "Weekly GEX", 3, symbol, candidate_date, spot
                )
                last_gex_res = gex_res
                return candidate_date, options_res, weekly_res, gex_res

            return requested_date, last_options_res, last_weekly_res, last_gex_res

        with st.spinner(f"Analyzing market structure for {symbol}..."):
            effective_date, options_res, weekly_res, gex_res = fetch_core_market_data(date)
            st.session_state["effective_expiry_date"] = effective_date
            st.session_state["options_result"] = options_res

            options_payload = (st.session_state.get("options_result") or {}).get("data", {})
            direct_auth = options_payload.get("direct_auth") if isinstance(options_payload, dict) else None
            if isinstance(direct_auth, dict) and direct_auth.get("cookie_header"):
                st.session_state["barchart_direct_auth"] = direct_auth

            st.session_state["weekly_result"] = weekly_res
            st.session_state["gex_result"] = gex_res

            try:
                st.session_state["hist_df"] = fetch_price_history(symbol).copy()
            except Exception as e:
                st.warning(f"Could not load price history: {e}")
                st.session_state["hist_df"] = pd.DataFrame()

            st.session_state["spot_at_fetch"] = spot
    elif "effective_expiry_date" not in st.session_state:
        st.session_state["effective_expiry_date"] = None

    options_result = st.session_state.get("options_result")
    weekly_result = st.session_state.get("weekly_result")
    gex_result = st.session_state.get("gex_result")
    hist_df = st.session_state.get("hist_df")
    analysis_date = st.session_state.get("effective_expiry_date") or date

    has_chain_data = bool(options_result and options_result.get("success"))
    chain_df_for_playbook = (
        pd.DataFrame(options_result["data"].get("data", []))
        if has_chain_data
        else pd.DataFrame()
    )

    has_core_data = bool(options_result and weekly_result and options_result.get("success"))
    df = pd.DataFrame()
    w = {}
    totals, pcr = {}, {}
    gex_df = pd.DataFrame()

    if has_core_data:
        df = pd.DataFrame(options_result["data"].get("data", []))
        w = weekly_result["data"]
        totals = w.get("totals", {})
        pcr = w.get("pcr", {})
        gex_df = pd.DataFrame(gex_result["data"].get("data", [])) if gex_result and gex_result.get("success") else pd.DataFrame()
        spot_for_gex_views = float(w.get("spot") or st.session_state.get("spot_at_fetch") or spot)

        if analysis_date != date:
            st.info(f"Using weekly expiry fallback {analysis_date} for chain and gamma calculations.")

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
    elif fetch_btn and api_ok:
        st.error("Data fetch failed after multiple retries. Please check the backend connection.")
    else:
        spot_for_gex_views = float(st.session_state.get("spot_at_fetch") or spot)

    # ---------------- Stateful navigation (keeps selected tab across reruns) ----------------
    tab_labels = [
        "📋 Chain",
        "📊 OI",
        "📈 Trending OI",
        "📌 Weekly GEX",
        "🧲 Map",
        "🎰 Microstructure",
        "🧮 Greeks",
        "🏆 Pro Edge",
        "🔳 Folding",
        "🧠 Nobel Pattern",
        "🧠 Nobel Predictor (FC)",
        "📦 Expected Move",
        "🧲 Gamma Flip",
        "🧾 IV Term Structure",
        "📈 VWAP",
        "🎯 Vol Cone",
        "🔮 Friday Predictor",
        "🧠 Friday Predictor+",
        "🗓️ Friday Calculation (6 Weeks)",
        "📜 Friday Playbook",
        "🌊 Vanna/Charm",
        "📊 Orderflow/Delta",
        "🧠 Interpretation",
        "🧾 Share Stats",
        "📈 Yahoo Data",
        "📡 Market Signals",
        "🧭 Envelope Gator",
        "🧠 ML RSI Pro",
        "🧭 Crossover Strategy Guide",
        "📉 Trend Engine",
        "💸 Capital Flow",
    ]
    if "active_main_tab" not in st.session_state or st.session_state["active_main_tab"] not in tab_labels:
        st.session_state["active_main_tab"] = tab_labels[0]

    active_tab = st.pills(
        "Dashboard Tabs",
        options=tab_labels,
        selection_mode="single",
        key="active_main_tab",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_tab is None:
        active_tab = st.session_state.get("active_main_tab", tab_labels[0])

    def _show_core_fetch_hint():
        st.info("Click `🔄 Fetch Data` in the sidebar to load this tab.")

    if active_tab == "📋 Chain":
        if has_core_data:
            render_tab_options_chain(df)
        else:
            _show_core_fetch_hint()

    elif active_tab == "📊 OI":
        if has_core_data:
            render_tab_oi_charts(df)
        else:
            _show_core_fetch_hint()

    elif active_tab == "📈 Trending OI":
        if has_core_data:
            render_tab_trending_oi(df=df, spot=spot, symbol=symbol, expiry_date=str(analysis_date))
        else:
            _show_core_fetch_hint()

    elif active_tab == "📌 Weekly GEX":
        if has_core_data:
            render_tab_weekly_gamma(pcr, totals, w, spot_for_gex_views, gex_df)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🧲 Map":
        if has_core_data:
            render_tab_gamma_map_filters(symbol, analysis_date, spot_for_gex_views, gex_df if not gex_df.empty else pd.DataFrame())
        else:
            _show_core_fetch_hint()

    elif active_tab == "🎰 Microstructure":
        if has_core_data and not gex_df.empty:
            render_tab_microstructure_engine(
                symbol=symbol,
                spot=spot,
                gex_df=gex_df,
                expected_move=None,
                gamma_flip_strike=None,
                iv_annual=None,
            )
        else:
            st.info("Run Fetch Data (needs Weekly GEX table).")

    elif active_tab == "🧮 Greeks":
        if has_core_data:
            render_tab_vol_greeks(df, spot, symbol, analysis_date)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🏆 Pro Edge":
        if has_core_data:
            render_tab_pro_edge(symbol, analysis_date, spot, hist_df, totals, df)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🔳 Folding":
        if has_core_data:
            render_tab_market_folding(symbol)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🧠 Nobel Pattern":
        render_tab_nobel_pattern(symbol=symbol, spot=spot, hist_df=hist_df)

    elif active_tab == "🧠 Nobel Predictor (FC)":
        render_tab_nobel_predictor_firecrawl(symbol=symbol)

    elif active_tab == "📦 Expected Move":
        if has_core_data:
            render_tab_expected_move(df=df, spot=spot, expiry_date=analysis_date, symbol=symbol)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🧲 Gamma Flip":
        if has_core_data and not gex_df.empty:
            render_tab_gamma_flip_detector(gex_df=gex_df, spot=spot, symbol=symbol)
        else:
            st.info("Run Fetch Data (needs Weekly GEX table).")

    elif active_tab == "🧾 IV Term Structure":
        if has_core_data:
            render_tab_iv_term_structure(df=df, spot=spot, expiry_date=analysis_date, symbol=symbol)
        else:
            _show_core_fetch_hint()

    elif active_tab == "📈 VWAP":
        if has_core_data:
            render_tab_vwap_obv(symbol)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🎯 Vol Cone":
        if has_core_data:
            render_tab_vol_cone(symbol)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🔮 Friday Predictor":
        if has_core_data:
            render_tab_friday_predictor(symbol, analysis_date, hist_df, spot)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🧠 Friday Predictor+":
        if has_core_data:
            render_tab_friday_predictor_plus(symbol, w, hist_df, spot)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🗓️ Friday Calculation (6 Weeks)":
        render_tab_friday_calculation_6_weeks(
            symbol=symbol,
            spot=spot,
            direct_auth=st.session_state.get("barchart_direct_auth"),
        )

    elif active_tab == "📜 Friday Playbook":
        render_tab_friday_playbook(
            symbol,
            spot,
            chain_df_for_playbook,
            gex_df if has_core_data and not gex_df.empty else pd.DataFrame(),
            twelve_interval=friday_twelve_interval,
            twelve_outputsize=friday_twelve_outputsize,
        )

    elif active_tab == "🌊 Vanna/Charm":
        if has_core_data:
            render_tab_vanna_charm(symbol, analysis_date, spot, hist_df)
        else:
            _show_core_fetch_hint()

    elif active_tab == "📊 Orderflow/Delta":
        if has_core_data:
            render_tab_orderflow_delta(symbol, hist_df, spot)
        else:
            _show_core_fetch_hint()

    elif active_tab == "🧠 Interpretation":
        if has_core_data:
            render_tab_interpretation_engine(symbol, spot, df, hist_df, expiry_date=str(analysis_date))
        else:
            _show_core_fetch_hint()

    elif active_tab == "🧾 Share Stats":
        if has_core_data:
            render_tab_share_statistics(symbol, gex_df=gex_df, spot=spot)
        else:
            _show_core_fetch_hint()

    elif active_tab == "📈 Yahoo Data":
        render_tab_yahoo_data(symbol)

    elif active_tab == "📡 Market Signals":
        render_tab_market_signals(symbol)

    elif active_tab == "🧭 Envelope Gator":
        render_tab_envelope_gator_signals(symbol)

    elif active_tab == "🧠 ML RSI Pro":
        render_tab_ml_rsi_pro(symbol)

    elif active_tab == "🧭 Crossover Strategy Guide":
        render_tab_crossover_strategy_guide(symbol)

    elif active_tab == "📉 Trend Engine":
        render_tab_trend_engine(symbol)

    elif active_tab == "💸 Capital Flow":
        if has_core_data:
            render_tab_capital_flow(df, spot=spot, expiry_date=analysis_date, symbol=symbol)
        else:
            _show_core_fetch_hint()


if __name__ == "__main__":
    main()
