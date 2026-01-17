import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt_obj
import yfinance as yf
from ..helpers.calculations import compute_ma_stack_and_regime, realized_vol_annualized, structure_label, compute_key_levels, compute_opening_range, iv_proxy_rank, bs_vanna_charm, build_trade_bias, confidence_score
from ..helpers.data_fetching import fetch_price_history, fetch_intraday
from ..helpers.ui_components import st_df

def render_tab_pro_edge(symbol, date, spot_now, hist_df, totals, df):
    if spot_now is None or spot_now <= 0:
        st.warning("Spot price is not available. Enter a manual spot to unlock all calculations.")
        return

    hist_daily = hist_df.copy() if hist_df is not None and not hist_df.empty else pd.DataFrame()
    if hist_daily.empty or not set(["High", "Low", "Close"]).issubset(set(hist_daily.columns)):
        try:
            hist_daily = fetch_price_history(symbol, period="1y", interval="1d")
        except Exception: pass

    if "Date" not in hist_daily.columns:
        hist_daily = hist_daily.reset_index().rename(columns={"index": "Date"})

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
        st.warning("Not enough daily history to compute MA regime.")

    st.markdown("### Structure (HH/HL vs LH/LL)")
    struct = structure_label(hist_daily, lookback=50)
    if struct["ok"]:
        st.write(struct["label"])
    else:
        st.info("Structure label unavailable.")

    st.markdown("### Key Levels (Support/Resistance)")
    lvl = compute_key_levels(hist_daily)
    if lvl["ok"]:
        level_df = pd.DataFrame([{"Prev High": lvl["prev_high"], "Prev Low": lvl["prev_low"], "Prev Close": lvl["prev_close"], "Weekly High (5d)": lvl["wk_high"], "Weekly Low (5d)": lvl["wk_low"]}])
        st_df(level_df, height=80)

    st.markdown("### Opening Range (first 30 minutes)")
    intra = fetch_intraday(symbol, period="5d", interval="5m")
    orr = compute_opening_range(intra, minutes=30)
    if orr["ok"]:
        or_high, or_low = orr["or_high"], orr["or_low"]
        st.write(f"Session: **{orr['session_date']}** | OR High: **{or_high:.2f}** | OR Low: **{or_low:.2f}**")
        if spot_now > or_high: st.success("OR Breakout ↑ (spot above OR high)")
        elif spot_now < or_low: st.error("OR Breakdown ↓ (spot below OR low)")
        else: st.info("Inside Opening Range (chop risk)")

    st.markdown("### Gamma Regime")
    try: net_gex = float(totals.get("net_gex", totals.get("Net GEX", float("nan"))))
    except Exception: net_gex = None

    if net_gex is None or pd.isna(net_gex):
        st.info("Weekly GEX totals not available.")
        gex_regime = "UNKNOWN"
    else:
        if net_gex < 0:
            gex_regime = "NEGATIVE GAMMA 💥"
            st.error(f"{gex_regime} | Net GEX: {net_gex:,.0f}")
        elif net_gex > 0:
            gex_regime = "POSITIVE GAMMA 🧱"
            st.success(f"{gex_regime} | Net GEX: {net_gex:,.0f}")
        else:
            gex_regime = "NEUTRAL GAMMA"
            st.write(f"{gex_regime} | Net GEX: {net_gex:,.0f}")

    st.markdown("### IV + Vanna + Charm (Approx)")
    atm_iv = None
    try:
        if not df.empty and "Strike" in df.columns:
            tmp = df.copy()
            tmp["Strike"] = pd.to_numeric(tmp["Strike"], errors="coerce")
            tmp = tmp.dropna(subset=["Strike"])
            tmp["dist"] = (tmp["Strike"] - float(spot_now)).abs()
            atm_row = tmp.sort_values("dist").iloc[0]
            c_iv, p_iv = pd.to_numeric(atm_row.get("Call IV", np.nan), errors="coerce"), pd.to_numeric(atm_row.get("Put IV", np.nan), errors="coerce")
            if pd.notna(c_iv) and pd.notna(p_iv): atm_iv = float((c_iv + p_iv) / 2.0)
            elif pd.notna(c_iv): atm_iv = float(c_iv)
            elif pd.notna(p_iv): atm_iv = float(p_iv)
    except Exception: pass

    iv_rank_proxy = None
    if atm_iv is not None and atm_iv > 0:
        iv_rank = iv_proxy_rank(atm_iv, hist_daily, window=20)
        iv_rank_proxy = float(iv_rank["iv_proxy_rank"]) if iv_rank.get("ok") else None
        c1, c2, c3 = st.columns(3)
        c1.metric("ATM IV", f"{atm_iv:.2%}")
        if iv_rank_proxy is not None: c2.metric("IV Rank (proxy)", f"{iv_rank_proxy:.0f}/100")
        rv = realized_vol_annualized(hist_daily, window=20)
        if pd.notna(rv) and atm_iv > 0: c3.metric("RV20 vs IV", f"{rv/atm_iv:.2f}x")

    st.markdown("### Trade Bias + Checklist")
    bias_text = build_trade_bias(ma["label"] if ma.get("ok") else "N/A", gex_regime, iv_rank_proxy)
    st.markdown(bias_text)
    
    structure_ok = struct.get("ok") and ("BULL" in struct.get("label", "") or "BEAR" in struct.get("label", ""))
    or_ok = bool(orr.get("ok")) and (spot_now > orr.get("or_high", float("inf")) or spot_now < orr.get("or_low", float("-inf")))
    iv_ok = 30 <= iv_rank_proxy <= 80 if iv_rank_proxy is not None else False
    
    score = confidence_score(ma.get("strength", 0), structure_ok, iv_ok, gex_regime != "UNKNOWN", or_ok)
    st.metric("Confidence Score", score)
