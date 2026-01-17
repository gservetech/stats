# stats_app/tabs/tab_friday_predictor.py
"""Friday Expiry Predictor: AAPL 3-Engine Risk Book & Staged Timing"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from stats_app.helpers.api_client import API_BASE_URL
from stats_app.helpers.ui_components import st_plot


# -------------------------
# 1. CORE DATA & GEX MATH
# -------------------------

def _fetch_weekly_gex_table(symbol: str, expiry_date: str, spot: float) -> pd.DataFrame:
    r = requests.get(
        f"{API_BASE_URL}/weekly/gex",
        params={"symbol": symbol.upper(), "date": str(expiry_date), "spot": float(spot)},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    df = pd.DataFrame(rows)
    for c in ["strike", "call_gex", "put_gex", "net_gex"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)


def _get_gamma_levels(gex_df: pd.DataFrame, spot: float) -> dict:
    d = gex_df.copy()
    d["abs_net"] = d["net_gex"].abs()
    d["dist"] = (d["strike"] - float(spot)).abs()
    d["pin_score"] = d["abs_net"] / (d["dist"] + 1.0)
    magnet_row = d.loc[d["pin_score"].idxmax()]
    put_wall_row = d.loc[d["put_gex"].idxmax()]
    call_wall_row = d.loc[d["call_gex"].idxmax()]
    return {
        "put_wall": float(put_wall_row["strike"]),
        "call_wall": float(call_wall_row["strike"]),
        "magnet": float(magnet_row["strike"])
    }


def _nearest_strike(strikes, target, side="nearest"):
    if len(strikes) == 0: return float(target)
    if side == "below":
        candidates = strikes[strikes <= target]
        return float(candidates.max()) if len(candidates) > 0 else float(strikes.min())
    if side == "above":
        candidates = strikes[strikes >= target]
        return float(candidates.min()) if len(candidates) > 0 else float(strikes.max())
    return float(strikes[np.argmin(np.abs(strikes - target))])


# -------------------------
# 2. TAB RENDERER
# -------------------------

def render_tab_friday_predictor(symbol: str, expiry_date: str, hist_df: pd.DataFrame, spot: float):
    st.subheader(f"🧠 Professional Risk Book & Gamma Map: {symbol}")

    with st.sidebar:
        st.header("🎯 Broker Sync")
        # Updated to your AAPL Ameritrade VWAP
        vwap_last = st.number_input("Ameritrade VWAP Pivot", value=255.33, step=0.01)
        st.divider()
        st.header("⚙️ Book Construction")
        wing_width = st.slider("Condor Wing Width", 5, 20, 10)
        hedge_width = st.slider("Hedge Spread Width", 5, 30, 20)

    if hist_df is None or hist_df.empty:
        st.warning("Please fetch market data.")
        return

    try:
        gex_df = _fetch_weekly_gex_table(symbol, expiry_date, spot)
        levels = _get_gamma_levels(gex_df, spot)
        strikes = gex_df["strike"].values
    except Exception as e:
        st.error(f"GEX Error: {e}")
        return

    # Strength threshold (0.2% buffer) and Weakness threshold
    strength_threshold = vwap_last * 1.002
    is_strong = spot >= strength_threshold
    is_weak = spot < vwap_last

    # --- 🧲 GAMMA MAP ---
    st.write("### 🧲 Gamma Map (Magnets / Walls / Pivot)")
    fig_map = go.Figure()
    fig_map.add_vline(x=levels["put_wall"], line_dash="dash", line_color="red", annotation_text="Lower Wall")
    fig_map.add_vline(x=levels["call_wall"], line_dash="dash", line_color="green", annotation_text="Upper Wall")
    fig_map.add_vline(x=levels["magnet"], line_color="gold", line_width=4, annotation_text="Magnet")
    fig_map.add_vline(x=vwap_last, line_dash="dot", line_color="white", annotation_text="VWAP Pivot")
    fig_map.add_trace(go.Scatter(x=[spot], y=[0], mode="markers+text", text=[f"${spot:.2f}"],
                                 marker=dict(size=15, color="cyan", symbol="diamond")))
    fig_map.update_layout(template="plotly_dark", height=300, yaxis_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    st_plot(fig_map)

    # --- 📈 3-ENGINE PAYOFF ---
    condor = {
        "sell_put": _nearest_strike(strikes, levels["put_wall"], "below"),
        "buy_put": _nearest_strike(strikes, levels["put_wall"] - wing_width, "below"),
        "sell_call": _nearest_strike(strikes, levels["call_wall"], "above"),
        "buy_call": _nearest_strike(strikes, levels["call_wall"] + wing_width, "above")
    }
    call_h = {"buy": _nearest_strike(strikes, levels["call_wall"], "nearest"),
              "sell": _nearest_strike(strikes, levels["call_wall"] + hedge_width, "above")}
    put_h = {"buy": _nearest_strike(strikes, levels["put_wall"], "nearest"),
             "sell": _nearest_strike(strikes, levels["put_wall"] - hedge_width, "below")}

    st.write("### 📈 3-Engine Payoff Diagram")
    x_range = np.linspace(spot * 0.8, spot * 1.2, 300)
    y_pl = np.zeros_like(x_range)
    # IC Payoff Logic
    y_pl += np.where(x_range < condor["sell_put"], x_range - condor["sell_put"], 0)
    y_pl -= np.where(x_range < condor["buy_put"], x_range - condor["buy_put"], 0)
    y_pl += np.where(x_range > condor["sell_call"], condor["sell_call"] - x_range, 0)
    y_pl -= np.where(x_range > condor["buy_call"], condor["buy_call"] - x_range, 0)
    # Hedge Logic
    y_pl += np.where(x_range > call_h["buy"], x_range - call_h["buy"], 0)
    y_pl -= np.where(x_range > call_h["sell"], x_range - call_h["sell"], 0)
    y_pl -= np.where(x_range < put_h["buy"], put_h["buy"] - x_range, 0)
    y_pl += np.where(x_range < put_h["sell"], put_h["sell"] - x_range, 0)

    fig_pl = go.Figure()
    fig_pl.add_trace(
        go.Scatter(x=x_range, y=y_pl, fill='tozeroy', line=dict(color='cyan', width=3), name="Full Book P&L"))
    fig_pl.update_layout(template="plotly_dark", height=400, xaxis_title="Expiry Price", yaxis_title="Risk Book P&L",
                         margin=dict(l=20, r=20, t=20, b=20))
    st_plot(fig_pl)

    # --- 🧠 THE PROFESSIONAL SOLUTION: TIMING + STAGING ---
    st.divider()
    st.markdown("### 🧠 The Professional Solution: TIMING + STAGING")
    st.markdown(
        f"**Rule of Thumb**: Never sell puts into weakness (< {vwap_last:.2f}). Sell puts into strength (> {strength_threshold:.2f}).")

    col_mon, col_tue = st.columns(2)
    with col_mon:
        st.markdown(f"""
        #### 🥇 MONDAY (Discovery)
        * **Action**: Do NOT sell the put spread yet.
        * **Goal**: Let market show its hand.
        * **Hedges**: 3W Call Spread can be opened early.
        """)

    with col_tue:
        if is_strong:
            st.success(f"""
            #### 🥈 TUESDAY (STRENGTH)
            * **Status**: Price stable above {strength_threshold:.2f}.
            * **Action**: ✅ Complete the Iron Condor structure.
            * **Logic**: Market accepted higher prices; downside risk is lower.
            """)
        elif is_weak:
            st.error(f"""
            #### 🥈 TUESDAY (WEAKNESS)
            * **Status**: Price below {vwap_last:.2f}.
            * **Action**: ❌ **DO NOT** sell {condor['sell_put']} yet.
            * **Action**: ✅ Wait for lower support or move strikes lower.
            """)
        else:
            st.warning(
                f"#### 🥈 TUESDAY (NEUTRAL)\n- **Status**: Price in 'No-Man's Land' (${vwap_last:.2f} - ${strength_threshold:.2f}).\n- **Action**: Wait for {strength_threshold:.2f} to break before selling puts.")

    # --- EXIT RULES & LEVELS ---
    st.divider()
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**🛑 CRITICAL EXIT RULES**")
        st.markdown("- **Weekly Condor**: Close at +70-80% profit or if one side = 2x credit.")
        st.markdown("- **3W Hedges**: Take profit at +80% to +150%.")
        st.markdown("- **Safety**: Never hold past Thursday afternoon.")
    with e2:
        st.markdown("**🧱 Current GEX Strategy Levels**")
        df_levels = pd.DataFrame([
            {"Engine": "Weekly Condor (P)", "Sell/Buy Zone": f"{condor['sell_put']} / {condor['buy_put']}"},
            {"Engine": "Weekly Condor (C)", "Sell/Buy Zone": f"{condor['sell_call']} / {condor['buy_call']}"},
            {"Engine": "3W Call Hedge", "Sell/Buy Zone": f"{call_h['buy']} / {call_h['sell']}"},
            {"Engine": "3W Put Hedge", "Sell/Buy Zone": f"{put_h['buy']} / {put_h['sell']}"}
        ])
        st.table(df_levels)