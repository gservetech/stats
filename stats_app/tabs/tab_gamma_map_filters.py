import streamlit as st
import pandas as pd
from ..helpers.api_client import fetch_weekly_gex
from ..helpers.calculations import compute_gamma_map_artifacts
from ..helpers.ui_components import st_plot, st_df, create_top_strikes_chart
from ..helpers.filters import plot_filters, kalman_message
from ..helpers.data_fetching import fetch_price_history

def short_interest_bias(
    short_shares: float,
    float_shares: float,
    avg_vol_10d: float | None = None,
    short_shares_prior: float | None = None,
    short_ratio: float | None = None
):
    """
    Returns a lightweight bias signal based on short interest.
    This is NOT a price prediction - it's positioning context.
    """

    # Guardrails
    if not float_shares or float_shares <= 0:
        return {"label": "N/A", "direction": "Unknown", "score": 0, "notes": ["Float missing/invalid."]}

    short_pct_float = (short_shares / float_shares) * 100.0

    # Core scoring: higher short% can mean bearish positioning OR squeeze potential,
    # but "direction" depends on trend/flow elsewhere. We'll label it as "crowding".
    score = 0
    notes = []

    # Short % of float tiers (crowding)
    if short_pct_float < 3:
        score += 1
        notes.append(f"Low short interest ({short_pct_float:.2f}% of float) -> not a strong bearish signal.")
    elif short_pct_float < 8:
        score += 0
        notes.append(f"Moderate short interest ({short_pct_float:.2f}% of float) -> mixed/neutral.")
    elif short_pct_float < 15:
        score -= 1
        notes.append(f"High short interest ({short_pct_float:.2f}% of float) -> bearish positioning / squeeze watch.")
    else:
        score -= 2
        notes.append(f"Very high short interest ({short_pct_float:.2f}% of float) -> crowded short / squeeze conditions possible.")

    # Days to cover / short ratio (covering pressure)
    if short_ratio is not None:
        if short_ratio < 2:
            score += 1
            notes.append(f"Low days-to-cover ({short_ratio:.2f}) -> shorts can exit easily (less squeeze fuel).")
        elif short_ratio < 5:
            notes.append(f"Medium days-to-cover ({short_ratio:.2f}) -> some cover pressure possible.")
        else:
            score -= 1
            notes.append(f"High days-to-cover ({short_ratio:.2f}) -> potential cover pressure (squeeze risk if trend flips up).")

    # Change in short shares
    if short_shares_prior is not None and short_shares_prior > 0:
        delta = short_shares - short_shares_prior
        delta_pct = (delta / short_shares_prior) * 100.0
        if delta_pct > 5:
            score -= 1
            notes.append(f"Shorts increased meaningfully (+{delta_pct:.2f}%).")
        elif delta_pct < -5:
            score += 1
            notes.append(f"Shorts decreased meaningfully ({delta_pct:.2f}%).")
        else:
            notes.append(f"Shorts changed modestly ({delta_pct:+.2f}%).")

    # Optional: short shares vs 10d volume proxy
    if avg_vol_10d:
        cover_days_proxy = short_shares / avg_vol_10d
        notes.append(f"Cover proxy (Short/AvgVol10d): {cover_days_proxy:.2f} days")

    # Translate score to label (contextual bias, not a trade signal)
    if score >= 2:
        direction = "Neutral -> Slightly Bullish"
        label = "LOW SHORT PRESSURE"
    elif score == 1:
        direction = "Neutral"
        label = "LIGHT SHORTING"
    elif score == 0:
        direction = "Neutral"
        label = "MIXED"
    else:
        direction = "Neutral -> Slightly Bearish"
        label = "RISING/HEAVIER SHORTING"

    return {
        "label": label,
        "direction": direction,
        "score": score,
        "short_pct_float": short_pct_float,
        "notes": notes
    }

def render_tab_gamma_map_filters(symbol, date, spot):
    st.subheader("⌛ Gamma Map (Magnets / Walls / Box)")
    
    # Imports for plotting inside tab
    import plotly.graph_objects as go
    
    def plot_net_gex_map(df, spot, art):
        df = df.copy()
        df["net_gex"] = pd.to_numeric(df["net_gex"], errors="coerce").fillna(0.0)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(0.0)
        df = df.sort_values("strike")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["strike"], y=df["net_gex"], name="Net GEX", marker_color=df["net_gex"].apply(lambda x: "#00d775" if x >= 0 else "#ff4757")))
        
        if spot:
            fig.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot:g}")
            
        if art:
            if art.get("call_wall"):
                fig.add_vline(x=art["call_wall"], line_color="#00d775", line_width=2, annotation_text="Call Wall")
            if art.get("put_wall"):
                fig.add_vline(x=art["put_wall"], line_color="#ff4757", line_width=2, annotation_text="Put Wall")
            if art.get("zero_gamma"):
                fig.add_vline(x=art["zero_gamma"], line_dash="dot", line_color="orange", annotation_text="Zero Gamma")

        fig.update_layout(template="plotly_dark", height=500, title="Net GEX by Strike (Gamma Map)", xaxis_title="Strike", yaxis_title="Net GEX ($)")
        return fig

    with st.spinner("Loading per-strike GEX (weekly/gex) ..."):
        r_val = st.session_state.get("r_in", 0.041)
        q_val = st.session_state.get("q_in", 0.004)
        gex_result = fetch_weekly_gex(symbol, date, spot, r=r_val, q=q_val)

    if not gex_result.get("success"):
        st.warning(f"Could not load /weekly/gex: {gex_result.get('error')}")
    else:
        gex_payload = gex_result["data"]
        gex_df = pd.DataFrame(gex_payload.get("data", []) or [])

        if gex_df.empty:
            st.warning("No per-strike GEX returned from backend.")
        else:
            # Use the single source of truth function
            art = compute_gamma_map_artifacts(gex_df, spot=spot, top_n=10)
            
            if not art:
                st.warning("Could not compute gamma levels.")
            else:
                cA, cB, cC, cD = st.columns(4)
                cA.metric("Main Magnet", f"{art['magnet']:g}" if art["magnet"] is not None else "N/A")
                cB.metric("Put Wall (Lower)", f"{art['put_wall']:g}" if art["put_wall"] is not None else "N/A")
                cC.metric("Call Wall (Upper)", f"{art['call_wall']:g}" if art["call_wall"] is not None else "N/A")
                cD.metric("Spot Used", f"{art['spot_used']:.2f}" if art["spot_used"] is not None else "N/A")

                st_plot(plot_net_gex_map(gex_df, spot=spot, art=art))

                st.markdown("### 🧲 Gamma Walls (Top Call/Put GEX)")
                w1, w2 = st.columns(2)
                
                top_call = art["top_call"]
                top_put = art["top_put"]
                
                with w1:
                    st.markdown("**Top Call GEX**")
                    if not top_call.empty:
                        st_df(top_call)
                        if {"strike", "call_gex"}.issubset(top_call.columns):
                            st_plot(create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX"))
                    else:
                        st.info("Call GEX data not available.")
                with w2:
                    st.markdown("**Top Put GEX**")
                    if not top_put.empty:
                        st_df(top_put)
                        if {"strike", "put_gex"}.issubset(top_put.columns):
                            st_plot(create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX"))
                    else:
                        st.info("Put GEX data not available.")

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

        km = kalman_message(px["Close"].values, kf_series.values, lookback=20, band_pct=0.003)
        st.markdown(
            f"""
    **Kalman Read:** {km['msg']}

    - **Regime:** **{km.get('regime', 'N/A')}**
    - **Trend:** **{km.get('trend', 'N/A')}**
    - **Bias:** **{km.get('bias', 'N/A')}**
    - **Trend strength:** **{km.get('trend_strength', 'N/A')}**
    - **Structure:** **{km.get('structure', 'N/A')}**
    - **Chop (crossings/20):** **{km.get('crossings', 'N/A')}**
    """
        )
        st.caption("Tip: McGinley adapts to speed, KAMA adapts via Efficiency Ratio, Kalman adapts via Q/R confidence.")
