import streamlit as st
import pandas as pd
from ..helpers.api_client import fetch_weekly_gex, fetch_flashalpha_gex
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

def render_tab_gamma_map_filters(symbol, date, spot, gex_df_input: pd.DataFrame | None = None):
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

    gex_df = pd.DataFrame()
    if gex_df_input is not None and not gex_df_input.empty:
        gex_df = gex_df_input.copy()
        st.caption("Using cached weekly GEX from main fetch (same dataset as other tabs).")
    else:
        with st.spinner("Loading per-strike GEX (weekly/gex) ..."):
            r_val = st.session_state.get("r_in", 0.041)
            q_val = st.session_state.get("q_in", 0.004)
            gex_result = fetch_weekly_gex(symbol, date, spot, r=r_val, q=q_val)

        if not gex_result.get("success"):
            st.warning(f"Could not load /weekly/gex: {gex_result.get('error')}")
        else:
            gex_payload = gex_result["data"]
            gex_df = pd.DataFrame(gex_payload.get("data", []) or [])

    if not gex_df.empty:

        # Use the single source of truth function
        art = compute_gamma_map_artifacts(gex_df, spot=spot, top_n=10)
        
        if not art:
            st.warning("Could not compute gamma levels.")
        else:
            gex_totals = art.get("totals", {})

            st.markdown("### Total Net GEX")
            cA, cB = st.columns(2)
            cA.metric("Total Net GEX", f"{float(gex_totals.get('net_gex', 0.0)):,.0f}")
            cB.metric("Spot Used", f"{art['spot_used']:.2f}" if art["spot_used"] is not None else "N/A")
            st.caption("This total is summed from the same per-strike weekly GEX table used by the map below.")

            st.markdown("### Key Levels")
            cE, cF, cG = st.columns(3)
            cE.metric("Main Magnet", f"{art['magnet']:g}" if art["magnet"] is not None else "N/A")
            cF.metric("Put Wall (Lower)", f"{art['put_wall']:g}" if art["put_wall"] is not None else "N/A")
            cG.metric("Call Wall (Upper)", f"{art['call_wall']:g}" if art["call_wall"] is not None else "N/A")

            st_plot(plot_net_gex_map(gex_df, spot=spot, art=art), key="gamma_map_net_gex")

            # ── Fetch FlashAlpha data upfront so we can render side-by-side ──
            fa_art = None
            fa_data = None
            with st.spinner("Fetching FlashAlpha GEX data..."):
                fa_result = fetch_flashalpha_gex(symbol, str(date))
            if fa_result.get("success"):
                fa_data = fa_result["data"]
                fa_strikes = fa_data.get("strikes", [])
                if fa_strikes:
                    fa_df = pd.DataFrame(fa_strikes)
                    fa_art = compute_gamma_map_artifacts(fa_df, spot=fa_data.get("underlying_price", spot), top_n=10)

            st.markdown("### 🧲 Gamma Walls — Barchart vs FlashAlpha")

            top_call = art["top_call"]
            top_put = art["top_put"]
            top_net = art["top_net"]

            fa_top_call = fa_art["top_call"] if fa_art else None
            fa_top_put = fa_art["top_put"] if fa_art else None
            fa_top_net = fa_art["top_net"] if fa_art else None

            # --- Row 1: Call GEX ---
            st.markdown("#### Call GEX")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🧲 Our Top Call GEX**")
                if not top_call.empty:
                    st_df(top_call)
                    if {"strike", "call_gex"}.issubset(top_call.columns):
                        st_plot(create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX"), key="gamma_map_top_call_gex")
                else:
                    st.info("No data.")
            with c2:
                st.markdown("**🔄 FlashAlpha Top Call GEX**")
                if fa_top_call is not None and not fa_top_call.empty:
                    st_df(fa_top_call)
                    if {"strike", "call_gex"}.issubset(fa_top_call.columns):
                        st_plot(create_top_strikes_chart(fa_top_call, "strike", "call_gex", "FA Top Call GEX"), key="fa_top_call_gex")
                else:
                    st.info("No data.")

            # --- Row 2: Put GEX ---
            st.markdown("#### Put GEX")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**🧲 Our Top Put GEX**")
                if not top_put.empty:
                    st_df(top_put)
                    if {"strike", "put_gex"}.issubset(top_put.columns):
                        st_plot(create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX"), key="gamma_map_top_put_gex")
                else:
                    st.info("No data.")
            with p2:
                st.markdown("**🔄 FlashAlpha Top Put GEX**")
                if fa_top_put is not None and not fa_top_put.empty:
                    st_df(fa_top_put)
                    if {"strike", "put_gex"}.issubset(fa_top_put.columns):
                        st_plot(create_top_strikes_chart(fa_top_put, "strike", "put_gex", "FA Top Put GEX"), key="fa_top_put_gex")
                else:
                    st.info("No data.")

            # --- Row 3: Net GEX ---
            st.markdown("#### Net GEX (abs)")
            n1, n2 = st.columns(2)
            with n1:
                st.markdown("**🧲 Our Top Net GEX (abs)**")
                if not top_net.empty:
                    top_net = top_net.copy()
                    if "net_gex_abs" not in top_net.columns and "net_gex" in top_net.columns:
                        top_net["net_gex_abs"] = top_net["net_gex"].abs()
                    if "net_gex_abs" in top_net.columns:
                        top_net = top_net.sort_values("net_gex_abs", ascending=False)
                    st_df(top_net)
                    if {"strike", "net_gex"}.issubset(top_net.columns):
                        st_plot(create_top_strikes_chart(top_net, "strike", "net_gex", "Top Net GEX"), key="gamma_map_top_net_gex")
                else:
                    st.info("No data.")
            with n2:
                st.markdown("**🔄 FlashAlpha Top Net GEX (abs)**")
                if fa_top_net is not None and not fa_top_net.empty:
                    fa_top_net = fa_top_net.copy()
                    if "net_gex_abs" not in fa_top_net.columns and "net_gex" in fa_top_net.columns:
                        fa_top_net["net_gex_abs"] = fa_top_net["net_gex"].abs()
                    if "net_gex_abs" in fa_top_net.columns:
                        fa_top_net = fa_top_net.sort_values("net_gex_abs", ascending=False)
                    st_df(fa_top_net)
                    if {"strike", "net_gex"}.issubset(fa_top_net.columns):
                        st_plot(create_top_strikes_chart(fa_top_net, "strike", "net_gex", "FA Top Net GEX"), key="fa_top_net_gex")
                else:
                    st.info("No data.")

            # FlashAlpha metadata
            if fa_data:
                st.markdown("---")
                st.caption("FlashAlpha metadata")
                fa_c1, fa_c2, fa_c3, fa_c4 = st.columns(4)
                fa_c1.metric("FA Underlying", f"{fa_data.get('underlying_price', 0):,.2f}")
                fa_c2.metric("FA Gamma Flip", f"{fa_data.get('gamma_flip', 0):,.2f}")
                fa_c3.metric("FA Net GEX", f"{fa_data.get('net_gex', 0):,.0f}")
                fa_c4.metric("FA As Of", fa_data.get("as_of", "N/A")[:19] if fa_data.get("as_of") else "N/A")
            elif not fa_result.get("success"):
                st.warning(f"FlashAlpha API unavailable: {fa_result.get('error', 'Unknown error')}")

    else:
        st.warning("No per-strike GEX returned from backend.")

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
