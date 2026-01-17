import streamlit as st
import pandas as pd
from ..helpers.api_client import fetch_weekly_gex
from ..helpers.calculations import build_gamma_levels
from ..helpers.ui_components import st_plot
from ..helpers.filters import plot_filters, kalman_message
from ..helpers.data_fetching import fetch_price_history

def render_tab_gamma_map_filters(symbol, date, spot):
    st.subheader("⌛ Gamma Map (Magnets / Walls / Box)")
    
    # Imports for plotting inside tab
    import plotly.graph_objects as go
    
    def plot_net_gex_map(df, spot, levels):
        df = df.copy()
        df["net_gex"] = pd.to_numeric(df["net_gex"], errors="coerce").fillna(0.0)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce").fillna(0.0)
        df = df.sort_values("strike")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["strike"], y=df["net_gex"], name="Net GEX", marker_color=df["net_gex"].apply(lambda x: "#00d775" if x >= 0 else "#ff4757")))
        
        if spot:
            fig.add_vline(x=spot, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot:g}")
            
        if levels:
            if levels.get("major_call_wall"):
                fig.add_vline(x=levels["major_call_wall"], line_color="#00d775", line_width=2, annotation_text="Call Wall")
            if levels.get("major_put_wall"):
                fig.add_vline(x=levels["major_put_wall"], line_color="#ff4757", line_width=2, annotation_text="Put Wall")
            if levels.get("zero_gamma"):
                fig.add_vline(x=levels["zero_gamma"], line_dash="dot", line_color="orange", annotation_text="Zero Gamma")

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
            levels = build_gamma_levels(gex_df, spot=spot, top_n=5)
            if not levels:
                st.warning("Could not compute gamma levels.")
            else:
                cA, cB, cC, cD = st.columns(4)
                mag = float(levels['magnets'].iloc[0]['strike']) if not levels["magnets"].empty else None
                lower = levels["gamma_box"]["lower"]
                upper = levels["gamma_box"]["upper"]
                zg = levels.get("zero_gamma")

                cA.metric("Main Magnet", f"{mag:g}" if mag is not None else "N/A")
                cB.metric("Put Wall (Lower)", f"{lower:g}" if lower is not None else "N/A")
                cC.metric("Call Wall (Upper)", f"{upper:g}" if upper is not None else "N/A")
                cD.metric("Zero Gamma", f"{zg:g}" if zg is not None else "N/A")

                st_plot(plot_net_gex_map(gex_df, spot=spot, levels=levels))

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
