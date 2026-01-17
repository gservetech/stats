import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from ..helpers.calculations import build_vol_cone_analysis
from ..helpers.ui_components import st_plot, st_df

def render_tab_vol_cone(symbol):
    st.subheader("🎯 Volatility Cone Analysis")
    
    with st.expander("Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: period = st.selectbox("History", ["1y", "2y", "5y", "max"], index=1, key="cone_period")
        with c2: focus_win = st.number_input("Focus (days)", 5, 252, 20, key="cone_win")
        with c3: show_table = st.checkbox("Show Table", value=True)

    with st.spinner(f"Loading {symbol} data..."):
        try:
            raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
            hist = raw.reset_index()
        except Exception: hist = pd.DataFrame()

    if hist.empty:
        st.warning("No data.")
        return

    try:
        cone, summary, df = build_vol_cone_analysis(hist, focus_window=focus_win)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Regime", summary["vol_regime"])
        m2.metric("Direction", summary["direction"])
        m3.metric("Suggestion", summary["suggestion"])

        fig = go.Figure()
        pcts = [5, 25, 50, 75, 95]
        for p in pcts:
            fig.add_trace(go.Scatter(x=cone.index, y=cone[f"p{p}"], name=f"p{p}", line=dict(dash="dash" if p in [5, 95] else "solid", width=1)))
        fig.add_trace(go.Scatter(x=cone.index, y=cone["latest_rv"], name="Latest RV", line=dict(color="white", width=3)))
        
        fig.update_layout(template="plotly_dark", height=500, title=f"Volatility Cone for {symbol}", xaxis_title="Days to Expiry / Window", yaxis_title="Annualized Realized Vol")
        st_plot(fig)
        
        if show_table:
            st_df(cone.reset_index())

    except Exception as e:
        st.error(f"Error: {e}")
