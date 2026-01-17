import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..helpers.calculations import build_vwap_obv_analysis
from ..helpers.ui_components import st_plot, st_df

def render_tab_vwap_obv(symbol):
    st.subheader("📈 VWAP + OBV Analysis")
    period = st.selectbox("Lookback", ["3mo", "6mo", "1y", "2y"], index=1, key="vwap_period")
    confirm_days = st.slider("Confirmation Days", 1, 5, 2, key="vwap_confirm")

    with st.spinner(f"Loading {symbol} history..."):
        try:
            raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
            hist = raw.reset_index()
        except Exception: hist = pd.DataFrame()

    if hist.empty:
        st.warning("No data.")
        return

    df, summary = build_vwap_obv_analysis(hist, confirm_days=confirm_days)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{summary['close']:.2f}")
    c2.metric("VWAP", f"{summary['vwap']:.2f}" if summary['vwap'] else "N/A")
    c3.metric("OBV", f"{summary['obv']:,.0f}")
    c4.metric("Signals", f"B: {summary['total_buys']} | S: {summary['total_sells']}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Price", line=dict(color="white")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], name="VWAP", line=dict(color="orange", dash="dash")), row=1, col=1)
    
    buys = df[df["BUY_MARK"]]
    fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Close"]*0.98, mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=12, color="#00d775")), row=1, col=1)
    
    sells = df[df["SELL_MARK"]]
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Close"]*1.02, mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=12, color="#ff4757")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df["Date"], y=df["OBV"], name="OBV", line=dict(color="cyan")), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
    st_plot(fig)
    
    with st.expander("Show signal details", expanded=False):
        st_df(df[["Date", "Close", "VWAP", "OBV", "BUY", "SELL"]].tail(20))
