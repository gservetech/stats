import streamlit as st
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ..helpers.calculations import build_market_folding
from ..helpers.ui_components import st_plot

def render_tab_market_folding(symbol):
    st.subheader("Market Folding (Eigen-Entropy)")
    st.caption("Analyzes market structure using eigen-entropy of technical indicators.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        period_idx = st.selectbox("History Period", ["6mo", "1y", "2y", "5y"], index=2, key="fold_period")
    with c2:
        interval_idx = st.selectbox("Interval", ["1d", "1wk"], index=0, key="fold_interval")
    with c3:
        window_size = st.number_input("Correlation window", min_value=10, max_value=200, value=50, step=1, key="fold_window")
    with c4:
        smooth = st.number_input("Smoothing", min_value=1, max_value=50, value=5, step=1, key="fold_smooth")

    c5, c6 = st.columns(2)
    with c5:
        entropy_base = st.number_input("Entropy base", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="fold_entropy")
    with c6:
        low_q, high_q = st.slider("Regime quantiles", min_value=0.05, max_value=0.95, value=(0.25, 0.75), step=0.05, key="fold_q")

    if high_q <= low_q:
        st.warning("High quantile must be above low quantile.")
        return

    with st.spinner(f"Loading {symbol} data..."):
        try: hist_yf = yf.download(symbol, period=period_idx, interval=interval_idx, auto_adjust=False, progress=False)
        except Exception: hist_yf = pd.DataFrame()

    if hist_yf.empty:
        st.warning("No data returned.")
        return

    if isinstance(hist_yf.columns, pd.MultiIndex): hist_yf.columns = hist_yf.columns.droplevel(1)
    hist_yf = hist_yf.reset_index()
    if "Date" not in hist_yf.columns: hist_yf["Date"] = hist_yf["Datetime"] if "Datetime" in hist_yf.columns else hist_yf.index

    fold_df, loq, hiq = build_market_folding(hist_yf, int(window_size), int(smooth), float(entropy_base), float(low_q), float(high_q))
    plot_df = fold_df.dropna(subset=["Folding_Score"]).sort_values("Date")
    if plot_df.empty:
        st.warning("Not enough data.")
        return

    last = plot_df.iloc[-1]
    st.columns(3)[0].metric("Latest Close", f"{float(last['Close']):.2f}")
    st.columns(3)[1].metric("Folding Score", f"{float(last['Folding_Score']):.4f}")
    st.columns(3)[2].metric("Regime", str(last["Regime"]))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    colors = {"COLLAPSED": "rgba(255, 77, 77, 0.18)", "TRANSITION": "rgba(200, 200, 200, 0.14)", "COMPLEX_FOLDED": "rgba(102, 204, 102, 0.18)"}
    
    # Shading logic simplified
    last_reg, start_dt = None, None
    for _, row in plot_df.iterrows():
        if row["Regime"] != last_reg:
            if last_reg:
                for r in [1, 2]: fig.add_vrect(x0=start_dt, x1=row["Date"], fillcolor=colors.get(last_reg), opacity=0.35, row=r, col=1)
            start_dt, last_reg = row["Date"], row["Regime"]
    
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Close"], name="Close", line=dict(color="white", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Folding_Score"], name="Folding Score", line=dict(color="cyan")), row=2, col=1)
    if loq: fig.add_hline(y=loq, line_dash="dot", line_color="red", row=2, col=1)
    if hiq: fig.add_hline(y=hiq, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=700)
    st_plot(fig)
