import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ..helpers.calculations import approx_skew_25d, _build_spot_move_matrix, _fib_levels_from_swing
from ..helpers.ui_components import st_plot

def render_tab_vol_greeks(df, spot, symbol, date):
    st.subheader("🧮 Volatility & Greeks (from this expiry chain)")

    def plot_iv_and_greeks(df, spot, T, r, q):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from ..helpers.calculations import _to_float_series, _find_col, _bs_greeks
        
        d = df.copy()
        d["strike_num"] = _to_float_series(d["Strike"])
        d = d.dropna(subset=["strike_num"]).sort_values("strike_num")
        
        c_iv_col = _find_col(d, "call", "iv")
        p_iv_col = _find_col(d, "put", "iv")
        
        if c_iv_col: d["call_iv"] = _to_float_series(d[c_iv_col])
        if p_iv_col: d["put_iv"] = _to_float_series(d[p_iv_col])
        
        # Black-Scholes Greeks fallback if not in CSV
        f_cols = ["Call Delta", "Put Delta", "Gamma", "Vega", "Call Theta", "Put Theta"]
        missing = [c for c in f_cols if c not in d.columns]
        
        if missing and spot and T:
            res = d["strike_num"].apply(lambda k: _bs_greeks(spot, k, T, d.loc[d["strike_num"] == k, "call_iv"].iloc[0] if "call_iv" in d.columns else 0.25, r, q))
            for i, c in enumerate(f_cols):
                d[c] = res.apply(lambda x: x[i])
        
        fig_iv = go.Figure()
        if "call_iv" in d.columns: fig_iv.add_trace(go.Scatter(x=d["strike_num"], y=d["call_iv"], name="Call IV"))
        if "put_iv" in d.columns: fig_iv.add_trace(go.Scatter(x=d["strike_num"], y=d["put_iv"], name="Put IV"))
        fig_iv.update_layout(template="plotly_dark", height=400, title="IV Smile / Skew", xaxis_title="Strike", yaxis_title="Implied Vol")
        if spot: fig_iv.add_vline(x=spot, line_dash="dash", line_color="white")

        fig_g = make_subplots(rows=2, cols=2, subplot_titles=("Delta", "Gamma", "Vega", "Theta"))
        if "Call Delta" in d.columns:
            fig_g.add_trace(go.Scatter(x=d["strike_num"], y=d["Call Delta"], name="Call Delta"), row=1, col=1)
            fig_g.add_trace(go.Scatter(x=d["strike_num"], y=d["Put Delta"], name="Put Delta"), row=1, col=1)
        if "Gamma" in d.columns: fig_g.add_trace(go.Scatter(x=d["strike_num"], y=d["Gamma"], name="Gamma"), row=1, col=2)
        if "Vega" in d.columns: fig_g.add_trace(go.Scatter(x=d["strike_num"], y=d["Vega"], name="Vega"), row=2, col=1)
        if "Call Theta" in d.columns:
            fig_g.add_trace(go.Scatter(x=d["strike_num"], y=d["Call Theta"], name="Call Theta"), row=2, col=2)
            fig_g.add_trace(go.Scatter(x=d["strike_num"], y=d["Put Theta"], name="Put Theta"), row=2, col=2)
        
        fig_g.update_layout(template="plotly_dark", height=600, showlegend=False)
        
        atm_row = d.iloc[(d["strike_num"] - (spot or 0)).abs().argsort()[:1]].to_dict('records')
        atm = atm_row[0] if atm_row else None
        if atm: atm["atm_strike"] = atm["strike_num"]
        
        return fig_iv, fig_g, atm

    if df.empty:
        st.info("No options data loaded yet.")
        return

    with st.expander('Greek Inputs (Black-Scholes fallback)', expanded=False):
        r_in = st.number_input('Risk-free rate r (annual, decimal)', value=0.041, step=0.001, format='%.4f', key="r_in")
        q_in = st.number_input('Dividend yield q (annual, decimal)', value=0.004, step=0.001, format='%.4f', key="q_in")
        spot_override = st.number_input('Spot override (0 = auto)', value=0.0, step=0.1, format='%.2f', key="tab5_spot")
        use_trading_days = st.checkbox('Use trading-day year (252) for T', value=False, key="tab5_tday")

    spot_for_greeks = float(spot_override) if spot_override > 0 else float(spot)
    
    _now_ts = pd.Timestamp.now()
    _exp_ts = pd.Timestamp(date) + pd.Timedelta(hours=16)
    if use_trading_days:
        T = max(int((_exp_ts.normalize() - _now_ts.normalize()).days) / 252.0, 1e-6)
    else:
        T = max(float((_exp_ts - _now_ts).total_seconds()) / (365.0 * 24 * 3600), 1e-6)

    fig_iv, fig_greeks, atm = plot_iv_and_greeks(df, spot=spot_for_greeks, T=T, r=float(r_in), q=float(q_in))

    if not atm:
        st.warning("Could not compute ATM snapshot.")
    else:
        st_plot(fig_iv)
        st_plot(fig_greeks)
        
        st.markdown(f"**ATM strike:** `{atm['atm_strike']:g}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Call IV", f"{atm.get('call_iv', 0):.4f}")
        m2.metric("Put IV", f"{atm.get('put_iv', 0):.4f}")
        m3.metric("Call Delta", f"{atm.get('Call Delta', 0):.3f}")
        m4.metric("Put Delta", f"{atm.get('Put Delta', 0):.3f}")

        st.markdown("### 📊 Spot Move Matrix (Delta + Gamma)")
        if "Gamma" in atm:
            df_matrix = _build_spot_move_matrix(float(spot_for_greeks), float(atm.get("Call Delta", 0)), float(atm.get("Put Delta", 0)), float(atm.get("Gamma", 0)))
            st.dataframe(df_matrix, width="stretch", height=300)

    st.markdown("---")
    st.subheader("🧲 Skew Analysis (25-Delta)")
    skew = approx_skew_25d(df, spot=spot_for_greeks, T=T)
    if skew:
        c1, c2, c3 = st.columns(3)
        c1.metric("25d Call IV", f"{skew['call_25d_iv']:.2%}")
        c2.metric("25d Put IV", f"{skew['put_25d_iv']:.2%}")
        c3.metric("Skew (C-P)", f"{skew['skew_call_minus_put']:.4f}")
    else:
        st.info("Skew data unavailable.")

def get_today_open_from_yahoo(symbol: str) -> float | None:
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        d = t.history(period="1d")
        if not d.empty: return float(d["Open"].iloc[0])
    except Exception: pass
    return None
