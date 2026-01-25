import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_tab_orderflow_delta(symbol: str, hist_df: pd.DataFrame, spot: float):
    st.subheader("📊 Order Flow & Volume Delta (Proxy)")
    st.caption("⚠️ This uses OHLCV candle data, not real bid/ask. It is a PROXY for buying/selling pressure.")

    if hist_df is None or hist_df.empty:
        st.warning("No price history data available.")
        return

    df = hist_df.copy()

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close", "Close")
    open_col = cols.get("open", "Open")
    vol_col = cols.get("volume", "Volume")

    df = df[[open_col, close_col, vol_col]].dropna()
    df.columns = ["Open", "Close", "Volume"]

    # -----------------------------
    # Up / Down Volume Proxy
    # -----------------------------
    df["UpVol"] = np.where(df["Close"] > df["Open"], df["Volume"], 0)
    df["DownVol"] = np.where(df["Close"] < df["Open"], df["Volume"], 0)

    up_vol = df["UpVol"].tail(20).sum()
    down_vol = df["DownVol"].tail(20).sum()
    delta_proxy = up_vol - down_vol

    # -----------------------------
    # OBV
    # -----------------------------
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])

    df["OBV"] = obv

    obv_slope = df["OBV"].tail(10).iloc[-1] - df["OBV"].tail(10).iloc[0]

    # -----------------------------
    # Absorption Proxy
    # -----------------------------
    df["Body"] = (df["Close"] - df["Open"]).abs()
    avg_vol = df["Volume"].rolling(20).mean()
    small_body = df["Body"] < df["Body"].rolling(20).mean()
    high_vol = df["Volume"] > avg_vol

    df["Absorption"] = small_body & high_vol
    absorption_recent = df["Absorption"].tail(10).any()

    # -----------------------------
    # Display Metrics
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Up Volume (20 bars)", f"{up_vol:,.0f}")
    c2.metric("Down Volume (20 bars)", f"{down_vol:,.0f}")
    c3.metric("Volume Delta (Proxy)", f"{delta_proxy:,.0f}")
    c4.metric("OBV Slope (10 bars)", f"{obv_slope:,.0f}")

    if absorption_recent:
        st.warning("⚠️ Absorption detected recently: High volume but little price movement.")

    # -----------------------------
    # Interpretation
    # -----------------------------
    if delta_proxy > 0 and obv_slope > 0:
        st.success("🟢 Buying pressure dominant (proxy). Buyers appear more aggressive.")
    elif delta_proxy < 0 and obv_slope < 0:
        st.error("🔴 Selling pressure dominant (proxy). Sellers appear more aggressive.")
    else:
        st.info("🟡 Mixed / balanced order flow (proxy). Market likely in chop or rotation.")

    # -----------------------------
    # Plot OBV
    # -----------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV"))
    fig.update_layout(
        title=f"{symbol} — OBV (Cumulative Volume Flow Proxy)",
        template="plotly_dark",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Education Table
    # -----------------------------
    st.markdown("## 🧠 How to Read This")

    st.markdown(
        """
| What you see | What it means |
|--------------|---------------|
| High volume + price up + positive delta | Real buying |
| High volume + price down + negative delta | Real selling |
| High volume + flat price | Absorption |
| Price up + delta down | Short covering or passive buying |
| Price down + delta up | Long liquidation or passive selling |
        """
    )

    st.caption("⚠️ Remember: This is based on candle proxies. True delta requires bid/ask tick data.")
