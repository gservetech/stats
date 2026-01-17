def render_friday_predictor(symbol, expiry_date, weekly_data, hist_df, spot):
    """
    Friday Expiry Forecast (for the selected expiry_date)

    This version matches your standalone script logic:
    - Fetches per-strike weekly GEX table from /weekly/gex
    - Builds Gamma Map: put wall, call wall, magnet, box
    - Computes VWAP + ADL from recent daily candles (hist_df)
    - Predicts Friday close using blended gamma pin + trend score
    """
    import streamlit as st
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    from stats_app.helpers.api_client import API_BASE_URL
    from stats_app.helpers.ui_components import st_plot

    st.subheader(f"🔮 Friday Expiry Forecast: {symbol} ({expiry_date})")

    # ----------------------------
    # 0) Validate history
    # ----------------------------
    if hist_df is None or hist_df.empty:
        st.warning("No price history loaded. Click 'Fetch Data' first.")
        return

    need_cols = {"High", "Low", "Close", "Volume"}
    missing = need_cols - set(hist_df.columns)
    if missing:
        st.error(f"Price history missing columns: {sorted(list(missing))}")
        return

    # Ensure datetime index (recommended)
    if not hasattr(hist_df.index, "max"):
        st.error("hist_df index is not a DatetimeIndex.")
        return

    # ----------------------------
    # 1) Fetch per-strike weekly GEX table (same as your script)
    # ----------------------------
    try:
        r = requests.get(
            f"{API_BASE_URL}/weekly/gex",
            params={"symbol": symbol, "expiry": expiry_date},
            timeout=60,
        )
        r.raise_for_status()
        gex_df = pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"Failed to fetch /weekly/gex: {e}")
        return

    required_gex_cols = {"strike", "call_gex", "put_gex", "net_gex"}
    if not required_gex_cols.issubset(set(gex_df.columns)):
        st.error(f"/weekly/gex missing required columns: {sorted(list(required_gex_cols - set(gex_df.columns)))}")
        st.write("Columns received:", list(gex_df.columns))
        return

    # Make sure numeric
    for c in ["strike", "call_gex", "put_gex", "net_gex"]:
        gex_df[c] = pd.to_numeric(gex_df[c], errors="coerce")
    gex_df = gex_df.dropna(subset=["strike"]).sort_values("strike")

    # ----------------------------
    # 2) Gamma Map (your script logic)
    # ----------------------------
    def analyze_gamma(df, spot_price: float):
        d = df.copy()
        d["abs_net"] = d["net_gex"].abs()

        # Walls (global maxima)
        put_wall_row = d.loc[d["put_gex"].idxmax()]
        call_wall_row = d.loc[d["call_gex"].idxmax()]

        # Magnet: big net gamma + close to spot
        d["dist"] = (d["strike"] - spot_price).abs()
        d["pin_score"] = d["abs_net"] / (d["dist"] + 1.0)
        magnet_row = d.loc[d["pin_score"].idxmax()]

        low = float(min(put_wall_row["strike"], call_wall_row["strike"]))
        high = float(max(put_wall_row["strike"], call_wall_row["strike"]))

        return {
            "put_wall": put_wall_row,
            "call_wall": call_wall_row,
            "magnet": magnet_row,
            "box_low": low,
            "box_high": high,
        }

    gamma = analyze_gamma(gex_df, float(spot))
    put_wall = float(gamma["put_wall"]["strike"])
    call_wall = float(gamma["call_wall"]["strike"])
    magnet = float(gamma["magnet"]["strike"])
    box_low = float(gamma["box_low"])
    box_high = float(gamma["box_high"])

    # ----------------------------
    # 3) Compute VWAP + ADL (recent window)
    #    (same math as your script; using hist_df)
    # ----------------------------
    cutoff = hist_df.index.max() - pd.Timedelta(days=140)  # ~90 trading days
    recent = hist_df.loc[hist_df.index >= cutoff].copy()
    if recent.empty:
        recent = hist_df.tail(120).copy()

    # VWAP series (cumulative)
    tp = (recent["High"] + recent["Low"] + recent["Close"]) / 3.0
    vwap = (tp * recent["Volume"]).cumsum() / recent["Volume"].cumsum()
    vwap = vwap.replace([np.inf, -np.inf], np.nan).dropna()

    # ADL series (cumulative)
    hl = (recent["High"] - recent["Low"]).replace(0, np.nan)
    mfm = ((recent["Close"] - recent["Low"]) - (recent["High"] - recent["Close"])) / hl
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    adl = (mfm * recent["Volume"]).cumsum()

    # Money flow label
    adl_delta = float(adl.diff().tail(10).mean()) if len(adl) > 10 else float(adl.diff().iloc[-1] if len(adl) > 1 else 0.0)
    adl_trend = "🟢 Accumulation" if adl_delta > 0 else "🔴 Distribution"

    # ----------------------------
    # 4) Predict Friday close (your script blend)
    # ----------------------------
    def predict_friday_close(spot_price: float, magnet_row, vwap_series, adl_series):
        if vwap_series is None or len(vwap_series) < 5 or adl_series is None or len(adl_series) < 5:
            return float(magnet_row["strike"]), 0.0, 0.0

        # trend slopes
        vwap_slope = float(np.polyfit(np.arange(len(vwap_series)), vwap_series.values, 1)[0])
        adl_slope = float(np.polyfit(np.arange(len(adl_series)), adl_series.values, 1)[0])

        trend_score = float(np.tanh(vwap_slope * 1000.0 + adl_slope / 1e7))

        # gamma pin strength
        net = float(magnet_row["net_gex"])
        gamma_strength = float(np.tanh(abs(net) / 1e9))

        target = float(magnet_row["strike"])
        blended = float(
            spot_price
            + gamma_strength * (target - spot_price)
            + trend_score * 0.5 * (spot_price - float(vwap_series.iloc[-1]))
        )

        return blended, trend_score, gamma_strength

    pred, trend_score, gamma_strength = predict_friday_close(float(spot), gamma["magnet"], vwap, adl)

    # Optional clamp to the gamma box (prevents wild outputs)
    if box_low <= box_high:
        pred_clamped = float(np.clip(pred, box_low, box_high))
    else:
        pred_clamped = pred

    # Confidence (simple, explainable)
    confidence = int(np.clip((abs(gamma_strength) * 60 + abs(trend_score) * 40) * 100, 0, 100))

    # ----------------------------
    # 5) UI output
    # ----------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Friday Close", f"${pred_clamped:.2f}")
    c2.metric("Gamma Magnet", f"{magnet:.0f}")
    c3.metric("Market VWAP", f"${float(vwap.iloc[-1]) if len(vwap) else float(recent['Close'].iloc[-1]):.2f}")
    c4.metric("Confidence", f"{confidence}%")

    st.caption(
        f"Put Wall: {put_wall:.0f} | Call Wall: {call_wall:.0f} | Box: {box_low:.0f}→{box_high:.0f} | "
        f"TrendScore: {trend_score:.3f} | GammaStrength: {gamma_strength:.3f} | "
        f"History: {recent.index.min().date()} → {recent.index.max().date()}"
    )

    # ----------------------------
    # 6) Gamma Map plot
    # ----------------------------
    st.write("### 🧲 Gamma Map (Magnets / Walls / Box)")

    fig = go.Figure()

    # Box shading (as two vertical bounds)
    fig.add_vline(x=box_low, line_dash="dot", annotation_text="Box Low", annotation_position="top left")
    fig.add_vline(x=box_high, line_dash="dot", annotation_text="Box High", annotation_position="top right")

    fig.add_vline(x=put_wall, line_dash="dash", annotation_text="Put Wall")
    fig.add_vline(x=call_wall, line_dash="dash", annotation_text="Call Wall")
    fig.add_vline(x=magnet, line_width=4, annotation_text="Magnet")

    fig.add_trace(go.Scatter(
        x=[float(spot)], y=[0],
        mode="markers+text",
        text=[f"Spot {float(spot):.2f}"],
        textposition="top center",
        marker=dict(size=12),
        name="Spot"
    ))

    fig.add_trace(go.Scatter(
        x=[pred_clamped], y=[0],
        mode="markers+text",
        text=[f"Pred {pred_clamped:.2f}"],
        textposition="bottom center",
        marker=dict(size=12),
        name="Pred"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=320,
        xaxis_title="Strike",
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=35, b=10),
    )
    st_plot(fig)
