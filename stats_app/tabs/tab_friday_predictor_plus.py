import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# If you already use st_plot/st_df helpers, import them
from stats_app.helpers.ui_components import st_plot, st_df


def render_tab_friday_predictor_plus(symbol: str, weekly_data: dict, hist_df: pd.DataFrame, spot: float):
    st_title = f"🧠 Friday Predictor+ (VWAP + ADL + Gamma Box) — {symbol}"
    import streamlit as st
    st.subheader(st_title)

    if hist_df is None or hist_df.empty:
        st.info("No price history yet. Click 'Fetch Data' first.")
        return

    # -----------------------------
    # VWAP + ADL (from hist_df)
    # -----------------------------
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(set(hist_df.columns)):
        st.error(f"Price history missing: {sorted(list(required - set(hist_df.columns)))}")
        return

    tp = (hist_df["High"] + hist_df["Low"] + hist_df["Close"]) / 3.0
    hist_df = hist_df.copy()
    hist_df["VWAP"] = (tp * hist_df["Volume"]).cumsum() / hist_df["Volume"].cumsum()
    current_vwap = float(hist_df["VWAP"].iloc[-1])

    hl_range = (hist_df["High"] - hist_df["Low"]).replace(0, np.nan)
    clv = ((hist_df["Close"] - hist_df["Low"]) - (hist_df["High"] - hist_df["Close"])) / hl_range
    adl = (clv.fillna(0) * hist_df["Volume"]).cumsum()

    # Trend stats (last 20 bars)
    n = min(20, len(hist_df))
    vwap_slope = float(np.polyfit(np.arange(n), hist_df["VWAP"].tail(n).values, 1)[0]) if n >= 5 else 0.0
    adl_slope = float(np.polyfit(np.arange(n), adl.tail(n).values, 1)[0]) if n >= 5 else float(adl.diff().iloc[-1])

    money_flow = "🟢 Accumulation" if adl_slope > 0 else "🔴 Distribution"

    # -----------------------------
    # Gamma Map Levels (from weekly_data)
    # -----------------------------
    top = (weekly_data or {}).get("top_strikes", {}) if isinstance(weekly_data, dict) else {}
    call_wall = top.get("call_gex", [{}])[0].get("strike")
    put_wall  = top.get("put_gex",  [{}])[0].get("strike")
    magnet    = top.get("net_gex_abs", [{}])[0].get("strike")

    if call_wall is not None and put_wall is not None:
        box_low, box_high = min(call_wall, put_wall), max(call_wall, put_wall)
    elif magnet is not None:
        box_low, box_high = float(magnet) * 0.98, float(magnet) * 1.02
    else:
        box_low, box_high = spot * 0.98, spot * 1.02

    # -----------------------------
    # Optional: Finnhub OI (best-effort)
    # -----------------------------
    def _get_finnhub_key():
        try:
            import streamlit as st
            k = st.secrets.get("FINNHUB_API_KEY", None)
            if k:
                return k
        except Exception:
            pass
        return os.getenv("FINNHUB_API_KEY")

    def _fetch_finnhub_oi(sym: str, key: str):
        """
        Best-effort parsing. If your Finnhub plan/schema doesn't return OI, this returns None.
        """
        url = "https://finnhub.io/api/v1/stock/option-chain"
        r = requests.get(url, params={"symbol": sym, "token": key}, timeout=20)
        r.raise_for_status()
        data = r.json()

        rows = []
        chain = data.get("data") or data.get("options") or []
        for exp_block in chain:
            contracts = exp_block.get("options") or exp_block.get("contracts") or []
            for c in contracts:
                strike = c.get("strike")
                opt_type = (c.get("type") or c.get("optionType") or "").lower()
                oi = c.get("openInterest") or c.get("oi")
                if strike is None or oi is None:
                    continue
                rows.append({"type": opt_type, "strike": float(strike), "oi": float(oi)})

        if not rows:
            return None

        odf = pd.DataFrame(rows)
        call_oi = float(odf[odf["type"].str.contains("c", na=False)]["oi"].sum())
        put_oi  = float(odf[odf["type"].str.contains("p", na=False)]["oi"].sum())
        pcr_oi  = (put_oi / call_oi) if call_oi > 0 else None

        top_calls = (
            odf[odf["type"].str.contains("c", na=False)]
            .groupby("strike")["oi"].sum().sort_values(ascending=False).head(5).reset_index()
        )
        top_puts = (
            odf[odf["type"].str.contains("p", na=False)]
            .groupby("strike")["oi"].sum().sort_values(ascending=False).head(5).reset_index()
        )

        return {"call_oi": call_oi, "put_oi": put_oi, "pcr_oi": pcr_oi, "top_calls": top_calls, "top_puts": top_puts}

    oi_pack = None
    with st.expander("📌 Finnhub Open Interest (optional)", expanded=False):
        key = _get_finnhub_key()
        if not key:
            st.info("Add FINNHUB_API_KEY to `.streamlit/secrets.toml` to enable this section.")
        else:
            try:
                oi_pack = _fetch_finnhub_oi(symbol, key)
                if not oi_pack:
                    st.warning("No OI returned (plan/schema may not include open interest).")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Call OI", f"{oi_pack['call_oi']:.0f}")
                    c2.metric("Put OI", f"{oi_pack['put_oi']:.0f}")
                    c3.metric("OI PCR", f"{oi_pack['pcr_oi']:.2f}" if oi_pack["pcr_oi"] is not None else "N/A")

                    cc, pp = st.columns(2)
                    with cc:
                        st.write("Top Call OI strikes")
                        st_df(oi_pack["top_calls"])
                    with pp:
                        st.write("Top Put OI strikes")
                        st_df(oi_pack["top_puts"])
            except Exception as e:
                st.error(f"Finnhub OI fetch failed: {e}")

    # -----------------------------
    # Friday projection (simple + transparent)
    # -----------------------------
    # Trend bias
    bullish = (spot > current_vwap) and (adl_slope > 0) and (vwap_slope >= 0)
    bearish = (spot < current_vwap) and (adl_slope < 0) and (vwap_slope <= 0)

    pin_zone = spot * 0.005
    near_vwap = abs(spot - current_vwap) <= pin_zone

    bias = "NEUTRAL"
    if bullish:
        bias = "BULLISH"
    elif bearish:
        bias = "BEARISH"

    # Choose target: wall when trend is clear, magnet when pinning zone
    target = None
    if near_vwap and magnet is not None:
        target = float(magnet)
        bias = "NEUTRAL (Magnet Pin)"
    else:
        if bias == "BULLISH" and call_wall is not None:
            target = float(call_wall)
        elif bias == "BEARISH" and put_wall is not None:
            target = float(put_wall)

    if target is None:
        target = float(magnet) if magnet is not None else float(current_vwap)

    # Confidence score
    conf = 0
    if call_wall is not None and put_wall is not None and magnet is not None:
        conf += 40
    conf += 25 if (bullish or bearish) else 10
    conf += 15 if near_vwap else 5

    if oi_pack and oi_pack.get("pcr_oi") is not None:
        pcr = oi_pack["pcr_oi"]
        # rough confirmation
        if bias.startswith("BEAR") and pcr > 1.05:
            conf += 20
        elif bias.startswith("BULL") and pcr < 0.95:
            conf += 20
        else:
            conf += 5

    conf = int(max(0, min(100, conf)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Friday Close", f"${target:.2f}", delta=bias)
    c2.metric("VWAP", f"${current_vwap:.2f}")
    c3.metric("Money Flow", money_flow)
    c4.metric("Confidence", f"{conf}%")

    # -----------------------------
    # Gamma Map chart (Walls / Magnet / Box)
    # -----------------------------
    st.write("### 🧲 Gamma Map (Magnets / Walls / Box)")
    fig = go.Figure()

    fig.add_vrect(x0=box_low, x1=box_high, opacity=0.15, line_width=0,
                  annotation_text="Gamma Box", annotation_position="top left")

    if put_wall is not None:
        fig.add_vline(x=float(put_wall), line_dash="dash",
                      annotation_text="Put Wall", annotation_position="top left")
    if call_wall is not None:
        fig.add_vline(x=float(call_wall), line_dash="dash",
                      annotation_text="Call Wall", annotation_position="top right")
    if magnet is not None:
        fig.add_vline(x=float(magnet), line_width=4,
                      annotation_text="Magnet", annotation_position="top")

    fig.add_trace(go.Scatter(
        x=[spot], y=[0], mode="markers+text",
        text=[f"Spot ${spot:.2f}"], textposition="top center",
        marker=dict(size=12), name="Spot"
    ))
    fig.add_trace(go.Scatter(
        x=[target], y=[0], mode="markers+text",
        text=[f"Fri ${target:.2f}"], textposition="bottom center",
        marker=dict(size=12), name="Forecast"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=320,
        xaxis_title="Strike / Price Level",
        yaxis_visible=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=35, b=10),
    )
    st_plot(fig)

    with st.expander("How this is computed", expanded=False):
        st.write(
            "- If price is **near VWAP**, we assume **pinning** → target leans to the **Magnet**.\n"
            "- If trend is clear (VWAP slope + ADL slope), target leans to **Call Wall** (bull) or **Put Wall** (bear).\n"
            "- Confidence increases when walls+magnet exist, trend aligns, and (optionally) OI PCR confirms."
        )
