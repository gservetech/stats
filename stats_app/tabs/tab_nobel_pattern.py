import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    from stats_app.helpers.data_fetching import fetch_cnbc_chart_data
except Exception:
    fetch_cnbc_chart_data = None


# -----------------------------
# Data normalize helpers
# -----------------------------
def _to_ohlc_df_from_cnbc(data: dict) -> pd.DataFrame:
    bars = (data or {}).get("priceBars") or []
    df = pd.DataFrame(bars)
    if df.empty:
        return df

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    date_col = None
    for col in ["tradeTimeinMills", "tradeTime", "dateTime"]:
        if col in df.columns:
            date_col = col
            break

    if date_col == "tradeTimeinMills":
        df["date"] = pd.to_datetime(
            pd.to_numeric(df["tradeTimeinMills"], errors="coerce"),
            unit="ms",
            errors="coerce",
        )
    elif date_col == "tradeTime":
        df["date"] = pd.to_datetime(df["tradeTime"], format="%Y%m%d%H%M%S", errors="coerce")
    elif date_col == "dateTime":
        df["date"] = pd.to_datetime(df["dateTime"], errors="coerce")
    else:
        return pd.DataFrame()

    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    return df


def _to_ohlc_df_from_hist_df(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()

    df = hist_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            df["date"] = pd.to_datetime(df.index, errors="coerce")
    else:
        df["date"] = df.index

    colmap = {c.lower(): c for c in df.columns}

    def pick(name: str):
        return colmap.get(name)

    o = pick("open")
    h = pick("high")
    l = pick("low")
    c = pick("close")

    if c is None:
        if "Close" in df.columns:
            c = "Close"
        else:
            return pd.DataFrame()

    if o is None:
        df["open"] = df[c].shift(1)
    else:
        df["open"] = pd.to_numeric(df[o], errors="coerce")

    if h is None:
        df["high"] = pd.to_numeric(df[c], errors="coerce")
    else:
        df["high"] = pd.to_numeric(df[h], errors="coerce")

    if l is None:
        df["low"] = pd.to_numeric(df[c], errors="coerce")
    else:
        df["low"] = pd.to_numeric(df[l], errors="coerce")

    df["close"] = pd.to_numeric(df[c], errors="coerce")

    out = df[["date", "open", "high", "low", "close"]].dropna().sort_values("date")
    return out


# -----------------------------
# Indicator logic helpers
# -----------------------------
def _candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["body"] = (d["close"] - d["open"]).abs()
    d["rng"] = (d["high"] - d["low"]).replace(0, np.nan)
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["bull"] = d["close"] >= d["open"]
    d["bear"] = ~d["bull"]

    d["doji"] = (d["body"] / d["rng"]) <= 0.15
    d["hammer"] = (d["lower_wick"] >= 2.0 * d["body"]) & (d["upper_wick"] <= 0.5 * d["body"]) & d["bull"]
    d["shooting_star"] = (d["upper_wick"] >= 2.0 * d["body"]) & (d["lower_wick"] <= 0.5 * d["body"]) & d["bear"]

    prev_open = d["open"].shift(1)
    prev_close = d["close"].shift(1)
    prev_bull = prev_close >= prev_open
    prev_bear = ~prev_bull

    d["bull_engulf"] = prev_bear & d["bull"] & (d["open"] <= prev_close) & (d["close"] >= prev_open)
    d["bear_engulf"] = prev_bull & d["bear"] & (d["open"] >= prev_close) & (d["close"] <= prev_open)

    # ATR
    tr = pd.concat(
        [
            (d["high"] - d["low"]).abs(),
            (d["high"] - d["close"].shift(1)).abs(),
            (d["low"] - d["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    return d


def _pivots(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    d = df.copy()
    w = int(max(window, 3))
    d["pivot_high"] = d["high"] == d["high"].rolling(w, center=True).max()
    d["pivot_low"] = d["low"] == d["low"].rolling(w, center=True).min()
    return d


def _last_swing(df: pd.DataFrame):
    piv_high = df[df.get("pivot_high", False)]
    piv_low = df[df.get("pivot_low", False)]
    if piv_high.empty or piv_low.empty:
        return None, None

    last_low = piv_low.iloc[-1]
    highs_after_low = piv_high[piv_high["date"] > last_low["date"]]
    last_high = highs_after_low.iloc[-1] if not highs_after_low.empty else piv_high.iloc[-1]
    return (last_low["date"], float(last_low["low"])), (last_high["date"], float(last_high["high"]))


def _fib_levels(low_price: float, high_price: float) -> dict:
    rng = high_price - low_price
    if rng <= 0:
        return {}
    return {
        "0.382": high_price - 0.382 * rng,
        "0.500": high_price - 0.500 * rng,
        "0.618": high_price - 0.618 * rng,
        "0.650": high_price - 0.650 * rng,
    }


def _add_right_price_tag(fig: go.Figure, y: float, text: str):
    fig.add_annotation(
        x=1.005,
        xref="paper",
        y=y,
        yref="y",
        text=text,
        showarrow=False,
        align="left",
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.70)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        borderpad=4,
    )


def _sr_step_lines(df: pd.DataFrame, window: int = 25):
    w = int(max(10, window))
    sup = df["low"].rolling(w).min()
    res = df["high"].rolling(w).max()
    return sup, res


def _signal_engine(df: pd.DataFrame, fib: dict, pocket_tol_atr: float = 0.35):
    """
    Produces BUY/SELL events with:
      - conf (%) : heuristic strength
      - signal_price : EXACT price to buy/sell (we use the candle CLOSE as the signal price)

    IMPORTANT:
      - This is deterministic from your data; not a broker execution price.
      - If you want "next candle open" as entry, change signal_price assignment below.
    """
    d = df.copy()
    d["signal"] = None
    d["conf"] = np.nan
    d["signal_price"] = np.nan  # <- exact "system price"

    atr = d["atr"].copy().replace(0, np.nan)
    atr = atr.fillna(atr.median()).fillna(1e-6)

    sup, res = _sr_step_lines(d, window=25)
    d["support"] = sup
    d["resistance"] = res

    has_fib = bool(fib)
    if has_fib:
        gp_a = float(fib["0.618"])
        gp_b = float(fib["0.650"])
        gp_low = min(gp_a, gp_b)
        gp_high = max(gp_a, gp_b)
    else:
        gp_low = gp_high = None

    ma_fast = d["close"].rolling(10).mean()
    ma_slow = d["close"].rolling(30).mean()

    for i in range(2, len(d)):
        c = float(d.loc[i, "close"])
        a = float(atr.loc[i])
        tol = pocket_tol_atr * a

        near_support = np.isfinite(d.loc[i, "support"]) and abs(c - float(d.loc[i, "support"])) <= (0.6 * a)
        near_resist = np.isfinite(d.loc[i, "resistance"]) and abs(c - float(d.loc[i, "resistance"])) <= (0.6 * a)

        in_pocket = False
        near_pocket = False
        if has_fib:
            in_pocket = (c >= gp_low) and (c <= gp_high)
            near_pocket = (c >= gp_low - tol) and (c <= gp_high + tol)

        bull_pattern = bool(d.loc[i, "hammer"] or d.loc[i, "bull_engulf"])
        bear_pattern = bool(d.loc[i, "shooting_star"] or d.loc[i, "bear_engulf"])

        score = 0.0
        if bull_pattern:
            score += 0.45
        if bear_pattern:
            score -= 0.45

        if in_pocket:
            score += 0.20
        elif near_pocket:
            score += 0.10

        if near_support:
            score += 0.20
        if near_resist:
            score -= 0.20

        trend_up = None
        if np.isfinite(ma_fast.loc[i]) and np.isfinite(ma_slow.loc[i]):
            trend_up = bool(ma_fast.loc[i] >= ma_slow.loc[i])

        if trend_up is True and score > 0:
            score += 0.05
        if trend_up is False and score < 0:
            score -= 0.05

        # Decide signal
        if score >= 0.55:
            d.loc[i, "signal"] = "BUY"
        elif score <= -0.55:
            d.loc[i, "signal"] = "SELL"

        if d.loc[i, "signal"] is not None:
            conf = min(0.98, max(0.55, abs(score)))  # clamp
            d.loc[i, "conf"] = 100.0 * conf

            # ✅ EXACT "system price" = candle close
            d.loc[i, "signal_price"] = float(d.loc[i, "close"])

            # If you want NEXT candle open as the "system price", use:
            # if i + 1 < len(d): d.loc[i, "signal_price"] = float(d.loc[i + 1, "open"])

    return d


# -----------------------------
# Main renderer (THIS is what app.py imports)
# -----------------------------
def render_tab_nobel_pattern(symbol: str, spot: float, hist_df: pd.DataFrame):
    st.subheader("🏅 Nobel Signals (MT5-style overlay)")
    st.caption("Support/Resistance steps + Golden Pocket + BUY/SELL arrows (green/red) with confidence and exact signal price.")

    # Controls
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        source = st.selectbox("Price source", ["CNBC (intraday if available)", "History (hist_df)"], index=0)
    with c2:
        tf = st.selectbox("Timeframe", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "ALL"], index=1)
    with c3:
        pivot_window = st.slider("Swing window", 3, 21, 7, step=2)
    with c4:
        lookback = st.slider("Bars to display", 150, 1500, 650, step=50)

    o1, o2, o3, o4, o5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])
    with o1:
        show_pivots = st.checkbox("Show Pivots", value=False)
    with o2:
        show_patterns = st.checkbox("Show Patterns", value=False)
    with o3:
        show_signals = st.checkbox("Show BUY/SELL Arrows", value=True)
    with o4:
        remove_weekends = st.checkbox("Remove Weekends", value=True)
    with o5:
        big_chart = st.checkbox("Bigger Chart", value=True)

    # Load OHLC
    if source.startswith("CNBC") and fetch_cnbc_chart_data:
        with st.spinner(f"Fetching {symbol} ({tf}) from CNBC..."):
            data = fetch_cnbc_chart_data(symbol, tf)
        df = _to_ohlc_df_from_cnbc(data)
    else:
        df = _to_ohlc_df_from_hist_df(hist_df)

    if df is None or df.empty:
        st.warning("No OHLC data available for this tab.")
        return

    df = df.tail(int(lookback)).reset_index(drop=True)
    df = _candlestick_features(df)
    df = _pivots(df, window=int(pivot_window))

    last_close = float(df["close"].iloc[-1])
    last_open = float(df["open"].iloc[-1])
    direction = "BULLISH" if last_close >= last_open else "BEARISH"

    swing_low, swing_high = _last_swing(df)
    fib = {}
    if swing_low and swing_high:
        fib = _fib_levels(swing_low[1], swing_high[1])

    df_sig = _signal_engine(df, fib)

    # Left info box
    sup_now = float(df_sig["support"].iloc[-1]) if "support" in df_sig and df_sig["support"].notna().any() else np.nan
    res_now = float(df_sig["resistance"].iloc[-1]) if "resistance" in df_sig and df_sig["resistance"].notna().any() else np.nan

    last_signal_row = df_sig[df_sig["signal"].notna()].tail(1)
    if not last_signal_row.empty:
        last_sig = str(last_signal_row["signal"].iloc[0])
        last_conf = float(last_signal_row["conf"].iloc[0])
        last_sig_price = float(last_signal_row["signal_price"].iloc[0])
    else:
        last_sig = "NONE"
        last_conf = 0.0
        last_sig_price = np.nan

    left, right = st.columns([1.1, 3.5], gap="large")
    with left:
        st.markdown("### 🏅 NOBEL SIGNALS")
        st.write(f"**Trend:** {direction}")

        if last_sig != "NONE" and np.isfinite(last_sig_price):
            if last_sig_price < 10:
                st.write(f"**Last Signal:** {last_sig} ({last_conf:.0f}%) @ **{last_sig_price:.5f}**")
            else:
                st.write(f"**Last Signal:** {last_sig} ({last_conf:.0f}%) @ **{last_sig_price:.2f}**")
        else:
            st.write("**Last Signal:** NONE")

        if last_close < 10:
            st.write(f"**Price:** {last_close:,.5f}")
        else:
            st.write(f"**Price:** {last_close:,.2f}")

        if np.isfinite(sup_now) and np.isfinite(res_now):
            if sup_now < 10:
                st.write(f"**Support:** {sup_now:,.5f}")
                st.write(f"**Resistance:** {res_now:,.5f}")
            else:
                st.write(f"**Support:** {sup_now:,.2f}")
                st.write(f"**Resistance:** {res_now:,.2f}")

        if fib:
            gp_a = float(fib["0.618"])
            gp_b = float(fib["0.650"])
            gp_low = min(gp_a, gp_b)
            gp_high = max(gp_a, gp_b)
            if gp_low < 10:
                st.write(f"**Golden Pocket:** {gp_low:.5f} – {gp_high:.5f}")
            else:
                st.write(f"**Golden Pocket:** {gp_low:.2f} – {gp_high:.2f}")

        st.caption("Signal price shown is the candle CLOSE where the signal triggers (deterministic from your data).")

    with right:
        fig = go.Figure()

        # Candles
        fig.add_trace(
            go.Candlestick(
                x=df_sig["date"],
                open=df_sig["open"],
                high=df_sig["high"],
                low=df_sig["low"],
                close=df_sig["close"],
                name="OHLC",
            )
        )

        # Tight y-axis
        ymin = float(df_sig["low"].min())
        ymax = float(df_sig["high"].max())
        pad = max((ymax - ymin) * 0.06, 0.0004)
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

        # Remove weekend gaps
        if remove_weekends:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        # Support / Resistance lines (green/red)
        if "support" in df_sig and df_sig["support"].notna().any():
            fig.add_trace(go.Scatter(
                x=df_sig["date"], y=df_sig["support"],
                mode="lines", name="Support",
                line=dict(width=2, color="lime"),
                opacity=0.85,
                hovertemplate="Support: %{y:.5f}<extra></extra>" if float(df_sig["support"].iloc[-1]) < 10 else "Support: %{y:.2f}<extra></extra>",
            ))
        if "resistance" in df_sig and df_sig["resistance"].notna().any():
            fig.add_trace(go.Scatter(
                x=df_sig["date"], y=df_sig["resistance"],
                mode="lines", name="Resistance",
                line=dict(width=2, color="red"),
                opacity=0.85,
                hovertemplate="Resistance: %{y:.5f}<extra></extra>" if float(df_sig["resistance"].iloc[-1]) < 10 else "Resistance: %{y:.2f}<extra></extra>",
            ))

        # Fib + Golden Pocket (clear right tags)
        if fib:
            levels = [
                ("Fib 0.382", fib["0.382"]),
                ("Fib 0.500", fib["0.500"]),
                ("Fib 0.618", fib["0.618"]),
                ("Fib 0.650", fib["0.650"]),
            ]
            for name, y in levels:
                fig.add_shape(
                    type="line",
                    x0=df_sig["date"].iloc[0], x1=df_sig["date"].iloc[-1],
                    y0=y, y1=y,
                    line=dict(width=1, dash="dot", color="rgba(255,255,255,0.35)"),
                )
                yy = float(y)
                _add_right_price_tag(fig, yy, f"{name}: {yy:.5f}" if yy < 10 else f"{name}: {yy:.2f}")

            gp_a = float(fib["0.618"])
            gp_b = float(fib["0.650"])
            gp_low = min(gp_a, gp_b)
            gp_high = max(gp_a, gp_b)

            fig.add_shape(
                type="rect",
                x0=df_sig["date"].iloc[0], x1=df_sig["date"].iloc[-1],
                y0=gp_low, y1=gp_high,
                fillcolor="rgba(255, 215, 0, 0.10)",
                line=dict(width=0),
                layer="below",
            )
            _add_right_price_tag(
                fig,
                float(gp_high),
                f"Golden Pocket: {gp_low:.5f}–{gp_high:.5f}" if gp_low < 10 else f"Golden Pocket: {gp_low:.2f}–{gp_high:.2f}",
            )

        # Optional pivots
        if show_pivots:
            piv_hi = df_sig[df_sig["pivot_high"]]
            piv_lo = df_sig[df_sig["pivot_low"]]
            if not piv_hi.empty:
                fig.add_trace(go.Scatter(
                    x=piv_hi["date"], y=piv_hi["high"],
                    mode="markers", name="Pivot High",
                    marker=dict(size=5, symbol="triangle-up", color="orange"),
                ))
            if not piv_lo.empty:
                fig.add_trace(go.Scatter(
                    x=piv_lo["date"], y=piv_lo["low"],
                    mode="markers", name="Pivot Low",
                    marker=dict(size=5, symbol="triangle-down", color="cyan"),
                ))

        # Optional patterns
        if show_patterns:
            patt = df_sig[df_sig["bull_engulf"] | df_sig["bear_engulf"] | df_sig["doji"] | df_sig["hammer"] | df_sig["shooting_star"]].copy()
            if not patt.empty:
                def _label_row(r):
                    labels = []
                    if r["bull_engulf"]: labels.append("Bull Engulf")
                    if r["bear_engulf"]: labels.append("Bear Engulf")
                    if r["doji"]: labels.append("Doji")
                    if r["hammer"]: labels.append("Hammer")
                    if r["shooting_star"]: labels.append("Shooting Star")
                    return ", ".join(labels) if labels else "Pattern"

                patt["label"] = patt.apply(_label_row, axis=1)
                fig.add_trace(go.Scatter(
                    x=patt["date"], y=patt["close"],
                    mode="markers", name="Patterns",
                    marker=dict(size=4, symbol="circle", color="magenta"),
                    text=patt["label"],
                    hovertemplate="%{text}<br>Close: %{y:.5f}<extra></extra>" if float(patt["close"].iloc[-1]) < 10 else "%{text}<br>Close: %{y:.2f}<extra></extra>",
                ))

        # BUY/SELL arrows + confidence + EXACT SIGNAL PRICE
        if show_signals:
            srows = df_sig[df_sig["signal"].notna()].copy()
            if not srows.empty:
                buys = srows[srows["signal"] == "BUY"].copy()
                sells = srows[srows["signal"] == "SELL"].copy()

                if not buys.empty:
                    buy_price_fmt = [
                        (f"{p:.5f}" if float(p) < 10 else f"{p:.2f}")
                        for p in buys["signal_price"].astype(float).values
                    ]
                    fig.add_trace(go.Scatter(
                        x=buys["date"],
                        y=buys["low"] - (0.25 * buys["atr"].fillna(buys["atr"].median())),
                        mode="markers+text",
                        marker=dict(
                            symbol="triangle-up",
                            size=14,
                            color="lime",
                            line=dict(width=1, color="rgba(0,0,0,0.6)"),
                        ),
                        text=[
                            f"BUY @ {pp} ({c:.0f}%)"
                            for pp, c in zip(buy_price_fmt, buys["conf"].astype(float).values)
                        ],
                        textposition="top center",
                        textfont=dict(color="lime", size=12),
                        name="BUY",
                        hovertemplate="BUY @ %{customdata}<extra></extra>",
                        customdata=buy_price_fmt,
                    ))

                if not sells.empty:
                    sell_price_fmt = [
                        (f"{p:.5f}" if float(p) < 10 else f"{p:.2f}")
                        for p in sells["signal_price"].astype(float).values
                    ]
                    fig.add_trace(go.Scatter(
                        x=sells["date"],
                        y=sells["high"] + (0.25 * sells["atr"].fillna(sells["atr"].median())),
                        mode="markers+text",
                        marker=dict(
                            symbol="triangle-down",
                            size=14,
                            color="red",
                            line=dict(width=1, color="rgba(0,0,0,0.6)"),
                        ),
                        text=[
                            f"SELL @ {pp} ({c:.0f}%)"
                            for pp, c in zip(sell_price_fmt, sells["conf"].astype(float).values)
                        ],
                        textposition="bottom center",
                        textfont=dict(color="red", size=12),
                        name="SELL",
                        hovertemplate="SELL @ %{customdata}<extra></extra>",
                        customdata=sell_price_fmt,
                    ))

        fig.update_layout(
            template="plotly_dark",
            height=860 if big_chart else 620,
            margin=dict(l=0, r=60, t=25, b=0),
            showlegend=False,
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", side="right")

        st.plotly_chart(fig, use_container_width=True)

        # Optional table: all signals with exact prices
        with st.expander("Signals Table (exact prices)"):
            sig_tbl = df_sig[df_sig["signal"].notna()][["date", "signal", "signal_price", "conf"]].copy()
            if not sig_tbl.empty:
                sig_tbl["signal_price"] = sig_tbl["signal_price"].astype(float)
                sig_tbl["conf"] = sig_tbl["conf"].astype(float)
                st.dataframe(sig_tbl.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("No signals in current window/timeframe.")