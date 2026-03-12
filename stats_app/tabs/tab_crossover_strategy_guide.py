from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from stats_app.tabs.tab_yahoo_data import _build_chart_df, _fetch_chart_payload

_fragment = getattr(st, "fragment", lambda f: f)

RANGE_TO_YAHOO = {
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "MAX": "max",
}

RANGE_INTERVALS = {
    "6M": ["1d", "1wk"],
    "1Y": ["1d", "1wk"],
    "2Y": ["1d", "1wk", "1mo"],
    "5Y": ["1wk", "1mo"],
    "MAX": ["1wk", "1mo", "3mo"],
}

DEFAULT_INTERVAL = {
    "6M": "1d",
    "1Y": "1d",
    "2Y": "1d",
    "5Y": "1wk",
    "MAX": "1wk",
}

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]


def normalize_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in out.columns
        ]

    rename_map = {}
    for col in out.columns:
        c = str(col).strip().lower()
        if c in {"date", "datetime"}:
            rename_map[col] = "Date"
        elif c == "open":
            rename_map[col] = "Open"
        elif c == "high":
            rename_map[col] = "High"
        elif c == "low":
            rename_map[col] = "Low"
        elif c == "close":
            rename_map[col] = "Close"
        elif c in {"adj close", "adj_close", "adjclose"}:
            rename_map[col] = "Adj Close"
        elif c == "volume":
            rename_map[col] = "Volume"

    out = out.rename(columns=rename_map)

    if "Date" not in out.columns:
        out = out.reset_index()
        if "index" in out.columns:
            out = out.rename(columns={"index": "Date"})

    if "Date" not in out.columns:
        raise ValueError("No Date column found.")

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "Volume" in out.columns:
        out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")
    else:
        out["Volume"] = np.nan

    return out.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def _fetch_yahoo_price_df(symbol: str, range_key: str, interval: str) -> pd.DataFrame:
    payload = _fetch_chart_payload(symbol=symbol, range_key=range_key, interval=interval)
    raw_df, _ = _build_chart_df(payload)
    if raw_df.empty:
        return pd.DataFrame()

    mapped = raw_df[["datetime", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return normalize_yahoo_columns(mapped)


def compute_crossover_indicators(
        df: pd.DataFrame,
        price_ma_length: int = 20,
        fast_ma: int = 20,
        slow_ma: int = 50,
        ma_type: str = "EMA",
) -> pd.DataFrame:
    out = df.copy()
    ma_type = ma_type.upper()

    if ma_type == "SMA":
        out["PRICE_MA"] = out["Close"].rolling(price_ma_length, min_periods=1).mean()
        out["FAST_MA"] = out["Close"].rolling(fast_ma, min_periods=1).mean()
        out["SLOW_MA"] = out["Close"].rolling(slow_ma, min_periods=1).mean()
    else:
        out["PRICE_MA"] = out["Close"].ewm(span=price_ma_length, adjust=False).mean()
        out["FAST_MA"] = out["Close"].ewm(span=fast_ma, adjust=False).mean()
        out["SLOW_MA"] = out["Close"].ewm(span=slow_ma, adjust=False).mean()

    out["PRICE_CROSS_BUY"] = (
            (out["Close"].shift(1) <= out["PRICE_MA"].shift(1))
            & (out["Close"] > out["PRICE_MA"])
    )
    out["PRICE_CROSS_SELL"] = (
            (out["Close"].shift(1) >= out["PRICE_MA"].shift(1))
            & (out["Close"] < out["PRICE_MA"])
    )

    out["GOLDEN_CROSS"] = (
            (out["FAST_MA"].shift(1) <= out["SLOW_MA"].shift(1))
            & (out["FAST_MA"] > out["SLOW_MA"])
    )
    out["DEATH_CROSS"] = (
            (out["FAST_MA"].shift(1) >= out["SLOW_MA"].shift(1))
            & (out["FAST_MA"] < out["SLOW_MA"])
    )

    ma_gap_pct = ((out["FAST_MA"] - out["SLOW_MA"]).abs() / out["Close"].replace(0, np.nan)) * 100
    out["SIDEWAYS_ZONE"] = ma_gap_pct < 1.0
    return out


def build_interpretation(df: pd.DataFrame) -> dict[str, Any]:
    last = df.iloc[-1]

    if bool(last["GOLDEN_CROSS"]):
        double_ma_state = "Golden Cross just triggered"
    elif bool(last["DEATH_CROSS"]):
        double_ma_state = "Death Cross just triggered"
    elif last["FAST_MA"] > last["SLOW_MA"]:
        double_ma_state = "Fast MA above Slow MA"
    elif last["FAST_MA"] < last["SLOW_MA"]:
        double_ma_state = "Fast MA below Slow MA"
    else:
        double_ma_state = "Neutral"

    if bool(last["PRICE_CROSS_BUY"]):
        price_cross_state = "Price just crossed above MA"
    elif bool(last["PRICE_CROSS_SELL"]):
        price_cross_state = "Price just crossed below MA"
    elif last["Close"] > last["PRICE_MA"]:
        price_cross_state = "Price above MA"
    elif last["Close"] < last["PRICE_MA"]:
        price_cross_state = "Price below MA"
    else:
        price_cross_state = "Price on MA"

    market_condition = "Sideways / noisy" if bool(last["SIDEWAYS_ZONE"]) else "Trending"

    if bool(last["SIDEWAYS_ZONE"]):
        takeaway = "Higher risk of false crossover signals."
    elif last["FAST_MA"] > last["SLOW_MA"] and last["Close"] > last["PRICE_MA"]:
        takeaway = "Bullish alignment: trend and price are on the same side."
    elif last["FAST_MA"] < last["SLOW_MA"] and last["Close"] < last["PRICE_MA"]:
        takeaway = "Bearish alignment: trend and price are on the same side."
    else:
        takeaway = "Mixed signals: trend and price crossover are not fully aligned."

    return {
        "price_cross_state": price_cross_state,
        "double_ma_state": double_ma_state,
        "market_condition": market_condition,
        "takeaway": takeaway,
        "last_close": round(float(last["Close"]), 2),
        "price_ma": round(float(last["PRICE_MA"]), 2),
        "fast_ma": round(float(last["FAST_MA"]), 2),
        "slow_ma": round(float(last["SLOW_MA"]), 2),
    }


def make_price_crossover_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["PRICE_MA"],
            mode="lines",
            name="Single MA",
            line=dict(width=2, color="#f59e0b"),
        )
    )

    buy_df = df[df["PRICE_CROSS_BUY"]]
    sell_df = df[df["PRICE_CROSS_SELL"]]

    if not buy_df.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_df["Date"],
                y=buy_df["Low"] * 0.99,
                mode="markers",
                name="BUY",
                customdata=buy_df["Close"],
                hovertext=["Price Cross BUY"] * len(buy_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=20, symbol="triangle-up", color="#16a34a", line=dict(width=2, color="black")),
            )
        )

    if not sell_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_df["Date"],
                y=sell_df["High"] * 1.01,
                mode="markers",
                name="SELL",
                customdata=sell_df["Close"],
                hovertext=["Price Cross SELL"] * len(sell_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=20, symbol="triangle-down", color="#dc2626", line=dict(width=2, color="black")),
            )
        )

    fig.update_layout(
        title="1. Price Crossover Strategy",
        height=520,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Price")
    return fig


def make_double_ma_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2.5, color="#67b7ff"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["FAST_MA"],
            mode="lines",
            name="Fast MA",
            line=dict(width=2.2, color="#2563eb"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["SLOW_MA"],
            mode="lines",
            name="Slow MA",
            line=dict(width=2.2, color="#d97706"),
        )
    )

    golden_df = df[df["GOLDEN_CROSS"]]
    death_df = df[df["DEATH_CROSS"]]

    if not golden_df.empty:
        fig.add_trace(
            go.Scatter(
                x=golden_df["Date"],
                y=golden_df["FAST_MA"] * 0.99,
                mode="markers",
                name="Golden Cross",
                customdata=golden_df["Close"],
                hovertext=["Golden Cross BUY"] * len(golden_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=20, symbol="triangle-up", color="#16a34a", line=dict(width=2, color="black")),
            )
        )

    if not death_df.empty:
        fig.add_trace(
            go.Scatter(
                x=death_df["Date"],
                y=death_df["FAST_MA"] * 1.01,
                mode="markers",
                name="Death Cross",
                customdata=death_df["Close"],
                hovertext=["Death Cross SELL"] * len(death_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=20, symbol="triangle-down", color="#dc2626", line=dict(width=2, color="black")),
            )
        )

    fig.update_layout(
        title="2. Double Moving Average Crossover",
        height=520,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Value")
    return fig


def make_sideways_warning_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(width=2.2, color="#67b7ff")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["FAST_MA"], mode="lines", name="Fast MA", line=dict(width=2, color="#2563eb")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["SLOW_MA"], mode="lines", name="Slow MA", line=dict(width=2, color="#d97706")),
        row=1,
        col=1,
    )

    # Calculate all signals for this chart
    buy_df = df[df["PRICE_CROSS_BUY"]]
    sell_df = df[df["PRICE_CROSS_SELL"]]
    golden_df = df[df["GOLDEN_CROSS"]]
    death_df = df[df["DEATH_CROSS"]]

    # Price Crosses (Slightly smaller, lighter colors)
    if not buy_df.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_df["Date"],
                y=buy_df["Low"] * 0.99,
                mode="markers",
                name="Price Cross BUY",
                customdata=buy_df["Close"],
                hovertext=["Price Cross BUY"] * len(buy_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=14, symbol="triangle-up", color="#00d084", line=dict(width=1, color="black")),
            ),
            row=1, col=1
        )
    if not sell_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_df["Date"],
                y=sell_df["High"] * 1.01,
                mode="markers",
                name="Price Cross SELL",
                customdata=sell_df["Close"],
                hovertext=["Price Cross SELL"] * len(sell_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=14, symbol="triangle-down", color="#ff6377", line=dict(width=1, color="black")),
            ),
            row=1, col=1
        )

    # MA Crosses (Larger, bolder colors)
    if not golden_df.empty:
        fig.add_trace(
            go.Scatter(
                x=golden_df["Date"],
                y=golden_df["FAST_MA"] * 0.99,
                mode="markers",
                name="Golden Cross",
                customdata=golden_df["Close"],
                hovertext=["Golden Cross BUY"] * len(golden_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=20, symbol="triangle-up", color="#16a34a", line=dict(width=2, color="black")),
            ),
            row=1, col=1
        )
    if not death_df.empty:
        fig.add_trace(
            go.Scatter(
                x=death_df["Date"],
                y=death_df["FAST_MA"] * 1.01,
                mode="markers",
                name="Death Cross",
                customdata=death_df["Close"],
                hovertext=["Death Cross SELL"] * len(death_df),
                hovertemplate="%{x}<br>Close Price: $%{customdata:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(size=20, symbol="triangle-down", color="#dc2626", line=dict(width=2, color="black")),
            ),
            row=1, col=1
        )

    # Sideways zone
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["SIDEWAYS_ZONE"].astype(int), name="Sideways Zone", marker_color="#ff6377"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="3. Why Crossovers Fail in Sideways Markets",
        height=560,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_yaxes(title="Price", row=1, col=1)
    fig.update_yaxes(title="Sideways", row=2, col=1, range=[0, 1.2])
    fig.update_xaxes(title="Date", row=2, col=1)
    return fig


@_fragment
def render_tab_crossover_strategy_guide(symbol: str) -> None:
    target_symbol = (symbol or "AAPL").upper().strip()
    st.subheader("Crossover Strategy Guide")
    st.caption("Visual guide for price and moving-average crossovers using Yahoo history data.")

    st.markdown("### What this tab explains")
    st.write(
        "This tab turns Yahoo Finance history into a live visual guide for crossover ideas:\n"
        "- Price crossing above or below one moving average\n"
        "- Fast MA crossing above or below slow MA\n"
        "- Golden Cross and Death Cross\n"
        "- Why sideways markets create false signals"
    )

    a, b = st.columns(2)
    with a:
        range_label = st.selectbox(
            "History",
            options=list(RANGE_TO_YAHOO.keys()),
            index=1,
            key=f"cross_guide_range_{target_symbol}",
        )
    with b:
        valid_intervals = RANGE_INTERVALS[range_label]
        interval = st.selectbox(
            "Interval",
            options=valid_intervals,
            index=valid_intervals.index(DEFAULT_INTERVAL[range_label]),
            key=f"cross_guide_interval_{target_symbol}_{range_label}",
        )

    try:
        df = _fetch_yahoo_price_df(target_symbol, RANGE_TO_YAHOO[range_label], interval)
    except Exception as exc:
        st.error(f"Unable to load Yahoo history: {exc}")
        return

    if df.empty:
        st.warning("Yahoo returned no price rows for this selection.")
        return

    st.markdown("### Settings")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ma_type = st.selectbox("MA Type", ["EMA", "SMA"], index=0, key=f"cross_guide_type_{target_symbol}")
    with c2:
        price_ma_length = st.number_input("Single MA Length", min_value=2, max_value=300, value=20, step=1,
                                          key=f"cross_guide_price_ma_{target_symbol}")
    with c3:
        fast_ma = st.number_input("Fast MA", min_value=2, max_value=300, value=20, step=1,
                                  key=f"cross_guide_fast_ma_{target_symbol}")
    with c4:
        slow_ma = st.number_input("Slow MA", min_value=3, max_value=400, value=50, step=1,
                                  key=f"cross_guide_slow_ma_{target_symbol}")

    if fast_ma >= slow_ma:
        st.warning("Fast MA should usually be smaller than Slow MA.")

    min_needed = max(int(price_ma_length), int(fast_ma), int(slow_ma)) + 5
    if len(df) < min_needed:
        st.warning(
            f"Not enough rows for current parameters ({len(df)} rows, need about {min_needed}+). Increase history or reduce lookbacks."
        )
        return

    try:
        result_df = compute_crossover_indicators(
            df=df,
            price_ma_length=int(price_ma_length),
            fast_ma=int(fast_ma),
            slow_ma=int(slow_ma),
            ma_type=str(ma_type),
        )
    except Exception as exc:
        st.error(f"Indicator error: {exc}")
        return

    interp = build_interpretation(result_df)

    st.markdown("### Current Interpretation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Close", interp["last_close"])
    m2.metric("Price vs MA", interp["price_cross_state"])
    m3.metric("Fast vs Slow", interp["double_ma_state"])
    m4.metric("Market Condition", interp["market_condition"])

    st.info(interp["takeaway"])
    st.json(
        {
            "Price MA": interp["price_ma"],
            "Fast MA": interp["fast_ma"],
            "Slow MA": interp["slow_ma"],
            "Data Source": f"Yahoo Finance ({range_label} / {interval})",
        },
        expanded=True,
    )

    st.markdown("### Strategy Rules")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("#### 1. Price Crossover Strategy")
        st.write(
            "**Buy signal:** when price crosses above a single moving average  \n"
            "**Sell signal:** when price crosses below a single moving average\n\n"
            "This strategy reacts faster, but it gives more false signals in choppy markets."
        )
    with e2:
        st.markdown("#### 2. Double Moving Average Crossover")
        st.write(
            "**Golden Cross:** fast MA crosses above slow MA  \n"
            "**Death Cross:** fast MA crosses below slow MA\n\n"
            "This strategy is slower, but usually cleaner than the single price crossover."
        )

    st.markdown("### Visual Charts")
    st.plotly_chart(
        make_price_crossover_chart(result_df),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"cross_guide_price_chart_{target_symbol}_{range_label}_{interval}",
    )
    st.plotly_chart(
        make_double_ma_chart(result_df),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"cross_guide_double_chart_{target_symbol}_{range_label}_{interval}",
    )
    st.plotly_chart(
        make_sideways_warning_chart(result_df),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"cross_guide_sideways_chart_{target_symbol}_{range_label}_{interval}",
    )

    st.markdown("### Recent Events")
    price_events = result_df.loc[
        result_df["PRICE_CROSS_BUY"] | result_df["PRICE_CROSS_SELL"],
        ["Date", "Close", "PRICE_MA", "PRICE_CROSS_BUY", "PRICE_CROSS_SELL"],
    ].copy()

    if not price_events.empty:
        price_events["Signal"] = np.where(price_events["PRICE_CROSS_BUY"], "BUY", "SELL")
        price_events["Date"] = pd.to_datetime(price_events["Date"]).dt.strftime("%Y-%m-%d")
        st.markdown("#### Price Crossover Events")
        st.dataframe(
            price_events[["Date", "Signal", "Close", "PRICE_MA"]].tail(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No recent price crossover events.")

    ma_events = result_df.loc[
        result_df["GOLDEN_CROSS"] | result_df["DEATH_CROSS"],
        ["Date", "Close", "FAST_MA", "SLOW_MA", "GOLDEN_CROSS", "DEATH_CROSS"],
    ].copy()

    if not ma_events.empty:
        ma_events["Signal"] = np.where(ma_events["GOLDEN_CROSS"], "Golden Cross", "Death Cross")
        ma_events["Date"] = pd.to_datetime(ma_events["Date"]).dt.strftime("%Y-%m-%d")
        st.markdown("#### Double MA Events")
        st.dataframe(
            ma_events[["Date", "Signal", "Close", "FAST_MA", "SLOW_MA"]].tail(20),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No recent double moving average crossover events.")

    st.markdown("### When to Use Each Strategy")
    u1, u2 = st.columns(2)
    with u1:
        st.markdown("#### Price Crossover")
        st.write(
            "Better for:\n"
            "- quick entries\n"
            "- short-term trading\n"
            "- fast reaction to price change\n\n"
            "Weakness:\n"
            "- more false signals\n"
            "- vulnerable in sideways markets"
        )
    with u2:
        st.markdown("#### Double MA Crossover")
        st.write(
            "Better for:\n"
            "- swing trading\n"
            "- cleaner trend confirmation\n"
            "- fewer noisy entries\n\n"
            "Weakness:\n"
            "- slower signals\n"
            "- late entry after trend already started"
        )

    st.warning(
        "Crossovers work best in trending markets. In sideways markets, frequent crossing can create whipsaw signals.")