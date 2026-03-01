import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Volume"] / df["Volume"].rolling(period).mean()


def _build_strategy(
    df: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    breakout_lookback: int,
    atr_period: int,
    rvol_period: int,
    rvol_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["EMA_fast"] = _ema(out["Close"], ema_fast)
    out["EMA_slow"] = _ema(out["Close"], ema_slow)
    out["ATR"] = _atr(out, period=atr_period)
    out["RVOL"] = _rvol(out, period=rvol_period)

    out["bull"] = out["EMA_fast"] > out["EMA_slow"]
    out["bear"] = out["EMA_fast"] < out["EMA_slow"]

    out["break_high"] = out["Close"] > out["High"].rolling(breakout_lookback).max().shift(1)
    out["break_low"] = out["Close"] < out["Low"].rolling(breakout_lookback).min().shift(1)

    out["long"] = out["bull"] & out["break_high"] & (out["RVOL"] > rvol_threshold)
    out["short"] = out["bear"] & out["break_low"] & (out["RVOL"] > rvol_threshold)

    pos = 0
    pos_vals: list[int] = []
    for i in range(len(out)):
        if pos == 0:
            if bool(out["long"].iloc[i]):
                pos = 1
            elif bool(out["short"].iloc[i]):
                pos = -1
        else:
            if pos == 1 and bool(out["bear"].iloc[i]):
                pos = 0
            elif pos == -1 and bool(out["bull"].iloc[i]):
                pos = 0
        pos_vals.append(pos)

    out["pos"] = pos_vals
    out["ret"] = out["Close"].pct_change().fillna(0.0)
    out["equity"] = (1.0 + out["ret"] * out["pos"].shift(1).fillna(0.0)).cumprod()
    return out


def _add_regime_spans(fig: go.Figure, strategy_df: pd.DataFrame) -> None:
    if strategy_df.empty:
        return

    regime = np.where(strategy_df["bull"], 1, np.where(strategy_df["bear"], -1, 0))
    index_vals = strategy_df.index.to_list()
    start = 0

    for i in range(1, len(regime) + 1):
        end_segment = i == len(regime) or regime[i] != regime[start]
        if not end_segment:
            continue

        state = regime[start]
        if state != 0:
            color = "rgba(0, 180, 120, 0.08)" if state == 1 else "rgba(255, 89, 89, 0.08)"
            fig.add_vrect(
                x0=index_vals[start],
                x1=index_vals[i - 1],
                fillcolor=color,
                line_width=0,
                row=1,
                col=1,
            )
        start = i


def _build_figure(strategy_df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
        subplot_titles=(f"{symbol} Trend + Break Strategy", "Equity Curve"),
    )

    _add_regime_spans(fig, strategy_df)

    fig.add_trace(
        go.Scatter(
            x=strategy_df.index,
            y=strategy_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#67b7ff", width=1.8),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_df.index,
            y=strategy_df["EMA_fast"],
            mode="lines",
            name="EMA Fast",
            line=dict(color="#f5b700", width=1.2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_df.index,
            y=strategy_df["EMA_slow"],
            mode="lines",
            name="EMA Slow",
            line=dict(color="#ff5a5f", width=1.2),
        ),
        row=1,
        col=1,
    )

    long_df = strategy_df[strategy_df["long"]]
    short_df = strategy_df[strategy_df["short"]]

    fig.add_trace(
        go.Scatter(
            x=long_df.index,
            y=long_df["Close"],
            mode="markers",
            name="Long Signal",
            marker=dict(symbol="triangle-up", size=10, color="#00d084"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=short_df.index,
            y=short_df["Close"],
            mode="markers",
            name="Short Signal",
            marker=dict(symbol="triangle-down", size=10, color="#ff6377"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_df.index,
            y=strategy_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#9d8cff", width=2.0),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=860,
        template="plotly_dark",
        margin=dict(l=12, r=12, t=48, b=12),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(140,160,185,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(140,160,185,0.2)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(140,160,185,0.2)", row=2, col=1)
    return fig


@_fragment
def render_tab_trend_engine(symbol: str):
    target_symbol = (symbol or "AAPL").upper().strip()
    st.subheader("Trend Engine")
    st.caption("EMA trend filter + 20-bar breakout + RVOL trigger on Yahoo Finance history.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        range_label = st.selectbox(
            "History",
            options=list(RANGE_TO_YAHOO.keys()),
            index=1,
            key=f"trend_range_{target_symbol}",
        )
    with col_b:
        valid_intervals = RANGE_INTERVALS[range_label]
        interval = st.selectbox(
            "Interval",
            options=valid_intervals,
            index=valid_intervals.index(DEFAULT_INTERVAL[range_label]),
            key=f"trend_interval_{target_symbol}_{range_label}",
        )

    with st.expander("Strategy Parameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        ema_fast = p1.number_input("EMA Fast", min_value=2, max_value=200, value=21, step=1)
        ema_slow = p2.number_input("EMA Slow", min_value=3, max_value=300, value=55, step=1)
        breakout_lookback = p3.number_input("Breakout Lookback", min_value=5, max_value=200, value=20, step=1)

        p4, p5, p6 = st.columns(3)
        atr_period = p4.number_input("ATR Period", min_value=2, max_value=200, value=14, step=1)
        rvol_period = p5.number_input("RVOL Period", min_value=2, max_value=200, value=20, step=1)
        rvol_threshold = p6.number_input("RVOL Threshold", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

    try:
        payload = _fetch_chart_payload(
            symbol=target_symbol,
            range_key=RANGE_TO_YAHOO[range_label],
            interval=interval,
        )
        raw_df, _ = _build_chart_df(payload)
    except Exception as exc:
        st.error(f"Unable to load Yahoo history: {exc}")
        return

    if raw_df.empty:
        st.warning("Yahoo returned no price rows for this selection.")
        return

    strategy_input = raw_df[["datetime", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    strategy_input = strategy_input.set_index("Date").sort_index()

    min_needed = max(int(ema_slow), int(breakout_lookback), int(rvol_period), int(atr_period)) + 5
    if len(strategy_input) < min_needed:
        st.warning(
            f"Not enough rows for current parameters ({len(strategy_input)} rows, need about {min_needed}+). "
            "Increase history range or use smaller lookbacks."
        )
        return

    strategy_df = _build_strategy(
        strategy_input,
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        breakout_lookback=int(breakout_lookback),
        atr_period=int(atr_period),
        rvol_period=int(rvol_period),
        rvol_threshold=float(rvol_threshold),
    )

    entries = int(((strategy_df["pos"] != strategy_df["pos"].shift(1)) & (strategy_df["pos"] != 0)).sum())
    current_pos = int(strategy_df["pos"].iloc[-1])
    regime_txt = "Bullish" if bool(strategy_df["bull"].iloc[-1]) else ("Bearish" if bool(strategy_df["bear"].iloc[-1]) else "Neutral")
    pnl_pct = (float(strategy_df["equity"].iloc[-1]) - 1.0) * 100.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Last Close", f"{strategy_df['Close'].iloc[-1]:,.2f}")
    m2.metric("Regime", regime_txt)
    m3.metric("Open Position", "Long" if current_pos == 1 else ("Short" if current_pos == -1 else "Flat"))
    m4.metric("Strategy Return", f"{pnl_pct:+.2f}%")
    st.caption(f"Signals triggered: {entries}")

    fig = _build_figure(strategy_df, target_symbol)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"trend_engine_chart_{target_symbol}_{range_label}_{interval}",
    )

    with st.expander("Latest Signals"):
        cols = ["Close", "EMA_fast", "EMA_slow", "RVOL", "long", "short", "pos", "equity"]
        st.dataframe(strategy_df[cols].tail(30), width="stretch")
