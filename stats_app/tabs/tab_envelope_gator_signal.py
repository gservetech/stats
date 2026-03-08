import math
from dataclasses import dataclass
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


@dataclass
class EnvelopeGatorConfig:
    envelope_ma_period: int = 20
    envelope_atr_period: int = 14
    envelope_atr_mult: float = 1.25

    jaw_period: int = 13
    teeth_period: int = 8
    lips_period: int = 5

    stop_atr_mult: float = 1.3
    target_atr_mult: float = 2.0

    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 1.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _prepare_price_df(chart_payload: dict[str, Any]) -> pd.DataFrame:
    raw_df, _ = _build_chart_df(chart_payload)
    if raw_df.empty:
        return pd.DataFrame()

    base_cols = ["datetime", "open", "high", "low", "close"]
    if not all(col in raw_df.columns for col in base_cols):
        return pd.DataFrame()

    use_cols = base_cols + (["volume"] if "volume" in raw_df.columns else [])
    df = raw_df[use_cols].rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_indicators(df: pd.DataFrame, cfg: EnvelopeGatorConfig) -> pd.DataFrame:
    out = df.copy()

    out["ATR"] = calc_atr(out, cfg.envelope_atr_period)
    out["MIDLINE"] = out["Close"].ewm(span=cfg.envelope_ma_period, adjust=False).mean()
    out["UPPER_ENV"] = out["MIDLINE"] + (out["ATR"] * cfg.envelope_atr_mult)
    out["LOWER_ENV"] = out["MIDLINE"] - (out["ATR"] * cfg.envelope_atr_mult)
    out["ENV_WIDTH"] = out["UPPER_ENV"] - out["LOWER_ENV"]

    out["ENV_EXPANDING"] = out["ENV_WIDTH"] > out["ENV_WIDTH"].shift(1)
    out["ENV_SHRINKING"] = out["ENV_WIDTH"] < out["ENV_WIDTH"].shift(1)

    env_width_ma = out["ENV_WIDTH"].rolling(10).mean()
    out["COMPRESSION"] = out["ENV_WIDTH"] < env_width_ma

    out["JAW"] = out["Close"].rolling(cfg.jaw_period).mean()
    out["TEETH"] = out["Close"].rolling(cfg.teeth_period).mean()
    out["LIPS"] = out["Close"].rolling(cfg.lips_period).mean()

    out["GATOR_UPPER"] = (out["JAW"] - out["TEETH"]).abs()
    out["GATOR_LOWER"] = -1.0 * (out["TEETH"] - out["LIPS"]).abs()
    out["GATOR_TOTAL"] = out["GATOR_UPPER"] + out["GATOR_LOWER"].abs()

    out["GATOR_EXPANDING"] = (
        (out["GATOR_UPPER"] > out["GATOR_UPPER"].shift(1))
        & (out["GATOR_LOWER"].abs() > out["GATOR_LOWER"].abs().shift(1))
    )
    out["GATOR_SHRINKING"] = (
        (out["GATOR_UPPER"] < out["GATOR_UPPER"].shift(1))
        & (out["GATOR_LOWER"].abs() < out["GATOR_LOWER"].abs().shift(1))
    )

    out["TREND_UP"] = out["Close"] > out["MIDLINE"]
    out["TREND_DOWN"] = out["Close"] < out["MIDLINE"]

    recent_compression = out["COMPRESSION"].rolling(4).max().fillna(0).astype(bool)

    out["LONG_SIGNAL"] = (
        recent_compression
        & out["ENV_EXPANDING"]
        & out["GATOR_EXPANDING"]
        & out["TREND_UP"]
        & (out["Close"] > out["Close"].shift(1))
    )

    out["SHORT_SIGNAL"] = (
        recent_compression
        & out["ENV_EXPANDING"]
        & out["GATOR_EXPANDING"]
        & out["TREND_DOWN"]
        & (out["Close"] < out["Close"].shift(1))
    )

    out["EXIT_LONG_SIGNAL"] = (
        (out["Close"] >= out["UPPER_ENV"])
        & (out["ENV_SHRINKING"] | out["GATOR_SHRINKING"])
    )

    out["EXIT_SHORT_SIGNAL"] = (
        (out["Close"] <= out["LOWER_ENV"])
        & (out["ENV_SHRINKING"] | out["GATOR_SHRINKING"])
    )

    return out


def run_backtest(
    df: pd.DataFrame,
    cfg: EnvelopeGatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    capital = cfg.initial_capital
    position: dict[str, Any] | None = None

    start_idx = max(
        cfg.envelope_ma_period,
        cfg.envelope_atr_period,
        cfg.jaw_period,
        cfg.teeth_period,
        cfg.lips_period,
    ) + 2

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        date = row["Date"]
        equity_curve.append({"Date": date, "Equity": capital})

        if pd.isna(row["ATR"]) or pd.isna(row["MIDLINE"]):
            continue

        if position is None:
            risk_amount = capital * (cfg.risk_per_trade_pct / 100.0)

            if bool(row["LONG_SIGNAL"]):
                entry = float(row["Close"])
                stop = min(float(row["MIDLINE"]), float(row["Low"])) - float(row["ATR"]) * cfg.stop_atr_mult
                target = entry + float(row["ATR"]) * cfg.target_atr_mult
                risk_per_share = max(entry - stop, 1e-9)
                position = {
                    "type": "BUY",
                    "entry_idx": i,
                    "entry_date": date,
                    "entry_price": entry,
                    "stop": stop,
                    "target": target,
                    "shares": risk_amount / risk_per_share,
                }

            elif bool(row["SHORT_SIGNAL"]):
                entry = float(row["Close"])
                stop = max(float(row["MIDLINE"]), float(row["High"])) + float(row["ATR"]) * cfg.stop_atr_mult
                target = entry - float(row["ATR"]) * cfg.target_atr_mult
                risk_per_share = max(stop - entry, 1e-9)
                position = {
                    "type": "SELL",
                    "entry_idx": i,
                    "entry_date": date,
                    "entry_price": entry,
                    "stop": stop,
                    "target": target,
                    "shares": risk_amount / risk_per_share,
                }
            continue

        exit_price = None
        exit_reason = None

        if position["type"] == "BUY":
            if row["Low"] <= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "Stop"
            elif row["High"] >= position["target"]:
                exit_price = position["target"]
                exit_reason = "Target"
            elif bool(row["EXIT_LONG_SIGNAL"]):
                exit_price = float(row["Close"])
                exit_reason = "Momentum Exit"

            if exit_price is not None:
                pnl = (float(exit_price) - position["entry_price"]) * position["shares"]

        else:
            if row["High"] >= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "Stop"
            elif row["Low"] <= position["target"]:
                exit_price = position["target"]
                exit_reason = "Target"
            elif bool(row["EXIT_SHORT_SIGNAL"]):
                exit_price = float(row["Close"])
                exit_reason = "Momentum Exit"

            if exit_price is not None:
                pnl = (position["entry_price"] - float(exit_price)) * position["shares"]

        if exit_price is None:
            continue

        capital += pnl
        trades.append(
            {
                "Type": position["type"],
                "Entry Date": position["entry_date"],
                "Exit Date": date,
                "Entry Price": round(position["entry_price"], 4),
                "Exit Price": round(float(exit_price), 4),
                "Shares": round(float(position["shares"]), 2),
                "PnL": round(float(pnl), 2),
                "Result": "Win" if pnl > 0 else "Loss",
                "Exit Reason": exit_reason,
                "Entry Idx": int(position["entry_idx"]),
                "Exit Idx": int(i),
            }
        )
        position = None

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df = (
            equity_df.drop_duplicates(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
    return trades_df, equity_df


def compute_today_signal(df: pd.DataFrame) -> dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    close = float(last["Close"])
    midline = float(last["MIDLINE"])
    upper_env = float(last["UPPER_ENV"])
    lower_env = float(last["LOWER_ENV"])
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0.0

    trend_state = "bullish" if bool(last["TREND_UP"]) else "bearish" if bool(last["TREND_DOWN"]) else "neutral"
    expansion_state = "expanding" if bool(last["ENV_EXPANDING"]) else "compressing"
    momentum_state = (
        "strong"
        if bool(last["GATOR_EXPANDING"])
        else "weakening"
        if bool(last["GATOR_SHRINKING"])
        else "flat"
    )

    trend_strength = abs(close - midline) / max(close, 1e-9)
    momentum_1 = (close - float(prev["Close"])) / max(float(prev["Close"]), 1e-9)
    gator_strength = float(last["GATOR_TOTAL"]) / max(close, 1e-9)

    score = 0.0
    score += 9.0 * trend_strength
    score += 12.0 * momentum_1
    score += 15.0 * gator_strength

    if trend_state == "bullish":
        score += 0.25
    elif trend_state == "bearish":
        score -= 0.25

    if bool(last["ENV_EXPANDING"]):
        score += 0.15
    if bool(last["GATOR_SHRINKING"]):
        score -= 0.18

    prob_up = sigmoid(score)
    prob_down = 1.0 - prob_up

    if bool(last["LONG_SIGNAL"]):
        signal = "BUY EXPANSION"
    elif bool(last["SHORT_SIGNAL"]):
        signal = "SHORT EXPANSION"
    elif bool(last["EXIT_LONG_SIGNAL"]):
        signal = "EXIT LONG"
    elif bool(last["EXIT_SHORT_SIGNAL"]):
        signal = "EXIT SHORT"
    else:
        signal = "WAIT"

    confidence_raw = abs(prob_up - 0.5) * 2.0
    confidence = "high" if confidence_raw >= 0.65 else ("medium" if confidence_raw >= 0.35 else "low")

    return {
        "trend": trend_state,
        "expansion": expansion_state,
        "momentum": momentum_state,
        "signal": signal,
        "confidence": confidence,
        "next_move_probability_up": round(prob_up, 2),
        "next_move_probability_down": round(prob_down, 2),
        "expected_range": [round(close - atr, 2), round(close + atr, 2)],
        "last_close": round(close, 2),
        "midline": round(midline, 2),
        "upper_env": round(upper_env, 2),
        "lower_env": round(lower_env, 2),
        "atr": round(atr, 2),
        "trend_strength_pct": int(min(max(trend_strength * 1200, 0), 100)),
        "momentum_pct": int(min(max(abs(momentum_1) * 5000, 0), 100)),
        "risk_level_pct": int(min(max((atr / max(close, 1e-9)) * 1200, 0), 100)),
    }


def performance_summary(trades_df: pd.DataFrame, cfg: EnvelopeGatorConfig) -> dict[str, Any]:
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "ending_capital": cfg.initial_capital,
        }

    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] <= 0]

    gross_profit = float(wins["PnL"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["PnL"].sum())) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    net_pnl = float(trades_df["PnL"].sum())

    return {
        "trades": int(len(trades_df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round((len(wins) / len(trades_df)) * 100, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_win": round(float(wins["PnL"].mean()), 2) if not wins.empty else 0.0,
        "avg_loss": round(float(losses["PnL"].mean()), 2) if not losses.empty else 0.0,
        "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else None,
        "ending_capital": round(cfg.initial_capital + net_pnl, 2),
    }


def render_meter(label: str, value_pct: int, color: str) -> None:
    value_pct = int(max(0, min(100, value_pct)))
    st.markdown(
        f"""
        <div style="margin-bottom:14px;">
            <div style="font-weight:600; margin-bottom:6px;">{label}: {value_pct}%</div>
            <div style="height:10px; background:#20252b; border-radius:999px; overflow:hidden; border:1px solid #2d3640;">
                <div style="width:{value_pct}%; height:100%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_gator_y(df: pd.DataFrame, idx: int) -> float:
    val = df.iloc[idx]["GATOR_UPPER"]
    if pd.isna(val):
        return 0.0
    return float(val)


def _safe_gator_y_lower(df: pd.DataFrame, idx: int) -> float:
    val = df.iloc[idx]["GATOR_LOWER"]
    if pd.isna(val):
        return 0.0
    return float(val)


def build_trade_chart(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    max_labels: int = 12,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.24, 0.18],
        subplot_titles=(
            "Price + Envelope + Long / Short Signals",
            "Gator Oscillator + Same Trade Signals",
            "Stock Price Overlay",
        ),
    )

    # ---------------- Row 1: Candles + envelope ----------------
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#00d084",
            decreasing_line_color="#ff6377",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["UPPER_ENV"],
            mode="lines",
            name="Upper Envelope",
            line=dict(color="#cc4452", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["LOWER_ENV"],
            mode="lines",
            name="Lower Envelope",
            line=dict(color="#2d8f49", width=1.5),
            fill="tonexty",
            fillcolor="rgba(73,160,120,0.10)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MIDLINE"],
            mode="lines",
            name="Midline",
            line=dict(color="#4c6faf", width=1.2),
        ),
        row=1,
        col=1,
    )

    # ---------------- Legend helper traces ----------------
    helper_traces = [
        go.Scatter(
            x=[None], y=[None], mode="markers", name="Enter Buy",
            marker=dict(symbol="triangle-up", size=14, color="#00d084", line=dict(width=2, color="black"))
        ),
        go.Scatter(
            x=[None], y=[None], mode="markers", name="Enter Short",
            marker=dict(symbol="triangle-down", size=14, color="#ff6377", line=dict(width=2, color="black"))
        ),
        go.Scatter(
            x=[None], y=[None], mode="markers", name="Exit Buy",
            marker=dict(symbol="circle", size=12, color="#ffd166", line=dict(width=2, color="black"))
        ),
        go.Scatter(
            x=[None], y=[None], mode="markers", name="Exit Short",
            marker=dict(symbol="circle", size=12, color="#9d8cff", line=dict(width=2, color="black"))
        ),
        go.Scatter(
            x=[None], y=[None], mode="lines", name="Winning Trade",
            line=dict(color="#00d084", width=2, dash="dash")
        ),
        go.Scatter(
            x=[None], y=[None], mode="lines", name="Losing Trade",
            line=dict(color="#ff6377", width=2, dash="dash")
        ),
    ]
    for tr in helper_traces:
        fig.add_trace(tr, row=1, col=1)

    # ---------------- Row 2: Gator ----------------
    gator_colors_upper = np.where(
        df["GATOR_UPPER"] > df["GATOR_UPPER"].shift(1),
        "#2d8f49",
        "#a33b3b",
    )
    gator_colors_lower = np.where(
        df["GATOR_LOWER"].abs() > df["GATOR_LOWER"].abs().shift(1),
        "#2d8f49",
        "#a33b3b",
    )

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["GATOR_UPPER"],
            name="Gator Upper",
            marker_color=gator_colors_upper,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["GATOR_LOWER"],
            name="Gator Lower",
            marker_color=gator_colors_lower,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ---------------- Trade plotting on BOTH price + gator ----------------
    plot_trades = trades_df.head(max_labels).copy() if not trades_df.empty else pd.DataFrame()

    for _, trade in plot_trades.iterrows():
        entry_idx = int(trade["Entry Idx"])
        exit_idx = int(trade["Exit Idx"])

        x1 = df.iloc[entry_idx]["Date"]
        x2 = df.iloc[exit_idx]["Date"]
        y1 = float(trade["Entry Price"])
        y2 = float(trade["Exit Price"])
        pnl = float(trade["PnL"])

        connector_color = "#00d084" if pnl > 0 else "#ff6377"
        is_buy = trade["Type"] == "BUY"
        exit_color = "#ffd166" if is_buy else "#9d8cff"

        # ----- PRICE CHART -----
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color=connector_color, width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[y1],
                mode="markers",
                hovertemplate="Entry<br>%{x}<br>Price: $%{y:.2f}<extra></extra>",
                marker=dict(
                    symbol="triangle-up" if is_buy else "triangle-down",
                    size=18,
                    color="#00d084" if is_buy else "#ff6377",
                    line=dict(width=2, color="black"),
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x2],
                y=[y2],
                mode="markers",
                hovertemplate=f"Exit<br>%{{x}}<br>Price: $%{{y:.2f}}<br>PnL: {pnl:.2f}<extra></extra>",
                marker=dict(
                    symbol="circle",
                    size=15,
                    color=exit_color,
                    line=dict(width=2, color="black"),
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # ----- GATOR CHART -----
        entry_gator_y = _safe_gator_y(df, entry_idx) if is_buy else _safe_gator_y_lower(df, entry_idx)
        exit_gator_y = _safe_gator_y(df, exit_idx) if is_buy else _safe_gator_y_lower(df, exit_idx)

        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[entry_gator_y, exit_gator_y],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color=connector_color, width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[entry_gator_y],
                mode="markers",
                hovertemplate="Entry<br>%{x}<br>Gator: %{y:.4f}<extra></extra>",
                marker=dict(
                    symbol="triangle-up" if is_buy else "triangle-down",
                    size=16,
                    color="#00d084" if is_buy else "#ff6377",
                    line=dict(width=2, color="black"),
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x2],
                y=[exit_gator_y],
                mode="markers",
                hovertemplate=f"Exit<br>%{{x}}<br>Gator: %{{y:.4f}}<br>PnL: {pnl:.2f}<extra></extra>",
                marker=dict(
                    symbol="circle",
                    size=14,
                    color=exit_color,
                    line=dict(width=2, color="black"),
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # ---------------- Row 3: stock price again ----------------
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Stock Price",
            line=dict(color="#67b7ff", width=2.4),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MIDLINE"],
            mode="lines",
            name="Midline Overlay",
            line=dict(color="#9d8cff", width=1.5, dash="dot"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=1040,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=12, r=12, t=90, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.15,
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Gator", row=2, col=1)
    fig.update_yaxes(title_text="Stock Price", row=3, col=1)

    return fig


def build_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity_df.empty:
        fig.add_trace(
            go.Scatter(
                x=equity_df["Date"],
                y=equity_df["Equity"],
                mode="lines",
                name="Equity",
                line=dict(color="#9d8cff", width=2.4),
            )
        )
    fig.update_layout(
        title="Equity Curve",
        height=320,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=12, r=12, t=48, b=12),
        xaxis_title="Date",
        yaxis_title="Equity",
    )
    return fig


@_fragment
def render_tab_envelope_gator_signals(symbol: str) -> None:
    target_symbol = (symbol or "AAPL").upper().strip()
    st.subheader("Envelope Gator Signals")
    st.caption(
        "Yahoo price history + Envelope/Gator signal engine with long/short markers, "
        "same signals on Gator graph, stock price in recommendations, today bias panel, and backtest."
    )

    c1, c2 = st.columns(2)
    with c1:
        range_label = st.selectbox(
            "History",
            options=list(RANGE_TO_YAHOO.keys()),
            index=1,
            key=f"eg_range_{target_symbol}",
        )
    with c2:
        valid_intervals = RANGE_INTERVALS[range_label]
        interval = st.selectbox(
            "Interval",
            options=valid_intervals,
            index=valid_intervals.index(DEFAULT_INTERVAL[range_label]),
            key=f"eg_interval_{target_symbol}_{range_label}",
        )

    with st.expander("Strategy Parameters", expanded=False):
        a1, a2, a3 = st.columns(3)
        envelope_ma_period = a1.number_input("Envelope MA Period", 5, 100, 20, 1, key=f"eg_ma_{target_symbol}")
        envelope_atr_period = a2.number_input("Envelope ATR Period", 5, 50, 14, 1, key=f"eg_atr_{target_symbol}")
        envelope_atr_mult = a3.number_input("Envelope ATR Mult", 0.2, 10.0, 1.25, 0.05, key=f"eg_env_mult_{target_symbol}")

        b1, b2, b3 = st.columns(3)
        jaw_period = b1.number_input("Jaw Period", 3, 50, 13, 1, key=f"eg_jaw_{target_symbol}")
        teeth_period = b2.number_input("Teeth Period", 3, 50, 8, 1, key=f"eg_teeth_{target_symbol}")
        lips_period = b3.number_input("Lips Period", 2, 50, 5, 1, key=f"eg_lips_{target_symbol}")

        c1p, c2p, c3p = st.columns(3)
        stop_atr_mult = c1p.number_input("Stop ATR Mult", 0.5, 10.0, 1.3, 0.1, key=f"eg_stop_{target_symbol}")
        target_atr_mult = c2p.number_input("Target ATR Mult", 0.5, 10.0, 2.0, 0.1, key=f"eg_target_{target_symbol}")
        max_labels = c3p.number_input("Max Trade Markers", 2, 30, 12, 1, key=f"eg_labels_{target_symbol}")

        d1, d2 = st.columns(2)
        initial_capital = d1.number_input("Initial Capital", 1000.0, 100000000.0, 100000.0, 1000.0, key=f"eg_capital_{target_symbol}")
        risk_per_trade_pct = d2.number_input("Risk Per Trade %", 0.1, 10.0, 1.0, 0.1, key=f"eg_risk_{target_symbol}")

    cfg = EnvelopeGatorConfig(
        envelope_ma_period=int(envelope_ma_period),
        envelope_atr_period=int(envelope_atr_period),
        envelope_atr_mult=float(envelope_atr_mult),
        jaw_period=int(jaw_period),
        teeth_period=int(teeth_period),
        lips_period=int(lips_period),
        stop_atr_mult=float(stop_atr_mult),
        target_atr_mult=float(target_atr_mult),
        initial_capital=float(initial_capital),
        risk_per_trade_pct=float(risk_per_trade_pct),
    )

    try:
        payload = _fetch_chart_payload(
            symbol=target_symbol,
            range_key=RANGE_TO_YAHOO[range_label],
            interval=interval,
        )
        df = _prepare_price_df(payload)
    except Exception as exc:
        st.error(f"Unable to load Yahoo history: {exc}")
        return

    if df.empty:
        st.warning("Yahoo returned no price rows for this selection.")
        return

    min_needed = max(
        cfg.envelope_ma_period,
        cfg.envelope_atr_period,
        cfg.jaw_period,
        cfg.teeth_period,
        cfg.lips_period,
    ) + 8

    if len(df) < min_needed:
        st.warning(
            f"Not enough rows for current parameters ({len(df)} rows, need about {min_needed}+). "
            "Increase history or reduce lookbacks."
        )
        return

    strategy_df = compute_indicators(df, cfg)
    trades_df, equity_df = run_backtest(strategy_df, cfg)
    today = compute_today_signal(strategy_df)
    perf = performance_summary(trades_df, cfg)

    st.markdown("## Today Signal")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trend", today["trend"].upper())
    m2.metric("Signal", today["signal"])
    m3.metric("Confidence", today["confidence"].upper())
    m4.metric("Close", f'{today["last_close"]:.2f}')

    left, right = st.columns([1.15, 1.85])
    with left:
        st.markdown("### Visual Meters")
        render_meter("Trend Strength", today["trend_strength_pct"], "#0A84FF")
        render_meter("Momentum", today["momentum_pct"], "#00d084")
        render_meter("Risk Level", today["risk_level_pct"], "#ff6377")

    with right:
        st.markdown("### Today Read")
        st.write(f"**Stock Price:** ${today['last_close']:.2f}")
        st.write(f"**Midline:** {today['midline']}")
        st.write(f"**Upper Envelope:** {today['upper_env']}")
        st.write(f"**Lower Envelope:** {today['lower_env']}")
        st.write(f"**ATR:** {today['atr']}")

        if today["signal"] == "BUY EXPANSION":
            st.success("Bullish expansion. Wait for a controlled pullback toward the midline or lower half of the envelope.")
        elif today["signal"] == "SHORT EXPANSION":
            st.warning("Bearish expansion. Wait for a rally toward the midline or upper half of the envelope.")
        elif today["signal"] == "EXIT LONG":
            st.info("Long momentum is fading. Avoid fresh longs here and protect profits.")
        elif today["signal"] == "EXIT SHORT":
            st.info("Short momentum is fading. Avoid fresh shorts here and protect profits.")
        else:
            st.info("No clean edge right now. Waiting is better than forcing a trade.")

        st.json(
            {
                "stock_price": today["last_close"],
                "trend": today["trend"],
                "expansion": today["expansion"],
                "momentum": today["momentum"],
                "next_move_probability_up": today["next_move_probability_up"],
                "next_move_probability_down": today["next_move_probability_down"],
                "expected_range": today["expected_range"],
                "midline": today["midline"],
                "upper_env": today["upper_env"],
                "lower_env": today["lower_env"],
                "atr": today["atr"],
            },
            expanded=True,
        )

    st.markdown("## Charts")
    st.plotly_chart(
        build_trade_chart(strategy_df, trades_df, max_labels=int(max_labels)),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"eg_trade_chart_{target_symbol}_{range_label}_{interval}",
    )
    st.plotly_chart(
        build_equity_chart(equity_df),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
        key=f"eg_equity_chart_{target_symbol}_{range_label}_{interval}",
    )

    st.markdown("## Backtest Summary")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Trades", perf["trades"])
    s2.metric("Win Rate", f'{perf["win_rate"]}%')
    s3.metric("Net PnL", f'{perf["net_pnl"]:.2f}')
    s4.metric("Avg Win", f'{perf["avg_win"]:.2f}')
    s5.metric("Avg Loss", f'{perf["avg_loss"]:.2f}')
    s6.metric("Ending Capital", f'{perf["ending_capital"]:.2f}')
    st.caption(f"Profit Factor: {perf['profit_factor'] if perf['profit_factor'] is not None else 'Infinity'}")

    st.markdown("## Trade Table")
    if trades_df.empty:
        st.info("No trades were generated with the current settings.")
    else:
        display_df = trades_df.copy()
        display_df["Entry Date"] = pd.to_datetime(display_df["Entry Date"]).dt.strftime("%Y-%m-%d")
        display_df["Exit Date"] = pd.to_datetime(display_df["Exit Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(
            display_df[
                [
                    "Type",
                    "Entry Date",
                    "Exit Date",
                    "Entry Price",
                    "Exit Price",
                    "Shares",
                    "PnL",
                    "Result",
                    "Exit Reason",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Show latest indicator rows"):
        latest = strategy_df.tail(12).copy()
        latest["Date"] = latest["Date"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            latest[
                [
                    "Date",
                    "Close",
                    "MIDLINE",
                    "UPPER_ENV",
                    "LOWER_ENV",
                    "ATR",
                    "GATOR_UPPER",
                    "GATOR_LOWER",
                    "LONG_SIGNAL",
                    "SHORT_SIGNAL",
                    "EXIT_LONG_SIGNAL",
                    "EXIT_SHORT_SIGNAL",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )