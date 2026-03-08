import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
class StrategyConfig:
    fast_ema: int = 20
    slow_ema: int = 50
    atr_period: int = 14
    rsi_period: int = 14
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.0
    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 1.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _prepare_price_df(chart_payload: dict[str, Any]) -> pd.DataFrame:
    raw_df, _ = _build_chart_df(chart_payload)
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df[["datetime", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "datetime": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    out = df.copy()

    out["EMA_FAST"] = out["Close"].ewm(span=cfg.fast_ema, adjust=False).mean()
    out["EMA_SLOW"] = out["Close"].ewm(span=cfg.slow_ema, adjust=False).mean()

    prev_close = out["Close"].shift(1)
    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - prev_close).abs()
    tr3 = (out["Low"] - prev_close).abs()
    out["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR"] = out["TR"].rolling(cfg.atr_period).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(cfg.rsi_period).mean()
    avg_loss = loss.rolling(cfg.rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["RSI"] = out["RSI"].fillna(50.0)

    prev_ema_fast = out["EMA_FAST"].shift(1)
    out["LONG_SIGNAL"] = (
            (out["EMA_FAST"] > out["EMA_SLOW"])
            & (prev_close < prev_ema_fast)
            & (out["Close"] > out["EMA_FAST"])
    )
    out["SHORT_SIGNAL"] = (
            (out["EMA_FAST"] < out["EMA_SLOW"])
            & (prev_close > prev_ema_fast)
            & (out["Close"] < out["EMA_FAST"])
    )
    return out


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    capital = cfg.initial_capital
    position: dict[str, Any] | None = None

    start_idx = max(cfg.slow_ema, cfg.atr_period, cfg.rsi_period) + 2
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        date = row["Date"]
        equity_curve.append({"Date": date, "Equity": capital})

        if pd.isna(row["ATR"]):
            continue

        if position is None:
            risk_amount = capital * (cfg.risk_per_trade_pct / 100.0)
            if bool(row["LONG_SIGNAL"]):
                entry = float(row["Close"])
                stop = entry - cfg.stop_atr_mult * float(row["ATR"])
                target = entry + cfg.target_atr_mult * float(row["ATR"])
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
                stop = entry + cfg.stop_atr_mult * float(row["ATR"])
                target = entry - cfg.target_atr_mult * float(row["ATR"])
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
            elif row["Close"] < row["EMA_FAST"]:
                exit_price = float(row["Close"])
                exit_reason = "EMA Exit"

            if exit_price is not None:
                pnl = (exit_price - position["entry_price"]) * position["shares"]
        else:
            if row["High"] >= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "Stop"
            elif row["Low"] <= position["target"]:
                exit_price = position["target"]
                exit_reason = "Target"
            elif row["Close"] > row["EMA_FAST"]:
                exit_price = float(row["Close"])
                exit_reason = "EMA Exit"

            if exit_price is not None:
                pnl = (position["entry_price"] - exit_price) * position["shares"]

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
        equity_df = equity_df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return trades_df, equity_df


def compute_today_signal(df: pd.DataFrame) -> dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    ema_fast = float(last["EMA_FAST"])
    ema_slow = float(last["EMA_SLOW"])
    close = float(last["Close"])
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0.0
    rsi = float(last["RSI"])

    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    trend_strength = abs(ema_fast - ema_slow) / max(close, 1e-9)
    momentum_1 = (close - float(prev["Close"])) / max(float(prev["Close"]), 1e-9)
    distance_from_fast = (close - ema_fast) / max(close, 1e-9)
    atr_pct = atr / max(close, 1e-9)

    score = 0.0
    score += 5.5 * trend_strength
    score += 14.0 * momentum_1
    score += 4.0 * distance_from_fast
    if trend_up:
        score += 0.35
    if trend_down:
        score -= 0.35
    if rsi > 70:
        score -= 0.15
    elif rsi < 30:
        score += 0.15

    prob_up = sigmoid(score)
    prob_down = 1.0 - prob_up

    if trend_up and close >= ema_fast:
        signal = "BUY"
    elif trend_down and close <= ema_fast:
        signal = "SHORT"
    else:
        signal = "WAIT"

    confidence_raw = abs(prob_up - 0.5) * 2
    confidence = "high" if confidence_raw >= 0.65 else ("medium" if confidence_raw >= 0.35 else "low")
    trend = "bullish" if prob_up >= 0.55 else ("bearish" if prob_down >= 0.55 else "neutral")

    momentum_score = max(min(momentum_1 * 1000, 100), -100)
    if momentum_score > 20:
        momentum = "strong_up"
    elif momentum_score > 5:
        momentum = "moderate_up"
    elif momentum_score < -20:
        momentum = "strong_down"
    elif momentum_score < -5:
        momentum = "moderate_down"
    else:
        momentum = "flat"

    return {
        "trend": trend,
        "momentum": momentum,
        "next_move_probability_up": round(prob_up, 2),
        "next_move_probability_down": round(prob_down, 2),
        "expected_range": [round(close - atr, 2), round(close + atr, 2)],
        "signal": signal,
        "confidence": confidence,
        "trend_strength_pct": int(min(max(trend_strength * 1000, 0), 100)),
        "momentum_pct": int(min(max(abs(momentum_score), 0), 100)),
        "risk_level_pct": int(min(max(atr_pct * 1000, 0), 100)),
        "last_close": round(close, 2),
        "ema_fast": round(ema_fast, 2),
        "ema_slow": round(ema_slow, 2),
        "atr": round(atr, 2),
        "rsi": round(rsi, 2),
    }


def performance_summary(trades_df: pd.DataFrame, cfg: StrategyConfig) -> dict[str, Any]:
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


def build_trade_chart(df: pd.DataFrame, trades_df: pd.DataFrame, max_labels: int = 12) -> go.Figure:
    fig = go.Figure()

    # 1. Base Lines
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(color="#67b7ff", width=2.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_FAST"], mode="lines", name="EMA Fast",
                             line=dict(color="#00d084", width=1.7)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_SLOW"], mode="lines", name="EMA Slow",
                             line=dict(color="#ffb84d", width=1.7)))

    # 2. DUMMY TRACES FOR THE LEGEND (These stay at the top of the chart)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Enter Buy",
                             marker=dict(symbol="triangle-up", size=14, color="#00d084",
                                         line=dict(width=2, color="black"))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Enter Short",
                             marker=dict(symbol="triangle-down", size=14, color="#ff6377",
                                         line=dict(width=2, color="black"))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Exit Buy",
                             marker=dict(symbol="circle", size=12, color="#ffd166", line=dict(width=2, color="black"))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Exit Short",
                             marker=dict(symbol="circle", size=12, color="#9d8cff", line=dict(width=2, color="black"))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Winning Trade",
                             line=dict(color="#00d084", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Losing Trade",
                             line=dict(color="#ff6377", width=2, dash="dash")))

    # 3. Plot Actual Trades
    plot_trades = trades_df.head(max_labels).copy() if not trades_df.empty else pd.DataFrame()
    buy_num = 1
    sell_num = 1
    for _, trade in plot_trades.iterrows():
        x1 = df.iloc[int(trade["Entry Idx"])]["Date"]
        x2 = df.iloc[int(trade["Exit Idx"])]["Date"]
        y1 = float(trade["Entry Price"])
        y2 = float(trade["Exit Price"])
        pnl = float(trade["PnL"])

        connector_color = "#00d084" if pnl > 0 else "#ff6377"
        is_buy = trade["Type"] == "BUY"

        entry_label = f"Buy {buy_num}" if is_buy else f"Sell {sell_num}"
        exit_label = f"Exit Buy {buy_num}" if is_buy else f"Exit Sell {sell_num}"

        # Color specific to the exit type
        exit_color = "#ffd166" if is_buy else "#9d8cff"

        entry_hovertext = [f"{entry_label}"]
        exit_hovertext = [f"{exit_label}"]

        entry_customdata = [[y1, pnl]]
        exit_customdata = [[y2, pnl]]

        if is_buy:
            buy_num += 1
        else:
            sell_num += 1

        # Trade connector line
        fig.add_trace(
            go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color=connector_color, width=2, dash="dash"),
            )
        )

        # Entry Marker
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[y1],
                mode="markers",
                name=entry_label,
                customdata=entry_customdata,
                hovertext=entry_hovertext,
                hovertemplate="%{x}<br>Entry Price: $%{customdata[0]:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(
                    symbol="triangle-up" if is_buy else "triangle-down",
                    size=20,
                    color="#00d084" if is_buy else "#ff6377",
                    line=dict(width=2, color="black")
                ),
                showlegend=False,
            )
        )

        # Exit Marker
        fig.add_trace(
            go.Scatter(
                x=[x2],
                y=[y2],
                mode="markers",
                name=exit_label,
                customdata=exit_customdata,
                hovertext=exit_hovertext,
                hovertemplate="%{x}<br>Exit Price: $%{customdata[0]:.2f}<br>PnL: $%{customdata[1]:.2f}<br><b>%{hovertext}</b><extra></extra>",
                marker=dict(
                    symbol="circle",
                    size=16,
                    color=exit_color,
                    line=dict(width=2, color="black")
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Trade Chart",
        height=680,  # Made slightly taller to comfortably fit the legend
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=12, r=12, t=80, b=12),  # Expanded top margin
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)"  # Keeps the background clean behind the legend
        ),
        xaxis_title="Date",
        yaxis_title="Price",
    )
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


@_fragment
def render_tab_market_signals(symbol: str) -> None:
    target_symbol = (symbol or "AAPL").upper().strip()
    st.subheader("Market Signals")
    st.caption("Yahoo price history + EMA/ATR/RSI signal engine with backtest and today bias panel.")

    c1, c2 = st.columns(2)
    with c1:
        range_label = st.selectbox(
            "History",
            options=list(RANGE_TO_YAHOO.keys()),
            index=1,
            key=f"market_signals_range_{target_symbol}",
        )
    with c2:
        valid_intervals = RANGE_INTERVALS[range_label]
        interval = st.selectbox(
            "Interval",
            options=valid_intervals,
            index=valid_intervals.index(DEFAULT_INTERVAL[range_label]),
            key=f"market_signals_interval_{target_symbol}_{range_label}",
        )

    with st.expander("Strategy Parameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        fast_ema = p1.number_input("Fast EMA", min_value=5, max_value=100, value=20, step=1,
                                   key=f"ms_fast_{target_symbol}")
        slow_ema = p2.number_input("Slow EMA", min_value=10, max_value=200, value=50, step=1,
                                   key=f"ms_slow_{target_symbol}")
        atr_period = p3.number_input("ATR Period", min_value=5, max_value=50, value=14, step=1,
                                     key=f"ms_atr_{target_symbol}")

        p4, p5, p6 = st.columns(3)
        stop_atr_mult = p4.number_input(
            "Stop ATR Mult",
            min_value=0.5,
            max_value=10.0,
            value=1.5,
            step=0.1,
            key=f"ms_stop_{target_symbol}",
        )
        target_atr_mult = p5.number_input(
            "Target ATR Mult",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key=f"ms_target_{target_symbol}",
        )
        rsi_period = p6.number_input("RSI Period", min_value=5, max_value=50, value=14, step=1,
                                     key=f"ms_rsi_{target_symbol}")

        p7, p8, p9 = st.columns(3)
        initial_capital = p7.number_input(
            "Initial Capital",
            min_value=1000.0,
            value=100000.0,
            step=1000.0,
            key=f"ms_capital_{target_symbol}",
        )
        risk_per_trade_pct = p8.number_input(
            "Risk Per Trade %",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=f"ms_risk_{target_symbol}",
        )
        max_labels = p9.number_input(
            "Max Chart Labels",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
            key=f"ms_labels_{target_symbol}",
        )

    cfg = StrategyConfig(
        fast_ema=int(fast_ema),
        slow_ema=int(slow_ema),
        atr_period=int(atr_period),
        rsi_period=int(rsi_period),
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

    min_needed = max(cfg.slow_ema, cfg.atr_period, cfg.rsi_period) + 5
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

    c_left, c_right = st.columns([1.15, 1.85])
    with c_left:
        st.markdown("### Visual Meters")
        render_meter("Trend Strength", today["trend_strength_pct"], "#0A84FF")
        render_meter("Momentum", today["momentum_pct"], "#00d084")
        render_meter("Risk Level", today["risk_level_pct"], "#ff6377")
    with c_right:
        st.markdown("### Today Read")
        if today["signal"] == "BUY":
            st.success("Bullish condition. Prefer pullbacks toward the fast EMA before long entries.")
        elif today["signal"] == "SHORT":
            st.warning("Bearish condition. Prefer failed rallies toward the fast EMA before short entries.")
        else:
            st.info("No clean setup right now. Waiting is better than forcing a low-quality signal.")

        st.json(
            {
                "trend": today["trend"],
                "momentum": today["momentum"],
                "next_move_probability_up": today["next_move_probability_up"],
                "next_move_probability_down": today["next_move_probability_down"],
                "expected_range": today["expected_range"],
                "ema_fast": today["ema_fast"],
                "ema_slow": today["ema_slow"],
                "atr": today["atr"],
                "rsi": today["rsi"],
            },
            expanded=True,
        )

    st.markdown("## Charts")
    st.plotly_chart(
        build_trade_chart(strategy_df, trades_df, max_labels=int(max_labels)),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"market_signals_trade_chart_{target_symbol}_{range_label}_{interval}",
    )
    st.plotly_chart(
        build_equity_chart(equity_df),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
        key=f"market_signals_equity_chart_{target_symbol}_{range_label}_{interval}",
    )

    st.markdown("## Backtest Summary")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Trades", perf["trades"])
    s2.metric("Win Rate", f'{perf["win_rate"]}%')
    s3.metric("Net PnL", f'{perf["net_pnl"]:.2f}')
    s4.metric("Avg Win", f'{perf["avg_win"]:.2f}')
    s5.metric("Avg Loss", f'{perf["avg_loss"]:.2f}')
    s6.metric("Ending Capital", f'{perf["ending_capital"]:.2f}')
    st.caption(
        f"Profit Factor: {perf['profit_factor'] if perf['profit_factor'] is not None else 'Infinity'}"
    )

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
        latest = strategy_df.tail(10).copy()
        latest["Date"] = latest["Date"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            latest[
                [
                    "Date",
                    "Close",
                    "EMA_FAST",
                    "EMA_SLOW",
                    "ATR",
                    "RSI",
                    "LONG_SIGNAL",
                    "SHORT_SIGNAL",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )