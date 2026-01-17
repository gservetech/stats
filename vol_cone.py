from __future__ import annotations

"""
MU Volatility Cone + VWAP/OBV + “SUGGESTION” printed ON THE FIGURE (outside the plot area)

What you asked:
- Put the suggestion (BUY/SELL based on price vs VWAP/OBV + cone regime) directly on the graph,
  without hiding the cone lines.

How it works:
1) Cone decides the TRADE TYPE:
   - VOL CHEAP (<= p25): prefer BUYING options
   - VOL EXPENSIVE (>= p75): prefer SELLING premium
   - VOL NORMAL: selective / wait

2) VWAP/OBV decides DIRECTION:
   - Bullish bias: Close > VWAP and VWAP_SLOPE > 0 and OBV_SLOPE > 0
   - Bearish bias: Close < VWAP and VWAP_SLOPE < 0 and OBV_SLOPE < 0
   - Otherwise: neutral

3) Final "SUGGESTION" text is shown in the right panel:
   - “BUY CALL / CALL DEBIT SPREAD” or
   - “SELL PUT SPREAD / SELL PREMIUM” or
   - “WAIT / NEUTRAL”

Data:
- Finnhub first; if 403 blocked, fallback to yfinance automatically.

Install:
  pip install requests python-dotenv pandas numpy matplotlib yfinance
"""

import os
import time
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()

TRADING_DAYS = 252
FINNHUB_BASE = "https://finnhub.io/api/v1"

SYMBOL = "MU"
YEARS = 5
WINDOWS = [5, 10, 20, 30, 60, 90, 120]
PCTS = [5, 25, 50, 75, 95]


@dataclass
class Suggestion:
    vol_regime: str
    trade_type: str
    direction: str
    final: str


def _token() -> Optional[str]:
    tok = os.getenv("FINNHUB_API_KEY")
    return tok.strip() if tok else None


# =========================
# DATA FETCH
# =========================

def fetch_finnhub_daily_candles(symbol: str, lookback_days: int) -> pd.DataFrame:
    tok = _token()
    if not tok:
        raise RuntimeError("Missing FINNHUB_API_KEY in .env")

    end = int(time.time())
    start = end - int(lookback_days * 24 * 60 * 60)

    params = {
        "symbol": symbol.upper(),
        "resolution": "D",
        "from": start,
        "to": end,
        "token": tok,
    }
    r = requests.get(f"{FINNHUB_BASE}/stock/candle", params=params, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"Finnhub candles HTTP {r.status_code}: {r.text[:200]}")

    res = r.json()
    if not isinstance(res, dict) or res.get("s") != "ok":
        raise RuntimeError(f"Finnhub candles non-ok: {res}")

    df = pd.DataFrame(
        {
            "Open": res["o"],
            "High": res["h"],
            "Low": res["l"],
            "Close": res["c"],
            "Volume": res["v"],
        },
        index=pd.to_datetime(res["t"], unit="s"),
    ).sort_index()
    df.index.name = "Date"
    return df


def fetch_yfinance_daily(symbol: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol.upper(), period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out


def fetch_prices(symbol: str, years: int) -> Tuple[pd.DataFrame, str]:
    lookback_days = int(years * 365.25) + 30
    try:
        return fetch_finnhub_daily_candles(symbol, lookback_days=lookback_days), "Finnhub"
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "don't have access" in msg:
            print("[WARN] Finnhub candles blocked (403). Using yfinance for prices.")
        else:
            print(f"[WARN] Finnhub failed ({type(e).__name__}). Using yfinance for prices.")
        return fetch_yfinance_daily(symbol, years=years), "yfinance"


# =========================
# INDICATORS
# =========================

def add_vwap_obv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TP"] = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["VWAP"] = (out["TP"] * out["Volume"]).cumsum() / out["Volume"].cumsum().replace(0, np.nan)
    out["OBV"] = (np.sign(out["Close"].diff()) * out["Volume"]).fillna(0).cumsum()
    out["VWAP_SLOPE"] = out["VWAP"].diff()
    out["OBV_SLOPE"] = out["OBV"].diff()
    return out


# =========================
# VOL CONE
# =========================

def annualized_realized_vol(close: pd.Series, window_days: int) -> pd.Series:
    logret = np.log(close.astype(float)).diff()
    return logret.rolling(window_days).std(ddof=0) * math.sqrt(TRADING_DAYS)


def build_vol_cone(close: pd.Series, windows: List[int], percentiles: List[int]) -> pd.DataFrame:
    rows = []
    for w in windows:
        rv = annualized_realized_vol(close, w).dropna()
        if rv.empty:
            continue
        row = {"window_days": w, "latest_rv": float(rv.iloc[-1])}
        for p in percentiles:
            row[f"p{p}"] = float(np.nanpercentile(rv.values, p))
        rows.append(row)
    return pd.DataFrame(rows).set_index("window_days").sort_index()


def vol_regime_for_window(latest_rv: float, row: pd.Series) -> Tuple[str, str]:
    """
    Returns (vol_regime, trade_type).
    """
    p25 = float(row.get("p25", np.nan))
    p75 = float(row.get("p75", np.nan))
    if np.isnan([p25, p75]).any():
        return "UNKNOWN", "WAIT"

    if latest_rv <= p25:
        return "VOL CHEAP (<= p25)", "BUY OPTIONS"
    if latest_rv >= p75:
        return "VOL EXPENSIVE (>= p75)", "SELL PREMIUM"
    return "VOL NORMAL (p25–p75)", "SELECTIVE / WAIT"


def direction_from_vwap_obv(close: float, vwap: float, vwap_slope: float, obv_slope: float) -> str:
    bullish = (close > vwap) and (vwap_slope > 0) and (obv_slope > 0)
    bearish = (close < vwap) and (vwap_slope < 0) and (obv_slope < 0)
    if bullish:
        return "BULLISH"
    if bearish:
        return "BEARISH"
    return "NEUTRAL"


def combine_suggestion(vol_trade_type: str, direction: str) -> str:
    """
    Translate trade type + direction into a concrete “what to do”.
    (Not financial advice; this is a rule-based suggestion display.)
    """
    if vol_trade_type == "BUY OPTIONS":
        if direction == "BULLISH":
            return "SUGGESTION: Buy CALL or CALL DEBIT SPREAD (trend up + vol cheap)"
        if direction == "BEARISH":
            return "SUGGESTION: Buy PUT or PUT DEBIT SPREAD (trend down + vol cheap)"
        return "SUGGESTION: Buy STRADDLE/STRANGLE (vol cheap but direction unclear)"
    if vol_trade_type == "SELL PREMIUM":
        if direction == "BULLISH":
            return "SUGGESTION: Sell PUT CREDIT SPREAD / CSP (bullish + vol expensive)"
        if direction == "BEARISH":
            return "SUGGESTION: Sell CALL CREDIT SPREAD / CC (bearish + vol expensive)"
        return "SUGGESTION: Iron Condor (range + vol expensive)"
    return "SUGGESTION: WAIT / NO EDGE (vol normal or direction unclear)"


# =========================
# PLOT (cone left, panel right)
# =========================

def plot_cone_with_suggestion(symbol: str, cone: pd.DataFrame, source: str, df: pd.DataFrame,
                              focus_window: int, sug: Suggestion) -> None:
    import matplotlib.gridspec as gridspec

    x = cone.index.values.astype(int)

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5.2, 1.8], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])
    axp.axis("off")

    # Cone lines
    pct_cols = [c for c in cone.columns if c.startswith("p")]
    for c in pct_cols:
        ax.plot(x, cone[c].values, label=c)

    # latest RV
    ax.plot(x, cone["latest_rv"].values, label="latest_rv", linewidth=2.6)

    # Focus dot
    y = float(cone.loc[focus_window, "latest_rv"])
    ax.scatter([focus_window], [y], s=140, marker="o", zorder=6, label=f"Focus RV @ {focus_window}d")
    ax.annotate(f"{focus_window}d RV={y:.2%}", (focus_window, y), textcoords="offset points", xytext=(10, 10))

    ax.set_title(f"{symbol.upper()} Volatility Cone (Realized Vol) — data: {source}")
    ax.set_xlabel("Lookback window (trading days)")
    ax.set_ylabel("Annualized volatility")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)

    # Panel text (includes suggestion)
    last = df.iloc[-1]
    close = float(last["Close"])
    vwap = float(last["VWAP"])
    vwap_slope = float(last["VWAP_SLOPE"] if pd.notna(last["VWAP_SLOPE"]) else 0.0)
    obv_slope = float(last["OBV_SLOPE"] if pd.notna(last["OBV_SLOPE"]) else 0.0)

    panel = (
        "TODAY'S SIGNALS\n"
        "--------------\n"
        f"Symbol      : {symbol.upper()}\n"
        f"Date        : {df.index[-1].date()}\n"
        f"Close       : {close:.2f}\n"
        f"VWAP        : {vwap:.2f}\n"
        f"VWAP slope  : {vwap_slope:.4f}\n"
        f"OBV slope   : {obv_slope:.2f}\n"
        f"Focus win   : {focus_window}d\n"
        f"Focus RV    : {y:.2%}\n\n"
        "VOL CONE READ\n"
        "------------\n"
        f"{sug.vol_regime}\n"
        f"Trade type  : {sug.trade_type}\n\n"
        "DIRECTION READ\n"
        "-------------\n"
        f"{sug.direction}\n\n"
        "FINAL\n"
        "-----\n"
        f"{sug.final}\n\n"
        "HOW TO READ\n"
        "----------\n"
        "• p5/p25/p50/p75/p95 are historical RV bands\n"
        "• latest_rv shows today's RV for each horizon\n"
        "• If RV <= p25: volatility is cheap -> prefer buying options\n"
        "• If RV >= p75: volatility is rich -> prefer selling premium\n"
        "• Direction uses VWAP/OBV: above+up= bullish, below+down=bearish\n"
    )

    axp.text(
        0.02, 0.98, panel,
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.96, edgecolor="gray")
    )

    plt.show()


# =========================
# RUN
# =========================

def run(symbol: str = SYMBOL, years: int = YEARS) -> None:
    df, source = fetch_prices(symbol, years=years)
    df = add_vwap_obv(df)

    close_series = df["Close"].dropna()
    cone = build_vol_cone(close_series, WINDOWS, PCTS)

    focus_window = 20 if 20 in cone.index else int(cone.index.values[0])
    focus_rv = float(cone.loc[focus_window, "latest_rv"])

    vol_regime, trade_type = vol_regime_for_window(focus_rv, cone.loc[focus_window])

    last = df.iloc[-1]
    close = float(last["Close"])
    vwap = float(last["VWAP"])
    vwap_slope = float(last["VWAP_SLOPE"] if pd.notna(last["VWAP_SLOPE"]) else 0.0)
    obv_slope = float(last["OBV_SLOPE"] if pd.notna(last["OBV_SLOPE"]) else 0.0)

    direction = direction_from_vwap_obv(close, vwap, vwap_slope, obv_slope)
    final = combine_suggestion(trade_type, direction)

    sug = Suggestion(
        vol_regime=vol_regime,
        trade_type=trade_type,
        direction=direction,
        final=final
    )

    # Console summary
    print("\n" + "=" * 90)
    print(f"{symbol.upper()} — TODAY SUGGESTION")
    print("=" * 90)
    print(f"Data source : {source}")
    print(f"Date        : {df.index[-1].date()}")
    print(f"Close       : {close:.2f}")
    print(f"VWAP        : {vwap:.2f} | slope {vwap_slope:.4f}")
    print(f"OBV slope   : {obv_slope:.2f}")
    print(f"Focus RV    : {focus_rv:.2%} @ {focus_window}d")
    print(f"Vol regime  : {vol_regime}")
    print(f"Trade type  : {trade_type}")
    print(f"Direction   : {direction}")
    print(final)
    print("=" * 90 + "\n")

    plot_cone_with_suggestion(symbol, cone, source, df, focus_window, sug)


if __name__ == "__main__":
    run(symbol="AAPL", years=5)
