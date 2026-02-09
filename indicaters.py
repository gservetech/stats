from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Optional: if you installed finnhub-python
try:
    import get_volume
    HAS_FINNHUB = True
except Exception:
    HAS_FINNHUB = False


# =========================
# DATA FETCH
# =========================

def fetch_finnhub_candles(symbol: str, resolution: str = "D", lookback_days: int = 240) -> pd.DataFrame:
    if not HAS_FINNHUB:
        raise RuntimeError("finnhub-python not installed. Run: pip install finnhub-python")

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FINNHUB_API_KEY in .env")

    client = finnhub.Client(api_key=api_key)

    end = int(time.time())
    start = end - (lookback_days * 24 * 60 * 60)

    res = client.stock_candles(symbol.upper(), resolution, start, end)

    if not isinstance(res, dict) or res.get("s") != "ok":
        raise RuntimeError(f"Finnhub returned non-ok: {res}")

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


def fetch_yfinance_candles(symbol: str, lookback_days: int = 240) -> pd.DataFrame:
    import yfinance as yf

    period = f"{max(lookback_days, 60)}d"
    df = yf.download(symbol.upper(), period=period, interval="1d", auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index.name = "Date"
    return out


# =========================
# INDICATORS
# =========================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # VWAP (cumulative)
    out["TP"] = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["VWAP"] = (out["TP"] * out["Volume"]).cumsum() / out["Volume"].cumsum().replace(0, np.nan)

    # OBV
    out["OBV"] = (np.sign(out["Close"].diff()) * out["Volume"]).fillna(0).cumsum()

    # ADL (extra info)
    hl = (out["High"] - out["Low"]).replace(0, np.nan)
    out["MFM"] = ((out["Close"] - out["Low"]) - (out["High"] - out["Close"])) / hl
    out["ADL"] = (out["MFM"].fillna(0) * out["Volume"]).cumsum()

    return out


# =========================
# SIGNAL LOGIC (TREND-FOLLOWING, CONSISTENT)
# =========================

def add_acc_dist_and_signals(
    df: pd.DataFrame,
    confirm_days: int = 2,
    show_tags_every: int = 3,
) -> pd.DataFrame:
    out = df.copy()

    close_chg = out["Close"].diff()
    obv_chg = out["OBV"].diff()

    # Pressure days
    out["ACC"] = (close_chg > 0) & (obv_chg > 0)      # buying pressure
    out["DIST"] = (close_chg < 0) & (obv_chg < 0)     # selling pressure

    # Confirmed streak strength
    out["ACC_STR"] = out["ACC"].rolling(confirm_days).sum()
    out["DIST_STR"] = out["DIST"].rolling(confirm_days).sum()

    # VWAP slope filter (helps reduce whipsaws)
    out["VWAP_SLOPE"] = out["VWAP"].diff()

    # BUY only if today is ACC + confirmed + above VWAP + VWAP rising
    out["BUY"] = (
        out["ACC"]
        & (out["ACC_STR"] >= confirm_days)
        & (out["Close"] > out["VWAP"])
        & (out["VWAP_SLOPE"] > 0)
    )

    # SELL only if today is DIST + confirmed + below VWAP + VWAP falling
    out["SELL"] = (
        out["DIST"]
        & (out["DIST_STR"] >= confirm_days)
        & (out["Close"] < out["VWAP"])
        & (out["VWAP_SLOPE"] < 0)
    )

    # Mark only first day of signal streak (no repeated arrows)
    out["BUY_MARK"] = out["BUY"] & (~out["BUY"].shift(1).fillna(False))
    out["SELL_MARK"] = out["SELL"] & (~out["SELL"].shift(1).fillna(False))

    # Density control for ACC/DIST labels
    out["TAG_OK"] = False
    if show_tags_every and show_tags_every > 0:
        idx = np.arange(len(out))
        out.loc[out.index[idx % show_tags_every == 0], "TAG_OK"] = True

    return out


# =========================
# PLOTTING (PRICE + VWAP, VOLUME, OBV + RIGHT PANEL INFO)
# =========================

def plot_chart(df: pd.DataFrame, symbol: str, source: str) -> None:
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[5.2, 1.8],   # left charts, right info
        height_ratios=[2, 1, 1],   # price, volume, obv
        wspace=0.15,
        hspace=0.25
    )

    ax1 = fig.add_subplot(gs[0, 0])                 # price + vwap
    axv = fig.add_subplot(gs[1, 0], sharex=ax1)     # volume
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)     # obv
    axg = fig.add_subplot(gs[:, 1])                 # info panel
    axg.axis("off")

    # ---- PRICE + VWAP ----
    ax1.plot(df.index, df["Close"], label="Price", color="black", alpha=0.6, linewidth=1.6, zorder=2)
    ax1.plot(df.index, df["VWAP"], label="VWAP", color="blue", lw=2.6, ls="--", zorder=5)

    # ---- BUY / SELL ----
    buys = df[df["BUY_MARK"]]
    sells = df[df["SELL_MARK"]]

    ax1.scatter(buys.index, buys["Close"], marker="^", s=160, edgecolors="black", linewidths=1.2, label="BUY", zorder=7)
    ax1.scatter(sells.index, sells["Close"], marker="v", s=160, edgecolors="black", linewidths=1.2, label="SELL", zorder=7)

    # ---- ACC / DIST small tags ----
    for i in range(1, len(df)):
        if not bool(df["TAG_OK"].iloc[i]):
            continue
        if bool(df["ACC"].iloc[i]):
            ax1.text(df.index[i], df["Close"].iloc[i], "ACC", fontsize=8, fontweight="bold", zorder=8)
        elif bool(df["DIST"].iloc[i]):
            ax1.text(df.index[i], df["Close"].iloc[i], "DIST", fontsize=8, fontweight="bold", zorder=8)

    ax1.set_title(f"{symbol.upper()} — VWAP + Volume + OBV BUY/SELL System")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper left")

    # ---- VOLUME (colored by up/down close) ----
    vol_colors = np.where(df["Close"].diff() >= 0, "#2ecc71", "#e74c3c")
    axv.bar(df.index, df["Volume"], color=vol_colors, alpha=0.7)
    axv.set_title("Volume")
    axv.grid(alpha=0.2)

    # ---- OBV ----
    ax2.plot(df.index, df["OBV"], label="OBV", color="green", linewidth=1.6)
    ax2.set_title("OBV (Volume Momentum)")
    ax2.grid(alpha=0.2)

    # ---- INFO PANEL ----
    last = df.iloc[-1]

    info = (
        "LATEST BAR\n"
        "----------\n"
        f"Symbol     : {symbol.upper()}\n"
        f"Date       : {df.index[-1].date()}\n"
        f"Close      : {float(last['Close']):.2f}\n"
        f"VWAP       : {float(last['VWAP']):.2f}\n"
        f"VWAP slope : {float(last['VWAP_SLOPE'] or 0):.4f}\n"
        f"ACC_STR    : {int(last['ACC_STR'] or 0)}\n"
        f"DIST_STR   : {int(last['DIST_STR'] or 0)}\n"
        f"BUY now    : {bool(last['BUY'])}\n"
        f"SELL now   : {bool(last['SELL'])}\n"
        f"Source     : {source}\n"
        "\n"
        "GUIDE\n"
        "-----\n"
        "ACC  = Close↑ & OBV↑ (buy pressure)\n"
        "DIST = Close↓ & OBV↓ (sell pressure)\n\n"
        "BUY  ▲ = ACC today + confirmed\n"
        "        + Close > VWAP + VWAP rising\n\n"
        "SELL ▼ = DIST today + confirmed\n"
        "        + Close < VWAP + VWAP falling\n\n"
        "VOLUME COLORS\n"
        "-------------\n"
        "Green volume = Price closed UP vs prior day\n"
        "Red volume   = Price closed DOWN vs prior day\n"
        "(Helps show who controlled the day)\n\n"
        "TUNING\n"
        "------\n"
        "confirm_days=1 → more signals\n"
        "confirm_days=3 → fewer/stronger\n"
    )

    axg.text(
        0.02, 0.98, info,
        va="top",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.96, edgecolor="gray")
    )

    plt.show()


# =========================
# RUN
# =========================

def run(
    symbol: str = "MU",
    lookback_days: int = 240,
    resolution: str = "D",
    confirm_days: int = 2,
    show_tags_every: int = 3
) -> pd.DataFrame:

    # Try Finnhub; if blocked (403), fallback to yfinance
    source = "Finnhub"
    try:
        df = fetch_finnhub_candles(symbol=symbol, resolution=resolution, lookback_days=lookback_days)
    except Exception as e:
        source = f"yfinance (Finnhub blocked: {type(e).__name__})"
        df = fetch_yfinance_candles(symbol=symbol, lookback_days=lookback_days)

    df = add_indicators(df)
    df = add_acc_dist_and_signals(df, confirm_days=confirm_days, show_tags_every=show_tags_every)

    # Console quick view
    last = df.iloc[-1]
    print("\nLatest bar:")
    print(f"Symbol     : {symbol.upper()}")
    print(f"Source     : {source}")
    print(f"Date       : {df.index[-1].date()}")
    print(f"Close      : {float(last['Close']):.2f}")
    print(f"VWAP       : {float(last['VWAP']):.2f}")
    print(f"ACC_STR    : {int(last['ACC_STR'] or 0)}")
    print(f"DIST_STR   : {int(last['DIST_STR'] or 0)}")
    print(f"BUY now    : {bool(last['BUY'])}")
    print(f"SELL now   : {bool(last['SELL'])}")

    plot_chart(df, symbol, source)
    return df


if __name__ == "__main__":
    # Examples:
    # run(symbol="AAPL")
    # run(symbol="NVDA", confirm_days=3)
    run(symbol="MU", lookback_days=240, resolution="D", confirm_days=2, show_tags_every=3)
