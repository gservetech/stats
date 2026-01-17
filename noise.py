from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import entropy as shannon_entropy
import matplotlib.pyplot as plt


# -----------------------------
# Pure pandas/numpy indicators
# -----------------------------

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def bbands(close: pd.Series, length: int = 20, std: float = 2.0):
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    return lower, mid, upper


def cmf(high, low, close, volume, length=20):
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum().replace(0, np.nan)


def minmax_rolling(x, lookback):
    rmin = x.rolling(lookback).min()
    rmax = x.rolling(lookback).max()
    return (x - rmin) / (rmax - rmin).replace(0, np.nan)


# -----------------------------
# Market Folding (Eigen-Entropy)
# -----------------------------

def rolling_eigen_entropy(df, features, window, base=2.0):
    n = len(df)
    scores = np.full(n, np.nan)
    X = df[features].to_numpy()

    for i in range(window, n + 1):
        corr = np.corrcoef(X[i - window:i], rowvar=False)
        corr = np.nan_to_num(corr)
        w = np.abs(np.linalg.eigvalsh(corr))
        s = w.sum()
        if s > 0 and np.isfinite(s):
            scores[i - 1] = shannon_entropy(w / s, base=base)

    return pd.Series(scores, index=df.index)


def main():
    # -----------------------------
    # SETTINGS
    # -----------------------------
    ticker = "MU"
    period = "2y"
    interval = "1d"

    window_size = 50   # rolling correlation window
    smooth = 5         # smoothing for Folding_Score
    entropy_base = 2.0

    # regime quantiles
    low_q = 0.25
    high_q = 0.75

    print(f"Downloading {ticker} data ({period}, {interval}) ...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError("No data returned from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    required = {"High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    # -----------------------------
    # FEATURES
    # -----------------------------
    df["RSI"] = rsi(df["Close"]) / 100.0
    lo, mid, hi = bbands(df["Close"])
    df["BBW"] = minmax_rolling((hi - lo) / mid.replace(0, np.nan), 50)
    df["NATR"] = atr(df["High"], df["Low"], df["Close"]) / df["Close"].replace(0, np.nan)

    vslow = df["Volume"].rolling(10).mean()
    vfast = df["Volume"].rolling(5).mean()
    vol_osc = (vfast - vslow) / vslow.replace(0, np.nan)
    df["VolOsc"] = np.tanh(vol_osc)

    df["CMF"] = cmf(df["High"], df["Low"], df["Close"], df["Volume"])

    feats = ["RSI", "BBW", "NATR", "VolOsc", "CMF"]
    df = df.dropna(subset=feats).copy()

    # -----------------------------
    # FOLDING SCORE
    # -----------------------------
    df["Entropy"] = rolling_eigen_entropy(df, feats, window_size, base=entropy_base)
    df["Folding_Score"] = df["Entropy"].rolling(smooth).mean()

    # -----------------------------
    # REGIME LABELS
    # -----------------------------
    s = df["Folding_Score"].dropna()
    if s.empty:
        raise RuntimeError("Folding_Score is empty (not enough data after rolling windows).")

    loq = float(s.quantile(low_q))
    hiq = float(s.quantile(high_q))

    df["Regime"] = np.where(
        df["Folding_Score"] <= loq, "COLLAPSED",
        np.where(df["Folding_Score"] >= hiq, "COMPLEX_FOLDED", "TRANSITION")
    )

    # -----------------------------
    # LATEST (TODAY) SUMMARY
    # -----------------------------
    last = df.dropna(subset=["Folding_Score"]).iloc[-1]
    last_date = df.dropna(subset=["Folding_Score"]).index[-1]
    last_close = float(last["Close"])
    last_score = float(last["Folding_Score"])
    last_regime = str(last["Regime"])

    print("\n===== TODAY / LATEST BAR =====")
    print(f"Date         : {last_date}")
    print(f"Close        : {last_close:.2f}")
    print(f"FoldingScore : {last_score:.4f}")
    print(f"Regime       : {last_regime}")
    print("=============================\n")

    # -----------------------------
    # PLOT (chart + explanation)
    # -----------------------------
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Background regimes
    colors = {"COLLAPSED": "#ff4d4d", "TRANSITION": "#cccccc", "COMPLEX_FOLDED": "#66cc66"}
    last_r = None
    start = None
    for i in range(len(df)):
        r = df["Regime"].iloc[i]
        if r != last_r:
            if last_r is not None:
                ax[0].axvspan(df.index[start], df.index[i], color=colors.get(last_r, "#ffffff"), alpha=0.15)
            start = i
            last_r = r
    ax[0].axvspan(df.index[start], df.index[-1], color=colors.get(last_r, "#ffffff"), alpha=0.15)

    # Price line + entropy dots
    sc = ax[0].scatter(df.index, df["Close"], c=df["Folding_Score"], cmap="viridis", s=20, zorder=3)
    ax[0].plot(df.index, df["Close"], color="black", linewidth=1.2, zorder=2)
    plt.colorbar(sc, ax=ax[0], label="Folding Score (Eigen-Entropy)")

    ax[0].set_title(f"{ticker} — Market Folding (Eigen-Entropy Regimes)")
    ax[0].set_ylabel("Price")

    # --- Highlight latest point (today) ---
    ax[0].axvline(last_date, linestyle="--", linewidth=1.2, alpha=0.6)
    ax[0].scatter([last_date], [last_close], s=220, marker="o", edgecolors="black", linewidths=1.2, zorder=5)

    badge_text = (
        f"LATEST\n"
        f"{pd.to_datetime(last_date).date()}\n"
        f"Close: {last_close:.2f}\n"
        f"Score: {last_score:.4f}\n"
        f"Regime: {last_regime}"
    )

    # Place annotation near the last point (slight offset)
    ax[0].annotate(
        badge_text,
        xy=(last_date, last_close),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.85),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        zorder=6,
    )

    ax[0].grid(alpha=0.2)

    # -----------------------------
    # Explanation panel (under chart)
    # -----------------------------
    ax[1].axis("off")

    today_box = (
        "TODAY / LATEST BAR (highlighted above)\n"
        f"- Date: {pd.to_datetime(last_date).date()}\n"
        f"- Close: {last_close:.2f}\n"
        f"- Folding Score (Eigen-Entropy): {last_score:.4f}\n"
        f"- Regime: {last_regime}\n"
    )

    explanation = f"""{today_box}
WHAT IS MARKET FOLDING?
Market Folding treats the market as a geometric object. We build 5 “forces” (momentum, volatility,
trend stress, volume pressure, money flow), then measure how tightly they move together.

WHAT IS THE FOLDING SCORE (EIGEN-ENTROPY)?
We compute a rolling correlation matrix of those forces and take its eigenvalues.
Entropy of eigenvalues measures “effective dimensionality”:
- Higher entropy: drivers are more independent (complex/healthy structure)
- Lower entropy: drivers collapse together (fragile/stressed structure)

REGIMES:
🟢 COMPLEX_FOLDED (high score):
  - diverse drivers, structurally stable trends often persist
⚪ TRANSITION (mid score):
  - structure changing, consolidation or regime shift
🔴 COLLAPSED (low score):
  - correlations converge, fragile market; sharp moves can start here

INTERPRETATION:
- Falling score → structure collapsing (risk rising)
- Rising score  → structure opening (stability improving)
"""

    ax[1].text(0.01, 0.98, explanation, va="top", fontsize=11, family="monospace")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
