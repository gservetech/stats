import numpy as np
import pandas as pd
import plotly.graph_objects as go

def mcginley_dynamic(price: pd.Series, length: int = 14) -> pd.Series:
    p = pd.Series(price).astype(float).copy()
    out = pd.Series(index=p.index, dtype=float)
    md_prev = float(p.iloc[0])
    out.iloc[0] = md_prev
    for i in range(1, len(p)):
        pi = float(p.iloc[i])
        denom = length * (pi / md_prev) ** 4 if md_prev != 0 else length
        md_prev = md_prev + (pi - md_prev) / denom
        out.iloc[i] = md_prev
    return out

def kama(price: pd.Series, er_length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    p = pd.Series(price).astype(float).copy()
    out = pd.Series(index=p.index, dtype=float)
    fast_sc, slow_sc = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
    out.iloc[0] = float(p.iloc[0])
    abs_diff = p.diff().abs()
    for i in range(1, len(p)):
        if i < er_length:
            out.iloc[i] = out.iloc[i - 1]
            continue
        change = abs(float(p.iloc[i]) - float(p.iloc[i - er_length]))
        volatility = float(abs_diff.iloc[i - er_length + 1: i + 1].sum())
        er = (change / volatility) if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out.iloc[i] = out.iloc[i - 1] + sc * (float(p.iloc[i]) - float(out.iloc[i - 1]))
    return out

def kalman_filter_1d(close, process_var=1e-5, meas_var=1e-2) -> np.ndarray:
    if isinstance(close, pd.DataFrame): z = close.iloc[:, 0].to_numpy()
    elif isinstance(close, pd.Series): z = close.to_numpy()
    else: z = np.asarray(close)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = int(len(z))
    if n == 0: return np.array([], dtype=float)
    x, p = np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    x[0], p[0] = float(z[0]), 1.0
    q, r = float(process_var), float(meas_var)
    for k in range(1, n):
        x_pred, p_pred = x[k - 1], p[k - 1] + q
        K = p_pred / (p_pred + r)
        x[k] = x_pred + K * (float(z[k]) - x_pred)
        p[k] = (1 - K) * p_pred
    return x

def kalman_message(close_series, kalman_series, lookback: int = 20, band_pct: float = 0.003):
    close, kf = np.asarray(close_series, dtype=float).reshape(-1), np.asarray(kalman_series, dtype=float).reshape(-1)
    n = int(min(len(close), len(kf)))
    if n < 10: return {"trend": "N/A", "msg": "Not enough data for Kalman interpretation.", "reasons": ["Need at least ~10 bars."]}
    close, kf = close[-n:], kf[-n:]
    lb = min(int(lookback), n - 1)
    c, k = close[-lb:], kf[-lb:]
    band, diff = abs(float(k[-1])) * float(band_pct), float(c[-1] - k[-1])
    bias = "PRICE ABOVE KALMAN" if diff > band else ("PRICE BELOW KALMAN" if diff < -band else "PRICE NEAR KALMAN")
    sign = np.sign(c - k)
    sign[sign == 0] = 1
    crossings = int(np.sum(sign[1:] != sign[:-1]))
    slope = float(k[-1] - k[0])
    abs_moves = np.abs(np.diff(c))
    vol = float(np.mean(abs_moves)) if len(abs_moves) else 0.0
    ratio = abs(slope) / (vol + 1e-9)
    strength = int(max(0, min(100, round(100 * (1 - np.exp(-0.35 * ratio))))))
    hi1, lo1 = float(np.max(c)), float(np.min(c))
    mid = lb // 2
    hi0, lo0 = (float(np.max(c[:mid])) if mid > 2 else hi1), (float(np.min(c[:mid])) if mid > 2 else lo1)
    structure = "HH/HL" if (hi1 > hi0 and lo1 > lo0) else ("LH/LL" if (hi1 < hi0 and lo1 < lo0) else "MIXED")
    chop_threshold = max(6, lb // 4)
    if crossings >= chop_threshold and strength < 35: trend, regime = "RANGE", "RANGEBOUND"
    else:
        trend = "UPTREND" if slope > 0 else ("DOWNTREND" if slope < 0 else "RANGE")
        regime = "TRANSITION" if (strength < 35 or crossings >= max(3, chop_threshold // 2)) else "TRENDING"
        if regime == "TRANSITION" and trend != "RANGE": trend = "TRANSITION"
    msg = f"Kalman regime: {regime}. Signal: {trend}. {bias}."
    if trend == "UPTREND" and "ABOVE" in bias: msg += " Strong trend; pullbacks toward Kalman can act as support."
    if trend == "UPTREND" and "BELOW" in bias: msg += " Trend up but price below Kalman → watch reclaim; failure can mean weakness."
    if trend == "DOWNTREND" and "BELOW" in bias: msg += " Downtrend; rallies toward Kalman often fade (resistance)."
    if trend == "DOWNTREND" and "ABOVE" in bias: msg += " Price above Kalman in downtrend → possible transition if it holds."
    if trend == "RANGE": msg += " Expect mean-reversion; Kalman can act as the midline."
    return {"trend": trend, "bias": bias, "crossings": crossings, "trend_strength": strength, "regime": regime, "structure": structure, "msg": msg, "reasons": [f"Kalman slope: {slope:+.4f}", f"Crossings: {crossings}", f"Trend strength: {strength}/100", f"Structure: {structure}", f"Bias: {bias}"]}

def plot_filters(df_prices: pd.DataFrame, length_md: int, kama_er: int, kama_fast: int, kama_slow: int, kf_q: float, kf_r: float):
    close = df_prices["Close"].astype(float)
    md = mcginley_dynamic(close, length=length_md)
    k = kama(close, er_length=kama_er, fast=kama_fast, slow=kama_slow)
    kf = pd.Series(kalman_filter_1d(close, process_var=kf_q, meas_var=kf_r), index=close.index, name="Kalman")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=md.index, y=md, mode="lines", name=f"McGinley ({length_md})"))
    fig.add_trace(go.Scatter(x=k.index, y=k, mode="lines", name=f"KAMA (ER={kama_er}, {kama_fast}/{kama_slow})"))
    fig.add_trace(go.Scatter(x=kf.index, y=kf, mode="lines", name=f"Kalman (Q={kf_q:g}, R={kf_r:g})"))
    fig.update_layout(template="plotly_dark", height=520, title="📈 Market Noise Filters (McGinley / KAMA / Kalman)", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    return fig, kf
