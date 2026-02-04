import math
import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt_obj
import re

def _pick_first_series(obj):
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] >= 1:
            return obj.iloc[:, 0]
        return pd.Series(dtype="float64")
    return obj

def _scalar_from_value(val, default=float("nan")):
    try:
        if isinstance(val, pd.DataFrame):
            if val.empty: return default
            val = val.iloc[:, 0]
        if isinstance(val, pd.Series):
            if val.empty: return default
            val = val.iloc[0]
        if val is None: return default
        return float(val)
    except Exception: return default

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce")

def _find_col(df: pd.DataFrame, side: str, key: str):
    side, key = side.lower(), key.lower()
    cols = list(df.columns)
    patterns = [
        rf"^{side}\s*{key}$", rf"^{side}\s*{key}\b",
        rf"^{side}\b.*\b{key}$", rf"^.*\b{side}\b.*\b{key}\b.*$",
        rf"^.*\b{key}\b.*\b{side}\b.*$",
    ]
    for pat in patterns:
        for c in cols:
            if re.search(pat, str(c), flags=re.IGNORECASE): return c
    return None

def _bs_greeks(S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0: return (float('nan'),) * 6
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    Nd1, pdf_d1 = _norm_cdf(d1), _norm_pdf(d1)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    call_delta, put_delta = disc_q * Nd1, disc_q * (Nd1 - 1.0)
    gamma = (disc_q * pdf_d1) / (S * sigma * sqrtT)
    vega_per_1pct = (S * disc_q * pdf_d1 * sqrtT) / 100.0
    call_theta_y = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * disc_r * _norm_cdf(d2) + q * S * disc_q * Nd1
    put_theta_y = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrtT) + r * K * disc_r * _norm_cdf(-d2) - q * S * disc_q * _norm_cdf(-d1)
    return call_delta, put_delta, gamma, vega_per_1pct, call_theta_y / 365.0, put_theta_y / 365.0

def _build_spot_move_matrix(spot: float, call_delta: float, put_delta: float, gamma: float) -> pd.DataFrame:
    moves = [-20, -15, -10, -7.5, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7.5, 10, 15, 20]
    rows = []
    for dS in moves:
        call_chg = (call_delta * dS) + 0.5 * gamma * (dS ** 2)
        put_chg = (put_delta * dS) + 0.5 * gamma * (dS ** 2)
        rows.append({"Spot Move ($)": dS, "New Spot": spot + dS, "Call \u0394+\u0393 Est. Change": call_chg, "Put \u0394+\u0393 Est. Change": put_chg})
    return pd.DataFrame(rows)

def _swing_high_low_from_history(hist_df: pd.DataFrame, lookback_days: int):
    try:
        sub = hist_df.tail(int(lookback_days)).copy()
        if sub.empty: return None
        lo = float(sub["Low"].min()) if "Low" in sub.columns else float(sub["Close"].min())
        hi = float(sub["High"].max()) if "High" in sub.columns else float(sub["Close"].max())
        return (lo, hi) if lo != hi else None
    except Exception: return None

def _fib_levels_from_swing(swing_low: float, swing_high: float):
    lo, hi = float(swing_low), float(swing_high)
    rng = hi - lo
    if rng == 0: return None
    retr = {"0% (Low)": lo, "23.6%": hi - 0.236 * rng, "38.2%": hi - 0.382 * rng, "50.0%": hi - 0.5 * rng, "61.8%": hi - 0.618 * rng, "78.6%": hi - 0.786 * rng, "100% (High)": hi}
    ext = {"Upper 127.2%": hi + 0.272 * rng, "Upper 161.8%": hi + 0.618 * rng, "Lower -27.2%": lo - 0.272 * rng, "Lower -61.8%": lo - 0.618 * rng}
    return retr, ext

def approx_skew_25d(df: pd.DataFrame, spot: float = None, T: float = None) -> dict:
    d = df.copy()
    if "Strike" not in d.columns: return {}
    d["strike_num"] = _to_float_series(d["Strike"])
    d = d.dropna(subset=["strike_num"]).sort_values("strike_num")
    
    c_iv_col = _find_col(d, "call", "iv")
    p_iv_col = _find_col(d, "put", "iv")
    if not (c_iv_col and p_iv_col): return {}
    
    d["call_iv"] = _to_float_series(d[c_iv_col])
    d["put_iv"] = _to_float_series(d[p_iv_col])
    
    c_del_col = _find_col(d, "call", "delta")
    p_del_col = _find_col(d, "put", "delta")
    
    # If Delta is missing, try to calculate it or use ATM IV as proxy
    if not (c_del_col and p_del_col) and spot and T:
        d["call_delta"] = d.apply(lambda r: bs_delta(spot, r["strike_num"], T, 0.04, 0.0, r["call_iv"], True) if pd.notna(r["call_iv"]) else np.nan, axis=1)
        d["put_delta"] = d.apply(lambda r: bs_delta(spot, r["strike_num"], T, 0.04, 0.0, r["put_iv"], False) if pd.notna(r["put_iv"]) else np.nan, axis=1)
    elif c_del_col and p_del_col:
        d["call_delta"] = _to_float_series(d[c_del_col])
        d["put_delta"] = _to_float_series(d[p_del_col])
    else:
        # Last resort: absolute distance from spot as proxy for ATM skew if delta calculation is impossible
        if spot:
            d["dist"] = (d["strike_num"] - spot).abs()
            d = d.sort_values("dist")
            # This isn't really 25d skew but it's better than nothing? Actually let's just fail gracefully if no delta.
            return {}
        return {}

    dc = d.dropna(subset=["call_iv", "call_delta"])
    dp = d.dropna(subset=["put_iv", "put_delta"])
    if dc.empty or dp.empty: return {}

    c_row = dc.iloc[(dc["call_delta"] - 0.25).abs().argsort()[:1]].iloc[0]
    p_row = dp.iloc[(dp["put_delta"] + 0.25).abs().argsort()[:1]].iloc[0]
    
    return {
        "call_25d_strike": float(c_row["strike_num"]), 
        "call_25d_iv": float(c_row["call_iv"]), 
        "call_25d_delta": float(c_row["call_delta"]), 
        "put_25d_strike": float(p_row["strike_num"]), 
        "put_25d_iv": float(p_row["put_iv"]), 
        "put_25d_delta": float(p_row["put_delta"]), 
        "skew_call_minus_put": float(c_row["call_iv"] - p_row["put_iv"])
    }

def _slope(series: pd.Series, lookback: int = 5) -> float:
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) <= lookback: return float("nan")
        return float(s.iloc[-1] - s.iloc[-1 - lookback])
    except Exception: return float("nan")

def _ensure_flat_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly flattens yfinance MultiIndex columns and returns a focused OHLCV DataFrame."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        col_map = {}
        for col in d.columns:
            for target in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
                if target not in col_map and any(str(x).strip().lower() == target.lower() for x in col):
                    col_map[target] = col
        new_df = pd.DataFrame(index=d.index)
        for target, tup in col_map.items():
            new_df[target] = d[tup]
        d = new_df
    
    # Standardize column names to Title Case
    rename_map = {}
    for col in d.columns:
        for target in ["Open", "High", "Low", "Close", "Volume"]:
            if str(col).strip().lower() == target.lower():
                rename_map[col] = target
    d = d.rename(columns=rename_map)
    
    # Fallback for "Close" if only "Adj Close" exists
    if "Close" not in d.columns and "Adj Close" in d.columns:
        d["Close"] = d["Adj Close"]
    
    # Convert all to numeric 1D series
    for col in d.columns:
        if col in ["Open", "High", "Low", "Close", "Volume"]:
            v = d[col]
            if isinstance(v, pd.DataFrame): 
                v = v.iloc[:, 0] if v.shape[1] >= 1 else pd.Series(dtype=float)
            d[col] = pd.to_numeric(v, errors='coerce')
    
    return d

def compute_ma_stack_and_regime(hist) -> dict:
    out = {"ok": False, "label": "N/A", "strength": 0, "details": {}}
    if hist is None: return out
    h = _ensure_flat_ohlcv(hist) if isinstance(hist, pd.DataFrame) else pd.DataFrame()
    if h.empty:
        if isinstance(hist, pd.Series):
             h = pd.DataFrame({"Close": pd.to_numeric(hist, errors="coerce")}).dropna()
        else: return out
    
    if len(h) < 200: return out
    close = h["Close"]
    for w in [20, 50, 200]: h[f"SMA{w}"] = close.rolling(w).mean()
    short_windows = [15, 20, 30, 45, 60]
    for w in short_windows: h[f"MA{w}"] = close.rolling(w).mean()
    last = h.iloc[-1]
    c, sma20, sma50, sma200 = float(last["Close"]), float(last["SMA20"]), float(last["SMA50"]), float(last["SMA200"])
    s20, s50, s200 = float(last["SMA20"] - h["SMA20"].iloc[-11]), float(last["SMA50"] - h["SMA50"].iloc[-11]), float(last["SMA200"] - h["SMA200"].iloc[-11])
    up_slopes = sum(1 for v in (s20, s50, s200) if v > 0)
    dn_slopes = sum(1 for v in (s20, s50, s200) if v < 0)
    bull_stack, bear_stack = (c > sma20 > sma50 > sma200), (c < sma20 < sma50 < sma200)
    ma_vals = [float(last[f"MA{w}"]) for w in short_windows]
    short_bull_stack = all(ma_vals[i] > ma_vals[i+1] for i in range(len(ma_vals)-1))
    short_bear_stack = all(ma_vals[i] < ma_vals[i+1] for i in range(len(ma_vals)-1))
    strength = (30 if bull_stack or bear_stack else 0) + (20 if short_bull_stack or short_bear_stack else 0) + (15 if up_slopes >= 2 or dn_slopes >= 2 else 0) + (10 if (c > sma200 and not bear_stack) or (c < sma200 and not bull_stack) else 0)
    if bull_stack and up_slopes >= 2: label = "STRONG UPTREND 📈"
    elif bear_stack and dn_slopes >= 2: label = "STRONG DOWNTREND 📉"
    elif c > sma200: label = "WEAK / CHOPPY UPTREND ⚠️"
    elif c < sma200: label = "WEAK / CHOPPY DOWNTREND ⚠️"
    else: label = "NO TREND 💤"
    out.update({"ok": True, "label": label, "strength": int(min(100, strength)), "details": {"close": c, "SMA20": sma20, "SMA50": sma50, "SMA200": sma200, "bull_stack": bool(bull_stack), "bear_stack": bool(bear_stack)}})
    return out

def realized_vol_annualized(hist: pd.DataFrame, window: int = 20) -> float:
    try:
        h = _ensure_flat_ohlcv(hist)
        if h.empty or "Close" not in h.columns: return float("nan")
        c = h["Close"]
        if len(c) < window + 1: return float("nan")
        rets = np.log(c / c.shift(1)).dropna()
        return float(rets.tail(window).std() * np.sqrt(252))
    except Exception: return float("nan")

def iv_proxy_rank(current_iv: float, hist: pd.DataFrame, window: int = 20) -> dict:
    out = {"ok": False, "iv_proxy_rank": None, "rv_min": None, "rv_max": None, "details": {}}
    if hist is None or hist.empty or not (current_iv and current_iv > 0): return out
    h = _ensure_flat_ohlcv(hist)
    if h.empty or "Close" not in h.columns: return out
    c = h["Close"]
    rv = (np.log(c / c.shift(1)).dropna().rolling(window).std() * math.sqrt(252)).dropna()
    if rv.empty: return out
    rv_min, rv_max = float(rv.min()), float(rv.max())
    rank = 100.0 * (float(current_iv) - rv_min) / (rv_max - rv_min) if rv_max != rv_min else 50.0
    out.update({"ok": True, "iv_proxy_rank": float(max(0.0, min(100.0, rank))), "rv_min": rv_min, "rv_max": rv_max})
    return out

def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0: return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * (_norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0)

def bs_vanna_charm(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> dict:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0: return {"delta": 0.0, "vanna": 0.0, "charm_per_year": 0.0, "charm_per_day": 0.0}
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    phi_d1, exp_qT = _norm_pdf(d1), math.exp(-q * T)
    vanna = -exp_qT * phi_d1 * d2 / sigma
    charm_y = (q * exp_qT * _norm_cdf(d1) - exp_qT * phi_d1 * ((r - q) / (sigma * sqrtT) - d2 / (2 * T))) if is_call else (-q * exp_qT * _norm_cdf(-d1) - exp_qT * phi_d1 * ((r - q) / (sigma * sqrtT) - d2 / (2 * T)))
    return {"delta": float(bs_delta(S, K, T, r, q, sigma, is_call)), "vanna": float(vanna), "charm_per_year": float(charm_y), "charm_per_day": float(charm_y / 365.0)}

def compute_key_levels(hist_daily: pd.DataFrame) -> dict:
    out = {'ok': False, 'details': {}}
    if hist_daily is None or hist_daily.empty: return out
    h = _ensure_flat_ohlcv(hist_daily)
    
    if "Date" not in h.columns:
        h = h.reset_index().rename(columns={h.index.name if h.index.name else "index": "Date"})
    
    if "Close" not in h.columns: return out
    h = h.dropna(subset=["Close"]).sort_values("Date")
    if len(h) < 2: return out
    
    prev = h.iloc[-2]
    last5 = h.tail(5)
    
    out.update({
        "ok": True, 
        "prev_high": _scalar_from_value(prev.get("High")), 
        "prev_low": _scalar_from_value(prev.get("Low")), 
        "prev_close": _scalar_from_value(prev.get("Close")), 
        "wk_high": _scalar_from_value(last5["High"].max()) if "High" in last5.columns else float("nan"), 
        "wk_low": _scalar_from_value(last5["Low"].min()) if "Low" in last5.columns else float("nan")
    })
    return out

def compute_opening_range(intra: pd.DataFrame, minutes: int = 30) -> dict:
    if intra is None or intra.empty: return {'ok': False}
    df = _ensure_flat_ohlcv(intra)
    if df.empty or "Close" not in df.columns: return {'ok': False}
    
    dt_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else None)
    if not dt_col:
        # If no obvious date col, check index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name if df.index.name else "index": "Datetime"})
            dt_col = "Datetime"
        else: return {'ok': False}

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col, "Close"])
    if df.empty: return {'ok': False}
    
    last_day = df[dt_col].dt.date.max()
    d = df[df[dt_col].dt.date == last_day].sort_values(dt_col)
    if d.empty: return {'ok': False}
    
    or_df = d[d[dt_col] < d.iloc[0][dt_col] + pd.Timedelta(minutes=minutes)]
    if or_df.empty or not all(c in or_df.columns for c in ["High", "Low"]): return {'ok': False}
    
    return {
        "ok": True, 
        "session_date": str(last_day), 
        "or_high": float(or_df["High"].max()), 
        "or_low": float(or_df["Low"].min()), 
        "last_close": float(d.iloc[-1]["Close"])
    }

def _as_1d_series(x):
    if isinstance(x, pd.DataFrame): return x.iloc[:, 0] if x.shape[1] >= 1 else pd.Series(dtype=float)
    return x if isinstance(x, pd.Series) else pd.Series(x)

def structure_label(hist_daily: pd.DataFrame, lookback: int = 40) -> dict:
    out = {"ok": False, "label": "N/A"}
    if hist_daily is None or hist_daily.empty: return out
    h = hist_daily.copy()
    
    # Flatten MultiIndex if present
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in h.columns]
        
    if "Date" not in h.columns:
        possible_dates = [c for c in h.columns if str(c).lower() in ["date", "datetime"]]
        if possible_dates: h = h.rename(columns={possible_dates[0]: "Date"})
        else: h = h.reset_index().rename(columns={h.index.name if h.index.name else "index": "Date"})

    # Stricter column check
    required = ["High", "Low", "Close"]
    if not all(c in h.columns for c in required): return out

    h = h.sort_values("Date").tail(int(lookback)).reset_index(drop=True)
    for c in required: h[c] = pd.to_numeric(_as_1d_series(h[c]), errors="coerce")
    
    # Final check before dropna to avoid KeyError if renaming failed somehow
    present = [c for c in required if c in h.columns]
    if len(present) < 3: return out
    
    h = h.dropna(subset=present)
    if len(h) < 10: return out
    win, piv_hi, piv_lo = 2, [], []
    for i in range(win, len(h) - win):
        if h["High"].iloc[i] == h["High"].iloc[i-win:i+win+1].max(): piv_hi.append((i, h["High"].iloc[i]))
        if h["Low"].iloc[i] == h["Low"].iloc[i-win:i+win+1].min(): piv_lo.append((i, h["Low"].iloc[i]))
    label = "RANGE / UNCLEAR 💤"
    if len(piv_hi) >= 2 and len(piv_lo) >= 2:
        hh, hl = piv_hi[-1][1] > piv_hi[-2][1], piv_lo[-1][1] > piv_lo[-2][1]
        lh, ll = piv_hi[-1][1] < piv_hi[-2][1], piv_lo[-1][1] < piv_lo[-2][1]
        if hh and hl: label = "BULL STRUCTURE 📈 (HH + HL)"
        elif lh and ll: label = "BEAR STRUCTURE 📉 (LH + LL)"
        elif hh and not hl: label = "RISKY UPTREND ⚠️ (HH but no HL)"
        elif ll and not lh: label = "RISKY DOWNTREND ⚠️ (LL but no LH)"
    out.update({"ok": True, "label": label, "pivot_highs": piv_hi[-3:], "pivot_lows": piv_lo[-3:]})
    return out

def build_trade_bias(trend_label: str, gex_regime: str, iv_rank_proxy: float | None) -> str:
    bias = []
    if "UPTREND" in trend_label: bias.append("Bias: **Bullish** → favor call spreads / put sells (defined risk).")
    elif "DOWNTREND" in trend_label: bias.append("Bias: **Bearish** → favor put spreads / call sells (defined risk).")
    else: bias.append("Bias: **Neutral/Chop** → favor premium-selling structures (iron condor / butterflies) when IV is elevated.")
    if "NEGATIVE" in gex_regime: bias.append("GEX regime: **Negative gamma** → expect faster moves + whipsaws; size smaller, use defined risk.")
    elif "PIN" in gex_regime: bias.append("GEX regime: **Pin/Mean-revert** → mean-reversion near big strikes can work better than breakouts.")
    if iv_rank_proxy is not None:
        if iv_rank_proxy >= 70: bias.append("IV is **high** (proxy) → buying naked options is harder; spreads/premium-selling often fit better.")
        elif iv_rank_proxy <= 30: bias.append("IV is **low** (proxy) → directional option buying can be more reasonable (still manage risk).")
    return "\n".join(bias)

def short_interest_bias(
    short_shares: float,
    float_shares: float,
    avg_vol_10d: float | None = None,
    short_shares_prior: float | None = None,
    short_ratio: float | None = None
) -> dict:
    def _to_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    short_shares = _to_float(short_shares)
    float_shares = _to_float(float_shares)
    avg_vol_10d = _to_float(avg_vol_10d)
    short_shares_prior = _to_float(short_shares_prior)
    short_ratio = _to_float(short_ratio)

    notes = []
    short_pct_float = None
    if short_shares is not None and float_shares and float_shares > 0:
        short_pct_float = (short_shares / float_shares) * 100.0
        notes.append(f"Short % of float: {short_pct_float:.2f}%")

    days_to_cover = short_ratio
    if days_to_cover is None and short_shares is not None and avg_vol_10d and avg_vol_10d > 0:
        days_to_cover = short_shares / avg_vol_10d
        notes.append("Short ratio estimated from short shares / avg 10d volume.")
    if days_to_cover is not None:
        notes.append(f"Short ratio (days to cover): {days_to_cover:.2f}")

    change_pct = None
    if short_shares is not None and short_shares_prior and short_shares_prior > 0:
        change_pct = (short_shares - short_shares_prior) / short_shares_prior * 100.0
        notes.append(f"Short shares change vs prior: {change_pct:+.1f}%")

    score = 0
    if short_pct_float is not None:
        if short_pct_float >= 10:
            score += 2
            notes.append("Short % of float is high (>=10%).")
        elif short_pct_float >= 5:
            score += 1
            notes.append("Short % of float is moderate (5–10%).")
        else:
            score -= 1
            notes.append("Short % of float is low (<5%).")

    if days_to_cover is not None:
        if days_to_cover >= 5:
            score += 2
            notes.append("Days to cover is high (>=5).")
        elif days_to_cover >= 3:
            score += 1
            notes.append("Days to cover is moderate (3–5).")
        elif days_to_cover < 2:
            score -= 1
            notes.append("Days to cover is low (<2).")

    if change_pct is not None:
        if change_pct >= 10:
            score += 1
            notes.append("Short interest increased materially vs prior.")
        elif change_pct <= -10:
            score -= 1
            notes.append("Short interest decreased materially vs prior.")

    if score >= 3:
        direction, label = "Bearish positioning / squeeze risk", "High short interest"
    elif score == 2:
        direction, label = "Bearish tilt", "Elevated short interest"
    elif score == 1:
        direction, label = "Mildly bearish", "Moderate short interest"
    elif score <= -2:
        direction, label = "Light short interest", "Low short interest"
    else:
        direction, label = "Neutral", "Balanced short interest"

    return {
        "short_pct_float": short_pct_float,
        "direction": direction,
        "label": label,
        "notes": notes,
        "score": score,
    }

def confidence_score(trend_strength: int, structure_ok: bool, vol_ok: bool, gex_ok: bool, or_ok: bool) -> int:
    score = int(trend_strength * 0.4) + (15 if structure_ok else 0) + (15 if vol_ok else 0) + (20 if gex_ok else 0) + (10 if or_ok else 0)
    return int(min(100, score))

def add_vwap_obv_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TP"] = (out["High"] + out["Low"] + out["Close"]) / 3.0
    vol_cumsum = out["Volume"].cumsum().replace(0, np.nan)
    out["VWAP"] = (out["TP"] * out["Volume"]).cumsum() / vol_cumsum
    out["OBV"] = (np.sign(out["Close"].diff()) * out["Volume"]).fillna(0).cumsum()
    hl = (out["High"] - out["Low"]).replace(0, np.nan)
    out["MFM"] = ((out["Close"] - out["Low"]) - (out["High"] - out["Close"])) / hl
    out["ADL"] = (out["MFM"].fillna(0) * out["Volume"]).cumsum()
    return out

def add_acc_dist_signals(df: pd.DataFrame, confirm_days: int = 2) -> pd.DataFrame:
    out = df.copy()
    close_chg, obv_chg = out["Close"].diff(), out["OBV"].diff()
    out["ACC"], out["DIST"] = (close_chg > 0) & (obv_chg > 0), (close_chg < 0) & (obv_chg < 0)
    out["ACC_STR"], out["DIST_STR"] = out["ACC"].rolling(confirm_days).sum(), out["DIST"].rolling(confirm_days).sum()
    out["VWAP_SLOPE"] = out["VWAP"].diff()
    out["BUY"] = out["ACC"] & (out["ACC_STR"] >= confirm_days) & (out["Close"] > out["VWAP"]) & (out["VWAP_SLOPE"] > 0)
    out["SELL"] = out["DIST"] & (out["DIST_STR"] >= confirm_days) & (out["Close"] < out["VWAP"]) & (out["VWAP_SLOPE"] < 0)
    buy_shifted, sell_shifted = out["BUY"].shift(1), out["SELL"].shift(1)
    out["BUY_MARK"] = out["BUY"] & (~buy_shifted.where(buy_shifted.notna(), False).astype(bool))
    out["SELL_MARK"] = out["SELL"] & (~sell_shifted.where(sell_shifted.notna(), False).astype(bool))
    return out

def build_vwap_obv_analysis(hist: pd.DataFrame, confirm_days: int = 2) -> tuple:
    df = hist.copy()
    if "Date" not in df.columns: df = df.reset_index().rename(columns={"index": "Date"})
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = add_vwap_obv_indicators(df)
    df = add_acc_dist_signals(df, confirm_days=confirm_days)
    last = df.iloc[-1]
    summary = {"date": last["Date"], "close": float(last["Close"]), "vwap": float(last["VWAP"]), "vwap_slope": float(last["VWAP_SLOPE"]), "obv": float(last["OBV"]), "adl": float(last["ADL"]), "acc_str": int(last["ACC_STR"]), "dist_str": int(last["DIST_STR"]), "is_acc": bool(last["ACC"]), "is_dist": bool(last["DIST"]), "buy_signal": bool(last["BUY"]), "sell_signal": bool(last["SELL"]), "total_buys": int(df["BUY_MARK"].sum()), "total_sells": int(df["SELL_MARK"].sum())}
    return df, summary

TRADING_DAYS = 252
def annualized_realized_vol_series(close: pd.Series, window_days: int) -> pd.Series:
    logret = np.log(close.astype(float)).diff()
    return logret.rolling(window_days).std(ddof=0) * math.sqrt(TRADING_DAYS)

def build_vol_cone(close: pd.Series, windows: list = None, percentiles: list = None) -> pd.DataFrame:
    windows, percentiles = windows or [5, 10, 20, 30, 60, 90, 120], percentiles or [5, 25, 50, 75, 95]
    rows = []
    for w in windows:
        rv = annualized_realized_vol_series(close, w).dropna()
        if rv.empty: continue
        row = {"window_days": w, "latest_rv": float(rv.iloc[-1])}
        for p in percentiles: row[f"p{p}"] = float(np.nanpercentile(rv.values, p))
        rows.append(row)
    return pd.DataFrame(rows).set_index("window_days").sort_index()

def vol_regime_for_window(latest_rv: float, row: pd.Series) -> tuple:
    p25, p75 = float(row.get("p25", np.nan)), float(row.get("p75", np.nan))
    if np.isnan([p25, p75]).any(): return "UNKNOWN", "WAIT"
    if latest_rv <= p25: return "VOL CHEAP (≤ p25)", "BUY OPTIONS"
    if latest_rv >= p75: return "VOL EXPENSIVE (≥ p75)", "SELL PREMIUM"
    return "VOL NORMAL (p25–p75)", "SELECTIVE / WAIT"

def direction_from_vwap_obv(close: float, vwap: float, vwap_slope: float, obv_slope: float) -> str:
    if (close > vwap) and (vwap_slope > 0) and (obv_slope > 0): return "BULLISH"
    if (close < vwap) and (vwap_slope < 0) and (obv_slope < 0): return "BEARISH"
    return "NEUTRAL"

def combine_vol_suggestion(vol_trade_type: str, direction: str) -> str:
    if vol_trade_type == "BUY OPTIONS":
        if direction == "BULLISH": return "Buy CALL or CALL DEBIT SPREAD (trend up + vol cheap)"
        if direction == "BEARISH": return "Buy PUT or PUT DEBIT SPREAD (trend down + vol cheap)"
        return "Buy STRADDLE/STRANGLE (vol cheap but direction unclear)"
    if vol_trade_type == "SELL PREMIUM":
        if direction == "BULLISH": return "Sell PUT CREDIT SPREAD / CSP (bullish + vol expensive)"
        if direction == "BEARISH": return "Sell CALL CREDIT SPREAD / CC (bearish + vol expensive)"
        return "Iron Condor (range + vol expensive)"
    return "WAIT / NO EDGE (vol normal or direction unclear)"

def build_vol_cone_analysis(hist: pd.DataFrame, focus_window: int = 20, windows: list = None, percentiles: list = None) -> tuple:
    df = hist.copy()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    if "Date" not in df.columns: df = df.reset_index().rename(columns={"index": "Date"})
    df = add_vwap_obv_indicators(df)
    df["VWAP_SLOPE"], df["OBV_SLOPE"] = df["VWAP"].diff(), df["OBV"].diff(5) / 5.0
    cone = build_vol_cone(df["Close"].dropna(), windows=windows, percentiles=percentiles)
    if cone.empty: raise ValueError("Could not build volatility cone - not enough data")
    if focus_window not in cone.index: focus_window = int(cone.index.values[len(cone) // 2])
    focus_rv = float(cone.loc[focus_window, "latest_rv"])
    vol_regime, trade_type = vol_regime_for_window(focus_rv, cone.loc[focus_window])
    last = df.iloc[-1]
    direction = direction_from_vwap_obv(float(last["Close"]), float(last["VWAP"]), float(last["VWAP_SLOPE"]), float(last["OBV_SLOPE"]))
    summary = {"date": last["Date"], "close": float(last["Close"]), "vwap": float(last["VWAP"]), "vwap_slope": float(last["VWAP_SLOPE"]), "obv_slope": float(last["OBV_SLOPE"]), "focus_window": focus_window, "focus_rv": focus_rv, "vol_regime": vol_regime, "trade_type": trade_type, "direction": direction, "suggestion": combine_vol_suggestion(trade_type, direction)}
    return cone, summary, df

def build_gamma_levels(gex_df: pd.DataFrame, spot: float, top_n: int = 5):
    df = gex_df.copy()
    if df.empty: return None
    for c in ["strike", "call_gex", "put_gex", "net_gex", "gamma"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    magnets = df.assign(net_abs=df["net_gex"].abs()).sort_values("net_abs", ascending=False).head(top_n)[["strike", "net_gex"]]
    all_call_walls = df.sort_values("call_gex", ascending=False).head(15)[["strike", "call_gex"]]
    all_put_walls = df.sort_values("put_gex", ascending=False).head(15)[["strike", "put_gex"]]
    major_call_wall = float(all_call_walls.iloc[0]["strike"]) if not all_call_walls.empty else None
    major_put_wall = float(all_put_walls.iloc[0]["strike"]) if not all_put_walls.empty else None
    put_below = all_put_walls[all_put_walls["strike"] <= spot].sort_values("strike", ascending=False)
    call_above = all_call_walls[all_call_walls["strike"] >= spot].sort_values("strike", ascending=True)
    nearest_lower = float(put_below.iloc[0]["strike"]) if not put_below.empty else major_put_wall
    nearest_upper = float(call_above.iloc[0]["strike"]) if not call_above.empty else major_call_wall
    zero_gamma = None
    try:
        df_range = df[df["strike"].between(spot * 0.8, spot * 1.2)].copy()
        if not df_range.empty: zero_gamma = float(df_range.loc[df_range["net_gex"].abs().idxmin()]["strike"])
    except Exception: pass
    return {"magnets": magnets, "major_call_wall": major_call_wall, "major_put_wall": major_put_wall, "nearest_lower": nearest_lower, "nearest_upper": nearest_upper, "zero_gamma": zero_gamma, "gamma_box": {"lower": nearest_lower, "upper": nearest_upper}}

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.ewm(alpha=1 / length, adjust=False).mean(), loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

def bbands(close: pd.Series, length: int = 20, std: float = 2.0):
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std(ddof=0)
    return mid - std * sd, mid, mid + std * sd

def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (mfm * volume).rolling(length).sum() / volume.rolling(length).sum().replace(0, np.nan)

def minmax_rolling(x: pd.Series, lookback: int) -> pd.Series:
    rmin, rmax = x.rolling(lookback).min(), x.rolling(lookback).max()
    return (x - rmin) / (rmax - rmin).replace(0, np.nan)

def _shannon_entropy_from_probs(p: np.ndarray, base: float = 2.0) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0: return float("nan")
    return float(-np.sum(p * np.log(p)) / (np.log(base) if base and base != 1 else 1.0))

def rolling_eigen_entropy(df: pd.DataFrame, features: list[str], window: int, base: float = 2.0) -> pd.Series:
    n, X = len(df), df[features].to_numpy()
    scores = np.full(n, np.nan)
    for i in range(window, n + 1):
        corr = np.nan_to_num(np.corrcoef(X[i - window:i], rowvar=False))
        w = np.abs(np.linalg.eigvalsh(corr))
        if w.sum() > 0: scores[i - 1] = _shannon_entropy_from_probs(w / w.sum(), base=base)
    return pd.Series(scores, index=df.index)

def build_market_folding(df: pd.DataFrame, window_size: int, smooth: int, entropy_base: float, low_q: float, high_q: float) -> tuple:
    df = df.copy()
    for col in ["High", "Low", "Close", "Volume"]: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["RSI"] = rsi(df["Close"]) / 100.0
    lo, mid, hi = bbands(df["Close"])
    df["BBW"] = minmax_rolling((hi - lo) / mid.replace(0, np.nan), 50)
    df["NATR"] = atr(df["High"], df["Low"], df["Close"]) / df["Close"].replace(0, np.nan)
    vslow, vfast = df["Volume"].rolling(10).mean(), df["Volume"].rolling(5).mean()
    df["VolOsc"] = np.tanh((vfast - vslow) / vslow.replace(0, np.nan))
    df["CMF"] = cmf(df["High"], df["Low"], df["Close"], df["Volume"])
    feats = ["RSI", "BBW", "NATR", "VolOsc", "CMF"]
    df = df.dropna(subset=feats).copy()
    if df.empty: return df, None, None
    df["Entropy"] = rolling_eigen_entropy(df, feats, window_size, base=entropy_base)
    df["Folding_Score"] = df["Entropy"].rolling(smooth).mean()
    s = df["Folding_Score"].dropna()
    if s.empty: return df, None, None
    loq, hiq = float(s.quantile(low_q)), float(s.quantile(high_q))
    df["Regime"] = np.where(df["Folding_Score"] <= loq, "COLLAPSED", np.where(df["Folding_Score"] >= hiq, "COMPLEX_FOLDED", "TRANSITION"))
    return df, loq, hiq
