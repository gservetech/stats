# stats_app/tabs/tab_vanna_charm.py
# ------------------------------------------------------------
# 🌊 Vanna & Charm (Proxy Map)
#
# IMPORTANT:
# Your Barchart chain does NOT contain true vanna/charm Greeks.
# This tab computes EDUCATIONAL PROXIES using only what you have:
# Strike + Call OI + Put OI (+ IV if available).
#
# Robust to API payload shapes:
# - {"success":True, "data":[rows...]}
# - {"success":True, "data":{"data":[rows...]}}
# - {"success":True, "payload":{"data":[rows...]}}
#
# Tolerates both row formats:
# - /options (side-by-side chain): Strike, Call OI, Put OI, Call IV, Put IV...
# - /weekly/gex (computed): strike, call_oi, put_oi, Call IV, Put IV...
#
# NOTE ON "TRADES" IN THIS TAB:
# - Everything below is EDUCATIONAL, rule-based templates.
# - It is NOT a signal, not financial advice.
# - Your fills/prices are unknown here; sizing uses conservative MAX-LOSS math.
# - Timing/expiry is presented as pattern-based education (common market tendencies).
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import date as _date, timedelta

from ..helpers.api_client import fetch_weekly_gex
from ..helpers.ui_components import st_plot


# ----------------------------
# Small utils
# ----------------------------

def _to_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)

        s = str(x).strip()
        if s == "" or s.lower() in ("na", "n/a", "none", "-"):
            return default

        s = s.replace(",", "")

        # "35%" -> 0.35
        if s.endswith("%"):
            s = s[:-1]
            v = float(s)
            return v / 100.0

        return float(s)
    except Exception:
        return default


def _to_int(x, default=0):
    v = _to_float(x, None)
    if v is None:
        return int(default)
    try:
        return int(round(v))
    except Exception:
        return int(default)


def _unwrap_rows(api_result: dict):
    """Return list[dict] rows from various API shapes."""
    if not isinstance(api_result, dict):
        return []

    d = api_result.get("data")

    if isinstance(d, list):
        return d

    if isinstance(d, dict) and isinstance(d.get("data"), list):
        return d["data"]

    p = api_result.get("payload")
    if isinstance(p, dict) and isinstance(p.get("data"), list):
        return p["data"]

    if isinstance(api_result.get("rows"), list):
        return api_result["rows"]

    return []


def _parse_chain_df(rows: list[dict]) -> pd.DataFrame:
    """
    Normalize rows into:
      strike, call_oi, put_oi, call_iv, put_iv
    """
    df0 = pd.DataFrame(rows).copy()
    if df0.empty:
        return df0

    cols = {c.lower().strip(): c for c in df0.columns}

    def col(*names):
        for n in names:
            k = n.lower().strip()
            if k in cols:
                return cols[k]
        return None

    c_strike = col("Strike", "strike")

    c_call_oi = col(
        "Call OI", "call oi",
        "Call Open Int", "call open int",
        "Call Open Interest", "call open interest",
        "call_oi",
    )
    c_put_oi = col(
        "Put OI", "put oi",
        "Put Open Int", "put open int",
        "Put Open Interest", "put open interest",
        "put_oi",
    )

    c_call_iv = col("Call IV", "call iv", "call_iv")
    c_put_iv = col("Put IV", "put iv", "put_iv")

    missing = []
    if not c_strike:
        missing.append("Strike/strike")
    if not c_call_oi:
        missing.append("Call OI/call_oi")
    if not c_put_oi:
        missing.append("Put OI/put_oi")
    if missing:
        raise ValueError(f"Need {', '.join(missing)} columns in chain rows.")

    out = pd.DataFrame()
    out["strike"] = df0[c_strike].apply(lambda x: _to_float(x, None))
    out["call_oi"] = df0[c_call_oi].apply(lambda x: _to_int(x, 0))
    out["put_oi"] = df0[c_put_oi].apply(lambda x: _to_int(x, 0))

    # IV optional
    out["call_iv"] = df0[c_call_iv].apply(lambda x: _to_float(x, None)) if c_call_iv else np.nan
    out["put_iv"] = df0[c_put_iv].apply(lambda x: _to_float(x, None)) if c_put_iv else np.nan

    out = out.dropna(subset=["strike"]).sort_values("strike")
    return out


def _trend_context_block(spot: float, hist_df: pd.DataFrame):
    """Educational MA context computed from hist_df Close."""
    st.markdown("### 📊 Trend Context (Educational)")
    if hist_df is None or not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        st.info("📊 Trend Context: Price history not available.")
        return

    if "Close" not in hist_df.columns:
        st.info("📊 Trend Context: 'Close' column not found in history.")
        return

    close = pd.to_numeric(hist_df["Close"], errors="coerce").dropna()
    if close.empty:
        st.info("📊 Trend Context: Close series is empty after cleaning.")
        return

    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

    bullets = []
    if np.isfinite(ma20):
        bullets.append("• Price is **above MA20** → structure is stronger." if spot > ma20 else "• Price is **below MA20** → structure is weaker.")
    else:
        bullets.append("• MA20 not available (not enough history).")

    if np.isfinite(ma50):
        bullets.append("• Price is **above MA50** → structure is stronger." if spot > ma50 else "• Price is **below MA50** → structure is weaker.")
    else:
        bullets.append("• MA50 not available (not enough history).")

    if np.isfinite(ma200):
        bullets.append("• Price is **above MA200** → long-term bias stronger." if spot > ma200 else "• Price is **below MA200** → long-term bias weaker.")
    else:
        bullets.append("• MA200 not available (not enough history).")

    st.markdown("  \n".join(bullets))
    st.caption("⚠️ This is context, not a prediction.")


def _calc_levels(df: pd.DataFrame, spot: float):
    """
    Dynamic 'walls' + 'magnet' from ONLY your available columns.
    - put_wall: strike with highest put OI
    - call_wall: strike with highest call OI
    - magnet: strike with highest absolute charm_proxy (pinning pressure near spot)
    """
    put_wall = None
    call_wall = None
    magnet = None

    if df is None or df.empty:
        return put_wall, call_wall, magnet

    if "put_oi" in df.columns and df["put_oi"].notna().any():
        put_wall = float(df.loc[df["put_oi"].idxmax(), "strike"])
    if "call_oi" in df.columns and df["call_oi"].notna().any():
        call_wall = float(df.loc[df["call_oi"].idxmax(), "strike"])

    if "charm_proxy" in df.columns and df["charm_proxy"].notna().any():
        magnet = float(df.loc[df["charm_proxy"].abs().idxmax(), "strike"])

    # If magnet couldn't be computed, fallback to nearest strike to spot
    if magnet is None:
        try:
            magnet = float(df.iloc[(df["strike"] - float(spot)).abs().argsort().iloc[0]]["strike"])
        except Exception:
            magnet = None

    return put_wall, call_wall, magnet


def _regime_sentence(spot: float, put_wall, call_wall, net_vanna: float, net_charm: float):
    """
    One meaningful sentence you can glance at.
    Uses dynamic walls + proxy values (direction only, not "targets").
    """
    s = float(spot)

    if put_wall is None or call_wall is None:
        bias = "call-weighted" if net_charm > 0 else "put-weighted" if net_charm < 0 else "balanced"
        return f"Proxy flows are {bias}; focus on nearest high-OI strikes because the map suggests pinning/pressure around spot."

    pw = float(put_wall)
    cw = float(call_wall)

    directional = (
        "call-side pressure" if (net_charm > 0 and net_vanna > 0) else
        "put-side pressure" if (net_charm < 0 and net_vanna < 0) else
        "mixed pressure"
    )

    if s > cw:
        return f"🚀 Above Call Wall ({cw:.2f}): breakout mode — {directional} can accelerate moves higher."
    if s < pw:
        return f"🧨 Below Put Wall ({pw:.2f}): breakdown mode — {directional} can accelerate moves lower."

    dist_to_pw = abs(s - pw)
    dist_to_cw = abs(s - cw)
    near = f"near Call Wall ({cw:.2f})" if dist_to_cw < dist_to_pw else f"near Put Wall ({pw:.2f})"

    return f"🧲 Inside walls ({pw:.2f} → {cw:.2f}) {near}: expect chop/pinning unless price breaks; {directional} hints which side may win on breakout."


# ----------------------------
# Timing + Expiry guidance (Educational, pattern-based)
# ----------------------------

def _next_friday(from_dt: _date) -> _date:
    """Return the next Friday date (including today if Friday)."""
    # Monday=0 ... Sunday=6, Friday=4
    delta = (4 - from_dt.weekday()) % 7
    return from_dt + timedelta(days=delta)


def _add_weeks_to_friday(friday_dt: _date, weeks: int) -> _date:
    return friday_dt + timedelta(days=7 * int(max(0, weeks)))


def _infer_vol_regime(df: pd.DataFrame) -> str:
    """
    Super simple IV read:
    - If IV exists, we compute median call/put IV near spot (already converted to decimal elsewhere).
    - Return: "high_iv", "normal_iv", "unknown"
    """
    if df is None or df.empty:
        return "unknown"
    civ = pd.to_numeric(df.get("call_iv", np.nan), errors="coerce")
    piv = pd.to_numeric(df.get("put_iv", np.nan), errors="coerce")
    iv = pd.concat([civ, piv], ignore_index=True).dropna()
    if iv.empty:
        return "unknown"

    # normalize "35" -> 0.35 if user forgot earlier conversion (defensive)
    iv = iv.apply(lambda x: x / 100.0 if x > 3.0 else x)
    med = float(iv.median())

    # educational thresholds only (not "signals")
    if med >= 0.55:
        return "high_iv"
    if med >= 0.30:
        return "normal_iv"
    return "low_iv"


def _timing_expiry_guidance_block(symbol: str, df: pd.DataFrame, spot: float, put_wall, call_wall, magnet,
                                 total_vanna: float, total_charm: float):
    """
    EDUCATIONAL:
    Suggests an expiry BUCKET + common weekly timing windows based on:
    - Regime: pinned/inside vs breakout vs breakdown
    - Optional IV regime (high IV -> more time / avoid weeklies unless breakout)
    This does NOT know your broker fills, news calendar, earnings, etc.
    """
    st.markdown("### 🗓️ Expiry + Timing Guidance (Educational patterns)")
    st.caption(
        "Not financial advice. This is a **pattern-matching helper** based on your own wall/magnet structure + proxies. "
        "Always factor liquidity, news/earnings, and your personal risk plan."
    )

    S = float(spot)
    step = _infer_strike_step(df["strike"]) if df is not None and not df.empty else 1.0
    iv_regime = _infer_vol_regime(df)

    if put_wall is None or call_wall is None:
        st.info(
            "Walls not detected (missing OI). Without walls, expiry/timing suggestions are weaker. "
            "Education: if you expect chop, use **more time**; if you expect breakout, wait for confirmation then choose a defined-risk structure."
        )
        return

    pw = float(put_wall)
    cw = float(call_wall)
    m = float(magnet) if magnet is not None else S

    inside = (S >= pw) and (S <= cw)
    pinned = inside and (abs(S - m) <= 2.0 * step)
    breakout = S > cw
    breakdown = S < pw

    directional = (
        "call-side pressure" if (total_charm > 0 and total_vanna > 0) else
        "put-side pressure" if (total_charm < 0 and total_vanna < 0) else
        "mixed pressure"
    )

    # date suggestions (bucketed): next Friday vs +2 weeks vs +3 weeks
    today = _date.today()
    nf = _next_friday(today)
    f_plus_2 = _add_weeks_to_friday(nf, 2)
    f_plus_3 = _add_weeks_to_friday(nf, 3)

    # "best" expiry bucket (education)
    if pinned:
        expiry_bucket = "2–3 weeks out"
        expiry_dates = f"{f_plus_2.isoformat()} or {f_plus_3.isoformat()}"
        why = (
            f"Price is near **magnet** `{m:.2f}` inside `{pw:.2f} → {cw:.2f}`. "
            "Pinned/chop environments often punish very short-dated options (theta + whipsaw)."
        )
    elif breakout or breakdown:
        expiry_bucket = "1–2 weeks out (or same-week ONLY after confirmation)"
        expiry_dates = f"{nf.isoformat()} (aggressive) or {f_plus_2.isoformat()} (more forgiving)"
        why = (
            "Breakout/breakdown regimes can move faster. "
            "Education: you can choose shorter expiries **only after** a clear break/hold; otherwise use more time."
        )
    else:
        expiry_bucket = "1–3 weeks out (depends on conviction + IV)"
        expiry_dates = f"{nf.isoformat()} / {f_plus_2.isoformat()} / {f_plus_3.isoformat()}"
        why = (
            f"Inside the range but not perfectly pinned. "
            "Education: expiry depends on whether you expect rotation/chop (more time) or quick rejection/break (less time)."
        )

    # adjust for IV regime (education)
    if iv_regime == "high_iv":
        iv_note = "IV appears **high** → education: prefer **more time** and defined-risk structures; weeklies can decay quickly."
        if "2–3 weeks" not in expiry_bucket:
            expiry_bucket = "2–3 weeks out"
            expiry_dates = f"{f_plus_2.isoformat()} or {f_plus_3.isoformat()}"
    elif iv_regime == "low_iv":
        iv_note = "IV appears **low** → education: time decay is slower; shorter expiries may behave cleaner (still risky)."
    else:
        iv_note = "IV not available/usable → guidance relies mostly on wall/magnet structure."

    # timing windows (education, not instructions)
    timing_md = """
**Common timing tendencies (educational):**
- **Avoid the first 5–15 minutes** after the open (spreads + whipsaws can be worst).
- Many traders prefer **after the first pullback** or **after a clear reclaim/break** (confirmation > guessing).
- If you're using **credit spreads** in chop: education often favors **after a rejection is visible** (not at the exact level).
- If you're using **momentum/debit spreads**: education often favors **break + hold** (don’t front-run).
- **Last 30–60 minutes** can show “pinning” toward magnets, but it can also be very fast/volatile.
"""

    # day-of-week tendencies (education)
    dow_md = """
**Day-of-week tendencies (educational, not always true):**
- **Mon–Tue:** better for **2–3 week** positions if you want time for the thesis to play out.
- **Wed:** can be “mid-week churn”; some avoid opening new risk unless a clean break is happening.
- **Thu:** if targeting **next Friday**, education: this is when many start positioning (still depends on regime).
- **Fri:** 0DTE/weekly can be very sensitive to pinning + gamma; education: sizing must be smaller and exits tighter.
"""

    # pattern rules specific to current structure
    if pinned:
        pattern = (
            f"**Pattern match (current): PINNED** near magnet `{m:.2f}`. "
            f"Bias tone: **{directional}**.\n\n"
            f"Education: prioritize **range-friendly** templates (credit spreads) or wait for a confirmed break of `{cw:.2f}` / `{pw:.2f}`."
        )
    elif breakout:
        pattern = (
            f"**Pattern match (current): BREAKOUT** above call wall `{cw:.2f}`. Tone: **{directional}**.\n\n"
            f"Education: momentum templates become more relevant **after** it holds above `{cw:.2f}`. "
            "If it falls back inside the range, the breakout thesis weakens."
        )
    elif breakdown:
        pattern = (
            f"**Pattern match (current): BREAKDOWN** below put wall `{pw:.2f}`. Tone: **{directional}**.\n\n"
            f"Education: downside momentum templates become more relevant **after** it stays below `{pw:.2f}`. "
            "If it re-enters the range, breakdown thesis weakens."
        )
    else:
        near_side = "call wall" if abs(S - cw) < abs(S - pw) else "put wall"
        pattern = (
            f"**Pattern match (current): INSIDE RANGE** (closer to **{near_side}**). Tone: **{directional}**.\n\n"
            "Education: near resistance → bear-call/trim longs; near support → bull-put/trim shorts; "
            "but confirmation beats guessing."
        )

    st.markdown(
        f"""
**Symbol:** `{symbol}`  
**Spot:** `{S:.2f}`  
**Put Wall:** `{pw:.2f}` | **Call Wall:** `{cw:.2f}` | **Magnet:** `{m:.2f}`  

---

### Suggested expiry bucket (educational)
- **Bucket:** **{expiry_bucket}**
- **Example Friday dates:** `{expiry_dates}`
- **Why:** {why}

**IV note:** {iv_note}

---

### Pattern match
{pattern}
"""
    )
    st.markdown(timing_md)
    st.markdown(dow_md)

    st.caption(
        "Reminder: These are common tendencies, not guarantees. News, earnings, and liquidity can dominate the setup."
    )


# ----------------------------
# Trade template helpers (Educational)
# ----------------------------

def _infer_strike_step(strikes: pd.Series) -> float:
    """Infer typical strike spacing (e.g., 0.5 / 1 / 2.5 / 5) from chain strikes."""
    s = pd.to_numeric(strikes, errors="coerce").dropna().sort_values().unique()
    if len(s) < 3:
        return 1.0
    diffs = np.diff(s)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1.0
    step = float(np.median(diffs))
    if step <= 0:
        step = 1.0
    return step


def _nearest_strike(strikes: pd.Series, x: float) -> float:
    s = pd.to_numeric(strikes, errors="coerce").dropna()
    if s.empty:
        return float(x)
    idx = (s - float(x)).abs().idxmin()
    return float(s.loc[idx])


def _round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(round(float(x) / step) * step)


def _format_spread(name: str, leg1: str, k1: float, leg2: str, k2: float) -> str:
    return f"**{name}:** {leg1} `{k1:.2f}` / {leg2} `{k2:.2f}`"


def _sizing_math(risk_budget_usd: float, spread_width: float, credit_est: float = 0.0) -> dict:
    """
    Conservative sizing:
    - For credit spread: max_loss ≈ (width - credit) * 100
    - If credit unknown, use credit_est=0 -> worst-case width*100
    """
    width = max(float(spread_width), 0.0)
    credit = max(float(credit_est), 0.0)
    max_loss_per_1 = max((width - credit) * 100.0, 0.0)
    if max_loss_per_1 <= 0:
        return {"max_loss_per_spread": 0.0, "contracts": 0}
    contracts = int(np.floor(max(float(risk_budget_usd), 0.0) / max_loss_per_1))
    return {"max_loss_per_spread": max_loss_per_1, "contracts": max(contracts, 0)}


def _trade_templates_block(symbol: str, df: pd.DataFrame, spot: float, put_wall, call_wall, magnet, total_vanna: float, total_charm: float):
    """
    EDUCATIONAL spread templates (bull/bear) using dynamic strikes.
    No premiums/fills here, so we provide:
    - strike suggestions based on walls/magnet/spot
    - sizing based on user risk budget and spread width (max loss)
    """
    st.markdown("### 🧩 Trade Templates (Educational, rule-based)")
    st.caption(
        "Not financial advice. These are **templates** that adapt to any ticker using: spot, strike step, walls, magnet. "
        "Always check liquidity, bid/ask, and your own risk limits."
    )

    strikes = df["strike"]
    step = _infer_strike_step(strikes)
    S = float(spot)

    # offsets (in steps) so it works for 0.5 steps or 5 steps
    short_offset = 2
    long_offset = 5

    # Choose short strikes relative to spot but snapped to chain
    bull_short_put = _nearest_strike(strikes, _round_to_step(S - short_offset * step, step))
    bull_long_put = _nearest_strike(strikes, _round_to_step(bull_short_put - long_offset * step, step))

    bear_short_call = _nearest_strike(strikes, _round_to_step(S + short_offset * step, step))
    bear_long_call = _nearest_strike(strikes, _round_to_step(bear_short_call + long_offset * step, step))

    # Debit spreads keyed off magnet/walls
    ref_up = float(call_wall) if call_wall is not None else (float(magnet) if magnet is not None else S)

    bull_call_long = _nearest_strike(strikes, _round_to_step(min(S, ref_up) - 1 * step, step))
    bull_call_short = _nearest_strike(strikes, _round_to_step(bull_call_long + 4 * step, step))

    bear_put_long = _nearest_strike(strikes, _round_to_step((float(magnet) if magnet is not None else S) + 1 * step, step))
    bear_put_short = _nearest_strike(strikes, _round_to_step(bear_put_long - 4 * step, step))

    # Spread widths (strike distance)
    bull_put_width = abs(bull_short_put - bull_long_put)
    bear_call_width = abs(bear_long_call - bear_short_call)
    bull_call_width = abs(bull_call_short - bull_call_long)
    bear_put_width = abs(bear_put_long - bear_put_short)

    colA, colB, colC = st.columns([1.2, 1.2, 1.6])
    with colA:
        risk_budget = st.number_input("Risk budget per trade (USD)", min_value=0.0, value=150.0, step=25.0)
    with colB:
        credit_est = st.number_input("Credit estimate (optional, per spread)", min_value=0.0, value=0.0, step=0.05)
    with colC:
        st.write("**Sizing uses max-loss math**")
        st.caption("Credit spreads: max loss ≈ (width − credit) × 100. If you leave credit=0, it’s worst-case sizing.")

    size_bull_put = _sizing_math(risk_budget, bull_put_width, credit_est=credit_est)
    size_bear_call = _sizing_math(risk_budget, bear_call_width, credit_est=credit_est)

    st.markdown("#### ✅ Bullish Templates")
    st.markdown(
        "\n".join([
            _format_spread("Bull Put Credit (bullish / neutral)", "Sell Put", bull_short_put, "Buy Put", bull_long_put),
            f"- Why: benefits if price **stays above** short strike; often fits **pinning / chop** regimes.\n"
            f"- Strike step detected: `{step:.2f}`; width: `{bull_put_width:.2f}`",
            f"- Conservative sizing (educational): max loss ≈ `${size_bull_put['max_loss_per_spread']:.0f}` per 1 contract → "
            f"**{size_bull_put['contracts']}** contract(s) for `${risk_budget:.0f}` risk budget."
        ])
    )

    st.markdown(
        "\n".join([
            _format_spread("Bull Call Debit (bullish momentum)", "Buy Call", bull_call_long, "Sell Call", bull_call_short),
            "- Why: used when you expect **break & hold above magnet/call wall** (momentum regime).",
            "- Sizing: max loss = **debit paid × 100**. Use your broker’s mid price to compute contracts."
        ])
    )

    st.markdown("#### ✅ Bearish Templates")
    st.markdown(
        "\n".join([
            _format_spread("Bear Call Credit (bearish / neutral)", "Sell Call", bear_short_call, "Buy Call", bear_long_call),
            f"- Why: benefits if price **stays below** short strike; often fits **resistance / rejection** regimes.\n"
            f"- Strike step detected: `{step:.2f}`; width: `{bear_call_width:.2f}`",
            f"- Conservative sizing (educational): max loss ≈ `${size_bear_call['max_loss_per_spread']:.0f}` per 1 contract → "
            f"**{size_bear_call['contracts']}** contract(s) for `${risk_budget:.0f}` risk budget."
        ])
    )

    st.markdown(
        "\n".join([
            _format_spread("Bear Put Debit (bearish momentum)", "Buy Put", bear_put_long, "Sell Put", bear_put_short),
            "- Why: used when you expect **lose magnet / break lower** (momentum downside).",
            "- Sizing: max loss = **debit paid × 100**. Use your broker’s mid price to compute contracts."
        ])
    )

    st.markdown("#### 🧭 How to choose a template (Educational logic)")
    if put_wall is None or call_wall is None:
        st.info(
            "Walls were not detected (missing OI). In that case, prefer: "
            "• If price is ranging: credit spreads (wider, small size) "
            "• If price is breaking: debit spreads (defined risk)."
        )
        return

    pw = float(put_wall)
    cw = float(call_wall)
    m = float(magnet) if magnet is not None else S

    directional = (
        "call-side pressure" if (total_charm > 0 and total_vanna > 0) else
        "put-side pressure" if (total_charm < 0 and total_vanna < 0) else
        "mixed pressure"
    )

    if (S >= pw) and (S <= cw) and abs(S - m) <= 2 * step:
        st.success(
            f"Pinning zone: spot `{S:.2f}` is near magnet `{m:.2f}` inside `{pw:.2f} → {cw:.2f}`. "
            f"Education: chop risk is higher; **credit spreads** often match range/pin conditions. "
            f"Proxy tone: {directional}."
        )
    elif S > cw:
        st.warning(
            f"Breakout regime: spot `{S:.2f}` is above call wall `{cw:.2f}`. "
            f"Education: momentum conditions can favor **bull call debit** templates; manage risk if it re-enters the range."
        )
    elif S < pw:
        st.warning(
            f"Breakdown regime: spot `{S:.2f}` is below put wall `{pw:.2f}`. "
            f"Education: momentum conditions can favor **bear put debit** templates; manage risk if it re-enters the range."
        )
    else:
        st.info(
            f"Inside range but not pinned: `{pw:.2f} → {cw:.2f}`. "
            f"Education: use **where price is** (near resistance vs support) to select bear-call vs bull-put templates. "
            f"Proxy tone: {directional}."
        )


# ----------------------------
# Main render
# ----------------------------

def render_tab_vanna_charm(symbol, date, spot, hist_df=None):
    st.subheader(f"🌊 Vanna & Charm (Proxy Map): {symbol}")

    with st.expander("📌 Reminder (your API columns) + what this tab is doing", expanded=True):
        st.markdown(
            """
**Your chain provides (Calls & Puts):**  
**Latest, Bid, Ask, Change, Volume, Open Int, IV, Last Trade, Strike**.

**This tab converts that into:**  
- **Call OI / Put OI** (from the duplicated “Open Int” columns you already have)  
- **Call IV / Put IV** (if available)

**Important:**  
- **Net Dealer Vanna (proxy)** and **Net Dealer Charm (proxy)** are **NOT Open Interest** and **NOT contracts**.  
- They are **educational “dealer pressure / gravity” scores** computed **from OI (+ IV if available)** near spot.  
- If OI is clustered near a key strike, you can see large proxy values even if spot is not exactly on that strike.
            """
        )

    if hist_df is not None:
        _trend_context_block(float(spot), hist_df)
    else:
        st.markdown("### 📊 Trend Context (Educational)")
        st.info(
            "📊 Trend Context: Moving average columns were not detected in this table.\n\n"
            "If you add columns like MA20, MA50, MA200, this panel will explain the trend context automatically."
        )

    with st.spinner("Fetching chain + computed exposure table..."):
        r_val = st.session_state.get("r_in", 0.05)
        q_val = st.session_state.get("q_in", 0.0)
        api_result = fetch_weekly_gex(symbol, date, spot, r=r_val, q=q_val)

    if not isinstance(api_result, dict) or not api_result.get("success"):
        st.error("Greeks/chain not available for this ticker/date combo.")
        return

    rows = _unwrap_rows(api_result)
    if not rows:
        st.error("Chain parse error: No rows found inside API response.")
        st.write("Top-level keys detected:", list(api_result.keys()))
        st.write("Raw 'data' type:", type(api_result.get("data")).__name__)
        st.code(str(api_result)[:600])
        return

    try:
        df = _parse_chain_df(rows)
    except Exception as e:
        st.error(f"Chain parse error: {e}")
        sample_df = pd.DataFrame(rows)
        st.write("Detected columns:")
        st.write(list(sample_df.columns))
        st.write("First row sample:")
        st.write(sample_df.head(1))
        return

    if df.empty:
        st.warning("Parsed chain dataframe is empty after processing.")
        return

    # ----------------------------
    # Build proxies
    # ----------------------------
    S = float(spot)

    width = max(S * 0.02, 1.0)  # 2% of spot, min 1.0
    dist = (df["strike"] - S).abs()
    w = np.exp(-((dist / width) ** 2))

    call_iv = pd.to_numeric(df["call_iv"], errors="coerce").fillna(0.0)
    put_iv = pd.to_numeric(df["put_iv"], errors="coerce").fillna(0.0)

    # normalize IV: 35 -> 0.35
    call_iv = np.where(call_iv > 3.0, call_iv / 100.0, call_iv)
    put_iv = np.where(put_iv > 3.0, put_iv / 100.0, put_iv)

    df["vanna_proxy"] = w * ((df["call_oi"] * call_iv) - (df["put_oi"] * put_iv)) * S * 0.01
    df["charm_proxy"] = w * (df["call_oi"] - df["put_oi"])

    total_vanna = float(df["vanna_proxy"].sum())
    total_charm = float(df["charm_proxy"].sum())

    put_wall, call_wall, magnet = _calc_levels(df, S)

    st.markdown("### 🧭 Market Regime Summary")
    st.success(_regime_sentence(S, put_wall, call_wall, total_vanna, total_charm))

    with st.expander("📌 How to read this (using current levels)", expanded=True):
        if put_wall is None or call_wall is None:
            st.markdown(
                f"""
**Current spot:** `{S:.2f}`

I couldn’t detect **Put Wall / Call Wall** from the current chain rows (missing OI).  
Once OI is present, this box will automatically show the breakout/breakdown levels.
                """
            )
        else:
            st.markdown(
                f"""
### How to interpret today’s structure

**Current spot:** `{S:.2f}`  
**Put Wall (highest Put OI):** `{float(put_wall):.2f}`  
**Call Wall (highest Call OI):** `{float(call_wall):.2f}`  
**Magnet (highest pinning pressure):** `{float(magnet):.2f}`

---

### Scenario 1 — Inside the range

**If price stays between `{float(put_wall):.2f}` and `{float(call_wall):.2f}` and Vanna/Charm are elevated:**

🧲 **Market is pinned near a key strike; expect chop unless a breakout occurs.**  
This usually means **range trading / patience** works better than chasing.

---

### Scenario 2 — Bullish breakout

**If price breaks and holds ABOVE `{float(call_wall):.2f}`:**

🚀 **Bullish breakout zone — upside move likely to accelerate.**  
Because hedging flows can flip into **momentum-chasing**.

---

### Scenario 3 — Bearish breakdown

**If price breaks and holds BELOW `{float(put_wall):.2f}`:**

🧨 **Bearish breakdown zone — downside move likely to accelerate.**  
Because support hedging is removed and selling can **cascade**.

---

### Reminder

- These numbers are **NOT price targets**
- They are **dealer pressure / gravity zones**
- They help you judge **stall/chop vs accelerate**
                """
            )

    # ----------------------------
    # NEW: Expiry + Timing guidance (Educational)
    # ----------------------------
    with st.expander("🗓️ Expiry + Timing (Pattern Match) — Educational", expanded=True):
        _timing_expiry_guidance_block(symbol, df, S, put_wall, call_wall, magnet, total_vanna, total_charm)

    # ----------------------------
    # Trade templates (Educational) - for ANY ticker
    # ----------------------------
    with st.expander("🧩 Trade Templates (Bull/Bear Spreads + Sizing Math) — Educational", expanded=True):
        _trade_templates_block(symbol, df, S, put_wall, call_wall, magnet, total_vanna, total_charm)

    # ----------------------------
    # Metrics
    # ----------------------------
    c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 2])

    with c1:
        st.metric("Spot", f"{S:.2f}")

    with c2:
        st.markdown("### Net Dealer Vanna (proxy)")
        st.markdown(f"**{total_vanna:,.0f}**")
        st.caption(
            "How it's calculated (proxy):\n\n"
            "For each strike:\n"
            "`(Call OI × Call IV − Put OI × Put IV) × DistanceWeight × Spot × 0.01`\n\n"
            "Then sum across strikes.\n\n"
            "Meaning: **pressure from OI+IV near spot** (not contracts)."
        )

    with c3:
        st.markdown("### Net Dealer Charm (proxy)")
        st.markdown(f"**{total_charm:,.0f}**")
        st.caption(
            "How it's calculated (proxy):\n\n"
            "For each strike:\n"
            "`(Call OI − Put OI) × DistanceWeight`\n\n"
            "Then sum across strikes.\n\n"
            "Meaning: **pinning/drift pressure from OI near spot** (not contracts)."
        )

    with c4:
        st.markdown("### Put Wall")
        st.markdown(f"**{float(put_wall):.2f}**" if put_wall is not None else "**—**")
        st.caption("Strike with **highest Put OI** → typical downside support / acceleration level.")

    with c5:
        st.markdown("### Call Wall")
        st.markdown(f"**{float(call_wall):.2f}**" if call_wall is not None else "**—**")
        st.caption("Strike with **highest Call OI** → typical upside resistance / acceleration level.")

    st.caption(
        "This tab uses proxies because your chain does NOT contain true vanna/charm Greeks. "
        "Inputs used: Strike + Call OI + Put OI + (Call IV / Put IV if available)."
    )

    # ----------------------------
    # Charts (FIXED: labels + annotations no longer overlap)
    # ----------------------------
    left, right = st.columns(2)

    def _apply_axis_layout(fig: go.Figure, title: str, ytitle: str):
        fig.update_layout(
            template="plotly_dark",
            height=480,
            title=dict(text=title, x=0.02, xanchor="left"),
            margin=dict(l=55, r=20, t=55, b=110),
            bargap=0.15,
            xaxis=dict(
                title="Strike",
                type="linear",
                tickmode="auto",
                nticks=9,
                tickangle=-45,
                automargin=True,
                tickfont=dict(size=11),
                tickformat=".0f",
                showgrid=False,
            ),
            yaxis=dict(
                title=ytitle,
                automargin=True,
                showgrid=True,
                zeroline=True,
            ),
            showlegend=False,
        )
        return fig

    def _add_vline(fig: go.Figure, x: float, label: str, dash: str, color: str, xshift: int = 0):
        fig.add_vline(
            x=x,
            line_dash=dash,
            line_color=color,
            line_width=2,
            annotation_text=label,
            annotation_position="top",
            annotation_yshift=18,
            annotation_xshift=xshift,
            annotation_font_size=10,
            annotation_bgcolor="rgba(0,0,0,0.55)",
            annotation_bordercolor="rgba(255,255,255,0.25)",
            annotation_borderpad=3,
        )

    with left:
        st.write("### Net Dealer Vanna (proxy) by Strike")
        fig_v = go.Figure(go.Bar(x=df["strike"], y=df["vanna_proxy"], name="Vanna (proxy)"))

        _add_vline(fig_v, S, "SPOT", "dash", "white", xshift=0)
        if call_wall is not None:
            _add_vline(fig_v, float(call_wall), "CALL WALL", "dot", "orange", xshift=20)
        if put_wall is not None:
            _add_vline(fig_v, float(put_wall), "PUT WALL", "dot", "red", xshift=-20)

        fig_v = _apply_axis_layout(fig_v, "Vanna (proxy)", "Vanna (proxy)")
        st_plot(fig_v)

    with right:
        st.write("### Net Dealer Charm (proxy) by Strike")
        fig_c = go.Figure(go.Bar(x=df["strike"], y=df["charm_proxy"], name="Charm (proxy)"))

        _add_vline(fig_c, S, "SPOT", "dash", "white", xshift=0)
        if magnet is not None:
            _add_vline(fig_c, float(magnet), "MAGNET", "dot", "cyan", xshift=20)
        if call_wall is not None:
            _add_vline(fig_c, float(call_wall), "CALL WALL", "dot", "orange", xshift=40)
        if put_wall is not None:
            _add_vline(fig_c, float(put_wall), "PUT WALL", "dot", "red", xshift=-20)

        fig_c = _apply_axis_layout(fig_c, "Charm (proxy)", "Charm (proxy)")
        st_plot(fig_c)

    with st.expander("Debug / Normalized inputs used (Strike, OI, IV)", expanded=False):
        show = df[["strike", "call_oi", "put_oi", "call_iv", "put_iv", "vanna_proxy", "charm_proxy"]].copy()
        st.dataframe(show, use_container_width=True)
