import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from stats_app.helpers.api_client import fetch_weekly_gex
from stats_app.helpers.calculations import build_gamma_levels

# ---------------------------
# Interpretation Engine (Educational)
# ---------------------------
# This tab turns the same data you already fetch (spot, price history,
# options chain) into a simple, repeatable "What is happening today?"
# explanation that works on ANY symbol.
#
# IMPORTANT:
# - This is educational market structure context, NOT financial advice.
# - It uses proxies because your chain payload may not include true dealer
#   positioning, true gamma exposure, or true order-flow delta.


@dataclass
class StructureLevels:
    put_wall: Optional[float] = None
    call_wall: Optional[float] = None
    magnet: Optional[float] = None
    box_low: Optional[float] = None
    box_high: Optional[float] = None


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name that exists in df (case-insensitive match)."""
    if df is None or df.empty:
        return None

    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = lower_map.get(name.lower())
        if c:
            return c
    return None


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _compute_walls_and_magnet(chain_df: pd.DataFrame) -> StructureLevels:
    # Common column patterns seen across vendors
    strike_col = _first_existing_col(chain_df, ["strike", "Strike", "STRIKE"])

    call_oi_col = _first_existing_col(
        chain_df,
        [
            "call_oi",
            "calloi",
            "call_open_interest",
            "callopeninterest",
            "callOpenInterest",
            "Call OI",
            "CallOpenInterest",
            "callOpenInt",
        ],
    )
    put_oi_col = _first_existing_col(
        chain_df,
        [
            "put_oi",
            "putoi",
            "put_open_interest",
            "putopeninterest",
            "putOpenInterest",
            "Put OI",
            "PutOpenInterest",
            "putOpenInt",
        ],
    )

    if not strike_col:
        return StructureLevels()

    strikes = _to_numeric(chain_df[strike_col])

    call_oi = _to_numeric(chain_df[call_oi_col]) if call_oi_col else None
    put_oi = _to_numeric(chain_df[put_oi_col]) if put_oi_col else None

    levels = StructureLevels()

    if call_oi is not None and call_oi.notna().any():
        idx = call_oi.fillna(-1).astype(float).idxmax()
        levels.call_wall = float(strikes.loc[idx]) if idx in strikes.index else None

    if put_oi is not None and put_oi.notna().any():
        idx = put_oi.fillna(-1).astype(float).idxmax()
        levels.put_wall = float(strikes.loc[idx]) if idx in strikes.index else None

    # Magnet proxy: highest total OI (pinning / liquidity)
    if call_oi is not None and put_oi is not None:
        total_oi = call_oi.fillna(0).astype(float) + put_oi.fillna(0).astype(float)
        if total_oi.notna().any():
            idx = total_oi.idxmax()
            levels.magnet = float(strikes.loc[idx]) if idx in strikes.index else None

    # Box: walls define a range (if both exist)
    if levels.put_wall is not None and levels.call_wall is not None:
        levels.box_low = float(min(levels.put_wall, levels.call_wall))
        levels.box_high = float(max(levels.put_wall, levels.call_wall))

    return levels


def _compute_gex_levels(symbol: str, expiry_date: Optional[str], spot: float) -> Optional[StructureLevels]:
    if not symbol or not expiry_date or spot is None:
        return None

    r_val = st.session_state.get("r_in", 0.041)
    q_val = st.session_state.get("q_in", 0.004)
    gex_result = fetch_weekly_gex(symbol, expiry_date, spot, r=r_val, q=q_val)
    if not gex_result.get("success"):
        return None

    gex_payload = gex_result.get("data", {})
    gex_df = pd.DataFrame(gex_payload.get("data", []) or [])
    if gex_df.empty:
        return None

    levels = build_gamma_levels(gex_df, spot=spot, top_n=5)
    if not levels:
        return None

    out = StructureLevels()

    magnets = levels.get("magnets")
    if isinstance(magnets, pd.DataFrame) and not magnets.empty:
        out.magnet = float(magnets.iloc[0]["strike"])

    box = levels.get("gamma_box") or {}
    if box.get("lower") is not None:
        out.put_wall = float(box["lower"])
    if box.get("upper") is not None:
        out.call_wall = float(box["upper"])

    if out.put_wall is None and levels.get("major_put_wall") is not None:
        out.put_wall = float(levels["major_put_wall"])
    if out.call_wall is None and levels.get("major_call_wall") is not None:
        out.call_wall = float(levels["major_call_wall"])

    if out.put_wall is not None and out.call_wall is not None:
        out.box_low = float(min(out.put_wall, out.call_wall))
        out.box_high = float(max(out.put_wall, out.call_wall))

    return out


def _safe_pct(x: Optional[float], y: Optional[float]) -> Optional[float]:
    if x is None or y is None or y == 0:
        return None
    return 100.0 * (x / y)


def _compute_vwap_proxy(hist_df: pd.DataFrame) -> Optional[float]:
    if hist_df is None or hist_df.empty:
        return None

    close_col = _first_existing_col(hist_df, ["Close", "close"])
    high_col = _first_existing_col(hist_df, ["High", "high"])
    low_col = _first_existing_col(hist_df, ["Low", "low"])
    vol_col = _first_existing_col(hist_df, ["Volume", "volume"])

    if not close_col:
        return None

    px = hist_df.copy()
    close = _to_numeric(px[close_col])

    # If we have Volume + (High/Low), compute typical-price VWAP over last ~2 sessions
    if vol_col and (high_col and low_col):
        vol = _to_numeric(px[vol_col]).fillna(0)
        high = _to_numeric(px[high_col])
        low = _to_numeric(px[low_col])
        typical = (high + low + close) / 3.0

        window = min(len(px), 78)  # ~2 trading days of 5m bars OR last 2 daily bars
        typical = typical.tail(window)
        vol = vol.tail(window)

        denom = float(vol.sum())
        if denom > 0:
            return float((typical * vol).sum() / denom)

    # Fallback: use MA20 as a "VWAP-like" structure line
    if len(close.dropna()) >= 20:
        return float(close.rolling(20).mean().iloc[-1])

    return float(close.iloc[-1]) if close.notna().any() else None


def _market_regime(spot: float, levels: StructureLevels) -> Tuple[str, str]:
    # Proxy logic:
    # - Inside box => pinning / mean reversion bias
    # - Outside box => trend / acceleration bias
    if levels.box_low is not None and levels.box_high is not None:
        if levels.box_low <= spot <= levels.box_high:
            return (
                "RANGE / PINNING (proxy for dealer long-gamma)",
                "Inside the wall-to-wall range, moves often stall and chop unless a clean break happens.",
            )
        if spot > levels.box_high:
            return (
                "BREAKOUT ZONE (proxy for dealer short-gamma above walls)",
                "Above the range, hedging flows can amplify moves; breakouts may run faster.",
            )
        return (
            "BREAKDOWN ZONE (proxy for dealer short-gamma below walls)",
            "Below the range, hedging flows can amplify downside; breakdowns may run faster.",
        )

    # If we can't compute box, fall back to magnet proximity
    if levels.magnet is not None:
        dist = abs(spot - levels.magnet)
        if dist <= 0.005 * max(1.0, spot):
            return (
                "PINNING NEAR MAGNET (proxy)",
                "Price is near the highest-liquidity strike; chop/pinning is common.",
            )

    return (
        "MIXED / UNKNOWN (insufficient structure)",
        "Not enough structure fields were found to classify range vs trend reliably.",
    )


def _pressure_proxy(hist_df: pd.DataFrame, spot: float) -> Tuple[str, str, int]:
    """Buy/sell pressure proxy using price + volume.

    Truth: every trade has a buyer and seller, so we can't know "buy vs sell"
    from volume alone without bid/ask. But we CAN build a useful proxy:
      - Up-volume vs Down-volume (based on price change)
      - OBV-like flow (volume signed by price change)
      - Trend + VWAP/MA position
      - Absorption hint (high volume, little progress)
    """
    if hist_df is None or hist_df.empty:
        return ("UNKNOWN", "No price history loaded, so pressure proxy is unavailable.", 0)

    close_col = _first_existing_col(hist_df, ["Close", "close"])
    open_col = _first_existing_col(hist_df, ["Open", "open"])
    high_col = _first_existing_col(hist_df, ["High", "high"])
    low_col = _first_existing_col(hist_df, ["Low", "low"])
    vol_col = _first_existing_col(hist_df, ["Volume", "volume"])

    if not close_col:
        return ("UNKNOWN", "Price history missing Close column.", 0)

    df = hist_df.copy()
    close = _to_numeric(df[close_col])

    # ---- Trend (recent closes slope) ----
    close_clean = close.dropna()
    if len(close_clean) < 3:
        return ("UNKNOWN", "Not enough closes to estimate pressure.", 0)

    recent = close_clean.tail(min(len(close_clean), 20)).values
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0.0

    # ---- VWAP/MA proxy position ----
    vwap = _compute_vwap_proxy(df)
    above_vwap = (vwap is not None) and (spot >= vwap)

    # ---- Volume flow proxies ----
    vol_note = ""
    absorption_note = ""
    if vol_col:
        vol = _to_numeric(df[vol_col]).fillna(0)

        # Define "up" / "down" bars:
        # - Prefer Close vs previous Close (works even if Open is missing)
        # - If Open exists, use Close vs Open for intrabar intent
        if open_col and df[open_col].notna().any():
            o = _to_numeric(df[open_col])
            up_mask = (close > o)
            down_mask = (close < o)
        else:
            prev = close.shift(1)
            up_mask = (close > prev)
            down_mask = (close < prev)

        up_vol = float(vol[up_mask].sum())
        down_vol = float(vol[down_mask].sum())
        total = up_vol + down_vol
        delta = up_vol - down_vol
        dom = None if total <= 0 else (delta / total)

        # OBV-like: sign volume by direction and look at last N slope
        signed = np.where(up_mask, vol.values, np.where(down_mask, -vol.values, 0.0))
        obv = pd.Series(signed, index=df.index).cumsum()
        obv_clean = obv.dropna()
        obv_slope = None
        if len(obv_clean) >= 5:
            obv_recent = obv_clean.tail(min(len(obv_clean), 30)).values
            obv_slope = float(np.polyfit(np.arange(len(obv_recent)), obv_recent, 1)[0])

        # Absorption hint: very high volume bar with small real body
        if open_col and high_col and low_col:
            o = _to_numeric(df[open_col])
            h = _to_numeric(df[high_col])
            l = _to_numeric(df[low_col])
            body = (close - o).abs()
            rng = (h - l).abs().replace(0, np.nan)
            body_ratio = (body / rng).fillna(0)
            vol_z = (vol - vol.rolling(50).mean()) / (vol.rolling(50).std().replace(0, np.nan))
            # last bar only
            if len(df) >= 1 and pd.notna(vol_z.iloc[-1]):
                if vol_z.iloc[-1] >= 1.5 and body_ratio.iloc[-1] <= 0.25:
                    absorption_note = (
                        "⚠️ **Absorption hint:** volume is elevated but price made little progress "
                        "(big activity, small candle body). This often means a strong passive side is absorbing."
                    )

        # Build readable volume note
        if total > 0:
            vol_note = (
                f"Up-volume: `{up_vol:,.0f}` | Down-volume: `{down_vol:,.0f}` | "
                f"Volume-delta proxy: `{delta:,.0f}`"
            )
            if dom is not None:
                if dom >= 0.12:
                    vol_note += " → **buyers more aggressive (proxy)**"
                elif dom <= -0.12:
                    vol_note += " → **sellers more aggressive (proxy)**"
                else:
                    vol_note += " → **balanced / two-sided (proxy)**"

            if obv_slope is not None:
                vol_note += f" | OBV-slope proxy: `{obv_slope:,.0f}`"
        else:
            vol_note = "Volume column found, but usable up/down volume could not be computed."

    # ---- Final classification (combine trend + vwap + volume) ----
    # Score: trend and VWAP are the anchors; volume flow is a booster.
    score = 0
    if slope > 0:
        score += 1
    elif slope < 0:
        score -= 1

    if above_vwap:
        score += 1
    else:
        score -= 1

    # Volume booster
    if vol_col and "buyers more aggressive" in vol_note:
        score += 1
    elif vol_col and "sellers more aggressive" in vol_note:
        score -= 1

    note_parts = []
    # Narrative
    if score >= 2:
        label = "BUYERS IN CONTROL (proxy)"
        note_parts.append("Price trend + structure suggest buyers are pressing (and holding) levels.")
    elif score <= -2:
        label = "SELLERS IN CONTROL (proxy)"
        note_parts.append("Price trend + structure suggest sellers are pressing (and rejecting) levels.")
    else:
        label = "MIXED / CHOP (proxy)"
        note_parts.append("Signals are mixed — common near magnets/walls or during transition.")

    if vwap is not None:
        note_parts.append(f"VWAP/MA proxy: `{vwap:.2f}` (spot {'above' if above_vwap else 'below'}).")

    if vol_note:
        note_parts.append(vol_note)

    if absorption_note:
        note_parts.append(absorption_note)

    # Clamp score to -3..+3 (should already be within range)
    score = int(max(-3, min(3, score)))
    return (label, "  \\n".join(note_parts), score)


def _strategy_ideas(spot: float, levels: StructureLevels) -> List[str]:
    """Educational strategy templates (no sizing, no guarantees)."""
    ideas: List[str] = []

    if levels.box_low is not None and levels.box_high is not None:
        inside = levels.box_low <= spot <= levels.box_high
        near_call_wall = levels.call_wall is not None and abs(spot - levels.call_wall) <= 0.01 * spot
        near_put_wall = levels.put_wall is not None and abs(spot - levels.put_wall) <= 0.01 * spot

        if inside:
            ideas.append("Range environment template: consider *defined-risk* premium-selling structures (e.g., credit spreads / iron condor) **only if** you understand assignment risk.")
        else:
            ideas.append("Outside-range environment template: consider *defined-risk* directional structures (e.g., debit spreads) aligned with the break direction.")

        if near_call_wall:
            ideas.append("Near call wall: upside may stall/pin — many traders look for rejection/fade setups **or** wait for a clean acceptance above the wall.")
        if near_put_wall:
            ideas.append("Near put wall: downside may stall/bounce — many traders look for rejection/bounce setups **or** wait for a clean acceptance below the wall.")

    if levels.magnet is not None:
        ideas.append("Magnet concept: the highest-liquidity strike can act like a gravity zone (pinning) as expiry approaches.")

    ideas.append("Risk note: prefer defined-risk spreads over naked options; avoid oversizing; know your max loss before entering.")
    return ideas


from stats_app.tabs.tab_symbol_chart import render_symbol_chart

def render_tab_interpretation_engine(
    symbol: str,
    spot: float,
    chain_df: Optional[pd.DataFrame] = None,
    hist_df: Optional[pd.DataFrame] = None,
    expiry_date: Optional[str] = None,
):
    render_symbol_chart(symbol)
    
    st.markdown("## 🧠 Interpretation Engine (Any Symbol)")
    st.caption(
        "Educational context only — not investment advice. "
        "This uses proxies (walls/magnet from GEX when available; regime from spot vs walls; "
        "pressure from price+VWAP+volume proxy)."
    )

    if chain_df is None or chain_df.empty:
        st.info("No options chain data found for this symbol/date yet.")
        return

    levels = _compute_gex_levels(symbol, expiry_date, spot)
    if levels is None:
        levels = _compute_walls_and_magnet(chain_df)
    regime, regime_note = _market_regime(spot, levels)
    pressure, pressure_note, pressure_score = _pressure_proxy(hist_df, spot)
    vwap = _compute_vwap_proxy(hist_df) if hist_df is not None else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Spot", f"{spot:.2f}")
    with c2:
        st.metric("Put Wall (Lower)", "—" if levels.put_wall is None else f"{levels.put_wall:.2f}")
    with c3:
        st.metric("Call Wall (Upper)", "—" if levels.call_wall is None else f"{levels.call_wall:.2f}")
    with c4:
        st.metric("Main Magnet", "—" if levels.magnet is None else f"{levels.magnet:.2f}")

    if vwap is not None:
        st.write(f"**VWAP / MA proxy:** `{vwap:.2f}`")

    st.markdown("---")

    st.markdown("### 1) Structure")
    if levels.box_low is not None and levels.box_high is not None:
        st.write(f"**Gamma Box proxy:** `{levels.box_low:.2f} → {levels.box_high:.2f}` (walls)")
    else:
        st.write("Box proxy unavailable (missing one or both walls).")

    st.markdown("### 2) Regime")
    st.write(f"**{regime}**")
    st.caption(regime_note)

    st.markdown("### 3) Pressure (proxy)")

    # Pressure Score: -3 (strong selling) → +3 (strong buying)
    score_emoji = "🟢" if pressure_score >= 2 else ("🔴" if pressure_score <= -2 else "⚪")
    st.markdown(
        f"**Volume Pressure Score:** {score_emoji} `{pressure_score:+d}`  \n"
        f"**Pressure label:** **{pressure}**"
    )
    st.markdown(pressure_note)

    with st.expander("What does the -3 → +3 score mean?"):
        st.markdown(
            "- **+3** = strong buying pressure (proxy): up-volume dominates, OBV rising, trend+VWAP supportive.\n"
            "- **+2** = moderate buying pressure (proxy).\n"
            "- **+1** = slight buying bias (proxy).\n"
            "- **0** = neutral / balanced (proxy) — often chop/pinning.\n"
            "- **-1** = slight selling bias (proxy).\n"
            "- **-2** = moderate selling pressure (proxy).\n"
            "- **-3** = strong selling pressure (proxy): down-volume dominates, OBV falling, trend+VWAP weak.\n\n"
            "⚠️ **Important:** This is a proxy. True buy/sell requires bid/ask (footprint) data. "
            "Also, combine this with **walls/magnet**: a strong score **at a wall** can still reverse."
        )


    with st.expander("How the app infers buying vs selling (proxy)"):
        st.markdown(
            "- **Volume alone can’t tell buy vs sell** (every trade has both).\n"
            "- We use a **proxy**: compare **up-volume vs down-volume** (based on price change), plus an **OBV-slope** proxy.\n"
            "- **Up-volume ≫ down-volume** → buyers more aggressive (proxy).\n"
            "- **Down-volume ≫ up-volume** → sellers more aggressive (proxy).\n"
            "- If volume is high but candles show little progress, we flag an **absorption hint**."
        )

    st.markdown("---")

    st.markdown("## 🧲 Dynamic explanations you will see")
    # Build dynamic scenario text using the levels computed
    lines: List[str] = []

    if levels.put_wall is not None and levels.call_wall is not None:
        lines.append(
            f"If spot `{spot:.2f}` is **inside** `{levels.box_low:.2f} → {levels.box_high:.2f}`: expect **chop/pinning** unless a clean break happens."
        )

        lines.append(
            f"If spot **breaks above** call wall `{levels.call_wall:.2f}`: it can become a **bullish breakout zone** (moves may accelerate)."
        )

        lines.append(
            f"If spot **breaks below** put wall `{levels.put_wall:.2f}`: it can become a **bearish breakdown zone** (moves may accelerate)."
        )

    elif levels.magnet is not None:
        lines.append(
            f"If spot `{spot:.2f}` is near magnet `{levels.magnet:.2f}`: expect **pinning / gravity** behavior more often."
        )

    if vwap is not None:
        if spot >= vwap:
            lines.append(f"Spot is **above** VWAP/MA proxy `{vwap:.2f}` → structure is slightly stronger.")
        else:
            lines.append(f"Spot is **below** VWAP/MA proxy `{vwap:.2f}` → structure is slightly weaker.")

    for ln in lines:
        st.write("• " + ln)

    st.markdown("---")

    st.markdown("## 🧩 Strategy templates (educational)")
    for idea in _strategy_ideas(spot, levels):
        st.write("• " + idea)

    with st.expander("📌 Copy/paste summary"):
        summary = {
            "symbol": symbol,
            "expiry": expiry_date,
            "spot": round(float(spot), 4),
            "put_wall": None if levels.put_wall is None else round(float(levels.put_wall), 4),
            "call_wall": None if levels.call_wall is None else round(float(levels.call_wall), 4),
            "magnet": None if levels.magnet is None else round(float(levels.magnet), 4),
            "box": None
            if (levels.box_low is None or levels.box_high is None)
            else [round(float(levels.box_low), 4), round(float(levels.box_high), 4)],
            "vwap_proxy": None if vwap is None else round(float(vwap), 4),
            "regime": regime,
            "pressure_proxy": pressure,
            "notes": {
                "regime": regime_note,
                "pressure": pressure_note,
            },
        }
        st.code(summary, language="json")
