import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

from ..helpers.ui_components import st_df

try:
    import finnhub  # optional; only used if FINNHUB_API_KEY is set
except Exception:
    finnhub = None


# -----------------------------------------------------------------------------
# Friday Playbook (Options Chain + Finnhub Stock Confirmation) — FULL FILE
# -----------------------------------------------------------------------------
# Options side (PRIMARY):
#   - uses chain_df: call/put volume + OI + IV per strike
#   - derives walls, magnet, flow proxy (Volume / OI)
#
# Stock side (CONFIRMATION):
#   - uses Finnhub 5-min candles (Close/Volume)
#   - confirms if breakout has real participation or is likely fake
#
# Rule:
#   ✅ Options volume leads, stock volume confirms.
# -----------------------------------------------------------------------------


def _num(s, default=0.0):
    try:
        if s is None:
            return default
        return float(pd.to_numeric(s, errors="coerce"))
    except Exception:
        return default


def _fmt(x, nd=2):
    try:
        v = float(x)
        return f"{v:,.{nd}f}"
    except Exception:
        return "N/A"


def _prep_chain_df(chain_df: pd.DataFrame) -> pd.DataFrame:
    df = chain_df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = ["strike", "call_volume", "call_open_int", "call_iv", "put_volume", "put_open_int", "put_iv"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"chain_df missing required columns: {missing}")

    # Coerce numerics
    num_cols = [
        c for c in df.columns
        if any(k in c for k in ["bid", "ask", "change", "volume", "open_int", "iv", "strike"])
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["strike"]).sort_values("strike")

    # Derived features
    df["total_oi"] = df["call_open_int"].fillna(0) + df["put_open_int"].fillna(0)
    df["oi_skew"] = df["put_open_int"].fillna(0) - df["call_open_int"].fillna(0)

    df["call_flow_proxy"] = df["call_volume"].fillna(0) / (df["call_open_int"].fillna(0) + 1)
    df["put_flow_proxy"] = df["put_volume"].fillna(0) / (df["put_open_int"].fillna(0) + 1)
    df["flow_proxy"] = df["call_flow_proxy"] + df["put_flow_proxy"]

    return df


def _walls_and_magnet(df: pd.DataFrame):
    call_wall_row = df.loc[df["call_open_int"].fillna(0).idxmax()] if not df.empty else None
    put_wall_row = df.loc[df["put_open_int"].fillna(0).idxmax()] if not df.empty else None
    magnet_row = df.loc[df["total_oi"].fillna(0).idxmax()] if not df.empty else None

    call_wall = _num(call_wall_row["strike"], None) if call_wall_row is not None else None
    put_wall = _num(put_wall_row["strike"], None) if put_wall_row is not None else None
    magnet = _num(magnet_row["strike"], None) if magnet_row is not None else None

    return call_wall, put_wall, magnet, call_wall_row, put_wall_row, magnet_row


def _near(x, y, band_pct=0.006):
    x = _num(x, None)
    y = _num(y, None)
    if x is None or y is None or x == 0:
        return False
    return abs(x - y) / abs(x) <= band_pct


def _infer_regime_from_chain(df: pd.DataFrame, spot: float, call_wall: float, put_wall: float, magnet: float):
    s = _num(spot, None)
    if s is None or call_wall is None or put_wall is None:
        return "UNKNOWN", "Not enough data (spot/walls missing)."

    low = min(call_wall, put_wall)
    high = max(call_wall, put_wall)

    between = low <= s <= high
    outside = (s < low) or (s > high)
    near_magnet = _near(s, magnet)

    # Flow proxy near spot (closest 3 strikes)
    df2 = df.copy()
    df2["dist"] = (df2["strike"] - s).abs()
    near = df2.sort_values("dist").head(3)

    near_put_flow = float(near["put_flow_proxy"].fillna(0).mean()) if not near.empty else 0.0
    near_call_flow = float(near["call_flow_proxy"].fillna(0).mean()) if not near.empty else 0.0

    total_call_oi = float(df["call_open_int"].fillna(0).sum())
    total_put_oi = float(df["put_open_int"].fillna(0).sum())
    oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    if outside:
        return "BREAK / MOMENTUM", "Spot is outside the wall range → hedging can amplify moves."

    if between and near_magnet:
        if oi_ratio is None or (0.75 <= oi_ratio <= 1.35):
            return "PIN / CONTROL", "Spot is near the magnet between walls → pin/chop risk is high."
        return "PIN but SKEWED", "Near magnet, but OI is skewed → pin risk + sudden break risk."

    if between and (near_put_flow > 1.5 or near_call_flow > 1.5):
        side = "puts" if near_put_flow > near_call_flow else "calls"
        return "RISKY CHOP (FLOW BUILDING)", f"Between walls but {side} flow proxy is high → whipsaw / late break risk."

    return "MIXED", "No strong pin or break signals; wait for confirmation."


def _rulebook_markdown():
    return """
## 📜 Friday Gamma Rulebook (READ THIS EVERY FRIDAY)

*Fridays are GAMMA days — not conviction days.*  
Price can be driven by *dealer hedging + OI magnets + intraday option flow*, not “news” or feelings.

### 🔑 Core Truths
1. Walls control price until they break
2. Moving walls are targets, not resistance
3. Spreads near walls lose on Fridays (they remove gamma)
4. Long gamma beats being right
5. No break = no trade

### ⏰ Best Trading Windows (Toronto time)
- ✅ 10:30 AM – 1:30 PM
- ✅ 2:30 PM – 3:30 PM (fresh break only)
- 🚫 9:30 – 9:45 AM
- 🚫 3:30 – 4:00 PM
"""


def _options_vs_stock_volume_markdown():
    return """
## 🔥 Options Volume vs Stock Volume (How to use both)

*Options Volume (PRIMARY):* shows where dealer hedging pressure is building (flow proxy = Volume/OI).  
*Stock Volume (CONFIRMATION):* tells you if the hedge pressure is actually moving the tape.

✅ *Rule:* Options volume leads, stock volume confirms.
"""


@st.cache_resource
def _get_finnhub_client(api_key: str):
    if finnhub is None:
        return None
    return finnhub.Client(api_key=api_key)


@st.cache_data(ttl=20)
def fetch_finnhub_candles(symbol: str, api_key: str, resolution: str = "5", lookback_sec: int = 1800) -> pd.DataFrame:
    """
    Fetch last lookback_sec worth of candles from Finnhub.
    resolution: '1','5','15','30','60','D'
    """
    client = _get_finnhub_client(api_key)
    if client is None:
        return pd.DataFrame()

    now = int(time.time())
    past = now - int(lookback_sec)

    res = client.stock_candles(symbol, resolution, past, now)
    if not isinstance(res, dict) or res.get("s") == "no_data":
        return pd.DataFrame()

    df = pd.DataFrame(res)
    # expected keys: c,h,l,o,s,t,v
    if df.empty or "t" not in df.columns:
        return pd.DataFrame()

    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/Toronto")
    return df


def breakout_status_from_candles(candles: pd.DataFrame) -> Dict[str, Any]:
    """
    Produces:
      - latest close, high, low
      - latest volume
      - prev volume
      - vol_ratio_prev = latest / prev
      - vol_ratio_avg  = latest / avg(last N)
      - label: SURGE / QUIET / STEADY / NO_DATA
    """
    if candles is None or candles.empty or "v" not in candles.columns or "c" not in candles.columns:
        return {"label": "NO_DATA", "note": "No Finnhub candles (market closed or key missing)."}

    df = candles.copy().sort_values("t")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    latest_v = float(latest.get("v", 0))
    prev_v = float(prev.get("v", 0))
    avg_v = float(df["v"].tail(min(20, len(df))).mean()) if len(df) >= 2 else latest_v

    vol_ratio_prev = (latest_v / prev_v) if prev_v > 0 else None
    vol_ratio_avg = (latest_v / avg_v) if avg_v > 0 else None

    label = "STEADY"
    note = "Volume is stable."
    if vol_ratio_prev is not None and vol_ratio_prev >= 1.5:
        label = "SURGE"
        note = "🚀 Volume SURGE vs previous candle (>= 1.5x). Breakout confirmation is stronger."
    elif vol_ratio_prev is not None and vol_ratio_prev <= 0.8:
        label = "QUIET"
        note = "💤 Volume QUIET vs previous candle (<= 0.8x). Fake-out risk is higher."

    return {
        "label": label,
        "note": note,
        "time": latest.get("t"),
        "close": float(latest.get("c", 0)),
        "high": float(latest.get("h", 0)),
        "low": float(latest.get("l", 0)),
        "vol": latest_v,
        "prev_vol": prev_v,
        "avg_vol": avg_v,
        "vol_ratio_prev": vol_ratio_prev,
        "vol_ratio_avg": vol_ratio_avg,
    }


def _options_flow_snapshot(df: pd.DataFrame, spot: float) -> Dict[str, Any]:
    """
    Looks at the closest 3 strikes to spot and returns directional flow hints.
    """
    s = _num(spot, None)
    if s is None or df is None or df.empty:
        return {"note": "No options flow snapshot available."}

    d = df.copy()
    d["dist"] = (d["strike"] - s).abs()
    near = d.sort_values("dist").head(3)

    near_call_flow = float(near["call_flow_proxy"].fillna(0).mean()) if not near.empty else 0.0
    near_put_flow = float(near["put_flow_proxy"].fillna(0).mean()) if not near.empty else 0.0

    side = "CALLS" if near_call_flow >= near_put_flow else "PUTS"
    strength = max(near_call_flow, near_put_flow)

    return {
        "near_call_flow": near_call_flow,
        "near_put_flow": near_put_flow,
        "dominant_side": side,
        "dominant_strength": strength,
        "note": f"Near-spot flow proxy: calls={near_call_flow:.2f}, puts={near_put_flow:.2f} (dominant: {side})"
    }


def _go_nogo(options_regime: str, opt_flow: Dict[str, Any], stock_break: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines:
      - options regime
      - options dominant near-spot flow
      - Finnhub stock volume label (SURGE/QUIET/STEADY)
    into an actionable message.
    """
    label = "NO-GO"
    reason = "Wait for confirmation."

    stock_lbl = stock_break.get("label", "NO_DATA")
    flow_side = opt_flow.get("dominant_side", "CALLS")
    flow_strength = float(opt_flow.get("dominant_strength", 0.0))

    # Break regime + stock volume surge => GO
    if options_regime == "BREAK / MOMENTUM" and stock_lbl in ("SURGE", "STEADY"):
        if flow_strength >= 1.0:
            label = "GO"
            reason = f"Break regime + stock volume {stock_lbl} + strong near-spot {flow_side} flow."
        else:
            label = "CAUTION"
            reason = f"Break regime + stock volume {stock_lbl}, but near-spot option flow is not strong yet."

    # Risky chop + quiet => NO-GO
    if options_regime in ("PIN / CONTROL", "RISKY CHOP (FLOW BUILDING)") and stock_lbl == "QUIET":
        label = "NO-GO"
        reason = "Pin/chop risk + stock volume is quiet → high fake-out risk."

    # No data from Finnhub => caution
    if stock_lbl == "NO_DATA":
        label = "CAUTION"
        reason = "No real-time stock volume feed available; rely on options flow only."

    return {"label": label, "reason": reason}


def _get_finnhub_key() -> Optional[str]:
    try:
        key = st.secrets.get("FINNHUB_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("FINNHUB_API_KEY")


def render_tab_friday_playbook_from_chain(
    symbol: str,
    spot: float,
    chain_df: pd.DataFrame,
    finnhub_api_key: Optional[str] = None,
    finnhub_resolution: str = "5",
    finnhub_lookback_sec: int = 1800,
):
    st.subheader("📅 Friday Gamma Playbook (Chain-Driven + Finnhub Confirmation)")

    st.warning("📌 READ the rulebook below BEFORE placing any Friday trade.")
    st.markdown(_rulebook_markdown())
    st.markdown(_options_vs_stock_volume_markdown())

    if chain_df is None or chain_df.empty:
        st.info("No options chain data provided.")
        return

    # ---------------- Options: walls/flow ----------------
    try:
        df = _prep_chain_df(chain_df)
    except Exception as e:
        st.error(f"Chain data format issue: {e}")
        return

    call_wall, put_wall, magnet, call_wall_row, put_wall_row, magnet_row = _walls_and_magnet(df)
    regime, reason = _infer_regime_from_chain(df, spot, call_wall, put_wall, magnet)

    st.markdown("## 🧱 Today’s Structure (Walls & Magnet)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Symbol", symbol)
    c2.metric("Spot", _fmt(spot, 2))
    c3.metric("Call Wall (max Call OI)", _fmt(call_wall, 2))
    c4.metric("Put Wall (max Put OI)", _fmt(put_wall, 2))
    c5.metric("Magnet (max Total OI)", _fmt(magnet, 2))

    c6, c7, c8 = st.columns(3)
    total_call_oi = float(df["call_open_int"].fillna(0).sum())
    total_put_oi = float(df["put_open_int"].fillna(0).sum())
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
    c6.metric("Total Call OI", f"{total_call_oi:,.0f}")
    c7.metric("Total Put OI", f"{total_put_oi:,.0f}")
    c8.metric("Put/Call (OI)", f"{pcr_oi:.2f}" if pcr_oi is not None else "N/A")

    st.info(f"*Regime:* {regime} — {reason}")

    # Near-spot options flow snapshot
    opt_flow = _options_flow_snapshot(df, spot)

    # ---------------- Stock: Finnhub candles confirmation ----------------
    st.markdown("## 📈 Real-time Stock Volume Confirmation (Finnhub)")

    stock_break = {"label": "NO_DATA", "note": "Finnhub not configured."}
    if finnhub_api_key and finnhub is not None:
        candles = fetch_finnhub_candles(symbol, finnhub_api_key, finnhub_resolution, finnhub_lookback_sec)
        stock_break = breakout_status_from_candles(candles)

        if stock_break.get("label") == "NO_DATA":
            st.info(stock_break.get("note", "No candle data."))
        else:
            d1, d2, d3, d4 = st.columns(4)
            t = stock_break.get("time")
            t_str = t.strftime("%H:%M") if isinstance(t, pd.Timestamp) else "N/A"
            d1.metric("Latest Candle", t_str)
            d2.metric("Close", _fmt(stock_break.get("close"), 2))
            d3.metric("5m Volume", f"{stock_break.get('vol', 0):,.0f}")
            vr = stock_break.get("vol_ratio_prev")
            d4.metric("Vol vs Prev", f"{vr:.2f}x" if vr is not None else "N/A")
            st.caption(stock_break.get("note", ""))

            with st.expander("Show Finnhub candles (debug)", expanded=False):
                st_df(candles[["t", "o", "h", "l", "c", "v"]].tail(20))
    else:
        st.info("Set FINNHUB_API_KEY to enable real-time stock volume confirmation.")

    # ---------------- GO / NO-GO ----------------
    st.markdown("## ✅ Trade Readiness (GO / NO-GO)")
    go = _go_nogo(regime, opt_flow, stock_break)

    if go["label"] == "GO":
        st.success(f"*GO* — {go['reason']}")
    elif go["label"] == "CAUTION":
        st.warning(f"*CAUTION* — {go['reason']}")
    else:
        st.error(f"*NO-GO* — {go['reason']}")

    st.caption(opt_flow.get("note", ""))

    # ---------------- Options flow table ----------------
    st.markdown("## 📡 Option Flow Proxy (PRIMARY on Fridays)")
    top_flow = df.copy().sort_values("flow_proxy", ascending=False).head(12)
    show_cols = [
        "strike",
        "call_volume", "call_open_int", "call_flow_proxy",
        "put_volume", "put_open_int", "put_flow_proxy",
        "flow_proxy",
        "call_iv", "put_iv",
    ]
    st_df(top_flow[[c for c in show_cols if c in top_flow.columns]])

    # Wall details
    with st.expander("🧱 Wall / Magnet Detail Rows", expanded=False):
        cols = [
            "strike",
            "call_open_int", "call_volume", "call_iv", "call_flow_proxy",
            "put_open_int", "put_volume", "put_iv", "put_flow_proxy",
            "flow_proxy",
        ]
        view = pd.concat(
            [
                pd.DataFrame([call_wall_row]) if call_wall_row is not None else pd.DataFrame(),
                pd.DataFrame([put_wall_row]) if put_wall_row is not None else pd.DataFrame(),
                pd.DataFrame([magnet_row]) if magnet_row is not None else pd.DataFrame(),
            ],
            ignore_index=True
        ).drop_duplicates(subset=["strike"])
        if not view.empty:
            st_df(view[[c for c in cols if c in view.columns]])

    st.markdown("## 🚫 Friday 'Do Not' List")
    st.markdown("""
- Don’t trade the first 15 minutes
- Don’t hold 0DTE into close
- Don’t sell premium near magnets
- Don’t use tight debit spreads near walls
- Don’t fight wall migration
""")

    with st.expander("📚 Full Chain (sorted by strike)", expanded=False):
        st_df(df.sort_values("strike"))

    st.caption("Educational only — high risk. Options volume leads; stock volume confirms.")


def render_tab_friday_playbook(
    symbol: str,
    spot: float,
    chain_df: pd.DataFrame,
    gex_df: Optional[pd.DataFrame] = None,
):
    """
    Backward-compatible entrypoint used by app.py.
    gex_df is accepted for compatibility but not used in this chain-driven version.
    """
    _ = gex_df
    render_tab_friday_playbook_from_chain(
        symbol=symbol,
        spot=spot,
        chain_df=chain_df,
        finnhub_api_key=_get_finnhub_key(),
    )
