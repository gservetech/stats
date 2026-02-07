import os
from typing import Optional, Dict, Any

import pandas as pd
import streamlit as st

from ..helpers.calculations import compute_gamma_map_artifacts
from ..helpers.ui_components import st_df

try:
    from twelvedata import TDClient  # optional; only used if TWELVE_API_KEY is set
except Exception:
    TDClient = None


# -----------------------------------------------------------------------------
# Friday Playbook (Options Chain + TwelveData Stock Confirmation) — FULL FILE
# -----------------------------------------------------------------------------
# Options side (PRIMARY):
#   - uses chain_df: call/put volume + OI + IV per strike
#   - derives walls, magnet, flow proxy (Volume / OI)
#
# Stock side (CONFIRMATION):
#   - uses TwelveData 5-min candles (Close/Volume)
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


def _ensure_col(df: pd.DataFrame, canonical: str, aliases: list[str]):
    if canonical in df.columns:
        return
    for a in aliases:
        if a in df.columns:
            df[canonical] = df[a]
            return


def _prep_chain_df(chain_df: pd.DataFrame) -> pd.DataFrame:
    df = chain_df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Normalize common schema variants coming from backend/API sources.
    _ensure_col(df, "strike", ["strike_price", "k"])
    _ensure_col(df, "call_volume", ["call_vol", "calls_volume", "c_volume"])
    _ensure_col(df, "put_volume", ["put_vol", "puts_volume", "p_volume"])
    _ensure_col(df, "call_open_int", ["call_oi", "calls_oi", "call_open_interest"])
    _ensure_col(df, "put_open_int", ["put_oi", "puts_oi", "put_open_interest"])
    _ensure_col(df, "call_iv", ["calls_iv", "call_implied_volatility", "call_ivol"])
    _ensure_col(df, "put_iv", ["puts_iv", "put_implied_volatility", "put_ivol"])

    required = ["strike", "call_volume", "call_open_int", "call_iv", "put_volume", "put_open_int", "put_iv"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"chain_df missing required columns: {missing}")

    # Coerce numerics
    num_cols = [
        c for c in df.columns
        if any(k in c for k in ["bid", "ask", "change", "volume", "open_int", "oi", "iv", "strike"])
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


def _gex_levels(gex_df: pd.DataFrame, spot: float):
    if gex_df is None or gex_df.empty:
        return None
    art = compute_gamma_map_artifacts(gex_df, spot=spot, top_n=10)
    if not art:
        return None
    return {
        "magnet": art.get("magnet"),
        "put_wall": art.get("put_wall"),
        "call_wall": art.get("call_wall"),
        "spot_used": art.get("spot_used"),
    }


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


def _get_twelve_key() -> Optional[str]:
    try:
        key = st.secrets.get("TWELVE_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    key = os.getenv("TWELVE_API_KEY", "").strip()
    return key or None


@st.cache_resource
def _get_twelve_client(api_key: str):
    if TDClient is None or not api_key:
        return None
    return TDClient(apikey=api_key)


@st.cache_data(ttl=20)
def fetch_twelve_candles(symbol: str, api_key: str, interval: str = "5min", outputsize: int = 100) -> pd.DataFrame:
    """
    Fetch intraday candles from TwelveData and normalize columns to:
    t, o, h, l, c, v
    """
    td = _get_twelve_client(api_key)
    if td is None:
        return pd.DataFrame()

    try:
        ts = td.time_series(symbol=symbol, interval=interval, outputsize=outputsize)
        df = ts.as_pandas()
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_index()
    cols = {str(c).lower(): c for c in df.columns}
    c_open = cols.get("open")
    c_high = cols.get("high")
    c_low = cols.get("low")
    c_close = cols.get("close")
    c_volume = cols.get("volume")
    if c_close is None or c_volume is None:
        return pd.DataFrame()

    t_idx = pd.to_datetime(df.index, errors="coerce")
    if isinstance(t_idx, pd.DatetimeIndex):
        if t_idx.tz is None:
            t_idx = t_idx.tz_localize("America/Toronto", nonexistent="shift_forward", ambiguous="NaT")
        else:
            t_idx = t_idx.tz_convert("America/Toronto")

    out = pd.DataFrame({
        "t": t_idx,
        "o": pd.to_numeric(df[c_open], errors="coerce") if c_open is not None else pd.NA,
        "h": pd.to_numeric(df[c_high], errors="coerce") if c_high is not None else pd.NA,
        "l": pd.to_numeric(df[c_low], errors="coerce") if c_low is not None else pd.NA,
        "c": pd.to_numeric(df[c_close], errors="coerce"),
        "v": pd.to_numeric(df[c_volume], errors="coerce"),
    }).dropna(subset=["t", "c", "v"])

    return out


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
        return {"label": "NO_DATA", "note": "No stock candles (market closed or key missing)."}

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
      - stock volume label (SURGE/QUIET/STEADY)
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

    # No stock candles => caution
    if stock_lbl == "NO_DATA":
        label = "CAUTION"
        reason = "No real-time stock volume feed available; rely on options flow only."

    return {"label": label, "reason": reason}


def render_tab_friday_playbook_from_chain(
    symbol: str,
    spot: float,
    chain_df: pd.DataFrame,
    gex_df: Optional[pd.DataFrame] = None,
    twelve_api_key: Optional[str] = None,
    twelve_interval: str = "5min",
    twelve_outputsize: int = 100,
):
    st.subheader("📅 Friday Gamma Playbook (Chain-Driven + TwelveData Confirmation)")

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

    oi_call_wall, oi_put_wall, oi_magnet, call_wall_row, put_wall_row, magnet_row = _walls_and_magnet(df)
    gex_levels = _gex_levels(gex_df, spot)

    if gex_levels:
        call_wall = _num(gex_levels.get("call_wall"), oi_call_wall)
        put_wall = _num(gex_levels.get("put_wall"), oi_put_wall)
        magnet = _num(gex_levels.get("magnet"), oi_magnet)
        spot_used = _num(gex_levels.get("spot_used"), spot)
        source_note = "Using GEX-derived levels (same logic as Gamma Map tab)."
    else:
        call_wall, put_wall, magnet = oi_call_wall, oi_put_wall, oi_magnet
        spot_used = spot
        source_note = "Using OI-derived levels (fallback because GEX levels unavailable)."

    regime, reason = _infer_regime_from_chain(df, spot_used, call_wall, put_wall, magnet)

    st.markdown("## 🧱 Today’s Structure (Walls & Magnet)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Symbol", symbol)
    c2.metric("Spot Used", _fmt(spot_used, 2))
    c3.metric("Call Wall", _fmt(call_wall, 2))
    c4.metric("Put Wall", _fmt(put_wall, 2))
    c5.metric("Magnet", _fmt(magnet, 2))
    st.caption(source_note)

    c6, c7, c8 = st.columns(3)
    total_call_oi = float(df["call_open_int"].fillna(0).sum())
    total_put_oi = float(df["put_open_int"].fillna(0).sum())
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
    c6.metric("Total Call OI", f"{total_call_oi:,.0f}")
    c7.metric("Total Put OI", f"{total_put_oi:,.0f}")
    c8.metric("Put/Call (OI)", f"{pcr_oi:.2f}" if pcr_oi is not None else "N/A")

    st.info(f"*Regime:* {regime} — {reason}")

    # Near-spot options flow snapshot
    opt_flow = _options_flow_snapshot(df, spot_used)

    # ---------------- Stock: TwelveData candles confirmation ----------------
    st.markdown("## 📈 Real-time Stock Volume Confirmation (TwelveData)")

    stock_break = {"label": "NO_DATA", "note": "TwelveData not configured."}
    if twelve_api_key and TDClient is not None:
        candles = fetch_twelve_candles(symbol, twelve_api_key, twelve_interval, twelve_outputsize)
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

            with st.expander("Show TwelveData candles (debug)", expanded=False):
                st_df(candles[["t", "o", "h", "l", "c", "v"]].tail(20))
    else:
        st.info("Set TWELVE_API_KEY to enable real-time stock volume confirmation.")

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
    twelve_interval: str = "5min",
    twelve_outputsize: int = 100,
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
        gex_df=gex_df,
        twelve_api_key=_get_twelve_key(),
        twelve_interval=twelve_interval,
        twelve_outputsize=twelve_outputsize,
    )
