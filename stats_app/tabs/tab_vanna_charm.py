# stats_app/tabs/tab_vanna_charm.py
# ------------------------------------------------------------
# 🌊 Vanna & Charm (Proxy Map)
#
# IMPORTANT:
# Your Barchart chain does NOT contain true vanna/charm Greeks.
# So this tab computes EDUCATIONAL PROXIES using only what you
# actually have: Strike + Call OI + Put OI (+ IV if available).
#
# This file is robust to your API payload shapes:
# - {"success":True, "data":[rows...]}
# - {"success":True, "data":{"data":[rows...]}}
# - {"success":True, "payload":{"data":[rows...]}}
#
# It also tolerates BOTH row formats:
# - /options (side-by-side chain): Strike, Call OI, Put OI, Call IV, Put IV...
# - /weekly/gex (computed): strike, call_oi, put_oi, Call IV, Put IV...
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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
    Uses dynamic walls + your proxy values (direction + magnitude).
    """
    s = float(spot)

    # If we don't have walls, fallback sentence
    if put_wall is None or call_wall is None:
        bias = "call-weighted" if net_charm > 0 else "put-weighted" if net_charm < 0 else "balanced"
        return f"Proxy flows are {bias}; focus on nearest high-OI strikes because the map suggests pinning/pressure around spot."

    pw = float(put_wall)
    cw = float(call_wall)

    # define "high" as relative (no hard-coded symbol), using combined magnitude
    mag = abs(net_vanna) + abs(net_charm)
    is_high = mag > 0  # always true, but we’ll message as “higher” using relative words
    # We’ll keep it simple: interpret direction from charm (OI imbalance) & vanna (OI×IV imbalance)
    directional = "call-side pressure" if (net_charm > 0 and net_vanna > 0) else \
                  "put-side pressure" if (net_charm < 0 and net_vanna < 0) else \
                  "mixed pressure"

    if s > cw:
        return f"🚀 Above Call Wall ({cw:.2f}): breakout mode — {directional} can accelerate moves higher."
    if s < pw:
        return f"🧨 Below Put Wall ({pw:.2f}): breakdown mode — {directional} can accelerate moves lower."

    # inside walls
    # closer to which wall?
    dist_to_pw = abs(s - pw)
    dist_to_cw = abs(s - cw)
    near = f"near Call Wall ({cw:.2f})" if dist_to_cw < dist_to_pw else f"near Put Wall ({pw:.2f})"

    return f"🧲 Inside walls ({pw:.2f} → {cw:.2f}) {near}: expect chop/pinning unless price breaks; {directional} tells which side may win on breakout."


# ----------------------------
# Main render
# ----------------------------

def render_tab_vanna_charm(symbol, date, spot, hist_df=None):
    st.subheader(f"🌊 Vanna & Charm (Proxy Map): {symbol}")

    # Always show a reminder so you don't forget
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

    # Trend context panel
    if hist_df is not None:
        _trend_context_block(float(spot), hist_df)
    else:
        # user asked to show this message if MAs not detected / no history
        st.markdown("### 📊 Trend Context (Educational)")
        st.info("📊 Trend Context: Moving average columns were not detected in this table.\n\nIf you add columns like MA20, MA50, MA200, this panel will explain the trend context automatically.")

    # Fetch data (your existing /weekly/gex endpoint)
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
    # Build proxies (NO true greeks available)
    # ----------------------------
    S = float(spot)

    # weight nearby strikes more (so 400 matters when spot=398)
    width = max(S * 0.02, 1.0)  # 2% of spot, min 1.0
    dist = (df["strike"] - S).abs()
    w = np.exp(-((dist / width) ** 2))

    call_iv = pd.to_numeric(df["call_iv"], errors="coerce").fillna(0.0)
    put_iv = pd.to_numeric(df["put_iv"], errors="coerce").fillna(0.0)

    # if IV is 35 instead of 0.35, convert to decimal
    call_iv = np.where(call_iv > 3.0, call_iv / 100.0, call_iv)
    put_iv = np.where(put_iv > 3.0, put_iv / 100.0, put_iv)

    # Proxy formulas (educational)
    df["vanna_proxy"] = w * ((df["call_oi"] * call_iv) - (df["put_oi"] * put_iv)) * S * 0.01
    df["charm_proxy"] = w * (df["call_oi"] - df["put_oi"])

    total_vanna = float(df["vanna_proxy"].sum())
    total_charm = float(df["charm_proxy"].sum())

    # Dynamic levels
    put_wall, call_wall, magnet = _calc_levels(df, S)

    # ----------------------------
    # Summary line (meaningful sentence)
    # ----------------------------
    st.markdown("### 🧭 Market Regime Summary")
    st.success(_regime_sentence(S, put_wall, call_wall, total_vanna, total_charm))

    # ----------------------------
    # Dynamic “How to read this” block (uses CURRENT values)
    # ----------------------------
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
    # Metrics (with explanations beside them)
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
    # Charts
    # ----------------------------
    left, right = st.columns(2)

    with left:
        st.write("### Net Dealer Vanna (proxy) by Strike")
        fig_v = go.Figure(go.Bar(x=df["strike"], y=df["vanna_proxy"], name="Vanna (proxy)"))
        fig_v.add_vline(x=S, line_dash="dash", line_color="white", annotation_text="SPOT")
        if call_wall is not None:
            fig_v.add_vline(x=float(call_wall), line_dash="dot", line_color="orange", annotation_text="CALL WALL")
        if put_wall is not None:
            fig_v.add_vline(x=float(put_wall), line_dash="dot", line_color="red", annotation_text="PUT WALL")
        fig_v.update_layout(template="plotly_dark", height=420, xaxis_title="Strike", yaxis_title="Vanna (proxy)")
        st_plot(fig_v)

    with right:
        st.write("### Net Dealer Charm (proxy) by Strike")
        fig_c = go.Figure(go.Bar(x=df["strike"], y=df["charm_proxy"], name="Charm (proxy)"))
        fig_c.add_vline(x=S, line_dash="dash", line_color="white", annotation_text="SPOT")
        if magnet is not None:
            fig_c.add_vline(x=float(magnet), line_dash="dot", line_color="cyan", annotation_text="MAGNET")
        if call_wall is not None:
            fig_c.add_vline(x=float(call_wall), line_dash="dot", line_color="orange", annotation_text="CALL WALL")
        if put_wall is not None:
            fig_c.add_vline(x=float(put_wall), line_dash="dot", line_color="red", annotation_text="PUT WALL")
        fig_c.update_layout(template="plotly_dark", height=420, xaxis_title="Strike", yaxis_title="Charm (proxy)")
        st_plot(fig_c)

    # Debug table
    with st.expander("Debug / Normalized inputs used (Strike, OI, IV)", expanded=False):
        show = df[["strike", "call_oi", "put_oi", "call_iv", "put_iv", "vanna_proxy", "charm_proxy"]].copy()
        st.dataframe(show, use_container_width=True)
