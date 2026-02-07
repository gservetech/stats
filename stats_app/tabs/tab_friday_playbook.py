import streamlit as st
import pandas as pd
from ..helpers.ui_components import st_df


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
    """
    Normalizes chain columns into canonical names:
      strike, call_volume, call_open_int, call_iv, put_volume, put_open_int, put_iv
    Supports common source variants like:
      Call OI/Put OI, call_oi/put_oi, Call Volume/Put Volume, call_vol/put_vol, etc.
    """
    df = chain_df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

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

    num_cols = [
        c for c in df.columns
        if any(k in c for k in ["bid", "ask", "change", "volume", "open_int", "oi", "iv", "strike"])
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["strike"]).sort_values("strike")

    df["total_oi"] = df["call_open_int"].fillna(0) + df["put_open_int"].fillna(0)
    df["oi_skew"] = df["put_open_int"].fillna(0) - df["call_open_int"].fillna(0)
    df["call_flow_proxy"] = df["call_volume"].fillna(0) / (df["call_open_int"].fillna(0) + 1)
    df["put_flow_proxy"] = df["put_volume"].fillna(0) / (df["put_open_int"].fillna(0) + 1)
    df["flow_proxy"] = df["call_flow_proxy"] + df["put_flow_proxy"]

    if "call_bid" in df.columns and "call_ask" in df.columns:
        df["call_spread"] = df["call_ask"].fillna(0) - df["call_bid"].fillna(0)
    else:
        df["call_spread"] = pd.NA

    if "put_bid" in df.columns and "put_ask" in df.columns:
        df["put_spread"] = df["put_ask"].fillna(0) - df["put_bid"].fillna(0)
    else:
        df["put_spread"] = pd.NA

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
            return "PIN / CONTROL", "Spot is near the magnet between walls → pin/chop risk is high (Friday behavior)."
        return "PIN but SKEWED", "Near magnet, but OI is skewed → pin risk + sudden break risk."

    if between and (near_put_flow > 1.5 or near_call_flow > 1.5):
        side = "puts" if near_put_flow > near_call_flow else "calls"
        return "RISKY CHOP (FLOW BUILDING)", f"Between walls but {side} flow proxy is high near spot → whipsaw / late break risk."

    return "MIXED", "No strong pin or break signals; wait for confirmation."


def _rulebook_markdown():
    return """
## 📜 Friday Gamma Rulebook (READ THIS EVERY FRIDAY)

*Fridays are GAMMA days — not conviction days.*  
Price can be driven by *dealer hedging + OI magnets + intraday flow*, not “news” or feelings.

### 🔑 Core Truths
1. *Walls control price until they break*
2. *Moving walls are targets, not resistance*
3. *Spreads near walls lose on Fridays* (they remove your gamma)
4. *Long gamma beats being right*
5. *No break = no trade*

---

### 🧭 Decision Checklist (DO NOT SKIP)
*1️⃣ Is spot between Call Wall and Put Wall?*  
- YES → expect *chop / pin* → go to step 2  
- NO  → potential *momentum* → go to step 4

*2️⃣ Is spot near the Magnet (max total OI)?*  
- YES → *PIN risk high* → no trade / tiny scalps only  
- NO  → go to step 3

*3️⃣ Is flow proxy (Volume/OI) building near spot?*  
- YES → *whipsaw / late break* risk → wait for confirmation  
- NO  → likely controlled range → no trade

*4️⃣ Did a wall BREAK and hold (5–15 mins) with volume?*  
- YES → trade *LONG GAMMA* (single option, next-week expiry)  
- NO  → stand down

---

### 🚫 Always Avoid on Fridays
- Tight *debit spreads* near walls  
- *Credit spreads* near magnets  
- “It’s overextended” fades inside wall range  
- Holding 0DTE into the close  
- Fighting *wall migration*

---

### ⏰ Best Trading Windows (Toronto time)
- ✅ *10:30 AM – 1:30 PM*: best breakouts (fewer fake-outs)
- ✅ *2:30 PM – 3:30 PM: only if a **fresh break* starts
- 🚫 *9:30 – 9:45 AM*: common fake breaks / spread chaos
- 🚫 *3:30 – 4:00 PM*: pin games / spreads widen / randomness

---

### 🧠 If confused → DO NOTHING
Not trading is a position.
"""


def _strategy_details_markdown(regime: str):
    headline = {
        "BREAK / MOMENTUM": "✅ Today is shaping up as a *BREAK / MOMENTUM* day — focus on breakout long-gamma plays.",
        "PIN / CONTROL": "⚠️ Today is shaping up as a *PIN / CONTROL* day — patience is the edge.",
        "PIN but SKEWED": "⚠️ *Pin + Skew* — price can stick, then suddenly rip. Wait for confirmation.",
        "RISKY CHOP (FLOW BUILDING)": "🚨 *Risky chop* — flow building can create whipsaw. Wait for a clean break.",
        "MIXED": "ℹ️ *Mixed* — don’t force trades; wait for the market to show its hand.",
        "UNKNOWN": "ℹ️ *Unknown* — missing data; do not trade off this tab alone.",
    }.get(regime, "ℹ️ Follow the checklist and trade only confirmed structure.")

    return f"""
## 🧠 Friday Strategies (Detailed: What, When, How)

{headline}

---

### 🥇 Strategy 1: BREAKOUT → LONG GAMMA (Primary Friday Money Maker)
*When to use*
- Spot *breaks* Call Wall (bull) or Put Wall (bear)
- Break *holds 5–15 minutes* (no snap-back)
- Volume expands; flow proxy supports direction
- No major opposing wall immediately nearby

*Best time*
- *10:30 AM – 1:30 PM* (best)
- *2:30 – 3:30 PM* only if a NEW break starts

*What to trade*
- *Single call/put*
- *Next-week expiry* (not 0DTE)
- *ATM or 1 strike OTM*

*Entry*
- After confirmation (don’t buy first tick)
- Best: break → small pullback/retest → continuation

*Exit*
- Take profits into the next big OI cluster / visible stall
- Exit if price re-enters the wall range
- Don’t hold into close

*Why it works*
- You’re *long gamma* in a market that can force dealers to chase hedges.

---

### 🥈 Strategy 2: WALL MIGRATION RIDE (Advanced Momentum)
*What it is*
- The wall *moves with price* (walls become targets)
- Example: Call wall shifts upward as price rises (dealers chasing)

*When to use*
- Flow proxy builds at higher strikes (ahead of price)
- Old wall stops rejecting price
- Trend is clean (not wick-chaos)

*Best time*
- *11:30 AM – 2:30 PM*

*What to trade*
- Single option, *next-week expiry*
- Smaller size than Strategy 1

*Entry*
- Enter after migration is clearly underway (not first print)

*Exit*
- Exit on flow slowdown, stall candles, or when walls stop migrating

---

### 🥉 Strategy 3: PIN DAY = NO TRADE (Most Profitable Discipline)
*When*
- Spot between walls + *near Magnet*
- Flow proxy calms down midday
- Price keeps snapping back to magnet

*Best action*
- *No trade*
- If you must: tiny scalps, fast exits, no holding

*Why*
- Dealers dampen moves → you get chopped + theta’d

---

### ⚠️ Strategy 4: SHORT PREMIUM (Advanced/Optional — only if skilled)
*Only allowed when*
- Price glued to magnet for hours
- Volume fading, no migration
- You can monitor closely

*Best time*
- *After 1:30 PM*

*Trade*
- Small defined-risk (e.g., iron fly)

*Exit*
- Early profit (20–40%)
- Exit immediately if flow spikes or price leaves magnet

---

### 🚫 Why tight spreads fail on Fridays (your exact issue)
Spreads cap your convexity and remove gamma — you lose to:
- Chop near magnet
- Fast regime flips
- Widening spreads into close
"""


def render_tab_friday_playbook(symbol: str, spot: float, chain_df: pd.DataFrame):
    st.subheader("📅 Friday Gamma Playbook (Chain-Driven: Walls • Magnet • Flow • Strategies)")

    st.warning("📌 READ the rulebook below BEFORE placing any Friday trade.")
    st.markdown(_rulebook_markdown())

    if chain_df is None or chain_df.empty:
        st.info("No options chain data provided.")
        return

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
    st.markdown(_strategy_details_markdown(regime))

    with st.expander("🧱 Wall / Magnet Detail Rows (why these levels)", expanded=False):
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
            ignore_index=True,
        ).drop_duplicates(subset=["strike"])
        if not view.empty:
            show = view[[c for c in cols if c in view.columns]]
            st_df(show)
        else:
            st.caption("No wall rows could be built.")

    st.markdown("## 📡 Flow / “Live OI” Proxy (Volume ÷ OI)")
    st.caption("Official OI updates overnight. This shows where today’s activity is heavy relative to existing OI (pressure proxy).")

    top_flow = df.copy().sort_values("flow_proxy", ascending=False).head(12)
    show_cols = [
        "strike",
        "call_volume", "call_open_int", "call_flow_proxy",
        "put_volume", "put_open_int", "put_flow_proxy",
        "flow_proxy",
        "call_iv", "put_iv",
    ]
    st_df(top_flow[[c for c in show_cols if c in top_flow.columns]])

    st.markdown("## 🚫 Friday 'Do Not' List (read before clicking Buy/Sell)")
    st.markdown("""
- *Tight debit spreads near walls* (gamma removed → chop + theta)
- *Credit spreads near magnets* (pin breaks rip through strikes)
- *First 15 minutes* trades (fake breaks + spread chaos)
- *Holding 0DTE into close* (pin games + widening spreads)
- *Fighting wall migration* (moving walls often become targets)
""")

    st.markdown("## 📚 Full Chain (Sorted by Strike)")
    with st.expander("Show full chain table", expanded=False):
        st_df(df.sort_values("strike"))

    st.caption("Educational only — Friday options trading is high risk. Use small size, defined risk, and strict exits.")
