import streamlit as st
import pandas as pd
from ..helpers.ui_components import st_df

# -----------------------------------------------------------------------------
# Friday Playbook (Options Chain Only) — FULL FILE (LOCAL + GLOBAL WALLS)
#
# ✅ Always-visible Friday Rulebook + Detailed Strategies + Time Windows
# ✅ Robust chain parsing:
#    1) Name-based mapping (call_/put_ columns)
#    2) Position-based mapping (Barchart: Calls ... Strike ... Puts)
# ✅ Computes BOTH:
#    - GLOBAL walls/magnet (entire chain)
#    - LOCAL walls/magnet (tradeable band around spot)
# ✅ Explicit strategy permissions:
#    - Bull Debit / Bear Debit
#    - Bull Credit / Bear Credit
#    - Neutral short-premium (iron fly/condor) only in PIN regime late-day (flagged)
# ✅ Strike suggestions (Friday-safe):
#    - Bull Debit strikes (calls) when bull debit is allowed
#    - Bear Debit strikes (puts) when bear debit is allowed
#    - Neutral short premium center + wings (iron fly) when allowed (safe ranges)
#
# Inputs:
#   render_tab_friday_playbook_from_chain(symbol, spot, chain_df)
# -----------------------------------------------------------------------------

# -------------------------
# Small helpers
# -------------------------
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


def _finalize_playbook_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coerce numerics
    for c in ["strike", "call_open_int", "put_open_int", "call_volume", "put_volume", "call_iv", "put_iv"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["strike"]).sort_values("strike")

    # Fill missing numeric
    df["call_open_int"] = df["call_open_int"].fillna(0)
    df["put_open_int"] = df["put_open_int"].fillna(0)
    df["call_volume"] = df["call_volume"].fillna(0)
    df["put_volume"] = df["put_volume"].fillna(0)
    df["call_iv"] = df["call_iv"].fillna(0)
    df["put_iv"] = df["put_iv"].fillna(0)

    # Derived
    df["total_oi"] = df["call_open_int"] + df["put_open_int"]
    df["oi_skew"] = df["put_open_int"] - df["call_open_int"]

    # "Live OI" proxy = volume / (oi + 1)
    df["call_flow_proxy"] = df["call_volume"] / (df["call_open_int"] + 1)
    df["put_flow_proxy"] = df["put_volume"] / (df["put_open_int"] + 1)
    df["flow_proxy"] = df["call_flow_proxy"] + df["put_flow_proxy"]

    return df


def _prep_chain_df(chain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust mapping for two common layouts:

    A) Name-based:
       strike, call_volume, call_open_int, call_iv, put_volume, put_open_int, put_iv

    B) Barchart-style:
       [CALL cols ... volume, open_int, iv ...]  strike  [PUT cols ... volume, open_int, iv ...]

    Raises ValueError with debug details if mapping fails.
    """
    df = chain_df.copy()
    orig_cols = list(df.columns)

    # Normalize (preserve duplicates like volume.1)
    def norm(c: str) -> str:
        c = str(c).strip().lower()
        c = c.replace(" ", "_").replace("%", "pct")
        return c

    df.columns = [norm(c) for c in df.columns]
    cols = list(df.columns)

    # 1) NAME-BASED mapping
    def find_first(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    strike_col = find_first(["strike", "strk"]) or next((c for c in cols if "strike" in c), None)

    call_vol = find_first(["call_volume", "calls_volume", "c_volume", "callvol", "c_vol", "volume_calls", "calls_vol"])
    call_oi = find_first([
        "call_open_int", "call_open_interest", "call_oi", "calls_open_int", "calls_open_interest",
        "c_open_int", "c_open_interest", "c_oi", "open_int_calls", "open_interest_calls"
    ])
    call_iv = find_first(["call_iv", "call_imp_vol", "call_implied_vol", "callimpliedvol", "c_iv"])

    put_vol = find_first(["put_volume", "puts_volume", "p_volume", "putvol", "p_vol", "volume_puts", "puts_vol"])
    put_oi = find_first([
        "put_open_int", "put_open_interest", "put_oi", "puts_open_int", "puts_open_interest",
        "p_open_int", "p_open_interest", "p_oi", "open_int_puts", "open_interest_puts"
    ])
    put_iv = find_first(["put_iv", "put_imp_vol", "put_implied_vol", "putimpliedvol", "p_iv"])

    if strike_col and call_vol and call_oi and call_iv and put_vol and put_oi and put_iv:
        out = pd.DataFrame({
            "strike": df[strike_col],
            "call_volume": df[call_vol],
            "call_open_int": df[call_oi],
            "call_iv": df[call_iv],
            "put_volume": df[put_vol],
            "put_open_int": df[put_oi],
            "put_iv": df[put_iv],
        })
        return _finalize_playbook_df(out)

    # 2) POSITION-BASED mapping
    if strike_col is None:
        raise ValueError(
            "Could not locate strike column.\n"
            f"Original columns: {orig_cols}\n"
            f"Normalized columns: {cols}"
        )

    strike_idx = cols.index(strike_col)
    left = cols[:strike_idx]        # Calls side
    right = cols[strike_idx + 1:]   # Puts side

    def pick(side_cols, tokens):
        for tok in tokens:
            if tok in side_cols:
                return tok
            m = next((c for c in side_cols if tok in c), None)
            if m:
                return m
        return None

    call_vol = pick(left, ["volume", "vol"])
    call_oi = pick(left, ["open_int", "open_interest", "openint", "oi"])
    call_iv = pick(left, ["iv", "implied_vol", "implied"])

    put_vol = pick(right, ["volume", "vol"])
    put_oi = pick(right, ["open_int", "open_interest", "openint", "oi"])
    put_iv = pick(right, ["iv", "implied_vol", "implied"])

    missing = []
    if call_vol is None:
        missing.append("call_volume")
    if call_oi is None:
        missing.append("call_open_int")
    if call_iv is None:
        missing.append("call_iv")
    if put_vol is None:
        missing.append("put_volume")
    if put_oi is None:
        missing.append("put_open_int")
    if put_iv is None:
        missing.append("put_iv")

    if missing:
        raise ValueError(
            "Could not map required fields (your chain columns differ).\n"
            f"Missing: {missing}\n"
            f"Original columns: {orig_cols}\n"
            f"Normalized columns: {cols}\n"
            f"Strike col: {strike_col}\n"
            f"Left (CALLS): {left}\n"
            f"Right (PUTS): {right}\n"
            "Tip: Ensure Volume/Open Int/IV exist on both sides of Strike."
        )

    out = pd.DataFrame({
        "strike": df[strike_col],
        "call_volume": df[call_vol],
        "call_open_int": df[call_oi],
        "call_iv": df[call_iv],
        "put_volume": df[put_vol],
        "put_open_int": df[put_oi],
        "put_iv": df[put_iv],
    })

    return _finalize_playbook_df(out)


def _walls_and_magnet(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None, None, None, None, None

    call_wall_row = df.loc[df["call_open_int"].fillna(0).idxmax()]
    put_wall_row = df.loc[df["put_open_int"].fillna(0).idxmax()]
    magnet_row = df.loc[df["total_oi"].fillna(0).idxmax()]

    call_wall = _num(call_wall_row["strike"], None)
    put_wall = _num(put_wall_row["strike"], None)
    magnet = _num(magnet_row["strike"], None)

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

    # OI balance (PCR OI)
    total_call_oi = float(df["call_open_int"].fillna(0).sum())
    total_put_oi = float(df["put_open_int"].fillna(0).sum())
    oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    if outside:
        return "BREAK / MOMENTUM", "Spot is outside the wall range → hedging can amplify moves."

    if between and near_magnet:
        if oi_ratio is None or (0.75 <= oi_ratio <= 1.35):
            return "PIN / CONTROL", "Spot is near the magnet between walls → pin/chop risk is high (Friday behavior)."
        else:
            return "PIN but SKEWED", "Near magnet, but OI is skewed → pin risk + sudden break risk."

    if between and (near_put_flow > 1.5 or near_call_flow > 1.5):
        side = "puts" if near_put_flow > near_call_flow else "calls"
        return "RISKY CHOP (FLOW BUILDING)", f"Between walls but {side} flow proxy is high near spot → whipsaw / late break risk."

    return "MIXED", "No strong pin or break signals; wait for confirmation."


# -------------------------
# LOCAL band logic (fixes “everything looks the same”)
# -------------------------
def _local_slice(df: pd.DataFrame, spot: float, pct_band=0.12, min_dollars=10.0):
    """
    Tradeable window around spot:
      band = max(spot*pct_band, min_dollars)
    """
    s = _num(spot, None)
    if s is None or df is None or df.empty:
        return df

    band = max(abs(s) * pct_band, float(min_dollars))
    lo = s - band
    hi = s + band
    local = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    return local if not local.empty else df


# -------------------------
# Strike suggestion helpers
# -------------------------
def _suggest_bull_debit_strikes(df: pd.DataFrame, spot: float, call_wall: float, n=4):
    """
    Friday-safe call strikes:
      - Prefer ATM then 1-2 strikes OTM
      - Avoid buying AT/ABOVE call wall
      - Rank by closeness to spot and call_flow_proxy
    """
    if df is None or df.empty or spot is None or call_wall is None:
        return []

    d = df.copy()
    d["dist"] = (d["strike"] - spot).abs()

    # safe zone: below call wall (don’t buy into the wall)
    safe = d[d["strike"] < call_wall].copy()
    if safe.empty:
        return []

    # prefer near ATM, but also require some flow
    safe = safe.sort_values(["dist", "call_flow_proxy"], ascending=[True, False])

    out = []
    for _, r in safe.head(n).iterrows():
        strike = float(r["strike"])
        label = "ATM" if abs(strike - spot) <= 0.75 else ("OTM" if strike > spot else "ITM")
        out.append({
            "strike": strike,
            "label": label,
            "call_flow_proxy": float(r.get("call_flow_proxy", 0.0)),
            "call_oi": float(r.get("call_open_int", 0.0)),
            "call_vol": float(r.get("call_volume", 0.0)),
        })
    return out


def _suggest_bear_debit_strikes(df: pd.DataFrame, spot: float, put_wall: float, n=4):
    """
    Friday-safe put strikes:
      - Prefer ATM then 1-2 strikes OTM (downside)
      - Avoid buying AT/BELOW put wall
      - Rank by closeness to spot and put_flow_proxy
    """
    if df is None or df.empty or spot is None or put_wall is None:
        return []

    d = df.copy()
    d["dist"] = (d["strike"] - spot).abs()

    # safe zone: above put wall (don’t buy into the wall)
    safe = d[d["strike"] > put_wall].copy()
    if safe.empty:
        return []

    safe = safe.sort_values(["dist", "put_flow_proxy"], ascending=[True, False])

    out = []
    for _, r in safe.head(n).iterrows():
        strike = float(r["strike"])
        label = "ATM" if abs(strike - spot) <= 0.75 else ("OTM" if strike < spot else "ITM")
        out.append({
            "strike": strike,
            "label": label,
            "put_flow_proxy": float(r.get("put_flow_proxy", 0.0)),
            "put_oi": float(r.get("put_open_int", 0.0)),
            "put_vol": float(r.get("put_volume", 0.0)),
        })
    return out


def _suggest_iron_fly(df: pd.DataFrame, magnet: float, wing_pct=0.05):
    """
    Safe-ish neutral short-premium template (advanced):
      - Center at magnet strike (nearest available)
      - Wings = ±wing_pct of magnet
    """
    if df is None or df.empty or magnet is None:
        return None

    d = df.copy()
    d["dist"] = (d["strike"] - magnet).abs()
    center = float(d.sort_values("dist").iloc[0]["strike"])

    wing = max(abs(center) * wing_pct, 2.5)
    lower = center - wing
    upper = center + wing

    # snap wings to nearest available strikes
    def snap(target):
        dd = d.copy()
        dd["d2"] = (dd["strike"] - target).abs()
        return float(dd.sort_values("d2").iloc[0]["strike"])

    return {
        "center": center,
        "put_wing": snap(lower),
        "call_wing": snap(upper)
    }


# -------------------------
# Strategy engine (explicit bull/bear debit/credit)
# -------------------------
def _trade_permissions(regime: str, spot: float, local_call_wall: float, local_put_wall: float, local_magnet: float):
    out = {
        "readiness": "NO-GO",
        "direction": "NEUTRAL",
        "bull_debit": False,
        "bear_debit": False,
        "bull_credit": False,
        "bear_credit": False,
        "neutral_short_premium": False,
        "why": "",
        "suggestions": [],
    }

    s = _num(spot, None)

    if regime == "BREAK / MOMENTUM":
        out["readiness"] = "GO (on confirmed break)"

        if s is not None and local_call_wall is not None and s > local_call_wall:
            out["direction"] = "BULL"
            out["bull_debit"] = True
            out["why"] = "Above LOCAL call wall → breakout regime. Prefer long gamma (debit). Avoid credit spreads."
            out["suggestions"] = [
                "Bull Debit: Single Call (NEXT-WEEK expiry), ATM → 1 strike OTM.",
                "Optional: Call debit spread only if trend is clean and no nearby OI cluster overhead.",
            ]
            return out

        if s is not None and local_put_wall is not None and s < local_put_wall:
            out["direction"] = "BEAR"
            out["bear_debit"] = True
            out["why"] = "Below LOCAL put wall → breakdown regime. Prefer long gamma (debit). Avoid credit spreads."
            out["suggestions"] = [
                "Bear Debit: Single Put (NEXT-WEEK expiry), ATM → 1 strike OTM.",
                "Optional: Put debit spread only if trend is clean and no nearby OI cluster below.",
            ]
            return out

        # momentum but unclear side
        out["direction"] = "MOMENTUM (WAIT CONFIRM)"
        out["why"] = "Momentum detected but not clearly beyond LOCAL walls. Wait for break-and-hold (5–15 min) + volume."
        out["suggestions"] = ["Wait for confirmation candle + volume, then use single option (NEXT-WEEK)."]
        return out

    if regime in ["PIN / CONTROL", "PIN but SKEWED"]:
        out["readiness"] = "NO-GO"
        out["direction"] = "NEUTRAL"
        out["neutral_short_premium"] = True
        out["why"] = "Pin/control → chop + theta. Directional debit/credit spreads are low expectancy."
        out["suggestions"] = [
            "Best: No trade.",
            "Optional (advanced): Neutral defined-risk short premium ONLY if price is glued near magnet for hours (late-day).",
        ]
        return out

    if regime == "RISKY CHOP (FLOW BUILDING)":
        out["readiness"] = "NO-GO"
        out["direction"] = "WAIT"
        out["why"] = "Flow building inside range → whipsaw risk. Wait for break-and-hold before choosing a debit direction."
        out["suggestions"] = ["Wait for break-and-hold (5–15 min), then switch to Bull/Bear Debit (single, NEXT-WEEK)."]
        return out

    if regime == "MIXED":
        out["readiness"] = "NO-GO"
        out["direction"] = "WAIT"
        out["why"] = "Mixed signals. Don’t force trades. Only trade confirmed breakouts."
        out["suggestions"] = ["Stand down until: break + hold + volume.", "If you must: tiny scalps only, fast exits."]
        return out

    out["readiness"] = "NO-GO"
    out["direction"] = "UNKNOWN"
    out["why"] = "Missing data."
    return out


# -------------------------
# UI text blocks
# -------------------------
def _rulebook_markdown():
    return """
## 📜 Friday Gamma Rulebook (READ THIS EVERY FRIDAY)

**Fridays are GAMMA days — not conviction days.**  
Price can be driven by **dealer hedging + OI magnets + intraday flow**, not “news” or feelings.

### 🔑 Core Truths
1. **Walls control price until they break**
2. **Moving walls are targets, not resistance**
3. **Spreads near walls lose on Fridays** (they remove your gamma)
4. **Long gamma beats being right**
5. **No break = no trade**

---

### 🧭 Decision Checklist (DO NOT SKIP)
**1️⃣ Is spot between Call Wall and Put Wall?**  
- YES → expect **chop / pin** → go to step 2  
- NO  → potential **momentum** → go to step 4

**2️⃣ Is spot near the Magnet (max total OI)?**  
- YES → **PIN risk high** → *no trade / tiny scalps only*  
- NO  → go to step 3

**3️⃣ Is flow proxy (Volume/OI) building near spot?**  
- YES → **whipsaw / late break** risk → *wait for confirmation*  
- NO  → likely controlled range → *no trade*

**4️⃣ Did a wall BREAK and hold (5–15 mins) with volume?**  
- YES → trade **LONG GAMMA** (single option, next-week expiry)  
- NO  → *stand down*

---

### 🚫 Always Avoid on Fridays
- Tight **debit spreads** near walls  
- **Credit spreads** near magnets  
- “It’s overextended” fades inside wall range  
- Holding 0DTE into the close  
- Fighting **wall migration**

---

### ⏰ Best Trading Windows (Toronto time)
- ✅ **10:30 AM – 1:30 PM**: best breakouts (fewer fake-outs)
- ✅ **2:30 PM – 3:30 PM**: only if a **fresh break** starts
- 🚫 **9:30 – 9:45 AM**: common fake breaks / spread chaos
- 🚫 **3:30 – 4:00 PM**: pin games / spreads widen / randomness

---

### 🧠 If confused → DO NOTHING
Not trading is a position.
"""


def _strategy_details_markdown(regime: str):
    headline = {
        "BREAK / MOMENTUM": "✅ **BREAK / MOMENTUM** day — focus on breakout long-gamma (debit) plays.",
        "PIN / CONTROL": "⚠️ **PIN / CONTROL** day — discipline is the edge (most profits come from *not trading*).",
        "PIN but SKEWED": "⚠️ **Pin + Skew** — can stick, then suddenly rip. Wait for confirmation.",
        "RISKY CHOP (FLOW BUILDING)": "🚨 **Risky chop** — whipsaw risk. Wait for a clean break-and-hold.",
        "MIXED": "ℹ️ **Mixed** — don’t force trades; wait for the market to show its hand.",
        "UNKNOWN": "ℹ️ **Unknown** — missing data; do not trade off this tab alone."
    }.get(regime, "ℹ️ Follow the checklist and trade only confirmed structure.")

    return f"""
## 🧠 Friday Strategies (Detailed: What, When, How)

{headline}

---

### 🥇 Strategy 1: BREAKOUT → LONG GAMMA (Primary Friday Money Maker)
**Best time**
- **10:30 AM – 1:30 PM** (best)
- **2:30 – 3:30 PM** only if a NEW break starts

**Bullish breakout**
- **Single Call**, **NEXT-WEEK expiry**, **ATM → 1 strike OTM**
- Entry: break → pullback/retest → continuation
- Exit: into next big OI cluster / stall; exit if price re-enters range; no holding into close

**Bearish breakout**
- **Single Put**, **NEXT-WEEK expiry**, **ATM → 1 strike OTM**
- Same entry/exit rules

---

### 🥈 Strategy 2: WALL MIGRATION RIDE (Advanced Momentum)
- Use **single options only**, smaller size
- Best time: **11:30 AM – 2:30 PM**
- Exit on stall candles / flow slowdown / walls stop migrating

---

### 🥉 Strategy 3: PIN DAY = NO TRADE (Most Profitable Discipline)
- Spot between walls + near magnet → **no trade**
- If you must: tiny scalps, fast exits, no holding

---

### ⚠️ Strategy 4: SHORT PREMIUM (Advanced/Optional — only if skilled)
- Only if price is glued to magnet for hours + volume fading
- Best time: **after 1:30 PM**
- Defined risk only (iron fly/condor), take 20–40% profit, exit on flow spike
"""


# -------------------------
# Main renderer
# -------------------------
def render_tab_friday_playbook_from_chain(symbol: str, spot: float, chain_df: pd.DataFrame):
    st.subheader("📅 Friday Gamma Playbook (Chain-Driven: Walls • Magnet • Flow • Strategies)")

    st.warning("📌 READ the rulebook below BEFORE placing any Friday trade.")
    st.markdown(_rulebook_markdown())

    if chain_df is None or chain_df.empty:
        st.info("No options chain data provided.")
        return

    with st.expander("🔎 Debug: raw chain_df columns", expanded=False):
        st.write("Columns:", list(chain_df.columns))
        st_df(chain_df.head(5))

    # Parse chain
    try:
        df = _prep_chain_df(chain_df)
    except Exception as e:
        st.error(f"Chain data format issue: {e}")
        return

    # GLOBAL structure (context)
    g_call_wall, g_put_wall, g_magnet, g_call_row, g_put_row, g_mag_row = _walls_and_magnet(df)

    # LOCAL structure (execution)
    local_df = _local_slice(df, spot=spot, pct_band=0.12, min_dollars=10.0)
    l_call_wall, l_put_wall, l_magnet, l_call_row, l_put_row, l_mag_row = _walls_and_magnet(local_df)

    # Regime based on LOCAL structure (execution)
    regime, reason = _infer_regime_from_chain(local_df, spot, l_call_wall, l_put_wall, l_magnet)

    # -------------------------
    # Structure display
    # -------------------------
    st.markdown("## 🧱 Today’s Structure (LOCAL = execution, GLOBAL = context)")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Symbol", symbol)
    a2.metric("Spot", _fmt(spot, 2))
    a3.metric("LOCAL Magnet", _fmt(l_magnet, 2))
    a4.metric("GLOBAL Magnet", _fmt(g_magnet, 2))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("LOCAL Call Wall", _fmt(l_call_wall, 2))
    b2.metric("LOCAL Put Wall", _fmt(l_put_wall, 2))
    b3.metric("GLOBAL Call Wall", _fmt(g_call_wall, 2))
    b4.metric("GLOBAL Put Wall", _fmt(g_put_wall, 2))

    # PCR (from LOCAL window)
    total_call_oi = float(local_df["call_open_int"].sum()) if local_df is not None and not local_df.empty else 0.0
    total_put_oi = float(local_df["put_open_int"].sum()) if local_df is not None and not local_df.empty else 0.0
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    c1, c2, c3 = st.columns(3)
    c1.metric("LOCAL Total Call OI", f"{total_call_oi:,.0f}")
    c2.metric("LOCAL Total Put OI", f"{total_put_oi:,.0f}")
    c3.metric("LOCAL Put/Call (OI)", f"{pcr_oi:.2f}" if pcr_oi is not None else "N/A")

    st.info(f"**Regime (LOCAL):** {regime} — {reason}")

    # Strategy explanation (always visible)
    st.markdown(_strategy_details_markdown(regime))

    # -------------------------
    # Explicit strategy permissions
    # -------------------------
    perms = _trade_permissions(regime, spot, l_call_wall, l_put_wall, l_magnet)

    st.markdown("## ✅ Trade Readiness (GO / NO-GO) + Allowed Structures")
    if perms["readiness"].startswith("GO"):
        st.success(f"{perms['readiness']} | Direction: **{perms['direction']}**")
    else:
        st.error(f"{perms['readiness']} | Direction: **{perms['direction']}**")
    st.caption(perms["why"])

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Bull Debit", "✅" if perms["bull_debit"] else "❌")
    d2.metric("Bear Debit", "✅" if perms["bear_debit"] else "❌")
    d3.metric("Bull Credit", "✅" if perms["bull_credit"] else "❌")
    d4.metric("Bear Credit", "✅" if perms["bear_credit"] else "❌")
    d5.metric("Neutral Short Prem", "⚠️" if perms["neutral_short_premium"] else "❌")

    st.markdown("### 📌 What you should do (structure guidance)")
    for s in perms.get("suggestions", []):
        st.write(f"- {s}")

    # -------------------------
    # STRIKE SUGGESTIONS (THIS IS WHAT YOU ASKED FOR)
    # -------------------------
    st.markdown("## 🎯 Strike Suggestions (based on LOCAL structure + flow proxy)")
    st.caption("These are *templates* (not orders). Use NEXT-WEEK expiry. Enter only after confirmation candles.")

    if perms["bull_debit"]:
        st.markdown("### 🟢 Bull Debit (Single Calls) — Suggested Strikes")
        calls = _suggest_bull_debit_strikes(local_df, spot=spot, call_wall=l_call_wall, n=4)
        if calls:
            for x in calls:
                st.write(
                    f"- **{int(round(x['strike']))} Call** ({x['label']}) | "
                    f"Call Flow: {x['call_flow_proxy']:.2f} | "
                    f"Call Vol: {int(x['call_vol']):,} | "
                    f"Call OI: {int(x['call_oi']):,}"
                )
            st.caption(f"❌ Avoid buying into the LOCAL call wall (~{_fmt(l_call_wall,2)}).")
        else:
            st.warning("No clean bull-debit strikes found below the LOCAL call wall.")

    if perms["bear_debit"]:
        st.markdown("### 🔴 Bear Debit (Single Puts) — Suggested Strikes")
        puts = _suggest_bear_debit_strikes(local_df, spot=spot, put_wall=l_put_wall, n=4)
        if puts:
            for x in puts:
                st.write(
                    f"- **{int(round(x['strike']))} Put** ({x['label']}) | "
                    f"Put Flow: {x['put_flow_proxy']:.2f} | "
                    f"Put Vol: {int(x['put_vol']):,} | "
                    f"Put OI: {int(x['put_oi']):,}"
                )
            st.caption(f"❌ Avoid buying into the LOCAL put wall (~{_fmt(l_put_wall,2)}).")
        else:
            st.warning("No clean bear-debit strikes found above the LOCAL put wall.")

    if perms["neutral_short_premium"]:
        st.markdown("### 🟡 Neutral (Advanced) — Iron Fly Template (ONLY if pinned late-day)")
        fly = _suggest_iron_fly(local_df, magnet=l_magnet, wing_pct=0.05)
        if fly:
            st.write(f"- **Sell {int(round(fly['center']))} Call** + **Sell {int(round(fly['center']))} Put** (center)")
            st.write(f"- **Buy {int(round(fly['call_wing']))} Call** (upper wing)")
            st.write(f"- **Buy {int(round(fly['put_wing']))} Put** (lower wing)")
            st.caption("Only after 1:30 PM if price is glued to magnet for hours. Take 20–40% profit; exit on flow spike.")
        else:
            st.warning("Could not build an iron-fly template from the local strikes.")

    if not (perms["bull_debit"] or perms["bear_debit"] or perms["neutral_short_premium"]):
        st.info("No strike templates because the regime is NO-GO. Wait for break-and-hold + volume.")

    # -------------------------
    # Time rules
    # -------------------------
    st.markdown("## ⏰ Time rules (Toronto)")
    st.write("- ✅ **10:30–1:30** = best breakout window (if confirmed break)")
    st.write("- ✅ **2:30–3:30** = only if a *fresh* break starts")
    st.write("- 🚫 **9:30–9:45** = fake breaks / spread chaos")
    st.write("- 🚫 **3:30–4:00** = pin games / widening spreads")

    # -------------------------
    # Detail rows (LOCAL + GLOBAL)
    # -------------------------
    with st.expander("🧱 LOCAL Wall/Magnet Detail Rows (execution)", expanded=False):
        cols = [
            "strike",
            "call_open_int", "call_volume", "call_iv", "call_flow_proxy",
            "put_open_int", "put_volume", "put_iv", "put_flow_proxy",
            "flow_proxy",
        ]
        view = pd.concat(
            [
                pd.DataFrame([l_call_row]) if l_call_row is not None else pd.DataFrame(),
                pd.DataFrame([l_put_row]) if l_put_row is not None else pd.DataFrame(),
                pd.DataFrame([l_mag_row]) if l_mag_row is not None else pd.DataFrame(),
            ],
            ignore_index=True
        ).drop_duplicates(subset=["strike"])
        if not view.empty:
            st_df(view[[c for c in cols if c in view.columns]])
        else:
            st.caption("No local rows found.")

    with st.expander("🧱 GLOBAL Wall/Magnet Detail Rows (context)", expanded=False):
        cols = [
            "strike",
            "call_open_int", "call_volume", "call_iv", "call_flow_proxy",
            "put_open_int", "put_volume", "put_iv", "put_flow_proxy",
            "flow_proxy",
        ]
        view = pd.concat(
            [
                pd.DataFrame([g_call_row]) if g_call_row is not None else pd.DataFrame(),
                pd.DataFrame([g_put_row]) if g_put_row is not None else pd.DataFrame(),
                pd.DataFrame([g_mag_row]) if g_mag_row is not None else pd.DataFrame(),
            ],
            ignore_index=True
        ).drop_duplicates(subset=["strike"])
        if not view.empty:
            st_df(view[[c for c in cols if c in view.columns]])
        else:
            st.caption("No global rows found.")

    # -------------------------
    # Flow proxy heat (LOCAL)
    # -------------------------
    st.markdown("## 📡 Flow / “Live OI” Proxy (Volume ÷ OI) — LOCAL window")
    st.caption("Official OI updates overnight. This shows where today’s activity is heavy relative to existing OI (pressure proxy).")

    top_flow = local_df.copy().sort_values("flow_proxy", ascending=False).head(12)
    show_cols = [
        "strike",
        "call_volume", "call_open_int", "call_flow_proxy",
        "put_volume", "put_open_int", "put_flow_proxy",
        "flow_proxy",
        "call_iv", "put_iv",
    ]
    st_df(top_flow[[c for c in show_cols if c in top_flow.columns]])

    # Do-not list
    st.markdown("## 🚫 Friday 'Do Not' List (read before clicking Buy/Sell)")
    st.markdown("""
- **Tight debit spreads near walls** (gamma removed → chop + theta)
- **Credit spreads near magnets** (pin breaks rip through strikes)
- **First 15 minutes** trades (fake breaks + spread chaos)
- **Holding 0DTE into close** (pin games + widening spreads)
- **Fighting wall migration** (moving walls often become targets)
""")

    # Full chain viewer
    st.markdown("## 📚 Full Parsed Chain (Sorted by Strike)")
    with st.expander("Show FULL parsed chain table", expanded=False):
        st_df(df.sort_values("strike"))

    st.caption("Educational only — Friday options trading is high risk. Use small size, defined risk, and strict exits.")