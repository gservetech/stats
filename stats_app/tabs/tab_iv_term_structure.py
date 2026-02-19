import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from stats_app.helpers.ui_components import st_plot, st_df


def _lc_map(cols):
    return {c.lower(): c for c in cols}


def _find_col(df, candidates):
    cols = _lc_map(df.columns)
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _find_contains(df, *needles):
    for c in df.columns:
        cl = c.lower()
        if all(n.lower() in cl for n in needles):
            return c
    return None


def _to_num(s):
    if isinstance(s, pd.Series):
        s = (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("—", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(s, errors="coerce")


def _iv_to_percent(iv_series: pd.Series) -> pd.Series:
    """
    Your export IV is usually already percent (e.g. 23.5 or '23.5%').
    If it comes as decimal (0.235), convert to percent.
    """
    iv = _to_num(iv_series)
    med = float(np.nanmedian(iv)) if iv.notna().any() else np.nan
    if np.isfinite(med) and med <= 3.0:  # looks like 0.20..1.50 etc
        iv = iv * 100.0
    return iv


def _reshape_side_by_side_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles your exact format:
    Calls: Type, IV, Open Int, Volume, Bid, Ask, Latest, Strike
    Puts:  Type.1, IV.1, Open Int.1, Volume.1, Bid.1, Ask.1, Latest.1, Strike
    """
    cols = set(df.columns)
    if "Strike" not in cols:
        return pd.DataFrame()

    # Need at least IV and IV.1 or Type and Type.1
    if not (("IV" in cols and "IV.1" in cols) or ("Type" in cols and "Type.1" in cols)):
        return pd.DataFrame()

    out_call = pd.DataFrame({
        "strike": _to_num(df["Strike"]),
        "side": "CALL",
        "iv": _iv_to_percent(df["IV"]) if "IV" in cols else np.nan,
        "oi": _to_num(df["Open Int"]).fillna(0) if "Open Int" in cols else 0.0,
        "vol": _to_num(df["Volume"]).fillna(0) if "Volume" in cols else 0.0,
    })
    out_put = pd.DataFrame({
        "strike": _to_num(df["Strike"]),
        "side": "PUT",
        "iv": _iv_to_percent(df["IV.1"]) if "IV.1" in cols else np.nan,
        "oi": _to_num(df["Open Int.1"]).fillna(0) if "Open Int.1" in cols else 0.0,
        "vol": _to_num(df["Volume.1"]).fillna(0) if "Volume.1" in cols else 0.0,
    })

    out = pd.concat([out_call, out_put], ignore_index=True)
    out = out.dropna(subset=["strike", "iv"])
    return out


def _reshape_side_by_side_two_strikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles rare exports where Strike appears twice (CALL block then PUT block).
    """
    work = df.copy()
    cols = list(work.columns)
    strike_candidates = [c for c in cols if "strike" in c.lower()]
    if len(strike_candidates) < 2:
        return pd.DataFrame()

    call_strike = strike_candidates[0]
    put_strike = strike_candidates[1]

    def pick_block_cols(block="call"):
        if block == "call":
            end_idx = cols.index(put_strike)
            block_cols = cols[:end_idx + 1]
            strike_col = call_strike
        else:
            start_idx = cols.index(put_strike)
            block_cols = cols[start_idx:]
            strike_col = put_strike

        def first_match(names):
            for n in names:
                for c in block_cols:
                    if c.lower() == n.lower():
                        return c
            return None

        def contains_all(*needles):
            for c in block_cols:
                cl = c.lower()
                if all(n in cl for n in needles):
                    return c
            return None

        iv_col = first_match(["iv"]) or contains_all("iv")
        oi_col = first_match(["open int", "open interest", "openint", "oi"]) or contains_all("open", "int")
        vol_col = first_match(["volume", "vol"]) or contains_all("vol")
        return strike_col, iv_col, oi_col, vol_col

    c_strike, c_iv, c_oi, c_vol = pick_block_cols("call")
    p_strike, p_iv, p_oi, p_vol = pick_block_cols("put")

    if not c_iv or not p_iv:
        return pd.DataFrame()

    out_call = pd.DataFrame({
        "strike": _to_num(work[c_strike]),
        "side": "CALL",
        "iv": _iv_to_percent(work[c_iv]),
        "oi": _to_num(work[c_oi]).fillna(0) if c_oi else 0.0,
        "vol": _to_num(work[c_vol]).fillna(0) if c_vol else 0.0,
    })
    out_put = pd.DataFrame({
        "strike": _to_num(work[p_strike]),
        "side": "PUT",
        "iv": _iv_to_percent(work[p_iv]),
        "oi": _to_num(work[p_oi]).fillna(0) if p_oi else 0.0,
        "vol": _to_num(work[p_vol]).fillna(0) if p_vol else 0.0,
    })
    out = pd.concat([out_call, out_put], ignore_index=True)
    out = out.dropna(subset=["strike", "iv"])
    return out


def _reshape_wide_explicit_sides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles formats where columns explicitly mention Call/Put
    e.g., 'Call IV', 'Put IV', 'Strike'.
    """
    strike_col = _find_col(df, ["strike", "strikeprice", "k"]) or _find_contains(df, "strike")
    if not strike_col:
        return pd.DataFrame()

    # Find IV columns specific to Calls and Puts
    call_iv = _find_contains(df, "call", "iv")
    put_iv = _find_contains(df, "put", "iv")

    if not call_iv or not put_iv:
        return pd.DataFrame()

    # Optional: Find OI and Volume
    call_oi = _find_contains(df, "call", "open") or _find_contains(df, "call", "oi")
    put_oi = _find_contains(df, "put", "open") or _find_contains(df, "put", "oi")

    call_vol = _find_contains(df, "call", "vol")
    put_vol = _find_contains(df, "put", "vol")

    out_call = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": "CALL",
        "iv": _iv_to_percent(df[call_iv]),
        "oi": _to_num(df[call_oi]).fillna(0) if call_oi else 0.0,
        "vol": _to_num(df[call_vol]).fillna(0) if call_vol else 0.0,
    })

    out_put = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": "PUT",
        "iv": _iv_to_percent(df[put_iv]),
        "oi": _to_num(df[put_oi]).fillna(0) if put_oi else 0.0,
        "vol": _to_num(df[put_vol]).fillna(0) if put_vol else 0.0,
    })

    out = pd.concat([out_call, out_put], ignore_index=True)
    out = out.dropna(subset=["strike", "iv"])
    return out


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Your real format: .1 suffix
    r = _reshape_side_by_side_suffix(df)
    if not r.empty:
        return r

    # 2) Alternate: two strike columns
    r = _reshape_side_by_side_two_strikes(df)
    if not r.empty:
        return r

    # 3) Explicit prefixed/suffixed sides (Call IV, Put IV)
    r = _reshape_wide_explicit_sides(df)
    if not r.empty:
        return r

    # 4) Already-long format
    strike_col = _find_col(df, ["strike", "strikeprice", "k"]) or _find_contains(df, "strike")
    side_col = _find_col(df, ["type", "optiontype", "callput", "putcall", "right", "side"])
    iv_col = _find_col(df, ["iv", "impliedvolatility"]) or _find_contains(df, "iv")
    oi_col = _find_col(df, ["openint", "open interest", "open_interest", "oi"]) or _find_contains(df, "open", "int")
    vol_col = _find_col(df, ["volume", "vol"]) or _find_contains(df, "vol")

    if not strike_col or not side_col or not iv_col:
        return pd.DataFrame()

    s = df[side_col].astype(str).str.upper().str.strip()
    s = s.replace({"CALLS": "CALL", "PUTS": "PUT", "C": "CALL", "P": "PUT"})
    s = np.where(s.str.contains("C"), "CALL", np.where(s.str.contains("P"), "PUT", s))

    out = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": s,
        "iv": _iv_to_percent(df[iv_col]),
        "oi": _to_num(df[oi_col]).fillna(0) if oi_col else 0.0,
        "vol": _to_num(df[vol_col]).fillna(0) if vol_col else 0.0,
    })
    out = out.dropna(subset=["strike", "iv"])
    return out


def render_tab_iv_term_structure(df: pd.DataFrame, spot: float, expiry_date: str = None, symbol: str = None):
    st.subheader("🧾 IV Term Structure (Single Expiry view + sanity filters)")
    st.caption(
        "Single expiry = show clean smile + ATM IV. (If you add multiple expiries later, we can extend to true term structure.)")

    if df is None or df.empty:
        st.warning("No chain data loaded.")
        return
    if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
        st.error("Spot missing/invalid.")
        return
    spot = float(spot)

    long_df = _to_long(df)
    if long_df.empty:
        st.error("Required columns not found (need Strike + IV + Side).")
        st_df(pd.DataFrame({"columns": list(df.columns)}), height=220)
        return

    with st.expander("Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        min_oi = c1.number_input("Min OI", min_value=0, value=0, step=10)
        max_iv = c2.slider("Max IV (%)", 10, 500, 250)
        m_min = c3.slider("Moneyness min (K/S)", 0.20, 1.50, 0.60)
        m_max = c4.slider("Moneyness max (K/S)", 0.60, 3.00, 1.40)

        d1, d2, d3 = st.columns(3)
        show_calls = d1.checkbox("Show Calls", value=True)
        show_puts = d2.checkbox("Show Puts", value=True)
        show_raw = d3.checkbox("Show Raw Points", value=True)

    work = long_df.copy()
    work["iv"] = pd.to_numeric(work["iv"], errors="coerce")
    work["oi"] = pd.to_numeric(work["oi"], errors="coerce").fillna(0.0)
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work = work.dropna(subset=["strike", "iv"])
    work = work[(work["iv"] > 0) & (work["iv"] <= float(max_iv))]

    # OI usable = numeric + at least some non-zero
    oi_usable = work["oi"].notna().any() and float(work["oi"].sum()) > 0.0
    if not oi_usable:
        min_oi = 0
        st.warning("Open Interest looks empty/zero in this export → Min OI filter disabled.")

    work["m"] = work["strike"] / spot

    before = len(work)
    work = work[(work["m"] >= float(m_min)) & (work["m"] <= float(m_max))]
    after_window = len(work)

    if oi_usable and min_oi > 0:
        work = work[work["oi"] >= float(min_oi)]
    after_oi = len(work)

    st.caption(f"Rows kept: {after_window} (window) | After OI: {after_oi} | Started: {before}")

    if work.empty:
        st.error("After cleaning, no strikes remain. Try lower Min OI or widen moneyness window.")
        return

    # ATM IV estimate
    work["dist_atm"] = (work["m"] - 1.0).abs()
    near_atm = work.sort_values("dist_atm").head(12)
    atm_iv = float(near_atm["iv"].median()) if not near_atm.empty else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"{spot:,.2f}")
    c2.metric("ATM IV (approx)", f"{atm_iv:,.2f}%" if np.isfinite(atm_iv) else "—")
    c3.metric("Expiry", str(expiry_date) if expiry_date else "N/A")

    def _plot_smile(x_col: str, title: str):
        fig = go.Figure()

        def add_side(side, name):
            d = work[work["side"] == side].sort_values(x_col)
            if d.empty:
                return

            if show_raw:
                fig.add_trace(go.Scatter(
                    x=d[x_col], y=d["iv"],
                    mode="markers",
                    name=f"{name} (raw)",
                    hovertemplate=f"{name}<br>{x_col}: %{{x:.4f}}<br>IV: %{{y:.2f}}%<extra></extra>",
                ))

            # Smooth curve (polyfit)
            deg = 3
            x = d[x_col].astype(float).values
            y = d["iv"].astype(float).values
            if len(x) >= 6:
                w = d["oi"].astype(float).values if oi_usable else np.ones_like(x)
                w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
                try:
                    p = np.polyfit(x, y, deg=deg, w=w)
                    xs = np.linspace(np.min(x), np.max(x), 220)
                    ys = np.polyval(p, xs)
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys,
                        mode="lines",
                        name=f"{name} (smooth)",
                        hovertemplate=f"{name} smooth<br>{x_col}: %{{x:.4f}}<br>IV: %{{y:.2f}}%<extra></extra>",
                    ))
                except Exception:
                    pass

        if show_calls:
            add_side("CALL", "Calls")
        if show_puts:
            add_side("PUT", "Puts")

        if x_col == "strike":
            fig.add_vline(x=spot, line_width=1, line_dash="dot", annotation_text="Spot", annotation_position="top")
        else:
            fig.add_vline(x=1.0, line_width=1, line_dash="dot", annotation_text="ATM (K/S=1.0)",
                          annotation_position="top")

        fig.update_layout(
            template="plotly_dark",
            height=520,
            title=title,
            xaxis_title=x_col,
            yaxis_title="Implied Volatility (%)",
            legend_title_text="Series",
        )
        st_plot(fig)

    _plot_smile("strike", "IV vs Strike (Clean + Smoothed)")
    _plot_smile("m", "IV vs Moneyness (K/S) (Clean + Smoothed)")

    with st.expander("🔎 Cleaned rows (debug)", expanded=False):
        st_df(work[["side", "strike", "m", "iv", "oi", "vol"]].sort_values(["side", "strike"]), height=420)