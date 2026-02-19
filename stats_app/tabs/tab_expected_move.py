import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from stats_app.helpers.ui_components import st_plot, st_df


def _lc_map(cols):
    return {c.lower(): c for c in cols}


def _to_num(s):
    if isinstance(s, pd.Series):
        s = (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("—", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(s, errors="coerce")


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


def _maybe_expand_nested(df: pd.DataFrame) -> pd.DataFrame:
    # If backend returns nested dict column, expand first dict column
    for c in df.columns:
        nn = df[c].dropna()
        if nn.empty:
            continue
        v = nn.iloc[0]
        if isinstance(v, dict):
            expanded = pd.json_normalize(df[c]).add_prefix(f"{c}.")
            base = df.drop(columns=[c]).reset_index(drop=True)
            return pd.concat([base, expanded.reset_index(drop=True)], axis=1)
    return df


def _reshape_strike_level_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strike-level rows: 1 row per strike, columns for call/put bid/ask/last/iv/oi/vol.
    Returns long df with columns:
    strike, side, bid, ask, last, iv, oi, vol
    """
    strike_col = (
        _find_col(df, ["strike", "strikeprice", "strike_price", "k"])
        or _find_contains(df, "strike")
    )
    if not strike_col:
        return pd.DataFrame()

    # detect call/put columns by contains patterns
    call_bid = _find_contains(df, "call", "bid")
    call_ask = _find_contains(df, "call", "ask")
    call_last = _find_contains(df, "call", "last") or _find_contains(df, "call", "latest") or _find_contains(df, "call", "price")
    call_iv = _find_contains(df, "call", "iv")
    call_oi = _find_contains(df, "call", "open", "interest") or _find_contains(df, "call", "openinterest") or _find_contains(df, "call", "oi")
    call_vol = _find_contains(df, "call", "volume") or _find_contains(df, "call", "vol")

    put_bid = _find_contains(df, "put", "bid")
    put_ask = _find_contains(df, "put", "ask")
    put_last = _find_contains(df, "put", "last") or _find_contains(df, "put", "latest") or _find_contains(df, "put", "price")
    put_iv = _find_contains(df, "put", "iv")
    put_oi = _find_contains(df, "put", "open", "interest") or _find_contains(df, "put", "openinterest") or _find_contains(df, "put", "oi")
    put_vol = _find_contains(df, "put", "volume") or _find_contains(df, "put", "vol")

    # We need at least strike + some premium fields (bid/ask or last) on both sides
    call_has_prem = (call_bid and call_ask) or call_last
    put_has_prem = (put_bid and put_ask) or put_last
    if not (call_has_prem and put_has_prem):
        return pd.DataFrame()

    out_call = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": "CALL",
        "bid": _to_num(df[call_bid]) if call_bid else np.nan,
        "ask": _to_num(df[call_ask]) if call_ask else np.nan,
        "last": _to_num(df[call_last]) if call_last else np.nan,
        "iv": _to_num(df[call_iv]) if call_iv else np.nan,
        "oi": _to_num(df[call_oi]).fillna(0) if call_oi else 0.0,
        "vol": _to_num(df[call_vol]).fillna(0) if call_vol else 0.0,
    })

    out_put = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": "PUT",
        "bid": _to_num(df[put_bid]) if put_bid else np.nan,
        "ask": _to_num(df[put_ask]) if put_ask else np.nan,
        "last": _to_num(df[put_last]) if put_last else np.nan,
        "iv": _to_num(df[put_iv]) if put_iv else np.nan,
        "oi": _to_num(df[put_oi]).fillna(0) if put_oi else 0.0,
        "vol": _to_num(df[put_vol]).fillna(0) if put_vol else 0.0,
    })

    out = pd.concat([out_call, out_put], ignore_index=True)
    out = out.dropna(subset=["strike"])
    return out


def _reshape_side_by_side_two_strikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some exports repeat the Strike column twice (CALL block then PUT block).
    """
    cols = list(df.columns)
    strike_cols = [c for c in cols if "strike" in c.lower()]
    if len(strike_cols) < 2:
        return pd.DataFrame()

    call_strike = strike_cols[0]
    put_strike = strike_cols[1]

    def block_cols(block="call"):
        if block == "call":
            end = cols.index(put_strike)
            block = cols[: end + 1]
            strike = call_strike
        else:
            start = cols.index(put_strike)
            block = cols[start:]
            strike = put_strike

        def first_exact(name_list):
            for n in name_list:
                for c in block:
                    if c.lower() == n.lower():
                        return c
            return None

        def contains_any(*needles):
            for c in block:
                cl = c.lower()
                if all(n in cl for n in needles):
                    return c
            return None

        last = first_exact(["latest", "last"]) or contains_any("last") or contains_any("latest")
        bid = first_exact(["bid"]) or contains_any("bid")
        ask = first_exact(["ask"]) or contains_any("ask")
        iv = first_exact(["iv"]) or contains_any("iv")
        oi = first_exact(["open int", "open interest", "oi"]) or contains_any("open", "int") or contains_any("openinterest")
        vol = first_exact(["volume", "vol"]) or contains_any("vol")

        return strike, last, bid, ask, iv, oi, vol

    c_strike, c_last, c_bid, c_ask, c_iv, c_oi, c_vol = block_cols("call")
    p_strike, p_last, p_bid, p_ask, p_iv, p_oi, p_vol = block_cols("put")

    # Require premium fields
    if not (((c_bid and c_ask) or c_last) and ((p_bid and p_ask) or p_last)):
        return pd.DataFrame()

    out_call = pd.DataFrame({
        "strike": _to_num(df[c_strike]),
        "side": "CALL",
        "bid": _to_num(df[c_bid]) if c_bid else np.nan,
        "ask": _to_num(df[c_ask]) if c_ask else np.nan,
        "last": _to_num(df[c_last]) if c_last else np.nan,
        "iv": _to_num(df[c_iv]) if c_iv else np.nan,
        "oi": _to_num(df[c_oi]).fillna(0) if c_oi else 0.0,
        "vol": _to_num(df[c_vol]).fillna(0) if c_vol else 0.0,
    })
    out_put = pd.DataFrame({
        "strike": _to_num(df[p_strike]),
        "side": "PUT",
        "bid": _to_num(df[p_bid]) if p_bid else np.nan,
        "ask": _to_num(df[p_ask]) if p_ask else np.nan,
        "last": _to_num(df[p_last]) if p_last else np.nan,
        "iv": _to_num(df[p_iv]) if p_iv else np.nan,
        "oi": _to_num(df[p_oi]).fillna(0) if p_oi else 0.0,
        "vol": _to_num(df[p_vol]).fillna(0) if p_vol else 0.0,
    })
    out = pd.concat([out_call, out_put], ignore_index=True)
    out = out.dropna(subset=["strike"])
    return out


def _to_long_any(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries in order:
    1) expand nested dict cols
    2) strike-level reshape (call/put columns)
    3) side-by-side two-strike blocks
    4) already-long contract-level (type/side col)
    """
    work = _maybe_expand_nested(df.copy())

    # 1) strike-level (best for your dashboard)
    r1 = _reshape_strike_level_to_long(work)
    if not r1.empty:
        return r1

    # 2) side-by-side with two strike columns
    r2 = _reshape_side_by_side_two_strikes(work)
    if not r2.empty:
        return r2

    # 3) already long/contract-level
    strike_col = _find_col(work, ["strike", "strikeprice", "k"]) or _find_contains(work, "strike")
    side_col = _find_col(work, ["type", "optiontype", "callput", "putcall", "right", "side"])
    if not strike_col or not side_col:
        return pd.DataFrame()

    bid_col = _find_col(work, ["bid", "bidprice"]) or _find_contains(work, "bid")
    ask_col = _find_col(work, ["ask", "askprice"]) or _find_contains(work, "ask")
    last_col = _find_col(work, ["last", "latest", "price", "close"]) or _find_contains(work, "last")

    iv_col = _find_col(work, ["iv", "impliedvolatility"]) or _find_contains(work, "iv")
    oi_col = _find_col(work, ["openint", "open interest", "open_interest", "oi"]) or _find_contains(work, "open", "int")
    vol_col = _find_col(work, ["volume", "vol"]) or _find_contains(work, "vol")

    s = work[side_col].astype(str).str.upper().str.strip()
    s = s.replace({"CALLS": "CALL", "PUTS": "PUT", "C": "CALL", "P": "PUT"})
    s = np.where(s.str.contains("C"), "CALL", np.where(s.str.contains("P"), "PUT", s))

    out = pd.DataFrame({
        "strike": _to_num(work[strike_col]),
        "side": s,
        "bid": _to_num(work[bid_col]) if bid_col else np.nan,
        "ask": _to_num(work[ask_col]) if ask_col else np.nan,
        "last": _to_num(work[last_col]) if last_col else np.nan,
        "iv": _to_num(work[iv_col]) if iv_col else np.nan,
        "oi": _to_num(work[oi_col]).fillna(0) if oi_col else 0.0,
        "vol": _to_num(work[vol_col]).fillna(0) if vol_col else 0.0,
    })
    out = out.dropna(subset=["strike"])
    return out


def render_tab_expected_move(df: pd.DataFrame, spot: float, expiry_date: str = None, symbol: str = None):
    st.subheader("📦 Expected Move Engine (Single Expiry)")
    st.caption("Uses **ATM straddle** (ATM call mid + ATM put mid) as expected move for the selected expiry.")

    if df is None or df.empty:
        st.warning("No chain data loaded.")
        return
    if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
        st.error("Spot missing/invalid.")
        return
    spot = float(spot)

    long_df = _to_long_any(df)

    if long_df.empty:
        st.error("Could not understand your chain table format (no strike/side detected).")
        st_df(pd.DataFrame({"columns": list(df.columns)}), height=240)
        st.caption("Fix: Your chain should have either (a) a CALL/PUT side column, OR (b) strike-level call*/put* columns.")
        return

    # Compute mid
    long_df["mid"] = np.where(
        np.isfinite(long_df["bid"]) & np.isfinite(long_df["ask"]) & (long_df["ask"] > 0),
        (long_df["bid"] + long_df["ask"]) / 2.0,
        long_df["last"],
    )
    long_df["mid"] = pd.to_numeric(long_df["mid"], errors="coerce")

    # Must have both sides per strike
    valid_strikes = (
        long_df.dropna(subset=["strike"])
        .groupby("strike")["side"]
        .nunique()
        .reset_index()
    )
    valid_strikes = valid_strikes[valid_strikes["side"] >= 2]["strike"].values
    if len(valid_strikes) == 0:
        st.error("Need both CALL and PUT rows per strike to compute ATM straddle.")
        st_df(long_df.head(30), height=360)
        return

    atm_strike = float(min(valid_strikes, key=lambda k: abs(float(k) - spot)))

    call_row = long_df[(long_df["side"] == "CALL") & (long_df["strike"] == atm_strike)].head(1)
    put_row = long_df[(long_df["side"] == "PUT") & (long_df["strike"] == atm_strike)].head(1)

    call_mid = float(call_row["mid"].iloc[0]) if not call_row.empty and np.isfinite(call_row["mid"].iloc[0]) else np.nan
    put_mid = float(put_row["mid"].iloc[0]) if not put_row.empty and np.isfinite(put_row["mid"].iloc[0]) else np.nan

    if not np.isfinite(call_mid) or not np.isfinite(put_mid):
        st.error("ATM call/put premium missing. Ensure Bid/Ask or Latest/Last exists in your export.")
        st_df(pd.concat([call_row, put_row], ignore_index=True), height=220)
        return

    straddle = call_mid + put_mid
    exp_move_d = float(straddle)
    exp_move_pct = float((straddle / spot) * 100.0)

    lo = spot - exp_move_d
    hi = spot + exp_move_d

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"{spot:,.2f}")
    c2.metric("ATM Strike", f"{atm_strike:,.2f}")
    c3.metric("Expected Move ($)", f"±{exp_move_d:,.2f}")
    c4.metric("Expected Move (%)", f"±{exp_move_pct:,.2f}%")

    st.info(
        f"**Implied Range** {('for ' + symbol) if symbol else ''} {expiry_date or ''}: "
        f"**{lo:,.2f} ↔ {hi:,.2f}**  (ATM straddle)"
    )

    # Show context around ATM (closest strikes)
    tmp = long_df.copy()
    tmp["dist"] = (tmp["strike"] - spot).abs()
    show = tmp.sort_values(["dist", "side"]).head(24)[
        ["side", "strike", "bid", "ask", "mid", "last", "iv", "oi", "vol"]
    ]
    st_df(show, height=420)

    # Visual
    fig = go.Figure()
    fig.add_hline(y=spot, line_width=1, line_dash="dot")
    fig.add_hrect(y0=lo, y1=hi, opacity=0.15, line_width=0)
    fig.add_trace(go.Scatter(
        x=["Low", "Spot", "High"],
        y=[lo, spot, hi],
        mode="markers+text",
        text=[f"{lo:,.2f}", f"{spot:,.2f}", f"{hi:,.2f}"],
        textposition="top center",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        name="Expected Range",
    ))
    fig.update_layout(template="plotly_dark", height=360, title="Expected Move Range (ATM Straddle)", yaxis_title="Price")
    st_plot(fig)