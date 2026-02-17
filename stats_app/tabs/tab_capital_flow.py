import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from stats_app.helpers.ui_components import st_df


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
    return pd.to_numeric(s, errors="coerce")


def _pct_rank(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return x.rank(pct=True)


def _pick_col_ui(label, df, default=None, help_text=None):
    options = ["(none)"] + list(df.columns)
    idx = 0
    if default and default in df.columns:
        idx = options.index(default)
    chosen = st.selectbox(label, options=options, index=idx, help=help_text)
    return None if chosen == "(none)" else chosen


def _maybe_expand_nested(df: pd.DataFrame) -> pd.DataFrame:
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


def _reshape_strike_level_to_long(df: pd.DataFrame):
    strike_col = (
        _find_col(df, ["strike", "strikeprice", "strike_price", "k"])
        or _find_contains(df, "strike")
    )
    if not strike_col:
        return None, "No strike column found."

    call_oi = (
        _find_contains(df, "call", "open", "interest")
        or _find_contains(df, "call", "openinterest")
        or _find_contains(df, "call", "oi")
    )
    put_oi = (
        _find_contains(df, "put", "open", "interest")
        or _find_contains(df, "put", "openinterest")
        or _find_contains(df, "put", "oi")
    )

    call_vol = (
        _find_contains(df, "call", "volume")
        or _find_contains(df, "call", "vol")
    )
    put_vol = (
        _find_contains(df, "put", "volume")
        or _find_contains(df, "put", "vol")
    )

    if not (call_oi and put_oi and call_vol and put_vol):
        return None, (
            "Not strike-level format (missing call/put OI+Volume columns). "
            "Expected columns like callOpenInterest/putOpenInterest and callVolume/putVolume."
        )

    call_mid = _find_contains(df, "call", "mid") or _find_contains(df, "call", "mark")
    put_mid = _find_contains(df, "put", "mid") or _find_contains(df, "put", "mark")
    call_last = _find_contains(df, "call", "last") or _find_contains(df, "call", "price")
    put_last = _find_contains(df, "put", "last") or _find_contains(df, "put", "price")
    call_bid = _find_contains(df, "call", "bid")
    call_ask = _find_contains(df, "call", "ask")
    put_bid = _find_contains(df, "put", "bid")
    put_ask = _find_contains(df, "put", "ask")

    exp_col = (
        _find_contains(df, "expiration")
        or _find_contains(df, "expiry")
        or _find_contains(df, "exp")
    )
    dte_col = _find_contains(df, "dte") or _find_contains(df, "days")

    def _prem_series(mid, last, bid, ask):
        if mid:
            return _to_num(df[mid]).fillna(0).clip(lower=0)
        if last:
            return _to_num(df[last]).fillna(0).clip(lower=0)
        if bid and ask:
            return ((_to_num(df[bid]) + _to_num(df[ask])) / 2.0).fillna(0).clip(lower=0)
        return pd.Series([0.0] * len(df))

    left = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": "CALL",
        "oi": _to_num(df[call_oi]).fillna(0),
        "vol": _to_num(df[call_vol]).fillna(0),
        "prem": _prem_series(call_mid, call_last, call_bid, call_ask),
    })
    right = pd.DataFrame({
        "strike": _to_num(df[strike_col]),
        "side": "PUT",
        "oi": _to_num(df[put_oi]).fillna(0),
        "vol": _to_num(df[put_vol]).fillna(0),
        "prem": _prem_series(put_mid, put_last, put_bid, put_ask),
    })

    if exp_col:
        left["exp"] = df[exp_col].astype(str)
        right["exp"] = df[exp_col].astype(str)
    elif dte_col:
        dte = _to_num(df[dte_col]).fillna(-1).astype(int).astype(str)
        left["exp"] = "DTE " + dte
        right["exp"] = "DTE " + dte
    else:
        left["exp"] = "N/A"
        right["exp"] = "N/A"

    out = pd.concat([left, right], ignore_index=True)
    return out, None


def render_tab_capital_flow(df: pd.DataFrame, spot: float = None, expiry_date: str = None, symbol: str = None):
    st.subheader("💸 Capital Flow Engine (Barchart-Ready)")
    st.caption("Interactive hover charts (Plotly). Works with **contract-level** and **Barchart strike-level** data.")

    if df is None or df.empty:
        st.warning("No chain data loaded.")
        return

    if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
        st.error("Spot missing. Pass `spot=spot` from main app (sidebar spot).")
        return
    spot = float(spot)

    work = _maybe_expand_nested(df.copy())

    reshaped, reshape_err = _reshape_strike_level_to_long(work)
    if reshaped is not None:
        long_df = reshaped.copy()
        if expiry_date and "exp" in long_df.columns:
            long_df["exp"] = long_df["exp"].replace("N/A", str(expiry_date))
        _run_engine(long_df, spot=spot, symbol=symbol)
        return

    strike_col = (
        _find_col(work, ["strike", "strikeprice", "strike_price", "k"])
        or _find_contains(work, "strike")
    )
    oi_col = (
        _find_col(work, ["openinterest", "open_interest", "oi"])
        or _find_contains(work, "open", "interest")
    )
    vol_col = (
        _find_col(work, ["volume", "vol"])
        or _find_contains(work, "volume")
        or _find_contains(work, "vol")
    )
    side_col = _find_col(work, ["optiontype", "type", "callput", "putcall", "cp", "right", "side"])

    mid_col = _find_col(work, ["mid", "mark", "midprice", "markprice"])
    last_col = _find_col(work, ["last", "lastprice", "price", "close"])
    bid_col = _find_col(work, ["bid", "bidprice"])
    ask_col = _find_col(work, ["ask", "askprice"])

    with st.expander("🛠 Column Mapper (Contract-level only)", expanded=True):
        st.caption(
            "If your df is strike-level, you should NOT need this. "
            "If it is contract-level, map strike / type / OI / volume here."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            strike_col = _pick_col_ui("Strike column", work, default=strike_col)
        with c2:
            side_col = _pick_col_ui("Call/Put column", work, default=side_col)
        with c3:
            oi_col = _pick_col_ui("Open Interest column", work, default=oi_col)

        c4, c5, c6 = st.columns(3)
        with c4:
            vol_col = _pick_col_ui("Volume column", work, default=vol_col)
        with c5:
            mid_col = _pick_col_ui("Mid/Mark", work, default=mid_col)
        with c6:
            last_col = _pick_col_ui("Last", work, default=last_col)

        c7, c8 = st.columns(2)
        with c7:
            bid_col = _pick_col_ui("Bid", work, default=bid_col)
        with c8:
            ask_col = _pick_col_ui("Ask", work, default=ask_col)

        st.divider()
        st.caption("Debug: columns in your df")
        st_df(pd.DataFrame({"columns": list(work.columns)}), height=240)

    missing = []
    if not strike_col:
        missing.append("strike")
    if not oi_col:
        missing.append("open_interest (oi)")
    if not vol_col:
        missing.append("volume")
    if not side_col:
        missing.append("call/put type")

    if missing:
        st.error(
            "Still missing required columns: " + ", ".join(missing)
            + "\n\nIf you are using Barchart strike-level data, your df should contain columns like:\n"
              "• callOpenInterest / putOpenInterest\n"
              "• callVolume / putVolume\n"
              "(No `type` column is needed in strike-level mode.)\n\n"
              "This tab tried strike-level reshape but failed. See details below."
        )
        st.caption("Strike-level reshape failure reason:")
        st.code(str(reshape_err))
        st_df(work.head(30), height=420)
        return

    long_df = pd.DataFrame({
        "strike": _to_num(work[strike_col]),
        "side": work[side_col].astype(str),
        "oi": _to_num(work[oi_col]).fillna(0),
        "vol": _to_num(work[vol_col]).fillna(0),
    })

    prem = pd.Series([0.0] * len(work), index=work.index)
    if mid_col:
        prem = _to_num(work[mid_col]).fillna(0.0)
    elif last_col:
        prem = _to_num(work[last_col]).fillna(0.0)
    elif bid_col and ask_col:
        prem = ((_to_num(work[bid_col]) + _to_num(work[ask_col])) / 2.0).fillna(0.0)
    long_df["prem"] = prem.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    long_df["exp"] = str(expiry_date) if expiry_date else "N/A"
    _run_engine(long_df, spot=spot, symbol=symbol)


def _run_engine(long_df: pd.DataFrame, spot: float, symbol: str = None):
    f = long_df.copy()

    f["strike"] = _to_num(f["strike"])
    f["oi"] = _to_num(f["oi"]).fillna(0.0)
    f["vol"] = _to_num(f["vol"]).fillna(0.0)

    if "prem" not in f.columns:
        f["prem"] = 0.0
    f["prem"] = _to_num(f["prem"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    if "exp" not in f.columns:
        f["exp"] = "N/A"

    s = f["side"].astype(str).str.upper().str.strip()
    s = s.replace({"CALLS": "CALL", "PUTS": "PUT", "C": "CALL", "P": "PUT"})
    f["side"] = np.where(s.str.contains("C"), "CALL", np.where(s.str.contains("P"), "PUT", s))

    f["_dist_pct"] = (f["strike"] - float(spot)).abs() / float(spot)
    mult = 100.0
    f["_vol_notional"] = f["vol"] * f["prem"] * mult
    f["_oi_notional"] = f["oi"] * f["prem"] * mult
    f["_newness"] = f["vol"] / (f["oi"].replace(0, np.nan))
    f["_newness"] = f["_newness"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    with st.expander("🔎 Data Diagnostics (why you may see zeros)", expanded=False):
        st.write("Rows:", int(len(f)))
        st.write("Premium stats:", {
            "prem_nonzero_rows": int((f["prem"] > 0).sum()),
            "prem_max": float(f["prem"].max()) if len(f) else 0.0,
            "prem_median": float(f["prem"].median()) if len(f) else 0.0,
        })
        st.write("Volume stats:", {
            "vol_nonzero_rows": int((f["vol"] > 0).sum()),
            "vol_max": float(f["vol"].max()) if len(f) else 0.0,
            "vol_median": float(f["vol"].median()) if len(f) else 0.0,
        })
        st.write("OI stats:", {
            "oi_nonzero_rows": int((f["oi"] > 0).sum()),
            "oi_max": float(f["oi"].max()) if len(f) else 0.0,
            "oi_median": float(f["oi"].median()) if len(f) else 0.0,
        })
        st.caption(
            "If **prem_nonzero_rows = 0**, then Premium Flow ($) and OI Notional ($) will be zero. "
            "Use Raw Volume / Raw OI views."
        )
        st_df(f.head(30), height=280)

    with st.expander("⚙️ Engine Controls", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        side_pick = c1.multiselect("Side", ["CALL", "PUT"], default=["CALL", "PUT"])

        exp_pick = c2.multiselect(
            "Expirations",
            options=sorted(f["exp"].dropna().unique().tolist()),
            default=sorted(f["exp"].dropna().unique().tolist())[:6],
        )

        max_dist_pct = c3.slider("Max Distance from Spot (%)", 1, 100, 100) / 100.0

        min_oi = c4.number_input("Min OI", min_value=0, value=0)
        min_vol = c5.number_input("Min Vol", min_value=0, value=0)

        d1, d2, d3, d4 = st.columns(4)
        top_n = d1.slider("Top N strikes", 10, 150, 40)
        strike_bucket = d2.selectbox("Strike Bucket", ["$0.50", "$1", "$2.50", "$5", "$10"], index=1)

        prem_nonzero = int((f["prem"] > 0).sum())
        default_view = "Premium Flow ($)" if prem_nonzero > 0 else "Raw Volume"
        view_mode = d3.selectbox(
            "View",
            ["Premium Flow ($)", "OI Notional ($)", "Raw OI", "Raw Volume"],
            index=["Premium Flow ($)", "OI Notional ($)", "Raw OI", "Raw Volume"].index(default_view),
        )

        atm_window = d4.slider("ATM Focus Window (%)", 1, 50, 15) / 100.0

    bucket = float(strike_bucket.replace("$", ""))

    f2 = f.copy()
    f2 = f2[f2["side"].isin(side_pick)]
    if exp_pick:
        f2 = f2[f2["exp"].isin(exp_pick)]
    f2 = f2[f2["_dist_pct"] <= max_dist_pct]
    f2 = f2[f2["oi"] >= float(min_oi)]
    f2 = f2[f2["vol"] >= float(min_vol)]

    if f2.empty:
        st.warning("No rows after filters. Set Min OI/Min Vol to 0 and Max Distance to 100%.")
        return

    f2["_strike_b"] = (np.round(f2["strike"] / bucket) * bucket).round(2)

    if view_mode == "Premium Flow ($)":
        measure = "_vol_notional"
        measure_label = "Premium Flow ($)"
    elif view_mode == "OI Notional ($)":
        measure = "_oi_notional"
        measure_label = "OI Notional ($)"
    elif view_mode == "Raw OI":
        measure = "oi"
        measure_label = "Open Interest"
    else:
        measure = "vol"
        measure_label = "Volume"

    by = f2.groupby(["_strike_b", "side"], dropna=False)[measure].sum().reset_index()
    pivot = by.pivot_table(index="_strike_b", columns="side", values=measure, aggfunc="sum").fillna(0.0)

    if "CALL" not in pivot.columns:
        pivot["CALL"] = 0.0
    if "PUT" not in pivot.columns:
        pivot["PUT"] = 0.0

    pivot["NET"] = pivot["CALL"] - pivot["PUT"]
    pivot["TOTAL"] = pivot["CALL"] + pivot["PUT"]
    pivot["abs_net"] = pivot["NET"].abs()

    pivot["newness"] = f2.groupby("_strike_b")["_newness"].mean().reindex(pivot.index).fillna(0.0)
    pivot["s_total"] = _pct_rank(pivot["TOTAL"])
    pivot["s_abs_net"] = _pct_rank(pivot["abs_net"])
    pivot["s_new"] = _pct_rank(pivot["newness"])
    pivot["score"] = (0.55 * pivot["s_total"] + 0.30 * pivot["s_abs_net"] + 0.15 * pivot["s_new"])

    top_strikes = pivot["TOTAL"].sort_values(ascending=False).head(int(top_n)).index
    pivot_top = pivot.loc[top_strikes].sort_index()

    total_call = float(f2.loc[f2["side"] == "CALL", measure].sum())
    total_put = float(f2.loc[f2["side"] == "PUT", measure].sum())
    net = total_call - total_put
    dom = (total_call / (total_call + total_put)) if (total_call + total_put) > 0 else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"CALL {measure_label}", f"{total_call:,.0f}")
    k2.metric(f"PUT {measure_label}", f"{total_put:,.0f}")
    k3.metric("NET (Call - Put)", f"{net:,.0f}")
    k4.metric("Call Dominance", f"{(dom * 100):.1f}%" if np.isfinite(dom) else "—")

    st.divider()

    st.markdown("### 📊 Strike Concentration (Calls vs Puts)")
    fig1 = go.Figure()
    fig1.add_bar(
        x=pivot_top.index.astype(str),
        y=pivot_top["CALL"],
        name="CALL",
        hovertemplate="Strike: %{x}<br>Call: %{y:,.0f}<extra></extra>",
    )
    fig1.add_bar(
        x=pivot_top.index.astype(str),
        y=pivot_top["PUT"],
        name="PUT",
        hovertemplate="Strike: %{x}<br>Put: %{y:,.0f}<extra></extra>",
    )
    fig1.update_layout(
        barmode="stack",
        template="plotly_dark",
        xaxis_title="Strike (bucketed)",
        yaxis_title=measure_label,
        height=450,
        legend_title_text="Side",
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### ⚖️ Net Imbalance (CALL - PUT)")
    fig2 = go.Figure()
    fig2.add_bar(
        x=pivot_top.index.astype(str),
        y=pivot_top["NET"],
        name="NET",
        hovertemplate="Strike: %{x}<br>Net: %{y:,.0f}<extra></extra>",
    )
    fig2.add_hline(y=0, line_width=1)
    fig2.update_layout(
        template="plotly_dark",
        xaxis_title="Strike (bucketed)",
        yaxis_title="Net (Call - Put)",
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🗓️ Expiry Clustering")
    exp_agg = (
        f2.groupby(["exp", "side"], dropna=False)[measure]
        .sum()
        .reset_index()
        .pivot_table(index="exp", columns="side", values=measure, aggfunc="sum")
        .fillna(0.0)
    )
    if "CALL" not in exp_agg.columns:
        exp_agg["CALL"] = 0.0
    if "PUT" not in exp_agg.columns:
        exp_agg["PUT"] = 0.0
    exp_agg["TOTAL"] = exp_agg["CALL"] + exp_agg["PUT"]
    exp_agg = exp_agg.sort_values("TOTAL", ascending=False)
    st_df(exp_agg.reset_index().rename(columns={"exp": "expiry"}), height=280)

    st.markdown("### 🧠 New vs Old Money (Proxy)")
    st.caption("Higher = more volume relative to existing OI (more active/new attention).")

    newness = (
        f2.groupby(["_strike_b", "side"], dropna=False)["_newness"]
        .mean()
        .reset_index()
        .pivot_table(index="_strike_b", columns="side", values="_newness", aggfunc="mean")
        .fillna(0.0)
    )
    if "CALL" not in newness.columns:
        newness["CALL"] = 0.0
    if "PUT" not in newness.columns:
        newness["PUT"] = 0.0
    newness["MAX"] = newness[["CALL", "PUT"]].max(axis=1)

    new_top = newness["MAX"].sort_values(ascending=False).head(int(top_n)).index
    newness_top = newness.loc[new_top].sort_index()

    fig3 = go.Figure()
    fig3.add_scatter(
        x=newness_top.index.astype(str),
        y=newness_top["CALL"],
        mode="lines+markers",
        name="CALL newness",
        hovertemplate="Strike: %{x}<br>Call newness: %{y:.4f}<extra></extra>",
    )
    fig3.add_scatter(
        x=newness_top.index.astype(str),
        y=newness_top["PUT"],
        mode="lines+markers",
        name="PUT newness",
        hovertemplate="Strike: %{x}<br>Put newness: %{y:.4f}<extra></extra>",
    )
    fig3.update_layout(
        template="plotly_dark",
        xaxis_title="Strike (bucketed)",
        yaxis_title="Vol / OI (avg)",
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🎯 ATM Focus (Closest Strikes Only)")
    atm = f2[f2["_dist_pct"] <= atm_window].copy()
    if atm.empty:
        st.info("No rows inside ATM window. Increase the ATM Focus Window.")
    else:
        atm["_strike_b"] = (np.round(atm["strike"] / bucket) * bucket).round(2)
        atm_agg = (
            atm.groupby(["_strike_b", "side"], dropna=False)[measure]
            .sum()
            .reset_index()
            .pivot_table(index="_strike_b", columns="side", values=measure, aggfunc="sum")
            .fillna(0.0)
        )
        if "CALL" not in atm_agg.columns:
            atm_agg["CALL"] = 0.0
        if "PUT" not in atm_agg.columns:
            atm_agg["PUT"] = 0.0
        atm_agg["NET"] = atm_agg["CALL"] - atm_agg["PUT"]
        st_df(atm_agg.reset_index().rename(columns={"_strike_b": "strike"}), height=260)

    st.markdown("### 🏆 Capital Strikes Leaderboard")
    strike_score = pivot.sort_values("score", ascending=False).head(int(top_n))
    st_df(
        strike_score.reset_index().rename(columns={"_strike_b": "strike"}).round({"score": 4, "newness": 4}),
        height=420,
    )

    st.info(
        f"Spot used: **{spot:.2f}**"
        + (f" | Symbol: **{symbol}**" if symbol else "")
        + "\n\nHover any bar/point to see strike + values. "
          "If Premium Flow ($) is zero, your data likely has no premium fields. Switch View to Raw Volume/Raw OI."
    )
