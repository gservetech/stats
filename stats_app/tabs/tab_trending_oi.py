import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stats_app.helpers.ui_components import st_btn, st_df, st_plot


def _to_num(series_or_value):
    if isinstance(series_or_value, pd.Series):
        s = (
            series_or_value.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("—", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series_or_value, errors="coerce")


def _lc_map(cols):
    return {str(c).strip().lower(): c for c in cols}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lc = _lc_map(df.columns)
    for c in candidates:
        key = c.strip().lower()
        if key in lc:
            return lc[key]
    return None


def _find_contains(df: pd.DataFrame, *needles: str) -> str | None:
    for c in df.columns:
        cl = str(c).strip().lower()
        if all(n.lower() in cl for n in needles):
            return c
    return None


def _extract_oi_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize chain data to: Strike, call_oi, put_oi.
    Handles common dashboard and Barchart-like side-by-side variants.
    """
    strike_col = (
        _find_col(df, ["Strike", "strike", "strikePrice", "strike_price", "k"])
        or _find_contains(df, "strike")
    )
    if not strike_col:
        return pd.DataFrame()

    call_oi_col = (
        _find_col(df, ["Call OI", "call_oi", "callOpenInterest", "call_open_interest", "call_open_int"])
        or _find_contains(df, "call", "open", "interest")
        or _find_contains(df, "call", "openinterest")
        or _find_contains(df, "call", "oi")
    )
    put_oi_col = (
        _find_col(df, ["Put OI", "put_oi", "putOpenInterest", "put_open_interest", "put_open_int"])
        or _find_contains(df, "put", "open", "interest")
        or _find_contains(df, "put", "openinterest")
        or _find_contains(df, "put", "oi")
    )

    # Side-by-side fallback seen in CSV exports: Open Int + Open Int.1
    if not (call_oi_col and put_oi_col):
        side_call = _find_col(df, ["Open Int", "open int", "open_int", "openinterest"])
        side_put = _find_col(df, ["Open Int.1", "open int.1", "open_int.1", "openinterest.1"])
        if side_call and side_put:
            call_oi_col = side_call
            put_oi_col = side_put

    if not (call_oi_col and put_oi_col):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "Strike": _to_num(df[strike_col]),
            "call_oi": _to_num(df[call_oi_col]).fillna(0.0),
            "put_oi": _to_num(df[put_oi_col]).fillna(0.0),
        }
    )
    out = out.dropna(subset=["Strike"])
    if out.empty:
        return out

    out = (
        out.groupby("Strike", as_index=False)[["call_oi", "put_oi"]]
        .sum()
        .sort_values("Strike")
        .reset_index(drop=True)
    )
    return out


def _pick_atm_strike(strikes: np.ndarray, spot: float) -> float:
    arr = np.asarray(strikes, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(spot))))
    return float(arr[idx])


def _window_around_atm(df: pd.DataFrame, spot: float, n_each_side: int = 7):
    if df is None or df.empty:
        raise ValueError("Empty dataframe")

    work = df.sort_values("Strike").reset_index(drop=True).copy()
    strikes = work["Strike"].to_numpy(dtype=float)
    atm = _pick_atm_strike(strikes, spot)
    atm_idx = int(np.where(strikes == atm)[0][0])

    lo = max(0, atm_idx - int(n_each_side))
    hi = min(len(work) - 1, atm_idx + int(n_each_side))
    w = work.iloc[lo : hi + 1].copy()
    w["is_atm"] = w["Strike"] == atm
    return w.reset_index(drop=True), atm


def _summarize_snapshot(df_window: pd.DataFrame) -> dict:
    call_oi_total = float(df_window["call_oi"].sum(skipna=True))
    put_oi_total = float(df_window["put_oi"].sum(skipna=True))
    diff = put_oi_total - call_oi_total
    net_pcr = (put_oi_total / call_oi_total) if call_oi_total > 0 else np.nan

    sentiment = "Neutral"
    if diff > 0:
        sentiment = "Put OI > Call OI -> Bullish bias"
    elif diff < 0:
        sentiment = "Call OI > Put OI -> Bearish bias"

    return {
        "call_oi_total": call_oi_total,
        "put_oi_total": put_oi_total,
        "difference_put_minus_call": diff,
        "net_pcr": net_pcr,
        "sentiment": sentiment,
    }


def _compare_two_snapshots(prev_df: pd.DataFrame, cur_df: pd.DataFrame, spot: float, n_each_side: int = 7):
    prev_w, atm_prev = _window_around_atm(prev_df, spot, n_each_side)
    cur_w, atm_cur = _window_around_atm(cur_df, spot, n_each_side)

    base = cur_w[["Strike"]].copy()

    m = base.merge(prev_w[["Strike", "call_oi", "put_oi"]], on="Strike", how="left")
    m = m.rename(columns={"call_oi": "call_oi_prev", "put_oi": "put_oi_prev"})

    m = m.merge(cur_w[["Strike", "call_oi", "put_oi", "is_atm"]], on="Strike", how="left")
    m = m.rename(columns={"call_oi": "call_oi_cur", "put_oi": "put_oi_cur"})

    for col in ["call_oi_prev", "put_oi_prev", "call_oi_cur", "put_oi_cur"]:
        m[col] = _to_num(m[col]).fillna(0.0)

    m["d_call_oi"] = m["call_oi_cur"] - m["call_oi_prev"]
    m["d_put_oi"] = m["put_oi_cur"] - m["put_oi_prev"]
    m["net_change"] = m["d_put_oi"] - m["d_call_oi"]

    prev_summary = _summarize_snapshot(prev_w)
    cur_summary = _summarize_snapshot(cur_w)

    prev_diff = float(prev_summary["difference_put_minus_call"])
    cur_diff = float(cur_summary["difference_put_minus_call"])
    direction_change_pct = (
        ((cur_diff - prev_diff) / abs(prev_diff) * 100.0)
        if abs(prev_diff) > 1e-12
        else np.nan
    )

    dominance = "No clear direction (mixed or both rising)"
    if (cur_summary["put_oi_total"] > prev_summary["put_oi_total"]) and (
        cur_summary["call_oi_total"] < prev_summary["call_oi_total"]
    ):
        dominance = "Bullish trend (Put OI up, Call OI down)"
    elif (cur_summary["call_oi_total"] > prev_summary["call_oi_total"]) and (
        cur_summary["put_oi_total"] < prev_summary["put_oi_total"]
    ):
        dominance = "Bearish trend (Call OI up, Put OI down)"

    meta = {
        "atm_prev": atm_prev,
        "atm_cur": atm_cur,
        "prev_summary": prev_summary,
        "cur_summary": cur_summary,
        "direction_change_pct": direction_change_pct,
        "dominance": dominance,
    }

    return m.sort_values("Strike").reset_index(drop=True), meta


def _plot_net_change_by_strike(m: pd.DataFrame):
    colors = np.where(
        m["net_change"] > 0,
        "#1fbf75",
        np.where(m["net_change"] < 0, "#ff5c5c", "#8e98a8"),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=m["Strike"],
            y=m["net_change"],
            marker_color=colors,
            hovertemplate="Strike: %{x}<br>NetChange: %{y:,.0f}<extra></extra>",
            name="NetChange",
        )
    )
    fig.add_hline(y=0.0, line_width=1, line_color="#9aa4b2")
    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Buying vs Selling Pressure (NetChange = dPutOI - dCallOI)",
        xaxis_title="Strike (ATM window)",
        yaxis_title="NetChange",
        showlegend=False,
    )
    return fig


def _plot_pcr(prev_pcr: float, cur_pcr: float):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Prev PCR", "Cur PCR"],
            y=[prev_pcr, cur_pcr],
            marker_color=["#6ea8fe", "#f9c74f"],
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        title="Net PCR (Put OI / Call OI) in Active Strikes",
        yaxis_title="PCR",
    )
    return fig


def _snapshot_payload(oi_df: pd.DataFrame) -> dict:
    return {
        "captured_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "df": oi_df.copy(),
    }


def render_tab_trending_oi(df: pd.DataFrame, spot: float, symbol: str, expiry_date: str):
    st.subheader("Trending OI")
    st.caption(
        "Uses your loaded chain directly (no CSV). Window is ATM +/- N strikes. "
        "NetChange = dPutOI - dCallOI."
    )

    if df is None or df.empty:
        st.warning("No chain data loaded.")
        return
    if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
        st.error("Spot is missing or invalid.")
        return
    spot = float(spot)

    oi_df = _extract_oi_table(df)
    if oi_df.empty:
        st.error("Could not detect Strike/Call OI/Put OI columns from the loaded chain.")
        st_df(pd.DataFrame({"columns": list(df.columns)}), height=260)
        return

    n_each_side = int(
        st.slider(
            "Strikes above/below ATM",
            min_value=1,
            max_value=20,
            value=7,
            step=1,
            key=f"trend_oi_n_{symbol}_{expiry_date}",
        )
    )

    scope = f"{str(symbol).upper()}|{str(expiry_date)}"
    prev_key = f"trend_oi_prev_{scope}"
    cur_key = f"trend_oi_cur_{scope}"

    if prev_key not in st.session_state:
        st.session_state[prev_key] = None
    if cur_key not in st.session_state:
        st.session_state[cur_key] = None

    b1, b2, b3 = st.columns(3)
    with b1:
        if st_btn("Set Previous Snapshot = Current Chain", key=f"trend_prev_btn_{scope}"):
            st.session_state[prev_key] = _snapshot_payload(oi_df)
    with b2:
        if st_btn("Set Current Snapshot = Current Chain", key=f"trend_cur_btn_{scope}"):
            st.session_state[cur_key] = _snapshot_payload(oi_df)
    with b3:
        if st_btn("Clear Snapshots", key=f"trend_clear_btn_{scope}"):
            st.session_state[prev_key] = None
            st.session_state[cur_key] = None

    prev_snap = st.session_state.get(prev_key)
    cur_snap = st.session_state.get(cur_key)

    st.caption(
        "Previous snapshot: "
        + (prev_snap["captured_at"] if isinstance(prev_snap, dict) else "not set")
        + " | Current snapshot: "
        + (cur_snap["captured_at"] if isinstance(cur_snap, dict) else "not set")
    )

    # Always show current single-snapshot summary for quick read.
    live_window, live_atm = _window_around_atm(oi_df, spot=spot, n_each_side=n_each_side)
    live_sum = _summarize_snapshot(live_window)

    st.markdown("**Current Chain Snapshot (single-window summary)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ATM", f"{live_atm:,.2f}")
    c2.metric("Call OI Total", f"{live_sum['call_oi_total']:,.0f}")
    c3.metric("Put OI Total", f"{live_sum['put_oi_total']:,.0f}")
    c4.metric("Diff (Put-Call)", f"{live_sum['difference_put_minus_call']:,.0f}")
    st.caption(f"Net PCR: {live_sum['net_pcr']:.3f} | Sentiment: {live_sum['sentiment']}")

    if not (isinstance(prev_snap, dict) and isinstance(cur_snap, dict)):
        st.info("Capture both previous and current snapshots to view Trending OI comparison.")
        return

    prev_df = prev_snap.get("df", pd.DataFrame())
    cur_df = cur_snap.get("df", pd.DataFrame())
    if prev_df.empty or cur_df.empty:
        st.warning("Snapshots are empty. Re-capture them from current chain data.")
        return

    m, meta = _compare_two_snapshots(prev_df, cur_df, spot=spot, n_each_side=n_each_side)
    ps = meta["prev_summary"]
    cs = meta["cur_summary"]

    st.markdown("**Trending Comparison (Previous vs Current)**")
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("ATM (Prev)", f"{meta['atm_prev']:,.2f}")
    x2.metric("ATM (Cur)", f"{meta['atm_cur']:,.2f}")
    x3.metric(
        "Direction Change %",
        "N/A" if not np.isfinite(meta["direction_change_pct"]) else f"{meta['direction_change_pct']:.2f}%",
    )
    total_net = float(m["net_change"].sum())
    bias = "Neutral"
    if total_net > 0:
        bias = "Bullish (Put dOI dominates)"
    elif total_net < 0:
        bias = "Bearish (Call dOI dominates)"
    x4.metric("Overall Bias", bias)

    y1, y2, y3 = st.columns(3)
    y1.metric("Prev Diff (Put-Call)", f"{ps['difference_put_minus_call']:,.0f}")
    y2.metric("Cur Diff (Put-Call)", f"{cs['difference_put_minus_call']:,.0f}")
    y3.metric(
        "Net PCR (Prev -> Cur)",
        f"{ps['net_pcr']:.3f} -> {cs['net_pcr']:.3f}"
        if np.isfinite(ps["net_pcr"]) and np.isfinite(cs["net_pcr"])
        else "N/A",
    )
    st.caption(f"Dominance: {meta['dominance']}")

    st_plot(_plot_net_change_by_strike(m))
    st_plot(_plot_pcr(ps["net_pcr"], cs["net_pcr"]))

    show_cols = [
        "Strike",
        "is_atm",
        "d_call_oi",
        "d_put_oi",
        "net_change",
        "call_oi_prev",
        "call_oi_cur",
        "put_oi_prev",
        "put_oi_cur",
    ]
    out = m[show_cols].copy()
    for col in [c for c in out.columns if c != "is_atm"]:
        out[col] = _to_num(out[col]).round(2)
    st_df(out, height=420)
