import streamlit as st
import pandas as pd
from ..helpers.data_fetching import fetch_yahoo_share_statistics
from ..helpers.calculations import short_interest_bias
from ..helpers.ui_components import st_df

def _fmt_large(num: float | None):
    if num is None:
        return "N/A"
    try:
        n = float(num)
    except Exception:
        return "N/A"
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"{n/1e12:.2f}T"
    if abs_n >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs_n >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs_n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:,.0f}"

def render_tab_share_statistics(symbol: str):
    st.markdown("### 🧾 Short Interest Context (Float / Short / Cover)")

    if not symbol:
        st.info("Enter a symbol to load share statistics.")
        return

    stats = fetch_yahoo_share_statistics(symbol)
    if not stats or not stats.get("success"):
        err = stats.get("error") if isinstance(stats, dict) else None
        url = stats.get("url") if isinstance(stats, dict) else None
        msg = err or "Share statistics unavailable."
        if url:
            msg = f"{msg} ({url})"
        st.warning(msg)
        return

    if stats.get("html_error"):
        st.info(f"Yahoo HTML parse failed, using JSON fallback. Reason: {stats['html_error']}")

    short_shares = stats.get("short_shares")
    float_shares = stats.get("float_shares")
    avg_vol_10d = stats.get("avg_vol_10d")
    short_shares_prior = stats.get("short_shares_prior")
    short_ratio = stats.get("short_ratio")

    si = short_interest_bias(
        short_shares=short_shares,
        float_shares=float_shares,
        avg_vol_10d=avg_vol_10d,
        short_shares_prior=short_shares_prior,
        short_ratio=short_ratio
    )

    a, b, c, d = st.columns(4)
    short_pct = si.get("short_pct_float")
    a.metric("Short % of Float", f"{short_pct:.2f}%" if short_pct is not None else "N/A")
    b.metric("Short Ratio", f"{short_ratio:.2f}" if short_ratio is not None else "N/A")
    c.metric("Bias", si.get("direction", "N/A"))
    d.metric("Label", si.get("label", "N/A"))

    sub1, sub2, sub3 = st.columns(3)
    sub1.metric("Short Shares", _fmt_large(short_shares))
    sub2.metric("Float Shares", _fmt_large(float_shares))
    sub3.metric("Avg Vol (10d)", _fmt_large(avg_vol_10d))

    with st.expander("How this was interpreted"):
        for n in si.get("notes", []):
            st.write(f"- {n}")

    asof = stats.get("short_shares_asof") or stats.get("short_ratio_asof")
    if asof:
        st.caption(f"Yahoo Finance share statistics as of: {asof}")

    with st.expander("Raw Share Statistics (Yahoo)"):
        raw = stats.get("raw") or {}
        if raw:
            df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in raw.items()])
            st_df(df, height=320)
        else:
            st.info("No raw share statistics found.")

    st.caption("Note: Short interest is positioning context, not a standalone up/down predictor.")
