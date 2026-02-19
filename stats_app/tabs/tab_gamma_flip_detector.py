import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from stats_app.helpers.ui_components import st_plot, st_df


def _to_num(s):
    if isinstance(s, pd.Series):
        s = (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("—", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(s, errors="coerce")


def _find_contains(df, *needles):
    for c in df.columns:
        cl = c.lower()
        if all(n.lower() in cl for n in needles):
            return c
    return None


def render_tab_gamma_flip_detector(gex_df: pd.DataFrame, spot: float, symbol: str = None):
    st.subheader("🧲 Gamma Flip Detector")
    st.caption("Uses your **weekly GEX per strike** (preferred). Detects the strike where net gamma crosses ~0.")

    if spot is None or not np.isfinite(float(spot)) or float(spot) <= 0:
        st.error("Spot missing/invalid.")
        return
    spot = float(spot)

    if gex_df is None or gex_df.empty:
        st.warning("No GEX table loaded (gex_df is empty). Run Fetch Data first.")
        return

    work = gex_df.copy()

    strike_col = _find_contains(work, "strike") or _find_contains(work, "k")
    netgex_col = (
        _find_contains(work, "net", "gex")
        or _find_contains(work, "total", "gex")
        or _find_contains(work, "gex")
    )

    if not strike_col or not netgex_col:
        st.error("Could not find strike / net GEX columns in gex_df.")
        st_df(pd.DataFrame({"columns": list(work.columns)}), height=220)
        return

    work["strike"] = _to_num(work[strike_col])
    work["net_gex"] = _to_num(work[netgex_col])

    work = work.dropna(subset=["strike", "net_gex"]).sort_values("strike")
    if work.empty:
        st.warning("GEX data exists but strike/net_gex became empty after cleaning.")
        return

    # Find closest to zero net_gex as "flip" (best robust)
    flip_idx = (work["net_gex"].abs()).idxmin()
    gamma_flip = float(work.loc[flip_idx, "strike"])
    flip_val = float(work.loc[flip_idx, "net_gex"])

    # Determine sign regime around spot
    # interpolate net_gex at spot using nearest strikes
    near = work.iloc[(work["strike"] - spot).abs().argsort()[:6]].copy()
    near = near.sort_values("strike")
    approx_net_at_spot = float(near["net_gex"].mean()) if not near.empty else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"{spot:,.2f}")
    c2.metric("Gamma Flip (≈0 net GEX)", f"{gamma_flip:,.2f}")
    c3.metric("Net GEX near flip", f"{flip_val:,.0f}")

    if np.isfinite(approx_net_at_spot):
        regime = "LONG GAMMA (pin/mean-revert)" if approx_net_at_spot > 0 else "SHORT GAMMA (move/expand)"
        st.info(f"Approx dealer gamma regime near spot: **{regime}** (based on average net GEX near spot).")

    # Plot net GEX curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=work["strike"],
        y=work["net_gex"],
        mode="lines+markers",
        name="Net GEX",
        hovertemplate="Strike: %{x}<br>Net GEX: %{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=spot, line_width=1, line_dash="dot", annotation_text="Spot", annotation_position="top")
    fig.add_vline(x=gamma_flip, line_width=2, line_dash="dash", annotation_text="Gamma Flip", annotation_position="top")
    fig.add_hline(y=0, line_width=1)

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title=f"Net GEX by Strike {('— ' + symbol) if symbol else ''}",
        xaxis_title="Strike",
        yaxis_title="Net GEX",
    )
    st_plot(fig)

    with st.expander("🔎 Table (cleaned)", expanded=False):
        st_df(work[["strike", "net_gex"]], height=360)

    st.caption(
        "Interpretation:\n"
        "- Above 0 net GEX typically implies more **pinning / mean reversion**.\n"
        "- Below 0 net GEX typically implies more **trend / volatility expansion**.\n"
        "Gamma flip is a key inflection area, not a guaranteed reversal."
    )