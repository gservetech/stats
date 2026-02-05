import streamlit as st
from ..helpers.ui_components import st_df, st_plot, create_top_strikes_chart
from ..helpers.calculations import compute_gamma_map_artifacts


def render_tab_weekly_gamma(pcr, totals, w, spot, gex_df, art=None):
    """
    Render the Weekly Gamma / GEX tab.
    
    Args:
        pcr: Put/Call ratio data
        totals: Total GEX values
        w: Weekly data dict (for spot fallback)
        spot: Current spot price
        gex_df: Full per-strike GEX DataFrame
        art: Optional pre-computed gamma artifacts from compute_gamma_map_artifacts.
             If None, will compute from gex_df.
    """
    st.subheader("📌 Weekly Gamma / GEX (Dealer Positioning)")
    
    # Compute artifacts if not provided (single source of truth)
    if art is None and gex_df is not None and not gex_df.empty:
        art = compute_gamma_map_artifacts(gex_df, spot=spot, top_n=10)
    
    spot_used = art.get("spot_used") if art else (w.get("spot") or spot)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Put/Call Ratio (OI)", f"{(pcr.get('oi') or 0):.3f}" if pcr.get("oi") is not None else "N/A")
    c2.metric("Put/Call Ratio (Volume)", f"{(pcr.get('volume') or 0):.3f}" if pcr.get("volume") is not None else "N/A")
    c3.metric("Total Net GEX", f"{(totals.get('net_gex') or 0):,.0f}")
    c4.metric("Spot Used", f"{float(spot_used):,.2f}" if spot_used is not None else "N/A")

    # Get tables from artifacts (same df, same spot)
    top_call = art.get("top_call") if art else None
    top_put = art.get("top_put") if art else None
    top_net = art.get("top_net") if art else None
    
    # Handle legacy calls that pass these separately
    if top_call is None:
        import pandas as pd
        top_call = pd.DataFrame()
    if top_put is None:
        import pandas as pd
        top_put = pd.DataFrame()
    if top_net is None:
        import pandas as pd
        top_net = pd.DataFrame()

    st.markdown("### 🧲 Top Strikes (Gamma Walls / Magnets)")
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("**Top Call GEX**")
        if not top_call.empty:
            st_df(top_call)
            if {"strike", "call_gex"}.issubset(top_call.columns):
                st_plot(create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX"), key="weekly_gamma_top_call_gex")
        else:
            st.info("No top call GEX data returned.")

    with colB:
        st.markdown("**Top Put GEX**")
        if not top_put.empty:
            st_df(top_put)
            if {"strike", "put_gex"}.issubset(top_put.columns):
                st_plot(create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX"), key="weekly_gamma_top_put_gex")
        else:
            st.info("No top put GEX data returned.")

    with colC:
        st.markdown("**Top Net GEX (abs)**")
        if not top_net.empty:
            top_net = top_net.copy()
            if "net_gex_abs" not in top_net.columns and "net_gex" in top_net.columns:
                top_net["net_gex_abs"] = top_net["net_gex"].abs()
            if "net_gex_abs" in top_net.columns:
                top_net = top_net.sort_values("net_gex_abs", ascending=False)
            st_df(top_net)
            y_col = "net_gex"
            title = "Top Net GEX (Directional)"
            if {"strike", y_col}.issubset(top_net.columns):
                st_plot(create_top_strikes_chart(top_net, "strike", y_col, title), key="weekly_gamma_top_net_gex")
        else:
            st.info("No top net GEX data returned.")

    # Combined top strikes from intersecting call/put
    st.markdown("### 🧩 Combined Top Call/Put (same strikes)")
    if not top_call.empty and not top_put.empty:
        import pandas as pd
        top_combined = pd.merge(
            top_call[["strike", "call_gex"]],
            top_put[["strike", "put_gex"]],
            on="strike",
            how="inner"
        )
        if not top_combined.empty and "call_gex" in top_combined.columns and "put_gex" in top_combined.columns:
            top_combined["net_gex"] = top_combined["call_gex"] + top_combined["put_gex"]
            top_combined["net_gex_abs"] = top_combined["net_gex"].abs()
            top_combined = top_combined.sort_values("net_gex_abs", ascending=False)
            st_df(top_combined)
            if {"strike", "net_gex"}.issubset(top_combined.columns):
                st_plot(create_top_strikes_chart(top_combined, "strike", "net_gex", "Combined Net GEX (Top Call/Put Strikes)"), key="weekly_gamma_combined_net_gex")
        else:
            st.info("No overlapping strikes found between top call and put GEX.")
    else:
        st.info("No combined top strikes data returned.")

    st.markdown("### 📚 Full Per-Strike GEX Table (all strikes)")
    if gex_df is not None and not gex_df.empty:
        show_df = gex_df.copy()
        if "strike" in show_df.columns:
            show_df = show_df.sort_values("strike")
        with st.expander("Show full table", expanded=False):
            st_df(show_df)
    else:
        st.info("No per-strike GEX data returned.")

    st.caption("Note: GEX is an approximation from IV + OI using Black-Scholes gamma; educational only.")
