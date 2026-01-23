import streamlit as st
from ..helpers.ui_components import st_df, st_plot, create_top_strikes_chart

def render_tab_weekly_gamma(pcr, totals, w, spot, top_call, top_put, top_net):
    st.subheader("📌 Weekly Gamma / GEX (Dealer Positioning)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Put/Call Ratio (OI)", f"{(pcr.get('oi') or 0):.3f}" if pcr.get("oi") is not None else "N/A")
    c2.metric("Put/Call Ratio (Volume)", f"{(pcr.get('volume') or 0):.3f}" if pcr.get("volume") is not None else "N/A")
    c3.metric("Total Net GEX", f"{(totals.get('net_gex') or 0):,.0f}")
    c4.metric("Spot Used", f"{float(w.get('spot') or spot):,.2f}")

    st.markdown("### 🧲 Top Strikes (Gamma Walls / Magnets)")
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("**Top Call GEX**")
        if not top_call.empty:
            st_df(top_call)
            if {"strike", "call_gex"}.issubset(top_call.columns):
                st_plot(create_top_strikes_chart(top_call, "strike", "call_gex", "Top Call GEX"))
        else:
            st.info("No top call GEX data returned.")

    with colB:
        st.markdown("**Top Put GEX**")
        if not top_put.empty:
            st_df(top_put)
            if {"strike", "put_gex"}.issubset(top_put.columns):
                st_plot(create_top_strikes_chart(top_put, "strike", "put_gex", "Top Put GEX"))
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
            y_col = "net_gex_abs" if "net_gex_abs" in top_net.columns else "net_gex"
            if {"strike", y_col}.issubset(top_net.columns):
                st_plot(create_top_strikes_chart(top_net, "strike", y_col, "Top Net GEX (abs)"))
        else:
            st.info("No top net GEX data returned.")

    st.caption("Note: GEX is an approximation from IV + OI using Black-Scholes gamma; educational only.")
