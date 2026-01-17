import streamlit as st
import pandas as pd
from ..helpers.ui_components import create_oi_charts, create_top_strikes_chart, st_plot, st_df

def render_tab_oi_charts(df):
    required_cols = {"Strike", "Call OI", "Put OI"}
    if required_cols.issubset(set(df.columns)):
        bar_fig, line_fig = create_oi_charts(df)
        st_plot(bar_fig)
        st_plot(line_fig)
        
        c1, c2 = st.columns(2)
        with c1:
            top_calls = df.sort_values("Call OI", ascending=False).head(10)[["Strike", "Call OI"]]
            st_plot(create_top_strikes_chart(top_calls, "Strike", "Call OI", "🔥 Top 10 Call OI Strikes"))
        with c2:
            top_puts = df.sort_values("Put OI", ascending=False).head(10)[["Strike", "Put OI"]]
            st_plot(create_top_strikes_chart(top_puts, "Strike", "Put OI", "💧 Top 10 Put OI Strikes"))
    else:
        st.warning("Dataframe missing required columns for OI charts (Strike, Call OI, Put OI)")
