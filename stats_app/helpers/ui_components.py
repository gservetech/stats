import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

def st_df(df: pd.DataFrame, height=None, hide_index: bool = True):
    """
    Streamlit 2026+ prefers width=... and rejects height=None.
    This wrapper:
      - uses width="stretch" when supported
      - only passes height if it's a real value (int/"stretch"/"content")
      - falls back to use_container_width for older Streamlit
    """
    kwargs = {"hide_index": hide_index}
    kwargs["width"] = "stretch"

    if height is not None:
        if isinstance(height, str):
            kwargs["height"] = height
        else:
            kwargs["height"] = int(height)

    if kwargs.get("height", "__missing__") is None:
        kwargs.pop("height", None)

    try:
        st.dataframe(df, **kwargs)
    except TypeError:
        fallback_kwargs = {"hide_index": hide_index}
        if height is not None:
            fallback_kwargs["height"] = int(height)
        st.dataframe(df, **fallback_kwargs)

def st_plot(fig, key: str | None = None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, key=key)

def st_btn(label: str, disabled: bool = False, key: str | None = None):
    try:
        return st.button(label, width="stretch", disabled=disabled, key=key)
    except TypeError:
        return st.button(label, disabled=disabled, key=key)

def _parse_num(val):
    if pd.isna(val) or val == "":
        return 0.0
    return float(str(val).replace(",", ""))

def create_oi_charts(df: pd.DataFrame):
    df = df.copy()
    df["call_oi"] = df["Call OI"].apply(_parse_num)
    df["put_oi"] = df["Put OI"].apply(_parse_num)
    df["strike_num"] = df["Strike"].apply(_parse_num)
    df_sorted = df.sort_values("strike_num")

    bar_fig = make_subplots(rows=1, cols=2, subplot_titles=("📈 Calls OI (Bar)", "📉 Puts OI (Bar)"))
    bar_fig.add_trace(go.Bar(x=df_sorted["strike_num"], y=df_sorted["call_oi"], name="Calls"), row=1, col=1)
    bar_fig.add_trace(go.Bar(x=df_sorted["strike_num"], y=df_sorted["put_oi"], name="Puts"), row=1, col=2)
    bar_fig.update_layout(template="plotly_dark", height=350, showlegend=False)
    bar_fig.update_xaxes(title_text="Strike")
    bar_fig.update_yaxes(title_text="Open Interest")

    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(x=df_sorted["strike_num"], y=df_sorted["call_oi"], mode="lines+markers", name="Call OI"))
    line_fig.add_trace(
        go.Scatter(x=df_sorted["strike_num"], y=df_sorted["put_oi"], mode="lines+markers", name="Put OI"))
    line_fig.update_layout(
        title="📊 Call vs Put Open Interest by Strike",
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )
    line_fig.update_xaxes(title="Strike")
    line_fig.update_yaxes(title="Open Interest")

    return bar_fig, line_fig

def create_top_strikes_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x_col], y=df[y_col]))
    fig.update_layout(template="plotly_dark", height=320, title=title)
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title=y_col)
    return fig
