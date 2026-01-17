import streamlit as st
from ..helpers.ui_components import st_df

def render_tab_options_chain(df):
    st_df(df, height=520)
