import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stats_app.helpers.data_fetching import fetch_cnbc_chart_data

@st.fragment
def render_symbol_chart(symbol: str):
    # Timeframe selection matching the user's list
    timeframes = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "ALL"]
    
    # Initialize session state for timeframe if not present
    if "selected_chart_tf" not in st.session_state:
        st.session_state["selected_chart_tf"] = "1D"
        
    current_tf = st.session_state["selected_chart_tf"]

    # Header with Timeframe selector
    st.markdown("""
        <style>
        .tf-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 1px solid #3d4450;
            padding-bottom: 10px;
        }
        .tf-btn {
            background: none;
            border: none;
            color: #b0b5bc;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            padding: 5px 10px;
            border-radius: 4px;
        }
        .tf-btn:hover {
            color: white;
            background: #2a2e39;
        }
        .tf-btn.active {
            color: #00d775;
            background: rgba(0, 215, 117, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Timeframe selector row
    cols = st.columns(len(timeframes) + 2)
    for i, tf in enumerate(timeframes):
        if cols[i].button(tf, key=f"tf_btn_{tf}", width="stretch", 
                          type="primary" if current_tf == tf else "secondary"):
            st.session_state["selected_chart_tf"] = tf
            # No st.rerun() needed - fragment auto-reruns on widget interaction

    # Add a spacer and a full-screen icon lookalike (placeholder)
    cols[-1].markdown("""
        <div style="text-align: right; color: #b0b5bc; font-size: 20px; cursor: pointer; padding-top: 5px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
        </div>
    """, unsafe_allow_html=True)

    # Fetch data
    with st.spinner(f"Fetching {symbol} ({current_tf})..."):
        data = fetch_cnbc_chart_data(symbol, current_tf)
        
    if not data or "priceBars" not in data or not data["priceBars"]:
        st.warning(f"No chart data available for {symbol} ({current_tf})")
        return

    bars = data["priceBars"]
    df = pd.DataFrame(bars)
    
    if df.empty:
        st.info("The data pool for this symbol is currently empty.")
        return

    # Process data
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['open'] = pd.to_numeric(df['open'])
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'])
        
    # Find the date column
    date_col = None
    for col in ['tradeTime', 'tradeTimeinMills', 'dateTime']:
        if col in df.columns:
            date_col = col
            break
            
    if date_col == 'tradeTime':
        df['date'] = pd.to_datetime(df['tradeTime'], format='%Y%m%d%H%M%S', errors='coerce')
    elif date_col == 'tradeTimeinMills':
        df['date'] = pd.to_datetime(pd.to_numeric(df['tradeTimeinMills']), unit='ms', errors='coerce')
    elif date_col == 'dateTime':
        try:
            df['date'] = pd.to_datetime(df['dateTime'], format='%Y%m%d%H%M%S', errors='coerce')
        except:
            df['date'] = pd.to_datetime(df['dateTime'], errors='coerce')
    else:
        st.error(f"Could not find timestamp column in data. Keys: {list(df.columns)}")
        return

    df = df.sort_values('date')

    # Chart Header (Symbol + Price)
    last_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[0]
    change = last_price - prev_price
    pct_change = (change / prev_price) * 100 if prev_price != 0 else 0
    
    color = "#00d775" if change >= 0 else "#ff4b4b"
    arrow = "▲" if change >= 0 else "▼"
    
    col1, col2 = st.columns([2, 5])
    with col1:
        st.markdown(f"""
            <div style="padding: 10px 0;">
                <span style="font-size: 24px; font-weight: 700;">{symbol}</span>
                <span style="font-size: 24px; font-weight: 400; margin-left: 10px;">{last_price:,.2f}</span>
                <div style="color: {color}; font-size: 16px; margin-top: -5px;">
                    {arrow} {abs(change):.2f} ({abs(pct_change):.2f}%)
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Volume (or proxy) series for bottom bars
    has_volume = "volume" in df.columns and df["volume"].notna().any()
    if has_volume:
        vol_series = df["volume"].fillna(0)
        vol_label = "Volume"
    else:
        if "high" in df.columns and "low" in df.columns:
            vol_series = (df["high"] - df["low"]).abs().fillna(0)
        else:
            vol_series = df["close"].diff().abs().fillna(0)
        vol_label = "Activity (price-range proxy)"

    vol_display = vol_series.astype(float)

    # Price axis bounds (avoid 0 baseline flattening)
    if "low" in df.columns and df["low"].notna().any():
        price_min = float(df["low"].min())
    else:
        price_min = float(df["close"].min())
    if "high" in df.columns and df["high"].notna().any():
        price_max = float(df["high"].max())
    else:
        price_max = float(df["close"].max())
    price_pad = max((price_max - price_min) * 0.05, 0.5)

    # Up/Down coloring for volume bars
    if "open" in df.columns and df["open"].notna().any():
        up_mask = df["close"] >= df["open"]
    else:
        up_mask = df["close"] >= df["close"].shift(1)
    vol_colors = np.where(up_mask, "rgba(0,215,117,0.85)", "rgba(255,75,75,0.85)")

    # Bar width based on median time gap (thicker look)
    bar_kwargs = {}
    if len(df) > 1:
        gaps = df["date"].astype("int64").diff().dropna()
        if len(gaps) > 0:
            gap_ms = float(gaps.median()) / 1_000_000.0
            bar_kwargs["width"] = max(int(gap_ms * 0.8), 1)

    # Price chart
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=2.5),
            fillcolor=f'rgba({0 if color=="#ff4b4b" else 0}, {215 if color=="#00d775" else 75}, {117 if color=="#00d775" else 75}, 0.15)',
            name=symbol,
            hovertemplate="%{y:.2f}",
        )
    )
    price_fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    price_fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        showline=False,
        zeroline=False,
        tickfont=dict(color="#888"),
    )
    price_fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        side="right",
        showline=False,
        zeroline=False,
        tickfont=dict(color="#888"),
        tickprefix="",
        tickformat=".2f",
        range=[price_min - price_pad, price_max + price_pad],
    )

    # Volume chart (separate)
    volume_fig = go.Figure()
    volume_fig.add_trace(
        go.Bar(
            x=df["date"],
            y=vol_display,
            marker_color=vol_colors,
            name=vol_label,
            customdata=vol_series,
            hovertemplate=f"{vol_label}: %{{customdata:,.0f}}",
            opacity=0.95,
            marker_line_width=0,
            **bar_kwargs,
        )
    )
    volume_fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        bargap=0.0,
        bargroupgap=0.0,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    volume_fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        showline=False,
        zeroline=False,
        tickfont=dict(color="#888"),
    )
    vol_max = float(np.nanmax(vol_display)) if len(vol_display) else 0.0
    vol_upper = max(vol_max * 1.1, 10.0)
    volume_fig.update_yaxes(
        range=[0, vol_upper],
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        side="right",
        showline=False,
        zeroline=False,
        tickfont=dict(color="#888"),
        tickformat="~s",
        title=dict(text=vol_label, font=dict(color="#888", size=12)),
    )

    # Range indicator (vertical lines for day break if 5D)
    if current_tf == "5D":
        # Add subtle vertical lines at day boundaries
        days = df['date'].dt.date.unique()
        for day in days[1:]:
            day_start = pd.to_datetime(day)
            price_fig.add_vline(
                x=day_start,
                line_width=1,
                line_dash="dash",
                line_color="rgba(255,255,255,0.1)",
            )
            volume_fig.add_vline(
                x=day_start,
                line_width=1,
                line_dash="dash",
                line_color="rgba(255,255,255,0.1)",
            )

    st.plotly_chart(price_fig, width="stretch", config={'displayModeBar': False})
    st.plotly_chart(volume_fig, width="stretch", config={'displayModeBar': False})
