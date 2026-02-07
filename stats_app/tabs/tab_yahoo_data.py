from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

_fragment = getattr(st, "fragment", lambda f: f)

TIMEFRAME_TO_RANGE = {
    "1D": "1d",
    "5D": "5d",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "5Y": "5y",
    "ALL": "max",
}

TIMEFRAME_INTERVALS = {
    "1D": ["1m", "2m", "5m", "15m", "30m", "60m"],
    "5D": ["5m", "15m", "30m", "60m"],
    "1M": ["15m", "30m", "60m", "1d"],
    "3M": ["1d", "1wk"],
    "6M": ["1d", "1wk"],
    "YTD": ["1d", "1wk", "1mo"],
    "1Y": ["1d", "1wk", "1mo"],
    "5Y": ["1wk", "1mo", "3mo"],
    "ALL": ["1wk", "1mo", "3mo"],
}

DEFAULT_INTERVAL = {
    "1D": "1m",
    "5D": "15m",
    "1M": "30m",
    "3M": "1d",
    "6M": "1d",
    "YTD": "1d",
    "1Y": "1d",
    "5Y": "1wk",
    "ALL": "1wk",
}

_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://finance.yahoo.com/",
}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("parsedValue", "raw", "source"):
            if key in value:
                return _coerce_float(value[key])
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _interval_seconds(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    if interval.endswith("wk"):
        return int(interval[:-2]) * 7 * 86400
    if interval.endswith("mo"):
        return int(interval[:-2]) * 30 * 86400
    return 60


@st.cache_data(ttl=45, show_spinner=False)
def _fetch_chart_payload(symbol: str, range_key: str, interval: str) -> dict[str, Any]:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": range_key,
        "interval": interval,
        "includePrePost": "true",
        "events": "div|split|earn",
        "lang": "en-US",
        "region": "US",
        "source": "cosaic",
    }

    response = requests.get(url, params=params, headers=_YAHOO_HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(error.get("description") or str(error))

    result = chart.get("result")
    if not result:
        raise RuntimeError("Yahoo returned no chart result")

    return result[0]


def _build_chart_df(chart_payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = chart_payload.get("meta", {})
    timestamps = chart_payload.get("timestamp") or []
    quote = (chart_payload.get("indicators", {}).get("quote") or [{}])[0]

    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    rows: list[dict[str, Any]] = []
    timezone_name = meta.get("exchangeTimezoneName") or "America/New_York"

    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "timestamp": ts,
                "open": _coerce_float(opens[idx]) if idx < len(opens) else None,
                "high": _coerce_float(highs[idx]) if idx < len(highs) else None,
                "low": _coerce_float(lows[idx]) if idx < len(lows) else None,
                "close": _coerce_float(closes[idx]) if idx < len(closes) else None,
                "volume": _coerce_float(volumes[idx]) if idx < len(volumes) else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, meta

    dt_series = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    try:
        dt_series = dt_series.dt.tz_convert(timezone_name)
    except Exception:
        dt_series = dt_series.dt.tz_convert("America/New_York")
    df["datetime"] = dt_series

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "close"]).sort_values("datetime")
    return df, meta


def _build_main_chart(df: pd.DataFrame, symbol: str, interval: str) -> go.Figure:
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.86, 0.14],
        vertical_spacing=0.02,
    )

    close_series = df["close"].astype(float)
    gap_sec = df["datetime"].diff().dt.total_seconds().fillna(0)
    gap_threshold = max(_interval_seconds(interval) * 6, 4 * 3600)
    line_series = close_series.copy()
    line_series[gap_sec > gap_threshold] = np.nan

    move_up = close_series.diff().fillna(0) >= 0
    vol_colors = np.where(move_up, "#00c076", "#f24b5e")

    volume_raw = df["volume"].fillna(0).astype(float)
    if len(volume_raw) > 0:
        cap = float(volume_raw.quantile(0.995))
        volume_plot = volume_raw.clip(upper=cap) if cap > 0 else volume_raw
    else:
        volume_plot = volume_raw

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=line_series,
            mode="lines",
            name=f"{symbol} Close",
            line=dict(color="#0A84FF", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(10,132,255,0.24)",
            connectgaps=False,
            hovertemplate="%{x}<br>Price: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=volume_plot,
            marker=dict(color=vol_colors),
            customdata=volume_raw,
            opacity=0.95,
            hovertemplate="%{x}<br>Volume: %{customdata:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    last_price = float(close_series.iloc[-1])
    fig.add_hline(
        y=last_price,
        line_dash="dash",
        line_color="#ff3b5b",
        line_width=1,
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=1.0,
        xref="paper",
        y=last_price,
        yref="y",
        text=f"{last_price:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(color="white", size=12),
        bgcolor="#d53045",
        borderpad=4,
    )

    fig.update_layout(
        height=760,
        margin=dict(l=8, r=10, t=6, b=8),
        paper_bgcolor="#0a1220",
        plot_bgcolor="#0a1220",
        font=dict(color="#dbe5ef"),
        showlegend=False,
        bargap=0.0,
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(153,176,201,0.20)",
        tickfont=dict(size=12, color="#9eb1c7"),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(153,176,201,0.20)",
        tickfont=dict(size=12, color="#9eb1c7"),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="rgba(153,176,201,0.20)",
        tickformat=".2f",
        tickfont=dict(size=12, color="#9eb1c7"),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        side="right",
        showgrid=False,
        tickformat="~s",
        tickfont=dict(size=11, color="#9eb1c7"),
        row=2,
        col=1,
    )

    return fig


@_fragment
def render_tab_yahoo_data(symbol: str):
    symbol = (symbol or "AAPL").upper().strip()

    st.markdown(
        """
        <style>
        .yf-minimal-shell { background: #0a1220; border: 1px solid #1b2a3f; border-radius: 10px; padding: 8px 10px 10px 10px; }
        .yf-minimal-title { font-size: 22px; font-weight: 700; color: #e8f0fa; margin: 6px 0 8px 0; }
        .yf-minimal-up { color: #00d084; font-weight: 700; }
        .yf-minimal-down { color: #ff6377; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tf_key = f"yahoo_tf_{symbol}"
    iv_key = f"yahoo_iv_{symbol}"

    if tf_key not in st.session_state:
        st.session_state[tf_key] = "1D"
    if iv_key not in st.session_state:
        st.session_state[iv_key] = DEFAULT_INTERVAL[st.session_state[tf_key]]

    current_tf = st.session_state[tf_key]
    valid_intervals = TIMEFRAME_INTERVALS[current_tf]

    if st.session_state[iv_key] not in valid_intervals:
        st.session_state[iv_key] = DEFAULT_INTERVAL[current_tf]

    tf_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    timeframes = list(TIMEFRAME_TO_RANGE.keys())

    for idx, tf in enumerate(timeframes):
        clicked = tf_cols[idx].button(
            tf,
            key=f"yahoo_tf_btn_{symbol}_{tf}",
            type="primary" if current_tf == tf else "secondary",
            width="stretch",
        )
        if clicked:
            st.session_state[tf_key] = tf
            if st.session_state[iv_key] not in TIMEFRAME_INTERVALS[tf]:
                st.session_state[iv_key] = DEFAULT_INTERVAL[tf]
            current_tf = st.session_state[tf_key]
            valid_intervals = TIMEFRAME_INTERVALS[current_tf]

    selected_interval = tf_cols[-1].selectbox(
        "Interval",
        options=valid_intervals,
        index=valid_intervals.index(st.session_state[iv_key]),
        key=f"yahoo_iv_select_{symbol}_{current_tf}",
        label_visibility="collapsed",
    )
    st.session_state[iv_key] = selected_interval

    try:
        payload = _fetch_chart_payload(
            symbol=symbol,
            range_key=TIMEFRAME_TO_RANGE[current_tf],
            interval=st.session_state[iv_key],
        )
        df, meta = _build_chart_df(payload)
    except Exception as exc:
        st.error(f"Unable to load Yahoo chart data: {exc}")
        return

    if df.empty:
        st.warning("Yahoo returned no chart rows for this selection.")
        return

    close_now = float(df["close"].iloc[-1])
    prev_close = (
        _coerce_float(meta.get("chartPreviousClose"))
        or _coerce_float(meta.get("previousClose"))
        or float(df["close"].iloc[0])
    )
    change = close_now - prev_close
    pct = (change / prev_close) * 100 if prev_close else 0.0
    direction_class = "yf-minimal-up" if change >= 0 else "yf-minimal-down"

    st.markdown('<div class="yf-minimal-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin:4px 0 8px 0;">
            <div class="yf-minimal-title">{symbol}</div>
            <div style="text-align:right;">
                <div style="font-size:30px; font-weight:800; color:#eaf3ff;">{close_now:,.2f}</div>
                <div class="{direction_class}" style="font-size:14px;">{change:+.2f} ({pct:+.2f}%)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = _build_main_chart(df, symbol, st.session_state[iv_key])
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"yahoo_plot_{symbol}_{current_tf}_{st.session_state[iv_key]}",
    )

    st.markdown("</div>", unsafe_allow_html=True)
