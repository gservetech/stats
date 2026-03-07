from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stats_app.tabs.tab_yahoo_data import _build_chart_df, _fetch_chart_payload

_fragment = getattr(st, "fragment", lambda f: f)

RANGE_TO_YAHOO = {
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "MAX": "max",
}

RANGE_INTERVALS = {
    "6M": ["1d", "1wk"],
    "1Y": ["1d", "1wk"],
    "2Y": ["1d", "1wk", "1mo"],
    "5Y": ["1wk", "1mo"],
    "MAX": ["1wk", "1mo", "3mo"],
}

DEFAULT_INTERVAL = {
    "6M": "1d",
    "1Y": "1d",
    "2Y": "1d",
    "5Y": "1wk",
    "MAX": "1wk",
}

REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for col in out.columns:
        cl = str(col).strip().lower()
        if cl in {"date", "datetime"}:
            rename_map[col] = "Date"
        elif cl == "open":
            rename_map[col] = "Open"
        elif cl == "high":
            rename_map[col] = "High"
        elif cl == "low":
            rename_map[col] = "Low"
        elif cl == "close":
            rename_map[col] = "Close"
        elif cl in {"adj close", "adj_close"}:
            rename_map[col] = "Adj Close"
        elif cl == "volume":
            rename_map[col] = "Volume"

    out = out.rename(columns=rename_map)
    if "Date" not in out.columns:
        out = out.reset_index()
        if "Date" not in out.columns and "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "Date"})

    if "Date" not in out.columns:
        raise ValueError("Could not find a Date column in Yahoo history.")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in REQUIRED_PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.dropna(subset=REQUIRED_PRICE_COLUMNS).reset_index(drop=True)


def _fetch_yahoo_price_df(symbol: str, range_key: str, interval: str) -> pd.DataFrame:
    payload = _fetch_chart_payload(symbol=symbol, range_key=range_key, interval=interval)
    raw_df, _ = _build_chart_df(payload)
    if raw_df.empty:
        return pd.DataFrame()

    return standardize_price_columns(
        raw_df[["datetime", "open", "high", "low", "close", "volume"]].rename(
            columns={
                "datetime": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    )


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / max(length, 1)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).clip(0, 100)


def smooth_series(series: pd.Series, method: str = "EMA", length: int = 3) -> pd.Series:
    if length <= 1:
        return series.copy()
    if method.upper() == "SMA":
        return series.rolling(length, min_periods=1).mean()
    return series.ewm(span=length, adjust=False).mean()


def rolling_slope(series: pd.Series, window: int = 5) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=series.index)

    x = np.arange(window, dtype=float)

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x_mean = x.mean()
        y_mean = values.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - x_mean) * (values - y_mean)).sum() / denom)

    return series.rolling(window, min_periods=window).apply(lambda arr: _slope(np.asarray(arr, dtype=float)), raw=True)


def apply_kalman_like_smoothing(series: pd.Series, strength: float = 0.30) -> pd.Series:
    strength = float(np.clip(strength, 0.01, 0.99))
    values = series.astype(float).values.copy()
    if len(values) == 0:
        return series.copy()

    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        curr = values[i]
        prev = out[i - 1]
        if np.isnan(curr):
            out[i] = prev
        elif np.isnan(prev):
            out[i] = curr
        else:
            out[i] = prev + strength * (curr - prev)
    return pd.Series(out, index=series.index)


def build_features(
    df: pd.DataFrame,
    standard_rsi_col: str = "standard_rsi",
    rsi_mom_lag: int = 3,
    rsi_vol_window: int = 10,
    rsi_slope_window: int = 5,
    price_mom_lag: int = 5,
) -> pd.DataFrame:
    out = df.copy()
    out["rsi_mom"] = out[standard_rsi_col] - out[standard_rsi_col].shift(rsi_mom_lag)
    out["rsi_vol"] = out[standard_rsi_col].rolling(rsi_vol_window, min_periods=rsi_vol_window).std()
    out["rsi_slope"] = rolling_slope(out[standard_rsi_col], window=rsi_slope_window)
    out["price_mom"] = out["Close"] - out["Close"].shift(price_mom_lag)
    return out


def compute_ml_rsi(
    df: pd.DataFrame,
    rsi_length: int = 14,
    use_rsi_smoothing: bool = True,
    rsi_smoothing_method: str = "EMA",
    rsi_smoothing_length: int = 3,
    knn_neighbors: int = 5,
    knn_lookback: int = 100,
    knn_weight: float = 0.50,
    feature_count: int = 5,
    use_filter: bool = True,
    filter_strength: float = 0.30,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> pd.DataFrame:
    try:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import MinMaxScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("ML RSI Pro requires scikit-learn. Add `scikit-learn` to the environment.") from exc

    out = df.copy()
    out["rsi_raw"] = compute_rsi(out["Close"], length=rsi_length)
    out["standard_rsi"] = (
        smooth_series(out["rsi_raw"], rsi_smoothing_method, rsi_smoothing_length)
        if use_rsi_smoothing
        else out["rsi_raw"]
    )

    out = build_features(out, standard_rsi_col="standard_rsi")
    feature_candidates = ["standard_rsi", "rsi_mom", "rsi_vol", "rsi_slope", "price_mom"]
    feature_count = int(np.clip(feature_count, 2, len(feature_candidates)))
    selected_features = feature_candidates[:feature_count]

    out["knn_rsi"] = np.nan
    out["ml_rsi_raw"] = np.nan

    scaler = MinMaxScaler()
    knn_weight = float(np.clip(knn_weight, 0.0, 1.0))
    knn_neighbors = max(1, int(knn_neighbors))
    knn_lookback = max(knn_neighbors + 5, int(knn_lookback))

    valid_idx = out.dropna(subset=selected_features + ["standard_rsi"]).index.tolist()
    for idx in valid_idx:
        if idx < knn_lookback:
            continue

        window_df = out.iloc[idx - knn_lookback:idx].dropna(subset=selected_features + ["standard_rsi"])
        if len(window_df) < knn_neighbors:
            continue

        x_train = window_df[selected_features].values
        y_train = window_df["standard_rsi"].values
        x_curr = out.loc[[idx], selected_features].values
        if np.isnan(x_curr).any():
            continue

        x_train_scaled = scaler.fit_transform(x_train)
        x_curr_scaled = scaler.transform(x_curr)

        model = KNeighborsRegressor(n_neighbors=knn_neighbors, weights="distance")
        model.fit(x_train_scaled, y_train)

        pred = float(model.predict(x_curr_scaled)[0])
        base = float(out.at[idx, "standard_rsi"])
        ml_raw = ((1.0 - knn_weight) * base) + (knn_weight * pred)

        out.at[idx, "knn_rsi"] = pred
        out.at[idx, "ml_rsi_raw"] = ml_raw

    out["ml_rsi"] = out["ml_rsi_raw"].fillna(out["standard_rsi"])
    out["knn_rsi"] = out["knn_rsi"].fillna(out["standard_rsi"])

    if use_filter:
        out["ml_rsi"] = apply_kalman_like_smoothing(out["ml_rsi"], strength=filter_strength)

    out["ml_rsi"] = out["ml_rsi"].clip(0, 100)
    out["rsi_regime"] = np.where(
        out["ml_rsi"] >= overbought,
        "Overbought",
        np.where(out["ml_rsi"] <= oversold, "Oversold", np.where(out["ml_rsi"] >= 50, "Bullish", "Bearish")),
    )

    out["buy_signal"] = (
        (out["ml_rsi"].shift(1) < 50)
        & (out["ml_rsi"] >= 50)
        & (out["Close"] > out["Close"].rolling(20, min_periods=1).mean())
    )
    out["sell_signal"] = (
        (out["ml_rsi"].shift(1) > 50)
        & (out["ml_rsi"] <= 50)
        & (out["Close"] < out["Close"].rolling(20, min_periods=1).mean())
    )
    out["exit_long"] = (out["ml_rsi"].shift(1) > overbought) & (out["ml_rsi"] <= overbought)
    out["exit_short"] = (out["ml_rsi"].shift(1) < oversold) & (out["ml_rsi"] >= oversold)
    return out


def build_signal_summary(df: pd.DataFrame) -> dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    ml_rsi = float(last["ml_rsi"])
    slope = ml_rsi - float(prev["ml_rsi"])
    trend = "Bullish" if ml_rsi >= 50 else "Bearish"
    momentum = "Strong Up" if slope > 2 else "Up" if slope > 0 else "Strong Down" if slope < -2 else "Down" if slope < 0 else "Flat"

    if bool(last["buy_signal"]):
        action = "BUY"
    elif bool(last["sell_signal"]):
        action = "SHORT"
    elif bool(last["exit_long"]):
        action = "EXIT LONG"
    elif bool(last["exit_short"]):
        action = "EXIT SHORT"
    else:
        action = "HOLD"

    return {
        "ML RSI": round(ml_rsi, 2),
        "Trend": trend,
        "Momentum": momentum,
        "Regime": str(last["rsi_regime"]),
        "Signal": action,
        "Standard RSI": round(float(last["standard_rsi"]), 2),
        "KNN RSI": round(float(last["knn_rsi"]), 2),
    }


def make_ml_rsi_chart(df: pd.DataFrame, overbought: float = 70.0, oversold: float = 30.0, title: str = "ML RSI Pro") -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.62, 0.38],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    buy_df = df[df["buy_signal"]]
    sell_df = df[df["sell_signal"]]
    exit_long_df = df[df["exit_long"]]
    exit_short_df = df[df["exit_short"]]

    fig.add_trace(
        go.Scatter(
            x=buy_df["Date"],
            y=buy_df["Low"] * 0.995,
            mode="markers+text",
            name="BUY1",
            text=["BUY1"] * len(buy_df),
            textposition="top center",
            marker=dict(size=10, symbol="triangle-up", color="#00d084"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sell_df["Date"],
            y=sell_df["High"] * 1.005,
            mode="markers+text",
            name="SELL1",
            text=["SELL1"] * len(sell_df),
            textposition="top center",
            marker=dict(size=10, symbol="triangle-down", color="#ff6377"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=exit_long_df["Date"],
            y=exit_long_df["High"] * 1.01,
            mode="markers+text",
            name="EXIT LONG",
            text=["EXIT L"] * len(exit_long_df),
            textposition="top center",
            marker=dict(size=9, symbol="x", color="#ffd166"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=exit_short_df["Date"],
            y=exit_short_df["Low"] * 0.99,
            mode="markers+text",
            name="EXIT SHORT",
            text=["EXIT S"] * len(exit_short_df),
            textposition="bottom center",
            marker=dict(size=9, symbol="x", color="#9d8cff"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["standard_rsi"], mode="lines", name="Standard RSI", line=dict(width=1, dash="dot", color="#67b7ff")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["knn_rsi"], mode="lines", name="KNN RSI", line=dict(width=1, dash="dash", color="#ffd166")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["ml_rsi"], mode="lines", name="Final ML RSI", line=dict(width=3, color="#00d084")),
        row=2,
        col=1,
    )

    for level in (overbought, 50, oversold):
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(255,255,255,0.28)", row=2, col=1)

    fig.update_layout(
        title=title,
        height=850,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


def _classify_signal(row: pd.Series) -> str:
    if row["buy_signal"]:
        return "BUY1"
    if row["sell_signal"]:
        return "SELL1"
    if row["exit_long"]:
        return "EXIT LONG"
    if row["exit_short"]:
        return "EXIT SHORT"
    return "NONE"


@_fragment
def render_tab_ml_rsi_pro(symbol: str) -> None:
    target_symbol = (symbol or "AAPL").upper().strip()
    st.subheader("ML RSI Pro")
    st.caption("Adaptive RSI using KNN pattern matching and smoothing on Yahoo Finance history.")

    with st.expander("What this tab does", expanded=False):
        st.write(
            "- Computes standard RSI\n"
            "- Builds RSI momentum, volatility, slope, and price momentum features\n"
            "- Finds similar historical setups with KNN\n"
            "- Blends standard RSI with KNN-adjusted RSI\n"
            "- Smooths the final output and marks BUY / SHORT / EXIT events"
        )

    a, b = st.columns(2)
    with a:
        range_label = st.selectbox(
            "History",
            options=list(RANGE_TO_YAHOO.keys()),
            index=1,
            key=f"ml_rsi_range_{target_symbol}",
        )
    with b:
        valid_intervals = RANGE_INTERVALS[range_label]
        interval = st.selectbox(
            "Interval",
            options=valid_intervals,
            index=valid_intervals.index(DEFAULT_INTERVAL[range_label]),
            key=f"ml_rsi_interval_{target_symbol}_{range_label}",
        )

    try:
        df = _fetch_yahoo_price_df(target_symbol, RANGE_TO_YAHOO[range_label], interval)
    except Exception as exc:
        st.error(f"Unable to load Yahoo history: {exc}")
        return

    if df.empty:
        st.warning("Yahoo returned no price rows for this selection.")
        return

    if len(df) < 150:
        st.warning("This model works better with at least 150 rows of history.")

    st.markdown("### Parameters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rsi_length = st.number_input("RSI Length", min_value=2, max_value=100, value=14, step=1, key=f"ml_rsi_len_{target_symbol}")
        knn_neighbors = st.number_input("KNN Neighbors", min_value=1, max_value=50, value=5, step=1, key=f"ml_rsi_knn_n_{target_symbol}")
    with c2:
        knn_lookback = st.number_input("KNN Lookback", min_value=20, max_value=1000, value=100, step=10, key=f"ml_rsi_knn_lb_{target_symbol}")
        knn_weight = st.slider("KNN Weight", min_value=0.0, max_value=1.0, value=0.50, step=0.05, key=f"ml_rsi_knn_w_{target_symbol}")
    with c3:
        feature_count = st.slider("Feature Count", min_value=2, max_value=5, value=5, step=1, key=f"ml_rsi_feat_{target_symbol}")
        overbought = st.slider("Overbought", min_value=50, max_value=95, value=70, step=1, key=f"ml_rsi_ob_{target_symbol}")
    with c4:
        oversold = st.slider("Oversold", min_value=5, max_value=50, value=30, step=1, key=f"ml_rsi_os_{target_symbol}")
        filter_strength = st.slider("Filter Strength", min_value=0.01, max_value=0.99, value=0.30, step=0.01, key=f"ml_rsi_filter_{target_symbol}")

    s1, s2, s3 = st.columns(3)
    with s1:
        use_rsi_smoothing = st.checkbox("Use RSI Smoothing", value=True, key=f"ml_rsi_smooth_enable_{target_symbol}")
    with s2:
        rsi_smoothing_method = st.selectbox("RSI Smoothing Method", options=["EMA", "SMA"], index=0, key=f"ml_rsi_smooth_method_{target_symbol}")
    with s3:
        rsi_smoothing_length = st.number_input("RSI Smoothing Length", min_value=1, max_value=50, value=3, step=1, key=f"ml_rsi_smooth_len_{target_symbol}")

    use_filter = st.checkbox("Use Final Smoothing Filter", value=True, key=f"ml_rsi_filter_enable_{target_symbol}")

    min_needed = max(int(knn_lookback), int(rsi_length), int(rsi_smoothing_length), 150)
    if len(df) < min_needed:
        st.warning(
            f"Not enough rows for current parameters ({len(df)} rows, need about {min_needed}+). Increase history or reduce lookbacks."
        )
        return

    try:
        result_df = compute_ml_rsi(
            df=df,
            rsi_length=int(rsi_length),
            use_rsi_smoothing=bool(use_rsi_smoothing),
            rsi_smoothing_method=str(rsi_smoothing_method),
            rsi_smoothing_length=int(rsi_smoothing_length),
            knn_neighbors=int(knn_neighbors),
            knn_lookback=int(knn_lookback),
            knn_weight=float(knn_weight),
            feature_count=int(feature_count),
            use_filter=bool(use_filter),
            filter_strength=float(filter_strength),
            overbought=float(overbought),
            oversold=float(oversold),
        )
    except Exception as exc:
        st.error(f"Calculation error: {exc}")
        return

    summary = build_signal_summary(result_df)
    st.markdown("### Signal Box")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ML RSI", summary["ML RSI"])
    m2.metric("Trend", summary["Trend"])
    m3.metric("Momentum", summary["Momentum"])
    m4.metric("Signal", summary["Signal"])

    st.json(
        {
            "Regime": summary["Regime"],
            "Standard RSI": summary["Standard RSI"],
            "KNN RSI": summary["KNN RSI"],
            "Rows": len(result_df),
            "Data Source": f"Yahoo Finance ({range_label} / {interval})",
        },
        expanded=True,
    )

    st.markdown("### Chart")
    st.plotly_chart(
        make_ml_rsi_chart(result_df, overbought=float(overbought), oversold=float(oversold), title="ML RSI Pro"),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
        key=f"ml_rsi_chart_{target_symbol}_{range_label}_{interval}",
    )

    st.markdown("### Recent Signal Events")
    signal_rows = result_df.loc[
        result_df["buy_signal"] | result_df["sell_signal"] | result_df["exit_long"] | result_df["exit_short"],
        ["Date", "Close", "standard_rsi", "knn_rsi", "ml_rsi", "buy_signal", "sell_signal", "exit_long", "exit_short"],
    ].copy()

    if not signal_rows.empty:
        signal_rows["Signal"] = signal_rows.apply(_classify_signal, axis=1)
        signal_rows["Date"] = pd.to_datetime(signal_rows["Date"]).dt.strftime("%Y-%m-%d %H:%M")
        signal_rows = signal_rows[["Date", "Signal", "Close", "standard_rsi", "knn_rsi", "ml_rsi"]].tail(20)
        st.dataframe(signal_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No recent signal events found with the current settings.")

    st.markdown("### Export")
    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ML RSI Output CSV",
        data=csv_bytes,
        file_name=f"{target_symbol.lower()}_ml_rsi_pro_output.csv",
        mime="text/csv",
    )
