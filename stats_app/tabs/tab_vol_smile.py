import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re


# -----------------------------
# Helpers: column detection
# -----------------------------
def _norm(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def _find_col_exactish(df: pd.DataFrame, candidates: list[str]):
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _find_col_contains(df: pd.DataFrame, needles: list[str], prefer_suffix: str | None = None):
    scored = []
    for c in df.columns:
        nc = _norm(c)
        if all(n in nc for n in needles):
            score = 0
            if prefer_suffix and str(c).endswith(prefer_suffix):
                score += 6
            score += max(0, 3 - len(nc) // 16)
            scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _find_strike_col(df: pd.DataFrame):
    return (
        _find_col_exactish(df, ["Strike", "Strike Price", "StrikePrice"])
        or _find_col_contains(df, ["strike"])
    )


def _find_call_put_iv_cols(df: pd.DataFrame):
    # Side-by-side exports often have:
    # Call IV column = "IV"
    # Put  IV column = "IV.1"
    call_iv = _find_col_exactish(df, ["IV"]) or _find_col_contains(df, ["iv"])
    put_iv = _find_col_exactish(df, ["IV.1", "IV_1"]) or _find_col_contains(df, ["iv"], prefer_suffix=".1")
    if call_iv == put_iv:
        put_iv = None
    return call_iv, put_iv


def _find_open_int_cols(df: pd.DataFrame):
    # Side-by-side exports often have:
    # Call OI column = "Open Int"
    # Put  OI column = "Open Int.1"
    call_oi = _find_col_exactish(df, ["Open Int", "Open Interest"]) or _find_col_contains(df, ["open", "int"])
    put_oi = (
        _find_col_exactish(df, ["Open Int.1", "Open Interest.1", "Open Int_1", "Open Interest_1"])
        or _find_col_contains(df, ["open", "int"], prefer_suffix=".1")
    )
    if call_oi == put_oi:
        put_oi = None
    return call_oi, put_oi


# -----------------------------
# Helpers: parsing
# -----------------------------
def _to_float_series(s: pd.Series) -> pd.Series:
    # Handles "123.4%", "1,234.5", "—", blanks, etc.
    return (
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("—", "", regex=False)
        .str.replace("–", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def _parse_oi_col(series: pd.Series) -> pd.Series:
    # Handles "1,234", blanks, em-dash, etc.
    s = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("—", "", regex=False)
        .str.replace("–", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


# -----------------------------
# Helpers: smoothing
# -----------------------------
def _weighted_polyfit(x: np.ndarray, y: np.ndarray, w: np.ndarray, degree: int):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < max(degree + 2, 8):
        return None

    x_mu = float(np.mean(x))
    x_sd = float(np.std(x)) if float(np.std(x)) > 1e-9 else 1.0
    xn = (x - x_mu) / x_sd

    try:
        coefs = np.polyfit(xn, y, deg=degree, w=w)
        return coefs, x_mu, x_sd
    except Exception:
        return None


def _poly_eval(coefs_pack, x: np.ndarray):
    if coefs_pack is None:
        return None
    coefs, x_mu, x_sd = coefs_pack
    xn = (np.asarray(x, dtype=float) - x_mu) / x_sd
    return np.polyval(coefs, xn)


def _nearest_iv_at_moneyness(df: pd.DataFrame, m_target: float, iv_col: str):
    if df.empty or iv_col not in df.columns:
        return None
    idx = (df["m"] - m_target).abs().idxmin()
    v = df.loc[idx, iv_col]
    return float(v) if pd.notna(v) else None


# -----------------------------
# Main render
# -----------------------------
def render_tab_vol_smile(df: pd.DataFrame, spot: float):
    st.subheader("📐 Volatility Smile (Single Expiry)")
    st.caption("IV vs Strike and IV vs Moneyness (K/S) — cleaned + smoothed (OI-weighted when available).")

    if df is None or df.empty:
        st.warning("No chain data loaded. Click `🔄 Fetch Data` first.")
        return

    if not spot or float(spot) <= 0:
        st.warning("Spot is missing/invalid. Enter manual spot in sidebar or refresh spot.")
        return
    spot = float(spot)

    strike_col = _find_strike_col(df)
    call_iv_col, put_iv_col = _find_call_put_iv_cols(df)
    call_oi_col, put_oi_col = _find_open_int_cols(df)

    with st.expander("🔎 Debug: detected columns", expanded=False):
        st.write(
            {
                "strike_col": strike_col,
                "call_iv_col": call_iv_col,
                "put_iv_col": put_iv_col,
                "call_oi_col": call_oi_col,
                "put_oi_col": put_oi_col,
            }
        )
        st.write("All columns:", list(df.columns))

    if strike_col is None or call_iv_col is None:
        st.error("Required columns not found (need Strike + at least one IV column).")
        return

    # Controls
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.2, 1.2, 1.2, 1.2])
    with c1:
        min_oi = st.number_input("Min OI (auto-disabled if OI not usable)", min_value=0, max_value=5_000_000, value=50, step=10)
    with c2:
        max_iv = st.slider("Max IV (%)", min_value=50, max_value=500, value=250, step=10)
    with c3:
        m_low = st.slider("Moneyness min (K/S)", min_value=0.10, max_value=1.00, value=0.60, step=0.05)
    with c4:
        m_high = st.slider("Moneyness max (K/S)", min_value=1.00, max_value=3.00, value=1.40, step=0.05)
    with c5:
        degree = st.selectbox("Smooth degree (fit)", options=[2, 3, 4], index=1)

    show_calls = st.checkbox("Show Calls", value=True)
    show_puts = st.checkbox("Show Puts", value=True)
    show_raw = st.checkbox("Show Raw Points", value=True)

    # Prepare + parse
    data = df.copy()

    data["strike"] = _to_float_series(data[strike_col])
    data["call_iv"] = _to_float_series(data[call_iv_col])
    data["put_iv"] = _to_float_series(data[put_iv_col]) if put_iv_col else np.nan

    # Robust OI parse (this is the fix you needed)
    data["call_oi"] = _parse_oi_col(df[call_oi_col]) if call_oi_col else 0.0
    data["put_oi"] = _parse_oi_col(df[put_oi_col]) if put_oi_col else 0.0

    data = data.dropna(subset=["strike"]).sort_values("strike")
    data["m"] = data["strike"] / spot

    # OI Debug (shows if parsing is working)
    with st.expander("🧪 OI Debug", expanded=False):
        st.write("Detected call_oi_col:", call_oi_col)
        st.write("Detected put_oi_col:", put_oi_col)
        st.write("call_oi sum:", float(pd.Series(data["call_oi"]).sum()))
        st.write("put_oi sum:", float(pd.Series(data["put_oi"]).sum()))
        if call_oi_col:
            st.write("Sample call OI raw:", df[call_oi_col].head(10).tolist())
        st.write("Sample call OI parsed:", pd.Series(data["call_oi"]).head(10).tolist())
        if put_oi_col:
            st.write("Sample put OI raw:", df[put_oi_col].head(10).tolist())
            st.write("Sample put OI parsed:", pd.Series(data["put_oi"]).head(10).tolist())

    before_n = len(data)

    # Clean IV
    data.loc[(data["call_iv"] <= 0) | (data["call_iv"] > float(max_iv)), "call_iv"] = np.nan
    if "put_iv" in data.columns:
        data.loc[(data["put_iv"] <= 0) | (data["put_iv"] > float(max_iv)), "put_iv"] = np.nan

    # Moneyness window
    data = data[(data["m"] >= float(m_low)) & (data["m"] <= float(m_high))]
    after_window = len(data)

    # Determine if OI is usable
    oi_usable = (float(pd.Series(data["call_oi"]).sum()) + float(pd.Series(data["put_oi"]).sum())) > 0
    applied_min_oi = int(min_oi) if oi_usable else 0

    if not oi_usable and min_oi > 0:
        st.warning("Open Interest columns look empty/non-numeric in this export → Min OI filter auto-disabled.")

    # Apply OI filter
    if applied_min_oi > 0:
        data = data[(data["call_oi"] >= applied_min_oi) | (data["put_oi"] >= applied_min_oi)]
    after_oi = len(data)

    # Failsafe relax
    if data.empty:
        st.warning("No strikes after cleaning. Auto-relaxing filters (disabling OI filter)…")
        data = df.copy()
        data["strike"] = _to_float_series(data[strike_col])
        data["call_iv"] = _to_float_series(data[call_iv_col])
        data["put_iv"] = _to_float_series(data[put_iv_col]) if put_iv_col else np.nan
        data = data.dropna(subset=["strike"]).sort_values("strike")
        data["m"] = data["strike"] / spot

        data.loc[(data["call_iv"] <= 0) | (data["call_iv"] > float(max_iv)), "call_iv"] = np.nan
        if "put_iv" in data.columns:
            data.loc[(data["put_iv"] <= 0) | (data["put_iv"] > float(max_iv)), "put_iv"] = np.nan

        data = data[(data["m"] >= float(m_low)) & (data["m"] <= float(m_high))]

    if data.empty:
        st.error("Still no strikes available. This usually means IV values are missing/0 for this expiry export.")
        return

    dropped = before_n - len(data)
    st.caption(f"Rows kept: {len(data)} (dropped {dropped}). Window kept: {after_window}. After OI: {after_oi}.")

    calls = data.dropna(subset=["call_iv"]).copy()
    puts = data.dropna(subset=["put_iv"]).copy()

    # weights: OI when available else equal weights
    if oi_usable:
        calls["w"] = np.clip(calls["call_oi"].values.astype(float), 1.0, None)
        puts["w"] = np.clip(puts["put_oi"].values.astype(float), 1.0, None)
    else:
        calls["w"] = 1.0
        puts["w"] = 1.0

    # Fit in moneyness space
    call_fit = _weighted_polyfit(calls["m"].values, calls["call_iv"].values, calls["w"].values, degree=degree) if len(calls) else None
    put_fit = _weighted_polyfit(puts["m"].values, puts["put_iv"].values, puts["w"].values, degree=degree) if len(puts) else None

    m_grid = np.linspace(float(m_low), float(m_high), 180)
    k_grid = m_grid * spot
    call_smooth = _poly_eval(call_fit, m_grid) if call_fit else None
    put_smooth = _poly_eval(put_fit, m_grid) if put_fit else None

    # --------------------------
    # Metrics
    # --------------------------
    def _iv_at(m_target: float, side: str):
        if side == "call":
            if call_smooth is not None:
                i = int(np.argmin(np.abs(m_grid - m_target)))
                return float(call_smooth[i])
            return _nearest_iv_at_moneyness(calls, m_target, "call_iv")
        else:
            if put_smooth is not None:
                i = int(np.argmin(np.abs(m_grid - m_target)))
                return float(put_smooth[i])
            return _nearest_iv_at_moneyness(puts, m_target, "put_iv")

    atm_call = _iv_at(1.00, "call") if show_calls else None
    atm_put = _iv_at(1.00, "put") if show_puts else None
    iv90_put = _iv_at(0.90, "put") if show_puts else None
    iv110_call = _iv_at(1.10, "call") if show_calls else None

    atm_vals = [v for v in [atm_call, atm_put] if isinstance(v, (int, float)) and np.isfinite(v)]
    atm_iv = float(np.mean(atm_vals)) if atm_vals else None

    skew = (float(iv90_put) - atm_iv) if (atm_iv is not None and iv90_put is not None) else None
    curvature = (float(np.mean([iv90_put, iv110_call])) - atm_iv) if (atm_iv is not None and iv90_put is not None and iv110_call is not None) else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ATM IV (%)", f"{atm_iv:.2f}" if atm_iv is not None else "—")
    m2.metric("IV @ 0.90 (Put wing)", f"{iv90_put:.2f}" if iv90_put is not None else "—")
    m3.metric("IV @ 1.10 (Call wing)", f"{iv110_call:.2f}" if iv110_call is not None else "—")
    if skew is not None and curvature is not None:
        m4.metric("Skew / Curvature", f"{skew:+.2f} / {curvature:+.2f}")
    else:
        m4.metric("Skew / Curvature", "—")

    # --------------------------
    # Chart 1: IV vs Strike
    # --------------------------
    fig1 = go.Figure()

    if show_raw and show_calls and not calls.empty:
        fig1.add_trace(
            go.Scatter(
                x=calls["strike"],
                y=calls["call_iv"],
                mode="markers",
                name="Calls (raw)",
                opacity=0.25,
                hovertemplate="Strike: %{x}<br>Call IV: %{y:.2f}%<extra></extra>",
            )
        )
    if show_raw and show_puts and not puts.empty:
        fig1.add_trace(
            go.Scatter(
                x=puts["strike"],
                y=puts["put_iv"],
                mode="markers",
                name="Puts (raw)",
                opacity=0.25,
                hovertemplate="Strike: %{x}<br>Put IV: %{y:.2f}%<extra></extra>",
            )
        )

    if show_calls and call_smooth is not None:
        fig1.add_trace(
            go.Scatter(
                x=k_grid,
                y=call_smooth,
                mode="lines",
                name="Calls (smooth)",
                line=dict(width=3),
                hovertemplate="Strike: %{x:.2f}<br>Call IV (smooth): %{y:.2f}%<extra></extra>",
            )
        )
    if show_puts and put_smooth is not None:
        fig1.add_trace(
            go.Scatter(
                x=k_grid,
                y=put_smooth,
                mode="lines",
                name="Puts (smooth)",
                line=dict(width=3),
                hovertemplate="Strike: %{x:.2f}<br>Put IV (smooth): %{y:.2f}%<extra></extra>",
            )
        )

    fig1.add_vline(x=spot, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.35)")
    fig1.update_layout(
        template="plotly_dark",
        height=560,
        title="IV vs Strike (Clean + Smoothed)",
        xaxis_title="Strike",
        yaxis_title="IV (%)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig1.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig1.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig1, use_container_width=True)

    # --------------------------
    # Chart 2: IV vs Moneyness
    # --------------------------
    fig2 = go.Figure()

    if show_raw and show_calls and not calls.empty:
        fig2.add_trace(
            go.Scatter(
                x=calls["m"],
                y=calls["call_iv"],
                mode="markers",
                name="Calls (raw)",
                opacity=0.25,
                hovertemplate="K/S: %{x:.4f}<br>Call IV: %{y:.2f}%<extra></extra>",
            )
        )
    if show_raw and show_puts and not puts.empty:
        fig2.add_trace(
            go.Scatter(
                x=puts["m"],
                y=puts["put_iv"],
                mode="markers",
                name="Puts (raw)",
                opacity=0.25,
                hovertemplate="K/S: %{x:.4f}<br>Put IV: %{y:.2f}%<extra></extra>",
            )
        )

    if show_calls and call_smooth is not None:
        fig2.add_trace(
            go.Scatter(
                x=m_grid,
                y=call_smooth,
                mode="lines",
                name="Calls (smooth)",
                line=dict(width=3),
                hovertemplate="K/S: %{x:.4f}<br>Call IV (smooth): %{y:.2f}%<extra></extra>",
            )
        )
    if show_puts and put_smooth is not None:
        fig2.add_trace(
            go.Scatter(
                x=m_grid,
                y=put_smooth,
                mode="lines",
                name="Puts (smooth)",
                line=dict(width=3),
                hovertemplate="K/S: %{x:.4f}<br>Put IV (smooth): %{y:.2f}%<extra></extra>",
            )
        )

    fig2.add_vline(x=1.0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.35)")
    fig2.update_layout(
        template="plotly_dark",
        height=560,
        title="IV vs Moneyness (K/S) (Clean + Smoothed)",
        xaxis_title="Strike / Spot (K/S)",
        yaxis_title="IV (%)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig2.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig2, use_container_width=True)

    # Interpretation
    st.markdown("### 🧠 Interpretation (from cleaned smile)")
    bullets = []
    if atm_iv is not None:
        bullets.append(f"- **ATM IV ~ {atm_iv:.1f}%** (baseline implied movement).")
    if skew is not None:
        if skew > 0:
            bullets.append(f"- **Put skew positive ({skew:+.1f})** → downside hedges priced richer vs ATM.")
        else:
            bullets.append(f"- **Put skew weak/negative ({skew:+.1f})** → downside hedges not priced aggressively.")
    if curvature is not None:
        if curvature > 0:
            bullets.append(f"- **Curvature positive ({curvature:+.1f})** → wings priced higher than ATM (real smile).")
        else:
            bullets.append(f"- **Curvature flat/negative ({curvature:+.1f})** → shallow smile in this window.")
    bullets.append("- If you still see the OI warning, open **🧪 OI Debug** and check parsed OI samples.")
    st.write("\n".join(bullets))

    st.success("✅ Smile rendered. If OI parses, smoothing is OI-weighted; otherwise it falls back to equal weights.")