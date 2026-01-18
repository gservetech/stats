# stats_app/tabs/tab_friday_predictor.py
"""Friday Expiry Predictor: 3-Engine Risk Book + Professional Playbook (no extra debug spam)"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from io import StringIO  # ✅ added (needed for Stooq CSV)

from stats_app.helpers.api_client import API_BASE_URL
from stats_app.helpers.ui_components import st_plot


# =========================================================
# AAPL STREAMLIT CLOUD FIX (HISTORY FALLBACK ONLY IF hist_df EMPTY)
# =========================================================

@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    """
    Fallback daily OHLCV from Stooq (no API key).
    Only used when Streamlit Cloud returns empty hist_df for AAPL.
    """
    sym = symbol.upper().strip()
    url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Normalize to expected columns
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need = ["High", "Low", "Close", "Volume"]
    if not set(need).issubset(df.columns):
        return pd.DataFrame()

    return df.dropna(subset=need).copy()


def _repair_hist_df_if_empty(symbol: str, hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal repair: only if hist_df is empty AND symbol is AAPL.
    """
    if hist_df is not None and not hist_df.empty:
        return hist_df

    if symbol.upper().strip() != "AAPL":
        return hist_df

    try:
        df = _fetch_stooq_daily("AAPL")
        if df is not None and not df.empty:
            return df
    except Exception:
        return hist_df

    return hist_df


# =========================================================
# 1) CORE DATA & GEX
# =========================================================

def _fetch_weekly_gex_table(symbol: str, expiry_date: str, spot: float) -> pd.DataFrame:
    r = requests.get(
        f"{API_BASE_URL}/weekly/gex",
        params={"symbol": symbol.upper(), "date": str(expiry_date), "spot": float(spot)},
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    df = pd.DataFrame(rows)
    for c in ["strike", "call_gex", "put_gex", "net_gex"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    return df


def _get_gamma_levels(gex_df: pd.DataFrame, spot: float) -> dict:
    d = gex_df.copy()
    d["abs_net"] = d["net_gex"].abs()
    d["dist"] = (d["strike"] - float(spot)).abs()
    d["pin_score"] = d["abs_net"] / (d["dist"] + 1.0)

    magnet_row = d.loc[d["pin_score"].idxmax()]
    put_wall_row = d.loc[d["put_gex"].idxmax()]
    call_wall_row = d.loc[d["call_gex"].idxmax()]

    return {
        "put_wall": float(put_wall_row["strike"]),
        "call_wall": float(call_wall_row["strike"]),
        "magnet": float(magnet_row["strike"]),
    }


def _nearest_strike(strikes: np.ndarray, target: float, side: str = "nearest") -> float:
    if strikes is None or len(strikes) == 0:
        return float(target)

    strikes = np.asarray(strikes, dtype=float)

    if side == "below":
        candidates = strikes[strikes <= target]
        return float(candidates.max()) if len(candidates) > 0 else float(strikes.min())

    if side == "above":
        candidates = strikes[strikes >= target]
        return float(candidates.min()) if len(candidates) > 0 else float(strikes.max())

    idx = int(np.argmin(np.abs(strikes - target)))
    return float(strikes[idx])


def _to_friday(date_like) -> str:
    """Return the Friday (YYYY-MM-DD) for the week of date_like (or same day if already Friday)."""
    d = pd.to_datetime(date_like).normalize()
    # weekday: Mon=0 ... Fri=4
    delta = (4 - d.weekday()) % 7
    return str((d + pd.Timedelta(days=int(delta))).date())


def _add_weeks_to_friday(weekly_expiry: str, weeks: int = 3) -> str:
    d = pd.to_datetime(weekly_expiry) + pd.Timedelta(days=int(weeks * 7))
    return _to_friday(d)


# =========================================================
# 2) LIGHT TECH (VWAP proxy + ADL)
# =========================================================

def _anchored_vwap_proxy(hist_df: pd.DataFrame) -> tuple[float, str]:
    """
    No Finnhub required.
    Anchored VWAP proxy blend (1d/3d) using daily closes.
    """
    recent = hist_df.tail(5)
    vwap_1 = float(recent.tail(1)["Close"].mean())
    vwap_3 = float(recent.tail(3)["Close"].mean())
    vwap = 0.6 * vwap_1 + 0.4 * vwap_3
    return vwap, "Anchored VWAP proxy blend (1d/3d)"


def _adl_delta(hist_df: pd.DataFrame) -> float:
    recent = hist_df.tail(120).copy()
    hl = (recent["High"] - recent["Low"]).replace(0, np.nan)
    clv = ((recent["Close"] - recent["Low"]) - (recent["High"] - recent["Close"])) / hl
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    adl = (clv * recent["Volume"]).cumsum()
    return float(adl.diff().tail(10).mean())


# =========================================================
# 3) STRATEGY BUILDER (Weekly Condor + 3W Hedges)
# =========================================================

def _build_book(strikes: np.ndarray, levels: dict, wing_width: int, hedge_width: int) -> dict:
    put_wall = float(levels["put_wall"])
    call_wall = float(levels["call_wall"])
    magnet = float(levels["magnet"])

    box_low = min(put_wall, call_wall)
    box_high = max(put_wall, call_wall)

    # Weekly Iron Condor (income engine)
    condor = {
        "sell_put": _nearest_strike(strikes, box_low, "below"),
        "buy_put": _nearest_strike(strikes, box_low - wing_width, "below"),
        "sell_call": _nearest_strike(strikes, box_high, "above"),
        "buy_call": _nearest_strike(strikes, box_high + wing_width, "above"),
    }

    # 3W Call Debit (upside hedge)
    call_hedge = {
        "buy_call": _nearest_strike(strikes, magnet, "nearest"),
        "sell_call": _nearest_strike(strikes, magnet + hedge_width, "above"),
    }

    # 3W Put Debit (downside hedge)
    put_hedge = {
        "buy_put": _nearest_strike(strikes, box_low, "below"),
        "sell_put": _nearest_strike(strikes, box_low - hedge_width, "below"),
    }

    return {
        "box_low": box_low,
        "box_high": box_high,
        "condor": condor,
        "call_hedge": call_hedge,
        "put_hedge": put_hedge,
    }


# =========================================================
# 4) TAB RENDERER
# =========================================================

def render_tab_friday_predictor(symbol: str, expiry_date: str, hist_df: pd.DataFrame, spot: float):
    st.subheader(f"🧠 Professional Risk Book & Playbook: {symbol}")

    # -------- Sidebar controls (keep your style) --------
    with st.sidebar:
        st.header("🎯 Broker Sync")
        vwap_pivot = st.number_input(
            "Broker VWAP Pivot (manual)",
            value=float(spot) if spot else 0.0,
            step=0.01,
            help="Optional: paste your broker VWAP pivot here. Used for strength/weakness timing gates.",
        )
        st.divider()
        st.header("⚙️ Book Construction")
        wing_width = st.slider("Weekly Condor Wing Width", 5, 25, 10)
        hedge_width = st.slider("3W Hedge Spread Width", 5, 40, 20)

    # ✅ ONLY FIX: if AAPL hist_df is empty on Streamlit Cloud, repair it here
    hist_df = _repair_hist_df_if_empty(symbol, hist_df)

    if hist_df is None or hist_df.empty:
        st.warning("Please fetch market data.")
        return

    need_cols = {"High", "Low", "Close", "Volume"}
    if not need_cols.issubset(hist_df.columns):
        st.error(f"Missing columns in hist_df: {sorted(list(need_cols - set(hist_df.columns)))}")
        return

    # -------- Strength/Weakness gate (simple + consistent) --------
    pivot = float(vwap_pivot) if vwap_pivot and vwap_pivot > 0 else float(spot)
    strength_num = pivot * 1.002  # +0.2%
    weakness_num = pivot          # pivot
    is_strong = float(spot) >= strength_num
    is_weak = float(spot) < weakness_num

    # -------- GEX table + gamma levels --------
    try:
        gex_df = _fetch_weekly_gex_table(symbol, expiry_date, spot)
        levels = _get_gamma_levels(gex_df, spot)
        strikes = gex_df["strike"].values.astype(float)
    except Exception as e:
        st.error(f"GEX Error: {e}")
        return

    # -------- Build 3-engine book --------
    book = _build_book(strikes, levels, wing_width=wing_width, hedge_width=hedge_width)
    box_low = book["box_low"]
    box_high = book["box_high"]
    condor = book["condor"]
    call_hedge = book["call_hedge"]
    put_hedge = book["put_hedge"]

    # -------- Expiries --------
    weekly_expiry = str(expiry_date)
    hedge_expiry = _add_weeks_to_friday(weekly_expiry, weeks=3)

    # -------- VWAP proxy + ADL (no Finnhub/AlphaV) --------
    vwap_model, vwap_src = _anchored_vwap_proxy(hist_df)
    adl_d = _adl_delta(hist_df)
    money_flow = "🟢 Accumulation" if adl_d > 0 else "🔴 Distribution"

    # -------- BEFORE GRAPH: show everything clearly --------
    st.markdown("## ✅ This Week’s Plan (All Instructions BEFORE Charts)")

    st.markdown(f"""
### 1) Market Context
- **Spot:** {spot:.2f}
- **Broker Pivot (manual):** {pivot:.2f}
- **Strength gate:** spot ≥ **{strength_num:.2f}**
- **Weakness gate:** spot < **{weakness_num:.2f}**
- **VWAP proxy:** {vwap_model:.2f}  _(src: {vwap_src})_
- **Money flow:** {money_flow}

### 2) Gamma Structure
- **Put Wall:** {levels["put_wall"]:.0f}
- **Call Wall:** {levels["call_wall"]:.0f}
- **Magnet:** {levels["magnet"]:.0f}
- **Box:** {box_low:.0f} → {box_high:.0f}
""")

    # -------- Timing + staging (AAPL style but dynamic) --------
    status_txt = "✅ STRENGTH DETECTED" if is_strong else ("❌ WEAKNESS DETECTED" if is_weak else "⚪ NEUTRAL / MIXED")
    st.markdown(f"### 3) Timing + Staging (Mon→Thu) — Status: **{status_txt}**")

    timing_html = f"""
    <div style="background:#0e1117;padding:16px;border-radius:10px;border:1px solid #30363d;">
      <div style="color:#ff7b72;font-weight:700;">
        Rule: Never sell put-side risk into weakness (&lt; {weakness_num:.2f}). Prefer selling put-side only after strength (&gt; {strength_num:.2f}).
      </div>

      <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap;">
        <div style="flex:1;min-width:260px;background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d;">
          <div style="color:#d29922;font-weight:700;">🥇 MONDAY (Discovery)</div>
          <ul style="color:#c9d1d9;margin:8px 0 0 18px;">
            <li><b>Do not</b> start with full condor.</li>
            <li>Open <b>3W hedges</b> small (call + put) if you want protection early.</li>
            <li>If you must start weekly: open <b>call-side only</b> first (less “catching knife” risk).</li>
          </ul>
        </div>

        <div style="flex:1;min-width:260px;background:{'#23863622' if is_strong else '#da363322'};padding:12px;border-radius:8px;border:1px solid {'#238636' if is_strong else '#da3633'};">
          <div style="color:{'#3fb950' if is_strong else '#f85149'};font-weight:700;">🥈 TUESDAY (Execution)</div>
          <ul style="color:#c9d1d9;margin:8px 0 0 18px;">
            <li>Status: <b>{'Strength' if is_strong else ('Weakness' if is_weak else 'Mixed')}</b></li>
            <li>If <b>Strength</b>: add weekly <b>put-side</b> (complete condor gradually).</li>
            <li>If <b>Weakness</b>: do <b>not</b> sell put-side; wait or move strikes lower.</li>
          </ul>
        </div>

        <div style="flex:1;min-width:260px;background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d;">
          <div style="color:#58a6ff;font-weight:700;">🥉 WEDNESDAY (Commit)</div>
          <ul style="color:#c9d1d9;margin:8px 0 0 18px;">
            <li>If price remains inside box: complete/hold the weekly condor.</li>
            <li>If price trends toward a wall: reduce the threatened side early.</li>
          </ul>
        </div>

        <div style="flex:1;min-width:260px;background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d;">
          <div style="color:#f85149;font-weight:700;">🏁 THURSDAY (Defense)</div>
          <ul style="color:#c9d1d9;margin:8px 0 0 18px;">
            <li>Take profits early; avoid holding weekly risk late.</li>
            <li>Trim or close if threatened; don’t “hope” into Friday.</li>
          </ul>
        </div>
      </div>

      <div style="margin-top:12px;background:#161b22;padding:12px;border-radius:8px;border:1px solid #30363d;">
        <div style="color:#f85149;font-weight:700;">🛑 Exit Rules (Mechanical)</div>
        <ul style="color:#c9d1d9;margin:8px 0 0 18px;">
          <li><b>Weekly condor:</b> take profit at <b>+70% to +80%</b> OR cut a side if it hits <b>2× credit</b>.</li>
          <li><b>3W hedges:</b> take profit at <b>+80% to +150%</b> when they pay.</li>
          <li><b>Safety:</b> avoid holding weekly risk past <b>Thursday afternoon</b>.</li>
        </ul>
      </div>
    </div>
    """
    st.components.v1.html(timing_html, height=420)

    st.markdown("## 🧱 Strategy Levels (Weekly + 3W Risk Mitigation)")
    df_levels = pd.DataFrame([
        {
            "Engine": "Weekly Condor (Put Credit Spread)",
            "Expiry": weekly_expiry,
            "Legs": f"Sell Put {condor['sell_put']:.0f} / Buy Put {condor['buy_put']:.0f}",
        },
        {
            "Engine": "Weekly Condor (Call Credit Spread)",
            "Expiry": weekly_expiry,
            "Legs": f"Sell Call {condor['sell_call']:.0f} / Buy Call {condor['buy_call']:.0f}",
        },
        {
            "Engine": "3W Upside Hedge (Call Debit Spread)",
            "Expiry": hedge_expiry,
            "Legs": f"Buy Call {call_hedge['buy_call']:.0f} / Sell Call {call_hedge['sell_call']:.0f}",
        },
        {
            "Engine": "3W Downside Hedge (Put Debit Spread)",
            "Expiry": hedge_expiry,
            "Legs": f"Buy Put {put_hedge['buy_put']:.0f} / Sell Put {put_hedge['sell_put']:.0f}",
        },
    ])
    st.table(df_levels)

    st.divider()

    st.write("### 🧲 Gamma Map + Weekly P&L Snapshot")

    col_map, col_pl = st.columns(2)

    with col_map:
        fig_map = go.Figure()

        fig_map.add_vrect(x0=box_low, x1=box_high, fillcolor="rgba(0,255,0,0.06)", line_width=0)

        fig_map.add_vline(x=levels["put_wall"], line_dash="dash", line_color="red", annotation_text="Put Wall")
        fig_map.add_vline(x=levels["call_wall"], line_dash="dash", line_color="green", annotation_text="Call Wall")
        fig_map.add_vline(x=levels["magnet"], line_color="gold", line_width=4, annotation_text="Magnet")

        leg_lines = [
            ("W Sell Put", condor["sell_put"]),
            ("W Buy Put", condor["buy_put"]),
            ("W Sell Call", condor["sell_call"]),
            ("W Buy Call", condor["buy_call"]),
            ("3W Call Buy", call_hedge["buy_call"]),
            ("3W Call Sell", call_hedge["sell_call"]),
            ("3W Put Buy", put_hedge["buy_put"]),
            ("3W Put Sell", put_hedge["sell_put"]),
        ]
        for name, x in leg_lines:
            fig_map.add_vline(x=float(x), line_dash="dot", opacity=0.35, annotation_text=name)

        fig_map.add_trace(
            go.Scatter(
                x=[spot],
                y=[0],
                mode="markers+text",
                text=[f"Spot ${spot:.2f}"],
                textposition="top center",
                marker=dict(size=12, color="cyan"),
            )
        )

        fig_map.update_layout(
            template="plotly_dark",
            height=360,
            yaxis_visible=False,
            title="Gamma Map + Strategy Legs",
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st_plot(fig_map)

    with col_pl:
        x_range = np.linspace(spot * 0.85, spot * 1.15, 220)
        y_pl = np.zeros_like(x_range)

        y_pl += np.where(x_range < condor["sell_put"], x_range - condor["sell_put"], 0)
        y_pl -= np.where(x_range < condor["buy_put"], x_range - condor["buy_put"], 0)
        y_pl += np.where(x_range > condor["sell_call"], condor["sell_call"] - x_range, 0)
        y_pl -= np.where(x_range > condor["buy_call"], condor["buy_call"] - x_range, 0)

        fig_pl = go.Figure()
        fig_pl.add_trace(go.Scatter(x=x_range, y=y_pl, fill="tozeroy"))
        fig_pl.add_vline(x=spot, line_dash="dash", annotation_text="Spot")
        fig_pl.add_vline(x=box_low, line_dash="dot", annotation_text="Box Low")
        fig_pl.add_vline(x=box_high, line_dash="dot", annotation_text="Box High")
        fig_pl.update_layout(template="plotly_dark", height=360, title="Weekly Income Shape (Condor)")
        st_plot(fig_pl)
