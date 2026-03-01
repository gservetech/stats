# stats_app/tabs/tab_microstructure_engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================
# CSS (casino/dashboard look)
# =============================

def _apply_microstructure_css():
    st.markdown(
        """
<style>
/* Background */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(1100px 700px at 50% 0%, #2b3042 0%, #11162a 55%, #0b1020 100%);
}
[data-testid="stHeader"]{ background: rgba(0,0,0,0); }

/* Card (tile) */
.ms-card{
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.35);
  min-height: 140px;
}
.ms-title{
  font-size: 14px;
  font-weight: 800;
  letter-spacing: .35px;
  color: rgba(255,255,255,0.92);
  margin-bottom: 8px;
}
.ms-row{
  display:flex;
  justify-content: space-between;
  align-items: baseline;
  margin: 6px 0;
  color: rgba(255,255,255,0.86);
}
.ms-k{ font-size: 12px; opacity: 0.85; }
.ms-v{ font-size: 22px; font-weight: 900; }

.ms-badge{
  display:inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  border: 1px solid rgba(255,255,255,0.16);
}
.badge-green{ background: rgba(35, 168, 92, 0.16); color: #B7F7CF; }
.badge-yellow{ background: rgba(255, 193, 7, 0.14); color: #FFE7A3; }
.badge-red{ background: rgba(220, 53, 69, 0.18); color: #FFB7C0; }
.badge-blue{ background: rgba(13, 110, 253, 0.14); color: #BBD5FF; }

.ms-hr{ height:1px; background: rgba(255,255,255,0.10); margin: 10px 0; }

/* Plotly on dark */
.js-plotly-plot .plotly .main-svg{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}

/* --- Top Strikes (Gamma Walls / Magnets) --- */
.ts-wrap{ padding: 10px 8px 2px 8px; }
.ts-title{
  font-size: 28px;
  font-weight: 900;
  color: rgba(255,255,255,0.92);
  margin: 6px 0 14px 0;
}
.ts-chip{
  display:inline-block;
  background: rgba(0,100,255,0.80);
  color: white;
  font-weight: 900;
  padding: 6px 10px;
  border-radius: 4px;
  margin: 0 0 10px 0;
}
.ts-chip.gray{
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================
# Formatting helpers
# =============================

def _fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:,.{nd}f}"


def _fmt_money(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.0f}M"
    if ax >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.2f}"


def _card_html(title: str, rows: list[tuple[str, str]], badge_text: Optional[str] = None, badge_class: str = "badge-blue"):
    rows_html = "\n".join(
        [f'<div class="ms-row"><div class="ms-k">{k}</div><div class="ms-v">{v}</div></div>' for k, v in rows]
    )
    badge_html = f'<div class="ms-hr"></div><span class="ms-badge {badge_class}">{badge_text}</span>' if badge_text else ""
    return f"""
<div class="ms-card">
  <div class="ms-title">{title}</div>
  {rows_html}
  {badge_html}
</div>
"""


# =============================
# Data utilities
# =============================

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Finds a column by:
      1) exact match against candidate strings
      2) contains match against candidate strings
    """
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        hit = next((x for x in cols if c in x), None)
        if hit:
            return hit
    return None


def _find_col_contains_all(df: pd.DataFrame, must_contain: List[str]) -> Optional[str]:
    """
    Finds a column where ALL tokens are contained in the column name.
    Example: must_contain=["call","gex"] => "call_gex"
    """
    cols = list(df.columns)
    for c in cols:
        cl = c.lower()
        if all(t.lower() in cl for t in must_contain):
            return c
    return None


@dataclass
class GexCols:
    strike: str
    net_gex: str
    call_gex: Optional[str] = None
    put_gex: Optional[str] = None


def _resolve_gex_cols(gex_df: pd.DataFrame) -> GexCols:
    df = _norm_cols(gex_df)
    strike_col = _find_col(df, ["strike", "strk"])
    net_gex_col = _find_col_contains_all(df, ["net", "gex"]) or _find_col(df, ["net_gex", "netgamma", "net_gamma"])
    call_gex_col = _find_col_contains_all(df, ["call", "gex"])
    put_gex_col = _find_col_contains_all(df, ["put", "gex"])

    if not strike_col:
        raise ValueError("gex_df missing strike column (expected 'strike').")
    if not net_gex_col:
        raise ValueError("gex_df missing net_gex column (expected 'net_gex').")

    return GexCols(strike=strike_col, net_gex=net_gex_col, call_gex=call_gex_col, put_gex=put_gex_col)


# =============================
# Microstructure computations
# =============================

def _estimate_gamma_flip(df: pd.DataFrame, strike_col: str, net_gex_col: str) -> Optional[float]:
    d = df[[strike_col, net_gex_col]].copy()
    d[strike_col] = pd.to_numeric(d[strike_col], errors="coerce")
    d[net_gex_col] = pd.to_numeric(d[net_gex_col], errors="coerce")
    d = d.dropna().sort_values(strike_col)
    if d.empty:
        return None

    x = d[strike_col].to_numpy()
    y = d[net_gex_col].to_numpy()
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None

    flips = []
    for i in idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if (y1 - y0) == 0:
            flips.append(float(x0))
        else:
            flips.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))

    peak = float(x[int(np.argmax(np.abs(y)))])
    flips.sort(key=lambda f: abs(f - peak))
    return flips[0] if flips else None


def _gamma_concentration(df: pd.DataFrame, net_gex_col: str, top_n: int = 3) -> float:
    y = pd.to_numeric(df[net_gex_col], errors="coerce").fillna(0.0).to_numpy()
    ay = np.abs(y)
    total = float(np.sum(ay))
    if total <= 0:
        return 0.0
    top = float(np.sum(np.sort(ay)[-top_n:]))
    return float(top / total)


def _expected_move_from_iv(spot: float, iv_annual: Optional[float], days: int = 5) -> Optional[float]:
    if spot is None or not np.isfinite(spot) or spot <= 0:
        return None
    if iv_annual is None or not np.isfinite(iv_annual) or iv_annual <= 0:
        return None
    t = max(days, 1) / 365.0
    return float(spot * iv_annual * math.sqrt(t))


def _flip_risk(spot: float, flip: Optional[float], expected_move: Optional[float]) -> Optional[float]:
    if flip is None or expected_move is None or expected_move <= 0:
        return None
    return float(abs(spot - flip) / expected_move)


def _risk_badge(value: Optional[float], low: float = 0.6, high: float = 1.0) -> Tuple[str, str]:
    if value is None or not np.isfinite(value):
        return "UNKNOWN", "badge-blue"
    if value <= low:
        return "LOW", "badge-green"
    if value <= high:
        return "MODERATE", "badge-yellow"
    return "HIGH", "badge-red"


def _hedging_pressure_proxy(df: pd.DataFrame, strike_col: str, net_gex_col: str, spot: float, n: int = 7):
    """
    Proxy: mean(|net_gex|) among N strikes closest to spot.
    Regime proxy: stabilizing if mean(net_gex near spot) >= 0, else destabilizing.
    """
    d = df[[strike_col, net_gex_col]].copy()
    d[strike_col] = pd.to_numeric(d[strike_col], errors="coerce")
    d[net_gex_col] = pd.to_numeric(d[net_gex_col], errors="coerce")
    d = d.dropna().sort_values(strike_col)
    if d.empty:
        return None

    d["dist"] = (d[strike_col] - spot).abs()
    near = d.sort_values("dist").head(max(3, n))

    mag = float(np.mean(np.abs(near[net_gex_col].to_numpy()))) if not near.empty else 0.0
    net_near = float(np.mean(near[net_gex_col].to_numpy())) if not near.empty else 0.0
    stabilizing = net_near >= 0

    return {"sorted": d, "near": near, "mag": mag, "net_near": net_near, "stabilizing": stabilizing}


def _regime_score(conc: float, flip_risk: Optional[float], stabilizing: bool) -> int:
    """
    0-100 score:
      lower -> pin/control
      higher -> expansion risk
    """
    score = 50.0
    score -= (conc * 35.0)
    if flip_risk is not None and np.isfinite(flip_risk):
        score += (max(0.0, 1.2 - flip_risk) * 25.0)
    score += (20.0 if not stabilizing else -10.0)
    return int(max(0, min(100, round(score))))


def _score_label(score: int) -> Tuple[str, str]:
    if score >= 70:
        return "EXPANSION", "badge-red"
    if score >= 40:
        return "MIXED", "badge-yellow"
    return "PIN / CONTROL", "badge-green"


# =============================
# Plot
# =============================

def _plot_pressure(df: pd.DataFrame, strike_col: str, net_gex_col: str, spot: float, flip: Optional[float]) -> go.Figure:
    d = df[[strike_col, net_gex_col]].copy()
    d[strike_col] = pd.to_numeric(d[strike_col], errors="coerce")
    d[net_gex_col] = pd.to_numeric(d[net_gex_col], errors="coerce")
    d = d.dropna().sort_values(strike_col)

    x = d[strike_col].to_numpy()
    y = np.abs(d[net_gex_col].to_numpy())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Hedge Pressure (|net_gex|)"))

    fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot", annotation_position="top left")
    if flip is not None and np.isfinite(flip):
        fig.add_vline(x=flip, line_dash="dot", annotation_text="Gamma Flip", annotation_position="top right")

    fig.update_layout(
        title="Hedging Pressure by Strike (proxy)",
        template="plotly_dark",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Strike",
        yaxis_title="Pressure (proxy)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =============================
# Strategy helpers (2.5 / 5 increments)
# =============================

def _round_to_increment(x: float, inc: float) -> float:
    if inc <= 0:
        return float(x)
    return round(x / inc) * inc


def _strike_ladder(spot: float, inc: float, steps_each_side: int = 6) -> list[float]:
    atm = _round_to_increment(spot, inc)
    return [atm + i * inc for i in range(-steps_each_side, steps_each_side + 1)]


def _pick_spread_strikes(spot: float, inc: float, direction: str, width_steps: int = 2, itm_bias: int = 0):
    """
    Debit spreads:
      bull: buy call at/near ATM, sell higher call
      bear: buy put at/near ATM, sell lower put
    """
    atm = _round_to_increment(spot, inc)

    if direction == "bull":
        long_strike = atm - itm_bias * inc
        short_strike = long_strike + width_steps * inc
        return float(long_strike), float(short_strike)

    # bear
    long_strike = atm + itm_bias * inc
    short_strike = long_strike - width_steps * inc
    return float(long_strike), float(short_strike)


def _pick_credit_spread_strikes(spot: float, inc: float, side: str, width_steps: int = 2, distance_steps: int = 2):
    """
    Credit spreads:
      put: bull put credit (sell put OTM, buy further OTM)
      call: bear call credit (sell call OTM, buy further OTM)
    """
    atm = _round_to_increment(spot, inc)

    if side == "put":
        short_put = atm - distance_steps * inc
        long_put = short_put - width_steps * inc
        return float(short_put), float(long_put)

    # call
    short_call = atm + distance_steps * inc
    long_call = short_call + width_steps * inc
    return float(short_call), float(long_call)


def _pick_iron_condor_strikes(spot: float, inc: float, width_steps: int = 2, distance_steps: int = 2):
    sp, lp = _pick_credit_spread_strikes(spot, inc, side="put", width_steps=width_steps, distance_steps=distance_steps)
    sc, lc = _pick_credit_spread_strikes(spot, inc, side="call", width_steps=width_steps, distance_steps=distance_steps)
    return (sp, lp, sc, lc)


# =============================
# Top Strikes (like your screenshot)
# =============================

def render_top_strikes_section(gex_df: pd.DataFrame, top_n: int = 10):
    st.markdown('<div class="ts-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="ts-title">Top Strikes (Gamma Walls / Magnets)</div>', unsafe_allow_html=True)

    if gex_df is None or gex_df.empty:
        st.info("Weekly GEX not loaded. Click Fetch Data.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    d = _norm_cols(gex_df)
    cols = _resolve_gex_cols(d)

    # Ensure numeric
    d[cols.strike] = pd.to_numeric(d[cols.strike], errors="coerce")
    d[cols.net_gex] = pd.to_numeric(d[cols.net_gex], errors="coerce")
    if cols.call_gex:
        d[cols.call_gex] = pd.to_numeric(d[cols.call_gex], errors="coerce")
    if cols.put_gex:
        d[cols.put_gex] = pd.to_numeric(d[cols.put_gex], errors="coerce")

    d = d.dropna(subset=[cols.strike, cols.net_gex])

    # Tables
    top_call = None
    if cols.call_gex:
        top_call = (
            d.dropna(subset=[cols.call_gex])
            .sort_values(cols.call_gex, ascending=False)
            .loc[:, [cols.strike, cols.call_gex]]
            .head(top_n)
            .rename(columns={cols.strike: "strike", cols.call_gex: "call_gex"})
        )

    top_put = None
    if cols.put_gex:
        top_put = (
            d.dropna(subset=[cols.put_gex])
            .sort_values(cols.put_gex, ascending=False)
            .loc[:, [cols.strike, cols.put_gex]]
            .head(top_n)
            .rename(columns={cols.strike: "strike", cols.put_gex: "put_gex"})
        )

    top_net = (
        d.assign(net_gex_abs=lambda x: x[cols.net_gex].abs())
        .sort_values("net_gex_abs", ascending=False)
        .loc[:, [cols.strike, cols.net_gex, "net_gex_abs"]]
        .head(top_n)
        .rename(columns={cols.strike: "strike", cols.net_gex: "net_gex"})
    )

    c1, c2, c3 = st.columns([1.15, 1.15, 1.55])

    with c1:
        st.markdown('<div class="ts-chip">Top Call GEX</div>', unsafe_allow_html=True)
        if top_call is None or top_call.empty:
            st.caption("call_gex not available in this dataset.")
        else:
            st.dataframe(top_call, use_container_width=True, height=360)

    with c2:
        st.markdown('<div class="ts-chip">Top Put GEX</div>', unsafe_allow_html=True)
        if top_put is None or top_put.empty:
            st.caption("put_gex not available in this dataset.")
        else:
            st.dataframe(top_put, use_container_width=True, height=360)

    with c3:
        st.markdown('<div class="ts-chip gray">Top Net GEX (abs)</div>', unsafe_allow_html=True)
        st.dataframe(top_net, use_container_width=True, height=360)

    st.markdown("</div>", unsafe_allow_html=True)


# =============================
# Public render function (called from app.py)
# =============================

def render_tab_microstructure_engine(
    symbol: str,
    spot: float,
    gex_df: pd.DataFrame,
    expected_move: Optional[float] = None,
    gamma_flip_strike: Optional[float] = None,
    iv_annual: Optional[float] = None,
):
    """
    Minimal requirements:
      - symbol, spot, gex_df (weekly gex table)

    Optional:
      - expected_move
      - gamma_flip_strike
      - iv_annual (auto expected move)
    """
    _apply_microstructure_css()

    st.subheader("🎰 Microstructure Engine (Tiles + 1 Chart)")

    if gex_df is None or gex_df.empty:
        st.info("Run Fetch Data (needs Weekly GEX table).")
        return

    df = _norm_cols(gex_df)
    cols = _resolve_gex_cols(df)
    strike_col, net_gex_col = cols.strike, cols.net_gex

    # Controls row
    c1, c2, c3 = st.columns([1.4, 1.2, 1.4])
    with c1:
        st.metric("Symbol", symbol)
    with c2:
        st.metric("Spot", _fmt_num(float(spot), 2))
    with c3:
        view = st.radio("View", ["Minimal", "Quant"], horizontal=True, index=0)

    # Expected move
    if expected_move is None:
        expected_move = _expected_move_from_iv(spot=spot, iv_annual=iv_annual, days=5)

    # Flip
    if gamma_flip_strike is None:
        gamma_flip_strike = _estimate_gamma_flip(df, strike_col, net_gex_col)

    # Pressure proxy
    hp = _hedging_pressure_proxy(df, strike_col, net_gex_col, spot=spot, n=7)
    if not hp:
        st.warning("Could not compute hedging pressure proxy.")
        return

    mag = float(hp["mag"])
    stabilizing = bool(hp["stabilizing"])

    # heuristic scale so tiles look like "$ flows"
    scale = 1.0
    if mag != 0:
        lg = math.log10(abs(mag) + 1e-9)
        if lg < 3:
            scale = 1e6
        elif lg < 6:
            scale = 1e3
        elif lg > 10:
            scale = 1e-3

    flow_per_1 = mag * scale
    flow_per_em = flow_per_1 * (expected_move if expected_move else 0.0)

    conc = _gamma_concentration(df, net_gex_col, top_n=3)
    fr = _flip_risk(spot=spot, flip=gamma_flip_strike, expected_move=expected_move)
    fr_lbl, fr_badge = _risk_badge(fr, low=0.6, high=1.0)

    score = _regime_score(conc, fr, stabilizing)
    score_lbl, score_badge = _score_label(score)

    # ---- Tiles
    t1, t2, t3, t4 = st.columns(4)

    with t1:
        badge = "STABILIZING" if stabilizing else "DESTABILIZING"
        badge_class = "badge-green" if stabilizing else "badge-red"
        st.markdown(
            _card_html(
                "Dealer Hedging Pressure",
                [("Flow per $1", _fmt_money(flow_per_1)), ("Flow per EM", _fmt_money(flow_per_em))],
                badge_text=badge,
                badge_class=badge_class,
            ),
            unsafe_allow_html=True,
        )

    with t2:
        pin = "STRONG PIN" if conc >= 0.60 else ("MIXED" if conc >= 0.35 else "FRAGILE")
        pin_badge = "badge-green" if conc >= 0.60 else ("badge-yellow" if conc >= 0.35 else "badge-red")
        st.markdown(
            _card_html(
                "Gamma Concentration",
                [("Top-3/Total", f"{conc*100:,.0f}%"), ("Pin quality", pin)],
                badge_text=pin,
                badge_class=pin_badge,
            ),
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown(
            _card_html(
                "Flip Proximity Risk",
                [("FlipRisk", _fmt_num(fr, 2) if fr is not None else "N/A"), ("Level", fr_lbl)],
                badge_text=f"RISK: {fr_lbl}",
                badge_class=fr_badge,
            ),
            unsafe_allow_html=True,
        )

    with t4:
        st.markdown(
            _card_html(
                "Regime Score",
                [("Score", f"{score:d}"), ("Mode", score_lbl)],
                badge_text=score_lbl,
                badge_class=score_badge,
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    # ---- One chart (always)
    fig = _plot_pressure(df, strike_col, net_gex_col, spot=spot, flip=gamma_flip_strike)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Action Card (always)
    st.markdown("### ✅ Action Card")
    if score_lbl == "PIN / CONTROL":
        st.success("Pin/control likely → **Iron Condor / Credit spreads** preferred.")
    elif score_lbl == "EXPANSION":
        st.error("Expansion risk elevated → **Debit spreads** preferred; avoid tight short-gamma.")
    else:
        st.warning("Mixed regime → reduce size; prefer defined risk.")

    # ============================================================
    # ✅ ALWAYS SHOW STRATEGIES (Minimal + Quant)
    # ============================================================
    st.markdown("## 🧭 Strategy Ideas (Bull / Bear / Neutral)")

    cA, cB, cC = st.columns([1.3, 1.2, 1.5])
    with cA:
        strike_inc = st.selectbox("Strike increment", [2.5, 5.0], index=1)
    with cB:
        width_steps = st.slider("Spread width (steps)", 1, 4, 2)
    with cC:
        dist_steps = st.slider("Distance from ATM for credit spreads (steps)", 1, 6, 2)

    atm = _round_to_increment(spot, strike_inc)

    preferred = "MIXED (Selective)"
    if score_lbl == "PIN / CONTROL":
        preferred = "NEUTRAL (Premium Selling)"
    elif score_lbl == "EXPANSION":
        preferred = "DIRECTIONAL (Momentum)"

    st.caption(f"Preferred mode: **{preferred}** | ATM (rounded): **{atm:.2f}**")

    rows = []

    # Bull
    if score_lbl == "EXPANSION":
        lc, sc = _pick_spread_strikes(spot, strike_inc, direction="bull", width_steps=width_steps, itm_bias=0)
        rows.append(("BULL", "Debit Call Spread", f"Buy {lc:.2f} Call / Sell {sc:.2f} Call", "Defined risk; best in expansion"))
    else:
        sp, lp = _pick_credit_spread_strikes(spot, strike_inc, side="put", width_steps=width_steps, distance_steps=dist_steps)
        rows.append(("BULL", "Bull Put Credit Spread", f"Sell {sp:.2f} Put / Buy {lp:.2f} Put", "Premium selling; better in pin/control"))

    # Bear
    if score_lbl == "EXPANSION":
        lp_, sp_ = _pick_spread_strikes(spot, strike_inc, direction="bear", width_steps=width_steps, itm_bias=0)
        rows.append(("BEAR", "Debit Put Spread", f"Buy {lp_:.2f} Put / Sell {sp_:.2f} Put", "Defined risk; best in expansion"))
    else:
        sc2, lc2 = _pick_credit_spread_strikes(spot, strike_inc, side="call", width_steps=width_steps, distance_steps=dist_steps)
        rows.append(("BEAR", "Bear Call Credit Spread", f"Sell {sc2:.2f} Call / Buy {lc2:.2f} Call", "Premium selling; better in pin/control"))

    # Neutral
    if score_lbl == "PIN / CONTROL":
        sp2, lp2, sc3, lc3 = _pick_iron_condor_strikes(spot, strike_inc, width_steps=width_steps, distance_steps=dist_steps)
        rows.append(("NEUTRAL", "Iron Condor", f"Put: Sell {sp2:.2f}/Buy {lp2:.2f} | Call: Sell {sc3:.2f}/Buy {lc3:.2f}", "Range/pin structure"))
    else:
        rows.append(("NEUTRAL", "No Trade / Wait", "Wait for confirmation (flip + flow)", "Neutral weaker in expansion/mixed"))

    st.dataframe(pd.DataFrame(rows, columns=["Bias", "Strategy", "Strike Plan", "Why"]), use_container_width=True)

    with st.expander("Show strike ladder (rounded to increment)"):
        ladder = _strike_ladder(spot, strike_inc, steps_each_side=6)
        st.write(", ".join([f"{x:.2f}" for x in ladder]))

    # ============================================================
    # ✅ TOP STRIKES SECTION (like your screenshot)
    # ============================================================
    render_top_strikes_section(gex_df=gex_df, top_n=10)

    # ============================================================
    # Quant-only extras (optional)
    # ============================================================
    if view == "Quant":
        st.markdown("### 🧪 Stress Test (proxy)")
        q1, q2, q3 = st.columns([1.2, 1.2, 1.6])

        with q1:
            pmove = st.slider("Price move ($)", -10.0, 10.0, 3.0, 0.5)
        with q2:
            ivmove = st.slider("IV move (%)", -10, 10, 3, 1)
        with q3:
            moved_spot = spot + pmove
            moved_fr = _flip_risk(moved_spot, gamma_flip_strike, expected_move)
            moved_lbl, moved_badge = _risk_badge(moved_fr, low=0.6, high=1.0)

            proj_pressure = flow_per_1 * abs(pmove) * (1.0 + abs(ivmove) / 100.0)
            st.metric("Projected Pressure", _fmt_money(proj_pressure))
            st.markdown(f"**FlipRisk after move:** `{_fmt_num(moved_fr, 2) if moved_fr is not None else 'N/A'}`")
            st.markdown(f"<span class='ms-badge {moved_badge}'>FlipRisk: {moved_lbl}</span>", unsafe_allow_html=True)

        st.markdown("### 🧠 Convexity Alignment (your book)")
        a1, a2, a3, a4 = st.columns([1.1, 1.1, 1.1, 1.7])
        with a1:
            user_delta = st.number_input("Net Delta", value=0.0, step=100.0)
        with a2:
            user_gamma = st.number_input("Net Gamma", value=0.0, step=50.0)
        with a3:
            user_vega = st.number_input("Net Vega", value=0.0, step=100.0)

        aligned = (user_gamma <= 0) if stabilizing else (user_gamma >= 0)
        with a4:
            if aligned:
                st.success("✅ ALIGNED with regime")
            else:
                st.error("⚠️ MISALIGNED with regime")
            st.caption("Rule: Stabilizing (pin) → short gamma ok. Destabilizing (expansion) → prefer long gamma.")


# Optional alias if you want a shorter import name elsewhere
def render_tab_microstructure(symbol: str, spot: float, gex_df: pd.DataFrame):
    render_tab_microstructure_engine(symbol=symbol, spot=spot, gex_df=gex_df)