import datetime as dt

import pandas as pd
import streamlit as st

from stats_app.helpers.api_client import fetch_weekly_gex, fetch_weekly_summary
from stats_app.helpers.calculations import compute_gamma_map_artifacts
from stats_app.helpers.ui_components import st_btn, st_df


def _next_friday(d: dt.date) -> dt.date:
    days_ahead = (4 - d.weekday()) % 7
    return d + dt.timedelta(days=days_ahead)


def _to_num(val):
    try:
        return float(val)
    except Exception:
        return None


def _run_6_week_calc(symbol: str, spot: float, start_friday: dt.date) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []

    for i in range(6):
        expiry = start_friday + dt.timedelta(days=7 * i)
        expiry_str = expiry.isoformat()

        summary_res = fetch_weekly_summary(symbol, expiry_str, spot)
        gex_res = fetch_weekly_gex(symbol, expiry_str, spot)

        row = {
            "week": i + 1,
            "expiry": expiry_str,
            "spot_used": float(spot),
        }

        summary_ok = bool(summary_res and summary_res.get("success"))
        gex_ok = bool(gex_res and gex_res.get("success"))

        if summary_ok and gex_ok:
            summary_data = summary_res.get("data", {})
            totals = summary_data.get("totals", {}) if isinstance(summary_data, dict) else {}
            pcr = summary_data.get("pcr", {}) if isinstance(summary_data, dict) else {}

            gex_data = gex_res.get("data", {}) if isinstance(gex_res, dict) else {}
            gex_rows = gex_data.get("data", []) if isinstance(gex_data, dict) else []
            gex_df = pd.DataFrame(gex_rows)
            art = compute_gamma_map_artifacts(gex_df, spot=float(spot), top_n=10) if not gex_df.empty else {}

            row.update(
                {
                    "call_gex_total": _to_num(totals.get("call_gex")),
                    "put_gex_total": _to_num(totals.get("put_gex")),
                    "net_gex_total": _to_num(totals.get("net_gex")),
                    "pcr_oi": _to_num(pcr.get("oi")),
                    "pcr_volume": _to_num(pcr.get("volume")),
                    "call_wall": _to_num(art.get("call_wall")),
                    "put_wall": _to_num(art.get("put_wall")),
                    "magnet": _to_num(art.get("magnet")),
                    "zero_gamma": _to_num(art.get("zero_gamma")),
                    "status": "ok",
                }
            )
        else:
            summary_err = summary_res.get("error", "Unknown summary error") if isinstance(summary_res, dict) else "Summary request failed"
            gex_err = gex_res.get("error", "Unknown gex error") if isinstance(gex_res, dict) else "GEX request failed"
            row.update({"status": "error", "error": f"summary={summary_err}; gex={gex_err}"})
            errors.append({"expiry": expiry_str, "summary_error": summary_err, "gex_error": gex_err})

        rows.append(row)

    return pd.DataFrame(rows), errors


def render_tab_friday_calculation_6_weeks(symbol: str, spot: float):
    st.subheader("🗓️ Friday Calculation (6 Weeks)")

    if not symbol:
        st.warning("Symbol is required.")
        return

    spot_val = _to_num(spot)
    if spot_val is None or spot_val <= 0:
        st.warning("Valid spot price is required.")
        return

    start_friday = _next_friday(dt.date.today())
    end_friday = start_friday + dt.timedelta(days=35)
    st.caption(
        f"Nearest upcoming Friday: **{start_friday.isoformat()}**  |  "
        f"Range: **{start_friday.isoformat()} → {end_friday.isoformat()}**  |  "
        f"Spot used for all weeks: **{spot_val:,.2f}**"
    )

    cache_key = f"friday_calc_6w_{symbol}"
    refresh = st_btn("Run / Refresh 6-Week Calculation", key=f"friday_calc_6w_refresh_{symbol}")

    cached = st.session_state.get(cache_key)
    try:
        cached_spot = float((cached or {}).get("spot", 0.0))
    except Exception:
        cached_spot = 0.0
    needs_run = (
        refresh
        or not cached
        or cached.get("start_friday") != start_friday.isoformat()
        or abs(cached_spot - spot_val) > 1e-9
    )

    if needs_run:
        with st.spinner(f"Calculating 6 Friday snapshots for {symbol}..."):
            out_df, errors = _run_6_week_calc(symbol=symbol, spot=spot_val, start_friday=start_friday)
        st.session_state[cache_key] = {
            "start_friday": start_friday.isoformat(),
            "spot": float(spot_val),
            "rows": out_df,
            "errors": errors,
            "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    payload = st.session_state.get(cache_key, {})
    out_df = payload.get("rows", pd.DataFrame())
    errors = payload.get("errors", [])
    updated_at = payload.get("updated_at")

    if out_df.empty:
        st.info("No rows available yet. Click the run button.")
        return

    ok_rows = int((out_df["status"] == "ok").sum()) if "status" in out_df.columns else 0
    total_rows = int(len(out_df))
    c1, c2, c3 = st.columns(3)
    c1.metric("Weeks Loaded", f"{ok_rows}/{total_rows}")
    c2.metric("Symbol", symbol)
    c3.metric("Spot", f"{spot_val:,.2f}")

    show_cols = [
        "week",
        "expiry",
        "spot_used",
        "call_gex_total",
        "put_gex_total",
        "net_gex_total",
        "call_wall",
        "put_wall",
        "magnet",
        "zero_gamma",
        "pcr_oi",
        "pcr_volume",
        "status",
    ]
    present_cols = [c for c in show_cols if c in out_df.columns]
    st_df(out_df[present_cols], height=420)

    if updated_at:
        st.caption(f"Last updated: {updated_at}")

    if errors:
        with st.expander("Errors", expanded=False):
            st_df(pd.DataFrame(errors), height=200)
