import datetime as dt
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from stats_app.helpers.api_client import fetch_weekly_gex, fetch_weekly_summary
from stats_app.helpers.barchart_direct import BarchartDirectClient
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


def _to_date(date_str: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(date_str))
    except Exception:
        return None


def _week_expiry_candidates(target_expiry_str: str) -> list[str]:
    d = _to_date(target_expiry_str)
    if d is None:
        return [target_expiry_str]
    # Weekly expiries can shift earlier (e.g., Friday holiday -> Thursday).
    prev = d - dt.timedelta(days=1)
    return [d.isoformat(), prev.isoformat()]


def _is_retryable_error(status_code: int | None, err_text: str | None) -> bool:
    if status_code is not None:
        return int(status_code) in {408, 429, 500, 502, 503, 504}
    t = (err_text or "").lower()
    non_retryable_signals = (
        "no contracts found",
        "missing",
        "not found",
        "404",
    )
    if any(s in t for s in non_retryable_signals):
        return False
    return True


def _run_6_week_calc(
    symbol: str,
    spot: float,
    start_friday: dt.date,
    direct_auth: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []

    runtime_cookie = (direct_auth or {}).get("cookie_header")
    runtime_xsrf = (direct_auth or {}).get("xsrf_token")

    def _make_direct_client() -> BarchartDirectClient:
        if runtime_cookie:
            timeout_env = os.getenv("BARCHART_DIRECT_TIMEOUT_SECONDS", "30")
            try:
                timeout_seconds = max(5, int(timeout_env))
            except Exception:
                timeout_seconds = 30
            return BarchartDirectClient(
                cookie_input=str(runtime_cookie),
                xsrf_override=str(runtime_xsrf) if runtime_xsrf else None,
                timeout_seconds=timeout_seconds,
            )
        return BarchartDirectClient.from_env()

    direct_probe_client = _make_direct_client()
    direct_ready = direct_probe_client.ready
    direct_probe_client.close()

    def _fetch_week_direct(expiry_str: str) -> dict:
        if not direct_ready:
            return {
                "success": False,
                "error": "Direct cookie unavailable (no session-captured auth and no BARCHART_DIRECT_COOKIE).",
                "source": "direct_api",
                "retryable": False,
            }
        wk_client = _make_direct_client()
        try:
            res = wk_client.fetch_weekly_summary_and_gex(symbol=symbol, date=expiry_str, spot=float(spot))
        except Exception as exc:
            res = {"success": False, "error": str(exc)}
        finally:
            wk_client.close()
        if res.get("success"):
            res["source"] = "direct_api_session" if runtime_cookie else "direct_api"
            res["retryable"] = False
        else:
            res.setdefault("source", "direct_api")
            res["retryable"] = _is_retryable_error(None, res.get("error"))
        return res

    def _fetch_week_browser(expiry_str: str) -> dict:
        summary_res = fetch_weekly_summary(symbol=symbol, date=expiry_str, spot=float(spot))
        summary_ok = bool(summary_res.get("success"))
        summary_status = summary_res.get("status_code")
        if not summary_ok:
            summary_err = summary_res.get("error", "weekly summary failed")
            return {
                "success": False,
                "source": "browser_scrape",
                "summary_error": summary_err,
                "gex_error": "Skipped because summary failed.",
                "status_code": summary_status,
                "retryable": _is_retryable_error(summary_status, summary_err),
                "error": f"summary={summary_err}",
            }

        gex_res = fetch_weekly_gex(symbol=symbol, date=expiry_str, spot=float(spot))
        gex_ok = bool(gex_res.get("success"))
        gex_status = gex_res.get("status_code")
        if gex_ok:
            return {
                "success": True,
                "source": "browser_scrape",
                "expiration_type": "weekly",
                "summary": summary_res.get("data", {}),
                "gex": gex_res.get("data", {}),
                "retryable": False,
            }

        summary_err = summary_res.get("error", "weekly summary failed")
        gex_err = gex_res.get("error", "weekly gex failed")
        return {
            "success": False,
            "source": "browser_scrape",
            "summary_error": summary_err,
            "gex_error": gex_err,
            "status_code": gex_status,
            "retryable": _is_retryable_error(gex_status, gex_err),
            "error": f"summary={summary_err}; gex={gex_err}",
        }

    def _fetch_week_job(week_idx: int, target_expiry_str: str) -> tuple[int, str, dict]:
        candidates = _week_expiry_candidates(target_expiry_str)

        last_direct_res: dict | None = None
        for candidate in candidates:
            direct_res = _fetch_week_direct(candidate)
            last_direct_res = direct_res
            if direct_res.get("success"):
                direct_res["target_expiry"] = target_expiry_str
                direct_res["resolved_expiry"] = candidate
                return week_idx, candidate, direct_res

        last_browser_res: dict | None = None
        for candidate in candidates:
            browser_res = _fetch_week_browser(candidate)
            last_browser_res = browser_res
            if browser_res.get("success"):
                browser_res["direct_error"] = (last_direct_res or {}).get("error", "Direct request failed")
                browser_res["target_expiry"] = target_expiry_str
                browser_res["resolved_expiry"] = candidate
                return week_idx, candidate, browser_res

        direct_error = (last_direct_res or {}).get("error", "Direct request failed")
        browser_error = (last_browser_res or {}).get("error", "Browser fallback failed")
        combined_error = f"direct={direct_error} | browser={browser_error}"
        return week_idx, target_expiry_str, {
            "success": False,
            "source": "direct_api->browser_scrape",
            "error": combined_error,
            "direct_error": direct_error,
            "summary_error": (last_browser_res or {}).get("summary_error"),
            "gex_error": (last_browser_res or {}).get("gex_error"),
            "retryable": bool((last_direct_res or {}).get("retryable") or (last_browser_res or {}).get("retryable")),
            "target_expiry": target_expiry_str,
            "resolved_expiry": target_expiry_str,
        }

    work_items: list[tuple[int, str]] = []
    for i in range(6):
        expiry = start_friday + dt.timedelta(days=7 * i)
        work_items.append((i + 1, expiry.isoformat()))

    max_workers_env = os.getenv("FRIDAY_6W_MAX_WORKERS", "6")
    try:
        max_workers = max(1, min(len(work_items), int(max_workers_env)))
    except Exception:
        max_workers = min(len(work_items), 6)

    results_by_week: dict[int, tuple[str, dict]] = {}
    failed_weeks: list[tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_week_job, week_idx, expiry_str) for week_idx, expiry_str in work_items]
        for fut in as_completed(futures):
            try:
                week_idx, expiry_str, direct_res = fut.result()
            except Exception as exc:
                # Should be rare since _fetch_week_job already catches, but keep this as a safeguard.
                week_idx, expiry_str, direct_res = -1, "unknown", {"success": False, "error": str(exc), "retryable": True}

            if week_idx != -1:
                results_by_week[week_idx] = (expiry_str, direct_res)
                if not direct_res.get("success") and direct_res.get("retryable"):
                    failed_weeks.append((week_idx, expiry_str))

    # One retry pass for failures (sequential) to smooth transient API hiccups.
    for week_idx, expiry_str in failed_weeks:
        _, _, retry_res = _fetch_week_job(week_idx, expiry_str)
        if retry_res.get("success"):
            results_by_week[week_idx] = (expiry_str, retry_res)

    for week_idx, target_expiry_str in work_items:
        resolved_expiry, direct_res = (
            results_by_week.get(week_idx)
            or (target_expiry_str, {"success": False, "error": "Missing result", "target_expiry": target_expiry_str})
        )
        row = {
            "week": week_idx,
            "target_expiry": target_expiry_str,
            "expiry": resolved_expiry,
            "spot_used": float(spot),
        }

        if direct_res.get("success"):
            summary_data = direct_res.get("summary", {})
            totals = summary_data.get("totals", {}) if isinstance(summary_data, dict) else {}
            pcr = summary_data.get("pcr", {}) if isinstance(summary_data, dict) else {}

            gex_payload = direct_res.get("gex", {})
            gex_rows = gex_payload.get("data", []) if isinstance(gex_payload, dict) else []
            gex_df = pd.DataFrame(gex_rows)
            art = compute_gamma_map_artifacts(gex_df, spot=float(spot), top_n=10) if not gex_df.empty else {}

            row.update(
                {
                    "source": direct_res.get("source", "direct_api"),
                    "exp_type": direct_res.get("expiration_type", "weekly"),
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
            err = direct_res.get("error", "Direct weekly request failed")
            row.update({"source": direct_res.get("source", "direct_api"), "status": "error", "error": err})
            errors.append(
                {
                    "target_expiry": direct_res.get("target_expiry", target_expiry_str),
                    "expiry": direct_res.get("resolved_expiry", resolved_expiry),
                    "direct_error": direct_res.get("direct_error", err),
                    "summary_error": direct_res.get("summary_error", err),
                    "gex_error": direct_res.get("gex_error", err),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows), errors


def _render_6_week_calc_body(symbol: str, spot_val: float, start_friday: dt.date, direct_auth: dict | None = None):
    calculate_now = st.selectbox(
        "Calculate 6-week data now?",
        options=["No", "Yes"],
        index=0,
        key=f"friday_calc_6w_mode_{symbol}",
    )
    if calculate_now != "Yes":
        st.info("Set to 'Yes' to run the 6-week Friday calculation. Default is 'No' for performance.")
        return

    cache_key = f"friday_calc_6w_{symbol}"
    refresh = st_btn("Run / Refresh 6-Week Calculation", key=f"friday_calc_6w_refresh_{symbol}")

    cached = st.session_state.get(cache_key)
    try:
        cached_spot = float((cached or {}).get("spot", 0.0))
    except Exception:
        cached_spot = 0.0
    auth_stamp = (direct_auth or {}).get("captured_at") or ""
    needs_run = (
        refresh
        or not cached
        or cached.get("start_friday") != start_friday.isoformat()
        or abs(cached_spot - spot_val) > 1e-9
        or (cached.get("auth_stamp", "") != auth_stamp)
    )

    if needs_run:
        with st.spinner(f"Calculating 6 Friday snapshots for {symbol}..."):
            out_df, errors = _run_6_week_calc(
                symbol=symbol,
                spot=spot_val,
                start_friday=start_friday,
                direct_auth=direct_auth,
            )
        st.session_state[cache_key] = {
            "start_friday": start_friday.isoformat(),
            "spot": float(spot_val),
            "auth_stamp": auth_stamp,
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
        "target_expiry",
        "expiry",
        "source",
        "exp_type",
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


if hasattr(st, "fragment"):
    @st.fragment
    def _render_6_week_calc_fragment(symbol: str, spot_val: float, start_friday: dt.date, direct_auth: dict | None = None):
        _render_6_week_calc_body(symbol=symbol, spot_val=spot_val, start_friday=start_friday, direct_auth=direct_auth)
else:
    def _render_6_week_calc_fragment(symbol: str, spot_val: float, start_friday: dt.date, direct_auth: dict | None = None):
        _render_6_week_calc_body(symbol=symbol, spot_val=spot_val, start_friday=start_friday, direct_auth=direct_auth)


def render_tab_friday_calculation_6_weeks(symbol: str, spot: float, direct_auth: dict | None = None):
    st.subheader("🗓️ Friday Calculation (6 Weeks)")
    st.caption("Data source: direct Barchart API first (cookie-auth), with browser-scrape fallback per week.")

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

    if isinstance(direct_auth, dict) and direct_auth.get("captured_at"):
        st.caption(f"Session direct auth cached from Fetch Data at: {direct_auth.get('captured_at')}")

    _render_6_week_calc_fragment(
        symbol=symbol,
        spot_val=spot_val,
        start_friday=start_friday,
        direct_auth=direct_auth,
    )
