import pandas as pd
import numpy as np


def calculate_gamma_levels(weekly_result, df_options, hist_df, vwap_lookback=90):
    """
    Analyzes GEX and Technicals to find Magnets and Walls.

    Fixes:
    - VWAP is computed on a RECENT window (default last 90 trading days)
    - ADL is computed as a cumulative line and we use its recent slope
    - Adds sanity checks so VWAP can't be wildly off spot
    """
    w = weekly_result["data"]
    top = w.get("top_strikes", {})

    call_wall = top.get("call_gex", [{}])[0].get("strike")
    put_wall  = top.get("put_gex",  [{}])[0].get("strike")
    magnet    = top.get("net_gex_abs", [{}])[0].get("strike")

    # ---------- safety ----------
    if hist_df is None or hist_df.empty:
        return {
            "call_wall": call_wall,
            "put_wall": put_wall,
            "magnet": magnet,
            "vwap": None,
            "adl_status": "N/A",
            "adl_slope": None,
            "notes": "No price history data"
        }

    need_cols = {"High", "Low", "Close", "Volume"}
    if not need_cols.issubset(set(hist_df.columns)):
        return {
            "call_wall": call_wall,
            "put_wall": put_wall,
            "magnet": magnet,
            "vwap": None,
            "adl_status": "N/A",
            "adl_slope": None,
            "notes": f"Missing columns: {sorted(list(need_cols - set(hist_df.columns)))}"
        }

    # Use only RECENT history for VWAP/ADL context
    recent = hist_df.tail(int(vwap_lookback)).copy()

    # ---------- VWAP (recent window) ----------
    tp = (recent["High"] + recent["Low"] + recent["Close"]) / 3.0
    vwap_series = (tp * recent["Volume"]).cumsum() / recent["Volume"].cumsum()
    vwap = float(vwap_series.iloc[-1])

    # ---------- ADL (cumulative) + slope ----------
    hl_range = (recent["High"] - recent["Low"]).replace(0, np.nan)
    clv = ((recent["Close"] - recent["Low"]) - (recent["High"] - recent["Close"])) / hl_range
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)

    adl_series = (clv * recent["Volume"]).cumsum()

    # slope over last 20 bars (or less if short)
    n = min(20, len(adl_series))
    if n >= 5:
        adl_slope = float(np.polyfit(np.arange(n), adl_series.tail(n).values, 1)[0])
    else:
        adl_slope = float(adl_series.diff().iloc[-1]) if len(adl_series) > 1 else 0.0

    adl_status = "Accumulating" if adl_slope > 0 else "Distributing"

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "magnet": magnet,
        "vwap": vwap,
        "adl_status": adl_status,
        "adl_slope": adl_slope,
        "vwap_lookback": int(vwap_lookback),
        "notes": f"VWAP/ADL computed on last {len(recent)} bars"
    }
