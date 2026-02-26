from __future__ import annotations

import os
import re
from io import StringIO
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

PRICE_PATTERN_SIZE_DEFAULT = 20
TOP_K_DEFAULT = 7
SIM_THRESHOLD_DEFAULT = 0.6


def _get_secret_or_env(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def _to_int(value: Any, fallback: int) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return fallback


def _build_yahoo_history_url(symbol: str) -> str:
    clean = (symbol or "AAPL").strip().upper()
    encoded = quote(clean, safe="^$.-")
    return f"https://finance.yahoo.com/quote/{encoded}/history/"


def _string_has_table_hint(text: str) -> bool:
    low = text.lower()
    if "<table" in low or "<tr" in low:
        return True
    return (
        "| date |" in low
        and "| open |" in low
        and "| high |" in low
        and "| low |" in low
        and "| close" in low
    )


def _collect_table_candidates(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        if _string_has_table_hint(obj):
            out.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_table_candidates(value, out)
        return
    if isinstance(obj, list):
        for value in obj:
            _collect_table_candidates(value, out)


def _table_candidate_score(text: str) -> tuple[int, int]:
    low = text.lower()
    score = 0
    score += low.count("<tr") * 4
    score += low.count("\n|") * 2
    if "| date |" in low:
        score += 25
    if "| open |" in low:
        score += 10
    if "| high |" in low:
        score += 10
    if "| low |" in low:
        score += 10
    if "| close" in low:
        score += 10
    if "historical prices" in low:
        score += 10
    return score, len(text)


def _extract_table_text_from_response(payload: dict[str, Any]) -> str:
    candidates: list[str] = []
    _collect_table_candidates(payload, candidates)
    if not candidates:
        return ""
    return max(candidates, key=_table_candidate_score)


def _flatten_columns(columns: Any) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flat: list[str] = []
        for col in columns.to_list():
            flat.append(" ".join(str(part) for part in col if str(part) != "nan").strip())
        return flat
    return [str(col).strip() for col in columns]


def _normalized_col_name(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("\u2212", "-", regex=False)
        .str.replace("--", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _find_history_table(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, int]]:
    required = ("date", "open", "high", "low", "close")
    best_table: pd.DataFrame | None = None
    best_map: dict[str, int] = {}
    best_score = -1

    for table in tables:
        if table is None or table.empty:
            continue

        cols = _flatten_columns(table.columns)
        col_map: dict[str, int] = {}
        for idx, col in enumerate(cols):
            norm = _normalized_col_name(col)
            if norm.startswith("date") and "date" not in col_map:
                col_map["date"] = idx
            elif norm.startswith("open") and "open" not in col_map:
                col_map["open"] = idx
            elif norm.startswith("high") and "high" not in col_map:
                col_map["high"] = idx
            elif norm.startswith("low") and "low" not in col_map:
                col_map["low"] = idx
            elif norm.startswith("close") and not norm.startswith("adjclose") and "close" not in col_map:
                col_map["close"] = idx
            elif norm.startswith("volume") and "volume" not in col_map:
                col_map["volume"] = idx

        score = sum(1 for key in required if key in col_map) + (1 if "volume" in col_map else 0)
        if score > best_score:
            best_table = table
            best_map = col_map
            best_score = score

    if best_table is None or any(key not in best_map for key in required):
        raise ValueError("Could not locate Yahoo historical price table in Firecrawl output.")

    return best_table, best_map


def _normalize_history_table(table: pd.DataFrame, col_map: dict[str, int]) -> pd.DataFrame:
    table = table.copy()
    table.columns = _flatten_columns(table.columns)

    normalized = pd.DataFrame(
        {
            "datetime": table.iloc[:, col_map["date"]],
            "open": table.iloc[:, col_map["open"]],
            "high": table.iloc[:, col_map["high"]],
            "low": table.iloc[:, col_map["low"]],
            "close": table.iloc[:, col_map["close"]],
            "volume": table.iloc[:, col_map["volume"]] if "volume" in col_map else np.nan,
        }
    )

    action_mask = (
        normalized.fillna("").astype(str).agg(" ".join, axis=1).str.contains(
            r"dividend|stock split|capital gain", case=False, regex=True
        )
    )
    normalized = normalized[~action_mask].copy()

    normalized["datetime"] = pd.to_datetime(normalized["datetime"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        normalized[col] = _coerce_numeric(normalized[col])

    normalized = normalized.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    normalized = normalized.sort_values("datetime").drop_duplicates(subset="datetime")
    normalized["volume"] = normalized["volume"].fillna(0.0)

    return normalized[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _parse_yahoo_history_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), displayed_only=False)
    table, col_map = _find_history_table(tables)
    return _normalize_history_table(table, col_map)


def _is_markdown_separator_line(line: str) -> bool:
    stripped = line.strip().strip("|").strip()
    return bool(stripped) and all(ch in "-: " for ch in stripped)


def _extract_markdown_table_blocks(markdown: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("|") and line.count("|") >= 2:
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append(current.copy())
            current = []

    if len(current) >= 2:
        blocks.append(current.copy())

    return blocks


def _markdown_block_to_df(lines: list[str]) -> pd.DataFrame:
    if not lines:
        return pd.DataFrame()

    headers = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
    if not headers:
        return pd.DataFrame()

    start = 1
    if len(lines) > 1 and _is_markdown_separator_line(lines[1]):
        start = 2

    rows: list[list[str]] = []
    width = len(headers)
    for line in lines[start:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        elif len(cells) > width:
            cells = cells[: width - 1] + [" | ".join(cells[width - 1 :])]
        rows.append(cells)

    if not rows:
        return pd.DataFrame(columns=headers)

    return pd.DataFrame(rows, columns=headers)


def _parse_yahoo_history_markdown(markdown: str) -> pd.DataFrame:
    blocks = _extract_markdown_table_blocks(markdown)
    tables = [_markdown_block_to_df(block) for block in blocks]
    tables = [table for table in tables if table is not None and not table.empty]
    if not tables:
        raise ValueError("No markdown tables found in Firecrawl output.")
    table, col_map = _find_history_table(tables)
    return _normalize_history_table(table, col_map)


def _parse_yahoo_history_payload_text(payload_text: str) -> pd.DataFrame:
    errors: list[str] = []

    if "<table" in payload_text.lower() or "<tr" in payload_text.lower():
        try:
            html_df = _parse_yahoo_history_html(payload_text)
            if not html_df.empty:
                return html_df
        except Exception as exc:
            errors.append(f"html parser: {exc}")

    try:
        markdown_df = _parse_yahoo_history_markdown(payload_text)
        if not markdown_df.empty:
            return markdown_df
    except Exception as exc:
        errors.append(f"markdown parser: {exc}")

    detail = "; ".join(errors[-2:]).strip()
    if detail:
        raise ValueError(f"Could not parse Yahoo historical table from Firecrawl output ({detail}).")
    raise ValueError("Could not parse Yahoo historical table from Firecrawl output.")


def _scrape_yahoo_history_with_firecrawl(
    symbol: str,
    force_fresh: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    api_key = (_get_secret_or_env("FIRECRAWL_API_KEY", "") or "").strip()
    if not api_key:
        raise ValueError("Missing FIRECRAWL_API_KEY (set it in .streamlit/secrets.toml or environment).")

    api_url = (_get_secret_or_env("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v2/scrape") or "").strip()
    max_age_ms = _to_int(_get_secret_or_env("FIRECRAWL_MAX_AGE_MS", "172800000"), 172800000)
    if force_fresh:
        max_age_ms = 0
    timeout_seconds = _to_int(_get_secret_or_env("FIRECRAWL_TIMEOUT_SECONDS", "60"), 60)

    payload = {
        "url": _build_yahoo_history_url(symbol),
        "onlyMainContent": False,
        "maxAge": max_age_ms,
        "parsers": ["pdf"],
        "formats": ["markdown", "html"],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(api_url, json=payload, headers=headers, timeout=timeout_seconds)
    if response.status_code >= 400:
        body_preview = response.text[:400].strip()
        raise RuntimeError(f"Firecrawl scrape failed ({response.status_code}): {body_preview}")

    response_json = response.json()
    payload_text = _extract_table_text_from_response(response_json)
    if not payload_text:
        raise RuntimeError("Firecrawl returned no parseable HTML/markdown table content.")

    parsed = _parse_yahoo_history_payload_text(payload_text)
    if parsed.empty:
        raise RuntimeError("Parsed OHLC table is empty after filtering.")
    return parsed, response_json


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(prices) / prices[:-1]


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-10:
        return 0.0
    sim = float(np.dot(v1, v2) / denom)
    return (sim + 1.0) / 2.0


def predict_next_bar(
    current_prices: np.ndarray,
    database: list[dict[str, Any]],
    top_k: int,
    sim_threshold: float,
) -> tuple[float, float]:
    current_returns = calculate_returns(current_prices)
    current_returns = normalize_vector(current_returns)

    sims_and_future: list[tuple[float, float]] = []
    for entry in database:
        sim = cosine_similarity(current_returns, entry["pattern"])
        sims_and_future.append((sim, entry["future_return"]))

    if not sims_and_future:
        return 0.0, 0.0

    sims_and_future.sort(reverse=True, key=lambda x: x[0])
    weighted_return = 0.0
    total_weight = 0.0
    top_sims: list[float] = []
    used = 0

    for sim, fut in sims_and_future[:top_k]:
        if sim < sim_threshold:
            continue
        weight = sim**2
        weighted_return += fut * weight
        total_weight += weight
        top_sims.append(sim)
        used += 1

    if total_weight == 0.0 or used == 0:
        fallback_count = min(len(sims_and_future), top_k)
        if fallback_count == 0:
            return 0.0, 0.0
        fallback_mean = np.mean([f for _, f in sims_and_future[:fallback_count]])
        fallback_conf = np.mean([s for s, _ in sims_and_future[:fallback_count]]) * 100.0
        return float(fallback_mean), float(fallback_conf)

    prediction = float(weighted_return / total_weight)
    confidence = float(np.mean(top_sims) * 100.0)
    return prediction, confidence


def run_walkforward(
    df: pd.DataFrame,
    pattern_size: int,
    train_start: int,
    train_end: int,
    top_k: int,
    sim_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    patterns: list[dict[str, Any]] = []
    closes = df["close"].values
    n = len(closes)

    for i in range(max(pattern_size, train_start), min(train_end, n - 2) + 1):
        window = closes[i - pattern_size : i + 1]
        returns = normalize_vector(calculate_returns(window))
        future_return = (closes[i + 1] - closes[i]) / closes[i]
        patterns.append({"pattern": returns, "future_return": float(future_return), "index": i})

    if len(patterns) < 5:
        raise ValueError("Not enough patterns in training window. Increase training interval or reduce pattern size.")

    records: list[dict[str, Any]] = []
    for i in range(min(train_end + 1, n - 2), n - 1):
        if i - pattern_size < 0:
            continue
        current_prices = closes[i - pattern_size : i + 1]
        pred_ret, conf = predict_next_bar(current_prices, patterns, top_k=top_k, sim_threshold=sim_threshold)
        predicted_price = closes[i] * (1 + pred_ret)
        actual_ret = (closes[i + 1] - closes[i]) / closes[i]
        correct = (pred_ret >= 0 and actual_ret >= 0) or (pred_ret < 0 and actual_ret < 0)
        records.append(
            {
                "bar_index": i,
                "time": df.index[i],
                "base_price": float(closes[i]),
                "predicted_return": float(pred_ret),
                "predicted_price": float(predicted_price),
                "confidence": float(conf),
                "actual_return": float(actual_ret),
                "correct": bool(correct),
            }
        )

    return pd.DataFrame(records), patterns


def _stratify_accuracy(preds_df: pd.DataFrame) -> dict[str, tuple[int, int, float]]:
    hi = preds_df[preds_df["confidence"] >= 75.0]
    mid = preds_df[(preds_df["confidence"] >= 65.0) & (preds_df["confidence"] < 75.0)]
    lo = preds_df[preds_df["confidence"] < 65.0]

    def _stats(sub: pd.DataFrame) -> tuple[int, int, float]:
        cnt = len(sub)
        ok = int(sub["correct"].sum()) if cnt > 0 else 0
        acc = float(100.0 * ok / cnt) if cnt > 0 else float("nan")
        return cnt, ok, acc

    return {"high": _stats(hi), "mid": _stats(mid), "low": _stats(lo)}


def _compute_iii_ema_break_signals(
    data_df: pd.DataFrame,
    wick_ratio_max: float,
    squeeze_threshold_pct: float,
    iii_flip_lookback: int,
    iii_max_flips: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data_df is None or data_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = data_df.copy().sort_values("datetime").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float).fillna(0.0)

    for period in (20, 50, 100, 200):
        df[f"ema{period}"] = close.ewm(span=period, adjust=False).mean()

    rng = (high - low).replace(0, np.nan)
    iip = ((2.0 * close - high - low) / rng).clip(-1.0, 1.0)
    money_flow = iip * volume
    vol_roll = volume.rolling(21, min_periods=5).sum()
    flow_roll = money_flow.rolling(21, min_periods=5).sum()
    df["iii"] = (100.0 * flow_roll / vol_roll).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    iii_blue = df["iii"] >= 0.0
    prev_iii_blue = iii_blue.shift(1, fill_value=False)
    df["iii_turn_blue"] = iii_blue & (~prev_iii_blue)
    df["iii_turn_red"] = (~iii_blue) & prev_iii_blue

    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    df["is_green"] = close > open_
    df["is_red"] = close < open_
    df["wick_clean_long"] = (upper_wick <= (body * wick_ratio_max)) & (body > 0)
    df["wick_clean_short"] = (lower_wick <= (body * wick_ratio_max)) & (body > 0)

    df["ribbon_top"] = df[["ema20", "ema50", "ema100", "ema200"]].max(axis=1)
    df["ribbon_bottom"] = df[["ema20", "ema50", "ema100", "ema200"]].min(axis=1)
    close_safe = close.replace(0, np.nan)
    df["ribbon_width_pct"] = ((df["ribbon_top"] - df["ribbon_bottom"]) / close_safe).abs().fillna(0.0)
    df["inside_ribbon"] = (close >= df["ribbon_bottom"]) & (close <= df["ribbon_top"])
    df["ribbon_squeeze"] = df["ribbon_width_pct"] <= squeeze_threshold_pct

    iii_flip = (iii_blue != iii_blue.shift(1)).astype(int).fillna(0)
    df["iii_flip_count"] = iii_flip.rolling(iii_flip_lookback, min_periods=1).sum()
    df["iii_chop"] = df["iii_flip_count"] > float(iii_max_flips)

    df["bull_trend"] = (
        (df["ema20"] > df["ema50"])
        & (df["ema50"] > df["ema100"])
        & (df["ema100"] > df["ema200"])
    )
    df["bear_trend"] = (
        (df["ema20"] < df["ema50"])
        & (df["ema50"] < df["ema100"])
        & (df["ema100"] < df["ema200"])
    )

    df["close_above_ribbon"] = close > df["ribbon_top"]
    df["close_below_ribbon"] = close < df["ribbon_bottom"]
    df["avoid_zone"] = df["inside_ribbon"] | df["ribbon_squeeze"] | df["iii_chop"]

    long_setup = (
        df["iii_turn_blue"]
        & df["is_green"]
        & df["wick_clean_long"]
        & df["bull_trend"]
        & df["close_above_ribbon"]
        & (~df["avoid_zone"])
    )
    short_setup = (
        df["iii_turn_red"]
        & df["is_red"]
        & df["wick_clean_short"]
        & df["bear_trend"]
        & df["close_below_ribbon"]
        & (~df["avoid_zone"])
    )

    prev_swing_low = low.rolling(5, min_periods=1).min().shift(1)
    prev_swing_high = high.rolling(5, min_periods=1).max().shift(1)
    stop_long = pd.concat([df["ema50"], prev_swing_low], axis=1).min(axis=1)
    stop_short = pd.concat([df["ema50"], prev_swing_high], axis=1).max(axis=1)

    entry_long = high
    entry_short = low
    risk_long = entry_long - stop_long
    risk_short = stop_short - entry_short

    valid_long = long_setup & (risk_long > 0)
    valid_short = short_setup & (risk_short > 0)

    df["side"] = np.select([valid_long, valid_short], ["LONG", "SHORT"], default="")
    df["entry"] = np.where(valid_long, entry_long, np.where(valid_short, entry_short, np.nan))
    df["stop"] = np.where(valid_long, stop_long, np.where(valid_short, stop_short, np.nan))
    df["target_1_5r"] = np.where(
        valid_long,
        entry_long + (risk_long * 1.5),
        np.where(valid_short, entry_short - (risk_short * 1.5), np.nan),
    )
    df["target_2r"] = np.where(
        valid_long,
        entry_long + (risk_long * 2.0),
        np.where(valid_short, entry_short - (risk_short * 2.0), np.nan),
    )

    signals = df[df["side"] != ""].copy()
    signals = signals[
        [
            "datetime",
            "side",
            "close",
            "iii",
            "entry",
            "stop",
            "target_1_5r",
            "target_2r",
            "ema20",
            "ema50",
            "ema100",
            "ema200",
            "volume",
        ]
    ].reset_index(drop=True)
    return df, signals


def _render_iii_ema_strategy(symbol: str, data_df: pd.DataFrame) -> None:
    st.subheader("III + EMA Break Strategy (Article-based)")
    st.caption(
        "Uses Intraday Intensity Index (III) + EMA 20/50/100/200 alignment on the scraped OHLCV bars. "
        "Long: III turns blue + bullish break above ribbon. Short: III turns red + bearish break below ribbon."
    )
    st.caption(
        "Note: this tab currently uses Yahoo historical bars from Firecrawl (usually daily history). "
        "For strict 3-minute execution, wire this same logic to an intraday 3m data source."
    )

    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        wick_ratio_max = float(
            st.slider(
                "Max wick/body ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key=f"fc_iii_wick_{symbol}",
            )
        )
    with k2:
        squeeze_threshold_pct = float(
            st.slider(
                "Ribbon squeeze threshold (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.60,
                step=0.05,
                key=f"fc_iii_squeeze_{symbol}",
            )
        ) / 100.0
    with k3:
        iii_flip_lookback = int(
            st.slider(
                "III flip lookback (bars)",
                min_value=3,
                max_value=20,
                value=6,
                step=1,
                key=f"fc_iii_flip_lb_{symbol}",
            )
        )
    with k4:
        iii_max_flips = int(
            st.slider(
                "Max III flips in lookback",
                min_value=0,
                max_value=10,
                value=3,
                step=1,
                key=f"fc_iii_flip_max_{symbol}",
            )
        )

    calc_df, sig_df = _compute_iii_ema_break_signals(
        data_df=data_df,
        wick_ratio_max=wick_ratio_max,
        squeeze_threshold_pct=squeeze_threshold_pct,
        iii_flip_lookback=iii_flip_lookback,
        iii_max_flips=iii_max_flips,
    )
    if calc_df.empty:
        st.info("No OHLCV data available for III + EMA strategy.")
        return

    long_count = int((calc_df["side"] == "LONG").sum())
    short_count = int((calc_df["side"] == "SHORT").sum())
    latest_iii = float(calc_df["iii"].iloc[-1]) if len(calc_df) else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("LONG setups", f"{long_count}")
    m2.metric("SHORT setups", f"{short_count}")
    m3.metric("Latest III", f"{latest_iii:.2f}")

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=calc_df["datetime"],
            y=calc_df["close"],
            name="Close",
            line=dict(color="#111111", width=1.8),
        )
    )
    price_fig.add_trace(go.Scatter(x=calc_df["datetime"], y=calc_df["ema20"], name="EMA20", line=dict(width=1.2)))
    price_fig.add_trace(go.Scatter(x=calc_df["datetime"], y=calc_df["ema50"], name="EMA50", line=dict(width=1.2)))
    price_fig.add_trace(go.Scatter(x=calc_df["datetime"], y=calc_df["ema100"], name="EMA100", line=dict(width=1.2)))
    price_fig.add_trace(go.Scatter(x=calc_df["datetime"], y=calc_df["ema200"], name="EMA200", line=dict(width=1.2)))

    long_points = calc_df[calc_df["side"] == "LONG"]
    short_points = calc_df[calc_df["side"] == "SHORT"]
    if not long_points.empty:
        price_fig.add_trace(
            go.Scatter(
                x=long_points["datetime"],
                y=long_points["entry"],
                mode="markers",
                name="LONG signal",
                marker=dict(symbol="triangle-up", size=12, color="#0b8a00"),
            )
        )
    if not short_points.empty:
        price_fig.add_trace(
            go.Scatter(
                x=short_points["datetime"],
                y=short_points["entry"],
                mode="markers",
                name="SHORT signal",
                marker=dict(symbol="triangle-down", size=12, color="#c62828"),
            )
        )
    price_fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(price_fig, use_container_width=True)

    iii_colors = np.where(calc_df["iii"] >= 0, "#1f77b4", "#d62728")
    iii_fig = go.Figure()
    iii_fig.add_trace(
        go.Bar(
            x=calc_df["datetime"],
            y=calc_df["iii"],
            name="III",
            marker_color=iii_colors,
        )
    )
    iii_fig.add_hline(y=0.0, line_color="#6b7280", line_dash="dash")
    iii_fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(iii_fig, use_container_width=True)

    st.write("### III + EMA signals")
    if sig_df.empty:
        st.info("No valid LONG/SHORT setups found with current filters.")
    else:
        st.dataframe(sig_df.sort_values("datetime", ascending=False).head(150), use_container_width=True)
        sig_csv = sig_df.to_csv(index=False)
        st.download_button(
            "Download III+EMA signals CSV",
            sig_csv,
            file_name=f"{symbol.lower()}_iii_ema_signals.csv",
            mime="text/csv",
        )


def _render_results(symbol: str, data_df: pd.DataFrame, preds_df: pd.DataFrame, params: dict[str, Any]) -> None:
    total = len(preds_df)
    correct = int(preds_df["correct"].sum()) if total else 0
    overall_acc = (100.0 * correct / total) if total else 0.0
    strat = _stratify_accuracy(preds_df)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Price chart + predictions")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_df["datetime"],
                y=data_df["close"],
                name=f"{symbol} close",
                line=dict(color="black"),
            )
        )
        if len(preds_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=preds_df["time"],
                    y=preds_df["predicted_price"],
                    name="predicted_price",
                    mode="markers+lines",
                    line=dict(dash="dash"),
                )
            )
            sigs = preds_df[preds_df["confidence"] >= 75.0]
            if not sigs.empty:
                bullish = sigs[sigs["predicted_return"] > 0]
                bearish = sigs[sigs["predicted_return"] < 0]
                if not bullish.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=bullish["time"],
                            y=bullish["base_price"],
                            mode="markers",
                            marker_symbol="triangle-up",
                            marker_size=12,
                            name="bull_signals",
                            marker=dict(color="green"),
                        )
                    )
                if not bearish.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=bearish["time"],
                            y=bearish["base_price"],
                            mode="markers",
                            marker_symbol="triangle-down",
                            marker_size=12,
                            name="bear_signals",
                            marker=dict(color="red"),
                        )
                    )
        fig.update_layout(height=620, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction table (last 100)")
        st.dataframe(preds_df.sort_values("time", ascending=False).head(100), height=300, use_container_width=True)

    with right:
        st.metric("Total predictions", f"{total}")
        st.metric("Overall accuracy %", f"{overall_acc:.2f}%")

        hi_cnt, hi_ok, hi_acc = strat["high"]
        mid_cnt, mid_ok, mid_acc = strat["mid"]
        lo_cnt, lo_ok, lo_acc = strat["low"]
        st.write("### Confidence stratification")
        st.write(
            {
                ">75%": f"{hi_ok}/{hi_cnt} ({0.0 if np.isnan(hi_acc) else hi_acc:.1f}%)",
                "65-75%": f"{mid_ok}/{mid_cnt} ({0.0 if np.isnan(mid_acc) else mid_acc:.1f}%)",
                "<65%": f"{lo_ok}/{lo_cnt} ({0.0 if np.isnan(lo_acc) else lo_acc:.1f}%)",
            }
        )
        st.write("### Parameters used")
        st.json(params)

    st.divider()
    _render_iii_ema_strategy(symbol=symbol, data_df=data_df)

    st.write("### Scraped data sample")
    st.dataframe(data_df.tail(30), use_container_width=True)

    ohlc_csv = data_df.to_csv(index=False)
    st.download_button(
        "Download normalized OHLC CSV",
        ohlc_csv,
        file_name=f"{symbol.lower()}_firecrawl_ohlc.csv",
        mime="text/csv",
    )

    preds_csv = preds_df.to_csv(index=False)
    st.download_button(
        "Download predictions CSV",
        preds_csv,
        file_name=f"{symbol.lower()}_nobel_predictions.csv",
        mime="text/csv",
    )


def render_tab_nobel_predictor_firecrawl(symbol: str) -> None:
    st.subheader("🧠 Nobel Predictor (Firecrawl Yahoo History)")
    st.caption(
        "Scrapes Yahoo historical data through Firecrawl, removes Dividend/Split rows, "
        "normalizes to datetime/open/high/low/close/volume, then runs the Nobel predictor."
    )

    symbol_default = (symbol or "AAPL").upper().strip()
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        target_symbol = st.text_input(
            "Symbol for scrape",
            value=symbol_default,
            key=f"fc_nobel_symbol_{symbol_default}",
        ).upper().strip()
    with c2:
        pattern_size = int(
            st.number_input(
                "Pattern length",
                min_value=5,
                max_value=200,
                value=PRICE_PATTERN_SIZE_DEFAULT,
                step=1,
                key=f"fc_nobel_pattern_{symbol_default}",
            )
        )
    with c3:
        top_k = int(
            st.number_input(
                "Top K matches",
                min_value=1,
                max_value=50,
                value=TOP_K_DEFAULT,
                step=1,
                key=f"fc_nobel_topk_{symbol_default}",
            )
        )
    with c4:
        sim_threshold = float(
            st.slider(
                "Similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=SIM_THRESHOLD_DEFAULT,
                step=0.01,
                key=f"fc_nobel_sim_{symbol_default}",
            )
        )

    c5, c6 = st.columns([1, 1])
    with c5:
        train_start_pct = int(
            st.slider(
                "Train start % of history",
                min_value=0,
                max_value=90,
                value=0,
                key=f"fc_nobel_train_start_{symbol_default}",
            )
        )
    with c6:
        train_end_pct = int(
            st.slider(
                "Train end % of history",
                min_value=10,
                max_value=99,
                value=70,
                key=f"fc_nobel_train_end_{symbol_default}",
            )
        )

    run_btn = st.button("Scrape Yahoo + Run Backtest", key=f"fc_nobel_run_{symbol_default}")
    result_key = f"fc_nobel_result_{target_symbol}"

    if run_btn:
        if not target_symbol:
            st.error("Enter a valid symbol.")
            return

        try:
            with st.spinner(f"Scraping Yahoo history for {target_symbol} via Firecrawl..."):
                parsed_df, raw_response = _scrape_yahoo_history_with_firecrawl(
                    target_symbol,
                    force_fresh=True,
                )
        except Exception as exc:
            st.error(f"Failed to scrape/parse history: {exc}")
            return

        if len(parsed_df) < pattern_size + 20:
            st.error(
                f"Not enough rows after cleaning ({len(parsed_df)}). "
                f"Need at least {pattern_size + 20} rows."
            )
            return

        indexed = parsed_df.copy().set_index("datetime")
        n = len(indexed)
        train_start = int(n * train_start_pct / 100.0)
        train_end = int(n * train_end_pct / 100.0)

        try:
            preds_df, _ = run_walkforward(
                indexed,
                pattern_size=pattern_size,
                train_start=train_start,
                train_end=train_end,
                top_k=top_k,
                sim_threshold=sim_threshold,
            )
        except Exception as exc:
            st.error(f"Error running predictor: {exc}")
            return

        st.session_state[result_key] = {
            "symbol": target_symbol,
            "data_df": parsed_df,
            "preds_df": preds_df,
            "params": {
                "pattern_size": pattern_size,
                "top_k": top_k,
                "sim_threshold": sim_threshold,
                "train_start_pct": train_start_pct,
                "train_end_pct": train_end_pct,
                "rows_used": len(parsed_df),
                "firecrawl_success": bool(raw_response.get("success", True)),
            },
        }

    result = st.session_state.get(result_key)
    if not result:
        st.info("Click `Scrape Yahoo + Run Backtest` to build data and run predictions.")
        return

    _render_results(
        symbol=result["symbol"],
        data_df=result["data_df"],
        preds_df=result["preds_df"],
        params=result["params"],
    )
