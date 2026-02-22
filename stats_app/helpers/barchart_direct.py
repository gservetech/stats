import datetime as dt
import math
import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

BASE_URL = "https://www.barchart.com/proxies/core-api/v1/options/get"

BASE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.barchart.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def _to_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "unch"}:
        return default
    s = s.replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return default


def _to_int(val: Any, default: int = 0) -> int:
    f = _to_float(val, None)
    if f is None:
        return int(default)
    return int(round(f))


def _iv_to_decimal(iv_val: Any) -> float | None:
    if iv_val is None:
        return None
    if isinstance(iv_val, (int, float)):
        v = float(iv_val)
        return v / 100.0 if v > 3 else v
    s = str(iv_val).strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "unch"}:
        return None
    if s.endswith("%"):
        v = _to_float(s[:-1], None)
        return (v / 100.0) if v is not None else None
    v = _to_float(s, None)
    if v is None:
        return None
    return v / 100.0 if v > 3 else v


def _years_to_expiry(date_yyyy_mm_dd: str) -> float:
    now = dt.datetime.now()
    exp = dt.datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").replace(
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )
    dt_seconds = (exp - now).total_seconds()
    return max(dt_seconds, 0.0) / (365.0 * 24 * 3600.0)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return (math.exp(-q * T) * _norm_pdf(d1)) / (S * sigma * math.sqrt(T))


def _parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in cookie_header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        cookies[key.strip()] = value.strip()
    return cookies


def _extract_cookie_from_input(cookie_input: str) -> str:
    value = (cookie_input or "").strip()
    if not value:
        return ""

    if "curl " in value:
        match = re.search(r"""(?:^|\s)-b\s+(['"])(.*?)\1""", value, flags=re.DOTALL)
        if match:
            return match.group(2).strip()
        match = re.search(
            r"""(?:^|\s)-H\s+(['"])cookie:\s*(.*?)\1""",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(2).strip()
    return value


def _cookie_value(cookie_jar: requests.cookies.RequestsCookieJar, name: str) -> str | None:
    values = [cookie.value for cookie in cookie_jar if cookie.name == name and cookie.value]
    return values[-1] if values else None


def _iter_contracts(payload: dict) -> list[dict]:
    data = payload.get("data", payload)
    contracts: list[dict] = []

    if isinstance(data, dict):
        if isinstance(data.get("Call"), list):
            contracts.extend([x for x in data["Call"] if isinstance(x, dict)])
        if isinstance(data.get("Put"), list):
            contracts.extend([x for x in data["Put"] if isinstance(x, dict)])
        if contracts:
            return contracts

        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and ("optionType" in item or "raw" in item):
                        contracts.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("options"), list):
                        contracts.extend([x for x in item["options"] if isinstance(x, dict)])
            elif isinstance(value, dict) and ("optionType" in value or "raw" in value):
                contracts.append(value)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and ("optionType" in item or "raw" in item):
                contracts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("options"), list):
                contracts.extend([x for x in item["options"] if isinstance(x, dict)])

    return contracts


def _contracts_to_gex_df(
    contracts: list[dict],
    spot: float,
    date: str,
    r: float = 0.05,
    q: float = 0.0,
    multiplier: int = 100,
) -> pd.DataFrame:
    per_strike: dict[float, dict[str, Any]] = {}

    for item in contracts:
        raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
        strike = _to_float(item.get("strikePrice"), _to_float(raw.get("strikePrice"), None))
        if strike is None:
            continue

        side = (
            str(item.get("optionType") or raw.get("optionType") or item.get("symbolType") or raw.get("symbolType") or "")
            .strip()
            .lower()
        )
        is_call = side.startswith("c")
        is_put = side.startswith("p")
        if not (is_call or is_put):
            continue

        entry = per_strike.setdefault(
            strike,
            {
                "strike": float(strike),
                "Call IV": "",
                "Put IV": "",
                "call_iv_dec": None,
                "put_iv_dec": None,
                "call_oi": 0,
                "put_oi": 0,
                "call_vol": 0,
                "put_vol": 0,
            },
        )

        iv_raw = item.get("volatility")
        if iv_raw in (None, ""):
            iv_raw = raw.get("volatility")
        iv_dec = _iv_to_decimal(iv_raw)
        iv_txt = str(iv_raw) if iv_raw not in (None, "") else ""

        oi = _to_int(item.get("openInterest"), _to_int(raw.get("openInterest"), 0))
        vol = _to_int(item.get("volume"), _to_int(raw.get("volume"), 0))

        if is_call:
            entry["Call IV"] = iv_txt
            entry["call_iv_dec"] = iv_dec
            entry["call_oi"] = oi
            entry["call_vol"] = vol
        if is_put:
            entry["Put IV"] = iv_txt
            entry["put_iv_dec"] = iv_dec
            entry["put_oi"] = oi
            entry["put_vol"] = vol

    if not per_strike:
        return pd.DataFrame()

    df = pd.DataFrame([v for _, v in sorted(per_strike.items(), key=lambda kv: kv[0])])
    T = _years_to_expiry(date)

    gammas_call = []
    gammas_put = []
    for _, row in df.iterrows():
        civ = row["call_iv_dec"]
        piv = row["put_iv_dec"]
        g_call = _bs_gamma(spot, row["strike"], T, r, q, civ) if civ else 0.0
        g_put = _bs_gamma(spot, row["strike"], T, r, q, piv) if piv else 0.0
        if not g_call and g_put:
            g_call = g_put
        if not g_put and g_call:
            g_put = g_call
        gammas_call.append(g_call)
        gammas_put.append(g_put)

    df["gamma_call"] = gammas_call
    df["gamma_put"] = gammas_put

    s2 = spot * spot
    df["call_gex"] = 0.01 * df["gamma_call"] * df["call_oi"] * multiplier * s2
    df["put_gex"] = 0.01 * df["gamma_put"] * df["put_oi"] * multiplier * s2
    df["net_gex"] = df["call_gex"] - df["put_gex"]

    return df[
        [
            "strike",
            "Call IV",
            "Put IV",
            "gamma_call",
            "gamma_put",
            "call_oi",
            "put_oi",
            "call_vol",
            "put_vol",
            "call_gex",
            "put_gex",
            "net_gex",
        ]
    ].copy()


def _build_weekly_summary_payload(symbol: str, date: str, spot: float, gex_df: pd.DataFrame) -> dict:
    total_call_oi = float(gex_df["call_oi"].sum())
    total_put_oi = float(gex_df["put_oi"].sum())
    total_call_vol = float(gex_df["call_vol"].sum())
    total_put_vol = float(gex_df["put_vol"].sum())

    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else None

    total_call_gex = float(gex_df["call_gex"].sum())
    total_put_gex = float(gex_df["put_gex"].sum())

    totals = {
        "call_oi": total_call_oi,
        "put_oi": total_put_oi,
        "call_volume": total_call_vol,
        "put_volume": total_put_vol,
        "call_gex": total_call_gex,
        "put_gex": total_put_gex,
        "net_gex": total_call_gex - total_put_gex,
    }

    return {
        "success": True,
        "symbol": symbol,
        "date": date,
        "spot": spot,
        "pcr": {"oi": pcr_oi, "volume": pcr_vol},
        "totals": totals,
    }


@dataclass
class BarchartDirectClient:
    cookie_input: str
    xsrf_override: str | None = None
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        self.cookie_header = _extract_cookie_from_input(self.cookie_input or "")
        self.cookies = _parse_cookie_header(self.cookie_header)
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(BASE_HEADERS)

        if self.cookies:
            self.session.cookies.update(self.cookies)

    @property
    def ready(self) -> bool:
        return bool(self.cookie_header)

    @classmethod
    def from_env(cls) -> "BarchartDirectClient":
        if load_dotenv is not None:
            load_dotenv()
        cookie_input = (
            os.getenv("BARCHART_DIRECT_COOKIE")
            or os.getenv("BARCHART_COOKIE")
            or os.getenv("BARCHART_CURL")
            or ""
        )
        xsrf_override = os.getenv("BARCHART_DIRECT_XSRF") or os.getenv("BARCHART_XSRF_TOKEN")
        timeout = _to_int(os.getenv("BARCHART_DIRECT_TIMEOUT_SECONDS"), 30)
        return cls(cookie_input=cookie_input, xsrf_override=xsrf_override, timeout_seconds=max(5, timeout))

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def _warmup_url(self, symbol: str, date: str) -> str:
        sym = (symbol or "").strip().upper().lstrip("$")
        return f"https://www.barchart.com/stocks/quotes/{sym}/options?expiration={date}-w&view=sbs"

    def fetch_options_payload(self, symbol: str, date: str, expiration_type: str = "weekly") -> dict:
        if not self.ready:
            raise ValueError("Barchart direct cookie is missing.")

        symbol_clean = (symbol or "").strip().upper()
        params = {
            "baseSymbol": symbol_clean,
            "fields": (
                "symbol,baseSymbol,strikePrice,expirationDate,moneyness,bidPrice,midpoint,askPrice,"
                "lastPrice,priceChange,percentChange,volume,openInterest,openInterestChange,volatility,"
                "delta,optionType,daysToExpiration,tradeTime,averageVolatility,historicVolatility30d,"
                "baseNextEarningsDate,dividendExDate,baseTimeCode,expirationType,impliedVolatilityRank1y,"
                "symbolCode,symbolType"
            ),
            "groupBy": "optionType",
            "expirationDate": date,
            "meta": "field.shortName,expirations,field.description",
            "orderBy": "strikePrice",
            "orderDir": "asc",
            "optionsOverview": "true",
            "expirationType": expiration_type,
            "raw": "1",
        }

        referer = self._warmup_url(symbol_clean, date)
        self.session.headers["Referer"] = referer

        try:
            self.session.get(referer, timeout=self.timeout_seconds)
        except requests.RequestException:
            pass

        xsrf_cookie = _cookie_value(self.session.cookies, "XSRF-TOKEN") or self.cookies.get("XSRF-TOKEN")
        if self.xsrf_override:
            self.session.headers["X-XSRF-TOKEN"] = self.xsrf_override
        elif xsrf_cookie:
            self.session.headers["X-XSRF-TOKEN"] = urllib.parse.unquote(xsrf_cookie)

        response = self.session.get(BASE_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def fetch_weekly_summary_and_gex(
        self,
        symbol: str,
        date: str,
        spot: float,
        r: float = 0.05,
        q: float = 0.0,
        multiplier: int = 100,
    ) -> dict:
        contracts: list[dict] = []
        used_expiration_type = None
        last_error = None

        for exp_type in ("weekly", "monthly"):
            try:
                payload = self.fetch_options_payload(symbol=symbol, date=date, expiration_type=exp_type)
            except Exception as exc:
                last_error = str(exc)
                continue
            contracts = _iter_contracts(payload)
            if contracts:
                used_expiration_type = exp_type
                break

        if not contracts:
            err = "No contracts found in direct API response."
            if last_error:
                err = f"{err} Last error: {last_error}"
            return {"success": False, "error": err}

        gex_df = _contracts_to_gex_df(
            contracts=contracts,
            spot=float(spot),
            date=date,
            r=float(r),
            q=float(q),
            multiplier=int(multiplier),
        )
        if gex_df.empty:
            return {"success": False, "error": "Unable to build GEX table from direct API response."}

        summary_payload = _build_weekly_summary_payload(symbol=symbol, date=date, spot=float(spot), gex_df=gex_df)
        gex_payload = {
            "success": True,
            "symbol": symbol,
            "date": date,
            "spot": float(spot),
            "count": int(len(gex_df)),
            "data": gex_df.to_dict(orient="records"),
        }

        return {
            "success": True,
            "summary": summary_payload,
            "gex": gex_payload,
            "contracts_count": len(contracts),
            "expiration_type": used_expiration_type or "weekly",
        }
