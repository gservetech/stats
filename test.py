import json
import os
import re
import sys
import urllib.parse

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

BASE_URL = "https://www.barchart.com/proxies/core-api/v1/options/get"

DEFAULT_PARAMS = {
    "baseSymbol": "AAPL",
    "fields": (
        "symbol,baseSymbol,strikePrice,expirationDate,moneyness,bidPrice,midpoint,askPrice,"
        "lastPrice,priceChange,percentChange,volume,openInterest,openInterestChange,volatility,"
        "delta,optionType,daysToExpiration,tradeTime,averageVolatility,historicVolatility30d,"
        "baseNextEarningsDate,dividendExDate,baseTimeCode,expirationType,impliedVolatilityRank1y,"
        "symbolCode,symbolType"
    ),
    "groupBy": "optionType",
    "expirationDate": "2026-02-23",
    "meta": "field.shortName,expirations,field.description",
    "orderBy": "strikePrice",
    "orderDir": "asc",
    "optionsOverview": "true",
    "expirationType": "weekly",
    "raw": "1",
}

BASE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.barchart.com",
    "Referer": "https://www.barchart.com/stocks/quotes/AAPL/options",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def build_warmup_url(params: dict) -> str:
    symbol = str(params.get("baseSymbol", "AAPL")).lstrip("$")
    expiration = str(params.get("expirationDate", "")).strip()
    if expiration:
        return (
            f"https://www.barchart.com/stocks/quotes/{symbol}/options"
            f"?expiration={expiration}-w&view=sbs"
        )
    return f"https://www.barchart.com/stocks/quotes/{symbol}/options"


def parse_cookie_header(cookie_header: str) -> dict:
    cookies = {}
    for part in cookie_header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        # requests/http cookie handling requires latin-1 encodable values.
        # Skip malformed/truncated entries copied from UI previews (e.g., containing unicode ellipsis).
        try:
            key.encode("latin-1")
            value.encode("latin-1")
        except UnicodeEncodeError:
            continue
        cookies[key] = value
    return cookies


def get_cookie_value_from_jar(cookie_jar: requests.cookies.RequestsCookieJar, name: str) -> str | None:
    # requests.cookies.get(name) can raise CookieConflictError when same name exists across domains/paths.
    values = [cookie.value for cookie in cookie_jar if cookie.name == name and cookie.value]
    if not values:
        return None
    return values[-1]


def extract_cookie_from_input(cookie_input: str) -> str:
    value = cookie_input.strip()
    if not value:
        return ""

    # Supports pasted DevTools curl command with -b 'k=v; ...'
    if "curl " in value:
        match = re.search(r"""(?:^|\s)-b\s+(['"])(.*?)\1""", value, flags=re.DOTALL)
        if match:
            return match.group(2).strip()

        # Fallback: parse cookie from -H 'cookie: ...'
        match = re.search(
            r"""(?:^|\s)-H\s+(['"])cookie:\s*(.*?)\1""",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(2).strip()

    return value


def fetch_options_data(cookie_input: str, params: dict | None = None) -> dict:
    request_params = params or DEFAULT_PARAMS
    cookie_header = extract_cookie_from_input(cookie_input)
    if not cookie_header:
        raise ValueError("Cookie header is required.")
    if cookie_header.startswith("http://") or cookie_header.startswith("https://"):
        raise ValueError(
            "BARCHART_COOKIE is a URL. It must be raw cookie text like "
            "'XSRF-TOKEN=...; laravel_session=...; ...'."
        )

    headers = dict(BASE_HEADERS)
    headers["Referer"] = build_warmup_url(request_params)

    cookies = parse_cookie_header(cookie_header)
    xsrf_override = os.getenv("BARCHART_XSRF_TOKEN") or os.getenv("BARCHART_DIRECT_XSRF")

    with requests.Session() as session:
        session.trust_env = False
        session.headers.update(headers)
        session.cookies.update(cookies)

        warmup_url = headers["Referer"]
        try:
            session.get(warmup_url, timeout=30)
        except requests.RequestException:
            pass

        xsrf_cookie = get_cookie_value_from_jar(session.cookies, "XSRF-TOKEN") or cookies.get("XSRF-TOKEN")
        if xsrf_override:
            session.headers["X-XSRF-TOKEN"] = xsrf_override
        elif xsrf_cookie:
            session.headers["X-XSRF-TOKEN"] = urllib.parse.unquote(xsrf_cookie)

        response = session.get(
            BASE_URL,
            params=request_params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    cookie_input = (
        os.getenv("BARCHART_COOKIE")
        or os.getenv("BARCHART_CURL")
        or os.getenv("BARCHART_DIRECT_COOKIE")
    )
    if len(sys.argv) > 1 and sys.argv[1].strip():
        cookie_input = sys.argv[1].strip()

    if not cookie_input:
        print(
            "Missing cookie auth. Set BARCHART_COOKIE/BARCHART_CURL/"
            "BARCHART_DIRECT_COOKIE or pass first argument.",
            file=sys.stderr,
        )
        return 1

    try:
        data = fetch_options_data(cookie_input)
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        if exc.response is not None:
            if exc.response.status_code == 401:
                print(
                    "401 Unauthorized: cookie is likely incomplete/expired. "
                    "Use a fresh full browser cookie including laravel_session + XSRF-TOKEN.",
                    file=sys.stderr,
                )
            print(exc.response.text[:2000], file=sys.stderr)
        return 2
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
