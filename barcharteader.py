import requests
import urllib.parse
import json

BASE_URL = "https://www.barchart.com/proxies/core-api/v1/options/get"

params = {
    "symbol": "AAPL",
    "fields": (
        "symbol,optionType,strikePrice,expirationDate,lastPrice,bidPrice,askPrice,"
        "priceChange,volume,openInterest,volatility,tradeTime,baseNextEarningsDate,"
        "dividendExDate,baseTimeCode,expirationType,impliedVolatilityRank1y"
    ),
    "groupBy": "strikePrice",
    "raw": "1",
    "expirationDate": "2026-04-02",
    "meta": "field.shortName,field.description,expirations,field.type",
    "orderBy": "strikePrice",
    "orderDir": "asc",
    "optionsOverview": "true",
}

COOKIE_STR = """curl 'https://www.barchart.com/proxies/core-api/v1/options/get?baseSymbol=AAPL&fields=symbol%2CbaseSymbol%2CstrikePrice%2CexpirationDate%2Cmoneyness%2CbidPrice%2Cmidpoint%2CaskPrice%2ClastPrice%2CpriceChange%2CpercentChange%2Cvolume%2CopenInterest%2CopenInterestChange%2Cvolatility%2Cdelta%2CoptionType%2CdaysToExpiration%2CexpirationDate%2CtradeTime%2CaverageVolatility%2ChistoricVolatility30d%2CbaseNextEarningsDate%2CdividendExDate%2CbaseTimeCode%2CexpirationType%2CimpliedVolatilityRank1y%2CsymbolCode%2CsymbolType&groupBy=optionType&expirationDate=2026-03-27&meta=field.shortName%2Cexpirations%2Cfield.description&orderBy=strikePrice&orderDir=asc&optionsOverview=true&expirationType=weekly&raw=1' \
  -H 'accept: application/json' \
  -H 'accept-language: en-CA,en;q=0.9,fr-CA;q=0.8,fr;q=0.7,en-GB;q=0.6,en-US;q=0.5' \
  -b '_gcl_au=1.1.84300422.1767226181; _ga=GA1.1.1625575563.1767226181; _fbp=fb.1.1767226181122.109975775380389463; usprivacy=1YNY; _scor_uid=c707d8dc050e49d5becd22c8ae845d01; _cc_id=51af77578339dde2c033b5dbb6c903b4; _li_ss=CgA; cto_bundle=vW2HN19PVFF4bEdJQXoxRTN2cmZBNnpBbmNsYm1BRFpWMmdIbzFsbHklMkZ1OHhBcWFxMThjJTJCY1YlMkZpVE1zc1A0aHFBeDlrYzlPcWZoM25CcUlDdVl1Wmh3YXdlUXAwMkUlMkYzYURMWiUyQnRPTGZXcExneU9OVzBCYXRrVEMyMXhwQXVYWE1xWlMlMkZqYXpzSSUyRnJBcGVuNXolMkYlMkZLYTZtbXclM0QlM0Q; cto_bidid=vwV9_F9oWmp2NFJ4R0VrYWw1cTM1cSUyRms2OXZlenRtZ05QJTJGUXF3alZjTFdWemlFYldaTDNiYVI2YXNoVXN5ZlM2TlUxMWtvJTJCVXJLQm1KaG8wTDRuMWdOR0FjNFN6UE0yNFUwR09ZWjhOdSUyQmt4JTJGQVklM0Q; _hjSessionUser_2563157=eyJpZCI6ImM3NWVjN2VkLTU2ODItNTRhNy1iMmM3LTk2OWM3MGM1ZGE4MSIsImNyZWF0ZWQiOjE3NjcyMjYyMDkyMDYsImV4aXN0aW5nIjp0cnVlfQ==; remember_web_59ba36addc2b2f9401580f014c7f58ea4e30989d=eyJpdiI6IlBBYTRmbFhGUGg1bnJaOWtjUEZlSFE9PSIsInZhbHVlIjoiUlF0Qm5CZjVYMEpxbVc2L0NpR3V4enJxOWNET1JpNm1jS0FpL2hhdlVXZysvd1EvUkd1WjkxZlg2WTBHNFlMVWltY2EyQUtGNmYzNU4wRUJsZmMrMDhIUng3ZGx0bjdsQXNIalY2ODhtQ1dHTnYzV1pubXR6UlNwVURzd2w0Uk5hWkE1amNJMGlycFVkNGw3akxpaGpRTlA5SUFzS3JubThuVXRoT0dCSXR4YUFzVkgvU0lqaEwrdi9OTGJZbmc0SEtKMWpETks0Vys0ekl4dG1TZDhTbFpUQy9QTFJMNFVLNWtWbGJTUHdLUT0iLCJtYWMiOiI3ZDI0YjM2ZjFmNDdmZjVkZDM1NTIwZTM2MzVhN2VkMDIzNmE4YzUxZWZlNjQwOWJiOTI2YWYyOGVkZjk0NDU5IiwidGFnIjoiIn0%3D; panoramaId=8aa72d231cdfd844993ec577e02b16d53938c7641becdb4ddb45a7fd9a229e8e; cnx_userId=1-5a5cfed82b43470288ab587af709fff1; _pubcid=e581b8cb-2745-4fc5-b4a8-c9f0793e9057; panoramaId_expiry=1772069889856; panoramaIdType=panoIndiv; g_state={"i_l":0,"i_ll":1771465091250,"i_b":"6MEOQ812kRySvoe8f4L92VgUmDzQmnAiciRoElWRC6w","i_e":{"enable_itp_optimization":0}}; remember_web_3dc7a913ef5fd4b890ecabe3487085573e16cf82=eyJpdiI6ImdSbTBKeXdaWC9rWWdjZVRLekx3T1E9PSIsInZhbHVlIjoiUjBBZGtOQ3o1V01WdWFvTHNENGp3ZDIvei96ZW1vSVFCR3duUzV5UDJGRzBqS3RoclBjcS9hQVljcGFnZDZiN21TZEF6c1hZZkV3SEJFOUp2d1B2N2ZtNUlwZllwQ2tHb2xvTGYrSEVZL0V0blVKVGJXaHhQcEJCMWVaTmtudU5kRXAvNnkzVGRsc3VyNDNEcFZkMEtPTzA3b08zMDFyMzJ4eXp3V0FwWXZuRko4TXh4djRPeHNGRjhTN3VPUDdGcFJPQTNId2ptcG5rUEw5STFxeVFCM0VxY29RcGJUQnMyazZsdDdqRENXVT0iLCJtYWMiOiIyYjZjMzViYjFlZThjZDk0ZmJkOTliOGRlYzRjMTY1Mzc1OWJjZTA3NmE4MmYyZTZkNzk5ZWRmZmM4OWU3ZjkzIiwidGFnIjoiIn0%3D; market=eyJpdiI6IjZuNFNOZEtYdDlIMjU5VmNsTUNxNFE9PSIsInZhbHVlIjoiaUx5aXRmSS8zNTVxUVBGYmdwYzVGRXhpamg0eGpOVjdMMGVUUGczVWljenpLVGJROFE5U0N3bUk0V2d6ZlFtOSIsIm1hYyI6IjljNjQ4NzVhNGU0NGE3MzQxOTI0ZTkxNzUzM2IwNmNjODQ5YWViNzE2OWJiYmQwMGUzNTBkZjMzODZkY2MxZjUiLCJ0YWciOiIifQ%3D%3D; webinarClosed=353; ab.storage.deviceId.bf243db8-6e0e-43fa-9ce4-2afc85959789=g%3A2a84b4e2-b406-ade2-15d7-8116e7e1e48c%7Ce%3Aundefined%7Cc%3A1767226180520%7Cl%3A1771699914016; ab.storage.userId.bf243db8-6e0e-43fa-9ce4-2afc85959789=g%3A949e318ac976c36634e33f704ac94cd3365b8c034a9972a1abae1b9f630ec327%7Ce%3Aundefined%7Cc%3A1767237605715%7Cl%3A1771699914016; _clck=179ykku%5E2%5Eg3r%5E0%5E2192; _clsk=5syzob%5E1771699914858%5E1%5E1%5Ei.clarity.ms%2Fcollect; _li_dcdm_c=.barchart.com; _lc2_fpi=0963eb871108--01jt53cm0rk37mx9cy910nk3h2; _lc2_fpi_meta=%7B%22w%22%3A1752866470559%7D; AMZN-Token=v2FweLxPZ3lQVzBBY0NnSWY3d0J1OHFpcGZMSjBGZkpkVk5obTNYOS94RHpnaFdYY05oTGYzVG94ZGJhRE10em40NTd1b0gzNWNZSFMweGQ3dENFQWpHY1RmYmZWNFVhTnR1Q2R4SzliWUREaDBVYmMwOEZ3RlI5dHJkMlRlR2VnUU5UYVAwVVF2QThFNHJDUE53V2JnMFRCeFk5ZE5EL1c5ejM0STJBUkdjcjZMd0piZVBZT3lJRUNJZVRxZm5zPWJrdgFiaXZ4IDc3KzlJM0x2djcwZjc3Kzk3Nys5WDNYdnY3MFY3Nys5/w==; _lr_geo_location_state=ON; _lr_geo_location=CA; _lc2_fpi_js=0963eb871108--01jt53cm0rk37mx9cy910nk3h2; laravel_token=eyJpdiI6IkdWRnBiUVRGQnc3Zm5DSnpncTErYUE9PSIsInZhbHVlIjoiTHF4VUpuNmFFTXAzNVVnbkxpdG1pN1B0OTh1WVE1MHlUZ3N4ZW82V2xCYkhxRmFwdUxNVVNYVm9CNHlabUt3U0ptQjZzQy9lMFUvZCs0L0VSci9reW43UGVjWVJnVStxei9takdpUTU1bEZJdk1PY0FLdVpDWmF4M21LcGNoTFp3dzBkOHVQaks3Tlh5M0NVRU9RQ0NKc0V0OThlaDhhbDZ5U2dQTGhxUExveHNjQ0pYazBVOEdwMXh6NGp3NXQxSnFkSHdqZ2RjYXVFUWh3S0NRQVlGSDJpSUhMUW44cG5BTk5mR0Z5MUVxNjlwRU5JbUJMRE82SjU4bjFaYW1PWXEvbFpCRmUyRk5OVzJqV2lVNW9rMkpLejROS2JaRHM3dS9jR3NpYlhDRTFqR0ZVN2VJTHIvWFl3UTNFYjZWUE8iLCJtYWMiOiJiOWU5YTQxNmIyNWU2ZDJlZDNmMmI3MGI2YzA5ZDAzMzVhOWU0MzZiN2VlYzY4YjNiZjE4M2IwOGFmNGE5ZTkzIiwidGFnIjoiIn0%3D; XSRF-TOKEN=eyJpdiI6IlZ0V09DZUpvQXlORytPcU5DTEZRWkE9PSIsInZhbHVlIjoiaXN4VkQyVTR0bzc1M0JsQ3pNdi9VUDBGNXFxY0Y3WWFGVlE1U3d0L0NyU2VIUldwQ2V5U25iZGxtSzI0UkpHcGloZkJURGhjS3p0Q2ZtWUtXMzNqSWR5YlBZYXZBbzlkd1kwWWM3cVVveGJRZGhrTCtORkthdmF5UlVqMm8waEwiLCJtYWMiOiJhNzg2ZTRiODBmNGMyOTYxODFjOTU4MTliOWQxZWQ4MjA4YTg0YmM2OGQ0OTlmNTkzOTBlNjEwNmZlZTI4NDI3IiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6Ii9mREFyTHBwZ0xacDVLaDM5SEVSdmc9PSIsInZhbHVlIjoiOEpRTGhtc2srZHZOSHhKd3lqaHBXWEh5OTdTTytjN3NiQUc3Q2x6aGoxem5ic0tOZi9YaHlZQ2QybjJFaWo5YnJmUHBuR1lkQ2gwalFqdHd0bGpOV1JpQnNXZ3prc0IvdUNIczJWb1V5R3l3cUZaSXovWmdidGdBWitJRGZqWU8iLCJtYWMiOiI3NDU4NGU0YzcwOGMxNGRmNGEwMzA4YTlmMTY4YTIyYzFmMDJiNGRlY2Y1Nzc4Mjc5M2JkNTY5MzlmNDE1N2JmIiwidGFnIjoiIn0%3D; bcFreeUserPageView=3; _tfpvi=ZDM4ZGUzMGQtZDJhOS00YzExLThhOTktZjZkOTFhMjI5MmUxIy03LTE%3D; __gads=ID=a49ca376d9eb13e9:T=1767226687:RT=1771699923:S=ALNI_Mbs8ZFsqG2JCg_YSgqRkwMAtbUG5Q; __gpi=UID=000012e147357f01:T=1767226687:RT=1771699923:S=ALNI_MakPQLjDrNaeTIQBh66oqXXyvtn-g; __eoi=ID=d687990116d329fb:T=1767226687:RT=1771699923:S=AA-AfjZjxsFgDf-dwfbrs8jvlrWA; ab.storage.sessionId.bf243db8-6e0e-43fa-9ce4-2afc85959789=g%3A5fce5db9-252e-f447-0b9e-ac6c007437b9%7Ce%3A1771701726949%7Cc%3A1771699914015%7Cl%3A1771699926949; _ga_PE0FK9V6VN=GS2.1.s1771699914$o11$g1$t1771699954$j20$l0$h0' \
  -H 'priority: u=1, i' \
  -H 'referer: https://www.barchart.com/stocks/quotes/AAPL/options?expiration=2026-02-23-w&view=sbs' \
  -H 'sec-ch-ua: "Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-origin' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36' \
  -H 'x-xsrf-token: eyJpdiI6IlZ0V09DZUpvQXlORytPcU5DTEZRWkE9PSIsInZhbHVlIjoiaXN4VkQyVTR0bzc1M0JsQ3pNdi9VUDBGNXFxY0Y3WWFGVlE1U3d0L0NyU2VIUldwQ2V5U25iZGxtSzI0UkpHcGloZkJURGhjS3p0Q2ZtWUtXMzNqSWR5YlBZYXZBbzlkd1kwWWM3cVVveGJRZGhrTCtORkthdmF5UlVqMm8waEwiLCJtYWMiOiJhNzg2ZTRiODBmNGMyOTYxODFjOTU4MTliOWQxZWQ4MjA4YTg0YmM2OGQ0OTlmNTkzOTBlNjEwNmZlZTI4NDI3IiwidGFnIjoiIn0='"""

def parse_cookie_str(cookie_str: str) -> dict:
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)  # split ONLY on first "="
        cookies[k.strip()] = v.strip()
    return cookies

def main():
    cookies = parse_cookie_str(COOKIE_STR)

    # Sanity check: confirm we actually found it
    print("Has XSRF-TOKEN:", "XSRF-TOKEN" in cookies)
    print("Has laravel_session:", "laravel_session" in cookies)

    xsrf_cookie = cookies.get("XSRF-TOKEN")
    xsrf_value = urllib.parse.unquote(xsrf_cookie) if xsrf_cookie else None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.barchart.com/stocks/quotes/AAPL/options",
        "Origin": "https://www.barchart.com",
        "Cookie": COOKIE_STR,
    }

    if xsrf_value:
        headers["X-XSRF-TOKEN"] = xsrf_value

    with requests.Session() as s:
        r = s.get(BASE_URL, params=params, headers=headers, timeout=30)

        print("Status:", r.status_code)
        print("Sent X-XSRF-TOKEN:", "yes" if "X-XSRF-TOKEN" in headers else "no")

        r.raise_for_status()  # will stop if not 200

        data = r.json()

        print("\n===== FULL RESPONSE =====\n")
        print(json.dumps(data, indent=2))  # pretty print full JSON

if __name__ == "__main__":
    main()