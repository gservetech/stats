import requests
import os
import streamlit as st

def safe_cache_data(*dargs, **dkwargs):
    """No-op cache wrapper: always returns the original function."""
    def _decorator(func):
        return func
    return _decorator

def get_api_base_url() -> str:
    try:
        if "API_BASE_URL" in st.secrets:
            return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except Exception:
        pass
    env_url = os.getenv("API_BASE_URL")
    if env_url:
        return env_url.rstrip("/")
    return "http://localhost:8000"

API_BASE_URL = get_api_base_url()

@safe_cache_data(ttl=15, show_spinner=False)
def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

@safe_cache_data(ttl=5, show_spinner=False)
def fetch_spot_quote(symbol: str, date: str):
    try:
        r = requests.get(
            f"{API_BASE_URL}/spot",
            params={"symbol": symbol, "date": date},
            timeout=30,
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        try:
            detail = r.json().get("detail", f"HTTP {r.status_code}")
        except Exception:
            detail = f"HTTP {r.status_code}"
        return {"success": False, "error": detail, "status_code": r.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout calling backend spot endpoint.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}

@safe_cache_data(ttl=300, show_spinner=False)
def fetch_options(symbol: str, date: str):
    try:
        r = requests.get(
            f"{API_BASE_URL}/options",
            params={"symbol": symbol, "date": date},
            timeout=300
        )
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        try:
            detail = r.json().get("detail", f"HTTP {r.status_code}")
        except Exception:
            detail = f"HTTP {r.status_code}"
        return {"success": False, "error": detail, "status_code": r.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout calling backend.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}

@safe_cache_data(ttl=300, show_spinner=False)
def fetch_weekly_summary(symbol: str, date: str, spot: float, r: float = 0.05, q: float = 0.0, multiplier: int = 100):
    try:
        rqs = requests.get(
            f"{API_BASE_URL}/weekly/summary",
            params={"symbol": symbol, "date": date, "spot": spot, "r": r, "q": q, "multiplier": multiplier},
            timeout=300
        )
        if rqs.status_code == 200:
            return {"success": True, "data": rqs.json()}
        try:
            detail = rqs.json().get("detail", f"HTTP {rqs.status_code}")
        except Exception:
            detail = f"HTTP {rqs.status_code}"
        return {"success": False, "error": detail, "status_code": rqs.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout calculating weekly summary.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}

@safe_cache_data(ttl=300, show_spinner=False)
def fetch_weekly_gex(symbol: str, date: str, spot: float, r: float = 0.05, q: float = 0.0, multiplier: int = 100):
    try:
        rqs = requests.get(
            f"{API_BASE_URL}/weekly/gex",
            params={"symbol": symbol, "date": date, "spot": spot, "r": r, "q": q, "multiplier": multiplier},
            timeout=300
        )
        if rqs.status_code == 200:
            return {"success": True, "data": rqs.json()}
        try:
            detail = rqs.json().get("detail", f"HTTP {rqs.status_code}")
        except Exception:
            detail = f"HTTP {rqs.status_code}"
        return {"success": False, "error": detail, "status_code": rqs.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout fetching weekly gex.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}


@safe_cache_data(ttl=300, show_spinner=False)
def fetch_share_statistics(symbol: str):
    try:
        r = requests.get(
            f"{API_BASE_URL}/yahoo/share-statistics",
            params={"symbol": symbol},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get("detail", f"HTTP {r.status_code}")
        except Exception:
            detail = f"HTTP {r.status_code}"
        return {"success": False, "error": detail, "status_code": r.status_code}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout calling backend share statistics endpoint.", "status_code": 408}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend.", "status_code": 503}
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}
