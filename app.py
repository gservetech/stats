"""
Barchart Stock Options Dashboard - Streamlit Application
Real-time options data visualization using Barchart's internal API.

Features:
- Real-time options chain data from Barchart
- Side-by-side Calls/Puts straddle view
- Options volume and open interest charts
- Implied volatility analysis
- Barchart-inspired UI design
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Cache TTL in seconds (5 minutes = 300 seconds)
CACHE_TTL = 300

# Barchart API Configuration
BARCHART_BASE_URL = "https://www.barchart.com"
BARCHART_OPTIONS_API = "https://www.barchart.com/proxies/core-api/v1/options/get"
BARCHART_EXPIRATIONS_API = "https://www.barchart.com/proxies/core-api/v1/options-expirations/get"
BARCHART_QUOTE_API = "https://www.barchart.com/proxies/core-api/v1/quotes/get"

# Browser-like headers (same as scraping script would capture)
BARCHART_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.barchart.com/stocks/quotes/AAPL/options",
    "Origin": "https://www.barchart.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}

# Page configuration
st.set_page_config(
    page_title="Barchart Options Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barchart-inspired CSS styling (green theme)
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container - Barchart dark theme */
    .main {
        background: #1a1d21;
    }
    
    .stApp {
        background: linear-gradient(180deg, #1a1d21 0%, #0d0f11 100%);
    }
    
    /* Barchart Header styling */
    .barchart-header {
        background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
        padding: 1.5rem 2rem;
        border-radius: 0;
        margin: -1rem -1rem 1.5rem -1rem;
        box-shadow: 0 4px 20px rgba(0, 135, 90, 0.3);
    }
    
    .barchart-header h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .barchart-header p {
        color: rgba(255,255,255,0.85);
        margin-top: 0.3rem;
        font-size: 0.95rem;
    }
    
    /* Ticker display box */
    .ticker-box {
        background: linear-gradient(145deg, #252a30 0%, #1e2328 100%);
        border: 1px solid #3d4450;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .ticker-symbol {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d775;
    }
    
    .ticker-name {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 4px;
    }
    
    .ticker-price {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
    }
    
    .price-up {
        color: #00d775 !important;
    }
    
    .price-down {
        color: #ff4757 !important;
    }
    
    /* Metric cards - Barchart style */
    .metric-card {
        background: linear-gradient(145deg, #252a30 0%, #1e2328 100%);
        border: 1px solid #3d4450;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: white;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Options table styling */
    .options-table {
        background: #1e2328;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Call options - Green theme */
    .call-header {
        background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
        color: white;
        padding: 0.8rem;
        font-weight: 600;
        text-align: center;
    }
    
    /* Put options - Red theme */
    .put-header {
        background: linear-gradient(90deg, #dc3545 0%, #ff4757 100%);
        color: white;
        padding: 0.8rem;
        font-weight: 600;
        text-align: center;
    }
    
    /* Strike price column */
    .strike-header {
        background: #3d4450;
        color: white;
        padding: 0.8rem;
        font-weight: 600;
        text-align: center;
    }
    
    /* Status indicator */
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.4rem 0.8rem;
        background: rgba(0, 215, 117, 0.15);
        border: 1px solid #00d775;
        border-radius: 15px;
        color: #00d775;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .pulse {
        width: 8px;
        height: 8px;
        background: #00d775;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 215, 117, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(0, 215, 117, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 215, 117, 0); }
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2328 0%, #15181c 100%);
        border-right: 1px solid #3d4450;
    }
    
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: #252a30;
        border: 1px solid #3d4450;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00875a 0%, #00a86b 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 135, 90, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #252a30;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9ca3af;
        border-radius: 0;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #00875a;
        color: white;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #252a30;
        border-radius: 8px;
    }
    
    /* Info box styling */
    .info-box {
        background: rgba(0, 135, 90, 0.1);
        border: 1px solid #00875a;
        border-radius: 8px;
        padding: 1rem;
        color: #00d775;
    }
    
    .warning-box {
        background: rgba(255, 71, 87, 0.1);
        border: 1px solid #ff4757;
        border-radius: 8px;
        padding: 1rem;
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


# ========== HELPER FUNCTIONS (from scraping script) ==========

def _to_float(val, default=None):
    """Convert value to float, handling various formats."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ("", "N/A", "na", "None", "-"):
        return default
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in ("", "-", "."):
        return default
    try:
        return float(s)
    except:
        return default


def _to_int(val, default=0):
    """Convert value to int."""
    f = _to_float(val, None)
    return int(round(f)) if f is not None else default


def _fmt_price(x):
    """Format price for display."""
    if x is None:
        return ""
    return f"{x:,.2f}"


def _fmt_int(x):
    """Format integer for display."""
    if x is None:
        return ""
    return f"{int(x):,}"


def _fmt_iv(val):
    """Format implied volatility."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float)):
        return f"{val * 100:.2f}%" if val <= 10 else f"{val:.2f}%"
    return str(val)


def _pick(option_obj):
    """Extract and format option data (same as scraping script)."""
    if not option_obj:
        return {k: "" for k in ["Latest", "Bid", "Ask", "Change", "Volume", "Open Int", "IV", "Last Trade"]}
    
    raw = option_obj.get("raw") or {}
    latest = _to_float(option_obj.get("lastPrice"), raw.get("lastPrice"))
    bid = _to_float(option_obj.get("bidPrice"), raw.get("bidPrice"))
    ask = _to_float(option_obj.get("askPrice"), raw.get("askPrice"))
    volume = _to_int(option_obj.get("volume"), raw.get("volume"))
    oi = _to_int(option_obj.get("openInterest"), raw.get("openInterest"))
    iv_val = option_obj.get("volatility") or raw.get("volatility")
    
    return {
        "Latest": _fmt_price(latest),
        "Bid": _fmt_price(bid),
        "Ask": _fmt_price(ask),
        "Change": str(option_obj.get("priceChange") or ""),
        "Volume": _fmt_int(volume),
        "Open Int": _fmt_int(oi),
        "IV": _fmt_iv(iv_val),
        "Last Trade": str(option_obj.get("tradeTime") or ""),
        "raw_latest": latest,
        "raw_volume": volume,
        "raw_oi": oi,
        "raw_iv": _to_float(iv_val, 0)
    }


# ========== BARCHART API FUNCTIONS ==========

def get_barchart_session(ticker: str) -> requests.Session:
    """Create a session with cookies by visiting Barchart page first."""
    session = requests.Session()
    session.headers.update(BARCHART_HEADERS)
    
    try:
        # Visit the options page to get cookies (like the browser would)
        page_url = f"{BARCHART_BASE_URL}/stocks/quotes/{ticker}/options"
        session.headers['Referer'] = page_url
        response = session.get(page_url, timeout=15)
        
        if response.status_code == 200:
            # Extract XSRF token from cookies if present
            if 'XSRF-TOKEN' in session.cookies:
                token = session.cookies['XSRF-TOKEN']
                session.headers['X-XSRF-TOKEN'] = token
    except Exception as e:
        pass  # Continue without cookies
    
    return session


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_barchart_expirations(ticker: str) -> list:
    """Fetch available expiration dates from Barchart API."""
    try:
        session = get_barchart_session(ticker)
        
        params = {
            "symbol": ticker,
            "fields": "expirationDate,optionsCount,callsVolume,putsVolume,callsOpenInterest,putsOpenInterest"
        }
        
        response = session.get(BARCHART_EXPIRATIONS_API, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            expirations = []
            exp_details = []
            
            if data.get("data"):
                for exp in data["data"]:
                    exp_date = exp.get("expirationDate")
                    if exp_date:
                        expirations.append(exp_date)
                        exp_details.append({
                            "date": exp_date,
                            "optionsCount": exp.get("optionsCount", 0),
                            "callsVolume": _to_int(exp.get("callsVolume", 0)),
                            "putsVolume": _to_int(exp.get("putsVolume", 0)),
                            "callsOI": _to_int(exp.get("callsOpenInterest", 0)),
                            "putsOI": _to_int(exp.get("putsOpenInterest", 0)),
                        })
            
            return expirations, exp_details
    except Exception as e:
        st.warning(f"Could not fetch expirations: {e}")
    
    return [], []


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_barchart_options(ticker: str, expiration: str = None) -> dict:
    """
    Fetch options data from Barchart API.
    This mimics the same API call that the browser makes.
    """
    try:
        session = get_barchart_session(ticker)
        
        # Build params similar to what browser sends
        params = {
            "symbol": ticker,
            "fields": "strikePrice,optionType,lastPrice,bidPrice,askPrice,priceChange,percentChange,volume,openInterest,volatility,tradeTime,daysToExpiration,symbolCode",
            "groupBy": "optionType",
            "raw": "1",
            "meta": "field.shortName,field.type,field.description"
        }
        
        if expiration:
            params["expirationDate"] = expiration
        
        response = session.get(BARCHART_OPTIONS_API, params=params, timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Barchart API returned status {response.status_code}")
    except Exception as e:
        st.warning(f"Error fetching options: {e}")
    
    return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_barchart_quote(ticker: str) -> dict:
    """Fetch stock quote from Barchart."""
    try:
        session = get_barchart_session(ticker)
        
        params = {
            "symbol": ticker,
            "fields": "symbol,symbolName,lastPrice,priceChange,percentChange,open,high,low,previousClose,volume,tradeTime,averageVolume,fiftyTwoWkHigh,fiftyTwoWkLow"
        }
        
        response = session.get(BARCHART_QUOTE_API, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                quote = data["data"][0] if isinstance(data["data"], list) else data["data"]
                raw = quote.get("raw", {})
                
                return {
                    'symbol': ticker,
                    'name': quote.get('symbolName', ticker),
                    'lastPrice': _to_float(quote.get('lastPrice'), raw.get('lastPrice', 0)),
                    'priceChange': _to_float(quote.get('priceChange'), raw.get('priceChange', 0)),
                    'percentChange': _to_float(quote.get('percentChange'), raw.get('percentChange', 0)),
                    'open': _to_float(quote.get('open'), raw.get('open', 0)),
                    'high': _to_float(quote.get('high'), raw.get('high', 0)),
                    'low': _to_float(quote.get('low'), raw.get('low', 0)),
                    'previousClose': _to_float(quote.get('previousClose'), raw.get('previousClose', 0)),
                    'volume': _to_int(quote.get('volume'), raw.get('volume', 0)),
                    'avgVolume': _to_int(quote.get('averageVolume'), raw.get('averageVolume', 0)),
                    'week52High': _to_float(quote.get('fiftyTwoWkHigh'), raw.get('fiftyTwoWkHigh', 0)),
                    'week52Low': _to_float(quote.get('fiftyTwoWkLow'), raw.get('fiftyTwoWkLow', 0)),
                    'tradeTime': quote.get('tradeTime', ''),
                }
    except Exception as e:
        st.warning(f"Error fetching quote: {e}")
    
    return None


def process_options_data(options_json: dict) -> tuple:
    """
    Process Barchart options JSON into straddle format (same as scraping script).
    Returns calls_df, puts_df, straddle_df
    """
    if not options_json or not options_json.get("data"):
        return None, None, None
    
    data = options_json.get("data", {})
    rows = []
    calls_list = []
    puts_list = []
    
    # Handle different data structures from Barchart
    strike_items = {}
    
    if isinstance(data, dict):
        if "Call" in data or "Put" in data:
            # Standard grouped format
            for opt_type in ["Call", "Put"]:
                for item in data.get(opt_type, []):
                    strike = item.get("strikePrice")
                    if strike not in strike_items:
                        strike_items[strike] = []
                    strike_items[strike].append(item)
        else:
            # SBS format - keys are strike prices
            strike_items = data
    elif isinstance(data, list):
        # Flat list format
        for item in data:
            strike = item.get("strikePrice")
            if strike not in strike_items:
                strike_items[strike] = []
            strike_items[strike].append(item)
    
    # Build straddle rows
    for strike_str, items in strike_items.items():
        if not isinstance(items, list):
            items = [items]
        
        call_obj = next((i for i in items if i.get("optionType") == "Call"), None)
        put_obj = next((i for i in items if i.get("optionType") == "Put"), None)
        
        c_data = _pick(call_obj)
        p_data = _pick(put_obj)
        strike_num = _to_float(strike_str, 0)
        
        # Straddle row
        row = {
            "Call Latest": c_data["Latest"],
            "Call Bid": c_data["Bid"],
            "Call Ask": c_data["Ask"],
            "Call Change": c_data["Change"],
            "Call Volume": c_data["Volume"],
            "Call OI": c_data["Open Int"],
            "Call IV": c_data["IV"],
            "Strike": strike_num,
            "Put Latest": p_data["Latest"],
            "Put Bid": p_data["Bid"],
            "Put Ask": p_data["Ask"],
            "Put Change": p_data["Change"],
            "Put Volume": p_data["Volume"],
            "Put OI": p_data["Open Int"],
            "Put IV": p_data["IV"],
        }
        rows.append((strike_num, row))
        
        # Individual calls/puts for charts
        if call_obj:
            calls_list.append({
                'strike': strike_num,
                'lastPrice': c_data['raw_latest'],
                'volume': c_data['raw_volume'],
                'openInterest': c_data['raw_oi'],
                'iv': c_data['raw_iv'],
            })
        if put_obj:
            puts_list.append({
                'strike': strike_num,
                'lastPrice': p_data['raw_latest'],
                'volume': p_data['raw_volume'],
                'openInterest': p_data['raw_oi'],
                'iv': p_data['raw_iv'],
            })
    
    # Sort by strike and create DataFrames
    rows.sort(key=lambda x: x[0])
    straddle_df = pd.DataFrame([r for _, r in rows])
    
    calls_df = pd.DataFrame(calls_list).sort_values('strike') if calls_list else None
    puts_df = pd.DataFrame(puts_list).sort_values('strike') if puts_list else None
    
    return calls_df, puts_df, straddle_df


# ========== CHART FUNCTIONS ==========

def create_oi_chart(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create Open Interest chart (Barchart style)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('📈 Calls Open Interest', '📉 Puts Open Interest'),
        horizontal_spacing=0.1
    )
    
    if calls_df is not None and len(calls_df) > 0:
        fig.add_trace(
            go.Bar(
                x=calls_df['strike'],
                y=calls_df['openInterest'],
                name='Calls OI',
                marker_color='#00d775',
                opacity=0.85
            ),
            row=1, col=1
        )
    
    if puts_df is not None and len(puts_df) > 0:
        fig.add_trace(
            go.Bar(
                x=puts_df['strike'],
                y=puts_df['openInterest'],
                name='Puts OI',
                marker_color='#ff4757',
                opacity=0.85
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f'{ticker} Options - Open Interest by Strike',
        template='plotly_dark',
        paper_bgcolor='#1e2328',
        plot_bgcolor='#1e2328',
        font=dict(color='white', family='Inter'),
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Strike Price", gridcolor='#3d4450')
    fig.update_yaxes(title_text="Open Interest", gridcolor='#3d4450')
    
    return fig


def create_volume_chart(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create Volume chart (Barchart style)."""
    fig = go.Figure()
    
    if calls_df is not None and len(calls_df) > 0:
        fig.add_trace(
            go.Bar(
                x=calls_df['strike'],
                y=calls_df['volume'],
                name='Calls Volume',
                marker_color='#00d775',
                opacity=0.8
            )
        )
    
    if puts_df is not None and len(puts_df) > 0:
        fig.add_trace(
            go.Bar(
                x=puts_df['strike'],
                y=puts_df['volume'],
                name='Puts Volume',
                marker_color='#ff4757',
                opacity=0.8
            )
        )
    
    fig.update_layout(
        title=f'{ticker} Options Volume by Strike',
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='#1e2328',
        plot_bgcolor='#1e2328',
        font=dict(color='white', family='Inter'),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="Strike Price", gridcolor='#3d4450')
    fig.update_yaxes(title_text="Volume", gridcolor='#3d4450')
    
    return fig


def create_iv_smile_chart(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create Implied Volatility Smile chart."""
    fig = go.Figure()
    
    if calls_df is not None and len(calls_df) > 0:
        valid_calls = calls_df[calls_df['iv'] > 0]
        if len(valid_calls) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_calls['strike'],
                    y=valid_calls['iv'] * 100 if valid_calls['iv'].max() <= 10 else valid_calls['iv'],
                    mode='lines+markers',
                    name='Calls IV',
                    line=dict(color='#00d775', width=3),
                    marker=dict(size=8)
                )
            )
    
    if puts_df is not None and len(puts_df) > 0:
        valid_puts = puts_df[puts_df['iv'] > 0]
        if len(valid_puts) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_puts['strike'],
                    y=valid_puts['iv'] * 100 if valid_puts['iv'].max() <= 10 else valid_puts['iv'],
                    mode='lines+markers',
                    name='Puts IV',
                    line=dict(color='#ff4757', width=3),
                    marker=dict(size=8)
                )
            )
    
    fig.update_layout(
        title=f'{ticker} Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        template='plotly_dark',
        paper_bgcolor='#1e2328',
        plot_bgcolor='#1e2328',
        font=dict(color='white', family='Inter'),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(gridcolor='#3d4450')
    fig.update_yaxes(gridcolor='#3d4450')
    
    return fig


def create_put_call_ratio_gauge(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> go.Figure:
    """Create Put/Call ratio gauge chart."""
    if calls_df is None or puts_df is None:
        return None
    
    total_call_oi = calls_df['openInterest'].sum() if len(calls_df) > 0 else 0
    total_put_oi = puts_df['openInterest'].sum() if len(puts_df) > 0 else 0
    
    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pc_ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Put/Call Ratio", 'font': {'size': 18, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00d775" if pc_ratio < 1 else "#ff4757"},
            'bgcolor': "#3d4450",
            'borderwidth': 2,
            'bordercolor': "#3d4450",
            'steps': [
                {'range': [0, 0.7], 'color': 'rgba(0, 215, 117, 0.3)'},
                {'range': [0.7, 1.3], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [1.3, 2], 'color': 'rgba(255, 71, 87, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 1
            }
        },
        number={'font': {'color': 'white'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='#1e2328',
        font={'color': "white", 'family': 'Inter'},
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


# ========== MAIN APPLICATION ==========

def main():
    # Auto-refresh every 5 minutes
    count = st_autorefresh(interval=300000, limit=None, key="barchart_refresh")
    
    # Header
    st.markdown("""
    <div class="barchart-header">
        <h1>📊 Barchart Options Dashboard</h1>
        <p>Real-time options chain data and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔍 Symbol Lookup")
        
        ticker_input = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            help="Enter any US stock ticker symbol"
        )
        ticker = ticker_input.upper().strip()
        
        st.markdown("---")
        
        st.markdown("### 🔥 Popular Symbols")
        popular = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'SPY', 'QQQ']
        
        cols = st.columns(3)
        for i, t in enumerate(popular):
            with cols[i % 3]:
                if st.button(t, key=f"pop_{t}", use_container_width=True):
                    ticker = t
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Status
        st.markdown("""
        <div class="status-live">
            <div class="pulse"></div>
            <span>LIVE DATA</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**Data Source:** Barchart.com")
    
    # Main content
    if ticker:
        # Fetch quote data
        with st.spinner(f"Loading {ticker} data from Barchart..."):
            quote = fetch_barchart_quote(ticker)
            expirations, exp_details = fetch_barchart_expirations(ticker)
        
        if quote:
            # Quote display
            col1, col2, col3 = st.columns([2, 2, 3])
            
            with col1:
                price_class = "price-up" if quote['priceChange'] >= 0 else "price-down"
                change_symbol = "▲" if quote['priceChange'] >= 0 else "▼"
                
                st.markdown(f"""
                <div class="ticker-box">
                    <div class="ticker-symbol">{ticker}</div>
                    <div class="ticker-name">{quote['name']}</div>
                    <div class="ticker-price {price_class}">${quote['lastPrice']:,.2f}</div>
                    <div class="{price_class}">{change_symbol} {abs(quote['priceChange']):,.2f} ({abs(quote['percentChange']):,.2f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("Open", f"${quote['open']:,.2f}")
                st.metric("High", f"${quote['high']:,.2f}")
                st.metric("Low", f"${quote['low']:,.2f}")
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            with col3:
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("Volume", f"{quote['volume']:,}")
                with mcol2:
                    st.metric("52W High", f"${quote['week52High']:,.2f}")
                with mcol3:
                    st.metric("52W Low", f"${quote['week52Low']:,.2f}")
        
        st.markdown("---")
        
        # Options section
        if expirations:
            # Expiration selector
            st.markdown("### 📅 Expiration Date")
            selected_exp = st.selectbox(
                "Select expiration",
                expirations,
                format_func=lambda x: x,
                label_visibility="collapsed"
            )
            
            # Fetch options for selected expiration
            with st.spinner(f"Loading options chain for {selected_exp}..."):
                options_json = fetch_barchart_options(ticker, selected_exp)
            
            if options_json:
                calls_df, puts_df, straddle_df = process_options_data(options_json)
                
                # Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Options Chain",
                    "📊 Open Interest",
                    "📈 Volume & IV",
                    "🎯 Analysis"
                ])
                
                with tab1:
                    st.markdown("#### Side-by-Side Options Chain (Straddle View)")
                    
                    if straddle_df is not None and len(straddle_df) > 0:
                        # Create colored headers
                        st.markdown("""
                        <div style="display: flex; margin-bottom: 0;">
                            <div style="flex: 1; background: linear-gradient(90deg, #00875a, #00a86b); color: white; padding: 10px; text-align: center; font-weight: 600; border-radius: 8px 0 0 0;">
                                📈 CALLS
                            </div>
                            <div style="flex: 0 0 80px; background: #3d4450; color: white; padding: 10px; text-align: center; font-weight: 600;">
                                STRIKE
                            </div>
                            <div style="flex: 1; background: linear-gradient(90deg, #dc3545, #ff4757); color: white; padding: 10px; text-align: center; font-weight: 600; border-radius: 0 8px 0 0;">
                                📉 PUTS
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display the straddle table
                        st.dataframe(
                            straddle_df,
                            use_container_width=True,
                            height=500,
                            hide_index=True
                        )
                        
                        # Download button
                        csv = straddle_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Options Chain (CSV)",
                            data=csv,
                            file_name=f"{ticker}_options_{selected_exp}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No options data available for this expiration")
                
                with tab2:
                    if calls_df is not None and puts_df is not None:
                        oi_chart = create_oi_chart(calls_df, puts_df, ticker)
                        st.plotly_chart(oi_chart, use_container_width=True)
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        total_call_oi = calls_df['openInterest'].sum() if len(calls_df) > 0 else 0
                        total_put_oi = puts_df['openInterest'].sum() if len(puts_df) > 0 else 0
                        
                        with col1:
                            st.metric("Total Calls OI", f"{total_call_oi:,}")
                        with col2:
                            st.metric("Total Puts OI", f"{total_put_oi:,}")
                        with col3:
                            pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                            st.metric("P/C Ratio", f"{pc_ratio:.3f}")
                        with col4:
                            sentiment = "🐂 Bullish" if pc_ratio < 1 else "🐻 Bearish"
                            st.metric("Sentiment", sentiment)
                    else:
                        st.warning("No data available for charts")
                
                with tab3:
                    if calls_df is not None and puts_df is not None:
                        # Volume chart
                        vol_chart = create_volume_chart(calls_df, puts_df, ticker)
                        st.plotly_chart(vol_chart, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # IV Smile chart
                        iv_chart = create_iv_smile_chart(calls_df, puts_df, ticker)
                        st.plotly_chart(iv_chart, use_container_width=True)
                    else:
                        st.warning("No data available for charts")
                
                with tab4:
                    if calls_df is not None and puts_df is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # P/C Ratio gauge
                            gauge_chart = create_put_call_ratio_gauge(calls_df, puts_df)
                            if gauge_chart:
                                st.plotly_chart(gauge_chart, use_container_width=True)
                        
                        with col2:
                            # Top strikes by OI
                            st.markdown("#### 🎯 Top Strikes by Open Interest")
                            
                            if len(calls_df) > 0:
                                top_call_strikes = calls_df.nlargest(5, 'openInterest')[['strike', 'openInterest']]
                                top_call_strikes.columns = ['Strike', 'Call OI']
                                
                            if len(puts_df) > 0:
                                top_put_strikes = puts_df.nlargest(5, 'openInterest')[['strike', 'openInterest']]
                                top_put_strikes.columns = ['Strike', 'Put OI']
                            
                            st.markdown("**Top Call Strikes:**")
                            st.dataframe(top_call_strikes, use_container_width=True, hide_index=True)
                            
                            st.markdown("**Top Put Strikes:**")
                            st.dataframe(top_put_strikes, use_container_width=True, hide_index=True)
                        
                        # Expiration summary
                        if exp_details:
                            st.markdown("---")
                            st.markdown("#### 📅 All Expirations Summary")
                            exp_df = pd.DataFrame(exp_details)
                            exp_df.columns = ['Expiration', 'Options Count', 'Calls Volume', 'Puts Volume', 'Calls OI', 'Puts OI']
                            st.dataframe(exp_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No data available for analysis")
            else:
                st.error("Could not fetch options data. Barchart may be rate limiting requests.")
        else:
            st.warning(f"No options available for {ticker} or unable to connect to Barchart.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.5); padding: 1rem;">
        <p>📊 Barchart Options Dashboard | Data from Barchart.com | Built with Streamlit</p>
        <p style="font-size: 0.8rem;">⚠️ For informational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
