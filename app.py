"""
Yahoo Finance Stock Dashboard - Streamlit Application
Real-time stock data scraping and visualization dashboard.

Features:
- Real-time stock price tracking
- Options straddle data display
- Interactive charts and visualizations
- Auto-refresh every minute
- Historical data tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# Page configuration
st.set_page_config(
    page_title="Yahoo Finance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .dashboard-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .dashboard-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e3f 0%, #2d2d5a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .positive {
        color: #00d4aa !important;
    }
    
    .negative {
        color: #ff6b6b !important;
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(30, 30, 63, 0.8) !important;
        border-radius: 10px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Status indicators */
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 1rem;
        background: rgba(0, 212, 170, 0.2);
        border: 1px solid #00d4aa;
        border-radius: 20px;
        color: #00d4aa;
        font-weight: 600;
    }
    
    .pulse {
        width: 10px;
        height: 10px;
        background: #00d4aa;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 212, 170, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 212, 170, 0); }
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(30, 30, 63, 0.6);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 63, 0.6);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'stock_data_history' not in st.session_state:
    st.session_state.stock_data_history = {}
if 'options_data' not in st.session_state:
    st.session_state.options_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'ticker_list' not in st.session_state:
    st.session_state.ticker_list = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']


def get_stock_data(ticker: str) -> dict:
    """Fetch real-time stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d", interval="1m")
        
        # Get current price and changes
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        prev_close = info.get('previousClose', current_price)
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        return {
            'ticker': ticker,
            'name': info.get('shortName', ticker),
            'price': current_price,
            'change': change,
            'change_pct': change_pct,
            'open': info.get('open', 0),
            'high': info.get('dayHigh', 0),
            'low': info.get('dayLow', 0),
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'week_52_high': info.get('fiftyTwoWeekHigh', 0),
            'week_52_low': info.get('fiftyTwoWeekLow', 0),
            'history': hist,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None


def get_options_data(ticker: str) -> tuple:
    """Fetch options chain data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations:
            return None, None, []
        
        # Get the nearest expiration
        nearest_exp = expirations[0]
        opt = stock.option_chain(nearest_exp)
        
        calls_df = opt.calls.copy()
        puts_df = opt.puts.copy()
        
        return calls_df, puts_df, expirations
    except Exception as e:
        st.error(f"Error fetching options for {ticker}: {e}")
        return None, None, []


def create_price_chart(data: dict, ticker: str) -> go.Figure:
    """Create an interactive price chart with volume."""
    if data is None or data.get('history') is None or len(data['history']) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    hist = data['history']
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price',
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff6b6b'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#00d4aa' if close >= open_ else '#ff6b6b' 
              for close, open_ in zip(hist['Close'], hist['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} - Real-Time Price Chart',
        template='plotly_dark',
        paper_bgcolor='rgba(30, 30, 63, 0.8)',
        plot_bgcolor='rgba(30, 30, 63, 0.8)',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False,
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_options_chart(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create options visualization chart."""
    if calls_df is None or puts_df is None:
        fig = go.Figure()
        fig.add_annotation(text="No options data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Calls Open Interest', 'Puts Open Interest'),
        horizontal_spacing=0.1
    )
    
    # Calls
    fig.add_trace(
        go.Bar(
            x=calls_df['strike'],
            y=calls_df['openInterest'],
            name='Calls OI',
            marker_color='#00d4aa',
            opacity=0.8
        ),
        row=1, col=1
    )
    
    # Puts
    fig.add_trace(
        go.Bar(
            x=puts_df['strike'],
            y=puts_df['openInterest'],
            name='Puts OI',
            marker_color='#ff6b6b',
            opacity=0.8
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'{ticker} Options - Open Interest by Strike',
        template='plotly_dark',
        paper_bgcolor='rgba(30, 30, 63, 0.8)',
        plot_bgcolor='rgba(30, 30, 63, 0.8)',
        font=dict(color='white'),
        height=400,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Strike Price", gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Open Interest", gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_volatility_smile(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create volatility smile chart."""
    if calls_df is None or puts_df is None:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = go.Figure()
    
    # Filter valid IV data
    calls_valid = calls_df[calls_df['impliedVolatility'] > 0]
    puts_valid = puts_df[puts_df['impliedVolatility'] > 0]
    
    if len(calls_valid) > 0:
        fig.add_trace(
            go.Scatter(
                x=calls_valid['strike'],
                y=calls_valid['impliedVolatility'] * 100,
                mode='lines+markers',
                name='Calls IV',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=8)
            )
        )
    
    if len(puts_valid) > 0:
        fig.add_trace(
            go.Scatter(
                x=puts_valid['strike'],
                y=puts_valid['impliedVolatility'] * 100,
                mode='lines+markers',
                name='Puts IV',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8)
            )
        )
    
    fig.update_layout(
        title=f'{ticker} - Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(30, 30, 63, 0.8)',
        plot_bgcolor='rgba(30, 30, 63, 0.8)',
        font=dict(color='white'),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def create_straddle_table(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined straddle view table."""
    if calls_df is None or puts_df is None:
        return pd.DataFrame()
    
    # Merge on strike price
    calls_subset = calls_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
    calls_subset.columns = ['Strike', 'Call Last', 'Call Bid', 'Call Ask', 'Call Vol', 'Call OI', 'Call IV']
    
    puts_subset = puts_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
    puts_subset.columns = ['Strike', 'Put Last', 'Put Bid', 'Put Ask', 'Put Vol', 'Put OI', 'Put IV']
    
    straddle = pd.merge(calls_subset, puts_subset, on='Strike', how='outer')
    straddle = straddle.sort_values('Strike')
    
    # Format IV as percentage
    straddle['Call IV'] = (straddle['Call IV'] * 100).round(2).astype(str) + '%'
    straddle['Put IV'] = (straddle['Put IV'] * 100).round(2).astype(str) + '%'
    
    return straddle


def format_number(num):
    """Format large numbers for display."""
    if num is None or pd.isna(num):
        return "N/A"
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def main():
    # Auto-refresh every 60 seconds (60000 milliseconds)
    count = st_autorefresh(interval=60000, limit=None, key="stock_refresh")
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>📈 Yahoo Finance Dashboard</h1>
        <p>Real-time stock data, options analysis, and market visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Ticker input
        ticker_input = st.text_input(
            "Enter Stock Ticker",
            value="AAPL",
            help="Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        )
        ticker = ticker_input.upper().strip()
        
        st.markdown("---")
        
        # Quick ticker buttons
        st.markdown("### 🔥 Popular Tickers")
        popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']
        
        cols = st.columns(3)
        for i, t in enumerate(popular_tickers):
            with cols[i % 3]:
                if st.button(t, key=f"btn_{t}", use_container_width=True):
                    ticker = t
        
        st.markdown("---")
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Status indicator
        st.markdown("""
        <div class="status-live">
            <div class="pulse"></div>
            <span>LIVE DATA</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**Refresh Count:** {count}")
    
    # Main content
    if ticker:
        # Fetch data
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data = get_stock_data(ticker)
            calls_df, puts_df, expirations = get_options_data(ticker)
        
        if stock_data:
            # Key Metrics Row
            st.markdown("### 📊 Key Metrics")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                change_class = "positive" if stock_data['change'] >= 0 else "negative"
                change_symbol = "▲" if stock_data['change'] >= 0 else "▼"
                st.metric(
                    label="Current Price",
                    value=f"${stock_data['price']:.2f}",
                    delta=f"{change_symbol} {abs(stock_data['change']):.2f} ({abs(stock_data['change_pct']):.2f}%)"
                )
            
            with col2:
                st.metric(label="Open", value=f"${stock_data['open']:.2f}")
            
            with col3:
                st.metric(label="Day High", value=f"${stock_data['high']:.2f}")
            
            with col4:
                st.metric(label="Day Low", value=f"${stock_data['low']:.2f}")
            
            with col5:
                st.metric(label="Volume", value=format_number(stock_data['volume']).replace('$', ''))
            
            with col6:
                st.metric(label="Market Cap", value=format_number(stock_data['market_cap']))
            
            st.markdown("---")
            
            # Charts Section
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🎯 Options Chain", "📊 Options Analysis", "📋 Data Tables"])
            
            with tab1:
                st.markdown("#### Real-Time Price Chart")
                price_chart = create_price_chart(stock_data, ticker)
                st.plotly_chart(price_chart, use_container_width=True)
                
                # Additional price stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📈 **52W High:** ${stock_data['week_52_high']:.2f}")
                with col2:
                    st.info(f"📉 **52W Low:** ${stock_data['week_52_low']:.2f}")
                with col3:
                    pe = stock_data['pe_ratio']
                    pe_display = f"{pe:.2f}" if pe and pe > 0 else "N/A"
                    st.info(f"📊 **P/E Ratio:** {pe_display}")
            
            with tab2:
                st.markdown("#### Options Chain Data")
                
                if expirations:
                    selected_exp = st.selectbox(
                        "Select Expiration Date",
                        expirations,
                        key="exp_select"
                    )
                    
                    # Fetch new options data if expiration changed
                    if selected_exp:
                        stock = yf.Ticker(ticker)
                        opt = stock.option_chain(selected_exp)
                        calls_df = opt.calls
                        puts_df = opt.puts
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 📗 Calls")
                        if calls_df is not None and len(calls_df) > 0:
                            display_calls = calls_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                            display_calls.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'Open Int', 'IV']
                            display_calls['IV'] = (display_calls['IV'] * 100).round(2).astype(str) + '%'
                            st.dataframe(display_calls, use_container_width=True, height=400)
                        else:
                            st.warning("No calls data available")
                    
                    with col2:
                        st.markdown("##### 📕 Puts")
                        if puts_df is not None and len(puts_df) > 0:
                            display_puts = puts_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].copy()
                            display_puts.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'Open Int', 'IV']
                            display_puts['IV'] = (display_puts['IV'] * 100).round(2).astype(str) + '%'
                            st.dataframe(display_puts, use_container_width=True, height=400)
                        else:
                            st.warning("No puts data available")
                else:
                    st.warning(f"No options available for {ticker}")
            
            with tab3:
                st.markdown("#### Options Analysis Charts")
                
                if calls_df is not None and puts_df is not None:
                    # Open Interest Chart
                    oi_chart = create_options_chart(calls_df, puts_df, ticker)
                    st.plotly_chart(oi_chart, use_container_width=True)
                    
                    # Volatility Smile
                    vol_chart = create_volatility_smile(calls_df, puts_df, ticker)
                    st.plotly_chart(vol_chart, use_container_width=True)
                    
                    # Put/Call Ratio
                    col1, col2, col3 = st.columns(3)
                    
                    total_call_oi = calls_df['openInterest'].sum()
                    total_put_oi = puts_df['openInterest'].sum()
                    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                    
                    with col1:
                        st.metric("Total Call OI", f"{total_call_oi:,.0f}")
                    with col2:
                        st.metric("Total Put OI", f"{total_put_oi:,.0f}")
                    with col3:
                        sentiment = "🐻 Bearish" if pc_ratio > 1 else "🐂 Bullish"
                        st.metric("Put/Call Ratio", f"{pc_ratio:.3f} ({sentiment})")
                else:
                    st.warning("Options data not available for analysis")
            
            with tab4:
                st.markdown("#### 📋 Complete Data Tables")
                
                # Stock Info Table
                st.markdown("##### Stock Information")
                stock_info = {
                    'Metric': ['Name', 'Ticker', 'Price', 'Change', 'Change %', 'Open', 'High', 'Low', 
                               'Volume', 'Market Cap', 'P/E Ratio', '52W High', '52W Low'],
                    'Value': [
                        stock_data['name'], ticker, f"${stock_data['price']:.2f}",
                        f"${stock_data['change']:.2f}", f"{stock_data['change_pct']:.2f}%",
                        f"${stock_data['open']:.2f}", f"${stock_data['high']:.2f}",
                        f"${stock_data['low']:.2f}", f"{stock_data['volume']:,.0f}",
                        format_number(stock_data['market_cap']),
                        f"{stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] else "N/A",
                        f"${stock_data['week_52_high']:.2f}", f"${stock_data['week_52_low']:.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stock_info), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Straddle Table
                st.markdown("##### Straddle View (Calls & Puts Combined)")
                straddle_df = create_straddle_table(calls_df, puts_df)
                if len(straddle_df) > 0:
                    st.dataframe(straddle_df, use_container_width=True, height=400)
                    
                    # Download button
                    csv = straddle_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Straddle Data (CSV)",
                        data=csv,
                        file_name=f"{ticker}_straddle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No straddle data available")
                
                st.markdown("---")
                
                # Historical Data
                st.markdown("##### Recent Price History")
                if stock_data['history'] is not None and len(stock_data['history']) > 0:
                    hist_display = stock_data['history'].reset_index()
                    hist_display = hist_display.tail(50)  # Last 50 records
                    st.dataframe(hist_display, use_container_width=True, height=300)
                else:
                    st.warning("No historical data available")
        else:
            st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.5); padding: 1rem;">
        <p>📊 Yahoo Finance Dashboard | Data updates every minute | Built with Streamlit</p>
        <p>⚠️ Data provided by Yahoo Finance via yfinance. For informational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
