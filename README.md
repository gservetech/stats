# 📈 Yahoo Finance Dashboard

A real-time stock data dashboard built with Streamlit, featuring live stock prices, options chain analysis, and interactive visualizations.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## ✨ Features

- **Real-Time Stock Data**: Live stock prices, changes, and key metrics
- **Auto-Refresh**: Data updates automatically every minute
- **Interactive Charts**: Candlestick charts, volume analysis, and options visualizations
- **Options Chain**: Full calls/puts data with multiple expiration dates
- **Options Analysis**: Open interest charts and implied volatility smile
- **Data Export**: Download straddle data as CSV
- **Premium UI**: Dark theme with glassmorphism effects

## 🚀 Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/yahoo-finance-dashboard.git
cd yahoo-finance-dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **yfinance** - Yahoo Finance data API
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **BeautifulSoup** - Web scraping (optional features)

## 📊 Dashboard Sections

| Tab                 | Description                             |
| ------------------- | --------------------------------------- |
| 📈 Price Chart      | Real-time candlestick chart with volume |
| 🎯 Options Chain    | Full calls and puts data tables         |
| 📊 Options Analysis | Open interest and volatility charts     |
| 📋 Data Tables      | Complete data with CSV export           |

## 🔧 Configuration

The dashboard uses a custom dark theme. You can modify the theme in `.streamlit/config.toml`.

## ⚠️ Disclaimer

This dashboard is for informational purposes only. Data is provided by Yahoo Finance via the yfinance library. Always verify data before making investment decisions.

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
