import warnings
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore")

# -----------------------------
# 1) Download data
# -----------------------------
tlt = (
    vbt.YFData.download("AAPL", start="2025-01-01", end="2025-12-01")
    .get("Close")
    .to_frame()
)
close = tlt["Close"]

# Make index timezone-naive safely (vectorbt/yfinance can return tz-aware)
idx = tlt.index
if getattr(idx, "tz", None) is not None:
    idx_naive = idx.tz_convert(None)
else:
    idx_naive = idx

# -----------------------------
# 2) Build signal frames
# -----------------------------
short_entries = pd.DataFrame.vbt.signals.empty_like(close)
short_exits   = pd.DataFrame.vbt.signals.empty_like(close)
long_entries  = pd.DataFrame.vbt.signals.empty_like(close)
long_exits    = pd.DataFrame.vbt.signals.empty_like(close)

# First trading day of each month (True at the first row of each month)
first_of_month = ~idx_naive.to_period("M").duplicated()

# IMPORTANT: use .loc with the index mask (not .iloc)
short_entries.loc[first_of_month] = True

# Exit 5 trading days after the short entry
short_exits.loc[short_entries.shift(5).fillna(False)] = True

# Long entry 7 trading days before month-end (based on shift of first_of_month marker)
long_entries.loc[short_entries.shift(-7).fillna(False)] = True

# Long exit 1 trading day before month-end
long_exits.loc[short_entries.shift(-1).fillna(False)] = True

# -----------------------------
# 3) Run portfolio
# -----------------------------
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    freq="1D",
)

print(pf.stats())

# -----------------------------
# 4) Plot (reliable option)
# -----------------------------
fig = pf.plot()
fig.write_html("tlt_buy_and_hold_signals.html")
print("Saved chart to: tlt_buy_and_hold_signals.html (open in browser)")
