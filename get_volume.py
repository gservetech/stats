import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from twelvedata import TDClient
import pandas as pd

# 1. LOAD API KEY
load_dotenv(find_dotenv())
API_KEY = os.getenv("TWELVE_API_KEY")

if not API_KEY:
    print("❌ ERROR: Could not find 'TWELVE_API_KEY' in .env")
    exit()

# 2. CONFIGURATION
SYMBOL = "MU"
INTERVAL = "5min"
# 100 candles covers ~8 hours of 5-min data (Full Trading Day)
OUTPUT_SIZE = 100

td = TDClient(apikey=API_KEY)

print(f"--- FULL DAY MONITOR FOR {SYMBOL} ---")
print(f"✅ Fetching last {OUTPUT_SIZE} candles every 60s...")
print("Press Ctrl+C to stop.\n")

# 3. MAIN LOOP
while True:
    try:
        # Fetch Full Day Data (Costs same 1 Credit as fetching 2 candles)
        ts = td.time_series(
            symbol=SYMBOL,
            interval=INTERVAL,
            outputsize=OUTPUT_SIZE
        )

        # Convert to Pandas
        df = ts.as_pandas().sort_index()

        # --- CALCULATE DAY STATS ---
        # Filter for only "Today's" data (in case 100 candles goes back to yesterday)
        # Twelve Data returns time in local exchange time or UTC usually.
        # We'll just take the tail for simplicity in this view.

        day_open = df.iloc[0]['open']
        day_high = df['high'].max()
        day_low = df['low'].min()
        day_vol = df['volume'].sum()

        # Get Latest Candle for Breakout Logic
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(latest['close'])
        current_vol = int(latest['volume'])
        prev_vol = int(prev['volume'])

        now = datetime.now().strftime('%H:%M:%S')

        # --- DISPLAY OUTPUT ---
        print(f"\n[{now}] {SYMBOL} LIVE UPDATE")
        print(f"--------------------------------")
        print(f"🌞 DAY STATS:  High: ${day_high:.2f} | Low: ${day_low:.2f} | Vol: {int(day_vol):,}")
        print(f"⏱️ LATEST:     Price: ${current_price:.2f} | Vol (5m): {current_vol:,}")

        # --- BREAKOUT LOGIC ---
        vol_change_pct = ((current_vol - prev_vol) / prev_vol) * 100 if prev_vol > 0 else 0

        if current_vol > (prev_vol * 1.5):
            print(f"🚀 ALERT: Volume Spiking! (+{vol_change_pct:.0f}%)")
        elif current_vol < (prev_vol * 0.8):
            print(f"💤 QUIET: Volume dropping.")
        else:
            print(f"➡️ STEADY: Normal flow.")

        # Sleep to save credits (800 limit / day)
        time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(60)