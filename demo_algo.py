"""
📊 Real-time Algo Trading Bot (Demo Version)

⚠️ Disclaimer:
This demo is built strictly for educational purposes only.
It is NOT intended for live trading or financial use.
"""
# NEED FULL PROJECT DM ME ON LINKEDIN www.linkedin.com/in/parthmulay
import yfinance as yf
import pandas as pd
import numpy as np

# Fetch stock data (last 3 months for demo)
print("Fetching stock data for RELIANCE.NS...")
df = yf.download("RELIANCE.NS", period="3mo", interval="1d")

# Add Moving Average (MA) Indicator
df["MA_20"] = df["Close"].rolling(window=20).mean()

# Add Relative Strength Index (RSI)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# Print last 5 rows with indicators
print("\nSample Data with Indicators:")
print(df[["Close", "MA_20", "RSI_14"]].tail())

# Generate a simple Buy / Hold / Sell signal
latest = df.iloc[-1]
signal = "HOLD"
if latest["RSI_14"] < 30:
    signal = "BUY (Oversold)"
elif latest["RSI_14"] > 70:
    signal = "SELL (Overbought)"

print(f"\nTrading Signal for {latest.name.date()}: {signal}")

