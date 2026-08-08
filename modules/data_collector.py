# modules/data_collector.py

import pandas as pd
import yfinance as yf
from datetime import datetime

# Streamlit Cloud (Linux) වලදී MetaTrader5 import error නොවී handle කිරීමට:
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MT5_AVAILABLE = False
    mt5 = None

class DataCollector:
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol

    def fetch_mt5_data(self, num_bars=500):
        if not MT5_AVAILABLE:
            print("⚠️ MetaTrader5 is not available on this OS/Cloud environment.")
            return None

        if not mt5.initialize():
            print("❌ MetaTrader5 Initialization Failed!")
            return None

        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, num_bars)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['spread'] = df['spread'].astype(float)
        df['symbol'] = self.symbol
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'symbol']]

    def fetch_yfinance_data(self, period="7d", interval="15m"):
        yf_symbol = "GC=F" if self.symbol in ["XAUUSD", "GOLD"] else self.symbol
        print(f"🌐 Fetching Backup Data from Yahoo Finance ({yf_symbol})...")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return None

        df = df.reset_index()
        df.rename(columns={
            'Datetime': 'timestamp',
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        df['spread'] = 0.30
        df['symbol'] = self.symbol
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'symbol']]