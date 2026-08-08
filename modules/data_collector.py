# modules/data_collector.py

import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataCollector:
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol

    def connect_mt5(self):
        """MT5 Terminal එකට Connect වීම"""
        if not mt5.initialize():
            print("❌ MetaTrader5 Initialization Failed!")
            return False
        print("✅ MetaTrader5 Connected Successfully!")
        return True

    def fetch_mt5_data(self, timeframe=mt5.TIMEFRAME_M15, num_bars=2000):
        """MT5 එකෙන් Direct OHLCV Data Pull කිරීම"""
        if not self.connect_mt5():
            return None

        # Data Fetching
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, num_bars)
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            print(f"❌ Failed to fetch data for {self.symbol}")
            return None

        # DataFrame එකට Convert කිරීම
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        
        # Columns Rename & CleanUp
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df['symbol'] = self.symbol
        
        return df

    def fetch_yfinance_data(self, period="60d", interval="15m"):
        """MT5 නැති වෙලාවක Yahoo Finance එකෙන් Gold (GC=F) Data ගැනීම"""
        ticker = "GC=F" if self.symbol == "XAUUSD" else self.symbol
        print(f"🌐 Fetching Backup Data from Yahoo Finance ({ticker})...")
        
        df = yf.download(tickers=ticker, period=period, interval=interval)
        if df.empty:
            return None
            
        df = df.reset_index()
        df.columns = df.columns.get_level_values(0)  # MultiIndex remove කිරීම
        df.rename(columns={
            'Datetime': 'timestamp',
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        df['spread'] = 0.20  # Average Gold Spread Estimate
        df['symbol'] = self.symbol
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread', 'symbol']]

if __name__ == "__main__":
    # Test Data Collection Process
    collector = DataCollector(symbol="XAUUSD")
    
    print("Testing MT5 Data Fetch...")
    df = collector.fetch_mt5_data(timeframe=mt5.TIMEFRAME_M15, num_bars=500)
    
    # MT5 නැතිනම් Yahoo Finance මගින් Test කිරීම
    if df is None:
        print("Falling back to Yahoo Finance...")
        df = collector.fetch_yfinance_data(period="7d", interval="15m")
        
    if df is not None:
        print(f"\n✅ Data Fetch Successful! Total Rows: {len(df)}")
        print(df.tail())