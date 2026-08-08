# modules/feature_engine.py

import pandas as pd
import numpy as np
import ta

class FeatureEngine:
    def __init__(self, df):
        self.df = df.copy()

    # ---------------------------------------------------------
    # 1. SMART MONEY CONCEPTS (SMC) ENGINE
    # ---------------------------------------------------------
    def add_smc_features(self):
        """Liquidity Sweeps, Swing High/Low, Order Blocks, BOS/CHoCH"""
        df = self.df

        # Highs & Lows Swings
        df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))

        # Liquidity Sweeps
        df['bsl_sweep'] = np.where((df['high'] > df['high'].rolling(20).max().shift(1)) & 
                                   (df['close'] < df['high'].rolling(20).max().shift(1)), 1, 0)
        df['ssl_sweep'] = np.where((df['low'] < df['low'].rolling(20).min().shift(1)) & 
                                   (df['close'] > df['low'].rolling(20).min().shift(1)), 1, 0)

        # Structure Break (BOS)
        df['bos_bullish'] = np.where(df['close'] > df['high'].rolling(10).max().shift(1), 1, 0)
        df['bos_bearish'] = np.where(df['close'] < df['low'].rolling(10).min().shift(1), 1, 0)

        # Order Blocks (OB)
        df['bullish_ob'] = np.where((df['close'].shift(1) < df['open'].shift(1)) & 
                                    (df['close'] > df['high'].shift(1)), 1, 0)
        df['bearish_ob'] = np.where((df['close'].shift(1) > df['open'].shift(1)) & 
                                    (df['close'] < df['low'].shift(1)), 1, 0)
        self.df = df
        return self

    # ---------------------------------------------------------
    # 2. FAIR VALUE GAP (FVG) ENGINE
    # ---------------------------------------------------------
    def add_fvg_features(self):
        """Bullish/Bearish FVG and Gap Sizes"""
        df = self.df

        # Bullish FVG: Low of candle 3 > High of candle 1
        df['fvg_bullish'] = np.where(df['low'] > df['high'].shift(2), 1, 0)
        df['fvg_bullish_gap'] = np.where(df['fvg_bullish'] == 1, df['low'] - df['high'].shift(2), 0)

        # Bearish FVG: High of candle 3 < Low of candle 1
        df['fvg_bearish'] = np.where(df['high'] < df['low'].shift(2), 1, 0)
        df['fvg_bearish_gap'] = np.where(df['fvg_bearish'] == 1, df['low'].shift(2) - df['high'], 0)

        self.df = df
        return self

    # ---------------------------------------------------------
    # 3. INDICATORS & MARKET REGIME
    # ---------------------------------------------------------
    def add_indicators_and_regime(self):
        """EMA, RSI, MACD, ATR, Market Regime"""
        df = self.df

        # Moving Averages
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)

        # RSI & MACD
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])

        # Volatility & Market Regime
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        atr_mean = df['atr'].rolling(50).mean()

        df['regime_trend'] = np.where((df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200']), "BULLISH",
                             np.where((df['close'] < df['ema_50']) & (df['ema_50'] < df['ema_200']), "BEARISH", "SIDEWAYS"))
        df['regime_volatility'] = np.where(df['atr'] > atr_mean * 1.2, "HIGH", "NORMAL")

        # ML Target Variable (1 if Next 3 Candles Direction is UP, else 0)
        df['target'] = np.where(df['close'].shift(-3) > df['close'], 1, 0)

        self.df = df.dropna()
        return self

    def build_all_features(self):
        """Run complete extraction pipeline"""
        self.add_smc_features()
        self.add_fvg_features()
        self.add_indicators_and_regime()
        return self.df

if __name__ == "__main__":
    from data_collector import DataCollector
    
    collector = DataCollector(symbol="XAUUSD")
    raw_df = collector.fetch_mt5_data(num_bars=500)
    
    if raw_df is None:
        raw_df = collector.fetch_yfinance_data(period="7d", interval="15m")
        
    if raw_df is not None:
        fe = FeatureEngine(raw_df)
        processed_df = fe.build_all_features()
        print("\n✅ Feature Engine Pipeline Success!")
        print(f"Total Rows: {len(processed_df)} | Total Features: {processed_df.shape[1]}")
        print(processed_df[['timestamp', 'close', 'bsl_sweep', 'fvg_bullish', 'regime_trend', 'target']].tail(5))