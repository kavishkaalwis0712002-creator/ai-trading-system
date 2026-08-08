# main.py

import os
import sys

# Path Configuration
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.data_collector import DataCollector
from modules.feature_engine import FeatureEngine
from modules.models import ModelEnsemble
from modules.confluence import ConfluenceEngine

def run_terminal_system():
    print("==================================================")
    print(" 🚀 AI MULTI-CONFLUENCE TRADING SYSTEM (TERMINAL) ")
    print("==================================================\n")

    # 1. Ingest Data
    print("[1/4] Fetching Live Market Data (XAU/USD)...")
    collector = DataCollector(symbol="XAUUSD")
    raw_df = collector.fetch_mt5_data(num_bars=500)
    if raw_df is None:
        raw_df = collector.fetch_yfinance_data(period="7d", interval="15m")

    if raw_df is None:
        print("❌ Data collection failed!")
        return

    # 2. Extract Features
    print("[2/4] Engineering Features (SMC, FVG, Indicators)...")
    fe = FeatureEngine(raw_df)
    df_processed = fe.build_all_features()

    # 3. Train ML Ensemble & Predict
    print("[3/4] Training AI Ensemble Models (XGBoost + LightGBM + LSTM)...")
    ensemble = ModelEnsemble()
    ensemble.train_all_models(df_processed)
    predictions = ensemble.predict_latest(df_processed)

    # 4. Generate Signal
    print("[4/4] Evaluating Confluence & Risk Matrix...\n")
    risk_engine = ConfluenceEngine()
    sig = risk_engine.evaluate(df_processed, predictions)

    # Print Results
    print("==================================================")
    print(f" 📊 MARKET PAIR : {sig['symbol']}")
    print(f" 📈 TREND REGIME: {sig['regime_trend']} | VOLATILITY: {sig['regime_volatility']}")
    print(f" 🛡️ NEWS STATUS : {sig['news_status']}")
    print("--------------------------------------------------")
    print(f" 🎯 AI SIGNAL   : {sig['signal']} (Grade: {sig['grade']})")
    print(f" 🔥 CONFIDENCE  : {sig['confidence']}%")
    print("--------------------------------------------------")
    print(f" 📍 Entry Price : {sig['entry']}")
    print(f" 🛑 Stop Loss   : {sig['sl']}")
    print(f" 🎯 Take Profit 1: {sig['tp1']}")
    print(f" 🎯 Take Profit 2: {sig['tp2']}")
    print(f" 🎯 Take Profit 3: {sig['tp3']}")
    print(f" ⚖️ Risk:Reward : {sig['rr_ratio']}")
    print(f" 📦 Lot Size    : {sig['recommended_lot']} Lots (0.5% Risk)")
    print("--------------------------------------------------")
    print(" 🤖 MODEL PROBABILITIES:")
    print(f"    • XGBoost    : {round(predictions['p_xgb'] * 100, 1)}%")
    print(f"    • LightGBM   : {round(predictions['p_lgb'] * 100, 1)}%")
    print(f"    • PyTorch LSTM: {round(predictions['p_lstm'] * 100, 1)}%")
    print(f"    • Ensemble   : {round(predictions['ensemble_p'] * 100, 1)}%")
    print("==================================================\n")

if __name__ == "__main__":
    run_terminal_system()