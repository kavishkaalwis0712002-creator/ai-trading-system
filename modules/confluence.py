# modules/confluence.py

import os
import sys

# Project Root Directory එක Path එකට add කිරීම
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

# Dynamic Import Handling
try:
    from config.config import ACCOUNT_BALANCE, RISK_PER_TRADE
except ModuleNotFoundError:
    try:
        from config import ACCOUNT_BALANCE, RISK_PER_TRADE
    except ModuleNotFoundError:
        ACCOUNT_BALANCE = 1000.0
        RISK_PER_TRADE = 0.005

class ConfluenceEngine:
    def __init__(self, account_balance=ACCOUNT_BALANCE, risk_per_trade=RISK_PER_TRADE):
        self.balance = account_balance
        self.risk_pct = risk_per_trade

    def evaluate(self, df, ensemble_predictions):
        latest = df.iloc[-1]
        p_bullish = ensemble_predictions['ensemble_p']

        # 1. CONFLUENCE SCORING MATRIX
        smc_score = 90 if (latest['ssl_sweep'] or latest['bullish_ob'] or latest['bos_bullish']) else 40
        fvg_score = 85 if latest['fvg_bullish'] else 35
        indicator_score = 80 if (latest['rsi'] > 45 and latest['macd'] > 0 and latest['regime_trend'] == "BULLISH") else 40
        news_safe = 95 

        final_confidence = round(
            (p_bullish * 35) + 
            (smc_score * 0.25) + 
            (fvg_score * 0.15) + 
            (indicator_score * 0.15) + 
            (news_safe * 0.10), 2
        )

        # 2. DECISION LOGIC MATRIX
        signal = "WAIT"
        grade = "NO TRADE"

        if final_confidence >= 75 and p_bullish > 0.55:
            signal = "BUY"
            grade = "A+" if final_confidence >= 85 else "A"
        elif final_confidence <= 35 and p_bullish < 0.45:
            signal = "SELL"
            grade = "A+" if final_confidence <= 25 else "A"
        else:
            signal = "WAIT"
            grade = "NO TRADE"

        # 3. DYNAMIC RISK MANAGEMENT
        entry = float(latest['close'])
        atr = float(latest['atr']) if not np.isnan(latest['atr']) else 3.0

        if signal == "BUY":
            sl = round(entry - (1.5 * atr), 2)
            risk_dist = abs(entry - sl)
            tp1 = round(entry + (risk_dist * 1.5), 2)
            tp2 = round(entry + (risk_dist * 2.5), 2)
            tp3 = round(entry + (risk_dist * 4.0), 2)
        elif signal == "SELL":
            sl = round(entry + (1.5 * atr), 2)
            risk_dist = abs(entry - sl)
            tp1 = round(entry - (risk_dist * 1.5), 2)
            tp2 = round(entry - (risk_dist * 2.5), 2)
            tp3 = round(entry - (risk_dist * 4.0), 2)
        else:
            sl, tp1, tp2, tp3, risk_dist = entry, entry, entry, entry, 1.0

        risk_amount = self.balance * self.risk_pct
        lot_size = round(risk_amount / (risk_dist * 100), 2) if risk_dist > 0 else 0.01
        lot_size = max(lot_size, 0.01)

        rr_ratio = f"1:{round(abs(tp2 - entry) / max(risk_dist, 0.01), 1)}"

        return {
            "symbol": latest.get('symbol', 'XAUUSD'),
            "signal": signal,
            "grade": grade,
            "confidence": final_confidence,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr_ratio": rr_ratio,
            "recommended_lot": lot_size,
            "regime_trend": latest['regime_trend'],
            "regime_volatility": latest['regime_volatility'],
            "news_status": "SAFE" if news_safe > 50 else "HIGH RISK"
        }