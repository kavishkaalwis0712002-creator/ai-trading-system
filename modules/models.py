# modules/models.py

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------
# PyTorch Deep LSTM Model Architecture
# ---------------------------------------------------------
class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ---------------------------------------------------------
# AI Model Ensemble Manager
# ---------------------------------------------------------
class ModelEnsemble:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, eval_metric='logloss')
        self.lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, verbose=-1)
        self.scaler = StandardScaler()
        self.lstm_model = None
        
        # Core Features used for Training
        self.feature_cols = [
            'rsi', 'macd', 'atr', 'ema_20', 'ema_50', 
            'bsl_sweep', 'ssl_sweep', 'bos_bullish', 
            'fvg_bullish', 'fvg_bearish', 'bullish_ob', 'bearish_ob'
        ]

    def train_all_models(self, df):
        """Train XGBoost, LightGBM, and PyTorch LSTM models"""
        df = df.dropna().copy()
        
        X = df[self.feature_cols].values
        y = df['target'].values

        # Scaling
        X_scaled = self.scaler.fit_transform(X)

        # 1. Train XGBoost
        self.xgb_model.fit(X_scaled, y)

        # 2. Train LightGBM
        self.lgb_model.fit(X_scaled, y)

        # 3. Train PyTorch LSTM (Sequence creation)
        seq_len = 10
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - seq_len):
            X_seq.append(X_scaled[i:i+seq_len])
            y_seq.append(y[i+seq_len])

        X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)

        self.lstm_model = PyTorchLSTM(input_dim=len(self.feature_cols))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)

        self.lstm_model.train()
        for epoch in range(15):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_seq)
            loss = criterion(outputs, y_seq)
            loss.backward()
            optimizer.step()

        print("🤖 All AI Ensemble Models Trained Successfully!")

    def predict_latest(self, df):
        """Generate Probability outputs from all models for the latest candle"""
        df_latest = df.dropna().copy()
        X_latest = df_latest[self.feature_cols].values
        X_scaled = self.scaler.transform(X_latest)

        # Individual Model Probabilities
        p_xgb = float(self.xgb_model.predict_proba(X_scaled[-1:])[:, 1][0])
        p_lgb = float(self.lgb_model.predict_proba(X_scaled[-1:])[:, 1][0])

        # LSTM Prediction
        self.lstm_model.eval()
        seq_input = torch.tensor(np.array([X_scaled[-10:]]), dtype=torch.float32)
        with torch.no_grad():
            p_lstm = float(self.lstm_model(seq_input).item())

        # Weighted Ensemble Final Probability (Bullish Confidence)
        ensemble_p = (0.40 * p_xgb) + (0.35 * p_lgb) + (0.25 * p_lstm)

        return {
            "p_xgb": round(p_xgb, 4),
            "p_lgb": round(p_lgb, 4),
            "p_lstm": round(p_lstm, 4),
            "ensemble_p": round(ensemble_p, 4)
        }

if __name__ == "__main__":
    from data_collector import DataCollector
    from feature_engine import FeatureEngine

    collector = DataCollector(symbol="XAUUSD")
    raw_df = collector.fetch_mt5_data(num_bars=500)
    if raw_df is None:
        raw_df = collector.fetch_yfinance_data(period="7d", interval="15m")

    if raw_df is not None:
        fe = FeatureEngine(raw_df)
        df_processed = fe.build_all_features()

        ensemble = ModelEnsemble()
        ensemble.train_all_models(df_processed)
        predictions = ensemble.predict_latest(df_processed)

        print("\n🔥 Prediction Output Test:")
        print(predictions)