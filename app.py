# app.py

import os
import sys

# Root Path Configuration
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_collector import DataCollector
from modules.feature_engine import FeatureEngine
from modules.models import ModelEnsemble
from modules.confluence import ConfluenceEngine

st.set_page_config(page_title="AI Multi-Confluence Trading System (LIVE)", layout="wide", page_icon="🚀")

st.title("🚀 AI Multi-Confluence Trading Intelligence System (LIVE ENGINE)")
st.caption("Institutional Gold (XAU/USD) Live Signal Engine")
st.markdown("---")

# ⏱️ Native Streamlit Auto-Refresh Fragment (Runs every 30 seconds)
@st.fragment(run_every="30s")
def render_live_dashboard():
    with st.spinner("⚡ Fetching Live Market Data & Re-evaluating AI Ensemble..."):
        # 1. Pipeline Execution
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

            risk_engine = ConfluenceEngine()
            signal_data = risk_engine.evaluate(df_processed, predictions)

    # Top Bar Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Market Pair", signal_data['symbol'])
    col2.metric("Market Trend", signal_data['regime_trend'])
    col3.metric("Volatility", signal_data['regime_volatility'])
    col4.metric("Signal Grade", signal_data['grade'])
    col5.metric("Final Confidence", f"{signal_data['confidence']}%")

    st.markdown("---")

    # Main Content Layout
    left_col, right_col = st.columns([1.1, 2])
    latest_row = df_processed.iloc[-1]

    with left_col:
        st.subheader("🎯 Active AI Live Signal")
        sig = signal_data['signal']
        sig_icon = "🟢" if sig == "BUY" else ("🔴" if sig == "SELL" else "🟡")
        
        st.markdown(f"## {sig_icon} SIGNAL: **{sig}**")
        st.write(f"**Live Entry Price:** `${signal_data['entry']:.2f}`")
        st.write(f"**Stop Loss (SL):** `${signal_data['sl']:.2f}`")
        st.write(f"**Take Profit 1 (TP1):** `${signal_data['tp1']:.2f}`")
        st.write(f"**Take Profit 2 (TP2):** `${signal_data['tp2']:.2f}`")
        st.write(f"**Take Profit 3 (TP3):** `${signal_data['tp3']:.2f}`")
        st.write(f"**Risk : Reward Ratio:** `{signal_data['rr_ratio']}`")
        st.write(f"**Recommended Lot Size:** `{signal_data['recommended_lot']} Lots` (0.5% Risk)")

        st.markdown("---")
        st.subheader("🤖 Live Model Probabilities")
        st.write(f"• **XGBoost Prob:** `{round(predictions['p_xgb'] * 100, 1)}%`")
        st.write(f"• **LightGBM Prob:** `{round(predictions['p_lgb'] * 100, 1)}%`")
        st.write(f"• **PyTorch LSTM Prob:** `{round(predictions['p_lstm'] * 100, 1)}%`")
        st.write(f"• **Weighted Ensemble:** `{round(predictions['ensemble_p'] * 100, 1)}%`")

        st.markdown("---")
        st.subheader("💡 Why this Signal was Generated?")
        reasons = []
        if latest_row['ssl_sweep']:
            reasons.append("✅ **Sell-Side Liquidity Sweep:** Bullish reversal signal detected.")
        if latest_row['bsl_sweep']:
            reasons.append("⚠️ **Buy-Side Liquidity Sweep:** Bearish reversal signal detected.")
        if latest_row['bullish_ob']:
            reasons.append("✅ **Bullish Order Block:** Price reacting off an institutional zone.")
        if latest_row['bos_bullish']:
            reasons.append("✅ **Break of Structure (BOS):** Bullish trend continuation.")
        if latest_row['fvg_bullish']:
            reasons.append("✅ **Bullish Fair Value Gap:** Imbalance zone for entry retest.")
        
        if latest_row['rsi'] > 50:
            reasons.append(f"📈 **RSI:** `{latest_row['rsi']:.1f}` (Bullish Momentum)")
        else:
            reasons.append(f"📉 **RSI:** `{latest_row['rsi']:.1f}` (Bearish / Neutral Momentum)")

        if predictions['ensemble_p'] < 0.45 and sig == "WAIT":
            reasons.append("❌ **Low AI Probability:** Ensemble confidence below 75% threshold.")

        for r in reasons:
            st.write(r)

    with right_col:
        st.subheader("📊 Live Price & Indicator Analysis Chart")
        chart_df = df_processed.tail(100)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=chart_df['timestamp'], open=chart_df['open'], high=chart_df['high'],
            low=chart_df['low'], close=chart_df['close'], name="Gold Price"
        ), row=1, col=1)

        # EMAs
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['ema_20'], name="EMA 20", line=dict(color='orange', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['ema_50'], name="EMA 50", line=dict(color='cyan', width=1.2)), row=1, col=1)

        # Entry / SL / TP Visualization
        if sig in ["BUY", "SELL"]:
            fig.add_hline(y=signal_data['entry'], line_dash="dash", line_color="blue", annotation_text="ENTRY", row=1, col=1)
            fig.add_hline(y=signal_data['sl'], line_dash="solid", line_color="red", annotation_text="STOP LOSS", row=1, col=1)
            fig.add_hline(y=signal_data['tp1'], line_dash="dot", line_color="green", annotation_text="TP 1", row=1, col=1)

        # Volume & RSI
        fig.add_trace(go.Bar(x=chart_df['timestamp'], y=chart_df['volume'], name="Volume", marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['rsi'], name="RSI", line=dict(color='purple', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Run Auto-Refresh Fragment Function
render_live_dashboard()