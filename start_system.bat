@echo off
title AI Trading System Runner
cd /d "C:\Users\User\OneDrive\Desktop\ai_trading_system"
echo Starting Virtual Environment & Launching AI Trading System...
call venv\Scripts\activate
streamlit run app.py
pause