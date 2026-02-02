
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Mock Streamlit session state
class MockSessionState(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value

import streamlit as st
if not hasattr(st, 'session_state'):
    st.session_state = MockSessionState()
st.session_state["data_mode"] = "Preloaded Dataset"
# Mock other streamlit functions to avoid errors
st.cache_data = lambda func: func
st.error = print
st.warning = print
st.info = print
st.success = print

# Add root to path
sys.path.append(os.getcwd())

from src.market_data import load_preloaded_options, get_stock_history_vol
from pricing.heston_calibration import HestonCalibrator

def verify_calibration_logic():
    print("Starting Verification (Single Date, All Strikes)...")
    
    calib_ticker = "AAPL"
    calib_date = datetime(2025, 12, 17).date()
    calib_exp = datetime(2026, 2, 20).date()
    
    print(f"Ticker: {calib_ticker}, Date: {calib_date}, Exp: {calib_exp}")
    
    # 1. Load Data
    df_all = load_preloaded_options()
    if df_all.empty:
        print("Error: Preloaded options empty.")
        return

    # 2. Filter (Single Date, Call, Expiry)
    mask = (df_all['ticker'] == calib_ticker) & \
           (df_all['type'] == 'Call') & \
           (df_all['expiry'] == '2026-02-20') & \
           (df_all['Date'].dt.date == calib_date)
    
    calib_final_df = df_all[mask].copy().sort_values("Strike")
    print(f"Filtered Data Rows (Strikes): {len(calib_final_df)}")
    
    if calib_final_df.empty:
        print("Error: Final filtered dataset is empty.")
        return
        
    # 3. Spot Price
    S_real, _, _ = get_stock_history_vol(calib_ticker, "2025-12-17")
    print(f"Spot Price at {calib_date}: {S_real}")
    
    if not S_real:
        print("Error: No Spot Price")
        return

    # 4. Prepare Vectors
    # TimeToExp is constant
    T_exp = (pd.to_datetime(calib_exp) - pd.to_datetime(calib_date)).days / 365.0
    if T_exp < 0.001: T_exp = 0.001
    
    # Vector of constant T_exp
    calib_final_df['TimeToExp'] = T_exp 
    
    market_prices = calib_final_df['Close'].values
    strikes_arr = calib_final_df['Strike'].values
    maturities = calib_final_df['TimeToExp'].values
    
    print(f"Strikes Range: {strikes_arr.min()} - {strikes_arr.max()}")
    
    # 5. Run Calibration
    initial_guess = [0.04, 0.04, -0.7, 2.0, 0.3] 
    calibrator = HestonCalibrator(S_real, 0.05)
    
    print("Running Calibration (bounds + trf)...")
    # Note: user updated HestonCalibrator to use bounds and 'trf' internally
    res = calibrator.calibration(market_prices, strikes_arr, maturities, initial_guess)
    
    print(f"Calibration Success: {res.success}")
    print(f"Params: {res.x}")
    print("Verification Completed.")

if __name__ == "__main__":
    verify_calibration_logic()
