import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add root to path to import pricing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pricing.pricing import implied_volatility

def process_and_merge():
    # Paths
    new_data_path = os.path.join("data", "extended_options_data.csv")
    main_dataset_path = os.path.join("data", "extended dataset.csv")
    
    if not os.path.exists(new_data_path):
        print(f"Error: {new_data_path} not found.")
        return

    # Load new data
    df_new = pd.read_csv(new_data_path)
    
    # Constants
    r = 0.05
    model = "Black-Scholes-Merton"
    settlement = "European"
    expiry = "2026-02-20"
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    
    # Spot prices on 2025-12-17
    spot_prices = {
        "AAPL": 271.84,
        "NVDA": 170.94,
        "SPY": 671.4
    }

    # Prepare for calculation
    processed_rows = []
    
    print("Calculating IV for new data...")
    for index, row in df_new.iterrows():
        ticker = row["Ticker"]
        strike = row["Strike"]
        option_type = "C" if row["Type"] == "Call" else "P"
        price = row["Close"]
        date_str = row["Date"]
        date_dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        # S (Spot price)
        S = spot_prices.get(ticker)
        if S is None:
            print(f"Warning: No spot price for {ticker}. Skipping.")
            continue
            
        # T (Time to maturity in years)
        T = (expiry_dt - date_dt).days / 365.0
        
        # IV calculation
        iv, error = implied_volatility(price, S, strike, T, r, option_type)
        
        if error:
            print(f"  Warning for {row['Contract']}: {error}. Setting IV to NaN.")
            iv = np.nan
            
        # Map to main dataset schema
        # Date,ticker,expiry,time_to_maturity,type,Strike,Open,High,Low,Close,Volume,Implied Volatility,r,model,settlement,Symbol
        processed_row = {
            "Date": date_str,
            "ticker": ticker,
            "expiry": expiry,
            "time_to_maturity": T,
            "type": row["Type"],
            "Strike": strike,
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": price,
            "Volume": row["Volume"],
            "Implied Volatility": iv,
            "r": r,
            "model": model,
            "settlement": settlement,
            "Symbol": row["Contract"]
        }
        processed_rows.append(processed_row)

    df_processed = pd.DataFrame(processed_rows)
    
    # Merge with existing data
    if os.path.exists(main_dataset_path):
        df_main = pd.read_csv(main_dataset_path)
        # Avoid duplicates if script is run multiple times for same data
        # We can check Symbol AND Date
        df_combined = pd.concat([df_main, df_processed], ignore_index=True)
        # Optional: Drop exact duplicates
        df_combined = df_combined.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    else:
        df_combined = df_processed

    # Save
    df_combined.to_csv(main_dataset_path, index=False)
    print(f"\nSuccessfully integrated {len(df_processed)} rows into {main_dataset_path}")

if __name__ == "__main__":
    process_and_merge()
