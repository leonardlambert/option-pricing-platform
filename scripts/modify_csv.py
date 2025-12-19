import pandas as pd
import re
from datetime import datetime
import os
import sys

# Add project root to path to import pricing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from pricing.pricing import implied_volatility

def modify_csv(file_path, prices_path):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Reading {prices_path}...")
    prices_df = pd.read_csv(prices_path)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    
    # Store original column order to identify where to insert
    
    def parse_symbol(symbol):
        # Example O:AAPL260220C00230000
        match = re.search(r':([A-Z]+)(\d{6})([CP])', symbol)
        if match:
            ticker = match.group(1)
            expiry_str = match.group(2)
            type_char = match.group(3)
            
            # Turned 6 numbers into datetime object
            expiry_dt = datetime.strptime(expiry_str, '%y%m%d')
            type_val = 'Call' if type_char == 'C' else 'Put'
            
            return ticker, expiry_dt, type_val
        return None, None, None

    print("Parsing symbols and calculating time to maturity...")
    parsed_data = df['Symbol'].apply(parse_symbol)
    df['ticker'] = parsed_data.apply(lambda x: x[0])
    df['expiry'] = parsed_data.apply(lambda x: x[1])
    df['type'] = parsed_data.apply(lambda x: x[2])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['time_to_maturity'] = (df['expiry'] - df['Date']).dt.days / 365.0
    
    print("Calculating Implied Volatility...")
    def get_iv(row):
        date = row['Date']
        ticker = row['ticker']
        
        # Get spot price
        day_prices = prices_df[prices_df['Date'] == date]
        if day_prices.empty or ticker not in day_prices.columns:
            return None
        
        S = day_prices[ticker].values[0]
        K = row['Strike']
        T = row['time_to_maturity']
        market_price = row['Close']
        r = 0.05
        option_type_code = 'C' if row['type'] == 'Call' else 'P'
        
        iv, error = implied_volatility(market_price, S, K, T, r, option_type_code)
        return iv

    df['Implied Volatility'] = df.apply(get_iv, axis=1)
    
    # Add static columns
    df['r'] = 0.05
    df['model'] = "Black-Scholes-Merton"
    df['settlement'] = "European"
    
    # Reorder columns
    ordered_cols = [
        'Date', 'ticker', 'expiry', 'time_to_maturity', 'type',
        'Strike', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Implied Volatility', 'r', 'model', 'settlement', 'Symbol'
    ]
    
    df = df[ordered_cols]
    
    # Formatting for CSV
    df['expiry'] = df['expiry'].dt.strftime('%Y-%m-%d')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    # Round IV for readability
    df['Implied Volatility'] = df['Implied Volatility'].round(4)
    
    output_path = file_path
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    csv_path = r'd:\Code\antigravity\projects\market-risk-pricing-platform\data\preloaded_options.csv'
    prices_path = r'd:\Code\antigravity\projects\market-risk-pricing-platform\data\preloaded_prices.csv'
    
    # Backup first
    backup_path = csv_path.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(csv_path, backup_path)
        print(f"Backup created at {backup_path}")
        
    modify_csv(csv_path, prices_path)
