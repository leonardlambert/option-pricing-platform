import pandas as pd
import numpy as np
from datetime import datetime
import os
from massive import RESTClient
import time


client = RESTClient("m96R5pBNQ9fm3__LOjSR4AYniFwiqVKh")


# fetch parameters
tickers_strikes = {
    "AAPL": [235, 240, 245, 250, 260, 265, 275, 280, 290, 295, 300, 305],
    "NVDA": [150, 155],
    "SPY": [585, 600, 615, 630, 645, 695, 710, 725, 740, 755]
}

expiration = "260220" # Feb 20, 2026
option_types = ["C", "P"]

all_data = []

for ticker, strikes in tickers_strikes.items():
    print(f"Fetching data for {ticker}...")
    for option_type in option_types:
        print(f"  Fetching {('Calls' if option_type == 'C' else 'Puts')}...")
        for s in strikes:
            # Format strike for Polygon ticker (e.g., 150 -> 00150000)
            strike_val = int(s * 1000)
            strike_str = f"{strike_val:08d}"
            contract_symbol = f"O:{ticker}{expiration}{option_type}{strike_str}"
            
            print(f"    Requesting {contract_symbol}...")
            try:
                aggs = client.list_aggs(
                    contract_symbol, 
                    1,
                    "day",
                    "2025-12-17", 
                    "2025-12-17",
                    adjusted=True,
                    sort="asc",
                    limit=120
                )
                
                for a in aggs:
                    row = {
                        "Ticker": ticker,
                        "Strike": s,
                        "Type": "Call" if option_type == "C" else "Put",
                        "Contract": contract_symbol,
                        "Date": datetime.fromtimestamp(a.timestamp / 1000).strftime("%Y-%m-%d") if hasattr(a, 'timestamp') else None,
                        "Open": a.open,
                        "High": a.high,
                        "Low": a.low,
                        "Close": a.close,
                        "Volume": a.volume
                    }
                    all_data.append(row)
            except Exception as e:
                print(f"Error fetching {contract_symbol}: {e}")

            # Sleep to respect rate limits (60s / 5 calls for free tier usually, but user had 20s)
            time.sleep(20) 

# Create DataFrame and save
df = pd.DataFrame(all_data)
if not df.empty:
    output_path = os.path.join("data", "extended_options_data.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully fetched {len(df)} rows.")
    print(f"Data saved to {output_path}")
    print(df.head())
else:
    print("No data fetched.")