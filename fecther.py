import os
import time
from datetime import datetime, timedelta
from massive import RESTClient
import pandas as pd

API_KEY = "m96R5pBNQ9fm3__LOjSR4AYniFwiqVKh"
client = RESTClient(API_KEY)

def list_aggs_with_retry(symbol, multiplier, timespan, start, end, **kwargs):
    """Fetch aggregates with a 60s wait if rate limited (429)."""
    while True:
        try:
            return client.list_aggs(symbol, multiplier, timespan, start, end, **kwargs)
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit for {symbol}. Waiting 60s...")
                time.sleep(60)
                continue
            raise e

# Define tickers and date range
tickers = ["AAPL", "NVDA", "SPY"]
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# # --- Part 1: Underlying Retrieval ---
# print("\n--- Starting Underlying Retrieval ---")
# all_data = {}

# for ticker in tickers:
#     print(f"Fetching {ticker} from {start_date} to {end_date}...")
#     try:
#         aggs = list_aggs_with_retry(
#             ticker,
#             1,
#             "day",
#             start_date,
#             end_date,
#             adjusted="true",
#             sort="asc",
#             limit=500,
#         )
        
#         ticker_prices = []
#         for a in aggs:
#             # a.timestamp is usually in ms
#             date = datetime.fromtimestamp(a.timestamp / 1000).strftime('%Y-%m-%d')
#             ticker_prices.append({"Date": date, ticker: a.close})
        
#         df_ticker = pd.DataFrame(ticker_prices)
#         if not df_ticker.empty:
#             df_ticker.set_index("Date", inplace=True)
#             all_data[ticker] = df_ticker
#         else:
#             print(f"No data found for {ticker}")
            
#     except Exception as e:
#         print(f"Error fetching {ticker}: {e}")

# # Merge and Save
# if all_data:
#     new_df = pd.concat(all_data.values(), axis=1, join="outer").sort_index()
    
#     os.makedirs("data", exist_ok=True)
#     csv_path = "data/preloaded_prices.csv"
    
#     if os.path.exists(csv_path):
#         existing_df = pd.read_csv(csv_path, index_col=0)
#         # Combine, prioritize new data (though they should match), and sort
#         combined_df = pd.concat([existing_df, new_df], axis=0).groupby(level=0).last().sort_index()
#         combined_df.to_csv(csv_path)
#         print(f"\nSuccessfully updated prices in {csv_path}")
#     else:
#         new_df.to_csv(csv_path)
#         print(f"\nSuccessfully saved new prices to {csv_path}")
# else:
#     print("No data was fetched for any ticker.")


# --- Part 2: Options Retrieval ---
print("\n--- Starting Options Retrieval ---")
strikes = [570, 635, 670, 705, 770]
exp_str = "260220"  
ticker_opt = "SPY"

options_data = []

for strike in strikes:
    # Format symbol: O:NVDA251219C00145000
    # Strike is 8 digits, scaled by 1000
    strike_val = int(strike * 1000)
    strike_str = f"{strike_val:08d}"
    symbol = f"O:{ticker_opt}{exp_str}P{strike_str}"
    
    print(f"Fetching {symbol} from {start_date} to {end_date}...")
    try:
        aggs = list_aggs_with_retry(
            symbol,
            1,
            "day",
            start_date,
            end_date,
            adjusted="true",
            sort="asc",
            limit=500,
        )
        
        for a in aggs:
            date = datetime.fromtimestamp(a.timestamp / 1000).strftime('%Y-%m-%d')
            options_data.append({
                "Date": date,
                "Symbol": symbol,
                "Strike": strike,
                "Open": a.open,
                "High": a.high,
                "Low": a.low,
                "Close": a.close,
                "Volume": a.volume
            })
            
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

if options_data:
    df_new_options = pd.DataFrame(options_data)
    os.makedirs("data", exist_ok=True)
    options_csv_path = "data/preloaded_options.csv"
    
    if os.path.exists(options_csv_path):
        df_existing = pd.read_csv(options_csv_path)
        # Combine and drop duplicates based on Date and Symbol
        df_combined = pd.concat([df_existing, df_new_options], ignore_index=True)
        df_combined.drop_duplicates(subset=["Date", "Symbol"], keep="last", inplace=True)
        df_combined.sort_values(["Symbol", "Date"], inplace=True)
        df_combined.to_csv(options_csv_path, index=False)
        print(f"\nSuccessfully updated options data in {options_csv_path}")
    else:
        df_new_options.to_csv(options_csv_path, index=False)
        print(f"\nSuccessfully saved new options data to {options_csv_path}")
else:
    print("No options data was fetched.")