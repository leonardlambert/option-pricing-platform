import streamlit as st
from massive import RESTClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Default API Key (Fallback)
DEFAULT_API_KEY = "m96R5pBNQ9fm3__LOjSR4AYniFwiqVKh"

def get_api_key():
    """Returns the user-provided API key from session state, or the default one."""
    return st.session_state.get("user_api_key", DEFAULT_API_KEY)

def validate_api_key(api_key):
    """
    Attempts a simple API call to validate the key.
    Returns (bool, message)
    """
    client = RESTClient(api_key)
    try:
        # Try fetching a very basic aggregate for a common ticker
        # If it doesn't raise an Auth error, it's likely valid.
        _ = client.list_aggs("AAPL", 1, "day", "2023-01-01", "2023-01-02")
        return True, "API Key is valid."
    except Exception as e:
        return False, f"Validation Failed: {str(e)}"

def get_option_aggregates(ticker, expiration_date, option_type, strike, start_date, end_date, limit=120):
    """
    Fetch OHLCV aggregates for a specific option contract using Massive API.
    
    Format: O:{ticker}{expiration}{type}{strike}
    Expiration: YYMMDD
    Strike: 8 digits (scaled by 1000, e.g. 65.00 -> 00065000)
    
    Args:
        ticker (str): Underlying symbol, e.g., "SPY"
        expiration_date (datetime): Expiration date object
        option_type (str): "C" or "P"
        strike (float): Strike price
        start_date (str): YYYY-MM-DD
        end_date (str): YYYY-MM-DD
    """
    client = RESTClient(get_api_key())
    
    # Format Expiration: YYMMDD
    exp_str = expiration_date.strftime("%y%m%d")
  
    strike_val = int(strike*1000)
    strike_str = f"{strike_val:08d}"
    
    contract_symbol = f"O:{ticker}{exp_str}{option_type}{strike_str}"
    
    aggs = []
    try:
        results = client.list_aggs(
            contract_symbol,
            1,
            "day",
            start_date,
            end_date,
            adjusted="true",
            sort="asc",
            limit=limit,
        )
        for a in results:
            aggs.append(a)
            
        if not aggs:
            return pd.DataFrame(), None

        # Assuming 'a' has standard OHLC attributes. Convert to DataFrame.
        # Massive/Polygon objects usually have .timestamp, .open, .high, .low, .close, .volume
        data = []
        for a in aggs:
            data.append({
                "Date": datetime.fromtimestamp(a.timestamp / 1000) if hasattr(a, 'timestamp') else None,
                "Open": a.open,
                "High": a.high,
                "Low": a.low,
                "Close": a.close,
                "Volume": a.volume,
                "Transactions": getattr(a, 'transactions', 0)
            })
            
        return pd.DataFrame(data), None
        
    except Exception as e:
        if "429" in str(e):
            print("Rate limit hit. Waiting 60s...")
            time.sleep(60)
            return get_option_aggregates(ticker, expiration_date, option_type, strike, start_date, end_date, limit)
        # Return empty DF and error message
        return pd.DataFrame(), str(e)

def get_option_previous_close(ticker, expiration_date, option_type, strike):
    """
    Fetch previous day bar (Previous Close) for a specific option contract.
    Returns (DataFrame, error_message)
    """
    client = RESTClient(get_api_key())
    
    exp_str = expiration_date.strftime("%y%m%d")
    strike_val = int(strike * 1000)
    strike_str = f"{strike_val:08d}"
    
    contract_symbol = f"O:{ticker}{exp_str}{option_type}{strike_str}"
    
    try:
        agg = client.get_previous_close_agg(
            contract_symbol,
            adjusted="true"
        )
        
        # agg is likely a single object directly, or a list? 
        # User example just printed it. Assuming standard behaviour similar to aggs but single.
        # Let's wrap in list if typical object, or check type.
        results = [agg] if agg and not isinstance(agg, list) else (agg if agg else [])
        
        data = []
        for a in results:
            data.append({
                "Date": datetime.fromtimestamp(a.timestamp / 1000) if hasattr(a, 'timestamp') else None,
                "Open": a.open,
                "High": a.high,
                "Low": a.low,
                "Close": a.close,
                "Volume": a.volume,
                "Transactions": getattr(a, 'transactions', 0)
            })
            
        if not data:
             return pd.DataFrame(), "No previous close data found.", contract_symbol
             
        return pd.DataFrame(data), None, contract_symbol
        
    except Exception as e:
        if "429" in str(e):
             print("Rate limit hit (prev close). Waiting 60s...")
             time.sleep(60)
             return get_option_previous_close(ticker, expiration_date, option_type, strike)
        return pd.DataFrame(), str(e), contract_symbol

def get_stock_history_vol(ticker, end_date_str, retries=10, shift_days=0):
    """
    Fetch 1 year of daily history for the underlying ticker to:
    1. Get the Close price on 'end_date_str' (for S).
    2. Calculate Historical Volatility (annualized std of log returns).
    
    Args:
        ticker (str): "SPY" etc.
        end_date_str (str): YYYY-MM-DD (Previous Close Date)
        retries (int): Number of retries allowed.
        shift_days (int): Days to shift the start date forward (used in recursion).
        
    Returns:
        (last_close, volatility, error_message)
    """
    client = RESTClient(get_api_key())
    
    # Approx 1 year back, shifted by retry count
    dt_end = datetime.strptime(end_date_str,("%Y-%m-%d"))
    dt_start = (dt_end - timedelta(days=365)) + timedelta(days=shift_days)
    start_str = dt_start.strftime("%Y-%m-%d")
    
    try:
        results = client.list_aggs(
            ticker, 
            1, "day", start_str, end_date_str,
            adjusted="true", limit=500, sort="asc"
        )
        
        data = []
        data = []
        for a in results:
             data.append({"Close": a.close, "Date": datetime.fromtimestamp(a.timestamp / 1000)})
             
        # Fallback: Try "I:" prefix if empty (for Indices like NDX, SPX)
        if not data and not ticker.startswith("I:"):
            print(f"Retrying with I:{ticker}...")
            results_idx = client.list_aggs(
                f"I:{ticker}", 
                1, "day", start_str, end_date_str,
                adjusted="true", limit=500, sort="asc"
            )
            for a in results_idx:
                data.append({"Close": a.close, "Date": datetime.fromtimestamp(a.timestamp / 1000)})

        # Fallback: Try "A:" prefix if empty (for stocks)
        if not data and not ticker.startswith("A:"):
            print(f"Retrying with A:{ticker}...")
            results_idx = client.list_aggs(
                f"A:{ticker}", 
                1, "day", start_str, end_date_str,
                adjusted="true", limit=500, sort="asc"
            )
            for a in results_idx:
                data.append({"Close": a.close, "Date": datetime.fromtimestamp(a.timestamp / 1000)})

        if not data:
            if retries > 0:
                print(f"No underlying data for range {start_str} to {end_date_str}. Retrying in 10s (Start +1 day)...")
                time.sleep(10)
                # Next retry: same end date, start date + 1, decrement retries
                return get_stock_history_vol(ticker, end_date_str, retries=retries-1, shift_days=shift_days+1)
            else:
                return None, None, "No underlying data found after 10 retries."
            
        df = pd.DataFrame(data)
        df.sort_values("Date", inplace=True)
        
        # 1. Last Close
        last_close = df.iloc[-1]["Close"]
        
        # 2. Historical Volatility
        df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
        # Std dev of log returns * sqrt(252)
        daily_std = df["LogRet"].std()
        vol = daily_std * np.sqrt(252)
        
        return last_close, vol, None
        
    except Exception as e:
        if "429" in str(e):
             print("Rate limit hit (stock hist). Waiting 60s...")
             time.sleep(60)
             # Use same args to retry exact call
             return get_stock_history_vol(ticker, end_date_str, retries, shift_days)
        return None, None, str(e)

def get_underlying_history_range(ticker, start_date_str, end_date_str):
    """
    Fetch daily close prices for underlying between dates.
    Returns ({date_obj: close_price}, error_message)
    """
    client = RESTClient(get_api_key())
    try:
        results = client.list_aggs(
            ticker, 
            1, "day", start_date_str, end_date_str,
            adjusted="true", limit=500, sort="asc"
        )
        
        data = {}
        data = {}
        for a in results:
             d = datetime.fromtimestamp(a.timestamp / 1000).date()
             data[d] = a.close
             
        # Fallback: Try "I:" prefix
        if not data and not ticker.startswith("I:"):
            results_idx = client.list_aggs(
                f"I:{ticker}", 
                1, "day", start_date_str, end_date_str,
                adjusted="true", limit=500, sort="asc"
            )
            for a in results_idx:
                 d = datetime.fromtimestamp(a.timestamp / 1000).date()
                 data[d] = a.close

        # Fallback: Try "A:" prefix
        if not data and not ticker.startswith("A:"):
            results_idx = client.list_aggs(
                f"A:{ticker}", 
                1, "day", start_date_str, end_date_str,
                adjusted="true", limit=500, sort="asc"
            )
            for a in results_idx:
                 d = datetime.fromtimestamp(a.timestamp / 1000).date()
                 data[d] = a.close
             
        if not data:
            return {}, "No underlying data found for range."
            
        return data, None
        
    except Exception as e:
        if "429" in str(e):
             print("Rate limit hit (underlying range). Waiting 60s...")
             time.sleep(60)
             return get_underlying_history_range(ticker, start_date_str, end_date_str)
        return {}, str(e)
