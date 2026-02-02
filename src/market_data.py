import streamlit as st
from massive import RESTClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def get_api_key():
    """Returns the user-provided API key from session state, or None."""
    return st.session_state.get("user_api_key")

def validate_api_key(api_key):
    """
    Attempts a simple API call to validate the key.
    Returns (bool, message)
    """
    client = RESTClient(api_key)
    try:
        _ = client.list_aggs("AAPL", 1, "day", "2023-01-01", "2023-01-02")
        return True, "API Key is valid."
    except Exception as e:
        return False, f"Validation Failed: {str(e)}"

@st.cache_data
def load_preloaded_options():
    try:
        df = pd.read_csv("data/extended dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_preloaded_prices():
    try:
        df = pd.read_csv("data/preloaded_prices.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        return pd.DataFrame()

def get_option_aggregates(ticker, expiration_date, option_type, strike, start_date, end_date, limit=120):
    """Fetch OHLCV aggregates for a specific option contract."""
    if st.session_state.get("data_mode") == "Preloaded Dataset":
        try:
            df_opt = load_preloaded_options()
            if df_opt.empty:
                return pd.DataFrame(), "Preloaded options CSV not found or empty."
            
            exp_str = expiration_date.strftime("%y%m%d")
            strike_val = int(strike * 1000)
            strike_str = f"{strike_val:08d}"
            contract_symbol = f"O:{ticker}{exp_str}{option_type}{strike_str}"
            
            mask = (df_opt['Symbol'] == contract_symbol) & \
                   (df_opt['Date'] >= pd.to_datetime(start_date)) & \
                   (df_opt['Date'] <= pd.to_datetime(end_date))
            
            res = df_opt[mask].sort_values("Date").head(limit)
            if res.empty:
                return pd.DataFrame(), f"No data found in preloaded dataset for {contract_symbol}"
            
            # Match schema expected by app. Include Implied Volatility if present.
            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if "Implied Volatility" in res.columns:
                cols.append("Implied Volatility")
            
            res = res.copy()
            if 'Transactions' not in res.columns:
                res['Transactions'] = 0
            cols.append("Transactions")
                
            return res[cols], None
        except Exception as e:
            return pd.DataFrame(), f"Local Data Error: {str(e)}"

    client = RESTClient(get_api_key())
    exp_str = expiration_date.strftime("%y%m%d")
    strike_val = int(strike*1000)
    strike_str = f"{strike_val:08d}"
    contract_symbol = f"O:{ticker}{exp_str}{option_type}{strike_str}"
    
    aggs = []
    try:
        results = client.list_aggs(
            contract_symbol, 1, "day", start_date, end_date,
            adjusted="true", sort="asc", limit=limit,
        )
        for a in results:
            aggs.append(a)
        if not aggs:
            return pd.DataFrame(), None
        data = []
        for a in aggs:
            data.append({
                "Date": datetime.fromtimestamp(a.timestamp / 1000) if hasattr(a, 'timestamp') else None,
                "Open": a.open, "High": a.high, "Low": a.low, "Close": a.close, "Volume": a.volume,
                "Transactions": getattr(a, 'transactions', 0)
            })
        return pd.DataFrame(data), None
    except Exception as e:
        if "429" in str(e):
            time.sleep(60)
            return get_option_aggregates(ticker, expiration_date, option_type, strike, start_date, end_date, limit)
        return pd.DataFrame(), str(e)

def get_option_previous_close(ticker, expiration_date, option_type, strike):
    """Fetch previous day bar for a specific option contract."""
    exp_str = expiration_date.strftime("%y%m%d")
    strike_val = int(strike * 1000)
    strike_str = f"{strike_val:08d}"
    contract_symbol = f"O:{ticker}{exp_str}{option_type}{strike_str}"

    if st.session_state.get("data_mode") == "Preloaded Dataset":
        try:
            df_opt = load_preloaded_options()
            if df_opt.empty:
                return pd.DataFrame(), "Preloaded options CSV not found or empty.", contract_symbol
            
            res = df_opt[df_opt['Symbol'] == contract_symbol].sort_values("Date")
            if res.empty:
                return pd.DataFrame(), "No data in preloaded CSV", contract_symbol
            
            latest = res.tail(1).copy()
            
            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if "Implied Volatility" in latest.columns:
                cols.append("Implied Volatility")
            
            if 'Transactions' not in latest.columns:
                latest['Transactions'] = 0
            cols.append("Transactions")
            
            return latest[cols], None, contract_symbol
        except Exception as e:
            return pd.DataFrame(), f"Local Error: {str(e)}", contract_symbol

    client = RESTClient(get_api_key())
    try:
        agg = client.get_previous_close_agg(contract_symbol, adjusted="true")
        results = [agg] if agg and not isinstance(agg, list) else (agg if agg else [])
        data = []
        for a in results:
            data.append({
                "Date": datetime.fromtimestamp(a.timestamp / 1000) if hasattr(a, 'timestamp') else None,
                "Open": a.open, "High": a.high, "Low": a.low, "Close": a.close, "Volume": a.volume,
                "Transactions": getattr(a, 'transactions', 0)
            })
        if not data:
             return pd.DataFrame(), "No previous close data found.", contract_symbol
        return pd.DataFrame(data), None, contract_symbol
    except Exception as e:
        if "429" in str(e):
             time.sleep(60)
             return get_option_previous_close(ticker, expiration_date, option_type, strike)
        return pd.DataFrame(), str(e), contract_symbol

def get_stock_history_vol(ticker, end_date_str, retries=10, shift_days=0):
    """Fetch 1 year of daily history to get Close and Vol."""
    if st.session_state.get("data_mode") == "Preloaded Dataset":
        try:
            df_p = load_preloaded_prices()
            if df_p.empty:
                 return None, None, "Preloaded prices CSV not found or empty."
            
            if ticker not in df_p.columns:
                return None, None, f"Ticker {ticker} not in preloaded prices."
            
            # Use current date or given end_date_str
            dt_end = pd.to_datetime(end_date_str)
            mask = (df_p['Date'] <= dt_end)
            df_subset = df_p[mask].tail(252).copy() # Use last 252 available days up to end_date
            
            if df_subset.empty:
                 return None, None, "No historical data found for window."
            
            last_close = df_subset.iloc[-1][ticker]
            df_subset["LogRet"] = np.log(df_subset[ticker] / df_subset[ticker].shift(1))
            vol = df_subset["LogRet"].std() * np.sqrt(252)
            
            return last_close, vol, None
        except Exception as e:
            return None, None, f"Local Error: {str(e)}"

    client = RESTClient(get_api_key())
    dt_end = datetime.strptime(end_date_str,("%Y-%m-%d"))
    dt_start = (dt_end - timedelta(days=365)) + timedelta(days=shift_days)
    start_str = dt_start.strftime("%Y-%m-%d")
    
    try:
        results = client.list_aggs(ticker, 1, "day", start_str, end_date_str, adjusted="true", limit=500, sort="asc")
        data = []
        for a in results:
             data.append({"Close": a.close, "Date": datetime.fromtimestamp(a.timestamp / 1000)})
        if not data:
            if retries > 0:
                time.sleep(1)
                return get_stock_history_vol(ticker, end_date_str, retries=retries-1, shift_days=shift_days+1)
            return None, None, "No underlying data found."
        df = pd.DataFrame(data)
        last_close = df.iloc[-1]["Close"]
        df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
        vol = df["LogRet"].std() * np.sqrt(252)
        return last_close, vol, None
    except Exception as e:
        if "429" in str(e):
             time.sleep(60)
             return get_stock_history_vol(ticker, end_date_str, retries, shift_days)
        return None, None, str(e)

def get_underlying_history_range(ticker, start_date_str, end_date_str):
    """Fetch daily close prices for underlying between dates."""
    if st.session_state.get("data_mode") == "Preloaded Dataset":
        try:
            df_p = load_preloaded_prices()
            if df_p.empty:
                return {}, "Preloaded prices CSV not found or empty."
            
            if ticker not in df_p.columns:
                return {}, f"Ticker {ticker} not in preloaded prices."
            
            mask = (df_p['Date'] >= pd.to_datetime(start_date_str)) & \
                   (df_p['Date'] <= pd.to_datetime(end_date_str))
            
            res = df_p[mask]
            data = {}
            for _, row in res.iterrows():
                data[row['Date'].date()] = row[ticker]
            return data, None
        except Exception as e:
            return {}, f"Local Error: {str(e)}"

    client = RESTClient(get_api_key())
    try:
        results = client.list_aggs(ticker, 1, "day", start_date_str, end_date_str, adjusted="true", limit=500, sort="asc")
        data = {}
        for a in results:
             d = datetime.fromtimestamp(a.timestamp / 1000).date()
             data[d] = a.close
        if not data:
            return {}, "No data found."
        return data, None
    except Exception as e:
        if "429" in str(e):
             time.sleep(60)
             return get_underlying_history_range(ticker, start_date_str, end_date_str)
        return {}, str(e)

def get_available_dates(ticker, opt_type):
    """Returns a sorted list of dates present in both options and prices CSVs for a ticker/type."""
    if st.session_state.get("data_mode") != "Preloaded Dataset":
        return []
    
    try:
        df_opt = load_preloaded_options()
        df_p = load_preloaded_prices()
        
        if df_opt.empty or df_p.empty:
            return []
        
        # Options Filter: Ticker and Type
        mask_opt = (df_opt['ticker'] == ticker) & (df_opt['type'] == ('Call' if opt_type == 'C' else 'Put'))
        available_opt_dates = set(df_opt[mask_opt]['Date'].dt.date)
        
        # Prices Filter: Ticker column must exist and be non-null
        if ticker not in df_p.columns:
            return []
        available_price_dates = set(df_p[df_p[ticker].notna()]['Date'].dt.date)
        
        # Intersection
        common = sorted(list(available_opt_dates.intersection(available_price_dates)))
        return common
    except Exception as e:
        st.error(f"Error fetching available dates: {e}")
        return []
def get_all_preloaded_options(ticker, expiration_date, option_type, start_date, end_date):
    """Fetch all strikes and OHLCV for a ticker/expiry/type range from preloaded dataset."""
    if st.session_state.get("data_mode") != "Preloaded Dataset":
        return pd.DataFrame(), "Not in Preloaded Dataset mode."
    
    try:
        df_opt = load_preloaded_options()
        if df_opt.empty:
            return pd.DataFrame(), "Preloaded options CSV not found or empty."
        
        # Determine exact type string from 'C'/'P'
        type_str = 'Call' if option_type == 'C' else 'Put'
        
        # Filter by Ticker, Expiry (formatted as YYYY-MM-DD), Type, and Date Range
        exp_str = expiration_date.strftime("%Y-%m-%d")
        
        mask = (df_opt['ticker'] == ticker) & \
               (df_opt['expiry'] == exp_str) & \
               (df_opt['type'] == type_str) & \
               (df_opt['Date'] >= pd.to_datetime(start_date)) & \
               (df_opt['Date'] <= pd.to_datetime(end_date))
        
        res = df_opt[mask].sort_values(["Date", "Strike"])
        
        if res.empty:
            return pd.DataFrame(), f"No data found for {ticker} {type_str} expiring {exp_str} in range."
            
        # Ensure schema consistency
        cols = ["Date", "Strike", "Open", "High", "Low", "Close", "Volume"]
        if "Implied Volatility" in res.columns:
            cols.append("Implied Volatility")
        
        res = res.copy()
        if 'Transactions' not in res.columns:
            res['Transactions'] = 0
        cols.append("Transactions")
        
        return res[cols], None
    except Exception as e:
        return pd.DataFrame(), f"Local Data Error: {str(e)}"
