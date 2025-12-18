import streamlit as st
import json
import os
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime

BOOK_FILE = "data/option_book.json"

def initialize_session_state():
    if "book" not in st.session_state:
        st.session_state.book = load_book()
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "Live API"

def load_book():
    if os.path.exists(BOOK_FILE):
        try:
            with open(BOOK_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_book(book):
    os.makedirs("data", exist_ok=True)
    with open(BOOK_FILE, "w") as f:
        json.dump(book, f, indent=2)

def add_strategy_to_book(S0, T, r, sigma, legs, name="Untitled Strategy"):
    strategy = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "S0": S0, "T": T, "r": r, "sigma": sigma,
        "legs": legs
    }
    st.session_state.book.append(strategy)
    save_book(st.session_state.book)

def delete_strategy(index):
    if 0 <= index < len(st.session_state.book):
        st.session_state.book.pop(index)
        save_book(st.session_state.book)

def reset_book():
    st.session_state.book = []
    save_book([])

def interpolate_volatility(strikes, ivs, target_x):
    """
    Interpolates IV at a specific target x (strike or moneyness).
    Uses spline interpolation if enough points, else linear.
    """
    if len(strikes) < 2:
        return None
        
    try:
        # Ensure inputs are numpy arrays
        x_sorted = np.array(strikes)
        y_sorted = np.array(ivs)
        
        # Sort by strike
        sorted_indices = np.argsort(x_sorted)
        x_sorted = x_sorted[sorted_indices]
        y_sorted = y_sorted[sorted_indices]
        
        if len(x_sorted) >= 4:
            try:
                spline = make_interp_spline(x_sorted, y_sorted)
                return float(spline(target_x))
            except:
                return float(np.interp(target_x, x_sorted, y_sorted))
        else:
            return float(np.interp(target_x, x_sorted, y_sorted))
    except Exception as e:
        print(f"Interpolation Error: {e}")
        return None
