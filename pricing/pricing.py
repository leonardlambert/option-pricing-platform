import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type):
    """
    Calculate Black-Scholes option price.
    """
    if T <= 0:
        return max(0, S - K) if option_type == "C" else max(0, K - S)
        
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "P":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return 0.0

def compute_greeks(S, K, T, r, sigma, option_type):
    """
    Compute Delta, Gamma, Theta, Vega, Rho.
    """
    if T <= 0:
        return 0, 0, 0, 0, 0
        
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    if option_type == "C":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
    return delta, gamma, theta, vega, rho

def calculate_pnl_attribution(S, K, T, r, sigma, option_type, dS, dVol, dT):
    """
    Calculate Taylor PnL Attribution component-wise for a single option.
    """
    delta, gamma, theta, vega, _ = compute_greeks(S, K, T, r, sigma, option_type)
    
    delta_attr = delta * dS
    gamma_attr = 0.5 * gamma * (dS**2)
    vega_attr = vega * dVol
    theta_attr = theta * dT
    
    return {
        "Delta": delta_attr,
        "Gamma": gamma_attr,
        "Vega": vega_attr,
        "Theta": theta_attr
    }

import math

# --- GBS Limits & Constants ---
MIN_T = 1.0 / 1000.0
MIN_X = 0.01
MIN_FS = 0.01
MIN_V = 0.005 
MAX_V = 5.0
MAX_STEPS = 100
PRECISION = 1.0e-5

def _approx_implied_vol(option_type, fs, x, t, r, b, cp):
    """
    Brenner & Subrahmanyam (1988), Feinstein (1988) approximation.
    Used for initial guess.
    """
    # Prevent divide by zero / domain errors
    t = max(t, MIN_T)
    x = max(x, MIN_X)
    fs = max(fs, MIN_FS)

    ebrt = math.exp((b - r) * t)
    ert = math.exp(-r * t)

    denom = (fs * ebrt + x * ert)
    if denom == 0: return 0.2 # Safe fallback
    
    a = math.sqrt(2 * math.pi) / denom

    if option_type == "C":
        payoff = fs * ebrt - x * ert
    else:
        print("option is put")
        payoff = x * ert - fs * ebrt

    b_term = cp - payoff / 2
    c_term = (payoff ** 2) / math.pi

    # Safety for sqrt
    radicand = b_term ** 2 + c_term
    if radicand < 0: radicand = 0
    
    v = (a * (b_term + math.sqrt(radicand))) / math.sqrt(t)
    return max(MIN_V, v)

def implied_vol_bisection(price, S, K, T, r, option_type, b=None):
    """
    True Bisection Search for Implied Volatility.
    """
    if b is None: b = r

    # Validate inputs
    if price <= 0: return None, "Price <= 0"
    if T <= MIN_T: return None, "Expired"

    def val_fn(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type)

    # 1. Check Arbitrage Bounds (Intrinsic Value)
    if option_type == "C":
        intrinsic = max(0, S * math.exp((b-r)*T) - K * math.exp(-r*T)) # Generalized
    else:
        print("option is put")
        intrinsic = max(0, K * math.exp(-r*T) - S * math.exp((b-r)*T))
    
    if price < intrinsic:
        return None, f"Price Below Intrinsic ({intrinsic:.4f})"

    # 2. Bracket the Root
    # We use a wide range [MIN_V, MAX_V] directly for safety in fallback
    v_low = MIN_V
    v_high = MAX_V
    
    p_low = val_fn(v_low)
    p_high = val_fn(v_high)

    if price < p_low:
        return v_low, "Price too low for model (Vol < Min)"
    if price > p_high:
        return v_high, "Price too high for model (Vol > Max)"

    # 3. Bisection
    for _ in range(MAX_STEPS):
        v_mid = (v_low + v_high) / 2.0
        p_mid = val_fn(v_mid)
        
        if abs(price - p_mid) < PRECISION:
            return v_mid, None
            
        if p_mid < price:
            v_low = v_mid
        else:
            v_high = v_mid
            
    return v_mid, "Bisection did not fully converge"


def implied_volatility(price, S, K, T, r, option_type, tol=1e-5, max_iter=100):
                       
    cutoff = S * 2 if option_type == "C" else K * 2
    #print("option_type", option_type)
    b = r # Std Black Scholes
    # 1. Newton-Raphson
    # Initial Guess
    v = _approx_implied_vol(option_type, S, K, T, r, b, price)
    v = max(MIN_V, min(MAX_V, v))

    for i in range(max_iter):
        p_curr = black_scholes_price(S, K, T, r, v, option_type)
        diff = price - p_curr
        
        if abs(diff) < tol:
            return v, None
            
        # Get Vega
        _, _, _, vega, _ = compute_greeks(S, K, T, r, v, option_type)
        
        if vega < 1e-8:
            # Low vega -> Newton unstable -> Switch to Bisection
            break
            
        v_new = v + diff / vega  # Newton step
        
        if v_new < MIN_V or v_new > MAX_V:
             # Out of bounds -> Switch to Bisection
             break
        
        v = v_new
    
    # Fallback to Bisection if Newton failed or bailed
    return implied_vol_bisection(price, S, K, T, r, option_type, b=b)
