"""
Finite Difference Greek Computation for Non-Black-Scholes Models

This module provides numerical Greek computation using finite differences
for pricing models that don't have analytical Greek formulas.
"""
import numpy as np
from pricing.FFT_pricer import fft_pricer
from pricing.characteristic_functions import phi_vg, phi_merton


def compute_greeks_fd(S, K, T, r, sigma, option_type, model, model_params=None):
    """
    Compute Greeks using finite differences for FFT-based models.
    
    Parameters:
    -----------
    S : float - Spot price
    K : float - Strike price
    T : float - Time to maturity
    r : float - Risk-free rate
    sigma : float - Volatility
    option_type : str - 'C' for call, 'P' for put
    model : str - 'VG' or 'Merton'
    model_params : dict - Model-specific parameters
    
    Returns:
    --------
    dict : Dictionary containing delta, gamma, theta, vega, rho
    """
    if model_params is None:
        model_params = {}
    
    # Define pricing function based on model
    def price_fn(s, k, t, rate, vol):
        call = (option_type == "C")
        if model == "VG":
            theta = model_params.get('theta', -0.1)
            nu = model_params.get('nu', 0.2)
            return fft_pricer(k, s, t, rate, phi_vg, args=(vol, theta, nu), call=call)
        elif model == "Merton":
            lamb = model_params.get('lamb', 0.1)
            mu_j = model_params.get('mu_j', -0.05)
            sigma_j = model_params.get('sigma_j', 0.2)
            return fft_pricer(k, s, t, rate, phi_merton, args=(vol, lamb, mu_j, sigma_j), call=call)
        else:
            raise ValueError(f"Unknown model: {model}")
    
    # Epsilon values for finite differences
    eps_S = max(1e-4 * S, 0.01)
    eps_sigma = 1e-3
    eps_T = 1 / 365.0
    eps_r = 1e-4
    
    # Base price
    price = price_fn(S, K, T, r, sigma)
    
    # Delta: ∂V/∂S
    price_up_S = price_fn(S + eps_S, K, T, r, sigma)
    price_dn_S = price_fn(S - eps_S, K, T, r, sigma)
    delta = (price_up_S - price_dn_S) / (2 * eps_S)
    
    # Gamma: ∂²V/∂S²
    gamma = (price_up_S - 2 * price + price_dn_S) / (eps_S ** 2)
    
    # Vega: ∂V/∂σ
    price_up_sigma = price_fn(S, K, T, r, sigma + eps_sigma)
    price_dn_sigma = price_fn(S, K, T, r, sigma - eps_sigma)
    vega = (price_up_sigma - price_dn_sigma) / (2 * eps_sigma)
    
    # Theta: -∂V/∂T (negative because time decay)
    if T > eps_T:
        price_dn_T = price_fn(S, K, T - eps_T, r, sigma)
        theta = -(price - price_dn_T) / eps_T
    else:
        theta = 0.0
    
    # Rho: ∂V/∂r
    price_up_r = price_fn(S, K, T, r + eps_r, sigma)
    price_dn_r = price_fn(S, K, T, r - eps_r, sigma)
    rho = (price_up_r - price_dn_r) / (2 * eps_r)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
